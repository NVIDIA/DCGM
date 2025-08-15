# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dcgm_structs
import dcgm_fields
import dcgm_errors
import logger
import test_utils
import DcgmMnDiag
from MockMnubergemmController import MockMnubergemmController

import os
from ctypes import *
import json

# ------------------------------------------------------
def helper_setup_and_run_mndiag(handle, config):
    """
    Helper to setup and run multinode mndiag
    """
    try :
        mock_controller = MockMnubergemmController(test_utils.get_mock_mnubergemm_path(), test_utils.get_mock_nvidia_smi_path())

        if len(config["test_nodes"]) > 0:
            mock_controller.setup_multinode_mock_environment(config)
            hostList = config["hostList"] + [node["ip"] for node in config["test_nodes"]]
        else:
            mock_controller.setup_localhost_mock_environment(config)
            hostList = config["hostList"]

        error_entities = mock_controller.get_error_entities()
        driver_versions = mock_controller.get_driver_versions(config)
        testName = "mnubergemm"
        testParams = [
            "mnubergemm.time_to_run=1"
        ]

        # Run mndiag
        mndiag = DcgmMnDiag.DcgmMnDiag(hostList=hostList, testName=testName, parameters=testParams, handle=handle)
        response = mndiag.Execute(handle)

    except Exception:
        helper_print_stderr_output()
        raise
    
    finally:
        mock_controller.cleanup_multinode_mock_environment(config)
    
    return response, error_entities, driver_versions

# ------------------------------------------------------
def helper_validate_response(response, config, testName="MNUBERGEMM", success=None, error_entities={}, driver_versions=[]):
    """
    Helper to validate mndiag response structure
    """
    hostList = config["hostList"] + [node["ip"] for node in config["test_nodes"]]
    expected_num_errors = config.get("expected_num_errors", 0)

    assert response is not None, "Response should not be None"
    assert response.version == dcgm_structs.dcgmMnDiagResponse_version1, f"Got response version {response.version}, expected {dcgm_structs.dcgmMnDiagResponse_version1}"

    # Validate host information
    assert response.numHosts == len(hostList), f"numHosts {response.numHosts} should match hostList length {len(hostList)}"
    for i in range(response.numHosts):
        host = response.hosts[i]
        assert len(host.hostname) > 0, f"Host {i} has empty hostname"
        assert host.hostname in hostList, f"Host {i} hostname {host.hostname} not in hostList"

    # Validate entity results
    assert response.numEntities == response.numResults, f"numEntities {response.numEntities} should match numResults {response.numResults}"
    assert response.numEntities == config["expected_number_of_entities"], f"numEntities {response.numEntities} should match expected_number_of_entities {config['expected_number_of_entities']}"
    for i in range(response.numResults):
        result = response.results[i]
        assert result.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU, f"Entity {i} has invalid entityGroupId. Got {result.entity.entityGroupId}, expected {dcgm_fields.DCGM_FE_GPU}"
        assert result.testId == 0, f"Entity {i} has invalid testId. Got {result.testId}, expected 0"

        # Get hostname for this result
        assert result.hostId < response.numHosts, f"Entity {i} has invalid hostId. Got {result.hostId}, expected < {response.numHosts}"
        hostname = response.hosts[result.hostId].hostname
        assert hostname in hostList, f"Entity {i} has hostname {hostname} not in hostList"

        # Validate error entities
        if len(error_entities.keys()) > 0:
            entity_key = f"{hostname}:{result.entity.entityId}"
            if entity_key in error_entities.keys():
                assert result.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, f"Entity {i} with key {entity_key} has invalid result: {result.result}. Expected {dcgm_structs.DCGM_DIAG_RESULT_FAIL}"
            else:
                assert result.result == dcgm_structs.DCGM_DIAG_RESULT_PASS, f"Entity {i} with key {entity_key} has invalid result: {result.result}. Expected {dcgm_structs.DCGM_DIAG_RESULT_PASS}"

    # Validate test results
    assert response.numTests == 1, "numTests should be 1"
    for i in range(response.numTests):
        test = response.tests[i]
        assert test.name == testName, f"Test name mismatch: expected {testName}, got {test.name}"
        if success:
            assert test.result == dcgm_structs.DCGM_DIAG_RESULT_PASS, f"Test result invalid: {test.result}. Expected {dcgm_structs.DCGM_DIAG_RESULT_PASS}"
        else:
            assert test.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, f"Test result invalid: {test.result}. Expected {dcgm_structs.DCGM_DIAG_RESULT_FAIL}"

    # Validate error count matches expected errors
    assert response.numErrors == expected_num_errors, f"Expected {expected_num_errors} errors, got {response.numErrors}"
    if not success:
        for i in range(response.numErrors):
            error = response.errors[i]
            assert error.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU, f"Error {i} has invalid entityGroupId. Got {error.entity.entityGroupId}, expected {dcgm_fields.DCGM_FE_GPU}"
            assert error.testId == 0, f"Error {i} has invalid testId. Got {error.testId}, expected 0"
            assert error.code == dcgm_errors.DCGM_FR_UNKNOWN, f"Error {i} has invalid error code. Got {error.code}, expected {dcgm_errors.DCGM_FR_UNKNOWN}"
            assert error.category == dcgm_errors.DCGM_FR_EC_HARDWARE_OTHER, f"Error {i} has invalid category. Got {error.category}, expected {dcgm_errors.DCGM_FR_EC_HARDWARE_OTHER}"
            assert error.severity == dcgm_errors.DCGM_ERROR_TRIAGE, f"Error {i} has invalid severity. Got {error.severity}, expected {dcgm_errors.DCGM_ERROR_TRIAGE}"
            assert len(error.msg) > 0, f"Error {i} has empty message"

            all_error_msgs = [m.strip() for m in error.msg.split(';') if m.strip()]
            hostname = response.hosts[response.errors[i].hostId].hostname
            entity_key = f"{hostname}:{error.entity.entityId}"
            for expected_msg in error_entities[entity_key]:
                assert any(expected_msg in actual for actual in all_error_msgs), (
                    f"Expected error message '{expected_msg}' not found in actual error messages {all_error_msgs} "
                    f"for entity {entity_key} (error index {i})"
                )

    # Validate host entities are populated correctly
    for i in range(response.numHosts):
        host = response.hosts[i]
        dcgm_version = test_utils.get_dcgm_version()
        assert host.numEntities > 0, f"Host {i} has no entities. Got {host.numEntities}, expected > 0"
        assert host.dcgmVersion == dcgm_version, f"Host {i} dcgmVersion mismatch: got '{host.dcgmVersion}', expected '{dcgm_version}'"
        assert host.driverVersion in driver_versions, f"Host {i} driverVersion '{host.driverVersion}' not found in expected driver versions {driver_versions}"

# ------------------------------------------------------
def helper_validate_error_output(response, validation_config):
    """
    Flexible helper to validate mndiag output for both success and failure cases.

    validation_config: dict with keys:
        - gpu_required_error: list of substrings that must appear in errors[]
    """
    # Split error messages on ';' and flatten
    error_msgs = []
    for i in range(response.numErrors):
        error_msgs.extend([msg.strip() for msg in response.errors[i].msg.split(';') if msg.strip()])

    for gpu_id, required_error_msgs in validation_config.get("gpu_required_error", {}).items():
        for required_error in required_error_msgs:
            assert any(required_error in msg for msg in error_msgs), \
                f"Required error substring '{required_error}' not found in error messages for GPU {gpu_id}"

# ------------------------------------------------------
def helper_check_generated_log_file(config, mnubergemm_log_file):
    """
    Helper to check the generated log file for required info and error substrings
    Also validates test nodes' info and error messages if test nodes are present
    """
    # Check log exists
    assert os.path.exists(mnubergemm_log_file), f"Log file {mnubergemm_log_file} does not exist"

    # Open the logs and check if its not empty
    with open(mnubergemm_log_file, 'r') as f:
        log_lines = f.readlines()
        assert log_lines, f"Log file {mnubergemm_log_file} is empty"

    # len([2025-07-01 15:43:39.392]<space>) = 26
    timestamp_length = 26
    # Collect all lines that start with 'MNUB [I]'
    info_lines = [line.strip() for line in log_lines if line[timestamp_length:].startswith('MNUB [I]')]
    error_lines = [line.strip() for line in log_lines if line[timestamp_length:].startswith('MNUB [E]')]

    # Check required info substrings are present in info lines for head node
    for gpu_id, required_info_msgs in config.get("gpu_required_info", {}).items():
        for required_info in required_info_msgs:
            assert any(required_info in line for line in info_lines), f"Required info substring '{required_info}' not found in MNUB [I] lines for head node GPU {gpu_id}"

    # Check required error substrings are present in error lines for head node
    for gpu_id, required_error_msgs in config.get("gpu_required_error", {}).items():
        for required_error in required_error_msgs:
            assert any(required_error in line for line in error_lines), f"Required error substring '{required_error}' not found in MNUB [E] lines for head node GPU {gpu_id}"

    # If test nodes exist, validate their info and error messages
    test_nodes = config.get("test_nodes", [])
    if test_nodes:
        for node in test_nodes:
            # Check required info substrings for test node
            for gpu_id, required_info_msgs in node.get("gpu_required_info", {}).items():
                for required_info in required_info_msgs:
                    assert any(required_info in line for line in info_lines), f"Required info substring '{required_info}' not found in MNUB [I] lines for test node {node['ip']} GPU {gpu_id}"

            # Check required error substrings for test node
            for gpu_id, required_error_msgs in node.get("gpu_required_error", {}).items():
                for required_error in required_error_msgs:
                    assert any(required_error in line for line in error_lines), f"Required error substring '{required_error}' not found in MNUB [E] lines for test node {node['ip']} GPU {gpu_id}"


# -----------------------------------------------------
def helper_print_stderr_output():
    """
    Helper to print stderr output
    """
    logger.info("Printing stderr output")
    with open(test_utils.get_stderr_output_file_path(), 'r') as f:
        logger.info(f.read())

# ------------------------------------------------------
def helper_run_and_validate_mndiag(handle, config, success=True):
    """
    Helper to run and validate mndiag
    """
    # Run mndiag
    response, error_entities, driver_versions = helper_setup_and_run_mndiag(handle, config)

    # Validate the response
    helper_validate_response(response=response, config=config, success=success, error_entities=error_entities, driver_versions=driver_versions)
    helper_validate_error_output(response=response, validation_config=config)
    helper_check_generated_log_file(config, test_utils.get_mnubergemm_log_file_path())

# ------------------------------------------------------
def helper_setup_headnode_message(config, messages, entities):
    """
    Helper to setup headnode message
    """
    headnode_entities = set()

    config["hostList"] = ["localhost"] # Head node ip
    config["gpu_required_info"] = messages["head_node"]["gpu_required_info"]
    config["gpu_required_error"] = messages["head_node"]["gpu_required_error"]
    headnode_entities.update(config["gpu_required_info"].keys())
    headnode_entities.update(config["gpu_required_error"].keys())
    config["expected_number_of_entities"] = len(headnode_entities)
    for node in config["test_nodes"]:
        config["expected_number_of_entities"] += node["expected_number_of_entities"] 

    errors = 0
    for node in config["test_nodes"]:
        errors += len(node["gpu_required_error"].keys())
    config["expected_num_errors"] = len(config["gpu_required_error"].keys()) + errors

# ------------------------------------------------------
def helper_setup_testnode_message(config, messages, entities):
    """
    Helper to setup testnode message
    """
    for node in config["test_nodes"]:
        node["gpu_required_info"] = messages["test_nodes"]["gpu_required_info"]
        node["gpu_required_error"] = messages["test_nodes"]["gpu_required_error"]
        entities.update(node["gpu_required_info"].keys())
        entities.update(node["gpu_required_error"].keys())
        node["expected_number_of_entities"] = len(entities)

    for node in config["test_nodes"]:
        node["expected_num_errors"] = len(node["gpu_required_error"].keys())
    
def helper_setup_multinode_config(config):
    """
    Helper to setup multinode config
    """
    # Load test node config
    test_config_path = os.getcwd() + "/dcgm_mndiag_test_config.json"
    try:
        with open(test_config_path, 'r') as f:
            test_config = json.load(f)
    except (FileNotFoundError, PermissionError, OSError):
        test_utils.skip_test(f"Test config file {test_config_path} is unreadable or does not exist")

    for node in test_config["test_nodes"]:
        if len(node["ip"]) == 0 or len(node["sku"]) == 0 or len(node["hostname"]) == 0 or len(node["hostengine_path"]) == 0:
            test_utils.skip_test("Test node config is missing required fields")

    # Setup test node
    config["test_nodes"] = test_config["test_nodes"]

# ------------------------------------------------------
@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120,  heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_basic_success(handle, gpuIds):
    """Test basic successful mndiag execution with mock mnubergemm"""
    logger.info("Starting basic mndiag success test")
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    # Configure mock
    config = {
        "hostList": ["localhost"],
        "gpu_required_info": {
            0: [
                "INFO  hosthash=12345 B0I=56789",
                "OPENMPI/NCCL GEMM PULSE",
                "T:N FINISHED",
                "GLOBAL PERF G : 2265061.5 GFlops stddev 1138556.6",
                "GLOBAL PERF N : 160.54855 GB/s stddev 19.520765",
                "RANK PERF G : RANK 0 : 2183234.5 GFlops stddev 1091777.5",
                "RANK PERF G : RANK 1 : 2356335.5 GFlops stddev 1179427.5",
                "RANK PERF N : RANK 0 : 160.81259 GB/s stddev 19.73412",
                "RANK PERF N : RANK 1 : 160.74977 GB/s stddev 19.816277",
            ],
            1: [
                "INFO  hosthash=12345 B0I=56789",
                "OPENMPI/NCCL GEMM PULSE",
                "T:N FINISHED",
                "GLOBAL PERF N : 160.54855 GB/s stddev 19.520765",
                "RANK PERF G : RANK 0 : 2183234.5 GFlops stddev 1091777.5",
                "RANK PERF G : RANK 1 : 2356335.5 GFlops stddev 1179427.5",
                "RANK PERF N : RANK 0 : 160.81259 GB/s stddev 19.73412",
                "RANK PERF N : RANK 1 : 160.74977 GB/s stddev 19.816277",
            ]
        },
        "gpu_required_error": {},
        "test_nodes": [],
        "expected_number_of_entities": 2,
    }
    config["expected_num_errors"] = len(config["gpu_required_error"].keys())

    helper_run_and_validate_mndiag(handle, config, success=True)

    logger.info("Basic mndiag success test completed successfully")

@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120,  heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_basic_failure(handle, gpuIds):
    """Test mndiag when mock mnubergemm simulates all error messages"""
    logger.info("Starting mndiag basic failure test")
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    # Configure mock
    config = {
        "hostList": ["localhost"],
        "gpu_required_info": {
            0: [
                "INFO  hosthash=12345 B0I=56789",
            ],
            1: [
                "INFO  hosthash=12345 B0I=56789",
            ],
            2: [
                "INFO  hosthash=12345 B0I=56789",
            ],
            3: [
                "INFO  hosthash=12345 B0I=56789",
            ],
            4: [
                "INFO  hosthash=12345 B0I=56789",
            ],
            5: [
                "INFO  hosthash=12345 B0I=56789",
            ]
        },
        "gpu_required_error": {
            0: [
                "CUDA-capable device(s) is/are busy or unavailable",
            ],
            1: [
                "cuBLAS Error : init failed",
            ],
            2: [
                "NCCL Error : not initialized",
            ],
            3: [
                "IMEX Error : status is DOWN",
            ],
            4: [
                "cudaStreamQuery returned an illegal memory access was encountered"
            ],
            5: [
                "Unexpected abort",
            ]

        },
        "test_nodes": [],
        "expected_number_of_entities": 6,
    }
    config["expected_num_errors"] = len(config["gpu_required_error"].keys())

    helper_run_and_validate_mndiag(handle, config, success=False)

    logger.info("Basic failure mndiag test completed successfully")
    
@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120,  heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_mixed_pass_fail_1(handle, gpuIds):
    """Test mndiag mixed pass/fail scenario 1"""
    
    logger.info("Starting mixed pass/fail mndiag 1")
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    # Configure mock
    logger.info("Scenario 1 - 1 host, 3 gpus, 2 gpus pass, 1 gpu fail")
    config = {
        "hostList": ["localhost"],
        "gpu_required_info": {
            0: [
                "INFO  hosthash=12345 B0I=56789",
                "OPENMPI/NCCL GEMM PULSE",
                "T:N FINISHED"
            ],
            1: [
                "INFO  hosthash=12345 B0I=56789",
                "OPENMPI/NCCL GEMM PULSE",
                "T:N FINISHED"
            ],
            2: [
                "INFO  hosthash=12345 B0I=56789",
            ]
        },
        "gpu_required_error": {
            2: [
                "CUDA-capable device(s) is/are busy or unavailable",
                "cudaStreamQuery returned an illegal memory access was encountered"
            ]
        },
        "test_nodes": [],
        "expected_number_of_entities": 3,
    }
    config["expected_num_errors"] = len(config["gpu_required_error"].keys())

    helper_run_and_validate_mndiag(handle, config, success=False)
    
    logger.info("Mixed pass/fail 1 mndiag test completed successfully")

@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120,  heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_mixed_pass_fail_2(handle, gpuIds):
    """Test mndiag mixed pass/fail scenario 2"""

    logger.info("Starting mixed pass/fail mndiag 2")
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    # Configure mock
    logger.info("Scenario 1 - 1 host, 3 gpus, 2 gpus pass, 1 gpu fail")
    config = {
        "hostList": ["localhost"],
        "gpu_required_info": {
            0: [
                "INFO  hosthash=12345 B0I=56789",
                "OPENMPI/NCCL GEMM PULSE",
                "T:N FINISHED"
            ],
            1: [
                "INFO  hosthash=12345 B0I=56789",
            ],
            2: [
                "INFO  hosthash=12345 B0I=56789",
            ]
        },
        "gpu_required_error": {
            1: [
                "CUDA-capable device(s) is/are busy or unavailable",
            ],
            2: [
                "cudaStreamQuery returned an illegal memory access was encountered",
            ]
        },
        "test_nodes": [],
        "expected_number_of_entities": 3,
    }
    config["expected_num_errors"] = len(config["gpu_required_error"].keys())

    helper_run_and_validate_mndiag(handle, config, success=False)

    logger.info("Mixed pass/fail mndiag test completed successfully")

# ------------------------------------------------------
# Multinode tests
@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120, heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_multinode_basic_success(handle, gpuIds):
    """Test basic successful mndiag execution with mock mnubergemm on multiple nodes - pass on head node, pass on test nodes"""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)

    messages = {
        "head_node": {
            "gpu_required_info": {
                0: [
                    "OPENMPI/NCCL GEMM PULSE",
                ]
            },
            "gpu_required_error": {}
        },
        "test_nodes": {
            "gpu_required_info": {
                0: [
                        "INFO  hosthash=12345 B0I=56789",
                        "OPENMPI/NCCL GEMM PULSE",
                        "T:N FINISHED",
                        "GLOBAL PERF G : 2265061.5 GFlops stddev 1138556.6",
                        "GLOBAL PERF N : 160.54855 GB/s stddev 19.520765",
                        "RANK PERF G : RANK 0 : 2183234.5 GFlops stddev 1091777.5",
                        "RANK PERF G : RANK 1 : 2356335.5 GFlops stddev 1179427.5",
                        "RANK PERF N : RANK 0 : 160.81259 GB/s stddev 19.73412",
                        "RANK PERF N : RANK 1 : 160.74977 GB/s stddev 19.816277",
                    ],
                1: [
                    "INFO  hosthash=12345 B0I=56789",
                    "OPENMPI/NCCL GEMM PULSE",
                    "T:N FINISHED",
                    "GLOBAL PERF N : 160.54855 GB/s stddev 19.520765",
                    "RANK PERF G : RANK 0 : 2183234.5 GFlops stddev 1091777.5",
                    "RANK PERF G : RANK 1 : 2356335.5 GFlops stddev 1179427.5",
                    "RANK PERF N : RANK 0 : 160.81259 GB/s stddev 19.73412",
                    "RANK PERF N : RANK 1 : 160.74977 GB/s stddev 19.816277",
                ]
            },
            "gpu_required_error": {}
        }
    }
    
    helper_setup_testnode_message(config, messages, entities)
    helper_setup_headnode_message(config, messages, entities)
    helper_run_and_validate_mndiag(handle, config, success=True)


@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120, heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_multinode_basic_failure_1(handle, gpuIds):
    """Test basic failure mndiag execution with mock mnubergemm on multiple nodes, fail on head node 1 gpu, fail on test nodes 6 gpus"""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)

    # Currently configuring all test nodes with same message
    messages = {
        "head_node": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                ]
            },
            "gpu_required_error": {
                0: [
                    "cuBLAS Error : init failed",
                ],
            },
        },
        "test_nodes": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                1: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                2: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                3: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                4: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                5: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
            },
            "gpu_required_error": {
                0: [
                    "CUDA-capable device(s) is/are busy or unavailable",
                ],
                1: [
                    "cuBLAS Error : init failed",
                ],
                2: [
                    "NCCL Error : not initialized",
                    "NCCL Error : not initialized2",
                ],
                3: [
                    "IMEX Error : status is DOWN",
                ],
                4: [
                    "cudaStreamQuery returned an illegal memory access was encountered"
                ],
                5: [
                    "Unexpected abort",
                ]  
            }
        }
    }
    
    helper_setup_testnode_message(config, messages, entities)
    helper_setup_headnode_message(config, messages, entities)
    helper_run_and_validate_mndiag(handle, config, success=False)

@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120, heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env()})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_multinode_basic_failure_2(handle, gpuIds):
    """Test basic failure mndiag execution with mock mnubergemm on multiple node, fail on head node 1 gpu, fail on test nodes 1 gpu, same error"""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)

    # Currently configuring all test nodes with same message
    messages = {
        "head_node": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                ]
            },
            "gpu_required_error": {
                0: [
                    "CUDA-capable device(s) is/are busy or unavailable",
                ],
            },
        },
        "test_nodes": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
            },
            "gpu_required_error": {
                0: [
                    "CUDA-capable device(s) is/are busy or unavailable",
                ],
            }
        }
    }
    
    helper_setup_testnode_message(config, messages, entities)
    helper_setup_headnode_message(config, messages, entities)
    helper_run_and_validate_mndiag(handle, config, success=False)


@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120, heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_multinode_basic_failure_3(handle, gpuIds):
    """Test basic failure mndiag execution with mock mnubergemm on multiple node, fail on head node 2 gpus, fail on test nodes 2 gpus, same error"""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)

    # Currently configuring all test nodes with same message
    messages = {
        "head_node": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                1: [
                    "INFO  hosthash=12345 B0I=56789",
                ]
            },
            "gpu_required_error": {
                0: [
                    "cuBLAS Error : init failed",
                ],
                1: [
                    "cuBLAS Error : init failed",
                ],
            },
        },
        "test_nodes": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                1: [
                    "INFO  hosthash=12345 B0I=56789",
                ]
            },
            "gpu_required_error": {
                0: [
                    "cuBLAS Error : init failed",
                ],
                1: [
                    "cuBLAS Error : init failed",
                ],
            }
        }
    }
    
    helper_setup_testnode_message(config, messages, entities)
    helper_setup_headnode_message(config, messages, entities)
    helper_run_and_validate_mndiag(handle, config, success=False)


@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120, heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_multinode_headnode_pass_testnodes_fail(handle, gpuIds):
    """Test basic successful headnode and failure testnodes mndiag execution with mock mnubergemm on multiple nodes, pass on head node, fail on test nodes 2 gpus, different error"""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")
    config = {}
    entities = set()

    helper_setup_multinode_config(config)

    # Currently configuring all test nodes with same message
    messages = {
        "head_node": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                    "OPENMPI/NCCL GEMM PULSE",
                ]
            },
            "gpu_required_error": {},
        },
        "test_nodes": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                1: [
                    "INFO  hosthash=12345 B0I=56789",
                ]
            },
            "gpu_required_error": {
                0: [
                    "CUDA-capable device(s) is/are busy or unavailable",
                ],
                1: [
                    "cuBLAS Error : init failed",
                ],
            }
        }
    }
    
    helper_setup_testnode_message(config, messages, entities)
    helper_setup_headnode_message(config, messages, entities)
    helper_run_and_validate_mndiag(handle, config, success=False)


@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120, heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_multinode_headnode_fail_testnodes_pass(handle, gpuIds):
    """Test basic failure headnode and successful testnodes mndiag execution with mock mnubergemm on multiple nodes, fail on head node 2 gpus, pass on test nodes"""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)

    # Currently configuring all test nodes with same message
    messages = {
        "head_node": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                1: [
                    "INFO  hosthash=12345 B0I=56789",
                ]
            },
            "gpu_required_error": {
                0: [
                    "cuBLAS Error : init failed",
                ],
                1: [
                    "NCCL Error : not initialized",
                ],
            },
        },
        "test_nodes": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                    "OPENMPI/NCCL GEMM PULSE",
                ]
            },
            "gpu_required_error": {}
        }
    }
    
    helper_setup_testnode_message(config, messages, entities)
    helper_setup_headnode_message(config, messages, entities)
    helper_run_and_validate_mndiag(handle, config, success=False)


@test_utils.run_only_with_gpus_present()
@test_utils.run_if_mpirun_exists()
@test_utils.run_with_standalone_host_engine(120, heEnv={"PATH": test_utils.get_updated_env_path_variable(), "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(), "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(), "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_current_skus_env(), "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1"})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_all_same_sku_gpus()
def test_dcgm_mndiag_multinode_mixed_pass_fail(handle, gpuIds):
    """Test mixed pass/fail mndiag execution with mock mnubergemm on multiple nodes, pass on head node, pass on test nodes 1 gpu and fail on 2 gpus"""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)

    # Currently configuring all test nodes with same message
    messages = {
        "head_node": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                    "OPENMPI/NCCL GEMM PULSE",
                ]
            },
            "gpu_required_error": {},
        },
        "test_nodes": {
            "gpu_required_info": {
                0: [
                    "INFO  hosthash=12345 B0I=56789",
                    "OPENMPI/NCCL GEMM PULSE",
                ],
                1: [
                    "INFO  hosthash=12345 B0I=56789",
                ],
                2: [
                    "INFO  hosthash=12345 B0I=56789",
                ]

            },
            "gpu_required_error": {
                1: [
                    "cuBLAS Error : init failed",
                ],
                2: [
                    "NCCL Error : not initialized",
                ],
            }
        }
    }
    
    helper_setup_testnode_message(config, messages, entities)
    helper_setup_headnode_message(config, messages, entities)
    helper_run_and_validate_mndiag(handle, config, success=False)