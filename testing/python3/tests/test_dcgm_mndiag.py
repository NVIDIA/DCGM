# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
import test_helpers.helper_dcgm_mndiag
import logger
import test_utils
import DcgmMnDiag
from MockMnubergemmController import MockMnubergemmController
from MockNvloomController import MockNvloomController
from MockMnnvbandwidthController import MockMnnvbandwidthController

import os
from ctypes import *
import json

# ------------------------------------------------------


def helper_setup_and_run_mndiag(handle, config, run_as_non_root=False):
    """
    Helper to setup and run multinode mndiag
    """
    try:
        mock_controller = MockMnubergemmController(
            test_utils.get_mock_mnubergemm_path(),
            test_utils.get_mock_nvidia_smi_path())

        if len(config["test_nodes"]) > 0:
            mock_controller.setup_multinode_mock_environment(config)
            hostList = config["hostList"] + [node["ip"]
                                             for node in config["test_nodes"]]
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
        mndiag = DcgmMnDiag.DcgmMnDiag(
            hostList=hostList,
            testName=testName,
            parameters=testParams,
            handle=handle)
        if run_as_non_root:
            test_utils.RunAsNonRoot.is_supported(skip_if_not_supported=True)
            with test_utils.RunAsNonRoot(reload_driver=False):
                response = mndiag.Execute(handle)
        else:
            response = mndiag.Execute(handle)

    except Exception:
        helper_print_stderr_output()
        raise

    finally:
        mock_controller.cleanup_multinode_mock_environment(config)

    return response, error_entities, driver_versions

# ------------------------------------------------------


def helper_validate_response(
        response,
        config,
        testName="MNUBERGEMM",
        success=None,
        error_entities={},
        driver_versions=[]):
    """
    Helper to validate mndiag response structure
    """
    hostList = config["hostList"] + [node["ip"]
                                     for node in config["test_nodes"]]

    # Cap expected count at the response struct's hardcoded capacity, i.e. DCGM_MN_DIAG_RESPONSE_ERRORS_MAX
    # C++ silently truncates errors beyond this, the assertion below should reflect that
    expected_num_errors = config.get("expected_num_errors", 0)
    expected_num_errors = min(
        expected_num_errors, dcgm_structs.DCGM_MN_DIAG_RESPONSE_ERRORS_MAX)

    assert response is not None, "Response should not be None"
    assert response.version == dcgm_structs.dcgmMnDiagResponse_version2, (
        f"Got response version {response.version}, "
        f"expected {dcgm_structs.dcgmMnDiagResponse_version2}")

    # Validate host information
    assert response.numHosts == len(hostList), (
        f"numHosts {response.numHosts} should match hostList length "
        f"{len(hostList)}")
    for i in range(response.numHosts):
        host = response.hosts[i]
        assert len(host.hostname) > 0, f"Host {i} has empty hostname"
        assert host.hostname in hostList, (
            f"Host {i} hostname {host.hostname} not in hostList")

    # Validate entity results
    assert response.numEntities == response.numResults, (
        f"numEntities {response.numEntities} should match numResults "
        f"{response.numResults}")
    assert response.numEntities == config["expected_number_of_entities"], (
        f"numEntities {response.numEntities} should match "
        f"expected_number_of_entities "
        f"{config['expected_number_of_entities']}")
    for i in range(response.numResults):
        result = response.results[i]
        assert result.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU, (
            f"Entity {i} has invalid entityGroupId. "
            f"Got {result.entity.entityGroupId}, "
            f"expected {dcgm_fields.DCGM_FE_GPU}")
        assert result.testId == 0, (
            f"Entity {i} has invalid testId. Got {result.testId}, "
            f"expected 0")

        # Get hostname for this result
        assert result.hostId < response.numHosts, (
            f"Entity {i} has invalid hostId. Got {result.hostId}, "
            f"expected < {response.numHosts}")
        hostname = response.hosts[result.hostId].hostname
        assert hostname in hostList, f"Entity {i} has hostname {hostname} not in hostList"

        # Validate error entities
        if len(error_entities.keys()) > 0:
            entity_key = f"{hostname}:{result.entity.entityId}"
            if entity_key in error_entities.keys():
                expected = dcgm_structs.DCGM_DIAG_RESULT_FAIL
                assert result.result == expected, (
                    f"Entity {i} with key {entity_key} has invalid "
                    f"result: {result.result}. Expected {expected}")
            else:
                expected = dcgm_structs.DCGM_DIAG_RESULT_PASS
                assert result.result == expected, (
                    f"Entity {i} with key {entity_key} has invalid "
                    f"result: {result.result}. Expected {expected}")

    # Validate test results
    assert response.numTests == 1, "numTests should be 1"
    for i in range(response.numTests):
        test = response.tests[i]
        assert test.name == testName, (
            f"Test name mismatch: expected {testName}, got {test.name}")
        if success:
            expected = dcgm_structs.DCGM_DIAG_RESULT_PASS
            assert test.result == expected, (
                f"Test result invalid: {test.result}. Expected {expected}")
        else:
            expected = dcgm_structs.DCGM_DIAG_RESULT_FAIL
            assert test.result == expected, (
                f"Test result invalid: {test.result}. Expected {expected}")

    # Validate error count matches expected errors
    assert response.numErrors == expected_num_errors, (
        f"Expected {expected_num_errors} errors, got {response.numErrors}")
    if not success:
        for i in range(response.numErrors):
            error = response.errors[i]
            assert error.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU, (
                f"Error {i} has invalid entityGroupId. Got "
                f"{error.entity.entityGroupId}, expected "
                f"{dcgm_fields.DCGM_FE_GPU}")
            assert error.testId == 0, (
                f"Error {i} has invalid testId. Got {error.testId}, "
                f"expected 0")
            assert error.code == dcgm_errors.DCGM_FR_UNKNOWN, (
                f"Error {i} has invalid error code. Got {error.code}, "
                f"expected {dcgm_errors.DCGM_FR_UNKNOWN}")
            assert error.category == dcgm_errors.DCGM_FR_EC_HARDWARE_OTHER, (
                f"Error {i} has invalid category. Got {error.category}, "
                f"expected {dcgm_errors.DCGM_FR_EC_HARDWARE_OTHER}")
            assert error.severity == dcgm_errors.DCGM_ERROR_TRIAGE, (
                f"Error {i} has invalid severity. Got {error.severity}, "
                f"expected {dcgm_errors.DCGM_ERROR_TRIAGE}")
            assert len(error.msg) > 0, f"Error {i} has empty message"

            all_error_msgs = [m.strip()
                              for m in error.msg.split(';') if m.strip()]
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
        assert host.numEntities > 0, (
            f"Host {i} has no entities. Got {host.numEntities}, "
            f"expected > 0")
        assert host.dcgmVersion == dcgm_version, (
            f"Host {i} dcgmVersion mismatch: got '{host.dcgmVersion}', "
            f"expected '{dcgm_version}'")
        injection_driver_version = test_helpers.helper_dcgm_mndiag.get_injection_values()[
            "INJECTION_DRIVER_VERSION"]
        assert host.driverVersion == injection_driver_version, (
            f"Host {i} driverVersion '{host.driverVersion}' not equal to "
            f"expected driver version {injection_driver_version}")

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
        error_msgs.extend(
            [msg.strip() for msg in response.errors[i].msg.split(';') if msg.strip()])

    for gpu_id, required_error_msgs in validation_config.get(
            "gpu_required_error", {}).items():
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
    assert os.path.exists(
        mnubergemm_log_file), f"Log file {mnubergemm_log_file} does not exist"

    # Open the logs and check if its not empty
    with open(mnubergemm_log_file, 'r') as f:
        log_lines = f.readlines()
        assert log_lines, f"Log file {mnubergemm_log_file} is empty"

    info_lines = [line.strip() for line in log_lines if 'MNUB [I]' in line]
    error_lines = [line.strip() for line in log_lines if 'MNUB [E]' in line]

    # Check required info substrings are present in info lines for head node
    for gpu_id, required_info_msgs in config.get(
            "gpu_required_info", {}).items():
        for required_info in required_info_msgs:
            assert any(
                required_info in line for line in info_lines), f"Required info substring '{required_info}' not found in MNUB [I] lines for head node GPU {gpu_id}"

    # Check required error substrings are present in error lines for head node
    for gpu_id, required_error_msgs in config.get(
            "gpu_required_error", {}).items():
        for required_error in required_error_msgs:
            assert any(
                required_error in line for line in error_lines), f"Required error substring '{required_error}' not found in MNUB [E] lines for head node GPU {gpu_id}"

    # If test nodes exist, validate their info and error messages
    test_nodes = config.get("test_nodes", [])
    if test_nodes:
        for node in test_nodes:
            # Check required info substrings for test node
            for gpu_id, required_info_msgs in node.get(
                    "gpu_required_info", {}).items():
                for required_info in required_info_msgs:
                    assert any(required_info in line for line in info_lines), (
                        f"Required info substring '{required_info}' not "
                        f"found in MNUB [I] lines for test node "
                        f"{node['ip']} GPU {gpu_id}")

            # Check required error substrings for test node
            for gpu_id, required_error_msgs in node.get(
                    "gpu_required_error", {}).items():
                for required_error in required_error_msgs:
                    assert any(required_error in line for line in error_lines), (
                        f"Required error substring '{required_error}' not "
                        f"found in MNUB [E] lines for test node "
                        f"{node['ip']} GPU {gpu_id}")


# -----------------------------------------------------
def helper_print_stderr_output():
    """
    Helper to print stderr output
    """
    logger.info("Printing stderr output")
    with open(test_utils.get_stderr_output_file_path(), 'r') as f:
        logger.info(f.read())

# ------------------------------------------------------


def helper_run_and_validate_mndiag(
        handle, config, success=True, run_as_non_root=False):
    """
    Helper to run and validate mndiag
    """
    # Run mndiag
    response, error_entities, driver_versions = helper_setup_and_run_mndiag(
        handle, config, run_as_non_root=run_as_non_root)

    # Validate the response
    helper_validate_response(
        response=response,
        config=config,
        success=success,
        error_entities=error_entities,
        driver_versions=driver_versions)
    helper_validate_error_output(response=response, validation_config=config)
    helper_check_generated_log_file(
        config, test_utils.get_mnubergemm_log_file_path())

# ------------------------------------------------------


def helper_setup_headnode_message(config, messages, entities):
    """
    Helper to setup headnode message
    """
    headnode_entities = set()

    config["hostList"] = ["localhost"]  # Head node ip
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
    config["expected_num_errors"] = len(
        config["gpu_required_error"].keys()) + errors

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
        test_utils.skip_test(
            f"Test config file {test_config_path} is unreadable or does not exist")

    for node in test_config["test_nodes"]:
        if len(
            node.get("ip", "")) == 0 or len(
            node.get("sku", "")) == 0 or len(
            node.get("username", "")) == 0 or len(
                node.get("hostengine_path", "")) == 0 or len(
                node.get("nvml_injection_yaml_file_path", "")) == 0:
            test_utils.skip_test("Test node config is missing required fields")

    # Setup test node
    config["test_nodes"] = test_config["test_nodes"]

# ------------------------------------------------------


def helper_extract_mnubergemm_path_from_logs(
        log_path, hostname=None, ip=None, is_remote=False):
    """
    Helper method to extract mnubergemm path from log files.
    """
    custom_path_log_pattern = "Inferred custom mnubergemm path:"
    default_path_log_pattern = "Inferred default mnubergemm path:"
    grep_pattern_remote = "Setting expected test binary path to:"

    def extract_path_from_line(line):
        """Extract path from a log line that contains mnubergemm path"""
        if custom_path_log_pattern in line:
            start_idx = line.find(custom_path_log_pattern) + \
                len(custom_path_log_pattern)
        elif default_path_log_pattern in line:
            start_idx = line.find(default_path_log_pattern) + \
                len(default_path_log_pattern)
        elif grep_pattern_remote in line:
            start_idx = line.find(grep_pattern_remote) + \
                len(grep_pattern_remote)
        else:
            return None

        # For remote node logs, prioritize "on node:" over square bracket
        if grep_pattern_remote in line:
            # Look for " on node:" first
            end_idx = line.find(" on node:", start_idx)
            if end_idx == -1:
                # If no " on node:" found, look for square bracket
                end_idx = line.find('[', start_idx)
        else:
            # For head node logs, look for square bracket first
            end_idx = line.find('[', start_idx)
            if end_idx == -1:
                # If no square bracket found, look for " on node:" (fallback)
                end_idx = line.find(" on node:", start_idx)

        if end_idx == -1:
            # If no end marker found, take the rest of the line
            path = line[start_idx:].strip()
        else:
            path = line[start_idx:end_idx].strip()

        return path

    if is_remote:
        ssh_cmd = f"ssh {hostname}@{ip}"
        grep_cmd = f"{ssh_cmd} 'grep -E \"{grep_pattern_remote}\" {log_path} | tail -1'"

        try:
            result = os.popen(grep_cmd).read().strip()
            if result:
                return extract_path_from_line(result)
            else:
                logger.warning(
                    f"Could not get mnubergemm path from remote node {hostname}@{ip}")
                return None
        except Exception as e:
            logger.warning(
                f"Error getting mnubergemm path from remote node {hostname}@{ip}: {e}")
            return None
    else:
        # For head node, read the log file directly
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if custom_path_log_pattern in line or default_path_log_pattern in line:
                        return extract_path_from_line(line)
        except FileNotFoundError:
            logger.warning(f"Could not find {log_path} on head node")
        except Exception as e:
            logger.warning(f"Error reading head node log file: {e}")

        return None


def helper_validate_mnubergemm_path(
        config,
        head_node_log_path,
        remote_log_path):
    """
    Helper to validate mnubergemm path
    """
    # Get the mnubergemm path from head node logs
    head_node_path = helper_extract_mnubergemm_path_from_logs(
        head_node_log_path, is_remote=False)
    if not head_node_path:
        assert False, "Could not find mnubergemm path in head node logs"

    logger.info(f"Head node mnubergemm path: {head_node_path}")

    # Get mnubergemm paths from remote nodes
    remote_paths = []
    for test_node in config.get("test_nodes", []):
        username = test_node.get("username")
        ip = test_node.get("ip")
        if not username or not ip:
            logger.warning("Test node missing username or IP field")
            continue

        remote_path = helper_extract_mnubergemm_path_from_logs(
            remote_log_path, username, ip, is_remote=True)
        if remote_path:
            remote_paths.append((username, ip, remote_path))
            log_fn = logger.info if os.environ.get(
                "DCGM_MNDIAG_VERBOSE", "0") == "1" else logger.debug
            log_fn(
                f"Remote node {username}@{ip} mnubergemm path: {remote_path}")
        else:
            remote_paths.append((username, ip, None))

    logger.info(
        f"Collected mnubergemm paths from {len(remote_paths)} remote node(s)")

    # Compare paths - all should be the same
    all_paths_match = True
    for username, ip, remote_path in remote_paths:
        if remote_path is None:
            logger.error(
                f"Could not retrieve path from remote node {username}@{ip}")
            all_paths_match = False
        elif remote_path != head_node_path:
            logger.error(
                f"Path mismatch: head node has '{head_node_path}', remote node {username}@{ip} has '{remote_path}'")
            all_paths_match = False

    if all_paths_match:
        logger.info(
            "All mnubergemm paths match across head node and remote nodes")
    else:
        assert False, "Mnubergemm paths should be consistent across all nodes"

# ------------------------------------------------------


def helper_dcgm_mndiag_basic_success(
        handle, gpuIds, run_as_non_root=False):
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

    helper_run_and_validate_mndiag(
        handle,
        config,
        success=True,
        run_as_non_root=run_as_non_root)

    logger.info("Basic mndiag success test completed successfully")


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_basic_success(handle, gpuIds):
    """Verify mnubergemm all-pass on localhost with injected GPUs."""
    helper_dcgm_mndiag_basic_success(handle, gpuIds)


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_basic_success_non_root_caller(handle, gpuIds):
    """Test a non-root caller using MPI files created by the root hostengine."""
    helper_dcgm_mndiag_basic_success(
        handle,
        gpuIds,
        run_as_non_root=True)


def helper_dcgm_mndiag_basic_failure(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_basic_failure(handle, gpuIds):
    """Verify mnubergemm failure reporting on localhost with injected GPUs."""
    helper_dcgm_mndiag_basic_failure(handle, gpuIds)


def helper_dcgm_mndiag_mixed_pass_fail_1(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mixed_pass_fail_1(handle, gpuIds):
    """Verify mnubergemm mixed pass/fail (variant 1) on localhost."""
    helper_dcgm_mndiag_mixed_pass_fail_1(handle, gpuIds)


def helper_dcgm_mndiag_mixed_pass_fail_2(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mixed_pass_fail_2(handle, gpuIds):
    """Verify mnubergemm mixed pass/fail (variant 2) on localhost."""
    helper_dcgm_mndiag_mixed_pass_fail_2(handle, gpuIds)


# ------------------------------------------------------
# Multinode tests


def helper_dcgm_mndiag_multinode_basic_success(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_multinode_basic_success(handle, gpuIds):
    """Verify mnubergemm all-pass across multiple nodes."""
    helper_dcgm_mndiag_multinode_basic_success(handle, gpuIds)


def helper_dcgm_mndiag_multinode_basic_failure_1(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_multinode_basic_failure_1(handle, gpuIds):
    """Verify mnubergemm multinode failure: head and test nodes fail."""
    helper_dcgm_mndiag_multinode_basic_failure_1(handle, gpuIds)


def helper_dcgm_mndiag_multinode_basic_failure_2(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_multinode_basic_failure_2(handle, gpuIds):
    """Verify mnubergemm multinode failure: head fails, test nodes pass."""
    helper_dcgm_mndiag_multinode_basic_failure_2(handle, gpuIds)


def helper_dcgm_mndiag_multinode_basic_failure_3(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_multinode_basic_failure_3(handle, gpuIds):
    """Verify mnubergemm multinode failure: head passes, test nodes fail."""
    helper_dcgm_mndiag_multinode_basic_failure_3(handle, gpuIds)


def helper_dcgm_mndiag_multinode_headnode_pass_testnodes_fail(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_multinode_headnode_pass_testnodes_fail(handle, gpuIds):
    """Verify mnubergemm multinode: head passes, all test nodes fail."""
    helper_dcgm_mndiag_multinode_headnode_pass_testnodes_fail(handle, gpuIds)


def helper_dcgm_mndiag_multinode_headnode_fail_testnodes_pass(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_multinode_headnode_fail_testnodes_pass(handle, gpuIds):
    """Verify mnubergemm multinode: head fails, all test nodes pass."""
    helper_dcgm_mndiag_multinode_headnode_fail_testnodes_pass(handle, gpuIds)


def helper_dcgm_mndiag_multinode_mixed_pass_fail(handle, gpuIds):
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


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_multinode_mixed_pass_fail(handle, gpuIds):
    """Verify mnubergemm multinode mixed pass/fail across nodes."""
    helper_dcgm_mndiag_multinode_mixed_pass_fail(handle, gpuIds)


def helper_dcgm_mndiag_multinode_validate_mnubergemm_path(handle, gpuIds, hostengineApp=None):
    """Test mndiag validate mnubergemm path"""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    # When --force-logging is active, the host engine writes its trace to a
    # per-app file (app_NNN_dcgm_trace.log) instead of dcgm_trace.log.
    if hostengineApp and hostengineApp.dcgm_trace_fname:
        head_node_log_path = hostengineApp.dcgm_trace_fname
    else:
        head_node_log_path = os.path.join(
            os.getcwd(), "_out_runLogs/dcgm_trace.log")
    remote_log_path = "/var/log/nv-hostengine.log"

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
                    "INFO  hosthash=112233 B0I=556677",
                    "OPENMPI/NCCL GEMM PULSE",
                ]

            },
            "gpu_required_error": {}
        }
    }

    helper_setup_testnode_message(config, messages, entities)
    helper_setup_headnode_message(config, messages, entities)
    helper_run_and_validate_mndiag(handle, config, success=True)
    helper_validate_mnubergemm_path(
        config, head_node_log_path, remote_log_path)


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heArgs=["--log-level", "debug"], heEnv=test_helpers.helper_dcgm_mndiag.HE_Env, passAppAsArg=True)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_multinode_validate_mnubergemm_path(handle, gpuIds, hostengineApp=None):
    """Verify mnubergemm binary path consistency across all nodes."""
    helper_dcgm_mndiag_multinode_validate_mnubergemm_path(
        handle, gpuIds, hostengineApp)


# ------------------------------------------------------
# Backwards compatibility tests


def helper_dcgm_mndiag_backwards_compatibility_v1(handle, gpuIds):
    """Test that v1 structs are still accepted and produce valid results."""
    logger.info("Starting mndiag backwards compatibility v1 test")
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    mock_controller = MockMnubergemmController(
        test_utils.get_mock_mnubergemm_path(),
        test_utils.get_mock_nvidia_smi_path())

    config = {
        "hostList": ["localhost"],
        "gpu_required_info": {
            0: [
                "INFO  hosthash=12345 B0I=56789",
                "OPENMPI/NCCL GEMM PULSE",
                "T:N FINISHED",
            ],
            1: [
                "INFO  hosthash=12345 B0I=56789",
                "OPENMPI/NCCL GEMM PULSE",
                "T:N FINISHED",
            ]
        },
        "gpu_required_error": {},
        "test_nodes": [],
        "expected_number_of_entities": 2,
    }
    config["expected_num_errors"] = 0

    try:
        mock_controller.setup_localhost_mock_environment(config)

        mndiag = DcgmMnDiag.DcgmMnDiag(
            hostList=["localhost"],
            testName="mnubergemm",
            parameters=["mnubergemm.time_to_run=1"],
            handle=handle,
            version=dcgm_structs.dcgmRunMnDiag_version1)
        response = mndiag.Execute(handle)
    finally:
        mock_controller.cleanup_multinode_mock_environment(config)

    assert response is not None, "Response should not be None"
    assert response.version == dcgm_structs.dcgmMnDiagResponse_version1, (
        f"Got response version {response.version}, "
        f"expected {dcgm_structs.dcgmMnDiagResponse_version1}")
    assert response.numHosts == 1, f"numHosts {response.numHosts} should be 1"
    assert response.numTests == 1, f"numTests {response.numTests} should be 1"
    assert response.numEntities == 2, (
        f"numEntities {response.numEntities} should be 2")

    logger.info("Backwards compatibility v1 test completed successfully")


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_backwards_compatibility_v1(handle, gpuIds):
    """Verify backward compatibility with the v1 mndiag response struct."""
    helper_dcgm_mndiag_backwards_compatibility_v1(handle, gpuIds)


# =========================================================================
# NVLOOM E2E TESTS
# =========================================================================


def helper_setup_and_run_nvloom(handle, config):
    """Setup mock nvloom_cli environment and run mndiag with testName=nvloom."""
    try:
        mock_controller = MockNvloomController(
            test_utils.get_mock_nvloom_path(),
            test_utils.get_mock_nvidia_smi_path())

        if len(config.get("test_nodes", [])) > 0:
            mock_controller.setup_multinode_mock_environment(config)
            hostList = config["hostList"] + [node["ip"]
                                             for node in config["test_nodes"]]
        else:
            mock_controller.setup_localhost_mock_environment(config)
            hostList = config["hostList"]

        error_entities = mock_controller.get_error_entities()
        driver_versions = mock_controller.get_driver_versions(config)

        mndiag = DcgmMnDiag.DcgmMnDiag(
            hostList=hostList,
            testName="nvloom",
            parameters=["nvloom.time_to_run=1"],
            handle=handle)
        response = mndiag.Execute(handle)

    except Exception:
        helper_print_stderr_output()
        raise

    finally:
        if config.get("test_nodes"):
            mock_controller.cleanup_multinode_mock_environment(config)
        else:
            mock_controller.cleanup_mock_environment()

    return response, error_entities, driver_versions


def helper_validate_nvloom_response(response, config, success=True,
                                    error_entities=None):
    """Validate nvloom-format response structure."""
    if error_entities is None:
        error_entities = {}

    assert response is not None, "Response should not be None"
    assert response.version == dcgm_structs.dcgmMnDiagResponse_version2

    assert response.numTests == 1, f"numTests {response.numTests} should be 1"
    assert response.tests[0].name == "NVLOOM", (
        f"Test name should be NVLOOM, got {response.tests[0].name}")

    if success:
        error_detail = ""
        if response.numErrors > 0:
            error_msgs = [response.errors[i].msg
                          for i in range(response.numErrors)]
            error_detail = (f"; numErrors={response.numErrors}, "
                            f"errors={error_msgs}")
        assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_PASS, (
            f"Expected PASS, got {response.tests[0].result}"
            f"{error_detail}")
    else:
        assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, (
            f"Expected FAIL, got {response.tests[0].result}")

    assert response.numHosts > 0, "Should have at least one host"
    assert response.numResults > 0, "Should have at least one entity result"
    assert response.numEntities == response.numResults

    for i in range(response.numResults):
        result = response.results[i]
        assert result.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU
        assert result.testId == 0
        assert result.hostId < response.numHosts

        if len(error_entities.keys()) > 0:
            hostname = response.hosts[result.hostId].hostname
            entity_key = f"{hostname}:{result.entity.entityId}"
            if entity_key in error_entities.keys():
                assert result.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, (
                    f"Entity {i} with key {entity_key} should be FAIL, "
                    f"got {result.result}")
            else:
                assert result.result == dcgm_structs.DCGM_DIAG_RESULT_PASS, (
                    f"Entity {i} with key {entity_key} should be PASS, "
                    f"got {result.result}")

    if not success and error_entities:
        for i in range(response.numErrors):
            error = response.errors[i]
            assert error.code == dcgm_errors.DCGM_FR_UNKNOWN
            assert len(error.msg) > 0
            all_error_msgs = [m.strip()
                              for m in error.msg.split(';') if m.strip()]
            hostname = response.hosts[error.hostId].hostname
            entity_key = f"{hostname}:{error.entity.entityId}"
            for expected_msg in error_entities[entity_key]:
                assert any(expected_msg in actual for actual in all_error_msgs), (
                    f"Expected error message '{expected_msg}' not found in "
                    f"actual error messages {all_error_msgs} "
                    f"for entity {entity_key} (error index {i})"
                )


def helper_dcgm_nvloom_basic_success(handle, gpuIds):
    """Test nvloom mndiag when mock nvloom_cli simulates all-pass output."""
    logger.info("Starting nvloom basic success test")

    config = {
        "hostList": ["localhost"],
        "gpus_per_host": len(gpuIds),
        "gpu_required_info": {},
        "gpu_required_error": {},
        "test_suites": ["gpu_to_rack_device_to_device_bidir_read_ce"],
    }

    response, error_entities, _ = helper_setup_and_run_nvloom(handle, config)
    helper_validate_nvloom_response(response, config, success=True)

    logger.info("Nvloom basic success test completed")


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_nvloom_basic_success(handle, gpuIds):
    """Verify nvloom all-pass on localhost with injected GPUs."""
    helper_dcgm_nvloom_basic_success(handle, gpuIds)


def helper_dcgm_nvloom_basic_failure(handle, gpuIds):
    """Test nvloom mndiag when mock nvloom_cli simulates errors."""
    logger.info("Starting nvloom basic failure test")

    config = {
        "hostList": ["localhost"],
        "gpus_per_host": len(gpuIds),
        "gpu_required_info": {},
        "gpu_required_error": {
            0: [
                "[CUDA_ERROR_UNKNOWN] cuMulticastBindMem failed",
            ],
            1: [
                "[CUDA_ERROR_UNKNOWN] cuMulticastBindMem failed",
            ],
        },
        "test_suites": ["multicast_all_to_all_ce"],
    }

    response, error_entities, _ = helper_setup_and_run_nvloom(handle, config)
    helper_validate_nvloom_response(
        response, config, success=False, error_entities=error_entities)

    logger.info("Nvloom basic failure test completed")


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_nvloom_basic_failure(handle, gpuIds):
    """Verify nvloom failure reporting on localhost with injected GPUs."""
    helper_dcgm_nvloom_basic_failure(handle, gpuIds)


# ============================================================
# Multinode nvloom tests
# ============================================================

def helper_setup_nvloom_testnode_message(config, messages, entities):
    """Setup test node messages for nvloom multinode tests."""
    for node in config["test_nodes"]:
        node["gpu_required_info"] = messages["test_nodes"]["gpu_required_info"]
        node["gpu_required_error"] = messages["test_nodes"]["gpu_required_error"]
        entities.update(node["gpu_required_error"].keys())


def helper_setup_nvloom_headnode_message(config, messages, entities):
    """Setup head node messages for nvloom multinode tests."""
    config["hostList"] = ["localhost"]
    config["gpu_required_info"] = messages["head_node"]["gpu_required_info"]
    config["gpu_required_error"] = messages["head_node"]["gpu_required_error"]

    # For nvloom, entities come from "Process" lines, which are gpus_per_host per host
    gpus_per_host = config.get("gpus_per_host", 4)
    num_hosts = 1 + len(config["test_nodes"])
    config["expected_number_of_entities"] = gpus_per_host * num_hosts

    errors = 0
    for node in config["test_nodes"]:
        errors += len(node["gpu_required_error"].keys())
    config["expected_num_errors"] = len(
        config["gpu_required_error"].keys()) + errors


def helper_run_and_validate_nvloom(handle, config, success=True):
    """Run nvloom mndiag and validate the response."""
    response, error_entities, _ = helper_setup_and_run_nvloom(handle, config)

    hostList = config["hostList"] + [node["ip"]
                                     for node in config.get("test_nodes", [])]

    assert response is not None, "Response should not be None"
    assert response.version == dcgm_structs.dcgmMnDiagResponse_version2

    assert response.numTests == 1, f"numTests {response.numTests} should be 1"
    assert response.tests[0].name == "NVLOOM", (
        f"Test name should be NVLOOM, got {response.tests[0].name}")

    if success:
        error_detail = ""
        if response.numErrors > 0:
            error_msgs = [response.errors[i].msg
                          for i in range(response.numErrors)]
            error_detail = (f"; numErrors={response.numErrors}, "
                            f"errors={error_msgs}")
        assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_PASS, (
            f"Expected PASS, got {response.tests[0].result}"
            f"{error_detail}")
    else:
        assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, (
            f"Expected FAIL, got {response.tests[0].result}")

    assert response.numHosts == len(hostList), (
        f"numHosts {response.numHosts} should match hostList length "
        f"{len(hostList)}")
    for i in range(response.numHosts):
        host = response.hosts[i]
        assert len(host.hostname) > 0, f"Host {i} has empty hostname"
        assert host.hostname in hostList, (
            f"Host {i} hostname {host.hostname} not in hostList")

    assert response.numEntities == response.numResults
    assert response.numEntities == config["expected_number_of_entities"], (
        f"numEntities {response.numEntities} should match expected "
        f"{config['expected_number_of_entities']}")

    for i in range(response.numResults):
        result = response.results[i]
        assert result.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU
        assert result.testId == 0
        assert result.hostId < response.numHosts

        if len(error_entities.keys()) > 0:
            hostname = response.hosts[result.hostId].hostname
            entity_key = f"{hostname}:{result.entity.entityId}"
            if entity_key in error_entities.keys():
                assert result.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, (
                    f"Entity {i} with key {entity_key} should be FAIL, "
                    f"got {result.result}")
            else:
                assert result.result == dcgm_structs.DCGM_DIAG_RESULT_PASS, (
                    f"Entity {i} with key {entity_key} should be PASS, "
                    f"got {result.result}")

    expected_num_errors = min(
        config.get("expected_num_errors", 0),
        dcgm_structs.DCGM_MN_DIAG_RESPONSE_ERRORS_MAX)
    assert response.numErrors == expected_num_errors, (
        f"numErrors {response.numErrors} should be {expected_num_errors}")

    if not success:
        for i in range(response.numErrors):
            error = response.errors[i]
            assert error.code == dcgm_errors.DCGM_FR_UNKNOWN
            assert len(error.msg) > 0
            all_error_msgs = [m.strip()
                              for m in error.msg.split(';') if m.strip()]
            hostname = response.hosts[error.hostId].hostname
            entity_key = f"{hostname}:{error.entity.entityId}"
            for expected_msg in error_entities[entity_key]:
                assert any(expected_msg in actual for actual in all_error_msgs), (
                    f"Expected error message '{expected_msg}' not found in "
                    f"actual error messages {all_error_msgs} "
                    f"for entity {entity_key} (error index {i})"
                )


def helper_dcgm_nvloom_multinode_basic_success(handle, gpuIds):
    """Test nvloom on multiple nodes: pass on head node, pass on test nodes."""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)
    config["gpus_per_host"] = 4
    config["test_suites"] = ["gpu_to_rack_device_to_device_bidir_read_ce"]

    messages = {
        "head_node": {
            "gpu_required_info": {},
            "gpu_required_error": {}
        },
        "test_nodes": {
            "gpu_required_info": {},
            "gpu_required_error": {}
        }
    }

    helper_setup_nvloom_testnode_message(config, messages, entities)
    helper_setup_nvloom_headnode_message(config, messages, entities)
    helper_run_and_validate_nvloom(handle, config, success=True)


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_nvloom_multinode_basic_success(handle, gpuIds):
    """Verify nvloom all-pass across multiple nodes."""
    helper_dcgm_nvloom_multinode_basic_success(handle, gpuIds)


def helper_dcgm_nvloom_multinode_basic_failure_1(handle, gpuIds):
    """Test nvloom on multiple nodes: fail on head node 1 gpu, fail on test nodes multiple gpus."""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)
    config["gpus_per_host"] = 4
    config["test_suites"] = ["multicast_all_to_all_ce"]

    messages = {
        "head_node": {
            "gpu_required_info": {},
            "gpu_required_error": {
                0: [
                    "[CUDA_ERROR_UNKNOWN] cuMulticastBindMem failed",
                ],
            },
        },
        "test_nodes": {
            "gpu_required_info": {},
            "gpu_required_error": {
                0: [
                    "[CUDA_ERROR_UNKNOWN] cuMulticastBindMem failed",
                ],
                1: [
                    "NVLINK Error: link degraded",
                ],
                2: [
                    "[CUDA_ERROR_NOT_READY] cuStreamSynchronize failed",
                    "Unexpected abort during memory transfer",
                ],
                3: [
                    "IMEX Error: status is DOWN",
                ],
            }
        }
    }

    helper_setup_nvloom_testnode_message(config, messages, entities)
    helper_setup_nvloom_headnode_message(config, messages, entities)
    helper_run_and_validate_nvloom(handle, config, success=False)


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_nvloom_multinode_basic_failure_1(handle, gpuIds):
    """Verify nvloom multinode failure: head and test nodes fail on multiple GPUs."""
    helper_dcgm_nvloom_multinode_basic_failure_1(handle, gpuIds)


def helper_dcgm_nvloom_multinode_basic_failure_2(handle, gpuIds):
    """Test nvloom on multiple nodes: fail on head node only, test nodes pass."""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)
    config["gpus_per_host"] = 4
    config["test_suites"] = ["gpu_to_rack_device_to_device_bidir_read_ce"]

    messages = {
        "head_node": {
            "gpu_required_info": {},
            "gpu_required_error": {
                0: [
                    "[CUDA_ERROR_UNKNOWN] cuMulticastBindMem failed",
                ],
            },
        },
        "test_nodes": {
            "gpu_required_info": {},
            "gpu_required_error": {}
        }
    }

    helper_setup_nvloom_testnode_message(config, messages, entities)
    helper_setup_nvloom_headnode_message(config, messages, entities)
    helper_run_and_validate_nvloom(handle, config, success=False)


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_nvloom_multinode_basic_failure_2(handle, gpuIds):
    """Verify nvloom multinode failure: head fails, test nodes pass."""
    helper_dcgm_nvloom_multinode_basic_failure_2(handle, gpuIds)


def helper_dcgm_nvloom_multinode_basic_failure_3(handle, gpuIds):
    """Test nvloom on multiple nodes: head node passes, test nodes fail with same error."""
    logger.info(f"mpirun_path: {test_utils.get_mpirun_path()}")

    config = {}
    entities = set()

    helper_setup_multinode_config(config)
    config["gpus_per_host"] = 4
    config["test_suites"] = ["multicast_all_to_all_ce"]

    messages = {
        "head_node": {
            "gpu_required_info": {},
            "gpu_required_error": {}
        },
        "test_nodes": {
            "gpu_required_info": {},
            "gpu_required_error": {
                0: [
                    "[CUDA_ERROR_UNKNOWN] cuMulticastBindMem failed",
                ],
            }
        }
    }

    helper_setup_nvloom_testnode_message(config, messages, entities)
    helper_setup_nvloom_headnode_message(config, messages, entities)
    helper_run_and_validate_nvloom(handle, config, success=False)


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_nvloom_multinode_basic_failure_3(handle, gpuIds):
    """Verify nvloom multinode failure: head passes, test nodes fail."""
    helper_dcgm_nvloom_multinode_basic_failure_3(handle, gpuIds)


# =========================================================================
# Multinode nvbandwidth tests
# =========================================================================

def helper_mnnvbandwidth_config(gpuIds):
    return {
        "mock_hosts": [{
            "hostname": "localhost",
            "gpu_ids": list(gpuIds),
        }],
        "test_nodes": [],
        "stdout_lines": [],
    }


def helper_add_mnnvbandwidth_test_nodes(config, max_nodes=None):
    helper_setup_multinode_config(config)
    if max_nodes is not None:
        config["test_nodes"] = config["test_nodes"][:max_nodes]
    for node in config["test_nodes"]:
        config["mock_hosts"].append({
            "hostname": node["ip"],
            "gpu_ids": [0, 1],
        })


def helper_execute_mnnvbandwidth(handle, config, parameters=()):
    mock_controller = MockMnnvbandwidthController(
        test_utils.get_mock_mnnvbandwidth_path(),
        test_utils.get_mock_nvidia_smi_path())

    try:
        if config["test_nodes"]:
            mock_controller.setup_multinode_mock_environment(config)
        else:
            mock_controller.setup_localhost_mock_environment(config)

        host_list = [
            host["hostname"] for host in config["mock_hosts"]
        ]
        mndiag = DcgmMnDiag.DcgmMnDiag(
            hostList=host_list,
            testName="mnnvbandwidth",
            parameters=parameters,
            handle=handle)
        return mndiag.Execute(handle)
    finally:
        mock_controller.cleanup_multinode_mock_environment(config)


def helper_mnnvbandwidth_process_lines(config):
    lines = []
    rank = 0
    for host in config["mock_hosts"]:
        for gpu_id in host["gpu_ids"]:
            lines.append(
                f"Process {rank} ({host['hostname']}): device {gpu_id}: "
                f"NVIDIA GB200 (00000000:{rank:02x}:00)")
            rank += 1
    return lines


def helper_mnnvbandwidth_rank(config, hostname, gpu_id):
    rank = 0
    for host in config["mock_hosts"]:
        for current_gpu_id in host["gpu_ids"]:
            if (host["hostname"], current_gpu_id) == (hostname, gpu_id):
                return rank
            rank += 1
    raise ValueError(f"No rank found for {hostname} GPU {gpu_id}")


def helper_set_mnnvbandwidth_success_output(config, testcase):
    process_lines = helper_mnnvbandwidth_process_lines(config)
    device_count = len(process_lines)
    matrix_lines = [
        "  " + "".join(f"{column:>10}" for column in range(device_count)),
    ]
    matrix_lines.extend(
        f"{row:>2}" + "".join(
            f"{'N/A' if row == column else '800.00':>10}"
            for column in range(device_count))
        for row in range(device_count)
    )
    config["stdout_lines"] = process_lines + [
        f"Running {testcase}.",
        "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)",
    ] + matrix_lines + [
        f"SUM {testcase} {800.0 * device_count * (device_count - 1)}",
    ]


def helper_mnnvbandwidth_entity_pairs(response):
    pairs = set()
    for index in range(response.numEntities):
        entity = response.entities[index]
        hostname = response.hosts[entity.hostId].hostname
        pairs.add((hostname, entity.entity.entityId))
    return pairs


def helper_mnnvbandwidth_results(response):
    results = {}
    for index in range(response.numResults):
        result = response.results[index]
        assert result.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU
        hostname = response.hosts[result.hostId].hostname
        entity = (hostname, result.entity.entityId)
        assert entity not in results
        results[entity] = result.result
    return results


def helper_validate_mnnvbandwidth_response(
        response,
        expected_result,
        expected_entities,
        expected_results,
        expected_num_errors,
        error_group=None,
        error_entity_id=None,
        error_hostname=None,
        error_substrings=()):
    actual_errors = [
        response.errors[index].msg
        for index in range(response.numErrors)
    ]
    assert response.version == dcgm_structs.dcgmMnDiagResponse_version2
    assert response.numTests == 1
    assert response.tests[0].name == "MNNVBANDWIDTH"
    assert response.tests[0].result == expected_result, (
        f"Expected result {expected_result}, got {response.tests[0].result}; "
        f"errors: {actual_errors!r}")
    assert response.numEntities == len(expected_entities)
    assert response.numResults == len(expected_results)
    assert response.numErrors == expected_num_errors
    assert response.tests[0].numResults == len(expected_results)
    assert response.tests[0].numErrors == expected_num_errors
    assert response.numHosts == len({
        hostname for hostname, _ in expected_entities
    })
    assert helper_mnnvbandwidth_entity_pairs(
        response) == set(expected_entities)
    assert helper_mnnvbandwidth_results(response) == expected_results

    if error_group is not None:
        assert response.numErrors > 0
        assert response.errors[0].entity.entityGroupId == error_group
    if error_entity_id is not None:
        assert response.numErrors > 0
        assert response.errors[0].entity.entityId == error_entity_id
    if error_hostname is not None:
        assert response.numErrors > 0
        error = response.errors[0]
        assert response.hosts[error.hostId].hostname == error_hostname

    for expected_error in error_substrings:
        assert any(
            expected_error in error for error in actual_errors), (
                f"Expected error substring {expected_error!r} not found "
                f"in {actual_errors!r}")


def helper_expected_mnnvbandwidth_entities(config):
    return {
        (host["hostname"], gpu_id)
        for host in config["mock_hosts"]
        for gpu_id in host["gpu_ids"]
    }


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_text_success(handle, gpuIds):
    config = helper_mnnvbandwidth_config(gpuIds)
    helper_set_mnnvbandwidth_success_output(
        config, "device_to_device_memcpy_read_ce")
    response = helper_execute_mnnvbandwidth(handle, config)
    expected_entities = helper_expected_mnnvbandwidth_entities(config)

    helper_validate_mnnvbandwidth_response(
        response,
        dcgm_structs.DCGM_DIAG_RESULT_PASS,
        expected_entities,
        dict.fromkeys(
            expected_entities, dcgm_structs.DCGM_DIAG_RESULT_PASS),
        0)


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_uses_required_defaults(handle, gpuIds):
    """Verify mnnvbandwidth selects the required tests and all pairs for a small run."""
    config = helper_mnnvbandwidth_config(gpuIds)
    config["required_args"] = [
        "--testcase",
        "multinode_device_to_device_memcpy_write_ce",
        "multinode_device_to_device_memcpy_read_ce",
        "--targetNumPairs",
        "-1",
    ]
    helper_set_mnnvbandwidth_success_output(
        config, "multinode_device_to_device_memcpy_read_ce")
    response = helper_execute_mnnvbandwidth(handle, config)
    expected_entities = helper_expected_mnnvbandwidth_entities(config)

    helper_validate_mnnvbandwidth_response(
        response,
        dcgm_structs.DCGM_DIAG_RESULT_PASS,
        expected_entities,
        dict.fromkeys(
            expected_entities, dcgm_structs.DCGM_DIAG_RESULT_PASS),
        0)


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_gpu_pairs_all(handle, gpuIds):
    """Verify gpuPairs=all executes and aggregates every injected GPU pair."""
    assert len(gpuIds) == 2, "The GB200 mock SKU must expose exactly two GPUs"
    config = helper_mnnvbandwidth_config(gpuIds)
    config["required_args"] = [
        "--pair",
        "--testcase",
        "multinode_device_to_device_memcpy_read_sm",
        "--bufferSize",
        "1",
    ]
    helper_set_mnnvbandwidth_success_output(
        config, "multinode_device_to_device_memcpy_read_sm")

    response = helper_execute_mnnvbandwidth(
        handle, config, parameters=["mnnvbandwidth.gpuPairs=all"])
    expected_entities = helper_expected_mnnvbandwidth_entities(config)

    helper_validate_mnnvbandwidth_response(
        response,
        dcgm_structs.DCGM_DIAG_RESULT_PASS,
        expected_entities,
        dict.fromkeys(
            expected_entities, dcgm_structs.DCGM_DIAG_RESULT_PASS),
        0)
    assert response.tests[0].auxData.data.startswith(
        "MNNVBANDWIDTH_SUMMARY planned=1;passed=1;failed=0;")
    assert ";stdout=" in response.tests[0].auxData.data
    assert ";stderr=" in response.tests[0].auxData.data
    assert all(
        response.entities[index].serialNum == ""
        for index in range(response.numEntities))


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_text_testcase_failure(
        handle, gpuIds):
    config = helper_mnnvbandwidth_config(gpuIds)
    config["stdout_lines"] = helper_mnnvbandwidth_process_lines(config)
    config["stderr_lines"] = [
        "ERROR: Testcase missing_test not found!",
    ]
    config["exit_code"] = 1
    response = helper_execute_mnnvbandwidth(handle, config)

    helper_validate_mnnvbandwidth_response(
        response,
        dcgm_structs.DCGM_DIAG_RESULT_FAIL,
        helper_expected_mnnvbandwidth_entities(config),
        {},
        1,
        error_group=dcgm_fields.DCGM_FE_NONE,
        error_substrings=("missing_test", "not found"))


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_text_rank_failure(
        handle, gpuIds):
    gpu_id = gpuIds[0]
    config = helper_mnnvbandwidth_config([gpu_id])
    config.update({
        "stdout_lines": [
            f"Process 0 (localhost): device {gpu_id}: "
            "NVIDIA GB200 (00000000:01:00)",
        ],
        "stderr_lines": [
            (
                "[CUDA_ERROR_ILLEGAL_ADDRESS] an illegal memory access was "
                "encountered in expression cudaMemcpy(dst, src, size) on "
                "localhost, rank = 0 in copy() : multinode_memcpy.cpp:58"
            ),
        ],
        "exit_code": 1,
    })
    response = helper_execute_mnnvbandwidth(handle, config)

    helper_validate_mnnvbandwidth_response(
        response,
        dcgm_structs.DCGM_DIAG_RESULT_FAIL,
        {("localhost", gpu_id)},
        {("localhost", gpu_id): dcgm_structs.DCGM_DIAG_RESULT_FAIL},
        1,
        error_group=dcgm_fields.DCGM_FE_GPU,
        error_entity_id=gpu_id,
        error_substrings=("CUDA_ERROR_ILLEGAL_ADDRESS",))


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_text_mpi_failure(
        handle, gpuIds):
    config = helper_mnnvbandwidth_config(gpuIds)
    config.update({
        "stderr_lines": [
            (
                "MPI_ABORT was invoked on rank 0 in communicator "
                "MPI_COMM_WORLD"
            ),
        ],
        "exit_code": 1,
    })
    response = helper_execute_mnnvbandwidth(handle, config)

    helper_validate_mnnvbandwidth_response(
        response,
        dcgm_structs.DCGM_DIAG_RESULT_FAIL,
        set(),
        {},
        1,
        error_group=dcgm_fields.DCGM_FE_NONE,
        error_substrings=("MPI_ABORT",))


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_multinode_remote_exit_failure(
        handle, gpuIds):
    config = helper_mnnvbandwidth_config(gpuIds)
    helper_add_mnnvbandwidth_test_nodes(config, max_nodes=1)
    helper_set_mnnvbandwidth_success_output(
        config, "multinode_device_to_device_memcpy_read_ce")
    # Keep every rank alive through process detection. Without this, the
    # launcher can finish before the manager observes it and the test only
    # exercises the generic no-process failure path.
    config["run_seconds"] = 1
    failed_node = config["test_nodes"][0]
    failed_node["exit_code"] = 1

    response = helper_execute_mnnvbandwidth(handle, config)

    assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL
    assert response.tests[0].auxData.data.startswith("MPI exited with code:")


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_multinode_text_success(
        handle, gpuIds):
    config = helper_mnnvbandwidth_config(gpuIds)
    helper_add_mnnvbandwidth_test_nodes(config)
    helper_set_mnnvbandwidth_success_output(
        config, "multinode_device_to_device_memcpy_read_ce")
    response = helper_execute_mnnvbandwidth(handle, config)
    expected_entities = helper_expected_mnnvbandwidth_entities(config)

    helper_validate_mnnvbandwidth_response(
        response,
        dcgm_structs.DCGM_DIAG_RESULT_PASS,
        expected_entities,
        dict.fromkeys(
            expected_entities, dcgm_structs.DCGM_DIAG_RESULT_PASS),
        0)


# NO HARDWARE


@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_mndiag_mnnvbandwidth_multinode_text_failure(
        handle, gpuIds):
    config = helper_mnnvbandwidth_config(gpuIds)
    helper_add_mnnvbandwidth_test_nodes(config)
    assert len(config["mock_hosts"]) > 1
    failed_host = config["mock_hosts"][-1]
    assert len(failed_host["gpu_ids"]) > 1
    failed_hostname = failed_host["hostname"]
    failed_gpu_id = failed_host["gpu_ids"][-1]
    failed_rank = helper_mnnvbandwidth_rank(
        config, failed_hostname, failed_gpu_id)
    assert failed_rank > 0
    config["stdout_lines"] = helper_mnnvbandwidth_process_lines(config)
    config["stderr_lines"] = [
        (
            "[CUDA_ERROR_ILLEGAL_ADDRESS] an illegal memory access was "
            "encountered in expression cudaMemcpy(dst, src, size) on "
            f"{failed_hostname}, rank = {failed_rank} in copy() : "
            "multinode_memcpy.cpp:58"
        ),
    ]
    response = helper_execute_mnnvbandwidth(handle, config)

    helper_validate_mnnvbandwidth_response(
        response,
        dcgm_structs.DCGM_DIAG_RESULT_FAIL,
        helper_expected_mnnvbandwidth_entities(config),
        {(failed_hostname, failed_gpu_id):
            dcgm_structs.DCGM_DIAG_RESULT_FAIL},
        1,
        error_group=dcgm_fields.DCGM_FE_GPU,
        error_entity_id=failed_gpu_id,
        error_hostname=failed_hostname,
        error_substrings=("CUDA_ERROR_ILLEGAL_ADDRESS",))


def helper_dcgm_nvloom_validate_duration(handle, gpuIds):
    """Verify that mndiag computes --duration from time_to_run correctly.

    Sets nvloom.time_to_run=300 and checks that the mock nvloom_cli receives
    --duration equal to floor((300 - 30) / 8) = 33. The default gpu-to-rack
    suite has 8 testcases in both NVLoom 1.4.1 and 2.0.0.
    """
    logger.info("Starting nvloom duration validation test")

    config = {
        "hostList": ["localhost"],
        "gpus_per_host": len(gpuIds),
        "gpu_required_info": {},
        "gpu_required_error": {},
        "test_suites": ["gpu_to_rack_device_to_device_bidir_read_ce"],
    }

    mock_controller = MockNvloomController(
        test_utils.get_mock_nvloom_path(),
        test_utils.get_mock_nvidia_smi_path())

    try:
        mock_controller.cleanup_captured_args()
        mock_controller.setup_localhost_mock_environment(config)

        time_to_run = 300
        margin = 30
        gpu_to_rack_count = 8
        expected_duration = (time_to_run - margin) // gpu_to_rack_count

        mndiag = DcgmMnDiag.DcgmMnDiag(
            hostList=config["hostList"],
            testName="nvloom",
            parameters=[f"nvloom.time_to_run={time_to_run}"],
            handle=handle)
        mndiag.Execute(handle)

        captured_duration = mock_controller.get_captured_duration()
        assert captured_duration is not None, (
            "Mock nvloom_cli did not capture --duration argument")
        assert captured_duration == str(expected_duration), (
            f"Expected --duration={expected_duration}, "
            f"got --duration={captured_duration}")

        logger.info(
            f"Nvloom duration validation passed: --duration={captured_duration}")

    except Exception:
        helper_print_stderr_output()
        raise

    finally:
        mock_controller.cleanup_captured_args()
        mock_controller.cleanup_mock_environment()


# NO HARDWARE

@test_utils.run_if_mpirun_exists()
@test_utils.run_with_injection_nvml_using_specific_sku(test_helpers.helper_dcgm_mndiag.get_injection_values()["INJECTION_YAML"])
@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_dcgm_mndiag.HE_Env)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_nvloom_validate_duration(handle, gpuIds):
    """Verify mndiag passes computed --duration to nvloom_cli."""
    helper_dcgm_nvloom_validate_duration(handle, gpuIds)
