# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import contextmanager
import json
import os

import apps
import dcgm_agent_internal
import dcgm_errors
import dcgm_fields
import dcgm_structs
import DcgmDiag
import test_helpers.helper_sdc_plugin
import test_utils


CPU_SDC_TEST_NAME = "cpu_sdc"
CPU_EUD_DISABLE_PARAM = "cpu_eud.is_allowed=false"
CPU_EUD_MISSING_TEST_MSG = "test 'cpu_eud' does not match any loaded tests"
CPU_SDC_MISSING_PACKAGE_MSG = "Please ensure that the [cpu_sdc] dependent package is installed"
CPU_SDC_DEFAULT_BINARY_PATH = "/path/to/real/sdc"
EXPECTED_SDC_BINARY_VERSION = "1.2.3"
CPU_SDC_ENV_VARS = (
    "DCGM_CPU_SDC_BIN_PATH",
    "DCGM_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED",
    "DCGM_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS",
    "DCGM_CPU_SDC_SKIP_BIN_PERMISSION_CHECK",
    "DCGM_MOCKED_SDC_RETURN_PATH",
    "DCGM_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC",
)


@contextmanager
def _without_cpu_sdc_env():
    original_env = {name: os.environ.pop(
        name, None) for name in CPU_SDC_ENV_VARS}
    try:
        yield
    finally:
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _find_cpu_sdc_test(response):
    for idx in range(response.numTests):
        if response.tests[idx].name == CPU_SDC_TEST_NAME:
            return idx, response.tests[idx]
    raise AssertionError("Should have cpu_sdc test result.")


def _run_cpu_sdc_diag(handle, cpuIds, paramsStr=""):
    dd = DcgmDiag.DcgmDiag(
        cpuIds=cpuIds,
        testNamesStr=CPU_SDC_TEST_NAME,
        paramsStr=paramsStr)
    response = test_utils.diag_execute_wrapper(dd, handle)
    return response


def _run_diag_level(handle, cpuIds, runLevel, paramsStr=""):
    dd = DcgmDiag.DcgmDiag(
        cpuIds=cpuIds,
        testNamesStr=runLevel,
        paramsStr=paramsStr,
        timeout=30)
    response = test_utils.diag_execute_wrapper(dd, handle)
    return response


def _run_diag_level_with_optional_cpu_eud_skip(handle, cpuIds, runLevel):
    try:
        return _run_diag_level(
            handle, cpuIds, runLevel, paramsStr=CPU_EUD_DISABLE_PARAM)
    except dcgm_structs.DCGMError_NvvsNoAvailableTest as error:
        if CPU_EUD_MISSING_TEST_MSG not in str(error):
            raise
        return _run_diag_level(handle, cpuIds, runLevel)


def _assert_diag_level_includes_cpu_sdc(handle, cpuIds, skipCpuEud=False):
    for runLevel in ["3", "4"]:
        if skipCpuEud:
            response = _run_diag_level_with_optional_cpu_eud_skip(
                handle, cpuIds, runLevel)
        else:
            response = _run_diag_level(handle, cpuIds, runLevel)
        test_idx, sdc_test = _find_cpu_sdc_test(response)

        assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_PASS, (
            f"Actual result for r{runLevel} is {sdc_test.result}")
        assert sdc_test.numErrors == 0, (
            f"Actual number of errors for r{runLevel} is {sdc_test.numErrors}")
        _assert_cpu_sdc_binary_version(sdc_test)
        _assert_cpu_sdc_results(
            response,
            test_idx,
            sdc_test,
            _expected_result_map(cpuIds, dcgm_structs.DCGM_DIAG_RESULT_PASS))


def _require_cpu0(cpuIds):
    if 0 not in cpuIds:
        test_utils.skip_test(
            "The mocked_cpu_sdc CPU-specific fixtures report CPU 0, "
            f"but CPU IDs are {cpuIds}.")


def _expected_result_map(cpuIds, default_result, overrides=None):
    expected = {cpuId: default_result for cpuId in cpuIds}
    if overrides:
        expected.update(overrides)
    return expected


def _assert_cpu_sdc_results(response, test_idx, sdc_test, expected_results):
    assert sdc_test.numResults == len(expected_results), (
        f"Should have [{len(expected_results)}] results, but found "
        f"[{sdc_test.numResults}]")

    actual_results = {}
    for idx in range(sdc_test.numResults):
        result_idx = sdc_test.resultIndices[idx]
        result = response.results[result_idx]
        assert result.testId == test_idx, f"Actual testId is {result.testId}"
        assert result.entity.entityGroupId == dcgm_fields.DCGM_FE_CPU, (
            f"Actual entityGroupId is {result.entity.entityGroupId}")
        actual_results[result.entity.entityId] = result.result

    assert actual_results == expected_results, (
        f"Actual results are {actual_results}, expected {expected_results}")


def _assert_cpu_sdc_error(
        response,
        sdc_test,
        expected_entity_group_id,
        expected_entity_id,
        expected_message):
    assert sdc_test.numErrors == 1, (
        f"Expected 1 error, got {sdc_test.numErrors}")
    error_idx = sdc_test.errorIndices[0]
    error = response.errors[error_idx]
    assert error.code == dcgm_errors.DCGM_FR_CPU_SDC_TEST_FAILED, (
        f"Actual err.code is {error.code}")
    assert error.category == dcgm_errors.DCGM_FR_EC_SOFTWARE_CPU_SDC, (
        f"Actual err.category is {error.category}")
    assert error.severity == dcgm_errors.DCGM_ERROR_TRIAGE, (
        f"Actual err.severity is {error.severity}")
    assert error.entity.entityGroupId == expected_entity_group_id, (
        f"Actual err.entity.entityGroupId is {error.entity.entityGroupId}")
    assert error.entity.entityId == expected_entity_id, (
        f"Actual err.entity.entityId is {error.entity.entityId}")
    assert expected_message in error.msg, f"Actual err.msg is {error.msg}"


def _assert_cpu_sdc_binary_version(sdc_test):
    assert sdc_test.auxData.version == dcgm_structs.dcgmDiagTestAuxData_version1, (
        f"Actual auxData.version is {sdc_test.auxData.version}")
    aux_json = json.loads(sdc_test.auxData.data)
    assert aux_json["version"] == EXPECTED_SDC_BINARY_VERSION, (
        f"Actual aux data is {sdc_test.auxData.data}")


def _assert_cpu_sdc_result_field(handle, expected_error_code):
    field_values = dcgm_agent_internal.dcgmGetLatestValuesForFields(
        handle, 0, [dcgm_fields.DCGM_FI_DEV_DIAG_CPU_SDC_RESULT])

    assert len(field_values) == 1, (
        f"Expected one CPU SDC result field, got {len(field_values)}")
    field_value = field_values[0]
    assert field_value.fieldId == dcgm_fields.DCGM_FI_DEV_DIAG_CPU_SDC_RESULT, (
        f"Actual field ID is {field_value.fieldId}")
    assert field_value.status == dcgm_structs.DCGM_ST_OK, (
        f"Actual field status is {field_value.status}")
    assert field_value.value.i64 == expected_error_code, (
        f"Actual CPU SDC result field value is {field_value.value.i64}, "
        f"expected {expected_error_code}")


def _cpu_sdc_params(is_allowed=True):
    return f"{CPU_SDC_TEST_NAME}.is_allowed={is_allowed}"


def _run_cpu_sdc_dcgmi(json_output=False):
    args = ["diag", "-r", CPU_SDC_TEST_NAME, "-i", "cpu:*"]
    if json_output:
        args.append("-j")

    ret_value, stdout_lines, stderr_lines = test_utils.run_dcgmi_command(args)
    assert ret_value == 0, (
        f"dcgmi returned {ret_value}. stdout={stdout_lines}, "
        f"stderr={stderr_lines}")
    return stdout_lines


def _dcgmi_stdout(stdout_lines):
    return "\n".join(stdout_lines)


def _dcgmi_json(stdout_lines):
    stdout = _dcgmi_stdout(stdout_lines)
    start = stdout.find("{")
    end = stdout.rfind("}")
    assert start >= 0 and end >= start, f"dcgmi JSON output not found: {stdout}"
    return json.loads(stdout[start:end + 1])


def _find_dcgmi_json_test(output, test_name):
    for category in output.get("DCGM Diagnostic", {}).get("test_categories", []):
        for test in category.get("tests", []):
            if test.get("name") == test_name:
                return test
    raise AssertionError(
        f"dcgmi JSON output did not contain test {test_name}: {output}")


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env(
        "system_pass",
        heartbeat_timeout_seconds=1,
        mocked_missed_heartbeat_interval_sec=4),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_heartbeat_timeout_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_cpu_sdc_heartbeat_timeout_seconds(handle, cpuIds):
    response = _run_cpu_sdc_diag(handle, cpuIds)
    test_idx, sdc_test = _find_cpu_sdc_test(response)

    assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, (
        f"Actual result is {sdc_test.result}")
    _assert_cpu_sdc_results(
        response,
        test_idx,
        sdc_test,
        _expected_result_map(cpuIds, dcgm_structs.DCGM_DIAG_RESULT_FAIL))
    _assert_cpu_sdc_error(
        response,
        sdc_test,
        dcgm_fields.DCGM_FE_NONE,
        0,
        "Child process heartbeat timed out")


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env(
        "system_pass",
        heartbeat_timeout_disabled=1,
        heartbeat_timeout_seconds=1,
        mocked_missed_heartbeat_interval_sec=4),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_heartbeat_disabled_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_cpu_sdc_heartbeat_timeout_disabled(handle, cpuIds):
    response = _run_cpu_sdc_diag(handle, cpuIds)
    test_idx, sdc_test = _find_cpu_sdc_test(response)

    assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_PASS, (
        f"Actual result is {sdc_test.result}")
    assert sdc_test.numErrors == 0, (
        f"Actual number of errors is {sdc_test.numErrors}")
    _assert_cpu_sdc_binary_version(sdc_test)
    _assert_cpu_sdc_results(
        response,
        test_idx,
        sdc_test,
        _expected_result_map(cpuIds, dcgm_structs.DCGM_DIAG_RESULT_PASS))


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists("/var/tmp")
def test_dcgmi_cpu_sdc_missing_package_without_binary_env():
    if os.path.exists(CPU_SDC_DEFAULT_BINARY_PATH):
        test_utils.skip_test(
            f"Skip, because CPU SDC binary exists at {CPU_SDC_DEFAULT_BINARY_PATH}.")

    with _without_cpu_sdc_env():
        with test_utils.RunStandaloneHostEngine(
                120,
                heEnv={"DCGM_SUPPORT_NON_NVIDIA_CPU": "1"},
                heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_missing_package_home_dir']):
            dcgmi = apps.DcgmiApp(
                ["diag", "-r", CPU_SDC_TEST_NAME, "-i", "cpu:*"])
            dcgmi.start(timeout=30)
            ret_value = dcgmi.wait()
            dcgmi.validate()
            stdout_lines = dcgmi.stdout_lines
            stderr_lines = dcgmi.stderr_lines

    output = "\n".join(stdout_lines + stderr_lines)
    assert ret_value != 0, (
        f"dcgmi should fail without the CPU SDC package. output={output}")
    assert "cannot find dependent package" in output, f"Actual dcgmi output is {output}"
    assert CPU_SDC_MISSING_PACKAGE_MSG in output, f"Actual dcgmi output is {output}"
    assert "DCGM proprietary package" not in output, f"Actual dcgmi output is {output}"


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("cpu0_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("cpu0_pass"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_cpu_sdc_with_cpu_pass(handle, cpuIds):
    _require_cpu0(cpuIds)
    response = _run_cpu_sdc_diag(handle, cpuIds)
    test_idx, sdc_test = _find_cpu_sdc_test(response)

    assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_PASS, (
        f"Actual result is {sdc_test.result}")
    assert sdc_test.numErrors == 0, (
        f"Actual number of errors is {sdc_test.numErrors}")
    _assert_cpu_sdc_binary_version(sdc_test)
    _assert_cpu_sdc_results(
        response,
        test_idx,
        sdc_test,
        _expected_result_map(
            cpuIds,
            dcgm_structs.DCGM_DIAG_RESULT_SKIP,
            {0: dcgm_structs.DCGM_DIAG_RESULT_PASS}))
    _assert_cpu_sdc_result_field(
        handle, dcgm_errors.DCGM_FR_OK)


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("cpu0_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("cpu0_pass", exit_code=7),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_cpu_sdc_with_cpu_pass_nonzero_exit(handle, cpuIds):
    _require_cpu0(cpuIds)
    response = _run_cpu_sdc_diag(handle, cpuIds)
    test_idx, sdc_test = _find_cpu_sdc_test(response)

    assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, (
        f"Actual result is {sdc_test.result}")
    _assert_cpu_sdc_results(
        response,
        test_idx,
        sdc_test,
        _expected_result_map(cpuIds, dcgm_structs.DCGM_DIAG_RESULT_FAIL))
    _assert_cpu_sdc_error(
        response,
        sdc_test,
        dcgm_fields.DCGM_FE_NONE,
        0,
        "Child process exited with non-zero code, 7.")


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("cpu0_fail_then_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("cpu0_fail_then_pass"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_cpu_sdc_with_cpu_fail_then_pass(handle, cpuIds):
    _require_cpu0(cpuIds)
    response = _run_cpu_sdc_diag(handle, cpuIds)
    test_idx, sdc_test = _find_cpu_sdc_test(response)

    assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, (
        f"Actual result is {sdc_test.result}")
    _assert_cpu_sdc_results(
        response,
        test_idx,
        sdc_test,
        _expected_result_map(
            cpuIds,
            dcgm_structs.DCGM_DIAG_RESULT_SKIP,
            {0: dcgm_structs.DCGM_DIAG_RESULT_FAIL}))
    _assert_cpu_sdc_error(
        response,
        sdc_test,
        dcgm_fields.DCGM_FE_CPU,
        0,
        "Failed first.")
    _assert_cpu_sdc_result_field(
        handle, dcgm_errors.DCGM_FR_CPU_SDC_TEST_FAILED)


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("system_pass"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_cpu_sdc_with_system_pass(handle, cpuIds):
    response = _run_cpu_sdc_diag(handle, cpuIds)
    test_idx, sdc_test = _find_cpu_sdc_test(response)

    assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_PASS, (
        f"Actual result is {sdc_test.result}")
    assert sdc_test.numErrors == 0, (
        f"Actual number of errors is {sdc_test.numErrors}")
    _assert_cpu_sdc_binary_version(sdc_test)
    _assert_cpu_sdc_results(
        response,
        test_idx,
        sdc_test,
        _expected_result_map(cpuIds, dcgm_structs.DCGM_DIAG_RESULT_PASS))


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("system_pass"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_r3_r4_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_diag_r_3_4_include_cpu_sdc(handle, cpuIds):
    _assert_diag_level_includes_cpu_sdc(handle, cpuIds, skipCpuEud=True)


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("system_pass"),
    heArgs=[
        '--service-account',
        'dcgm-tests-service-account',
        '--home-dir',
        '/var/tmp/test_dcgm_sdc_r3_r4_service_account_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_diag_r_3_4_include_cpu_sdc_with_service_account(handle, cpuIds):
    _assert_diag_level_includes_cpu_sdc(handle, cpuIds, skipCpuEud=True)


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_fail_then_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("system_fail_then_pass"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgm_cpu_sdc_with_system_fail_then_pass(handle, cpuIds):
    response = _run_cpu_sdc_diag(handle, cpuIds)
    test_idx, sdc_test = _find_cpu_sdc_test(response)

    assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, (
        f"Actual result is {sdc_test.result}")
    _assert_cpu_sdc_results(
        response,
        test_idx,
        sdc_test,
        _expected_result_map(cpuIds, dcgm_structs.DCGM_DIAG_RESULT_FAIL))
    _assert_cpu_sdc_error(
        response,
        sdc_test,
        dcgm_fields.DCGM_FE_NONE,
        0,
        "failed")


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("cpu0_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("cpu0_pass"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_cpu_sdc_plugin_is_allowed(handle, cpuIds):
    response = _run_cpu_sdc_diag(
        handle, cpuIds, paramsStr=_cpu_sdc_params(is_allowed=False))
    test_idx, sdc_test = _find_cpu_sdc_test(response)

    assert sdc_test.result == dcgm_structs.DCGM_DIAG_RESULT_SKIP, (
        f"Actual result is {sdc_test.result}")
    assert sdc_test.numErrors == 0, (
        f"Actual number of errors is {sdc_test.numErrors}")
    _assert_cpu_sdc_results(
        response,
        test_idx,
        sdc_test,
        _expected_result_map(cpuIds, dcgm_structs.DCGM_DIAG_RESULT_SKIP))


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("system_pass"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_dcgmi_json_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgmi_cpu_sdc_json_output_has_sdc_binary_version(handle, cpuIds):
    output = _dcgmi_json(_run_cpu_sdc_dcgmi(json_output=True))

    assert output["metadata"]["CPU SDC Binary Version"] == EXPECTED_SDC_BINARY_VERSION, (
        f"Actual dcgmi JSON output is {output}")


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_pass_no_version"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("system_pass_no_version"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_dcgmi_no_version_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgmi_cpu_sdc_output_skips_sdc_binary_version_without_version_payload(handle, cpuIds):
    output = _dcgmi_json(_run_cpu_sdc_dcgmi(json_output=True))
    cpu_sdc_test = _find_dcgmi_json_test(output, CPU_SDC_TEST_NAME)

    assert cpu_sdc_test["test_summary"]["status"] == "Pass", (
        f"Actual dcgmi JSON output is {output}")
    assert "CPU SDC Binary Version" not in output.get("metadata", {}), (
        f"Actual dcgmi JSON output is {output}")

    stdout = _dcgmi_stdout(_run_cpu_sdc_dcgmi())

    assert "CPU SDC Binary Version" not in stdout, f"Actual dcgmi output is {stdout}"


# NO HARDWARE

@test_utils.run_only_as_root()
@test_utils.run_only_on_numa_systems()
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.MOCKED_SDC_PATH)
@test_utils.run_only_when_path_exists(
    test_helpers.helper_sdc_plugin.result_file_path("system_pass"))
@test_utils.run_only_when_path_exists("/var/tmp")
@test_utils.run_with_standalone_host_engine(
    120,
    heEnv=test_helpers.helper_sdc_plugin.HE_Env("system_pass"),
    heArgs=['--home-dir', '/var/tmp/test_dcgm_sdc_dcgmi_cli_home_dir'])
@test_utils.run_only_with_live_cpus()
def test_dcgmi_cpu_sdc_cli_output_has_sdc_binary_version(handle, cpuIds):
    stdout = _dcgmi_stdout(_run_cpu_sdc_dcgmi())

    assert "CPU SDC Binary Version" in stdout, f"Actual dcgmi output is {stdout}"
    assert EXPECTED_SDC_BINARY_VERSION in stdout, f"Actual dcgmi output is {stdout}"
