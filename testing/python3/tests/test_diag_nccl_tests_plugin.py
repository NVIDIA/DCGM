# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.

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

import DcgmDiag
import test_utils
import dcgm_structs
import dcgm_errors
import test_helpers.helper_nccl_tests_plugin


def _print_diag_response(response):
    """Print basic diagnostic response details for debugging."""
    test_names = []
    for i in range(response.numTests):
        name = response.tests[i].name
        test_names.append(name.decode() if isinstance(
            name, (bytes, bytearray)) else str(name))
    errors = []
    for i in range(response.numErrors):
        err = response.errors[i]
        errors.append({"code": err.code, "msg": err.msg})
    print(f"diag tests: {test_names}")
    print(f"diag errors: {errors}")


def _assert_nccl_present(response):
    idx = get_nccl_test_index(response)
    if idx is None:
        _print_diag_response(response)
    assert idx is not None, "nccl_tests not found in response"
    return idx


def get_nccl_test_index(response):
    """Find the index of nccl_tests in the response."""
    for i in range(response.numTests):
        name = response.tests[i].name
        if isinstance(name, (bytes, bytearray)):
            name = name.decode(errors="replace")
        if str(name) == "nccl_tests":
            return i
    return None


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_skips_when_env_not_set(handle, gpuIds):
    """
    Test that requesting nccl_tests without DCGM_NCCL_TESTS_BIN_PATH yields no available test.
    """
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nccl_tests")
    try:
        _ = test_utils.diag_execute_wrapper(dd, handle)
    except dcgm_structs.DCGMError_NvvsNoAvailableTest:
        return
    assert False, "Expected NVVS to report no available tests when env is unset"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_nccl_tests_plugin.HE_Env("pass"))
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_pass(handle, gpuIds):
    """
    Test that nccl_tests plugin passes when mock outputs passing results.
    """
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nccl_tests")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nccl_test_idx = _assert_nccl_present(response)
    assert response.tests[nccl_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_PASS, \
        f"Expected PASS, got {response.tests[nccl_test_idx].result}"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_nccl_tests_plugin.HE_Env("pass_exit_1"))
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_pass_output_nonzero_exit(handle, gpuIds):
    """
    Test that nccl_tests plugin fails when output is OK but exit code is non-zero.
    """
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nccl_tests")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nccl_test_idx = _assert_nccl_present(response)
    assert response.tests[nccl_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, \
        f"Expected FAIL, got {response.tests[nccl_test_idx].result}"
    assert response.tests[nccl_test_idx].numErrors > 0, "Expected at least one error"

    err_idx = response.tests[nccl_test_idx].errorIndices[0]
    err = response.errors[err_idx]
    assert err.code == dcgm_errors.DCGM_FR_NCCL_ERROR, \
        f"Expected DCGM_FR_NCCL_ERROR ({dcgm_errors.DCGM_FR_NCCL_ERROR}), got {err.code}"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_nccl_tests_plugin.HE_Env("fail_oob"))
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_fail_oob(handle, gpuIds):
    """
    Test that nccl_tests plugin fails with DCGM_FR_NCCL_ERROR on out-of-bounds failure.
    """
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nccl_tests")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nccl_test_idx = _assert_nccl_present(response)
    assert response.tests[nccl_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, \
        f"Expected FAIL, got {response.tests[nccl_test_idx].result}"
    assert response.tests[nccl_test_idx].numErrors > 0, "Expected at least one error"

    err_idx = response.tests[nccl_test_idx].errorIndices[0]
    err = response.errors[err_idx]
    assert err.code == dcgm_errors.DCGM_FR_NCCL_ERROR, \
        f"Expected DCGM_FR_NCCL_ERROR ({dcgm_errors.DCGM_FR_NCCL_ERROR}), got {err.code}"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_nccl_tests_plugin.HE_Env("fail_bw"))
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_fail_bandwidth(handle, gpuIds):
    """
    Test that nccl_tests plugin fails with DCGM_FR_NCCL_ERROR on bandwidth failure.
    """
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nccl_tests")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nccl_test_idx = _assert_nccl_present(response)
    assert response.tests[nccl_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, \
        f"Expected FAIL, got {response.tests[nccl_test_idx].result}"
    assert response.tests[nccl_test_idx].numErrors > 0, "Expected at least one error"

    err_idx = response.tests[nccl_test_idx].errorIndices[0]
    err = response.errors[err_idx]
    assert err.code == dcgm_errors.DCGM_FR_NCCL_ERROR, \
        f"Expected DCGM_FR_NCCL_ERROR ({dcgm_errors.DCGM_FR_NCCL_ERROR}), got {err.code}"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_nccl_tests_plugin.HE_Env("none_exit_1"))
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_nonzero_exit_no_output(handle, gpuIds):
    """
    Test that nccl_tests plugin fails when output is missing expected result lines.
    """
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nccl_tests")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nccl_test_idx = _assert_nccl_present(response)
    assert response.tests[nccl_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, \
        f"Expected FAIL, got {response.tests[nccl_test_idx].result}"
    assert response.tests[nccl_test_idx].numErrors > 0, "Expected at least one error"

    err_idx = response.tests[nccl_test_idx].errorIndices[0]
    err = response.errors[err_idx]
    assert err.code == dcgm_errors.DCGM_FR_NCCL_ERROR, \
        f"Expected DCGM_FR_NCCL_ERROR ({dcgm_errors.DCGM_FR_NCCL_ERROR}), got {err.code}"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_nccl_tests_plugin.HE_Env("none_exit_0"))
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_zero_exit_no_output(handle, gpuIds):
    """
    Test that nccl_tests plugin fails when output is missing expected result lines.
    """
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nccl_tests")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nccl_test_idx = _assert_nccl_present(response)
    assert response.tests[nccl_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, \
        f"Expected FAIL, got {response.tests[nccl_test_idx].result}"
    assert response.tests[nccl_test_idx].numErrors > 0, "Expected at least one error"

    err_idx = response.tests[nccl_test_idx].errorIndices[0]
    err = response.errors[err_idx]
    assert err.code == dcgm_errors.DCGM_FR_NCCL_ERROR, \
        f"Expected DCGM_FR_NCCL_ERROR ({dcgm_errors.DCGM_FR_NCCL_ERROR}), got {err.code}"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_nccl_tests_plugin.HE_Env_Live())
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_live_pass(handle, gpuIds):
    """
    Test that nccl_tests plugin passes with real all_reduce_perf binary.
    """
    if not test_helpers.helper_nccl_tests_plugin.is_live_binary_available():
        test_utils.skip_test("Live all_reduce_perf binary not available")

    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="nccl_tests")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nccl_test_idx = _assert_nccl_present(response)
    assert response.tests[nccl_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_PASS, \
        f"Expected PASS, got {response.tests[nccl_test_idx].result}"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_helpers.helper_nccl_tests_plugin.HE_Env_Live({"NCCL_TESTS_MIN_BW": "99999"}))
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_nccl_tests_live_fail_min_bw(handle, gpuIds):
    """
    Test that nccl_tests plugin fails when minimum bandwidth threshold is unreachable.
    """
    if not test_helpers.helper_nccl_tests_plugin.is_live_binary_available():
        test_utils.skip_test("Live all_reduce_perf binary not available")

    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="nccl_tests")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nccl_test_idx = _assert_nccl_present(response)
    assert response.tests[nccl_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, \
        f"Expected FAIL, got {response.tests[nccl_test_idx].result}"
    assert response.tests[nccl_test_idx].numErrors > 0, "Expected at least one error"

    err_idx = response.tests[nccl_test_idx].errorIndices[0]
    err = response.errors[err_idx]
    assert err.code == dcgm_errors.DCGM_FR_NCCL_ERROR, \
        f"Expected DCGM_FR_NCCL_ERROR ({dcgm_errors.DCGM_FR_NCCL_ERROR}), got {err.code}"
