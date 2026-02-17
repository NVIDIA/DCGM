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

import DcgmDiag
import test_utils
import dcgm_structs
import dcgm_agent_internal
import nvml_injection
import nvml_injection_structs
import dcgm_nvml
import dcgm_errors
from _test_helpers import skip_test_if_no_dcgm_nvml


def mock_mem_copy_util(handle, gpuId, nvmlReturn, value):
    injectedRet = nvml_injection.c_injectNvmlRet_t()
    injectedRet.nvmlRet = nvmlReturn
    injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UTILIZATION
    injectedRet.values[0].value.Utilization.memory = value
    injectedRet.values[0].value.Utilization.gpu = 0
    injectedRet.valueCount = 1
    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(
        handle, gpuId, "UtilizationRates", None, 0, injectedRet)
    assert (ret == dcgm_structs.DCGM_ST_OK)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_current_system_injection_nvml(skuFileName="current_test_nvbandwidth_plugin_fail_if_mem_copy_util_is_larger_than_threshold.yaml")
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvbandwidth_plugin_fail_if_mem_copy_util_is_larger_than_threshold(handle, gpuIds):
    '''
    Test to verify that the nvbandwidth plugin fails if the memory copy util is larger than the threshold.
    '''
    mock_mem_copy_util(handle, gpuIds[0], dcgm_nvml.NVML_SUCCESS, 11)
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nvbandwidth",
                           paramsStr="nvbandwidth.is_allowed=true;nvbandwidth.testcases=0,1")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nvbandwidth_test_idx = 1
    assert response.numTests == 2
    assert response.tests[nvbandwidth_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL
    assert response.tests[nvbandwidth_test_idx].numErrors == 1
    errIdx = response.tests[nvbandwidth_test_idx].errorIndices[0]
    err = response.errors[errIdx]
    assert err.code == dcgm_errors.DCGM_FR_FIELD_THRESHOLD_TS
    assert err.category == dcgm_errors.DCGM_FR_EC_HARDWARE_MEMORY
    assert err.severity == dcgm_errors.DCGM_ERROR_ISOLATE
    assert err.msg.find(
        "is greater than 10%. This may affect the results of the nvbandwidth test.") != -1, f"Error message: {err.msg}"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_current_system_injection_nvml(skuFileName="current_test_nvbandwidth_plugin_will_not_fail_if_mem_copy_util_no_data.yaml")
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvbandwidth_plugin_will_not_fail_if_mem_copy_util_no_data(handle, gpuIds):
    '''
    Test to verify that the nvbandwidth plugin will not fail if the memory copy util is no data.
    '''
    mock_mem_copy_util(handle, gpuIds[0], dcgm_nvml.NVML_ERROR_UNKNOWN, 11)
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="nvbandwidth",
                           paramsStr="nvbandwidth.is_allowed=true;nvbandwidth.testcases=0,1")
    response = test_utils.diag_execute_wrapper(dd, handle)

    nvbandwidth_test_idx = 1
    assert response.numTests == 2
    assert response.tests[nvbandwidth_test_idx].result == dcgm_structs.DCGM_DIAG_RESULT_PASS
    assert response.tests[nvbandwidth_test_idx].numErrors == 0
