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
# test the nvml injection lib

import pydcgm
import dcgm_structs
import dcgm_agent_internal
import test_utils
import time
import dcgm_fields
from ctypes import *
import nvml_injection
import nvml_injection_structs
from _test_helpers import skip_test_if_no_dcgm_nvml, maybe_dcgm_nvml

def helper_inject_struct(handle, gpuIds):
    gpuId = gpuIds[0]
    injectedBar1Total = 12345 * 1024 * 1024
    injectedBar1Used = 45 * 1024 * 1024
    injectedBar1Free = 12300 * 1024 * 1024

    injectedRet = nvml_injection.c_injectNvmlRet_t()
    injectedRet.nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_BAR1MEMORY
    injectedRet.values[0].value.BAR1Memory.bar1Total = injectedBar1Total
    injectedRet.values[0].value.BAR1Memory.bar1Used = injectedBar1Used
    injectedRet.values[0].value.BAR1Memory.bar1Free = injectedBar1Free
    injectedRet.valueCount = 1

    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuId, "BAR1MemoryInfo", None, 0, injectedRet)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, dcgm_fields.DCGM_FI_DEV_BAR1_TOTAL, 60000000, 3600.0, 0)
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, dcgm_fields.DCGM_FI_DEV_BAR1_USED, 60000000, 3600.0, 0)
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, dcgm_fields.DCGM_FI_DEV_BAR1_FREE, 60000000, 3600.0, 0)
    dcgmSystem.UpdateAllFields(1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [dcgm_fields.DCGM_FI_DEV_BAR1_TOTAL, ])
    assert (values[0].value.i64 == 12345)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [dcgm_fields.DCGM_FI_DEV_BAR1_USED, ])
    assert (values[0].value.i64 == 45)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [dcgm_fields.DCGM_FI_DEV_BAR1_FREE, ])
    assert (values[0].value.i64 == 12300)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_inject_struct_standalone(handle, gpuIds):
    helper_inject_struct(handle, gpuIds)

def helper_inject_key_with_two_values(handle, gpuIds):
    gpuId = gpuIds[0]
    major = 2
    minor = 1

    injectedRet = nvml_injection.c_injectNvmlRet_t()
    injectedRet.nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRet.values[0].value.UInt = major
    injectedRet.values[1].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRet.values[1].value.UInt = minor
    injectedRet.valueCount = 2

    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuId, "CudaComputeCapability", None, 0, injectedRet)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    fieldId = dcgm_fields.DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, fieldId, 60000000, 3600.0, 0)
    dcgmSystem.UpdateAllFields(1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [fieldId, ])
    expected = major << 16 | minor
    assert (values[0].value.i64 == expected)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_inject_key_with_two_values_standalone(handle, gpuIds):
    helper_inject_key_with_two_values(handle, gpuIds)

def helper_inject_with_extra_keys(handle, gpuIds):
    gpuId = gpuIds[0]
    injectedVal = 123

    injectedRet = nvml_injection.c_injectNvmlRet_t()
    injectedRet.nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRet.values[0].value.UInt = injectedVal
    injectedRet.valueCount = 1

    extraKeys = nvml_injection_structs.c_injectNvmlVal_t()
    extraKeys.type = nvml_injection_structs.c_injectionArgType_t.INJECTION_CLOCKTYPE
    extraKeys.value.ClockType = 2
    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuId, "ApplicationsClock", extraKeys, 1, injectedRet)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    fieldId = dcgm_fields.DCGM_FI_DEV_APP_MEM_CLOCK
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, fieldId, 60000000, 3600.0, 0)
    dcgmSystem.UpdateAllFields(1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [fieldId, ])
    assert (values[0].value.i64 == injectedVal)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_inject_with_extra_keys_standalone(handle, gpuIds):
    helper_inject_with_extra_keys(handle, gpuIds)

def helper_inject_struct_for_following_calls(handle, gpuIds):
    gpuId = gpuIds[0]
    injectedBar1Total = 12345 * 1024 * 1024
    injectedBar1Used = 45 * 1024 * 1024
    injectedBar1Free = 12300 * 1024 * 1024

    injectedRetsArray = nvml_injection.c_injectNvmlRet_t * 2
    injectedRets = injectedRetsArray()
    injectedRets[0].nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRets[0].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_BAR1MEMORY
    injectedRets[0].values[0].value.BAR1Memory.bar1Total = injectedBar1Total
    injectedRets[0].values[0].value.BAR1Memory.bar1Used = injectedBar1Used
    injectedRets[0].values[0].value.BAR1Memory.bar1Free = injectedBar1Free
    injectedRets[0].valueCount = 1
    injectedRets[1].nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRets[1].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_BAR1MEMORY
    injectedRets[1].values[0].value.BAR1Memory.bar1Total = injectedBar1Total
    injectedRets[1].values[0].value.BAR1Memory.bar1Used = 0
    injectedRets[1].values[0].value.BAR1Memory.bar1Free = injectedBar1Total
    injectedRets[1].valueCount = 1

    ret = dcgm_agent_internal.dcgmInjectNvmlDeviceForFollowingCalls(handle, gpuId, "BAR1MemoryInfo", None, 0, injectedRets, 2)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, dcgm_fields.DCGM_FI_DEV_BAR1_TOTAL, 1000000, 0, 1)
    dcgmSystem.UpdateAllFields(1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [dcgm_fields.DCGM_FI_DEV_BAR1_TOTAL, ])
    assert (values[0].value.i64 == 12345)
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, dcgm_fields.DCGM_FI_DEV_BAR1_USED, 1000000, 0, 1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [dcgm_fields.DCGM_FI_DEV_BAR1_USED, ])
    dcgmSystem.UpdateAllFields(1)
    assert (values[0].value.i64 == 0)
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, dcgm_fields.DCGM_FI_DEV_BAR1_FREE, 1000000, 0, 1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [dcgm_fields.DCGM_FI_DEV_BAR1_FREE, ])
    dcgmSystem.UpdateAllFields(1)
    assert (values[0].value.i64 != 12345)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_inject_struct_for_following_calls_standalone(handle, gpuIds):
    helper_inject_struct_for_following_calls(handle, gpuIds)

def helper_inject_key_with_two_values_for_following_calls(handle, gpuIds):
    gpuId = gpuIds[0]
    major = 2
    minor = 1

    injectedRetsArray = nvml_injection.c_injectNvmlRet_t * 1
    injectedRets = injectedRetsArray()
    injectedRets[0].nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRets[0].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRets[0].values[0].value.UInt = major
    injectedRets[0].values[1].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRets[0].values[1].value.UInt = minor
    injectedRets[0].valueCount = 2

    ret = dcgm_agent_internal.dcgmInjectNvmlDeviceForFollowingCalls(handle, gpuId, "CudaComputeCapability", None, 0, injectedRets, 1)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    fieldId = dcgm_fields.DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, fieldId, 500000, 0, 1)
    dcgmSystem.UpdateAllFields(1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [fieldId, ])
    expected = major << 16 | minor
    assert (values[0].value.i64 == expected)
    time.sleep(1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [fieldId, ])
    assert (values[0].value.i64 != expected)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_inject_key_with_two_values_for_following_calls_standalone(handle, gpuIds):
    helper_inject_key_with_two_values_for_following_calls(handle, gpuIds)

def helper_inject_with_extra_keys_for_following_calls(handle, gpuIds):
    gpuId = gpuIds[0]
    injectedVal = 123

    injectedRetsArray = nvml_injection.c_injectNvmlRet_t * 1
    injectedRets = injectedRetsArray()
    injectedRets[0].nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRets[0].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRets[0].values[0].value.UInt = injectedVal
    injectedRets[0].valueCount = 1

    extraKeys = nvml_injection_structs.c_injectNvmlVal_t()
    extraKeys.type = nvml_injection_structs.c_injectionArgType_t.INJECTION_CLOCKTYPE
    extraKeys.value.ClockType = 2
    ret = dcgm_agent_internal.dcgmInjectNvmlDeviceForFollowingCalls(handle, gpuId, "ApplicationsClock", extraKeys, 1, injectedRets, 1)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    fieldId = dcgm_fields.DCGM_FI_DEV_APP_MEM_CLOCK
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgm_agent_internal.dcgmWatchFieldValue(dcgmHandle.handle, gpuId, fieldId, 60000000, 3600.0, 0)
    dcgmSystem.UpdateAllFields(1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(dcgmHandle.handle, gpuId, [fieldId, ])
    assert (values[0].value.i64 == injectedVal)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_inject_with_extra_keys_for_following_calls_standalone(handle, gpuIds):
    helper_inject_with_extra_keys_for_following_calls(handle, gpuIds)
