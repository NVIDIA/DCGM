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

import pydcgm
import dcgm_field_helpers
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent
import dcgm_agent_internal
import logger
import test_utils
import dcgm_fields
import dcgm_fields_internal
import dcgm_nvml
import dcgmvalue
import time
import ctypes
import apps
from dcgm_structs import dcgmExceptionClass
from _test_helpers import skip_test_if_no_dcgm_nvml
import nvml_injection
import nvml_injection_structs
import dcgm_nvml
import utils
import os
from ctypes import *

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_inject_sram_threshold(handle, gpuIds):
    gpuId = gpuIds[0]

    def mock_sram_threshold_counter(handle, gpuId, threshold, nvmlRet):
        injectedRet = nvml_injection.c_injectNvmlRet_t()
        injectedRet.nvmlRet = nvmlRet
        injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_ECCSRAMERRORSTATUS
        injectedRet.values[0].value.EccSramErrorStatus.bThresholdExceeded = threshold
        injectedRet.valueCount = 1

        ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuId, "SramEccErrorStatus", None, 0, injectedRet)
        assert (ret == dcgm_structs.DCGM_ST_OK)

    def validate_sram_threshold(handle, gpuId, threshold):
        entity = dcgm_structs.c_dcgmGroupEntityPair_t()
        entity.entityGroupId = dcgm_fields.DCGM_FE_GPU
        entity.entityId = gpuId

        fieldIDs = [ dcgm_fields.DCGM_FI_DEV_THRESHOLD_SRM ]

        responses = {}

        for fieldId in fieldIDs:
            dictKey = "%d:%d:%d" % (dcgm_fields.DCGM_FE_GPU, gpuId, fieldId)
            responses[dictKey] = 0 #0 responses so far

        fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(handle, [entity], fieldIDs, dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)
        
        for i, fieldValue in enumerate(fieldValues):
            assert(fieldValue.version == dcgm_structs.dcgmFieldValue_version2), "idx %d Version was x%X. not x%X" % (i, fieldValue.version, dcgm_structs.dcgmFieldValue_version2)
            dictKey = "%d:%d:%d" % (fieldValue.entityGroupId, fieldValue.entityId, fieldValue.fieldId)
            assert dictKey in responses and responses[dictKey] == 0, "Mismatch on dictKey %s. Responses: %s" % (dictKey, str(responses))
            assert(fieldValue.status == dcgm_structs.DCGM_ST_OK), "idx %d status was %d" % (i, fieldValue.status)
            assert(fieldValue.ts != 0), "idx %d timestamp was 0" % i
            assert(fieldValue.unused == 0), "idx %d unused was %d" % (i, fieldValue.unused)
            assert(fieldValue.fieldType == ord(dcgm_fields.DCGM_FT_INT64)), "Field %d type is wrong: %c %c" % (fieldValue.fieldId, fieldValue.fieldType, dcgm_fields.DCGM_FT_INT64)
            assert(fieldValue.value.i64 == threshold), "Field %d value wrong: %d != %d" % (fieldValue.fieldId, fieldValue.value.i64, threshold)
            responses[dictKey] += 1

    threshold = 1337
    
    status = dcgm_nvml.c_nvmlEccSramErrorStatus_v1_t()
    
    status.bThresholdExceeded = threshold # Technically a bool, but we verify int

    mock_sram_threshold_counter(handle, gpuId, threshold, dcgm_nvml.NVML_SUCCESS)
    
    validate_sram_threshold(handle, gpuId, threshold)
