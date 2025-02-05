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
def test_inject_memory_error_counter(handle, gpuIds):
    gpuId = gpuIds[0]

    def mock_memory_error_counter(handle, gpuId, errorType, counterType, locationType, nvmlRet, value):
        injectedRet = nvml_injection.c_injectNvmlRet_t()
        injectedRet.nvmlRet = nvmlRet
        injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_ULONG_LONG
        injectedRet.values[0].value.ULongLong = value
        injectedRet.valueCount = 1

        extraKeysType = nvml_injection_structs.c_injectNvmlVal_t * 3
        extraKeys = extraKeysType()
        extraKeys[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_MEMORYERRORTYPE
        extraKeys[0].value.MemoryErrorType = errorType
        extraKeys[1].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_ECCCOUNTERTYPE
        extraKeys[1].value.EccCounterType = counterType
        extraKeys[2].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_MEMORYLOCATION
        extraKeys[2].value.MemoryLocation = locationType

        ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuId, "MemoryErrorCounter", extraKeys, 3, injectedRet)
        assert (ret == dcgm_structs.DCGM_ST_OK)

    def validate_ecc_values(handle, gpuId, fieldIds):
        entity = dcgm_structs.c_dcgmGroupEntityPair_t()
        entity.entityGroupId = dcgm_fields.DCGM_FE_GPU
        entity.entityId = gpuId

        responses = {}

        for fieldId in fieldIds:
            dictKey = "%d:%d:%d" % (dcgm_fields.DCGM_FE_GPU, gpuId, fieldId)
            responses[dictKey] = 0 #0 responses so far

            fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(handle, [entity], fieldIds, dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)
        
        for i, fieldValue in enumerate(fieldValues):
            assert(fieldValue.version == dcgm_structs.dcgmFieldValue_version2), "idx %d Version was x%X. not x%X" % (i, fieldValue.version, dcgm_structs.dcgmFieldValue_version2)
            dictKey = "%d:%d:%d" % (fieldValue.entityGroupId, fieldValue.entityId, fieldValue.fieldId)
            assert dictKey in responses and responses[dictKey] == 0, "Mismatch on dictKey %s. Responses: %s" % (dictKey, str(responses))
            assert(fieldValue.status == dcgm_structs.DCGM_ST_OK), "idx %d status was %d" % (i, fieldValue.status)
            assert(fieldValue.ts != 0), "idx %d timestamp was 0" % i
            assert(fieldValue.unused == 0), "idx %d unused was %d" % (i, fieldValue.unused)
            assert(fieldValue.fieldType == ord(dcgm_fields.DCGM_FT_INT64)), "Field %d type is wrong: %c %c" % (fieldValue.fieldId, fieldValue.fieldType, dcgm_fields.DCGM_FT_INT64)
            assert(fieldValue.value.i64 == fieldValue.fieldId), "Field %d value wrong: %d" % (fieldValue.fieldId, fieldValue.value.i64)
            responses[dictKey] += 1
        
    """
    We map the field Id to the set of nvmlDEviceGetMemoryErrorCounter index
    parameters.

    With the exception of fields that are supported bit Turing, and
    Turing and later, the following normally are only supported either
    pre-Turing or Turing and later. However, nvml-injection allows is to 
    fake-inject all of them to ensure the right parameters are passed to
    nvmlDeviceGetMemoryErrorCounter().
    """

    fieldMap = {  dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_DEV : # Turing & later
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY ],
                   
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_DEV : 
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY ],

                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_DEV : 
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY ],
                 
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_DEV : 
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY ],

                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_SRM : 
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_SRAM ],

                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_SRM : 
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_SRAM ],
                 
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_SRM :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_SRAM ],

                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_SRM : 
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_SRAM ],
                   
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_L1: # pre-Turing
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_L1_CACHE ],

                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_L1: 
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_L1_CACHE ],

                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_L1 :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_L1_CACHE ],

                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_L1 :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_L1_CACHE ],
    
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_L2 :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_L2_CACHE ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_L2 :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_L2_CACHE ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_L2 :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_L2_CACHE ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_L2 :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_L2_CACHE ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_REG :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_REGISTER_FILE ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_REG :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_REGISTER_FILE ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_REG :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_REGISTER_FILE ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_REG :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_REGISTER_FILE ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TEX :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_TEXTURE_MEMORY ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TEX :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_TEXTURE_MEMORY ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TEX :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_TEXTURE_MEMORY ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TEX :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_TEXTURE_MEMORY ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_SHM :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_TEXTURE_SHM ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_SHM :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_TEXTURE_SHM ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_SHM :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_TEXTURE_SHM ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_SHM :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_TEXTURE_SHM ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_CBU :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_CBU ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_CBU :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_VOLATILE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_CBU ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_CBU :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_CBU ],
                  
                  dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_CBU :
                  [ dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    dcgm_nvml.NVML_AGGREGATE_ECC,
                    dcgm_nvml.NVML_MEMORY_LOCATION_CBU ]
                }

    # We use the fieldId itself as the valid return value for the nvml call.
    fieldIds = []
    for fieldId, keys in fieldMap.items():
        mock_memory_error_counter(handle, gpuId, keys[0], keys[1], keys[2], dcgm_nvml.NVML_SUCCESS, fieldId)
        fieldIds.append(fieldId)

        
    validate_ecc_values(handle, gpuId, fieldIds)
