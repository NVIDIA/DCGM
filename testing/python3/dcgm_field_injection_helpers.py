
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
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_fields
import dcgm_structs

import time

# Stores the parameters in a field value of type DCGM_FT_INT64
def get_field_value_i64(fieldId, value, offset, entityGroupId=dcgm_fields.DCGM_FE_GPU):
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = fieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()+offset) * 1000000.0)
    field.value.i64 = value

    return field

# Stores the parameters in a field value of type DCGM_FT_DOUBLE
def get_field_value_fp64(fieldId, value, offset, entityGroupId=dcgm_fields.DCGM_FE_GPU):
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = fieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_DOUBLE)
    field.ts = int((time.time()+offset) * 1000000.0)
    field.value.dbl = value

    return field

'''
        inject_nvml_value - injects a value into injection NVML

    handle    - the handle to DCGM
    gpuId     - the id of the GPU we're injecting
    fieldId   - the DCGM field id of what we're injecting into NVML
    value     - the value we're injecting
    offset    - the offset in seconds for the timestamp the value should have
'''
def inject_nvml_value(handle, gpuId, fieldId, value, offset):
    fieldType = dcgm_fields.DcgmFieldGetById(fieldId).fieldType
    if fieldType == dcgm_fields.DCGM_FT_INT64:
        field = get_field_value_i64(fieldId, value, offset)
        ret = dcgm_agent_internal.dcgmInjectEntityFieldValueToNvml(handle, dcgm_fields.DCGM_FE_GPU, gpuId, field)
    else:
        field = get_field_value_fp64(fieldId, value, offset)
        ret = dcgm_agent_internal.dcgmInjectEntityFieldValueToNvml(handle, dcgm_fields.DCGM_FE_GPU, gpuId, field)

    return ret

'''
        inject_value - injects a value into DCGM's cache

    handle          - the handle to DCGM
    entityId        - the id of the entity we're injecting the value for
    fieldId         - the id of the field we're injecting a value into
    value           - the value we're injecting
    offset          - the offset - in seconds - for the timestamp the value should have
    verifyInsertion - True if we should fail if the value couldn't be injected, False = ignore. (default to True)
    entityType      - the type of entity we're injecting the value for, defaults to GPU
    repeatCount     - the number of repeated times we should inject the value, defaults to 0, meaning 1 injection
    repeatOffset    - how many seconds to increment the offset by in each subsequent injection
'''
def inject_value(handle, entityId, fieldId, value, offset, verifyInsertion=True,
                 entityType=dcgm_fields.DCGM_FE_GPU, repeatCount=0, repeatOffset=1):
    fieldType = dcgm_fields.DcgmFieldGetById(fieldId).fieldType

    if fieldType == dcgm_fields.DCGM_FT_INT64:
        ret = inject_field_value_i64(handle, entityId, fieldId, value, offset, entityGroupId=entityType)

        for i in range(0, repeatCount):
            if ret != dcgm_structs.DCGM_ST_OK:
                # Don't continue inserting if it isn't working
                break

            offset = offset + repeatOffset
            ret = inject_field_value_i64(handle, entityId, fieldId, value, offset, entityGroupId=entityType)

    elif fieldType == dcgm_fields.DCGM_FT_DOUBLE:

        ret = inject_field_value_fp64(handle, entityId, fieldId, value, offset, entityGroupId=entityType)
        for i in range(0, repeatCount):
            if ret != dcgm_structs.DCGM_ST_OK:
                # Don't continue inserting if it isn't working
                break

            offset = offset + repeatOffset
            ret = inject_field_value_fp64(handle, entityId, fieldId, value, offset, entityGroupId=entityType)
    else:
        assert False, "Cannot inject field type '%s', only INT64 and DOUBLE are supported" % fieldType

    if verifyInsertion:
        assert ret == dcgm_structs.DCGM_ST_OK, "Could not inject value %s in field id %s" % (value, fieldId)
    return ret

# Injects a field value of type DCGM_FT_INT64 into DCGM's cache
def inject_field_value_i64(handle, entityId, fieldId, value, offset, entityGroupId=dcgm_fields.DCGM_FE_GPU):
    field = get_field_value_i64(fieldId, value, offset, entityGroupId)

    return dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, entityGroupId, entityId, field)

# Injects a field value of type DCGM_FT_DOUBLE into DCGM's cache
def inject_field_value_fp64(handle, entityId, fieldId, value, offset, entityGroupId=dcgm_fields.DCGM_FE_GPU):
    field = get_field_value_fp64(fieldId, value, offset, entityGroupId)

    return dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, entityGroupId, entityId, field)
