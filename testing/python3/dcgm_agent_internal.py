# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
##
# Python bindings for the internal API of DCGM library (dcgm_test_apis.h)
##

from ctypes import *
from ctypes.util import find_library
import dcgm_agent
import dcgm_structs
import dcgm_structs_internal


DCGM_EMBEDDED_HANDLE = c_void_p(0x7fffffff)

# Utils
_dcgmIntCheckReturn = dcgm_structs._dcgmCheckReturn
dcgmDeviceConfig_t  = dcgm_structs.c_dcgmDeviceConfig_v1
dcgmRecvUpdates_t = dcgm_structs._dcgmRecvUpdates_t
dcgmStatsFileType_t = dcgm_structs_internal._dcgmStatsFileType_t
dcgmInjectFieldValue_t = dcgm_structs_internal.c_dcgmInjectFieldValue_v1

""" 
Corresponding Calls
"""

@dcgm_agent.ensure_byte_strings()
def dcgmServerRun(portNumber, socketPath, isConnectionTcp):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmEngineRun")
    ret = fn(portNumber, socketPath, isConnectionTcp)
    _dcgmIntCheckReturn(ret)
    return ret

@dcgm_agent.ensure_byte_strings()
def dcgmGetLatestValuesForFields(dcgmHandle, gpuId, fieldIds):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmGetLatestValuesForFields")
    field_values = (dcgm_structs.c_dcgmFieldValue_v1 * len(fieldIds))()
    id_values = (c_uint * len(fieldIds))(*fieldIds)
    ret = fn(dcgmHandle, c_int(gpuId), id_values, c_uint(len(fieldIds)), field_values)
    _dcgmIntCheckReturn(ret)
    return field_values

@dcgm_agent.ensure_byte_strings()
def dcgmGetMultipleValuesForField(dcgmHandle, gpuId, fieldId, maxCount, startTs, endTs, order):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmGetMultipleValuesForField")
    localMaxCount = c_int(maxCount) #Going to pass by ref
    #Make space to return up to maxCount records
    max_field_values = (dcgm_structs.c_dcgmFieldValue_v1 * maxCount)()
    ret = fn(dcgmHandle, c_int(gpuId), c_uint(fieldId), byref(localMaxCount), c_int64(startTs), c_int64(endTs), c_uint(order), max_field_values)
    _dcgmIntCheckReturn(ret)
    localMaxCount = localMaxCount.value #Convert to int
    #We may have gotten less records back than we requested. If so, truncate our array
    return max_field_values[:int(localMaxCount)]

# This method is used to tell the cache manager to watch a field value
@dcgm_agent.ensure_byte_strings()
def dcgmWatchFieldValue(dcgmHandle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmWatchFieldValue")
    ret = fn(dcgmHandle, c_int(gpuId), c_uint(fieldId), c_longlong(updateFreq), c_double(maxKeepAge), c_int(maxKeepEntries))
    _dcgmIntCheckReturn(ret)
    return ret

# This method is used to tell the cache manager to unwatch a field value
def dcgmUnwatchFieldValue(dcgmHandle, gpuId, fieldId, clearCache):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmUnwatchFieldValue")
    ret = fn(dcgmHandle, c_int(gpuId), c_uint(fieldId), c_int(clearCache))
    _dcgmIntCheckReturn(ret)
    return ret

def dcgmInjectFieldValue(dcgmHandle, gpuId, value):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmInjectFieldValue")
    ret = fn(dcgmHandle, c_uint(gpuId), byref(value))
    _dcgmIntCheckReturn(ret)
    return ret

@dcgm_agent.ensure_byte_strings()
def dcgmInjectEntityFieldValue(dcgmHandle, entityGroupId, entityId, value):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmInjectEntityFieldValue")
    ret = fn(dcgmHandle, c_uint(entityGroupId), c_uint(entityId), byref(value))
    _dcgmIntCheckReturn(ret)
    return ret

@dcgm_agent.ensure_byte_strings()
def dcgmSetEntityNvLinkLinkState(dcgmHandle, entityGroupId, entityId, linkId, linkState):
    linkStateStruct = dcgm_structs_internal.c_dcgmSetNvLinkLinkState_v1()
    linkStateStruct.version = dcgm_structs_internal.dcgmSetNvLinkLinkState_version1
    linkStateStruct.entityGroupId = entityGroupId
    linkStateStruct.entityId = entityId
    linkStateStruct.linkId = linkId
    linkStateStruct.linkState = linkState
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmSetEntityNvLinkLinkState")
    ret = fn(dcgmHandle, byref(linkStateStruct))
    _dcgmIntCheckReturn(ret)
    return ret

@dcgm_agent.ensure_byte_strings()
def dcgmGetCacheManagerFieldInfo(dcgmHandle, gpuId, fieldId):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmGetCacheManagerFieldInfo")
    cmfi = dcgm_structs_internal.dcgmCacheManagerFieldInfo_v3()

    cmfi.gpuId = gpuId
    cmfi.fieldId = fieldId

    ret = fn(dcgmHandle, byref(cmfi))
    _dcgmIntCheckReturn(ret)
    return cmfi


@dcgm_agent.ensure_byte_strings()
def dcgmCreateFakeEntities(dcgmHandle, cfe):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmCreateFakeEntities")
    
    cfe.version = dcgm_structs_internal.dcgmCreateFakeEntities_version2

    ret = fn(dcgmHandle, byref(cfe))
    _dcgmIntCheckReturn(ret)

    return cfe


#First parameter below is the return type
dcgmFieldValueEnumeration_f = CFUNCTYPE(c_int32, c_uint32, POINTER(dcgm_structs.c_dcgmFieldValue_v1), c_int32, c_void_p)

@dcgm_agent.ensure_byte_strings()
def dcgmGetFieldValuesSince(dcgmHandle, groupId, sinceTimestamp, fieldIds, enumCB, userData):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmGetFieldValuesSince")
    c_fieldIds = (c_uint32 * len(fieldIds))(*fieldIds)
    c_nextSinceTimestamp = c_int64()
    ret = fn(dcgmHandle, groupId, c_int64(sinceTimestamp), c_fieldIds, c_int32(len(fieldIds)), byref(c_nextSinceTimestamp), enumCB, py_object(userData))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_nextSinceTimestamp.value

def dcgmMetadataStateSetRunInterval(dcgmHandle, runIntervalMs):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmMetadataStateSetRunInterval")
    ret = fn(dcgmHandle, c_uint(runIntervalMs))
    _dcgmIntCheckReturn(ret)

@dcgm_agent.ensure_byte_strings()
def dcgmIntrospectGetFieldExecTime(dcgm_handle, fieldId, waitIfNoData=True):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmIntrospectGetFieldExecTime")
    
    execTime = dcgm_structs.c_dcgmIntrospectFullFieldsExecTime_v2()
    execTime.version = dcgm_structs.dcgmIntrospectFullFieldsExecTime_version2
    
    ret = fn(dcgm_handle, fieldId, byref(execTime), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return execTime

@dcgm_agent.ensure_byte_strings()
def dcgmIntrospectGetFieldMemoryUsage(dcgm_handle, fieldId, waitIfNoData=True):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmIntrospectGetFieldMemoryUsage")
    
    memInfo = dcgm_structs.c_dcgmIntrospectFullMemory_v1()
    memInfo.version = dcgm_structs.dcgmIntrospectFullMemory_version1
    
    ret = fn(dcgm_handle, fieldId, byref(memInfo), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return memInfo

@dcgm_agent.ensure_byte_strings()
def dcgmVgpuConfigSet(dcgm_handle, group_id, configToSet, status_handle):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmVgpuConfigSet")
    configToSet.version = dcgm_structs.dcgmDeviceVgpuConfig_version1
    ret = fn(dcgm_handle, group_id, byref(configToSet), status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret

def dcgmVgpuConfigGet(dcgm_handle, group_id, reqCfgType, count, status_handle):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmVgpuConfigGet")

    vgpu_config_values_array = count * dcgm_structs.c_dcgmDeviceVgpuConfig_v1
    c_config_values = vgpu_config_values_array()

    for index in range(0, count):
        c_config_values[index].version = dcgm_structs.dcgmDeviceVgpuConfig_version1

    ret = fn(dcgm_handle, group_id, reqCfgType, count, c_config_values, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return list(c_config_values[0:count])

@dcgm_agent.ensure_byte_strings()
def dcgmVgpuConfigEnforce(dcgm_handle, group_id, status_handle):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmVgpuConfigEnforce")
    ret = fn(dcgm_handle, group_id, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret

@dcgm_agent.ensure_byte_strings()
def dcgmGetVgpuDeviceAttributes(dcgm_handle, gpuId):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmGetVgpuDeviceAttributes")
    device_values = dcgm_structs.c_dcgmVgpuDeviceAttributes_v6()
    device_values.version = dcgm_structs.dcgmVgpuDeviceAttributes_version6
    ret = fn(dcgm_handle, c_int(gpuId), byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values

@dcgm_agent.ensure_byte_strings()
def dcgmGetVgpuInstanceAttributes(dcgm_handle, vgpuId):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmGetVgpuInstanceAttributes")
    device_values = dcgm_structs.c_dcgmVgpuInstanceAttributes_v1()
    device_values.version = dcgm_structs.dcgmVgpuInstanceAttributes_version1
    ret = fn(dcgm_handle, c_int(vgpuId), byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values

def dcgmStopDiagnostic(dcgm_handle):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmStopDiagnostic")
    ret = fn(dcgm_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret
