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
# APIs that use a versioned structure from dcgm_agent.py:

# dcgmConfigGet
# dcgmConfigSet
# dcgmConnect_v2
# dcgmFieldGroupGetAll - Bug found
# dcgmFieldGroupGetInfo
# dcgmGetDeviceAttributes
# dcgmGetPidInfo - Bug found
# dcgmGroupGetInfo
# dcgmHealthCheck 
# dcgmIntrospectGetFieldsExecTime - Bug found
# vtDcgmIntrospectGetFieldsMemoryUsage - Bug found
# dcgmIntrospectGetHostengineCpuUtilization
# dcgmIntrospectGetHostengineMemoryUsage
# dcgmJobGetStats
# dcgmPolicyGet
# dcgmRunDiagnostic

# APIs that use a versioned structure from dcgm_agent_internal:

# vtDcgmGetVgpuDeviceAttributes
# dcgmGetVgpuInstanceAttributes
# dcgmIntrospectGetFieldExecTime
# dcgmIntrospectGetFieldMemoryUsage
# dcgmVgpuConfigGet
# dcgmVgpuConfigSet
##

# If a new API that uses a versioned structure is added, the corresponding test should be added in this file

import apps
import logger
import test_utils
import dcgm_agent
import dcgm_agent_internal
import pydcgm
import dcgm_structs
import dcgm_fields
from ctypes import *
from dcgm_structs import dcgmExceptionClass

# Provides access to functions from dcgm_agent_internal
dcgmFP = dcgm_structs._dcgmGetFunctionPointer

def vtDcgmConnect_v2(ip_address, connectParams, versionTest):
    connectParams = dcgm_structs.c_dcgmConnectV2Params_v1()
    connectParams.version = dcgm_structs.make_dcgm_version(connectParams, 1)
    logger.debug("Structure version: %d" % connectParams.version)
    connectParams.version = versionTest
    dcgm_handle = c_void_p()
    fn = dcgmFP("dcgmConnect_v2")
    ret = fn(ip_address, byref(connectParams), byref(dcgm_handle))
    dcgm_structs._dcgmCheckReturn(ret)
    return dcgm_handle

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_connect_validate(handle, gpuIds):

    """
    Validates structure version
    """
    fieldGroupFieldIds = [dcgm_fields.DCGM_FI_DEV_GPU_TEMP, ]
    connectParams = dcgm_structs.c_dcgmConnectV2Params_v1()
    connectParams.persistAfterDisconnect = 0
    
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmConnect_v2('localhost', connectParams, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmConnect_v2('localhost', connectParams, versionTest)
        
def vtDcgmGetDeviceAttributes(dcgm_handle, gpuId, versionTest):
    fn = dcgmFP("dcgmGetDeviceAttributes")
    device_values = dcgm_structs.c_dcgmDeviceAttributes_v1()
    device_values.version = dcgm_structs.make_dcgm_version(device_values, 1)
    logger.debug("Structure version: %d" % device_values.version)

    device_values.version = versionTest
    ret = fn(dcgm_handle, c_int(gpuId), byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_get_device_attributes_validate(handle, gpuIds):
    """
    Validates structure version
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    ## Add first GPU to the group
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against

    #Make sure the device attributes and config fields have updated
    systemObj.UpdateAllFields(1)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmGetDeviceAttributes(handle, gpuIds[0], versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random invalid version
        ret = vtDcgmGetDeviceAttributes(handle, gpuIds[0], versionTest)

def vtDcgmGroupGetInfo(dcgm_handle, group_id, versionTest):
    fn = dcgmFP("dcgmGroupGetInfo")
    device_values = dcgm_structs.c_dcgmGroupInfo_v2()
    device_values.version = dcgm_structs.make_dcgm_version(device_values, 2)
    logger.debug("Structure version: %d" % device_values.version)

    device_values.version = versionTest
    ret = fn(dcgm_handle, group_id, byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values

@test_utils.run_with_embedded_host_engine()
def test_dcgm_group_get_info_validate(handle):

    """
    Validates structure version
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmGroupGetInfo(handle, groupId, versionTest)
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmGroupGetInfo(handle, groupId, versionTest)

def vtDcgmFieldGroupGetInfo(dcgm_handle, fieldGroupId, versionTest):
    
    c_fieldGroupInfo = dcgm_structs.c_dcgmFieldGroupInfo_v1()
    c_fieldGroupInfo.version = dcgm_structs.make_dcgm_version(c_fieldGroupInfo, 1)
    logger.debug("Structure version: %d" % c_fieldGroupInfo.version)

    c_fieldGroupInfo.version = versionTest
    c_fieldGroupInfo.fieldGroupId = fieldGroupId
    fn = dcgmFP("dcgmFieldGroupGetInfo")
    ret = fn(dcgm_handle, byref(c_fieldGroupInfo))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_fieldGroupInfo

@test_utils.run_with_embedded_host_engine()
def test_dcgm_field_group_get_info_validate(handle):

    """
    Validates structure version
    """
    fieldIds = [dcgm_fields.DCGM_FI_DRIVER_VERSION, dcgm_fields.DCGM_FI_DEV_NAME, dcgm_fields.DCGM_FI_DEV_BRAND]
    handle = pydcgm.DcgmHandle(handle)
    fieldGroup = pydcgm.DcgmFieldGroup(handle, "mygroup", fieldIds)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmFieldGroupGetInfo(handle.handle, fieldGroup.fieldGroupId, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmFieldGroupGetInfo(handle.handle, fieldGroup.fieldGroupId, versionTest)

def vtDcgmFieldGroupGetAll(dcgm_handle, versionTest):
    c_allGroupInfo = dcgm_structs.c_dcgmAllFieldGroup_v1()
    c_allGroupInfo.version = dcgm_structs.make_dcgm_version(c_allGroupInfo, 1)
    logger.debug("Structure version: %d" % c_allGroupInfo.version)

    c_allGroupInfo.version = versionTest
    fn = dcgmFP("dcgmFieldGroupGetAll")
    ret = fn(dcgm_handle, byref(c_allGroupInfo))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_allGroupInfo

@test_utils.run_with_embedded_host_engine()
def test_dcgm_field_group_get_all_validate(handle):

    """
    Validates structure version
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert len(gpuIdList) >= 0, "Not able to find devices on the node for embedded case"
    
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        vtDcgmFieldGroupGetAll(handle, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        vtDcgmFieldGroupGetAll(handle, versionTest)

def vtDcgmConfigSet(dcgm_handle, group_id, configToSet, status_handle, versionTest):
    fn = dcgmFP("dcgmConfigSet")
    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.version = dcgm_structs.make_dcgm_version(config_values, 1)
    logger.debug("Structure version: %d" % config_values.version)
    configToSet.version = versionTest
    ret = fn(dcgm_handle, group_id, byref(configToSet), status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret

@test_utils.run_with_embedded_host_engine()
def test_dcgm_config_set_validate(handle):
    """
    Validates structure version
    """
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")
    status_handle = dcgm_agent.dcgmStatusCreate()
    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmConfigSet(handle,groupId,config_values, status_handle, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random invalid version
        ret = vtDcgmConfigSet(handle,groupId,config_values, status_handle, versionTest)

def vtDcgmConfigGet(dcgm_handle, group_id, reqCfgType, count, status_handle, versionTest):
    fn = dcgmFP("dcgmConfigGet")

    config_values_array = count * dcgm_structs.c_dcgmDeviceConfig_v1
    c_config_values = config_values_array()

    for index in range(0, count):
        c_config_values[index].version = versionTest

    ret = fn(dcgm_handle, group_id, reqCfgType, count, c_config_values, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return list(c_config_values[0:count])

@test_utils.run_with_embedded_host_engine()
def test_dcgm_config_get_validate(handle):

    """
    Validates structure version
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert len(gpuIdList) >= 0, "Not able to find devices on the node for embedded case"
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")
    groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId)
    status_handle = dcgm_agent.dcgmStatusCreate()

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle, versionTest)

def vtDcgmPolicyGet(dcgm_handle, group_id, count, status_handle, versionTest):
    fn = dcgmFP("dcgmPolicyGet")
    policy_array = count * dcgm_structs.c_dcgmPolicy_v1

    c_policy_values = policy_array()

    policy = dcgm_structs.c_dcgmPolicy_v1()
    policy.version = dcgm_structs.make_dcgm_version(policy, 1)
    logger.debug("Structure version: %d" % policy.version)

    policyCallback = dcgm_structs.c_dcgmPolicyCallbackResponse_v1()
    policyCallback.version = dcgm_structs.make_dcgm_version(policyCallback, 1)
    logger.debug("Structure version: %d" % policyCallback.version)

    for index in range(0, count):
        c_policy_values[index].version = versionTest

    ret = fn(dcgm_handle, group_id, count, c_policy_values, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return c_policy_values[0:count]

@test_utils.run_with_embedded_host_engine()
def test_dcgm_policy_get_validate(handle):
    
    """
    Validates structure version
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert len(gpuIdList) >= 0, "Not able to find devices on the node for embedded case"
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")
    status_handle = dcgm_agent.dcgmStatusCreate()
    count = 1

    diagLevel = dcgm_structs.DCGM_DIAG_LVL_SHORT

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmPolicyGet(handle, groupId, count, status_handle, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmPolicyGet(handle, groupId, count, status_handle, versionTest)

def vtDcgmHealthCheck(dcgm_handle, groupId, versionTest):
    c_results = dcgm_structs.c_dcgmHealthResponse_v4()
    c_results.version = dcgm_structs.make_dcgm_version(c_results, 4)
    logger.debug("Structure version: %d" % c_results.version)

    c_results.version = versionTest
    fn = dcgmFP("dcgmHealthCheck")
    ret = fn(dcgm_handle, groupId, byref(c_results))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_results

@test_utils.run_with_embedded_host_engine()
def test_dcgm_health_check_validate(handle):

    """
    Validates structure version
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmHealthCheck(handle, groupId, versionTest)
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmHealthCheck(handle, groupId, versionTest)

def vtDcgmActionValidate_v2(dcgm_handle, runDiagInfo, versionTest):
    response = dcgm_structs.c_dcgmDiagResponse_v6()
    response.version = dcgm_structs.make_dcgm_version(response, 6)
    logger.debug("Structure version: %d" % response.version)

    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v7()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version7
    logger.debug("Structure version: %d" % runDiagInfo.version)

    runDiagInfo.version = versionTest
    response.version = versionTest
    fn = dcgmFP("dcgmActionValidate_v2")
    ret = fn(dcgm_handle, byref(runDiagInfo), byref(response))
    dcgm_structs._dcgmCheckReturn(ret)
    return response

def vtDcgmActionValidate(dcgm_handle, group_id, validate, versionTest):
    response = dcgm_structs.c_dcgmDiagResponse_v6()
    response.version = versionTest
    
    # Put the group_id and validate into a dcgmRunDiag struct
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v7()
    runDiagInfo.version = versionTest
    runDiagInfo.validate = validate
    runDiagInfo.groupId = group_id

    fn = dcgmFP("dcgmActionValidate_v2")
    ret = fn(dcgm_handle, byref(runDiagInfo), byref(response))
    dcgm_structs._dcgmCheckReturn(ret)
    return response

def vtDcgmRunDiagnostic(dcgm_handle, group_id, diagLevel, versionTest):
    response = dcgm_structs.c_dcgmDiagResponse_v6()
    response.version = versionTest
    fn = dcgmFP("dcgmRunDiagnostic")
    ret = fn(dcgm_handle, group_id, diagLevel, byref(response))
    dcgm_structs._dcgmCheckReturn(ret)
    return response

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_with_max_power_limit_set()
def test_dcgm_run_diagnostic_validate(handle, gpuIds):

    """
    Validates structure version
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert len(gpuIdList) >= 0, "Not able to find devices on the node for embedded case"
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")
    groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId)
    status_handle = dcgm_agent.dcgmStatusCreate()

    diagLevel = dcgm_structs.DCGM_DIAG_LVL_SHORT

    gpuIdStr = ""
    for i, gpuId in enumerate(gpuIds):
        if i > 0:
            gpuIdStr += ","
        gpuIdStr += str(gpuId)

    drd = dcgm_structs.c_dcgmRunDiag_t()
    drd.version = dcgm_structs.dcgmRunDiag_version
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    drd.groupId = groupId
    drd.gpuList = gpuIdStr

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmActionValidate_v2(handle, drd, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmActionValidate_v2(handle, drd, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmActionValidate(handle, drd.groupId, drd.validate, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmActionValidate(handle, drd.groupId, drd.validate, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmRunDiagnostic(handle, drd.groupId, diagLevel, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmRunDiagnostic(handle, drd.groupId, diagLevel, versionTest)

def vtDcgmGetPidInfo(dcgm_handle, groupId, pid, versionTest):
    fn = dcgmFP("dcgmGetPidInfo")
    pidInfo = dcgm_structs.c_dcgmPidInfo_v2()
    pidInfo.version = dcgm_structs.make_dcgm_version(dcgm_structs.c_dcgmPidInfo_v2, 2)
    logger.debug("Structure version: %d" % pidInfo.version)

    pidInfo.version = versionTest
    pidInfo.pid = pid

    ret = fn(dcgm_handle, groupId, byref(pidInfo))
    dcgm_structs._dcgmCheckReturn(ret)
    return pidInfo

def StartAppOnGpus(handle):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    allGpuIds = dcgmSystem.discovery.GetAllGpuIds()

    gpuInfoList = []
    addedPids = []

    for gpuId in allGpuIds:
        gpuAttrib = dcgmSystem.discovery.GetGpuAttributes(gpuId)
        gpuInfoList.append((gpuId, gpuAttrib.identifiers.pciBusId))

    for info in gpuInfoList:
        gpuId = info[0]
        busId = info[1]
        appTimeout = int(1000) #miliseconds

        #Start a cuda app so we have something to accounted
        appParams = ["--ctxCreate", busId,
                        "--busyGpu", busId, str(appTimeout),
                        "--ctxDestroy", busId]
        app = apps.CudaCtxCreateAdvancedApp(appParams, env=test_utils.get_cuda_visible_devices_env(handle, gpuId))
        app.start(appTimeout*2)
        pid = app.getpid()
        addedPids.append(pid)
        app.wait()
        app.terminate()
        app.validate()
        logger.info("Started PID %d." % pid)

    return addedPids

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_get_pid_info_validate(handle, gpuIds):

    """
    Validates structure version
    """
    
    pidList = StartAppOnGpus(handle)
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")

    for pid in pidList:
        with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
            versionTest = 0 #invalid version
            ret = vtDcgmGetPidInfo(handle, groupId, pid, versionTest)

        with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
            versionTest = 50 #random number version
            ret = vtDcgmGetPidInfo(handle, groupId, pid, versionTest)

def vtDcgmJobGetStats(dcgm_handle, jobid, versionTest):
    fn = dcgmFP("dcgmJobGetStats")
    jobInfo = dcgm_structs.c_dcgmJobInfo_v3()
    jobInfo.version = dcgm_structs.make_dcgm_version(jobInfo, 3)
    logger.debug("Structure version: %d" % jobInfo.version)

    jobInfo.version = versionTest

    ret = fn(dcgm_handle, jobid, byref(jobInfo))
    dcgm_structs._dcgmCheckReturn(ret)
    return jobInfo

@test_utils.run_with_embedded_host_engine()
def test_dcgm_job_get_stats_validate(handle):
    """
    Validates structure version
    """
    
    jobid = "1"

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmJobGetStats(handle, jobid, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmJobGetStats(handle, jobid, versionTest)

def vtDcgmIntrospectGetHostengineMemoryUsage(dcgm_handle, versionTest, waitIfNoData=True):
    fn = dcgmFP("dcgmIntrospectGetHostengineMemoryUsage")
    
    memInfo = dcgm_structs.c_dcgmIntrospectMemory_v1()
    memInfo.version = dcgm_structs.make_dcgm_version(memInfo, 1)
    logger.debug("Structure version: %d" % memInfo.version)

    memInfo.version = versionTest
    
    ret = fn(dcgm_handle, byref(memInfo), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return memInfo
    
@test_utils.run_with_embedded_host_engine()
def test_dcgm_introspect_get_hostengine_memory_usage_validate(handle):
    
    """
    Validates structure version
    """

    waitIfNoData = True

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmIntrospectGetHostengineMemoryUsage(handle, versionTest, waitIfNoData)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmIntrospectGetHostengineMemoryUsage(handle, versionTest, waitIfNoData)

def vtDcgmIntrospectGetHostengineCpuUtilization(dcgm_handle, versionTest , waitIfNoData=True):
    fn = dcgmFP("dcgmIntrospectGetHostengineCpuUtilization")
    
    cpuUtil = dcgm_structs.c_dcgmIntrospectCpuUtil_v1()
    cpuUtil.version = dcgm_structs.make_dcgm_version(cpuUtil, 1)
    logger.debug("Structure version: %d" % cpuUtil.version)

    cpuUtil.version = versionTest
    
    ret = fn(dcgm_handle, byref(cpuUtil), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return cpuUtil

@test_utils.run_with_embedded_host_engine()
def test_dcgm_introspect_get_hostengine_cpu_utilization_validate(handle):
    
    """
    Validates structure version
    """

    waitIfNoData = True

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmIntrospectGetHostengineCpuUtilization(handle, versionTest, waitIfNoData)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmIntrospectGetHostengineCpuUtilization(handle, versionTest, waitIfNoData)


def vtDcgmIntrospectGetFieldsExecTime(dcgm_handle, introspectContext, versionTest, waitIfNoData=True):
    fn = dcgmFP("dcgmIntrospectGetFieldsExecTime")

    
    execTime = dcgm_structs.c_dcgmIntrospectFieldsExecTime_v1()
    execTime.version = dcgm_structs.make_dcgm_version(execTime, 1)
    logger.debug("Structure version: %d" % execTime.version)
    

    fullExecTime = dcgm_structs.c_dcgmIntrospectFullFieldsExecTime_v2()
    fullExecTime.version = dcgm_structs.make_dcgm_version(fullExecTime, 2)
    logger.debug("Structure version: %d" % fullExecTime.version)

    fullExecTime.version = versionTest
    execTime.version = versionTest

    introspectContext = dcgm_structs.c_dcgmIntrospectContext_v1()
    introspectContext.version = dcgm_structs.make_dcgm_version(introspectContext, 1)
    logger.debug("Structure version: %d" % introspectContext.version)

    introspectContext.version = versionTest

    ret = fn(dcgm_handle, byref(introspectContext), byref(execTime), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return execTime

@test_utils.run_with_embedded_host_engine()
def test_dcgm_introspect_get_fields_exec_time_validate(handle):
    
    """
    Validates structure version
    """
    introspectContext = dcgm_structs.c_dcgmIntrospectContext_v1()
    waitIfNoData = True

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmIntrospectGetFieldsExecTime(handle, introspectContext, versionTest, waitIfNoData)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmIntrospectGetFieldsExecTime(handle, introspectContext, versionTest, waitIfNoData)

def vtDcgmIntrospectGetFieldsMemoryUsage(dcgm_handle, introspectContext, versionTest, waitIfNoData=True):
    fn = dcgmFP("dcgmIntrospectGetFieldsMemoryUsage")
    
    memInfo = dcgm_structs.c_dcgmIntrospectFullMemory_v1()
    memInfo.version = dcgm_structs.make_dcgm_version(memInfo, 1)
    logger.debug("Structure version: %d" % memInfo.version)

    memInfo.version = versionTest
    
    introspectContext = dcgm_structs.c_dcgmIntrospectContext_v1()
    introspectContext.version = versionTest

    ret = fn(dcgm_handle, byref(introspectContext), byref(memInfo), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return memInfo

@test_utils.run_with_embedded_host_engine()
def test_dcgm_introspect_get_fields_memory_usage_validate(handle):
    
    """
    Validates structure version
    """
    introspectContext = dcgm_structs.c_dcgmIntrospectContext_v1()
    waitIfNoData = True

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmIntrospectGetFieldsMemoryUsage(handle, introspectContext, versionTest, waitIfNoData)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmIntrospectGetFieldsMemoryUsage(handle, introspectContext, versionTest, waitIfNoData)

########### dcgm_agent_internal.py ###########

def vtDcgmGetVgpuDeviceAttributes(dcgm_handle, gpuId, versionTest):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmGetVgpuDeviceAttributes")
    device_values = dcgm_structs.c_dcgmVgpuDeviceAttributes_v6()
    device_values.version = dcgm_structs.make_dcgm_version(device_values, 1)
    logger.debug("Structure version: %d" % device_values.version)

    device_values.version = versionTest
    ret = fn(dcgm_handle, c_int(gpuId), byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_get_vgpu_device_attributes_validate(handle, gpuIds):
    """
    Verifies that vGPU attributes are properly queried
    """

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmGetVgpuDeviceAttributes(handle, gpuIds[0], versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmGetVgpuDeviceAttributes(handle, gpuIds[0], versionTest)


def vtDcgmGetVgpuInstanceAttributes(dcgm_handle, vgpuId, versionTest):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmGetVgpuInstanceAttributes")
    device_values = dcgm_structs.c_dcgmVgpuInstanceAttributes_v1()
    device_values.version = dcgm_structs.make_dcgm_version(device_values, 1)
    logger.debug("Structure version: %d" % device_values.version)

    device_values.version = versionTest
    ret = fn(dcgm_handle, c_int(vgpuId), byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_get_vgpu_instance_attributes_validate(handle, gpuIds):
    """
    Verifies that vGPU attributes are properly queried
    """

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmGetVgpuInstanceAttributes(handle, gpuIds[0], versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmGetVgpuInstanceAttributes(handle, gpuIds[0], versionTest)

def vtDcgmVgpuConfigSet(dcgm_handle, group_id, configToSet, status_handle, versionTest):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmVgpuConfigSet")
    configToSet.version = versionTest
    ret = fn(dcgm_handle, group_id, byref(configToSet), status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret

@test_utils.run_with_embedded_host_engine()
def test_dcgm_vgpu_config_set_validate(handle):
    """
    Validates structure version
    """
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")
    status_handle = dcgm_agent.dcgmStatusCreate()
    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmVgpuConfigSet(handle, groupId, config_values, status_handle, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random invalid version
        ret = vtDcgmVgpuConfigSet(handle, groupId, config_values, status_handle, versionTest)


def vtDcgmVgpuConfigGet(dcgm_handle, group_id, reqCfgType, count, status_handle, versionTest):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmVgpuConfigSet")

    vgpu_config_values_array = count * dcgm_structs.c_dcgmDeviceVgpuConfig_v1
    c_config_values = vgpu_config_values_array()
    
    vgpuConfig = dcgm_structs.c_dcgmDeviceVgpuConfig_v1()
    vgpuConfig.version = dcgm_structs.make_dcgm_version(vgpuConfig, 1)
    logger.debug("Structure version: %d" % vgpuConfig.version)

    for index in range(0, count):
        c_config_values[index].version = versionTest

    ret = fn(dcgm_handle, group_id, c_config_values, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return list(c_config_values[0:count])


@test_utils.run_with_embedded_host_engine()
def test_dcgm_vgpu_config_get_validate(handle):

    """
    Validates structure version
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert len(gpuIdList) >= 0, "Not able to find devices on the node for embedded case"
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "test1")
    groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId)
    status_handle = dcgm_agent.dcgmStatusCreate()

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmVgpuConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle, versionTest)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmVgpuConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle, versionTest)


def vtDcgmIntrospectGetFieldExecTime(dcgm_handle, fieldId, versionTest, waitIfNoData=True):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmIntrospectGetFieldExecTime")
    
    execTime = dcgm_structs.c_dcgmIntrospectFullFieldsExecTime_v2()
    execTime.version = versionTest
    
    ret = fn(dcgm_handle, fieldId, byref(execTime), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return execTime

@test_utils.run_with_embedded_host_engine()
def test_dcgm_introspect_get_field_exec_time_validate(handle):
    
    """
    Validates structure version
    """
    fieldId = dcgm_fields.DCGM_FI_DEV_GPU_TEMP
    waitIfNoData = True

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmIntrospectGetFieldExecTime(handle, fieldId, versionTest, waitIfNoData)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmIntrospectGetFieldExecTime(handle, fieldId, versionTest, waitIfNoData)


def vtDcgmIntrospectGetFieldMemoryUsage(dcgm_handle, fieldId, versionTest, waitIfNoData=True):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmIntrospectGetFieldMemoryUsage")
    
    memInfo = dcgm_structs.c_dcgmIntrospectFullMemory_v1()
    memInfo.version = versionTest
    
    ret = fn(dcgm_handle, fieldId, byref(memInfo), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return memInfo

@test_utils.run_with_embedded_host_engine()
def test_dcgm_introspect_get_field_memory_usage_validate(handle):
    
    """
    Validates structure version
    """
    fieldId = dcgm_fields.DCGM_FI_DEV_GPU_TEMP
    waitIfNoData = True

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 0 #invalid version
        ret = vtDcgmIntrospectGetFieldMemoryUsage(handle, fieldId, versionTest, waitIfNoData)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        versionTest = 50 #random number version
        ret = vtDcgmIntrospectGetFieldMemoryUsage(handle, fieldId, versionTest, waitIfNoData)
