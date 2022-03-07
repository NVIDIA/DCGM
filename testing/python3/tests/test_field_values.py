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

import pydcgm
import dcgm_field_helpers
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent
import dcgm_agent_internal
import logger
import test_utils
import dcgm_fields
import dcgmvalue
import time
import ctypes
import apps
from dcgm_structs import dcgmExceptionClass
import utils
import os

g_profilingFieldIds = [
    dcgm_fields.DCGM_FI_PROF_GR_ENGINE_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
    dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE,
    dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES,
    dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES,
    dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES,
    dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES
]

def get_usec_since_1970():
    sec = time.time()
    return int(sec * 1000000.0)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_field_values_since_agent(handle, gpuIds):

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    ## Add first GPU to the group
    gpuId = gpuIds[0]
    groupObj.AddGpu(gpuId)
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against

    expectedValueCount = 0

    #Make a base value that is good for starters
    fvGood = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    fvGood.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    fvGood.fieldId = dcgm_fields.DCGM_FI_DEV_POWER_USAGE
    fvGood.status = 0
    fvGood.fieldType = ord(dcgm_fields.DCGM_FT_DOUBLE)
    fvGood.ts = get_usec_since_1970()
    fvGood.value.dbl = 100.0

    fieldGroupObj = pydcgm.DcgmFieldGroup(handleObj, "my_field_group", [fvGood.fieldId, ])

    #This will throw an exception if it fails
    dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvGood)
    expectedValueCount += 1

    operationMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO #Todo: Read from handleObj
    updateFreq = 1000000
    maxKeepAge = 86400.0
    maxKeepSamples = 0
    fieldWatcher = dcgm_field_helpers.DcgmFieldGroupWatcher(handle, groupObj.GetId(), fieldGroupObj,
                                                            operationMode, updateFreq, maxKeepAge, maxKeepSamples, 0)

    #DcgmFieldCollectionWatcher triggers a field value update so expectedValueCount should be incremented here as well
    expectedValueCount += 1

    assert len(fieldWatcher.values[gpuId][fvGood.fieldId]) == expectedValueCount, "%d != %d" % (len(fieldWatcher.values[gpuId][fvGood.fieldId]), expectedValueCount)
    #Cheat a bit by getting nextSinceTimestamp from the fieldWatcher. We are trying to
    #insert after the last time records were read
    nextSinceTimestamp = fieldWatcher._nextSinceTimestamp

    #insert more than one value at a time so we can track
    for numValuesPerLoop in range(0, 10):

        fvGood.ts = nextSinceTimestamp

        for i in range(numValuesPerLoop):
            if i > 0:
                fvGood.ts += 1
            fvGood.value.dbl += 0.1
            dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvGood)
            expectedValueCount += 1

        newValueCount = fieldWatcher.GetAllSinceLastCall()
        nextSinceTimestamp = fieldWatcher._nextSinceTimestamp

        assert nextSinceTimestamp != 0, "Got 0 nextSinceTimestamp"
        
        #A field value is always returned. If no data exists, then a single field value is returned with status set to NO DATA
        if numValuesPerLoop == 0:
            assert newValueCount == 1, "newValueCount %d != 1" % newValueCount
        else:
            assert newValueCount == numValuesPerLoop, "newValueCount %d != numValuesPerLoop %d" % (newValueCount, numValuesPerLoop)

def helper_dcgm_values_since(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuId = gpuIds[0]

    fieldId = dcgm_fields.DCGM_FI_DEV_BRAND #Should be a field id in fieldCollectionId

    fieldGroupObj = pydcgm.DcgmFieldGroup(handleObj, "my_field_group", [fieldId, ])

    operationMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO #Todo: Read from handleObj
    updateFreq = 100000
    maxKeepAge = 86400.0
    maxKeepSamples = 0
    #Watch and read initial values
    fieldWatcher = dcgm_field_helpers.DcgmFieldGroupWatcher(handle, groupObj.GetId(), fieldGroupObj,
                                                            operationMode, updateFreq, maxKeepAge, maxKeepSamples, 0)

    firstReadSize = len(fieldWatcher.values[gpuId][fieldId])
    assert firstReadSize > 0, "Expected values after first read. Got 0"

    sleepFor = (updateFreq / 1000000.0) * 2 #Sleep for 2x update freq to allow an update to occur
    time.sleep(sleepFor)

    numRead = fieldWatcher.GetAllSinceLastCall()
    secondReadSize = len(fieldWatcher.values[gpuId][fieldId])

    assert fieldWatcher._nextSinceTimestamp != 0, "Expected nonzero nextSinceTimestamp"
    assert numRead > 0, "Expected callbacks to be called from dcgmEngineGetValuesSince"
    assert secondReadSize > firstReadSize, "Expected more records. 2nd %d. 1st %d" % (secondReadSize, firstReadSize)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_values_since_agent(handle, gpuIds):
    helper_dcgm_values_since(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_values_since_remote(handle, gpuIds):
    helper_dcgm_values_since(handle, gpuIds)

def helper_dcgm_values_since_entities(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    entityId = gpuIds[0]
    entityGroupId = dcgm_fields.DCGM_FE_GPU

    fieldId = dcgm_fields.DCGM_FI_DEV_BRAND #Should be a field id in fieldCollectionId

    fieldGroupObj = pydcgm.DcgmFieldGroup(handleObj, "my_field_group", [fieldId, ])

    operationMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO #Todo: Read from handleObj
    updateFreq = 100000
    maxKeepAge = 86400.0
    maxKeepSamples = 0
    #Watch and read initial values
    fieldWatcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(handle, groupObj.GetId(), fieldGroupObj,
                                                                  operationMode, updateFreq, maxKeepAge, maxKeepSamples, 0)

    firstReadSize = len(fieldWatcher.values[entityGroupId][entityId][fieldId])
    assert firstReadSize > 0, "Expected values after first read. Got 0"

    sleepFor = (updateFreq / 1000000.0) * 2 #Sleep for 2x update freq to allow an update to occur
    time.sleep(sleepFor)

    numRead = fieldWatcher.GetAllSinceLastCall()
    secondReadSize = len(fieldWatcher.values[entityGroupId][entityId][fieldId])

    assert fieldWatcher._nextSinceTimestamp != 0, "Expected nonzero nextSinceTimestamp"
    assert numRead > 0, "Expected callbacks to be called from dcgmEngineGetValuesSince"
    assert secondReadSize > firstReadSize, "Expected more records. 2nd %d. 1st %d" % (secondReadSize, firstReadSize)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_values_since_entities_agent(handle, gpuIds):
    helper_dcgm_values_since_entities(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_values_since_entities_remote(handle, gpuIds):
    helper_dcgm_values_since_entities(handle, gpuIds)

def helper_dcgm_entity_get_latest_values(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuId = gpuIds[0]

    fieldIds = [dcgm_fields.DCGM_FI_DEV_BRAND, dcgm_fields.DCGM_FI_DEV_GPU_TEMP]
    fieldGroupObj = pydcgm.DcgmFieldGroup(handleObj, "my_field_group", fieldIds)

    updateFreq = 100000
    maxKeepAge = 86400.0
    maxKeepSamples = 0

    groupObj.samples.WatchFields(fieldGroupObj, updateFreq, maxKeepAge, maxKeepSamples)

    #Make sure our new fields have updated
    systemObj.UpdateAllFields(True)

    fieldValues = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, gpuId, fieldIds)
    
    for i, fieldValue in enumerate(fieldValues):
        logger.info(str(fieldValue))
        assert(fieldValue.version != 0), "idx %d Version was 0" % i
        assert(fieldValue.fieldId == fieldIds[i]), "idx %d fieldValue.fieldId %d != fieldIds[i] %d" % (i, fieldValue.fieldId, fieldIds[i])
        assert(fieldValue.status == dcgm_structs.DCGM_ST_OK), "idx %d status was %d" % (i, fieldValue.status)
        assert(fieldValue.ts != 0), "idx %d timestamp was 0" % i
        

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_entity_get_latest_values_embedded(handle, gpuIds):
    helper_dcgm_entity_get_latest_values(handle, gpuIds)

'''
Verify that the returned field values match the requested ones for dcgmEntitiesGetLatestValues
'''
def helper_validate_entities_lastest_values_request(handle, gpuIds, fieldIds):
    entityPairList = []
    responses = {}
    
    for gpuId in gpuIds:
        entityPairList.append(dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId))
        for fieldId in fieldIds:
            dictKey = "%d:%d:%d" % (dcgm_fields.DCGM_FE_GPU, gpuId, fieldId)
            responses[dictKey] = 0 #0 responses so far

    flags = dcgm_structs.DCGM_FV_FLAG_LIVE_DATA
    fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entityPairList, fieldIds, flags)
    
    for i, fieldValue in enumerate(fieldValues):
        logger.info(str(fieldValue))
        assert(fieldValue.version == dcgm_structs.dcgmFieldValue_version2), "idx %d Version was x%X. not x%X" % (i, fieldValue.version, dcgm_structs.dcgmFieldValue_version2)
        dictKey = "%d:%d:%d" % (fieldValue.entityGroupId, fieldValue.entityId, fieldValue.fieldId)
        assert dictKey in responses and responses[dictKey] == 0, "Mismatch on dictKey %s. Responses: %s" % (dictKey, str(responses))
        assert(fieldValue.status == dcgm_structs.DCGM_ST_OK), "idx %d status was %d" % (i, fieldValue.status)
        assert(fieldValue.ts != 0), "idx %d timestamp was 0" % i
        assert(fieldValue.unused == 0), "idx %d unused was %d" % (i, fieldValue.unused)
        responses[dictKey] += 1

def helper_dcgm_entities_get_latest_values(handle, gpuIds):
    #Request various combinations of DCGM field IDs. We're mixing field IDs that 
    #have NVML mappings and those that don't in order to try and cause failures

    #First, just field IDs that don't have mappings
    nonMappedFieldIds = [dcgm_fields.DCGM_FI_DEV_BRAND, 
                         dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
                         dcgm_fields.DCGM_FI_DEV_SM_CLOCK,
                         dcgm_fields.DCGM_FI_DEV_MEM_CLOCK,
                         dcgm_fields.DCGM_FI_DEV_VIDEO_CLOCK]
    fieldIds = nonMappedFieldIds
    helper_validate_entities_lastest_values_request(handle, gpuIds, fieldIds)

    #Now just field IDs that have mappings
    fieldIds = []
    for fieldId in range(dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TEX):
        fieldIds.append(fieldId)
    helper_validate_entities_lastest_values_request(handle, gpuIds, fieldIds)

    #Now a mix of both
    fieldIds = []
    for i in range(len(nonMappedFieldIds)):
        fieldIds.append(nonMappedFieldIds[i])
        fieldIds.append(i + dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL)
    helper_validate_entities_lastest_values_request(handle, gpuIds, fieldIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_entities_get_latest_values_embedded(handle, gpuIds):
    helper_dcgm_entities_get_latest_values(handle, gpuIds)

#Skip this test when running in injection-only mode
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled() # This test relies on accounting data, which doesn't work with MIG mode
def test_dcgm_live_accounting_data(handle, gpuIds):
    if test_utils.is_nvswitch_detected():
        test_utils.skip_test("Skipping GPU Cuda tests on NvSwitch systems since they require the FM to be loaded")

    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "mygroup")
    gpuId = dcgm_agent.dcgmGroupGetInfo(handle, groupId).entityList[0].entityId

    #Get the busid of the GPU
    fieldId = dcgm_fields.DCGM_FI_DEV_PCI_BUSID
    updateFreq = 1000000
    maxKeepAge = 3600.0 #one hour
    maxKeepEntries = 0 #no limit
    dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)
    dcgm_agent.dcgmUpdateAllFields(handle, 1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fieldId,])
    busId = values[0].value.str

    fieldId = dcgm_fields.DCGM_FI_DEV_ACCOUNTING_DATA
    updateFreq = 100000
    maxKeepAge = 3600.0 #one hour
    maxKeepEntries = 0 #no limit

    try:
        dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_REQUIRES_ROOT) as e:
        test_utils.skip_test("Skipping test for non-root due to accounting mode not being watched ahead of time.")
        return #Nothing further to do for non-root without accounting data

    #Start a cuda app so we have something to accounted
    appTimeout = 1000
    appParams = ["--ctxCreate", busId,
                 "--busyGpu", busId, str(appTimeout),
                 "--ctxDestroy", busId]
    app = apps.CudaCtxCreateAdvancedApp(appParams, env=test_utils.get_cuda_visible_devices_env(handle, gpuId))
    app.start(appTimeout*2)
    appPid = app.getpid()

    #Force an update
    dcgm_agent.dcgmUpdateAllFields(handle, 1)

    app.wait()

    #Wait for RM to think the app has exited. Querying immediately after app.wait() did not work
    time.sleep(1.0)

    #Force an update after app exits
    dcgm_agent.dcgmUpdateAllFields(handle, 1)

    maxCount = 2000 #Current accounting buffer is 1920
    startTs = 0
    endTs = 0
    values = dcgm_agent_internal.dcgmGetMultipleValuesForField(handle, gpuId, fieldId, maxCount, startTs, endTs, dcgm_structs.DCGM_ORDER_ASCENDING)

    foundOurPid = False

    #There could be multiple accounted processes in our data structure. Look for the PID we just ran
    for value in values:
        accStats = dcgm_structs.c_dcgmDevicePidAccountingStats_v1()
        ctypes.memmove(ctypes.addressof(accStats), value.value.blob, accStats.FieldsSizeof())

        #print "Pid %d: %s" % (accStats.pid, str(accStats))

        if appPid == accStats.pid:
            #Found it!
            foundOurPid = True
            break

    assert foundOurPid, "Did not find app PID of %d in list of %d PIDs for gpuId %d" % (appPid, len(values), gpuId)

def helper_dcgm_values_pid_stats(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    
    ## Add first GPU to the group (only need single GPU)
    gpuId = gpuIds[0]
    groupObj.AddGpu(gpuId)
    busId = _get_gpu_bus_id(gpuId, handle)
    
    #watch the process info fields
    if utils.is_root():
        groupObj.stats.WatchPidFields(updateFreq=100000, maxKeepAge=3600, maxKeepSamples=0)
    else:
        try:
            groupObj.stats.WatchPidFields(updateFreq=100000, maxKeepAge=3600, maxKeepSamples=0)
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_REQUIRES_ROOT) as e:
            return
    
    #Start a cuda app so we have something to accounted
    #Start a 2nd app that will appear in the compute app list
    appTimeout = 5000
    app = _create_cuda_app_for_pid_stats(handle, busId, appTimeout, gpuId)
    app2 = _create_cuda_app_for_pid_stats(handle, busId, appTimeout, gpuId)
    app3 = _create_cuda_assert_app_for_pid_stats(handle, busId, appTimeout, gpuId)

    app.wait()
    app2.wait()
    app3.wait()
    app3.terminate()
    app3.validate()
    
    #Wait for RM to think the app has exited. Querying immediately after app.wait() did not work
    time.sleep(1.0)

    _assert_pid_utilization_rate(systemObj, groupObj, app.getpid())
    _assert_other_compute_pid_seen(systemObj, groupObj, app.getpid(), app2.getpid())
    _assert_pid_cuda_assert_occurence(systemObj, groupObj, app.getpid())

def _assert_pid_cuda_assert_occurence(dcgmSystem, dcgmGroup, appPid):
    ''' Force an update and verifies that a xid error occurred during the life of the process '''
    dcgmSystem.UpdateAllFields(1)
    pidInfo = dcgmGroup.stats.GetPidInfo(appPid)
    
    assert pidInfo.summary.numXidCriticalErrors > 0, "At least one Xid error should have been caught, but (%d) were found" % pidInfo.summary.numXidCriticalErrors
    
    for index in range(pidInfo.summary.numXidCriticalErrors):
        assert pidInfo.summary.xidCriticalErrorsTs[index] != 0, "Unable to find a valid timestamp for the Xid Error %d" % index
        logger.debug("Xid Timestamp: " + str(pidInfo.summary.xidCriticalErrorsTs[index]))

def _assert_pid_utilization_rate(dcgmSystem, dcgmGroup, appPid):
    '''Force an update and then assert that utilization rates are recorded for a PID'''
    dcgmSystem.UpdateAllFields(1)
    pidInfo = dcgmGroup.stats.GetPidInfo(appPid)

    assert pidInfo.gpus[0].processUtilization.pid == appPid, " Expected PID %d, got PID %d"  % (appPid, pidInfo.gpus[0].processUtilization.pid)
    utilizationRate = 0
    
    if pidInfo.gpus[0].processUtilization.smUtil > 0 :
        utilizationRate = 1
    if pidInfo.gpus[0].processUtilization.memUtil > 0 :
        utilizationRate = 1

    #TODO: DCGM-1418 - Uncomment the following line again
    #assert utilizationRate, "Expected non-zero utilization rates for the PID %d"  %appPid
    
def _assert_other_compute_pid_seen(dcgmSystem, dcgmGroup, app1Pid, app2Pid):
    '''Force an update and then assert that PID 1 stats see PID2'''
    dcgmSystem.UpdateAllFields(1)
    pidInfo = dcgmGroup.stats.GetPidInfo(app1Pid)

    assert pidInfo.summary.numOtherComputePids >= 1, "Expected other pid of %d" % app2Pid

    #Check for the expected PID in the range of OtherCompute Pids in the process stats
    pidFound = False
    for pid in range(0, pidInfo.summary.numOtherComputePids):
        if app2Pid == pidInfo.summary.otherComputePids[pid]:
            pidFound = True
            break
            
    assert pidFound,  "Expected other compute pid %d, number of other Compute Pids - %d \
                                     . PIDs Found %d, %d, %d , %d, %d, %d, %d, %d, %d, %d"\
                                     % (app2Pid, pidInfo.summary.numOtherComputePids, pidInfo.summary.otherComputePids[0],  pidInfo.summary.otherComputePids[1],\
                                           pidInfo.summary.otherComputePids[2], pidInfo.summary.otherComputePids[3], pidInfo.summary.otherComputePids[4], pidInfo.summary.otherComputePids[5],\
                                           pidInfo.summary.otherComputePids[6], pidInfo.summary.otherComputePids[7], pidInfo.summary.otherComputePids[8], pidInfo.summary.otherComputePids[9])

def _create_cuda_app_for_pid_stats(handle, busId, appTimeout, gpuId):
    app = apps.CudaCtxCreateAdvancedApp(["--ctxCreate", busId,
                                         "--busyGpu", busId, str(appTimeout),
                                         "--ctxDestroy", busId], env=test_utils.get_cuda_visible_devices_env(handle, gpuId))
    app.start(appTimeout*2)
    return app

def _create_cuda_assert_app_for_pid_stats(handle, busId, appTimeout, gpuId):
    app = apps.RunCudaAssert(["--ctxCreate", busId,
                              "--cuMemAlloc", busId, "200",
                              "--cuMemFree", busId,
                              "--assertGpu", busId, str(appTimeout)], env=test_utils.get_cuda_visible_devices_env(handle, gpuId))
    
    app.start(appTimeout*2)
    return app

def _get_gpu_bus_id(gpuId, handle):
    #Get the busid of the GPU
    fieldId = dcgm_fields.DCGM_FI_DEV_PCI_BUSID
    updateFreq = 100000
    maxKeepAge = 3600.0 #one hour
    maxKeepEntries = 0 #no limit
    dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)
    dcgm_agent.dcgmUpdateAllFields(handle, 1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fieldId,])
    return values[0].value.str
    
def helper_dcgm_values_pid_stats_realtime(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    
    ## Add first GPU to the group (only need single GPU)
    gpuId = gpuIds[0]
    groupObj.AddGpu(gpuId)
    busId = _get_gpu_bus_id(gpuId, handle)
    
    #watch the process info fields
    try:
        groupObj.stats.WatchPidFields(updateFreq=100000,
                                  maxKeepAge=3600,
                                  maxKeepSamples=0)
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_REQUIRES_ROOT) as e:
        test_utils.skip_test("Skipping test for non-root due to accounting mode not being watched ahead of time.")
        return
    
    #Start a cuda app so we have something to accounted
    appTimeout = 10000
    app = _create_cuda_app_for_pid_stats(handle, busId, appTimeout, gpuId)
    appPid = app.getpid()
    app.stdout_readtillmatch(lambda s: s.find("Calling cuInit") != -1)
    
    #Start a 2nd app that will appear in the compute app list
    app2 = _create_cuda_app_for_pid_stats(handle, busId, appTimeout, gpuId)
    app2Pid = app2.getpid()
    app2.stdout_readtillmatch(lambda s: s.find("Calling cuInit") != -1)

    time.sleep(1.0)
    _assert_other_compute_pid_seen(systemObj, groupObj, appPid, app2Pid)
    
    app.wait()
    app2.wait()
    
    _assert_pid_utilization_rate(systemObj, groupObj,appPid)
    _assert_pid_utilization_rate(systemObj, groupObj,app2Pid)    
    
    ## Make sure the stats can be fetched after the process is complete
    for count in range(0,2):
        time.sleep(1.0)
        _assert_other_compute_pid_seen(systemObj, groupObj, appPid, app2Pid)


#Skip this test when running in injection-only mode
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled() # This test relies on accounting data, which doesn't work with MIG mode
def test_dcgm_values_pid_stats_embedded(handle, gpuIds):
    if test_utils.is_nvswitch_detected():
        test_utils.skip_test("Skipping GPU Cuda tests on NvSwitch systems since they require the FM to be loaded")
    helper_dcgm_values_pid_stats(handle, gpuIds)

#Skip this test when running in injection-only mode
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled() # This test relies on accounting data, which doesn't work with MIG mode
def test_dcgm_live_pid_stats_remote(handle, gpuIds):

    if test_utils.is_nvswitch_detected():
        test_utils.skip_test("Skipping GPU Cuda tests on NvSwitch systems since they require the FM to be loaded")

    helper_dcgm_values_pid_stats(handle, gpuIds)
    
#Skip this test when running in injection-only mode
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled() # This test relies on accounting data, which doesn't work with MIG mode
def test_dcgm_values_pid_stats_realtime_embedded(handle, gpuIds):

    if test_utils.is_nvswitch_detected():
        test_utils.skip_test("Skipping GPU Cuda tests on NvSwitch systems since they require the FM to be loaded")

    helper_dcgm_values_pid_stats_realtime(handle, gpuIds)

#Skip this test when running in injection-only mode
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled() # This test relies on accounting data, which doesn't work with MIG mode
def test_dcgm_values_pid_stats_realtime_remote(handle, gpuIds):

    if test_utils.is_nvswitch_detected():
        test_utils.skip_test("Skipping GPU Cuda tests on NvSwitch systems since they require the FM to be loaded")

    helper_dcgm_values_pid_stats_realtime(handle, gpuIds)    

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_values_job_stats_remove(handle, gpuIds):

    if test_utils.is_nvswitch_detected():
        test_utils.skip_test("Skipping GPU Cuda tests on NvSwitch systems since they require the FM to be loaded")

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")

    jobId = "my_important_job"

    #Fetch an unknown job
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA)):
        groupObj.stats.RemoveJob(jobId)

    #Stop an unknown job
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA)):
        groupObj.stats.StopJobStats(jobId)

    #Watch the underlying job fields so that the underlying job stats work
    groupObj.stats.WatchJobFields(10000000, 3600.0, 0)

    #Add the job then remove it
    groupObj.stats.StartJobStats(jobId)
    groupObj.stats.RemoveJob(jobId)

    #Remove it again. This will fail
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA)):
        groupObj.stats.RemoveJob(jobId)

    #Re-use the key again
    groupObj.stats.StartJobStats(jobId)
    groupObj.stats.StopJobStats(jobId)
    #Use the mass-remove this time
    groupObj.stats.RemoveAllJobs()

    #Remove the job we deleted with RemoveAllJobs(). This should fail
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA)):
        groupObj.stats.RemoveJob(jobId)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled() # This test relies on accounting data, which doesn't work with MIG mode
def test_dcgm_values_job_stats_get(handle, gpuIds):
    if test_utils.is_nvswitch_detected():
        test_utils.skip_test("Skipping GPU Cuda tests on NvSwitch systems since they require the FM to be loaded")

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")

    ## Add first GPU to the group
    gpuId = gpuIds[-1]
    groupObj.AddGpu(gpuId)
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against

    #Get the busid of the GPU
    fieldId = dcgm_fields.DCGM_FI_DEV_PCI_BUSID
    updateFreq = 100000
    maxKeepAge = 3600.0 #one hour
    maxKeepEntries = 0 #no limit
    dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)
    dcgm_agent.dcgmUpdateAllFields(handle, 1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fieldId,])
    busId = values[0].value.str
    appTimeout = 2000 #Give the cache manager a chance to record stats

    jobId = "jobIdTest"

    ## Notify DCGM to start collecting stats
    try:
        groupObj.stats.WatchJobFields(updateFreq, maxKeepAge, maxKeepEntries)
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_REQUIRES_ROOT) as e:
        test_utils.skip_test("Skipping test for non-root due to accounting mode not being watched ahead of time.")
        return

    groupObj.stats.StartJobStats(jobId)

    environ = test_utils.get_cuda_visible_devices_env(handle, gpuId)

    #Start a few cuda apps so we have something to accounted
    appParams = ["--ctxCreate", busId,
                "--busyGpu", busId, str(appTimeout),
                "--ctxDestroy", busId]
    
    xidArgs = ["--ctxCreate", busId,
               "--cuMemAlloc", busId, "200",
               "--cuMemFree", busId,
               "--assertGpu", busId, str(appTimeout)]
    
    #app.start(appTimeout*2)
    #app_xid.start(appTimeout*2)
    
    appPid = []
    appObjs = []

    #Start all three apps at once
    for i in range (0,3):
        app = apps.CudaCtxCreateAdvancedApp(appParams, env=environ)
        app.start(appTimeout*2)
        
        app_xid = apps.RunCudaAssert(xidArgs, env=environ)
        app_xid.start(appTimeout)
        
        appPid.append(app.getpid())
        
        appObjs.append(app)
        appObjs.append(app_xid)

    for app in appObjs:
        app.wait()
        
    for app_xid in appObjs:
        app_xid.wait()
        app_xid.terminate()
        app_xid.validate()

    #Wait for RM to think the app has exited. Querying immediately after app.wait() did not work
    time.sleep(1.0)

    ## Notify DCGM to stop collecting stats
    groupObj.stats.StopJobStats(jobId)

    #Force an update after app exits
    systemObj.UpdateAllFields(1)

    # get job stats
    jobInfo = groupObj.stats.GetJobStats(jobId)

    pidFound = 0

    assert jobInfo.summary.numComputePids >= 3, "Not all CUDA process ran captured during job. Expected 3, got %d" %jobInfo.summary.numComputePids
    for i in range (0,2):
        pidFound = 0
        for j in range (0,dcgm_structs.DCGM_MAX_PID_INFO_NUM):
            if appPid[i] == jobInfo.summary.computePids[j].pid:
                pidFound = 1
                break
        assert pidFound, "CUDA Process PID not captured during job. Missing: %d" % appPid[i]
        
    #Validate the values in the summary and the gpu Info
    assert jobInfo.summary.energyConsumed >= jobInfo.gpus[0].energyConsumed, "energyConsumed in the job stat summary %d is less than the one consumed by a gpu %d" % \
                                                                              (jobInfo.summary.energyConsumed, jobInfo.gpus[0].energyConsumed)
    assert jobInfo.summary.pcieReplays >= jobInfo.gpus[0].pcieReplays, "pice replays in the job stat summary %d is less than the one found by a gpu %d" %\
                                                                        (jobInfo.summary.pcieReplays, jobInfo.gpus[0].pcieReplays)
    assert jobInfo.summary.startTime == jobInfo.gpus[0].startTime, "Start Time in the job stat summary %d is different than the one stored in gpu Info %d" %\
                                                                        (jobInfo.summary.startTime, jobInfo.gpus[0].startTime)
    assert jobInfo.summary.endTime == jobInfo.gpus[0].endTime, "End Time in the job stat summary %d is different than the one stored in gpu Info %d" %\
                                                                        (jobInfo.summary.endTime, jobInfo.gpus[0].endTime)
    assert jobInfo.summary.eccSingleBit >= jobInfo.gpus[0].eccSingleBit, "ecc single bit in the job stat summary %d is less than the one stored in a gpu Info %d" %\
                                                                        (jobInfo.summary.eccSingleBit, jobInfo.gpus[0].eccSingleBit)
    assert jobInfo.summary.eccDoubleBit >= jobInfo.gpus[0].eccDoubleBit, "ecc double bit in the job stat summary %d is less than the one stored in a gpu Info %d" %\
                                                                        (jobInfo.summary.eccDoubleBit, jobInfo.gpus[0].eccDoubleBit)
    assert jobInfo.summary.thermalViolationTime >= jobInfo.gpus[0].thermalViolationTime, "thermal violation time in the job stat summary %d is less than the one stored in a gpu Info %d" %\
                                                                        (jobInfo.summary.thermalViolationTime, jobInfo.gpus[0].thermalViolationTime)
    assert jobInfo.summary.powerViolationTime >= jobInfo.gpus[0].powerViolationTime, "power violation time in the job stat summary %d is less than the one stored in a gpu Info %d" %\
                                                                        (jobInfo.summary.powerViolationTime, jobInfo.gpus[0].powerViolationTime)
    assert jobInfo.summary.maxGpuMemoryUsed >= jobInfo.gpus[0].maxGpuMemoryUsed, "Max GPU memory used in the job stat summary %d is less than the one stored in a gpu Info %d" %\
                                                                        (jobInfo.summary.maxGpuMemoryUsed, jobInfo.gpus[0].maxGpuMemoryUsed)
    assert jobInfo.summary.syncBoostTime >= jobInfo.gpus[0].syncBoostTime, "Sync Boost time in the job stat summary %d is less than the one stored in a gpu Info %d" %\
                                                                        (jobInfo.summary.syncBoostTime, jobInfo.gpus[0].syncBoostTime)
    assert jobInfo.summary.overallHealth == jobInfo.gpus[0].overallHealth, "Over all Health in the job summary (%d) is different from the one in the gpu Info (%d)" %\
                                                                        (jobInfo.summary.overallHealth, jobInfo.gpus[0].overallHealth)
    assert jobInfo.summary.numXidCriticalErrors == jobInfo.gpus[0].numXidCriticalErrors, "At least (%d) Xid error should have been caught, but (%d) were found" %\
                                                                        (jobInfo.summary.numXidCriticalErrors, jobInfo.gpus[0].numXidCriticalErrors)
    assert jobInfo.summary.numXidCriticalErrors > 0, "At least one Xid error should have been caught, but (%d) were found" % jobInfo.summary.numXidCriticalErrors   
    
    for index in range(jobInfo.summary.numXidCriticalErrors):
        assert jobInfo.summary.xidCriticalErrorsTs[index] != 0, "Unable to find a valid timestamp for the Xid Error %d" % index
        logger.debug("Xid Timestamp: " + str(jobInfo.summary.xidCriticalErrorsTs[index]))
   
    #Start another job with the same job ID and it should return DUPLICATE KEY Error
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_DUPLICATE_KEY)):
        groupObj.stats.StartJobStats(jobId)
        
    #print str(jobInfo.summary)
    #print ""
    #print str(jobInfo.gpus[0])


@test_utils.run_with_embedded_host_engine()
def test_dcgm_field_by_id(handle):

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    fieldInfo = systemObj.fields.GetFieldById(dcgm_fields.DCGM_FI_DEV_BRAND)
    assert fieldInfo.fieldId == dcgm_fields.DCGM_FI_DEV_BRAND, "Field id %d" % fieldInfo.fieldId
    assert fieldInfo.fieldType == dcgm_fields.DCGM_FT_STRING, "Field type %s" % fieldInfo.fieldType
    assert fieldInfo.scope == dcgm_fields.DCGM_FS_DEVICE, "Field scope %d" % fieldInfo.scope
    assert fieldInfo.tag == "brand", "Field tag %s" % fieldInfo.tag

    bogusFieldInfo = systemObj.fields.GetFieldById(dcgm_fields.DCGM_FI_MAX_FIELDS)
    assert bogusFieldInfo == None, "Expected null fieldInfo for dcgm_fields.DCGM_FI_MAX_FIELDS"

@test_utils.run_with_embedded_host_engine()
def test_dcgm_field_by_tag(handle):

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    fieldInfo = systemObj.fields.GetFieldByTag('brand')
    assert fieldInfo.fieldId == dcgm_fields.DCGM_FI_DEV_BRAND, "Field id %d" % fieldInfo.fieldId
    assert fieldInfo.fieldType == dcgm_fields.DCGM_FT_STRING, "Field type %d" % fieldInfo.fieldType
    assert fieldInfo.scope == dcgm_fields.DCGM_FS_DEVICE, "Field scope %d" % fieldInfo.scope
    assert fieldInfo.tag == "brand", "Field tag %s" % fieldInfo.tag

    bogusFieldInfo = systemObj.fields.GetFieldByTag('no_way_this_is_a_field')
    assert bogusFieldInfo == None, "Expected null fieldInfo for bogus tag"

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_fields_all_fieldids_valid(handle, gpuIds):
    """
    Test that any field IDs that are defined are retrievable
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)

    #Some field IDs don't generate data by default. For instance, accounting data only works if accounting
    #mode is enabled and processes are running. Field IDs in this list fall into this category and have
    #already been confirmed to generate data within the cache manager
    exceptionFieldIds = [dcgm_fields.DCGM_FI_DEV_COMPUTE_PIDS,
                         dcgm_fields.DCGM_FI_DEV_ACCOUNTING_DATA,
                         dcgm_fields.DCGM_FI_DEV_GRAPHICS_PIDS,
                         dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
                         dcgm_fields.DCGM_FI_DEV_VGPU_VM_ID,
                         dcgm_fields.DCGM_FI_DEV_VGPU_VM_NAME,
                         dcgm_fields.DCGM_FI_DEV_VGPU_TYPE,
                         dcgm_fields.DCGM_FI_DEV_VGPU_UUID,
                         dcgm_fields.DCGM_FI_DEV_VGPU_DRIVER_VERSION,
                         dcgm_fields.DCGM_FI_DEV_VGPU_MEMORY_USAGE,
                         dcgm_fields.DCGM_FI_DEV_VGPU_LICENSE_INSTANCE_STATUS,
                         dcgm_fields.DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT,
                         dcgm_fields.DCGM_FI_DEV_VGPU_ENC_STATS,
                         dcgm_fields.DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO,
                         dcgm_fields.DCGM_FI_DEV_VGPU_FBC_STATS,
                         dcgm_fields.DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO,
                         dcgm_fields.DCGM_FI_DEV_GPU_NVLINK_ERRORS,
                         dcgm_fields.DCGM_FI_DEV_GPU_UTIL_SAMPLES,
                         dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES]
    exceptionFieldIds.extend(g_profilingFieldIds)
    
    gpuId = gpuIds[0]
    fieldIdVars = {}

    #Find all of the numerical field IDs by looking at the dcgm_fields module's attributes
    for moduleAttribute in list(dcgm_fields.__dict__.keys()):
        if moduleAttribute.find("DCGM_FI_") == 0 and moduleAttribute != 'DCGM_FI_MAX_FIELDS' and \
           moduleAttribute != 'DCGM_FI_UNKNOWN':
            fieldIdVars[moduleAttribute] = dcgm_fields.__dict__[moduleAttribute]

    numErrors = 0

    #Add watches on all known fieldIds
    for fieldIdName in list(fieldIdVars.keys()):
        fieldId = fieldIdVars[fieldIdName]

        updateFreq = 10 * 1000000
        maxKeepAge = 3600.0
        maxKeepEntries = 100
        try:
            if test_utils.is_mig_mode_enabled() and fieldId == dcgm_fields.DCGM_FI_DEV_ACCOUNTING_DATA:
                # We cannot enable accounting mode with MIG mode enabled - CUDANVML-153
                continue

            dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_REQUIRES_ROOT):
            logger.info("Skipping field %d that requires root" % fieldId)
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM):
            logger.error("Unable to watch field %s (id %d). Unknown field?" % (fieldIdName, fieldId))
            numErrors += 1

    #Force all fields to possibly update so we can fetch data for them
    handleObj.GetSystem().UpdateAllFields(1)

    #fieldIdVars = {'DCGM_FI_GPU_TOPOLOGY_NVLINK' : 61}

    for fieldIdName in list(fieldIdVars.keys()):
        fieldId = fieldIdVars[fieldIdName]

        #Verify that we can fetch field metadata. This call will throw an exception on error
        fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)

        #Fetch each fieldId individually so we can check for errors
        fieldValue = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fieldId, ])[0]

        if fieldId in exceptionFieldIds:
            continue #Don't check fields that are excluded from testing
        #Skip NvSwitch fields since they are pushed from fabric manager rather than polled
        if fieldId >= dcgm_fields.DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P00 and fieldId <= dcgm_fields.DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS:
            continue

        if fieldValue.status == dcgm_structs.DCGM_ST_NOT_SUPPORTED:
            #It's ok for fields to not be supported. We got a useful error code
            logger.info("field %s (id %d) returned st DCGM_ST_NOT_SUPPORTED (OK)" % (fieldIdName, fieldId))
        elif fieldValue.status != dcgm_structs.DCGM_ST_OK:
            logger.error("No value for field %s (id %d). status: %d" % (fieldIdName, fieldId, fieldValue.status))
            numErrors += 1

    assert numErrors == 0, "Got %d errors" % numErrors


def test_dcgm_verify_manual_mode_behavior():
    """  
    Test to verify that field values cannot be
    retrieved automatically in manual operation mode
    """

    # Gets the handle and set operation mode to manual
    handleObj = pydcgm.DcgmHandle(ipAddress=None, opMode=dcgm_structs.DCGM_OPERATION_MODE_MANUAL)

    # Creates a default group with all GPUs
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuId = groupObj.GetGpuIds()[0]


    fieldId = dcgm_fields.DCGM_FI_DEV_POWER_USAGE
    updateFreq = 100000 #100 miliseconds
    maxKeepAge = 3600.0 #one hour
    maxKeepEntries = 0 #no limit

    # watch the fieldvalues list
    dcgm_agent_internal.dcgmWatchFieldValue(handleObj.handle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)

    # trigger update for all fields once, wait for it complete and get timestamp
    systemObj.UpdateAllFields(waitForUpdate=True)
    initialValues = dcgm_agent_internal.dcgmGetLatestValuesForFields(handleObj.handle, gpuId, [fieldId,])
    firstTimestamp = initialValues[0].ts

    for i in range(1,10):
        values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handleObj.handle, gpuId, [fieldId,])
        otherTimestamp = values[0].ts
        time.sleep(300.0 / 1000.0) # sleep for 300 miliseconds
        
        assert firstTimestamp == otherTimestamp, "Fields got updated automatically, that should not happen in MANUAL OPERATION MODE"

    # trigger update manually to make sure fields got updated
    # and have a different timestamp now
    systemObj.UpdateAllFields(waitForUpdate=True)
    time.sleep(300.0 / 1000.0)
    postUpdateValues = dcgm_agent_internal.dcgmGetLatestValuesForFields(handleObj.handle, gpuId, [fieldId,])
    latestTimestamp = postUpdateValues[0].ts

    handleObj.Shutdown()

    assert firstTimestamp != latestTimestamp, "Fields did not get updated after manually trigerring an update"


def test_dcgm_verify_auto_mode_behavior():
    """  
    Test to verify that field values can be retrieved
    automatically in manual operation mode
    """

    # Gets the handle and set operation mode to automatic
    handleObj = pydcgm.DcgmHandle(ipAddress=None, opMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO)

    # Creates a default group with all GPUs
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuId = groupObj.GetGpuIds()[0]


    fieldId = dcgm_fields.DCGM_FI_DEV_POWER_USAGE
    updateFreq = 100000 #100 miliseconds
    maxKeepAge = 3600.0 #one hour
    maxKeepEntries = 0 #no limit

    # watch the fieldvalues list
    dcgm_agent_internal.dcgmWatchFieldValue(handleObj.handle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)

    # trigger update for all fields once, wait for it complete and get timestamp
    systemObj.UpdateAllFields(waitForUpdate=True)
    initialValues = dcgm_agent_internal.dcgmGetLatestValuesForFields(handleObj.handle, gpuId, [fieldId,])
    firstTimestamp = initialValues[0].ts

    time.sleep(300.0 / 1000.0) # sleep for 300 miliseconds
    otherValues = dcgm_agent_internal.dcgmGetLatestValuesForFields(handleObj.handle, gpuId, [fieldId,])
    nextTimestamp = otherValues[0].ts

    handleObj.Shutdown()
            
    assert firstTimestamp != nextTimestamp, "Failed to update Fields automatically, that should not happen in AUTO OPERATION MODE"

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_device_attributes(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    for gpuId in gpuIds:
        gpuAttrib = systemObj.discovery.GetGpuAttributes(gpuId)

        #Validate field values
        assert gpuAttrib.version != 0, "gpuAttrib.version == 0"
        assert len(gpuAttrib.identifiers.brandName) > 0 and not dcgmvalue.DCGM_STR_IS_BLANK(gpuAttrib.identifiers.brandName), \
            "gpuAttrib.identifiers.brandName: '%s'" % gpuAttrib.identifiers.brandName
        assert len(gpuAttrib.identifiers.deviceName) > 0 and not dcgmvalue.DCGM_STR_IS_BLANK(gpuAttrib.identifiers.deviceName), \
            "gpuAttrib.identifiers.deviceName: '%s'" % gpuAttrib.identifiers.deviceName
        assert len(gpuAttrib.identifiers.pciBusId) > 0 and not dcgmvalue.DCGM_STR_IS_BLANK(gpuAttrib.identifiers.pciBusId), \
            "gpuAttrib.identifiers.pciBusId: '%s'" % gpuAttrib.identifiers.pciBusId
        assert len(gpuAttrib.identifiers.uuid) > 0 and not dcgmvalue.DCGM_STR_IS_BLANK(gpuAttrib.identifiers.uuid), \
            "gpuAttrib.identifiers.uuid: '%s'" % gpuAttrib.identifiers.uuid
        assert len(gpuAttrib.identifiers.vbios) > 0 and not dcgmvalue.DCGM_STR_IS_BLANK(gpuAttrib.identifiers.vbios), \
            "gpuAttrib.identifiers.vbios: '%s'" % gpuAttrib.identifiers.vbios
        assert len(gpuAttrib.identifiers.driverVersion) > 0 and not dcgmvalue.DCGM_STR_IS_BLANK(gpuAttrib.identifiers.driverVersion), \
            "gpuAttrib.identifiers.driverVersion: '%s'" % gpuAttrib.identifiers.driverVersion
        assert gpuAttrib.identifiers.pciDeviceId != 0, "gpuAttrib.identifiers.pciDeviceId: %08X" % gpuAttrib.identifiers.pciDeviceId
        assert gpuAttrib.identifiers.pciSubSystemId != 0, "gpuAttrib.identifiers.pciSubSystemId: %08X" % gpuAttrib.identifiers.pciSubSystemId

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_device_attributes_bad_gpuid(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    gpuIds = [-1, dcgm_structs.DCGM_MAX_NUM_DEVICES]

    #None of these should crash
    for gpuId in gpuIds:
        gpuAttrib = systemObj.discovery.GetGpuAttributes(gpuId)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_nvlink_bandwidth(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuId = gpuIds[0]

    fieldIds = [ dcgm_fields.DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL, ]

    fieldGroupObj = pydcgm.DcgmFieldGroup(handleObj, "my_group", fieldIds)

    operationMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO
    updateFreq = 100000
    maxKeepAge = 86400.0
    maxKeepSamples = 0

    fieldWatcher = dcgm_field_helpers.DcgmFieldGroupWatcher(handle, groupObj.GetId(), fieldGroupObj, operationMode, updateFreq, maxKeepAge, maxKeepSamples, 0)

    assert len(fieldWatcher.values[gpuId]) == len(fieldIds), "Expected %d NVlink bandwidth values, got %d" % (len(fieldIds), len(fieldWatcher.values[gpuId]))

    for fieldId in fieldIds:
        for value in fieldWatcher.values[gpuId][fieldId]:
            # Either the GPU supports NvLink, in which case a
            # non-blank, actual value should be read, or the GPU does
            # not support NvLink, in which case the field should be
            # blank and the value should reflect that it's not
            # supported.
            assert ((value.isBlank == True) or (value.isBlank == False and value.value >= 0)), "Unexpected error reading field %d on GPU %d" % (fieldId, gpuId)

def helper_nvswitch_monitoring(handle, switchIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    entities = []
    for switchId in switchIds:
        entity = dcgm_structs.c_dcgmGroupEntityPair_t()
        entity.entityGroupId = dcgm_fields.DCGM_FE_SWITCH
        entity.entityId = switchId
        entities.append(entity)

    groupObj = systemObj.GetGroupWithEntities('SwitchGroup', entities)

    fieldIds = []
    for i in range(dcgm_fields.DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P00, dcgm_fields.DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P05):
        fieldIds.append(i)

    fieldGroupObj = pydcgm.DcgmFieldGroup(handleObj, "my_group", fieldIds)

    switchId = switchIds[0]

    operationMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO
    updateFreq = 100000
    maxKeepAge = 86400.0
    maxKeepSamples = 0

    fieldWatcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(handle, groupObj.GetId(), fieldGroupObj, operationMode, updateFreq, maxKeepAge, maxKeepSamples, 0)
    
    msg = "Expected %d NVlink bandwidth values, got %d" % (len(fieldIds), len(fieldWatcher.values[dcgm_fields.DCGM_FE_SWITCH][switchId]))
    assert len(fieldWatcher.values[dcgm_fields.DCGM_FE_SWITCH][switchId]) == len(fieldIds), msg

    # Check that the values are the appropriate dummy values
    for fieldId in fieldIds:
        for value in fieldWatcher.values[dcgm_fields.DCGM_FE_SWITCH][switchId][fieldId]:
            # For now, these should all be blank values. This test may be updated or deleted later
            # when the NSCQ library exists
            assert (value.isBlank == True), "Unexpected error reading field %d on Switch %d" % (fieldId, switchId)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_nvswitches(switchCount=2)
def test_nvswitch_monitoring_embedded(handle, switchIds):
    helper_nvswitch_monitoring(handle, switchIds)

@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_nvswitches(switchCount=2)
def test_nvswitch_monitoring_standalone(handle, switchIds):
    helper_nvswitch_monitoring(handle, switchIds)
