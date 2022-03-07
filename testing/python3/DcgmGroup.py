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
import dcgm_agent
import dcgm_structs
import dcgm_fields
import dcgm_field_helpers

class DcgmGroupConfig:
    def __init__(self, dcgmHandle, groupId, dcgmGroup):
        self._dcgmHandle = dcgmHandle
        self._groupId = groupId
        self._dcgmGroup = dcgmGroup

    '''
    Set configuration for this group

    config should be an instance of dcgm_structs.c_dcgmDeviceConfig_v1

    Will throw an exception on error
    '''
    def Set(self, config):
        status = pydcgm.DcgmStatus()
        ret = dcgm_structs.DCGM_ST_OK

        try:
            ret = dcgm_agent.dcgmConfigSet(self._dcgmHandle.handle, self._groupId, config, status.handle)
        except dcgm_structs.DCGMError as e:
            pass

        #Throw specific errors before return error
        status.ThrowExceptionOnErrors()
        #Throw an appropriate exception on error
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Get configuration for this group

    configType is a DCGM_CONFIG_? constant

    Returns an array of dcgm_structs.c_dcgmDeviceConfig_v1 objects
    Throws an exception on error
    '''
    def Get(self, configType):
        status = pydcgm.DcgmStatus()

        gpuIds = self._dcgmGroup.GetGpuIds()
        configList = dcgm_agent.dcgmConfigGet(self._dcgmHandle.handle, self._groupId, configType, len(gpuIds), status.handle)
        #Throw specific errors before return error
        status.ThrowExceptionOnErrors()
        return configList

    '''
    Enforce the configuration that has been set with Set()

    Throws an exception on error
    '''
    def Enforce(self):
        status = pydcgm.DcgmStatus()
        ret = dcgm_structs.DCGM_ST_OK
        try:
            ret = dcgm_agent.dcgmConfigEnforce(self._dcgmHandle.handle, self._groupId, status.handle)
        except dcgm_structs.DCGMError as e:
            pass
            
        #Throw specific errors before return error
        status.ThrowExceptionOnErrors()
        #Throw an appropriate exception on error
        dcgm_structs._dcgmCheckReturn(ret)

class DcgmGroupSamples:

    def __init__(self, dcgmHandle, groupId, dcgmGroup):
        self._dcgmHandle = dcgmHandle
        self._groupId = groupId
        self._dcgmGroup = dcgmGroup

    '''
    Tell DCGM to start recording samples for the given field group

    fieldGroup: DcgmFieldGroup() instance tracking the fields we want to watch.
    updateFreq: How often to update these fields in usec
    maxKeepAge: How long to keep data for these fields in seconds
    maxKeepSamples: Maximum number of samples to keep per field. 0=no limit

    Once the field collection is watched, it will update whenever the next update
    loop occurs. If you want to query these values immediately, use
    handle.UpdateAllFields(True) to make sure that the fields have updated at least once.
    '''
    def WatchFields(self, fieldGroup, updateFreq, maxKeepAge, maxKeepSamples):
        ret = dcgm_agent.dcgmWatchFields(self._dcgmHandle.handle, self._groupId, fieldGroup.fieldGroupId, updateFreq, maxKeepAge, maxKeepSamples)
        dcgm_structs._dcgmCheckReturn(ret)

    
    '''
    tell DCGM to stop recording samples for a given field group

    fieldGroup: DcgmFieldGroup() instance tracking the fields we want to unwatch.
    '''
    def UnwatchFields(self, fieldGroup):
        ret = dcgm_agent.dcgmUnwatchFields(self._dcgmHandle.handle, self._groupId, fieldGroup.fieldGroupId)
        dcgm_structs._dcgmCheckReturn(ret)
    
    '''
    Get the most recent values for each field in a field collection

    fieldGroup: DcgmFieldGroup() instance tracking the fields we want to watch.

    Returns DcgmFieldValueCollection object. Use its .values[gpuId][fieldId][0].value to access values
    '''
    def GetLatest(self, fieldGroup):
        dfvc = dcgm_field_helpers.DcgmFieldValueCollection(self._dcgmHandle.handle, self._groupId)
        dfvc.GetLatestValues(fieldGroup)
        return dfvc
    
    '''
    Get the most recent values for each field in a field collection

    fieldGroup: DcgmFieldGroup() instance tracking the fields we want to watch.

    Returns DcgmFieldValueEntityCollection object. Use its .values[entityGroupId][entityId][fieldId][0].value to access values
    '''
    def GetLatest_v2(self, fieldGroup):
        dfvec = dcgm_field_helpers.DcgmFieldValueEntityCollection(self._dcgmHandle.handle, self._groupId)
        dfvec.GetLatestValues(fieldGroup)
        return dfvec

    '''
    Get the new values for each field in a field collection since the last
    collection.

    dfvc:       DcgmFieldValueCollection() instance. Will return a
                DcgmFieldValueCollection with values since the one passed in.
                Pass None for the first call to get one for subsequent calls.
                On subsequent calls, pass what was returned.
    fieldGroup: DcgmFieldGroup() instance tracking the fields we want to watch.

    Returns DcgmFieldValueCollection object. Use its .values[gpuId][fieldId][*].value to access values
    '''
    def GetAllSinceLastCall(self, dfvc, fieldGroup):
        if dfvc == None:
            dfvc = dcgm_field_helpers.DcgmFieldValueCollection(self._dcgmHandle.handle, self._groupId)
            dfvc.GetLatestValues(fieldGroup)
        else:
            dfvc.GetAllSinceLastCall(fieldGroup)
            # We used to expect at least one value (GetLatestValues), so this
            # ensures we provide one at the risk of repetition. This should not
            # happen if we call this function infrequently enough (slower than
            # the sampling rate).
            if dfvc.values.len() == 0:
                dfvc.GetLatestValues(fieldGroup)
        return dfvc

    '''
    Gets more values for each field in a field entity collection

    dfvec:      DcgmFieldValueEntityCollection() instance. Will return a
                DcgmFieldValueEntityCollection with values since the one passed
                in. Pass None for the first call to get one for subsequent
                calls. On subsequent calls, pass what was returned.

    fieldGroup: DcgmFieldGroup() instance tracking the fields we want to watch.

    Returns DcgmFieldValueEntityCollection object. Use its .values[entityGroupId][entityId][fieldId][*].value to access values
    '''
    def GetAllSinceLastCall_v2(self, dvfec, fieldGroup):
        if dfvec == None:
            dfvec = dcgm_field_helpers.DcgmFieldValueEntityCollection(self._dcgmHandle.handle, self._groupId)
            dfvec.GetLastestValues_v2(fieldGroup)
        else:
            dfvec.GetAllSinceLastCall_v2(fieldGroup)
            # We used to expect at least one value (GetLatestValues), so this
            # ensures we provide one at the risk of repetition. This should not
            # happen if we call this function infrequently enough (slower than
            # the sampling rate).
            if dfvec.values.len() == 0:
                dfvec.GetLatestValues_v2(fieldGroup)

        return dfvec

    '''
    Convenience alias for DcgmHandle.UpdateAllFields(). All fields on the system will be updated, not
    just this group's.
    '''
    def UpdateAllFields(self, waitForUpdate):
        self._dcgmHandle.UpdateAllFields(waitForUpdate)

class DcgmGroupHealth:
    def __init__(self, dcgmHandle, groupId, dcgmGroup):
        self._dcgmHandle = dcgmHandle
        self._groupId = groupId
        self._dcgmGroup = dcgmGroup

    '''
    Enable health checks for this group

    systems: A bitmask of dcgm_structs.DCGM_HEALTH_WATCH_? definitions of which health checks to enable
    updateInterval: How often DCGM should request new health data from the driver in usec
    maxKeepAge: How long DCGM should keep health data around once it has been retrieved from the driver in seconds
    '''
    def Set(self, systems, updateInterval = None, maxKeepAge = None):
        if updateInterval is None or maxKeepAge is None:
            ret = dcgm_agent.dcgmHealthSet(self._dcgmHandle.handle, self._groupId, systems)
        else:
            ret = dcgm_agent.dcgmHealthSet_v2(self._dcgmHandle.handle, self._groupId, systems, updateInterval, maxKeepAge)
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Retrieve the current state of the DCGM health check system

    Returns a bitmask of dcgm_structs.DCGM_HEALTH_WATCH_? definitions of which health checks are currently enabled
    '''
    def Get(self):
        systems = dcgm_agent.dcgmHealthGet(self._dcgmHandle.handle, self._groupId)
        return systems

    '''
    Check the configured watches for any errors/failures/warnings that have occurred
    since the last time this check was invoked.  On the first call, stateful information
    about all of the enabled watches within a group is created but no error results are
    provided.  On subsequent calls, any error information will be returned.

    @param version    IN: Allows the caller to use an older version of this request. Should be 
                          dcgm_structs.dcgmHealthResponse_version4

    Returns a dcgm_structs.c_dcgmHealthResponse_* object that contains results for each GPU/entity
    '''
    def Check(self, version = dcgm_structs.dcgmHealthResponse_version4):
        resp = dcgm_agent.dcgmHealthCheck(self._dcgmHandle.handle, self._groupId, version)
        return resp

class DcgmGroupPolicy:
    def __init__(self, dcgmHandle, groupId, dcgmGroup):
        self._dcgmHandle = dcgmHandle
        self._groupId = groupId
        self._dcgmGroup = dcgmGroup
        
    '''
    Get the current violation policy inside the policy manager. Given a groupId, a number of 
    policy structures are retrieved.
    
    @param statusHandle              IN/OUT: pydcgm.DcgmStatus for the resulting status of the operation. Pass it as None 
                                             if the detailed error information for the operation is not needed (default).
            
    Returns a list of dcgm_structs.c_dcgmPolicy_v1 with the same length as the number of GPUs in the group.  
    The index of an entry corresponds to a given GPU ID in the group.  Throws an exception on error.
    '''
    def Get(self, statusHandle=None):
        if statusHandle:
            statusHandle = statusHandle.handle
        count = len(self._dcgmGroup.GetGpuIds())
        if count <= 0:
            raise pydcgm.DcgmException("This group has no GPUs, cannot retrieve policies")
        return dcgm_agent.dcgmPolicyGet(self._dcgmHandle.handle, self._groupId, count, statusHandle)
    
    '''
    Set the current violation policy inside the policy manager.  Given the conditions within "policy", 
    if a violation has occurred, subsequent action(s) may be performed to either 
    report or contain the failure.

    This API is only supported on Tesla GPUs and will throw DCGMError_NotSupported if called on non-Tesla GPUs.
    
    @param policy                        IN: dcgm_structs.c_dcgmPolicy_v1 that will be applied to all GPUs in the group
    
    @param statusHandle              IN/OUT: pydcgm.DcgmStatus for the resulting status for the operation. Pass it as 
                                             None if the detailed error information for the operation is not needed (default).
            
    Returns Nothing. Throws an exception on error
    '''
    def Set(self, policy, statusHandle=None):
        if statusHandle:
            statusHandle = statusHandle.handle
        dcgm_agent.dcgmPolicySet(self._dcgmHandle.handle, self._groupId, policy, statusHandle)
    
    '''
    Register a function to be called when a specific policy condition (see dcgm_structs.c_dcgmPolicy_v1.condition) 
    has been violated.  This callback(s) will be called automatically when in DCGM_OPERATION_MODE_AUTO mode and only after 
    DcgmPolicy.Trigger when in DCGM_OPERATION_MODE_MANUAL mode.  
    All callbacks are made within a separate thread.

    This API is only supported on Tesla GPUs and will throw DCGMError_NotSupported if called on non-Tesla GPUs.
  
    @param condition                     IN: The set of conditions specified as an OR'd list 
                                             (see dcgm_structs.DCGM_POLICY_COND_*)
                                             for which to register a callback function
            
    @param beginCallback                 IN: A function that should be called should a violation occur.  This 
                                             function will be called prior to any actions specified by the policy are taken.
            
    @param finishCallback                IN: A reference to a function that should be called should a violation occur.  
                                             This function will be called after any action specified by the policy are completed.
    
    At least one callback must be provided that is not None.
    
    Returns Nothing. Throws an exception on error.
    '''
    def Register(self, condition, beginCallback=None, finishCallback=None):
        if beginCallback is None and finishCallback is None:
            raise pydcgm.DcgmException("At least 1 callback must be provided to register that is not None")
        dcgm_agent.dcgmPolicyRegister(self._dcgmHandle.handle, self._groupId, condition, beginCallback, finishCallback)
    
    '''
    Unregister a function to be called for a specific policy condition (see dcgm_structs.c_dcgmPolicy_v1.condition) .
    This function will unregister all callbacks for a given condition.
 
    @param condition                     IN: The set of conditions specified as an OR'd list 
                                             (see dcgm_structs.DCGM_POLICY_COND_*) 
                                             for which to unregister a callback function

    Returns Nothing. Throws an exception on error.
    '''
    def Unregister(self, condition):
        dcgm_agent.dcgmPolicyUnregister(self._dcgmHandle.handle, self._groupId, condition)

    '''
    Inform the policy manager loop to perform an iteration and trigger the callbacks of any
    registered functions. Callback functions will be called from a separate thread as the calling function.
 
    Note: The GPU monitoring and management agent must call this method periodically if the operation 
    mode is set to manual mode (DCGM_OPERATION_MODE_MANUAL) during initialization 
    (\ref DcgmHandle.__init__).
    
    Returns Nothing. Throws an exception if there is a generic error that the 
    policy manager was unable to perform another iteration.
    '''
    def Trigger(self):
        dcgm_agent.dcgmPolicyTrigger(self._dcgmHandle.handle)

class DcgmGroupDiscovery:
    def __init__(self, dcgmHandle, groupId, dcgmGroup):
        self._dcgmHandle = dcgmHandle
        self._groupId = groupId
        self._dcgmGroup = dcgmGroup

    '''
    Get the topology for this group

    Returns a c_dcgmGroupTopology_v1 object representing the topology for this group
    '''
    def GetTopology(self):
        return dcgm_agent.dcgmGetGroupTopology(self._dcgmHandle.handle, self._groupId)

class DcgmGroupStats:
    def __init__(self, dcgmHandle, groupId, dcgmGroup):
        self._dcgmHandle = dcgmHandle
        self._groupId = groupId
        self._dcgmGroup = dcgmGroup

    '''
    Tell DCGM to start recording samples for fields returned from GetPidInfo()

    updateFreq: How often to update these fields in usec
    maxKeepAge: How long to keep data for these fields in seconds
    maxKeepSamples: Maximum number of samples to keep per field. 0=no limit

    Once the field collection is watched, it will update whenever the next update
    loop occurs. If you want to query these values immediately, use
    handle.UpdateAllFields(True) to make sure that the fields have updated at least once.
    '''
    def WatchPidFields(self, updateFreq, maxKeepAge, maxKeepSamples):
        ret = dcgm_agent.dcgmWatchPidFields(self._dcgmHandle.handle, self._groupId, updateFreq, maxKeepAge, maxKeepSamples)
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Get process stats for a given PID on this GPU group

    You must call WatchPidFields() before this query for this method to return any results

    Returns a dcgm_structs.c_dcgmPidInfo_v2 structure
    '''
    def GetPidInfo(self, pid):
        return dcgm_agent.dcgmGetPidInfo(self._dcgmHandle.handle, self._groupId, pid)

    '''
    Tell DCGM to start recording samples for fields returned from GetJobStats()

    updateFreq: How often to update these fields in usec
    maxKeepAge: How long to keep data for these fields in seconds
    maxKeepSamples: Maximum number of samples to keep per field. 0=no limit

    Once the fields are watched, they will update whenever the next update
    loop occurs. If you want to query these values immediately, use
    handle.UpdateAllFields(True) to make sure that the fields have updated at least once.
    '''
    def WatchJobFields(self, updateFreq, maxKeepAge, maxKeepSamples):
        ret = dcgm_agent.dcgmWatchJobFields(self._dcgmHandle.handle, self._groupId, updateFreq, maxKeepAge, maxKeepSamples)
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Start collecting stats for a named job for this GPU group

    Calling this will tell DCGM to start tracking stats for the given jobId. Stats tracking
    will end when StopJobStats() is called

    You must call WatchJobFields() before this call to tell DCGM to start sampling the fields
    that are returned from GetJobStats().

    jobId is a unique string identifier for this job. An exception will be thrown if this is not unique

    Returns Nothing (Will throw exception on error)
    '''
    def StartJobStats(self, jobId):
        ret = dcgm_agent.dcgmJobStartStats(self._dcgmHandle.handle, self._groupId, jobId)
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Stop collecting stats for a named job

    Calling this will tell DCGM to stop collecting stats for a job that was previously started
    with StartJobStats().

    jobId is the unique string that was passed as jobId to StartJobStats.

    Returns Nothing (Will throw exception on error)
    '''
    def StopJobStats(self, jobId):
        ret = dcgm_agent.dcgmJobStopStats(self._dcgmHandle.handle, jobId)
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Get stats for a job that was started with StartJobStats. If StopJobStats has not been called yet,
    this will get stats from when the job started until now. If StopJob was called prior to
    this, the returned Stats will go from when StartJobStats was called to when StopJobStats was called.

    jobId is the unique string that was passed as jobId to StartJobStats and StopJobStats

    Returns a dcgm_structs.c_dcgmJobInfo_v3 structure. Throws an exception on error
    '''
    def GetJobStats(self, jobId):
        ret = dcgm_agent.dcgmJobGetStats(self._dcgmHandle.handle, jobId)
        return ret

    '''
    This API tells DCGM to stop tracking the job given by jobId. After this call, you will no longer
    be able to call GetJobStats() on this jobId. However, you will be able to reuse jobId after
    this call.

    jobId is the unique string that was passed as jobId to StartJobStats and StopJobStats

    Returns Nothing (Will throw exception on error)
    '''
    def RemoveJob(self, jobId):
        ret = dcgm_agent.dcgmJobRemove(self._dcgmHandle.handle, jobId)
        return ret

    '''
    This API tells DCGM to stop tracking all jobs. After this call, you will no longer
    be able to call dcgmJobGetStats() any jobs until you call StartJobStats() again.
    You will be able to reuse any previously-used jobIds after this call.

    Returns Nothing (Will throw exception on error)
    '''
    def RemoveAllJobs(self):
        ret = dcgm_agent.dcgmJobRemoveAll(self._dcgmHandle.handle)
        return ret

class DcgmGroupAction:
    def __init__(self, dcgmHandle, groupId, dcgmGroup):
        self._dcgmHandle = dcgmHandle
        self._groupId = groupId
        self._dcgmGroup = dcgmGroup

    '''
    Inform the action manager to perform a manual validation of a group of GPUs on the system

    validate is what sort of validation to do. See dcgm_structs.DCGM_POLICY_VALID_* defines.

    Returns a dcgm_structs.c_dcgmDiagResponse_v5 instance
    '''
    def Validate(self, validate):
        runDiagInfo = dcgm_structs.c_dcgmRunDiag_v7()
        runDiagInfo.version = dcgm_structs.dcgmRunDiag_version7
        runDiagInfo.validate = validate
        runDiagInfo.groupId = self._groupId

        ret = dcgm_agent.dcgmActionValidate_v2(self._dcgmHandle.handle, runDiagInfo)
        return ret

    '''
    Run a diagnostic on this group of GPUs.

    diagLevel is the level of diagnostic desired. See dcgm_structs.DCGM_DIAG_LVL_* constants.

    Returns a dcgm_structs.c_dcgmDiagResponse_v5 instance
    '''
    def RunDiagnostic(self, diagLevel):
        ret = dcgm_agent.dcgmRunDiagnostic(self._dcgmHandle.handle, self._groupId, diagLevel)
        return ret

    '''
    Run a specific diagnostic test on this group of GPUs.
    testName is the name of the specific test that should be invoked.
    Returns a dcgm_structs.c_dcgmDiagResponse_v5 instance
    '''
    def RunSpecificTest(self, testName):
        runDiagInfo = dcgm_structs.c_dcgmRunDiag_v7()
        runDiagInfo.version = dcgm_structs.dcgmRunDiag_version7
        for i in range(len(testName)):
            runDiagInfo.testNames[0][i] = testName[i]
        runDiagInfo.groupId = self._groupId
        runDiagInfo.validate = dcgm_structs.DCGM_POLICY_VALID_NONE
        response = dcgm_agent.dcgmActionValidate_v2(self._dcgmHandle.handle, runDiagInfo)
        return response

class DcgmGroupProfiling:
    def __init__(self, dcgmHandle, groupId, dcgmGroup):
        self._dcgmHandle = dcgmHandle
        self._groupId = groupId
        self._dcgmGroup = dcgmGroup

    '''
    Get a list of the profiling metric groups available for this group of entities

    Returns a dcgm_structs.c_dcgmProfGetMetricGroups_v2 instance
    '''
    def GetSupportedMetricGroups(self):
        ret = dcgm_agent.dcgmProfGetSupportedMetricGroups(self._dcgmHandle.handle, self._groupId)
        return ret
    
    '''
    Watch a list of profiling field IDs

    Returns a dcgm_structs.c_dcgmProfWatchFields_v1 instance
    '''
    def WatchFields(self, fieldIds, updateFreq, maxKeepAge, maxKeepSamples):
        ret = dcgm_agent.dcgmProfWatchFields(self._dcgmHandle.handle, fieldIds, self._groupId, 
                                             updateFreq, maxKeepAge, maxKeepSamples)
        return ret
    
    '''
    Unwatch all profiling fields that were watched with WatchFields

    Returns a dcgm_structs.c_dcgmProfUnwatchFields_v1 instance
    '''
    def UnwatchFields(self):
        ret = dcgm_agent.dcgmProfUnwatchFields(self._dcgmHandle.handle, self._groupId)
        return ret

class DcgmGroup:
    '''
    Constructor.

    Either groupId OR groupName must be provided as a parameter.
    This will set which GPU group this object is bound to

    groupId=DCGM_GROUP_ALL_GPUS creates a group with all GPUs. Passing an existing groupId will
    not create an additional group.
    If groupName is provided, an empty group (No GPUs) of name groupName will be created. This group
    will be destroyed when this object goes out of scope or is deleted with del().
    groupType is the type of group to create. See dcgm_structs.DCGM_GROUP_? constants.
    '''
    def __init__(self, dcgmHandle, groupId=None, groupName=None, groupType=dcgm_structs.DCGM_GROUP_EMPTY):
        self._dcgmHandle = dcgmHandle

        if groupId is None and groupName is None:
            raise pydcgm.DcgmException("Either groupId or groupName is required")

        if groupId is not None:
            self._groupId = groupId
        else:
            self._groupId = dcgm_agent.dcgmGroupCreate(self._dcgmHandle.handle, groupType, groupName)

        #Create namespace classes
        self.config = DcgmGroupConfig(self._dcgmHandle, self._groupId, self)
        self.samples = DcgmGroupSamples(self._dcgmHandle, self._groupId, self)
        self.health = DcgmGroupHealth(self._dcgmHandle, self._groupId, self)
        self.policy = DcgmGroupPolicy(self._dcgmHandle, self._groupId, self)
        self.discovery = DcgmGroupDiscovery(self._dcgmHandle, self._groupId, self)
        self.stats = DcgmGroupStats(self._dcgmHandle, self._groupId, self)
        self.action = DcgmGroupAction(self._dcgmHandle, self._groupId, self)
        self.profiling = DcgmGroupProfiling(self._dcgmHandle, self._groupId, self)

    '''
    Remove this group from DCGM. This object will no longer be valid after this call.
    '''
    def Delete(self):
        del self.config
        self.config = None
        del self.samples
        self.samples = None
        del self.health
        self.health = None
        del self.policy
        self.policy = None
        del self.discovery
        self.discovery = None
        del self.stats
        self.stats = None
        del self.action
        self.action = None
        del self.profiling
        self.profiling = None

        #Delete the group we created if we're not using the special all-GPU group
        if self._groupId is not None and not self._IsGroupIdStatic():
            ret = dcgm_agent.dcgmGroupDestroy(self._dcgmHandle.handle, self._groupId)
            dcgm_structs._dcgmCheckReturn(ret)

        self._groupId = None

    '''
    Private method to determine if our groupId is a predefined one
    '''
    def _IsGroupIdStatic(self):
        if self._groupId == dcgm_structs.DCGM_GROUP_ALL_GPUS or \
           self._groupId == dcgm_structs.DCGM_GROUP_ALL_NVSWITCHES:
            return True
        return False

    '''
    Add a GPU to this group

    gpuId is the GPU ID to add to our group

    Returns Nothing. Throws an exception on error
    '''
    def AddGpu(self, gpuId):
        if self._IsGroupIdStatic():
            raise pydcgm.DcgmException("Can't add a GPU to a static group")

        ret = dcgm_agent.dcgmGroupAddDevice(self._dcgmHandle.handle, self._groupId, gpuId)
        dcgm_structs._dcgmCheckReturn(ret)
    
    '''
    Add an entity to this group

    entityGroupId is DCGM_FE_? constant of the entity group this entity belongs to
    entityId is the entity to add to this group

    Returns Nothing. Throws an exception on error
    '''
    def AddEntity(self, entityGroupId, entityId):
        if self._IsGroupIdStatic():
            raise pydcgm.DcgmException("Can't add an entity to a static group")

        ret = dcgm_agent.dcgmGroupAddEntity(self._dcgmHandle.handle, self._groupId, entityGroupId, entityId)
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Remove a GPU from this group

    gpuId is the GPU ID to remove from our group

    Returns Nothing. Throws an exception on error
    '''
    def RemoveGpu(self, gpuId):
        if self._IsGroupIdStatic():
            raise pydcgm.DcgmException("Can't remove a GPU from a static group")

        ret = dcgm_agent.dcgmGroupRemoveDevice(self._dcgmHandle.handle, self._groupId, gpuId)
        dcgm_structs._dcgmCheckReturn(ret)
    
    '''
    Remove an entity from this group

    entityGroupId is DCGM_FE_? constant of the entity group this entity belongs to
    entityId is the entity to remove from this group

    Returns Nothing. Throws an exception on error
    '''
    def RemoveEntity(self, entityGroupId, entityId):
        if self._IsGroupIdStatic():
            raise pydcgm.DcgmException("Can't remove an entity from a static group")

        ret = dcgm_agent.dcgmGroupRemoveEntity(self._dcgmHandle.handle, self._groupId, entityGroupId, entityId)
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Get an array of GPU ids that are part of this group

    Note: this ignores non-GPU members of the group

    Returns a list of GPU ids. Throws an exception on error
    '''
    def GetGpuIds(self):
        groupInfo = dcgm_agent.dcgmGroupGetInfo(self._dcgmHandle.handle, self._groupId)
        groupGpuIds = []
        for i in range(groupInfo.count):
            if groupInfo.entityList[i].entityGroupId != dcgm_fields.DCGM_FE_GPU:
                continue
            groupGpuIds.append(groupInfo.entityList[i].entityId)
        return groupGpuIds

    '''
    Get an array of entities that are part of this group

    Returns a list of c_dcgmGroupEntityPair_t structs. Throws an exception on error
    '''
    def GetEntities(self):
        groupInfo = dcgm_agent.dcgmGroupGetInfo(self._dcgmHandle.handle, self._groupId)
        entities = groupInfo.entityList[0:groupInfo.count]
        return entities

    '''
    Get the groupId of this object

    Returns our groupId
    '''
    def GetId(self):
        return self._groupId

