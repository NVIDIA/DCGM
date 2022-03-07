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
import ctypes

class DcgmSystemDiscovery:
    '''
    Constructor
    '''
    def __init__(self, dcgmHandle):
        self._dcgmHandle = dcgmHandle

    '''
    Get all IDs of the GPUs that DCGM knows about. To get only GPUs that DCGM support,
    use GetAllSupportedGpuIds().

    Returns an array of GPU IDs. Each of these can be passed to DcgmGroup::AddGpu()
    '''
    def GetAllGpuIds(self):
        gpuIds = dcgm_agent.dcgmGetAllDevices(self._dcgmHandle.handle)
        return gpuIds

    '''
    Get all of IDs of the GPUs that DCGM supports. This will exclude unsupported
    GPUs

    Returns an array of GPU IDs. Each of these can be passed to DcgmGroup::AddGpu()
    '''
    def GetAllSupportedGpuIds(self):
        gpuIds = dcgm_agent.dcgmGetAllSupportedDevices(self._dcgmHandle.handle)
        return gpuIds

    '''
    Get some basic GPU attributes for a given GPU ID.

    Returns a dcgm_structs.c_dcgmDeviceAttributes_v1() object for the given GPU
    '''
    def GetGpuAttributes(self, gpuId):
        return dcgm_agent.dcgmGetDeviceAttributes(self._dcgmHandle.handle, gpuId)

    '''
    Get topology information for a given GPU ID

    Returns a dcgm_structs.c_dcgmDeviceTopology_v1 structure representing the topology for the given GPU
    '''
    def GetGpuTopology(self, gpuId):
        return dcgm_agent.dcgmGetDeviceTopology(self._dcgmHandle.handle, gpuId)

    '''
    Get all entityIds of the entities that DCGM knows about.

    entityGroupId IN: DCGM_FE_? constant of the entity group to fetch the entities of
    onlyActive    IN: Boolean as to whether to fetch entities that are supported by DCGM (True)
                      or all entity IDs (False)

    Returns an array of entity IDs. Each of these can be passed to DcgmGroup::AddEntity()
    '''
    def GetEntityGroupEntities(self, entityGroupId, onlySupported):
        flags = 0
        if onlySupported:
            flags |= dcgm_structs.DCGM_GEGE_FLAG_ONLY_SUPPORTED
        entityIds = dcgm_agent.dcgmGetEntityGroupEntities(self._dcgmHandle.handle, entityGroupId, flags)
        return entityIds
    
    '''
    Get the status of all of the NvLink links in the system.

    Returns a dcgm_structs.c_dcgmNvLinkStatus_v2 object.
    '''
    def GetNvLinkLinkStatus(self):
        return dcgm_agent.dcgmGetNvLinkLinkStatus(self._dcgmHandle.handle)
    
    '''
    From a bitmask of input gpu ids, return a bitmask of numGpus GPUs which identifies the topologically
    closest GPUs to use for a single job. DCGM will consider CPU affinities and NVLink connection speeds
    to determine the closest.
    hintFlags can instruct DCGM to consider GPU health or not. By default, unhealthy GPUs are excluded from
    consideration.
    '''
    def SelectGpusByTopology(self, inputGpuIds, numGpus, hintFlags):
        return dcgm_agent.dcgmSelectGpusByTopology(self._dcgmHandle.handle, inputGpuIds, numGpus, hintFlags)

class DcgmSystemIntrospect:
    '''
    Class to access the system-wide introspection modules of DCGM
    '''
    
    def __init__(self, dcgmHandle):
        self._handle = dcgmHandle
        self.state = DcgmSystemIntrospectState(dcgmHandle)
        self.memory = DcgmSystemIntrospectMemory(dcgmHandle)
        self.execTime = DcgmSystemIntrospectExecTime(dcgmHandle)
        self.cpuUtil = DcgmSystemIntrospectCpuUtil(dcgmHandle)
        
    def UpdateAll(self, waitForUpdate=True):
        dcgm_agent.dcgmIntrospectUpdateAll(self._handle.handle, waitForUpdate)
        
class DcgmSystemIntrospectState:
    '''
    Class to access the state of DCGM introspection gathering
    '''
    
    def __init__(self, dcgmHandle):
        self._dcgmHandle = dcgmHandle
    
    '''
    Toggle the state of dcgm introspection data collection
    
    enabledState: any property of dcgm_structs.DCGM_INTROSPECT_STATE
    '''
    def toggle(self, enabledState):
        dcgm_agent.dcgmIntrospectToggleState(self._dcgmHandle.handle, enabledState)

class DcgmSystemIntrospectMemory:
    '''
    Class to access information about the memory usage of DCGM itself
    '''
    
    def __init__(self, dcgmHandle):
        self._dcgmHandle = dcgmHandle;
        
    def GetForFieldGroup(self, fieldGroup, waitIfNoData=True):
        '''
        Get the current amount of memory used to store a field group that DCGM is watching.
        
        fieldGroup:        DcgmFieldGroup() instance
        waitIfNoData:      wait for metadata to be updated if it's not available
                      
        Returns a dcgm_structs.c_dcgmIntrospectFullMemory_v1 object
        Raises an exception for DCGM_ST_NOT_WATCHED if the field group is not watched.
        Raises an exception for DCGM_ST_NO_DATA if no data is available yet and \ref waitIfNoData is False
        '''
        introspectContext = dcgm_structs.c_dcgmIntrospectContext_v1()
        introspectContext.version = dcgm_structs.dcgmIntrospectContext_version1
        introspectContext.introspectLvl = dcgm_structs.DCGM_INTROSPECT_LVL.FIELD_GROUP
        introspectContext.fieldGroupId = fieldGroup.fieldGroupId
        
        return dcgm_agent.dcgmIntrospectGetFieldsMemoryUsage(self._dcgmHandle.handle, 
                                                             introspectContext, 
                                                             waitIfNoData)

    def GetForAllFields(self, waitIfNoData=True):
        '''
        Get the current amount of memory used to store all fields that DCGM is watching.
        
        waitIfNoData: wait for metadata to be updated if it's not available.
                      
        Returns a dcgm_structs.c_dcgmIntrospectFullMemory_v1 object
        Raises an exception for DCGM_ST_NO_DATA if no data is available yet and \ref waitIfNoData is False
        '''
        introspectContext = dcgm_structs.c_dcgmIntrospectContext_v1()
        introspectContext.version = dcgm_structs.dcgmIntrospectContext_version1
        introspectContext.introspectLvl = dcgm_structs.DCGM_INTROSPECT_LVL.ALL_FIELDS
        
        return dcgm_agent.dcgmIntrospectGetFieldsMemoryUsage(self._dcgmHandle.handle, 
                                                             introspectContext, 
                                                             waitIfNoData)
    
    def GetForHostengine(self, waitIfNoData=True):
        '''
        Retrieve the total amount of virtual memory that the hostengine process is currently using.
        This measurement represents both the resident set size (what is currently in RAM) and
        the swapped memory that belongs to the process.
        
        waitIfNoData:      wait for metadata to be updated if it's not available
                      
        Returns a dcgm_structs.c_dcgmIntrospectMemory_v1 object
        Raises an exception for DCGM_ST_NO_DATA if no data is available yet and \ref waitIfNoData is False
        '''
        return dcgm_agent.dcgmIntrospectGetHostengineMemoryUsage(self._dcgmHandle.handle, waitIfNoData)
        
    
class DcgmSystemIntrospectExecTime:
    '''
    Class to access information about the execution time of DCGM itself
    '''
    
    def __init__(self, dcgmHandle):
        self._dcgmHandle = dcgmHandle;
        
    def GetForFieldGroup(self, fieldGroup, waitIfNoData=True):
        '''
        Get the total execution time since startup that was used for updating a 
        field group that DCGM is watching.
        
        fieldGroup:        DcgmFieldGroup() instance
        waitIfNoData:      wait for metadata to be updated if it's not available
                      
        Returns a dcgm_structs.c_dcgmIntrospectFullFieldsExecTime_v2 object
        Raises an exception for DCGM_ST_NOT_WATCHED if the field group is not watched.
        Raises an exception for DCGM_ST_NO_DATA if no data is available yet and \ref waitIfNoData is False
        '''
        introspectContext = dcgm_structs.c_dcgmIntrospectContext_v1()
        introspectContext.version = dcgm_structs.dcgmIntrospectContext_version1
        introspectContext.introspectLvl = dcgm_structs.DCGM_INTROSPECT_LVL.FIELD_GROUP
        introspectContext.fieldGroupId = fieldGroup.fieldGroupId
        
        return dcgm_agent.dcgmIntrospectGetFieldsExecTime(self._dcgmHandle.handle, 
                                                          introspectContext, 
                                                          waitIfNoData)

    def GetForAllFields(self, waitIfNoData=True):
        '''
        Get the total execution time since startup that was used for updating 
        all fields that DCGM is watching.
        
        waitIfNoData:      wait for metadata to be updated if it's not available
                      
        Returns a dcgm_structs.c_dcgmIntrospectFullFieldsExecTime_v2 object
        Raises an exception for DCGM_ST_NOT_WATCHED if the field group is not watched.
        Raises an exception for DCGM_ST_NO_DATA if no data is available yet and \ref waitIfNoData is False
        '''
        introspectContext = dcgm_structs.c_dcgmIntrospectContext_v1()
        introspectContext.version = dcgm_structs.dcgmIntrospectContext_version1
        introspectContext.introspectLvl = dcgm_structs.DCGM_INTROSPECT_LVL.ALL_FIELDS
        
        return dcgm_agent.dcgmIntrospectGetFieldsExecTime(self._dcgmHandle.handle, 
                                                          introspectContext, 
                                                          waitIfNoData)

class DcgmSystemIntrospectCpuUtil:
    '''
    Class to access information about the CPU Utilization of DCGM
    '''
    
    def __init__(self, dcgmHandle):
        self._dcgmHandle = dcgmHandle
        
    def GetForHostengine(self, waitIfNoData=True):
        '''
        Get the current CPU Utilization of the hostengine process.
        
        waitIfNoData:      wait for metadata to be updated if it's not available
                      
        Returns a dcgm_structs.c_dcgmIntrospectCpuUtil_v1 object
        Raises an exception for DCGM_ST_NO_DATA if no data is available yet and \ref waitIfNoData is False
        '''
        return dcgm_agent.dcgmIntrospectGetHostengineCpuUtilization(self._dcgmHandle.handle, waitIfNoData)

'''
Class to encapsulate DCGM field-metadata requests
'''
class DcgmSystemFields:

    def GetFieldById(self, fieldId):
        '''
        Get a field's metadata by its dcgm_fields.DCGM_FI_* field ID

        fieldId: dcgm_fields.DCGM_FI_* field ID of the field

        Returns a dcgm_fields.c_dcgm_field_meta_t struct on success or None on error.
        '''
        return dcgm_fields.DcgmFieldGetById(fieldId)

    def GetFieldByTag(self, tag):
        '''
        Get a field's metadata by its tag name. Ex: 'brand'

        tag: Tag name of the field

        Returns a dcgm_fields.c_dcgm_field_meta_t struct on success or None on error.
        '''
        return dcgm_fields.DcgmFieldGetByTag(tag)

'''
Class to encapsulate DCGM module management and introspection
'''
class DcgmSystemModules:
    '''
    Constructor
    '''
    def __init__(self, dcgmHandle): 
        self._dcgmHandle = dcgmHandle
    
    '''
    Blacklist a module from being loaded by DCGM.

    moduleId a dcgm_structs.dcgmModuleId* ID of the module to blacklist

    Returns: Nothing.
    Raises a DCGM_ST_IN_USE exception if the module was already loaded
    '''
    def Blacklist(self, moduleId):
        dcgm_agent.dcgmModuleBlacklist(self._dcgmHandle.handle, moduleId)

    '''
    Get the statuses of all of the modules in DCGM

    Returns: a dcgm_structs.c_dcgmModuleGetStatuses_v1 structure.
    '''
    def GetStatuses(self):
        return dcgm_agent.dcgmModuleGetStatuses(self._dcgmHandle.handle)


'''
Class to encapsulate DCGM profiling
'''
class DcgmSystemProfiling:
    '''
    Constructor
    '''
    def __init__(self, dcgmHandle): 
        self._dcgmHandle = dcgmHandle
    
    '''
    Pause profiling activities in DCGM. This should be used when you are monitoring profiling fields
    from DCGM but want to be able to still run developer tools like nvprof, nsight systems, and nsight compute.
    Profiling fields start with DCGM_PROF_ and are in the field ID range 1001-1012.
    
    Call this API before you launch one of those tools and Resume() after the tool has completed.
    
    DCGM will save BLANK values while profiling is paused. 
    Calling this while profiling activities are already paused is fine and will be treated as a no-op.
    '''
    def Pause(self):
        return dcgm_agent.dcgmProfPause(self._dcgmHandle.handle)
    
    '''
    Resume profiling activities in DCGM that were previously paused with Pause().

    Call this API after you have completed running other NVIDIA developer tools to reenable DCGM
    profiling metrics.
    
    DCGM will save BLANK values while profiling is paused. 
    
    Calling this while profiling activities have already been resumed is fine and will be treated as a no-op.
    '''
    def Resume(self):
        return dcgm_agent.dcgmProfResume(self._dcgmHandle.handle)

'''
Class to encapsulate global DCGM methods. These apply to a single DcgmHandle, provided to the constructor
'''
class DcgmSystem:
    '''
    Constructor

    dcgmHandle is a pydcgm.DcgmHandle instance of the connection that will be used by all methods of this class
    '''
    def __init__(self, dcgmHandle):
        self._dcgmHandle = dcgmHandle

        #Child classes
        self.discovery = DcgmSystemDiscovery(self._dcgmHandle)
        self.introspect = DcgmSystemIntrospect(self._dcgmHandle)
        self.fields = DcgmSystemFields()
        self.modules = DcgmSystemModules(self._dcgmHandle)
        self.profiling = DcgmSystemProfiling(self._dcgmHandle)

    '''
    Request that the host engine perform a field value update cycle. If the host
    engine was starting in DCGM_OPERATION_MODE_MANUAL, calling this method is
    the only way that field values will be updated.

    Note that performing a field value update cycle does not update every field.
    It only update fields that are newly watched or fields that haven't updated
    in enough time to warrant updating again, based on their update frequency.

    waitForUpdate specifies whether this function call should block until the
    field value update loop is complete or not. Use True if you intend to query
    values immediately after calling this.
    '''
    def UpdateAllFields(self, waitForUpdate):
        ret = dcgm_agent.dcgmUpdateAllFields(self._dcgmHandle.handle, waitForUpdate)
        #Throw an exception on error
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Get a DcgmGroup instance for the default all-GPUs group. This object is used to
    perform operations on a group of GPUs. See DcgmGroup.py for details.

    AddGpu() and RemoveGpu() operations are not allowed on the default group
    '''
    def GetDefaultGroup(self):
        return pydcgm.DcgmGroup(self._dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)

    '''
    Get an instance of DcgmGroup with no GPUs. Call AddGpu() on the returned
    object with GPU IDs from GetAllGpuIds() before performing actions on
    the returned DcgmGroup instance.

    groupName is the name of the group to create in the host engine. This name must be
    unique.

    Note: The group will be deleted from the host engine when the returned object goes out of scope
    '''
    def GetEmptyGroup(self, groupName):
        return pydcgm.DcgmGroup(self._dcgmHandle, groupName=groupName)

    '''
    Get an instance of DcgmGroup populated with the gpuIds provided

    groupName is the name of the group to create in the host engine. This name must be
    unique.
    gpuIds is the list of GPU IDs to add to the group

    Note: The group will be deleted from the host engine when the returned object goes out of scope
    '''
    def GetGroupWithGpuIds(self, groupName, gpuIds):
        newGroup = pydcgm.DcgmGroup(self._dcgmHandle, groupName=groupName)
        for gpuId in gpuIds:
            newGroup.AddGpu(gpuId)
        return newGroup

    '''
    Get an instance of DcgmGroup populated with the provided entities

    groupName is the name of the group to create in the host engine. This name must be
    unique.
    entities is the list of entity pairs (type and id) to add to the group

    Note: The group will be deleted from the host engine when the returned object goes out of scope
    '''
    def GetGroupWithEntities(self, groupName, entities):
        group = pydcgm.DcgmGroup(self._dcgmHandle, groupName=groupName)
        for entity in entities:
            group.AddEntity(entity.entityGroupId, entity.entityId)

        return group

    '''
    Get ids of all DcgmGroups of GPUs. This returns a list containing the ids of the DcgmGroups.
    '''

    def GetAllGroupIds(self):
        return dcgm_agent.dcgmGroupGetAllIds(self._dcgmHandle.handle)

    '''
    Get all all of the field groups in the system
    '''
    def GetAllFieldGroups(self):
        return dcgm_agent.dcgmFieldGroupGetAll(self._dcgmHandle.handle)

    '''
    Get a field group's id by its name.

    Returns: Field group ID if found
             None if not found
    '''
    def GetFieldGroupIdByName(self, name):
        allGroups = self.GetAllFieldGroups()
        for i in range(0, allGroups.numFieldGroups):
            if allGroups.fieldGroups[i].fieldGroupName == name:
                return ctypes.c_void_p(allGroups.fieldGroups[i].fieldGroupId)

        return None


