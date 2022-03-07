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
import subprocess
import signal, os
import pydcgm
import dcgm_structs
import threading
import dcgm_fields
import sys
import logging

defaultFieldIds = [
    dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
    dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
    dcgm_fields.DCGM_FI_DEV_SM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
    dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
    dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
    dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_FREE,
    dcgm_fields.DCGM_FI_DEV_FB_USED,
    dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
    dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
    dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_MEM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP,
    dcgm_fields.DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
    dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL,
    dcgm_fields.DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
    dcgm_fields.DCGM_FI_DEV_PCIE_RX_THROUGHPUT
    ]

def entity_group_id_to_string(entityGroupId):
    if entityGroupId == dcgm_fields.DCGM_FE_GPU:
        return 'GPU'
    elif entityGroupId == dcgm_fields.DCGM_FE_VGPU:
        return 'VGPU'
    elif entityGroupId == dcgm_fields.DCGM_FE_SWITCH:
        return 'NVSWITCH'
    elif entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
        return 'GPU INSTANCE'
    elif entityGroupId == dcgm_fields.DCGM_FE_GPU_CI:
        return 'COMPUTE INSTANCE'
    else:
        return ''

class DcgmReader(object):
    ###########################################################################
    '''
    This function can be implemented as a callback in the class that inherits from DcgmReader
    to handle each field individually.
    By default, it passes a string with the gpu, field tag, and value to LogInfo()
    @params:
    gpuId : the id of the GPU this field is reporting on
    fieldId : the id of the field (ignored by default, may be useful for children)
    fieldTag : the string representation of the field id
    val : the value class that comes from DCGM (v.value is the value for the field)
    '''
    def CustomFieldHandler(self, gpuId, fieldId, fieldTag, val):
        print "GPU %s field %s=%s" % (str(gpuId), fieldTag, str(val.value))
                    
    ###########################################################################
    '''
    This function can be implemented as a callback in the class that inherits from DcgmReader
    to handle each field individually.
    By default, it passes a string with the gpu, field tag, and value to LogInfo()
    @params:
    entityGroupId : the type of entity this field is reporting on
    entityId : the id of the entity this field is reporting on
    fieldId : the id of the field (ignored by default, may be useful for children)
    fieldTag : the string representation of the field id
    val : the value class that comes from DCGM (v.value is the value for the field)
    '''
    def CustomFieldHandler_v2(self, entityGroupId, entityId, fieldId, fieldTag, val):
        print "%s %s field %s=%s" % (entity_group_id_to_string(entityGroupId), str(entityId), fieldTag, str(val.value))

    ###########################################################################
    '''
    This function can be implemented as a callback in the class that inherits from DcgmReader
    to handle all of the data queried from DCGM.
    By default, it will simply print the field tags and values for each GPU
    @params:
    fvs : Data in the format entityGroupId -> entityId -> values (dictionary of dictionaries)
    '''
    def CustomDataHandler_v2(self,fvs):
        for entityGroupId in fvs.keys():
            entityGroup = fvs[entityGroupId]

            for entityId in entityGroup.keys():
                entityFv = entityGroup[entityId]
                for fieldId in entityFv.keys():
                    if fieldId in self.m_dcgmIgnoreFields:
                        continue

                    val = entityFv[fieldId][-1]

                    if val.isBlank:
                        continue

                    fieldTag = self.m_fieldIdToInfo[fieldId].tag

                    self.CustomFieldHandler_v2(entityGroupId, entityId, fieldId, fieldTag, val)

    ###########################################################################
    '''
    This function can be implemented as a callback in the class that inherits from DcgmReader
    to handle all of the data queried from DCGM.
    By default, it will simply print the field tags and values for each GPU
    @params:
    fvs : Dictionary with gpuID as key and values as Value
    '''
    def CustomDataHandler(self, fvs):
        for gpuId in fvs.keys():
            gpuFv = fvs[gpuId]

            for fieldId in gpuFv.keys():
                if fieldId in self.m_dcgmIgnoreFields:
                    continue

                val = gpuFv[fieldId][-1]

                if val.isBlank:
                    continue

                fieldTag = self.m_fieldIdToInfo[fieldId].tag

                self.CustomFieldHandler(gpuId, fieldId, fieldTag, val)

    ###########################################################################
    def SetupGpuIdUUIdMappings(self):
        '''
        Populate the m_gpuIdToUUId map
        '''

        gpuIds = self.m_dcgmGroup.GetGpuIds()
        for gpuId in gpuIds:
            gpuInfo = self.m_dcgmSystem.discovery.GetGpuAttributes(gpuId)
            self.m_gpuIdToUUId[gpuId] = gpuInfo.identifiers.uuid

    ###########################################################################
    '''
    Constructor
    @params:
    hostname        : Address:port of the host to connect. Defaults to localhost
    fieldIds        : List of the field ids to publish. If it isn't specified, our default list is used.
    updateFrequency : Frequency of update in microseconds. Defauls to 10 seconds or 10000000 microseconds
    maxKeepAge      : Max time to keep data from NVML, in seconds. Default is 3600.0 (1 hour)
    ignoreList      : List of the field ids we want to query but not publish.
    gpuIds          : List of GPU IDs to monitor. If not provided, DcgmReader will monitor all GPUs on the system
    '''
    def __init__(self, hostname='localhost', fieldIds=defaultFieldIds, updateFrequency=10000000,
            maxKeepAge=3600.0, ignoreList=[], fieldGroupName='dcgm_fieldgroupData', gpuIds=None,
            entities=None):
        self.m_dcgmHostName = hostname
        self.m_publishFieldIds = fieldIds
        self.m_updateFreq = updateFrequency
        self.m_fieldGroupName = fieldGroupName
        self.m_requestedGpuIds = gpuIds
        self.m_requestedEntities = entities

        self.m_dcgmIgnoreFields = ignoreList #Fields not to publish
        self.m_maxKeepAge = maxKeepAge
        self.m_dcgmHandle = None
        self.m_dcgmSystem = None
        self.m_dcgmGroup = None
        self.m_closeHandle = False

        self.m_gpuIdToBusId = {} #GpuID => PCI-E busId string
        self.m_gpuIdToUUId = {} # FieldId => dcgm_fields.dcgm_field_meta_t
        self.m_fieldIdToInfo = {} #FieldId => dcgm_fields.dcgm_field_meta_t
        self.m_lock = threading.Lock() #DCGM connection start-up/shutdown is not thread safe. Just lock pessimistically
        self.m_debug = False

    ###########################################################################
    '''
    Define what should happen to this object at the beginning of a with
    block. In this case, nothing more is needed since the constructor should've
    been called.
    '''
    def __enter__(self):
        return self

    ###########################################################################
    '''
    Define the cleanup
    '''
    def __exit__(self, type, value, traceback):
        self.Shutdown()

    ###########################################################################
    '''
    This function intializes the dcgm from the specified directory
    and connects to host engine.
    '''
    def InitWrapped(self, path=None):
        dcgm_structs._dcgmInit(libDcgmPath=path)
        self.Reconnect()

    ###########################################################################
    '''
    This function tries to connect to hostengine and calls initwrapped to initialize
    the dcgm.
    '''
    def Init(self, libpath=None):
        with self.m_lock:
            try:
                self.InitWrapped(path=libpath)
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID):
                self.LogError("Can't connect to nv-hostengine. Is it down?")
                self.SetDisconnected()

    ###########################################################################
    '''
    Delete the dcgm group, dcgm system and dcgm handle and clear the attributes on shutdown.
    '''
    def SetDisconnected(self):
        #Force destructors since DCGM currently doesn't support more than one client connection per process
        if self.m_dcgmGroup is not None:
            del(self.m_dcgmGroup)
            self.m_dcgmGroup = None
        if self.m_dcgmSystem is not None:
            del(self.m_dcgmSystem)
            self.m_dcgmSystem = None
        if self.m_dcgmHandle is not None:
            del(self.m_dcgmHandle)
            self.m_dcgmHandle = None

    ##########################################################################
    '''
    This function calls the SetDisconnected function which disconnect from dcgm and clears
    dcgm handle and dcgm group.
    '''
    def Shutdown(self):
        with self.m_lock:
            if self.m_closeHandle == True:
                self.SetDisconnected()

    ############################################################################
    '''
    Turns debugging output on
    '''
    def AddDebugOutput(self):
        self.m_debug = True

    ############################################################################
    '''
    '''
    def InitializeFromHandle(self):
        self.m_dcgmSystem = self.m_dcgmHandle.GetSystem()

        if not self.m_requestedGpuIds and not self.m_requestedEntities:
            self.m_dcgmGroup = self.m_dcgmSystem.GetDefaultGroup()
        else:
            groupName = "dcgmreader_%d" % os.getpid()

            if self.m_requestedGpuIds:
                self.m_dcgmGroup = self.m_dcgmSystem.GetGroupWithGpuIds(groupName, self.m_requestedGpuIds)
                if self.m_requestedEntities:
                    for entity in self.m_requestedEntities:
                        self.m_dcgmGroup.AddEntity(entity.entityGroupId, entity.entityId)
            else:
                self.m_dcgmGroup = self.m_dcgmSystem.GetGroupWithEntities(groupName, self.m_requestedEntities)

        self.SetupGpuIdBusMappings()
        self.SetupGpuIdUUIdMappings()
        self.GetFieldMetadata()
        self.AddFieldWatches()


    ############################################################################
    '''
    Has DcgmReader use but not own a handle. Currently for the unit tests.
    '''
    def SetHandle(self, handle):
        self.m_dcgmHandle = pydcgm.DcgmHandle(handle)
        self.InitializeFromHandle()

    ############################################################################
    '''
    Reconnect function checks if connection handle is present. If the handle is
    none, it creates the handle and gets the default dcgm group. It then maps gpuIds to
    BusID, set the meta data of the field ids and adds watches to the field Ids mentioned in the idToWatch list.
    '''
    def Reconnect(self):
        if self.m_dcgmHandle is not None:
            return

        self.LogDebug("Connection handle is None. Trying to reconnect")

        self.m_dcgmHandle = pydcgm.DcgmHandle(None, self.m_dcgmHostName, dcgm_structs.DCGM_OPERATION_MODE_AUTO)
        self.m_closeHandle = True

        self.LogDebug("Connected to nv-hostengine")

        self.InitializeFromHandle()

    ###########################################################################
    '''
    Populate the g_gpuIdToBusId map. This map contains mapping from
    gpuID to the BusID.
    '''
    def SetupGpuIdBusMappings(self):
        self.m_gpuIdToBusId = {}

        gpuIds = self.m_dcgmGroup.GetGpuIds()
        for gpuId in gpuIds:
            gpuInfo = self.m_dcgmSystem.discovery.GetGpuAttributes(gpuId)
            self.m_gpuIdToBusId[gpuId] = gpuInfo.identifiers.pciBusId

    ###########################################################################
    '''
    Add watches to the fields which are passed in init function in idToWatch list.
    It also updates the field values for the first time.
    '''
    def AddFieldWatches(self):
        maxKeepSamples = 0 #No limit. Handled by m_maxKeepAge
        self.m_dcgmGroup.samples.WatchFields(self.m_fieldGroup, self.m_updateFreq, self.m_maxKeepAge, maxKeepSamples)
        self.m_dcgmSystem.UpdateAllFields(1)

    ###########################################################################
    '''
    If the groupID already exists, we delete that group and create a new fieldgroup with
    the fields mentioned in idToWatch. Then information of each field is acquired from its id.
    '''
    def GetFieldMetadata(self):
        self.m_fieldIdToInfo = {}

        findByNameId = self.m_dcgmSystem.GetFieldGroupIdByName(self.m_fieldGroupName)

        #Remove our field group if it exists already
        if findByNameId is not None:
            delFieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle=self.m_dcgmHandle, fieldGroupId=findByNameId)
            delFieldGroup.Delete()
            del(delFieldGroup)

        self.m_fieldGroup = pydcgm.DcgmFieldGroup(self.m_dcgmHandle, self.m_fieldGroupName, self.m_publishFieldIds)

        for fieldId in self.m_fieldGroup.fieldIds:
            self.m_fieldIdToInfo[fieldId] = self.m_dcgmSystem.fields.GetFieldById(fieldId)
            if self.m_fieldIdToInfo[fieldId] == 0 or self.m_fieldIdToInfo[fieldId] == None:
                self.LogError("Cannot get field tag for field id %d. Please check dcgm_fields to see if it is valid." % (fieldId))
                raise dcgm_structs.DCGMError(dcgm_structs.DCGM_ST_UNKNOWN_FIELD)

    ###########################################################################
    '''
    This function attempts to connect to dcgm and calls the implemented CustomDataHandler in the child class with field values.
    @params:
    self.m_dcgmGroup.samples.GetLatest(self.m_fieldGroup).values : The field values for each field. This dictionary contains fieldInfo
                                                                   for each field id requested to be watches.
    '''
    def Process(self):
        with self.m_lock:
            try:
                self.Reconnect()
                if not self.m_requestedEntities:
                    return self.CustomDataHandler(self.m_dcgmGroup.samples.GetLatest(self.m_fieldGroup).values)
                else:
                    return self.CustomDataHandler_v2(self.m_dcgmGroup.samples.GetLatest_v2(self.m_fieldGroup).values)
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID):
                self.LogError("Can't connect to nv-hostengine. Is it down?")
                self.SetDisconnected()

    ###########################################################################
    def LogInfo(self, msg):
        logging.info(msg)

    ###########################################################################
    def LogDebug(self, msg):
        logging.debug(msg)

    ###########################################################################
    def LogError(self, msg):
        logging.error(msg)


    ###########################################################################
    '''
    This function gets each value as a dictionary of dictionaries. The dictionary
    returned is each gpu id mapped to a dictionary of it's field values. Each
    field value dictionary is the field name mapped to the value or the field
    id mapped to value depending on the parameter mapById.
    '''
    def GetLatestGpuValuesAsDict(self, mapById):
        systemDictionary = {}

        with self.m_lock:
            try:
                self.Reconnect()
                fvs = self.m_dcgmGroup.samples.GetLatest(self.m_fieldGroup).values
                for gpuId in fvs.keys():
                    systemDictionary[gpuId] = {} # initialize the gpu's dictionary
                    gpuFv = fvs[gpuId]

                    for fieldId in gpuFv.keys():
                        val = gpuFv[fieldId][-1]

                        if val.isBlank:
                            continue

                        if mapById == False:
                            fieldTag = self.m_fieldIdToInfo[fieldId].tag
                            systemDictionary[gpuId][fieldTag] = val.value
                        else:
                            systemDictionary[gpuId][fieldId] = val.value
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID):
                self.LogError("Can't connection to nv-hostengine. Please verify that it is running.")
                self.SetDisconnected()

        return systemDictionary

    ###########################################################################
    def GetLatestGpuValuesAsFieldIdDict(self):
        return self.GetLatestGpuValuesAsDict(True)

    ###########################################################################
    def GetLatestGpuValuesAsFieldNameDict(self):
        return self.GetLatestGpuValuesAsDict(False)
