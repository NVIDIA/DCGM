/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*****************************************************************************
 * APIs for Internal testing
 *****************************************************************************/
DCGM_ENTRY_POINT(dcgmInjectFieldValue,
                 tsapiEngineInjectFieldValue,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmInjectFieldValue_t *pDcgmInjectFieldValue),
                 "({} {}, {})",
                 pDcgmHandle,
                 gpuId,
                 pDcgmInjectFieldValue)

DCGM_ENTRY_POINT(dcgmGetCacheManagerFieldInfo,
                 tsapiEngineGetCacheManagerFieldInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmCacheManagerFieldInfo_v4_t *fieldInfo),
                 "({} {})",
                 pDcgmHandle,
                 fieldInfo)

DCGM_ENTRY_POINT(dcgmGetGpuStatus,
                 tsapiGetGpuStatus,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, DcgmEntityStatus_t *status),
                 "({} {} {})",
                 pDcgmHandle,
                 gpuId,
                 status)


DCGM_ENTRY_POINT(dcgmCreateFakeEntities,
                 tsapiCreateFakeEntities,
                 (dcgmHandle_t pDcgmHandle, dcgmCreateFakeEntities_v2 *createFakeEntities),
                 "({} {})",
                 pDcgmHandle,
                 createFakeEntities)

DCGM_ENTRY_POINT(dcgmInjectEntityFieldValue,
                 tsapiInjectEntityFieldValue,
                 (dcgmHandle_t pDcgmHandle,
                  dcgm_field_entity_group_t entityGroupId,
                  dcgm_field_eid_t entityId,
                  dcgmInjectFieldValue_t *pDcgmInjectFieldValue),
                 "({} {} {} {})",
                 pDcgmHandle,
                 entityGroupId,
                 entityId,
                 pDcgmInjectFieldValue)

DCGM_ENTRY_POINT(dcgmSetEntityNvLinkLinkState,
                 tsapiSetEntityNvLinkLinkState,
                 (dcgmHandle_t pDcgmHandle, dcgmSetNvLinkLinkState_v1 *linkState),
                 "({} {})",
                 pDcgmHandle,
                 linkState)

/*****************************************************************************
 * Consolidated Standalone and Embedded APIs
 *****************************************************************************/

DCGM_ENTRY_POINT(dcgmEngineRun,
                 tsapiEngineRun,
                 (unsigned short portNumber, char const *socketPath, unsigned int isConnectionTCP),
                 "({} {} {})",
                 portNumber,
                 socketPath,
                 isConnectionTCP)

DCGM_ENTRY_POINT(dcgmGetAllDevices,
                 tsapiEngineGetAllDevices,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES], int *count),
                 "({} {} {})",
                 pDcgmHandle,
                 gpuIdList,
                 count)

DCGM_ENTRY_POINT(dcgmGetAllSupportedDevices,
                 tsapiEngineGetAllSupportedDevices,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES], int *count),
                 "({} {} {})",
                 pDcgmHandle,
                 gpuIdList,
                 count)

DCGM_ENTRY_POINT(dcgmGetDeviceAttributes,
                 tsapiEngineGetDeviceAttributes,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmDeviceAttributes_t *pDcgmDeviceAttr),
                 "({} {} {})",
                 pDcgmHandle,
                 gpuId,
                 pDcgmDeviceAttr)

DCGM_ENTRY_POINT(dcgmGetEntityGroupEntities,
                 tsapiGetEntityGroupEntities,
                 (dcgmHandle_t dcgmHandle,
                  dcgm_field_entity_group_t entityGroup,
                  dcgm_field_eid_t *entities,
                  int *numEntities,
                  unsigned int flags),
                 "({} {} {}, {}, x{:X})",
                 dcgmHandle,
                 entityGroup,
                 entities,
                 numEntities,
                 flags)

DCGM_ENTRY_POINT(dcgmGetGpuChipArchitecture,
                 tsapiGetGpuChipArchitecture,
                 (dcgmHandle_t dcgmHandle, unsigned int gpuId, dcgmChipArchitecture_t *chipArchitecture),
                 "({} {} {})",
                 dcgmHandle,
                 gpuId,
                 chipArchitecture)

DCGM_ENTRY_POINT(dcgmGetGpuInstanceHierarchy,
                 tsapiGetGpuInstanceHierarchy,
                 (dcgmHandle_t dcgmHandle, dcgmMigHierarchy_v2 *hierarchy),
                 "({} {})",
                 dcgmHandle,
                 hierarchy)

DCGM_ENTRY_POINT(dcgmCreateMigEntity,
                 tsapiCreateMigEntity,
                 (dcgmHandle_t dcgmHandle, dcgmCreateMigEntity_t *cme),
                 "({} {})",
                 dcgmHandle,
                 cme)

DCGM_ENTRY_POINT(dcgmDeleteMigEntity,
                 tsapiDeleteMigEntity,
                 (dcgmHandle_t dcgmHandle, dcgmDeleteMigEntity_t *dme),
                 "({} {})",
                 dcgmHandle,
                 dme)

DCGM_ENTRY_POINT(dcgmGetNvLinkLinkStatus,
                 tsapiGetNvLinkLinkStatus,
                 (dcgmHandle_t dcgmHandle, dcgmNvLinkStatus_v4 *linkStatus),
                 "({} {})",
                 dcgmHandle,
                 linkStatus)

DCGM_ENTRY_POINT(dcgmGetNvLinkP2PStatus,
                 tsapiGetNvLinkP2PStatus,
                 (dcgmHandle_t dcgmHandle, dcgmNvLinkP2PStatus_v1 *linkStatus),
                 "({} {})",
                 dcgmHandle,
                 linkStatus)

DCGM_ENTRY_POINT(dcgmGetCpuHierarchy,
                 tsapiGetCpuHierarchy,
                 (dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v1 *cpuHierarchy),
                 "({} {})",
                 dcgmHandle,
                 cpuHierarchy)

DCGM_ENTRY_POINT(dcgmGetCpuHierarchy_v2,
                 tsapiGetCpuHierarchy_v2,
                 (dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v2 *cpuHierarchy),
                 "({} {})",
                 dcgmHandle,
                 cpuHierarchy)

DCGM_ENTRY_POINT(dcgmGetVgpuDeviceAttributes,
                 tsapiEngineGetVgpuDeviceAttributes,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmVgpuDeviceAttributes_t *pDcgmVgpuDeviceAttr),
                 "({} {} {})",
                 pDcgmHandle,
                 gpuId,
                 pDcgmVgpuDeviceAttr)

DCGM_ENTRY_POINT(dcgmGetVgpuInstanceAttributes,
                 tsapiEngineGetVgpuInstanceAttributes,
                 (dcgmHandle_t pDcgmHandle, unsigned int vgpuId, dcgmVgpuInstanceAttributes_t *pDcgmVgpuInstanceAttr),
                 "({} {} {})",
                 pDcgmHandle,
                 vgpuId,
                 pDcgmVgpuInstanceAttr)

DCGM_ENTRY_POINT(dcgmGroupCreate,
                 tsapiEngineGroupCreate,
                 (dcgmHandle_t pDcgmHandle, dcgmGroupType_t type, const char *groupName, dcgmGpuGrp_t *pDcgmGrpId),
                 "({} {} {} {})",
                 pDcgmHandle,
                 type,
                 groupName,
                 pDcgmGrpId)

DCGM_ENTRY_POINT(dcgmGroupDestroy,
                 tsapiEngineGroupDestroy,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId),
                 "({} {})",
                 pDcgmHandle,
                 groupId)

DCGM_ENTRY_POINT(dcgmGroupAddDevice,
                 tsapiEngineGroupAddDevice,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId),
                 "({} {} {})",
                 pDcgmHandle,
                 groupId,
                 gpuId)

DCGM_ENTRY_POINT(dcgmGroupAddEntity,
                 tsapiGroupAddEntity,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgm_field_entity_group_t entityGroupId,
                  dcgm_field_eid_t entityId),
                 "({} {} {}, {})",
                 pDcgmHandle,
                 groupId,
                 entityGroupId,
                 entityId)

DCGM_ENTRY_POINT(dcgmGroupRemoveDevice,
                 tsapiEngineGroupRemoveDevice,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId),
                 "({} {} {})",
                 pDcgmHandle,
                 groupId,
                 gpuId)

DCGM_ENTRY_POINT(dcgmGroupRemoveEntity,
                 tsapiGroupRemoveEntity,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgm_field_entity_group_t entityGroupId,
                  dcgm_field_eid_t entityId),
                 "({} {} {}, {})",
                 pDcgmHandle,
                 groupId,
                 entityGroupId,
                 entityId)

DCGM_ENTRY_POINT(dcgmGroupGetInfo,
                 tsapiEngineGroupGetInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupInfo_t *pDcgmGroupInfo),
                 "({} {} {})",
                 pDcgmHandle,
                 groupId,
                 pDcgmGroupInfo)

DCGM_ENTRY_POINT(dcgmGroupGetAllIds,
                 tsapiGroupGetAllIds,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupIdList[], unsigned int *count),
                 "({} {} {})",
                 pDcgmHandle,
                 groupIdList,
                 count)

DCGM_ENTRY_POINT(dcgmStatusCreate, tsapiStatusCreate, (dcgmStatus_t * statusHandle), "({})", statusHandle)

DCGM_ENTRY_POINT(dcgmStatusDestroy, tsapiStatusDestroy, (dcgmStatus_t statusHandle), "({})", statusHandle)

DCGM_ENTRY_POINT(dcgmStatusGetCount,
                 tsapiStatusGetCount,
                 (dcgmStatus_t statusHandle, unsigned int *count),
                 "({} {})",
                 statusHandle,
                 count)

DCGM_ENTRY_POINT(dcgmStatusPopError,
                 tsapiStatusPopError,
                 (dcgmStatus_t statusHandle, dcgmErrorInfo_t *pDcgmErrorInfo),
                 "({} {})",
                 statusHandle,
                 pDcgmErrorInfo)

DCGM_ENTRY_POINT(dcgmStatusClear, tsapiStatusClear, (dcgmStatus_t statusHandle), "({})", statusHandle)

DCGM_ENTRY_POINT(
    dcgmConfigSet,
    tsapiEngineConfigSet,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmConfig_t *pDeviceConfig, dcgmStatus_t statusHandle),
    "({} {} {} {})",
    pDcgmHandle,
    groupId,
    pDeviceConfig,
    statusHandle)

DCGM_ENTRY_POINT(
    dcgmVgpuConfigSet,
    tsapiEngineVgpuConfigSet,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmVgpuConfig_t *pDeviceConfig, dcgmStatus_t statusHandle),
    "({} {} {} {})",
    pDcgmHandle,
    groupId,
    pDeviceConfig,
    statusHandle)

DCGM_ENTRY_POINT(dcgmConfigGet,
                 tsapiEngineConfigGet,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmConfigType_t type,
                  int count,
                  dcgmConfig_t deviceConfigList[],
                  dcgmStatus_t statusHandle),
                 "({} {} {} {} {} {})",
                 pDcgmHandle,
                 groupId,
                 type,
                 count,
                 deviceConfigList,
                 statusHandle)

DCGM_ENTRY_POINT(dcgmVgpuConfigGet,
                 tsapiEngineVgpuConfigGet,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmConfigType_t type,
                  int count,
                  dcgmVgpuConfig_t deviceConfigList[],
                  dcgmStatus_t statusHandle),
                 "({} {} {} {} {} {})",
                 pDcgmHandle,
                 groupId,
                 type,
                 count,
                 deviceConfigList,
                 statusHandle)

DCGM_ENTRY_POINT(dcgmConfigEnforce,
                 tsapiEngineConfigEnforce,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle),
                 "({} {} {})",
                 pDcgmHandle,
                 groupId,
                 statusHandle)

DCGM_ENTRY_POINT(dcgmVgpuConfigEnforce,
                 tsapiEngineVgpuConfigEnforce,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle),
                 "({} {} {})",
                 pDcgmHandle,
                 groupId,
                 statusHandle)

DCGM_ENTRY_POINT(
    dcgmGetLatestValuesForFields,
    tsapiEngineGetLatestValuesForFields,
    (dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldIds[], unsigned int count, dcgmFieldValue_v1 values[]),
    "({} {} {} {} {})",
    pDcgmHandle,
    gpuId,
    fieldIds,
    count,
    values)

DCGM_ENTRY_POINT(dcgmEntityGetLatestValues,
                 tsapiEngineEntityGetLatestValues,
                 (dcgmHandle_t pDcgmHandle,
                  dcgm_field_entity_group_t entityGroup,
                  int entityId,
                  unsigned short fieldIds[],
                  unsigned int count,
                  dcgmFieldValue_v1 values[]),
                 "({} {} {} {} {} {})",
                 pDcgmHandle,
                 entityGroup,
                 entityId,
                 fieldIds,
                 count,
                 values)

DCGM_ENTRY_POINT(dcgmGetMultipleValuesForField,
                 tsapiEngineGetMultipleValuesForField,
                 (dcgmHandle_t pDcgmHandle,
                  int gpuId,
                  unsigned short fieldId,
                  int *count,
                  long long startTs,
                  long long endTs,
                  dcgmOrder_t order,
                  dcgmFieldValue_v1 values[]),
                 "({} {} {} {} {} {} {} {})",
                 pDcgmHandle,
                 gpuId,
                 fieldId,
                 count,
                 startTs,
                 endTs,
                 order,
                 values)

DCGM_ENTRY_POINT(dcgmWatchFieldValue,
                 tsapiEngineWatchFieldValue,
                 (dcgmHandle_t pDcgmHandle,
                  int gpuId,
                  unsigned short fieldId,
                  long long updateFreq,
                  double maxKeepAge,
                  int maxKeepSamples),
                 "({} {} {} {} {} {})",
                 pDcgmHandle,
                 gpuId,
                 fieldId,
                 updateFreq,
                 maxKeepAge,
                 maxKeepSamples)

DCGM_ENTRY_POINT(dcgmUnwatchFieldValue,
                 tsapiEngineUnwatchFieldValue,
                 (dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldId, int clearCache),
                 "({} {} {} {})",
                 pDcgmHandle,
                 gpuId,
                 fieldId,
                 clearCache)

DCGM_ENTRY_POINT(dcgmUpdateAllFields,
                 tsapiEngineUpdateAllFields,
                 (dcgmHandle_t pDcgmHandle, int waitForUpdate),
                 "({} {})",
                 pDcgmHandle,
                 waitForUpdate)

DCGM_ENTRY_POINT(dcgmPolicySet,
                 tsapiEnginePolicySet,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicy_t *policy, dcgmStatus_t statusHandle),
                 "({} {}, {}, {})",
                 pDcgmHandle,
                 groupId,
                 policy,
                 statusHandle)

DCGM_ENTRY_POINT(
    dcgmPolicyGet,
    tsapiEnginePolicyGet,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, int count, dcgmPolicy_t policy[], dcgmStatus_t statusHandle),
    "({} {}, {}, {}, {})",
    pDcgmHandle,
    groupId,
    count,
    policy,
    statusHandle)

DCGM_ENTRY_POINT(dcgmPolicyTrigger, tsapiEnginePolicyTrigger, (dcgmHandle_t pDcgmHandle), "{}", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmPolicyRegister_v2,
                 tsapiEnginePolicyRegister,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmPolicyCondition_t condition,
                  fpRecvUpdates callback,
                  uint64_t userData),
                 "({} {}, {}, {}, {})",
                 pDcgmHandle,
                 groupId,
                 condition,
                 callback,
                 userData)

DCGM_ENTRY_POINT(dcgmPolicyUnregister,
                 tsapiEnginePolicyUnregister,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition),
                 "({} {}, {})",
                 pDcgmHandle,
                 groupId,
                 condition)

DCGM_ENTRY_POINT(dcgmGetFieldValuesSince,
                 tsapiEngineGetFieldValuesSince,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  long long sinceTimestamp,
                  unsigned short *fieldIds,
                  int numFieldIds,
                  long long *nextSinceTimestamp,
                  dcgmFieldValueEnumeration_f enumCB,
                  void *userData),
                 "({} {} {} {} {} {} {} {})",
                 pDcgmHandle,
                 groupId,
                 sinceTimestamp,
                 fieldIds,
                 numFieldIds,
                 nextSinceTimestamp,
                 enumCB,
                 userData)

DCGM_ENTRY_POINT(dcgmGetValuesSince,
                 tsapiEngineGetValuesSince,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmFieldGrp_t fieldGroupId,
                  long long sinceTimestamp,
                  long long *nextSinceTimestamp,
                  dcgmFieldValueEnumeration_f enumCB,
                  void *userData),
                 "({} {} {} {} {} {} {})",
                 pDcgmHandle,
                 groupId,
                 fieldGroupId,
                 sinceTimestamp,
                 nextSinceTimestamp,
                 enumCB,
                 userData)

DCGM_ENTRY_POINT(dcgmGetValuesSince_v2,
                 tsapiEngineGetValuesSince_v2,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmFieldGrp_t fieldGroupId,
                  long long sinceTimestamp,
                  long long *nextSinceTimestamp,
                  dcgmFieldValueEntityEnumeration_f enumCB,
                  void *userData),
                 "({} {} {} {} {} {} {})",
                 pDcgmHandle,
                 groupId,
                 fieldGroupId,
                 sinceTimestamp,
                 nextSinceTimestamp,
                 enumCB,
                 userData)

DCGM_ENTRY_POINT(dcgmGetLatestValues,
                 tsapiEngineGetLatestValues,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmFieldGrp_t fieldGroupId,
                  dcgmFieldValueEnumeration_f enumCB,
                  void *userData),
                 "({} {} {} {} {})",
                 pDcgmHandle,
                 groupId,
                 fieldGroupId,
                 enumCB,
                 userData)

DCGM_ENTRY_POINT(dcgmGetLatestValues_v2,
                 tsapiEngineGetLatestValues_v2,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmFieldGrp_t fieldGroupId,
                  dcgmFieldValueEntityEnumeration_f enumCB,
                  void *userData),
                 "({} {} {} {} {})",
                 pDcgmHandle,
                 groupId,
                 fieldGroupId,
                 enumCB,
                 userData)

DCGM_ENTRY_POINT(dcgmEntitiesGetLatestValues,
                 tsapiEntitiesGetLatestValues,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGroupEntityPair_t entities[],
                  unsigned int entityCount,
                  unsigned short fields[],
                  unsigned int fieldCount,
                  unsigned int flags,
                  dcgmFieldValue_v2 values[]),
                 "({} {} {} {} {} {} {})",
                 pDcgmHandle,
                 entities,
                 entityCount,
                 fields,
                 fieldCount,
                 flags,
                 values)

DCGM_ENTRY_POINT(dcgmWatchFields,
                 tsapiWatchFields,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmFieldGrp_t fieldGroupId,
                  long long updateFreq,
                  double maxKeepAge,
                  int maxKeepSamples),
                 "({} {}, {}, {}, {}, {})",
                 pDcgmHandle,
                 groupId,
                 fieldGroupId,
                 updateFreq,
                 maxKeepAge,
                 maxKeepSamples)

DCGM_ENTRY_POINT(dcgmUnwatchFields,
                 tsapiUnwatchFields,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId),
                 "({} {}, {})",
                 pDcgmHandle,
                 groupId,
                 fieldGroupId)

DCGM_ENTRY_POINT(dcgmFieldGroupCreate,
                 tsapiFieldGroupCreate,
                 (dcgmHandle_t pDcgmHandle,
                  int numFieldIds,
                  const unsigned short *fieldIds,
                  const char *fieldGroupName,
                  dcgmFieldGrp_t *dcgmFieldGroupId),
                 "({} {}, {}, {}, {})",
                 pDcgmHandle,
                 numFieldIds,
                 fieldIds,
                 fieldGroupName,
                 dcgmFieldGroupId)

DCGM_ENTRY_POINT(dcgmFieldGroupDestroy,
                 tsapiFieldGroupDestroy,
                 (dcgmHandle_t pDcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId),
                 "({} {})",
                 pDcgmHandle,
                 dcgmFieldGroupId)

DCGM_ENTRY_POINT(dcgmFieldGroupGetInfo,
                 tsapiFieldGroupGetInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmFieldGroupInfo_t *fieldGroupInfo),
                 "({} {})",
                 pDcgmHandle,
                 fieldGroupInfo)

DCGM_ENTRY_POINT(dcgmFieldGroupGetAll,
                 tsapiFieldGroupGetAll,
                 (dcgmHandle_t pDcgmHandle, dcgmAllFieldGroup_t *allGroupInfo),
                 "({} {})",
                 pDcgmHandle,
                 allGroupInfo)

DCGM_ENTRY_POINT(dcgmHealthSet,
                 tsapiEngineHealthSet,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t systems),
                 "({} {}, {})",
                 pDcgmHandle,
                 groupId,
                 systems)

DCGM_ENTRY_POINT(dcgmHealthSet_v2,
                 tsapiEngineHealthSet_v2,
                 (dcgmHandle_t pDcgmHandle, dcgmHealthSetParams_v2 *params),
                 "({} {})",
                 pDcgmHandle,
                 params)

DCGM_ENTRY_POINT(dcgmHealthGet,
                 tsapiEngineHealthGet,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems),
                 "({} {}, {})",
                 pDcgmHandle,
                 groupId,
                 systems)

DCGM_ENTRY_POINT(dcgmHealthCheck,
                 tsapiEngineHealthCheck,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_t *results),
                 "({} {}, {})",
                 pDcgmHandle,
                 groupId,
                 results)

DCGM_ENTRY_POINT(dcgmActionValidate_v2,
                 tsapiEngineActionValidate_v2,
                 (dcgmHandle_t pDcgmHandle, dcgmRunDiag_v10 *drd, dcgmDiagResponse_v12 *response),
                 "({}, {}, {})",
                 pDcgmHandle,
                 drd,
                 response)

DCGM_ENTRY_POINT(
    dcgmActionValidate,
    tsapiEngineActionValidate,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyValidation_t validate, dcgmDiagResponse_v12 *response),
    "({} {}, {}, {})",
    pDcgmHandle,
    groupId,
    validate,
    response)

DCGM_ENTRY_POINT(dcgmRunDiagnostic,
                 tsapiEngineRunDiagnostic,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmDiagnosticLevel_t diagLevel,
                  dcgmDiagResponse_v12 *diagResponse),
                 "({} {}, {}, {})",
                 pDcgmHandle,
                 groupId,
                 diagLevel,
                 diagResponse)

DCGM_ENTRY_POINT(dcgmStopDiagnostic, tsapiEngineStopDiagnostic, (dcgmHandle_t pDcgmHandle), "({})", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmRunMnDiagnostic,
                 tsapiEngineRunMnDiagnostic,
                 (dcgmHandle_t pDcgmHandle, dcgmRunMnDiag_v1 const *drmnd, dcgmMnDiagResponse_v1 *response),
                 "({}, {}, {})",
                 pDcgmHandle,
                 drmnd,
                 response)

DCGM_ENTRY_POINT(dcgmStopMnDiagnostic, tsapiEngineStopMnDiagnostic, (dcgmHandle_t pDcgmHandle), "({})", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmMultinodeRequest,
                 tsapiEngineMultinodeRequest,
                 (dcgmHandle_t pDcgmHandle, dcgmMultinodeRequest_t *request),
                 "({} {})",
                 pDcgmHandle,
                 request)

DCGM_ENTRY_POINT(
    dcgmWatchPidFields,
    tsapiEngineWatchPidFields,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, long long updateFreq, double maxKeepAge, int maxKeepSamples),
    "({} {}, {}, {}, {})",
    pDcgmHandle,
    groupId,
    updateFreq,
    maxKeepAge,
    maxKeepSamples)

DCGM_ENTRY_POINT(dcgmGetPidInfo,
                 tsapiEngineGetPidInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo),
                 "({} {} {})",
                 pDcgmHandle,
                 groupId,
                 pidInfo)

DCGM_ENTRY_POINT(dcgmGetDeviceWorkloadPowerProfileInfo,
                 tsapiEngineGetWorkloadPowerProfileInfo,
                 (dcgmHandle_t pDcgmHandle,
                  unsigned int gpuId,
                  dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo,
                  dcgmDeviceWorkloadPowerProfilesStatus_v1 *profilesStatus),
                 "({} {} {} {})",
                 pDcgmHandle,
                 gpuId,
                 profilesInfo,
                 profilesStatus)

DCGM_ENTRY_POINT(dcgmGetDeviceTopology,
                 tsapiEngineGetDeviceTopology,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmDeviceTopology_t *deviceTopology),
                 "({} {} {})",
                 pDcgmHandle,
                 gpuId,
                 deviceTopology)

DCGM_ENTRY_POINT(dcgmGetGroupTopology,
                 tsapiEngineGroupTopology,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupTopology_t *groupTopology),
                 "({} {} {})",
                 pDcgmHandle,
                 groupId,
                 groupTopology)

DCGM_ENTRY_POINT(
    dcgmWatchJobFields,
    tsapiEngineWatchJobFields,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, long long updateFreq, double maxKeepAge, int maxKeepSamples),
    "({} {}, {}, {}, {})",
    pDcgmHandle,
    groupId,
    updateFreq,
    maxKeepAge,
    maxKeepSamples)


DCGM_ENTRY_POINT(dcgmJobStartStats,
                 tsapiEngineJobStartStats,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64]),
                 "({} {} {})",
                 pDcgmHandle,
                 groupId,
                 jobId)

DCGM_ENTRY_POINT(dcgmJobStopStats,
                 tsapiEngineJobStopStats,
                 (dcgmHandle_t pDcgmHandle, char jobId[64]),
                 "({} {})",
                 pDcgmHandle,
                 jobId)

DCGM_ENTRY_POINT(dcgmJobGetStats,
                 tsapiEngineJobGetStats,
                 (dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo),
                 "({} {} {})",
                 pDcgmHandle,
                 jobId,
                 pJobInfo)

DCGM_ENTRY_POINT(dcgmJobRemove,
                 tsapiEngineJobRemove,
                 (dcgmHandle_t pDcgmHandle, char jobId[64]),
                 "({} {})",
                 pDcgmHandle,
                 jobId)

DCGM_ENTRY_POINT(dcgmJobRemoveAll, tsapiEngineJobRemoveAll, (dcgmHandle_t pDcgmHandle), "({})", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmIntrospectGetHostengineCpuUtilization,
                 tsapiIntrospectGetHostengineCpuUtilization,
                 (dcgmHandle_t pDcgmHandle, dcgmIntrospectCpuUtil_t *cpuUtil, int waitIfNoData),
                 "({} {} {})",
                 pDcgmHandle,
                 cpuUtil,
                 waitIfNoData)

DCGM_ENTRY_POINT(dcgmIntrospectGetHostengineMemoryUsage,
                 tsapiIntrospectGetHostengineMemoryUsage,
                 (dcgmHandle_t pDcgmHandle, dcgmIntrospectMemory_t *memoryInfo, int waitIfNoData),
                 "({} {} {})",
                 pDcgmHandle,
                 memoryInfo,
                 waitIfNoData)

DCGM_ENTRY_POINT(
    dcgmSelectGpusByTopology,
    tsapiSelectGpusByTopology,
    (dcgmHandle_t pDcgmHandle, uint64_t inputGpuIds, uint32_t numGpus, uint64_t *outputGpuIds, uint64_t hintFlags),
    "({}, {}, {}, {}, {})",
    pDcgmHandle,
    inputGpuIds,
    numGpus,
    outputGpuIds,
    hintFlags)

DCGM_ENTRY_POINT(dcgmGetFieldSummary,
                 tsapiGetFieldSummary,
                 (dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request),
                 "({}, {})",
                 pDcgmHandle,
                 request)

DCGM_ENTRY_POINT(dcgmModuleDenylist,
                 tsapiModuleDenylist,
                 (dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId),
                 "({}, {})",
                 pDcgmHandle,
                 moduleId)

DCGM_ENTRY_POINT(dcgmModuleGetStatuses,
                 tsapiModuleGetStatuses,
                 (dcgmHandle_t pDcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses),
                 "({}, {})",
                 pDcgmHandle,
                 moduleStatuses)

DCGM_ENTRY_POINT(dcgmProfGetSupportedMetricGroups,
                 tsapiProfGetSupportedMetricGroups,
                 (dcgmHandle_t pDcgmHandle, dcgmProfGetMetricGroups_t *metricGroups),
                 "({}, {})",
                 pDcgmHandle,
                 metricGroups)

DCGM_ENTRY_POINT(dcgmProfPause, tsapiProfPause, (dcgmHandle_t pDcgmHandle), "({})", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmProfResume, tsapiProfResume, (dcgmHandle_t pDcgmHandle), "({})", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmHostengineSetLoggingSeverity,
                 tsapiHostengineSetLoggingSeverity,
                 (dcgmHandle_t pDcgmHandle, dcgmSettingsSetLoggingSeverity_t *logging),
                 "({}, {})",
                 pDcgmHandle,
                 logging)

DCGM_ENTRY_POINT(dcgmVersionInfo, tsapiVersionInfo, (dcgmVersionInfo_t * pVersionInfo), "({})", pVersionInfo)

DCGM_ENTRY_POINT(dcgmHostengineVersionInfo,
                 tsapiHostengineVersionInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmVersionInfo_t *pVersionInfo),
                 "({}, {})",
                 pDcgmHandle,
                 pVersionInfo)

DCGM_ENTRY_POINT(dcgmHostengineIsHealthy,
                 tsapiHostengineIsHealthy,
                 (dcgmHandle_t dcgmHandle, dcgmHostengineHealth_t *heHealth),
                 "({}, {})",
                 dcgmHandle,
                 heHealth)

DCGM_ENTRY_POINT(dcgmModuleIdToName,
                 tsapiDcgmModuleIdToName,
                 (dcgmModuleId_t id, char const **name),
                 "({}, {})",
                 id,
                 name)

DCGM_ENTRY_POINT(dcgmInjectEntityFieldValueToNvml,
                 tsapiInjectEntityFieldValueToNvml,
                 (dcgmHandle_t pDcgmHandle,
                  dcgm_field_entity_group_t entityGroupId,
                  dcgm_field_eid_t entityId,
                  dcgmInjectFieldValue_t *pDcgmInjectFieldValue),
                 "({} {} {} {})",
                 pDcgmHandle,
                 entityGroupId,
                 entityId,
                 pDcgmInjectFieldValue)

DCGM_ENTRY_POINT(dcgmCreateNvmlInjectionGpu,
                 tsapiCreateNvmlInjectionGpu,
                 (dcgmHandle_t dcgmHandle, unsigned int index),
                 "({} {})",
                 dcgmHandle,
                 index)

#ifdef INJECTION_LIBRARY_AVAILABLE
DCGM_ENTRY_POINT(dcgmInjectNvmlDevice,
                 tsapiInjectNvmlDevice,
                 (dcgmHandle_t dcgmHandle,
                  unsigned int gpuId,
                  const char *key,
                  const injectNvmlVal_t *extraKeys,
                  unsigned int extraKeyCount,
                  const injectNvmlRet_t *injectNvmlRet),
                 "({}, {}, {}, {}, {}, {})",
                 dcgmHandle,
                 gpuId,
                 key,
                 extraKeys,
                 extraKeyCount,
                 injectNvmlRet)

DCGM_ENTRY_POINT(dcgmInjectNvmlDeviceForFollowingCalls,
                 tsapiInjectNvmlDeviceForFollowingCalls,
                 (dcgmHandle_t dcgmHandle,
                  unsigned int gpuId,
                  const char *key,
                  const injectNvmlVal_t *extraKeys,
                  unsigned int extraKeyCount,
                  const injectNvmlRet_t *injectNvmlRets,
                  unsigned int retCount),
                 "({}, {}, {}, {}, {}, {}, {})",
                 dcgmHandle,
                 gpuId,
                 key,
                 extraKeys,
                 extraKeyCount,
                 injectNvmlRets,
                 retCount)

DCGM_ENTRY_POINT(dcgmInjectedNvmlDeviceReset,
                 tsapiInjectedNvmlDeviceReset,
                 (dcgmHandle_t dcgmHandle, unsigned int gpuId),
                 "({}, {})",
                 dcgmHandle,
                 gpuId)

DCGM_ENTRY_POINT(dcgmGetNvmlInjectFuncCallCount,
                 tsapiGetNvmlInjectFuncCallCount,
                 (dcgmHandle_t dcgmHandle, injectNvmlFuncCallCounts_t *nvmlFuncCallCounts),
                 "({}, {})",
                 dcgmHandle,
                 nvmlFuncCallCounts)

DCGM_ENTRY_POINT(dcgmResetNvmlInjectFuncCallCount,
                 tsapiResetNvmlInjectFuncCallCount,
                 (dcgmHandle_t dcgmHandle),
                 "({})",
                 dcgmHandle)

DCGM_ENTRY_POINT(dcgmRemoveNvmlInjectedGpu,
                 tsapiRemoveNvmlInjectedGpu,
                 (dcgmHandle_t dcgmHandle, char const *uuid),
                 "({}, {})",
                 dcgmHandle,
                 uuid)

DCGM_ENTRY_POINT(dcgmRestoreNvmlInjectedGpu,
                 tsapiRestoreNvmlInjectedGpu,
                 (dcgmHandle_t dcgmHandle, char const *uuid),
                 "({}, {})",
                 dcgmHandle,
                 uuid)
#endif

DCGM_ENTRY_POINT(dcgmPauseTelemetryForDiag, tsapiPause, (dcgmHandle_t pDcgmHandle), "({})", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmResumeTelemetryForDiag, tsapiResume, (dcgmHandle_t pDcgmHandle), "({})", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmNvswitchGetBackend,
                 tsapiNvswitchGetBackend,
                 (dcgmHandle_t pDcgmHandle, bool *active, char *backendName, unsigned int backendNameLength),
                 "({}, {}, {}, {})",
                 pDcgmHandle,
                 active,
                 backendName,
                 backendNameLength)

/*****************************************************************************/
