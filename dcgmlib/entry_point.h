/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
                 "(%p %d, %p)",
                 pDcgmHandle,
                 gpuId,
                 pDcgmInjectFieldValue)

DCGM_ENTRY_POINT(dcgmGetCacheManagerFieldInfo,
                 tsapiEngineGetCacheManagerFieldInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmCacheManagerFieldInfo_t *fieldInfo),
                 "(%p %p)",
                 pDcgmHandle,
                 fieldInfo)

DCGM_ENTRY_POINT(dcgmGetGpuStatus,
                 tsapiGetGpuStatus,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, DcgmEntityStatus_t *status),
                 "(%p %u %p)",
                 pDcgmHandle,
                 gpuId,
                 status)


DCGM_ENTRY_POINT(dcgmCreateFakeEntities,
                 tsapiCreateFakeEntities,
                 (dcgmHandle_t pDcgmHandle, dcgmCreateFakeEntities_v2 *createFakeEntities),
                 "(%p %p)",
                 pDcgmHandle,
                 createFakeEntities)

DCGM_ENTRY_POINT(dcgmInjectEntityFieldValue,
                 tsapiInjectEntityFieldValue,
                 (dcgmHandle_t pDcgmHandle,
                  dcgm_field_entity_group_t entityGroupId,
                  dcgm_field_eid_t entityId,
                  dcgmInjectFieldValue_t *pDcgmInjectFieldValue),
                 "(%p %u %u %p)",
                 pDcgmHandle,
                 entityGroupId,
                 entityId,
                 pDcgmInjectFieldValue)

DCGM_ENTRY_POINT(dcgmSetEntityNvLinkLinkState,
                 tsapiSetEntityNvLinkLinkState,
                 (dcgmHandle_t pDcgmHandle, dcgmSetNvLinkLinkState_v1 *linkState),
                 "(%p %p)",
                 pDcgmHandle,
                 linkState)

/*****************************************************************************
 * Consolidated Standalone and Embedded APIs
 *****************************************************************************/

DCGM_ENTRY_POINT(dcgmEngineRun,
                 tsapiEngineRun,
                 (unsigned short portNumber, char const *socketPath, unsigned int isConnectionTCP),
                 "(%d %p %d)",
                 portNumber,
                 socketPath,
                 isConnectionTCP)

DCGM_ENTRY_POINT(dcgmGetAllDevices,
                 tsapiEngineGetAllDevices,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES], int *count),
                 "(%p %p %p)",
                 pDcgmHandle,
                 gpuIdList,
                 count)

DCGM_ENTRY_POINT(dcgmGetAllSupportedDevices,
                 tsapiEngineGetAllSupportedDevices,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES], int *count),
                 "(%p %p %p)",
                 pDcgmHandle,
                 gpuIdList,
                 count)

DCGM_ENTRY_POINT(dcgmGetDeviceAttributes,
                 tsapiEngineGetDeviceAttributes,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmDeviceAttributes_t *pDcgmDeviceAttr),
                 "(%p %d %p)",
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
                 "(%p %u %p, %p, x%X)",
                 dcgmHandle,
                 entityGroup,
                 entities,
                 numEntities,
                 flags)

DCGM_ENTRY_POINT(dcgmGetGpuInstanceHierarchy,
                 tsapiGetGpuInstanceHierarchy,
                 (dcgmHandle_t dcgmHandle, dcgmMigHierarchy_v2 *hierarchy),
                 "(%p %p)",
                 dcgmHandle,
                 hierarchy)

DCGM_ENTRY_POINT(dcgmCreateMigEntity,
                 tsapiCreateMigEntity,
                 (dcgmHandle_t dcgmHandle, dcgmCreateMigEntity_t *cme),
                 "(%p %p)",
                 dcgmHandle,
                 cme)

DCGM_ENTRY_POINT(dcgmDeleteMigEntity,
                 tsapiDeleteMigEntity,
                 (dcgmHandle_t dcgmHandle, dcgmDeleteMigEntity_t *dme),
                 "(%p %p)",
                 dcgmHandle,
                 dme)

DCGM_ENTRY_POINT(dcgmGetNvLinkLinkStatus,
                 tsapiGetNvLinkLinkStatus,
                 (dcgmHandle_t dcgmHandle, dcgmNvLinkStatus_v2 *linkStatus),
                 "(%p %p)",
                 dcgmHandle,
                 linkStatus)

DCGM_ENTRY_POINT(dcgmGetVgpuDeviceAttributes,
                 tsapiEngineGetVgpuDeviceAttributes,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmVgpuDeviceAttributes_t *pDcgmVgpuDeviceAttr),
                 "(%p %d %p)",
                 pDcgmHandle,
                 gpuId,
                 pDcgmVgpuDeviceAttr)

DCGM_ENTRY_POINT(dcgmGetVgpuInstanceAttributes,
                 tsapiEngineGetVgpuInstanceAttributes,
                 (dcgmHandle_t pDcgmHandle, unsigned int vgpuId, dcgmVgpuInstanceAttributes_t *pDcgmVgpuInstanceAttr),
                 "(%p %d %p)",
                 pDcgmHandle,
                 vgpuId,
                 pDcgmVgpuInstanceAttr)

DCGM_ENTRY_POINT(dcgmGroupCreate,
                 tsapiEngineGroupCreate,
                 (dcgmHandle_t pDcgmHandle, dcgmGroupType_t type, char *groupName, dcgmGpuGrp_t *pDcgmGrpId),
                 "(%p %d %p %p)",
                 pDcgmHandle,
                 type,
                 groupName,
                 pDcgmGrpId)

DCGM_ENTRY_POINT(dcgmGroupDestroy,
                 tsapiEngineGroupDestroy,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId),
                 "(%p %p)",
                 pDcgmHandle,
                 groupId)

DCGM_ENTRY_POINT(dcgmGroupAddDevice,
                 tsapiEngineGroupAddDevice,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId),
                 "(%p %p %d)",
                 pDcgmHandle,
                 groupId,
                 gpuId)

DCGM_ENTRY_POINT(dcgmGroupAddEntity,
                 tsapiGroupAddEntity,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgm_field_entity_group_t entityGroupId,
                  dcgm_field_eid_t entityId),
                 "(%p %p %u, %u)",
                 pDcgmHandle,
                 groupId,
                 entityGroupId,
                 entityId)

DCGM_ENTRY_POINT(dcgmGroupRemoveDevice,
                 tsapiEngineGroupRemoveDevice,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId),
                 "(%p %p %d)",
                 pDcgmHandle,
                 groupId,
                 gpuId)

DCGM_ENTRY_POINT(dcgmGroupRemoveEntity,
                 tsapiGroupRemoveEntity,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgm_field_entity_group_t entityGroupId,
                  dcgm_field_eid_t entityId),
                 "(%p %p %u, %u)",
                 pDcgmHandle,
                 groupId,
                 entityGroupId,
                 entityId)

DCGM_ENTRY_POINT(dcgmGroupGetInfo,
                 tsapiEngineGroupGetInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupInfo_t *pDcgmGroupInfo),
                 "(%p %p %p)",
                 pDcgmHandle,
                 groupId,
                 pDcgmGroupInfo)

DCGM_ENTRY_POINT(dcgmGroupGetAllIds,
                 tsapiGroupGetAllIds,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupIdList[], unsigned int *count),
                 "(%p %p %p)",
                 pDcgmHandle,
                 groupIdList,
                 count)

DCGM_ENTRY_POINT(dcgmStatusCreate, tsapiStatusCreate, (dcgmStatus_t * statusHandle), "(%p)", statusHandle)

DCGM_ENTRY_POINT(dcgmStatusDestroy, tsapiStatusDestroy, (dcgmStatus_t statusHandle), "(%p)", statusHandle)

DCGM_ENTRY_POINT(dcgmStatusGetCount,
                 tsapiStatusGetCount,
                 (dcgmStatus_t statusHandle, unsigned int *count),
                 "(%p %p)",
                 statusHandle,
                 count)

DCGM_ENTRY_POINT(dcgmStatusPopError,
                 tsapiStatusPopError,
                 (dcgmStatus_t statusHandle, dcgmErrorInfo_t *pDcgmErrorInfo),
                 "(%p %p)",
                 statusHandle,
                 pDcgmErrorInfo)

DCGM_ENTRY_POINT(dcgmStatusClear, tsapiStatusClear, (dcgmStatus_t statusHandle), "(%p)", statusHandle)

DCGM_ENTRY_POINT(
    dcgmConfigSet,
    tsapiEngineConfigSet,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmConfig_t *pDeviceConfig, dcgmStatus_t statusHandle),
    "(%p %p %p %p)",
    pDcgmHandle,
    groupId,
    pDeviceConfig,
    statusHandle)

DCGM_ENTRY_POINT(
    dcgmVgpuConfigSet,
    tsapiEngineVgpuConfigSet,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmVgpuConfig_t *pDeviceConfig, dcgmStatus_t statusHandle),
    "(%p %p %p %p)",
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
                 "(%p %p %d %d %p %p)",
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
                 "(%p %p %d %d %p %p)",
                 pDcgmHandle,
                 groupId,
                 type,
                 count,
                 deviceConfigList,
                 statusHandle)

DCGM_ENTRY_POINT(dcgmConfigEnforce,
                 tsapiEngineConfigEnforce,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle),
                 "(%p %p %p)",
                 pDcgmHandle,
                 groupId,
                 statusHandle)

DCGM_ENTRY_POINT(dcgmVgpuConfigEnforce,
                 tsapiEngineVgpuConfigEnforce,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle),
                 "(%p %p %p)",
                 pDcgmHandle,
                 groupId,
                 statusHandle)

DCGM_ENTRY_POINT(
    dcgmGetLatestValuesForFields,
    tsapiEngineGetLatestValuesForFields,
    (dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldIds[], unsigned int count, dcgmFieldValue_v1 values[]),
    "(%p %d %p %d %p)",
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
                 "(%p %d %d %p %d %p)",
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
                 "(%p %d %d %p %lld %lld %d %p)",
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
                 "(%p %d %d %lld %f %d)",
                 pDcgmHandle,
                 gpuId,
                 fieldId,
                 updateFreq,
                 maxKeepAge,
                 maxKeepSamples)

DCGM_ENTRY_POINT(dcgmUnwatchFieldValue,
                 tsapiEngineUnwatchFieldValue,
                 (dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldId, int clearCache),
                 "(%p %d %d %d)",
                 pDcgmHandle,
                 gpuId,
                 fieldId,
                 clearCache)

DCGM_ENTRY_POINT(dcgmUpdateAllFields,
                 tsapiEngineUpdateAllFields,
                 (dcgmHandle_t pDcgmHandle, int waitForUpdate),
                 "(%p %d)",
                 pDcgmHandle,
                 waitForUpdate)

DCGM_ENTRY_POINT(dcgmPolicySet,
                 tsapiEnginePolicySet,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicy_t *policy, dcgmStatus_t statusHandle),
                 "(%p %p, %p, %p)",
                 pDcgmHandle,
                 groupId,
                 policy,
                 statusHandle)

DCGM_ENTRY_POINT(
    dcgmPolicyGet,
    tsapiEnginePolicyGet,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, int count, dcgmPolicy_t policy[], dcgmStatus_t statusHandle),
    "(%p %p, %d, %p, %p)",
    pDcgmHandle,
    groupId,
    count,
    policy,
    statusHandle)

DCGM_ENTRY_POINT(dcgmPolicyTrigger, tsapiEnginePolicyTrigger, (dcgmHandle_t pDcgmHandle), "%p", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmPolicyRegister,
                 tsapiEnginePolicyRegister,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmGpuGrp_t groupId,
                  dcgmPolicyCondition_t condition,
                  fpRecvUpdates beginCallback,
                  fpRecvUpdates finishCallback),
                 "(%p %p, %d, %p, %p)",
                 pDcgmHandle,
                 groupId,
                 condition,
                 beginCallback,
                 finishCallback)

DCGM_ENTRY_POINT(dcgmPolicyUnregister,
                 tsapiEnginePolicyUnregister,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition),
                 "(%p %p, %d)",
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
                 "(%p %p %lld %p %d %p %p %p)",
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
                 "(%p %p %p %lld %p %p %p)",
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
                 "(%p %p %p %lld %p %p %p)",
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
                 "(%p %p %p %p %p)",
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
                 "(%p %p %p %p %p)",
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
                 "(%p %p %u %p %u %u %p)",
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
                 "(%p %p, %p, %lld, %f, %d)",
                 pDcgmHandle,
                 groupId,
                 fieldGroupId,
                 updateFreq,
                 maxKeepAge,
                 maxKeepSamples)

DCGM_ENTRY_POINT(dcgmUnwatchFields,
                 tsapiUnwatchFields,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId),
                 "(%p %p, %p)",
                 pDcgmHandle,
                 groupId,
                 fieldGroupId)

DCGM_ENTRY_POINT(dcgmFieldGroupCreate,
                 tsapiFieldGroupCreate,
                 (dcgmHandle_t pDcgmHandle,
                  int numFieldIds,
                  unsigned short *fieldIds,
                  char *fieldGroupName,
                  dcgmFieldGrp_t *dcgmFieldGroupId),
                 "(%p %d, %p, %s, %p)",
                 pDcgmHandle,
                 numFieldIds,
                 fieldIds,
                 fieldGroupName,
                 dcgmFieldGroupId)

DCGM_ENTRY_POINT(dcgmFieldGroupDestroy,
                 tsapiFieldGroupDestroy,
                 (dcgmHandle_t pDcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId),
                 "(%p %p)",
                 pDcgmHandle,
                 dcgmFieldGroupId)

DCGM_ENTRY_POINT(dcgmFieldGroupGetInfo,
                 tsapiFieldGroupGetInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmFieldGroupInfo_t *fieldGroupInfo),
                 "(%p %p)",
                 pDcgmHandle,
                 fieldGroupInfo)

DCGM_ENTRY_POINT(dcgmFieldGroupGetAll,
                 tsapiFieldGroupGetAll,
                 (dcgmHandle_t pDcgmHandle, dcgmAllFieldGroup_t *allGroupInfo),
                 "(%p %p)",
                 pDcgmHandle,
                 allGroupInfo)

DCGM_ENTRY_POINT(dcgmHealthSet,
                 tsapiEngineHealthSet,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t systems),
                 "(%p %p, %d)",
                 pDcgmHandle,
                 groupId,
                 systems)

DCGM_ENTRY_POINT(dcgmHealthSet_v2,
                 tsapiEngineHealthSet_v2,
                 (dcgmHandle_t pDcgmHandle, dcgmHealthSetParams_v2 *params),
                 "(%p %p)",
                 pDcgmHandle,
                 params)

DCGM_ENTRY_POINT(dcgmHealthGet,
                 tsapiEngineHealthGet,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems),
                 "(%p %p, %p)",
                 pDcgmHandle,
                 groupId,
                 systems)

DCGM_ENTRY_POINT(dcgmHealthCheck,
                 tsapiEngineHealthCheck,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_t *results),
                 "(%p %p, %p)",
                 pDcgmHandle,
                 groupId,
                 results)

DCGM_ENTRY_POINT(dcgmActionValidate_v2,
                 tsapiEngineActionValidate_v2,
                 (dcgmHandle_t pDcgmHandle, dcgmRunDiag_t *drd, dcgmDiagResponse_t *response),
                 "(%p, %p, %p)",
                 pDcgmHandle,
                 drd,
                 response)

DCGM_ENTRY_POINT(
    dcgmActionValidate,
    tsapiEngineActionValidate,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyValidation_t validate, dcgmDiagResponse_t *response),
    "(%p %p, %d, %p)",
    pDcgmHandle,
    groupId,
    validate,
    response)

DCGM_ENTRY_POINT(
    dcgmRunDiagnostic,
    tsapiEngineRunDiagnostic,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmDiagnosticLevel_t diagLevel, dcgmDiagResponse_t *diagResponse),
    "(%p %p, %d, %p)",
    pDcgmHandle,
    groupId,
    diagLevel,
    diagResponse)

DCGM_ENTRY_POINT(dcgmStopDiagnostic, tsapiEngineStopDiagnostic, (dcgmHandle_t pDcgmHandle), "(%p)", pDcgmHandle)

DCGM_ENTRY_POINT(
    dcgmWatchPidFields,
    tsapiEngineWatchPidFields,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, long long updateFreq, double maxKeepAge, int maxKeepSamples),
    "(%p %p, %lld, %f, %d)",
    pDcgmHandle,
    groupId,
    updateFreq,
    maxKeepAge,
    maxKeepSamples)

DCGM_ENTRY_POINT(dcgmGetPidInfo,
                 tsapiEngineGetPidInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo),
                 "(%p %p %p)",
                 pDcgmHandle,
                 groupId,
                 pidInfo)

DCGM_ENTRY_POINT(dcgmGetDeviceTopology,
                 tsapiEngineGetDeviceTopology,
                 (dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmDeviceTopology_t *deviceTopology),
                 "(%p %d %p)",
                 pDcgmHandle,
                 gpuId,
                 deviceTopology)

DCGM_ENTRY_POINT(dcgmGetGroupTopology,
                 tsapiEngineGroupTopology,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupTopology_t *groupTopology),
                 "(%p %p %p)",
                 pDcgmHandle,
                 groupId,
                 groupTopology)

DCGM_ENTRY_POINT(
    dcgmWatchJobFields,
    tsapiEngineWatchJobFields,
    (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, long long updateFreq, double maxKeepAge, int maxKeepSamples),
    "(%p %p, %lld, %f, %d)",
    pDcgmHandle,
    groupId,
    updateFreq,
    maxKeepAge,
    maxKeepSamples)


DCGM_ENTRY_POINT(dcgmJobStartStats,
                 tsapiEngineJobStartStats,
                 (dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64]),
                 "(%p %p %p)",
                 pDcgmHandle,
                 groupId,
                 jobId)

DCGM_ENTRY_POINT(dcgmJobStopStats,
                 tsapiEngineJobStopStats,
                 (dcgmHandle_t pDcgmHandle, char jobId[64]),
                 "(%p %p)",
                 pDcgmHandle,
                 jobId)

DCGM_ENTRY_POINT(dcgmJobGetStats,
                 tsapiEngineJobGetStats,
                 (dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo),
                 "(%p %p %p)",
                 pDcgmHandle,
                 jobId,
                 pJobInfo)

DCGM_ENTRY_POINT(dcgmJobRemove,
                 tsapiEngineJobRemove,
                 (dcgmHandle_t pDcgmHandle, char jobId[64]),
                 "(%p %p)",
                 pDcgmHandle,
                 jobId)

DCGM_ENTRY_POINT(dcgmJobRemoveAll, tsapiEngineJobRemoveAll, (dcgmHandle_t pDcgmHandle), "(%p)", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmIntrospectToggleState,
                 tsapiMetadataToggleState,
                 (dcgmHandle_t pDcgmHandle, dcgmIntrospectState_t enabledState),
                 "(%p %d)",
                 pDcgmHandle,
                 enabledState)

DCGM_ENTRY_POINT(dcgmMetadataStateSetRunInterval,
                 tsapiMetadataStateSetRunInterval,
                 (dcgmHandle_t pDcgmHandle, unsigned int runIntervalMs),
                 "(%p %u)",
                 pDcgmHandle,
                 runIntervalMs)

DCGM_ENTRY_POINT(dcgmIntrospectUpdateAll,
                 tsapiMetadataUpdateAll,
                 (dcgmHandle_t pDcgmHandle, int waitForUpdate),
                 "(%p %d)",
                 pDcgmHandle,
                 waitForUpdate)

DCGM_ENTRY_POINT(
    dcgmIntrospectGetFieldExecTime,
    tsapiIntrospectGetFieldExecTime,
    (dcgmHandle_t pDcgmHandle, unsigned short fieldId, dcgmIntrospectFullFieldsExecTime_t *execTime, int waitIfNoData),
    "(%p %u %p %d)",
    pDcgmHandle,
    fieldId,
    execTime,
    waitIfNoData)

DCGM_ENTRY_POINT(dcgmIntrospectGetFieldsExecTime,
                 tsapiIntrospectGetFieldsExecTime,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmIntrospectContext_t *context,
                  dcgmIntrospectFullFieldsExecTime_t *execTime,
                  int waitIfNoData),
                 "(%p %p %p %d)",
                 pDcgmHandle,
                 context,
                 execTime,
                 waitIfNoData)

DCGM_ENTRY_POINT(dcgmIntrospectGetHostengineCpuUtilization,
                 tsapiIntrospectGetHostengineCpuUtilization,
                 (dcgmHandle_t pDcgmHandle, dcgmIntrospectCpuUtil_t *cpuUtil, int waitIfNoData),
                 "(%p %p %d)",
                 pDcgmHandle,
                 cpuUtil,
                 waitIfNoData)

DCGM_ENTRY_POINT(dcgmIntrospectGetFieldsMemoryUsage,
                 tsapiIntrospectGetFieldsMemoryUsage,
                 (dcgmHandle_t pDcgmHandle,
                  dcgmIntrospectContext_t *context,
                  dcgmIntrospectFullMemory_t *memoryInfo,
                  int waitIfNoData),
                 "(%p %p %p %d)",
                 pDcgmHandle,
                 context,
                 memoryInfo,
                 waitIfNoData)

DCGM_ENTRY_POINT(
    dcgmIntrospectGetFieldMemoryUsage,
    tsapiIntrospectGetFieldMemoryUsage,
    (dcgmHandle_t pDcgmHandle, unsigned short fieldId, dcgmIntrospectFullMemory_t *memoryInfo, int waitIfNoData),
    "(%p %u %p %d)",
    pDcgmHandle,
    fieldId,
    memoryInfo,
    waitIfNoData)

DCGM_ENTRY_POINT(dcgmIntrospectGetHostengineMemoryUsage,
                 tsapiIntrospectGetHostengineMemoryUsage,
                 (dcgmHandle_t pDcgmHandle, dcgmIntrospectMemory_t *memoryInfo, int waitIfNoData),
                 "(%p %p %d)",
                 pDcgmHandle,
                 memoryInfo,
                 waitIfNoData)

DCGM_ENTRY_POINT(
    dcgmSelectGpusByTopology,
    tsapiSelectGpusByTopology,
    (dcgmHandle_t pDcgmHandle, uint64_t inputGpuIds, uint32_t numGpus, uint64_t *outputGpuIds, uint64_t hintFlags),
    "(%p, %lu, %u, %p, %lu)",
    pDcgmHandle,
    inputGpuIds,
    numGpus,
    outputGpuIds,
    hintFlags)

DCGM_ENTRY_POINT(dcgmGetFieldSummary,
                 tsapiGetFieldSummary,
                 (dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request),
                 "(%p, %p)",
                 pDcgmHandle,
                 request)

DCGM_ENTRY_POINT(dcgmModuleBlacklist,
                 tsapiModuleBlacklist,
                 (dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId),
                 "(%p, %u)",
                 pDcgmHandle,
                 moduleId)

DCGM_ENTRY_POINT(dcgmModuleGetStatuses,
                 tsapiModuleGetStatuses,
                 (dcgmHandle_t pDcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses),
                 "(%p, %p)",
                 pDcgmHandle,
                 moduleStatuses)

DCGM_ENTRY_POINT(dcgmProfGetSupportedMetricGroups,
                 tsapiProfGetSupportedMetricGroups,
                 (dcgmHandle_t pDcgmHandle, dcgmProfGetMetricGroups_t *metricGroups),
                 "(%p, %p)",
                 pDcgmHandle,
                 metricGroups)

DCGM_ENTRY_POINT(dcgmProfWatchFields,
                 tsapiProfWatchFields,
                 (dcgmHandle_t pDcgmHandle, dcgmProfWatchFields_t *watchMetricGroup),
                 "(%p, %p)",
                 pDcgmHandle,
                 watchMetricGroup)

DCGM_ENTRY_POINT(dcgmProfUnwatchFields,
                 tsapiProfUnwatchFields,
                 (dcgmHandle_t pDcgmHandle, dcgmProfUnwatchFields_t *unwatchMetricGroup),
                 "(%p, %p)",
                 pDcgmHandle,
                 unwatchMetricGroup)

DCGM_ENTRY_POINT(dcgmProfPause, tsapiProfPause, (dcgmHandle_t pDcgmHandle), "(%p)", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmProfResume, tsapiProfResume, (dcgmHandle_t pDcgmHandle), "(%p)", pDcgmHandle)

DCGM_ENTRY_POINT(dcgmHostengineSetLoggingSeverity,
                 tsapiHostengineSetLoggingSeverity,
                 (dcgmHandle_t pDcgmHandle, dcgmSettingsSetLoggingSeverity_t *logging),
                 "(%p, %p)",
                 pDcgmHandle,
                 logging)

DCGM_ENTRY_POINT(dcgmVersionInfo, tsapiVersionInfo, (dcgmVersionInfo_t * pVersionInfo), "(%p)", pVersionInfo)

DCGM_ENTRY_POINT(dcgmHostengineVersionInfo,
                 tsapiHostengineVersionInfo,
                 (dcgmHandle_t pDcgmHandle, dcgmVersionInfo_t *pVersionInfo),
                 "(%p, %p)",
                 pDcgmHandle,
                 pVersionInfo)

DCGM_ENTRY_POINT(dcgmHostengineIsHealthy,
                 tsapiHostengineIsHealthy,
                 (dcgmHandle_t dcgmHandle, dcgmHostengineHealth_t *heHealth),
                 "(%p, %p)",
                 dcgmHandle,
                 heHealth)

DCGM_ENTRY_POINT(dcgmModuleIdToName,
                 tsapiDcgmModuleIdToName,
                 (dcgmModuleId_t id, char const **name),
                 "(%d, %p)",
                 id,
                 name)

/*****************************************************************************/
