/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include <dcgm_agent.h>
#include <dcgm_errors.h>
#include <dcgm_multinode_internal.h>
#include <dcgm_test_apis.h>

namespace DcgmNs
{

/**
 * Adapter for calling the public DCGM C API through a C++ type.
 *
 * This class provides the production implementation for components that accept
 * a DcgmOps-compatible object or a DcgmOpsAnyPtr-erased API handle.
 */
class DcgmOps final
{
public:
    static dcgmReturn_t dcgmInit();
    static dcgmReturn_t dcgmShutdown();
    static dcgmReturn_t dcgmConnect_v2(const char *ipAddress,
                                       dcgmConnectV2Params_t *connectParams,
                                       dcgmHandle_t *pDcgmHandle);
    static dcgmReturn_t dcgmConnect_v3(const char *connectionString,
                                       dcgmConnectV3Params_t *connectParams,
                                       dcgmHandle_t *pDcgmHandle);
    static dcgmReturn_t dcgmDisconnect(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmMultinodeRequest(dcgmHandle_t pDcgmHandle, dcgmMultinodeRequest_t *request);
    static dcgmReturn_t dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle);
    static dcgmReturn_t dcgmStartEmbedded_v2(dcgmStartEmbeddedV2Params_v1 *params);
    static dcgmReturn_t dcgmStopEmbedded(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmEngineRun(unsigned short portNumber, char const *socketPath, unsigned int connectionType);
    static dcgmReturn_t dcgmPauseTelemetryForDiag(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmResumeTelemetryForDiag(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmHostengineVersionInfo(dcgmHandle_t pDcgmHandle, dcgmVersionInfo_t *pVersionInfo);
    static dcgmReturn_t dcgmHostengineEnvironmentVariableInfo(dcgmHandle_t pDcgmHandle, dcgmEnvVarInfo_t *pEnvVarInfo);
    static dcgmReturn_t dcgmHostengineSetLoggingSeverity(dcgmHandle_t pDcgmHandle,
                                                         dcgmSettingsSetLoggingSeverity_t *logging);
    static dcgmReturn_t dcgmAttachDriver(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmDetachDriver(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmStatusCreate(dcgmStatus_t *statusHandle);
    static dcgmReturn_t dcgmStatusDestroy(dcgmStatus_t statusHandle);
    static dcgmReturn_t dcgmStatusPopError(dcgmStatus_t statusHandle, dcgmErrorInfo_t *errorInfo);
    static dcgmReturn_t dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                                    dcgmGroupEntityPair_t *entities,
                                                    unsigned int entityCount,
                                                    unsigned short *fields,
                                                    unsigned int fieldCount,
                                                    unsigned int flags,
                                                    dcgmFieldValue_v2 *values);
    static dcgmReturn_t dcgmGetGpuStatus(dcgmHandle_t pDcgmHandle, unsigned int gpuId, DcgmEntityStatus_t *status);
    static dcgmReturn_t dcgmGetAllDevices(dcgmHandle_t pDcgmHandle, unsigned int *gpuIdList, int *count);
    static dcgmReturn_t dcgmGetAllSupportedDevices(dcgmHandle_t pDcgmHandle, unsigned int *gpuIdList, int *count);
    static dcgmReturn_t dcgmGetGpuChipArchitecture(dcgmHandle_t dcgmHandle,
                                                   unsigned int gpuId,
                                                   dcgmChipArchitecture_t *chipArchitecture);
    static dcgmReturn_t dcgmGetNvLinkLinkStatus(dcgmHandle_t dcgmHandle, dcgmNvLinkStatus_v5 *linkStatus);
    static dcgmReturn_t dcgmGetDeviceWorkloadPowerProfileInfo(dcgmHandle_t pDcgmHandle,
                                                              unsigned int gpuId,
                                                              dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo,
                                                              dcgmDeviceWorkloadPowerProfilesStatus_v1 *profileStatus);
    static dcgmReturn_t dcgmGetGpuInstanceHierarchy(dcgmHandle_t dcgmHandle, dcgmMigHierarchy_v2 *hierarchy);
    static dcgmReturn_t dcgmGetCpuHierarchy_v2(dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v2 *cpuHierarchy);
    static dcgmReturn_t dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                        dcgmGroupType_t type,
                                        const char *groupName,
                                        dcgmGpuGrp_t *pDcgmGrpId);
    static dcgmReturn_t dcgmGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId);
    static dcgmReturn_t dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                           dcgmGpuGrp_t groupId,
                                           dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId);
    static dcgmReturn_t dcgmGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgm_field_entity_group_t entityGroupId,
                                              dcgm_field_eid_t entityId);
    static dcgmReturn_t dcgmGroupGetInfo(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgmGroupInfo_t *pDcgmGroupInfo);
    static dcgmReturn_t dcgmGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t *groupIdList, unsigned int *count);
    static dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                             int numFieldIds,
                                             const unsigned short *fieldIds,
                                             const char *fieldGroupName,
                                             dcgmFieldGrp_t *dcgmFieldGroupId);
    static dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId);
    static dcgmReturn_t dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId);
    static dcgmReturn_t dcgmFieldGroupGetInfo(dcgmHandle_t dcgmHandle, dcgmFieldGroupInfo_t *fieldGroupInfo);
    static dcgmReturn_t dcgmFieldGroupGetAll(dcgmHandle_t dcgmHandle, dcgmAllFieldGroup_t *allGroupInfo);
    static dcgmReturn_t dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmFieldGrp_t fieldGroupId,
                                        long long updateFreq,
                                        double maxKeepAge,
                                        int maxKeepSamples);
    static dcgmReturn_t dcgmUnwatchFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId);
    static dcgmReturn_t dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate);
    static dcgmReturn_t dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                               dcgmGpuGrp_t groupId,
                                               dcgmFieldGrp_t fieldGroupId,
                                               dcgmFieldValueEntityEnumeration_f enumCB,
                                               void *userData);
    static dcgmReturn_t dcgmGetValuesSince_v2(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgmFieldGrp_t fieldGroupId,
                                              long long sinceTimestamp,
                                              long long *nextSinceTimestamp,
                                              dcgmFieldValueEntityEnumeration_f enumCB,
                                              void *userData);
    static dcgmReturn_t dcgmGetLatestValuesForFields(dcgmHandle_t pDcgmHandle,
                                                     int gpuId,
                                                     unsigned short *fields,
                                                     unsigned int count,
                                                     dcgmFieldValue_v1 *values);
    static dcgmReturn_t dcgmGetMultipleValuesForField(dcgmHandle_t pDcgmHandle,
                                                      int gpuId,
                                                      unsigned short fieldId,
                                                      int *count,
                                                      long long startTs,
                                                      long long endTs,
                                                      dcgmOrder_t order,
                                                      dcgmFieldValue_v1 *values);
    static dcgmReturn_t dcgmGetFieldSummary(dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request);
    static dcgmReturn_t dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                              unsigned int gpuId,
                                              dcgmDeviceTopology_v2 *pDcgmDeviceTopology);
    static dcgmReturn_t dcgmGetGroupTopology(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmGroupTopology_v2 *pDcgmGroupTopology);
    static dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                unsigned int gpuId,
                                                dcgmDeviceAttributes_t *pDcgmAttr);
    static dcgmReturn_t dcgmGetEntityGroupEntities(dcgmHandle_t dcgmHandle,
                                                   dcgm_field_entity_group_t entityGroup,
                                                   dcgm_field_eid_t *entities,
                                                   int *numEntities,
                                                   unsigned int flags);
    static dcgmReturn_t dcgmConfigSet(dcgmHandle_t pDcgmHandle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmConfig_t *deviceConfig,
                                      dcgmStatus_t statusHandle);
    static dcgmReturn_t dcgmConfigGet(dcgmHandle_t pDcgmHandle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmConfigType_t reqType,
                                      int count,
                                      dcgmConfig_t *deviceConfigList,
                                      dcgmStatus_t statusHandle);
    static dcgmReturn_t dcgmConfigSetWorkloadPowerProfile(dcgmHandle_t pDcgmHandle,
                                                          dcgmWorkloadPowerProfile_t *workloadPowerProfile);
    static dcgmReturn_t dcgmConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle);
    /**
     * @name DeprecatedLegacyGroupPolicy
     * @deprecated Use the field-policy operations below.
     * @{
     */
    static dcgmReturn_t dcgmPolicySet(dcgmHandle_t pDcgmHandle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmPolicy_t *policy,
                                      dcgmStatus_t statusHandle);
    static dcgmReturn_t dcgmPolicyGet(dcgmHandle_t pDcgmHandle,
                                      dcgmGpuGrp_t groupId,
                                      int count,
                                      dcgmPolicy_t *policy,
                                      dcgmStatus_t statusHandle);
    static dcgmReturn_t dcgmPolicyUnregister(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmPolicyCondition_t condition);
    static dcgmReturn_t dcgmPolicyRegister_v2(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgmPolicyCondition_t condition,
                                              fpRecvUpdates callback,
                                              uint64_t userData);
    /** @} */

    /**
     * @name FieldPolicy
     * @{
     */
    static dcgmReturn_t dcgmPolicyCreate(dcgmHandle_t pDcgmHandle,
                                         unsigned short fieldId,
                                         const char *name,
                                         double threshold,
                                         dcgmPolicyOperator_t policyOperator,
                                         const dcgmGroupEntityPair_t *entities,
                                         unsigned int entityCount,
                                         unsigned int channelMask,
                                         unsigned int enabled,
                                         unsigned int rateLimitSec,
                                         uint64_t *policyId);
    static dcgmReturn_t dcgmPolicyModify(dcgmHandle_t pDcgmHandle, const dcgmPolicyInfo_t *policy);
    static dcgmReturn_t dcgmPolicyDelete(dcgmHandle_t pDcgmHandle,
                                         unsigned short fieldId,
                                         uint64_t policyId,
                                         unsigned int deleteAll);
    static dcgmReturn_t dcgmPolicyEnable(dcgmHandle_t pDcgmHandle, unsigned short fieldId, uint64_t policyId);
    static dcgmReturn_t dcgmPolicyDisable(dcgmHandle_t pDcgmHandle, unsigned short fieldId, uint64_t policyId);
    static dcgmReturn_t dcgmPolicyGetOne(dcgmHandle_t pDcgmHandle,
                                         unsigned short fieldId,
                                         uint64_t policyId,
                                         dcgmPolicyInfo_t *policy);
    static dcgmReturn_t dcgmPolicyGetAll(dcgmHandle_t pDcgmHandle, dcgmPolicyInfo_t *policies, unsigned int *count);
    static dcgmReturn_t dcgmPolicyImport(dcgmHandle_t pDcgmHandle, const char *path);
    static dcgmReturn_t dcgmPolicyGetViolations(dcgmHandle_t pDcgmHandle,
                                                int64_t sinceTimestamp,
                                                dcgm_policy_violation_t *violations,
                                                unsigned int *count);
    /** @} */

    static dcgmReturn_t dcgmHealthGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems);
    static dcgmReturn_t dcgmHealthSet_v2(dcgmHandle_t pDcgmHandle, dcgmHealthSetParams_v2 *params);
    static dcgmReturn_t dcgmHealthCheck(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_t *results);
    static dcgmReturn_t dcgmWatchPidFields(dcgmHandle_t pDcgmHandle,
                                           dcgmGpuGrp_t groupId,
                                           long long updateFreq,
                                           double maxKeepAge,
                                           int maxKeepSamples);
    static dcgmReturn_t dcgmGetPidInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo);
    static dcgmReturn_t dcgmJobStartStats(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64]);
    static dcgmReturn_t dcgmJobStopStats(dcgmHandle_t pDcgmHandle, char jobId[64]);
    static dcgmReturn_t dcgmJobRemove(dcgmHandle_t pDcgmHandle, char jobId[64]);
    static dcgmReturn_t dcgmJobRemoveAll(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmJobGetStats(dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo);
    static dcgmReturn_t dcgmIntrospectGetHostengineMemoryUsage(dcgmHandle_t pDcgmHandle,
                                                               dcgmIntrospectMemory_t *memoryInfo,
                                                               int waitIfNoData);
    static dcgmReturn_t dcgmIntrospectGetHostengineCpuUtilization(dcgmHandle_t pDcgmHandle,
                                                                  dcgmIntrospectCpuUtil_t *cpuUtil,
                                                                  int waitIfNoData);
    static dcgmReturn_t dcgmModuleDenylist(dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId);
    static dcgmReturn_t dcgmModuleGetStatuses(dcgmHandle_t pDcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses);
    static dcgmReturn_t dcgmModuleIdToName(dcgmModuleId_t id, char const **name);
    static dcgmReturn_t dcgmActionValidate_v2(dcgmHandle_t pDcgmHandle,
                                              dcgmRunDiag_v10 *drd,
                                              dcgmDiagResponse_v12 *response);
    static dcgmReturn_t dcgmStopDiagnostic(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmRunMnDiagnostic(dcgmHandle_t pDcgmHandle,
                                            dcgmRunMnDiag_v2 const *drmnd,
                                            dcgmMnDiagResponse_v2 *response);
    static dcgmReturn_t dcgmStopMnDiagnostic(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmDiagSendHeartbeat(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmProfGetSupportedMetricGroups(dcgmHandle_t pDcgmHandle,
                                                         dcgmProfGetMetricGroups_t *metricGroups);
    static dcgmReturn_t dcgmProfPause(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmProfResume(dcgmHandle_t pDcgmHandle);
    static dcgmReturn_t dcgmGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle,
                                                     dcgmCacheManagerFieldInfo_v4_t *fieldInfo);
    static dcgmReturn_t dcgmInjectFieldValue(dcgmHandle_t pDcgmHandle,
                                             unsigned int gpuId,
                                             dcgmInjectFieldValue_t *fieldValue);
    static dcgmReturn_t dcgmVgpuConfigGet(dcgmHandle_t pDcgmHandle,
                                          dcgmGpuGrp_t groupId,
                                          dcgmConfigType_t type,
                                          int count,
                                          dcgmVgpuConfig_t *deviceConfigList,
                                          dcgmStatus_t statusHandle);
    static dcgmReturn_t dcgmVgpuConfigSet(dcgmHandle_t pDcgmHandle,
                                          dcgmGpuGrp_t groupId,
                                          dcgmVgpuConfig_t *deviceConfig,
                                          dcgmStatus_t statusHandle);
    static dcgmReturn_t dcgmGetVgpuDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                    unsigned int gpuId,
                                                    dcgmVgpuDeviceAttributes_t *attributes);
    static dcgmReturn_t dcgmGetVgpuInstanceAttributes(dcgmHandle_t pDcgmHandle,
                                                      unsigned int vgpuId,
                                                      dcgmVgpuInstanceAttributes_t *attributes);
    static dcgmErrorSeverity_t dcgmErrorGetPriorityByCode(unsigned int code);
};

} //namespace DcgmNs
