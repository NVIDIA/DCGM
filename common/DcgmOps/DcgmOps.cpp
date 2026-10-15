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

#include "DcgmOps.h"

#include <dcgm_test_apis.h>

namespace DcgmNs
{

dcgmReturn_t DcgmOps::dcgmInit()
{
    return ::dcgmInit();
}

dcgmReturn_t DcgmOps::dcgmShutdown()
{
    return ::dcgmShutdown();
}

dcgmReturn_t DcgmOps::dcgmConnect_v2(const char *ipAddress,
                                     dcgmConnectV2Params_t *connectParams,
                                     dcgmHandle_t *pDcgmHandle)
{
    return ::dcgmConnect_v2(ipAddress, connectParams, pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmConnect_v3(const char *connectionString,
                                     dcgmConnectV3Params_t *connectParams,
                                     dcgmHandle_t *pDcgmHandle)
{
    return ::dcgmConnect_v3(connectionString, connectParams, pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmDisconnect(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmDisconnect(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmMultinodeRequest(dcgmHandle_t pDcgmHandle, dcgmMultinodeRequest_t *request)
{
    return ::dcgmMultinodeRequest(pDcgmHandle, request);
}

dcgmReturn_t DcgmOps::dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle)
{
    return ::dcgmStartEmbedded(opMode, pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmStartEmbedded_v2(dcgmStartEmbeddedV2Params_v1 *params)
{
    return ::dcgmStartEmbedded_v2(params);
}

dcgmReturn_t DcgmOps::dcgmStopEmbedded(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmStopEmbedded(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmEngineRun(unsigned short portNumber, char const *socketPath, unsigned int connectionType)
{
    return ::dcgmEngineRun(portNumber, socketPath, connectionType);
}

dcgmReturn_t DcgmOps::dcgmPauseTelemetryForDiag(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmPauseTelemetryForDiag(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmResumeTelemetryForDiag(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmResumeTelemetryForDiag(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmHostengineVersionInfo(dcgmHandle_t pDcgmHandle, dcgmVersionInfo_t *pVersionInfo)
{
    return ::dcgmHostengineVersionInfo(pDcgmHandle, pVersionInfo);
}

dcgmReturn_t DcgmOps::dcgmHostengineEnvironmentVariableInfo(dcgmHandle_t pDcgmHandle, dcgmEnvVarInfo_t *pEnvVarInfo)
{
    return ::dcgmHostengineEnvironmentVariableInfo(pDcgmHandle, pEnvVarInfo);
}

dcgmReturn_t DcgmOps::dcgmHostengineSetLoggingSeverity(dcgmHandle_t pDcgmHandle,
                                                       dcgmSettingsSetLoggingSeverity_t *logging)
{
    return ::dcgmHostengineSetLoggingSeverity(pDcgmHandle, logging);
}

dcgmReturn_t DcgmOps::dcgmAttachDriver(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmAttachDriver(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmDetachDriver(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmDetachDriver(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmStatusCreate(dcgmStatus_t *statusHandle)
{
    return ::dcgmStatusCreate(statusHandle);
}

dcgmReturn_t DcgmOps::dcgmStatusDestroy(dcgmStatus_t statusHandle)
{
    return ::dcgmStatusDestroy(statusHandle);
}

dcgmReturn_t DcgmOps::dcgmStatusPopError(dcgmStatus_t statusHandle, dcgmErrorInfo_t *errorInfo)
{
    return ::dcgmStatusPopError(statusHandle, errorInfo);
}

dcgmReturn_t DcgmOps::dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                                  dcgmGroupEntityPair_t entities[],
                                                  unsigned int entityCount,
                                                  unsigned short fields[],
                                                  unsigned int fieldCount,
                                                  unsigned int flags,
                                                  dcgmFieldValue_v2 values[])
{
    return ::dcgmEntitiesGetLatestValues(pDcgmHandle, entities, entityCount, fields, fieldCount, flags, values);
}

dcgmReturn_t DcgmOps::dcgmGetGpuStatus(dcgmHandle_t pDcgmHandle, unsigned int gpuId, DcgmEntityStatus_t *status)
{
    return ::dcgmGetGpuStatus(pDcgmHandle, gpuId, status);
}

dcgmReturn_t DcgmOps::dcgmGetAllDevices(dcgmHandle_t pDcgmHandle,
                                        unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                        int *count)
{
    return ::dcgmGetAllDevices(pDcgmHandle, gpuIdList, count);
}

dcgmReturn_t DcgmOps::dcgmGetAllSupportedDevices(dcgmHandle_t pDcgmHandle,
                                                 unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                                 int *count)
{
    return ::dcgmGetAllSupportedDevices(pDcgmHandle, gpuIdList, count);
}

dcgmReturn_t DcgmOps::dcgmGetGpuChipArchitecture(dcgmHandle_t dcgmHandle,
                                                 unsigned int gpuId,
                                                 dcgmChipArchitecture_t *chipArchitecture)
{
    return ::dcgmGetGpuChipArchitecture(dcgmHandle, gpuId, chipArchitecture);
}

dcgmReturn_t DcgmOps::dcgmGetNvLinkLinkStatus(dcgmHandle_t dcgmHandle, dcgmNvLinkStatus_v5 *linkStatus)
{
    return ::dcgmGetNvLinkLinkStatus(dcgmHandle, linkStatus);
}

dcgmReturn_t DcgmOps::dcgmGetDeviceWorkloadPowerProfileInfo(dcgmHandle_t pDcgmHandle,
                                                            unsigned int gpuId,
                                                            dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo,
                                                            dcgmDeviceWorkloadPowerProfilesStatus_v1 *profileStatus)
{
    return ::dcgmGetDeviceWorkloadPowerProfileInfo(pDcgmHandle, gpuId, profilesInfo, profileStatus);
}

dcgmReturn_t DcgmOps::dcgmGetGpuInstanceHierarchy(dcgmHandle_t dcgmHandle, dcgmMigHierarchy_v2 *hierarchy)
{
    return ::dcgmGetGpuInstanceHierarchy(dcgmHandle, hierarchy);
}

dcgmReturn_t DcgmOps::dcgmGetCpuHierarchy_v2(dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v2 *cpuHierarchy)
{
    return ::dcgmGetCpuHierarchy_v2(dcgmHandle, cpuHierarchy);
}

dcgmReturn_t DcgmOps::dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                      dcgmGroupType_t type,
                                      const char *groupName,
                                      dcgmGpuGrp_t *pDcgmGrpId)
{
    return ::dcgmGroupCreate(pDcgmHandle, type, groupName, pDcgmGrpId);
}

dcgmReturn_t DcgmOps::dcgmGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId)
{
    return ::dcgmGroupAddDevice(pDcgmHandle, groupId, gpuId);
}

dcgmReturn_t DcgmOps::dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgm_field_entity_group_t entityGroupId,
                                         dcgm_field_eid_t entityId)
{
    return ::dcgmGroupAddEntity(pDcgmHandle, groupId, entityGroupId, entityId);
}

dcgmReturn_t DcgmOps::dcgmGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId)
{
    return ::dcgmGroupRemoveEntity(pDcgmHandle, groupId, entityGroupId, entityId);
}

dcgmReturn_t DcgmOps::dcgmGroupGetInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupInfo_t *pDcgmGroupInfo)
{
    return ::dcgmGroupGetInfo(pDcgmHandle, groupId, pDcgmGroupInfo);
}

dcgmReturn_t DcgmOps::dcgmGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupIdList[], unsigned int *count)
{
    return ::dcgmGroupGetAllIds(pDcgmHandle, groupIdList, count);
}

dcgmReturn_t DcgmOps::dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                           int numFieldIds,
                                           const unsigned short *fieldIds,
                                           const char *fieldGroupName,
                                           dcgmFieldGrp_t *dcgmFieldGroupId)
{
    return ::dcgmFieldGroupCreate(dcgmHandle, numFieldIds, fieldIds, fieldGroupName, dcgmFieldGroupId);
}

dcgmReturn_t DcgmOps::dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId)
{
    return ::dcgmGroupDestroy(pDcgmHandle, groupId);
}

dcgmReturn_t DcgmOps::dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId)
{
    return ::dcgmFieldGroupDestroy(dcgmHandle, dcgmFieldGroupId);
}

dcgmReturn_t DcgmOps::dcgmFieldGroupGetInfo(dcgmHandle_t dcgmHandle, dcgmFieldGroupInfo_t *fieldGroupInfo)
{
    return ::dcgmFieldGroupGetInfo(dcgmHandle, fieldGroupInfo);
}

dcgmReturn_t DcgmOps::dcgmFieldGroupGetAll(dcgmHandle_t dcgmHandle, dcgmAllFieldGroup_t *allGroupInfo)
{
    return ::dcgmFieldGroupGetAll(dcgmHandle, allGroupInfo);
}

dcgmReturn_t DcgmOps::dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmFieldGrp_t fieldGroupId,
                                      long long updateFreq,
                                      double maxKeepAge,
                                      int maxKeepSamples)
{
    return ::dcgmWatchFields(pDcgmHandle, groupId, fieldGroupId, updateFreq, maxKeepAge, maxKeepSamples);
}

dcgmReturn_t DcgmOps::dcgmUnwatchFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId)
{
    return ::dcgmUnwatchFields(pDcgmHandle, groupId, fieldGroupId);
}

dcgmReturn_t DcgmOps::dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate)
{
    return ::dcgmUpdateAllFields(pDcgmHandle, waitForUpdate);
}

dcgmReturn_t DcgmOps::dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmFieldGrp_t fieldGroupId,
                                             dcgmFieldValueEntityEnumeration_f enumCB,
                                             void *userData)
{
    return ::dcgmGetLatestValues_v2(pDcgmHandle, groupId, fieldGroupId, enumCB, userData);
}

dcgmReturn_t DcgmOps::dcgmGetValuesSince_v2(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmFieldGrp_t fieldGroupId,
                                            long long sinceTimestamp,
                                            long long *nextSinceTimestamp,
                                            dcgmFieldValueEntityEnumeration_f enumCB,
                                            void *userData)
{
    return ::dcgmGetValuesSince_v2(
        pDcgmHandle, groupId, fieldGroupId, sinceTimestamp, nextSinceTimestamp, enumCB, userData);
}

dcgmReturn_t DcgmOps::dcgmGetLatestValuesForFields(dcgmHandle_t pDcgmHandle,
                                                   int gpuId,
                                                   unsigned short fields[],
                                                   unsigned int count,
                                                   dcgmFieldValue_v1 values[])
{
    return ::dcgmGetLatestValuesForFields(pDcgmHandle, gpuId, fields, count, values);
}

dcgmReturn_t DcgmOps::dcgmGetMultipleValuesForField(dcgmHandle_t pDcgmHandle,
                                                    int gpuId,
                                                    unsigned short fieldId,
                                                    int *count,
                                                    long long startTs,
                                                    long long endTs,
                                                    dcgmOrder_t order,
                                                    dcgmFieldValue_v1 values[])
{
    return ::dcgmGetMultipleValuesForField(pDcgmHandle, gpuId, fieldId, count, startTs, endTs, order, values);
}

dcgmReturn_t DcgmOps::dcgmGetFieldSummary(dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request)
{
    return ::dcgmGetFieldSummary(pDcgmHandle, request);
}

dcgmReturn_t DcgmOps::dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                            unsigned int gpuId,
                                            dcgmDeviceTopology_v2 *pDcgmDeviceTopology)
{
    return ::dcgmGetDeviceTopology(pDcgmHandle, gpuId, pDcgmDeviceTopology);
}

dcgmReturn_t DcgmOps::dcgmGetGroupTopology(dcgmHandle_t pDcgmHandle,
                                           dcgmGpuGrp_t groupId,
                                           dcgmGroupTopology_v2 *pDcgmGroupTopology)
{
    return ::dcgmGetGroupTopology(pDcgmHandle, groupId, pDcgmGroupTopology);
}

dcgmReturn_t DcgmOps::dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                              unsigned int gpuId,
                                              dcgmDeviceAttributes_t *pDcgmAttr)
{
    return ::dcgmGetDeviceAttributes(pDcgmHandle, gpuId, pDcgmAttr);
}

dcgmReturn_t DcgmOps::dcgmGetEntityGroupEntities(dcgmHandle_t dcgmHandle,
                                                 dcgm_field_entity_group_t entityGroup,
                                                 dcgm_field_eid_t *entities,
                                                 int *numEntities,
                                                 unsigned int flags)
{
    return ::dcgmGetEntityGroupEntities(dcgmHandle, entityGroup, entities, numEntities, flags);
}

dcgmReturn_t DcgmOps::dcgmConfigSet(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgmConfig_t *deviceConfig,
                                    dcgmStatus_t statusHandle)
{
    return ::dcgmConfigSet(pDcgmHandle, groupId, deviceConfig, statusHandle);
}

dcgmReturn_t DcgmOps::dcgmConfigGet(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgmConfigType_t reqType,
                                    int count,
                                    dcgmConfig_t deviceConfigList[],
                                    dcgmStatus_t statusHandle)
{
    return ::dcgmConfigGet(pDcgmHandle, groupId, reqType, count, deviceConfigList, statusHandle);
}

dcgmReturn_t DcgmOps::dcgmConfigSetWorkloadPowerProfile(dcgmHandle_t pDcgmHandle,
                                                        dcgmWorkloadPowerProfile_t *workloadPowerProfile)
{
    return ::dcgmConfigSetWorkloadPowerProfile(pDcgmHandle, workloadPowerProfile);
}

dcgmReturn_t DcgmOps::dcgmConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle)
{
    return ::dcgmConfigEnforce(pDcgmHandle, groupId, statusHandle);
}

dcgmReturn_t DcgmOps::dcgmPolicySet(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgmPolicy_t *policy,
                                    dcgmStatus_t statusHandle)
{
    return ::dcgmPolicySet(pDcgmHandle, groupId, policy, statusHandle);
}

dcgmReturn_t DcgmOps::dcgmPolicyGet(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    int count,
                                    dcgmPolicy_t *policy,
                                    dcgmStatus_t statusHandle)
{
    return ::dcgmPolicyGet(pDcgmHandle, groupId, count, policy, statusHandle);
}

dcgmReturn_t DcgmOps::dcgmPolicyUnregister(dcgmHandle_t pDcgmHandle,
                                           dcgmGpuGrp_t groupId,
                                           dcgmPolicyCondition_t condition)
{
    return ::dcgmPolicyUnregister(pDcgmHandle, groupId, condition);
}

dcgmReturn_t DcgmOps::dcgmPolicyRegister_v2(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmPolicyCondition_t condition,
                                            fpRecvUpdates callback,
                                            uint64_t userData)
{
    return ::dcgmPolicyRegister_v2(pDcgmHandle, groupId, condition, callback, userData);
}

dcgmReturn_t DcgmOps::dcgmPolicyCreate(dcgmHandle_t pDcgmHandle,
                                       unsigned short fieldId,
                                       const char *name,
                                       double threshold,
                                       dcgmPolicyOperator_t policyOperator,
                                       const dcgmGroupEntityPair_t *entities,
                                       unsigned int entityCount,
                                       unsigned int channelMask,
                                       unsigned int enabled,
                                       unsigned int rateLimitSec,
                                       uint64_t *policyId)
{
    return ::dcgmPolicyCreate(pDcgmHandle,
                              fieldId,
                              name,
                              threshold,
                              policyOperator,
                              entities,
                              entityCount,
                              channelMask,
                              enabled,
                              rateLimitSec,
                              policyId);
}

dcgmReturn_t DcgmOps::dcgmPolicyModify(dcgmHandle_t pDcgmHandle, const dcgmPolicyInfo_t *policy)
{
    return ::dcgmPolicyModify(pDcgmHandle, policy);
}

dcgmReturn_t DcgmOps::dcgmPolicyDelete(dcgmHandle_t pDcgmHandle,
                                       unsigned short fieldId,
                                       uint64_t policyId,
                                       unsigned int deleteAll)
{
    return ::dcgmPolicyDelete(pDcgmHandle, fieldId, policyId, deleteAll);
}

dcgmReturn_t DcgmOps::dcgmPolicyEnable(dcgmHandle_t pDcgmHandle, unsigned short fieldId, uint64_t policyId)
{
    return ::dcgmPolicyEnable(pDcgmHandle, fieldId, policyId);
}

dcgmReturn_t DcgmOps::dcgmPolicyDisable(dcgmHandle_t pDcgmHandle, unsigned short fieldId, uint64_t policyId)
{
    return ::dcgmPolicyDisable(pDcgmHandle, fieldId, policyId);
}

dcgmReturn_t DcgmOps::dcgmPolicyGetOne(dcgmHandle_t pDcgmHandle,
                                       unsigned short fieldId,
                                       uint64_t policyId,
                                       dcgmPolicyInfo_t *policy)
{
    return ::dcgmPolicyGetOne(pDcgmHandle, fieldId, policyId, policy);
}

dcgmReturn_t DcgmOps::dcgmPolicyGetAll(dcgmHandle_t pDcgmHandle, dcgmPolicyInfo_t *policies, unsigned int *count)
{
    return ::dcgmPolicyGetAll(pDcgmHandle, policies, count);
}

dcgmReturn_t DcgmOps::dcgmPolicyImport(dcgmHandle_t pDcgmHandle, const char *path)
{
    return ::dcgmPolicyImport(pDcgmHandle, path);
}

dcgmReturn_t DcgmOps::dcgmPolicyGetViolations(dcgmHandle_t pDcgmHandle,
                                              int64_t sinceTimestamp,
                                              dcgm_policy_violation_t *violations,
                                              unsigned int *count)
{
    return ::dcgmPolicyGetViolations(pDcgmHandle, sinceTimestamp, violations, count);
}

dcgmReturn_t DcgmOps::dcgmHealthGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems)
{
    return ::dcgmHealthGet(pDcgmHandle, groupId, systems);
}

dcgmReturn_t DcgmOps::dcgmHealthSet_v2(dcgmHandle_t pDcgmHandle, dcgmHealthSetParams_v2 *params)
{
    return ::dcgmHealthSet_v2(pDcgmHandle, params);
}

dcgmReturn_t DcgmOps::dcgmHealthCheck(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_t *results)
{
    return ::dcgmHealthCheck(pDcgmHandle, groupId, results);
}

dcgmReturn_t DcgmOps::dcgmWatchPidFields(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         long long updateFreq,
                                         double maxKeepAge,
                                         int maxKeepSamples)
{
    return ::dcgmWatchPidFields(pDcgmHandle, groupId, updateFreq, maxKeepAge, maxKeepSamples);
}

dcgmReturn_t DcgmOps::dcgmGetPidInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo)
{
    return ::dcgmGetPidInfo(pDcgmHandle, groupId, pidInfo);
}

dcgmReturn_t DcgmOps::dcgmJobStartStats(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64])
{
    return ::dcgmJobStartStats(pDcgmHandle, groupId, jobId);
}

dcgmReturn_t DcgmOps::dcgmJobStopStats(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    return ::dcgmJobStopStats(pDcgmHandle, jobId);
}

dcgmReturn_t DcgmOps::dcgmJobRemove(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    return ::dcgmJobRemove(pDcgmHandle, jobId);
}

dcgmReturn_t DcgmOps::dcgmJobRemoveAll(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmJobRemoveAll(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmJobGetStats(dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo)
{
    return ::dcgmJobGetStats(pDcgmHandle, jobId, pJobInfo);
}

dcgmReturn_t DcgmOps::dcgmIntrospectGetHostengineMemoryUsage(dcgmHandle_t pDcgmHandle,
                                                             dcgmIntrospectMemory_t *memoryInfo,
                                                             int waitIfNoData)
{
    return ::dcgmIntrospectGetHostengineMemoryUsage(pDcgmHandle, memoryInfo, waitIfNoData);
}

dcgmReturn_t DcgmOps::dcgmIntrospectGetHostengineCpuUtilization(dcgmHandle_t pDcgmHandle,
                                                                dcgmIntrospectCpuUtil_t *cpuUtil,
                                                                int waitIfNoData)
{
    return ::dcgmIntrospectGetHostengineCpuUtilization(pDcgmHandle, cpuUtil, waitIfNoData);
}

dcgmReturn_t DcgmOps::dcgmModuleDenylist(dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId)
{
    return ::dcgmModuleDenylist(pDcgmHandle, moduleId);
}

dcgmReturn_t DcgmOps::dcgmModuleGetStatuses(dcgmHandle_t pDcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses)
{
    return ::dcgmModuleGetStatuses(pDcgmHandle, moduleStatuses);
}

dcgmReturn_t DcgmOps::dcgmModuleIdToName(dcgmModuleId_t id, char const **name)
{
    return ::dcgmModuleIdToName(id, name);
}

dcgmReturn_t DcgmOps::dcgmActionValidate_v2(dcgmHandle_t pDcgmHandle,
                                            dcgmRunDiag_v10 *drd,
                                            dcgmDiagResponse_v12 *response)
{
    return ::dcgmActionValidate_v2(pDcgmHandle, drd, response);
}

dcgmReturn_t DcgmOps::dcgmStopDiagnostic(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmStopDiagnostic(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmRunMnDiagnostic(dcgmHandle_t pDcgmHandle,
                                          dcgmRunMnDiag_v2 const *drmnd,
                                          dcgmMnDiagResponse_v2 *response)
{
    return ::dcgmRunMnDiagnostic(pDcgmHandle, drmnd, response);
}

dcgmReturn_t DcgmOps::dcgmStopMnDiagnostic(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmStopMnDiagnostic(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmDiagSendHeartbeat(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmDiagSendHeartbeat(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmProfGetSupportedMetricGroups(dcgmHandle_t pDcgmHandle,
                                                       dcgmProfGetMetricGroups_t *metricGroups)
{
    return ::dcgmProfGetSupportedMetricGroups(pDcgmHandle, metricGroups);
}

dcgmReturn_t DcgmOps::dcgmProfPause(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmProfPause(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmProfResume(dcgmHandle_t pDcgmHandle)
{
    return ::dcgmProfResume(pDcgmHandle);
}

dcgmReturn_t DcgmOps::dcgmGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle, dcgmCacheManagerFieldInfo_v4_t *fieldInfo)
{
    return ::dcgmGetCacheManagerFieldInfo(pDcgmHandle, fieldInfo);
}

dcgmReturn_t DcgmOps::dcgmInjectFieldValue(dcgmHandle_t pDcgmHandle,
                                           unsigned int gpuId,
                                           dcgmInjectFieldValue_t *fieldValue)
{
    return ::dcgmInjectFieldValue(pDcgmHandle, gpuId, fieldValue);
}

dcgmReturn_t DcgmOps::dcgmVgpuConfigGet(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmConfigType_t type,
                                        int count,
                                        dcgmVgpuConfig_t deviceConfigList[],
                                        dcgmStatus_t statusHandle)
{
    return ::dcgmVgpuConfigGet(pDcgmHandle, groupId, type, count, deviceConfigList, statusHandle);
}

dcgmReturn_t DcgmOps::dcgmVgpuConfigSet(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmVgpuConfig_t *deviceConfig,
                                        dcgmStatus_t statusHandle)
{
    return ::dcgmVgpuConfigSet(pDcgmHandle, groupId, deviceConfig, statusHandle);
}

dcgmReturn_t DcgmOps::dcgmGetVgpuDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                  unsigned int gpuId,
                                                  dcgmVgpuDeviceAttributes_t *attributes)
{
    return ::dcgmGetVgpuDeviceAttributes(pDcgmHandle, gpuId, attributes);
}

dcgmReturn_t DcgmOps::dcgmGetVgpuInstanceAttributes(dcgmHandle_t pDcgmHandle,
                                                    unsigned int vgpuId,
                                                    dcgmVgpuInstanceAttributes_t *attributes)
{
    return ::dcgmGetVgpuInstanceAttributes(pDcgmHandle, vgpuId, attributes);
}

dcgmErrorSeverity_t DcgmOps::dcgmErrorGetPriorityByCode(unsigned int code)
{
    return ::dcgmErrorGetPriorityByCode(code);
}

} //namespace DcgmNs
