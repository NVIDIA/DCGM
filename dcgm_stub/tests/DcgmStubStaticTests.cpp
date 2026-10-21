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

#include <catch2/catch_test_macros.hpp>

#include <StderrCapture.hpp>
#include <dcgm_agent.h>
#include <dcgm_fields.h>
#include <dcgm_multinode_internal.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <dcgm_test_apis.h>
#include <nvml_injection_structs.h>

#include <string>

extern "C" unsigned int dcgmCpuHierarchyCpuOwnsCore(unsigned int coreId,
                                                    dcgmCpuHierarchyOwnedCores_v1 const *ownedCores);
extern "C" dcgmReturn_t dcgmInjectEntityFieldValue(dcgmHandle_t pDcgmHandle,
                                                   dcgm_field_entity_group_t entityGroupId,
                                                   dcgm_field_eid_t entityId,
                                                   dcgmInjectFieldValue_t *pDcgmInjectFieldValue);
extern "C" dcgmReturn_t dcgmInjectEntityFieldValueToNvml(dcgmHandle_t pDcgmHandle,
                                                         dcgm_field_entity_group_t entityGroupId,
                                                         dcgm_field_eid_t entityId,
                                                         dcgmInjectFieldValue_t *pDcgmInjectFieldValue);
extern "C" dcgmReturn_t dcgmCreateNvmlInjectionGpu(dcgmHandle_t dcgmHandle, unsigned int index);
extern "C" dcgmReturn_t dcgmInjectNvmlDevice(dcgmHandle_t dcgmHandle,
                                             unsigned int gpuId,
                                             const char *key,
                                             const injectNvmlVal_t *extraKeys,
                                             unsigned int extraKeyCount,
                                             const injectNvmlRet_t *injectNvmlRet);
extern "C" dcgmReturn_t dcgmInjectNvmlDeviceForFollowingCalls(dcgmHandle_t dcgmHandle,
                                                              unsigned int gpuId,
                                                              const char *key,
                                                              const injectNvmlVal_t *extraKeys,
                                                              unsigned int extraKeyCount,
                                                              const injectNvmlRet_t *injectNvmlRets,
                                                              unsigned int retCount);
extern "C" dcgmReturn_t dcgmInjectedNvmlDeviceReset(dcgmHandle_t dcgmHandle, unsigned int gpuId);
extern "C" dcgmReturn_t dcgmGetNvmlInjectFuncCallCount(dcgmHandle_t dcgmHandle,
                                                       injectNvmlFuncCallCounts_t *nvmlFuncCallCounts);
extern "C" dcgmReturn_t dcgmResetNvmlInjectFuncCallCount(dcgmHandle_t dcgmHandle);
extern "C" dcgmReturn_t dcgmRemoveNvmlInjectedGpu(dcgmHandle_t dcgmHandle, char const *uuid);
extern "C" dcgmReturn_t dcgmRestoreNvmlInjectedGpu(dcgmHandle_t dcgmHandle, char const *uuid);


SCENARIO("dcgm_stub reports uninitialized state before the real library is loaded")
{
    GIVEN("the static forwarding library has not loaded libdcgm")
    {
        WHEN("errorString is called")
        {
            StderrCapture capture;
            char const *result = errorString(DCGM_ST_OK);

            THEN("the call reports the wrong-library diagnostic and returns no string")
            {
                CHECK(result == nullptr);
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }

        WHEN("dcgmShutdown is called")
        {
            StderrCapture capture;
            dcgmReturn_t const result = dcgmShutdown();

            THEN("the call is rejected as uninitialized")
            {
                CHECK(result == DCGM_ST_UNINITIALIZED);
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }

        WHEN("a generated DCGM API wrapper is called")
        {
            StderrCapture capture;
            unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] {};
            int count                 = 0;
            dcgmReturn_t const result = dcgmGetAllDevices(0, gpuIds, &count);

            THEN("the dynamic wrapper rejects the request before touching output buffers")
            {
                CHECK(result == DCGM_ST_UNINITIALIZED);
                CHECK(count == 0);
                CHECK(gpuIds[0] == 0);
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }

        WHEN("another generated wrapper with unsigned count is called")
        {
            StderrCapture capture;
            dcgmGpuGrp_t groupIds[DCGM_MAX_NUM_GROUPS] {};
            unsigned int count        = 7;
            dcgmReturn_t const result = dcgmGroupGetAllIds(0, groupIds, &count);

            THEN("the generated wrapper returns the same uninitialized contract")
            {
                CHECK(result == DCGM_ST_UNINITIALIZED);
                CHECK(count == 7);
                CHECK(groupIds[0] == 0);
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }


        WHEN("additional generated wrappers with varied signatures are called")
        {
            StderrCapture capture;
            unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] {};
            int gpuCount = 4;
            dcgmDeviceAttributes_t attributes {};
            DcgmEntityStatus_t status = DcgmEntityStatusOk;
            dcgm_field_eid_t entities[4] { 9, 8, 7, 6 };
            int entityCount              = 4;
            dcgmReturn_t const supported = dcgmGetAllSupportedDevices(0, gpuIds, &gpuCount);
            dcgmReturn_t const attr      = dcgmGetDeviceAttributes(0, 3, &attributes);
            dcgmReturn_t const gpuStatus = dcgmGetGpuStatus(0, 3, &status);
            dcgmReturn_t const entityRet = dcgmGetEntityGroupEntities(0, DCGM_FE_GPU, entities, &entityCount, 0);
            dcgmReturn_t const engineRet = dcgmEngineRun(5555, "/tmp/dcgm.sock", DCGM_OPERATION_MODE_AUTO);

            THEN("each generated wrapper reports the same uninitialized state without modifying caller buffers")
            {
                CHECK(supported == DCGM_ST_UNINITIALIZED);
                CHECK(attr == DCGM_ST_UNINITIALIZED);
                CHECK(gpuStatus == DCGM_ST_UNINITIALIZED);
                CHECK(entityRet == DCGM_ST_UNINITIALIZED);
                CHECK(engineRet == DCGM_ST_UNINITIALIZED);
                CHECK(gpuCount == 4);
                CHECK(status == DcgmEntityStatusOk);
                CHECK(entityCount == 4);
                CHECK(entities[0] == 9);
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }

        WHEN("generated wrappers across DCGM subsystems are called")
        {
            StderrCapture capture;
            auto handle = static_cast<dcgmHandle_t>(0x1234);

            dcgmStatus_t statusHandle = 0;
            unsigned int count        = 3;
            int gpuCount              = 4;
            dcgmErrorInfo_t errorInfo {};
            dcgmGroupInfo_t groupInfo {};
            dcgmConfig_t config {};
            dcgmVgpuConfig_t vgpuConfig {};
            dcgmPolicy_t policy {};
            unsigned short fieldIds[2] { DCGM_FI_DEV_BOARD_POWER_WATTS, DCGM_FI_DEV_GPU_TEMP_CELSIUS };
            dcgmFieldValue_v1 fieldValues[2] {};
            dcgmFieldValue_v2 entityValues[2] {};
            dcgmGroupEntityPair_t entities[1] {};
            long long nextSinceTimestamp = 7;
            dcgmFieldGrp_t fieldGroupId  = static_cast<dcgmFieldGrp_t>(0x4321);
            dcgmFieldGroupInfo_t fieldGroupInfo {};
            dcgmAllFieldGroup_t allFieldGroupInfo {};
            dcgmHealthSystems_t healthSystems = DCGM_HEALTH_WATCH_PCIE;
            dcgmHealthResponse_t healthResponse {};
            dcgmDiagResponse_v12 diagResponse {};
            dcgmRunDiag_v10 runDiag {};
            dcgmRunMnDiag_v2 mnDiag {};
            dcgmMnDiagResponse_v2 mnDiagResponse {};
            dcgmMultinodeRequest_t multinodeRequest {};
            dcgmPidInfo_t pidInfo {};
            dcgmDeviceTopology_t deviceTopology {};
            dcgmGroupTopology_t groupTopology {};
            char jobId[64] = "stub-job";
            dcgmJobInfo_t jobInfo {};
            dcgmIntrospectCpuUtil_t cpuUtil {};
            dcgmIntrospectMemory_t memory {};
            uint64_t outputGpuIds = 0;
            dcgmFieldSummaryRequest_t summaryRequest {};
            dcgmModuleGetStatuses_t moduleStatuses {};
            dcgmProfGetMetricGroups_t metricGroups {};
            dcgmSettingsSetLoggingSeverity_t logging {};
            dcgmVersionInfo_t versionInfo {};
            dcgmHostengineHealth_t hostengineHealth {};
            char const *name = nullptr;
            dcgmWorkloadPowerProfile_t workloadPowerProfile {};
            dcgmEnvVarInfo_t envVarInfo {};
            dcgmModulesReloadable_v1 reloadable {};
            dcgmNvLinkStatus_v5 nvLinkStatus {};
            dcgmNvLinkP2PStatus_v1 nvLinkP2PStatus {};
            dcgmCpuHierarchy_v1 cpuHierarchy {};
            dcgmCpuHierarchy_v2 cpuHierarchyV2 {};
            dcgmMigHierarchy_v2 migHierarchy {};
            dcgmCreateMigEntity_t createMig {};
            dcgmDeleteMigEntity_t deleteMig {};

            THEN("each wrapper reports the same uninitialized forwarding-library contract")
            {
                CHECK(dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, "stub-group", nullptr) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGroupDestroy(handle, 1) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGroupAddDevice(handle, 1, 2) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGroupAddEntity(handle, 1, DCGM_FE_GPU, 2) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGroupRemoveDevice(handle, 1, 2) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGroupRemoveEntity(handle, 1, DCGM_FE_GPU, 2) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGroupGetInfo(handle, 1, &groupInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmStatusCreate(&statusHandle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmStatusDestroy(statusHandle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmStatusGetCount(statusHandle, &count) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmStatusPopError(statusHandle, &errorInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmStatusClear(statusHandle) == DCGM_ST_UNINITIALIZED);

                CHECK(dcgmConfigSet(handle, 1, &config, statusHandle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmVgpuConfigSet(handle, 1, &vgpuConfig, statusHandle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmConfigGet(handle, 1, DCGM_CONFIG_CURRENT_STATE, 1, &config, statusHandle)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmVgpuConfigGet(handle, 1, DCGM_CONFIG_CURRENT_STATE, 1, &vgpuConfig, statusHandle)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmConfigEnforce(handle, 1, statusHandle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmVgpuConfigEnforce(handle, 1, statusHandle) == DCGM_ST_UNINITIALIZED);

                CHECK(dcgmGetLatestValuesForFields(handle, 2, fieldIds, 2, fieldValues) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmEntityGetLatestValues(handle, DCGM_FE_GPU, 2, fieldIds, 2, fieldValues)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetMultipleValuesForField(
                          handle, 2, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpuCount, 1, 2, DCGM_ORDER_ASCENDING, fieldValues)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmWatchFieldValue(handle, 2, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 5.0, 3)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmUnwatchFieldValue(handle, 2, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmUpdateAllFields(handle, 1) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmEntitiesGetLatestValues(handle, entities, 1, fieldIds, 2, 0, entityValues)
                      == DCGM_ST_UNINITIALIZED);

                CHECK(dcgmPolicySet(handle, 1, &policy, statusHandle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmPolicyGet(handle, 1, 1, &policy, statusHandle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmPolicyTrigger(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmPolicyRegister_v2(handle, 1, static_cast<dcgmPolicyCondition_t>(0), nullptr, 0)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmPolicyUnregister(handle, 1, static_cast<dcgmPolicyCondition_t>(0)) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetFieldValuesSince(handle, 1, 1, fieldIds, 2, &nextSinceTimestamp, nullptr, nullptr)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetValuesSince(handle, 1, fieldGroupId, 1, &nextSinceTimestamp, nullptr, nullptr)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetValuesSince_v2(handle, 1, fieldGroupId, 1, &nextSinceTimestamp, nullptr, nullptr)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetLatestValues(handle, 1, fieldGroupId, nullptr, nullptr) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetLatestValues_v2(handle, 1, fieldGroupId, nullptr, nullptr) == DCGM_ST_UNINITIALIZED);

                CHECK(dcgmFieldGroupCreate(handle, 2, fieldIds, "fields", &fieldGroupId) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmFieldGroupDestroy(handle, fieldGroupId) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmFieldGroupGetInfo(handle, &fieldGroupInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmFieldGroupGetAll(handle, &allFieldGroupInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmHealthSet(handle, 1, healthSystems) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmHealthGet(handle, 1, &healthSystems) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmHealthCheck(handle, 1, &healthResponse) == DCGM_ST_UNINITIALIZED);

                CHECK(dcgmActionValidate_v2(handle, &runDiag, &diagResponse) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmActionValidate(handle, 1, static_cast<dcgmPolicyValidation_t>(0), &diagResponse)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmRunDiagnostic(handle, 1, static_cast<dcgmDiagnosticLevel_t>(0), &diagResponse)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmStopDiagnostic(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmRunMnDiagnostic(handle, &mnDiag, &mnDiagResponse) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmStopMnDiagnostic(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmMultinodeRequest(handle, &multinodeRequest) == DCGM_ST_UNINITIALIZED);

                CHECK(dcgmWatchPidFields(handle, 1, 1000, 10.0, 5) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetPidInfo(handle, 1, &pidInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetDeviceTopology(handle, 2, &deviceTopology) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetGroupTopology(handle, 1, &groupTopology) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmWatchJobFields(handle, 1, 1000, 10.0, 5) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmJobStartStats(handle, 1, jobId) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmJobStopStats(handle, jobId) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmJobGetStats(handle, jobId, &jobInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmJobRemove(handle, jobId) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmJobRemoveAll(handle) == DCGM_ST_UNINITIALIZED);

                CHECK(dcgmIntrospectGetHostengineCpuUtilization(handle, &cpuUtil, 0) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmIntrospectGetHostengineMemoryUsage(handle, &memory, 0) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmSelectGpusByTopology(handle, 0x3, 1, &outputGpuIds, 0) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetFieldSummary(handle, &summaryRequest) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmModuleDenylist(handle, DcgmModuleIdCore) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmModuleGetStatuses(handle, &moduleStatuses) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmProfGetSupportedMetricGroups(handle, &metricGroups) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmProfPause(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmProfResume(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmHostengineSetLoggingSeverity(handle, &logging) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmVersionInfo(&versionInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmHostengineVersionInfo(handle, &versionInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmHostengineIsHealthy(handle, &hostengineHealth) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmModuleIdToName(DcgmModuleIdCore, &name) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmPowerProfileIdToName(static_cast<dcgmPowerProfileType_t>(0), &name) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmPauseTelemetryForDiag(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmResumeTelemetryForDiag(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmConfigSetWorkloadPowerProfile(handle, &workloadPowerProfile) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmDiagSendHeartbeat(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmHostengineEnvironmentVariableInfo(handle, &envVarInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmAttachDriver(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmDetachDriver(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmEmptyCache(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmMarkModulesReloadable(handle, &reloadable) == DCGM_ST_UNINITIALIZED);

                CHECK(dcgmGetGpuInstanceHierarchy(handle, &migHierarchy) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmCreateMigEntity(handle, &createMig) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmDeleteMigEntity(handle, &deleteMig) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetNvLinkLinkStatus(handle, &nvLinkStatus) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetNvLinkP2PStatus(handle, &nvLinkP2PStatus) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetCpuHierarchy(handle, &cpuHierarchy) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetCpuHierarchy_v2(handle, &cpuHierarchyV2) == DCGM_ST_UNINITIALIZED);

                auto const output = capture.GetOutput();
                CHECK(output.find("Wrong version of the DCGM library is linked") != std::string::npos);
                CHECK(gpuCount == 4);
                CHECK(nextSinceTimestamp == 7);
                CHECK(outputGpuIds == 0);
            }
        }

        WHEN("the CPU hierarchy helper is called before initialization")
        {
            StderrCapture capture;
            dcgmCpuHierarchyOwnedCores_v1 ownedCores {};
            auto const result = dcgmCpuHierarchyCpuOwnsCore(7, &ownedCores);

            THEN("the helper reports that no core is owned and emits the same diagnostic")
            {
                CHECK(result == 0);
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }
    }
}

SCENARIO("dcgm_stub generated testing and injection wrappers preserve caller state when uninitialized")
{
    GIVEN("the static forwarding library has not loaded libdcgm")
    {
        auto handle = static_cast<dcgmHandle_t>(0x1234);

        WHEN("testing-only generated wrappers are called")
        {
            StderrCapture capture;
            dcgmInjectFieldValue_t injectedValue {};
            dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
            dcgmCreateFakeEntities_v2 fakeEntities {};
            dcgmSetNvLinkLinkState_v1 linkState {};
            dcgmChipArchitecture_t chipArchitecture = DCGM_CHIP_ARCH_UNKNOWN;
            dcgmVgpuDeviceAttributes_t vgpuDeviceAttributes {};
            dcgmVgpuInstanceAttributes_t vgpuInstanceAttributes {};

            THEN("each wrapper returns uninitialized without modifying visible output state")
            {
                CHECK(dcgmInjectFieldValue(handle, 1, &injectedValue) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetCacheManagerFieldInfo(handle, &fieldInfo) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmCreateFakeEntities(handle, &fakeEntities) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmInjectEntityFieldValue(handle, DCGM_FE_GPU, 1, &injectedValue) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmSetEntityNvLinkLinkState(handle, &linkState) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetGpuChipArchitecture(handle, 1, &chipArchitecture) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetVgpuDeviceAttributes(handle, 1, &vgpuDeviceAttributes) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetVgpuInstanceAttributes(handle, 2, &vgpuInstanceAttributes) == DCGM_ST_UNINITIALIZED);
                CHECK(chipArchitecture == DCGM_CHIP_ARCH_UNKNOWN);
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }

        WHEN("watch, health, and backend helper wrappers are called")
        {
            StderrCapture capture;
            dcgmHealthSetParams_v2 healthParams {};
            bool backendActive   = true;
            char backendName[16] = "unchanged";

            THEN("the failure contract is consistent across helper signatures")
            {
                CHECK(dcgmWatchFields(handle, 1, 2, 1000, 5.0, 3) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmUnwatchFields(handle, 1, 2) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmHealthSet_v2(handle, &healthParams) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmNvswitchGetBackend(handle, &backendActive, backendName, sizeof(backendName))
                      == DCGM_ST_UNINITIALIZED);
                CHECK(backendActive);
                CHECK(std::string(backendName) == "unchanged");
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }

#ifdef INJECTION_LIBRARY_AVAILABLE
        WHEN("NVML injection forwarding wrappers are called")
        {
            StderrCapture capture;
            injectNvmlVal_t extraKey {};
            injectNvmlRet_t injectedReturn {};
            injectNvmlFuncCallCounts_t callCounts {};

            THEN("they reject the calls before mutating caller-owned data")
            {
                CHECK(dcgmInjectEntityFieldValueToNvml(handle, DCGM_FE_GPU, 1, nullptr) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmCreateNvmlInjectionGpu(handle, 1) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmInjectNvmlDevice(handle, 1, "key", &extraKey, 1, &injectedReturn) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmInjectNvmlDeviceForFollowingCalls(handle, 1, "key", &extraKey, 1, &injectedReturn, 1)
                      == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmInjectedNvmlDeviceReset(handle, 1) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmGetNvmlInjectFuncCallCount(handle, &callCounts) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmResetNvmlInjectFuncCallCount(handle) == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmRemoveNvmlInjectedGpu(handle, "GPU-fake") == DCGM_ST_UNINITIALIZED);
                CHECK(dcgmRestoreNvmlInjectedGpu(handle, "GPU-fake") == DCGM_ST_UNINITIALIZED);
                CHECK(capture.GetOutput().find("Wrong version of the DCGM library is linked") != std::string::npos);
            }
        }
#endif
    }
}
