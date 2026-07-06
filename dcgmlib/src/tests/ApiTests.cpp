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
#include <catch2/catch_all.hpp>
#include <cstring>
#include <dcgm_agent.h>
#include <dcgm_agent_3x.h>
#include <dcgm_multinode_internal.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <dcgm_test_apis.h>
#include <set>
#include <sysmon/dcgm_sysmon_structs.h>

#include "DcgmStatus.h"
#include "config/dcgm_config_structs.h"


dcgmGpuTopologyLevel_t GetSlowestPath(dcgmTopology_t &topology);
dcgmReturn_t helperFillCpuHierarchyV1(dcgmCpuHierarchy_v1 *cpuHierarchy, dcgm_sysmon_msg_get_cpus_t const &sysmonMsg);
dcgmReturn_t helperFillCpuHierarchyV2(dcgmCpuHierarchy_v2 *cpuHierarchy, dcgm_sysmon_msg_get_cpus_t const &sysmonMsg);
dcgmReturn_t helperRunMnDiag(dcgmHandle_t dcgmHandle, dcgmRunMnDiag_v1 const *drmnd, dcgmMnDiagResponse_v1 *response);
dcgmReturn_t helperGetMultipleValuesForFieldFV1s(dcgmHandle_t pDcgmHandle,
                                                 dcgm_field_entity_group_t entityGroup,
                                                 dcgm_field_eid_t entityId,
                                                 unsigned int fieldId,
                                                 int *count,
                                                 long long startTs,
                                                 long long endTs,
                                                 dcgmOrder_t order,
                                                 dcgmFieldValue_v1 values[]);
int helperUpdateErrorCodes(dcgmStatus_t statusHandle, int numStatuses, dcgm_config_status_t *statuses);


TEST_CASE("DcgmApi: Test GetSlowestPath")
{
    dcgmTopology_t topology;
    topology.numElements     = 0;
    topology.element[0].path = DCGM_TOPOLOGY_NVLINK6;
    topology.element[1].path = DCGM_TOPOLOGY_NVLINK5;

    // Make sure num elements gates what is evaluated
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_UNINITIALIZED);

    topology.numElements = 2;
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_NVLINK5);

    topology.element[2].path = DCGM_TOPOLOGY_MULTIPLE;
    topology.numElements     = 3;
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_MULTIPLE);

    topology.element[3].path = DCGM_TOPOLOGY_HOSTBRIDGE;
    topology.numElements     = 4;
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_HOSTBRIDGE);

    topology.element[4].path = DCGM_TOPOLOGY_BOARD;
    topology.numElements     = 5;
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_HOSTBRIDGE);
}

extern "C" dcgmReturn_t dcgmPowerProfileIdToName(dcgmPowerProfileType_t id, char const **name);

TEST_CASE("DcgmApi: Test dcgmPowerProfileIdToName")
{
    std::set<std::string> names;

    for (unsigned int profileType = 0; profileType < DCGM_POWER_PROFILE_MAX; profileType++)
    {
        char const *namePtr = nullptr;

        REQUIRE(dcgmPowerProfileIdToName(static_cast<dcgmPowerProfileType_t>(profileType), &namePtr) == DCGM_ST_OK);

        REQUIRE(names.insert(std::string(namePtr)).second == true);
    }
}

TEST_CASE("FillCpuHierarchyV1: resets numCpus before copying")
{
    dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
    sysmonMsg.cpuCount                      = 2;
    sysmonMsg.cpus[0].cpuId                 = 3;
    sysmonMsg.cpus[1].cpuId                 = 7;
    sysmonMsg.cpus[0].ownedCores.bitmask[0] = 0x3;
    sysmonMsg.cpus[1].ownedCores.bitmask[0] = 0xC;

    dcgmCpuHierarchy_v1 hierarchy {};
    hierarchy.version = dcgmCpuHierarchy_version1;
    hierarchy.numCpus = 99;

    REQUIRE(helperFillCpuHierarchyV1(&hierarchy, sysmonMsg) == DCGM_ST_OK);
    CHECK(hierarchy.numCpus == 2);
    CHECK(hierarchy.cpus[0].cpuId == 3);
    CHECK(hierarchy.cpus[1].cpuId == 7);
    CHECK(hierarchy.cpus[0].ownedCores.bitmask[0] == 0x3);
    CHECK(hierarchy.cpus[1].ownedCores.bitmask[0] == 0xC);
}

TEST_CASE("FillCpuHierarchyV1: clamps copied CPUs to destination capacity")
{
    dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
    sysmonMsg.cpuCount = DCGM_MAX_NUM_CPUS + 2;
    for (unsigned int i = 0; i < DCGM_MAX_NUM_CPUS; i++)
    {
        sysmonMsg.cpus[i].cpuId = i + 10;
    }

    dcgmCpuHierarchy_v1 hierarchy {};
    hierarchy.version = dcgmCpuHierarchy_version1;

    REQUIRE(helperFillCpuHierarchyV1(&hierarchy, sysmonMsg) == DCGM_ST_OK);
    CHECK(hierarchy.numCpus == DCGM_MAX_NUM_CPUS);
    CHECK(hierarchy.cpus[0].cpuId == 10);
    CHECK(hierarchy.cpus[DCGM_MAX_NUM_CPUS - 1].cpuId == (DCGM_MAX_NUM_CPUS - 1) + 10);
}

TEST_CASE("FillCpuHierarchyV2: resets numCpus before copying")
{
    dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
    sysmonMsg.cpuCount                      = 2;
    sysmonMsg.cpus[0].cpuId                 = 5;
    sysmonMsg.cpus[1].cpuId                 = 9;
    sysmonMsg.cpus[0].ownedCores.bitmask[0] = 0x30;
    sysmonMsg.cpus[1].ownedCores.bitmask[0] = 0xC0;
    std::strncpy(sysmonMsg.cpus[0].serial, "cpu-five", sizeof(sysmonMsg.cpus[0].serial) - 1);
    std::strncpy(sysmonMsg.cpus[1].serial, "cpu-nine", sizeof(sysmonMsg.cpus[1].serial) - 1);

    dcgmCpuHierarchy_v2 hierarchy {};
    hierarchy.version = dcgmCpuHierarchy_version2;
    hierarchy.numCpus = 99;

    REQUIRE(helperFillCpuHierarchyV2(&hierarchy, sysmonMsg) == DCGM_ST_OK);
    CHECK(hierarchy.numCpus == 2);
    CHECK(hierarchy.cpus[0].cpuId == 5);
    CHECK(hierarchy.cpus[1].cpuId == 9);
    CHECK(hierarchy.cpus[0].ownedCores.bitmask[0] == 0x30);
    CHECK(hierarchy.cpus[1].ownedCores.bitmask[0] == 0xC0);
    CHECK(std::string(hierarchy.cpus[0].serial) == "cpu-five");
    CHECK(std::string(hierarchy.cpus[1].serial) == "cpu-nine");
}

TEST_CASE("FillCpuHierarchyV2: clamps copied CPUs to destination capacity")
{
    dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
    sysmonMsg.cpuCount = DCGM_MAX_NUM_CPUS + 1;
    for (unsigned int i = 0; i < DCGM_MAX_NUM_CPUS; i++)
    {
        sysmonMsg.cpus[i].cpuId = i + 20;
        std::strncpy(sysmonMsg.cpus[i].serial, "serial", sizeof(sysmonMsg.cpus[i].serial) - 1);
    }

    dcgmCpuHierarchy_v2 hierarchy {};
    hierarchy.version = dcgmCpuHierarchy_version2;

    REQUIRE(helperFillCpuHierarchyV2(&hierarchy, sysmonMsg) == DCGM_ST_OK);
    CHECK(hierarchy.numCpus == DCGM_MAX_NUM_CPUS);
    CHECK(hierarchy.cpus[0].cpuId == 20);
    CHECK(hierarchy.cpus[DCGM_MAX_NUM_CPUS - 1].cpuId == (DCGM_MAX_NUM_CPUS - 1) + 20);
    CHECK(std::string(hierarchy.cpus[0].serial) == "serial");
}

TEST_CASE("helperRunMnDiag: null pointer input validation")
{
    dcgmRunMnDiag_v1 params {};
    params.version = dcgmRunMnDiag_version1;

    dcgmMnDiagResponse_v1 resp {};
    resp.version = dcgmMnDiagResponse_version1;

    SECTION("null drmnd returns DCGM_ST_BADPARAM")
    {
        REQUIRE(helperRunMnDiag((dcgmHandle_t)0, nullptr, &resp) == DCGM_ST_BADPARAM);
    }

    SECTION("null response returns DCGM_ST_BADPARAM")
    {
        REQUIRE(helperRunMnDiag((dcgmHandle_t)0, &params, nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("both null returns DCGM_ST_BADPARAM")
    {
        REQUIRE(helperRunMnDiag((dcgmHandle_t)0, nullptr, nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("unexpected response version returns generic error before dispatch")
    {
        resp.version = 0;
        REQUIRE(helperRunMnDiag((dcgmHandle_t)0, &params, &resp) == DCGM_ST_GENERIC_ERROR);
    }
}

TEST_CASE("helperGetMultipleValuesForFieldFV1s: input validation")
{
    dcgmFieldValue_v1 values[10] = {};
    int count                    = 10;

    SECTION("null count returns BADPARAM")
    {
        REQUIRE(helperGetMultipleValuesForFieldFV1s(
                    (dcgmHandle_t)0, DCGM_FE_GPU, 0, 1, nullptr, 0, 0, DCGM_ORDER_ASCENDING, values)
                == DCGM_ST_BADPARAM);
    }

    SECTION("non-positive count returns BADPARAM")
    {
        count = GENERATE(0, -1);
        REQUIRE(helperGetMultipleValuesForFieldFV1s(
                    (dcgmHandle_t)0, DCGM_FE_GPU, 0, 1, &count, 0, 0, DCGM_ORDER_ASCENDING, values)
                == DCGM_ST_BADPARAM);
    }

    SECTION("null values returns BADPARAM")
    {
        REQUIRE(helperGetMultipleValuesForFieldFV1s(
                    (dcgmHandle_t)0, DCGM_FE_GPU, 0, 1, &count, 0, 0, DCGM_ORDER_ASCENDING, nullptr)
                == DCGM_ST_BADPARAM);
    }

    SECTION("zero field id returns BADPARAM before dispatch")
    {
        REQUIRE(helperGetMultipleValuesForFieldFV1s(
                    (dcgmHandle_t)0, DCGM_FE_GPU, 0, 0, &count, 0, 0, DCGM_ORDER_ASCENDING, values)
                == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("helperUpdateErrorCodes: invalid inputs are rejected")
{
    dcgm_config_status_t statuses[1] {};

    CHECK(helperUpdateErrorCodes(0, 1, statuses) == -1);
    CHECK(helperUpdateErrorCodes(static_cast<dcgmStatus_t>(1), 0, statuses) == -1);
    CHECK(helperUpdateErrorCodes(static_cast<dcgmStatus_t>(1), 1, nullptr) == -1);
}

TEST_CASE("helperUpdateErrorCodes: enqueues status entries in order")
{
    DcgmStatus status;
    dcgm_config_status_t statuses[2] {};
    statuses[0].gpuId     = 1;
    statuses[0].fieldId   = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
    statuses[0].errorCode = DCGM_ST_BADPARAM;
    statuses[1].gpuId     = 2;
    statuses[1].fieldId   = DCGM_FI_DEV_BOARD_POWER_WATTS;
    statuses[1].errorCode = DCGM_ST_NOT_SUPPORTED;

    REQUIRE(helperUpdateErrorCodes(reinterpret_cast<dcgmStatus_t>(&status), 2, statuses) == 0);
    REQUIRE(status.GetNumErrors() == 2);

    dcgmErrorInfo_t popped {};
    REQUIRE(status.Dequeue(&popped) == 0);
    CHECK(popped.gpuId == 1);
    CHECK(popped.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    CHECK(popped.status == DCGM_ST_BADPARAM);

    REQUIRE(status.Dequeue(&popped) == 0);
    CHECK(popped.gpuId == 2);
    CHECK(popped.fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
    CHECK(popped.status == DCGM_ST_NOT_SUPPORTED);
    CHECK(status.IsEmpty());
}

TEST_CASE("DcgmApi public entry points return uninitialized before dcgmInit")
{
    dcgmHandle_t handle = 0;
    dcgmGpuGrp_t groupId {};
    dcgmFieldGrp_t fieldGroupId {};
    dcgmStatus_t statusHandle {};
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] {};
    int count {};
    unsigned int uintValue {};
    unsigned short fieldIds[] = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
    dcgmFieldValue_v1 fieldValues[2] {};
    dcgmFieldValue_v2 fieldValuesV2[2] {};
    dcgmGroupEntityPair_t entities[] = { { DCGM_FE_GPU, 0 } };
    long long nextSince {};
    uint64_t outputGpuIds {};
    char backendName[64] {};
    bool backendActive {};
    char jobId[64] {};

    dcgmDeviceAttributes_t deviceAttributes {};
    dcgmVgpuDeviceAttributes_t vgpuDeviceAttributes {};
    dcgmVgpuInstanceAttributes_t vgpuInstanceAttributes {};
    dcgmGroupInfo_t groupInfo {};
    dcgmConfig_t config {};
    dcgmVgpuConfig_t vgpuConfig {};
    dcgmPolicy_t policy {};
    dcgmSetNvLinkLinkState_v1 linkState {};
    dcgmFieldGroupInfo_t fieldGroupInfo {};
    dcgmAllFieldGroup_t allFieldGroups {};
    dcgmHealthSystems_t healthSystems {};
    dcgmHealthSetParams_v2 healthSetParams {};
    dcgmHealthResponse_t healthResponse {};
    dcgmRunDiag_v10 runDiag {};
    dcgmDiagResponse_v12 diagResponse {};
    dcgmRunMnDiag_v1 runMnDiag {};
    dcgmMnDiagResponse_v1 mnDiagResponse {};
    dcgmMultinodeRequest_t multinodeRequest {};
    dcgmPidInfo_t pidInfo {};
    dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo {};
    dcgmDeviceWorkloadPowerProfilesStatus_v1 profilesStatus {};
    dcgmDeviceTopology_v2 deviceTopology {};
    dcgmDeviceTopology_v1 deviceTopologyV1 {};
    dcgmGroupTopology_v2 groupTopology {};
    dcgmGroupTopology_v1 groupTopologyV1 {};
    dcgmJobInfo_t jobInfo {};
    dcgmIntrospectCpuUtil_t cpuUtil {};
    dcgmIntrospectMemory_t memoryInfo {};
    dcgmFieldSummaryRequest_t fieldSummary {};
    dcgmModuleGetStatuses_t moduleStatuses {};
    dcgmProfGetMetricGroups_t metricGroups {};
    dcgmVersionInfo_t versionInfo {};
    dcgmHostengineHealth_t hostengineHealth {};
    dcgmInjectFieldValue_t injectFieldValue {};
    dcgmCreateFakeEntities_v2 fakeEntities {};
    dcgmChipArchitecture_t chipArchitecture {};
    dcgmMigHierarchy_v2 migHierarchy {};
    dcgmCreateMigEntity_t createMig {};
    dcgmDeleteMigEntity_t deleteMig {};
    dcgmNvLinkStatus_v5 nvLinkStatus {};
    dcgmNvLinkStatus_v4 nvLinkStatusV4 {};
    dcgmNvLinkP2PStatus_v1 nvLinkP2PStatus {};
    dcgmCpuHierarchy_v1 cpuHierarchy {};
    dcgmCpuHierarchy_v2 cpuHierarchyV2 {};
    dcgmWorkloadPowerProfile_t workloadPowerProfile {};
    dcgmEnvVarInfo_t envVarInfo {};
    dcgmModulesReloadable_v1 modulesReloadable {};

    dcgmReturn_t shutdownResult = dcgmShutdown();
    REQUIRE((shutdownResult == DCGM_ST_OK || shutdownResult == DCGM_ST_UNINITIALIZED));

    SECTION("GIVEN uninitialized DCGM WHEN group and field APIs are called THEN they fail before dispatch")
    {
        CHECK(dcgmGetAllDevices(handle, gpuIds, &count) != DCGM_ST_OK);
        CHECK(dcgmGetAllSupportedDevices(handle, gpuIds, &count) != DCGM_ST_OK);
        CHECK(dcgmGetDeviceAttributes(handle, 0, &deviceAttributes) != DCGM_ST_OK);
        CHECK(dcgmGetVgpuDeviceAttributes(handle, 0, &vgpuDeviceAttributes) != DCGM_ST_OK);
        CHECK(dcgmGetVgpuInstanceAttributes(handle, 0, &vgpuInstanceAttributes) != DCGM_ST_OK);
        CHECK(dcgmGroupCreate(handle, DCGM_GROUP_DEFAULT, "group", &groupId) != DCGM_ST_OK);
        CHECK(dcgmGroupDestroy(handle, groupId) != DCGM_ST_OK);
        CHECK(dcgmGroupAddDevice(handle, groupId, 0) != DCGM_ST_OK);
        CHECK(dcgmGroupAddEntity(handle, groupId, DCGM_FE_GPU, 0) != DCGM_ST_OK);
        CHECK(dcgmGroupRemoveDevice(handle, groupId, 0) != DCGM_ST_OK);
        CHECK(dcgmGroupRemoveEntity(handle, groupId, DCGM_FE_GPU, 0) != DCGM_ST_OK);
        CHECK(dcgmGroupGetInfo(handle, groupId, &groupInfo) != DCGM_ST_OK);
        CHECK(dcgmGroupGetAllIds(handle, &groupId, &uintValue) != DCGM_ST_OK);
        CHECK(dcgmFieldGroupCreate(handle, 1, fieldIds, "fields", &fieldGroupId) != DCGM_ST_OK);
        CHECK(dcgmFieldGroupDestroy(handle, fieldGroupId) != DCGM_ST_OK);
        CHECK(dcgmFieldGroupGetInfo(handle, &fieldGroupInfo) != DCGM_ST_OK);
        CHECK(dcgmFieldGroupGetAll(handle, &allFieldGroups) != DCGM_ST_OK);
    }

    SECTION("GIVEN uninitialized DCGM WHEN value and watch APIs are called THEN they fail before dispatch")
    {
        CHECK(dcgmGetLatestValuesForFields(handle, 0, fieldIds, 1, fieldValues) != DCGM_ST_OK);
        CHECK(dcgmEntityGetLatestValues(handle, DCGM_FE_GPU, 0, fieldIds, 1, fieldValues) != DCGM_ST_OK);
        CHECK(dcgmGetMultipleValuesForField(
                  handle, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &count, 0, 0, DCGM_ORDER_ASCENDING, fieldValues)
              != DCGM_ST_OK);
        CHECK(dcgmWatchFieldValue(handle, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000000, 10.0, 10) != DCGM_ST_OK);
        CHECK(dcgmUnwatchFieldValue(handle, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0) != DCGM_ST_OK);
        CHECK(dcgmUpdateAllFields(handle, 0) != DCGM_ST_OK);
        CHECK(dcgmGetFieldValuesSince(handle, groupId, 0, fieldIds, 1, &nextSince, nullptr, nullptr) != DCGM_ST_OK);
        CHECK(dcgmGetValuesSince(handle, groupId, fieldGroupId, 0, &nextSince, nullptr, nullptr) != DCGM_ST_OK);
        CHECK(dcgmGetValuesSince_v2(handle, groupId, fieldGroupId, 0, &nextSince, nullptr, nullptr) != DCGM_ST_OK);
        CHECK(dcgmGetLatestValues(handle, groupId, fieldGroupId, nullptr, nullptr) != DCGM_ST_OK);
        CHECK(dcgmGetLatestValues_v2(handle, groupId, fieldGroupId, nullptr, nullptr) != DCGM_ST_OK);
        CHECK(dcgmEntitiesGetLatestValues(handle, entities, 1, fieldIds, 1, 0, fieldValuesV2) != DCGM_ST_OK);
        CHECK(dcgmWatchFields(handle, groupId, fieldGroupId, 1000000, 10.0, 10) != DCGM_ST_OK);
        CHECK(dcgmUnwatchFields(handle, groupId, fieldGroupId) != DCGM_ST_OK);
    }

    SECTION("GIVEN uninitialized DCGM WHEN policy, config, and health APIs are called THEN they fail before dispatch")
    {
        CHECK(dcgmConfigSet(handle, groupId, &config, statusHandle) != DCGM_ST_OK);
        CHECK(dcgmVgpuConfigSet(handle, groupId, &vgpuConfig, statusHandle) != DCGM_ST_OK);
        CHECK(dcgmConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, &config, statusHandle) != DCGM_ST_OK);
        CHECK(dcgmVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, &vgpuConfig, statusHandle)
              != DCGM_ST_OK);
        CHECK(dcgmConfigEnforce(handle, groupId, statusHandle) != DCGM_ST_OK);
        CHECK(dcgmVgpuConfigEnforce(handle, groupId, statusHandle) != DCGM_ST_OK);
        CHECK(dcgmPolicySet(handle, groupId, &policy, statusHandle) != DCGM_ST_OK);
        CHECK(dcgmPolicyGet(handle, groupId, 1, &policy, statusHandle) != DCGM_ST_OK);
        CHECK(dcgmPolicyRegister_v2(handle, groupId, DCGM_POLICY_COND_DBE, nullptr, 0) != DCGM_ST_OK);
        CHECK(dcgmPolicyUnregister(handle, groupId, DCGM_POLICY_COND_DBE) != DCGM_ST_OK);
        CHECK(dcgmHealthSet(handle, groupId, DCGM_HEALTH_WATCH_ALL) != DCGM_ST_OK);
        CHECK(dcgmHealthSet_v2(handle, &healthSetParams) != DCGM_ST_OK);
        CHECK(dcgmHealthGet(handle, groupId, &healthSystems) != DCGM_ST_OK);
        CHECK(dcgmHealthCheck(handle, groupId, &healthResponse) != DCGM_ST_OK);
    }

    SECTION("GIVEN uninitialized DCGM WHEN diag and topology APIs are called THEN they fail before dispatch")
    {
        CHECK(dcgmActionValidate_v2(handle, &runDiag, &diagResponse) != DCGM_ST_OK);
        CHECK(dcgmActionValidate(handle, groupId, DCGM_POLICY_VALID_NONE, &diagResponse) != DCGM_ST_OK);
        CHECK(dcgmRunDiagnostic(handle, groupId, DCGM_DIAG_LVL_SHORT, &diagResponse) != DCGM_ST_OK);
        CHECK(dcgmStopDiagnostic(handle) != DCGM_ST_OK);
        CHECK(dcgmRunMnDiagnostic(handle, &runMnDiag, &mnDiagResponse) != DCGM_ST_OK);
        CHECK(dcgmStopMnDiagnostic(handle) != DCGM_ST_OK);
        CHECK(dcgmMultinodeRequest(handle, &multinodeRequest) != DCGM_ST_OK);
        CHECK(dcgmGetDeviceTopology(handle, 0, &deviceTopology) != DCGM_ST_OK);
        CHECK(dcgmGetDeviceTopology(handle, 0, reinterpret_cast<dcgmDeviceTopology_v2 *>(&deviceTopologyV1))
              != DCGM_ST_OK);
        CHECK(dcgmGetGroupTopology(handle, groupId, &groupTopology) != DCGM_ST_OK);
        CHECK(dcgmGetGroupTopology(handle, groupId, reinterpret_cast<dcgmGroupTopology_v2 *>(&groupTopologyV1))
              != DCGM_ST_OK);
        CHECK(dcgmSelectGpusByTopology(handle, 1, 1, &outputGpuIds, 0) != DCGM_ST_OK);
        CHECK(dcgmGetFieldSummary(handle, &fieldSummary) != DCGM_ST_OK);
        CHECK(dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, gpuIds, &count, 0) != DCGM_ST_OK);
        CHECK(dcgmGetGpuChipArchitecture(handle, 0, &chipArchitecture) != DCGM_ST_OK);
        CHECK(dcgmGetGpuInstanceHierarchy(handle, &migHierarchy) != DCGM_ST_OK);
        CHECK(dcgmCreateMigEntity(handle, &createMig) != DCGM_ST_OK);
        CHECK(dcgmDeleteMigEntity(handle, &deleteMig) != DCGM_ST_OK);
        CHECK(dcgmGetNvLinkLinkStatus(handle, &nvLinkStatus) != DCGM_ST_OK);
        CHECK(dcgmGetNvLinkLinkStatus(handle, reinterpret_cast<dcgmNvLinkStatus_v5 *>(&nvLinkStatusV4)) != DCGM_ST_OK);
        CHECK(dcgmGetNvLinkP2PStatus(handle, &nvLinkP2PStatus) != DCGM_ST_OK);
    }

    SECTION(
        "GIVEN uninitialized DCGM WHEN job, introspection, and hostengine APIs are called THEN they fail before dispatch")
    {
        CHECK(dcgmWatchPidFields(handle, groupId, 1000000, 10.0, 10) != DCGM_ST_OK);
        CHECK(dcgmGetPidInfo(handle, groupId, &pidInfo) != DCGM_ST_OK);
        CHECK(dcgmGetDeviceWorkloadPowerProfileInfo(handle, 0, &profilesInfo, &profilesStatus) != DCGM_ST_OK);
        CHECK(dcgmWatchJobFields(handle, groupId, 1000000, 10.0, 10) != DCGM_ST_OK);
        CHECK(dcgmJobStartStats(handle, groupId, jobId) != DCGM_ST_OK);
        CHECK(dcgmJobStopStats(handle, jobId) != DCGM_ST_OK);
        CHECK(dcgmJobGetStats(handle, jobId, &jobInfo) != DCGM_ST_OK);
        CHECK(dcgmJobRemove(handle, jobId) != DCGM_ST_OK);
        CHECK(dcgmJobRemoveAll(handle) != DCGM_ST_OK);
        CHECK(dcgmIntrospectGetHostengineCpuUtilization(handle, &cpuUtil, 0) != DCGM_ST_OK);
        CHECK(dcgmIntrospectGetHostengineMemoryUsage(handle, &memoryInfo, 0) != DCGM_ST_OK);
        CHECK(dcgmModuleDenylist(handle, DcgmModuleIdCore) != DCGM_ST_OK);
        CHECK(dcgmModuleGetStatuses(handle, &moduleStatuses) != DCGM_ST_OK);
        CHECK(dcgmProfGetSupportedMetricGroups(handle, &metricGroups) != DCGM_ST_OK);
        CHECK(dcgmProfPause(handle) != DCGM_ST_OK);
        CHECK(dcgmProfResume(handle) != DCGM_ST_OK);
        CHECK(dcgmHostengineVersionInfo(handle, &versionInfo) != DCGM_ST_OK);
        CHECK(dcgmHostengineIsHealthy(handle, &hostengineHealth) != DCGM_ST_OK);
    }

    SECTION("GIVEN uninitialized DCGM WHEN injection and misc APIs are called THEN they fail before dispatch")
    {
        CHECK(dcgmInjectFieldValue(handle, 0, &injectFieldValue) != DCGM_ST_OK);
        CHECK(dcgmGetCacheManagerFieldInfo(handle, nullptr) != DCGM_ST_OK);
        CHECK(dcgmGetGpuStatus(handle, 0, nullptr) != DCGM_ST_OK);
        CHECK(dcgmCreateFakeEntities(handle, &fakeEntities) != DCGM_ST_OK);
        CHECK(dcgmSetEntityNvLinkLinkState(handle, &linkState) != DCGM_ST_OK);
        CHECK(dcgmPauseTelemetryForDiag(handle) != DCGM_ST_OK);
        CHECK(dcgmResumeTelemetryForDiag(handle) != DCGM_ST_OK);
        CHECK(dcgmNvswitchGetBackend(handle, &backendActive, backendName, sizeof(backendName)) != DCGM_ST_OK);
        CHECK(dcgmConfigSetWorkloadPowerProfile(handle, &workloadPowerProfile) != DCGM_ST_OK);
        CHECK(dcgmDiagSendHeartbeat(handle) != DCGM_ST_OK);
        CHECK(dcgmHostengineEnvironmentVariableInfo(handle, &envVarInfo) != DCGM_ST_OK);
        CHECK(dcgmAttachDriver(handle) != DCGM_ST_OK);
        CHECK(dcgmDetachDriver(handle) != DCGM_ST_OK);
        CHECK(dcgmEmptyCache(handle) != DCGM_ST_OK);
        CHECK(dcgmMarkModulesReloadable(handle, &modulesReloadable) != DCGM_ST_OK);
        CHECK(dcgmGetCpuHierarchy(handle, &cpuHierarchy) != DCGM_ST_OK);
        CHECK(dcgmGetCpuHierarchy_v2(handle, &cpuHierarchyV2) != DCGM_ST_OK);
    }
}

TEST_CASE("DcgmApi public entry points validate parameters after dcgmInit")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);

    dcgmHandle_t handle = 0;
    dcgmGpuGrp_t groupId {};
    dcgmFieldGrp_t fieldGroupId {};
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] {};
    int count = 1;
    unsigned int uintValue {};
    unsigned short fieldIds[] = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
    dcgmFieldValue_v1 fieldValues[2] {};
    dcgmFieldValue_v2 fieldValuesV2[2] {};
    dcgmGroupEntityPair_t entities[] = { { DCGM_FE_GPU, 0 } };
    long long nextSince {};
    uint64_t outputGpuIds {};
    dcgmDeviceAttributes_t deviceAttributes {};
    dcgmGroupInfo_t groupInfo {};
    dcgmFieldGroupInfo_t fieldGroupInfo {};
    dcgmAllFieldGroup_t allFieldGroups {};
    dcgmHealthSystems_t healthSystems {};
    dcgmHealthResponse_t healthResponse {};
    dcgmRunDiag_v10 runDiag {};
    dcgmDiagResponse_v12 diagResponse {};
    dcgmRunMnDiag_v1 runMnDiag {};
    dcgmMnDiagResponse_v1 mnDiagResponse {};
    dcgmMultinodeRequest_t multinodeRequest {};
    dcgmFieldSummaryRequest_t fieldSummary {};
    dcgmVersionInfo_t versionInfo {};
    dcgmHostengineHealth_t hostengineHealth {};
    dcgmInjectFieldValue_t injectFieldValue {};
    dcgmCreateFakeEntities_v2 fakeEntities {};
    dcgmChipArchitecture_t chipArchitecture {};
    dcgmMigHierarchy_v2 migHierarchy {};
    dcgmNvLinkStatus_v5 nvLinkStatus {};
    dcgmNvLinkP2PStatus_v1 nvLinkP2PStatus {};
    dcgmCpuHierarchy_v1 cpuHierarchy {};
    dcgmCpuHierarchy_v2 cpuHierarchyV2 {};

    SECTION("GIVEN null output pointers WHEN group and field APIs are called THEN bad parameters are returned")
    {
        CHECK(dcgmGetAllDevices(handle, nullptr, &count) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetAllDevices(handle, gpuIds, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetAllSupportedDevices(handle, nullptr, &count) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetDeviceAttributes(handle, 0, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGroupCreate(handle, DCGM_GROUP_DEFAULT, nullptr, &groupId) == DCGM_ST_BADPARAM);
        CHECK(dcgmGroupCreate(handle, DCGM_GROUP_DEFAULT, "group", nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGroupGetInfo(handle, groupId, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGroupGetAllIds(handle, nullptr, &uintValue) == DCGM_ST_BADPARAM);
        CHECK(dcgmGroupGetAllIds(handle, &groupId, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmFieldGroupCreate(handle, 1, nullptr, "fields", &fieldGroupId) == DCGM_ST_BADPARAM);
        CHECK(dcgmFieldGroupCreate(handle, 1, fieldIds, nullptr, &fieldGroupId) == DCGM_ST_BADPARAM);
        CHECK(dcgmFieldGroupCreate(handle, 1, fieldIds, "fields", nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmFieldGroupGetInfo(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmFieldGroupGetAll(handle, nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN invalid value request buffers WHEN value APIs are called THEN bad parameters are returned")
    {
        CHECK(dcgmGetLatestValuesForFields(handle, 0, nullptr, 1, fieldValues) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetLatestValuesForFields(handle, 0, fieldIds, 1, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmEntityGetLatestValues(handle, DCGM_FE_GPU, 0, nullptr, 1, fieldValues) == DCGM_ST_BADPARAM);
        CHECK(dcgmEntityGetLatestValues(handle, DCGM_FE_GPU, 0, fieldIds, 1, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetMultipleValuesForField(
                  handle, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr, 0, 0, DCGM_ORDER_ASCENDING, fieldValues)
              == DCGM_ST_BADPARAM);
        CHECK(dcgmGetMultipleValuesForField(
                  handle, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &count, 0, 0, DCGM_ORDER_ASCENDING, nullptr)
              == DCGM_ST_BADPARAM);
        CHECK(dcgmGetFieldValuesSince(handle, groupId, 0, nullptr, 1, &nextSince, nullptr, nullptr)
              == DCGM_ST_BADPARAM);
        CHECK(dcgmGetFieldValuesSince(handle, groupId, 0, fieldIds, 1, nullptr, nullptr, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmEntitiesGetLatestValues(handle, nullptr, 1, fieldIds, 1, 0, fieldValuesV2) == DCGM_ST_BADPARAM);
        CHECK(dcgmEntitiesGetLatestValues(handle, entities, 1, nullptr, 1, 0, fieldValuesV2) == DCGM_ST_BADPARAM);
        CHECK(dcgmEntitiesGetLatestValues(handle, entities, 1, fieldIds, 1, 0, nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN null control payloads WHEN module APIs are called THEN bad parameters are returned")
    {
        CHECK(dcgmHealthGet(handle, groupId, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmHealthCheck(handle, groupId, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmActionValidate_v2(handle, nullptr, &diagResponse) == DCGM_ST_BADPARAM);
        CHECK(dcgmActionValidate_v2(handle, &runDiag, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmRunMnDiagnostic(handle, nullptr, &mnDiagResponse) == DCGM_ST_BADPARAM);
        CHECK(dcgmRunMnDiagnostic(handle, &runMnDiag, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmMultinodeRequest(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetDeviceTopology(handle, 0, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetGroupTopology(handle, groupId, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmSelectGpusByTopology(handle, 1, 1, nullptr, 0) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetFieldSummary(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, nullptr, &count, 0) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, gpuIds, nullptr, 0) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetGpuChipArchitecture(handle, 0, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetGpuInstanceHierarchy(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetNvLinkLinkStatus(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetNvLinkP2PStatus(handle, nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN null misc payloads WHEN misc APIs are called THEN bad parameters are returned")
    {
        CHECK(dcgmHostengineVersionInfo(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmHostengineIsHealthy(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmInjectFieldValue(handle, 0, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetCpuHierarchy(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetCpuHierarchy_v2(handle, nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN status handles WHEN lifecycle APIs are called THEN parameters and state are validated")
    {
        dcgmStatus_t statusHandle {};
        dcgmErrorInfo_t errorInfo {};
        unsigned int statusCount = 7;

        CHECK(dcgmStatusCreate(nullptr) == DCGM_ST_BADPARAM);
        REQUIRE(dcgmStatusCreate(&statusHandle) == DCGM_ST_OK);
        REQUIRE(statusHandle != 0);

        CHECK(dcgmStatusGetCount(0, &statusCount) == DCGM_ST_BADPARAM);
        CHECK(dcgmStatusGetCount(statusHandle, nullptr) == DCGM_ST_BADPARAM);
        REQUIRE(dcgmStatusGetCount(statusHandle, &statusCount) == DCGM_ST_OK);
        CHECK(statusCount == 0);

        CHECK(dcgmStatusPopError(0, &errorInfo) == DCGM_ST_BADPARAM);
        CHECK(dcgmStatusPopError(statusHandle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmStatusPopError(statusHandle, &errorInfo) == DCGM_ST_NO_DATA);

        CHECK(dcgmStatusClear(0) == DCGM_ST_BADPARAM);
        CHECK(dcgmStatusClear(statusHandle) == DCGM_ST_OK);

        CHECK(dcgmStatusDestroy(0) == DCGM_ST_BADPARAM);
        CHECK(dcgmStatusDestroy(statusHandle) == DCGM_ST_OK);
    }

    SECTION("GIVEN enum name helpers WHEN called THEN names and bad parameters are handled")
    {
        char const *name = nullptr;

        CHECK(dcgmModuleIdToName(DcgmModuleIdCore, nullptr) == DCGM_ST_BADPARAM);
        REQUIRE(dcgmModuleIdToName(DcgmModuleIdCore, &name) == DCGM_ST_OK);
        CHECK(std::strcmp(name, "Core") == 0);

        name = nullptr;
        CHECK(dcgmModuleIdToName(DcgmModuleIdCount, &name) == DCGM_ST_BADPARAM);

        CHECK(dcgmPowerProfileIdToName(DCGM_POWER_PROFILE_MAX, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmPowerProfileIdToName(DCGM_POWER_PROFILE_MAX, &name) == DCGM_ST_BADPARAM);
    }

    CHECK(dcgmShutdown() == DCGM_ST_OK);
    (void)deviceAttributes;
    (void)groupInfo;
    (void)fieldGroupInfo;
    (void)allFieldGroups;
    (void)healthSystems;
    (void)healthResponse;
    (void)multinodeRequest;
    (void)fieldSummary;
    (void)versionInfo;
    (void)hostengineHealth;
    (void)injectFieldValue;
    (void)fakeEntities;
    (void)chipArchitecture;
    (void)migHierarchy;
    (void)nvLinkStatus;
    (void)nvLinkP2PStatus;
    (void)cpuHierarchy;
    (void)cpuHierarchyV2;
    (void)outputGpuIds;
}
