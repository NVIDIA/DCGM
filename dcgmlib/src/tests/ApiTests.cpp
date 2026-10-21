/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <Defer.hpp>
#include <array>
#include <catch2/catch_all.hpp>
#include <cstring>
#include <dcgm_agent.h>
#include <dcgm_agent_3x.h>
#include <dcgm_core_structs.h>
#include <dcgm_multinode_internal.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <dcgm_test_apis.h>
#include <diag/dcgm_diag_structs.h>
#include <health/dcgm_health_structs.h>
#include <introspect/dcgm_introspect_structs.h>
#include <mndiag/dcgm_mndiag_structs.hpp>
#include <profiling/dcgm_profiling_structs.h>
#include <set>
#include <string>
#include <sysmon/dcgm_sysmon_structs.h>
#include <utility>

#include "DcgmFvBuffer.h"
#include "DcgmStatus.h"
#include "DcgmStringHelpers.h"
#include "UniquePtrUtil.h"
#include "config/dcgm_config_structs.h"
#include <policy/dcgm_policy_structs.h>


dcgmGpuTopologyLevel_t GetSlowestPath(dcgmTopology_t &topology);
dcgmReturn_t helperFillCpuHierarchyV1(dcgmCpuHierarchy_v1 *cpuHierarchy, dcgm_sysmon_msg_get_cpus_t const &sysmonMsg);
dcgmReturn_t helperFillCpuHierarchyV2(dcgmCpuHierarchy_v2 *cpuHierarchy, dcgm_sysmon_msg_get_cpus_t const &sysmonMsg);
dcgmReturn_t helperRunMnDiag(dcgmHandle_t dcgmHandle, dcgmRunMnDiag_v2 const *drmnd, dcgmMnDiagResponse_v2 *response);
void UpgradeMnDiagParams_v1_to_v2(dcgmRunMnDiag_v2 &dst, dcgmRunMnDiag_v1 const &src);
void DowngradeMnDiagResponse_v2_to_v1(dcgmMnDiagResponse_v1 &dst, dcgmMnDiagResponse_v2 const &src);
dcgmReturn_t helperGetCpuHierarchySysmonMsg(dcgmHandle_t dcgmHandle, dcgm_sysmon_msg_get_cpus_t &sysmonMsg);
dcgmReturn_t tsapiGetCpuHierarchy(dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v1 *cpuHierarchy);
dcgmReturn_t tsapiGetCpuHierarchy_v2(dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v2 *cpuHierarchy);
dcgmReturn_t helperGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t grpId, unsigned int gpuId);
dcgmReturn_t sendClientLogin(dcgmHandle_t dcgmHandle, dcgmConnectV3Params_t *connectParams);
dcgmReturn_t helperGroupRemoveDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t grpId, unsigned int gpuId);
dcgmReturn_t tsapiGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId);
dcgmReturn_t tsapiGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId);
dcgmReturn_t helperRunMnDiag(dcgmHandle_t dcgmHandle, dcgmRunMnDiag_v1 const *drmnd, dcgmMnDiagResponse_v1 *response);
dcgmReturn_t helperJobStatCmd(dcgmHandle_t pDcgmHandle,
                              unsigned int groupId,
                              const char JobId[64],
                              unsigned int jobCmd);
dcgmReturn_t helperGroupGetInfo(dcgmHandle_t pDcgmHandle,
                                dcgmGpuGrp_t groupId,
                                dcgmGroupInfo_t *groupInfo,
                                long long *endQueryTimestamp);
dcgmReturn_t helperWatchFieldValue(dcgmHandle_t pDcgmHandle,
                                   int gpuId,
                                   unsigned short fieldId,
                                   long long updateFreq,
                                   double maxKeepAge,
                                   int maxKeepSamples);
dcgmReturn_t helperUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate);
dcgmReturn_t helperDeviceGetAttributes(dcgmHandle_t pDcgmHandle, int gpuId, dcgmDeviceAttributes_t *pDcgmDeviceAttr);
dcgmReturn_t helperVgpuDeviceGetAttributes(dcgmHandle_t pDcgmHandle,
                                           int gpuId,
                                           dcgmVgpuDeviceAttributes_t *pDcgmVgpuDeviceAttr);
dcgmReturn_t helperVgpuInstanceGetAttributes(dcgmHandle_t pDcgmHandle,
                                             int vgpuId,
                                             dcgmVgpuInstanceAttributes_t *pDcgmVgpuInstanceAttr);
dcgmReturn_t helperGetTopologyAffinity(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmAffinity_t *groupAffinity);
dcgmReturn_t helperSelectGpusByTopology(dcgmHandle_t pDcgmHandle,
                                        uint64_t inputGpuIds,
                                        uint32_t numGpus,
                                        uint64_t *outputGpuIds,
                                        uint64_t hintFlags);
dcgmReturn_t helperGetTopologyPci(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmTopology_t *groupTopology);

dcgmReturn_t helperGetMultipleValuesForFieldFV1s(dcgmHandle_t pDcgmHandle,
                                                 dcgm_field_entity_group_t entityGroup,
                                                 dcgm_field_eid_t entityId,
                                                 unsigned int fieldId,
                                                 int *count,
                                                 long long startTs,
                                                 long long endTs,
                                                 dcgmOrder_t order,
                                                 dcgmFieldValue_v1 values[]);
dcgmReturn_t helperGetLatestValuesForFields(dcgmHandle_t dcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmGroupEntityPair_t *entityList,
                                            unsigned int entityListCount,
                                            dcgmFieldGrp_t fieldGroupId,
                                            unsigned short fieldIdList[],
                                            unsigned int fieldIdListCount,
                                            DcgmFvBuffer *fvBuffer,
                                            unsigned int flags);
int helperUpdateErrorCodes(dcgmStatus_t statusHandle, int numStatuses, dcgm_config_status_t *statuses);
dcgmReturn_t helperConfigSet(dcgmHandle_t dcgmHandle,
                             dcgmGpuGrp_t groupId,
                             dcgmConfig_t *pDeviceConfig,
                             dcgmStatus_t pDcgmStatusList);
dcgmReturn_t helperVgpuConfigSet(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgmVgpuConfig_t *pDeviceConfig,
                                 dcgmStatus_t pDcgmStatusList);
dcgmReturn_t helperConfigGet(dcgmHandle_t dcgmHandle,
                             dcgmGpuGrp_t groupId,
                             dcgmConfigType_t reqType,
                             int count,
                             dcgmConfig_t *pDeviceConfigList,
                             dcgmStatus_t pDcgmStatusList);
dcgmReturn_t helperVgpuConfigGet(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgmConfigType_t reqType,
                                 int count,
                                 dcgmVgpuConfig_t *pDeviceConfigList,
                                 dcgmStatus_t pDcgmStatusList);
dcgmReturn_t helperConfigEnforce(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t pDcgmStatusList);
dcgmReturn_t helperVgpuConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t pDcgmStatusList);
dcgmReturn_t helperInjectFieldValue(dcgmHandle_t pDcgmHandle,
                                    unsigned int gpuId,
                                    dcgmInjectFieldValue_t *pDcgmInjectFieldValue);
dcgmReturn_t helperPolicyGet(dcgmHandle_t dcgmHandle,
                             dcgmGpuGrp_t groupId,
                             int count,
                             dcgmPolicy_t dcgmPolicy[],
                             dcgmStatus_t dcgmStatusList);
dcgmReturn_t helperPolicySet(dcgmHandle_t dcgmHandle,
                             dcgmGpuGrp_t groupId,
                             dcgmPolicy_t *dcgmPolicy,
                             dcgmStatus_t dcgmStatusList);
dcgmReturn_t helperHealthSet(dcgmHandle_t dcgmHandle, dcgmHealthSetParams_v2 *healthSetParams);
dcgmReturn_t helperHealthGet(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems);
dcgmReturn_t helperActionManager(dcgmHandle_t dcgmHandle,
                                 dcgmRunDiag_v10 *drd,
                                 dcgmPolicyAction_t action,
                                 dcgmDiagResponse_v12 *response);
dcgmReturn_t helperStopDiag(dcgmHandle_t dcgmHandle);
dcgmReturn_t helperStopMnDiag(dcgmHandle_t dcgmHandle);
dcgmReturn_t cmHelperGetAllDevices(dcgmHandle_t pDcgmHandle, unsigned int *pGpuIdList, int *pCount, int onlySupported);
extern "C" dcgmReturn_t dcgmInjectEntityFieldValueToNvml(dcgmHandle_t dcgmHandle,
                                                         dcgm_field_entity_group_t entityGroupId,
                                                         dcgm_field_eid_t entityId,
                                                         dcgmInjectFieldValue_t *value);
extern "C" dcgmReturn_t dcgmCreateNvmlInjectionGpu(dcgmHandle_t dcgmHandle, unsigned int index);
#ifdef INJECTION_LIBRARY_AVAILABLE
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
extern "C" dcgmReturn_t dcgmGetNvmlInjectFuncCallCount(dcgmHandle_t dcgmHandle,
                                                       injectNvmlFuncCallCounts_t *nvmlFuncCallCounts);
extern "C" dcgmReturn_t dcgmResetNvmlInjectFuncCallCount(dcgmHandle_t dcgmHandle);
extern "C" dcgmReturn_t dcgmInjectedNvmlDeviceReset(dcgmHandle_t dcgmHandle, unsigned int gpuId);
extern "C" dcgmReturn_t dcgmRemoveNvmlInjectedGpu(dcgmHandle_t dcgmHandle, char const *uuid);
extern "C" dcgmReturn_t dcgmRestoreNvmlInjectedGpu(dcgmHandle_t dcgmHandle, char const *uuid);
#endif


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
    unsigned long constexpr firstCpuCoreMask  = 0x3;
    unsigned long constexpr secondCpuCoreMask = 0xC;
    dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
    sysmonMsg.cpuCount                      = 2;
    sysmonMsg.cpus[0].cpuId                 = 3;
    sysmonMsg.cpus[1].cpuId                 = 7;
    sysmonMsg.cpus[0].ownedCores.bitmask[0] = firstCpuCoreMask;
    sysmonMsg.cpus[1].ownedCores.bitmask[0] = secondCpuCoreMask;

    dcgmCpuHierarchy_v1 hierarchy {};
    hierarchy.version = dcgmCpuHierarchy_version1;
    hierarchy.numCpus = 99;

    REQUIRE(helperFillCpuHierarchyV1(&hierarchy, sysmonMsg) == DCGM_ST_OK);
    CHECK(hierarchy.numCpus == 2);
    CHECK(hierarchy.cpus[0].cpuId == 3);
    CHECK(hierarchy.cpus[1].cpuId == 7);
    CHECK(hierarchy.cpus[0].ownedCores.bitmask[0] == firstCpuCoreMask);
    CHECK(hierarchy.cpus[1].ownedCores.bitmask[0] == secondCpuCoreMask);
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
    unsigned long constexpr firstCpuCoreMask  = 0x30;
    unsigned long constexpr secondCpuCoreMask = 0xC0;
    dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
    sysmonMsg.cpuCount                      = 2;
    sysmonMsg.cpus[0].cpuId                 = 5;
    sysmonMsg.cpus[1].cpuId                 = 9;
    sysmonMsg.cpus[0].ownedCores.bitmask[0] = firstCpuCoreMask;
    sysmonMsg.cpus[1].ownedCores.bitmask[0] = secondCpuCoreMask;
    SafeCopyTo(sysmonMsg.cpus[0].serial, "cpu-five");
    SafeCopyTo(sysmonMsg.cpus[1].serial, "cpu-nine");

    dcgmCpuHierarchy_v2 hierarchy {};
    hierarchy.version = dcgmCpuHierarchy_version2;
    hierarchy.numCpus = 99;

    REQUIRE(helperFillCpuHierarchyV2(&hierarchy, sysmonMsg) == DCGM_ST_OK);
    CHECK(hierarchy.numCpus == 2);
    CHECK(hierarchy.cpus[0].cpuId == 5);
    CHECK(hierarchy.cpus[1].cpuId == 9);
    CHECK(hierarchy.cpus[0].ownedCores.bitmask[0] == firstCpuCoreMask);
    CHECK(hierarchy.cpus[1].ownedCores.bitmask[0] == secondCpuCoreMask);
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
        SafeCopyTo(sysmonMsg.cpus[i].serial, "serial");
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
    auto params     = MakeUniqueZero<dcgmRunMnDiag_v2>();
    params->version = dcgmRunMnDiag_version2;

    auto resp     = MakeUniqueZero<dcgmMnDiagResponse_v2>();
    resp->version = dcgmMnDiagResponse_version2;

    SECTION("null drmnd returns DCGM_ST_BADPARAM")
    {
        REQUIRE(helperRunMnDiag((dcgmHandle_t)0, nullptr, resp.get()) == DCGM_ST_BADPARAM);
    }

    SECTION("null response returns DCGM_ST_BADPARAM")
    {
        REQUIRE(helperRunMnDiag((dcgmHandle_t)0, params.get(), nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("both null returns DCGM_ST_BADPARAM")
    {
        REQUIRE(helperRunMnDiag((dcgmHandle_t)0,
                                static_cast<dcgmRunMnDiag_v2 const *>(nullptr),
                                static_cast<dcgmMnDiagResponse_v2 *>(nullptr))
                == DCGM_ST_BADPARAM);
    }

    SECTION("unexpected response version returns version mismatch before dispatch")
    {
        resp->version = 0;
        REQUIRE(helperRunMnDiag((dcgmHandle_t)0, params.get(), resp.get()) == DCGM_ST_VER_MISMATCH);
    }

    SECTION("v1 response version is accepted (not rejected as version mismatch)")
    {
        dcgmRunMnDiag_v1 v1Params {};
        v1Params.version = dcgmRunMnDiag_version1;
        SafeCopyTo(v1Params.hostList[0], "localhost");
        SafeCopyTo(v1Params.testName, "mnubergemm");

        dcgmMnDiagResponse_v1 v1Resp {};
        v1Resp.version = dcgmMnDiagResponse_version1;

        auto *castResp   = reinterpret_cast<dcgmMnDiagResponse_v2 *>(&v1Resp);
        auto *castParams = reinterpret_cast<dcgmRunMnDiag_v2 const *>(&v1Params);
        dcgmReturn_t ret = helperRunMnDiag((dcgmHandle_t)0, castParams, castResp);
        REQUIRE(ret == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("UpgradeMnDiagParams_v1_to_v2: preserves all v1 host entries and fields")
{
    dcgmRunMnDiag_v1 v1 {};
    v1.version = dcgmRunMnDiag_version1;

    SafeCopyTo(v1.hostList[0], "host-0");
    SafeCopyTo(v1.hostList[1], "host-1");
    SafeCopyTo(v1.hostList[DCGM_MAX_NUM_HOSTS - 1], "host-last");
    SafeCopyTo(v1.testName, "mnubergemm");
    SafeCopyTo(v1.testParms[0], "mnubergemm.time_to_run=60");

    auto v2 = MakeUniqueZero<dcgmRunMnDiag_v2>();
    UpgradeMnDiagParams_v1_to_v2(*v2, v1);

    CHECK(v2->version == dcgmRunMnDiag_version2);
    CHECK(std::string(v2->hostList[0]) == "host-0");
    CHECK(std::string(v2->hostList[1]) == "host-1");
    CHECK(std::string(v2->hostList[DCGM_MAX_NUM_HOSTS - 1]) == "host-last");
    CHECK(std::string(v2->hostList[DCGM_MAX_NUM_HOSTS]) == "");
    CHECK(std::string(v2->testName) == "mnubergemm");
    CHECK(std::string(v2->testParms[0]) == "mnubergemm.time_to_run=60");
}

TEST_CASE("DowngradeMnDiagResponse_v2_to_v1: preserves fields within v1 limits")
{
    auto v2         = MakeUniqueZero<dcgmMnDiagResponse_v2>();
    v2->version     = dcgmMnDiagResponse_version2;
    v2->numHosts    = 5;
    v2->numTests    = 1;
    v2->numErrors   = 2;
    v2->numInfos    = 1;
    v2->numEntities = 10;
    v2->numResults  = 10;

    SafeCopyTo(v2->hosts[0].hostname, "host-alpha");
    SafeCopyTo(v2->hosts[4].hostname, "host-epsilon");
    SafeCopyTo(v2->tests[0].name, "mnubergemm");
    v2->tests[0].result     = DCGM_DIAG_RESULT_PASS;
    v2->tests[0].numResults = 10;
    for (unsigned short i = 0; i < 10; i++)
    {
        v2->tests[0].resultIndices[i] = i;
    }

    dcgmMnDiagResponse_v1 v1 {};
    DowngradeMnDiagResponse_v2_to_v1(v1, *v2);

    CHECK(v1.version == dcgmMnDiagResponse_version1);
    CHECK(v1.numHosts == 5);
    CHECK(v1.numTests == 1);
    CHECK(v1.numErrors == 2);
    CHECK(v1.numInfos == 1);
    CHECK(v1.numEntities == 10);
    CHECK(v1.numResults == 10);
    CHECK(std::string(v1.hosts[0].hostname) == "host-alpha");
    CHECK(std::string(v1.hosts[4].hostname) == "host-epsilon");
    CHECK(std::string(v1.tests[0].name) == "mnubergemm");
    CHECK(v1.tests[0].result == DCGM_DIAG_RESULT_PASS);
    CHECK(v1.tests[0].numResults == 10);
    CHECK(v1.tests[0].resultIndices[0] == 0);
    CHECK(v1.tests[0].resultIndices[9] == 9);
}

TEST_CASE("DowngradeMnDiagResponse_v2_to_v1: clamps when v2 exceeds v1 capacity")
{
    auto v2                 = MakeUniqueZero<dcgmMnDiagResponse_v2>();
    v2->version             = dcgmMnDiagResponse_version2;
    v2->numHosts            = 200;
    v2->numEntities         = DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX_V1 + 100;
    v2->numResults          = DCGM_MN_DIAG_RESPONSE_RESULTS_MAX_V1 + 50;
    v2->numTests            = 1;
    v2->tests[0].numResults = DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX_V1 + 100;

    dcgmMnDiagResponse_v1 v1 {};
    DowngradeMnDiagResponse_v2_to_v1(v1, *v2);

    CHECK(v1.numHosts == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX_V1);
    CHECK(v1.numEntities == DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX_V1);
    CHECK(v1.numResults == DCGM_MN_DIAG_RESPONSE_RESULTS_MAX_V1);
    CHECK(v1.tests[0].numResults == DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX_V1);
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

SCENARIO("DcgmApi: helper job command rejects unsupported commands before dispatch")
{
    dcgmHandle_t handle                   = static_cast<dcgmHandle_t>(1);
    dcgmGpuGrp_t constexpr groupId        = 1;
    std::size_t constexpr jobIdBufferSize = 64;
    char validJobId[jobIdBufferSize]      = "jobA";

    WHEN("an unsupported job command is sent through the shared helper")
    {
        unsigned int constexpr unsupportedJobCommand = 0xFFFFFFFFU;
        dcgmReturn_t const ret = helperJobStatCmd(handle, groupId, validJobId, unsupportedJobCommand);

        THEN("the command is rejected before module dispatch")
        {
            CHECK(ret == DCGM_ST_GENERIC_ERROR);
        }
    }
}

SCENARIO("DcgmApi: helper validation rejects bad inputs before dispatch")
{
    dcgmHandle_t handle            = static_cast<dcgmHandle_t>(1);
    dcgmGpuGrp_t constexpr groupId = 1;
    unsigned short fieldIds[]      = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
    dcgmFieldGrp_t fieldGroupId {};

    GIVEN("field watch helper inputs that are invalid without hostengine state")
    {
        long long constexpr updateFreqUsec = 1000000;
        double constexpr maxKeepAge        = 1.0;
        int constexpr maxKeepSamples       = 1;

        WHEN("watch and unwatch helpers validate scalar field arguments")
        {
            THEN("bad parameters are returned before module dispatch")
            {
                CHECK(helperWatchFieldValue(handle, 0, 0, updateFreqUsec, maxKeepAge, maxKeepSamples)
                      == DCGM_ST_BADPARAM);
                CHECK(helperWatchFieldValue(handle, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, maxKeepAge, maxKeepSamples)
                      == DCGM_ST_BADPARAM);
                CHECK(helperWatchFieldValue(handle, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, updateFreqUsec, 0.0, 0)
                      == DCGM_ST_BADPARAM);
                CHECK(dcgmUnwatchFieldValue(handle, 0, 0, 0) == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("configuration helper payloads with invalid pointers, counts, and versions")
    {
        unsigned int constexpr configCount           = 2;
        dcgmConfigType_t constexpr invalidConfigType = static_cast<dcgmConfigType_t>(99);
        dcgmConfig_t config {};
        config.version = dcgmConfig_version;
        std::array<dcgmConfig_t, configCount> configs {};
        configs[0].version = dcgmConfig_version;
        configs[1].version = dcgmConfig_version;
        dcgmVgpuConfig_t vgpuConfig {};
        vgpuConfig.version = dcgmVgpuConfig_version1;
        dcgmVgpuConfig_t vgpuConfigs[configCount] {};
        vgpuConfigs[0].version = dcgmVgpuConfig_version1;
        vgpuConfigs[1].version = dcgmVgpuConfig_version1;

        WHEN("configuration helpers validate inputs before sending module commands")
        {
            THEN("bad parameters, version mismatches, and unsupported vGPU no-ops are explicit")
            {
                CHECK(helperConfigSet(handle, groupId, nullptr, 0) == DCGM_ST_BADPARAM);

                config.version = 0;
                CHECK(helperConfigSet(handle, groupId, &config, 0) == DCGM_ST_VER_MISMATCH);

                CHECK(helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, nullptr, 0) == DCGM_ST_BADPARAM);
                CHECK(helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 0, configs.data(), 0)
                      == DCGM_ST_BADPARAM);
                CHECK(helperConfigGet(handle, groupId, invalidConfigType, 1, configs.data(), 0) == DCGM_ST_BADPARAM);

                configs[1].version = 0;
                CHECK(helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, configCount, configs.data(), 0)
                      == DCGM_ST_VER_MISMATCH);
                configs[1].version = dcgmConfig_version;
                configs[0].version = 0;
                CHECK(helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, configs.data(), 0)
                      == DCGM_ST_VER_MISMATCH);

                CHECK(helperVgpuConfigSet(handle, groupId, nullptr, 0) == DCGM_ST_BADPARAM);
                vgpuConfig.version = 0;
                CHECK(helperVgpuConfigSet(handle, groupId, &vgpuConfig, 0) == DCGM_ST_VER_MISMATCH);
                vgpuConfig.version = dcgmVgpuConfig_version1;
                CHECK(helperVgpuConfigSet(handle, groupId, &vgpuConfig, 0) == DCGM_ST_NOT_SUPPORTED);

                CHECK(helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, nullptr, 0)
                      == DCGM_ST_BADPARAM);
                CHECK(helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 0, vgpuConfigs, 0)
                      == DCGM_ST_BADPARAM);
                vgpuConfigs[1].version = 0;
                CHECK(helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, configCount, vgpuConfigs, 0)
                      == DCGM_ST_VER_MISMATCH);
                vgpuConfigs[1].version = dcgmVgpuConfig_version1;
                CHECK(helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, vgpuConfigs, 0)
                      == DCGM_ST_NOT_SUPPORTED);
                CHECK(helperVgpuConfigEnforce(handle, groupId, 0) == DCGM_ST_NOT_SUPPORTED);
            }
        }
    }

    GIVEN("policy and field-group helpers with invalid request envelopes")
    {
        unsigned int constexpr policyCount = 2;
        dcgmPolicy_t policy {};
        policy.version = dcgmPolicy_version;
        std::array<dcgmPolicy_t, policyCount> policies {};
        policies[0].version = dcgmPolicy_version;
        policies[1].version = dcgmPolicy_version;
        dcgmFieldGroupInfo_t fieldGroupInfo {};
        dcgmAllFieldGroup_t allFieldGroups {};
        std::string const tooLongName(DCGM_MAX_STR_LENGTH, 'x');

        WHEN("the helper APIs validate their local contracts")
        {
            THEN("local errors are reported without relying on module dispatch")
            {
                CHECK(helperPolicyGet(handle, groupId, 1, nullptr, 0) == DCGM_ST_BADPARAM);
                CHECK(helperPolicyGet(handle, groupId, 0, policies.data(), 0) == DCGM_ST_BADPARAM);
                policies[1].version = 0;
                CHECK(helperPolicyGet(handle, groupId, policyCount, policies.data(), 0) == DCGM_ST_VER_MISMATCH);
                CHECK(helperPolicySet(handle, groupId, nullptr, 0) == DCGM_ST_BADPARAM);
                policy.version = 0;
                CHECK(helperPolicySet(handle, groupId, &policy, 0) == DCGM_ST_VER_MISMATCH);

                CHECK(dcgmFieldGroupCreate(handle, 0, fieldIds, "fields", &fieldGroupId) == DCGM_ST_BADPARAM);
                CHECK(
                    dcgmFieldGroupCreate(handle, DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP, fieldIds, "fields", &fieldGroupId)
                    == DCGM_ST_BADPARAM);
                CHECK(dcgmFieldGroupCreate(handle, 1, nullptr, "fields", &fieldGroupId) == DCGM_ST_BADPARAM);
                CHECK(dcgmFieldGroupCreate(handle, 1, fieldIds, nullptr, &fieldGroupId) == DCGM_ST_BADPARAM);
                CHECK(dcgmFieldGroupCreate(handle, 1, fieldIds, "", &fieldGroupId) == DCGM_ST_BADPARAM);
                CHECK(dcgmFieldGroupCreate(handle, 1, fieldIds, tooLongName.c_str(), &fieldGroupId)
                      == DCGM_ST_BADPARAM);
                CHECK(dcgmFieldGroupCreate(handle, 1, fieldIds, "fields", nullptr) == DCGM_ST_BADPARAM);

                CHECK(dcgmFieldGroupGetInfo(handle, nullptr) == DCGM_ST_BADPARAM);
                fieldGroupInfo.version = 0;
                CHECK(dcgmFieldGroupGetInfo(handle, &fieldGroupInfo) == DCGM_ST_VER_MISMATCH);

                CHECK(dcgmFieldGroupGetAll(handle, nullptr) == DCGM_ST_BADPARAM);
                allFieldGroups.version = 0;
                CHECK(dcgmFieldGroupGetAll(handle, &allFieldGroups) == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    GIVEN("MIG create and delete helper requests with invalid envelopes")
    {
        dcgmCreateMigEntity_t createMig {};
        createMig.version = dcgmCreateMigEntity_version;
        dcgmDeleteMigEntity_t deleteMig {};
        deleteMig.version = dcgmDeleteMigEntity_version;

        WHEN("MIG helper APIs validate versions, profiles, and entity ids")
        {
            THEN("invalid requests fail before dispatch")
            {
                CHECK(dcgmCreateMigEntity(handle, nullptr) == DCGM_ST_BADPARAM);
                createMig.version = 0;
                CHECK(dcgmCreateMigEntity(handle, &createMig) == DCGM_ST_VER_MISMATCH);

                createMig.version      = dcgmCreateMigEntity_version;
                createMig.createOption = DcgmMigCreateComputeInstance;
                createMig.profile      = DcgmMigProfileNone;
                CHECK(dcgmCreateMigEntity(handle, &createMig) == DCGM_ST_BADPARAM);

                CHECK(dcgmDeleteMigEntity(handle, nullptr) == DCGM_ST_BADPARAM);
                deleteMig.version = 0;
                CHECK(dcgmDeleteMigEntity(handle, &deleteMig) == DCGM_ST_VER_MISMATCH);

                deleteMig.version       = dcgmDeleteMigEntity_version;
                deleteMig.entityGroupId = DCGM_FE_GPU;
                CHECK(dcgmDeleteMigEntity(handle, &deleteMig) == DCGM_ST_BADPARAM);

                deleteMig.entityGroupId = DCGM_FE_GPU_I;
                deleteMig.entityId      = DCGM_MAX_INSTANCES;
                CHECK(dcgmDeleteMigEntity(handle, &deleteMig) == DCGM_ST_BADPARAM);

                deleteMig.entityGroupId = DCGM_FE_GPU_CI;
                deleteMig.entityId      = DCGM_MAX_COMPUTE_INSTANCES;
                CHECK(dcgmDeleteMigEntity(handle, &deleteMig) == DCGM_ST_BADPARAM);
            }
        }
    }
}

SCENARIO("DcgmApi: embedded and connection entry points validate local contracts")
{
    GIVEN("embedded-start parameters across supported ABI versions")
    {
        dcgmOperationMode_t constexpr invalidOperationMode = static_cast<dcgmOperationMode_t>(999);
        dcgmStartEmbeddedV2Params_v1 v1 {};
        v1.version = dcgmStartEmbeddedV2Params_version1;
        v1.opMode  = invalidOperationMode;

        dcgmStartEmbeddedV2Params_v2 v2 {};
        v2.version = dcgmStartEmbeddedV2Params_version2;
        v2.opMode  = invalidOperationMode;

        dcgmStartEmbeddedV2Params_v3 v3 {};
        v3.version = dcgmStartEmbeddedV2Params_version3;
        v3.opMode  = invalidOperationMode;

        WHEN("the public v2 entry point receives invalid operation modes")
        {
            auto const v1Ret = dcgmStartEmbedded_v2(&v1);
            auto const v2Ret = dcgmStartEmbedded_v2(reinterpret_cast<dcgmStartEmbeddedV2Params_v1 *>(&v2));
            auto const v3Ret = dcgmStartEmbedded_v2(reinterpret_cast<dcgmStartEmbeddedV2Params_v1 *>(&v3));

            THEN("each ABI version rejects the request before starting hostengine")
            {
                CHECK(v1Ret == DCGM_ST_BADPARAM);
                CHECK(v2Ret == DCGM_ST_BADPARAM);
                CHECK(v3Ret == DCGM_ST_BADPARAM);
                CHECK(v1.dcgmHandle == 0);
                CHECK(v2.dcgmHandle == 0);
                CHECK(v3.dcgmHandle == 0);
            }
        }

        WHEN("the start helpers receive null or unknown-version inputs")
        {
            unsigned int constexpr preservedHandleValue = 0x55;
            dcgmHandle_t const preservedHandle          = static_cast<dcgmHandle_t>(preservedHandleValue);
            dcgmHandle_t handle                         = preservedHandle;
            dcgmStartEmbeddedV2Params_v1 badVersion {};
            badVersion.version = 0;

            THEN("the calls fail locally and preserve caller state")
            {
                CHECK(dcgmStartEmbedded_v2(nullptr) == DCGM_ST_BADPARAM);
                CHECK(dcgmStartEmbedded_v2(&badVersion) == DCGM_ST_VER_MISMATCH);
                CHECK(dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, nullptr) == DCGM_ST_BADPARAM);
                CHECK(handle == preservedHandle);
            }
        }
    }

    GIVEN("connect parameters before and after library initialization")
    {
        dcgmConnectV2Params_t v2Params {};
        v2Params.version = dcgmConnectV2Params_version;
        dcgmConnectV3Params_t v3Params {};
        v3Params.version    = dcgmConnectV3Params_version;
        dcgmHandle_t handle = 0;

        WHEN("required pointers and addresses are missing")
        {
            THEN("bad parameters are reported before global state is consulted")
            {
                CHECK(dcgmConnect(nullptr, &handle) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect("", &handle) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect("127.0.0.1", nullptr) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect_v2(nullptr, &v2Params, &handle) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect_v2("", &v2Params, &handle) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect_v2("127.0.0.1", nullptr, &handle) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect_v2("127.0.0.1", &v2Params, nullptr) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect_v3(nullptr, &v3Params, &handle) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect_v3("", &v3Params, &handle) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect_v3("tcp://127.0.0.1", nullptr, &handle) == DCGM_ST_BADPARAM);
                CHECK(dcgmConnect_v3("tcp://127.0.0.1", &v3Params, nullptr) == DCGM_ST_BADPARAM);
            }
        }

        WHEN("versioned connect parameters are stale after dcgmInit")
        {
            REQUIRE(dcgmInit() == DCGM_ST_OK);
            DcgmNs::Defer shutdown([] { dcgmShutdown(); });
            v2Params.version = 0;
            v3Params.version = 0;

            THEN("version mismatches are returned without opening connections")
            {
                CHECK(dcgmConnect_v2("127.0.0.1", &v2Params, &handle) == DCGM_ST_VER_MISMATCH);
                CHECK(dcgmConnect_v3("tcp://127.0.0.1", &v3Params, &handle) == DCGM_ST_VER_MISMATCH);
                unsigned int constexpr invalidEmbeddedHandleValue = 0x1234;
                dcgmHandle_t const invalidEmbeddedHandle = static_cast<dcgmHandle_t>(invalidEmbeddedHandleValue);
                CHECK(dcgmStopEmbedded(invalidEmbeddedHandle) == DCGM_ST_BADPARAM);
            }
        }
    }
}


TEST_CASE("DcgmApi: public entry points return uninitialized before dcgmInit")
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
    auto runMnDiagPtr      = MakeUniqueZero<dcgmRunMnDiag_v2>();
    auto &runMnDiag        = *runMnDiagPtr;
    auto mnDiagResponsePtr = MakeUniqueZero<dcgmMnDiagResponse_v2>();
    auto &mnDiagResponse   = *mnDiagResponsePtr;
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

TEST_CASE("DcgmApi: public entry points validate parameters after dcgmInit")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { dcgmShutdown(); });

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
    auto runMnDiagPtr      = MakeUniqueZero<dcgmRunMnDiag_v2>();
    auto &runMnDiag        = *runMnDiagPtr;
    auto mnDiagResponsePtr = MakeUniqueZero<dcgmMnDiagResponse_v2>();
    auto &mnDiagResponse   = *mnDiagResponsePtr;
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
    dcgm_policy_violation_t policyViolations[1] {};
    unsigned int policyViolationCount = 1;
    std::array<dcgmGroupEntityPair_t, DCGM_MAX_NUM_DEVICES + 1> policyEntities {};
    dcgmPolicyInfo_t policyInfo {};
    uint64_t policyId {};

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

    SECTION("GIVEN invalid field-policy API payloads WHEN called THEN the public API rejects them")
    {
        constexpr char policyName[] = "policy";
        constexpr char emptyPath[]  = "";

        CHECK(dcgmPolicyCreate(handle,
                               DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                               nullptr,
                               80.0,
                               DCGM_POLICY_OP_GT,
                               nullptr,
                               0,
                               DCGM_POLICY_CHANNEL_CALLBACK,
                               1,
                               0,
                               &policyId)
              == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyCreate(handle,
                               DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                               policyName,
                               80.0,
                               DCGM_POLICY_OP_GT,
                               nullptr,
                               0,
                               DCGM_POLICY_CHANNEL_CALLBACK,
                               1,
                               0,
                               nullptr)
              == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyCreate(handle,
                               DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                               policyName,
                               80.0,
                               DCGM_POLICY_OP_GT,
                               nullptr,
                               1,
                               DCGM_POLICY_CHANNEL_CALLBACK,
                               1,
                               0,
                               &policyId)
              == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyCreate(handle,
                               DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                               policyName,
                               80.0,
                               DCGM_POLICY_OP_GT,
                               policyEntities.data(),
                               static_cast<unsigned int>(policyEntities.size()),
                               DCGM_POLICY_CHANNEL_CALLBACK,
                               1,
                               0,
                               &policyId)
              == DCGM_ST_BADPARAM);

        policyEntities[0].entityGroupId = static_cast<dcgm_field_entity_group_t>(DCGM_FE_GPU + 1);
        CHECK(dcgmPolicyCreate(handle,
                               DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                               policyName,
                               80.0,
                               DCGM_POLICY_OP_GT,
                               policyEntities.data(),
                               1,
                               DCGM_POLICY_CHANNEL_CALLBACK,
                               1,
                               0,
                               &policyId)
              == DCGM_ST_BADPARAM);

        CHECK(dcgmPolicyModify(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyGetOne(handle, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, nullptr) == DCGM_ST_BADPARAM);

        policyInfo.version = 0;
        CHECK(dcgmPolicyModify(handle, &policyInfo) == DCGM_ST_VER_MISMATCH);
        CHECK(dcgmPolicyGetOne(handle, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, &policyInfo) == DCGM_ST_VER_MISMATCH);
        unsigned int policyInfoCount = 1;
        CHECK(dcgmPolicyGetAll(handle, &policyInfo, &policyInfoCount) == DCGM_ST_VER_MISMATCH);

        policyInfo.version     = dcgmPolicyInfo_version;
        policyInfo.entityCount = DCGM_MAX_NUM_DEVICES + 1;
        CHECK(dcgmPolicyModify(handle, &policyInfo) == DCGM_ST_BADPARAM);

        CHECK(dcgmPolicyGetAll(handle, nullptr, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyGetAll(handle, nullptr, &policyViolationCount) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyGetViolations(handle, 0, nullptr, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyGetViolations(handle, 0, nullptr, &policyViolationCount) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyGetViolations(handle, 0, policyViolations, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyImport(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyImport(handle, emptyPath) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyRegister_v3(handle, 0, 0, nullptr, 0) == DCGM_ST_BADPARAM);

        CHECK(dcgmPolicyDelete(handle, 0, 0, 0) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyEnable(handle, 0, 0) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyDisable(handle, 0, 0) == DCGM_ST_BADPARAM);
        CHECK(dcgmPolicyUnregister_v3(handle, 0, 0) == DCGM_ST_BADPARAM);
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


TEST_CASE("DcgmApi: initialized boundary checks cover local validation paths")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { dcgmShutdown(); });
    dcgmHandle_t handle = 0;

    SECTION("GIVEN fake entity and NvLink payloads WHEN required envelopes are missing THEN local errors are returned")
    {
        dcgmCreateFakeEntities_v2 fakeEntities {};
        fakeEntities.version = dcgmCreateFakeEntities_version;
        dcgmSetNvLinkLinkState_v1 linkState {};
        linkState.version = dcgmSetNvLinkLinkState_version1;
        dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
        DcgmEntityStatus_t gpuStatus = DcgmEntityStatusOk;

        CHECK(dcgmCreateFakeEntities(handle, nullptr) == DCGM_ST_BADPARAM);
        fakeEntities.version = 0;
        CHECK(dcgmCreateFakeEntities(handle, &fakeEntities) == DCGM_ST_VER_MISMATCH);

        CHECK(dcgmSetEntityNvLinkLinkState(handle, nullptr) == DCGM_ST_BADPARAM);
        linkState.version = 0;
        CHECK(dcgmSetEntityNvLinkLinkState(handle, &linkState) == DCGM_ST_VER_MISMATCH);

        CHECK(dcgmGetCacheManagerFieldInfo(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetGpuStatus(handle, 0, nullptr) == DCGM_ST_BADPARAM);
        CHECK(gpuStatus == DcgmEntityStatusOk);
        CHECK(fieldInfo.version == 0);
    }

    SECTION("GIVEN value API requests WHEN counts are zero or output buffers are absent THEN BADPARAM is returned")
    {
        dcgmGroupEntityPair_t entities[] = { { DCGM_FE_GPU, 0 } };
        unsigned short fieldIds[]        = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        dcgmFieldValue_v1 values[1] {};
        dcgmFieldValue_v2 valuesV2[1] {};
        int count = 0;

        CHECK(dcgmGetLatestValuesForFields(handle, 0, fieldIds, 0, values) == DCGM_ST_BADPARAM);
        CHECK(dcgmEntityGetLatestValues(handle, DCGM_FE_GPU, 0, fieldIds, 0, values) == DCGM_ST_BADPARAM);
        CHECK(dcgmEntitiesGetLatestValues(handle, entities, 0, fieldIds, 1, 0, valuesV2) == DCGM_ST_BADPARAM);
        CHECK(dcgmEntitiesGetLatestValues(handle, entities, 1, fieldIds, 0, 0, valuesV2) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetMultipleValuesForField(
                  handle, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &count, 0, 0, DCGM_ORDER_ASCENDING, values)
              == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN versioned module and hostengine payloads WHEN versions are invalid THEN mismatches are explicit")
    {
        dcgmModuleGetStatuses_t moduleStatuses {};
        moduleStatuses.version = dcgmModuleGetStatuses_version;
        dcgmProfGetMetricGroups_t metricGroups {};
        metricGroups.version = dcgmProfGetMetricGroups_version3;
        dcgmEnvVarInfo_t envVarInfo {};
        envVarInfo.version = dcgmEnvVarInfo_version;
        dcgmModulesReloadable_v1 modulesReloadable {};
        modulesReloadable.version = dcgmModulesReloadable_version;
        dcgmWorkloadPowerProfile_t workloadPowerProfile {};
        workloadPowerProfile.version = dcgmWorkloadPowerProfile_version1;

        CHECK(dcgmModuleDenylist(handle, DcgmModuleIdCore) == DCGM_ST_BADPARAM);
        CHECK(dcgmModuleDenylist(handle, DcgmModuleIdCount) == DCGM_ST_BADPARAM);

        CHECK(dcgmModuleGetStatuses(handle, nullptr) == DCGM_ST_BADPARAM);
        moduleStatuses.version = 0;
        CHECK(dcgmModuleGetStatuses(handle, &moduleStatuses) == DCGM_ST_VER_MISMATCH);

        CHECK(dcgmProfGetSupportedMetricGroups(handle, nullptr) == DCGM_ST_BADPARAM);
        metricGroups.version = 0;
        CHECK(dcgmProfGetSupportedMetricGroups(handle, &metricGroups) == DCGM_ST_VER_MISMATCH);

        CHECK(dcgmHostengineEnvironmentVariableInfo(handle, nullptr) == DCGM_ST_BADPARAM);
        envVarInfo.version = 0;
        CHECK(dcgmHostengineEnvironmentVariableInfo(handle, &envVarInfo) == DCGM_ST_VER_MISMATCH);

        CHECK(dcgmMarkModulesReloadable(handle, nullptr) == DCGM_ST_BADPARAM);
        modulesReloadable.version = 0;
        CHECK(dcgmMarkModulesReloadable(handle, &modulesReloadable) == DCGM_ST_VER_MISMATCH);

        CHECK(dcgmConfigSetWorkloadPowerProfile(handle, nullptr) == DCGM_ST_BADPARAM);
        workloadPowerProfile.version = 0;
        CHECK(dcgmConfigSetWorkloadPowerProfile(handle, &workloadPowerProfile) == DCGM_ST_VER_MISMATCH);
    }

    SECTION(
        "GIVEN workload power profile info requests WHEN paired payloads are absent or stale THEN validation stops locally")
    {
        dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo {};
        profilesInfo.version = dcgmWorkloadPowerProfileProfilesInfo_version;
        dcgmDeviceWorkloadPowerProfilesStatus_v1 profilesStatus {};
        profilesStatus.version = dcgmDeviceWorkloadPowerProfilesStatus_version;

        CHECK(dcgmGetDeviceWorkloadPowerProfileInfo(handle, 0, nullptr, &profilesStatus) == DCGM_ST_BADPARAM);
        CHECK(dcgmGetDeviceWorkloadPowerProfileInfo(handle, 0, &profilesInfo, nullptr) == DCGM_ST_BADPARAM);

        profilesInfo.version = 0;
        CHECK(dcgmGetDeviceWorkloadPowerProfileInfo(handle, 0, &profilesInfo, &profilesStatus) == DCGM_ST_VER_MISMATCH);
        profilesInfo.version   = dcgmWorkloadPowerProfileProfilesInfo_version;
        profilesStatus.version = 0;
        CHECK(dcgmGetDeviceWorkloadPowerProfileInfo(handle, 0, &profilesInfo, &profilesStatus) == DCGM_ST_VER_MISMATCH);
    }
}


SCENARIO("DcgmApi: helpers reject invalid local inputs and preserve caller buffers")
{
    dcgmHandle_t handle = static_cast<dcgmHandle_t>(1);

    GIVEN("group info requests with invalid envelopes")
    {
        unsigned int constexpr preservedGroupCount = 77;
        long long constexpr preservedTimestamp     = 1234;
        dcgmGroupInfo_t groupInfo {};
        groupInfo.version   = dcgmGroupInfo_version;
        groupInfo.count     = preservedGroupCount;
        long long timestamp = preservedTimestamp;

        WHEN("the group helper validates pointers and versions")
        {
            THEN("bad parameters are rejected before dispatch and caller data is preserved")
            {
                CHECK(helperGroupGetInfo(handle, 1, nullptr, &timestamp) == DCGM_ST_BADPARAM);
                CHECK(timestamp == preservedTimestamp);

                groupInfo.version = dcgmGroupInfo_version3 - 1;
                CHECK(helperGroupGetInfo(handle, 1, &groupInfo, &timestamp) == DCGM_ST_VER_MISMATCH);
                CHECK(groupInfo.count == preservedGroupCount);

                groupInfo.version = dcgmGroupInfo_version + 1;
                CHECK(helperGroupGetInfo(handle, 1, &groupInfo, nullptr) == DCGM_ST_VER_MISMATCH);
                CHECK(groupInfo.count == preservedGroupCount);
            }
        }
    }

    GIVEN("multiple-value requests with unsupported field ids")
    {
        unsigned short constexpr firstUnsupportedFieldId  = 0xABCD;
        unsigned short constexpr secondUnsupportedFieldId = 0xBCDE;
        dcgmFieldValue_v1 values[2] {};
        values[0].fieldId   = firstUnsupportedFieldId;
        values[0].value.i64 = 123;
        values[1].fieldId   = secondUnsupportedFieldId;
        values[1].value.i64 = 456;
        int count           = 2;

        WHEN("the FV1 conversion helper is called")
        {
            auto ret = helperGetMultipleValuesForFieldFV1s(
                handle, DCGM_FE_GPU, 0, DCGM_FI_MAX_FIELDS, &count, 0, 0, DCGM_ORDER_ASCENDING, values);

            THEN("the field is rejected locally, the count is consumed, and output values are cleared")
            {
                CHECK(ret == DCGM_ST_UNKNOWN_FIELD);
                CHECK(count == 0);
                CHECK(values[0].fieldId == 0);
                CHECK(values[0].value.i64 == 0);
                CHECK(values[1].fieldId == 0);
                CHECK(values[1].value.i64 == 0);
            }
        }
    }

    GIVEN("watch requests with invalid field metadata")
    {
        WHEN("the watch helper is called with zero and unknown fields")
        {
            THEN("both paths return bad parameters before module dispatch")
            {
                CHECK(helperWatchFieldValue(handle, 0, 0, 1000, 10.0, 1) == DCGM_ST_BADPARAM);
                CHECK(helperWatchFieldValue(handle, 0, DCGM_FI_MAX_FIELDS, 1000, 10.0, 1) == DCGM_ST_BADPARAM);
            }
        }
    }
}


SCENARIO("DcgmApi: status handles expose queued errors in FIFO order")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { dcgmShutdown(); });

    GIVEN("a public status handle populated through the shared error helper")
    {
        dcgmStatus_t statusHandle {};
        REQUIRE(dcgmStatusCreate(&statusHandle) == DCGM_ST_OK);
        DcgmNs::Defer destroyStatus([&] { dcgmStatusDestroy(statusHandle); });

        int constexpr queuedStatusCount        = 3;
        unsigned int constexpr firstStatusGpu  = 7;
        unsigned int constexpr secondStatusGpu = 8;
        unsigned int constexpr thirdStatusGpu  = 9;
        dcgm_config_status_t statuses[queuedStatusCount] {};
        statuses[0].gpuId     = firstStatusGpu;
        statuses[0].fieldId   = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        statuses[0].errorCode = DCGM_ST_BADPARAM;
        statuses[1].gpuId     = secondStatusGpu;
        statuses[1].fieldId   = DCGM_FI_DEV_BOARD_POWER_WATTS;
        statuses[1].errorCode = DCGM_ST_NOT_SUPPORTED;
        statuses[2].gpuId     = thirdStatusGpu;
        statuses[2].fieldId   = DCGM_FI_DEV_ECC_SBE_VOL_TOTAL;
        statuses[2].errorCode = DCGM_ST_NO_DATA;

        REQUIRE(helperUpdateErrorCodes(statusHandle, queuedStatusCount, statuses) == 0);

        WHEN("errors are counted and popped through the public API")
        {
            unsigned int count = 0;
            REQUIRE(dcgmStatusGetCount(statusHandle, &count) == DCGM_ST_OK);

            dcgmErrorInfo_t first {};
            dcgmErrorInfo_t second {};
            dcgmErrorInfo_t third {};
            auto const firstRet  = dcgmStatusPopError(statusHandle, &first);
            auto const secondRet = dcgmStatusPopError(statusHandle, &second);
            auto const thirdRet  = dcgmStatusPopError(statusHandle, &third);

            THEN("the queued errors are returned in insertion order")
            {
                CHECK(count == queuedStatusCount);
                REQUIRE(firstRet == DCGM_ST_OK);
                CHECK(first.gpuId == firstStatusGpu);
                CHECK(first.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(first.status == DCGM_ST_BADPARAM);

                REQUIRE(secondRet == DCGM_ST_OK);
                CHECK(second.gpuId == secondStatusGpu);
                CHECK(second.fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(second.status == DCGM_ST_NOT_SUPPORTED);

                REQUIRE(thirdRet == DCGM_ST_OK);
                CHECK(third.gpuId == thirdStatusGpu);
                CHECK(third.fieldId == DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
                CHECK(third.status == DCGM_ST_NO_DATA);

                CHECK(dcgmStatusGetCount(statusHandle, &count) == DCGM_ST_OK);
                CHECK(count == 0);
                CHECK(dcgmStatusPopError(statusHandle, &first) == DCGM_ST_NO_DATA);
            }
        }

        WHEN("the status handle is cleared before popping")
        {
            unsigned int constexpr nonZeroInitialCount = 99;
            unsigned int count                         = nonZeroInitialCount;
            auto const clearRet                        = dcgmStatusClear(statusHandle);
            auto const countRet                        = dcgmStatusGetCount(statusHandle, &count);

            THEN("the handle reports an empty queue")
            {
                CHECK(clearRet == DCGM_ST_OK);
                CHECK(countRet == DCGM_ST_OK);
                CHECK(count == 0);
                dcgmErrorInfo_t popped {};
                CHECK(dcgmStatusPopError(statusHandle, &popped) == DCGM_ST_NO_DATA);
            }
        }
    }
}

SCENARIO("DcgmApi: enum-name helpers validate output pointers and boundaries")
{
    GIVEN("power profile and module id name lookups")
    {
        WHEN("valid ids are queried with missing output pointers")
        {
            THEN("the helpers reject the request before writing names")
            {
                CHECK(dcgmPowerProfileIdToName(static_cast<dcgmPowerProfileType_t>(DCGM_POWER_PROFILE_MAX - 1), nullptr)
                      == DCGM_ST_BADPARAM);
                CHECK(dcgmModuleIdToName(DcgmModuleIdCore, nullptr) == DCGM_ST_BADPARAM);
            }
        }

        WHEN("boundary ids are queried with valid output pointers")
        {
            char const *powerProfileName = nullptr;
            char const *moduleName       = nullptr;

            auto const powerRet = dcgmPowerProfileIdToName(
                static_cast<dcgmPowerProfileType_t>(DCGM_POWER_PROFILE_MAX - 1), &powerProfileName);
            auto const badPowerRet  = dcgmPowerProfileIdToName(DCGM_POWER_PROFILE_MAX, &powerProfileName);
            auto const moduleRet    = dcgmModuleIdToName(DcgmModuleIdCore, &moduleName);
            auto const badModuleRet = dcgmModuleIdToName(DcgmModuleIdCount, &moduleName);

            THEN("inclusive valid boundaries succeed and exclusive boundaries fail")
            {
                REQUIRE(powerRet == DCGM_ST_OK);
                REQUIRE(powerProfileName != nullptr);
                CHECK(std::strlen(powerProfileName) > 0);
                CHECK(badPowerRet == DCGM_ST_BADPARAM);

                REQUIRE(moduleRet == DCGM_ST_OK);
                CHECK(std::strcmp(moduleName, "Core") == 0);
                CHECK(badModuleRet == DCGM_ST_BADPARAM);
            }
        }
    }
}


SCENARIO("DcgmApi: attribute and topology helpers validate local contracts before dispatch")
{
    dcgmHandle_t handle = static_cast<dcgmHandle_t>(1);

    GIVEN("device attribute requests with null and unsupported versions")
    {
        unsigned int constexpr preservedPciDeviceId     = 0x1234;
        unsigned int constexpr preservedPersistenceMode = 7;
        dcgmDeviceAttributes_t deviceAttrs {};
        deviceAttrs.version                         = dcgmDeviceAttributes_version;
        deviceAttrs.identifiers.pciDeviceId         = preservedPciDeviceId;
        deviceAttrs.settings.persistenceModeEnabled = preservedPersistenceMode;

        WHEN("the helper validates the envelope")
        {
            auto const nullRet       = helperDeviceGetAttributes(handle, 0, nullptr);
            deviceAttrs.version      = dcgmDeviceAttributes_version + 1;
            auto const badVersionRet = helperDeviceGetAttributes(handle, 0, &deviceAttrs);

            THEN("invalid requests are rejected without mutating caller data")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(badVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(deviceAttrs.identifiers.pciDeviceId == preservedPciDeviceId);
                CHECK(deviceAttrs.settings.persistenceModeEnabled == preservedPersistenceMode);
            }
        }
    }

    GIVEN("vGPU device attribute requests with null and unsupported versions")
    {
        unsigned int constexpr preservedSupportedVgpuTypeCount  = 11;
        unsigned int constexpr preservedCreatableVgpuTypeCount  = 12;
        unsigned int constexpr preservedActiveVgpuInstanceCount = 13;
        dcgmVgpuDeviceAttributes_t vgpuDeviceAttrs {};
        vgpuDeviceAttrs.version                 = dcgmVgpuDeviceAttributes_version7;
        vgpuDeviceAttrs.supportedVgpuTypeCount  = preservedSupportedVgpuTypeCount;
        vgpuDeviceAttrs.creatableVgpuTypeCount  = preservedCreatableVgpuTypeCount;
        vgpuDeviceAttrs.activeVgpuInstanceCount = preservedActiveVgpuInstanceCount;

        WHEN("the helper validates the vGPU device envelope")
        {
            auto const nullRet       = helperVgpuDeviceGetAttributes(handle, 0, nullptr);
            vgpuDeviceAttrs.version  = dcgmVgpuDeviceAttributes_version7 - 1;
            auto const badVersionRet = helperVgpuDeviceGetAttributes(handle, 0, &vgpuDeviceAttrs);

            THEN("invalid requests fail before dispatch and preserve caller data")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(badVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(vgpuDeviceAttrs.supportedVgpuTypeCount == preservedSupportedVgpuTypeCount);
                CHECK(vgpuDeviceAttrs.creatableVgpuTypeCount == preservedCreatableVgpuTypeCount);
                CHECK(vgpuDeviceAttrs.activeVgpuInstanceCount == preservedActiveVgpuInstanceCount);
            }
        }
    }

    GIVEN("vGPU instance attribute requests at version boundaries")
    {
        unsigned int constexpr preservedVgpuTypeId = 99;
        char constexpr preservedVmName[]           = "preserved-vm";
        dcgmVgpuInstanceAttributes_t instanceAttrs {};
        instanceAttrs.version    = dcgmVgpuInstanceAttributes_version1;
        instanceAttrs.vgpuTypeId = preservedVgpuTypeId;
        std::strncpy(instanceAttrs.vmName, preservedVmName, sizeof(instanceAttrs.vmName) - 1);

        WHEN("the helper validates null, too-old, and too-new versions")
        {
            auto const nullRet    = helperVgpuInstanceGetAttributes(handle, 0, nullptr);
            instanceAttrs.version = dcgmVgpuInstanceAttributes_version1 - 1;
            auto const oldRet     = helperVgpuInstanceGetAttributes(handle, 0, &instanceAttrs);
            instanceAttrs.version = dcgmVgpuInstanceAttributes_version + 1;
            auto const newRet     = helperVgpuInstanceGetAttributes(handle, 0, &instanceAttrs);

            THEN("only broad-contract validation is performed before dispatch")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(oldRet == DCGM_ST_VER_MISMATCH);
                CHECK(newRet == DCGM_ST_VER_MISMATCH);
                CHECK(instanceAttrs.vgpuTypeId == preservedVgpuTypeId);
                CHECK(std::string(instanceAttrs.vmName) == preservedVmName);
            }
        }
    }

    GIVEN("topology requests with invalid local inputs")
    {
        dcgmAffinity_t affinity {};
        dcgmTopology_t topology {};
        uint64_t constexpr preservedGpuMask = 0xDEADBEEF;
        uint64_t constexpr inputGpuMask     = 0x3;
        uint64_t selectedGpus               = preservedGpuMask;

        WHEN("topology helpers validate output pointers and handles")
        {
            auto const affinityNullRet     = helperGetTopologyAffinity(handle, 1, nullptr);
            auto const pciNullRet          = helperGetTopologyPci(handle, 1, nullptr);
            auto const selectNullHandleRet = helperSelectGpusByTopology(
                static_cast<dcgmHandle_t>(0), inputGpuMask, 2, &selectedGpus, DCGM_TOPO_HINT_F_IGNOREHEALTH);
            auto const selectNullOutputRet
                = helperSelectGpusByTopology(handle, inputGpuMask, 2, nullptr, DCGM_TOPO_HINT_F_IGNOREHEALTH);

            THEN("bad parameters are rejected before module dispatch")
            {
                CHECK(affinityNullRet == DCGM_ST_BADPARAM);
                CHECK(pciNullRet == DCGM_ST_BADPARAM);
                CHECK(selectNullHandleRet == DCGM_ST_BADPARAM);
                CHECK(selectNullOutputRet == DCGM_ST_BADPARAM);
                CHECK(selectedGpus == preservedGpuMask);
                CHECK(affinity.numGpus == 0);
                CHECK(topology.numElements == 0);
            }
        }
    }
}
SCENARIO("DcgmApi: configuration, policy, health, and field-value helpers reject local contract violations")
{
    dcgmHandle_t handle            = static_cast<dcgmHandle_t>(1);
    dcgmGpuGrp_t constexpr groupId = 1;

    GIVEN("configuration set and get requests")
    {
        unsigned int constexpr configCount = 2;
        dcgmConfig_t configs[configCount] {};
        configs[0].version = dcgmConfig_version;
        configs[1].version = dcgmConfig_version - 1;

        WHEN("null, bad request type, and mismatched versions are supplied")
        {
            auto const setNullRet    = helperConfigSet(handle, groupId, nullptr, 0);
            configs[0].version       = dcgmConfig_version - 1;
            auto const setVersionRet = helperConfigSet(handle, groupId, &configs[0], 0);
            configs[0].version       = dcgmConfig_version;
            auto const getNullRet    = helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, nullptr, 0);
            auto const getCountRet   = helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 0, configs, 0);
            dcgmConfigType_t constexpr invalidConfigType = static_cast<dcgmConfigType_t>(99);
            auto const getTypeRet = helperConfigGet(handle, groupId, invalidConfigType, 1, configs, 0);
            auto const getVersionRet
                = helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, configCount, configs, 0);

            THEN("each invalid envelope is rejected before module dispatch")
            {
                CHECK(setNullRet == DCGM_ST_BADPARAM);
                CHECK(setVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(getNullRet == DCGM_ST_BADPARAM);
                CHECK(getCountRet == DCGM_ST_BADPARAM);
                CHECK(getTypeRet == DCGM_ST_BADPARAM);
                CHECK(getVersionRet == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    GIVEN("vGPU configuration requests")
    {
        unsigned int constexpr configCount = 2;
        dcgmVgpuConfig_t configs[configCount] {};
        configs[0].version = dcgmVgpuConfig_version1;
        configs[1].version = dcgmVgpuConfig_version1;

        WHEN("the vGPU helpers validate inputs locally")
        {
            auto const setNullRet      = helperVgpuConfigSet(handle, groupId, nullptr, 0);
            configs[0].version         = dcgmVgpuConfig_version1 - 1;
            auto const setVersionRet   = helperVgpuConfigSet(handle, groupId, &configs[0], 0);
            configs[0].version         = dcgmVgpuConfig_version1;
            auto const setSupportedRet = helperVgpuConfigSet(handle, groupId, &configs[0], 0);
            auto const getNullRet      = helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, nullptr, 0);
            auto const getCountRet     = helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 0, configs, 0);
            configs[1].version         = dcgmVgpuConfig_version1 + 1;
            auto const getVersionRet
                = helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, configCount, configs, 0);

            THEN("bad envelopes fail and valid envelopes report the unsupported vGPU implementation")
            {
                CHECK(setNullRet == DCGM_ST_BADPARAM);
                CHECK(setVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(setSupportedRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(getNullRet == DCGM_ST_BADPARAM);
                CHECK(getCountRet == DCGM_ST_BADPARAM);
                CHECK(getVersionRet == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    GIVEN("policy and health requests")
    {
        unsigned int constexpr policyCount = 2;
        dcgmPolicy_t policies[policyCount] {};
        policies[0].version = dcgmPolicy_version;
        policies[1].version = dcgmPolicy_version - 1;
        dcgmHealthSetParams_v2 healthSet {};
        healthSet.version = dcgmHealthSetParams_version2 - 1;

        WHEN("local validation is applied")
        {
            auto const policyGetNullRet    = helperPolicyGet(handle, groupId, 1, nullptr, 0);
            auto const policyGetCountRet   = helperPolicyGet(handle, groupId, 0, policies, 0);
            auto const policyGetVersionRet = helperPolicyGet(handle, groupId, policyCount, policies, 0);
            auto const policySetNullRet    = helperPolicySet(handle, groupId, nullptr, 0);
            auto const policySetVersionRet = helperPolicySet(handle, groupId, &policies[1], 0);
            auto const healthSetNullRet    = helperHealthSet(handle, nullptr);
            auto const healthSetVersionRet = helperHealthSet(handle, &healthSet);
            auto const healthGetNullRet    = helperHealthGet(handle, groupId, nullptr);

            THEN("invalid requests return precise status codes before dispatch")
            {
                CHECK(policyGetNullRet == DCGM_ST_BADPARAM);
                CHECK(policyGetCountRet == DCGM_ST_BADPARAM);
                CHECK(policyGetVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(policySetNullRet == DCGM_ST_BADPARAM);
                CHECK(policySetVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(healthSetNullRet == DCGM_ST_BADPARAM);
                CHECK(healthSetVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(healthGetNullRet == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("latest field-value requests")
    {
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 0 };
        unsigned int constexpr entityCount = 1;
        unsigned int constexpr fieldCount  = 1;
        unsigned short fieldId             = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        DcgmFvBuffer fvBuffer;

        WHEN("entity lists, field lists, and output buffers violate preconditions")
        {
            auto const entityCountRet
                = helperGetLatestValuesForFields(handle, groupId, &entity, 0, 0, &fieldId, 1, &fvBuffer, 0);
            auto const fieldCountRet
                = helperGetLatestValuesForFields(handle, groupId, &entity, entityCount, 0, &fieldId, 0, &fvBuffer, 0);
            auto const bufferRet = helperGetLatestValuesForFields(
                handle, groupId, &entity, entityCount, 0, &fieldId, fieldCount, nullptr, 0);

            THEN("bad parameters are rejected without contacting the host engine")
            {
                CHECK(entityCountRet == DCGM_ST_BADPARAM);
                CHECK(fieldCountRet == DCGM_ST_BADPARAM);
                CHECK(bufferRet == DCGM_ST_BADPARAM);
            }
        }
    }
}


SCENARIO("DcgmApi: initialization, connection, and hostengine helpers validate local contracts")
{
    dcgmReturn_t const shutdownRet = dcgmShutdown();
    REQUIRE((shutdownRet == DCGM_ST_OK || shutdownRet == DCGM_ST_UNINITIALIZED));

    GIVEN("embedded start requests before initialization")
    {
        dcgmHandle_t handle                                = 0;
        dcgmOperationMode_t constexpr invalidOperationMode = static_cast<dcgmOperationMode_t>(99);
        dcgmStartEmbeddedV2Params_v1 startParams {};
        startParams.version = dcgmStartEmbeddedV2Params_version1;
        startParams.opMode  = invalidOperationMode;
        dcgmStartEmbeddedV2Params_v2 startParamsV2 {};
        startParamsV2.version = dcgmStartEmbeddedV2Params_version2;
        startParamsV2.opMode  = invalidOperationMode;
        dcgmStartEmbeddedV2Params_v3 startParamsV3 {};
        startParamsV3.version = dcgmStartEmbeddedV2Params_version3;
        startParamsV3.opMode  = invalidOperationMode;

        WHEN("null, unknown-version, and bad-operation requests are submitted")
        {
            auto const nullRet    = dcgmStartEmbedded_v2(nullptr);
            startParams.version   = 0;
            auto const versionRet = dcgmStartEmbedded_v2(&startParams);
            startParams.version   = dcgmStartEmbeddedV2Params_version1;
            auto const badModeRet = dcgmStartEmbedded_v2(&startParams);
            auto const badModeV2Ret
                = dcgmStartEmbedded_v2(reinterpret_cast<dcgmStartEmbeddedV2Params_v1 *>(&startParamsV2));
            auto const badModeV3Ret
                = dcgmStartEmbedded_v2(reinterpret_cast<dcgmStartEmbeddedV2Params_v1 *>(&startParamsV3));
            auto const nullHandleRet    = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, nullptr);
            auto const uninitializedRet = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &handle);

            THEN("validation fails before hostengine construction")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(versionRet == DCGM_ST_VER_MISMATCH);
                CHECK(badModeRet == DCGM_ST_BADPARAM);
                CHECK(badModeV2Ret == DCGM_ST_BADPARAM);
                CHECK(badModeV3Ret == DCGM_ST_BADPARAM);
                CHECK(nullHandleRet == DCGM_ST_BADPARAM);
                CHECK(uninitializedRet == DCGM_ST_UNINITIALIZED);
                CHECK(handle == 0);
                CHECK(startParamsV2.dcgmHandle == 0);
                CHECK(startParamsV3.dcgmHandle == 0);
            }
        }
    }

    GIVEN("initialized DCGM with local-only connection and version requests")
    {
        REQUIRE(dcgmInit() == DCGM_ST_OK);
        DcgmNs::Defer shutdown([] { CHECK(dcgmShutdown() == DCGM_ST_OK); });

        char const *legacyAddress             = "127.0.0.1";
        char const *tcpAddress                = "tcp://127.0.0.1";
        std::size_t constexpr backendNameSize = 4;
        dcgmHandle_t handle                   = 0;
        dcgmConnectV2Params_t connectV2 {};
        connectV2.version = dcgmConnectV2Params_version;
        dcgmConnectV3Params_t connectV3 {};
        connectV3.version = dcgmConnectV3Params_version;
        dcgmVersionInfo_t versionInfo {};
        versionInfo.version = dcgmVersionInfo_version;
        dcgmSettingsSetLoggingSeverity_t logging {};
        logging.version = dcgmSettingsSetLoggingSeverity_version - 1;
        char backendName[backendNameSize] {};
        bool active = true;

        WHEN("pointers, versions, and local buffer sizes are invalid")
        {
            auto const legacyConnectNullAddressRet = dcgmConnect(nullptr, &handle);
            auto const legacyConnectNullHandleRet  = dcgmConnect(legacyAddress, nullptr);
            auto const connectNullAddressRet       = dcgmConnect_v2(nullptr, &connectV2, &handle);
            auto const connectEmptyAddressRet      = dcgmConnect_v2("", &connectV2, &handle);
            auto const connectNullParamsRet        = dcgmConnect_v2(legacyAddress, nullptr, &handle);
            auto const connectNullHandleRet        = dcgmConnect_v2(legacyAddress, &connectV2, nullptr);
            connectV2.version                      = 0;
            auto const connectV2VersionRet         = dcgmConnect_v2(legacyAddress, &connectV2, &handle);

            auto const connectV3NullAddressRet  = dcgmConnect_v3(nullptr, &connectV3, &handle);
            auto const connectV3EmptyAddressRet = dcgmConnect_v3("", &connectV3, &handle);
            auto const connectV3NullParamsRet   = dcgmConnect_v3(tcpAddress, nullptr, &handle);
            auto const connectV3NullHandleRet   = dcgmConnect_v3(tcpAddress, &connectV3, nullptr);
            connectV3.version                   = 0;
            auto const connectV3VersionRet      = dcgmConnect_v3(tcpAddress, &connectV3, &handle);

            auto const versionNullRet        = dcgmVersionInfo(nullptr);
            versionInfo.version              = 0;
            auto const versionMismatchRet    = dcgmVersionInfo(&versionInfo);
            auto const loggingVersionRet     = dcgmHostengineSetLoggingSeverity(0, &logging);
            auto const stopBadHandleRet      = dcgmStopEmbedded(0);
            auto const disconnectNoClientRet = dcgmDisconnect(0);
            auto const backendSmallRet       = dcgmNvswitchGetBackend(0, &active, backendName, sizeof(backendName));

            THEN("all errors are reported before network or hostengine dispatch")
            {
                CHECK(legacyConnectNullAddressRet == DCGM_ST_BADPARAM);
                CHECK(legacyConnectNullHandleRet == DCGM_ST_BADPARAM);
                CHECK(connectNullAddressRet == DCGM_ST_BADPARAM);
                CHECK(connectEmptyAddressRet == DCGM_ST_BADPARAM);
                CHECK(connectNullParamsRet == DCGM_ST_BADPARAM);
                CHECK(connectNullHandleRet == DCGM_ST_BADPARAM);
                CHECK(connectV2VersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(connectV3NullAddressRet == DCGM_ST_BADPARAM);
                CHECK(connectV3EmptyAddressRet == DCGM_ST_BADPARAM);
                CHECK(connectV3NullParamsRet == DCGM_ST_BADPARAM);
                CHECK(connectV3NullHandleRet == DCGM_ST_BADPARAM);
                CHECK(connectV3VersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(versionNullRet == DCGM_ST_BADPARAM);
                CHECK(versionMismatchRet == DCGM_ST_VER_MISMATCH);
                CHECK(loggingVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(stopBadHandleRet == DCGM_ST_BADPARAM);
                CHECK(disconnectNoClientRet == DCGM_ST_OK);
                CHECK(backendSmallRet == DCGM_ST_INSUFFICIENT_SIZE);
                CHECK(handle == 0);
            }
        }
    }
}


SCENARIO("DcgmApi: version helper validates envelopes and fills build metadata")
{
    GIVEN("a caller-owned version info structure")
    {
        dcgmVersionInfo_t versionInfo {};
        versionInfo.version = dcgmVersionInfo_version;

        WHEN("the public version API receives a valid envelope")
        {
            auto const ret = dcgmVersionInfo(&versionInfo);

            THEN("the build information string is populated without requiring a hostengine")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(std::strlen(versionInfo.rawBuildInfoString) > 0);
                CHECK(versionInfo.rawBuildInfoString[sizeof(versionInfo.rawBuildInfoString) - 1] == '\0');
            }
        }

        WHEN("the public version API receives invalid envelopes")
        {
            char const sentinel   = versionInfo.rawBuildInfoString[0];
            auto const nullRet    = dcgmVersionInfo(nullptr);
            versionInfo.version   = 0;
            auto const versionRet = dcgmVersionInfo(&versionInfo);

            THEN("bad parameters and version mismatches are reported before writing build data")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(versionRet == DCGM_ST_VER_MISMATCH);
                CHECK(versionInfo.rawBuildInfoString[0] == sentinel);
            }
        }
    }
}


SCENARIO("DcgmApi: helper envelopes validate local-only failure cases")
{
    dcgmHandle_t handle            = static_cast<dcgmHandle_t>(1);
    dcgmGpuGrp_t constexpr groupId = 1;

    GIVEN("group info and field injection helper inputs")
    {
        long long constexpr preservedHostTimestamp = 1234;
        dcgmGroupInfo_t groupInfo {};
        groupInfo.version       = 0;
        long long hostTimestamp = preservedHostTimestamp;

        WHEN("null pointers and version mismatches are supplied")
        {
            auto const groupNullRet    = helperGroupGetInfo(handle, 1, nullptr, &hostTimestamp);
            auto const groupVersionRet = helperGroupGetInfo(handle, 1, &groupInfo, &hostTimestamp);
            auto const injectNullRet   = helperInjectFieldValue(handle, 0, nullptr);

            THEN("the helpers reject them without mutating caller-owned outputs")
            {
                CHECK(groupNullRet == DCGM_ST_BADPARAM);
                CHECK(groupVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(hostTimestamp == preservedHostTimestamp);
                CHECK(injectNullRet == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("public policy and health APIs with invalid local arguments")
    {
        dcgmHealthSystems_t systems = DCGM_HEALTH_WATCH_ALL;

        WHEN("bad callbacks, bad conditions, and null health pointers are used")
        {
            auto const registerNullCallbackRet
                = dcgmPolicyRegister_v2(handle, groupId, DCGM_POLICY_COND_DBE, nullptr, 0);
            auto const unregisterInvalidConditionRet
                = dcgmPolicyUnregister(handle, groupId, static_cast<dcgmPolicyCondition_t>(0));
            auto const healthSetNullRet     = dcgmHealthSet_v2(handle, nullptr);
            auto const healthSetBadGroupRet = dcgmHealthSet(handle, DCGM_GROUP_NULL, systems);
            auto const healthGetNullRet     = dcgmHealthGet(handle, groupId, nullptr);

            THEN("public entry points report local validation failures consistently")
            {
                CHECK(registerNullCallbackRet != DCGM_ST_OK);
                CHECK(unregisterInvalidConditionRet != DCGM_ST_OK);
                CHECK(healthSetNullRet == DCGM_ST_BADPARAM);
                CHECK(healthSetBadGroupRet != DCGM_ST_OK);
                CHECK(healthGetNullRet == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("configuration enforcement wrappers with unsupported vGPU and normal GPU paths")
    {
        DcgmStatus status;

        WHEN("vGPU enforcement is requested")
        {
            auto const vgpuRet = helperVgpuConfigEnforce(handle, groupId, reinterpret_cast<dcgmStatus_t>(&status));

            THEN("the retired vGPU path is reported as unsupported without dispatch")
            {
                CHECK(vgpuRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(status.IsEmpty());
            }
        }

        WHEN("GPU enforcement is called with an invalid handle")
        {
            auto const enforceRet = helperConfigEnforce(0, groupId, reinterpret_cast<dcgmStatus_t>(&status));

            THEN("the helper returns the dispatch-layer error and leaves the status queue empty")
            {
                CHECK(enforceRet != DCGM_ST_OK);
                CHECK(status.IsEmpty());
            }
        }
    }
}


SCENARIO("DcgmApi: diagnostic helper validates version compatibility before dispatch")
{
    dcgmHandle_t handle = static_cast<dcgmHandle_t>(0);

    GIVEN("missing diagnostic request and response pointers")
    {
        dcgmRunDiag_v10 runDiag {};
        runDiag.version = dcgmRunDiag_version10;
        dcgmDiagResponse_v12 response {};
        response.version = dcgmDiagResponse_version12;

        WHEN("the shared diagnostic helper validates required envelopes")
        {
            THEN("null inputs are rejected before constructing a module request")
            {
                CHECK(helperActionManager(handle, nullptr, DCGM_POLICY_ACTION_NONE, &response) == DCGM_ST_BADPARAM);
                CHECK(helperActionManager(handle, &runDiag, DCGM_POLICY_ACTION_NONE, nullptr) == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("diagnostic request and response versions that cannot be paired")
    {
        dcgmRunDiag_v10 runDiag {};
        dcgmDiagResponse_v12 response {};

        WHEN("unknown and incompatible versions are supplied")
        {
            runDiag.version          = 0;
            response.version         = dcgmDiagResponse_version12;
            auto const unknownRunRet = helperActionManager(handle, &runDiag, DCGM_POLICY_ACTION_NONE, &response);

            runDiag.version              = dcgmRunDiag_version8;
            response.version             = dcgmDiagResponse_version12;
            auto const responseTooNewRet = helperActionManager(handle, &runDiag, DCGM_POLICY_ACTION_NONE, &response);

            runDiag.version              = dcgmRunDiag_version9;
            response.version             = dcgmDiagResponse_version10;
            auto const responseTooOldRet = helperActionManager(handle, &runDiag, DCGM_POLICY_ACTION_NONE, &response);

            runDiag.version               = dcgmRunDiag_version7;
            response.version              = 0;
            auto const unknownResponseRet = helperActionManager(handle, &runDiag, DCGM_POLICY_ACTION_NONE, &response);

            THEN("the helper returns version mismatches without module dispatch")
            {
                CHECK(unknownRunRet == DCGM_ST_VER_MISMATCH);
                CHECK(responseTooNewRet == DCGM_ST_VER_MISMATCH);
                CHECK(responseTooOldRet == DCGM_ST_VER_MISMATCH);
                CHECK(unknownResponseRet == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    GIVEN("all supported diagnostic compatibility pairs")
    {
        struct VersionPair
        {
            unsigned int runVersion;
            unsigned int responseVersion;
            unsigned int copiedResponseVersion;
        };

        auto const versionPair
            = GENERATE(VersionPair { dcgmRunDiag_version10, dcgmDiagResponse_version12, dcgmDiagResponse_version12 },
                       VersionPair { dcgmRunDiag_version10, dcgmDiagResponse_version11, dcgmDiagResponse_version11 },
                       VersionPair { dcgmRunDiag_version9, dcgmDiagResponse_version12, dcgmDiagResponse_version11 },
                       VersionPair { dcgmRunDiag_version9, dcgmDiagResponse_version11, dcgmDiagResponse_version11 },
                       VersionPair { dcgmRunDiag_version8, dcgmDiagResponse_version10, dcgmDiagResponse_version10 },
                       VersionPair { dcgmRunDiag_version7, dcgmDiagResponse_version10, dcgmDiagResponse_version10 },
                       VersionPair { dcgmRunDiag_version7, dcgmDiagResponse_version9, dcgmDiagResponse_version9 },
                       VersionPair { dcgmRunDiag_version7, dcgmDiagResponse_version8, dcgmDiagResponse_version8 },
                       VersionPair { dcgmRunDiag_version7, dcgmDiagResponse_version7, dcgmDiagResponse_version7 });

        WHEN("a compatible request reaches the dispatch boundary with an invalid handle")
        {
            THEN("the request shape is accepted and the response envelope matches the translated message")
            {
                CAPTURE(versionPair.runVersion, versionPair.responseVersion);

                dcgmRunDiag_v10 runDiag {};
                runDiag.version = versionPair.runVersion;
                dcgmDiagResponse_v12 response {};
                response.version = versionPair.responseVersion;

                auto const ret = helperActionManager(handle, &runDiag, DCGM_POLICY_ACTION_NONE, &response);
                CHECK(ret != DCGM_ST_OK);
                CHECK(response.version == versionPair.copiedResponseVersion);
            }
        }
    }

    GIVEN("diagnostic stop helpers with no live hostengine")
    {
        WHEN("stop requests are sent to the dispatch boundary")
        {
            auto const stopDiagRet   = helperStopDiag(handle);
            auto const stopMnDiagRet = helperStopMnDiag(handle);

            THEN("the helpers build requests but report the dispatch-layer failure")
            {
                CHECK(stopDiagRet != DCGM_ST_OK);
                CHECK(stopMnDiagRet != DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("DcgmApi: local helper boundaries preserve caller buffers on failure")
{
    dcgmHandle_t handle = static_cast<dcgmHandle_t>(0);

    GIVEN("device-list helper requests with invalid output buffers")
    {
        int constexpr preservedDeviceCount = 17;
        unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] {};
        int count = preservedDeviceCount;

        WHEN("required pointers are absent")
        {
            auto const nullListRet  = cmHelperGetAllDevices(handle, nullptr, &count, 0);
            auto const nullCountRet = cmHelperGetAllDevices(handle, gpuIds, nullptr, 1);

            THEN("bad parameters are returned before count or list mutation")
            {
                CHECK(nullListRet == DCGM_ST_BADPARAM);
                CHECK(nullCountRet == DCGM_ST_BADPARAM);
                CHECK(count == preservedDeviceCount);
                CHECK(gpuIds[0] == 0);
            }
        }
    }

    GIVEN("entity-group and chip-architecture helpers at the dispatch boundary")
    {
        dcgm_field_eid_t constexpr firstEntityId           = 99;
        dcgm_field_eid_t constexpr secondEntityId          = 100;
        int constexpr preservedEntityCapacity              = 2;
        dcgm_field_eid_t entities[preservedEntityCapacity] = { firstEntityId, secondEntityId };
        int entityCapacity                                 = preservedEntityCapacity;
        dcgmChipArchitecture_t arch                        = DCGM_CHIP_ARCH_HOPPER;

        WHEN("required pointers are missing or dispatch fails")
        {
            auto const entitiesNullRet = dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, nullptr, &entityCapacity, 0);
            auto const countNullRet    = dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, entities, nullptr, 0);
            auto const dispatchRet     = dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, entities, &entityCapacity, 0);
            auto const archNullRet     = dcgmGetGpuChipArchitecture(handle, 0, nullptr);
            auto const archRet         = dcgmGetGpuChipArchitecture(handle, 0, &arch);

            THEN("local validation is precise and failed dispatch leaves caller-owned outputs unchanged")
            {
                CHECK(entitiesNullRet == DCGM_ST_BADPARAM);
                CHECK(countNullRet == DCGM_ST_BADPARAM);
                CHECK(dispatchRet != DCGM_ST_OK);
                CHECK(entityCapacity == preservedEntityCapacity);
                CHECK(entities[0] == firstEntityId);
                CHECK(entities[1] == secondEntityId);
                CHECK(archNullRet == DCGM_ST_BADPARAM);
                CHECK(archRet != DCGM_ST_OK);
                CHECK(arch == DCGM_CHIP_ARCH_HOPPER);
            }
        }
    }

    GIVEN("cache-manager field-info helper inputs")
    {
        dcgm_field_eid_t constexpr fieldInfoEntityId = 7;

        dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
        fieldInfo.version       = 0;
        fieldInfo.entityGroupId = DCGM_FE_GPU;
        fieldInfo.entityId      = fieldInfoEntityId;
        fieldInfo.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;

        WHEN("null and dispatch-failing requests are issued")
        {
            auto const nullRet     = dcgmGetCacheManagerFieldInfo(handle, nullptr);
            auto const dispatchRet = dcgmGetCacheManagerFieldInfo(handle, &fieldInfo);

            THEN("null input is rejected and failed dispatch preserves the original query envelope")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(dispatchRet != DCGM_ST_OK);
                CHECK(fieldInfo.version == 0);
                CHECK(fieldInfo.entityGroupId == DCGM_FE_GPU);
                CHECK(fieldInfo.entityId == fieldInfoEntityId);
                CHECK(fieldInfo.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
            }
        }
    }
}


SCENARIO("DcgmApi: local helper validation covers oversized and dispatch-boundary requests")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { CHECK(dcgmShutdown() == DCGM_ST_OK); });

    dcgmHandle_t handle            = static_cast<dcgmHandle_t>(0);
    dcgmGpuGrp_t constexpr groupId = 1;

    GIVEN("latest-value helper requests larger than the fixed protocol envelopes")
    {
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 0 };
        unsigned int constexpr validEntityCount = 1;
        unsigned int constexpr validFieldCount  = 1;
        unsigned short fieldId                  = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        DcgmFvBuffer fvBuffer;

        WHEN("the caller exceeds either entity or field-list capacity")
        {
            auto const tooManyEntitiesRet
                = helperGetLatestValuesForFields(handle,
                                                 groupId,
                                                 &entity,
                                                 DCGM_GROUP_MAX_ENTITIES_V2 + validEntityCount,
                                                 0,
                                                 &fieldId,
                                                 validFieldCount,
                                                 &fvBuffer,
                                                 0);
            auto const tooManyFieldsRet
                = helperGetLatestValuesForFields(handle,
                                                 groupId,
                                                 &entity,
                                                 validEntityCount,
                                                 0,
                                                 &fieldId,
                                                 DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP + validFieldCount,
                                                 &fvBuffer,
                                                 0);

            THEN("both are rejected before any request buffer copy")
            {
                CHECK(tooManyEntitiesRet == DCGM_ST_BADPARAM);
                CHECK(tooManyFieldsRet == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("public job statistics requests with invalid local arguments")
    {
        std::size_t constexpr jobIdBufferSize = 64;
        char emptyJobId[jobIdBufferSize] {};
        char validJobId[jobIdBufferSize] {};
        std::strncpy(validJobId, "dcgm-job", sizeof(validJobId) - 1);
        unsigned int constexpr preservedJobGpuCount   = 77;
        unsigned int constexpr preservedXidErrorCount = 88;
        unsigned int constexpr unsupportedJobCommand  = 0xFFFFFFFF;

        dcgmJobInfo_t jobInfo {};
        jobInfo.version                      = dcgmJobInfo_version;
        jobInfo.numGpus                      = preservedJobGpuCount;
        jobInfo.summary.numXidCriticalErrors = preservedXidErrorCount;

        WHEN("the public wrappers validate job ids, output pointers, and versions")
        {
            auto const startNullRet      = dcgmJobStartStats(handle, groupId, nullptr);
            auto const startEmptyRet     = dcgmJobStartStats(handle, groupId, emptyJobId);
            auto const stopNullRet       = dcgmJobStopStats(handle, nullptr);
            auto const stopEmptyRet      = dcgmJobStopStats(handle, emptyJobId);
            auto const getNullJobRet     = dcgmJobGetStats(handle, nullptr, &jobInfo);
            auto const getNullInfoRet    = dcgmJobGetStats(handle, validJobId, nullptr);
            jobInfo.version              = 0;
            auto const getBadVersionRet  = dcgmJobGetStats(handle, validJobId, &jobInfo);
            jobInfo.version              = dcgmJobInfo_version;
            auto const getDispatchRet    = dcgmJobGetStats(handle, validJobId, &jobInfo);
            auto const removeNullRet     = dcgmJobRemove(handle, nullptr);
            auto const removeEmptyRet    = dcgmJobRemove(handle, emptyJobId);
            auto const removeDispatchRet = dcgmJobRemove(handle, validJobId);

            THEN("local errors are precise and dispatch failures preserve caller-owned job info")
            {
                CHECK(startNullRet == DCGM_ST_BADPARAM);
                CHECK(startEmptyRet == DCGM_ST_BADPARAM);
                CHECK(stopNullRet == DCGM_ST_BADPARAM);
                CHECK(stopEmptyRet == DCGM_ST_BADPARAM);
                CHECK(getNullJobRet == DCGM_ST_BADPARAM);
                CHECK(getNullInfoRet == DCGM_ST_BADPARAM);
                CHECK(getBadVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(getDispatchRet != DCGM_ST_OK);
                CHECK(jobInfo.numGpus == preservedJobGpuCount);
                CHECK(jobInfo.summary.numXidCriticalErrors == preservedXidErrorCount);
                CHECK(removeNullRet == DCGM_ST_BADPARAM);
                CHECK(removeEmptyRet == DCGM_ST_BADPARAM);
                CHECK(removeDispatchRet != DCGM_ST_OK);
            }
        }

        WHEN("the shared command helper receives command ids at and outside the supported set")
        {
            auto const unsupportedRet = helperJobStatCmd(handle, groupId, validJobId, unsupportedJobCommand);
            auto const removeAllRet   = helperJobStatCmd(handle, groupId, validJobId, DCGM_CORE_SR_JOB_REMOVE_ALL);

            THEN("unknown commands are rejected locally while supported commands reach dispatch")
            {
                CHECK(unsupportedRet == DCGM_ST_GENERIC_ERROR);
                CHECK(removeAllRet != DCGM_ST_OK);
            }
        }
    }

    GIVEN("hostengine and environment metadata payloads")
    {
        dcgmVersionInfo_t versionInfo {};
        versionInfo.version = dcgmVersionInfo_version;
        std::strncpy(versionInfo.rawBuildInfoString, "preserved-build", sizeof(versionInfo.rawBuildInfoString) - 1);
        dcgmEnvVarInfo_t envVarInfo {};
        envVarInfo.version = dcgmEnvVarInfo_version;
        std::strncpy(envVarInfo.envVarName, "DCGM_TEST_ENV", sizeof(envVarInfo.envVarName) - 1);

        WHEN("the wrappers receive missing, stale, and dispatch-boundary envelopes")
        {
            auto const hostVersionNullRet = dcgmHostengineVersionInfo(handle, nullptr);
            auto const hostVersionRet     = dcgmHostengineVersionInfo(handle, &versionInfo);
            auto const envNullRet         = dcgmHostengineEnvironmentVariableInfo(handle, nullptr);
            envVarInfo.version            = 0;
            auto const envVersionRet      = dcgmHostengineEnvironmentVariableInfo(handle, &envVarInfo);
            envVarInfo.version            = dcgmEnvVarInfo_version;
            auto const envDispatchRet     = dcgmHostengineEnvironmentVariableInfo(handle, &envVarInfo);

            THEN("bad inputs stop locally and failed dispatch leaves caller buffers intact")
            {
                CHECK(hostVersionNullRet == DCGM_ST_BADPARAM);
                CHECK(hostVersionRet != DCGM_ST_OK);
                CHECK(std::string(versionInfo.rawBuildInfoString) == "preserved-build");
                CHECK(envNullRet == DCGM_ST_BADPARAM);
                CHECK(envVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(envDispatchRet != DCGM_ST_OK);
                CHECK(std::string(envVarInfo.envVarName) == "DCGM_TEST_ENV");
            }
        }
    }
}

SCENARIO("DcgmApi: module profiling and hostengine wrappers preserve caller buffers at dispatch boundary")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { CHECK(dcgmShutdown() == DCGM_ST_OK); });

    dcgmHandle_t handle = static_cast<dcgmHandle_t>(0);

    GIVEN("module status and profiling metric-group requests with valid local envelopes")
    {
        unsigned int constexpr moduleStatusCount = 2;
        unsigned int constexpr metricGpuId       = 7;
        unsigned int constexpr metricGroupCount  = 3;

        dcgmModuleGetStatuses_t moduleStatuses {};
        moduleStatuses.version            = dcgmModuleGetStatuses_version;
        moduleStatuses.numStatuses        = moduleStatusCount;
        moduleStatuses.statuses[0].id     = DcgmModuleIdCore;
        moduleStatuses.statuses[0].status = DcgmModuleStatusLoaded;
        dcgmProfGetMetricGroups_t metricGroups {};
        metricGroups.version                 = dcgmProfGetMetricGroups_version3;
        metricGroups.gpuId                   = metricGpuId;
        metricGroups.numMetricGroups         = metricGroupCount;
        auto const originalMetricGroupCount  = metricGroups.numMetricGroups;
        auto const originalModuleStatusCount = moduleStatuses.numStatuses;

        WHEN("the requests reach the module dispatch boundary without a live hostengine")
        {
            auto const validDenylistRet = dcgmModuleDenylist(handle, DcgmModuleIdPolicy);
            auto const statusesRet      = dcgmModuleGetStatuses(handle, &moduleStatuses);
            auto const metricGroupsRet  = dcgmProfGetSupportedMetricGroups(handle, &metricGroups);
            auto const pauseRet         = dcgmProfPause(handle);
            auto const resumeRet        = dcgmProfResume(handle);

            THEN("dispatch failures are reported and caller-owned request buffers remain intact")
            {
                CHECK(validDenylistRet != DCGM_ST_OK);
                CHECK(statusesRet != DCGM_ST_OK);
                CHECK(moduleStatuses.version == dcgmModuleGetStatuses_version);
                CHECK(moduleStatuses.numStatuses == originalModuleStatusCount);
                CHECK(moduleStatuses.statuses[0].id == DcgmModuleIdCore);
                CHECK(moduleStatuses.statuses[0].status == DcgmModuleStatusLoaded);
                CHECK(metricGroupsRet != DCGM_ST_OK);
                CHECK(metricGroups.version == dcgmProfGetMetricGroups_version3);
                CHECK(metricGroups.gpuId == metricGpuId);
                CHECK(metricGroups.numMetricGroups == originalMetricGroupCount);
                CHECK(pauseRet != DCGM_ST_OK);
                CHECK(resumeRet != DCGM_ST_OK);
            }
        }
    }

    GIVEN("hostengine health and logging requests")
    {
        unsigned int constexpr preservedOverallHealth = 1234;
        unsigned int constexpr targetLogger           = 7;

        dcgmHostengineHealth_t health {};
        health.version       = dcgmHostengineHealth_version;
        health.overallHealth = preservedOverallHealth;
        dcgmSettingsSetLoggingSeverity_t logging {};
        logging.version        = dcgmSettingsSetLoggingSeverity_version;
        logging.targetLogger   = targetLogger;
        logging.targetSeverity = DcgmLoggingSeverityDebug;

        WHEN("null stale and dispatch-boundary envelopes are submitted")
        {
            auto const healthNullRet     = dcgmHostengineIsHealthy(handle, nullptr);
            health.version               = 0;
            auto const healthVersionRet  = dcgmHostengineIsHealthy(handle, &health);
            health.version               = dcgmHostengineHealth_version;
            auto const healthDispatchRet = dcgmHostengineIsHealthy(handle, &health);

            auto const loggingDispatchRet = dcgmHostengineSetLoggingSeverity(handle, &logging);

            THEN("local validation is precise and failed dispatch does not overwrite caller outputs")
            {
                CHECK(healthNullRet == DCGM_ST_BADPARAM);
                CHECK(healthVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(healthDispatchRet != DCGM_ST_OK);
                CHECK(health.version == dcgmHostengineHealth_version);
                CHECK(health.overallHealth == preservedOverallHealth);
                CHECK(loggingDispatchRet != DCGM_ST_OK);
                CHECK(logging.version == dcgmSettingsSetLoggingSeverity_version);
                CHECK(logging.targetLogger == targetLogger);
                CHECK(logging.targetSeverity == DcgmLoggingSeverityDebug);
            }
        }
    }

    GIVEN("a field summary request with a valid local envelope")
    {
        dcgm_field_eid_t constexpr entityId = 3;
        long long constexpr startTime       = 11;
        long long constexpr endTime         = 22;
        unsigned int constexpr summaryCount = 5;

        dcgmFieldSummaryRequest_t request {};
        request.version               = dcgmFieldSummaryRequest_version1;
        request.fieldId               = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        request.entityGroupId         = DCGM_FE_GPU;
        request.entityId              = entityId;
        request.summaryTypeMask       = DCGM_SUMMARY_MIN | DCGM_SUMMARY_MAX;
        request.startTime             = startTime;
        request.endTime               = endTime;
        request.response.summaryCount = summaryCount;

        WHEN("the wrapper reaches dispatch without a live hostengine")
        {
            auto const ret = dcgmGetFieldSummary(handle, &request);

            THEN("the dispatch failure preserves the request and response envelope")
            {
                CHECK(ret != DCGM_ST_OK);
                CHECK(request.version == dcgmFieldSummaryRequest_version1);
                CHECK(request.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(request.entityGroupId == DCGM_FE_GPU);
                CHECK(request.entityId == entityId);
                CHECK(request.summaryTypeMask == (DCGM_SUMMARY_MIN | DCGM_SUMMARY_MAX));
                CHECK(request.startTime == startTime);
                CHECK(request.endTime == endTime);
                CHECK(request.response.summaryCount == summaryCount);
            }
        }
    }
}


SCENARIO("DcgmApi: watch-field helper builds GPU and global watch envelopes")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { CHECK(dcgmShutdown() == DCGM_ST_OK); });
    dcgmHandle_t handle                    = static_cast<dcgmHandle_t>(0);
    dcgm_field_eid_t constexpr gpuId       = 3;
    dcgm_field_eid_t constexpr globalGpuId = 99;
    long long constexpr updateFreqUsec     = 1000000;
    double constexpr maxKeepAge            = 30.0;
    int constexpr maxKeepSamples           = 10;

    GIVEN("valid field watch parameters for entity-scoped and global fields")
    {
        WHEN("the public watch helper reaches dispatch without a live hostengine")
        {
            auto const gpuFieldRet = helperWatchFieldValue(
                handle, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, updateFreqUsec, maxKeepAge, maxKeepSamples);
            auto const globalFieldRet = dcgmWatchFieldValue(
                handle, globalGpuId, DCGM_FI_SYSTEM_DRIVER_VERSION, updateFreqUsec, maxKeepAge, maxKeepSamples);

            THEN("both valid local envelopes are accepted before dispatch failure")
            {
                CHECK(gpuFieldRet == DCGM_ST_BADPARAM);
                CHECK(globalFieldRet == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("single-fault invalid watch parameters")
    {
        WHEN("each broad precondition is violated independently")
        {
            THEN("the helper rejects the request locally")
            {
                CHECK(helperWatchFieldValue(handle, gpuId, 0, updateFreqUsec, maxKeepAge, maxKeepSamples)
                      == DCGM_ST_BADPARAM);
                CHECK(helperWatchFieldValue(handle, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, maxKeepAge, maxKeepSamples)
                      == DCGM_ST_BADPARAM);
                CHECK(helperWatchFieldValue(handle, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, updateFreqUsec, 0.0, 0)
                      == DCGM_ST_BADPARAM);
                CHECK(helperWatchFieldValue(
                          handle, gpuId, DCGM_FI_MAX_FIELDS + 1, updateFreqUsec, maxKeepAge, maxKeepSamples)
                      == DCGM_ST_BADPARAM);
            }
        }
    }
}

SCENARIO("DcgmApi: multinode request validates envelopes and preserves payloads at dispatch boundary")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { CHECK(dcgmShutdown() == DCGM_ST_OK); });

    dcgmHandle_t handle = static_cast<dcgmHandle_t>(0);

    int constexpr headNodeId = 17;

    STATIC_REQUIRE(std::to_underlying(dcgmMultinodeTestType_t::mnubergemm) == 0);
    STATIC_REQUIRE(std::to_underlying(dcgmMultinodeTestType_t::nvloom) == 1);
    STATIC_REQUIRE(std::to_underlying(dcgmMultinodeTestType_t::mnnvbandwidth) == 2);

    auto makeRequest = [headNodeId](dcgmMultinodeRequestType_t requestType,
                                    dcgmMultinodeTestType_t testType = dcgmMultinodeTestType_t::mnubergemm) {
        dcgmMultinodeRequest_t request {};
        request.version                         = dcgmMultinodeRequest_version3;
        request.testType                        = testType;
        request.requestType                     = requestType;
        request.requestData.resource.headNodeId = headNodeId;
        request.requestData.resource.response   = RESERVED;
        return request;
    };

    GIVEN("locally invalid multinode envelopes")
    {
        WHEN("null version test-type and request-type faults are isolated")
        {
            int constexpr invalidMultinodeType = 77;

            auto badVersion            = makeRequest(ReserveResources);
            badVersion.version         = dcgmMultinodeRequest_version1;
            auto badTestType           = makeRequest(ReserveResources);
            badTestType.testType       = static_cast<dcgmMultinodeTestType_t>(invalidMultinodeType);
            auto badRequestType        = makeRequest(ReserveResources);
            badRequestType.requestType = static_cast<dcgmMultinodeRequestType_t>(invalidMultinodeType);

            THEN("the public API rejects each local contract violation before dispatch")
            {
                CHECK(dcgmMultinodeRequest(handle, nullptr) == DCGM_ST_BADPARAM);
                CHECK(dcgmMultinodeRequest(handle, &badVersion) == DCGM_ST_VER_MISMATCH);
                CHECK(dcgmMultinodeRequest(handle, &badTestType) == DCGM_ST_BADPARAM);
                CHECK(dcgmMultinodeRequest(handle, &badRequestType) == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("valid multinode requests for each generated subcommand family")
    {
        auto const requestType = GENERATE(ReserveResources,
                                          ReleaseResources,
                                          DetectProcess,
                                          AuthorizeConnection,
                                          RevokeAuthorization,
                                          BroadcastRunParameters,
                                          GetNodeInfo);
        auto const testType    = GENERATE(dcgmMultinodeTestType_t::mnubergemm, dcgmMultinodeTestType_t::mnnvbandwidth);

        WHEN("the request reaches dispatch without a live hostengine")
        {
            THEN("dispatch failure is reported without corrupting the caller-owned request envelope")
            {
                CAPTURE(std::to_underlying(requestType));
                CAPTURE(std::to_underlying(testType));
                auto request               = makeRequest(requestType, testType);
                auto const originalVersion = request.version;
                auto const originalType    = request.testType;
                auto const originalRequest = request.requestType;
                auto const ret             = dcgmMultinodeRequest(handle, &request);

                CHECK(ret != DCGM_ST_OK);
                CHECK(request.version == originalVersion);
                CHECK(request.testType == originalType);
                CHECK(request.requestType == originalRequest);
            }
        }
    }
}


SCENARIO("DcgmApi: command helpers validate parameters before dispatch and build valid envelopes")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { CHECK(dcgmShutdown() == DCGM_ST_OK); });
    auto handle                    = static_cast<dcgmHandle_t>(0);
    dcgmGpuGrp_t constexpr groupId = 1;

    GIVEN("configuration helper inputs with isolated local faults")
    {
        unsigned int constexpr configCount           = 2;
        dcgmConfigType_t constexpr invalidConfigType = static_cast<dcgmConfigType_t>(99);
        dcgmConfig_t config {};
        config.version = dcgmConfig_version;
        std::array<dcgmConfig_t, configCount> configs {};
        configs[0].version = dcgmConfig_version;
        configs[1].version = dcgmConfig_version;

        WHEN("the set get and enforce helpers are called with invalid and valid envelopes")
        {
            auto wrongVersion     = config;
            wrongVersion.version  = dcgmConfig_version + 1;
            auto mismatched       = configs;
            mismatched[1].version = dcgmConfig_version + 1;

            THEN("local validation rejects bad parameters and valid envelopes reach dispatch")
            {
                CHECK(helperConfigSet(handle, groupId, nullptr, 0) == DCGM_ST_BADPARAM);
                CHECK(helperConfigSet(handle, groupId, &wrongVersion, 0) == DCGM_ST_VER_MISMATCH);
                CHECK(helperConfigSet(handle, groupId, &config, 0) == DCGM_ST_BADPARAM);

                CHECK(helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, nullptr, 0) == DCGM_ST_BADPARAM);
                CHECK(helperConfigGet(handle, groupId, invalidConfigType, 1, configs.data(), 0) == DCGM_ST_BADPARAM);
                CHECK(helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 0, configs.data(), 0)
                      == DCGM_ST_BADPARAM);
                CHECK(helperConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, configCount, mismatched.data(), 0)
                      == DCGM_ST_VER_MISMATCH);
                CHECK(helperConfigGet(handle, groupId, DCGM_CONFIG_TARGET_STATE, configCount, configs.data(), 0)
                      == DCGM_ST_BADPARAM);

                CHECK(helperConfigEnforce(handle, groupId, 0) == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("vGPU configuration helper inputs")
    {
        unsigned int constexpr configCount           = 2;
        dcgmConfigType_t constexpr invalidConfigType = static_cast<dcgmConfigType_t>(99);
        dcgmVgpuConfig_t config {};
        config.version = dcgmVgpuConfig_version;
        std::array<dcgmVgpuConfig_t, configCount> configs {};
        configs[0].version = dcgmVgpuConfig_version;
        configs[1].version = dcgmVgpuConfig_version;

        WHEN("private vGPU helpers receive supported and unsupported envelopes")
        {
            auto wrongVersion     = config;
            wrongVersion.version  = dcgmVgpuConfig_version + 1;
            auto mismatched       = configs;
            mismatched[1].version = dcgmVgpuConfig_version + 1;

            THEN("the private APIs keep their documented local validation behavior")
            {
                CHECK(helperVgpuConfigSet(handle, groupId, nullptr, 0) == DCGM_ST_BADPARAM);
                CHECK(helperVgpuConfigSet(handle, groupId, &wrongVersion, 0) == DCGM_ST_VER_MISMATCH);
                CHECK(helperVgpuConfigSet(handle, groupId, &config, 0) == DCGM_ST_NOT_SUPPORTED);

                CHECK(helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 1, nullptr, 0)
                      == DCGM_ST_BADPARAM);
                CHECK(helperVgpuConfigGet(handle, groupId, invalidConfigType, 1, configs.data(), 0)
                      == DCGM_ST_NOT_SUPPORTED);
                CHECK(helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, 0, configs.data(), 0)
                      == DCGM_ST_BADPARAM);
                CHECK(helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, configCount, mismatched.data(), 0)
                      == DCGM_ST_VER_MISMATCH);
                CHECK(helperVgpuConfigGet(handle, groupId, DCGM_CONFIG_TARGET_STATE, configCount, configs.data(), 0)
                      == DCGM_ST_NOT_SUPPORTED);
                CHECK(helperVgpuConfigEnforce(handle, groupId, 0) == DCGM_ST_NOT_SUPPORTED);
            }
        }
    }

    GIVEN("policy helper inputs")
    {
        unsigned int constexpr policyCount = 2;
        dcgmPolicy_t policy {};
        policy.version = dcgmPolicy_version;
        std::array<dcgmPolicy_t, policyCount> policies {};
        policies[0].version = dcgmPolicy_version;
        policies[1].version = dcgmPolicy_version;

        WHEN("policy get and set helpers receive invalid and valid caller storage")
        {
            auto wrongVersion     = policy;
            wrongVersion.version  = dcgmPolicy_version + 1;
            auto mismatched       = policies;
            mismatched[1].version = dcgmPolicy_version + 1;

            THEN("bad local contracts fail before dispatch and valid envelopes reach dispatch")
            {
                CHECK(helperPolicyGet(handle, groupId, 1, nullptr, 0) == DCGM_ST_BADPARAM);
                CHECK(helperPolicyGet(handle, groupId, 0, policies.data(), 0) == DCGM_ST_BADPARAM);
                CHECK(helperPolicyGet(handle, groupId, 1, &wrongVersion, 0) == DCGM_ST_VER_MISMATCH);
                CHECK(helperPolicyGet(handle, groupId, policyCount, mismatched.data(), 0) == DCGM_ST_VER_MISMATCH);
                CHECK(helperPolicyGet(handle, groupId, policyCount, policies.data(), 0) == DCGM_ST_BADPARAM);

                CHECK(helperPolicySet(handle, groupId, nullptr, 0) == DCGM_ST_BADPARAM);
                CHECK(helperPolicySet(handle, groupId, &wrongVersion, 0) == DCGM_ST_VER_MISMATCH);
                CHECK(helperPolicySet(handle, groupId, &policy, 0) == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("field-watch helper inputs")
    {
        dcgm_field_eid_t constexpr gpuId        = 3;
        unsigned short constexpr invalidFieldId = DCGM_FI_MAX_FIELDS + 1;

        WHEN("valid public envelopes reach the module send boundary without a hostengine")
        {
            auto updateRet        = helperUpdateAllFields(handle, groupId);
            auto unwatchGpuRet    = dcgmUnwatchFieldValue(handle, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0);
            auto unwatchGlobalRet = dcgmUnwatchFieldValue(handle, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, 1);

            THEN("the helpers report dispatch failure but not local parameter failure")
            {
                CHECK(updateRet == DCGM_ST_BADPARAM);
                CHECK(unwatchGpuRet == DCGM_ST_BADPARAM);
                CHECK(unwatchGlobalRet == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the unwatch helper receives invalid field ids")
        {
            THEN("field metadata validation rejects them locally")
            {
                CHECK(dcgmUnwatchFieldValue(handle, gpuId, 0, 0) == DCGM_ST_BADPARAM);
                CHECK(dcgmUnwatchFieldValue(handle, gpuId, invalidFieldId, 0) == DCGM_ST_BADPARAM);
            }
        }
    }
}


SCENARIO("DcgmApi: public watch, history, health, and introspection boundaries preserve caller state")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { CHECK(dcgmShutdown() == DCGM_ST_OK); });

    dcgmHandle_t handle                     = static_cast<dcgmHandle_t>(0);
    dcgmGpuGrp_t constexpr groupId          = 1;
    dcgmGpuGrp_t constexpr dispatchGroupId  = 9;
    dcgmFieldGrp_t constexpr fieldGroupId   = 7;
    dcgm_field_eid_t constexpr gpuId        = 7;
    long long constexpr startTs             = 123;
    long long constexpr preservedNextSince  = 777;
    long long constexpr watchUpdateFreqUsec = 1000000;
    double constexpr watchMaxKeepAge        = 30.0;
    int constexpr watchMaxKeepSamples       = 4;

    auto fieldCallback = [](unsigned int, dcgmFieldValue_v1 *, int, void *) {
        return 0;
    };
    auto entityFieldCallback = [](dcgm_field_entity_group_t, dcgm_field_eid_t, dcgmFieldValue_v1 *, int, void *) {
        return 0;
    };

    GIVEN("watch-field group APIs with isolated local faults and valid dispatch-boundary envelopes")
    {
        WHEN("required group ids are missing or otherwise valid requests reach dispatch")
        {
            auto const watchBadGroupRet = dcgmWatchFields(
                handle, DCGM_GROUP_NULL, fieldGroupId, watchUpdateFreqUsec, watchMaxKeepAge, watchMaxKeepSamples);
            auto const unwatchBadGroupRet = dcgmUnwatchFields(handle, DCGM_GROUP_NULL, fieldGroupId);
            auto const watchDispatchRet   = dcgmWatchFields(
                handle, dispatchGroupId, fieldGroupId, watchUpdateFreqUsec, watchMaxKeepAge, watchMaxKeepSamples);
            auto const unwatchDispatchRet = dcgmUnwatchFields(handle, dispatchGroupId, fieldGroupId);

            THEN("local group validation is precise and valid envelopes report the dispatch-layer failure")
            {
                CHECK(watchBadGroupRet == DCGM_ST_BADPARAM);
                CHECK(unwatchBadGroupRet == DCGM_ST_BADPARAM);
                CHECK(watchDispatchRet == DCGM_ST_BADPARAM);
                CHECK(unwatchDispatchRet == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("historical and latest-value callback APIs")
    {
        unsigned short fieldIds[] = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        long long nextSince       = preservedNextSince;

        WHEN("callbacks or timestamp outputs are missing")
        {
            auto const fieldNoCallbackRet
                = dcgmGetFieldValuesSince(handle, groupId, startTs, fieldIds, 1, &nextSince, nullptr, nullptr);
            auto const fieldNoNextRet
                = dcgmGetFieldValuesSince(handle, groupId, startTs, fieldIds, 1, nullptr, fieldCallback, nullptr);
            auto const valuesNoCallbackRet
                = dcgmGetValuesSince(handle, groupId, gpuId, startTs, &nextSince, nullptr, nullptr);
            auto const valuesNoNextRet
                = dcgmGetValuesSince(handle, groupId, gpuId, startTs, nullptr, fieldCallback, nullptr);
            auto const valuesV2NoCallbackRet
                = dcgmGetValuesSince_v2(handle, groupId, gpuId, startTs, &nextSince, nullptr, nullptr);
            auto const latestNoCallbackRet   = dcgmGetLatestValues(handle, groupId, gpuId, nullptr, nullptr);
            auto const latestV2NoCallbackRet = dcgmGetLatestValues_v2(handle, groupId, gpuId, nullptr, nullptr);

            THEN("each missing local dependency is rejected before dispatch")
            {
                CHECK(fieldNoCallbackRet == DCGM_ST_BADPARAM);
                CHECK(fieldNoNextRet == DCGM_ST_BADPARAM);
                CHECK(valuesNoCallbackRet == DCGM_ST_BADPARAM);
                CHECK(valuesNoNextRet == DCGM_ST_BADPARAM);
                CHECK(valuesV2NoCallbackRet == DCGM_ST_BADPARAM);
                CHECK(latestNoCallbackRet == DCGM_ST_BADPARAM);
                CHECK(latestV2NoCallbackRet == DCGM_ST_BADPARAM);
                CHECK(nextSince == preservedNextSince);
            }
        }

        WHEN("valid callback envelopes reach dispatch without a hostengine")
        {
            nextSince = preservedNextSince;
            auto const fieldDispatchRet
                = dcgmGetFieldValuesSince(handle, groupId, startTs, fieldIds, 1, &nextSince, fieldCallback, nullptr);
            nextSince = preservedNextSince;
            auto const valuesDispatchRet
                = dcgmGetValuesSince(handle, groupId, gpuId, startTs, &nextSince, fieldCallback, nullptr);
            nextSince = preservedNextSince;
            auto const valuesV2DispatchRet
                = dcgmGetValuesSince_v2(handle, groupId, gpuId, startTs, &nextSince, entityFieldCallback, nullptr);
            auto const latestDispatchRet = dcgmGetLatestValues(handle, groupId, gpuId, fieldCallback, nullptr);
            auto const latestV2DispatchRet
                = dcgmGetLatestValues_v2(handle, groupId, gpuId, entityFieldCallback, nullptr);

            THEN("dispatch failures are reported without invoking the caller callbacks")
            {
                CHECK(fieldDispatchRet == DCGM_ST_BADPARAM);
                CHECK(valuesDispatchRet == DCGM_ST_BADPARAM);
                CHECK(valuesV2DispatchRet == DCGM_ST_BADPARAM);
                CHECK(latestDispatchRet == DCGM_ST_BADPARAM);
                CHECK(latestV2DispatchRet == DCGM_ST_BADPARAM);
                CHECK(nextSince == startTs);
            }
        }
    }

    GIVEN("health-check and introspection payloads")
    {
        unsigned int constexpr preservedIncidentCount = 17;
        long long constexpr preservedBytesUsed        = 1234;
        double constexpr preservedCpuTotal            = 55.5;

        dcgmHealthResponse_t health {};
        health.version       = dcgmHealthResponse_version;
        health.overallHealth = DCGM_HEALTH_RESULT_PASS;
        health.incidentCount = preservedIncidentCount;
        dcgmIntrospectMemory_t mem {};
        mem.version   = dcgmIntrospectMemory_version1;
        mem.bytesUsed = preservedBytesUsed;
        dcgmIntrospectCpuUtil_t cpu {};
        cpu.version = dcgmIntrospectCpuUtil_version1;
        cpu.total   = preservedCpuTotal;

        WHEN("null, stale, and dispatch-boundary requests are submitted")
        {
            auto const healthNullRet     = dcgmHealthCheck(handle, 1, nullptr);
            auto badHealth               = health;
            badHealth.version            = 0;
            auto const healthVersionRet  = dcgmHealthCheck(handle, 1, &badHealth);
            auto const healthDispatchRet = dcgmHealthCheck(handle, 1, &health);

            auto const memNullRet     = dcgmIntrospectGetHostengineMemoryUsage(handle, nullptr, 0);
            auto badMem               = mem;
            badMem.version            = 0;
            auto const memVersionRet  = dcgmIntrospectGetHostengineMemoryUsage(handle, &badMem, 0);
            auto const memDispatchRet = dcgmIntrospectGetHostengineMemoryUsage(handle, &mem, 1);

            auto const cpuNullRet     = dcgmIntrospectGetHostengineCpuUtilization(handle, nullptr, 0);
            auto badCpu               = cpu;
            badCpu.version            = 0;
            auto const cpuVersionRet  = dcgmIntrospectGetHostengineCpuUtilization(handle, &badCpu, 0);
            auto const cpuDispatchRet = dcgmIntrospectGetHostengineCpuUtilization(handle, &cpu, 1);

            THEN("local validation is precise and failed dispatch preserves caller-visible values")
            {
                CHECK(healthNullRet == DCGM_ST_BADPARAM);
                CHECK(healthVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(healthDispatchRet == DCGM_ST_BADPARAM);
                CHECK(health.version == dcgmHealthResponse_version);
                CHECK(health.overallHealth == DCGM_HEALTH_RESULT_PASS);
                CHECK(health.incidentCount == preservedIncidentCount);

                CHECK(memNullRet == DCGM_ST_BADPARAM);
                CHECK(memVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(memDispatchRet == DCGM_ST_BADPARAM);
                CHECK(mem.version == dcgmIntrospectMemory_version1);
                CHECK(mem.bytesUsed == preservedBytesUsed);

                CHECK(cpuNullRet == DCGM_ST_BADPARAM);
                CHECK(cpuVersionRet == DCGM_ST_VER_MISMATCH);
                CHECK(cpuDispatchRet == DCGM_ST_BADPARAM);
                CHECK(cpu.version == dcgmIntrospectCpuUtil_version1);
                CHECK(cpu.total == preservedCpuTotal);
            }
        }
    }
}

SCENARIO("DcgmApi: NVML injection wrappers validate local envelopes before dispatch")
{
    REQUIRE(dcgmInit() == DCGM_ST_OK);
    DcgmNs::Defer shutdown([] { CHECK(dcgmShutdown() == DCGM_ST_OK); });

    dcgmHandle_t handle = static_cast<dcgmHandle_t>(0);

    GIVEN("field-value injection requests")
    {
        long long constexpr preservedFieldValue = 44;
        dcgmInjectFieldValue_t value {};
        value.version   = dcgmInjectFieldValue_version;
        value.fieldId   = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        value.fieldType = DCGM_FT_INT64;
        value.value.i64 = preservedFieldValue;

        WHEN("the payload is missing or dispatch fails")
        {
            auto const nullRet     = dcgmInjectEntityFieldValueToNvml(handle, DCGM_FE_GPU, 0, nullptr);
            auto const dispatchRet = dcgmInjectEntityFieldValueToNvml(handle, DCGM_FE_GPU, 0, &value);

            THEN("null payloads are rejected and failed dispatch leaves the caller value intact")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(dispatchRet == DCGM_ST_BADPARAM);
                CHECK(value.version == dcgmInjectFieldValue_version);
                CHECK(value.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(value.value.i64 == preservedFieldValue);
            }
        }
    }

    GIVEN("fake NVML GPU creation requests")
    {
        WHEN("the request reaches dispatch without a hostengine")
        {
            unsigned int constexpr injectedGpuIndex = 7;
            auto const ret                          = dcgmCreateNvmlInjectionGpu(handle, injectedGpuIndex);

            THEN("the dispatch-layer failure is reported through the public wrapper")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }
    }

#ifdef INJECTION_LIBRARY_AVAILABLE
    GIVEN("single-call NVML device injection requests")
    {
        injectNvmlVal_t extraKey {};
        injectNvmlRet_t injectedRet {};
        injectedRet.nvmlRet    = NVML_ERROR_UNKNOWN;
        injectedRet.valueCount = 0;

        WHEN("individual broad-contract preconditions are violated")
        {
            auto tooManyExtraKeys    = injectedRet;
            auto tooManyValues       = injectedRet;
            tooManyValues.valueCount = NVML_INJECTION_MAX_VALUES + 1;

            auto successWithoutValue    = injectedRet;
            successWithoutValue.nvmlRet = NVML_SUCCESS;

            auto const nullKeyRet        = dcgmInjectNvmlDevice(handle, 0, nullptr, nullptr, 0, &injectedRet);
            auto const nullReturnRet     = dcgmInjectNvmlDevice(handle, 0, "Key", nullptr, 0, nullptr);
            auto const successNoValueRet = dcgmInjectNvmlDevice(handle, 0, "Key", nullptr, 0, &successWithoutValue);
            auto const extraCountRet
                = dcgmInjectNvmlDevice(handle, 0, "Key", nullptr, NVML_INJECTION_MAX_EXTRA_KEYS + 1, &tooManyExtraKeys);
            auto const extraPointerOnlyRet = dcgmInjectNvmlDevice(handle, 0, "Key", &extraKey, 0, &injectedRet);
            auto const valueCountRet       = dcgmInjectNvmlDevice(handle, 0, "Key", nullptr, 0, &tooManyValues);

            THEN("each invalid envelope is rejected locally with a precise status")
            {
                CHECK(nullKeyRet == DCGM_ST_BADPARAM);
                CHECK(nullReturnRet == DCGM_ST_BADPARAM);
                CHECK(successNoValueRet == DCGM_ST_BADPARAM);
                CHECK(extraCountRet == DCGM_ST_BADPARAM);
                CHECK(extraPointerOnlyRet == DCGM_ST_BADPARAM);
                CHECK(valueCountRet == DCGM_ST_BADPARAM);
            }
        }

        WHEN("a syntactically valid request reaches dispatch")
        {
            auto const ret = dcgmInjectNvmlDevice(handle, 0, "Key", nullptr, 0, &injectedRet);

            THEN("the wrapper reports the dispatch failure without mutating the request")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                CHECK(injectedRet.nvmlRet == NVML_ERROR_UNKNOWN);
                CHECK(injectedRet.valueCount == 0);
            }
        }
    }

    GIVEN("multi-call NVML device injection requests")
    {
        injectNvmlVal_t extraKey {};
        injectNvmlRet_t injectedRets[NVML_INJECTION_MAX_RETURNS + 1] {};
        injectedRets[0].nvmlRet    = NVML_ERROR_UNKNOWN;
        injectedRets[0].valueCount = 0;

        WHEN("individual broad-contract preconditions are violated")
        {
            auto tooManyValues       = injectedRets[0];
            tooManyValues.valueCount = NVML_INJECTION_MAX_VALUES + 1;

            auto successWithoutValue    = injectedRets[0];
            successWithoutValue.nvmlRet = NVML_SUCCESS;

            auto const nullKeyRet = dcgmInjectNvmlDeviceForFollowingCalls(handle, 0, nullptr, nullptr, 0, nullptr, 0);
            auto const extraCountRet = dcgmInjectNvmlDeviceForFollowingCalls(
                handle, 0, "Key", nullptr, NVML_INJECTION_MAX_EXTRA_KEYS + 1, nullptr, 0);
            auto const extraPointerOnlyRet
                = dcgmInjectNvmlDeviceForFollowingCalls(handle, 0, "Key", &extraKey, 0, nullptr, 0);
            auto const retCountRet = dcgmInjectNvmlDeviceForFollowingCalls(
                handle, 0, "Key", nullptr, 0, injectedRets, NVML_INJECTION_MAX_RETURNS + 1);
            auto const missingReturnsRet
                = dcgmInjectNvmlDeviceForFollowingCalls(handle, 0, "Key", nullptr, 0, nullptr, 1);
            auto const valueCountRet
                = dcgmInjectNvmlDeviceForFollowingCalls(handle, 0, "Key", nullptr, 0, &tooManyValues, 1);
            auto const successNoValueRet
                = dcgmInjectNvmlDeviceForFollowingCalls(handle, 0, "Key", nullptr, 0, &successWithoutValue, 1);

            THEN("each invalid envelope is rejected locally with a precise status")
            {
                CHECK(nullKeyRet == DCGM_ST_BADPARAM);
                CHECK(extraCountRet == DCGM_ST_BADPARAM);
                CHECK(extraPointerOnlyRet == DCGM_ST_BADPARAM);
                CHECK(retCountRet == DCGM_ST_BADPARAM);
                CHECK(missingReturnsRet == DCGM_ST_BADPARAM);
                CHECK(valueCountRet == DCGM_ST_BADPARAM);
                CHECK(successNoValueRet == DCGM_ST_BADPARAM);
            }
        }

        WHEN("a syntactically valid request and call-count query reach dispatch")
        {
            unsigned int constexpr preservedCallCount = 99;
            injectNvmlFuncCallCounts_t callCounts {};
            callCounts.numFuncs = preservedCallCount;

            auto const injectRet = dcgmInjectNvmlDeviceForFollowingCalls(handle, 0, "Key", nullptr, 0, injectedRets, 1);
            auto const nullCountsRet = dcgmGetNvmlInjectFuncCallCount(handle, nullptr);
            auto const countsRet     = dcgmGetNvmlInjectFuncCallCount(handle, &callCounts);

            THEN("local validation is precise and failed dispatch preserves caller-owned buffers")
            {
                CHECK(injectRet == DCGM_ST_BADPARAM);
                CHECK(nullCountsRet == DCGM_ST_BADPARAM);
                CHECK(countsRet == DCGM_ST_BADPARAM);
                CHECK(injectedRets[0].nvmlRet == NVML_ERROR_UNKNOWN);
                CHECK(injectedRets[0].valueCount == 0);
                CHECK(callCounts.numFuncs == preservedCallCount);
            }
        }
    }
#endif
}
