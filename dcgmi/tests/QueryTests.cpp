/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <DcgmStringHelpers.h>
#include <DcgmVariantHelper.hpp>
#include <DcgmiOutput.h>
#include <MockDcgmOps.h>
#include <Query.h>
#include <dcgm_agent.h>
#include <dcgmi_common.h>

#include <cstring>
#include <memory>
#include <string.h>

struct CpuRangeCase
{
    uint64_t bitmask[DCGM_MAX_NUM_CPU_CORES / sizeof(uint64_t) / CHAR_BIT];
    uint32_t bitmaskNumBits;
    std::vector<std::pair<uint32_t, uint32_t>> rangeSet;
};

TEST_CASE("HelperGetCpuRangesFromBitmask")
{
    CpuRangeCase testRanges[]
        = { /* Nominal cases here are defined as having a valid range and a configuration
             * of CPUs, of which there are nearly infinite so we will take ranges that
             * check the counts, range widths, firsts/lasts, and gaps.
             */
            { { 0xFFFFFFFFFFFFFFFFull, 0x00000000000000FFull }, DCGM_MAX_NUM_CPU_CORES, { { 0, 71 } } },
            { { 0x0F0E0C08ull }, DCGM_MAX_NUM_CPU_CORES, { { 3, 3 }, { 10, 11 }, { 17, 19 }, { 24, 27 } } },
            /* Edge cases here are defined as having a valid bitmask, but an unusual
             * characteristic. E.g.: having no CPUs specified, extremas of the mask,
             * and being completely full.
             */
            { { 0ull }, DCGM_MAX_NUM_CPU_CORES, {} },
            { { 1ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x8000000000000000ull },
              DCGM_MAX_NUM_CPU_CORES,
              { { 0, 0 }, { 1023, 1023 } } },
            { { (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull) },
              DCGM_MAX_NUM_CPU_CORES,
              { { 0, 1023 } } },
            { // special edge case where the mask size is zero, should just return no ranges
              { 0ull },
              0,
              {} }
          };

    for (unsigned long i = 0; i < (sizeof(testRanges) / sizeof(CpuRangeCase)); i++)
    {
        CpuRangeCase tc = testRanges[i];
        auto ranges     = HelperGetCpuRangesFromBitmask(tc.bitmask, tc.bitmaskNumBits);
        REQUIRE(ranges == tc.rangeSet);
    }
}

TEST_CASE("HelperBuildCpuListFromRanges")
{
    GIVEN("CPU core ranges")
    {
        SECTION("No ranges")
        {
            CHECK(HelperBuildCpuListFromRanges({}).empty());
        }

        SECTION("Single range")
        {
            CHECK(HelperBuildCpuListFromRanges({ { 4, 9 } }) == "Cores: 4-9");
        }

        SECTION("Multiple ranges")
        {
            CHECK(HelperBuildCpuListFromRanges({ { 0, 0 }, { 2, 5 }, { 9, 11 } }) == "Cores: 0-0,2-5,9-11");
        }
    }
}

TEST_CASE("HelperPopulateGpuDeviceOutput")
{
    dcgmDeviceAttributes_t attrs {};
    attrs.version = dcgmDeviceAttributes_version;
    std::strncpy(attrs.identifiers.deviceName, "TestGPU", sizeof(attrs.identifiers.deviceName) - 1);
    std::strncpy(attrs.identifiers.pciBusId, "0000:01:00.0", sizeof(attrs.identifiers.pciBusId) - 1);
    std::strncpy(attrs.identifiers.uuid, "GPU-12345678", sizeof(attrs.identifiers.uuid) - 1);

    auto getOutput = [&](DcgmEntityStatus_t status, bool showAll) {
        DcgmiOutputTree tree(20, 60);
        HelperPopulateGpuDeviceOutput(tree, 0, status, attrs, showAll);
        return tree.str();
    };

    SECTION("Device info always present")
    {
        auto output = getOutput(DcgmEntityStatusOk, false);
        CHECK(output.find("TestGPU") != std::string::npos);
        CHECK(output.find("0000:01:00.0") != std::string::npos);
        CHECK(output.find("GPU-12345678") != std::string::npos);
    }

    SECTION("Status shown only when showAll=true and status != OK")
    {
        CHECK(getOutput(DcgmEntityStatusOk, false).find("Status:") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusOk, true).find("Status:") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusDetached, false).find("Status:") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusDetached, true).find("Status: DETACHED") != std::string::npos);
    }

    SECTION("Cached suffix shown for non-OK, non-Fake status")
    {
        CHECK(getOutput(DcgmEntityStatusOk, false).find("(last known)") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusFake, true).find("(last known)") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusDetached, true).find("(last known)") != std::string::npos);
    }
}

TEST_CASE("Query::DisplayDeviceInfo rejects invalid attribute flags before querying DCGM")
{
    GIVEN("a Query object and an invalid attribute selector")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        Query query(&mockDcgmOps);

        SECTION("Duplicate attribute")
        {
            WHEN("device info is requested")
            {
                CHECK(query.DisplayDeviceInfo(0, 0, "aa") == DCGM_ST_BADPARAM);
            }
        }

        SECTION("Unknown attribute")
        {
            WHEN("device info is requested")
            {
                CHECK(query.DisplayDeviceInfo(0, 0, "az") == DCGM_ST_BADPARAM);
            }
        }

        SECTION("Too many attributes")
        {
            WHEN("device info is requested")
            {
                CHECK(query.DisplayDeviceInfo(0, 0, "aptcwa") == DCGM_ST_BADPARAM);
            }
        }
    }
}

TEST_CASE("Query::DisplayDeviceInfo")
{
    GIVEN("a Query object and mocked GPU attributes")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        Query query(&mockDcgmOps);
        auto handle = static_cast<dcgmHandle_t>(0x90);

        SECTION("Identifier attributes are displayed")
        {
            CoutCapture capture;

            CHECK(query.DisplayDeviceInfo(handle, 0, "a") == DCGM_ST_OK);
            CHECK(capture.str().find("GPU ID: 0") != std::string::npos);
            CHECK(capture.str().find("Mock GPU") != std::string::npos);
            CHECK(capture.str().find("GPU-mock-uuid") != std::string::npos);
        }

        SECTION("Power, thermal, and workload attributes are displayed")
        {
            CoutCapture capture;

            CHECK(query.DisplayDeviceInfo(handle, 0, "ptw") == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_queryState.workloadCallCount == 1);
            CHECK(capture.str().find("Power") != std::string::npos);
            CHECK(capture.str().find("Temperature") != std::string::npos);
        }

        SECTION("Workload power profile failures do not fail the outer device info request")
        {
            mockDcgmOps.m_queryState.workloadReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;

            CHECK(query.DisplayDeviceInfo(handle, 0, "w") == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_queryState.workloadCallCount == 1);
            CHECK(capture.str().find("Unable to get GPU info") != std::string::npos);
        }
    }
}

TEST_CASE("Query::DisplayDiscoveredDevices")
{
    GIVEN("a Query object and mocked entity lists")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        Query query(&mockDcgmOps);
        auto handle = static_cast<dcgmHandle_t>(0x93);

        SECTION("Empty entity lists are displayed successfully")
        {
            mockDcgmOps.m_entityListState.returnCode  = DCGM_ST_OK;
            mockDcgmOps.m_entityListState.gpuEntities = std::vector<dcgm_field_eid_t> {};
            CoutCapture capture;

            CHECK(query.DisplayDiscoveredDevices(handle, false) == DCGM_ST_OK);
            CHECK(capture.str().find("0 NvSwitches found.") != std::string::npos);
            CHECK(capture.str().find("0 ConnectX found.") != std::string::npos);
            CHECK(capture.str().find("0 CPUs found.") != std::string::npos);
        }

        SECTION("GPU list errors are returned")
        {
            mockDcgmOps.m_entityListState.returnCode  = DCGM_ST_BADPARAM;
            mockDcgmOps.m_entityListState.gpuEntities = std::vector<dcgm_field_eid_t> {};
            CoutCapture capture;

            CHECK(query.DisplayDiscoveredDevices(handle, false) == DCGM_ST_BADPARAM);
            CHECK(capture.str().find("Cannot get GPU list") != std::string::npos);
        }
    }
}

struct QueryTestFixture
{
    struct DiscoveryOutput
    {
        dcgmReturn_t result;
        std::string output;
    };

    void AddCpuWithoutModel()
    {
        mockDcgmOps.AddMockedEntity(DcgmNs::MockDcgmEntity(c_cpuEntity));
    }

    void AddCpuWithModel(char const *model)
    {
        DcgmNs::MockDcgmEntity cpu(c_cpuEntity);
        dcgmFieldValue_v2 modelValue {};
        modelValue.version       = dcgmFieldValue_version2;
        modelValue.entityGroupId = c_cpuEntity.entityGroupId;
        modelValue.entityId      = c_cpuEntity.entityId;
        modelValue.fieldId       = DCGM_FI_DEV_CPU_MODEL;
        modelValue.fieldType     = DCGM_FT_STRING;
        modelValue.status        = DCGM_ST_OK;
        SafeCopyTo(modelValue.value.str, model);
        cpu.InjectFieldValue(DCGM_FI_DEV_CPU_MODEL, modelValue);
        mockDcgmOps.AddMockedEntity(cpu);
    }

    DiscoveryOutput RunDiscovery()
    {
        CoutCapture capture;
        dcgmReturn_t const result = query.DisplayDiscoveredDevices(mockDcgmOps.m_handleId, false);
        return { result, capture.str() };
    }

    static constexpr dcgmGroupEntityPair_t c_cpuEntity { .entityGroupId = DCGM_FE_CPU, .entityId = 0 };
    DcgmNs::MockDcgmOps mockDcgmOps;
    Query query { &mockDcgmOps };
};

SCENARIO_METHOD(QueryTestFixture, "Query: displays the CPU model during discovery")
{
    constexpr char c_testCpuModel[]       = "Test CPU Model";
    constexpr char c_unknownModelOutput[] = "Name: Unknown";

    GIVEN("a CPU with a model value")
    {
        AddCpuWithModel(c_testCpuModel);

        WHEN("discovered devices are displayed")
        {
            auto const discovery = RunDiscovery();

            THEN("the model is displayed")
            {
                CHECK(discovery.result == DCGM_ST_OK);
                CHECK(discovery.output.find(std::string("Name: ") + c_testCpuModel) != std::string::npos);
            }
        }
    }

    GIVEN("a CPU whose model field has no cached data")
    {
        AddCpuWithoutModel();

        WHEN("discovered devices are displayed")
        {
            auto const discovery = RunDiscovery();

            THEN("the model is reported as unknown")
            {
                CHECK(discovery.result == DCGM_ST_OK);
                CHECK(discovery.output.find(c_unknownModelOutput) != std::string::npos);
            }
        }
    }

    GIVEN("a CPU whose model field is queried")
    {
        AddCpuWithModel(c_testCpuModel);

        WHEN("discovered devices are displayed")
        {
            auto const discovery = RunDiscovery();

            THEN("the temporary CPU model watch is removed before returning")
            {
                CHECK(discovery.result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps.m_fieldGroupState.unwatchCallCount == 1);
                CHECK(mockDcgmOps.m_fieldGroupState.lastUnwatchedGroupId
                      == mockDcgmOps.m_fieldGroupState.lastWatchedGroupId);
                CHECK(mockDcgmOps.m_fieldGroupState.lastUnwatchedFieldGroupId
                      == mockDcgmOps.m_fieldGroupState.lastWatchedFieldGroupId);
            }
        }
    }

    GIVEN("a CPU whose model field contains the DCGM blank string")
    {
        AddCpuWithModel(DCGM_STR_BLANK);

        WHEN("discovered devices are displayed")
        {
            auto const discovery = RunDiscovery();

            THEN("the model is reported as unknown")
            {
                CHECK(discovery.result == DCGM_ST_OK);
                CHECK(discovery.output.find(c_unknownModelOutput) != std::string::npos);
            }
        }
    }
}

TEST_CASE("Query::DisplayCpuInfo")
{
    GIVEN("a Query object and mocked CPU hierarchy")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        Query query(&mockDcgmOps);
        auto handle = static_cast<dcgmHandle_t>(0x91);

        SECTION("CPU attributes are displayed")
        {
            CoutCapture capture;

            CHECK(query.DisplayCpuInfo(handle, 0, "a") == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_queryState.cpuHierarchyCallCount == 1);
            CHECK(capture.str().find("CPU ID: 0") != std::string::npos);
            CHECK(capture.str().find("cpu-serial-0") != std::string::npos);
            CHECK(capture.str().find("Cores: 0-3") != std::string::npos);
        }

        SECTION("Missing CPUs are rejected")
        {
            CoutCapture capture;

            CHECK(query.DisplayCpuInfo(handle, 3, "a") == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_queryState.cpuHierarchyCallCount == 1);
        }

        SECTION("CPU hierarchy failures are returned")
        {
            mockDcgmOps.m_queryState.cpuHierarchyReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;

            CHECK(query.DisplayCpuInfo(handle, 0, "a") == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_queryState.cpuHierarchyCallCount == 1);
        }
    }
}

TEST_CASE("Query::DisplayGroupInfo")
{
    GIVEN("a Query object and mocked group info")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        Query query(&mockDcgmOps);
        auto handle = static_cast<dcgmHandle_t>(0x92);
        mockDcgmOps.m_groupInfoState.overrideInfo
            = DcgmNs::MockDcgmOps::MakeGroupInfo("query-group", { { DCGM_FE_GPU, 0 }, { DCGM_FE_CPU, 1 } });

        SECTION("Verbose group output displays GPU details and non-GPU entities")
        {
            CoutCapture capture;

            CHECK(query.DisplayGroupInfo(handle, 7, "a", true) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
            CHECK(mockDcgmOps.m_groupInfoState.lastGroupId == 7);
            CHECK(capture.str().find("Device info") != std::string::npos);
            CHECK(capture.str().find("GPU ID: 0") != std::string::npos);
            CHECK(capture.str().find("CPU id: 1") != std::string::npos);
        }

        SECTION("Non-verbose group output summarizes shared attributes")
        {
            CoutCapture capture;

            CHECK(query.DisplayGroupInfo(handle, 7, "a", false) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
            CHECK(capture.str().find("Group of 2 GPUs") != std::string::npos);
            CHECK(capture.str().find("Non-homogenous settings across group. Use with -v flag") != std::string::npos);
        }

        SECTION("Empty groups are accepted")
        {
            mockDcgmOps.m_groupInfoState.overrideInfo = DcgmNs::MockDcgmOps::MakeGroupInfo("query-group");
            CoutCapture capture;

            CHECK(query.DisplayGroupInfo(handle, 7, "a", false) == DCGM_ST_OK);
            CHECK(capture.str().find("No devices in group") != std::string::npos);
        }

        SECTION("Group lookup failures are returned")
        {
            mockDcgmOps.m_groupInfoState.returnCode = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;

            CHECK(query.DisplayGroupInfo(handle, 7, "a", false) == DCGM_ST_NOT_CONFIGURED);
            CHECK(capture.str().find("The Group is not found") != std::string::npos);
        }
    }
}

TEST_CASE("FormatMigHierarchy")
{
    GIVEN("a MIG hierarchy with a GPU instance and compute instance")
    {
        dcgmMigHierarchy_v2 hierarchy {};
        hierarchy.version = dcgmMigHierarchy_version2;

        auto &instance                = hierarchy.entityList[hierarchy.count++];
        instance.entity.entityGroupId = DCGM_FE_GPU_I;
        instance.entity.entityId      = 11;
        instance.parent.entityGroupId = DCGM_FE_GPU;
        instance.parent.entityId      = 3;
        std::strncpy(instance.info.gpuUuid, "GPU-test-uuid", sizeof(instance.info.gpuUuid) - 1);
        instance.info.nvmlGpuIndex          = 7;
        instance.info.nvmlInstanceId        = 2;
        instance.info.nvmlComputeInstanceId = static_cast<unsigned int>(-1);

        auto &computeInstance                      = hierarchy.entityList[hierarchy.count++];
        computeInstance.entity.entityGroupId       = DCGM_FE_GPU_CI;
        computeInstance.entity.entityId            = 12;
        computeInstance.parent.entityGroupId       = DCGM_FE_GPU_I;
        computeInstance.parent.entityId            = 11;
        computeInstance.info.nvmlGpuIndex          = 7;
        computeInstance.info.nvmlInstanceId        = 2;
        computeInstance.info.nvmlComputeInstanceId = 4;

        WHEN("the hierarchy is formatted")
        {
            auto output = FormatMigHierarchy(hierarchy);

            CHECK(output.find("Instance Hierarchy") != std::string::npos);
            CHECK(output.find("GPU GPU-test-uuid (EntityID: 3)") != std::string::npos);
            CHECK(output.find("GPU Instance (EntityID: 11)") != std::string::npos);
            CHECK(output.find("Compute Instance (EntityID: 12)") != std::string::npos);
            CHECK(output.find("CI 7/2/4") != std::string::npos);
        }
    }
}
