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

#include <Defer.hpp>
#include <DeviceMonitor.h>
#include <MockDcgmOps.h>
#include <StderrCapture.hpp>
#include <StdoutCapture.hpp>
#include <dcgm_fields.h>

#include <array>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

std::vector<dcgmGroupEntityPair_t> helperParseCpuEntities(std::string const &inputCpuIds);

namespace
{
class TestDeviceMonitor : public DeviceMonitor
{
public:
    using DeviceMonitor::DeviceMonitor;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class DeviceMonitorTestFixture
{
public:
    TestDeviceMonitor MakeMonitor(std::string const &entityIds,
                                  std::string const &groupId,
                                  std::string const &fieldIds,
                                  bool listMode                   = false,
                                  int fieldGroupId                = -1,
                                  std::chrono::milliseconds delay = std::chrono::milliseconds(1))
    {
        return TestDeviceMonitor(
            "localhost", entityIds, groupId, fieldIds, fieldGroupId, delay, 1, listMode, &mockDcgmOps);
    }

    void AddGpuInt64Field(unsigned int gpuId, unsigned short fieldId, long long value)
    {
        dcgmGroupEntityPair_t gpuEntity { DCGM_FE_GPU, gpuId };
        DcgmNs::MockDcgmEntity gpu(gpuEntity);
        InjectInt64Field(gpu, fieldId, value);
        mockDcgmOps.AddMockedEntity(std::move(gpu));
    }

    DcgmNs::MockDcgmOps mockDcgmOps;
    dcgmHandle_t handle = static_cast<dcgmHandle_t>(0xc8763);

private:
    static void InjectInt64Field(DcgmNs::MockDcgmEntity &entity, unsigned short fieldId, long long value)
    {
        dcgmFieldValue_v2 fieldValue {};
        fieldValue.version       = dcgmFieldValue_version2;
        fieldValue.entityGroupId = entity.GetEntity().entityGroupId;
        fieldValue.entityId      = entity.GetEntity().entityId;
        fieldValue.fieldId       = fieldId;
        fieldValue.fieldType     = DCGM_FT_INT64;
        fieldValue.status        = DCGM_ST_OK;
        fieldValue.value.i64     = value;
        entity.InjectFieldValue(fieldId, fieldValue);
    }
};

constexpr unsigned int c_noComputeInstanceId = static_cast<unsigned int>(-1);

} // namespace


SCENARIO_METHOD(DeviceMonitorTestFixture, "DeviceMonitor: lists field metadata")
{
    GIVEN("a DeviceMonitor created with the list option")
    {
        auto monitor = MakeMonitor("-1", "-1", "-1", true);

        WHEN("the command executes")
        {
            StdoutCapture capture;

            dcgmReturn_t result = monitor.RunWithHandle(handle);
            auto output         = capture.GetOutput();

            THEN("field metadata is listed")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(output.find("Long Name") != std::string::npos);
                CHECK(output.find("Short Name") != std::string::npos);
                CHECK(output.find("Field ID") != std::string::npos);
                CHECK(output.size() > 100);
            }
        }
    }
}

SCENARIO_METHOD(DeviceMonitorTestFixture, "DeviceMonitor: rejects non-numeric group identifiers")
{
    GIVEN("a DeviceMonitor created with a group identifier that is not special or numeric")
    {
        auto monitor = MakeMonitor("-1", "not-a-group", "100");

        WHEN("the command executes")
        {
            dcgmReturn_t result = monitor.RunWithHandle(handle);

            THEN("the command reports a bad parameter before creating field watches")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }
}

SCENARIO_METHOD(DeviceMonitorTestFixture,
                "DeviceMonitor: validates entity and field group inputs before watching fields")
{
    GIVEN("a DeviceMonitor with a malformed entity selector")
    {
        auto monitor = MakeMonitor("gpu:", "-1", "100");

        WHEN("the command executes")
        {
            dcgmReturn_t result = monitor.RunWithHandle(handle);

            THEN("the malformed selector is rejected before creating watches")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("a DeviceMonitor with neither field ids nor a field group")
    {
        auto monitor = MakeMonitor("-1", "-1", "-1");

        WHEN("the command executes")
        {
            dcgmReturn_t result = monitor.RunWithHandle(handle);

            THEN("the field-group validation reports a bad parameter")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("a DeviceMonitor with an invalid field id list")
    {
        auto monitor = MakeMonitor("-1", "-1", "not-a-field");

        WHEN("the command executes")
        {
            dcgmReturn_t result = monitor.RunWithHandle(handle);

            THEN("the parser failure is returned before watch creation")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }
}


SCENARIO_METHOD(DeviceMonitorTestFixture, "DeviceMonitor: uses mock groups, field groups, and watches")
{
    REQUIRE(DcgmFieldsInit() == 0);
    DcgmNs::Defer cleanupFields([] { DcgmFieldsTerm(); });

    AddGpuInt64Field(0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 72);
    handle = mockDcgmOps.m_handleId;

    SECTION("one iteration prints headers and records watch parameters")
    {
        auto monitor = MakeMonitor(
            "-1", "-1", std::to_string(DCGM_FI_DEV_GPU_TEMP_CELSIUS), false, -1, std::chrono::milliseconds(25));
        StdoutCapture capture;

        auto const result = monitor.RunWithHandle(handle);
        auto output       = capture.GetOutput();

        CHECK(result == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_groupManagementState.createCallCount == 1);
        CHECK(mockDcgmOps.m_groupManagementState.lastGroupType == DCGM_GROUP_DEFAULT);
        CHECK(mockDcgmOps.m_fieldGroupState.createCallCount == 1);
        CHECK(mockDcgmOps.m_fieldGroupState.watchCallCount == 1);
        CHECK(mockDcgmOps.m_fieldGroupState.lastUpdateFreq == 25000);
        CHECK(mockDcgmOps.m_fieldGroupState.lastMaxKeepSamples == 2);
        CHECK(output.find("#Entity") != std::string::npos);
    }

    SECTION("watch setup failures propagate configured DCGM errors")
    {
        std::array<dcgmReturn_t, 3> const cases {
            DCGM_ST_REQUIRES_ROOT,
            DCGM_ST_INSUFFICIENT_DRIVER_VERSION,
            DCGM_ST_BADPARAM,
        };

        for (auto const &tc : cases)
        {
            CAPTURE(tc);
            mockDcgmOps.m_fieldGroupState.watchReturn = tc;
            auto monitor = MakeMonitor("-1", "-1", std::to_string(DCGM_FI_DEV_GPU_TEMP_CELSIUS));
            StderrCapture capture;

            CHECK(monitor.RunWithHandle(handle) == tc);
            auto const output = capture.GetOutput();
            CHECK(output.find("Error setting watches") != std::string::npos);
            mockDcgmOps.m_fieldGroupState.watchReturn = DCGM_ST_OK;
        }
    }
}

SCENARIO_METHOD(DeviceMonitorTestFixture, "DeviceMonitor: reports group and field-group lookup failures")
{
    REQUIRE(DcgmFieldsInit() == 0);
    DcgmNs::Defer cleanupFields([] { DcgmFieldsTerm(); });

    AddGpuInt64Field(0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 72);
    handle = mockDcgmOps.m_handleId;

    SECTION("sorted entity lookup failure is returned before watching fields")
    {
        mockDcgmOps.m_groupInfoState.returnCode = DCGM_ST_NOT_CONFIGURED;
        auto monitor                            = MakeMonitor("-1", "-1", std::to_string(DCGM_FI_DEV_GPU_TEMP_CELSIUS));

        CHECK(monitor.RunWithHandle(handle) == DCGM_ST_NOT_CONFIGURED);
        CHECK(mockDcgmOps.m_fieldGroupState.watchCallCount == 0);
    }

    SECTION("explicit field group lookup failure is reported as a bad parameter")
    {
        constexpr int c_fieldGroupId                = 77;
        mockDcgmOps.m_fieldGroupState.getInfoReturn = DCGM_ST_NOT_CONFIGURED;
        auto monitor                                = MakeMonitor("-1", "-1", "-1", false, c_fieldGroupId);

        CHECK(monitor.RunWithHandle(handle) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.m_fieldGroupState.getInfoCallCount == 1);
        CHECK(mockDcgmOps.m_fieldGroupState.lastRequestedInfoId == c_fieldGroupId);
        CHECK(mockDcgmOps.m_fieldGroupState.watchCallCount == 0);
    }
}

SCENARIO("DeviceMonitor: parses comma-separated CPU entities")
{
    GIVEN("a comma-separated CPU entity list")
    {
        constexpr unsigned int c_cpuId0 = 0;
        constexpr unsigned int c_cpuId1 = 7;
        constexpr unsigned int c_cpuId2 = 42;
        std::string const input
            = std::to_string(c_cpuId0) + "," + std::to_string(c_cpuId1) + "," + std::to_string(c_cpuId2);

        WHEN("the list is parsed")
        {
            auto entities = helperParseCpuEntities(input);

            THEN("each token becomes a CPU entity in input order")
            {
                REQUIRE(entities.size() == 3);
                CHECK(entities[0].entityGroupId == DCGM_FE_CPU);
                CHECK(entities[0].entityId == c_cpuId0);
                CHECK(entities[1].entityGroupId == DCGM_FE_CPU);
                CHECK(entities[1].entityId == c_cpuId1);
                CHECK(entities[2].entityGroupId == DCGM_FE_CPU);
                CHECK(entities[2].entityId == c_cpuId2);
            }
        }
    }
}

SCENARIO("DeviceMonitor: merges GPU and MIG entities")
{
    GIVEN("group info containing GPUs and MIG entities")
    {
        constexpr dcgmGroupEntityPair_t c_gpu0 { DCGM_FE_GPU, 0 };
        constexpr dcgmGroupEntityPair_t c_gpu2 { DCGM_FE_GPU, 2 };
        constexpr dcgmGroupEntityPair_t c_gpuI { DCGM_FE_GPU_I, 40 };
        constexpr dcgmGroupEntityPair_t c_gpuCi { DCGM_FE_GPU_CI, 50 };
        constexpr dcgmGroupEntityPair_t c_switch { DCGM_FE_SWITCH, 9 };
        constexpr unsigned int c_instanceId        = 1;
        constexpr unsigned int c_computeInstanceId = 3;

        dcgmGroupInfo_t groupInfo {};
        groupInfo.version       = dcgmGroupInfo_version;
        groupInfo.count         = 5;
        groupInfo.entityList[0] = c_gpuCi;
        groupInfo.entityList[1] = c_gpu2;
        groupInfo.entityList[2] = c_gpuI;
        groupInfo.entityList[3] = c_gpu0;
        groupInfo.entityList[4] = c_switch;

        dcgmMigHierarchy_v2 hierarchy {};
        hierarchy.version                                  = dcgmMigHierarchy_version2;
        hierarchy.count                                    = 2;
        hierarchy.entityList[0].entity                     = c_gpuI;
        hierarchy.entityList[0].parent                     = c_gpu0;
        hierarchy.entityList[0].info.nvmlGpuIndex          = c_gpu0.entityId;
        hierarchy.entityList[0].info.nvmlInstanceId        = c_instanceId;
        hierarchy.entityList[0].info.nvmlComputeInstanceId = c_noComputeInstanceId;
        hierarchy.entityList[1].entity                     = c_gpuCi;
        hierarchy.entityList[1].parent                     = c_gpuI;
        hierarchy.entityList[1].info.nvmlGpuIndex          = c_gpu0.entityId;
        hierarchy.entityList[1].info.nvmlInstanceId        = c_instanceId;
        hierarchy.entityList[1].info.nvmlComputeInstanceId = c_computeInstanceId;

        SECTION("MIG entities are sorted under their parent GPU")
        {
            dcgmReturn_t result = DeviceMonitor::MergeGpuAndMigEntities(groupInfo, hierarchy);

            THEN("the GPU and its MIG entities are grouped before other entities")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(groupInfo.entityList[0].entityGroupId == c_gpu0.entityGroupId);
                CHECK(groupInfo.entityList[0].entityId == c_gpu0.entityId);
                CHECK(groupInfo.entityList[1].entityGroupId == c_gpuI.entityGroupId);
                CHECK(groupInfo.entityList[1].entityId == c_gpuI.entityId);
                CHECK(groupInfo.entityList[2].entityGroupId == c_gpuCi.entityGroupId);
                CHECK(groupInfo.entityList[2].entityId == c_gpuCi.entityId);
                CHECK(groupInfo.entityList[3].entityGroupId == c_gpu2.entityGroupId);
                CHECK(groupInfo.entityList[3].entityId == c_gpu2.entityId);
                CHECK(groupInfo.entityList[4].entityGroupId == c_switch.entityGroupId);
                CHECK(groupInfo.entityList[4].entityId == c_switch.entityId);
            }
        }

        SECTION("entities are sorted by group and id when no MIG hierarchy is available")
        {
            hierarchy.count = 0;

            dcgmReturn_t result = DeviceMonitor::MergeGpuAndMigEntities(groupInfo, hierarchy);

            THEN("non-MIG entities keep a stable group/id ordering")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(groupInfo.entityList[0].entityGroupId == c_gpu0.entityGroupId);
                CHECK(groupInfo.entityList[0].entityId == c_gpu0.entityId);
                CHECK(groupInfo.entityList[1].entityGroupId == c_gpu2.entityGroupId);
                CHECK(groupInfo.entityList[1].entityId == c_gpu2.entityId);
                CHECK(groupInfo.entityList[2].entityGroupId == c_switch.entityGroupId);
                CHECK(groupInfo.entityList[2].entityId == c_switch.entityId);
                CHECK(groupInfo.entityList[3].entityGroupId == c_gpuI.entityGroupId);
                CHECK(groupInfo.entityList[3].entityId == c_gpuI.entityId);
                CHECK(groupInfo.entityList[4].entityGroupId == c_gpuCi.entityGroupId);
                CHECK(groupInfo.entityList[4].entityId == c_gpuCi.entityId);
            }
        }
    }
}

SCENARIO("DeviceMonitor: merges GPU and MIG entities orders MIG siblings by hierarchy coordinates")
{
    GIVEN("a group with multiple compute instances from the same GPU instance")
    {
        constexpr dcgmGroupEntityPair_t c_gpu { DCGM_FE_GPU, 3 };
        constexpr dcgmGroupEntityPair_t c_gpuI { DCGM_FE_GPU_I, 41 };
        constexpr dcgmGroupEntityPair_t c_gpuCi1 { DCGM_FE_GPU_CI, 51 };
        constexpr dcgmGroupEntityPair_t c_gpuCi2 { DCGM_FE_GPU_CI, 52 };
        constexpr dcgmGroupEntityPair_t c_gpuCi3 { DCGM_FE_GPU_CI, 53 };
        constexpr unsigned int c_instanceId         = 2;
        constexpr unsigned int c_computeInstanceId1 = 1;
        constexpr unsigned int c_computeInstanceId2 = 2;
        constexpr unsigned int c_computeInstanceId3 = 3;

        dcgmGroupInfo_t groupInfo {};
        groupInfo.version       = dcgmGroupInfo_version;
        groupInfo.count         = 5;
        groupInfo.entityList[0] = c_gpuCi3;
        groupInfo.entityList[1] = c_gpuCi1;
        groupInfo.entityList[2] = c_gpuI;
        groupInfo.entityList[3] = c_gpu;
        groupInfo.entityList[4] = c_gpuCi2;

        dcgmMigHierarchy_v2 hierarchy {};
        hierarchy.version = dcgmMigHierarchy_version2;
        hierarchy.count   = 4;

        hierarchy.entityList[0].entity                     = c_gpuI;
        hierarchy.entityList[0].parent                     = c_gpu;
        hierarchy.entityList[0].info.nvmlGpuIndex          = c_gpu.entityId;
        hierarchy.entityList[0].info.nvmlInstanceId        = c_instanceId;
        hierarchy.entityList[0].info.nvmlComputeInstanceId = c_noComputeInstanceId;

        hierarchy.entityList[1].entity                     = c_gpuCi3;
        hierarchy.entityList[1].parent                     = c_gpuI;
        hierarchy.entityList[1].info.nvmlGpuIndex          = c_gpu.entityId;
        hierarchy.entityList[1].info.nvmlInstanceId        = c_instanceId;
        hierarchy.entityList[1].info.nvmlComputeInstanceId = c_computeInstanceId3;

        hierarchy.entityList[2].entity                     = c_gpuCi1;
        hierarchy.entityList[2].parent                     = c_gpuI;
        hierarchy.entityList[2].info.nvmlGpuIndex          = c_gpu.entityId;
        hierarchy.entityList[2].info.nvmlInstanceId        = c_instanceId;
        hierarchy.entityList[2].info.nvmlComputeInstanceId = c_computeInstanceId1;

        hierarchy.entityList[3].entity                     = c_gpuCi2;
        hierarchy.entityList[3].parent                     = c_gpuI;
        hierarchy.entityList[3].info.nvmlGpuIndex          = c_gpu.entityId;
        hierarchy.entityList[3].info.nvmlInstanceId        = c_instanceId;
        hierarchy.entityList[3].info.nvmlComputeInstanceId = c_computeInstanceId2;

        WHEN("the entities are merged")
        {
            dcgmReturn_t result = DeviceMonitor::MergeGpuAndMigEntities(groupInfo, hierarchy);

            THEN("the GPU instance precedes compute instances sorted by compute-instance id")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(groupInfo.entityList[0].entityGroupId == c_gpu.entityGroupId);
                CHECK(groupInfo.entityList[0].entityId == c_gpu.entityId);
                CHECK(groupInfo.entityList[1].entityGroupId == c_gpuI.entityGroupId);
                CHECK(groupInfo.entityList[1].entityId == c_gpuI.entityId);
                CHECK(groupInfo.entityList[2].entityGroupId == c_gpuCi1.entityGroupId);
                CHECK(groupInfo.entityList[2].entityId == c_gpuCi1.entityId);
                CHECK(groupInfo.entityList[3].entityGroupId == c_gpuCi2.entityGroupId);
                CHECK(groupInfo.entityList[3].entityId == c_gpuCi2.entityId);
                CHECK(groupInfo.entityList[4].entityGroupId == c_gpuCi3.entityGroupId);
                CHECK(groupInfo.entityList[4].entityId == c_gpuCi3.entityId);
            }
        }
    }
}

SCENARIO("DeviceMonitor: merges GPU and MIG entities orders MIG entities across multiple GPUs")
{
    GIVEN("a group with MIG entities from different parent GPUs")
    {
        constexpr dcgmGroupEntityPair_t c_gpu0 { DCGM_FE_GPU, 0 };
        constexpr dcgmGroupEntityPair_t c_gpu1 { DCGM_FE_GPU, 1 };
        constexpr dcgmGroupEntityPair_t c_gpuI70 { DCGM_FE_GPU_I, 70 };
        constexpr dcgmGroupEntityPair_t c_gpuI71 { DCGM_FE_GPU_I, 71 };
        constexpr dcgmGroupEntityPair_t c_gpuCi80 { DCGM_FE_GPU_CI, 80 };
        constexpr dcgmGroupEntityPair_t c_gpuCi81 { DCGM_FE_GPU_CI, 81 };
        constexpr dcgmGroupEntityPair_t c_switch { DCGM_FE_SWITCH, 4 };
        constexpr unsigned int c_gpu1InstanceId = 2;
        constexpr unsigned int c_gpu0InstanceId = 1;

        dcgmGroupInfo_t groupInfo {};
        groupInfo.version       = dcgmGroupInfo_version;
        groupInfo.count         = 6;
        groupInfo.entityList[0] = c_gpuCi80;
        groupInfo.entityList[1] = c_gpu1;
        groupInfo.entityList[2] = c_gpuI70;
        groupInfo.entityList[3] = c_gpu0;
        groupInfo.entityList[4] = c_gpuCi81;
        groupInfo.entityList[5] = c_switch;

        dcgmMigHierarchy_v2 hierarchy {};
        hierarchy.version = dcgmMigHierarchy_version2;
        hierarchy.count   = 3;

        hierarchy.entityList[0].entity                     = c_gpuCi80;
        hierarchy.entityList[0].parent                     = c_gpuI70;
        hierarchy.entityList[0].info.nvmlGpuIndex          = c_gpu1.entityId;
        hierarchy.entityList[0].info.nvmlInstanceId        = c_gpu1InstanceId;
        hierarchy.entityList[0].info.nvmlComputeInstanceId = 0;

        hierarchy.entityList[1].entity                     = c_gpuI70;
        hierarchy.entityList[1].parent                     = c_gpu1;
        hierarchy.entityList[1].info.nvmlGpuIndex          = c_gpu1.entityId;
        hierarchy.entityList[1].info.nvmlInstanceId        = c_gpu1InstanceId;
        hierarchy.entityList[1].info.nvmlComputeInstanceId = c_noComputeInstanceId;

        hierarchy.entityList[2].entity                     = c_gpuCi81;
        hierarchy.entityList[2].parent                     = c_gpuI71;
        hierarchy.entityList[2].info.nvmlGpuIndex          = c_gpu0.entityId;
        hierarchy.entityList[2].info.nvmlInstanceId        = c_gpu0InstanceId;
        hierarchy.entityList[2].info.nvmlComputeInstanceId = 0;

        WHEN("the entities are merged")
        {
            dcgmReturn_t result = DeviceMonitor::MergeGpuAndMigEntities(groupInfo, hierarchy);

            THEN("each MIG entity is sorted by its hierarchy coordinates while non-MIG entities remain ordered")
            {
                std::vector<dcgmGroupEntityPair_t> expected {
                    c_gpu0, c_gpuCi81, c_gpu1, c_gpuI70, c_gpuCi80, c_switch,
                };

                CHECK(result == DCGM_ST_OK);
                REQUIRE(groupInfo.count == expected.size());
                for (size_t i = 0; i < expected.size(); ++i)
                {
                    CAPTURE(i);
                    CHECK(groupInfo.entityList[i].entityGroupId == expected[i].entityGroupId);
                    CHECK(groupInfo.entityList[i].entityId == expected[i].entityId);
                }
            }
        }
    }
}


SCENARIO("DeviceMonitor: merges GPU and MIG entities handles hierarchy gaps and duplicates")
{
    GIVEN("a group containing duplicate GPUs, orphan MIG entries, and non-GPU entities")
    {
        constexpr dcgmGroupEntityPair_t c_gpu { DCGM_FE_GPU, 4 };
        constexpr dcgmGroupEntityPair_t c_gpuI { DCGM_FE_GPU_I, 89 };
        constexpr dcgmGroupEntityPair_t c_gpuCiKnown { DCGM_FE_GPU_CI, 90 };
        constexpr dcgmGroupEntityPair_t c_gpuCiOrphan { DCGM_FE_GPU_CI, 88 };
        constexpr dcgmGroupEntityPair_t c_switch { DCGM_FE_SWITCH, 1 };
        constexpr dcgmGroupEntityPair_t c_cpu { DCGM_FE_CPU, 2 };
        constexpr unsigned int c_instanceId        = 2;
        constexpr unsigned int c_computeInstanceId = 7;

        dcgmGroupInfo_t groupInfo {};
        groupInfo.version       = dcgmGroupInfo_version;
        groupInfo.count         = 7;
        groupInfo.entityList[0] = c_gpuCiKnown;
        groupInfo.entityList[1] = c_cpu;
        groupInfo.entityList[2] = c_gpu;
        groupInfo.entityList[3] = c_gpuI;
        groupInfo.entityList[4] = c_gpu;
        groupInfo.entityList[5] = c_switch;
        groupInfo.entityList[6] = c_gpuCiOrphan;

        dcgmMigHierarchy_v2 hierarchy {};
        hierarchy.version = dcgmMigHierarchy_version2;
        hierarchy.count   = 2;

        hierarchy.entityList[0].entity                     = c_gpuCiKnown;
        hierarchy.entityList[0].parent                     = c_gpuI;
        hierarchy.entityList[0].info.nvmlGpuIndex          = c_gpu.entityId;
        hierarchy.entityList[0].info.nvmlInstanceId        = c_instanceId;
        hierarchy.entityList[0].info.nvmlComputeInstanceId = c_computeInstanceId;

        hierarchy.entityList[1].entity                     = c_gpuI;
        hierarchy.entityList[1].parent                     = c_gpu;
        hierarchy.entityList[1].info.nvmlGpuIndex          = c_gpu.entityId;
        hierarchy.entityList[1].info.nvmlInstanceId        = c_instanceId;
        hierarchy.entityList[1].info.nvmlComputeInstanceId = c_noComputeInstanceId;

        WHEN("the entities are merged")
        {
            dcgmReturn_t result = DeviceMonitor::MergeGpuAndMigEntities(groupInfo, hierarchy);

            THEN("known MIG entries sort by hierarchy and orphan entries fall back to group/id ordering")
            {
                std::vector<dcgmGroupEntityPair_t> expected {
                    c_gpu, c_gpu, c_gpuI, c_gpuCiKnown, c_switch, c_gpuCiOrphan, c_cpu,
                };

                CHECK(result == DCGM_ST_OK);
                REQUIRE(groupInfo.count == expected.size());
                for (size_t i = 0; i < expected.size(); ++i)
                {
                    CAPTURE(i);
                    CHECK(groupInfo.entityList[i].entityGroupId == expected[i].entityGroupId);
                    CHECK(groupInfo.entityList[i].entityId == expected[i].entityId);
                }
            }
        }
    }
}

SCENARIO("DeviceMonitor: merges GPU and MIG entities preserves pure entity ordering without MIG data")
{
    GIVEN("a group containing only non-MIG entity types")
    {
        constexpr dcgmGroupEntityPair_t c_cpu { DCGM_FE_CPU, 7 };
        constexpr dcgmGroupEntityPair_t c_switch { DCGM_FE_SWITCH, 1 };
        constexpr dcgmGroupEntityPair_t c_gpu2 { DCGM_FE_GPU, 2 };
        constexpr dcgmGroupEntityPair_t c_connectX { DCGM_FE_CONNECTX, 3 };
        constexpr dcgmGroupEntityPair_t c_gpu0 { DCGM_FE_GPU, 0 };

        dcgmGroupInfo_t groupInfo {};
        groupInfo.version       = dcgmGroupInfo_version;
        groupInfo.count         = 5;
        groupInfo.entityList[0] = c_cpu;
        groupInfo.entityList[1] = c_switch;
        groupInfo.entityList[2] = c_gpu2;
        groupInfo.entityList[3] = c_connectX;
        groupInfo.entityList[4] = c_gpu0;

        dcgmMigHierarchy_v2 hierarchy {};
        hierarchy.version = dcgmMigHierarchy_version2;

        WHEN("the group is merged")
        {
            dcgmReturn_t result = DeviceMonitor::MergeGpuAndMigEntities(groupInfo, hierarchy);

            THEN("entities are ordered by entity group and id without changing the count")
            {
                std::vector<dcgmGroupEntityPair_t> expected {
                    c_gpu0, c_gpu2, c_switch, c_cpu, c_connectX,
                };

                CHECK(result == DCGM_ST_OK);
                REQUIRE(groupInfo.count == expected.size());
                for (size_t i = 0; i < expected.size(); ++i)
                {
                    CAPTURE(i);
                    CHECK(groupInfo.entityList[i].entityGroupId == expected[i].entityGroupId);
                    CHECK(groupInfo.entityList[i].entityId == expected[i].entityId);
                }
            }
        }
    }
}

SCENARIO("DeviceMonitor: merges GPU and MIG entities is stable for empty and single-entity groups")
{
    GIVEN("an empty group and an empty hierarchy")
    {
        dcgmGroupInfo_t groupInfo {};
        groupInfo.version = dcgmGroupInfo_version;

        dcgmMigHierarchy_v2 hierarchy {};
        hierarchy.version = dcgmMigHierarchy_version2;

        WHEN("the empty group is merged")
        {
            dcgmReturn_t result = DeviceMonitor::MergeGpuAndMigEntities(groupInfo, hierarchy);

            THEN("the operation succeeds without manufacturing entities")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(groupInfo.count == 0);
            }
        }
    }

    GIVEN("a single ConnectX entity")
    {
        constexpr dcgmGroupEntityPair_t c_connectX { DCGM_FE_CONNECTX, 42 };
        constexpr unsigned int c_entityCount = 1;

        dcgmGroupInfo_t groupInfo {};
        groupInfo.version       = dcgmGroupInfo_version;
        groupInfo.count         = c_entityCount;
        groupInfo.entityList[0] = c_connectX;

        dcgmMigHierarchy_v2 hierarchy {};
        hierarchy.version = dcgmMigHierarchy_version2;

        WHEN("the single entity is merged")
        {
            dcgmReturn_t result = DeviceMonitor::MergeGpuAndMigEntities(groupInfo, hierarchy);

            THEN("the entity is preserved exactly")
            {
                CHECK(result == DCGM_ST_OK);
                REQUIRE(groupInfo.count == c_entityCount);
                CHECK(groupInfo.entityList[0].entityGroupId == c_connectX.entityGroupId);
                CHECK(groupInfo.entityList[0].entityId == c_connectX.entityId);
            }
        }
    }
}
