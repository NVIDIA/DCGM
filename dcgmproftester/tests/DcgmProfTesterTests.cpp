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
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DcgmProfTester.h"
#include "MockDistributedCudaContextRuntime.hpp"
#include "MockDistributedCudaContextRuntimeFactory.hpp"
#include "PhysicalGpu.h"
#include <MockCudaWorkerThreadFactory.hpp>
#include <MockDcgmProfTester.hpp>

#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <ScopedSignalHandler.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace DcgmNs::ProfTester;
using namespace DcgmNs::ProfTester::Test;
using namespace Catch::Matchers;

namespace
{
Arguments_t::Parameters MakeOfflineParameters()
{
    Arguments_t::Parameters parameters {};
    parameters.m_noDcgmValidation = true;
    parameters.m_fieldId          = DCGM_FI_PROF_SM_UTIL_RATIO;
    parameters.m_syncCount        = 3;
    parameters.m_cublas           = true;
    return parameters;
}

auto MakeValidGpuLatestValuesHandler()
{
    return [](dcgmHandle_t,
              dcgmGroupEntityPair_t *entity,
              unsigned int,
              unsigned short *fieldIds,
              unsigned int,
              unsigned int,
              dcgmFieldValue_v2 *value) {
        value->status = DCGM_ST_OK;
        if (*fieldIds == DCGM_FI_DEV_GPU_VIRTUAL_MODE)
        {
            value->value.i64 = DCGM_GPU_VIRTUALIZATION_MODE_NONE;
        }
        else if (*fieldIds == DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY)
        {
            value->value.i64 = (8ULL << 16);
        }
        else if (*fieldIds == DCGM_FI_CUDA_GPU_VISIBLE_DEVICES)
        {
            std::snprintf(value->value.str, sizeof(value->value.str), "cuda-%u", entity->entityId);
        }
        return DCGM_ST_OK;
    };
}

struct DcgmProfTesterFixture
{
    MockDistributedCudaContextRuntimeState workerRuntimeState;
    MockDistributedCudaContextRuntime workerRuntime { &workerRuntimeState };
    std::unique_ptr<DcgmNs::MockDcgmOps> dcgmOps = std::make_unique<DcgmNs::MockDcgmOps>();
    DcgmNs::MockDcgmOps *mockDcgmOps             = dcgmOps.get();
    MockCudaOps cudaOps;
    std::unique_ptr<DcgmProfTester> tester = std::make_unique<DcgmProfTester>(
        std::make_unique<MockDistributedCudaContextRuntimeFactory>(&workerRuntimeState),
        std::make_unique<MockCudaWorkerThreadFactory>(),
        std::move(dcgmOps),
        &cudaOps);
};
} // namespace

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester processes an empty argument snapshot list")
{
    GIVEN("a newly constructed profiler tester with no parsed argument snapshots")
    {
        unsigned int callbackCount { 0 };

        WHEN("the queued argument snapshots are processed")
        {
            dcgmReturn_t const result = tester->Process([&callbackCount](Arguments_t const &) {
                ++callbackCount;
                return DCGM_ST_OK;
            });

            THEN("processing succeeds without invoking the callback")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(callbackCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester initializes command-line snapshots for processing")
{
    GIVEN("two requested fields and shared command-line options")
    {
        char program[]  = "dcgmproftester";
        char fields[]   = "1001,1002";
        char gpuIds[]   = "4,6";
        char duration[] = "2";
        char report[]   = "0.5";
        char *argv[]    = { program,
                            const_cast<char *>("-t"),
                            fields,
                            const_cast<char *>("-i"),
                            gpuIds,
                            const_cast<char *>("--no-dcgm-validation"),
                            const_cast<char *>("--duration"),
                            duration,
                            const_cast<char *>("--report"),
                            report };
        REQUIRE(tester->Init(static_cast<int>(std::size(argv)), argv) == DCGM_ST_OK);

        WHEN("the parsed snapshots are processed")
        {
            std::vector<unsigned int> fieldIds;
            std::vector<std::vector<unsigned int>> gpuLists;
            auto const result = tester->Process([&](Arguments_t const &arguments) {
                fieldIds.push_back(arguments.m_parameters.m_fieldId);
                gpuLists.push_back(arguments.m_gpuIds);
                CHECK(arguments.m_parameters.m_noDcgmValidation);
                CHECK(arguments.m_parameters.m_duration == 2.0);
                CHECK(arguments.m_parameters.m_reportInterval == 0.5);
                return DCGM_ST_OK;
            });

            THEN("one callback is invoked per field with shared command-line options")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_hostEngineState.startEmbeddedCallCount == 1);
                CHECK(fieldIds
                      == std::vector<unsigned int> { DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, DCGM_FI_PROF_SM_UTIL_RATIO });
                REQUIRE(gpuLists.size() == 2);
                CHECK(gpuLists[0] == std::vector<unsigned int> { 4, 6 });
                CHECK(gpuLists[1] == std::vector<unsigned int> { 4, 6 });
            }
        }
    }

    GIVEN("a parsed command line whose processor reports failures")
    {
        char program[] = "dcgmproftester";
        char fields[]  = "1001,1002";
        char *argv[]   = { program, const_cast<char *>("-t"), fields };
        REQUIRE(tester->Init(static_cast<int>(std::size(argv)), argv) == DCGM_ST_OK);

        WHEN("one callback invocation fails")
        {
            unsigned int callbackCount = 0;
            auto const result          = tester->Process([&](Arguments_t const &) {
                ++callbackCount;
                return callbackCount == 1 ? DCGM_ST_BADPARAM : DCGM_ST_OK;
            });

            THEN("processing continues but reports a generic failure")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(callbackCount == 2);
            }
        }
    }

    GIVEN("an invalid command line")
    {
        char program[] = "dcgmproftester";
        char fields[]  = "not-a-field";
        char *argv[]   = { program, const_cast<char *>("-t"), fields };

        WHEN("initialization parses the arguments")
        {
            auto const result = tester->Init(static_cast<int>(std::size(argv)), argv);

            THEN("the parse error is returned before DCGM is initialized")
            {
                CHECK(result != DCGM_ST_OK);
                CHECK(mockDcgmOps->m_hostEngineState.startEmbeddedCallCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester tracks shared test coordination state")
{
    GIVEN("a newly constructed profiler tester")
    {
        WHEN("the first-tick and next-part values are updated")
        {
            tester->SetFirstTick(true);
            tester->SetNextPart(7);

            THEN("the public queries report the updated values")
            {
                CHECK(tester->IsFirstTick());
                CHECK(tester->GetNextPart() == 7);
            }
        }

        WHEN("the first-tick value is cleared")
        {
            tester->SetFirstTick(true);
            tester->SetFirstTick(false);

            THEN("the public query reports that the current tick is not first")
            {
                CHECK_FALSE(tester->IsFirstTick());
            }
        }
    }
}

namespace
{

Arguments_t MakeArguments(std::vector<unsigned int> gpuIds = {})
{
    Arguments_t arguments {};
    arguments.m_gpuIds                        = std::move(gpuIds);
    arguments.m_parameters                    = MakeOfflineParameters();
    arguments.m_parameters.m_noDcgmValidation = false;
    arguments.m_parameters.m_fieldId          = DCGM_FI_PROF_SM_UTIL_RATIO;
    return arguments;
}

} // namespace

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester shuts down DCGM when its owning scope ends")
{
    GIVEN("an initialized tester with observable DCGM operations")
    {
        unsigned int shutdownCalls { 0 };
        mockDcgmOps->m_hostEngineState.shutdownCallback = [&shutdownCalls] {
            ++shutdownCalls;
        };
        char program[] = "dcgmproftester";
        char *argv[]   = { program };

        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);
        tester.reset();

        THEN("the tester destructor shuts DCGM down exactly once")
        {
            CHECK(shutdownCalls == 1);
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester initializes fake GPU groups")
{
    GIVEN("fake DCGM ops and two discovered GPUs")
    {
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        auto arguments                                          = MakeArguments();
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;

        WHEN("GPU initialization succeeds")
        {
            auto result = tester->InitializeGpus(arguments);

            THEN("DCGM group and field group collaborators observe the GPU entities")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_migHierarchyState.callCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 2);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.lastCreatedFieldIds
                      == std::vector<unsigned short> { DCGM_FI_PROF_SM_UTIL_RATIO });
            }
        }
    }

    GIVEN("fake DCGM ops and an explicit GPU subset")
    {
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        auto arguments                                          = MakeArguments({ 2 });
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;

        WHEN("only one requested GPU is present")
        {
            auto result = tester->InitializeGpus(arguments);

            THEN("only that GPU is added to the DCGM group")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == 1);
            }
        }
    }

    GIVEN("a tester that has already completed one GPU initialization cycle")
    {
        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;
        REQUIRE(tester->InitializeGpus(MakeArguments()) == DCGM_ST_OK);
        REQUIRE(tester->ShutdownGpus() == DCGM_ST_OK);

        auto secondArguments                   = MakeArguments({ 2 });
        secondArguments.m_parameters.m_fieldId = DCGM_FI_PROF_FP32_UTIL_RATIO;
        auto const previousAddCalls            = mockDcgmOps->m_groupManagementState.addCallCount;
        auto const previousFieldCreates        = mockDcgmOps->m_fieldGroupState.createCallCount;

        WHEN("a later snapshot narrows the GPU list and changes the field")
        {
            auto const result = tester->InitializeGpus(secondArguments);

            THEN("only the requested GPU is registered with the updated field group")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == previousAddCalls + 1);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == previousFieldCreates + 1);
                CHECK(mockDcgmOps->m_fieldGroupState.lastCreatedFieldIds
                      == std::vector<unsigned short> { DCGM_FI_PROF_FP32_UTIL_RATIO });
                CHECK(mockDcgmOps->m_groupManagementState.lastEntityGroup == DCGM_FE_GPU);
                CHECK(mockDcgmOps->m_groupManagementState.lastEntityId == 2);
            }
        }
    }

    GIVEN("initialized GPUs that remain owned across argument snapshots")
    {
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;
        REQUIRE(tester->InitializeGpus(MakeArguments()) == DCGM_ST_OK);

        auto retargeted                   = MakeArguments();
        retargeted.m_parameters.m_fieldId = DCGM_FI_PROF_FP32_UTIL_RATIO;
        auto const previousFieldCreates   = mockDcgmOps->m_fieldGroupState.createCallCount;

        WHEN("a later snapshot keeps the same GPUs but changes the field")
        {
            auto const result = tester->InitializeGpus(retargeted);

            THEN("the updated field group is created for the retained GPU set")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == previousFieldCreates + 1);
                CHECK(mockDcgmOps->m_fieldGroupState.lastCreatedFieldIds
                      == std::vector<unsigned short> { DCGM_FI_PROF_FP32_UTIL_RATIO });
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester handles sparse requested GPU sets")
{
    GIVEN("an explicit GPU list containing one present GPU and one missing GPU")
    {
        mockDcgmOps->m_discoveryState.allDevices                = { 2 };
        auto arguments                                          = MakeArguments({ 2, 99 });
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;

        WHEN("GPUs are initialized")
        {
            auto const result = tester->InitializeGpus(arguments);

            THEN("only the discovered requested GPU is kept in the DCGM group")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.lastEntityGroup == DCGM_FE_GPU);
                CHECK(mockDcgmOps->m_groupManagementState.lastEntityId == 2);
            }
        }
    }

    GIVEN("one enumerated GPU fails physical GPU initialization")
    {
        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        mockDcgmOps->m_discoveryState.allDevices            = { 1, 2 };
        mockDcgmOps->m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();

        DcgmNs::MockDcgmEntity healthyGpu({ DCGM_FE_GPU, 1 });
        healthyGpu.SetDevAttr(DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes());
        mockDcgmOps->AddMockedEntity(healthyGpu);

        WHEN("all GPUs are initialized")
        {
            auto const result = tester->InitializeGpus(MakeArguments());

            THEN("the failed GPU is removed and the healthy GPU remains testable")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.lastEntityGroup == DCGM_FE_GPU);
                CHECK(mockDcgmOps->m_groupManagementState.lastEntityId == 1);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester initializes MIG hierarchy entities")
{
    GIVEN("one physical GPU with one GPU instance and two compute instances")
    {
        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        mockDcgmOps->m_discoveryState.allDevices                = { 3 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

        auto &hierarchy                = mockDcgmOps->m_migHierarchyState.hierarchy;
        hierarchy.version              = dcgmMigHierarchy_version2;
        hierarchy.count                = 3;
        hierarchy.entityList[0].entity = { DCGM_FE_GPU_I, 30 };
        hierarchy.entityList[0].parent = { DCGM_FE_GPU, 3 };
        hierarchy.entityList[1].entity = { DCGM_FE_GPU_CI, 31 };
        hierarchy.entityList[1].parent = { DCGM_FE_GPU_I, 30 };
        hierarchy.entityList[2].entity = { DCGM_FE_GPU_CI, 32 };
        hierarchy.entityList[2].parent = { DCGM_FE_GPU_I, 30 };

        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 3 }));
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU_I, 30 }));
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU_CI, 31 }));
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU_CI, 32 }));

        WHEN("GPUs are initialized")
        {
            auto const result = tester->InitializeGpus(MakeArguments());

            dcgmGroupInfo_t groupInfo {};
            groupInfo.version = dcgmGroupInfo_version;
            REQUIRE(mockDcgmOps->dcgmGroupGetInfo(mockDcgmOps->m_handleId, 1, &groupInfo) == DCGM_ST_OK);

            THEN("GPU, GPU-instance, and compute-instance entities are added to the DCGM group")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 4);
                REQUIRE(groupInfo.count == 4);
                CHECK(groupInfo.entityList[0].entityGroupId == DCGM_FE_GPU_I);
                CHECK(groupInfo.entityList[0].entityId == 30);
                CHECK(groupInfo.entityList[1].entityGroupId == DCGM_FE_GPU_CI);
                CHECK(groupInfo.entityList[1].entityId == 31);
                CHECK(groupInfo.entityList[2].entityGroupId == DCGM_FE_GPU_CI);
                CHECK(groupInfo.entityList[2].entityId == 32);
                CHECK(groupInfo.entityList[3].entityGroupId == DCGM_FE_GPU);
                CHECK(groupInfo.entityList[3].entityId == 3);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester propagates GPU initialization failures")
{
    GIVEN("device enumeration fails")
    {
        mockDcgmOps->m_discoveryState.allDevicesReturn = DCGM_ST_CONNECTION_NOT_VALID;

        WHEN("GPUs are initialized")
        {
            auto result = tester->InitializeGpus(MakeArguments());

            THEN("the DCGM error is returned before group creation")
            {
                CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 0);
            }
        }
    }

    GIVEN("DCGM reports no devices")
    {
        mockDcgmOps->m_discoveryState.allDevices.clear();

        WHEN("GPUs are initialized")
        {
            auto result = tester->InitializeGpus(MakeArguments());

            THEN("the operation reports that no work is available")
            {
                CHECK(result != DCGM_ST_OK);
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 0);
            }
        }
    }

    GIVEN("MIG hierarchy enumeration fails after GPUs initialize")
    {
        mockDcgmOps->m_discoveryState.allDevices    = { 1, 2 };
        mockDcgmOps->m_migHierarchyState.returnCode = DCGM_ST_GENERIC_ERROR;

        WHEN("GPUs are initialized")
        {
            auto result = tester->InitializeGpus(MakeArguments());

            THEN("the hierarchy error is returned before group creation")
            {
                CHECK(result != DCGM_ST_OK);
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_migHierarchyState.callCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 0);
            }
        }
    }

    GIVEN("group creation fails after entity discovery")
    {
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_BADPARAM;
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

        WHEN("GPUs are initialized")
        {
            auto result = tester->InitializeGpus(MakeArguments());

            THEN("the group creation error is returned")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 0);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == 0);
            }
        }
    }

    GIVEN("field group creation fails after entity group population")
    {
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;
        mockDcgmOps->m_fieldGroupState.createReturn             = DCGM_ST_BADPARAM;
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

        WHEN("GPUs are initialized")
        {
            auto result = tester->InitializeGpus(MakeArguments());

            THEN("the field group error is returned after GPU entities were added")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 2);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == 1);
            }
        }
    }

    GIVEN("device enumeration succeeds but physical GPU initialization rejects every GPU")
    {
        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        mockDcgmOps->m_discoveryState.allDevices         = { 2 };
        mockDcgmOps->m_deviceAttributesState.returnCode  = DCGM_ST_BADPARAM;
        mockDcgmOps->m_groupManagementState.createReturn = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn    = DCGM_ST_OK;

        WHEN("GPUs are initialized and then shut down")
        {
            auto const initResult                                = tester->InitializeGpus(MakeArguments());
            mockDcgmOps->m_groupManagementState.destroyCallCount = 0;
            mockDcgmOps->m_fieldGroupState.destroyCallCount      = 0;
            auto const shutdownResult                            = tester->ShutdownGpus();

            THEN("no entities are watched and the empty DCGM groups are still destroyed")
            {
                CHECK(initResult == DCGM_ST_OK);
                CHECK(shutdownResult == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 0);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.destroyCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.destroyCallCount == 1);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester handles group lifecycle edge cases")
{
    GIVEN("fake DCGM ops and group entity addition failure")
    {
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_BADPARAM;

        WHEN("GPUs are initialized")
        {
            auto const result = tester->InitializeGpus(MakeArguments());

            THEN("the first add error is returned before field group creation")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 1);
                CHECK(mockDcgmOps->m_groupManagementState.addCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == 0);
            }
        }
    }

    GIVEN("fake DCGM ops with a successful initialization")
    {
        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;

        REQUIRE(tester->InitializeGpus(MakeArguments()) == DCGM_ST_OK);
        mockDcgmOps->m_groupManagementState.destroyCallCount = 0;
        mockDcgmOps->m_fieldGroupState.destroyCallCount      = 0;

        WHEN("group destruction fails")
        {
            mockDcgmOps->m_groupManagementState.destroyReturn = DCGM_ST_GENERIC_ERROR;
            auto const result                                 = tester->ShutdownGpus();

            THEN("shutdown returns the group destruction error after PhysicalGpu field-group cleanup")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(mockDcgmOps->m_groupManagementState.destroyCallCount == 1);
                // Two GPUs × AbortChildren; successful destroy skips the destructor path.
                CHECK(mockDcgmOps->m_fieldGroupState.destroyCallCount == 2);
            }
        }
    }

    GIVEN("fake DCGM ops with a successful initialization and group destroy")
    {
        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;

        REQUIRE(tester->InitializeGpus(MakeArguments()) == DCGM_ST_OK);
        mockDcgmOps->m_groupManagementState.destroyCallCount = 0;
        mockDcgmOps->m_fieldGroupState.destroyCallCount      = 0;

        WHEN("field group destruction fails")
        {
            mockDcgmOps->m_fieldGroupState.destroyReturn = DCGM_ST_BADPARAM;
            auto const result                            = tester->ShutdownGpus();

            THEN("shutdown propagates the field group error after destroying the group")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps->m_groupManagementState.destroyCallCount == 1);
                // Four PhysicalGpu cleanups + one DcgmProfTester::DestroyDcgmGroups call.
                CHECK(mockDcgmOps->m_fieldGroupState.destroyCallCount == 5);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester propagates embedded DCGM startup failures")
{
    GIVEN("fake DCGM lifecycle ops with startup failure")
    {
        char program[] = "dcgmproftester";
        char *argv[]   = { program };

        WHEN("embedded DCGM startup fails")
        {
            mockDcgmOps->m_hostEngineState.startEmbeddedReturn = DCGM_ST_GENERIC_ERROR;

            auto const result = tester->Init(1, argv);

            THEN("the startup error is returned and shutdown is not requested")
            {
                CHECK(result != DCGM_ST_OK);
                CHECK(mockDcgmOps->m_hostEngineState.startEmbeddedCallCount == 1);
                CHECK(mockDcgmOps->m_hostEngineState.stopEmbeddedCallCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester accepts pre-Turing GPUs for compatibility filtering")
{
    GIVEN("fake devices with a pre-Turing chip architecture")
    {
        mockDcgmOps->m_discoveryState.allDevices       = { 0, 1, 2 };
        mockDcgmOps->m_discoveryState.chipArchitecture = DCGM_CHIP_ARCH_VOLTA;

        WHEN("the compatibility list is requested")
        {
            auto result = tester->GetMaxwellPascalVoltaArchGpus();

            THEN("the pre-Turing GPU IDs are returned")
            {
                REQUIRE(result.has_value());
                CHECK(*result == std::vector<dcgm_field_eid_t> { 0, 1, 2 });
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_discoveryState.chipArchitectureCallCount == 3);
            }
        }

        WHEN("chip architecture lookup fails")
        {
            mockDcgmOps->m_discoveryState.chipArchitectureReturn = DCGM_ST_BADPARAM;
            auto result                                          = tester->GetMaxwellPascalVoltaArchGpus();

            THEN("the DCGM error is converted into a diagnostic string")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK_THAT(result.error(), ContainsSubstring("failed to get chip architecture"));
                CHECK_THAT(result.error(), ContainsSubstring("error"));
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester filters explicit GPU architecture lists")
{
    GIVEN("an explicit GPU list parsed from command line arguments")
    {
        mockDcgmOps->m_discoveryState.allDevices               = { 0, 1, 2, 3 };
        mockDcgmOps->m_discoveryState.chipArchitecturesByGpuId = {
            { 7, DCGM_CHIP_ARCH_MAXWELL },
            { 8, DCGM_CHIP_ARCH_TURING },
            { 9, DCGM_CHIP_ARCH_VOLTA },
        };

        char program[] = "dcgmproftester";
        char ids[]     = "7,8,9";
        char *argv[]   = { program, const_cast<char *>("-i"), ids };
        REQUIRE(tester->Init(3, argv) == DCGM_ST_OK);

        WHEN("pre-Turing GPUs are requested")
        {
            auto const result = tester->GetMaxwellPascalVoltaArchGpus();

            THEN("only matching IDs from the explicit list are returned without enumerating all devices")
            {
                REQUIRE(result.has_value());
                CHECK(*result == std::vector<dcgm_field_eid_t> { 7, 9 });
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 0);
                CHECK(mockDcgmOps->m_discoveryState.chipArchitectureCallCount == 3);
            }
        }
    }

    GIVEN("an invalid explicit GPU list")
    {
        char program[] = "dcgmproftester";
        char ids[]     = "7,not-a-gpu";
        char *argv[]   = { program, const_cast<char *>("-i"), ids };
        REQUIRE(tester->Init(3, argv) == DCGM_ST_OK);

        WHEN("the architecture filter parses the explicit list")
        {
            auto const result = tester->GetMaxwellPascalVoltaArchGpus();

            THEN("the parse error is surfaced as a diagnostic string")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK_THAT(result.error(), ContainsSubstring("Invalid GPU ID"));
                CHECK(mockDcgmOps->m_discoveryState.chipArchitectureCallCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester reports architecture-list discovery failures")
{
    GIVEN("all GPUs are requested but DCGM device enumeration fails")
    {
        mockDcgmOps->m_discoveryState.allDevicesReturn = DCGM_ST_CONNECTION_NOT_VALID;

        WHEN("pre-Turing GPUs are requested")
        {
            auto const result = tester->GetMaxwellPascalVoltaArchGpus();

            THEN("the enumeration failure is returned as a diagnostic string")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK_THAT(result.error(), ContainsSubstring("failed to get all devices"));
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_discoveryState.chipArchitectureCallCount == 0);
            }
        }
    }

    GIVEN("an explicit GPU id does not fit in an integer")
    {
        char program[] = "dcgmproftester";
        char ids[]     = "999999999999999999999999999999";
        char *argv[]   = { program, const_cast<char *>("-i"), ids };
        REQUIRE(tester->Init(static_cast<int>(std::size(argv)), argv) == DCGM_ST_OK);

        WHEN("the architecture filter parses the explicit list")
        {
            auto const result = tester->GetMaxwellPascalVoltaArchGpus();

            THEN("the range error is surfaced without querying architectures")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK_THAT(result.error(), ContainsSubstring("GPU ID out of range"));
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 0);
                CHECK(mockDcgmOps->m_discoveryState.chipArchitectureCallCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester filters all discovered GPU architecture lists")
{
    GIVEN("all GPUs are requested and DCGM reports a mixed architecture set")
    {
        mockDcgmOps->m_discoveryState.allDevices               = { 0, 1, 2, 3, 4 };
        mockDcgmOps->m_discoveryState.chipArchitecturesByGpuId = {
            { 0, DCGM_CHIP_ARCH_MAXWELL }, { 1, DCGM_CHIP_ARCH_PASCAL }, { 2, DCGM_CHIP_ARCH_VOLTA },
            { 3, DCGM_CHIP_ARCH_TURING },  { 4, DCGM_CHIP_ARCH_AMPERE },
        };

        WHEN("pre-Turing GPUs are requested")
        {
            auto const result = tester->GetMaxwellPascalVoltaArchGpus();

            THEN("only Maxwell, Pascal, and Volta GPUs are returned")
            {
                REQUIRE(result.has_value());
                CHECK(*result == std::vector<dcgm_field_eid_t> { 0, 1, 2 });
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_discoveryState.chipArchitectureCallCount == 5);
            }
        }
    }

    GIVEN("all GPUs are requested and none are pre-Turing")
    {
        mockDcgmOps->m_discoveryState.allDevices               = { 8, 9 };
        mockDcgmOps->m_discoveryState.chipArchitecturesByGpuId = {
            { 8, DCGM_CHIP_ARCH_TURING },
            { 9, DCGM_CHIP_ARCH_HOPPER },
        };

        WHEN("pre-Turing GPUs are requested")
        {
            auto const result = tester->GetMaxwellPascalVoltaArchGpus();

            THEN("the successful result is an empty list")
            {
                REQUIRE(result.has_value());
                CHECK(result->empty());
                CHECK(mockDcgmOps->m_discoveryState.allDevicesCallCount == 1);
                CHECK(mockDcgmOps->m_discoveryState.chipArchitectureCallCount == 2);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester field watch helpers expose DCGM collaborator outcomes")
{
    GIVEN("initialized GPUs whose field watches are exercised through RunTests")
    {
        auto arguments                                          = MakeArguments();
        arguments.m_parameters.m_noDcgmValidation               = true;
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;
        mockDcgmOps->m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 1 }));
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 2 }));
        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);

        WHEN("watching fields reports each documented profiler setup error")
        {
            std::vector<dcgmReturn_t> const statuses {
                DCGM_ST_REQUIRES_ROOT, DCGM_ST_PROFILING_NOT_SUPPORTED, DCGM_ST_INSUFFICIENT_DRIVER_VERSION,
                DCGM_ST_IN_USE,        DCGM_ST_NOT_SUPPORTED,           DCGM_ST_GROUP_INCOMPATIBLE,
                DCGM_ST_GENERIC_ERROR
            };

            THEN("the same DCGM status is returned to the caller")
            {
                for (auto status : statuses)
                {
                    CAPTURE(status);
                    mockDcgmOps->m_fieldGroupState.watchReturn = status;
                    CHECK(tester->RunTests(0.25, 2.0, DCGM_FI_PROF_TENSOR_UTIL_RATIO, 0) == status);
                }
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == static_cast<int>(statuses.size()));
            }
        }
    }

    GIVEN("a parsed command line with DCGM validation disabled")
    {
        char program[] = "dcgmproftester";
        char fields[]  = "1002";
        char *argv[]   = { program, const_cast<char *>("-t"), fields, const_cast<char *>("--no-dcgm-validation") };
        REQUIRE(tester->Init(static_cast<int>(std::size(argv)), argv) == DCGM_ST_OK);

        mockDcgmOps->m_discoveryState.allDevices                = { 1 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 1 }));

        WHEN("the parsed snapshot initializes GPUs")
        {
            auto const result = tester->Process([this](Arguments_t const &arguments) {
                CHECK(arguments.m_parameters.m_noDcgmValidation);
                return tester->InitializeGpus(arguments);
            });

            THEN("DCGM group and watch collaborators are skipped")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_groupManagementState.createCallCount == 0);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == 0);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 0);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 0);
                CHECK(mockDcgmOps->m_groupManagementState.destroyCallCount == 0);
                CHECK(mockDcgmOps->m_fieldGroupState.destroyCallCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester propagates CUDA-visible mapping failures before worker launch")
{
    GIVEN("initialized GPUs whose CUDA-visible device lookup later fails")
    {
        auto arguments                                          = MakeArguments();
        arguments.m_parameters.m_noDcgmValidation               = true;
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_groupManagementState.createReturn        = DCGM_ST_OK;
        mockDcgmOps->m_groupManagementState.addReturn           = DCGM_ST_OK;

        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);
        mockDcgmOps->m_entitiesGetLatestValuesState.handler = [](dcgmHandle_t,
                                                                 dcgmGroupEntityPair_t *,
                                                                 unsigned int,
                                                                 unsigned short *,
                                                                 unsigned int,
                                                                 unsigned int,
                                                                 dcgmFieldValue_v2 *value) {
            value->status = DCGM_ST_GENERIC_ERROR;
            return DCGM_ST_OK;
        };

        WHEN("tests are run and worker creation needs CUDA visibility")
        {
            auto const result = tester->RunTests(0.1, 0.2, DCGM_FI_PROF_SM_UTIL_RATIO, 0);

            THEN("the mapping failure is converted to a generic run failure before pipes or watchers are used")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(workerRuntime.GetCallCounts().pipe == 0);
                CHECK(workerRuntime.GetCallCounts().fork == 0);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 0);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester runs fake CUDA workers through scheduling and cleanup")
{
    GIVEN("two initialized GPUs and fake worker pipes")
    {
        // Feeder writes can race teardown, ignore SIGPIPE so write returns EPIPE instead of killing the test
        ScopedSignalHandler ignoreSigPipe(SIGPIPE, SIG_IGN);

        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        auto arguments                                          = MakeArguments();
        arguments.m_parameters.m_noDcgmValidation               = true;
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 1 }));
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 2 }));

        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);

        auto waitForWorkers = [this](std::size_t count) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                auto workers = workerRuntime.GetWorkers(count);
                if (workers.size() == count)
                {
                    return workers;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.GetWorkers(count);
        };

        auto waitForCommand = [this](auto const &worker, std::string_view token) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (workerRuntime.HasCommand(worker, token))
                {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.HasCommand(worker, token);
        };

        WHEN("both workers report start, pass, messages, and errors")
        {
            std::atomic_bool feederOk { false };
            std::jthread feeder([&] {
                auto workers = waitForWorkers(2);
                if (workers.size() != 2)
                {
                    return;
                }
                feederOk
                    = workerRuntime.WriteResponse(workers[0], "P\n") && workerRuntime.WriteResponse(workers[1], "P\n")
                      && waitForCommand(workers[0], "R ") && workerRuntime.WriteResponse(workers[0], "P\nM 0\nE 0\n")
                      && waitForCommand(workers[1], "R ") && workerRuntime.WriteResponse(workers[1], "P\nM 0\nE 0\n");
            });

            auto const result = tester->RunTests(0.25, 0.5, DCGM_FI_PROF_SM_UTIL_RATIO, 1);
            feeder.join();
            auto workers = waitForWorkers(2);

            THEN("the run succeeds, watches fields once, and drives both workers")
            {
                REQUIRE(workers.size() == 2);
                CHECK(feederOk.load());
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.GetCallCounts().pipe == 4);
                CHECK(workerRuntime.GetCallCounts().fork == 2);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.lastUpdateFreq == 250000);
                CHECK(mockDcgmOps->m_fieldGroupState.lastMaxKeepAge == 0.5);
                CHECK(workerRuntime.HasCommand(workers[0], "R "));
                CHECK(workerRuntime.HasCommand(workers[0], "M\nE\n"));
                CHECK(workerRuntime.HasCommand(workers[1], "R "));
                CHECK(workerRuntime.HasCommand(workers[1], "M\nE\n"));
            }
        }

        WHEN("watching fields fails after workers are started")
        {
            mockDcgmOps->m_fieldGroupState.watchReturn = DCGM_ST_IN_USE;

            auto const result = tester->RunTests(0.1, 0.2, DCGM_FI_PROF_SM_UTIL_RATIO, 0);
            auto workers      = waitForWorkers(2);

            THEN("the watch error is returned and workers are force-cleaned")
            {
                REQUIRE(workers.size() == 2);
                CHECK(result == DCGM_ST_IN_USE);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 0);
                CHECK(workerRuntime.GetCallCounts().fork == 2);
                CHECK(workerRuntime.HasCommand(workers[0], "X\n"));
                CHECK(workerRuntime.HasCommand(workers[1], "X\n"));
            }
        }

        WHEN("GPU parallelism is limited to one worker at a time")
        {
            std::atomic_bool feederOk { false };
            std::jthread feeder([&] {
                auto workers = waitForWorkers(2);
                if (workers.size() != 2)
                {
                    return;
                }
                feederOk = workerRuntime.WriteResponse(workers[0], "P\n") && waitForCommand(workers[0], "R ")
                           && workerRuntime.WriteResponse(workers[0], "P\nM 0\nE 0\n")
                           && workerRuntime.WriteResponse(workers[1], "P\n") && waitForCommand(workers[1], "R ")
                           && workerRuntime.WriteResponse(workers[1], "P\nM 0\nE 0\n");
            });

            auto const result = tester->RunTests(0.25, 0.5, DCGM_FI_PROF_SM_UTIL_RATIO, 1);
            feeder.join();
            auto workers = waitForWorkers(2);

            THEN("the second worker is queued until the first GPU reports complete")
            {
                REQUIRE(workers.size() == 2);
                CHECK(feederOk.load());
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 1);
                CHECK(workerRuntime.HasCommand(workers[0], "R "));
                CHECK(workerRuntime.HasCommand(workers[1], "R "));
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester serializes bandwidth test runs through fake workers")
{
    GIVEN("two initialized GPUs configured for a PCIe bandwidth field")
    {
        ScopedSignalHandler ignoreSigPipe(SIGPIPE, SIG_IGN);

        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        auto arguments                                          = MakeArguments();
        arguments.m_parameters.m_noDcgmValidation               = true;
        arguments.m_parameters.m_fieldId                        = DCGM_FI_PROF_PCIE_RX_BYTES;
        mockDcgmOps->m_discoveryState.allDevices                = { 1, 2 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 1 }));
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 2 }));

        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);

        auto waitForWorkers = [this](std::size_t count) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                auto workers = workerRuntime.GetWorkers(count);
                if (workers.size() == count)
                {
                    return workers;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.GetWorkers(count);
        };

        auto waitForCommand = [this](auto const &worker, std::string_view token) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (workerRuntime.HasCommand(worker, token))
                {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.HasCommand(worker, token);
        };

        WHEN("a PCIe test is requested with a high parallelism limit")
        {
            std::atomic_bool feederOk { false };
            std::jthread feeder([&] {
                auto workers = waitForWorkers(2);
                if (workers.size() != 2)
                {
                    return;
                }

                auto const completeWorkers = [&]() {
                    workerRuntime.WriteResponse(workers[0], "P\nM 0\nE 0\n");
                    workerRuntime.WriteResponse(workers[1], "P\nM 0\nE 0\n");
                };

                if (!workerRuntime.WriteResponse(workers[0], "P\n") || !workerRuntime.WriteResponse(workers[1], "P\n"))
                {
                    completeWorkers();
                    return;
                }

                auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                int readyIdx        = -1;
                while (std::chrono::steady_clock::now() < deadline)
                {
                    bool const firstReady  = workerRuntime.HasCommand(workers[0], "R ");
                    bool const secondReady = workerRuntime.HasCommand(workers[1], "R ");
                    if (firstReady && secondReady)
                    {
                        completeWorkers();
                        return;
                    }
                    if (firstReady)
                    {
                        readyIdx = 0;
                        break;
                    }
                    if (secondReady)
                    {
                        readyIdx = 1;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if (readyIdx < 0)
                {
                    completeWorkers();
                    return;
                }

                auto const otherIdx = 1 - readyIdx;
                if (workerRuntime.HasCommand(workers[otherIdx], "R "))
                {
                    completeWorkers();
                    return;
                }

                if (!workerRuntime.WriteResponse(workers[readyIdx], "P\nM 0\nE 0\n"))
                {
                    completeWorkers();
                    return;
                }

                feederOk = waitForCommand(workers[otherIdx], "R ")
                           && workerRuntime.WriteResponse(workers[otherIdx], "P\nM 0\nE 0\n");
                if (!feederOk)
                {
                    completeWorkers();
                }
            });

            auto const result = tester->RunTests(0.2, 0.4, DCGM_FI_PROF_PCIE_RX_BYTES, 99);
            feeder.join();
            auto workers = waitForWorkers(2);

            THEN("the bandwidth field forces serial GPU runs despite the high parallelism limit")
            {
                REQUIRE(workers.size() == 2);
                CHECK(feederOk.load());
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.GetCallCounts().fork == 2);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.lastUpdateFreq == 200000);
                CHECK(mockDcgmOps->m_fieldGroupState.lastMaxKeepAge == 0.4);
                CHECK(workerRuntime.HasCommand(workers[0], "R "));
                CHECK(workerRuntime.HasCommand(workers[1], "R "));
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester creates and drives workers for MIG compute instances")
{
    GIVEN("one MIG-enabled GPU with one compute instance")
    {
        ScopedSignalHandler ignoreSigPipe(SIGPIPE, SIG_IGN);

        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        auto attributes                                         = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        attributes.settings.migModeEnabled                      = 1;
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = attributes;
        unsigned int migCiVisibleLookups                        = 0;
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = [&migCiVisibleLookups](dcgmHandle_t,
                                                                                         dcgmGroupEntityPair_t *entity,
                                                                                         unsigned int,
                                                                                         unsigned short *fieldIds,
                                                                                         unsigned int,
                                                                                         unsigned int,
                                                                                         dcgmFieldValue_v2 *value) {
            value->status = DCGM_ST_OK;
            if (*fieldIds == DCGM_FI_DEV_GPU_VIRTUAL_MODE)
            {
                value->value.i64 = DCGM_GPU_VIRTUALIZATION_MODE_NONE;
            }
            else if (*fieldIds == DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY)
            {
                value->value.i64 = (8ULL << 16);
            }
            else if (*fieldIds == DCGM_FI_CUDA_GPU_VISIBLE_DEVICES)
            {
                if (entity->entityGroupId == DCGM_FE_GPU_CI)
                {
                    ++migCiVisibleLookups;
                    std::snprintf(value->value.str, sizeof(value->value.str), "MIG-CI-%u", entity->entityId);
                }
                else if (entity->entityGroupId == DCGM_FE_GPU)
                {
                    std::snprintf(value->value.str, sizeof(value->value.str), "MIG-GPU-%u", entity->entityId);
                }
                else
                {
                    std::snprintf(value->value.str, sizeof(value->value.str), "MIG-CI-%u", entity->entityId);
                }
            }
            return DCGM_ST_OK;
        };
        mockDcgmOps->m_topologyState.deviceReturn = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps->m_discoveryState.allDevices  = { 140 };

        auto &hierarchy                = mockDcgmOps->m_migHierarchyState.hierarchy;
        hierarchy.version              = dcgmMigHierarchy_version2;
        hierarchy.count                = 2;
        hierarchy.entityList[0].entity = { DCGM_FE_GPU_I, 141 };
        hierarchy.entityList[0].parent = { DCGM_FE_GPU, 140 };
        hierarchy.entityList[1].entity = { DCGM_FE_GPU_CI, 142 };
        hierarchy.entityList[1].parent = { DCGM_FE_GPU_I, 141 };

        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 140 }));
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU_I, 141 }));
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU_CI, 142 }));

        auto arguments                            = MakeArguments({ 140 });
        arguments.m_parameters.m_noDcgmValidation = true;
        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);

        auto waitForWorkers = [this](std::size_t count) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                auto workers = workerRuntime.GetWorkers(count);
                if (workers.size() == count)
                {
                    return workers;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.GetWorkers(count);
        };

        auto waitForCommand = [this](auto const &worker, std::string_view token) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (workerRuntime.HasCommand(worker, token))
                {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.HasCommand(worker, token);
        };

        WHEN("a standard profiler field is run")
        {
            std::atomic_bool feederOk { false };
            std::jthread feeder([&] {
                auto workers = waitForWorkers(1);
                if (workers.size() != 1)
                {
                    return;
                }
                feederOk = workerRuntime.WriteResponse(workers[0], "P\n") && waitForCommand(workers[0], "R ")
                           && workerRuntime.WriteResponse(workers[0], "P\nM 0\nE 0\n");
            });

            auto const result = tester->RunTests(0.2, 0.4, DCGM_FI_PROF_SM_UTIL_RATIO, 4);
            feeder.join();
            auto workers = waitForWorkers(1);

            THEN("one MIG worker is launched using the compute-instance CUDA mapping")
            {
                REQUIRE(workers.size() == 1);
                CHECK(feederOk.load());
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.GetCallCounts().fork == 1);
                CHECK(migCiVisibleLookups >= 1);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 1);
                CHECK(workerRuntime.HasCommand(workers[0], "R "));
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester collects diagnostics after running workers fail")
{
    GIVEN("one initialized GPU whose worker sends a running-phase failure report")
    {
        ScopedSignalHandler ignoreSigPipe(SIGPIPE, SIG_IGN);

        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        auto arguments                                          = MakeArguments({ 1 });
        arguments.m_parameters.m_noDcgmValidation               = true;
        mockDcgmOps->m_discoveryState.allDevices                = { 1 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 1 }));

        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);

        auto waitForWorkers = [this](std::size_t count) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                auto workers = workerRuntime.GetWorkers(count);
                if (workers.size() == count)
                {
                    return workers;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.GetWorkers(count);
        };

        auto waitForCommand = [this](auto const &worker, std::string_view token) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (workerRuntime.HasCommand(worker, token))
                {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.HasCommand(worker, token);
        };

        WHEN("the fake worker fails during the running phase and then reports diagnostics")
        {
            std::atomic_bool feederOk { false };
            std::jthread feeder([&] {
                auto workers = waitForWorkers(1);
                if (workers.size() != 1)
                {
                    return;
                }
                feederOk = workerRuntime.WriteResponse(workers[0], "P\n") && waitForCommand(workers[0], "R ")
                           && workerRuntime.WriteResponse(workers[0], "F\nM 9\nrun note\nE 10\nrun error\n");
            });

            auto const result = tester->RunTests(0.1, 0.2, DCGM_FI_PROF_SM_UTIL_RATIO, 0);
            feeder.join();
            auto workers = waitForWorkers(1);

            THEN("the run drains worker reports and completes field cleanup")
            {
                REQUIRE(workers.size() == 1);
                CHECK(feederOk.load());
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 1);
                CHECK(workerRuntime.HasCommand(workers[0], "M\nE\n"));
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester cleans up fake workers when run dispatch fails")
{
    GIVEN("one initialized GPU whose worker cannot run the requested profiler field")
    {
        ScopedSignalHandler ignoreSigPipe(SIGPIPE, SIG_IGN);

        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        auto arguments                                          = MakeArguments({ 1 });
        arguments.m_parameters.m_noDcgmValidation               = false;
        arguments.m_parameters.m_fieldId                        = DCGM_FI_DEV_GPU_NAME;
        mockDcgmOps->m_discoveryState.allDevices                = { 1 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 1 }));

        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);

        auto waitForWorkers = [this](std::size_t count) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                auto workers = workerRuntime.GetWorkers(count);
                if (workers.size() == count)
                {
                    return workers;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.GetWorkers(count);
        };

        auto waitForCommand = [this](auto const &worker, std::string_view token) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (workerRuntime.HasCommand(worker, token))
                {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.HasCommand(worker, token);
        };

        WHEN("the worker reports startup success for an unsupported run field")
        {
            std::atomic_bool feederOk { false };
            std::jthread feeder([&] {
                auto workers = waitForWorkers(1);
                if (workers.size() != 1)
                {
                    return;
                }
                feederOk = workerRuntime.WriteResponse(workers[0], "P\n") && waitForCommand(workers[0], "X\n");
            });

            auto const result = tester->RunTests(0.1, 0.2, DCGM_FI_DEV_GPU_NAME, 0);
            feeder.join();
            auto workers = waitForWorkers(1);

            THEN("the run fails after field watch cleanup and force-shutting down the child")
            {
                REQUIRE(workers.size() == 1);
                CHECK(feederOk.load());
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 1);
                CHECK(workerRuntime.HasCommand(workers[0], "X\n"));
            }
        }
    }

    GIVEN("one initialized GPU whose worker reports startup failure")
    {
        ScopedSignalHandler ignoreSigPipe(SIGPIPE, SIG_IGN);

        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        auto arguments                                          = MakeArguments({ 1 });
        arguments.m_parameters.m_noDcgmValidation               = false;
        mockDcgmOps->m_discoveryState.allDevices                = { 1 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 1 }));

        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);

        auto waitForWorkers = [this](std::size_t count) {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline)
            {
                auto workers = workerRuntime.GetWorkers(count);
                if (workers.size() == count)
                {
                    return workers;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return workerRuntime.GetWorkers(count);
        };

        WHEN("the worker cannot initialize but still returns diagnostics")
        {
            std::atomic_bool feederOk { false };
            std::jthread feeder([&] {
                auto workers = waitForWorkers(1);
                if (workers.size() != 1)
                {
                    return;
                }
                feederOk = workerRuntime.WriteResponse(workers[0], "F\nM 13\nstartup note\nE 12\nstartup err\n");
            });

            auto const result = tester->RunTests(0.1, 0.2, DCGM_FI_PROF_SM_UTIL_RATIO, 0);
            feeder.join();
            auto workers = waitForWorkers(1);

            THEN("startup diagnostics are drained and field cleanup still runs")
            {
                REQUIRE(workers.size() == 1);
                CHECK(feederOk.load());
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 1);
                CHECK(mockDcgmOps->m_fieldGroupState.unwatchCallCount == 1);
                CHECK(workerRuntime.HasCommand(workers[0], "M\nE\n"));
            }
        }
    }
}

SCENARIO_METHOD(DcgmProfTesterFixture, "DcgmProfTester reports DCGM setup failures before worker dispatch")
{
    GIVEN("one initialized GPU and a NvLink run whose per-GPU field group cannot be created")
    {
        char program[] = "dcgmproftester";
        char *argv[]   = { program };
        REQUIRE(tester->Init(1, argv) == DCGM_ST_OK);

        auto arguments                                          = MakeArguments({ 1 });
        arguments.m_parameters.m_noDcgmValidation               = false;
        arguments.m_parameters.m_fieldId                        = DCGM_FI_PROF_NVLINK_RX_BYTES;
        mockDcgmOps->m_discoveryState.allDevices                = { 1 };
        mockDcgmOps->m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps->m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps->m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps->AddMockedEntity(DcgmNs::MockDcgmEntity({ DCGM_FE_GPU, 1 }));

        REQUIRE(tester->InitializeGpus(arguments) == DCGM_ST_OK);
        REQUIRE(mockDcgmOps->m_groupManagementState.createCallCount >= 1);
        auto const fieldCreatesAfterInit = mockDcgmOps->m_fieldGroupState.createCallCount;
        REQUIRE(fieldCreatesAfterInit >= 1);

        mockDcgmOps->m_fieldGroupState.createReturn = DCGM_ST_BADPARAM;

        WHEN("the run starts")
        {
            auto const result  = tester->RunTests(0.1, 0.2, DCGM_FI_PROF_NVLINK_RX_BYTES, 0);
            auto const workers = workerRuntime.GetWorkers(1);

            THEN("the field-group setup failure is returned without watching fields or sending a run command")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps->m_fieldGroupState.createCallCount == fieldCreatesAfterInit + 1);
                CHECK(mockDcgmOps->m_fieldGroupState.watchCallCount == 0);
                REQUIRE(workers.size() == 1);
                CHECK_FALSE(workerRuntime.HasCommand(workers[0], "R "));
            }
        }
    }
}
