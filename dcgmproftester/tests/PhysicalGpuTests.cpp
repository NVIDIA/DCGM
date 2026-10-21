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
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DcgmProfTester.h"
#include "DistributedCudaContext.h"
#include "MockDistributedCudaContextRuntime.hpp"
#include "MockDistributedCudaContextRuntimeFactory.hpp"
#include "PhysicalGpu.h"
#include <MockCudaWorkerThreadFactory.hpp>
#include <MockDcgmProfTester.hpp>

#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <tuple>
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

struct PhysicalGpuFixture
{
    MockDistributedCudaContextRuntimeState workerRuntimeState;
    MockDistributedCudaContextRuntime workerRuntime { &workerRuntimeState };
    MockDistributedCudaContextRuntimeFactory runtimeFactory { &workerRuntimeState };
    MockDcgmProfTester tester;
    Arguments_t::Parameters parameters = MakeOfflineParameters();
    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps { &mockDcgmOps };
    MockCudaOps mockCudaOps;
    CudaOpsAnyPtr cudaOps { &mockCudaOps };
    CudaWorkerThreadFactoryAnyUniquePtr cudaWorkerFactory { std::make_unique<MockCudaWorkerThreadFactory>() };
};

auto MakeValidGpuLatestValuesHandler(uint64_t computeCapability = (8ULL << 16))
{
    return [computeCapability](dcgmHandle_t,
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
            value->value.i64 = computeCapability;
        }
        else if (*fieldIds == DCGM_FI_CUDA_GPU_VISIBLE_DEVICES)
        {
            std::snprintf(value->value.str, sizeof(value->value.str), "cuda-%u", entity->entityId);
        }
        return DCGM_ST_OK;
    };
}

auto MakePcieMetadataLatestValuesHandler(uint64_t computeCapability = (8ULL << 16))
{
    return [computeCapability](dcgmHandle_t handle,
                               dcgmGroupEntityPair_t *entity,
                               unsigned int entityCount,
                               unsigned short *fieldIds,
                               unsigned int fieldCount,
                               unsigned int flags,
                               dcgmFieldValue_v2 *value) {
        if (*fieldIds == DCGM_FI_DEV_PCIE_LINK_GEN)
        {
            value->status    = DCGM_ST_OK;
            value->value.i64 = 7;
            return DCGM_ST_OK;
        }
        if (*fieldIds == DCGM_FI_DEV_PCIE_LINK_WIDTH)
        {
            value->status    = DCGM_ST_OK;
            value->value.i64 = 8;
            return DCGM_ST_OK;
        }
        return MakeValidGpuLatestValuesHandler(computeCapability)(
            handle, entity, entityCount, fieldIds, fieldCount, flags, value);
    };
}

auto MakePcieLinkWidthFailsLatestValuesHandler(uint64_t computeCapability = (8ULL << 16))
{
    return [computeCapability](dcgmHandle_t handle,
                               dcgmGroupEntityPair_t *entity,
                               unsigned int entityCount,
                               unsigned short *fieldIds,
                               unsigned int fieldCount,
                               unsigned int flags,
                               dcgmFieldValue_v2 *value) {
        if (*fieldIds == DCGM_FI_DEV_PCIE_LINK_GEN)
        {
            value->status    = DCGM_ST_OK;
            value->value.i64 = 4;
            return DCGM_ST_OK;
        }
        if (*fieldIds == DCGM_FI_DEV_PCIE_LINK_WIDTH)
        {
            return DCGM_ST_GENERIC_ERROR;
        }
        return MakeValidGpuLatestValuesHandler(computeCapability)(
            handle, entity, entityCount, fieldIds, fieldCount, flags, value);
    };
}
} // namespace

TEST_CASE("MockDistributedCudaContextRuntime factories share only their owning test state")
{
    MockDistributedCudaContextRuntimeState firstState;
    MockDistributedCudaContextRuntime firstRuntime { &firstState };
    MockDistributedCudaContextRuntimeFactory factory { &firstState };
    auto first  = factory.Create();
    auto second = factory.Create();

    CHECK(first.AsAnyPtr().Fork() == 4200);
    CHECK(second.AsAnyPtr().Fork() == 4201);
    CHECK(firstRuntime.GetCallCounts().fork == 2);

    MockDistributedCudaContextRuntimeState isolatedState;
    MockDistributedCudaContextRuntime isolatedRuntime { &isolatedState };
    CHECK(isolatedRuntime.GetCallCounts().fork == 0);
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu exposes configured offline test properties")
{
    WHEN("the physical GPU is constructed")
    {
        PhysicalGpu gpu(
            &tester, 2, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        THEN("its public queries reflect the configured test parameters")
        {
            CHECK(gpu.GetGpuId() == 2);
            CHECK(gpu.GetFieldId() == DCGM_FI_PROF_SM_UTIL_RATIO);
            CHECK(gpu.GetDcgmValidation() == false);
            CHECK(gpu.IsSynchronous());
            CHECK(gpu.UseCublas());
            CHECK(gpu.GetHandle() == 0);
        }

        THEN("its worker state starts empty and valid")
        {
            CHECK_FALSE(gpu.WorkerStarted());
            CHECK_FALSE(gpu.WorkerRunning());
            CHECK_FALSE(gpu.WorkerFinished());
            CHECK_FALSE(gpu.WorkerReported());
            CHECK(gpu.AllStarted());
            CHECK(gpu.AllFinished());
            CHECK(gpu.AllReported());
            CHECK_FALSE(gpu.AnyWorkerRequestFailed());
            CHECK(gpu.IsValidated());
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu updates test parameters through the public setter")
{
    GIVEN("a physical GPU created with offline parameters")
    {
        PhysicalGpu gpu(
            &tester, 3, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        WHEN("new parameters are applied")
        {
            Arguments_t::Parameters next {};
            next.m_noDcgmValidation = false;
            next.m_fieldId          = DCGM_FI_PROF_TENSOR_UTIL_RATIO;
            next.m_syncCount        = 0;
            next.m_cublas           = false;
            gpu.SetParameters(next);

            THEN("the observable configuration reflects the new parameters")
            {
                CHECK(gpu.GetGpuId() == 3);
                CHECK(gpu.GetFieldId() == DCGM_FI_PROF_TENSOR_UTIL_RATIO);
                CHECK(gpu.GetDcgmValidation());
                CHECK_FALSE(gpu.IsSynchronous());
                CHECK_FALSE(gpu.UseCublas());

                next.m_noDcgmValidation = true;
                gpu.SetParameters(next);
                CHECK_FALSE(gpu.GetDcgmValidation());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu creates and destroys per-test field groups directly")
{
    GIVEN("a GPU with DCGM validation enabled")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        auto gpu                                               = std::make_unique<PhysicalGpu>(
            &tester, 51, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu->Init(0) == DCGM_ST_OK);

        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 51;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 51 };
        REQUIRE(gpu->AddSlice(entities, entity, "cuda-51") != nullptr);

        WHEN("the per-test group lifecycle is invoked")
        {
            auto const createResult = gpu->StartTests();
            auto const fieldGroupId = gpu->GetFieldGroupId();
            gpu->AbortChildren();

            THEN("the configured field id is passed through and the returned group is destroyed")
            {
                CHECK(createResult == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_fieldGroupState.createCallCount == 1);
                CHECK(mockDcgmOps.m_fieldGroupState.destroyCallCount == 1);
                CHECK(mockDcgmOps.m_fieldGroupState.lastCreatedFieldIds
                      == std::vector<unsigned short> { DCGM_FI_PROF_SM_UTIL_RATIO });
                CHECK(fieldGroupId == mockDcgmOps.m_fieldGroupState.lastDestroyedFieldGroupId);
            }
        }
    }

    GIVEN("a physical GPU with DCGM validation disabled")
    {
        parameters.m_noDcgmValidation                      = true;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 52, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 52;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 52 };
        REQUIRE(gpu->AddSlice(entities, entity, "cuda-52") != nullptr);

        WHEN("group creation and destruction are requested")
        {
            auto const createResult = gpu->StartTests();
            gpu->AbortChildren();

            THEN("the DCGM collaborators are skipped")
            {
                CHECK(createResult == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_fieldGroupState.createCallCount == 0);
                CHECK(mockDcgmOps.m_fieldGroupState.destroyCallCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu reports per-test field group creation and destruction failures")
{
    GIVEN("a physical GPU with validation enabled and a failing field-group create")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps.m_fieldGroupState.createReturn             = DCGM_ST_BADPARAM;
        auto gpu                                               = std::make_unique<PhysicalGpu>(
            &tester, 82, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu->Init(0) == DCGM_ST_OK);

        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 82;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 82 };
        REQUIRE(gpu->AddSlice(entities, entity, "cuda-82") != nullptr);

        WHEN("the per-test field group is created")
        {
            auto const result = gpu->StartTests();

            THEN("the DCGM create error is propagated")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(gpu->GetFieldGroupId() == 0);
            }
        }
    }

    GIVEN("a physical GPU with validation enabled and a failing field-group destroy")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        auto gpu                                               = std::make_unique<PhysicalGpu>(
            &tester, 83, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu->Init(0) == DCGM_ST_OK);

        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 83;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 83 };
        REQUIRE(gpu->AddSlice(entities, entity, "cuda-83") != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);
        mockDcgmOps.m_fieldGroupState.destroyReturn = DCGM_ST_GENERIC_ERROR;

        WHEN("the per-test field group is destroyed")
        {
            gpu->AbortChildren();

            THEN("the destroy collaborator is invoked and validation remains enabled")
            {
                CHECK(mockDcgmOps.m_fieldGroupState.destroyCallCount == 1);
                CHECK(gpu->GetDcgmValidation());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu advances synchronous workers only when peers are not behind")
{
    GIVEN("a physical GPU with two active CUDA contexts")
    {
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        parameters.m_syncCount                             = 3;
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 80, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        auto makeEntities = [](dcgm_field_eid_t id) {
            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
            entities[DCGM_FE_GPU] = id;
            return entities;
        };
        dcgmGroupEntityPair_t firstEntity { DCGM_FE_GPU, 80 };
        dcgmGroupEntityPair_t secondEntity { DCGM_FE_GPU, 81 };
        auto first  = gpu->AddSlice(makeEntities(80), firstEntity, "cuda-80");
        auto second = gpu->AddSlice(makeEntities(81), secondEntity, "cuda-81");
        REQUIRE(first != nullptr);
        REQUIRE(second != nullptr);

        WHEN("the requesting worker has already failed validation")
        {
            first->SetValidated(false);
            first->SetActivity(1);
            second->SetActivity(1);
            second->SetTries(0);

            THEN("the GPU refuses to advance that worker")
            {
                CHECK_FALSE(gpu->Advance(*first, 1));
                CHECK(first->GetActivity() == 1);
                CHECK(second->GetActivity() == 1);
                CHECK(second->GetTries() == 0);
            }
        }

        WHEN("another worker is still behind the requested activity")
        {
            first->SetValidated(true);
            first->SetParts(1, 2);
            second->SetParts(1, 2);
            first->SetActivity(4);
            second->SetActivity(2);
            first->SetTries(1);
            second->SetTries(0);

            THEN("the requesting worker keeps its activity and gets its retry back")
            {
                CHECK_FALSE(gpu->Advance(*first, 4));
                CHECK(first->GetActivity() == 4);
                CHECK(first->GetTries() == 2);
                CHECK(second->GetActivity() == 2);
                CHECK(second->GetTries() == 0);
            }
        }

        WHEN("another worker is still on an earlier part")
        {
            first->SetValidated(true);
            first->SetParts(2, 2);
            second->SetParts(1, 2);
            first->SetActivity(1);
            second->SetActivity(1);
            first->SetTries(1);
            second->SetTries(0);

            THEN("the requesting worker keeps its activity and gets its retry back")
            {
                CHECK_FALSE(gpu->Advance(*first, 1));
                CHECK(first->GetActivity() == 1);
                CHECK(first->GetTries() == 2);
                CHECK(second->GetActivity() == 1);
                CHECK(second->GetTries() == 0);
            }
        }

        WHEN("all peer workers have reached the same part and activity")
        {
            first->SetValidated(true);
            first->SetParts(1, 2);
            second->SetParts(1, 2);
            first->SetActivity(1);
            second->SetActivity(1);
            second->SetTries(0);

            THEN("the requesting worker advances to the next activity")
            {
                CHECK(gpu->Advance(*first, 1));
                CHECK(first->GetActivity() == 2);
                CHECK(second->GetActivity() == 1);
                CHECK(second->GetTries() == 0);
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu handles empty worker sets without external dependencies")
{
    GIVEN("a physical GPU with no CUDA workers")
    {
        PhysicalGpu gpu(
            &tester, 4, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        WHEN("test execution is started without workers")
        {
            dcgmReturn_t const result = gpu.StartTests();

            THEN("the command reports no available data and preserves empty worker state")
            {
                CHECK(result == DCGM_ST_NO_DATA);
                CHECK_FALSE(gpu.WorkerStarted());
                CHECK_FALSE(gpu.WorkerRunning());
                CHECK_FALSE(gpu.WorkerFinished());
                CHECK_FALSE(gpu.WorkerReported());
                CHECK(gpu.AllStarted());
                CHECK(gpu.AllFinished());
                CHECK(gpu.AllReported());
            }
        }

        WHEN("run is requested after all zero workers are already finished")
        {
            dcgmReturn_t const result = gpu.RunTests();

            THEN("the run is a successful no-op")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(gpu.AllFinished());
                CHECK(gpu.IsValidated());
            }
        }

        WHEN("children are aborted or nuked before any workers are added")
        {
            gpu.AbortChildren();
            gpu.NukeChildren(true);

            THEN("the empty worker state remains complete and reportable")
            {
                CHECK(gpu.AllStarted());
                CHECK(gpu.AllFinished());
                CHECK(gpu.AllReported());
                CHECK_FALSE(gpu.AnyWorkerRequestFailed());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu rejects invalid CUDA-visible entity requests")
{
    GIVEN("a physical GPU configured for offline validation")
    {
        PhysicalGpu gpu(
            &tester, 5, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        dcgmGroupEntityPair_t entity { DCGM_FE_SWITCH, 2 };
        std::string cudaVisibleGpus = "unchanged";

        WHEN("CUDA visible devices are requested for a non-GPU entity")
        {
            dcgmReturn_t const result = gpu.HelperGetCudaVisibleGPUs(cudaVisibleGpus, entity);

            THEN("the request is rejected and the caller buffer is not rewritten")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(cudaVisibleGpus == "unchanged");
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu propagates DCGM initialization failures")
{
    parameters.m_noDcgmValidation = false;

    WHEN("device attributes cannot be read")
    {
        mockDcgmOps.m_deviceAttributesState.returnCode = DCGM_ST_GENERIC_ERROR;
        PhysicalGpu gpu(
            &tester, 6, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        THEN("initialization returns the DCGM error")
        {
            CHECK(gpu.Init(0) == DCGM_ST_GENERIC_ERROR);
        }
    }

    WHEN("the virtualization mode is unsupported")
    {
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = [](dcgmHandle_t,
                                                                    dcgmGroupEntityPair_t *,
                                                                    unsigned int,
                                                                    unsigned short *,
                                                                    unsigned int,
                                                                    unsigned int,
                                                                    dcgmFieldValue_v2 *value) {
            value->status    = DCGM_ST_OK;
            value->value.i64 = DCGM_GPU_VIRTUALIZATION_MODE_HOST_VGPU;
            return DCGM_ST_OK;
        };

        PhysicalGpu gpu(
            &tester, 6, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        THEN("initialization reports profiling not supported")
        {
            CHECK(gpu.Init(0) == DCGM_ST_PROFILING_NOT_SUPPORTED);
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu initialization tolerates unavailable virtualization telemetry")
{
    GIVEN("a physical GPU whose virtualization field lookup fails")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        auto const validLatestValues                           = MakeValidGpuLatestValuesHandler();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = [validLatestValues](dcgmHandle_t handle,
                                                                                     dcgmGroupEntityPair_t *entity,
                                                                                     unsigned int entityCount,
                                                                                     unsigned short *fieldIds,
                                                                                     unsigned int fieldCount,
                                                                                     unsigned int flags,
                                                                                     dcgmFieldValue_v2 *value) {
            if (*fieldIds == DCGM_FI_DEV_GPU_VIRTUAL_MODE)
            {
                return DCGM_ST_GENERIC_ERROR;
            }
            return validLatestValues(handle, entity, entityCount, fieldIds, fieldCount, flags, value);
        };
        PhysicalGpu gpu(
            &tester, 90, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        WHEN("the GPU is initialized")
        {
            auto const result = gpu.Init(0);

            THEN("initialization succeeds so PerfWorks can report any later unsupported mode")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(gpu.GetGpuId() == 90);
                CHECK(gpu.GetGpuBusId() == "0000:01:00.0");
                CHECK_FALSE(gpu.IsMIG());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu initialization accepts supported virtualization modes")
{
    GIVEN("a GPU whose virtualization mode is passthrough")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = [](dcgmHandle_t,
                                                                    dcgmGroupEntityPair_t *entity,
                                                                    unsigned int,
                                                                    unsigned short *fieldIds,
                                                                    unsigned int,
                                                                    unsigned int,
                                                                    dcgmFieldValue_v2 *value) {
            value->status = DCGM_ST_OK;
            if (*fieldIds == DCGM_FI_DEV_GPU_VIRTUAL_MODE)
            {
                value->value.i64 = DCGM_GPU_VIRTUALIZATION_MODE_PASSTHROUGH;
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
        PhysicalGpu gpu(
            &tester, 93, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());

        WHEN("the GPU is initialized")
        {
            auto const result = gpu.Init(0);

            THEN("the supported passthrough mode does not block profiling setup")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(gpu.GetGpuId() == 93);
                CHECK(gpu.GetGpuBusId() == "0000:01:00.0");
                CHECK_FALSE(gpu.IsMIG());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu reuses MIG slice workers across field selections")
{
    GIVEN("a MIG GPU with two compute instances in the same GPU instance")
    {
        auto attributes                                        = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        attributes.settings.migModeEnabled                     = 1;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = attributes;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

        parameters.m_fieldId = DCGM_FI_PROF_SM_UTIL_RATIO;
        auto gpu             = std::make_unique<PhysicalGpu>(
            &tester, 7, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu->Init(0) == DCGM_ST_OK);

        auto makeEntities = [](dcgm_field_eid_t ci) {
            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
            entities[DCGM_FE_GPU]    = 7;
            entities[DCGM_FE_GPU_I]  = 1;
            entities[DCGM_FE_GPU_CI] = ci;
            return entities;
        };
        dcgmGroupEntityPair_t ci10 { DCGM_FE_GPU_CI, 10 };
        dcgmGroupEntityPair_t ci11 { DCGM_FE_GPU_CI, 11 };

        auto first  = gpu->AddSlice(makeEntities(10), ci10, "MIG-10");
        auto second = gpu->AddSlice(makeEntities(11), ci11, "MIG-11");
        REQUIRE(first != nullptr);
        REQUIRE(second != nullptr);
        REQUIRE(first != second);

        WHEN("a DRAM test allows only one CI per GI")
        {
            parameters.m_fieldId = DCGM_FI_PROF_DRAM_UTIL_RATIO;
            gpu->SetParameters(parameters);
            auto dramWorker = gpu->AddSlice(makeEntities(11), ci11, "MIG-11");

            THEN("the first CI worker is reused for the sibling CI")
            {
                REQUIRE(dramWorker == first);
            }
        }

        WHEN("a later multi-CI test requests the inactive CI again")
        {
            parameters.m_fieldId = DCGM_FI_PROF_DRAM_UTIL_RATIO;
            gpu->SetParameters(parameters);
            REQUIRE(gpu->AddSlice(makeEntities(11), ci11, "MIG-11") == first);

            parameters.m_fieldId = DCGM_FI_PROF_SM_UTIL_RATIO;
            gpu->SetParameters(parameters);
            auto reactivated = gpu->AddSlice(makeEntities(11), ci11, "MIG-11");

            THEN("the suspended worker is restored instead of allocating a replacement")
            {
                CHECK(reactivated == second);
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu maps CUDA-visible devices through DCGM")
{
    GIVEN("a non-MIG GPU with an NvLink peer")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

        dcgmDeviceTopology_v2 topology {};
        topology.version                           = dcgmDeviceTopology_version2;
        topology.numGpus                           = 1;
        topology.gpuPaths[0].gpuId                 = 9;
        mockDcgmOps.m_topologyState.deviceTopology = topology;

        PhysicalGpu gpu(
            &tester, 8, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 8 };
        std::string cudaVisibleGpus;

        WHEN("CUDA_VISIBLE_DEVICES is requested")
        {
            auto result = gpu.HelperGetCudaVisibleGPUs(cudaVisibleGpus, entity);

            THEN("the local GPU and peer mappings are returned in order")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(cudaVisibleGpus == "cuda-8,cuda-9");
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu maps CUDA visibility error and MIG edge cases")
{
    GIVEN("a non-MIG GPU whose topology is not supported")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps.m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;

        PhysicalGpu gpu(
            &tester, 12, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 12 };
        std::string cudaVisibleGpus = "old";

        WHEN("CUDA visibility is mapped")
        {
            auto const result = gpu.HelperGetCudaVisibleGPUs(cudaVisibleGpus, entity);

            THEN("the helper falls back to the local GPU mapping only")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(cudaVisibleGpus == "cuda-12");
            }
        }
    }

    GIVEN("a non-MIG GPU whose topology query fails")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps.m_topologyState.deviceReturn               = DCGM_ST_GENERIC_ERROR;

        PhysicalGpu gpu(
            &tester, 13, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 13 };
        std::string cudaVisibleGpus = "unchanged";

        WHEN("CUDA visibility is mapped")
        {
            auto const result = gpu.HelperGetCudaVisibleGPUs(cudaVisibleGpus, entity);

            THEN("the topology error is returned before the caller buffer is rewritten")
            {
                CHECK(result != DCGM_ST_OK);
                CHECK(cudaVisibleGpus == "unchanged");
            }
        }
    }

    GIVEN("a MIG compute instance with a MIG-formatted visible device string")
    {
        parameters.m_noDcgmValidation = false;

        auto attributes                                        = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        attributes.settings.migModeEnabled                     = 1;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = attributes;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = [](dcgmHandle_t,
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
                if (entity->entityGroupId == DCGM_FE_GPU)
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

        dcgmDeviceTopology_v2 topology {};
        topology.version                           = dcgmDeviceTopology_version2;
        topology.numGpus                           = 0;
        mockDcgmOps.m_topologyState.deviceTopology = topology;

        PhysicalGpu gpu(
            &tester, 14, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU_CI, 44 };
        std::string cudaVisibleGpus;

        WHEN("CUDA visibility is mapped for the compute instance")
        {
            auto const result = gpu.HelperGetCudaVisibleGPUs(cudaVisibleGpus, entity);

            THEN("the MIG compute-instance mapping is preserved")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(cudaVisibleGpus == "MIG-CI-44");
            }
        }
    }

    GIVEN("DCGM cannot map a peer entity to CUDA visibility")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = [](dcgmHandle_t,
                                                                    dcgmGroupEntityPair_t *,
                                                                    unsigned int,
                                                                    unsigned short *fieldIds,
                                                                    unsigned int,
                                                                    unsigned int,
                                                                    dcgmFieldValue_v2 *value) {
            if (*fieldIds == DCGM_FI_DEV_GPU_VIRTUAL_MODE)
            {
                value->status    = DCGM_ST_OK;
                value->value.i64 = DCGM_GPU_VIRTUALIZATION_MODE_NONE;
                return DCGM_ST_OK;
            }
            if (*fieldIds == DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY)
            {
                value->status    = DCGM_ST_OK;
                value->value.i64 = (8ULL << 16);
                return DCGM_ST_OK;
            }
            value->status = DCGM_ST_NO_DATA;
            return DCGM_ST_OK;
        };

        dcgmDeviceTopology_v2 topology {};
        topology.version                           = dcgmDeviceTopology_version2;
        topology.numGpus                           = 1;
        topology.gpuPaths[0].gpuId                 = 9;
        mockDcgmOps.m_topologyState.deviceTopology = topology;

        PhysicalGpu gpu(
            &tester, 15, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 15 };
        std::string cudaVisibleGpus;

        WHEN("CUDA visibility is mapped")
        {
            auto const result = gpu.HelperGetCudaVisibleGPUs(cudaVisibleGpus, entity);

            THEN("the helper reports a generic mapping failure")
            {
                CHECK(result != DCGM_ST_OK);
                CHECK(cudaVisibleGpus.empty());
            }
        }
    }

    GIVEN("a non-MIG GPU whose CUDA-visible mapping reports a MIG identifier")
    {
        parameters.m_noDcgmValidation                          = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = [](dcgmHandle_t,
                                                                    dcgmGroupEntityPair_t *,
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
                std::snprintf(value->value.str, sizeof(value->value.str), "MIG-GPU-only");
            }
            return DCGM_ST_OK;
        };
        mockDcgmOps.m_topologyState.deviceReturn = DCGM_ST_NOT_SUPPORTED;

        PhysicalGpu gpu(
            &tester, 130, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 130 };
        std::string cudaVisibleGpus = "unchanged";

        WHEN("CUDA visibility is mapped")
        {
            auto const result = gpu.HelperGetCudaVisibleGPUs(cudaVisibleGpus, entity);

            THEN("the MIG-only mapping is ignored without turning into a hard failure")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(cudaVisibleGpus.empty());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu propagates NvLink peer selection through RunTests")
{
    auto startReadyWorker = [this](PhysicalGpu &gpu, unsigned int gpuId) {
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = gpuId;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, gpuId };
        auto worker = gpu.AddSlice(entities, entity, "cuda-" + std::to_string(gpuId));
        REQUIRE(worker != nullptr);
        REQUIRE(gpu.StartTests() == DCGM_ST_OK);
        auto workers = workerRuntime.GetWorkers(1);
        REQUIRE(workers.size() == 1);
        REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
        REQUIRE(gpu.ProcessResponse(worker) == DCGM_ST_OK);
        REQUIRE(gpu.AllStarted());
        return workers[0];
    };

    GIVEN("a GPU with multiple NvLink peers")
    {
        parameters.m_noDcgmValidation                      = true;
        parameters.m_fieldId                               = DCGM_FI_PROF_NVLINK_TX_BYTES;
        parameters.m_report                                = false;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();

        auto addGpuWithBusId = [this](unsigned int gpuId, char const *busId) {
            DcgmNs::MockDcgmEntity entity({ DCGM_FE_GPU, gpuId });
            auto attributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
            std::snprintf(attributes.identifiers.pciBusId, sizeof(attributes.identifiers.pciBusId), "%s", busId);
            entity.SetDevAttr(attributes);
            mockDcgmOps.AddMockedEntity(entity);
        };
        constexpr char const *sourceBusId = "0000:3e:00.0";
        constexpr char const *peerBusId   = "0000:72:00.0";
        addGpuWithBusId(62, sourceBusId);
        addGpuWithBusId(70, "0000:46:00.0");
        addGpuWithBusId(71, "0000:47:00.0");
        addGpuWithBusId(72, peerBusId);

        dcgmDeviceTopology_v2 topology {};
        topology.version                           = dcgmDeviceTopology_version2;
        topology.numGpus                           = 3;
        topology.gpuPaths[0].gpuId                 = 70;
        topology.gpuPaths[0].path                  = DCGM_TOPOLOGY_BOARD;
        topology.gpuPaths[0].localNvLinkIds        = 0;
        topology.gpuPaths[1].gpuId                 = 71;
        topology.gpuPaths[1].path                  = DCGM_TOPOLOGY_NVLINK1;
        topology.gpuPaths[1].localNvLinkIds        = 0x1;
        topology.gpuPaths[2].gpuId                 = 72;
        topology.gpuPaths[2].path                  = DCGM_TOPOLOGY_NVLINK2;
        topology.gpuPaths[2].localNvLinkIds        = 0xF;
        mockDcgmOps.m_topologyState.deviceTopology = topology;

        PhysicalGpu gpu(
            &tester, 62, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(mockDcgmOps.m_handleId) == DCGM_ST_OK);
        auto workerHandle = startReadyWorker(gpu, 62);

        WHEN("NvLink bandwidth tests are run")
        {
            auto const result = gpu.RunTests();

            THEN("the peer with the most local NvLink ids is passed to the worker")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(workerHandle, "R "));
                CHECK(workerRuntime.HasCommand(workerHandle, peerBusId));
                CHECK_FALSE(workerRuntime.HasCommand(workerHandle, sourceBusId));
            }
        }
    }

    GIVEN("a GPU with no usable NvLink path")
    {
        parameters.m_noDcgmValidation                          = true;
        parameters.m_fieldId                                   = DCGM_FI_PROF_NVLINK_TX_BYTES;
        parameters.m_report                                    = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

        dcgmDeviceTopology_v2 topology {};
        topology.version                           = dcgmDeviceTopology_version2;
        topology.numGpus                           = 1;
        topology.gpuPaths[0].gpuId                 = 73;
        topology.gpuPaths[0].path                  = DCGM_TOPOLOGY_BOARD;
        topology.gpuPaths[0].localNvLinkIds        = 0;
        mockDcgmOps.m_topologyState.deviceTopology = topology;

        PhysicalGpu gpu(
            &tester, 63, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        auto workerHandle = startReadyWorker(gpu, 63);

        WHEN("NvLink bandwidth tests are run")
        {
            auto const result = gpu.RunTests();

            THEN("the run reports that NvLink peers are unsupported without sending a run command")
            {
                CHECK(result == DCGM_ST_NOT_SUPPORTED);
                CHECK_FALSE(workerRuntime.HasCommand(workerHandle, "R "));
                CHECK_FALSE(gpu.WorkerFinished());
                CHECK_FALSE(gpu.AllReported());
            }
        }
    }

    GIVEN("a GPU whose topology query fails")
    {
        parameters.m_noDcgmValidation                          = true;
        parameters.m_fieldId                                   = DCGM_FI_PROF_NVLINK_TX_BYTES;
        parameters.m_report                                    = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();
        mockDcgmOps.m_topologyState.deviceReturn               = DCGM_ST_GENERIC_ERROR;

        PhysicalGpu gpu(
            &tester, 68, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        startReadyWorker(gpu, 68);

        WHEN("NvLink bandwidth tests are run")
        {
            auto const result = gpu.RunTests();

            THEN("the topology failure is reported as unsupported")
            {
                CHECK(result == DCGM_ST_NOT_SUPPORTED);
            }
        }
    }

    GIVEN("a GPU whose selected peer attributes cannot be read")
    {
        parameters.m_noDcgmValidation                          = true;
        parameters.m_fieldId                                   = DCGM_FI_PROF_NVLINK_TX_BYTES;
        parameters.m_report                                    = false;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

        dcgmDeviceTopology_v2 topology {};
        topology.version                           = dcgmDeviceTopology_version2;
        topology.numGpus                           = 3;
        topology.gpuPaths[0].gpuId                 = 70;
        topology.gpuPaths[0].path                  = DCGM_TOPOLOGY_BOARD;
        topology.gpuPaths[0].localNvLinkIds        = 0;
        topology.gpuPaths[1].gpuId                 = 71;
        topology.gpuPaths[1].path                  = DCGM_TOPOLOGY_NVLINK1;
        topology.gpuPaths[1].localNvLinkIds        = 0x1;
        topology.gpuPaths[2].gpuId                 = 72;
        topology.gpuPaths[2].path                  = DCGM_TOPOLOGY_NVLINK2;
        topology.gpuPaths[2].localNvLinkIds        = 0xF;
        mockDcgmOps.m_topologyState.deviceTopology = topology;

        PhysicalGpu gpu(
            &tester, 64, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);
        auto workerHandle                              = startReadyWorker(gpu, 64);
        mockDcgmOps.m_deviceAttributesState.returnCode = DCGM_ST_BADPARAM;

        WHEN("NvLink bandwidth tests are run")
        {
            auto const result = gpu.RunTests();

            THEN("the peer attribute error is returned without sending a run command")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK_FALSE(workerRuntime.HasCommand(workerHandle, "R "));
                CHECK_FALSE(gpu.WorkerFinished());
                CHECK_FALSE(gpu.AllReported());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu rejects NvLink runs in MIG mode before worker dispatch")
{
    GIVEN("a started MIG worker configured for an NvLink profiler field")
    {
        parameters.m_noDcgmValidation = true;
        parameters.m_report           = false;
        parameters.m_fieldId          = DCGM_FI_PROF_NVLINK_RX_BYTES;

        auto attributes                                        = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        attributes.settings.migModeEnabled                     = 1;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = attributes;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

        dcgmDeviceTopology_v2 topology {};
        topology.version                           = dcgmDeviceTopology_version2;
        topology.numGpus                           = 3;
        topology.gpuPaths[0].gpuId                 = 70;
        topology.gpuPaths[0].path                  = DCGM_TOPOLOGY_BOARD;
        topology.gpuPaths[0].localNvLinkIds        = 0;
        topology.gpuPaths[1].gpuId                 = 71;
        topology.gpuPaths[1].path                  = DCGM_TOPOLOGY_NVLINK1;
        topology.gpuPaths[1].localNvLinkIds        = 0x1;
        topology.gpuPaths[2].gpuId                 = 72;
        topology.gpuPaths[2].path                  = DCGM_TOPOLOGY_NVLINK2;
        topology.gpuPaths[2].localNvLinkIds        = 0xF;
        mockDcgmOps.m_topologyState.deviceTopology = topology;

        PhysicalGpu gpu(
            &tester, 153, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu.Init(0) == DCGM_ST_OK);

        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU]    = 153;
        entities[DCGM_FE_GPU_I]  = 154;
        entities[DCGM_FE_GPU_CI] = 155;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU_CI, 155 };
        auto worker = gpu.AddSlice(entities, entity, "MIG-CI-155");
        REQUIRE(worker != nullptr);
        REQUIRE(gpu.StartTests() == DCGM_ST_OK);
        auto workers = workerRuntime.GetWorkers(1);
        REQUIRE(workers.size() == 1);
        REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
        REQUIRE(gpu.ProcessResponse(worker) == DCGM_ST_OK);

        WHEN("the NvLink subtest is dispatched")
        {
            auto const result = gpu.RunTests();

            THEN("MIG mode is rejected without sending a worker run command")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK_FALSE(workerRuntime.HasCommand(workers[0], "R "));
                CHECK_FALSE(gpu.WorkerFinished());
                CHECK_FALSE(gpu.AllReported());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu dispatches run commands for profiler field variants")
{
    auto dispatchRunCommand =
        [this](unsigned int gpuId, unsigned int fieldId, bool targetMaxValue, bool cublas, bool needsNvLinkPeer) {
            parameters.m_noDcgmValidation = true;
            parameters.m_report           = false;
            parameters.m_fieldId          = fieldId;
            parameters.m_duration         = 1.25;
            parameters.m_reportInterval   = 0.125;
            parameters.m_targetMaxValue   = targetMaxValue;
            parameters.m_cublas           = cublas;

            auto attributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
            if (needsNvLinkPeer)
            {
                std::snprintf(attributes.identifiers.pciBusId, sizeof(attributes.identifiers.pciBusId), "0000:72:00.0");
                dcgmDeviceTopology_v2 topology {};
                topology.version                           = dcgmDeviceTopology_version2;
                topology.numGpus                           = 3;
                topology.gpuPaths[0].gpuId                 = 70;
                topology.gpuPaths[0].path                  = DCGM_TOPOLOGY_BOARD;
                topology.gpuPaths[0].localNvLinkIds        = 0;
                topology.gpuPaths[1].gpuId                 = 71;
                topology.gpuPaths[1].path                  = DCGM_TOPOLOGY_NVLINK1;
                topology.gpuPaths[1].localNvLinkIds        = 0x1;
                topology.gpuPaths[2].gpuId                 = 72;
                topology.gpuPaths[2].path                  = DCGM_TOPOLOGY_NVLINK2;
                topology.gpuPaths[2].localNvLinkIds        = 0xF;
                mockDcgmOps.m_topologyState.deviceTopology = topology;
            }
            else
            {
                mockDcgmOps.m_topologyState.deviceReturn = DCGM_ST_NOT_SUPPORTED;
            }
            mockDcgmOps.m_deviceAttributesState.overrideAttributes = attributes;
            mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler();

            PhysicalGpu gpu(
                &tester, gpuId, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
            REQUIRE(gpu.Init(0) == DCGM_ST_OK);

            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
            entities[DCGM_FE_GPU] = gpuId;
            dcgmGroupEntityPair_t entity { DCGM_FE_GPU, gpuId };
            auto worker = gpu.AddSlice(entities, entity, "cuda-" + std::to_string(gpuId));
            REQUIRE(worker != nullptr);
            REQUIRE(gpu.StartTests() == DCGM_ST_OK);
            auto workers = workerRuntime.GetWorkers(1);
            REQUIRE(workers.size() == 1);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu.ProcessResponse(worker) == DCGM_ST_OK);

            auto const result = gpu.RunTests();
            return std::tuple { result, workers[0] };
        };

    GIVEN("started fake CUDA workers for omitted profiler field variants")
    {
        WHEN("GR engine activity is dispatched without targeting max")
        {
            auto const fieldId = DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO;
            CAPTURE(fieldId);
            auto const [result, worker] = dispatchRunCommand(110, fieldId, false, false, false);

            THEN("the worker receives a GR engine run command")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(worker, "R " + std::to_string(fieldId) + " 1.250 0.125 false"));
            }
        }

        WHEN("SM util targets max occupancy")
        {
            auto const fieldId = DCGM_FI_PROF_SM_UTIL_RATIO;
            CAPTURE(fieldId);
            auto const [result, worker] = dispatchRunCommand(111, fieldId, true, false, false);

            THEN("the worker receives an SM util run command with target-max enabled")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(worker, "R " + std::to_string(fieldId) + " 1.250 0.125 true"));
            }
        }

        WHEN("SM occupancy is dispatched without targeting max")
        {
            auto const fieldId = DCGM_FI_PROF_SM_OCCUPANCY_RATIO;
            CAPTURE(fieldId);
            auto const [result, worker] = dispatchRunCommand(112, fieldId, false, false, false);

            THEN("the worker receives an SM occupancy run command")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(worker, "R " + std::to_string(fieldId) + " 1.250 0.125 false"));
            }
        }

        WHEN("PCIe TX bandwidth is dispatched")
        {
            auto const fieldId = DCGM_FI_PROF_PCIE_TX_BYTES;
            CAPTURE(fieldId);
            auto const [result, worker] = dispatchRunCommand(113, fieldId, false, false, false);

            THEN("the worker receives a PCIe TX run command")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(worker, "R " + std::to_string(fieldId) + " 1.250 0.125 false"));
            }
        }

        WHEN("FP32 util is dispatched without CUBLAS")
        {
            auto const fieldId = DCGM_FI_PROF_FP32_UTIL_RATIO;
            CAPTURE(fieldId);
            auto const [result, worker] = dispatchRunCommand(114, fieldId, false, false, false);

            THEN("the worker receives an FP32 run command")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(worker, "R " + std::to_string(fieldId) + " 1.250 0.125 false"));
            }
        }

        WHEN("NvLink TX bandwidth is dispatched with a peer")
        {
            auto const fieldId = DCGM_FI_PROF_NVLINK_TX_BYTES;
            CAPTURE(fieldId);
            auto const [result, worker] = dispatchRunCommand(116, fieldId, false, false, true);

            THEN("the worker receives an NvLink run command that includes the peer bus id")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(worker, "R " + std::to_string(fieldId) + " 1.250 0.125 false"));
                CHECK(workerRuntime.HasCommand(worker, "0000:72:00.0"));
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu formats worker run commands for profiling field variants")
{
    struct CommandCase
    {
        unsigned int fieldId;
        bool targetMaxValue;
        bool cublas;
        bool needsNvLinkPeer;
    };

    auto const testCase = GENERATE(CommandCase { DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, true, false, false },
                                   CommandCase { DCGM_FI_PROF_SM_UTIL_RATIO, false, false, false },
                                   CommandCase { DCGM_FI_PROF_SM_OCCUPANCY_RATIO, true, false, false },
                                   CommandCase { DCGM_FI_PROF_FP16_UTIL_RATIO, false, false, false },
                                   CommandCase { DCGM_FI_PROF_FP64_UTIL_RATIO, false, true, false },
                                   CommandCase { DCGM_FI_PROF_TENSOR_UTIL_RATIO, false, true, false },
                                   CommandCase { DCGM_FI_PROF_DRAM_UTIL_RATIO, false, false, false },
                                   CommandCase { DCGM_FI_PROF_NVLINK_RX_BYTES, false, false, true });

    GIVEN("a single fake CUDA worker for a representative profiling field")
    {
        CAPTURE(testCase.fieldId);
        parameters.m_noDcgmValidation                      = true;
        parameters.m_fieldId                               = testCase.fieldId;
        parameters.m_duration                              = 1.25;
        parameters.m_reportInterval                        = 0.125;
        parameters.m_targetMaxValue                        = testCase.targetMaxValue;
        parameters.m_cublas                                = testCase.cublas;
        parameters.m_report                                = false;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();

        if (testCase.needsNvLinkPeer)
        {
            auto addGpuWithBusId = [this](unsigned int gpuId, char const *busId) {
                DcgmNs::MockDcgmEntity entity({ DCGM_FE_GPU, gpuId });
                auto attributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
                std::snprintf(attributes.identifiers.pciBusId, sizeof(attributes.identifiers.pciBusId), "%s", busId);
                entity.SetDevAttr(attributes);
                mockDcgmOps.AddMockedEntity(entity);
            };
            constexpr char const *sourceBusId = "0000:6e:00.0";
            constexpr char const *peerBusId   = "0000:72:00.0";
            addGpuWithBusId(110, sourceBusId);
            addGpuWithBusId(70, "0000:46:00.0");
            addGpuWithBusId(71, "0000:47:00.0");
            addGpuWithBusId(72, peerBusId);

            dcgmDeviceTopology_v2 topology {};
            topology.version                           = dcgmDeviceTopology_version2;
            topology.numGpus                           = 3;
            topology.gpuPaths[0].gpuId                 = 70;
            topology.gpuPaths[0].path                  = DCGM_TOPOLOGY_BOARD;
            topology.gpuPaths[0].localNvLinkIds        = 0;
            topology.gpuPaths[1].gpuId                 = 71;
            topology.gpuPaths[1].path                  = DCGM_TOPOLOGY_NVLINK1;
            topology.gpuPaths[1].localNvLinkIds        = 0x1;
            topology.gpuPaths[2].gpuId                 = 72;
            topology.gpuPaths[2].path                  = DCGM_TOPOLOGY_NVLINK2;
            topology.gpuPaths[2].localNvLinkIds        = 0xF;
            mockDcgmOps.m_topologyState.deviceTopology = topology;
        }
        else
        {
            mockDcgmOps.m_topologyState.deviceReturn = DCGM_ST_NOT_SUPPORTED;
        }

        auto gpu = std::make_unique<PhysicalGpu>(
            &tester, 110, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        if (testCase.needsNvLinkPeer)
        {
            REQUIRE(gpu->Init(mockDcgmOps.m_handleId) == DCGM_ST_OK);
        }

        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 110;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 110 };
        auto worker = gpu->AddSlice(entities, entity, "cuda-110");
        REQUIRE(worker != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);

        auto workers = workerRuntime.GetWorkers(1);
        REQUIRE(workers.size() == 1);
        REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
        REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);

        WHEN("the field-specific RunTests path is dispatched")
        {
            auto const result             = gpu->RunTests();
            auto const expectedRunCommand = "R " + std::to_string(testCase.fieldId) + " 1.250 0.125 "
                                            + (testCase.targetMaxValue ? "true" : "false");

            THEN("the worker receives a correctly formatted run command")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.GetCallCounts().fork == 1);
                CHECK(workerRuntime.HasCommand(workers[0], expectedRunCommand));
                if (testCase.needsNvLinkPeer)
                {
                    CHECK(workerRuntime.HasCommand(workers[0], "0000:72:00.0"));
                    CHECK_FALSE(workerRuntime.HasCommand(workers[0], "0000:6e:00.0"));
                }
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu processes fake worker startup responses")
{
    GIVEN("a physical GPU with one fake worker process")
    {
        parameters.m_noDcgmValidation                      = true;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 20, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 20;

        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 20 };
        auto worker = gpu->AddSlice(entities, entity, "cuda-20");

        REQUIRE(worker != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);

        auto workers = workerRuntime.GetWorkers(1);
        REQUIRE(workers.size() == 1);

        WHEN("the worker reports startup failure")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "F\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the GPU records a failed but completed startup request")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(gpu->WorkerStarted());
                CHECK(gpu->AllStarted());
                CHECK(gpu->AllFinished());
                CHECK_FALSE(gpu->AnyWorkerRequestFailed());
                CHECK(workerRuntime.HasCommand(workers[0], "M\nE\n"));
            }
        }

        WHEN("the worker sends an error report before startup completes")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "E 5\nhello\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the report is consumed as unexpected report data without marking startup complete")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllStarted());
                CHECK_FALSE(gpu->AnyWorkerRequestFailed());
            }
        }

        WHEN("no complete response is available yet")
        {
            auto const result = gpu->ProcessResponse(worker);

            THEN("startup remains pending without changing worker state")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerStarted());
                CHECK_FALSE(gpu->WorkerRunning());
                CHECK_FALSE(gpu->AllStarted());
            }
        }

        WHEN("a running marker arrives before the startup pass marker")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "S\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the notification is consumed as unexpected running state")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(gpu->WorkerRunning());
                CHECK_FALSE(gpu->WorkerStarted());
                CHECK_FALSE(gpu->AllStarted());
            }
        }

        WHEN("a part-complete marker arrives before startup completes")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "D 0 1\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the notification is ignored without marking the worker started")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerStarted());
                CHECK_FALSE(gpu->AllStarted());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu reports multi-worker startup failures independently")
{
    GIVEN("two fake workers on one physical GPU")
    {
        parameters.m_noDcgmValidation                      = true;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 131, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> firstEntities { { DCGM_FE_GPU, 131 } };
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> secondEntities { { DCGM_FE_GPU, 132 } };
        dcgmGroupEntityPair_t firstEntity { DCGM_FE_GPU, 131 };
        dcgmGroupEntityPair_t secondEntity { DCGM_FE_GPU, 132 };
        auto first  = gpu->AddSlice(firstEntities, firstEntity, "cuda-131");
        auto second = gpu->AddSlice(secondEntities, secondEntity, "cuda-132");
        REQUIRE(first != nullptr);
        REQUIRE(second != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);
        auto workers = workerRuntime.GetWorkers(2);
        REQUIRE(workers.size() == 2);

        WHEN("one worker fails startup before the other reports success")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "F\n"));
            auto const firstResult = gpu->ProcessResponse(first);

            THEN("startup remains incomplete and the failed request is observable")
            {
                CHECK(firstResult == DCGM_ST_OK);
                CHECK(gpu->WorkerStarted());
                CHECK_FALSE(gpu->AllStarted());
                CHECK_FALSE(gpu->AllFinished());
                CHECK(gpu->AnyWorkerRequestFailed());
            }

            AND_WHEN("the second worker reports startup success")
            {
                REQUIRE(workerRuntime.WriteResponse(workers[1], "P\n"));
                auto const secondResult = gpu->ProcessResponse(second);

                THEN("all startup requests are complete but the failed worker still prevents validation")
                {
                    CHECK(secondResult == DCGM_ST_OK);
                    CHECK(gpu->WorkerStarted());
                    CHECK(gpu->AllStarted());
                    CHECK_FALSE(gpu->AllFinished());
                    CHECK(gpu->AnyWorkerRequestFailed());
                    CHECK_FALSE(gpu->IsValidated());
                }
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu processes running and finished worker responses")
{
    GIVEN("a physical GPU with one started fake worker process")
    {
        parameters.m_noDcgmValidation                      = true;
        parameters.m_fieldId                               = DCGM_FI_PROF_SM_UTIL_RATIO;
        parameters.m_report                                = false;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 53, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 53;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 53 };
        auto worker = gpu->AddSlice(entities, entity, "cuda-53");
        REQUIRE(worker != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);

        auto workers = workerRuntime.GetWorkers(1);
        REQUIRE(workers.size() == 1);
        REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
        REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
        REQUIRE(gpu->AllStarted());
        REQUIRE(gpu->RunTests() == DCGM_ST_OK);

        WHEN("the worker reports that it has started running")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "S\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the running transition is exposed through the public worker state")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->AllFinished());
            }
        }

        WHEN("the worker reports a completed part")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "D 2 5\n"));
            auto const result  = gpu->ProcessResponse(worker);
            unsigned int part  = 0;
            unsigned int parts = 0;
            worker->GetParts(part, parts);

            THEN("the part counters are parsed without finishing the worker")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(part == 2);
                CHECK(parts == 5);
                CHECK_FALSE(gpu->AllFinished());
            }
        }

        WHEN("the worker sends a tick while synchronous mode is enabled")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "T 42\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the default tick handler consumes the notification and advances the worker")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(workers[0], "A\n"));
                CHECK_FALSE(gpu->AllFinished());
            }
        }

        WHEN("the worker sends a side-band message during the running phase")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "M 7\nmessage"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the message is consumed without finishing or completing reports")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllFinished());
                CHECK_FALSE(gpu->AllReported());
            }
        }

        WHEN("the worker sends a side-band error during the running phase")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "E 0\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the error is consumed as a report event without finishing the run")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllFinished());
                CHECK_FALSE(gpu->AllReported());
            }
        }

        WHEN("an unknown running notification is received")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "Z ignored\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the line is consumed and the run remains active")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerFinished());
                CHECK_FALSE(gpu->WorkerReported());
            }
        }

        WHEN("the worker passes and then returns message and error reports")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            auto const passResult           = gpu->ProcessResponse(worker);
            bool const workerFinishedOnPass = gpu->WorkerFinished();
            bool const allFinishedOnPass    = gpu->AllFinished();
            REQUIRE(workerRuntime.HasCommand(workers[0], "M\nE\n"));

            REQUIRE(workerRuntime.WriteResponse(workers[0], "M 4\nnote"));
            auto const messageResult = gpu->ProcessResponse(worker);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "E 0\n"));
            auto const errorResult = gpu->ProcessResponse(worker);

            THEN("the GPU advances through finished and reported states")
            {
                CHECK(passResult == DCGM_ST_OK);
                CHECK(messageResult == DCGM_ST_OK);
                CHECK(errorResult == DCGM_ST_OK);
                CHECK(workerFinishedOnPass);
                CHECK(allFinishedOnPass);
                CHECK(gpu->WorkerReported());
                CHECK(gpu->AllReported());
                CHECK(gpu->IsValidated());
            }
        }

        WHEN("a finished worker reports a failure while reports are being collected")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->AllFinished());
            REQUIRE(workerRuntime.HasCommand(workers[0], "M\nE\n"));

            REQUIRE(workerRuntime.WriteResponse(workers[0], "F\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the finished-response state records the worker failure")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(worker->Failed());
                CHECK_FALSE(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllReported());
                CHECK(gpu->IsValidated());
            }
        }

        WHEN("a finished worker emits an unexpected status line")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->AllFinished());
            REQUIRE(workerRuntime.HasCommand(workers[0], "M\nE\n"));

            REQUIRE(workerRuntime.WriteResponse(workers[0], "D 1 1\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the status is consumed without marking the worker reported")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllReported());
            }
        }

        WHEN("another pass marker arrives while reports are expected")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->AllFinished());

            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the unexpected marker is consumed without completing reports")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllReported());
            }
        }

        WHEN("an empty message report arrives before the error report")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->AllFinished());

            REQUIRE(workerRuntime.WriteResponse(workers[0], "M 0\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("message collection alone does not mark the worker fully reported")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllReported());
            }
        }

        WHEN("the running worker reports failure")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "F\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the failure is returned and report collection is requested")
            {
                CHECK(result != DCGM_ST_OK);
                CHECK(workerRuntime.HasCommand(workers[0], "M\nE\n"));
            }
        }

        WHEN("the running worker has no complete response available")
        {
            auto const result = gpu->ProcessResponse(worker);

            THEN("the run remains active and no transition flags are raised")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerFinished());
                CHECK_FALSE(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllFinished());
            }
        }

        WHEN("report collection has no complete response yet")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->AllFinished());

            auto const result = gpu->ProcessResponse(worker);

            THEN("the finished state remains pending for message and error reports")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->WorkerReported());
                CHECK_FALSE(gpu->AllReported());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu reports unsupported and duplicate worker response transitions")
{
    GIVEN("a started fake worker on a GPU configured with an unsupported test field")
    {
        parameters.m_noDcgmValidation                      = true;
        parameters.m_fieldId                               = DCGM_FI_DEV_GPU_NAME;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 91, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 91;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 91 };
        auto worker = gpu->AddSlice(entities, entity, "cuda-91");
        REQUIRE(worker != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);

        auto workers = workerRuntime.GetWorkers(1);
        REQUIRE(workers.size() == 1);
        REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
        REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);

        WHEN("running tests dispatches the unsupported field")
        {
            auto const result = gpu->RunTests();

            THEN("the unsupported field path returns a generic error without flagging a process signal")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(gpu->AllStarted());
                CHECK_FALSE(gpu->AnyWorkerRequestFailed());
            }
        }
    }

    GIVEN("a fake worker that already finished and reported")
    {
        parameters.m_noDcgmValidation                      = true;
        parameters.m_fieldId                               = DCGM_FI_PROF_SM_UTIL_RATIO;
        parameters.m_report                                = false;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 92, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 92;
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 92 };
        auto worker = gpu->AddSlice(entities, entity, "cuda-92");
        REQUIRE(worker != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);

        auto workers = workerRuntime.GetWorkers(1);
        REQUIRE(workers.size() == 1);
        REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
        REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
        REQUIRE(gpu->RunTests() == DCGM_ST_OK);
        REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
        REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
        REQUIRE(workerRuntime.HasCommand(workers[0], "M\nE\n"));
        REQUIRE(workerRuntime.WriteResponse(workers[0], "M 0\n"));
        REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
        REQUIRE(workerRuntime.WriteResponse(workers[0], "E 0\n"));
        REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
        REQUIRE(gpu->AllReported());

        WHEN("another finished-state report is processed")
        {
            REQUIRE(workerRuntime.WriteResponse(workers[0], "E 0\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the duplicate report is rejected explicitly")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(gpu->AllReported());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu destroys its worker contexts with its owning scope")
{
    GIVEN("a physical GPU with a started fake worker")
    {
        parameters.m_noDcgmValidation                      = true;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 21, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities { { DCGM_FE_GPU, 21 } };
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 21 };

        REQUIRE(gpu->AddSlice(std::move(entities), entity, "cuda-21") != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);
        REQUIRE(workerRuntime.GetWorkers(1).size() == 1);

        WHEN("the GPU owner is destroyed")
        {
            gpu.reset();

            THEN("the worker destructor reaps its child process without SIGKILL")
            {
                CHECK(workerRuntime.GetCallCounts().waitPid == 1);
                CHECK(workerRuntime.GetCallCounts().kill == 0);
                CHECK(tester.m_workersFailed == 1);
            }
        }

        WHEN("the GPU explicitly releases its workers")
        {
            gpu->NukeChildren();

            THEN("the worker is deregistered and its child process is reaped")
            {
                CHECK(workerRuntime.GetCallCounts().waitPid == 1);
                CHECK(tester.m_workersFailed == 1);
                CHECK(gpu->AllStarted());
                CHECK(gpu->AllFinished());
                CHECK(gpu->AllReported());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu covers slice creation and child cleanup edge cases")
{
    GIVEN("a GPU whose compute capability cannot be read while creating the first worker")
    {
        parameters.m_noDcgmValidation                          = true;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_entitiesGetLatestValuesState.handler     = [](dcgmHandle_t,
                                                                    dcgmGroupEntityPair_t *,
                                                                    unsigned int,
                                                                    unsigned short *fieldIds,
                                                                    unsigned int,
                                                                    unsigned int,
                                                                    dcgmFieldValue_v2 *value) {
            value->status = DCGM_ST_OK;
            if (*fieldIds == DCGM_FI_DEV_GPU_VIRTUAL_MODE)
            {
                value->value.i64 = DCGM_GPU_VIRTUALIZATION_MODE_NONE;
                return DCGM_ST_OK;
            }
            if (*fieldIds == DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY)
            {
                return DCGM_ST_GENERIC_ERROR;
            }
            return DCGM_ST_OK;
        };

        auto gpu = std::make_unique<PhysicalGpu>(
            &tester, 120, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        REQUIRE(gpu->Init(0) == DCGM_ST_OK);
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities { { DCGM_FE_GPU, 120 } };
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 120 };

        WHEN("the first slice is added")
        {
            auto worker = gpu->AddSlice(entities, entity, "cuda-120");

            THEN("worker creation fails before any child process exists")
            {
                CHECK(worker == nullptr);
                CHECK(gpu->StartTests() == DCGM_ST_NO_DATA);
                CHECK(workerRuntime.GetCallCounts().fork == 0);
            }
        }
    }

    GIVEN("a non-MIG GPU slice that is added more than once")
    {
        parameters.m_noDcgmValidation                      = true;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 121, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities { { DCGM_FE_GPU, 121 } };
        dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 121 };
        auto first = gpu->AddSlice(entities, entity, "cuda-121");
        REQUIRE(first != nullptr);

        WHEN("the same entity is added again")
        {
            auto second            = gpu->AddSlice(entities, entity, "cuda-121-again");
            auto const startResult = gpu->StartTests();
            auto workers           = workerRuntime.GetWorkers(1);

            THEN("the existing worker is reused and only one process is launched")
            {
                CHECK(second == first);
                CHECK(startResult == DCGM_ST_OK);
                REQUIRE(workers.size() == 1);
                CHECK(workerRuntime.GetCallCounts().fork == 1);
            }
        }
    }

    GIVEN("two fake workers that have been launched for one physical GPU")
    {
        parameters.m_noDcgmValidation                      = true;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 122, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> firstEntities { { DCGM_FE_GPU, 122 } };
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> secondEntities { { DCGM_FE_GPU, 123 } };
        dcgmGroupEntityPair_t firstEntity { DCGM_FE_GPU, 122 };
        dcgmGroupEntityPair_t secondEntity { DCGM_FE_GPU, 123 };
        REQUIRE(gpu->AddSlice(firstEntities, firstEntity, "cuda-122") != nullptr);
        REQUIRE(gpu->AddSlice(secondEntities, secondEntity, "cuda-123") != nullptr);
        REQUIRE(gpu->StartTests() == DCGM_ST_OK);
        auto workers = workerRuntime.GetWorkers(2);
        REQUIRE(workers.size() == 2);

        WHEN("the children are nuked gently")
        {
            gpu->NukeChildren(false);

            THEN("each worker receives an exit command and the GPU returns to an empty state")
            {
                CHECK(workerRuntime.HasCommand(workers[0], "X\n"));
                CHECK(workerRuntime.HasCommand(workers[1], "X\n"));
                CHECK(gpu->AllStarted());
                CHECK(gpu->AllFinished());
                CHECK(gpu->AllReported());
            }
        }

        WHEN("the children are aborted forcefully")
        {
            gpu->AbortChildren();

            THEN("the worker list is cleared and failed workers are reported to the tester")
            {
                CHECK(gpu->AllStarted());
                CHECK(gpu->AllFinished());
                CHECK(gpu->AllReported());
                CHECK(workerRuntime.GetCallCounts().waitPid == 2);
                CHECK(tester.m_workersFailed == 2);
            }
        }
    }

    GIVEN("two fake workers where the second process fails to fork")
    {
        parameters.m_noDcgmValidation                      = true;
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler();
        workerRuntimeState.failOnForkCall                  = 2;
        auto gpu                                           = std::make_unique<PhysicalGpu>(
            &tester, 124, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> firstEntities { { DCGM_FE_GPU, 124 } };
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> secondEntities { { DCGM_FE_GPU, 125 } };
        dcgmGroupEntityPair_t firstEntity { DCGM_FE_GPU, 124 };
        dcgmGroupEntityPair_t secondEntity { DCGM_FE_GPU, 125 };
        REQUIRE(gpu->AddSlice(firstEntities, firstEntity, "cuda-124") != nullptr);
        REQUIRE(gpu->AddSlice(secondEntities, secondEntity, "cuda-125") != nullptr);

        WHEN("worker startup reaches the second process")
        {
            auto const result = gpu->StartTests();

            THEN("the startup error is returned and previously started workers are cleaned up")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(workerRuntime.GetCallCounts().fork == 2);
                CHECK(gpu->AllStarted());
                CHECK(gpu->AllFinished());
                CHECK(gpu->AllReported());
            }
        }
    }
}

SCENARIO_METHOD(PhysicalGpuFixture, "PhysicalGpu processes fake worker ticks for architecture-dependent run paths")
{
    GIVEN("deterministic worker tick payloads")
    {
        WHEN("an Ampere 8.6 FP32 data-type worker reports progress with a mid-band sample")
        {
            parameters.m_noDcgmValidation                          = false;
            parameters.m_validate                                  = true;
            parameters.m_report                                    = false;
            parameters.m_fieldId                                   = DCGM_FI_PROF_FP32_UTIL_RATIO;
            parameters.m_duration                                  = 1.25;
            parameters.m_reportInterval                            = 0.125;
            parameters.m_cublas                                    = false;
            mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
            mockDcgmOps.m_entitiesGetLatestValuesState.handler = MakeValidGpuLatestValuesHandler((8ULL << 16) | 6ULL);
            mockDcgmOps.m_topologyState.deviceReturn           = DCGM_ST_NOT_SUPPORTED;

            dcgmFieldValue_v1 sample {};
            sample.version                               = dcgmFieldValue_version1;
            sample.fieldId                               = DCGM_FI_PROF_FP32_UTIL_RATIO;
            sample.fieldType                             = DCGM_FT_DOUBLE;
            sample.status                                = DCGM_ST_OK;
            sample.value.dbl                             = 0.45;
            mockDcgmOps.m_valuesSinceState.values        = std::vector<dcgmFieldValue_v1> { sample };
            mockDcgmOps.m_valuesSinceState.entityGroupId = DCGM_FE_GPU;
            mockDcgmOps.m_valuesSinceState.entityId      = 190;

            auto gpu = std::make_unique<PhysicalGpu>(
                &tester, 190, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
            REQUIRE(gpu->Init(0) == DCGM_ST_OK);
            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities { { DCGM_FE_GPU, 190 } };
            dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 190 };
            auto worker = gpu->AddSlice(entities, entity, "cuda-190");
            REQUIRE(worker != nullptr);
            REQUIRE(gpu->StartTests() == DCGM_ST_OK);
            auto workers = workerRuntime.GetWorkers(1);
            REQUIRE(workers.size() == 1);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->RunTests() == DCGM_ST_OK);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "T 0 0.750 0.250 0.500\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the unoptimized Ampere limit accepts the sample and advances the worker")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(worker->GetValidated());
                CHECK_FALSE(gpu->AllFinished());
                CHECK(workerRuntime.HasCommand(workers[0], "A\n"));
            }
        }

        WHEN("a non-unoptimized FP32 worker reports the same mid-band sample")
        {
            parameters.m_noDcgmValidation                          = false;
            parameters.m_validate                                  = true;
            parameters.m_report                                    = false;
            parameters.m_fieldId                                   = DCGM_FI_PROF_FP32_UTIL_RATIO;
            parameters.m_duration                                  = 1.25;
            parameters.m_reportInterval                            = 0.125;
            parameters.m_cublas                                    = false;
            mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
            mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler(8ULL << 16);
            mockDcgmOps.m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;

            dcgmFieldValue_v1 sample {};
            sample.version                               = dcgmFieldValue_version1;
            sample.fieldId                               = DCGM_FI_PROF_FP32_UTIL_RATIO;
            sample.fieldType                             = DCGM_FT_DOUBLE;
            sample.status                                = DCGM_ST_OK;
            sample.value.dbl                             = 0.45;
            mockDcgmOps.m_valuesSinceState.values        = std::vector<dcgmFieldValue_v1> { sample };
            mockDcgmOps.m_valuesSinceState.entityGroupId = DCGM_FE_GPU;
            mockDcgmOps.m_valuesSinceState.entityId      = 195;

            auto gpu = std::make_unique<PhysicalGpu>(
                &tester, 195, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
            REQUIRE(gpu->Init(0) == DCGM_ST_OK);
            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities { { DCGM_FE_GPU, 195 } };
            dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 195 };
            auto worker = gpu->AddSlice(entities, entity, "cuda-195");
            REQUIRE(worker != nullptr);
            REQUIRE(gpu->StartTests() == DCGM_ST_OK);
            auto workers = workerRuntime.GetWorkers(1);
            REQUIRE(workers.size() == 1);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->RunTests() == DCGM_ST_OK);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "T 0 0.750 0.250 0.500\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the stricter limit rejects the sample and does not advance")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(worker->GetValidated());
                CHECK_FALSE(workerRuntime.HasCommand(workers[0], "A\n"));
            }
        }

        WHEN("an NvLink worker reports progress")
        {
            parameters.m_noDcgmValidation = true;
            parameters.m_report           = false;
            parameters.m_fieldId          = DCGM_FI_PROF_NVLINK_RX_BYTES;
            parameters.m_duration         = 1.25;
            parameters.m_reportInterval   = 0.125;
            auto attributes               = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
            std::snprintf(attributes.identifiers.pciBusId, sizeof(attributes.identifiers.pciBusId), "0000:72:00.0");
            mockDcgmOps.m_deviceAttributesState.overrideAttributes = attributes;
            mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler(10ULL << 16);

            dcgmDeviceTopology_v2 topology {};
            topology.version                           = dcgmDeviceTopology_version2;
            topology.numGpus                           = 3;
            topology.gpuPaths[0].gpuId                 = 70;
            topology.gpuPaths[0].path                  = DCGM_TOPOLOGY_BOARD;
            topology.gpuPaths[0].localNvLinkIds        = 0;
            topology.gpuPaths[1].gpuId                 = 71;
            topology.gpuPaths[1].path                  = DCGM_TOPOLOGY_NVLINK1;
            topology.gpuPaths[1].localNvLinkIds        = 0x1;
            topology.gpuPaths[2].gpuId                 = 72;
            topology.gpuPaths[2].path                  = DCGM_TOPOLOGY_NVLINK2;
            topology.gpuPaths[2].localNvLinkIds        = 0xF;
            mockDcgmOps.m_topologyState.deviceTopology = topology;

            auto gpu = std::make_unique<PhysicalGpu>(
                &tester, 191, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
            REQUIRE(gpu->Init(0) == DCGM_ST_OK);
            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities { { DCGM_FE_GPU, 191 } };
            dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 191 };
            auto worker = gpu->AddSlice(entities, entity, "cuda-191");
            REQUIRE(worker != nullptr);
            REQUIRE(gpu->StartTests() == DCGM_ST_OK);
            auto workers = workerRuntime.GetWorkers(1);
            REQUIRE(workers.size() == 1);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->RunTests() == DCGM_ST_OK);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "T 5 0.500 100.000 200.000\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the NvLink tick handler advances the worker")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(worker->GetValidated());
                CHECK(workerRuntime.HasCommand(workers[0], "A\n"));
            }
        }

        WHEN("a PCIe worker reports zero bandwidth on a newer link generation")
        {
            parameters.m_noDcgmValidation                          = true;
            parameters.m_report                                    = false;
            parameters.m_fieldId                                   = DCGM_FI_PROF_PCIE_TX_BYTES;
            parameters.m_duration                                  = 1.25;
            parameters.m_reportInterval                            = 0.125;
            mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
            mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakePcieMetadataLatestValuesHandler();
            mockDcgmOps.m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;

            auto gpu = std::make_unique<PhysicalGpu>(
                &tester, 193, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
            REQUIRE(gpu->Init(0) == DCGM_ST_OK);
            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities { { DCGM_FE_GPU, 193 } };
            dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 193 };
            auto worker = gpu->AddSlice(entities, entity, "cuda-193");
            REQUIRE(worker != nullptr);
            REQUIRE(gpu->StartTests() == DCGM_ST_OK);
            auto workers = workerRuntime.GetWorkers(1);
            REQUIRE(workers.size() == 1);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->RunTests() == DCGM_ST_OK);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "T 0 0.750 0.000 0.000\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the PCIe metadata is accepted, capped, and the worker advances")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(worker->GetValidated());
                CHECK(gpu->IsValidated());
                CHECK(workerRuntime.HasCommand(workers[0], "A\n"));
            }
        }

        WHEN("PCIe link width lookup fails during a tick")
        {
            parameters.m_noDcgmValidation                          = true;
            parameters.m_report                                    = false;
            parameters.m_fieldId                                   = DCGM_FI_PROF_PCIE_RX_BYTES;
            parameters.m_duration                                  = 1.25;
            parameters.m_reportInterval                            = 0.125;
            mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
            mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakePcieLinkWidthFailsLatestValuesHandler();
            mockDcgmOps.m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;

            auto gpu = std::make_unique<PhysicalGpu>(
                &tester, 194, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
            REQUIRE(gpu->Init(0) == DCGM_ST_OK);
            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities { { DCGM_FE_GPU, 194 } };
            dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 194 };
            auto worker = gpu->AddSlice(entities, entity, "cuda-194");
            REQUIRE(worker != nullptr);
            REQUIRE(gpu->StartTests() == DCGM_ST_OK);
            auto workers = workerRuntime.GetWorkers(1);
            REQUIRE(workers.size() == 1);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->RunTests() == DCGM_ST_OK);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "T 0 0.750 0.000 0.000\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the GPU marks the sample invalid and asks the worker to quit")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_FALSE(gpu->IsValidated());
                CHECK(workerRuntime.HasCommand(workers[0], "Q\n"));
            }
        }

        WHEN("a MIG DRAM worker reports progress")
        {
            parameters.m_noDcgmValidation                          = true;
            parameters.m_report                                    = false;
            parameters.m_fieldId                                   = DCGM_FI_PROF_DRAM_UTIL_RATIO;
            parameters.m_duration                                  = 1.25;
            parameters.m_reportInterval                            = 0.125;
            auto attributes                                        = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
            attributes.settings.migModeEnabled                     = 1;
            mockDcgmOps.m_deviceAttributesState.overrideAttributes = attributes;
            mockDcgmOps.m_entitiesGetLatestValuesState.handler     = MakeValidGpuLatestValuesHandler(9ULL << 16);
            mockDcgmOps.m_topologyState.deviceReturn               = DCGM_ST_NOT_SUPPORTED;

            auto gpu = std::make_unique<PhysicalGpu>(
                &tester, 192, parameters, dcgmOps, cudaOps, cudaWorkerFactory.AsAnyPtr(), runtimeFactory.Create());
            REQUIRE(gpu->Init(0) == DCGM_ST_OK);
            std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities {
                { DCGM_FE_GPU, 192 },
                { DCGM_FE_GPU_I, 193 },
                { DCGM_FE_GPU_CI, 194 },
            };
            dcgmGroupEntityPair_t entity { DCGM_FE_GPU_CI, 194 };
            auto worker = gpu->AddSlice(entities, entity, "MIG-CI-194");
            REQUIRE(worker != nullptr);
            REQUIRE(gpu->StartTests() == DCGM_ST_OK);
            auto workers = workerRuntime.GetWorkers(1);
            REQUIRE(workers.size() == 1);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "P\n"));
            REQUIRE(gpu->ProcessResponse(worker) == DCGM_ST_OK);
            REQUIRE(gpu->RunTests() == DCGM_ST_OK);
            REQUIRE(workerRuntime.WriteResponse(workers[0], "T 1 0.500 0.100 0.250 1024.000 2048.000 1\n"));
            auto const result = gpu->ProcessResponse(worker);

            THEN("the DRAM handler aggregates the compute-instance sample and advances the worker")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(worker->GetValidated());
                CHECK(workerRuntime.HasCommand(workers[0], "A\n"));
            }
        }
    }
}
