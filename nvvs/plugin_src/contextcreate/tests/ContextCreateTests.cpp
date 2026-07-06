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
#include <ContextCreate.h>
#include <ContextCreatePlugin.h>
#include <PluginStrings.h>
#include <catch2/catch_all.hpp>

#include <memory.h>

#include <DcgmLib.h>
#include <MockDcgmLib.h>

namespace
{

dcgmHandle_t GetDcgmHandle(DcgmNs::MockDcgmLib &dcgmMockLib)
{
    dcgmHandle_t ret;
    REQUIRE(dcgmMockLib.dcgmInit() == DCGM_ST_OK);
    REQUIRE(dcgmMockLib.dcgmConnect_v2("localhost", nullptr, &ret) == DCGM_ST_OK);
    return ret;
}

} //namespace

auto cuCtxCreateMock
    = [](CUcontext *pctx [[maybe_unused]], unsigned int flags [[maybe_unused]], CUdevice device [[maybe_unused]]) {
          return CUDA_ERROR_INVALID_DEVICE;
      };

auto cuCtxCreateSuccessMock = [](CUcontext *pctx, unsigned int flags [[maybe_unused]], CUdevice device) {
    *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(device + 1));
    return CUDA_SUCCESS;
};

auto cuCtxCreateSkipMock
    = [](CUcontext *pctx [[maybe_unused]], unsigned int flags [[maybe_unused]], CUdevice device [[maybe_unused]]) {
          return CUDA_ERROR_UNKNOWN;
      };

auto cuCtxDestroyMock = [](CUcontext pctx [[maybe_unused]]) {
    return CUDA_SUCCESS;
};

auto cuGetErrorStringMock = [](CUresult cuSt [[maybe_unused]], const char **errStr) {
    *errStr = "ERROR";
    return CUDA_SUCCESS;
};

dcgmDiagPluginEntityList_v1 MakeContextCreateEntityList(unsigned int gpuCount)
{
    dcgmDiagPluginEntityList_v1 entityList {};
    entityList.numEntities = gpuCount;
    for (unsigned int i = 0; i < gpuCount; ++i)
    {
        entityList.entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList.entities[i].entity.entityId      = i;
        entityList.entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    return entityList;
}

std::unique_ptr<dcgmDiagPluginEntityList_v1> MakeContextCreateEntityListPtr(unsigned int gpuCount)
{
    auto entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    *entityList     = MakeContextCreateEntityList(gpuCount);
    return entityList;
}

dcgmDiagPluginTestParameter_t MakeStringParameter(std::string_view name, std::string_view value)
{
    dcgmDiagPluginTestParameter_t param {};
    snprintf(param.parameterName, sizeof(param.parameterName), "%.*s", static_cast<int>(name.size()), name.data());
    snprintf(param.parameterValue, sizeof(param.parameterValue), "%.*s", static_cast<int>(value.size()), value.data());
    param.type = DcgmPluginParamString;
    return param;
}

TEST_CASE("ContextCreate::CanCreateContext")
{
    SECTION("successful context creation returns CTX_CREATED")
    {
        auto plugin     = std::make_unique<ContextCreatePlugin>((dcgmHandle_t)0);
        auto entityList = MakeContextCreateEntityListPtr(2);
        plugin->InitializeForEntityList(plugin->GetCtxCreateTestName(), *entityList);

        ContextCreate testContextCreate(plugin->GetInfoStruct().defaultTestParameters, plugin.get(), (dcgmHandle_t)0);
        std::vector<std::unique_ptr<ContextCreateDevice>> theDevices(entityList->numEntities);
        for (size_t i = 0; auto &ptr : theDevices)
        {
            ptr           = std::make_unique<ContextCreateDevice>();
            ptr->gpuId    = i;
            ptr->cuDevice = i++;
        }

        int result = testContextCreate.CanCreateContext(theDevices, cuCtxCreateSuccessMock, cuCtxDestroyMock);

        REQUIRE(result == CTX_CREATED);
        CHECK(plugin->GetErrors(plugin->GetCtxCreateTestName()).empty());
    }

    SECTION("CUDA unknown marks the context creation as skipped")
    {
        auto plugin     = std::make_unique<ContextCreatePlugin>((dcgmHandle_t)0);
        auto entityList = MakeContextCreateEntityListPtr(1);
        plugin->InitializeForEntityList(plugin->GetCtxCreateTestName(), *entityList);

        ContextCreate testContextCreate(plugin->GetInfoStruct().defaultTestParameters, plugin.get(), (dcgmHandle_t)0);
        std::vector<std::unique_ptr<ContextCreateDevice>> theDevices(entityList->numEntities);
        theDevices[0]        = std::make_unique<ContextCreateDevice>();
        theDevices[0]->gpuId = 0;

        int result = testContextCreate.CanCreateContext(theDevices, cuCtxCreateSkipMock, cuCtxDestroyMock);

        REQUIRE(result == (CTX_CREATED | CTX_SKIP));
        CHECK(plugin->GetErrors(plugin->GetCtxCreateTestName()).empty());
    }

    SECTION("Negative testing of DCGM_FR_CUDA_CONTEXT")
    {
        std::unique_ptr<DcgmNs::MockDcgmLib> dcgmMockLib = std::make_unique<DcgmNs::MockDcgmLib>();
        dcgmHandle_t handle                              = GetDcgmHandle(*dcgmMockLib);
        auto plugin                                      = std::make_unique<ContextCreatePlugin>(handle);

        auto pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

        dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

        dcgmGroupEntityPair_t entity0 = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        dcgmGroupEntityPair_t entity1 = { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
        dcgmGroupEntityPair_t entity2 = { .entityGroupId = DCGM_FE_GPU, .entityId = 2 };
        dcgmGroupEntityPair_t entity3 = { .entityGroupId = DCGM_FE_GPU, .entityId = 3 };

        entityList.numEntities        = 4;
        entityList.entities[0].entity = entity0;
        entityList.entities[1].entity = entity1;
        entityList.entities[2].entity = entity2;
        entityList.entities[3].entity = entity3;

        plugin->InitializeForEntityList(plugin->GetCtxCreateTestName(), entityList);

        ContextCreate testContextCreate(plugin->GetInfoStruct().defaultTestParameters, plugin.get(), handle);

        std::vector<std::unique_ptr<ContextCreateDevice>> theDevices(entityList.numEntities);
        for (size_t i = 0; auto &ptr : theDevices)
        {
            ptr        = std::make_unique<ContextCreateDevice>();
            ptr->gpuId = i++;
        }

        int result
            = testContextCreate.CanCreateContext(theDevices, cuCtxCreateMock, cuCtxDestroyMock, cuGetErrorStringMock);
        REQUIRE(result == (CTX_CREATED | CTX_FAIL));

        auto errors = plugin->GetErrors(plugin->GetCtxCreateTestName());
        REQUIRE(errors.size() == 4);

        for (unsigned int i = 0; auto const &error : errors)
        {
            DcgmError d { i };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_CONTEXT, d, i, "ERROR");
            REQUIRE(d.GetMessage() == error.GetMessage());
            i++;
        }

        nvvsPluginEntityErrors_t errorsPerEntity = plugin->GetEntityErrors(plugin->GetCtxCreateTestName());

        unsigned int count { 0 };
        for (auto const &[entityPair, diagErrors] : errorsPerEntity)
        {
            // Make sure each of the GPU entity has exact one error, other entities don't have any error
            if (entityPair.entityGroupId == DCGM_FE_GPU)
            {
                REQUIRE(diagErrors.size() == 1);
                REQUIRE(diagErrors[0].entity.entityGroupId == entityPair.entityGroupId);
                REQUIRE(diagErrors[0].entity.entityId == entityPair.entityId);
                DcgmError expectedError { entityPair.entityId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_CONTEXT, expectedError, entityPair.entityId, "ERROR");
                REQUIRE(std::string(diagErrors[0].msg) == expectedError.GetMessage());
                count++;
            }
            else
            {
                REQUIRE(diagErrors.size() == 0);
            }
        }
        REQUIRE(count == entityList.numEntities);
    }
}


TEST_CASE("ContextCreate initializes and reports safe setup failures")
{
    auto plugin     = std::make_unique<ContextCreatePlugin>((dcgmHandle_t)0);
    auto entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    plugin->InitializeForEntityList(plugin->GetCtxCreateTestName(), *entityList);
    ContextCreate testContextCreate(plugin->GetInfoStruct().defaultTestParameters, plugin.get(), (dcgmHandle_t)0);

    SECTION("Init reports invalid DCGM group handle")
    {
        std::string error = testContextCreate.Init(*entityList);
        CHECK_FALSE(error.empty());
    }

    SECTION("Run fails when initialization reports an error")
    {
        CHECK(testContextCreate.Run(*entityList) == CONTEXT_CREATE_FAIL);
    }

    SECTION("ContextCreateDevice reports DCGM attribute lookup failures")
    {
        DcgmHandle handle((dcgmHandle_t)0);
        CHECK_THROWS_AS(ContextCreateDevice(plugin->GetCtxCreateTestName(), 0, "", plugin.get(), handle), DcgmError);
    }
}

TEST_CASE("ContextCreatePlugin::Go handles disabled and empty-input cases")
{
    GIVEN("a context create plugin")
    {
        auto plugin = std::make_unique<ContextCreatePlugin>((dcgmHandle_t)0);

        SECTION("unknown test names return without creating results")
        {
            auto entityList = MakeContextCreateEntityListPtr(1);

            plugin->Go("not_context_create", entityList.get(), 0, nullptr);

            CHECK_THROWS_AS(plugin->GetErrors(plugin->GetCtxCreateTestName()), std::out_of_range);
        }

        SECTION("null entity info returns without creating results")
        {
            plugin->Go(plugin->GetCtxCreateTestName(), nullptr, 0, nullptr);

            CHECK_THROWS_AS(plugin->GetErrors(plugin->GetCtxCreateTestName()), std::out_of_range);
        }

        SECTION("disabled context create test is skipped without running CUDA setup")
        {
            auto entityList                     = MakeContextCreateEntityListPtr(1);
            dcgmDiagPluginTestParameter_t param = MakeStringParameter(CTXCREATE_IS_ALLOWED, "False");

            plugin->Go(plugin->GetCtxCreateTestName(), entityList.get(), 1, &param);

            REQUIRE(plugin->GetGpuResults(plugin->GetCtxCreateTestName()).at(0) == NVVS_RESULT_SKIP);
            CHECK(plugin->GetErrors(plugin->GetCtxCreateTestName()).empty());
        }

        SECTION("empty GPU list fails with an empty-list error")
        {
            auto entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

            plugin->Go(plugin->GetCtxCreateTestName(), entityList.get(), 0, nullptr);

            auto const &errors = plugin->GetErrors(plugin->GetCtxCreateTestName());
            REQUIRE(errors.size() == 1);
            CHECK(errors[0].GetCode() == DCGM_FR_EMPTY_GPU_LIST);
        }
    }
}
