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
#include <catch2/catch_all.hpp>

#include <memory.h>

#include <DcgmLib.h>
#include <DcgmLibMock.h>

namespace
{

dcgmHandle_t GetDcgmHandle(DcgmNs::DcgmLibMock &dcgmMockLib)
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

auto cuCtxDestroyMock = [](CUcontext pctx [[maybe_unused]]) {
    return CUDA_SUCCESS;
};

auto cuGetErrorStringMock = [](CUresult cuSt [[maybe_unused]], const char **errStr) {
    *errStr = "ERROR";
    return CUDA_SUCCESS;
};

TEST_CASE("ContextCreate::CanCreateContext")
{
    SECTION("Negative testing of DCGM_FR_CUDA_CONTEXT")
    {
        std::unique_ptr<DcgmNs::DcgmLibMock> dcgmMockLib = std::make_unique<DcgmNs::DcgmLibMock>();
        dcgmHandle_t handle                              = GetDcgmHandle(*dcgmMockLib);
        ContextCreatePlugin plugin(handle);

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

        plugin.InitializeForEntityList(plugin.GetCtxCreateTestName(), entityList);

        ContextCreate testContextCreate(plugin.GetInfoStruct().defaultTestParameters, &plugin, handle);

        std::vector<std::unique_ptr<ContextCreateDevice>> theDevices(entityList.numEntities);
        for (size_t i = 0; auto &ptr : theDevices)
        {
            ptr        = std::make_unique<ContextCreateDevice>();
            ptr->gpuId = i++;
        }

        int result
            = testContextCreate.CanCreateContext(theDevices, cuCtxCreateMock, cuCtxDestroyMock, cuGetErrorStringMock);
        REQUIRE(result == (CTX_CREATED | CTX_FAIL));

        auto errors = plugin.GetErrors(plugin.GetCtxCreateTestName());
        REQUIRE(errors.size() == 4);

        for (unsigned int i = 0; auto const &error : errors)
        {
            DcgmError d { i };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_CONTEXT, d, i, "ERROR");
            REQUIRE(d.GetMessage() == error.GetMessage());
            i++;
        }

        nvvsPluginEntityErrors_t errorsPerEntity = plugin.GetEntityErrors(plugin.GetCtxCreateTestName());

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
