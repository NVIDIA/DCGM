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

#include "FakeDcgmDynamicLib.h"

#include <catch2/catch_test_macros.hpp>

#include <StderrCapture.hpp>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <string>

extern "C" unsigned int dcgmCpuHierarchyCpuOwnsCore(unsigned int coreId,
                                                    dcgmCpuHierarchyOwnedCores_v1 const *ownedCores);


SCENARIO("dcgm_stub forwards calls through a dynamically loaded libdcgm")
{
    GIVEN("a fake libdcgm colocated with the test executable")
    {
        WHEN("the stub initializes and forwards supported and unsupported symbols")
        {
            dcgmHandle_t handle        = 0;
            dcgmReturn_t const initRet = dcgmInit();
            REQUIRE(initRet == DCGM_ST_OK);

            dcgmReturn_t const startRet = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &handle);
            REQUIRE(startRet == DCGM_ST_OK);

            unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] {};
            int gpuCount                 = 0;
            dcgmReturn_t const deviceRet = dcgmGetAllDevices(handle, gpuIds, &gpuCount);
            REQUIRE(deviceRet == DCGM_ST_OK);
            REQUIRE(handle == FAKE_DCGM_HANDLE);
            REQUIRE(gpuCount == FAKE_DCGM_GPU_COUNT);
            REQUIRE(gpuIds[0] == FAKE_DCGM_GPU_ID_0);
            REQUIRE(gpuIds[1] == FAKE_DCGM_GPU_ID_1);

            dcgmGpuGrp_t groupIds[DCGM_MAX_NUM_GROUPS] {};
            unsigned int groupCount     = 0;
            dcgmReturn_t const groupRet = dcgmGroupGetAllIds(handle, groupIds, &groupCount);

            dcgmDeviceAttributes_t invalidAttributes {};
            dcgmReturn_t const invalidAttrRet = dcgmGetDeviceAttributes(handle, gpuIds[0], &invalidAttributes);

            dcgmDeviceAttributes_t attributes {};
            attributes.version         = dcgmDeviceAttributes_version;
            dcgmReturn_t const attrRet = dcgmGetDeviceAttributes(handle, gpuIds[0], &attributes);

            char const *moduleName     = nullptr;
            dcgmReturn_t const nameRet = dcgmModuleIdToName(DcgmModuleIdCore, &moduleName);
            char const *errorMessage   = errorString(DCGM_ST_BADPARAM);
            dcgmCpuHierarchyOwnedCores_v1 ownedCores {};
            unsigned int const ownsCore = dcgmCpuHierarchyCpuOwnsCore(FAKE_DCGM_OWNED_CORE_ID, &ownedCores);

            StderrCapture capture;
            DcgmEntityStatus_t status       = DcgmEntityStatusOk;
            dcgmReturn_t const missingRet   = dcgmGetGpuStatus(handle, gpuIds[0], &status);
            dcgmReturn_t const stopRet      = dcgmStopEmbedded(handle);
            dcgmReturn_t const shutdownRet  = dcgmShutdown();
            std::string const stderrMessage = capture.GetOutput();

            THEN("return values and out parameters come from the loaded library")
            {
                CHECK(groupRet == DCGM_ST_OK);
                CHECK(groupCount == FAKE_DCGM_GROUP_COUNT);
                CHECK(groupIds[0] == FAKE_DCGM_GROUP_ID);
                CHECK(invalidAttrRet == DCGM_ST_VER_MISMATCH);
                CHECK(attrRet == DCGM_ST_OK);
                CHECK(std::string(attributes.identifiers.deviceName) == FAKE_DCGM_DEVICE_NAME);
                CHECK(nameRet == DCGM_ST_OK);
                REQUIRE(moduleName != nullptr);
                CHECK(std::string(moduleName) == FAKE_DCGM_CORE_MODULE_NAME);
                REQUIRE(errorMessage != nullptr);
                CHECK(std::string(errorMessage) == FAKE_DCGM_BADPARAM_STRING);
                CHECK(ownsCore == 1);
                CHECK(missingRet == DCGM_ST_UNINITIALIZED);
                CHECK(status == DcgmEntityStatusOk);
                CHECK(stderrMessage.find("Failed to find dcgmGetGpuStatus") != std::string::npos);
                CHECK(stopRet == DCGM_ST_OK);
                CHECK(shutdownRet == DCGM_ST_OK);
            }
        }
    }
}
