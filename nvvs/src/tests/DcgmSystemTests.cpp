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

#include <DcgmSystem.h>
#include <MockDcgmOps.h>

#include <catch2/catch_test_macros.hpp>

#include <map>
#include <vector>

class DcgmSystemTestFixture
{
protected:
    DcgmNs::MockDcgmOps mockDcgmOps;
    long long gpuMaxOpTempVal    = 0;
    long long memoryMaxOpTempVal = 0;
    long long cudaDriverVersion  = 0;
    std::map<unsigned short, dcgmReturn_t> fieldRet;

    DcgmSystem MakeSystem(dcgmHandle_t handle)
    {
        return DcgmSystem(DcgmNs::DcgmOpsAnyPtr(&mockDcgmOps), handle);
    }

    void InstallLatestValuesHandler()
    {
        mockDcgmOps.m_entitiesGetLatestValuesState.handler = [this](dcgmHandle_t,
                                                                    dcgmGroupEntityPair_t *entities,
                                                                    unsigned int entityCount,
                                                                    unsigned short *fields,
                                                                    unsigned int fieldCount,
                                                                    unsigned int,
                                                                    dcgmFieldValue_v2 *values) -> dcgmReturn_t {
            if (entities == nullptr || fields == nullptr || values == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }

            unsigned int valueIndex = 0;
            for (unsigned int entityIndex = 0; entityIndex < entityCount; ++entityIndex)
            {
                for (unsigned int fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex)
                {
                    auto const fieldId = fields[fieldIndex];
                    if (auto retIt = fieldRet.find(fieldId); retIt != fieldRet.end() && retIt->second != DCGM_ST_OK)
                    {
                        return retIt->second;
                    }

                    auto &value         = values[valueIndex++];
                    value.entityGroupId = entities[entityIndex].entityGroupId;
                    value.entityId      = entities[entityIndex].entityId;
                    value.fieldId       = fieldId;
                    value.fieldType     = DCGM_FT_INT64;
                    value.status        = DCGM_ST_OK;
                    if (fieldId == DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS)
                    {
                        value.value.i64 = gpuMaxOpTempVal;
                    }
                    else if (fieldId == DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS)
                    {
                        value.value.i64 = memoryMaxOpTempVal;
                    }
                    else if (fieldId == DCGM_FI_CUDA_DRIVER_VERSION)
                    {
                        value.value.i64 = cudaDriverVersion;
                    }
                    else
                    {
                        value.value.i64 = 0;
                    }
                }
            }

            return DCGM_ST_OK;
        };
    }
};

SCENARIO_METHOD(DcgmSystemTestFixture, "DcgmSystemBase: reports unsupported operations")
{
    GIVEN("a base system interface")
    {
        DcgmSystemBase system;

        WHEN("callers use the default implementation")
        {
            dcgmDeviceAttributes_t attributes {};
            DcgmEntityStatus_t gpuStatus {};
            std::vector<unsigned int> gpuIds { 7 };
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
            dcgmFieldValue_v2 value {};

            auto const deviceAttributesRet = system.GetDeviceAttributes(0, attributes);
            auto const gpuStatusRet        = system.GetGpuStatus(0, &gpuStatus);
            auto const supportedDevicesRet = system.GetAllSupportedDevices(gpuIds);
            auto const allDevicesRet       = system.GetAllDevices(gpuIds);
            auto const latestValueRet      = system.GetGpuLatestValue(0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, value);
            auto const latestValuesRet     = system.GetLatestValuesForGpus({ 0 }, fieldIds, 0, nullptr, nullptr);
            auto const cudaMajor           = system.GetCudaMajorVersion();
            auto const cudaMinor           = system.GetCudaMinorVersion();
            auto const initialized         = system.IsInitialized();
            auto const chipArchitecture    = system.GetGpuChipArchitecture(0);

            THEN("all DCGM queries report not supported without mutating initialization state")
            {
                CHECK(deviceAttributesRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(gpuStatusRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(supportedDevicesRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(allDevicesRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(latestValueRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(latestValuesRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(cudaMajor == 0);
                CHECK(cudaMinor == 0);
                CHECK_FALSE(initialized);
                REQUIRE_FALSE(chipArchitecture.has_value());
                CHECK(chipArchitecture.error() == DCGM_ST_NOT_SUPPORTED);
            }
        }
    }
}

SCENARIO_METHOD(DcgmSystemTestFixture, "DcgmSystem: validates handle-dependent requests before calling DCGM")
{
    GIVEN("a system without a DCGM handle")
    {
        DcgmSystem system(DcgmNs::DcgmOpsAnyPtr(&mockDcgmOps), 0);

        WHEN("handle-dependent queries are made")
        {
            dcgmDeviceAttributes_t attributes {};
            DcgmEntityStatus_t gpuStatus {};
            std::vector<unsigned int> gpuIds { 1, 2 };
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
            dcgmFieldValue_v2 value {};

            auto const deviceAttributesRet = system.GetDeviceAttributes(0, attributes);
            auto const gpuStatusRet        = system.GetGpuStatus(0, &gpuStatus);
            auto const supportedDevicesRet = system.GetAllSupportedDevices(gpuIds);
            auto const allDevicesRet       = system.GetAllDevices(gpuIds);
            auto const latestValueRet      = system.GetGpuLatestValue(0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, value);
            auto const latestValuesRet     = system.GetLatestValuesForGpus({ 0 }, fieldIds, 0, nullptr, nullptr);
            auto const chipArchitecture    = system.GetGpuChipArchitecture(0);

            THEN("the calls fail with bad parameter, preserve caller-owned lists, and do not call DCGM")
            {
                CHECK_FALSE(system.IsInitialized());
                CHECK(deviceAttributesRet == DCGM_ST_BADPARAM);
                CHECK(gpuStatusRet == DCGM_ST_BADPARAM);
                CHECK(supportedDevicesRet == DCGM_ST_BADPARAM);
                CHECK(gpuIds == std::vector<unsigned int> { 1, 2 });
                CHECK(allDevicesRet == DCGM_ST_BADPARAM);
                CHECK(gpuIds == std::vector<unsigned int> { 1, 2 });
                CHECK(latestValueRet == DCGM_ST_BADPARAM);
                CHECK(latestValuesRet == DCGM_ST_BADPARAM);
                REQUIRE_FALSE(chipArchitecture.has_value());
                CHECK(chipArchitecture.error() == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_deviceAttributesState.callCount == 0);
                CHECK(mockDcgmOps.m_queryState.gpuStatusCallCount == 0);
                CHECK(mockDcgmOps.m_discoveryState.chipArchitectureCallCount == 0);
                CHECK(mockDcgmOps.m_discoveryState.allSupportedDevicesCallCount == 0);
                CHECK(mockDcgmOps.m_discoveryState.allDevicesCallCount == 0);
                CHECK(mockDcgmOps.m_entitiesGetLatestValuesState.callCount == 0);
            }
        }

        WHEN("the CUDA version is requested")
        {
            auto const major = system.GetCudaMajorVersion();
            auto const minor = system.GetCudaMinorVersion();

            THEN("the missing handle is reported as an unknown CUDA version")
            {
                CHECK(major == 0);
                CHECK(minor == 0);
            }
        }
    }

    GIVEN("a system with a non-null handle")
    {
        DcgmSystem system(DcgmNs::DcgmOpsAnyPtr(&mockDcgmOps), static_cast<dcgmHandle_t>(1));

        WHEN("a GPU id is outside the valid device range")
        {
            dcgmDeviceAttributes_t attributes {};
            DcgmEntityStatus_t gpuStatus {};

            auto const deviceAttributesRet = system.GetDeviceAttributes(DCGM_MAX_NUM_DEVICES, attributes);
            auto const gpuStatusRet        = system.GetGpuStatus(DCGM_MAX_NUM_DEVICES, &gpuStatus);

            THEN("validation fails before the DCGM API is called")
            {
                CHECK(system.IsInitialized());
                CHECK(deviceAttributesRet == DCGM_ST_BADPARAM);
                CHECK(gpuStatusRet == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_deviceAttributesState.callCount == 0);
                CHECK(mockDcgmOps.m_queryState.gpuStatusCallCount == 0);
            }
        }

        WHEN("GetGpuStatus is called with a null status pointer")
        {
            auto const gpuStatusRet = system.GetGpuStatus(0, nullptr);

            THEN("validation fails before the DCGM API is called")
            {
                CHECK(gpuStatusRet == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_queryState.gpuStatusCallCount == 0);
            }
        }
    }

    GIVEN("a system constructed with the mock's valid handle id")
    {
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_queryState.gpuStatus                     = DcgmEntityStatusOk;
        DcgmSystem system                                      = MakeSystem(mockDcgmOps.m_handleId);

        WHEN("device attributes and GPU status are requested for a valid GPU id")
        {
            dcgmDeviceAttributes_t attributes {};
            DcgmEntityStatus_t gpuStatus {};

            auto const deviceAttributesRet = system.GetDeviceAttributes(0, attributes);
            auto const gpuStatusRet        = system.GetGpuStatus(0, &gpuStatus);

            THEN("the requests succeed through DCGM")
            {
                CHECK(system.IsInitialized());
                CHECK(deviceAttributesRet == DCGM_ST_OK);
                CHECK(gpuStatusRet == DCGM_ST_OK);
                CHECK(gpuStatus == DcgmEntityStatusOk);
                CHECK(mockDcgmOps.m_deviceAttributesState.callCount == 1);
                CHECK(mockDcgmOps.m_queryState.gpuStatusCallCount == 1);
            }
        }
    }
}

SCENARIO_METHOD(DcgmSystemTestFixture, "DcgmSystem: forwards successful discovery and attribute requests")
{
    DcgmSystem system = MakeSystem(mockDcgmOps.m_handleId);

    GIVEN("mocked supported and all-device lists")
    {
        mockDcgmOps.m_discoveryState.allSupportedDevices = { 0, 2 };
        mockDcgmOps.m_discoveryState.allDevices          = { 0, 1, 2 };
        std::vector<unsigned int> gpuIds { 99 };

        WHEN("supported devices are queried")
        {
            auto const ret = system.GetAllSupportedDevices(gpuIds);

            THEN("the caller-owned vector is replaced with the returned IDs")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(gpuIds == std::vector<unsigned int> { 0, 2 });
            }
        }

        WHEN("all devices are queried")
        {
            auto const ret = system.GetAllDevices(gpuIds);

            THEN("unsupported devices are also surfaced")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(gpuIds == std::vector<unsigned int> { 0, 1, 2 });
            }
        }
    }

    GIVEN("mocked device attributes and status")
    {
        auto attributesOverride                                = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        attributesOverride.identifiers.pciDeviceId             = 0x23350003;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = attributesOverride;
        mockDcgmOps.m_queryState.gpuStatus                     = DcgmEntityStatusDetached;

        dcgmDeviceAttributes_t attributes {};
        DcgmEntityStatus_t status = DcgmEntityStatusUnknown;

        WHEN("a valid GPU is queried")
        {
            auto const attrRet   = system.GetDeviceAttributes(3, attributes);
            auto const statusRet = system.GetGpuStatus(4, &status);

            THEN("the calls are forwarded and outputs are populated")
            {
                CHECK(attrRet == DCGM_ST_OK);
                CHECK(attributes.identifiers.pciDeviceId == 0x23350003);
                CHECK(mockDcgmOps.m_deviceAttributesState.lastGpuId == 3);
                CHECK(statusRet == DCGM_ST_OK);
                CHECK(status == DcgmEntityStatusDetached);
                CHECK(mockDcgmOps.m_queryState.lastGpuId == 4);
            }
        }
    }

    GIVEN("a mocked chip architecture query")
    {
        mockDcgmOps.m_discoveryState.chipArchitecture = DCGM_CHIP_ARCH_HOPPER;

        WHEN("the architecture is queried")
        {
            auto result = system.GetGpuChipArchitecture(5);

            THEN("the architecture value is returned through the expected wrapper")
            {
                REQUIRE(result.has_value());
                CHECK(result.value() == DCGM_CHIP_ARCH_HOPPER);
                CHECK(mockDcgmOps.m_discoveryState.lastGpuId == 5);
            }
        }
    }
}

SCENARIO_METHOD(DcgmSystemTestFixture, "DcgmSystem: preserves caller vectors on discovery errors")
{
    DcgmSystem system = MakeSystem(mockDcgmOps.m_handleId);

    GIVEN("discovery APIs return errors")
    {
        std::vector<unsigned int> gpuIds { 8, 9 };
        mockDcgmOps.m_discoveryState.allSupportedDevicesReturn = DCGM_ST_GENERIC_ERROR;
        mockDcgmOps.m_discoveryState.allDevicesReturn          = DCGM_ST_NOT_SUPPORTED;

        WHEN("supported devices fail")
        {
            auto const ret = system.GetAllSupportedDevices(gpuIds);

            THEN("the error is returned and existing contents remain available")
            {
                CHECK(ret == DCGM_ST_GENERIC_ERROR);
                CHECK(gpuIds == std::vector<unsigned int> { 8, 9 });
            }
        }

        WHEN("all devices fail")
        {
            auto const ret = system.GetAllDevices(gpuIds);

            THEN("the error is returned and existing contents remain available")
            {
                CHECK(ret == DCGM_ST_NOT_SUPPORTED);
                CHECK(gpuIds == std::vector<unsigned int> { 8, 9 });
            }
        }
    }

    GIVEN("attribute and architecture APIs return errors")
    {
        mockDcgmOps.m_deviceAttributesState.returnCode      = DCGM_ST_GPU_IS_LOST;
        mockDcgmOps.m_queryState.gpuStatusReturn            = DCGM_ST_NO_DATA;
        mockDcgmOps.m_discoveryState.chipArchitectureReturn = DCGM_ST_NOT_SUPPORTED;

        dcgmDeviceAttributes_t attributes {};
        DcgmEntityStatus_t status = DcgmEntityStatusUnknown;

        WHEN("queries are forwarded")
        {
            auto const attrRet   = system.GetDeviceAttributes(2, attributes);
            auto const statusRet = system.GetGpuStatus(2, &status);
            auto archResult      = system.GetGpuChipArchitecture(2);

            THEN("the collaborator errors are preserved")
            {
                CHECK(attrRet == DCGM_ST_GPU_IS_LOST);
                CHECK(statusRet == DCGM_ST_NO_DATA);
                REQUIRE_FALSE(archResult.has_value());
                CHECK(archResult.error() == DCGM_ST_NOT_SUPPORTED);
            }
        }
    }
}

SCENARIO_METHOD(DcgmSystemTestFixture, "DcgmSystem: forwards latest-value and CUDA-version requests")
{
    InstallLatestValuesHandler();

    GIVEN("a system initialized with a DCGM handle")
    {
        DcgmSystem system = MakeSystem(mockDcgmOps.m_handleId);

        WHEN("one latest GPU field is requested")
        {
            dcgmFieldValue_v2 value {};
            gpuMaxOpTempVal = 93;
            auto const ret  = system.GetGpuLatestValue(7, DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS, 0, value);

            THEN("the request is forwarded and the field value is copied back")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(value.entityGroupId == DCGM_FE_GPU);
                CHECK(value.entityId == 7);
                CHECK(value.fieldId == DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS);
                CHECK(value.fieldType == DCGM_FT_INT64);
                CHECK(value.value.i64 == 93);
            }
        }

        WHEN("multiple GPUs and fields are requested")
        {
            struct ObservedValue
            {
                dcgm_field_entity_group_t m_entityGroupId;
                dcgm_field_eid_t m_entityId;
                unsigned short m_fieldId;
                int64_t m_value;
            };
            std::vector<ObservedValue> observed;
            auto callback = [](dcgm_field_entity_group_t entityGroupId,
                               dcgm_field_eid_t entityId,
                               dcgmFieldValue_v1 *values,
                               int numValues,
                               void *userData) -> int {
                auto *out = static_cast<std::vector<ObservedValue> *>(userData);
                for (int i = 0; i < numValues; ++i)
                {
                    out->push_back({ entityGroupId, entityId, values[i].fieldId, values[i].value.i64 });
                }
                return 0;
            };
            std::vector<unsigned int> gpuIds { 2, 3 };
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS,
                                                   DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS };
            gpuMaxOpTempVal    = 88;
            memoryMaxOpTempVal = 101;

            auto const ret
                = system.GetLatestValuesForGpus(gpuIds, fieldIds, DCGM_FV_FLAG_LIVE_DATA, callback, &observed);

            THEN("each entity-field pair is delivered to the callback")
            {
                CHECK(ret == DCGM_ST_OK);
                REQUIRE(observed.size() == 4);
                CHECK(observed[0].m_entityGroupId == DCGM_FE_GPU);
                CHECK(observed[0].m_entityId == 2);
                CHECK(observed[0].m_fieldId == DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS);
                CHECK(observed[0].m_value == 88);
                CHECK(observed[1].m_entityGroupId == DCGM_FE_GPU);
                CHECK(observed[1].m_entityId == 2);
                CHECK(observed[1].m_fieldId == DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS);
                CHECK(observed[1].m_value == 101);
                CHECK(observed[2].m_entityGroupId == DCGM_FE_GPU);
                CHECK(observed[2].m_entityId == 3);
                CHECK(observed[2].m_fieldId == DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS);
                CHECK(observed[2].m_value == 88);
                CHECK(observed[3].m_entityGroupId == DCGM_FE_GPU);
                CHECK(observed[3].m_entityId == 3);
                CHECK(observed[3].m_fieldId == DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS);
                CHECK(observed[3].m_value == 101);
            }
        }

        WHEN("the callback asks iteration to stop early")
        {
            int callbackCount = 0;
            auto callback
                = [](dcgm_field_entity_group_t, dcgm_field_eid_t, dcgmFieldValue_v1 *, int, void *userData) -> int {
                ++(*static_cast<int *>(userData));
                return 1;
            };
            std::vector<unsigned int> gpuIds { 0, 1 };
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS,
                                                   DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS };

            auto const ret = system.GetLatestValuesForGpus(gpuIds, fieldIds, 0, callback, &callbackCount);

            THEN("the operation still reports success after honoring the stop request")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(callbackCount == 1);
            }
        }

        WHEN("GetLatestValuesForGpus is called with a null checker")
        {
            auto const callCountBefore = mockDcgmOps.m_entitiesGetLatestValuesState.callCount;
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS };

            auto const ret = system.GetLatestValuesForGpus({ 0 }, fieldIds, 0, nullptr, nullptr);

            THEN("validation fails before the DCGM API is called")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_entitiesGetLatestValuesState.callCount == callCountBefore);
            }
        }

        WHEN("a latest-value query fails in DCGM")
        {
            fieldRet[DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS] = DCGM_ST_GENERIC_ERROR;
            dcgmFieldValue_v2 value {};
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS };
            auto callback = [](dcgm_field_entity_group_t, dcgm_field_eid_t, dcgmFieldValue_v1 *, int, void *) -> int {
                return 0;
            };

            auto const latestValueRet  = system.GetGpuLatestValue(0, DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS, 0, value);
            auto const latestValuesRet = system.GetLatestValuesForGpus({ 0 }, fieldIds, 0, callback, nullptr);

            THEN("the collaborator error is returned")
            {
                CHECK(latestValueRet == DCGM_ST_GENERIC_ERROR);
                CHECK(latestValuesRet == DCGM_ST_GENERIC_ERROR);
            }
        }
    }

    GIVEN("a system with CUDA driver values supplied by MockDcgmOps")
    {
        WHEN("the driver version is available")
        {
            DcgmSystem system = MakeSystem(mockDcgmOps.m_handleId);
            cudaDriverVersion = 12080;

            auto const majorBefore = system.GetCudaMajorVersion();
            auto const minorBefore = system.GetCudaMinorVersion();
            cudaDriverVersion      = 0;
            auto const majorAfter  = system.GetCudaMajorVersion();
            auto const minorAfter  = system.GetCudaMinorVersion();

            THEN("the major and minor CUDA versions are decoded and cached")
            {
                CHECK(majorBefore == 12);
                CHECK(minorBefore == 8);
                CHECK(majorAfter == 12);
                CHECK(minorAfter == 8);
            }
        }

        WHEN("the driver version is blank")
        {
            DcgmSystem system = MakeSystem(mockDcgmOps.m_handleId);
            cudaDriverVersion = DCGM_INT64_BLANK;

            auto const major = system.GetCudaMajorVersion();
            auto const minor = system.GetCudaMinorVersion();

            THEN("the CUDA version is reported as unknown")
            {
                CHECK(major == 0);
                CHECK(minor == 0);
            }
        }

        WHEN("the DCGM query fails")
        {
            DcgmSystem system                     = MakeSystem(mockDcgmOps.m_handleId);
            fieldRet[DCGM_FI_CUDA_DRIVER_VERSION] = DCGM_ST_NO_DATA;

            auto const major = system.GetCudaMajorVersion();
            auto const minor = system.GetCudaMinorVersion();

            THEN("the CUDA version is reported as unknown")
            {
                CHECK(major == 0);
                CHECK(minor == 0);
            }
        }
    }
}
