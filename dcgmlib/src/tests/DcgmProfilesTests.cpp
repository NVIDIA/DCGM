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
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <DcgmProfiles.h>

#include <array>
#include <span>

namespace
{
class FakeWorkloadPowerProfileReader : public DcgmWorkloadPowerProfileReader
{
public:
    nvmlReturn_t profilesInfoReturn    = NVML_SUCCESS;
    nvmlReturn_t currentProfilesReturn = NVML_SUCCESS;

    nvmlWorkloadPowerProfileProfilesInfo_t profilesInfo {};
    nvmlWorkloadPowerProfileCurrentProfiles_t currentProfiles {};

    unsigned int profilesInfoCallCount    = 0;
    unsigned int currentProfilesCallCount = 0;
    unsigned int observedProfilesInfoVersion {};
    unsigned int observedCurrentProfilesVersion {};
    SafeNvmlHandle observedDevice {};

    nvmlReturn_t GetWorkloadPowerProfileProfilesInfo(SafeNvmlHandle device,
                                                     nvmlWorkloadPowerProfileProfilesInfo_t *output) noexcept override
    {
        profilesInfoCallCount++;
        observedDevice = device;
        if (output != nullptr)
        {
            observedProfilesInfoVersion = output->version;
            *output                     = profilesInfo;
        }
        return profilesInfoReturn;
    }

    nvmlReturn_t GetWorkloadPowerProfileCurrentProfiles(
        SafeNvmlHandle device,
        nvmlWorkloadPowerProfileCurrentProfiles_t *output) noexcept override
    {
        currentProfilesCallCount++;
        observedDevice = device;
        if (output != nullptr)
        {
            observedCurrentProfilesVersion = output->version;
            *output                        = currentProfiles;
        }
        return currentProfilesReturn;
    }
};

} // namespace

SCENARIO("DcgmProfiles: workload power profile info validates inputs")
{
    auto const [caseName, missingGpuInfo, missingOutput] = GENERATE(table<char const *, bool, bool>({
        { "missing GPU info", true, false },
        { "missing output", false, true },
    }));
    CAPTURE(caseName);

    FakeWorkloadPowerProfileReader reader;
    dcgm_power_profile_helper_t gpuInfo {};
    dcgmWorkloadPowerProfileProfilesInfo_v1 output {};

    WHEN("profile info is queried")
    {
        auto const result = DcgmGetWorkloadPowerProfilesInfo(
            reader, missingGpuInfo ? nullptr : &gpuInfo, missingOutput ? nullptr : &output);

        THEN("the query fails before calling the reader")
        {
            CHECK(result == DCGM_ST_BADPARAM);
            CHECK(reader.profilesInfoCallCount == 0);
        }
    }
}

SCENARIO("DcgmProfiles: workload power profile info maps reader outcomes")
{
    FakeWorkloadPowerProfileReader reader;
    dcgm_power_profile_helper_t gpuInfo { .gpuId = 7, .nvmlDevice = SafeNvmlHandle { nullptr, 3, 11 } };
    dcgmWorkloadPowerProfileProfilesInfo_v1 output {};

    GIVEN("the reader reports an NVML error")
    {
        auto const [caseName, nvmlReturn, expectedReturn] = GENERATE(table<char const *, nvmlReturn_t, dcgmReturn_t>({
            { "unsupported", NVML_ERROR_NOT_SUPPORTED, DCGM_ST_NOT_SUPPORTED },
            { "unknown", NVML_ERROR_UNKNOWN, DCGM_ST_NVML_ERROR },
        }));
        CAPTURE(caseName);
        reader.profilesInfoReturn = nvmlReturn;

        WHEN("profile info is queried")
        {
            auto const result = DcgmGetWorkloadPowerProfilesInfo(reader, &gpuInfo, &output);

            THEN("the DCGM result reflects the NVML failure")
            {
                CHECK(result == expectedReturn);
                CHECK(reader.profilesInfoCallCount == 1);
                CHECK(reader.observedProfilesInfoVersion == nvmlWorkloadPowerProfileProfilesInfo_v1);
                CHECK(reader.observedDevice.nvmlIndex == gpuInfo.nvmlDevice.nvmlIndex);
                CHECK(reader.observedDevice.generation == gpuInfo.nvmlDevice.generation);
            }
        }
    }

    GIVEN("the reader returns no supported profiles")
    {
        WHEN("profile info is queried")
        {
            auto const result = DcgmGetWorkloadPowerProfilesInfo(reader, &gpuInfo, &output);

            THEN("the query succeeds with an empty DCGM profile list")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(output.profileCount == 0);
                CHECK(reader.profilesInfoCallCount == 1);
            }
        }
    }

    GIVEN("the reader returns sparse supported profile entries")
    {
        NVML_255_MASK_BIT_SET(2, reader.profilesInfo.perfProfilesMask);
        NVML_255_MASK_BIT_SET(65, reader.profilesInfo.perfProfilesMask);
        reader.profilesInfo.perfProfile[2].profileId                = NVML_POWER_PROFILE_COMPUTE;
        reader.profilesInfo.perfProfile[2].priority                 = 12;
        reader.profilesInfo.perfProfile[65].profileId               = NVML_POWER_PROFILE_HPC;
        reader.profilesInfo.perfProfile[65].priority                = 4;
        reader.profilesInfo.perfProfile[2].conflictingMask.mask[0]  = 0x00000005;
        reader.profilesInfo.perfProfile[65].conflictingMask.mask[2] = 0x00000020;

        WHEN("profile info is queried")
        {
            auto const result = DcgmGetWorkloadPowerProfilesInfo(reader, &gpuInfo, &output);

            THEN("only set mask entries are copied in index order")
            {
                CHECK(result == DCGM_ST_OK);
                REQUIRE(output.profileCount == 2);
                CHECK(output.workloadPowerProfile[0].profileId == DCGM_POWER_PROFILE_COMPUTE);
                CHECK(output.workloadPowerProfile[0].priority == 12);
                CHECK(output.workloadPowerProfile[0].conflictingMask[0] == 0x00000005);
                CHECK(output.workloadPowerProfile[1].profileId == DCGM_POWER_PROFILE_HPC);
                CHECK(output.workloadPowerProfile[1].priority == 4);
                CHECK(output.workloadPowerProfile[1].conflictingMask[2] == 0x00000020);
            }
        }
    }
}

SCENARIO("DcgmProfiles: workload power profile status validates inputs")
{
    auto const [caseName, missingGpuInfo, missingOutput] = GENERATE(table<char const *, bool, bool>({
        { "missing GPU info", true, false },
        { "missing output", false, true },
    }));
    CAPTURE(caseName);

    FakeWorkloadPowerProfileReader reader;
    dcgm_power_profile_helper_t gpuInfo {};
    dcgmDeviceWorkloadPowerProfilesStatus_v1 output {};

    WHEN("profile status is queried")
    {
        auto const result = DcgmGetWorkloadPowerProfilesStatus(
            reader, missingGpuInfo ? nullptr : &gpuInfo, missingOutput ? nullptr : &output);

        THEN("the query fails before calling the reader")
        {
            CHECK(result == DCGM_ST_BADPARAM);
            CHECK(reader.currentProfilesCallCount == 0);
        }
    }
}

SCENARIO("DcgmProfiles: workload power profile status maps reader outcomes")
{
    FakeWorkloadPowerProfileReader reader;
    dcgm_power_profile_helper_t gpuInfo { .gpuId = 9, .nvmlDevice = SafeNvmlHandle { nullptr, 5, 13 } };
    dcgmDeviceWorkloadPowerProfilesStatus_v1 output {};

    GIVEN("the reader reports an NVML error")
    {
        auto const [caseName, nvmlReturn, expectedReturn] = GENERATE(table<char const *, nvmlReturn_t, dcgmReturn_t>({
            { "unsupported", NVML_ERROR_NOT_SUPPORTED, DCGM_ST_NOT_SUPPORTED },
            { "unknown", NVML_ERROR_UNKNOWN, DCGM_ST_NVML_ERROR },
        }));
        CAPTURE(caseName);
        reader.currentProfilesReturn = nvmlReturn;

        WHEN("profile status is queried")
        {
            auto const result = DcgmGetWorkloadPowerProfilesStatus(reader, &gpuInfo, &output);

            THEN("the DCGM result reflects the NVML failure")
            {
                CHECK(result == expectedReturn);
                CHECK(reader.currentProfilesCallCount == 1);
                CHECK(reader.observedCurrentProfilesVersion == nvmlWorkloadPowerProfileCurrentProfiles_v1);
                CHECK(reader.observedDevice.nvmlIndex == gpuInfo.nvmlDevice.nvmlIndex);
                CHECK(reader.observedDevice.generation == gpuInfo.nvmlDevice.generation);
            }
        }
    }

    GIVEN("the reader returns current profile masks")
    {
        reader.currentProfiles.perfProfilesMask.mask[0]      = 0x00000003;
        reader.currentProfiles.perfProfilesMask.mask[7]      = 0x40000000;
        reader.currentProfiles.requestedProfilesMask.mask[1] = 0x00000008;
        reader.currentProfiles.enforcedProfilesMask.mask[2]  = 0x00000010;

        WHEN("profile status is queried")
        {
            auto const result = DcgmGetWorkloadPowerProfilesStatus(reader, &gpuInfo, &output);

            THEN("the DCGM status receives each NVML mask")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK_THAT(std::span { output.profileMask },
                           Catch::Matchers::RangeEquals(
                               std::array<unsigned int, 8> { 0x00000003, 0, 0, 0, 0, 0, 0, 0x40000000 }));
                CHECK_THAT(
                    std::span { output.requestedProfileMask },
                    Catch::Matchers::RangeEquals(std::array<unsigned int, 8> { 0, 0x00000008, 0, 0, 0, 0, 0, 0 }));
                CHECK_THAT(
                    std::span { output.enforcedProfileMask },
                    Catch::Matchers::RangeEquals(std::array<unsigned int, 8> { 0, 0, 0x00000010, 0, 0, 0, 0, 0 }));
                CHECK(reader.currentProfilesCallCount == 1);
            }
        }
    }
}

SCENARIO("DcgmProfiles: workload power profile NvmlTaskRunner overload validates inputs")
{
    NvmlTaskRunner nvmlDriver;
    dcgm_power_profile_helper_t gpuInfo {};
    dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo {};

    GIVEN("invalid arguments for the production reader overload")
    {
        WHEN("profile info is queried with a missing GPU info pointer")
        {
            auto const result = DcgmGetWorkloadPowerProfilesInfo(nvmlDriver, nullptr, &profilesInfo);

            THEN("the query fails before using NVML")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }

        WHEN("profile status is queried with a missing output pointer")
        {
            auto const result = DcgmGetWorkloadPowerProfilesStatus(nvmlDriver, &gpuInfo, nullptr);

            THEN("the query fails before using NVML")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }
}

SCENARIO("DcgmProfiles: workload power profile NvmlTaskRunner overload maps stale handles")
{
    NvmlTaskRunner nvmlDriver;
    REQUIRE(nvmlDriver.Start() == 0);
    dcgm_power_profile_helper_t gpuInfo { .gpuId = 4, .nvmlDevice = SafeNvmlHandle { nullptr, 1, 0 } };

    GIVEN("a safe handle from an old NVML generation")
    {
        WHEN("profile info is queried through the production overload")
        {
            dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo {};
            auto const result = DcgmGetWorkloadPowerProfilesInfo(nvmlDriver, &gpuInfo, &profilesInfo);

            THEN("the helper reports the runner rejection as an NVML error")
            {
                CHECK(result == DCGM_ST_NVML_ERROR);
            }
        }

        WHEN("profile status is queried through the production overload")
        {
            dcgmDeviceWorkloadPowerProfilesStatus_v1 profilesStatus {};
            auto const result = DcgmGetWorkloadPowerProfilesStatus(nvmlDriver, &gpuInfo, &profilesStatus);

            THEN("the helper reports the runner rejection as an NVML error")
            {
                CHECK(result == DCGM_ST_NVML_ERROR);
            }
        }
    }
}
