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
#include <DcgmVersion.hpp>

#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string_view>

namespace
{
std::size_t RequireBoundedCStringLength(char const *str, std::size_t maxSize)
{
    void const *const terminator = std::memchr(str, '\0', maxSize);
    REQUIRE(terminator != nullptr);
    return static_cast<char const *>(terminator) - str;
}
} // namespace

SCENARIO("GetVersionInfo validates its output argument")
{
    WHEN("the output argument is null")
    {
        dcgmReturn_t const result = GetVersionInfo(nullptr);

        THEN("the call fails with the documented bad-parameter status")
        {
            REQUIRE(result == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("a version info structure with an unsupported version")
    {
        dcgmVersionInfo_t versionInfo {};
        versionInfo.version = dcgmVersionInfo_version - 1;
        SafeCopyTo(versionInfo.rawBuildInfoString, "preserved-on-error");

        WHEN("version information is requested")
        {
            dcgmReturn_t const result = GetVersionInfo(&versionInfo);

            THEN("the call reports a version mismatch and leaves the payload unchanged")
            {
                REQUIRE(result == DCGM_ST_VER_MISMATCH);
                CHECK(versionInfo.version == dcgmVersionInfo_version - 1);
                std::size_t const preservedLen = RequireBoundedCStringLength(versionInfo.rawBuildInfoString,
                                                                             sizeof(versionInfo.rawBuildInfoString));
                CHECK(std::string_view(versionInfo.rawBuildInfoString, preservedLen) == "preserved-on-error");
            }
        }
    }
}

SCENARIO("GetVersionInfo returns the build information for a supported structure")
{
    GIVEN("a version info structure with the current version")
    {
        dcgmVersionInfo_t versionInfo {};
        versionInfo.version = dcgmVersionInfo_version;
        SafeCopyTo(versionInfo.rawBuildInfoString, "will-be-replaced");

        WHEN("version information is requested")
        {
            dcgmReturn_t const result = GetVersionInfo(&versionInfo);

            THEN("the call succeeds and copies the raw build information")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(versionInfo.version == dcgmVersionInfo_version);
                std::size_t const buildInfoLen = RequireBoundedCStringLength(versionInfo.rawBuildInfoString,
                                                                             sizeof(versionInfo.rawBuildInfoString));
                CHECK(std::string_view(versionInfo.rawBuildInfoString, buildInfoLen) == c_dcgmBuildInfo);
            }
        }
    }
}
