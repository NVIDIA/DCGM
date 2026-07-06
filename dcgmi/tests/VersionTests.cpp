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

#include <catch2/catch_all.hpp>

#include "TestHelpers.hpp"

#include <Version.h>
#include <dcgm_agent.h>

#include <cstring>
#include <string>

namespace
{
struct VersionApiState
{
    dcgmReturn_t versionReturn = DCGM_ST_OK;
    std::string rawBuildInfo   = "version:9.9.9;arch:test-arch;buildtype:Debug;buildid:test-build;builddate:2026-05-06;"
                                 "commit:test-commit;branch:test-branch;buildplatform:test-platform;crc:test-crc";
    dcgmHandle_t lastHandle {};
    unsigned int lastVersion = 0;
    int callCount            = 0;
};

VersionApiState g_versionApi;

void ResetVersionApi()
{
    g_versionApi = {};
}

class TestVersionInfo : public VersionInfo
{
public:
    using VersionInfo::VersionInfo;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};
} //namespace

extern "C" dcgmReturn_t dcgmHostengineVersionInfo(dcgmHandle_t handle, dcgmVersionInfo_t *versionInfo)
{
    g_versionApi.callCount++;
    g_versionApi.lastHandle  = handle;
    g_versionApi.lastVersion = versionInfo->version;

    if (g_versionApi.versionReturn == DCGM_ST_OK)
    {
        std::strncpy(versionInfo->rawBuildInfoString,
                     g_versionApi.rawBuildInfo.c_str(),
                     sizeof(versionInfo->rawBuildInfoString) - 1);
        versionInfo->rawBuildInfoString[sizeof(versionInfo->rawBuildInfoString) - 1] = '\0';
    }

    return g_versionApi.versionReturn;
}

TEST_CASE("VersionInfo")
{
    ResetVersionApi();
    auto handle = static_cast<dcgmHandle_t>(0x71);

    GIVEN("hostengine version information is available")
    {
        TestVersionInfo versionInfo("localhost");

        WHEN("the command runs with a connected handle")
        {
            CoutCapture capture;
            dcgmReturn_t result = versionInfo.RunWithHandle(handle);

            THEN("local and hostengine build information are displayed")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(g_versionApi.callCount == 1);
                CHECK(g_versionApi.lastHandle == handle);
                CHECK(g_versionApi.lastVersion == dcgmVersionInfo_version);
                CHECK(capture.str().find("Local build info:") != std::string::npos);
                CHECK(capture.str().find("Hostengine build info:") != std::string::npos);
                CHECK(capture.str().find("9.9.9") != std::string::npos);
            }
        }
    }

    GIVEN("hostengine version information fails")
    {
        g_versionApi.versionReturn = DCGM_ST_GENERIC_ERROR;
        TestVersionInfo versionInfo("localhost");

        WHEN("the command runs")
        {
            CoutCapture capture;
            dcgmReturn_t result = versionInfo.RunWithHandle(handle);

            THEN("the command returns a connection error after printing local build info")
            {
                CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(g_versionApi.callCount == 1);
                CHECK(capture.str().find("Local build info:") != std::string::npos);
                CHECK(capture.str().find("Hostengine build info:") == std::string::npos);
            }
        }
    }
}
