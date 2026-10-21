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

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <MockDcgmOps.h>
#include <Version.h>
#include <dcgm_agent.h>

#include <catch2/catch_all.hpp>

#include <memory>
#include <string>

namespace
{
std::string const &RawBuildInfo()
{
    static std::string const rawBuildInfo
        = "version:9.9.9;arch:test-arch;buildtype:Debug;buildid:test-build;builddate:2026-05-06;"
          "commit:test-commit;branch:test-branch;buildplatform:test-platform;crc:test-crc";
    return rawBuildInfo;
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

    dcgmReturn_t RunConnectionFailure(dcgmReturn_t connectionStatus)
    {
        return DoExecuteConnectionFailure(connectionStatus);
    }
};

struct VersionTestFixture
{
    VersionTestFixture()
        : dcgmOps(&mockDcgmOps)
    {
        mockDcgmOps.m_hostEngineState.rawBuildInfo = RawBuildInfo();
    }

    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps;
};
} //namespace

TEST_CASE_METHOD(VersionTestFixture, "VersionInfo")
{
    auto handle = static_cast<dcgmHandle_t>(0x71);

    GIVEN("hostengine version information is available")
    {
        TestVersionInfo versionInfo("localhost", dcgmOps);

        WHEN("the command runs with a connected handle")
        {
            CoutCapture capture;
            dcgmReturn_t result = versionInfo.RunWithHandle(handle);

            THEN("local and hostengine build information are displayed")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_hostEngineState.versionCallCount == 1);
                CHECK(mockDcgmOps.m_hostEngineState.lastVersionHandle == handle);
                CHECK(mockDcgmOps.m_hostEngineState.lastVersionVersion == dcgmVersionInfo_version);
                CHECK(capture.str().find("Local build info:") != std::string::npos);
                CHECK(capture.str().find("Hostengine build info:") != std::string::npos);
                CHECK(capture.str().find("9.9.9") != std::string::npos);
            }
        }
    }

    GIVEN("hostengine version information fails")
    {
        mockDcgmOps.m_hostEngineState.versionReturn = DCGM_ST_GENERIC_ERROR;
        TestVersionInfo versionInfo("localhost", dcgmOps);

        WHEN("the command runs")
        {
            CoutCapture capture;
            dcgmReturn_t result = versionInfo.RunWithHandle(handle);

            THEN("the command returns a connection error after printing local build info")
            {
                CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(mockDcgmOps.m_hostEngineState.versionCallCount == 1);
                CHECK(capture.str().find("Local build info:") != std::string::npos);
                CHECK(capture.str().find("Hostengine build info:") == std::string::npos);
            }
        }
    }

    GIVEN("hostengine is unreachable")
    {
        constexpr const char *c_hostName = "test-host";
        TestVersionInfo versionInfo(c_hostName, dcgmOps);

        WHEN("the connection failure handler runs")
        {
            CoutCapture capture;
            dcgmReturn_t result = versionInfo.RunConnectionFailure(DCGM_ST_CONNECTION_NOT_VALID);

            THEN("local build info is displayed with a note that hostengine build info is unavailable")
            {
                CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK_THAT(capture.str(), Catch::Matchers::ContainsSubstring("Local build info:"));
                CHECK_THAT(capture.str(), Catch::Matchers::ContainsSubstring("Hostengine build info not available"));
                CHECK_THAT(capture.str(), Catch::Matchers::ContainsSubstring(c_hostName));
            }
        }
    }
}
