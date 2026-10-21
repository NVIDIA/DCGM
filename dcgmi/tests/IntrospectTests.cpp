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

#include <Introspect.h>
#include <MockDcgmOps.h>

#include <catch2/catch_all.hpp>

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <memory>

namespace
{
struct IntrospectTestFixture
{
    IntrospectTestFixture()
        : dcgmOps(&mockDcgmOps)
    {
        dcgmIntrospectMemory_t memory {};
        memory.version                       = dcgmIntrospectMemory_version1;
        memory.bytesUsed                     = 1536;
        mockDcgmOps.m_introspectState.memory = memory;

        dcgmIntrospectCpuUtil_t cpu {};
        cpu.version                       = dcgmIntrospectCpuUtil_version1;
        cpu.total                         = 0.125;
        mockDcgmOps.m_introspectState.cpu = cpu;
    }

    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps;
};
} // namespace

TEST_CASE_METHOD(IntrospectTestFixture, "Introspect::DisplayStats renders hostengine stats")
{
    Introspect introspect(dcgmOps);
    CoutCapture capture;

    GIVEN("hostengine introspection data is available")
    {
        WHEN("hostengine stats are requested")
        {
            CHECK(introspect.DisplayStats(0x1234, true) == DCGM_ST_OK);

            THEN("memory and CPU data are fetched and displayed")
            {
                auto output = capture.str();
                CHECK(output.find("Hostengine Process") != std::string::npos);
                CHECK(output.find("Memory") != std::string::npos);
                CHECK(output.find("CPU Utilization") != std::string::npos);
                CHECK(output.find("1.5 KB") != std::string::npos);
                CHECK(mockDcgmOps.m_introspectState.memoryCallCount == 1);
                CHECK(mockDcgmOps.m_introspectState.cpuCallCount == 1);
                CHECK(mockDcgmOps.m_introspectState.lastHandle == 0x1234);
                CHECK(mockDcgmOps.m_introspectState.lastMemoryVersion == dcgmIntrospectMemory_version1);
                CHECK(mockDcgmOps.m_introspectState.lastCpuVersion == dcgmIntrospectCpuUtil_version1);
                CHECK(mockDcgmOps.m_introspectState.lastMemoryWait == 1);
                CHECK(mockDcgmOps.m_introspectState.lastCpuWait == 1);
            }
        }
    }
}

TEST_CASE_METHOD(IntrospectTestFixture, "Introspect::DisplayStats handles summary and error branches")
{
    Introspect introspect(dcgmOps);

    SECTION("summary without hostengine target only checks memory availability")
    {
        CoutCapture capture;

        CHECK(introspect.DisplayStats(0x2222, false) == DCGM_ST_OK);
        auto output = capture.str();
        CHECK(output.find("Introspection Information") != std::string::npos);
        CHECK(output.find("Hostengine Process") == std::string::npos);

        CHECK(mockDcgmOps.m_introspectState.memoryCallCount == 1);
        CHECK(mockDcgmOps.m_introspectState.cpuCallCount == 0);
    }

    SECTION("failed memory and CPU calls render Error values")
    {
        mockDcgmOps.m_introspectState.memoryReturn = DCGM_ST_NOT_CONFIGURED;
        mockDcgmOps.m_introspectState.cpuReturn    = DCGM_ST_NO_DATA;
        CoutCapture capture;

        CHECK(introspect.DisplayStats(0x3333, true) == DCGM_ST_OK);
        auto output = capture.str();
        CHECK(output.find("Error") != std::string::npos);

        CHECK(mockDcgmOps.m_introspectState.memoryCallCount == 1);
        CHECK(mockDcgmOps.m_introspectState.cpuCallCount == 1);
    }
}
