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

#include <DcgmModuleNvSwitch.h>
#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>

namespace DcgmNs
{

class TestDcgmModuleNvSwitch : public DcgmModuleNvSwitch
{
public:
    explicit TestDcgmModuleNvSwitch(dcgmCoreCallbacks_t &dcc)
        : DcgmModuleNvSwitch(dcc)
    {}

    // Make private methods public for testing
    using DcgmModuleNvSwitch::m_nvswitchMgr;
    using DcgmModuleNvSwitch::StopAndWait;

    using DcgmModuleNvSwitch::m_lastLinkStatusUpdateUsec;
};

} // namespace DcgmNs

using namespace DcgmNs;

TEST_CASE("RunOnce with active watches to test watch interval is honored.")
{
    // Common setup
    auto ret = DcgmFieldsInit();
    REQUIRE(ret == DCGM_ST_OK);

    dcgmCoreCallbacks_t dcc
        = { .version  = dcgmCoreCallbacks_version,
            .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
            .poster     = nullptr,
            .loggerfunc = [](void const *) { /* do nothing */ } };
    TestDcgmModuleNvSwitch nvSwitchModule(dcc);

    // Stop the module thread which calls RunOnce function and can modify the m_lastLinkStatusUpdateUsec.
    if (nvSwitchModule.StopAndWait(3000) == 1)
    {
        SKIP("Failed to stop the module thread, skipping test.");
    }
    // Reset the m_lastLinkStatusUpdateUsec to 0 since thread may have modified it before it is stopped.
    nvSwitchModule.m_lastLinkStatusUpdateUsec = 0;

    // Create a fake switch to watch
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1];
    ret = nvSwitchModule.m_nvswitchMgr.CreateFakeSwitches(fakeCount, fakeSwitchIds);
    REQUIRE(ret == DCGM_ST_OK);

    // Set up watch parameters on NvSwitch fields
    unsigned short fieldIds[2] = { DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT, DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT };
    DcgmWatcher watcher;
    std::chrono::milliseconds nextUpdateMs;
    std::chrono::milliseconds constexpr LINK_STATUS_RESCAN_INTERVAL_MS = std::chrono::seconds(30);
    std::chrono::milliseconds constexpr MIN_UPDATE_INTERVAL_MS         = std::chrono::milliseconds(1);
    std::chrono::microseconds constexpr shortInterval                  = std::chrono::seconds(20);

    auto setupWatch = [&](std::chrono::microseconds watchIntervalUsec) {
        return nvSwitchModule.m_nvswitchMgr.WatchField(DCGM_FE_SWITCH,
                                                       fakeSwitchIds[0],
                                                       2,
                                                       fieldIds,
                                                       watchIntervalUsec.count(),
                                                       watcher.watcherType,
                                                       watcher.connectionId,
                                                       true);
    };

    SECTION("No active watches")
    {
        // Don't set up any watches
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce()); // RunOnce returns time in milliseconds.
        // The return value should default to link status rescan interval (30 seconds)
        CHECK(nextUpdateMs == LINK_STATUS_RESCAN_INTERVAL_MS);
    }

    SECTION("Watch interval (20s) is less than link status rescan interval (30s)")
    {
        // Watch the fields with watch interval 20s.
        REQUIRE(setupWatch(shortInterval) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce()); // RunOnce returns time in milliseconds.
        // The returned value should be less than or equal to our watch interval.
        CHECK(std::chrono::duration_cast<std::chrono::microseconds>(nextUpdateMs) <= shortInterval);
    }

    SECTION("Watch interval (40s) is greater than link status rescan interval (30s)")
    {
        // Watch the fields with watch interval 40s.
        REQUIRE(setupWatch(shortInterval * 2) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce()); // RunOnce returns time in milliseconds.
        // The returned value should be equal to the link status rescan interval (30 seconds)
        CHECK(nextUpdateMs == LINK_STATUS_RESCAN_INTERVAL_MS);
    }

    SECTION("Multiple watches with different intervals")
    {
        // Set up first watch with 20s interval
        REQUIRE(setupWatch(shortInterval) == DCGM_ST_OK);
        // Set up second watch with 40s interval
        REQUIRE(setupWatch(shortInterval * 2) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce());
        // The returned value should be less than or equal to the minimum of the two intervals
        CHECK(std::chrono::duration_cast<std::chrono::microseconds>(nextUpdateMs) <= shortInterval);
    }

    SECTION("Minimum update interval is honored")
    {
        // Set up a watch with 1ms interval
        REQUIRE(setupWatch(MIN_UPDATE_INTERVAL_MS) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce());
        // The returned value should be equal to the minimum update interval (1ms)
        CHECK(nextUpdateMs == MIN_UPDATE_INTERVAL_MS);
    }

    SECTION("Minimum update interval is honored when watch interval is below minimum")
    {
        // Set up a watch with 0.5ms interval
        REQUIRE(setupWatch(std::chrono::milliseconds(MIN_UPDATE_INTERVAL_MS / 2)) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce());
        // The returned value should be equal to the minimum update interval (1ms)
        CHECK(nextUpdateMs == MIN_UPDATE_INTERVAL_MS);
    }
}
