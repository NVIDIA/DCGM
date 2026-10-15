/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmPolicyNotificationRouter.h"

#include <DcgmStringHelpers.h>
#include <dcgm_core_communication.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

class TestNotificationRouter
{
public:
    dcgmCoreCallbacks_t callbacks;

    /// Conditions seen by postfunc, in delivery order. Tests can assert against this.
    std::vector<dcgmPolicyCondition_t> deliveredConditions;

    TestNotificationRouter()
        : callbacks(createCallbacks())
    {}

private:
    static dcgmReturn_t recordingPostCallback(dcgm_module_command_header_t *req, void *poster)
    {
        if (poster != nullptr && req != nullptr)
        {
            auto *raw    = reinterpret_cast<dcgmCoreSendRawMessage_t *>(req);
            auto *notify = static_cast<dcgm_msg_policy_notify_t *>(raw->request.msgData);
            if (notify != nullptr)
            {
                static_cast<TestNotificationRouter *>(poster)->deliveredConditions.push_back(
                    notify->response.condition);
            }
        }
        return DCGM_ST_OK;
    }

    dcgmCoreCallbacks_t createCallbacks()
    {
        dcgmCoreCallbacks_t cb = {};
        cb.version             = dcgmCoreCallbacks_version;
        cb.postfunc            = recordingPostCallback;
        cb.poster              = this;
        cb.loggerfunc          = nullptr;
        return cb;
    }

public:
    dcgm_policy_violation_t CreateViolation(unsigned short fieldId   = DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                            int gpuId                = 0,
                                            double currentValue      = 85.0,
                                            double threshold         = 80.0,
                                            unsigned int channelMask = DCGM_POLICY_CHANNEL_CONSOLE)
    {
        dcgm_policy_violation_t violation = {};
        violation.policyId                = 0xDEADBEEFCAFEBABEull;
        SafeCopyTo(violation.policyName, "Test Policy");
        violation.fieldId = fieldId;
        SafeCopyTo(violation.fieldName, "GPU Temperature");
        violation.entity         = { .entityGroupId = DCGM_FE_GPU, .entityId = static_cast<dcgm_field_eid_t>(gpuId) };
        violation.timestamp      = timelib_usecSince1970();
        violation.currentValue   = currentValue;
        violation.thresholdValue = threshold;
        violation.policyOperator = DCGM_POLICY_OP_GT;
        violation.channelMask    = channelMask;
        return violation;
    }
};

TEST_CASE("DcgmPolicyNotificationRouter")
{
    SECTION("CONSOLE channel - basic routing")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto violation
            = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CONSOLE);

        REQUIRE(router.RouteViolation(violation) == DCGM_ST_OK);

        auto violations = router.GetRecentViolations(std::chrono::steady_clock::time_point::min());
        REQUIRE(violations.size() == 1);
        REQUIRE(violations[0].fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        REQUIRE(violations[0].currentValue == 85.0);
    }

    SECTION("CONSOLE channel - circular buffer")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto violation
            = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CONSOLE);

        for (int i = 0; i < 1100; i++)
        {
            router.RouteViolation(violation);
        }

        REQUIRE(router.GetRecentViolations(std::chrono::steady_clock::time_point::min()).size() == 1000);
    }

    SECTION("CONSOLE channel - duration filtering")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        timelib64_t const now = timelib_usecSince1970();

        auto violation
            = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CONSOLE);

        // old: 10 seconds ago
        violation.timestamp = now - 10000000;
        router.RouteViolation(violation);
        // recent: 2 seconds ago
        violation.timestamp = now - 2000000;
        router.RouteViolation(violation);
        // very recent: 0.5 seconds ago
        violation.timestamp = now - 500000;
        router.RouteViolation(violation);

        // 5-second lookback: should return the 2 recent violations
        auto violations = router.GetRecentViolations(std::chrono::steady_clock::now() - std::chrono::seconds(5));
        REQUIRE(violations.size() == 2);

        // 1-second lookback: should return only the most recent
        violations = router.GetRecentViolations(std::chrono::steady_clock::now() - std::chrono::seconds(1));
        REQUIRE(violations.size() == 1);
        REQUIRE(violations[0].timestamp == now - 500000);

        // min() time_point: returns all
        REQUIRE(router.GetRecentViolations(std::chrono::steady_clock::time_point::min()).size() == 3);
    }

    SECTION("CONSOLE channel - maxCount limiting")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto violation
            = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CONSOLE);

        // Add 5 violations with different timestamps
        for (int i = 1; i <= 5; i++)
        {
            violation.timestamp    = i * 1000;
            violation.currentValue = 80.0 + i;
            router.RouteViolation(violation);
        }

        // Get last 2 violations
        auto violations = router.GetRecentViolations(static_cast<size_t>(2));
        REQUIRE(violations.size() == 2);
        REQUIRE(violations[0].timestamp == 4000); // 2nd most recent
        REQUIRE(violations[1].timestamp == 5000); // Most recent

        // Get last 10 (more than available) - should return all 5
        violations = router.GetRecentViolations(static_cast<size_t>(10));
        REQUIRE(violations.size() == 5);

        // Get 0 - should return empty
        violations = router.GetRecentViolations(static_cast<size_t>(0));
        REQUIRE(violations.empty());
    }

    SECTION("FILE channel - JSON logging")
    {
        TestNotificationRouter fixture;

        std::string testLogPath = "/tmp/dcgm-policy-test-" + std::to_string(timelib_usecSince1970()) + ".log";

        {
            DcgmPolicyNotificationRouter router(fixture.callbacks, std::make_unique<std::ofstream>(testLogPath));

            auto violation
                = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_FILE);

            REQUIRE(router.RouteViolation(violation) == DCGM_ST_OK);
        }

        std::ifstream logFile(testLogPath);
        REQUIRE(logFile.is_open());

        std::string content((std::istreambuf_iterator<char>(logFile)), std::istreambuf_iterator<char>());
        logFile.close();

        REQUIRE(!content.empty());
        REQUIRE(content.find("\"fieldId\":") != std::string::npos);
        REQUIRE(content.find("\"entityGroupId\":1") != std::string::npos);
        REQUIRE(content.find("\"entityId\":0") != std::string::npos);
        REQUIRE(content.find("\"currentValue\":85") != std::string::npos);

        std::remove(testLogPath.c_str());
    }

    SECTION("CALLBACK channel - registration")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        // RegisterCallback returns an RAII guard; has_value() indicates success
        auto guard1 = router.RegisterCallback(1, 100);
        REQUIRE(guard1.has_value());
        auto guard2 = router.RegisterCallback(1, 101);
        REQUIRE(guard2.has_value());
        auto guard3 = router.RegisterCallback(2, 200);
        REQUIRE(guard3.has_value());

        // Deregistration: RAII guard going out of scope marks the callback as pending removal
        {
            auto guard4 = router.RegisterCallback(3, 300);
            REQUIRE(guard4.has_value());
        } // guard4 destructs here — callback for (3, 300) is pending deregistration

        auto violation
            = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CALLBACK);
        REQUIRE(router.RouteViolation(violation) == DCGM_ST_OK);
    }

    SECTION("CALLBACK channel - basic routing")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto guard1 = router.RegisterCallback(1, 100);
        REQUIRE(guard1.has_value());
        auto guard2 = router.RegisterCallback(2, 200);
        REQUIRE(guard2.has_value());

        auto violation
            = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CALLBACK);
        REQUIRE(router.RouteViolation(violation) == DCGM_ST_OK);
    }

    SECTION("CALLBACK channel - rate limiting suppresses repeats of the same fieldId within the window")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto guard = router.RegisterCallback(1, 100);
        REQUIRE(guard.has_value());

        auto violation1
            = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CALLBACK);
        violation1.timestamp = 1000000;

        auto violation2      = violation1;
        violation2.timestamp = 1000000 + 4000000; // 4 seconds later (within 5s window — should be suppressed)

        auto violation3      = violation1;
        violation3.timestamp = 1000000 + 6000000; // 6 seconds later (past window — should be delivered)

        REQUIRE(router.RouteViolation(violation1) == DCGM_ST_OK);
        REQUIRE(router.RouteViolation(violation2) == DCGM_ST_OK);
        REQUIRE(router.RouteViolation(violation3) == DCGM_ST_OK);

        // Two deliveries: violation1 and violation3. violation2 was rate-limited.
        REQUIRE(fixture.deliveredConditions.size() == 2);
    }

    SECTION("CALLBACK channel - stale timestamps do not regress rate limit state")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto guard = router.RegisterCallback(1, 100);
        REQUIRE(guard.has_value());

        auto first = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CALLBACK);
        first.timestamp = 10000000;

        auto stale      = first;
        stale.timestamp = first.timestamp - 1000000;

        auto stillLimited      = first;
        stillLimited.timestamp = first.timestamp + 4000000;

        auto outsideWindow      = first;
        outsideWindow.timestamp = first.timestamp + 6000000;

        REQUIRE(router.RouteViolation(first) == DCGM_ST_OK);
        REQUIRE(router.RouteViolation(stale) == DCGM_ST_OK);
        REQUIRE(router.RouteViolation(stillLimited) == DCGM_ST_OK);
        REQUIRE(router.RouteViolation(outsideWindow) == DCGM_ST_OK);

        // first and outsideWindow deliver. stale is ignored and must not move the marker backwards;
        // otherwise stillLimited would incorrectly be delivered.
        REQUIRE(fixture.deliveredConditions.size() == 2);
    }

    SECTION("CALLBACK channel - rate limiting is per (subscriber, fieldId), not per subscriber")
    {
        // Regression guard: an earlier violation on one fieldId must not suppress a near-simultaneous
        // violation on a different fieldId. Matches legacy DcgmPolicyManager::SetViolation semantics.
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto guard = router.RegisterCallback(1, 100);
        REQUIRE(guard.has_value());

        auto thermal
            = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, 85.0, 80.0, DCGM_POLICY_CHANNEL_CALLBACK);
        thermal.timestamp = 1000000;

        auto power
            = fixture.CreateViolation(DCGM_FI_DEV_BOARD_POWER_WATTS, 0, 350.0, 300.0, DCGM_POLICY_CHANNEL_CALLBACK);
        power.timestamp = 1000000 + 1000000; // 1 second later — well inside the 5s window

        auto thermalAgain      = thermal;
        thermalAgain.timestamp = 1000000 + 2000000; // 2s after thermal — within window, must be suppressed

        REQUIRE(router.RouteViolation(thermal) == DCGM_ST_OK);
        REQUIRE(router.RouteViolation(power) == DCGM_ST_OK);
        REQUIRE(router.RouteViolation(thermalAgain) == DCGM_ST_OK);

        // thermal + power delivered; thermalAgain suppressed by the per-fieldId thermal window.
        REQUIRE(fixture.deliveredConditions.size() == 2);
        REQUIRE(fixture.deliveredConditions[0] == DCGM_POLICY_COND_THERMAL);
        REQUIRE(fixture.deliveredConditions[1] == DCGM_POLICY_COND_POWER);
    }

    SECTION("CALLBACK channel - field to condition mapping")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto guard = router.RegisterCallback(1, 100);
        REQUIRE(guard.has_value());

        auto testField = [&](unsigned short fieldId, double value) {
            auto violation = fixture.CreateViolation(fieldId, 0, value, 0.0, DCGM_POLICY_CHANNEL_CALLBACK);
            REQUIRE(router.RouteViolation(violation) == DCGM_ST_OK);
        };

        testField(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 85.0);   // THERMAL
        testField(DCGM_FI_DEV_BOARD_POWER_WATTS, 350.0); // POWER
        testField(DCGM_FI_DEV_ECC_DBE_VOL_DEV, 5.0);     // DBE
        testField(DCGM_FI_DEV_XID_ERROR, 79.0);          // XID
        testField(9999, 100.0);                          // FIELD_VIOLATION (unmapped)
    }

    SECTION("Multi-channel - simultaneous routing")
    {
        TestNotificationRouter fixture;

        DcgmPolicyNotificationRouter router(fixture.callbacks);

        auto guard = router.RegisterCallback(1, 100);
        REQUIRE(guard.has_value());

        auto violation = fixture.CreateViolation(DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                 0,
                                                 85.0,
                                                 80.0,
                                                 DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_FILE
                                                     | DCGM_POLICY_CHANNEL_CALLBACK);

        REQUIRE(router.RouteViolation(violation) == DCGM_ST_OK);
        REQUIRE(router.GetRecentViolations(std::chrono::steady_clock::time_point::min()).size() == 1);
    }
}
