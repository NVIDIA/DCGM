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

#include <DcgmPolicy.h>
#include <DcgmPolicyRegistry.h>
#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <thread>
#include <vector>

static bool EntitiesEqual(std::vector<dcgmGroupEntityPair_t> const &left,
                          std::vector<dcgmGroupEntityPair_t> const &right)
{
    return left.size() == right.size()
           && std::equal(left.begin(), left.end(), right.begin(), [](auto const &leftEntity, auto const &rightEntity) {
                  return leftEntity.entityGroupId == rightEntity.entityGroupId
                         && leftEntity.entityId == rightEntity.entityId;
              });
}

static dcgm_policy_violation_t CreateViolation(DcgmPolicy const &policy, timelib64_t timestamp)
{
    dcgm_policy_violation_t violation = {};
    violation.policyId                = policy.GetPolicyId();
    violation.fieldId                 = policy.GetFieldId();
    violation.entity                  = { DCGM_FE_GPU, 0 };
    violation.currentValue            = policy.GetThreshold() + 1.0;
    violation.thresholdValue          = policy.GetThreshold();
    violation.policyOperator          = policy.GetPolicyOperator();
    violation.channelMask             = policy.GetChannelMask();
    violation.timestamp               = timestamp;
    return violation;
}

TEST_CASE("DcgmPolicy")
{
    SECTION("Constructor and getters")
    {
        DcgmPolicy policy(1,
                          "test",
                          0.0,
                          0.0,
                          DCGM_POLICY_OP_EQ,
                          { { DCGM_FE_GPU, 1 }, { DCGM_FE_GPU, 2 }, { DCGM_FE_GPU, 3 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);

        CHECK(policy.GetPolicyId() != 0);
        REQUIRE(policy.GetFieldId() == 1);
        REQUIRE(policy.GetName() == "test");
        REQUIRE(policy.GetValue() == 0.0f);
        REQUIRE(policy.GetThreshold() == 0.0);
        REQUIRE(policy.GetPolicyOperator() == DCGM_POLICY_OP_EQ);
        REQUIRE(EntitiesEqual(policy.GetEntities(), { { DCGM_FE_GPU, 1 }, { DCGM_FE_GPU, 2 }, { DCGM_FE_GPU, 3 } }));
        REQUIRE(policy.GetChannelMask() == static_cast<unsigned int>(DCGM_POLICY_CHANNEL_CONSOLE));
        REQUIRE(policy.IsEnabled() == true);
        REQUIRE(policy.GetRateLimit() == std::chrono::seconds { 10 });
        REQUIRE(policy.GetCreatedAt() > 0);
    }

    SECTION("Updates via registry produce new immutable policy instances")
    {
        unsigned short const fieldId = 100;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "temperature_policy",
                          0.0,
                          80.0,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                          DCGM_POLICY_CHANNEL_FILE,
                          DCGM_POLICY_STATE_DISABLED,
                          5);
        uint64_t const pid = policy.GetPolicyId();
        REQUIRE(registry.AddPolicy(policy) == DCGM_ST_OK);

        // UpdatePolicyThreshold produces a new policy with the new threshold
        REQUIRE(registry.UpdatePolicyThreshold(fieldId, pid, 90.0) == DCGM_ST_OK);
        std::vector<DcgmPolicy> policies;
        (void)registry.GetAllPolicies(std::back_inserter(policies));
        REQUIRE(policies.size() == 1);
        REQUIRE(policies[0].GetThreshold() == 90.0);

        // UpdatePolicyOperator
        REQUIRE(registry.UpdatePolicyOperator(fieldId, pid, DCGM_POLICY_OP_LT) == DCGM_ST_OK);
        policies.clear();
        (void)registry.GetAllPolicies(std::back_inserter(policies));
        REQUIRE(policies[0].GetPolicyOperator() == DCGM_POLICY_OP_LT);

        // UpdatePolicyEntities
        std::vector<dcgmGroupEntityPair_t> newEntities
            = { { DCGM_FE_GPU, 2 }, { DCGM_FE_GPU, 3 }, { DCGM_FE_GPU, 4 }, { DCGM_FE_GPU, 5 } };
        REQUIRE(registry.UpdatePolicyEntities(fieldId, pid, newEntities) == DCGM_ST_OK);
        policies.clear();
        (void)registry.GetAllPolicies(std::back_inserter(policies));
        REQUIRE(EntitiesEqual(policies[0].GetEntities(), newEntities));

        // UpdatePolicyChannels
        unsigned int newChannelMask
            = DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_CALLBACK | DCGM_POLICY_CHANNEL_FILE;
        REQUIRE(registry.UpdatePolicyChannels(fieldId, pid, newChannelMask) == DCGM_ST_OK);
        policies.clear();
        (void)registry.GetAllPolicies(std::back_inserter(policies));
        REQUIRE(policies[0].GetChannelMask() == newChannelMask);

        // SetPolicyState
        REQUIRE(registry.SetPolicyState(fieldId, pid, DCGM_POLICY_STATE_ENABLED) == DCGM_ST_OK);
        policies.clear();
        (void)registry.GetAllPolicies(std::back_inserter(policies));
        REQUIRE(policies[0].IsEnabled() == true);

        // UpdatePolicyRateLimit
        REQUIRE(registry.UpdatePolicyRateLimit(fieldId, pid, 20) == DCGM_ST_OK);
        policies.clear();
        (void)registry.GetAllPolicies(std::back_inserter(policies));
        REQUIRE(policies[0].GetRateLimit() == std::chrono::seconds { 20 });
    }

    SECTION("Unique policy IDs")
    {
        DcgmPolicy policy1(150,
                           "power_policy",
                           0.0,
                           250.0,
                           DCGM_POLICY_OP_GT,
                           { { DCGM_FE_GPU, 0 } },
                           DCGM_POLICY_CHANNEL_CONSOLE,
                           DCGM_POLICY_STATE_ENABLED,
                           10);
        DcgmPolicy policy2(150,
                           "power_policy",
                           0.0,
                           250.0,
                           DCGM_POLICY_OP_GT,
                           { { DCGM_FE_GPU, 0 } },
                           DCGM_POLICY_CHANNEL_CONSOLE,
                           DCGM_POLICY_STATE_ENABLED,
                           10);
        DcgmPolicy policy3(200,
                           "different_policy",
                           0.0,
                           100.0,
                           DCGM_POLICY_OP_EQ,
                           { { DCGM_FE_GPU, 1 }, { DCGM_FE_GPU, 2 } },
                           DCGM_POLICY_CHANNEL_CALLBACK,
                           DCGM_POLICY_STATE_DISABLED,
                           5);

        // Even with identical parameters, policy IDs should be unique
        REQUIRE(policy1.GetPolicyId() != policy2.GetPolicyId());
        REQUIRE(policy1.GetPolicyId() != policy3.GetPolicyId());
        REQUIRE(policy2.GetPolicyId() != policy3.GetPolicyId());
    }

    SECTION("Violation tracking - initial state")
    {
        DcgmPolicy policy(150,
                          "test_policy",
                          0.0,
                          80.0,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          0);

        // Initial state should have zero violations
        REQUIRE(policy.GetViolationCount() == 0);
        REQUIRE(policy.GetLastTriggeredTimestamp() == 0);
    }

    SECTION("Violation tracking - RecordViolation via registry increments count and updates timestamp")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(150,
                          "test_policy",
                          0.0,
                          80.0,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          0);
        REQUIRE(registry.AddPolicy(policy) == DCGM_ST_OK);

        auto getPolicy = [&]() {
            std::vector<DcgmPolicy> policies;
            (void)registry.GetAllPolicies(std::back_inserter(policies));
            REQUIRE(policies.size() == 1);
            return policies[0];
        };

        timelib64_t timestamp1 = 1000000;
        REQUIRE(registry.RecordViolation(CreateViolation(policy, timestamp1)) == DCGM_ST_OK);
        REQUIRE(getPolicy().GetViolationCount() == 1);
        REQUIRE(getPolicy().GetLastTriggeredTimestamp() == timestamp1);

        // Record another violation
        timelib64_t timestamp2 = 2000000;
        REQUIRE(registry.RecordViolation(CreateViolation(policy, timestamp2)) == DCGM_ST_OK);
        REQUIRE(getPolicy().GetViolationCount() == 2);
        REQUIRE(getPolicy().GetLastTriggeredTimestamp() == timestamp2);

        // Record multiple violations
        for (int i = 0; i < 10; i++)
        {
            CHECK(registry.RecordViolation(CreateViolation(policy, timestamp2 + i * 1000)) == DCGM_ST_OK);
        }
        REQUIRE(getPolicy().GetViolationCount() == 12);
    }

    SECTION("Violation tracking - RecordViolation returns DCGM_ST_NO_DATA for unknown fieldId")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(150,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);

        // Empty registry: every fieldId is unknown
        REQUIRE(registry.RecordViolation(CreateViolation(policy, 1000)) == DCGM_ST_NO_DATA);

        // Adding an unrelated field leaves the target fieldId still unknown
        REQUIRE(registry.AddPolicy(policy) == DCGM_ST_OK);
        REQUIRE(registry.RecordViolation(CreateViolation(policy, 1000)) == DCGM_ST_OK);

        auto missingFieldViolation    = CreateViolation(policy, 2000);
        missingFieldViolation.fieldId = 999;
        REQUIRE(registry.RecordViolation(missingFieldViolation) == DCGM_ST_NO_DATA);
    }
}
