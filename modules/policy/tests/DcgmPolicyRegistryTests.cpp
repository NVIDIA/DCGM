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

#include <DcgmPolicyRegistry.h>
#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <chrono>

static std::vector<DcgmPolicy> GetAllPolicies(DcgmPolicyRegistry &registry)
{
    std::vector<DcgmPolicy> out;
    (void)registry.GetAllPolicies(std::back_inserter(out));
    return out;
}

static std::vector<DcgmPolicy> GetPoliciesForField(DcgmPolicyRegistry &registry, unsigned short fieldId)
{
    std::vector<DcgmPolicy> out;
    (void)registry.GetPoliciesForField(fieldId, std::back_inserter(out));
    return out;
}

static size_t RegistryPolicyCount(DcgmPolicyRegistry &registry)
{
    return GetAllPolicies(registry).size();
}

static dcgm_policy_violation_t CreateViolation(DcgmPolicy const &policy,
                                               dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 },
                                               timelib64_t timestamp        = 1000)
{
    dcgm_policy_violation_t violation = {};
    violation.policyId                = policy.GetPolicyId();
    violation.fieldId                 = policy.GetFieldId();
    violation.entity                  = entity;
    violation.currentValue            = policy.GetThreshold() + 1.0;
    violation.thresholdValue          = policy.GetThreshold();
    violation.policyOperator          = policy.GetPolicyOperator();
    violation.channelMask             = policy.GetChannelMask();
    violation.timestamp               = timestamp;
    return violation;
}

TEST_CASE("DcgmPolicyRegistry: AddPolicy")
{
    SECTION("AddPolicy succeeds and stores correct properties")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(150,
                          "temperature_policy",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                          DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_FILE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);

        REQUIRE(registry.AddPolicy(policy) == DCGM_ST_OK);
        REQUIRE(registry.HasPolicy(150));
        REQUIRE(RegistryPolicyCount(registry) == 1);

        auto stored = GetPoliciesForField(registry, 150);
        REQUIRE(stored.size() == 1);
        REQUIRE(stored[0].GetFieldId() == 150);
        REQUIRE(stored[0].GetName() == "temperature_policy");
        REQUIRE(stored[0].GetThreshold() == 80.0f);
        REQUIRE(stored[0].GetPolicyOperator() == DCGM_POLICY_OP_GT);
        REQUIRE(stored[0].IsEnabled() == true);
        REQUIRE(stored[0].GetRateLimit() == std::chrono::seconds { 10 });
    }

    SECTION("AddPolicy rejects duplicate fieldId")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy1(150,
                           "first_policy",
                           0.0f,
                           80.0f,
                           DCGM_POLICY_OP_GT,
                           { { DCGM_FE_GPU, 0 } },
                           DCGM_POLICY_CHANNEL_CONSOLE,
                           DCGM_POLICY_STATE_ENABLED,
                           10);
        DcgmPolicy policy2(150,
                           "second_policy",
                           0.0f,
                           90.0f,
                           DCGM_POLICY_OP_LT,
                           { { DCGM_FE_GPU, 1 } },
                           DCGM_POLICY_CHANNEL_FILE,
                           DCGM_POLICY_STATE_DISABLED,
                           5);

        REQUIRE(registry.AddPolicy(policy1) == DCGM_ST_OK);
        REQUIRE(registry.AddPolicy(policy2) == DCGM_ST_DUPLICATE_KEY);
        REQUIRE(RegistryPolicyCount(registry) == 1);

        auto stored = GetPoliciesForField(registry, 150);
        REQUIRE(stored.size() == 1);
        REQUIRE(stored[0].GetName() == "first_policy");
    }
}

TEST_CASE("DcgmPolicyRegistry: RemovePolicy")
{
    SECTION("RemovePolicy by policyId removes the correct policy")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(150,
                          "temperature_policy",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        uint64_t const pid = policy.GetPolicyId();

        registry.AddPolicy(policy);

        REQUIRE(registry.RemovePolicy(150, pid) == DCGM_ST_OK);
        REQUIRE_FALSE(registry.HasPolicy(150));
        REQUIRE(RegistryPolicyCount(registry) == 0);
    }

    SECTION("RemovePolicy returns DCGM_ST_NO_DATA for unknown fieldId")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy dummy(150,
                         "x",
                         0.0f,
                         0.0f,
                         DCGM_POLICY_OP_GT,
                         { { DCGM_FE_GPU, 0 } },
                         DCGM_POLICY_CHANNEL_CONSOLE,
                         DCGM_POLICY_STATE_ENABLED,
                         10);
        REQUIRE(registry.RemovePolicy(999, dummy.GetPolicyId()) == DCGM_ST_NO_DATA);
    }

    SECTION("RemovePolicy returns DCGM_ST_NO_DATA for stale policyId on existing fieldId")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(150,
                          "x",
                          0.0f,
                          0.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        registry.AddPolicy(policy);
        REQUIRE(registry.RemovePolicy(150, policy.GetPolicyId() + 1) == DCGM_ST_NO_DATA);
        REQUIRE(registry.HasPolicy(150));
    }
}


TEST_CASE("DcgmPolicyRegistry: GetPolicyById")
{
    SECTION("Returns policy when it exists")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(150,
                          "temperature_policy",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        registry.AddPolicy(policy);

        auto result = registry.GetPolicyById(policy.GetPolicyId());
        REQUIRE(result.has_value());
        REQUIRE(result->GetPolicyId() == policy.GetPolicyId());
        REQUIRE(result->GetFieldId() == 150);
        REQUIRE(result->GetName() == "temperature_policy");
        REQUIRE(result->GetThreshold() == 80.0f);
    }

    SECTION("Returns nullopt for non-existent policyId")
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
        registry.AddPolicy(policy);
        uint64_t const unusedId = policy.GetPolicyId() + 1;
        REQUIRE_FALSE(registry.GetPolicyById(unusedId).has_value());
    }

    SECTION("Finds correct policy among multiple policies across fields")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy1(150,
                           "temp",
                           0.0f,
                           80.0f,
                           DCGM_POLICY_OP_GT,
                           { { DCGM_FE_GPU, 0 } },
                           DCGM_POLICY_CHANNEL_CONSOLE,
                           DCGM_POLICY_STATE_ENABLED,
                           10);
        DcgmPolicy policy2(155,
                           "power",
                           0.0f,
                           90.0f,
                           DCGM_POLICY_OP_LT,
                           { { DCGM_FE_GPU, 1 } },
                           DCGM_POLICY_CHANNEL_FILE,
                           DCGM_POLICY_STATE_ENABLED,
                           5);
        DcgmPolicy policy3(160,
                           "memory",
                           0.0f,
                           100.0f,
                           DCGM_POLICY_OP_EQ,
                           { { DCGM_FE_GPU, 2 } },
                           DCGM_POLICY_CHANNEL_CALLBACK,
                           DCGM_POLICY_STATE_DISABLED,
                           15);

        registry.AddPolicy(policy1);
        registry.AddPolicy(policy2);
        registry.AddPolicy(policy3);

        auto result = registry.GetPolicyById(policy2.GetPolicyId());
        REQUIRE(result.has_value());
        REQUIRE(result->GetFieldId() == 155);
        REQUIRE(result->GetName() == "power");
    }

    SECTION("Returns nullopt after the policy is removed")
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
        uint64_t const pid = policy.GetPolicyId();
        registry.AddPolicy(policy);
        REQUIRE(registry.GetPolicyById(pid).has_value());
        REQUIRE(registry.RemovePolicy(150, pid) == DCGM_ST_OK);
        REQUIRE_FALSE(registry.GetPolicyById(pid).has_value());
    }
}


TEST_CASE("DcgmPolicyRegistry: GetAllPolicies")
{
    SECTION("Returns all policies across all fields")
    {
        DcgmPolicyRegistry registry;
        registry.AddPolicy(DcgmPolicy(150,
                                      "temp",
                                      0.0f,
                                      80.0f,
                                      DCGM_POLICY_OP_GT,
                                      { { DCGM_FE_GPU, 0 } },
                                      DCGM_POLICY_CHANNEL_CONSOLE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      10));
        registry.AddPolicy(DcgmPolicy(155,
                                      "power",
                                      0.0f,
                                      250.0f,
                                      DCGM_POLICY_OP_GT,
                                      { { DCGM_FE_GPU, 1 } },
                                      DCGM_POLICY_CHANNEL_FILE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      5));
        registry.AddPolicy(DcgmPolicy(160,
                                      "memory",
                                      0.0f,
                                      90.0f,
                                      DCGM_POLICY_OP_LT,
                                      { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                                      DCGM_POLICY_CHANNEL_CALLBACK,
                                      DCGM_POLICY_STATE_DISABLED,
                                      15));

        auto policies = GetAllPolicies(registry);
        REQUIRE(policies.size() == 3);

        bool found150 = false, found155 = false, found160 = false;
        for (auto const &p : policies)
        {
            if (p.GetFieldId() == 150)
                found150 = true;
            if (p.GetFieldId() == 155)
                found155 = true;
            if (p.GetFieldId() == 160)
                found160 = true;
        }
        REQUIRE(found150);
        REQUIRE(found155);
        REQUIRE(found160);
    }

    SECTION("Count changes correctly with add and remove")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy p1(150,
                      "p1",
                      0.0f,
                      80.0f,
                      DCGM_POLICY_OP_EQ,
                      { { DCGM_FE_GPU, 0 } },
                      DCGM_POLICY_CHANNEL_CONSOLE,
                      DCGM_POLICY_STATE_ENABLED,
                      10);
        DcgmPolicy p2(155,
                      "p2",
                      0.0f,
                      90.0f,
                      DCGM_POLICY_OP_GT,
                      { { DCGM_FE_GPU, 1 } },
                      DCGM_POLICY_CHANNEL_FILE,
                      DCGM_POLICY_STATE_ENABLED,
                      5);

        REQUIRE(RegistryPolicyCount(registry) == 0);
        uint64_t const p1Id = p1.GetPolicyId();
        registry.AddPolicy(p1);
        REQUIRE(RegistryPolicyCount(registry) == 1);
        registry.AddPolicy(p2);
        REQUIRE(RegistryPolicyCount(registry) == 2);
        registry.RemovePolicy(p1.GetFieldId(), p1Id);
        REQUIRE(RegistryPolicyCount(registry) == 1);
    }
}

TEST_CASE("DcgmPolicyRegistry: RemoveAllPolicies")
{
    SECTION("RemoveAllPolicies removes all policies")
    {
        DcgmPolicyRegistry registry;
        registry.AddPolicy(DcgmPolicy(150,
                                      "p1",
                                      0.0f,
                                      80.0f,
                                      DCGM_POLICY_OP_EQ,
                                      { { DCGM_FE_GPU, 0 } },
                                      DCGM_POLICY_CHANNEL_CONSOLE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      10));
        registry.AddPolicy(DcgmPolicy(155,
                                      "p2",
                                      0.0f,
                                      90.0f,
                                      DCGM_POLICY_OP_GT,
                                      { { DCGM_FE_GPU, 1 } },
                                      DCGM_POLICY_CHANNEL_FILE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      5));

        REQUIRE(RegistryPolicyCount(registry) == 2);
        registry.RemoveAllPolicies();
        REQUIRE(RegistryPolicyCount(registry) == 0);
        REQUIRE_FALSE(registry.HasPolicy(150));
        REQUIRE_FALSE(registry.HasPolicy(155));
    }

    SECTION("RemoveAllPolicies on empty registry succeeds and remains empty")
    {
        DcgmPolicyRegistry registry;
        REQUIRE(registry.RemoveAllPolicies() == DCGM_ST_OK);
        REQUIRE(RegistryPolicyCount(registry) == 0);
    }
}

TEST_CASE("DcgmPolicyRegistry: ReplacePoliciesForField")
{
    SECTION("ReplacePoliciesForField replaces the policy for a field")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy original(fieldId,
                            "original",
                            0.0f,
                            80.0f,
                            DCGM_POLICY_OP_GT,
                            { { DCGM_FE_GPU, 0 } },
                            DCGM_POLICY_CHANNEL_CONSOLE,
                            DCGM_POLICY_STATE_ENABLED,
                            10);
        registry.AddPolicy(original);

        DcgmPolicy replacement(fieldId,
                               "replaced",
                               0.0f,
                               95.0f,
                               DCGM_POLICY_OP_LT,
                               { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                               DCGM_POLICY_CHANNEL_FILE,
                               DCGM_POLICY_STATE_ENABLED,
                               20);

        REQUIRE(registry.ReplacePoliciesForField(fieldId, { replacement }) == DCGM_ST_OK);

        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored.size() == 1);
        REQUIRE(stored[0].GetName() == "replaced");
        REQUIRE(stored[0].GetThreshold() == 95.0f);
        REQUIRE(stored[0].GetPolicyOperator() == DCGM_POLICY_OP_LT);
    }

    SECTION("ReplacePoliciesForField returns DCGM_ST_NO_DATA for non-existent field")
    {
        unsigned short const fieldId = 999;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_EQ,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        REQUIRE(registry.ReplacePoliciesForField(fieldId, { policy }) == DCGM_ST_NO_DATA);
    }

    SECTION("ReplacePoliciesForField returns DCGM_ST_BADPARAM when policy has mismatched fieldId")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy original(fieldId,
                            "original",
                            0.0f,
                            80.0f,
                            DCGM_POLICY_OP_EQ,
                            { { DCGM_FE_GPU, 0 } },
                            DCGM_POLICY_CHANNEL_CONSOLE,
                            DCGM_POLICY_STATE_ENABLED,
                            10);
        DcgmPolicy wrongField(999,
                              "wrong",
                              0.0f,
                              90.0f,
                              DCGM_POLICY_OP_EQ,
                              { { DCGM_FE_GPU, 0 } },
                              DCGM_POLICY_CHANNEL_CONSOLE,
                              DCGM_POLICY_STATE_ENABLED,
                              10);

        registry.AddPolicy(original);
        REQUIRE(registry.ReplacePoliciesForField(fieldId, { wrongField }) == DCGM_ST_BADPARAM);
    }

    SECTION("ReplacePoliciesForField returns DCGM_ST_BADPARAM for empty list")
    {
        DcgmPolicyRegistry registry;
        REQUIRE(registry.ReplacePoliciesForField(150, {}) == DCGM_ST_BADPARAM);
    }

    SECTION("ReplacePoliciesForField installs replacement identity and resets violation history")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy original(fieldId,
                            "original",
                            0.0f,
                            80.0f,
                            DCGM_POLICY_OP_GT,
                            { { DCGM_FE_GPU, 0 } },
                            DCGM_POLICY_CHANNEL_CONSOLE,
                            DCGM_POLICY_STATE_ENABLED,
                            0);
        uint64_t const originalPid = original.GetPolicyId();
        registry.AddPolicy(original);

        // Build up some violation history on the original
        REQUIRE(registry.RecordViolation(CreateViolation(original, { DCGM_FE_GPU, 0 }, 1000)) == DCGM_ST_OK);
        REQUIRE(registry.RecordViolation(CreateViolation(original, { DCGM_FE_GPU, 0 }, 2000)) == DCGM_ST_OK);

        DcgmPolicy replacement(fieldId,
                               "replaced",
                               0.0f,
                               95.0f,
                               DCGM_POLICY_OP_LT,
                               { { DCGM_FE_GPU, 1 } },
                               DCGM_POLICY_CHANNEL_FILE,
                               DCGM_POLICY_STATE_ENABLED,
                               20);
        uint64_t const newPid          = replacement.GetPolicyId();
        timelib64_t const newCreatedAt = replacement.GetCreatedAt();
        REQUIRE(newPid != originalPid);

        REQUIRE(registry.ReplacePoliciesForField(fieldId, { replacement }) == DCGM_ST_OK);

        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored.size() == 1);
        REQUIRE(stored[0].GetPolicyId() == newPid);        // replacement's id wins
        REQUIRE(stored[0].GetCreatedAt() == newCreatedAt); // replacement's createdAt wins
        REQUIRE(stored[0].GetViolationCount() == 0);       // history is NOT carried over
        REQUIRE(stored[0].GetLastTriggeredTimestamp() == 0);
    }

    SECTION("After ReplacePoliciesForField, the original policyId is stale and rejected by Update/Remove")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy original(fieldId,
                            "original",
                            0.0f,
                            80.0f,
                            DCGM_POLICY_OP_GT,
                            { { DCGM_FE_GPU, 0 } },
                            DCGM_POLICY_CHANNEL_CONSOLE,
                            DCGM_POLICY_STATE_ENABLED,
                            10);
        uint64_t const originalPid = original.GetPolicyId();
        registry.AddPolicy(original);

        DcgmPolicy replacement(fieldId,
                               "replaced",
                               0.0f,
                               95.0f,
                               DCGM_POLICY_OP_LT,
                               { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                               DCGM_POLICY_CHANNEL_FILE,
                               DCGM_POLICY_STATE_ENABLED,
                               20);
        uint64_t const newPid = replacement.GetPolicyId();
        REQUIRE(newPid != originalPid);
        REQUIRE(registry.ReplacePoliciesForField(fieldId, { replacement }) == DCGM_ST_OK);

        // Every mutating op keyed by the stale (original) policyId must fail
        CHECK(registry.RemovePolicy(fieldId, originalPid) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyValue(fieldId, originalPid, 1.0f) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyThreshold(fieldId, originalPid, 1.0f) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyOperator(fieldId, originalPid, DCGM_POLICY_OP_EQ) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyEntities(fieldId, originalPid, { { DCGM_FE_GPU, 0 } }) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyChannels(fieldId, originalPid, DCGM_POLICY_CHANNEL_CONSOLE) == DCGM_ST_NO_DATA);
        CHECK(registry.SetPolicyState(fieldId, originalPid, DCGM_POLICY_STATE_DISABLED) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyRateLimit(fieldId, originalPid, 5) == DCGM_ST_NO_DATA);

        // The new policyId is accepted
        REQUIRE(registry.UpdatePolicyThreshold(fieldId, newPid, 100.0f) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored[0].GetThreshold() == 100.0f);
    }

    SECTION("ReplacePoliciesForField clears rate-limit state when the policy ID is preserved")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy original(fieldId,
                            "original",
                            0.0f,
                            80.0f,
                            DCGM_POLICY_OP_GT,
                            { { DCGM_FE_GPU, 0 } },
                            DCGM_POLICY_CHANNEL_CONSOLE,
                            DCGM_POLICY_STATE_ENABLED,
                            10);
        REQUIRE(registry.AddPolicy(original) == DCGM_ST_OK);
        REQUIRE(registry.RecordViolation(CreateViolation(original, { DCGM_FE_GPU, 0 }, 1000000)) == DCGM_ST_OK);

        DcgmPolicy replacement(fieldId,
                               "replacement",
                               0.0f,
                               80.0f,
                               DCGM_POLICY_OP_GT,
                               { { DCGM_FE_GPU, 0 } },
                               DCGM_POLICY_CHANNEL_CONSOLE,
                               DCGM_POLICY_STATE_ENABLED,
                               10,
                               original.GetPolicyId(),
                               original.GetCreatedAt(),
                               0,
                               0);
        REQUIRE(registry.ReplacePoliciesForField(fieldId, { replacement }) == DCGM_ST_OK);
        REQUIRE(registry.RecordViolation(CreateViolation(replacement, { DCGM_FE_GPU, 0 }, 1000001)) == DCGM_ST_OK);
    }

    SECTION("ReplacePoliciesForField rejects multiple policies for one field")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy original(fieldId,
                            "original",
                            0.0f,
                            80.0f,
                            DCGM_POLICY_OP_GT,
                            { { DCGM_FE_GPU, 0 } },
                            DCGM_POLICY_CHANNEL_CONSOLE,
                            DCGM_POLICY_STATE_ENABLED,
                            10);
        registry.AddPolicy(original);

        DcgmPolicy first(fieldId,
                         "first",
                         0.0f,
                         11.0f,
                         DCGM_POLICY_OP_LT,
                         { { DCGM_FE_GPU, 1 } },
                         DCGM_POLICY_CHANNEL_FILE,
                         DCGM_POLICY_STATE_ENABLED,
                         5);
        DcgmPolicy second(fieldId,
                          "second",
                          0.0f,
                          22.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 2 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_DISABLED,
                          7);

        REQUIRE(registry.ReplacePoliciesForField(fieldId, { first, second }) == DCGM_ST_BADPARAM);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored.size() == 1);
        REQUIRE(stored[0].GetName() == "original");
        REQUIRE(stored[0].GetThreshold() == 80.0f);
    }
}

TEST_CASE("DcgmPolicyRegistry: Granular Updates")
{
    SECTION("UpdatePolicyValue updates only value")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        uint64_t const pid = policy.GetPolicyId();
        registry.AddPolicy(policy);

        REQUIRE(registry.UpdatePolicyValue(fieldId, pid, 75.5f) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored[0].GetValue() == 75.5f);
        REQUIRE(stored[0].GetThreshold() == 80.0f); // unchanged
    }

    SECTION("UpdatePolicyThreshold updates only threshold")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        uint64_t const pid = policy.GetPolicyId();
        registry.AddPolicy(policy);

        REQUIRE(registry.UpdatePolicyThreshold(fieldId, pid, 95.0f) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored[0].GetThreshold() == 95.0f);
        REQUIRE(stored[0].GetPolicyOperator() == DCGM_POLICY_OP_GT); // unchanged
    }

    SECTION("UpdatePolicyOperator updates only operator")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        uint64_t const pid = policy.GetPolicyId();
        registry.AddPolicy(policy);

        REQUIRE(registry.UpdatePolicyOperator(fieldId, pid, DCGM_POLICY_OP_LT) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored[0].GetPolicyOperator() == DCGM_POLICY_OP_LT);
        REQUIRE(stored[0].GetThreshold() == 80.0f); // unchanged
    }

    SECTION("UpdatePolicyEntities updates only GPU IDs")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        uint64_t const pid = policy.GetPolicyId();
        registry.AddPolicy(policy);

        std::vector<dcgmGroupEntityPair_t> newEntities
            = { { DCGM_FE_GPU, 2 }, { DCGM_FE_GPU, 3 }, { DCGM_FE_GPU, 4 }, { DCGM_FE_GPU, 5 } };
        REQUIRE(registry.UpdatePolicyEntities(fieldId, pid, newEntities) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored[0].GetEntities().size() == newEntities.size());
        REQUIRE(stored[0].GetEntities()[0].entityId == newEntities[0].entityId);
        REQUIRE(stored[0].IsEnabled() == true); // unchanged
    }

    SECTION("UpdatePolicyChannels updates only channel mask")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        uint64_t const pid = policy.GetPolicyId();
        registry.AddPolicy(policy);

        unsigned int newMask = DCGM_POLICY_CHANNEL_FILE | DCGM_POLICY_CHANNEL_CALLBACK;
        REQUIRE(registry.UpdatePolicyChannels(fieldId, pid, newMask) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored[0].GetChannelMask() == newMask);
        REQUIRE(stored[0].GetRateLimit() == std::chrono::seconds { 10 }); // unchanged
    }

    SECTION("SetPolicyState updates only state")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        uint64_t const pid = policy.GetPolicyId();
        registry.AddPolicy(policy);

        REQUIRE(registry.SetPolicyState(fieldId, pid, DCGM_POLICY_STATE_DISABLED) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored[0].IsEnabled() == false);
        REQUIRE(stored[0].GetName() == "test"); // unchanged
    }

    SECTION("UpdatePolicyRateLimit updates only rate limit")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        uint64_t const pid = policy.GetPolicyId();
        registry.AddPolicy(policy);

        REQUIRE(registry.UpdatePolicyRateLimit(fieldId, pid, 25) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored[0].GetRateLimit() == std::chrono::seconds { 25 });
        REQUIRE(stored[0].GetThreshold() == 80.0f); // unchanged
    }

    SECTION("All granular updates return DCGM_ST_NO_DATA for non-existent (fieldId, policyId)")
    {
        unsigned short const fieldId        = 150;
        unsigned short const missingFieldId = 999;
        DcgmPolicyRegistry registry;
        DcgmPolicy dummy(fieldId,
                         "x",
                         0.0f,
                         0.0f,
                         DCGM_POLICY_OP_GT,
                         { { DCGM_FE_GPU, 0 } },
                         DCGM_POLICY_CHANNEL_CONSOLE,
                         DCGM_POLICY_STATE_ENABLED,
                         10);
        registry.AddPolicy(dummy);
        uint64_t const stalePid = dummy.GetPolicyId() + 1; // guaranteed different from any real id

        // Unknown fieldId
        CHECK(registry.UpdatePolicyValue(missingFieldId, dummy.GetPolicyId(), 75.0f) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyThreshold(missingFieldId, dummy.GetPolicyId(), 90.0f) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyOperator(missingFieldId, dummy.GetPolicyId(), DCGM_POLICY_OP_EQ) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyEntities(missingFieldId, dummy.GetPolicyId(), { { DCGM_FE_GPU, 0 } })
              == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyChannels(missingFieldId, dummy.GetPolicyId(), DCGM_POLICY_CHANNEL_CONSOLE)
              == DCGM_ST_NO_DATA);
        CHECK(registry.SetPolicyState(missingFieldId, dummy.GetPolicyId(), DCGM_POLICY_STATE_ENABLED)
              == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyRateLimit(missingFieldId, dummy.GetPolicyId(), 10) == DCGM_ST_NO_DATA);

        // Known fieldId with stale policyId
        CHECK(registry.UpdatePolicyValue(fieldId, stalePid, 75.0f) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyThreshold(fieldId, stalePid, 90.0f) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyOperator(fieldId, stalePid, DCGM_POLICY_OP_EQ) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyEntities(fieldId, stalePid, { { DCGM_FE_GPU, 0 } }) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyChannels(fieldId, stalePid, DCGM_POLICY_CHANNEL_CONSOLE) == DCGM_ST_NO_DATA);
        CHECK(registry.SetPolicyState(fieldId, stalePid, DCGM_POLICY_STATE_ENABLED) == DCGM_ST_NO_DATA);
        CHECK(registry.UpdatePolicyRateLimit(fieldId, stalePid, 10) == DCGM_ST_NO_DATA);
    }

    SECTION("RecordViolation rejects stale queued violations")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        registry.AddPolicy(policy);

        auto validViolation = CreateViolation(policy, { DCGM_FE_GPU, 0 }, 1000);
        REQUIRE(registry.RecordViolation(validViolation) == DCGM_ST_OK);

        auto mismatchedPolicyId     = validViolation;
        mismatchedPolicyId.policyId = validViolation.policyId + 1;
        CHECK(registry.RecordViolation(mismatchedPolicyId) == DCGM_ST_NO_DATA);

        auto mismatchedFieldId    = validViolation;
        mismatchedFieldId.fieldId = fieldId + 1;
        CHECK(registry.RecordViolation(mismatchedFieldId) == DCGM_ST_NO_DATA);

        auto mismatchedOperator           = validViolation;
        mismatchedOperator.policyOperator = DCGM_POLICY_OP_LT;
        CHECK(registry.RecordViolation(mismatchedOperator) == DCGM_ST_NO_DATA);

        auto mismatchedThreshold           = validViolation;
        mismatchedThreshold.thresholdValue = 90.0;
        CHECK(registry.RecordViolation(mismatchedThreshold) == DCGM_ST_NO_DATA);

        auto mismatchedChannel        = validViolation;
        mismatchedChannel.channelMask = DCGM_POLICY_CHANNEL_FILE;
        CHECK(registry.RecordViolation(mismatchedChannel) == DCGM_ST_NO_DATA);

        auto mismatchedEntity   = validViolation;
        mismatchedEntity.entity = { DCGM_FE_GPU, 1 };
        CHECK(registry.RecordViolation(mismatchedEntity) == DCGM_ST_NO_DATA);

        REQUIRE(registry.SetPolicyState(fieldId, policy.GetPolicyId(), DCGM_POLICY_STATE_DISABLED) == DCGM_ST_OK);
        CHECK(registry.RecordViolation(validViolation) == DCGM_ST_NOT_CONFIGURED);

        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored.size() == 1);
        REQUIRE(stored[0].GetViolationCount() == 1);
        REQUIRE(stored[0].GetLastTriggeredTimestamp() == validViolation.timestamp);
    }

    SECTION("RecordViolation rate limits each entity independently")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "rate-limited",
                          0.0,
                          80.0,
                          DCGM_POLICY_OP_GT,
                          {},
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        REQUIRE(registry.AddPolicy(policy) == DCGM_ST_OK);

        timelib64_t constexpr firstTimestamp = 1000000;
        auto const firstGpu0                 = CreateViolation(policy, { DCGM_FE_GPU, 0 }, firstTimestamp);
        REQUIRE(registry.RecordViolation(firstGpu0) == DCGM_ST_OK);

        auto const suppressedGpu0 = CreateViolation(policy, { DCGM_FE_GPU, 0 }, firstTimestamp + 9999999);
        CHECK(registry.RecordViolation(suppressedGpu0) == DCGM_ST_NO_DATA);

        auto const firstGpu1 = CreateViolation(policy, { DCGM_FE_GPU, 1 }, firstTimestamp + 1000);
        REQUIRE(registry.RecordViolation(firstGpu1) == DCGM_ST_OK);

        auto const resumedGpu0 = CreateViolation(policy, { DCGM_FE_GPU, 0 }, firstTimestamp + 10000000);
        REQUIRE(registry.RecordViolation(resumedGpu0) == DCGM_ST_OK);

        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored.size() == 1);
        CHECK(stored[0].GetViolationCount() == 3);
        CHECK(stored[0].GetLastTriggeredTimestamp() == resumedGpu0.timestamp);
    }

    SECTION("RecordViolation does not rate limit when the configured rate is zero")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "unlimited",
                          0.0,
                          80.0,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          0);
        REQUIRE(registry.AddPolicy(policy) == DCGM_ST_OK);

        REQUIRE(registry.RecordViolation(CreateViolation(policy, { DCGM_FE_GPU, 0 }, 1000000)) == DCGM_ST_OK);
        REQUIRE(registry.RecordViolation(CreateViolation(policy, { DCGM_FE_GPU, 0 }, 1000001)) == DCGM_ST_OK);

        auto stored = GetPoliciesForField(registry, fieldId);
        REQUIRE(stored.size() == 1);
        CHECK(stored[0].GetViolationCount() == 2);
    }

    SECTION("Granular updates preserve violation history")
    {
        unsigned short const fieldId = 150;
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(fieldId,
                          "test",
                          0.0f,
                          80.0f,
                          DCGM_POLICY_OP_GT,
                          { { DCGM_FE_GPU, 0 } },
                          DCGM_POLICY_CHANNEL_CONSOLE,
                          DCGM_POLICY_STATE_ENABLED,
                          0);
        registry.AddPolicy(policy);

        // Record two violations before any config update
        timelib64_t const ts1 = 1000000;
        timelib64_t const ts2 = 2000000;
        REQUIRE(registry.RecordViolation(CreateViolation(policy, { DCGM_FE_GPU, 0 }, ts1)) == DCGM_ST_OK);
        REQUIRE(registry.RecordViolation(CreateViolation(policy, { DCGM_FE_GPU, 0 }, ts2)) == DCGM_ST_OK);

        auto before = GetPoliciesForField(registry, fieldId);
        REQUIRE(before[0].GetViolationCount() == 2);
        REQUIRE(before[0].GetLastTriggeredTimestamp() == ts2);

        uint64_t const originalPolicyId     = before[0].GetPolicyId();
        timelib64_t const originalCreatedAt = before[0].GetCreatedAt();

        // Each update method must carry violation state and identity forward
        REQUIRE(registry.UpdatePolicyThreshold(fieldId, originalPolicyId, 90.0f) == DCGM_ST_OK);
        auto after = GetPoliciesForField(registry, fieldId);
        REQUIRE(after[0].GetViolationCount() == 2);
        REQUIRE(after[0].GetLastTriggeredTimestamp() == ts2);
        REQUIRE(after[0].GetThreshold() == 90.0f);
        REQUIRE(after[0].GetPolicyId() == originalPolicyId);
        REQUIRE(after[0].GetCreatedAt() == originalCreatedAt);

        REQUIRE(registry.UpdatePolicyValue(fieldId, originalPolicyId, 5.0f) == DCGM_ST_OK);
        after = GetPoliciesForField(registry, fieldId);
        REQUIRE(after[0].GetViolationCount() == 2);
        REQUIRE(after[0].GetLastTriggeredTimestamp() == ts2);
        REQUIRE(after[0].GetPolicyId() == originalPolicyId);
        REQUIRE(after[0].GetCreatedAt() == originalCreatedAt);

        REQUIRE(registry.UpdatePolicyOperator(fieldId, originalPolicyId, DCGM_POLICY_OP_LT) == DCGM_ST_OK);
        after = GetPoliciesForField(registry, fieldId);
        REQUIRE(after[0].GetViolationCount() == 2);
        REQUIRE(after[0].GetLastTriggeredTimestamp() == ts2);
        REQUIRE(after[0].GetPolicyId() == originalPolicyId);
        REQUIRE(after[0].GetCreatedAt() == originalCreatedAt);

        REQUIRE(registry.UpdatePolicyEntities(fieldId, originalPolicyId, { { DCGM_FE_GPU, 1 }, { DCGM_FE_GPU, 2 } })
                == DCGM_ST_OK);
        after = GetPoliciesForField(registry, fieldId);
        REQUIRE(after[0].GetViolationCount() == 2);
        REQUIRE(after[0].GetLastTriggeredTimestamp() == ts2);
        REQUIRE(after[0].GetPolicyId() == originalPolicyId);
        REQUIRE(after[0].GetCreatedAt() == originalCreatedAt);

        REQUIRE(registry.UpdatePolicyChannels(fieldId, originalPolicyId, DCGM_POLICY_CHANNEL_FILE) == DCGM_ST_OK);
        after = GetPoliciesForField(registry, fieldId);
        REQUIRE(after[0].GetViolationCount() == 2);
        REQUIRE(after[0].GetLastTriggeredTimestamp() == ts2);
        REQUIRE(after[0].GetPolicyId() == originalPolicyId);
        REQUIRE(after[0].GetCreatedAt() == originalCreatedAt);

        REQUIRE(registry.SetPolicyState(fieldId, originalPolicyId, DCGM_POLICY_STATE_DISABLED) == DCGM_ST_OK);
        after = GetPoliciesForField(registry, fieldId);
        REQUIRE(after[0].GetViolationCount() == 2);
        REQUIRE(after[0].GetLastTriggeredTimestamp() == ts2);
        REQUIRE(after[0].GetPolicyId() == originalPolicyId);
        REQUIRE(after[0].GetCreatedAt() == originalCreatedAt);

        REQUIRE(registry.UpdatePolicyRateLimit(fieldId, originalPolicyId, 30) == DCGM_ST_OK);
        after = GetPoliciesForField(registry, fieldId);
        REQUIRE(after[0].GetViolationCount() == 2);
        REQUIRE(after[0].GetLastTriggeredTimestamp() == ts2);
        REQUIRE(after[0].GetPolicyId() == originalPolicyId);
        REQUIRE(after[0].GetCreatedAt() == originalCreatedAt);
    }
}


TEST_CASE("DcgmPolicyRegistry: AddPolicies")
{
    SECTION("Adds all policies when registry is empty")
    {
        DcgmPolicyRegistry registry;
        std::vector<DcgmPolicy> batch = {
            DcgmPolicy(150,
                       "temp",
                       0.0f,
                       80.0f,
                       DCGM_POLICY_OP_GT,
                       { { DCGM_FE_GPU, 0 } },
                       DCGM_POLICY_CHANNEL_CONSOLE,
                       DCGM_POLICY_STATE_ENABLED,
                       10),
            DcgmPolicy(155,
                       "power",
                       0.0f,
                       250.0f,
                       DCGM_POLICY_OP_GT,
                       { { DCGM_FE_GPU, 1 } },
                       DCGM_POLICY_CHANNEL_FILE,
                       DCGM_POLICY_STATE_ENABLED,
                       5),
            DcgmPolicy(160,
                       "memory",
                       0.0f,
                       90.0f,
                       DCGM_POLICY_OP_LT,
                       { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                       DCGM_POLICY_CHANNEL_CALLBACK,
                       DCGM_POLICY_STATE_DISABLED,
                       15),
        };

        REQUIRE(registry.AddPolicies(batch.data(), batch.data() + batch.size()) == DCGM_ST_OK);
        REQUIRE(RegistryPolicyCount(registry) == 3);
        REQUIRE(registry.HasPolicy(150));
        REQUIRE(registry.HasPolicy(155));
        REQUIRE(registry.HasPolicy(160));
    }

    SECTION("Rejects the entire batch when any fieldId already exists")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy existing(150,
                            "existing",
                            0.0f,
                            80.0f,
                            DCGM_POLICY_OP_GT,
                            { { DCGM_FE_GPU, 0 } },
                            DCGM_POLICY_CHANNEL_CONSOLE,
                            DCGM_POLICY_STATE_ENABLED,
                            10);
        uint64_t const existingPid          = existing.GetPolicyId();
        timelib64_t const existingCreatedAt = existing.GetCreatedAt();
        registry.AddPolicy(existing);

        std::vector<DcgmPolicy> batch = {
            DcgmPolicy(150,
                       "duplicate",
                       0.0f,
                       99.0f,
                       DCGM_POLICY_OP_LT,
                       { { DCGM_FE_GPU, 1 } },
                       DCGM_POLICY_CHANNEL_FILE,
                       DCGM_POLICY_STATE_DISABLED,
                       5),
            DcgmPolicy(155,
                       "new",
                       0.0f,
                       250.0f,
                       DCGM_POLICY_OP_GT,
                       { { DCGM_FE_GPU, 2 } },
                       DCGM_POLICY_CHANNEL_FILE,
                       DCGM_POLICY_STATE_ENABLED,
                       5),
        };

        REQUIRE(registry.AddPolicies(batch.data(), batch.data() + batch.size()) == DCGM_ST_DUPLICATE_KEY);
        REQUIRE(RegistryPolicyCount(registry) == 1);

        // Original policy for field 150 must be unchanged (identity preserved, not clobbered)
        auto stored = GetPoliciesForField(registry, 150);
        REQUIRE(stored[0].GetName() == "existing");
        REQUIRE(stored[0].GetThreshold() == 80.0f);
        REQUIRE(stored[0].GetPolicyId() == existingPid);
        REQUIRE(stored[0].GetCreatedAt() == existingCreatedAt);

        // No partial insert: field 155 was not added because field 150 collided.
        REQUIRE_FALSE(registry.HasPolicy(155));
    }

    SECTION("Rejects the entire batch when it contains duplicate fieldIds")
    {
        DcgmPolicyRegistry registry;
        std::vector<DcgmPolicy> batch = {
            DcgmPolicy(150,
                       "first",
                       0.0f,
                       80.0f,
                       DCGM_POLICY_OP_GT,
                       { { DCGM_FE_GPU, 0 } },
                       DCGM_POLICY_CHANNEL_CONSOLE,
                       DCGM_POLICY_STATE_ENABLED,
                       10),
            DcgmPolicy(150,
                       "second",
                       0.0f,
                       90.0f,
                       DCGM_POLICY_OP_LT,
                       { { DCGM_FE_GPU, 1 } },
                       DCGM_POLICY_CHANNEL_FILE,
                       DCGM_POLICY_STATE_DISABLED,
                       5),
            DcgmPolicy(155,
                       "other",
                       0.0f,
                       70.0f,
                       DCGM_POLICY_OP_EQ,
                       { { DCGM_FE_GPU, 2 } },
                       DCGM_POLICY_CHANNEL_CALLBACK,
                       DCGM_POLICY_STATE_ENABLED,
                       15),
        };

        REQUIRE(registry.AddPolicies(batch.data(), batch.data() + batch.size()) == DCGM_ST_DUPLICATE_KEY);
        REQUIRE(RegistryPolicyCount(registry) == 0);
        REQUIRE_FALSE(registry.HasPolicy(150));
        REQUIRE_FALSE(registry.HasPolicy(155));
    }

    SECTION("Empty batch succeeds with no changes")
    {
        DcgmPolicyRegistry registry;
        registry.AddPolicy(DcgmPolicy(150,
                                      "existing",
                                      0.0f,
                                      80.0f,
                                      DCGM_POLICY_OP_GT,
                                      { { DCGM_FE_GPU, 0 } },
                                      DCGM_POLICY_CHANNEL_CONSOLE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      10));

        std::vector<DcgmPolicy> empty;
        REQUIRE(registry.AddPolicies(empty.data(), empty.data() + empty.size()) == DCGM_ST_OK);
        REQUIRE(RegistryPolicyCount(registry) == 1);
    }

    SECTION("RemoveAllPolicies then AddPolicies atomically replaces registry contents")
    {
        DcgmPolicyRegistry registry;
        registry.AddPolicy(DcgmPolicy(150,
                                      "old",
                                      0.0f,
                                      80.0f,
                                      DCGM_POLICY_OP_GT,
                                      { { DCGM_FE_GPU, 0 } },
                                      DCGM_POLICY_CHANNEL_CONSOLE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      10));
        registry.AddPolicy(DcgmPolicy(155,
                                      "old2",
                                      0.0f,
                                      90.0f,
                                      DCGM_POLICY_OP_GT,
                                      { { DCGM_FE_GPU, 1 } },
                                      DCGM_POLICY_CHANNEL_FILE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      5));

        std::vector<DcgmPolicy> newPolicies = {
            DcgmPolicy(160,
                       "new1",
                       0.0f,
                       70.0f,
                       DCGM_POLICY_OP_LT,
                       { { DCGM_FE_GPU, 2 } },
                       DCGM_POLICY_CHANNEL_CALLBACK,
                       DCGM_POLICY_STATE_ENABLED,
                       20),
            DcgmPolicy(165,
                       "new2",
                       0.0f,
                       60.0f,
                       DCGM_POLICY_OP_EQ,
                       { { DCGM_FE_GPU, 3 } },
                       DCGM_POLICY_CHANNEL_CONSOLE,
                       DCGM_POLICY_STATE_DISABLED,
                       30),
        };

        registry.RemoveAllPolicies();
        REQUIRE(registry.AddPolicies(newPolicies.data(), newPolicies.data() + newPolicies.size()) == DCGM_ST_OK);

        REQUIRE(RegistryPolicyCount(registry) == 2);
        REQUIRE_FALSE(registry.HasPolicy(150));
        REQUIRE_FALSE(registry.HasPolicy(155));
        REQUIRE(registry.HasPolicy(160));
        REQUIRE(registry.HasPolicy(165));

        auto stored = GetPoliciesForField(registry, 160);
        REQUIRE(stored[0].GetName() == "new1");
        REQUIRE(stored[0].GetThreshold() == 70.0f);
    }
}

TEST_CASE("DcgmPolicyRegistry: HasPolicy")
{
    SECTION("Returns false on empty registry")
    {
        DcgmPolicyRegistry registry;
        REQUIRE_FALSE(registry.HasPolicy(150));
    }

    SECTION("Tracks presence through add and remove")
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
        uint64_t const pid = policy.GetPolicyId();
        REQUIRE(registry.AddPolicy(policy) == DCGM_ST_OK);
        REQUIRE(registry.HasPolicy(150));
        REQUIRE_FALSE(registry.HasPolicy(151));
        REQUIRE(registry.RemovePolicy(150, pid) == DCGM_ST_OK);
        REQUIRE_FALSE(registry.HasPolicy(150));
    }
}

TEST_CASE("DcgmPolicyRegistry: GetPoliciesForField")
{
    SECTION("Returns empty for unknown fieldId (no side effects)")
    {
        DcgmPolicyRegistry registry;
        REQUIRE(GetPoliciesForField(registry, 150).empty());
        REQUIRE_FALSE(registry.HasPolicy(150));
        REQUIRE(RegistryPolicyCount(registry) == 0);
    }

    SECTION("Returns a single policy when present (1 policy per field today)")
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
        REQUIRE(registry.AddPolicy(policy) == DCGM_ST_OK);
        auto stored = GetPoliciesForField(registry, 150);
        REQUIRE(stored.size() == 1);
        REQUIRE(stored[0].GetFieldId() == 150);
        REQUIRE(stored[0].GetPolicyId() == policy.GetPolicyId());
    }

    SECTION("Does not return policies from other fields")
    {
        DcgmPolicyRegistry registry;
        registry.AddPolicy(DcgmPolicy(150,
                                      "a",
                                      0.0f,
                                      80.0f,
                                      DCGM_POLICY_OP_GT,
                                      { { DCGM_FE_GPU, 0 } },
                                      DCGM_POLICY_CHANNEL_CONSOLE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      10));
        registry.AddPolicy(DcgmPolicy(155,
                                      "b",
                                      0.0f,
                                      90.0f,
                                      DCGM_POLICY_OP_LT,
                                      { { DCGM_FE_GPU, 1 } },
                                      DCGM_POLICY_CHANNEL_FILE,
                                      DCGM_POLICY_STATE_ENABLED,
                                      5));

        auto stored = GetPoliciesForField(registry, 150);
        REQUIRE(stored.size() == 1);
        REQUIRE(stored[0].GetFieldId() == 150);
    }
}
