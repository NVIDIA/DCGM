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
#include <DcgmPolicyEvaluator.h>
#include <DcgmPolicyRegistry.h>
#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

namespace
{

/**
 * Helper function to evaluate a policy for a specific GPU and return violations
 */
std::vector<dcgm_policy_violation_t> EvaluateAndGetViolations(DcgmPolicyEvaluator &evaluator,
                                                              unsigned short fieldId,
                                                              int gpuId,
                                                              double value)
{
    std::vector<dcgm_policy_violation_t> violations;
    evaluator.EvaluatePolicyForEntity(
        fieldId, { DCGM_FE_GPU, static_cast<dcgm_field_eid_t>(gpuId) }, value, violations);
    return violations;
}

/**
 * Test data structure for operator tests
 */
struct OperatorTestCase
{
    dcgmPolicyOperator_t op;
    double threshold;
    double testValue;
    bool expectViolation;
    std::string description;
};

} // anonymous namespace

TEST_CASE("DcgmPolicyEvaluator: Basic Operations")
{
    SECTION("Non-existent policy returns no violations")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicyEvaluator evaluator(registry);

        auto violations = EvaluateAndGetViolations(evaluator, 150, 0, 85.0);

        REQUIRE(violations.empty());
    }

    SECTION("Disabled policies are not evaluated")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(150,
                                 "disabled_policy",
                                 0.0,
                                 80.0,
                                 DCGM_POLICY_OP_GT,
                                 std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 } },
                                 DCGM_POLICY_CHANNEL_CONSOLE,
                                 DCGM_POLICY_STATE_DISABLED,
                                 60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);
        auto violations = EvaluateAndGetViolations(evaluator, 150, 0, 90.0);

        REQUIRE(violations.empty());
    }
}

TEST_CASE("DcgmPolicyEvaluator: Comparison Operators")
{
    SECTION("All operators correctly detect violations and non-violations")
    {
        // Comprehensive test data for all operators
        std::vector<OperatorTestCase> const testCases = {
            // GREATER_THAN operator
            { DCGM_POLICY_OP_GT, 80.0, 85.0, true, "GT: value > threshold" },
            { DCGM_POLICY_OP_GT, 80.0, 80.0, false, "GT: value == threshold" },
            { DCGM_POLICY_OP_GT, 80.0, 75.0, false, "GT: value < threshold" },

            // GREATER_THAN_OR_EQUAL operator
            { DCGM_POLICY_OP_GE, 80.0, 85.0, true, "GTE: value > threshold" },
            { DCGM_POLICY_OP_GE, 80.0, 80.0, true, "GTE: value == threshold" },
            { DCGM_POLICY_OP_GE, 80.0, 75.0, false, "GTE: value < threshold" },

            // LESS_THAN operator
            { DCGM_POLICY_OP_LT, 80.0, 75.0, true, "LT: value < threshold" },
            { DCGM_POLICY_OP_LT, 80.0, 80.0, false, "LT: value == threshold" },
            { DCGM_POLICY_OP_LT, 80.0, 85.0, false, "LT: value > threshold" },

            // LESS_THAN_OR_EQUAL operator
            { DCGM_POLICY_OP_LE, 80.0, 75.0, true, "LTE: value < threshold" },
            { DCGM_POLICY_OP_LE, 80.0, 80.0, true, "LTE: value == threshold" },
            { DCGM_POLICY_OP_LE, 80.0, 85.0, false, "LTE: value > threshold" },

            // EQUAL operator
            { DCGM_POLICY_OP_EQ, 80.0, 80.0, true, "EQ: value == threshold" },
            { DCGM_POLICY_OP_EQ, 80.0, 79.0, false, "EQ: value < threshold" },
            { DCGM_POLICY_OP_EQ, 80.0, 81.0, false, "EQ: value > threshold" },

            // NOT_EQUAL operator
            { DCGM_POLICY_OP_NE, 80.0, 79.0, true, "NE: value != threshold (less)" },
            { DCGM_POLICY_OP_NE, 80.0, 81.0, true, "NE: value != threshold (greater)" },
            { DCGM_POLICY_OP_NE, 80.0, 80.0, false, "NE: value == threshold" },
        };

        for (auto const &testCase : testCases)
        {
            DYNAMIC_SECTION(testCase.description)
            {
                DcgmPolicyRegistry registry;
                auto policy = DcgmPolicy(150,
                                         "test_policy",
                                         0.0,
                                         testCase.threshold,
                                         testCase.op,
                                         std::vector<dcgmGroupEntityPair_t> {},
                                         DCGM_POLICY_CHANNEL_CONSOLE,
                                         DCGM_POLICY_STATE_ENABLED,
                                         60);
                registry.AddPolicy(policy);

                DcgmPolicyEvaluator evaluator(registry);
                auto violations = EvaluateAndGetViolations(evaluator, 150, 0, testCase.testValue);

                if (testCase.expectViolation)
                {
                    REQUIRE(violations.size() == 1);
                    REQUIRE(violations[0].fieldId == 150);
                    REQUIRE(violations[0].currentValue == testCase.testValue);
                    REQUIRE(violations[0].thresholdValue == testCase.threshold);
                    REQUIRE(violations[0].policyOperator == testCase.op);
                }
                else
                {
                    REQUIRE(violations.empty());
                }
            }
        }
    }
}

TEST_CASE("DcgmPolicyEvaluator: CHANGED Operator")
{
    SECTION("Value change detection")
    {
        struct ChangedTestCase
        {
            double baselineValue;
            double testValue;
            bool expectViolation;
            std::string description;
        };

        std::vector<ChangedTestCase> const testCases = {
            { 75.0, 75.0, false, "Same value does not trigger" },
            { 75.0, 80.0, true, "Different value triggers" },
            { 0.0, 0.0, false, "Zero baseline, same value does not trigger" },
            { 0.0, 1.0, true, "Zero baseline, non-zero value triggers" },
        };

        for (auto const &testCase : testCases)
        {
            DYNAMIC_SECTION(testCase.description)
            {
                DcgmPolicyRegistry registry;
                auto policy = DcgmPolicy(150,
                                         "changed_policy",
                                         0.0,
                                         0.0,
                                         DCGM_POLICY_OP_CHANGED,
                                         std::vector<dcgmGroupEntityPair_t> {},
                                         DCGM_POLICY_CHANNEL_CONSOLE,
                                         DCGM_POLICY_STATE_ENABLED,
                                         60);
                registry.AddPolicy(policy);

                DcgmPolicyEvaluator evaluator(registry);

                // First evaluation always establishes baseline - never triggers
                REQUIRE(EvaluateAndGetViolations(evaluator, 150, 0, testCase.baselineValue).empty());

                auto violations = EvaluateAndGetViolations(evaluator, 150, 0, testCase.testValue);
                if (testCase.expectViolation)
                {
                    REQUIRE(violations.size() == 1);
                    REQUIRE(violations[0].currentValue == testCase.testValue);
                    REQUIRE(violations[0].policyOperator == DCGM_POLICY_OP_CHANGED);
                }
                else
                {
                    REQUIRE(violations.empty());
                }
            }
        }
    }

    SECTION("Previous values tracked per field and GPU")
    {
        DcgmPolicyRegistry registry;

        auto policy1 = DcgmPolicy(150,
                                  "field150",
                                  0.0,
                                  0.0,
                                  DCGM_POLICY_OP_CHANGED,
                                  std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                                  DCGM_POLICY_CHANNEL_CONSOLE,
                                  DCGM_POLICY_STATE_ENABLED,
                                  60);
        auto policy2 = DcgmPolicy(200,
                                  "field200",
                                  0.0,
                                  0.0,
                                  DCGM_POLICY_OP_CHANGED,
                                  std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                                  DCGM_POLICY_CHANNEL_CONSOLE,
                                  DCGM_POLICY_STATE_ENABLED,
                                  60);

        registry.AddPolicy(policy1);
        registry.AddPolicy(policy2);

        DcgmPolicyEvaluator evaluator(registry);

        // Establish baseline for field 150 GPU 0
        EvaluateAndGetViolations(evaluator, 150, 0, 75.0);

        // Same value - no trigger
        auto violations = EvaluateAndGetViolations(evaluator, 150, 0, 75.0);
        REQUIRE(violations.empty());

        // Changed value for field 150 GPU 0 - triggers
        violations = EvaluateAndGetViolations(evaluator, 150, 0, 80.0);
        REQUIRE(violations.size() == 1);

        // First evaluation for field 150 GPU 1 - establishes baseline, no trigger
        violations = EvaluateAndGetViolations(evaluator, 150, 1, 75.0);
        REQUIRE(violations.empty());

        // Changed value for GPU 1 - triggers (per-GPU tracking)
        violations = EvaluateAndGetViolations(evaluator, 150, 1, 80.0);
        REQUIRE(violations.size() == 1);

        // First evaluation for field 200 GPU 0 - establishes baseline, no trigger
        violations = EvaluateAndGetViolations(evaluator, 200, 0, 100.0);
        REQUIRE(violations.empty());

        // Changed value for field 200 - triggers (per-field tracking)
        violations = EvaluateAndGetViolations(evaluator, 200, 0, 200.0);
        REQUIRE(violations.size() == 1);
    }
}

TEST_CASE("DcgmPolicyEvaluator: GPU Handling")
{
    SECTION("Multiple GPU IDs - each GPU evaluated separately")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(
            150,
            "multi_gpu",
            0.0,
            80.0,
            DCGM_POLICY_OP_GT,
            std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 }, { DCGM_FE_GPU, 2 } },
            DCGM_POLICY_CHANNEL_CONSOLE,
            DCGM_POLICY_STATE_ENABLED,
            60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);

        // Evaluate GPU 0
        auto violations = EvaluateAndGetViolations(evaluator, 150, 0, 85.0);
        REQUIRE(violations.size() == 1);
        REQUIRE(violations[0].entity.entityId == 0);

        // Evaluate GPU 1
        violations = EvaluateAndGetViolations(evaluator, 150, 1, 85.0);
        REQUIRE(violations.size() == 1);
        REQUIRE(violations[0].entity.entityId == 1);

        // Evaluate GPU 2
        violations = EvaluateAndGetViolations(evaluator, 150, 2, 85.0);
        REQUIRE(violations.size() == 1);
        REQUIRE(violations[0].entity.entityId == 2);
    }

    SECTION("Empty GPU list accepts any GPU")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(150,
                                 "all_gpus",
                                 0.0,
                                 80.0,
                                 DCGM_POLICY_OP_GT,
                                 std::vector<dcgmGroupEntityPair_t> {},
                                 DCGM_POLICY_CHANNEL_CONSOLE,
                                 DCGM_POLICY_STATE_ENABLED,
                                 60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);

        // Any GPU should be accepted
        auto violations = EvaluateAndGetViolations(evaluator, 150, 5, 85.0);
        REQUIRE(violations.size() == 1);
        REQUIRE(violations[0].entity.entityId == 5);

        violations = EvaluateAndGetViolations(evaluator, 150, 10, 85.0);
        REQUIRE(violations.size() == 1);
        REQUIRE(violations[0].entity.entityId == 10);
    }

    SECTION("GPU not in policy list is rejected")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(150,
                                 "specific_gpus",
                                 0.0,
                                 80.0,
                                 DCGM_POLICY_OP_GT,
                                 std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                                 DCGM_POLICY_CHANNEL_CONSOLE,
                                 DCGM_POLICY_STATE_ENABLED,
                                 60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);

        // GPU 5 not in policy's list {0, 1} - should return no violations
        auto violations = EvaluateAndGetViolations(evaluator, 150, 5, 85.0);
        REQUIRE(violations.empty());

        // GPU 0 is in list - should return violation
        violations = EvaluateAndGetViolations(evaluator, 150, 0, 85.0);
        REQUIRE(violations.size() == 1);
    }
}

TEST_CASE("DcgmPolicyEvaluator: Multiple Policies")
{
    SECTION("Multiple policies evaluated independently")
    {
        DcgmPolicyRegistry registry;

        auto policy1 = DcgmPolicy(150,
                                  "temp_policy",
                                  0.0,
                                  80.0,
                                  DCGM_POLICY_OP_GT,
                                  std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 } },
                                  DCGM_POLICY_CHANNEL_CONSOLE,
                                  DCGM_POLICY_STATE_ENABLED,
                                  60);
        auto policy2 = DcgmPolicy(200,
                                  "power_policy",
                                  0.0,
                                  250.0,
                                  DCGM_POLICY_OP_LT,
                                  std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 1 } },
                                  DCGM_POLICY_CHANNEL_CONSOLE,
                                  DCGM_POLICY_STATE_ENABLED,
                                  60);
        auto policy3 = DcgmPolicy(300,
                                  "mem_policy",
                                  0.0,
                                  100.0,
                                  DCGM_POLICY_OP_EQ,
                                  std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 2 } },
                                  DCGM_POLICY_CHANNEL_CONSOLE,
                                  DCGM_POLICY_STATE_ENABLED,
                                  60);

        registry.AddPolicy(policy1);
        registry.AddPolicy(policy2);
        registry.AddPolicy(policy3);

        DcgmPolicyEvaluator evaluator(registry);

        // Evaluate each policy/GPU combination
        auto v1 = EvaluateAndGetViolations(evaluator, 150, 0, 85.0);  // Violation (85 > 80)
        auto v2 = EvaluateAndGetViolations(evaluator, 200, 1, 240.0); // Violation (240 < 250)
        auto v3 = EvaluateAndGetViolations(evaluator, 300, 2, 90.0);  // No violation (90 != 100)

        REQUIRE(v1.size() == 1);
        REQUIRE(v1[0].fieldId == 150);
        REQUIRE(v2.size() == 1);
        REQUIRE(v2[0].fieldId == 200);
        REQUIRE(v3.empty());
    }
}

TEST_CASE("DcgmPolicyEvaluator: Violation Structure")
{
    SECTION("Violation structure contains correct data")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicy policy(static_cast<unsigned short>(150),
                          "test_policy",
                          0.0,
                          80.0,
                          DCGM_POLICY_OP_GT,
                          std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 5 } },
                          DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_FILE,
                          DCGM_POLICY_STATE_ENABLED,
                          10);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);
        auto violations = EvaluateAndGetViolations(evaluator, 150, 5, 90.0);

        REQUIRE(violations.size() == 1);

        auto const &v = violations[0];
        REQUIRE(v.policyId == policy.GetPolicyId());
        REQUIRE(std::string(v.policyName) == "test_policy");
        REQUIRE(v.fieldId == 150);
        REQUIRE(v.entity.entityId == 5);
        REQUIRE(v.currentValue == 90.0);
        REQUIRE(v.thresholdValue == 80.0);
        REQUIRE(v.policyOperator == DCGM_POLICY_OP_GT);
        REQUIRE(__builtin_popcount(v.channelMask) == 2);
        REQUIRE((v.channelMask & DCGM_POLICY_CHANNEL_CONSOLE) != 0u);
        REQUIRE((v.channelMask & DCGM_POLICY_CHANNEL_FILE) != 0u);
        REQUIRE(v.timestamp > 0);
    }
}

TEST_CASE("DcgmPolicyEvaluator: Return Values")
{
    SECTION("Non-existent policy returns DCGM_ST_NO_DATA")
    {
        DcgmPolicyRegistry registry;
        DcgmPolicyEvaluator evaluator(registry);

        std::vector<dcgm_policy_violation_t> violations;
        dcgmReturn_t result = evaluator.EvaluatePolicyForEntity(999, { DCGM_FE_GPU, 0 }, 85.0, violations);

        REQUIRE(result == DCGM_ST_NO_DATA);
        REQUIRE(violations.empty());
    }

    SECTION("Disabled policy returns DCGM_ST_NOT_CONFIGURED")
    {
        DcgmPolicyRegistry registry;
        auto disabledPolicy = DcgmPolicy(300,
                                         "disabled",
                                         0.0,
                                         50.0,
                                         DCGM_POLICY_OP_GT,
                                         std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 } },
                                         DCGM_POLICY_CHANNEL_CONSOLE,
                                         DCGM_POLICY_STATE_DISABLED,
                                         60);
        registry.AddPolicy(disabledPolicy);

        DcgmPolicyEvaluator evaluator(registry);

        std::vector<dcgm_policy_violation_t> violations;
        dcgmReturn_t result = evaluator.EvaluatePolicyForEntity(300, { DCGM_FE_GPU, 0 }, 60.0, violations);

        REQUIRE(result == DCGM_ST_NOT_CONFIGURED);
        REQUIRE(violations.empty());
    }

    SECTION("Policy with no violation returns DCGM_ST_NO_DATA")
    {
        DcgmPolicyRegistry registry;
        auto noViolationPolicy = DcgmPolicy(400,
                                            "no_viol",
                                            0.0,
                                            100.0,
                                            DCGM_POLICY_OP_GT,
                                            std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 } },
                                            DCGM_POLICY_CHANNEL_CONSOLE,
                                            DCGM_POLICY_STATE_ENABLED,
                                            60);
        registry.AddPolicy(noViolationPolicy);

        DcgmPolicyEvaluator evaluator(registry);

        std::vector<dcgm_policy_violation_t> violations;
        dcgmReturn_t result
            = evaluator.EvaluatePolicyForEntity(400, { DCGM_FE_GPU, 0 }, 50.0, violations); // Below threshold

        REQUIRE(result == DCGM_ST_NO_DATA);
        REQUIRE(violations.empty());
    }

    SECTION("GPU not in policy list returns DCGM_ST_NO_DATA")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(500,
                                 "specific_gpus",
                                 0.0,
                                 80.0,
                                 DCGM_POLICY_OP_GT,
                                 std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 }, { DCGM_FE_GPU, 1 } },
                                 DCGM_POLICY_CHANNEL_CONSOLE,
                                 DCGM_POLICY_STATE_ENABLED,
                                 60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);

        std::vector<dcgm_policy_violation_t> violations;
        dcgmReturn_t result
            = evaluator.EvaluatePolicyForEntity(500, { DCGM_FE_GPU, 5 }, 90.0, violations); // GPU 5 not in {0, 1}

        REQUIRE(result == DCGM_ST_NO_DATA);
        REQUIRE(violations.empty());
    }

    SECTION("Violation returns DCGM_ST_OK")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(600,
                                 "violation",
                                 0.0,
                                 80.0,
                                 DCGM_POLICY_OP_GT,
                                 std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 } },
                                 DCGM_POLICY_CHANNEL_CONSOLE,
                                 DCGM_POLICY_STATE_ENABLED,
                                 60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);

        std::vector<dcgm_policy_violation_t> violations;
        dcgmReturn_t result = evaluator.EvaluatePolicyForEntity(600, { DCGM_FE_GPU, 0 }, 90.0, violations);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(violations.size() == 1);
    }
}

TEST_CASE("DcgmPolicyEvaluator: ClearFieldState")
{
    SECTION("ClearFieldState resets previous value tracking")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(150,
                                 "changed_policy",
                                 0.0,
                                 0.0,
                                 DCGM_POLICY_OP_CHANGED,
                                 std::vector<dcgmGroupEntityPair_t> {},
                                 DCGM_POLICY_CHANNEL_CONSOLE,
                                 DCGM_POLICY_STATE_ENABLED,
                                 60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);

        // First evaluation establishes baseline - no trigger
        auto violations = EvaluateAndGetViolations(evaluator, 150, 0, 75.0);
        REQUIRE(violations.empty());

        // Changed value - triggers
        violations = EvaluateAndGetViolations(evaluator, 150, 0, 80.0);
        REQUIRE(violations.size() == 1);

        // Same value - no trigger
        violations = EvaluateAndGetViolations(evaluator, 150, 0, 80.0);
        REQUIRE(violations.empty());

        // Clear state
        evaluator.ClearFieldState(150);

        // Same value after clear - re-establishes baseline, no trigger
        violations = EvaluateAndGetViolations(evaluator, 150, 0, 80.0);
        REQUIRE(violations.empty());

        // Changed value after clear - triggers
        violations = EvaluateAndGetViolations(evaluator, 150, 0, 90.0);
        REQUIRE(violations.size() == 1);
    }
}

TEST_CASE("DcgmPolicyEvaluator: No Rate Limiting or Duplicate Suppression")
{
    // Note: Rate limiting and duplicate suppression are handled by NotificationRouter,
    // not by DcgmPolicyEvaluator. These tests verify that Evaluator does NOT filter.

    SECTION("Same violation value triggers every time (no dup suppression in Evaluator)")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(150,
                                 "no_dup_suppress",
                                 0.0,
                                 80.0,
                                 DCGM_POLICY_OP_GT,
                                 std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 } },
                                 DCGM_POLICY_CHANNEL_CONSOLE,
                                 DCGM_POLICY_STATE_ENABLED,
                                 60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);

        // Every evaluation with violating value should trigger
        auto v1 = EvaluateAndGetViolations(evaluator, 150, 0, 85.0);
        auto v2 = EvaluateAndGetViolations(evaluator, 150, 0, 85.0);
        auto v3 = EvaluateAndGetViolations(evaluator, 150, 0, 85.0);

        REQUIRE(v1.size() == 1);
        REQUIRE(v2.size() == 1);
        REQUIRE(v3.size() == 1);
    }

    SECTION("Rapid evaluations all trigger (no rate limiting in Evaluator)")
    {
        DcgmPolicyRegistry registry;
        auto policy = DcgmPolicy(150,
                                 "no_rate_limit",
                                 0.0,
                                 80.0,
                                 DCGM_POLICY_OP_GT,
                                 std::vector<dcgmGroupEntityPair_t> { { DCGM_FE_GPU, 0 } },
                                 DCGM_POLICY_CHANNEL_CONSOLE,
                                 DCGM_POLICY_STATE_ENABLED,
                                 60);
        registry.AddPolicy(policy);

        DcgmPolicyEvaluator evaluator(registry);

        // Rapid evaluations should all trigger
        for (int i = 0; i < 10; ++i)
        {
            auto violations = EvaluateAndGetViolations(evaluator, 150, 0, 85.0 + i);
            REQUIRE(violations.size() == 1);
        }
    }
}
