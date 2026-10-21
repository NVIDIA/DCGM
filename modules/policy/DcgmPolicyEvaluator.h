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

#ifndef DCGMPOLICYEVALUATOR_H
#define DCGMPOLICYEVALUATOR_H

#include "DcgmPolicyRegistry.h"
#include "dcgm_policy_structs.h"
#include <dcgm_structs.h>
#include <map>
#include <timelib.h>
#include <vector>

/**
 * Policy Evaluator - Field policy comparison engine
 *
 * Evaluates field value updates against configured policy thresholds to determine
 * if violations have occurred. Operates generically on any DCGM field with any
 * comparison operator.
 *
 * This is a near-stateless evaluation engine. The only state maintained is for
 * the CHANGED operator which requires tracking previous values per (fieldId, entity).
 * That state is runtime evaluation state, not policy configuration.
 *
 * Ownership note: evaluation is expected to run on the serialized policy-module
 * worker.
 *
 * Rate limiting and duplicate suppression are handled by NotificationRouter,
 * not by this class.
 */
class DcgmPolicyEvaluator
{
public:
    /**
     * Constructs a new DcgmPolicyEvaluator instance
     *
     * @param[in] policyRegistry Reference to the policy registry for querying policies
     */
    explicit DcgmPolicyEvaluator(DcgmPolicyRegistry &policyRegistry);

    ~DcgmPolicyEvaluator() = default;

    // Non-copyable
    DcgmPolicyEvaluator(DcgmPolicyEvaluator const &)            = delete;
    DcgmPolicyEvaluator &operator=(DcgmPolicyEvaluator const &) = delete;

    /**
     * Evaluates live field data for a specific entity, on a single policy.
     *
     * Performs policy comparison without rate limiting or duplicate suppression.
     * Those responsibilities belong to NotificationRouter.
     *
     * @note This is the cache-manager/live-sample path. For DCGM_POLICY_OP_CHANGED,
     *       this method mutates the previous-value baseline after comparison. Do not
     *       use it for read-only user/API queries that should not affect future
     *       CHANGED notifications.
     *
     * @param[in] fieldId The field ID of the policy to evaluate
     * @param[in] entity The specific entity to evaluate.
     * @param[in] value The current field value for this entity.
     * @param[out] violations Vector to append any detected violations
     *
     * @return DCGM_ST_OK if violation detected
     * @return DCGM_ST_NO_DATA if no policy exists, entity not in policy, or no violation
     * @return DCGM_ST_NOT_CONFIGURED if policy is disabled
     */
    dcgmReturn_t EvaluatePolicyForEntity(unsigned short fieldId,
                                         dcgmGroupEntityPair_t entity,
                                         double value,
                                         std::vector<dcgm_policy_violation_t> &violations);

    /**
     * Clears tracking state for a specific field
     *
     * Should be called from the policy delete path to clean up previous value tracking
     * for DCGM_POLICY_OP_CHANGED.
     *
     * @param[in] fieldId Field ID to clear state for
     */
    void ClearFieldState(unsigned short fieldId);

private:
    /**
     * Checks if a value violates a policy threshold.
     *
     * Handles all operators including DCGM_POLICY_OP_CHANGED. For CHANGED on the live path,
     * compares against the stored previous value and updates it unconditionally before returning.
     *
     * @param[in] fieldId Field ID
     * @param[in] entity Entity being evaluated.
     * @param[in] currentValue Current field value
     * @param[in] threshold Policy threshold value
     * @param[in] op Comparison operator
     *
     * @return true if violation detected, false otherwise
     */
    bool CheckViolation(unsigned short fieldId,
                        dcgmGroupEntityPair_t entity,
                        double currentValue,
                        double threshold,
                        dcgmPolicyOperator_t op);

    /**
     * Evaluates a policy for a specific entity with a given value.
     *
     * @param[in] policy The policy to evaluate
     * @param[in] entity The entity to evaluate
     * @param[in] value The current value for the entity
     * @param[out] violations Vector to append any detected violations
     *
     * @return DCGM_ST_OK if violation detected
     */
    dcgmReturn_t EvaluatePolicy(DcgmPolicy const &policy,
                                dcgmGroupEntityPair_t entity,
                                double value,
                                std::vector<dcgm_policy_violation_t> &violations);

    /**
     * Creates a violation record from a policy and current values
     *
     * @param[in] policy The policy that was violated
     * @param[in] entity The entity that violated the policy
     * @param[in] value The current value that caused the violation
     *
     * @return Populated violation structure
     */
    dcgm_policy_violation_t CreateViolationRecord(DcgmPolicy const &policy, dcgmGroupEntityPair_t entity, double value);

    // The registry outlives this evaluator.
    DcgmPolicyRegistry *m_policyRegistry;

    // Tracks previous values for each field+entity (required for CHANGED operator)
    std::map<unsigned short, std::map<std::pair<dcgm_field_entity_group_t, dcgm_field_eid_t>, double>> m_previousValues;
};

#endif // DCGMPOLICYEVALUATOR_H
