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

#ifndef DCGMPOLICYREGISTRY_H
#define DCGMPOLICYREGISTRY_H

#include "DcgmPolicy.h"
#include "dcgm_policy_structs.h"
#include "dcgm_structs.h"

#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

class DcgmPolicyRegistry
{
public:
    DcgmPolicyRegistry();
    ~DcgmPolicyRegistry() = default;

    /**
     * Adds a new policy to the registry.
     *
     * @param[in] policy The policy to add
     *
     * @return DCGM_ST_OK on success, DCGM_ST_DUPLICATE_KEY if a policy for this fieldId already exists,
     *         DCGM_ST_BADPARAM if the registry is full
     */
    dcgmReturn_t AddPolicy(DcgmPolicy const &policy);

    /**
     * Adds a batch of policies atomically.
     *
     * Either every policy is inserted, or the registry is unchanged.
     *
     * @param[in] first Start of the policy range (may be null when first == last)
     * @param[in] last  One past the last policy
     *
     * @return DCGM_ST_OK if all policies were added, DCGM_ST_DUPLICATE_KEY if any fieldId
     *         is duplicated in the batch or already exists, DCGM_ST_BADPARAM if the batch
     *         would exceed DCGM_POLICY_MAX_POLICIES
     */
    dcgmReturn_t AddPolicies(DcgmPolicy *first, DcgmPolicy *last);

    /**
     * Atomically replace every registered policy with a validated batch.
     *
     * @param[in] first Start of the replacement policy range.
     * @param[in] last One past the end of the replacement policy range.
     * @return DCGM_ST_OK on success, or an error without changing the registry.
     */
    dcgmReturn_t ReplaceAllPolicies(DcgmPolicy *first, DcgmPolicy *last);

    /**
     * Removes the policy identified by (fieldId, policyId).
     *
     * @param[in] fieldId  Field ID the policy belongs to
     * @param[in] policyId The policy ID to remove
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no policy with this (fieldId, policyId)
     */
    dcgmReturn_t RemovePolicy(unsigned short fieldId, uint64_t policyId);

    /**
     * Removes all policies from the registry.
     *
     * Used for hot reload to clear registry before loading from file.
     *
     * @return DCGM_ST_OK on success
     */
    dcgmReturn_t RemoveAllPolicies();

    /**
     * Replaces all policies for a given field ID.
     *
     * Today exactly one policy per field is supported, so only the first element is applied.
     * When multiple policies per field are needed, this method will apply all replacements
     * with no change to callers.
     *
     * All policies in the list must carry the same fieldId as the lookup key.
     *
     * @param[in] fieldId  The field ID whose policies are being replaced
     * @param[in] policies The replacement policies (must be non-empty; first is used today)
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no policy exists for the field,
     *         DCGM_ST_BADPARAM if policies is empty or any policy has a mismatched fieldId
     */
    dcgmReturn_t ReplacePoliciesForField(unsigned short fieldId, std::vector<DcgmPolicy> policies);

    /**
     * Updates the current value for the policy identified by (fieldId, policyId).
     *
     * @param[in] fieldId  Field ID the policy belongs to
     * @param[in] policyId The policy ID
     * @param[in] value    The new current value
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no matching policy
     */
    dcgmReturn_t UpdatePolicyValue(unsigned short fieldId, uint64_t policyId, double value);

    /**
     * Updates the threshold for the policy identified by (fieldId, policyId).
     *
     * @param[in] fieldId   Field ID the policy belongs to
     * @param[in] policyId  The policy ID
     * @param[in] threshold The new threshold value
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no matching policy
     */
    dcgmReturn_t UpdatePolicyThreshold(unsigned short fieldId, uint64_t policyId, double threshold);

    /**
     * Updates the operator for the policy identified by (fieldId, policyId).
     *
     * @param[in] fieldId        Field ID the policy belongs to
     * @param[in] policyId       The policy ID
     * @param[in] policyOperator The new policy operator
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no matching policy
     */
    dcgmReturn_t UpdatePolicyOperator(unsigned short fieldId, uint64_t policyId, dcgmPolicyOperator_t policyOperator);

    /**
     * Updates the target entities for the policy identified by (fieldId, policyId).
     *
     * @param[in] fieldId  Field ID the policy belongs to
     * @param[in] policyId The policy ID
     * @param[in] entities The new vector of target entities.
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no matching policy
     */
    dcgmReturn_t UpdatePolicyEntities(unsigned short fieldId,
                                      uint64_t policyId,
                                      std::vector<dcgmGroupEntityPair_t> const &entities);

    /**
     * Updates the channel mask for the policy identified by (fieldId, policyId).
     *
     * @param[in] fieldId     Field ID the policy belongs to
     * @param[in] policyId    The policy ID
     * @param[in] channelMask Bitmask of dcgmPolicyChannel_t values
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no matching policy
     */
    dcgmReturn_t UpdatePolicyChannels(unsigned short fieldId, uint64_t policyId, unsigned int channelMask);

    /**
     * Sets the enabled/disabled state for the policy identified by (fieldId, policyId).
     *
     * @param[in] fieldId  Field ID the policy belongs to
     * @param[in] policyId The policy ID
     * @param[in] state    DCGM_POLICY_STATE_ENABLED or DCGM_POLICY_STATE_DISABLED
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no matching policy
     */
    dcgmReturn_t SetPolicyState(unsigned short fieldId, uint64_t policyId, dcgmPolicyState_t state);

    /**
     * Updates the rate limit for the policy identified by (fieldId, policyId).
     *
     * @param[in] fieldId   Field ID the policy belongs to
     * @param[in] policyId  The policy ID
     * @param[in] rateLimit The new rate limit value
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if no matching policy
     */
    dcgmReturn_t UpdatePolicyRateLimit(unsigned short fieldId, uint64_t policyId, unsigned int rateLimit);

    /**
     * Records a violation after validating it still matches the current policy and rate limit.
     *
     * @param[in] violation Violation produced by the evaluator.
     *
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if the policy no longer exists, no longer matches, or the
     *         notification is rate limited; DCGM_ST_NOT_CONFIGURED if the policy is disabled
     */
    dcgmReturn_t RecordViolation(dcgm_policy_violation_t const &violation);

    /**
     * Gets a policy by its policyId.
     *
     * @param[in] policyId The policy ID to look up
     *
     * @return Optional containing the policy if found, std::nullopt otherwise
     */
    [[nodiscard]] std::optional<DcgmPolicy> GetPolicyById(uint64_t policyId) const;

    /**
     * Checks if any policy exists for a given field ID.
     *
     * @param[in] fieldId The field ID to check
     *
     * @return True if a policy exists for the field, false otherwise
     */
    [[nodiscard]] bool HasPolicy(unsigned short fieldId) const;

    /**
     * Writes a snapshot of all policies for a given field ID into [out).
     *
     * Today at most one policy exists per field ID; when multiple policies per
     * field are supported this writes all of them with no change to callers.
     * Caller controls allocation — pass std::back_inserter(vec) to append, or
     * a raw iterator to write into a pre-allocated buffer.
     *
     * @param[in] fieldId The field ID to look up
     * @param[in] out     Output iterator to write matching policies into
     *
     * @return Iterator past the last written element
     */
    template <std::output_iterator<DcgmPolicy> OutputIt>
    [[nodiscard]] OutputIt GetPoliciesForField(unsigned short fieldId, OutputIt out) const
    {
        auto it = m_policies.find(fieldId);
        if (it != m_policies.end())
        {
            *out++ = it->second;
        }
        return out;
    }

    /**
     * Writes a snapshot of all policies into [out).
     *
     * Caller controls allocation — pass std::back_inserter(vec) to append, or
     * a raw iterator to write into a pre-allocated buffer.
     *
     * @param[in] out Output iterator to write policies into
     *
     * @return Iterator past the last written element
     */
    template <std::output_iterator<DcgmPolicy> OutputIt>
    [[nodiscard]] OutputIt GetAllPolicies(OutputIt out) const
    {
        for (auto const &[fieldId, policy] : m_policies)
        {
            *out++ = policy;
        }
        return out;
    }

private:
    static dcgmReturn_t ValidatePolicyBatchHelper(DcgmPolicy *first, DcgmPolicy *last, std::string_view operation);

    template <typename Builder>
    dcgmReturn_t ModifyPolicy(unsigned short fieldId, uint64_t policyId, Builder &&builder);

    void ClearRateLimitState(uint64_t policyId);

    // fieldId -> DcgmPolicy. One policy per field ID today.
    // When multiple policies per field are needed, change to unordered_map<unsigned short, vector<DcgmPolicy>>
    std::unordered_map<unsigned short, DcgmPolicy> m_policies;

    // policyId, entity group, entity id -> last notification timestamp. This runtime-only state is deliberately
    // not persisted: rate limits begin fresh after a hostengine restart.
    std::map<std::tuple<uint64_t, dcgm_field_entity_group_t, dcgm_field_eid_t>, timelib64_t> m_lastNotifications;
};

#endif // DCGMPOLICYREGISTRY_H
