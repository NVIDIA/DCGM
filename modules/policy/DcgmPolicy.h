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

#ifndef DCGMPOLICY_H
#define DCGMPOLICY_H

#include "DcgmPolicyUtils.h"
#include "dcgm_policy_structs.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <timelib.h>

class DcgmPolicy
{
public:
    /**
     * Constructs a new DcgmPolicy instance
     *
     * @param[in] fieldId Field identifier for the policy
     * @param[in] name Name of the policy
     * @param[in] value Current observed value for the field
     * @param[in] threshold Threshold value for the policy
     * @param[in] policyOperator Operator to use for comparisons
     * @param[in] entities List of entities this policy applies to
     * @param[in] channelMask Bitmask of dcgmPolicyChannel_t notification channels
     * @param[in] state Whether the policy is enabled or disabled
     * @param[in] rateLimit Rate limit for policy notifications, in seconds
     */
    DcgmPolicy(unsigned short fieldId,
               std::string const &name,
               double value,
               double threshold,
               dcgmPolicyOperator_t policyOperator,
               std::vector<dcgmGroupEntityPair_t> entities,
               unsigned int channelMask,
               dcgmPolicyState_t state,
               unsigned int rateLimit)
        : m_policyId(DcgmPolicyUtils::GeneratePolicyId())
        , m_fieldId(fieldId)
        , m_name(name)
        , m_value(value)
        , m_threshold(threshold)
        , m_operator(policyOperator)
        , m_entities(std::move(entities))
        , m_channelMask(channelMask)
        , m_state(state)
        , m_rateLimit(std::chrono::seconds { rateLimit })
        , m_createdAt(timelib_usecSince1970())
        , m_violationCount(0)
        , m_lastTriggeredTimestamp(0)
    {
        EnforceSizeLimits();
    }

    /**
     * Constructs an updated DcgmPolicy carrying forward identity and runtime state from an existing policy.
     *
     * Used by DcgmPolicyRegistry::ModifyPolicy — preserves policyId, createdAt, violationCount,
     * and lastTriggeredTimestamp so granular config updates don't silently change the policy's
     * identity or reset accumulated violation history.
     *
     * @param[in] fieldId                  Field identifier
     * @param[in] name                     Policy name
     * @param[in] value                    Current observed value
     * @param[in] threshold                Threshold value
     * @param[in] policyOperator           Comparison operator
     * @param[in] entities                 Entities this policy applies to
     * @param[in] channelMask              Notification channel bitmask
     * @param[in] state                    Enabled or disabled
     * @param[in] rateLimit                Notification rate limit, in seconds
     * @param[in] policyId                 UUID carried forward from the original policy
     * @param[in] createdAt                Creation timestamp carried forward from the original policy
     * @param[in] violationCount           Violation count carried forward from the original policy
     * @param[in] lastTriggeredTimestamp   Last triggered timestamp carried forward from the original policy
     */
    DcgmPolicy(unsigned short fieldId,
               std::string const &name,
               double value,
               double threshold,
               dcgmPolicyOperator_t policyOperator,
               std::vector<dcgmGroupEntityPair_t> entities,
               unsigned int channelMask,
               dcgmPolicyState_t state,
               unsigned int rateLimit,
               uint64_t policyId,
               timelib64_t createdAt,
               unsigned int violationCount,
               timelib64_t lastTriggeredTimestamp)
        : m_policyId(policyId)
        , m_fieldId(fieldId)
        , m_name(name)
        , m_value(value)
        , m_threshold(threshold)
        , m_operator(policyOperator)
        , m_entities(std::move(entities))
        , m_channelMask(channelMask)
        , m_state(state)
        , m_rateLimit(std::chrono::seconds { rateLimit })
        , m_createdAt(createdAt)
        , m_violationCount(violationCount)
        , m_lastTriggeredTimestamp(lastTriggeredTimestamp)
    {
        EnforceSizeLimits();
    }

    /**
     * Scratch ctor: m_name/m_entities pre-reserved to their DCGM maxes, other fields inert.
     *
     * Used as a copy destination in DcgmPolicyRegistry::ModifyPolicy so that copy-assignment
     * reuses the reserved buffers and avoids avoidable heap allocation.
     */
    DcgmPolicy();

    ~DcgmPolicy() = default;

    // ----------------------------------------
    // Getters
    /**
     * Get the policy id
     *
     * @return The policy ID
     */
    [[nodiscard]] uint64_t GetPolicyId() const;

    /**
     * Get the field id
     *
     * @return The field ID
     */
    [[nodiscard]] unsigned short GetFieldId() const;

    /**
     * Get the name
     *
     * @return The policy name
     */
    [[nodiscard]] std::string const &GetName() const;

    /**
     * Get the current value
     *
     * @return The current value
     */
    [[nodiscard]] double GetValue() const;

    /**
     * Get the threshold
     *
     * @return The threshold value
     */
    [[nodiscard]] double GetThreshold() const;

    /**
     * Get the policy operator
     *
     * @return The policy operator
     */
    [[nodiscard]] dcgmPolicyOperator_t GetPolicyOperator() const;

    /**
     * Get the target entities.
     *
     * @return Vector of target entities.
     */
    [[nodiscard]] std::vector<dcgmGroupEntityPair_t> const &GetEntities() const;

    /**
     * Get the channel mask
     *
     * @return Bitmask of dcgmPolicyChannel_t values
     */
    [[nodiscard]] unsigned int GetChannelMask() const;

    /**
     * Get the policy state
     *
     * @return DCGM_POLICY_STATE_ENABLED or DCGM_POLICY_STATE_DISABLED
     */
    [[nodiscard]] dcgmPolicyState_t GetState() const;

    /**
     * Check if the policy is enabled
     *
     * @return True if enabled, false otherwise
     */
    [[nodiscard]] bool IsEnabled() const;

    /**
     * Get the rate limit
     *
     * @return The rate limit duration between successive notifications
     */
    [[nodiscard]] std::chrono::seconds GetRateLimit() const;

    /**
     * Get the creation timestamp
     *
     * @return The creation timestamp
     */
    [[nodiscard]] timelib64_t GetCreatedAt() const;

    /**
     * Get the total violation count
     *
     * @return The number of violations detected for this policy
     */
    [[nodiscard]] unsigned int GetViolationCount() const;

    /**
     * Get the last triggered timestamp
     *
     * @return The timestamp of the last violation (0 if never triggered)
     */
    [[nodiscard]] timelib64_t GetLastTriggeredTimestamp() const;

    /**
     * Increments violation count and records the timestamp of the latest violation.
     *
     * @param[in] timestamp The timestamp of the violation.
     */
    void RecordViolation(timelib64_t timestamp);

    /**
     * Resets violation count and last triggered timestamp to zero.
     */
    void ResetViolationStats();

    // ----------------------------------------

private:
    /// Truncates m_name/m_entities if they exceed DCGM maxes (logs a warning).
    void EnforceSizeLimits();

    uint64_t m_policyId;
    unsigned short m_fieldId;
    std::string m_name;
    double m_value;
    double m_threshold;
    dcgmPolicyOperator_t m_operator;
    std::vector<dcgmGroupEntityPair_t> m_entities;
    unsigned int m_channelMask;
    dcgmPolicyState_t m_state;
    std::chrono::seconds m_rateLimit;
    timelib64_t m_createdAt;
    unsigned int m_violationCount;
    timelib64_t m_lastTriggeredTimestamp;
};

#endif // DCGMPOLICY_H
