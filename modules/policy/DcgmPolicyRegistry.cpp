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

#include "DcgmPolicyRegistry.h"

#include <DcgmLogging.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <string_view>
#include <unordered_map>

DcgmPolicyRegistry::DcgmPolicyRegistry()
{
    m_policies.reserve(DCGM_POLICY_MAX_POLICIES);
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::AddPolicy(DcgmPolicy const &policy)
{
    unsigned short const fieldId = policy.GetFieldId();

    if (m_policies.size() >= DCGM_POLICY_MAX_POLICIES)
    {
        log_error("Policy registry is full; cannot add fieldId {}", fieldId);
        return DCGM_ST_BADPARAM;
    }

    auto const [iterator, inserted] = m_policies.emplace(fieldId, policy);
    if (!inserted)
    {
        log_error("Policy with fieldId {} already exists", iterator->first);
        return DCGM_ST_DUPLICATE_KEY;
    }

    log_info("Added policy with fieldId {} (policyId: {})", fieldId, policy.GetPolicyId());
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::AddPolicies(DcgmPolicy *first, DcgmPolicy *last)
{
    // Empty range is a no-op success.
    size_t const inputCount = static_cast<size_t>(last - first);
    if (inputCount == 0)
    {
        return DCGM_ST_OK;
    }

    // Validate the batch of policies before adding them to the registry.
    if (auto const ret = ValidatePolicyBatchHelper(first, last, "AddPolicies"); ret != DCGM_ST_OK)
    {
        return ret;
    }

    // Reject if adding the batch to existing policies would exceed capacity.
    if (m_policies.size() + inputCount > DCGM_POLICY_MAX_POLICIES)
    {
        log_warning("AddPolicies: batch would exceed max policies {}; no policies added", DCGM_POLICY_MAX_POLICIES);
        return DCGM_ST_BADPARAM;
    }

    // Reject if any fieldId already exists in the registry.
    for (DcgmPolicy *it = first; it != last; ++it)
    {
        if (m_policies.contains(it->GetFieldId()))
        {
            log_warning("AddPolicies: fieldId {} already exists; no policies added", it->GetFieldId());
            return DCGM_ST_DUPLICATE_KEY;
        }
    }

    // All checks passed; commit the batch by moving each policy into the registry.
    for (DcgmPolicy *it = first; it != last; ++it)
    {
        m_policies.emplace(it->GetFieldId(), std::move(*it));
    }

    log_debug("AddPolicies: input={} added={}", inputCount, inputCount);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::ReplaceAllPolicies(DcgmPolicy *first, DcgmPolicy *last)
{
    size_t const inputCount = static_cast<size_t>(last - first);
    if (auto const ret = ValidatePolicyBatchHelper(first, last, "ReplaceAllPolicies"); ret != DCGM_ST_OK)
    {
        return ret;
    }

    std::unordered_map<unsigned short, DcgmPolicy> replacement;
    replacement.reserve(inputCount);
    for (DcgmPolicy *it = first; it != last; ++it)
    {
        replacement.emplace(it->GetFieldId(), std::move(*it));
    }

    m_policies.swap(replacement);
    log_debug("Replaced {} policies in registry", inputCount);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::ValidatePolicyBatchHelper(DcgmPolicy *first,
                                                           DcgmPolicy *last,
                                                           std::string_view operation)
{
    size_t const inputCount = static_cast<size_t>(last - first);
    if (inputCount > DCGM_POLICY_MAX_POLICIES)
    {
        log_warning("{}: batch size {} exceeds max policies {}", operation, inputCount, DCGM_POLICY_MAX_POLICIES);
        return DCGM_ST_BADPARAM;
    }

    for (DcgmPolicy *i = first; i != last; ++i)
    {
        for (DcgmPolicy *j = i + 1; j != last; ++j)
        {
            if (i->GetFieldId() == j->GetFieldId())
            {
                log_warning("{}: duplicate fieldId {} in batch", operation, i->GetFieldId());
                return DCGM_ST_DUPLICATE_KEY;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::RemovePolicy(unsigned short fieldId, uint64_t policyId)
{
    auto it = m_policies.find(fieldId);
    if (it == m_policies.end() || it->second.GetPolicyId() != policyId)
    {
        return DCGM_ST_NO_DATA;
    }

    ClearRateLimitState(policyId);
    m_policies.erase(it);
    log_debug("Removed policy with fieldId {} policyId {}", fieldId, policyId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::RemoveAllPolicies()
{
    auto const removedCount = m_policies.size();
    m_policies.clear();
    m_lastNotifications.clear();

    log_debug("Cleared all policies from registry (removed {} policies)", removedCount);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::ReplacePoliciesForField(unsigned short fieldId, std::vector<DcgmPolicy> policies)
{
    if (policies.empty())
    {
        log_error("ReplacePoliciesForField called with empty policy list for fieldId {}", fieldId);
        return DCGM_ST_BADPARAM;
    }

    // Check if all policies have the same fieldId
    for (auto const &p : policies)
    {
        if (p.GetFieldId() != fieldId)
        {
            log_error("ReplacePoliciesForField: fieldId mismatch (expected {}, got {})", fieldId, p.GetFieldId());
            return DCGM_ST_BADPARAM;
        }
    }

    if (policies.size() != 1)
    {
        log_error("ReplacePoliciesForField supports exactly one policy per field (received {})", policies.size());
        return DCGM_ST_BADPARAM;
    }

    DcgmPolicy replacement = std::move(policies.front());

    auto it = m_policies.find(fieldId);
    if (it == m_policies.end())
    {
        return DCGM_ST_NO_DATA;
    }

    ClearRateLimitState(it->second.GetPolicyId());
    it->second = std::move(replacement);

    log_debug("Replaced policy for fieldId {}", fieldId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
template <typename Builder>
dcgmReturn_t DcgmPolicyRegistry::ModifyPolicy(unsigned short fieldId, uint64_t policyId, Builder &&builder)
{
    auto it = m_policies.find(fieldId);
    if (it == m_policies.end() || it->second.GetPolicyId() != policyId)
    {
        return DCGM_ST_NO_DATA;
    }

    DcgmPolicy newPolicy = builder(it->second);

    it->second = std::move(newPolicy);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::UpdatePolicyValue(unsigned short fieldId, uint64_t policyId, double value)
{
    auto result = ModifyPolicy(fieldId, policyId, [&](DcgmPolicy const &p) {
        return DcgmPolicy(p.GetFieldId(),
                          p.GetName(),
                          value,
                          p.GetThreshold(),
                          p.GetPolicyOperator(),
                          p.GetEntities(),
                          p.GetChannelMask(),
                          p.GetState(),
                          p.GetRateLimit().count(),
                          p.GetPolicyId(),
                          p.GetCreatedAt(),
                          p.GetViolationCount(),
                          p.GetLastTriggeredTimestamp());
    });
    if (result == DCGM_ST_OK)
    {
        log_debug("Updated value for policyId {} to {}", policyId, value);
    }
    return result;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::UpdatePolicyThreshold(unsigned short fieldId, uint64_t policyId, double threshold)
{
    auto result = ModifyPolicy(fieldId, policyId, [&](DcgmPolicy const &p) {
        return DcgmPolicy(p.GetFieldId(),
                          p.GetName(),
                          p.GetValue(),
                          threshold,
                          p.GetPolicyOperator(),
                          p.GetEntities(),
                          p.GetChannelMask(),
                          p.GetState(),
                          p.GetRateLimit().count(),
                          p.GetPolicyId(),
                          p.GetCreatedAt(),
                          p.GetViolationCount(),
                          p.GetLastTriggeredTimestamp());
    });

    if (result == DCGM_ST_OK)
    {
        log_debug("Updated threshold for policyId {} to {}", policyId, threshold);
    }
    return result;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::UpdatePolicyOperator(unsigned short fieldId,
                                                      uint64_t policyId,
                                                      dcgmPolicyOperator_t policyOperator)
{
    auto result = ModifyPolicy(fieldId, policyId, [&](DcgmPolicy const &p) {
        return DcgmPolicy(p.GetFieldId(),
                          p.GetName(),
                          p.GetValue(),
                          p.GetThreshold(),
                          policyOperator,
                          p.GetEntities(),
                          p.GetChannelMask(),
                          p.GetState(),
                          p.GetRateLimit().count(),
                          p.GetPolicyId(),
                          p.GetCreatedAt(),
                          p.GetViolationCount(),
                          p.GetLastTriggeredTimestamp());
    });

    if (result == DCGM_ST_OK)
    {
        log_debug("Updated operator for policyId {} to {}", policyId, static_cast<int>(policyOperator));
    }
    return result;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::UpdatePolicyEntities(unsigned short fieldId,
                                                      uint64_t policyId,
                                                      std::vector<dcgmGroupEntityPair_t> const &entities)
{
    auto result = ModifyPolicy(fieldId, policyId, [&](DcgmPolicy const &p) {
        return DcgmPolicy(p.GetFieldId(),
                          p.GetName(),
                          p.GetValue(),
                          p.GetThreshold(),
                          p.GetPolicyOperator(),
                          entities,
                          p.GetChannelMask(),
                          p.GetState(),
                          p.GetRateLimit().count(),
                          p.GetPolicyId(),
                          p.GetCreatedAt(),
                          p.GetViolationCount(),
                          p.GetLastTriggeredTimestamp());
    });

    if (result == DCGM_ST_OK)
    {
        log_debug("Updated entities for policyId {}", policyId);
    }
    return result;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::UpdatePolicyChannels(unsigned short fieldId,
                                                      uint64_t policyId,
                                                      unsigned int channelMask)
{
    auto result = ModifyPolicy(fieldId, policyId, [&](DcgmPolicy const &p) {
        return DcgmPolicy(p.GetFieldId(),
                          p.GetName(),
                          p.GetValue(),
                          p.GetThreshold(),
                          p.GetPolicyOperator(),
                          p.GetEntities(),
                          channelMask,
                          p.GetState(),
                          p.GetRateLimit().count(),
                          p.GetPolicyId(),
                          p.GetCreatedAt(),
                          p.GetViolationCount(),
                          p.GetLastTriggeredTimestamp());
    });

    if (result == DCGM_ST_OK)
    {
        log_debug("Updated channel mask for policyId {}", policyId);
    }
    return result;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::SetPolicyState(unsigned short fieldId, uint64_t policyId, dcgmPolicyState_t state)
{
    auto result = ModifyPolicy(fieldId, policyId, [&](DcgmPolicy const &p) {
        return DcgmPolicy(p.GetFieldId(),
                          p.GetName(),
                          p.GetValue(),
                          p.GetThreshold(),
                          p.GetPolicyOperator(),
                          p.GetEntities(),
                          p.GetChannelMask(),
                          state,
                          p.GetRateLimit().count(),
                          p.GetPolicyId(),
                          p.GetCreatedAt(),
                          p.GetViolationCount(),
                          p.GetLastTriggeredTimestamp());
    });

    if (result == DCGM_ST_OK)
    {
        log_debug("Set state for policyId {} to {}", policyId, static_cast<int>(state));
    }
    return result;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::UpdatePolicyRateLimit(unsigned short fieldId,
                                                       uint64_t policyId,
                                                       unsigned int rateLimit)
{
    auto result = ModifyPolicy(fieldId, policyId, [&](DcgmPolicy const &p) {
        return DcgmPolicy(p.GetFieldId(),
                          p.GetName(),
                          p.GetValue(),
                          p.GetThreshold(),
                          p.GetPolicyOperator(),
                          p.GetEntities(),
                          p.GetChannelMask(),
                          p.GetState(),
                          rateLimit,
                          p.GetPolicyId(),
                          p.GetCreatedAt(),
                          p.GetViolationCount(),
                          p.GetLastTriggeredTimestamp());
    });

    if (result == DCGM_ST_OK)
    {
        log_debug("Updated rate limit for policyId {} to {}", policyId, rateLimit);
    }
    return result;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyRegistry::RecordViolation(dcgm_policy_violation_t const &violation)
{
    // Runtime stats are updated in place: no heap allocation in this method.
    // unordered_map lookup is average O(1) (worst-case O(n)).

    // The policy may have been deleted after the violation was produced.
    auto it = m_policies.find(violation.fieldId);
    if (it == m_policies.end())
    {
        return DCGM_ST_NO_DATA;
    }

    // The policy may have been recreated or modified since the violation was produced.
    DcgmPolicy &policy = it->second;
    if (policy.GetPolicyId() != violation.policyId || policy.GetPolicyOperator() != violation.policyOperator
        || policy.GetThreshold() != violation.thresholdValue || policy.GetChannelMask() != violation.channelMask)
    {
        return DCGM_ST_NO_DATA;
    }

    // Disabled policies should not accumulate violation stats or route notifications.
    if (!policy.IsEnabled())
    {
        return DCGM_ST_NOT_CONFIGURED;
    }

    // Entity filters may have changed after the violation was produced.
    auto const &entities = policy.GetEntities();
    if (!entities.empty())
    {
        auto const entityIt = std::find_if(entities.begin(), entities.end(), [&violation](auto const &entity) {
            return entity.entityGroupId == violation.entity.entityGroupId
                   && entity.entityId == violation.entity.entityId;
        });
        if (entityIt == entities.end())
        {
            return DCGM_ST_NO_DATA;
        }
    }

    auto const rateLimit = policy.GetRateLimit();
    if (rateLimit > std::chrono::seconds::zero())
    {
        auto const key
            = std::make_tuple(policy.GetPolicyId(), violation.entity.entityGroupId, violation.entity.entityId);
        auto const lastNotificationIt = m_lastNotifications.find(key);
        auto const rateLimitUsec      = std::chrono::duration_cast<std::chrono::microseconds>(rateLimit).count();
        if (lastNotificationIt != m_lastNotifications.end() && violation.timestamp >= lastNotificationIt->second
            && violation.timestamp - lastNotificationIt->second < rateLimitUsec)
        {
            log_debug("Rate limiting policyId {} for entity {}:{} until {}",
                      policy.GetPolicyId(),
                      violation.entity.entityGroupId,
                      violation.entity.entityId,
                      lastNotificationIt->second + rateLimitUsec);
            return DCGM_ST_NO_DATA;
        }

        m_lastNotifications[key] = violation.timestamp;
    }

    // All validation passed; record only the runtime counters/timestamp.
    policy.RecordViolation(violation.timestamp);
    return DCGM_ST_OK;
}

void DcgmPolicyRegistry::ClearRateLimitState(uint64_t policyId)
{
    std::erase_if(m_lastNotifications, [policyId](auto const &entry) { return std::get<0>(entry.first) == policyId; });
}

/*****************************************************************************/
std::optional<DcgmPolicy> DcgmPolicyRegistry::GetPolicyById(uint64_t policyId) const
{
    for (auto const &[fieldId, policy] : m_policies)
    {
        if (policy.GetPolicyId() == policyId)
        {
            return policy;
        }
    }
    return std::nullopt;
}

/*****************************************************************************/
bool DcgmPolicyRegistry::HasPolicy(unsigned short fieldId) const
{
    return m_policies.contains(fieldId);
}
