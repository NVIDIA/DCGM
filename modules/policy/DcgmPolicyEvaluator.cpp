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

#include "DcgmPolicyEvaluator.h"

#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <algorithm>
#include <dcgm_fields.h>
#include <dcgm_structs.h>
#include <iterator>
#include <optional>
#include <timelib.h>
#include <vector>

/*****************************************************************************/
DcgmPolicyEvaluator::DcgmPolicyEvaluator(DcgmPolicyRegistry &policyRegistry)
    : m_policyRegistry(&policyRegistry)
    , m_previousValues()
{}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyEvaluator::EvaluatePolicyForEntity(unsigned short fieldId,
                                                          dcgmGroupEntityPair_t entity,
                                                          double value,
                                                          std::vector<dcgm_policy_violation_t> &violations)
{
    std::vector<DcgmPolicy> policyVec;
    (void)m_policyRegistry->GetPoliciesForField(fieldId, std::back_inserter(policyVec));

    // Current design constraint: one policy per fieldId.
    // If multiple policies are registered, only the first is evaluated.
    if (policyVec.empty())
    {
        return DCGM_ST_NO_DATA;
    }

    return EvaluatePolicy(policyVec.front(), entity, value, violations);
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyEvaluator::EvaluatePolicy(DcgmPolicy const &policy,
                                                 dcgmGroupEntityPair_t entity,
                                                 double value,
                                                 std::vector<dcgm_policy_violation_t> &violations)
{
    if (!policy.IsEnabled())
    {
        return DCGM_ST_NOT_CONFIGURED;
    }

    std::vector<dcgmGroupEntityPair_t> const &configuredEntities = policy.GetEntities();
    if (!configuredEntities.empty())
    {
        auto const entityIt = std::find_if(configuredEntities.begin(),
                                           configuredEntities.end(),
                                           [entity](dcgmGroupEntityPair_t const &configuredEntity) {
                                               return configuredEntity.entityGroupId == entity.entityGroupId
                                                      && configuredEntity.entityId == entity.entityId;
                                           });
        if (entityIt == configuredEntities.end())
        {
            return DCGM_ST_NO_DATA;
        }
    }

    double threshold    = policy.GetThreshold();
    auto policyOperator = policy.GetPolicyOperator();
    auto const fieldId  = policy.GetFieldId();

    bool violated = CheckViolation(fieldId, entity, value, threshold, policyOperator);

    if (!violated)
    {
        return DCGM_ST_NO_DATA;
    }

    violations.push_back(CreateViolationRecord(policy, entity, value));

    log_debug("Policy violation detected: fieldId {} entity {}:{} value {} vs threshold {}",
              fieldId,
              entity.entityGroupId,
              entity.entityId,
              value,
              threshold);

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmPolicyEvaluator::ClearFieldState(unsigned short fieldId)
{
    m_previousValues.erase(fieldId);
}

/*****************************************************************************/
bool DcgmPolicyEvaluator::CheckViolation(unsigned short fieldId,
                                         dcgmGroupEntityPair_t entity,
                                         double currentValue,
                                         double threshold,
                                         dcgmPolicyOperator_t op)
{
    switch (op)
    {
        case DCGM_POLICY_OP_GT:
            return (currentValue > threshold);

        case DCGM_POLICY_OP_GE:
            return (currentValue >= threshold);

        case DCGM_POLICY_OP_LT:
            return (currentValue < threshold);

        case DCGM_POLICY_OP_LE:
            return (currentValue <= threshold);

        case DCGM_POLICY_OP_EQ:
            // Equality-style policy operators intentionally use exact comparisons.
            // They are intended for integer-valued fields represented as doubles.
            return (currentValue == threshold);

        case DCGM_POLICY_OP_NE:
            return (currentValue != threshold);

        case DCGM_POLICY_OP_CHANGED:
        {
            std::pair<dcgm_field_entity_group_t, dcgm_field_eid_t> entityKey { entity.entityGroupId, entity.entityId };
            auto &fieldValues = m_previousValues[fieldId];
            auto entityIt     = fieldValues.find(entityKey);

            // CHANGED uses the same exact comparison semantics as EQ/NE.
            bool const changed = entityIt != fieldValues.end() && currentValue != entityIt->second;

            fieldValues[entityKey] = currentValue;
            return changed;
        }

        default:
            return false;
    }
}

/*****************************************************************************/
dcgm_policy_violation_t DcgmPolicyEvaluator::CreateViolationRecord(DcgmPolicy const &policy,
                                                                   dcgmGroupEntityPair_t entity,
                                                                   double value)
{
    dcgm_policy_violation_t violation = {};
    violation.policyId                = policy.GetPolicyId();
    SafeCopyTo(violation.policyName, policy.GetName().c_str());
    auto const fieldMeta = DcgmFieldGetById(policy.GetFieldId());
    SafeCopyTo(violation.fieldName, fieldMeta != nullptr ? fieldMeta->tag : "UNKNOWN");
    violation.fieldId        = policy.GetFieldId();
    violation.entity         = entity;
    violation.currentValue   = value;
    violation.thresholdValue = policy.GetThreshold();
    violation.policyOperator = policy.GetPolicyOperator();
    violation.channelMask    = policy.GetChannelMask();
    violation.timestamp      = timelib_usecSince1970();

    return violation;
}
