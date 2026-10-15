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

#include "DcgmPolicy.h"

#include <DcgmLogging.h>
#include <dcgm_structs.h>

/*****************************************************************************/
DcgmPolicy::DcgmPolicy()
    : m_policyId(0)
    , m_fieldId(0)
    , m_value(0.0)
    , m_threshold(0.0)
    , m_operator(DCGM_POLICY_OP_GT)
    , m_channelMask(0)
    , m_state(DCGM_POLICY_STATE_DISABLED)
    , m_rateLimit(std::chrono::seconds::zero())
    , m_createdAt(0)
    , m_violationCount(0)
    , m_lastTriggeredTimestamp(0)
{
    // Pre-reserve max capacity. Subsequent copy-assignment from a real policy
    // reuses these buffers, so no heap allocation occurs on that copy.
    m_name.reserve(DCGM_MAX_STR_LENGTH);
    m_entities.reserve(DCGM_MAX_NUM_DEVICES);
}

/*****************************************************************************/
void DcgmPolicy::EnforceSizeLimits()
{
    if (m_name.size() > DCGM_MAX_STR_LENGTH)
    {
        log_warning("Policy name truncated from {} to {} characters", m_name.size(), DCGM_MAX_STR_LENGTH);
        m_name.resize(DCGM_MAX_STR_LENGTH);
    }
    if (m_entities.size() > DCGM_MAX_NUM_DEVICES)
    {
        log_warning("Policy entity list truncated from {} to {} entries", m_entities.size(), DCGM_MAX_NUM_DEVICES);
        m_entities.resize(DCGM_MAX_NUM_DEVICES);
    }
}

/*****************************************************************************/
// Getters
/*****************************************************************************/

uint64_t DcgmPolicy::GetPolicyId() const
{
    return m_policyId;
}

/*****************************************************************************/
unsigned short DcgmPolicy::GetFieldId() const
{
    return m_fieldId;
}

/*****************************************************************************/
std::string const &DcgmPolicy::GetName() const
{
    return m_name;
}

/*****************************************************************************/
double DcgmPolicy::GetValue() const
{
    return m_value;
}

/*****************************************************************************/
double DcgmPolicy::GetThreshold() const
{
    return m_threshold;
}

/*****************************************************************************/
dcgmPolicyOperator_t DcgmPolicy::GetPolicyOperator() const
{
    return m_operator;
}

/*****************************************************************************/
std::vector<dcgmGroupEntityPair_t> const &DcgmPolicy::GetEntities() const
{
    return m_entities;
}

/*****************************************************************************/
unsigned int DcgmPolicy::GetChannelMask() const
{
    return m_channelMask;
}

/*****************************************************************************/
dcgmPolicyState_t DcgmPolicy::GetState() const
{
    return m_state;
}

/*****************************************************************************/
bool DcgmPolicy::IsEnabled() const
{
    return m_state == DCGM_POLICY_STATE_ENABLED;
}

/*****************************************************************************/
std::chrono::seconds DcgmPolicy::GetRateLimit() const
{
    return m_rateLimit;
}

/*****************************************************************************/
timelib64_t DcgmPolicy::GetCreatedAt() const
{
    return m_createdAt;
}

/*****************************************************************************/
unsigned int DcgmPolicy::GetViolationCount() const
{
    return m_violationCount;
}

/*****************************************************************************/
timelib64_t DcgmPolicy::GetLastTriggeredTimestamp() const
{
    return m_lastTriggeredTimestamp;
}

/*****************************************************************************/
void DcgmPolicy::RecordViolation(timelib64_t timestamp)
{
    m_violationCount++;
    m_lastTriggeredTimestamp = timestamp;
}

/*****************************************************************************/
void DcgmPolicy::ResetViolationStats()
{
    m_violationCount         = 0;
    m_lastTriggeredTimestamp = 0;
}
