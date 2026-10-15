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

#pragma once

#include "DcgmProtocol.h"
#include "dcgm_policy_structs.h"

#include <timelib.h>

#include <atomic>
#include <unordered_map>
#include <utility>

enum class DcgmPolicyCallbackPayload
{
    Legacy,
    FieldViolation,
};

/**
 * Callback subscriber information
 *
 * Tracks callback registration for legacy and field-based policy callback APIs.
 */
struct DcgmPolicyCallbackInfo
{
    explicit DcgmPolicyCallbackInfo(dcgm_request_id_t reqId,
                                    unsigned short fieldIdFilterIn          = 0,
                                    uint64_t policyIdFilterIn               = 0,
                                    DcgmPolicyCallbackPayload payloadKindIn = DcgmPolicyCallbackPayload::Legacy)
        : requestId(reqId)
        , fieldIdFilter(fieldIdFilterIn)
        , policyIdFilter(policyIdFilterIn)
        , payloadKind(payloadKindIn)
        , pendingDeregistration(false)
    {}

    dcgm_request_id_t requestId;  //!< Request ID of subscriber
    unsigned short fieldIdFilter; //!< 0 means all fields
    uint64_t policyIdFilter;      //!< 0 means all policies
    DcgmPolicyCallbackPayload payloadKind;
    std::unordered_map<unsigned short, timelib64_t>
        lastSentTimestampPerField;           //!< Last notification time per fieldId (for rate limiting).
                                             //!< Matches legacy DcgmPolicyManager::SetViolation semantics:
                                             //!< rate limit is per (subscriber, fieldId), so an unrelated
                                             //!< field's violation does not suppress this one.
    std::atomic<bool> pendingDeregistration; //!< Whether the callback is pending deregistration
};

/**
 * RAII guard for policy callback deregistration
 *
 * On destruction, marks the associated callback as pending deregistration.
 * The callback is evicted from the active set on the next RouteViolation call.
 *
 * Move-only: copying is disabled to ensure single ownership.
 * Call Reset() to trigger deregistration immediately and disarm the guard.
 */
class DcgmPolicyCallbackRAII
{
public:
    explicit DcgmPolicyCallbackRAII(DcgmPolicyCallbackInfo &callbackInfo)
        : m_pendingDeregistration(&callbackInfo.pendingDeregistration)
    {}

    ~DcgmPolicyCallbackRAII()
    {
        if (m_pendingDeregistration != nullptr)
        {
            *m_pendingDeregistration = true;
        }
    }

    DcgmPolicyCallbackRAII(DcgmPolicyCallbackRAII &&other) noexcept
        : m_pendingDeregistration(std::exchange(other.m_pendingDeregistration, nullptr))
    {}

    DcgmPolicyCallbackRAII &operator=(DcgmPolicyCallbackRAII &&other) noexcept
    {
        std::swap(m_pendingDeregistration, other.m_pendingDeregistration);
        return *this;
    }

    DcgmPolicyCallbackRAII(const DcgmPolicyCallbackRAII &)            = delete;
    DcgmPolicyCallbackRAII &operator=(const DcgmPolicyCallbackRAII &) = delete;

    //! Trigger deregistration immediately and disarm this guard.
    void Reset() noexcept
    {
        auto guard = std::move(*this);
    }

private:
    std::atomic<bool> *m_pendingDeregistration;
};
