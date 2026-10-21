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

#include "DcgmPolicyCallbackRAII.h"
#include "dcgm_policy_structs.h"

#include <DcgmCoreProxy.h>
#include <dcgm_structs.h>

#include <boost/circular_buffer.hpp>

#include <json/json.h>

#include <algorithm>
#include <chrono>
#include <expected>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

/// Default minimum interval between repeated callback notifications for the same (subscriber, fieldId) pair
inline constexpr std::chrono::microseconds DCGM_POLICY_DEFAULT_CALLBACK_RATE_LIMIT
    = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(5));

/**
 * Notification Router - Dispatches policy violations to multiple channels
 *
 * Routes violation events to their configured destinations (console, file, callback)
 * with channel-specific formatting and rate limiting.
 */
class DcgmPolicyNotificationRouter
{
public:
    /**
     * Constructs with an injected output stream for the FILE channel.
     *
     * @param[in] coreCallbacks     Core callbacks for creating DcgmCoreProxy
     * @param[in] logStream         Output stream for FILE channel, or nullptr to disable it
     * @param[in] callbackRateLimit Minimum interval between repeated callback notifications
     *                              for the same (subscriber, fieldId) pair
     */
    DcgmPolicyNotificationRouter(dcgmCoreCallbacks_t coreCallbacks,
                                 std::unique_ptr<std::ostream> logStream     = nullptr,
                                 std::chrono::microseconds callbackRateLimit = DCGM_POLICY_DEFAULT_CALLBACK_RATE_LIMIT);

    ~DcgmPolicyNotificationRouter() = default;

    /**
     * Routes a single violation to all configured channels
     *
     * @note Not thread-safe. All router calls must be externally serialized by the policy manager.
     *
     * @param[in] violation The policy violation to route
     *
     * @return DCGM_ST_OK on success, error code on failure
     */
    dcgmReturn_t RouteViolation(dcgm_policy_violation_t const &violation);

    /**
     * Registers a callback subscriber for policy violations.
     *
     * @note Not thread-safe. All router calls must be externally serialized by the policy manager.
     *
     * @param[in] connectionId Connection ID of the subscriber
     * @param[in] requestId Request ID of the subscriber
     *
     * @return DcgmPolicyCallbackRAII RAII guard on success, or dcgmReturn_t error code on failure
     */
    [[nodiscard]] std::expected<DcgmPolicyCallbackRAII, dcgmReturn_t> RegisterCallback(
        dcgm_connection_id_t connectionId,
        dcgm_request_id_t requestId);

    /**
     * Registers a callback subscriber for field-based policy violation payloads.
     * fieldId == 0 and policyId == 0 act as wildcards.
     *
     * @note Not thread-safe. All router calls must be externally serialized by the policy manager.
     */
    [[nodiscard]] std::expected<DcgmPolicyCallbackRAII, dcgmReturn_t> RegisterViolationCallback(
        dcgm_connection_id_t connectionId,
        dcgm_request_id_t requestId,
        unsigned short fieldId,
        uint64_t policyId);

    /**
     * Writes violations that occurred at or after @p since into [@p out).
     * Pass std::chrono::steady_clock::time_point::min() to write all buffered violations.
     * Returns an iterator past the last written element.
     */
    template <std::output_iterator<dcgm_policy_violation_t> OutputIt>
    [[nodiscard]] OutputIt GetRecentViolations(OutputIt out, std::chrono::steady_clock::time_point since) const
    {
        timelib64_t cutoff = 0;
        if (since != std::chrono::steady_clock::time_point::min())
        {
            auto const nowSteady  = std::chrono::steady_clock::now();
            auto const nowUsec    = static_cast<timelib64_t>(timelib_usecSince1970());
            auto const offsetUsec = std::chrono::duration_cast<std::chrono::microseconds>(since - nowSteady).count();
            cutoff                = nowUsec + offsetUsec;
        }

        for (auto const &v : m_recentViolations.buffer)
        {
            if (v.timestamp >= cutoff)
            {
                *out++ = v;
            }
        }
        return out;
    }

    /**
     * Writes the @p maxCount most recent violations into [@p out).
     * Returns an iterator past the last written element.
     */
    template <std::output_iterator<dcgm_policy_violation_t> OutputIt>
    [[nodiscard]] OutputIt GetRecentViolations(OutputIt out, size_t maxCount) const
    {
        size_t const actualCount = std::min(maxCount, m_recentViolations.buffer.size());
        auto startIt             = m_recentViolations.buffer.end() - static_cast<ptrdiff_t>(actualCount);
        return std::copy(startIt, m_recentViolations.buffer.end(), out);
    }

    /// Convenience overload — allocates and returns a vector.
    [[nodiscard]] std::vector<dcgm_policy_violation_t> GetRecentViolations(
        std::chrono::steady_clock::time_point since) const;

    /// Convenience overload — allocates and returns a vector.
    [[nodiscard]] std::vector<dcgm_policy_violation_t> GetRecentViolations(size_t maxCount) const;

private:
    // Ownership contract:
    //   The router is owned by serialized policy-manager command processing. Notification commands,
    //   callback registration, recent-violation queries, and routing all run on that same path.
    //   As a result, router members are not internally synchronized.

    DcgmCoreProxy m_coreProxy; //!< Core proxy for IPC communication (callbacks)
    std::map<dcgm_connection_id_t, std::list<DcgmPolicyCallbackInfo>>
        m_callbacks; //!< Callback subscribers owned by the serialized policy-manager path.

    mutable struct
    {
        boost::circular_buffer<dcgm_policy_violation_t> buffer;
    } m_recentViolations;

    // File channel (not guarded; see ownership contract above)
    std::unique_ptr<std::ostream> m_logStream;
    std::unique_ptr<Json::StreamWriter> m_streamWriter;

    std::chrono::microseconds
        m_callbackRateLimitUs; //!< Minimum interval between repeated callback notifications per (subscriber, fieldId)

    /**
     * Build policy notification message from violation
     *
     * @param[in] violation The policy violation detected
     * @param[out] message The legacy notification message to populate
     */
    bool BuildPolicyNotifyMessage(dcgm_policy_violation_t const &violation, dcgm_msg_policy_notify_t &message);

    [[nodiscard]] std::expected<DcgmPolicyCallbackRAII, dcgmReturn_t> RegisterCallbackImpl(
        dcgm_connection_id_t connectionId,
        dcgm_request_id_t requestId,
        unsigned short fieldId,
        uint64_t policyId,
        DcgmPolicyCallbackPayload payloadKind);
};
