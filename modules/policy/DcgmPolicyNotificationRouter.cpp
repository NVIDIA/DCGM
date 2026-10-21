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

#include "DcgmPolicyNotificationRouter.h"
#include "DcgmPolicyUtils.h"

#include <TimeLib.hpp>

#include <DcgmLogging.h>
#include <DcgmProtocol.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <json/json.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fmt/format.h>
#include <utility>

/*****************************************************************************/
DcgmPolicyNotificationRouter::DcgmPolicyNotificationRouter(dcgmCoreCallbacks_t coreCallbacks,
                                                           std::unique_ptr<std::ostream> logStream,
                                                           std::chrono::microseconds callbackRateLimit)
    : m_coreProxy(coreCallbacks)
    , m_logStream(std::move(logStream))
    , m_callbackRateLimitUs(callbackRateLimit)
{
    if (m_logStream)
    {
        Json::StreamWriterBuilder writerBuilder;
        writerBuilder["indentation"] = "";
        m_streamWriter.reset(writerBuilder.newStreamWriter());
    }
    m_recentViolations.buffer.set_capacity(1000);
    log_debug("DcgmPolicyNotificationRouter initialized with circular buffer capacity 1000");
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyNotificationRouter::RouteViolation(dcgm_policy_violation_t const &violation)
{
    bool const hasConsole  = (violation.channelMask & DCGM_POLICY_CHANNEL_CONSOLE) != 0;
    bool const hasFile     = (violation.channelMask & DCGM_POLICY_CHANNEL_FILE) != 0;
    bool const hasCallback = (violation.channelMask & DCGM_POLICY_CHANNEL_CALLBACK) != 0;

    // Recent violations feed dcgmPolicyGetViolations / dcgmi watch, regardless of configured delivery channels.
    m_recentViolations.buffer.push_back(violation); // capacity fixed at construction; never allocates
    log_debug("Added violation to recent buffer: fieldId {} entity {}:{}",
              violation.fieldId,
              violation.entity.entityGroupId,
              violation.entity.entityId);

    // CONSOLE channel
    if (hasConsole)
    {
        log_debug("Console channel requested for fieldId {} entity {}:{}",
                  violation.fieldId,
                  violation.entity.entityGroupId,
                  violation.entity.entityId);
    }

    // FILE channel
    if (hasFile)
    {
        if (m_logStream == nullptr)
        {
            log_warning(
                "DCGM_POLICY_CHANNEL_FILE requested for fieldId {} but no log stream configured — violation dropped",
                violation.fieldId);
        }
        else
        {
            Json::Value entry;
            entry["timestamp"]     = static_cast<Json::Int64>(violation.timestamp);
            entry["timestampIso"]  = DcgmNs::Timelib::FormatTimestampIso(violation.timestamp);
            entry["entityGroupId"] = static_cast<Json::UInt>(violation.entity.entityGroupId);
            entry["entityId"]      = static_cast<Json::UInt>(violation.entity.entityId);
            entry["fieldId"]       = static_cast<Json::UInt>(violation.fieldId);
            entry["fieldName"]     = violation.fieldName;
            entry["policyName"]    = violation.policyName;
            entry["currentValue"]  = violation.currentValue;
            entry["threshold"]     = violation.thresholdValue;
            auto operatorStr       = DcgmPolicyUtils::OperatorToString(violation.policyOperator);
            entry["operator"]      = operatorStr.has_value() ? std::string(*operatorStr) : [&] {
                log_warning("Unknown policy operator {} for fieldId {} — writing raw value to log",
                            static_cast<int>(violation.policyOperator),
                            violation.fieldId);
                return fmt::format("UNKNOWN({})", static_cast<int>(violation.policyOperator));
            }();
            entry["status"]        = "VIOLATION";

            m_streamWriter->write(entry, m_logStream.get());
            *m_logStream << '\n';
            m_logStream->flush();

            log_debug("Logged violation to file: fieldId {} entity {}:{}",
                      violation.fieldId,
                      violation.entity.entityGroupId,
                      violation.entity.entityId);
        }
    }

    // CALLBACK channel
    if (hasCallback)
    {
        dcgm_msg_policy_notify_t notify {};
        bool legacyNotifyBuilt = false;
        dcgm_msg_policy_violation_t fieldNotify { violation };

        std::list<DcgmPolicyCallbackInfo> evicted;
        std::map<dcgm_connection_id_t, std::list<DcgmPolicyCallbackInfo>> emptyConnections;
        auto it = m_callbacks.begin();
        while (it != m_callbacks.end())
        {
            auto &[connectionId, infos] = *it;
            auto iter                   = infos.begin();
            while (iter != infos.end())
            {
                auto &callbackInfo = *iter;
                if (callbackInfo.pendingDeregistration.load())
                {
                    evicted.splice(evicted.end(), infos, iter++);
                    continue;
                }
                if ((callbackInfo.fieldIdFilter != 0 && callbackInfo.fieldIdFilter != violation.fieldId)
                    || (callbackInfo.policyIdFilter != 0 && callbackInfo.policyIdFilter != violation.policyId))
                {
                    ++iter;
                    continue;
                }

                // Rate limit is per (subscriber, fieldId) — matches legacy DcgmPolicyManager::SetViolation
                // semantics so an unrelated field's violation does not suppress this one.
                if (auto const lastIt = callbackInfo.lastSentTimestampPerField.find(violation.fieldId);
                    lastIt != callbackInfo.lastSentTimestampPerField.end())
                {
                    if (violation.timestamp <= lastIt->second)
                    {
                        ++iter;
                        continue;
                    }

                    timelib64_t const delta = violation.timestamp - lastIt->second;
                    if (delta < m_callbackRateLimitUs.count())
                    {
                        ++iter;
                        continue;
                    }
                }

                if (callbackInfo.payloadKind == DcgmPolicyCallbackPayload::Legacy && !legacyNotifyBuilt)
                {
                    legacyNotifyBuilt = BuildPolicyNotifyMessage(violation, notify);
                }

                if (callbackInfo.payloadKind == DcgmPolicyCallbackPayload::Legacy && !legacyNotifyBuilt)
                {
                    ++iter;
                    continue;
                }

                callbackInfo.lastSentTimestampPerField[violation.fieldId] = violation.timestamp;

                auto const msgType     = callbackInfo.payloadKind == DcgmPolicyCallbackPayload::Legacy
                                             ? DCGM_MSG_POLICY_NOTIFY
                                             : DCGM_MSG_POLICY_VIOLATION;
                auto *payload          = callbackInfo.payloadKind == DcgmPolicyCallbackPayload::Legacy
                                             ? static_cast<void *>(&notify)
                                             : static_cast<void *>(&fieldNotify);
                auto const payloadSize = callbackInfo.payloadKind == DcgmPolicyCallbackPayload::Legacy
                                             ? sizeof(notify)
                                             : sizeof(fieldNotify);

                log_debug(
                    "Sending callback notification to connectionId {} requestId {} for fieldId {} entity {}:{} msgType {}",
                    connectionId,
                    callbackInfo.requestId,
                    violation.fieldId,
                    violation.entity.entityGroupId,
                    violation.entity.entityId,
                    msgType);

                m_coreProxy.SendRawMessageToClient(
                    connectionId, msgType, callbackInfo.requestId, payload, payloadSize, DCGM_ST_OK);

                ++iter;
            }

            // Extract empty nodes into throwaway map — O(log n) tree relink, no allocation
            if (infos.empty())
            {
                emptyConnections.insert(m_callbacks.extract(it++));
            }
            else
            {
                ++it;
            }
        }
        // emptyConnections destructs here — list nodes freed outside the loop
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmPolicyNotificationRouter::BuildPolicyNotifyMessage(dcgm_policy_violation_t const &violation,
                                                            dcgm_msg_policy_notify_t &message)
{
    if (violation.entity.entityGroupId != DCGM_FE_GPU)
    {
        return false;
    }

    dcgmPolicyCallbackResponse_t &response = message.response;

    if (violation.entity.entityGroupId != DCGM_FE_GPU)
    {
        return false;
    }

    response.version = dcgmPolicyCallbackResponse_version;
    response.gpuId   = static_cast<unsigned int>(violation.entity.entityId);

    switch (violation.fieldId)
    {
        case DCGM_FI_DEV_GPU_TEMP_CELSIUS:
            response.condition                    = DCGM_POLICY_COND_THERMAL;
            response.val.thermal.timestamp        = violation.timestamp;
            response.val.thermal.thermalViolation = static_cast<unsigned int>(violation.currentValue);
            break;

        case DCGM_FI_DEV_BOARD_POWER_WATTS:
            response.condition                = DCGM_POLICY_COND_POWER;
            response.val.power.timestamp      = violation.timestamp;
            response.val.power.powerViolation = static_cast<unsigned int>(violation.currentValue);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_DEV:
        case DCGM_FI_DEV_ECC_DBE_VOL_TOTAL:
        case DCGM_FI_DEV_ECC_DBE_AGG_DEV:
        case DCGM_FI_DEV_ECC_DBE_AGG_TOTAL:
            response.condition         = DCGM_POLICY_COND_DBE;
            response.val.dbe.timestamp = violation.timestamp;
            response.val.dbe.location  = dcgmPolicyConditionDbe_t::DEVICE;
            response.val.dbe.numerrors = static_cast<unsigned int>(violation.currentValue);
            break;

        case DCGM_FI_DEV_PCIE_REPLAY_TOTAL:
            response.condition         = DCGM_POLICY_COND_PCI;
            response.val.pci.timestamp = violation.timestamp;
            response.val.pci.counter   = static_cast<unsigned int>(violation.currentValue);
            break;

        case DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL:
        case DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL:
        case DCGM_FI_DEV_PAGE_RETIRED_PENDING:
            response.condition         = DCGM_POLICY_COND_MAX_PAGES_RETIRED;
            response.val.mpr.timestamp = violation.timestamp;
            if (violation.fieldId == DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL)
            {
                response.val.mpr.sbepages = static_cast<unsigned int>(violation.currentValue);
                response.val.mpr.dbepages = 0;
            }
            else if (violation.fieldId == DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL)
            {
                response.val.mpr.sbepages = 0;
                response.val.mpr.dbepages = static_cast<unsigned int>(violation.currentValue);
            }
            else
            {
                // DCGM_FI_DEV_PAGE_RETIRED_PENDING provides only a total; split evenly (odd remainder -> sbe).
                // TODO(dcgm-docs): note this approximation in the policy API reference.
                auto const total          = static_cast<unsigned int>(violation.currentValue);
                response.val.mpr.sbepages = (total + 1) / 2; // ceiling half
                response.val.mpr.dbepages = total / 2;       // floor half
            }
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL:
            response.condition            = DCGM_POLICY_COND_NVLINK;
            response.val.nvlink.timestamp = violation.timestamp;
            response.val.nvlink.fieldId   = static_cast<unsigned short>(violation.fieldId);
            response.val.nvlink.counter   = static_cast<unsigned int>(violation.currentValue);
            break;

        case DCGM_FI_DEV_XID_ERROR:
            response.condition         = DCGM_POLICY_COND_XID;
            response.val.xid.timestamp = violation.timestamp;
            response.val.xid.errnum    = static_cast<unsigned int>(violation.currentValue);
            break;

        default:
            log_warning("No legacy condition mapping for fieldId {} - dropping notification", violation.fieldId);
            return false;
    }

    return true;
}

/*****************************************************************************/
std::expected<DcgmPolicyCallbackRAII, dcgmReturn_t> DcgmPolicyNotificationRouter::RegisterCallback(
    dcgm_connection_id_t connectionId,
    dcgm_request_id_t requestId)
{
    return RegisterCallbackImpl(connectionId, requestId, 0, 0, DcgmPolicyCallbackPayload::Legacy);
}

/*****************************************************************************/
std::expected<DcgmPolicyCallbackRAII, dcgmReturn_t> DcgmPolicyNotificationRouter::RegisterViolationCallback(
    dcgm_connection_id_t connectionId,
    dcgm_request_id_t requestId,
    unsigned short fieldId,
    uint64_t policyId)
{
    return RegisterCallbackImpl(connectionId, requestId, fieldId, policyId, DcgmPolicyCallbackPayload::FieldViolation);
}

/*****************************************************************************/
std::expected<DcgmPolicyCallbackRAII, dcgmReturn_t> DcgmPolicyNotificationRouter::RegisterCallbackImpl(
    dcgm_connection_id_t connectionId,
    dcgm_request_id_t requestId,
    unsigned short fieldId,
    uint64_t policyId,
    DcgmPolicyCallbackPayload payloadKind)
{
    // emplace_back instead of push_back — DcgmPolicyCallbackInfo is not movable (contains std::atomic).
    auto &callbackInfo = m_callbacks[connectionId].emplace_back(requestId, fieldId, policyId, payloadKind);

    log_debug("Registered callback for connectionId {} requestId {} fieldId {} policyId {} payloadKind {}",
              connectionId,
              requestId,
              fieldId,
              policyId,
              static_cast<int>(payloadKind));
    return std::expected<DcgmPolicyCallbackRAII, dcgmReturn_t>(std::in_place, callbackInfo);
}

/*****************************************************************************/
std::vector<dcgm_policy_violation_t> DcgmPolicyNotificationRouter::GetRecentViolations(
    std::chrono::steady_clock::time_point since) const
{
    std::vector<dcgm_policy_violation_t> out;
    out.reserve(m_recentViolations.buffer.capacity());
    (void)GetRecentViolations(std::back_inserter(out), since);
    return out;
}

/*****************************************************************************/
std::vector<dcgm_policy_violation_t> DcgmPolicyNotificationRouter::GetRecentViolations(size_t maxCount) const
{
    std::vector<dcgm_policy_violation_t> out;
    out.reserve(std::min(maxCount, m_recentViolations.buffer.capacity()));
    (void)GetRecentViolations(std::back_inserter(out), maxCount);
    return out;
}
