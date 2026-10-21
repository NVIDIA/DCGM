/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "Policy.h"

#include "CommandOutputController.h"
#include "DcgmiOutput.h"

#include <DcgmPolicyUtils.h>
#include <DcgmStringHelpers.h>
#include <TimeLib.hpp>
#include <dcgm_agent.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <fmt/format.h>
#include <json/json.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <ctype.h>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <string_view>
#include <thread>
#include <utility>

/***************************************************************************************/

/* Get Policy */
const std::string POLICY_HEADER  = "Policy Information";
const std::string NOT_APPLICABLE = "****";
const std::string NONE           = "None";
const std::string POLICY_UNKNOWN = "UNKNOWN";

namespace
{
constexpr char JSON_KEY_POLICY_ID[]            = "policyId";
constexpr char JSON_KEY_FIELD_ID[]             = "fieldId";
constexpr char JSON_KEY_FIELD_NAME[]           = "fieldName";
constexpr char JSON_KEY_CURRENT_VALUE[]        = "currentValue";
constexpr char JSON_KEY_THRESHOLD[]            = "threshold";
constexpr char JSON_KEY_OPERATOR[]             = "operator";
constexpr char JSON_KEY_CHANNELS[]             = "channels";
constexpr char TEXT_POLICY_MONITORING_STATUS[] = "Policy Monitoring Status";
constexpr char TEXT_POLICY_MODIFIED[]          = "Policy modified successfully";
constexpr char TEXT_POLICY_DELETED[]           = "Policy deleted successfully";
constexpr char LABEL_FIELD_ID[]                = "  Field ID:       ";
constexpr char LABEL_FIELD_NAME[]              = "  Field Name:     ";
constexpr char LABEL_POLICY_ID[]               = "  Policy ID:      ";
constexpr char LABEL_POLICY_NAME[]             = "  Policy Name:    ";
constexpr char LABEL_THRESHOLD[]               = "  Threshold:      ";
constexpr char LABEL_TARGET_GPUS[]             = "  Target GPUs:    ";
constexpr char LABEL_CHANNELS[]                = "  Channels:       ";
constexpr char LABEL_RATE_LIMIT[]              = "  Rate Limit:     ";
constexpr char TABLE_FIELD[]                   = "FIELD";
constexpr char TABLE_FIELD_ID[]                = "FIELD ID";
constexpr char TABLE_NAME[]                    = "NAME";
constexpr char TABLE_OPERATOR[]                = "OPERATOR";
constexpr char TABLE_THRESHOLD[]               = "THRESHOLD";
constexpr char TABLE_GPUS[]                    = "GPUS";
constexpr char TABLE_STATUS[]                  = "STATUS";
constexpr char TABLE_TIMESTAMP[]               = "TIMESTAMP";
constexpr char TABLE_GPU[]                     = "GPU";
constexpr char TABLE_VALUE[]                   = "VALUE";
constexpr char TABLE_POLICY[]                  = "POLICY";

std::string PolicyOperatorToString(dcgmPolicyOperator_t policyOperator)
{
    if (auto const operatorString = DcgmPolicyUtils::OperatorToString(policyOperator); operatorString.has_value())
    {
        return std::string(*operatorString);
    }
    return POLICY_UNKNOWN;
}

std::string PolicyStateToString(dcgmPolicyState_t state)
{
    return state == DCGM_POLICY_STATE_ENABLED ? "ENABLED" : "DISABLED";
}

std::string PolicyOperatorDisplay(dcgmPolicyOperator_t policyOperator)
{
    auto const symbol = DcgmPolicyUtils::OperatorToSymbol(policyOperator);
    auto const name   = DcgmPolicyUtils::OperatorToString(policyOperator);
    if (!symbol.has_value() || !name.has_value())
    {
        return POLICY_UNKNOWN;
    }

    if (policyOperator == DCGM_POLICY_OP_CHANGED)
    {
        return std::string(*name);
    }

    return fmt::format("{} ({})", *symbol, *name);
}

std::string PolicyChannelToString(dcgmPolicyChannel_t channel)
{
    if (auto const channelString = DcgmPolicyUtils::ChannelToString(channel); channelString.has_value())
    {
        return std::string(*channelString);
    }
    return POLICY_UNKNOWN;
}

std::string PolicyFieldName(unsigned short fieldId)
{
    if (auto const fieldMeta = DcgmFieldGetById(fieldId); fieldMeta != nullptr)
    {
        return fieldMeta->tag;
    }

    return POLICY_UNKNOWN;
}

Json::Value ChannelsToJson(unsigned int channelMask)
{
    Json::Value channels(Json::arrayValue);
    if ((channelMask & DCGM_POLICY_CHANNEL_CONSOLE) != 0)
    {
        channels.append(PolicyChannelToString(DCGM_POLICY_CHANNEL_CONSOLE));
    }
    if ((channelMask & DCGM_POLICY_CHANNEL_FILE) != 0)
    {
        channels.append(PolicyChannelToString(DCGM_POLICY_CHANNEL_FILE));
    }
    if ((channelMask & DCGM_POLICY_CHANNEL_CALLBACK) != 0)
    {
        channels.append(PolicyChannelToString(DCGM_POLICY_CHANNEL_CALLBACK));
    }
    return channels;
}

std::string ChannelsToString(unsigned int channelMask)
{
    std::string channels;

    auto appendChannel = [&channels](dcgmPolicyChannel_t channel) {
        if (!channels.empty())
        {
            channels += ',';
        }
        channels += PolicyChannelToString(channel);
    };

    if ((channelMask & DCGM_POLICY_CHANNEL_CONSOLE) != 0)
    {
        appendChannel(DCGM_POLICY_CHANNEL_CONSOLE);
    }
    if ((channelMask & DCGM_POLICY_CHANNEL_FILE) != 0)
    {
        appendChannel(DCGM_POLICY_CHANNEL_FILE);
    }
    if ((channelMask & DCGM_POLICY_CHANNEL_CALLBACK) != 0)
    {
        appendChannel(DCGM_POLICY_CHANNEL_CALLBACK);
    }

    return channels;
}

std::string EntitiesToString(dcgmPolicyInfo_t const &policy)
{
    if (policy.entityCount == 0)
    {
        return "All";
    }

    std::ostringstream entities;
    auto const entityCount = std::min(policy.entityCount, static_cast<unsigned int>(std::size(policy.entities)));
    for (unsigned int i = 0; i < entityCount; ++i)
    {
        if (i != 0)
        {
            entities << ',';
        }

        auto const &entity = policy.entities[i];
        if (entity.entityGroupId == DCGM_FE_GPU)
        {
            entities << entity.entityId;
        }
        else
        {
            entities << static_cast<unsigned int>(entity.entityGroupId) << ':' << entity.entityId;
        }
    }

    return entities.str();
}

std::string ThresholdToString(dcgmPolicyInfo_t const &policy)
{
    if (policy.policyOperator == DCGM_POLICY_OP_CHANGED)
    {
        return "-";
    }

    std::ostringstream threshold;
    threshold << policy.threshold;
    return threshold.str();
}

std::string FormatTimestamp(timelib64_t timestamp)
{
    if (timestamp == 0)
    {
        return "-";
    }

    auto timestampDisplay = DcgmNs::Timelib::FormatTimestampIso(static_cast<uint64_t>(timestamp));
    timestampDisplay[10]  = ' ';
    return timestampDisplay.substr(0, 19);
}

Json::Value PolicyToJson(dcgmPolicyInfo_t const &policy)
{
    Json::Value entry;
    entry[JSON_KEY_POLICY_ID]       = static_cast<Json::UInt64>(policy.policyId);
    entry[JSON_KEY_FIELD_ID]        = static_cast<Json::UInt>(policy.fieldId);
    entry[JSON_KEY_FIELD_NAME]      = PolicyFieldName(policy.fieldId);
    entry["name"]                   = policy.name;
    entry[JSON_KEY_CURRENT_VALUE]   = policy.currentValue;
    entry[JSON_KEY_THRESHOLD]       = policy.threshold;
    entry[JSON_KEY_OPERATOR]        = PolicyOperatorToString(policy.policyOperator);
    entry[JSON_KEY_CHANNELS]        = ChannelsToJson(policy.channelMask);
    entry["state"]                  = PolicyStateToString(policy.state);
    entry["rateLimitSec"]           = policy.rateLimitSec;
    entry["createdAt"]              = static_cast<Json::Int64>(policy.createdAt);
    entry["violationCount"]         = policy.violationCount;
    entry["lastTriggeredTimestamp"] = static_cast<Json::Int64>(policy.lastTriggeredTimestamp);

    Json::Value entities(Json::arrayValue);
    auto const entityCount = std::min(policy.entityCount, static_cast<unsigned int>(std::size(policy.entities)));
    for (unsigned int i = 0; i < entityCount; ++i)
    {
        Json::Value entity;
        entity["entityGroupId"] = static_cast<Json::UInt>(policy.entities[i].entityGroupId);
        entity["entityId"]      = static_cast<Json::UInt>(policy.entities[i].entityId);
        entities.append(entity);
    }
    entry["entities"] = entities;
    return entry;
}

Json::Value ViolationToJson(dcgm_policy_violation_t const &violation)
{
    Json::Value entry;
    entry[JSON_KEY_POLICY_ID]     = static_cast<Json::UInt64>(violation.policyId);
    entry["policyName"]           = violation.policyName;
    entry[JSON_KEY_FIELD_NAME]    = violation.fieldName;
    entry[JSON_KEY_FIELD_ID]      = static_cast<Json::UInt>(violation.fieldId);
    entry["entityGroupId"]        = static_cast<Json::UInt>(violation.entity.entityGroupId);
    entry["entityId"]             = static_cast<Json::UInt>(violation.entity.entityId);
    entry[JSON_KEY_CURRENT_VALUE] = violation.currentValue;
    entry[JSON_KEY_THRESHOLD]     = violation.thresholdValue;
    entry[JSON_KEY_OPERATOR]      = PolicyOperatorToString(violation.policyOperator);
    entry[JSON_KEY_CHANNELS]      = ChannelsToJson(violation.channelMask);
    entry["timestamp"]            = static_cast<Json::Int64>(violation.timestamp);
    return entry;
}

void PrintJson(Json::Value const &value)
{
    std::cout << value.toStyledString();
}

void PrintJsonLine(Json::Value const &value)
{
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    std::cout << Json::writeString(builder, value) << std::endl;
}

void PrintPolicyText(dcgmPolicyInfo_t const &policy)
{
    std::cout << "Policy Details: " << policy.name << '\n'
              << std::string(72, '=') << "\n\n"
              << "Configuration\n"
              << LABEL_FIELD_ID << policy.fieldId << '\n'
              << LABEL_FIELD_NAME << PolicyFieldName(policy.fieldId) << '\n'
              << LABEL_POLICY_ID << policy.policyId << '\n'
              << LABEL_POLICY_NAME << policy.name << '\n'
              << "  Condition:      " << PolicyOperatorDisplay(policy.policyOperator) << ' '
              << ThresholdToString(policy) << '\n'
              << LABEL_TARGET_GPUS << EntitiesToString(policy) << '\n'
              << "  Status:         " << PolicyStateToString(policy.state) << "\n\n"
              << "Notification Settings\n"
              << LABEL_CHANNELS << ChannelsToString(policy.channelMask) << '\n'
              << LABEL_RATE_LIMIT << policy.rateLimitSec << " seconds\n\n"
              << "Statistics\n"
              << "  Created:        " << FormatTimestamp(policy.createdAt) << '\n'
              << "  Total Violations: " << policy.violationCount << '\n'
              << "  Last Triggered: " << FormatTimestamp(policy.lastTriggeredTimestamp) << std::endl;
}

void PrintPolicies(std::vector<dcgmPolicyInfo_t> const &policies, bool json, std::string_view noPoliciesMessage)
{
    if (json)
    {
        Json::Value output(Json::arrayValue);
        for (auto const &policy : policies)
        {
            output.append(PolicyToJson(policy));
        }
        PrintJson(output);
    }
    else
    {
        DcgmiOutputColumns output;
        DcgmiOutputFieldSelector fieldSelector     = DcgmiOutputFieldSelector().child(TABLE_FIELD);
        DcgmiOutputFieldSelector nameSelector      = DcgmiOutputFieldSelector().child(TABLE_NAME);
        DcgmiOutputFieldSelector operatorSelector  = DcgmiOutputFieldSelector().child(TABLE_OPERATOR);
        DcgmiOutputFieldSelector thresholdSelector = DcgmiOutputFieldSelector().child(TABLE_THRESHOLD);
        DcgmiOutputFieldSelector gpusSelector      = DcgmiOutputFieldSelector().child(TABLE_GPUS);
        DcgmiOutputFieldSelector statusSelector    = DcgmiOutputFieldSelector().child(TABLE_STATUS);

        output.addHeader(TEXT_POLICY_MONITORING_STATUS);
        output.addColumn(9, TABLE_FIELD, fieldSelector);
        output.addColumn(29, TABLE_NAME, nameSelector);
        output.addColumn(22, TABLE_OPERATOR, operatorSelector);
        output.addColumn(13, TABLE_THRESHOLD, thresholdSelector);
        output.addColumn(14, TABLE_GPUS, gpusSelector);
        output.addColumn(12, TABLE_STATUS, statusSelector);

        for (auto const &policy : policies)
        {
            auto &row            = output[std::to_string(policy.policyId)];
            row[TABLE_FIELD]     = static_cast<unsigned int>(policy.fieldId);
            row[TABLE_NAME]      = policy.name;
            row[TABLE_OPERATOR]  = PolicyOperatorDisplay(policy.policyOperator);
            row[TABLE_THRESHOLD] = ThresholdToString(policy);
            row[TABLE_GPUS]      = EntitiesToString(policy);
            row[TABLE_STATUS]    = PolicyStateToString(policy.state);
        }

        std::cout << output.str();
        if (policies.empty())
        {
            std::cout << noPoliciesMessage << std::endl;
        }
    }
}

std::string ViolationRowKey(dcgm_policy_violation_t const &violation)
{
    return fmt::format("{}-{}-{}-{}",
                       violation.timestamp,
                       violation.policyId,
                       static_cast<unsigned int>(violation.entity.entityGroupId),
                       violation.entity.entityId);
}

void PrintViolationsText(std::vector<dcgm_policy_violation_t> const &violations)
{
    DcgmiOutputColumns output;
    DcgmiOutputFieldSelector timestampSelector = DcgmiOutputFieldSelector().child(TABLE_TIMESTAMP);
    DcgmiOutputFieldSelector gpuSelector       = DcgmiOutputFieldSelector().child(TABLE_GPU);
    DcgmiOutputFieldSelector fieldSelector     = DcgmiOutputFieldSelector().child(TABLE_FIELD_ID);
    DcgmiOutputFieldSelector valueSelector     = DcgmiOutputFieldSelector().child(TABLE_VALUE);
    DcgmiOutputFieldSelector thresholdSelector = DcgmiOutputFieldSelector().child(TABLE_THRESHOLD);
    DcgmiOutputFieldSelector policySelector    = DcgmiOutputFieldSelector().child(TABLE_POLICY);

    output.addHeader("Monitoring policy violations (Press Ctrl+C to stop)...");
    output.addColumn(21, TABLE_TIMESTAMP, timestampSelector);
    output.addColumn(9, TABLE_GPU, gpuSelector);
    output.addColumn(10, TABLE_FIELD_ID, fieldSelector);
    output.addColumn(13, TABLE_VALUE, valueSelector);
    output.addColumn(13, TABLE_THRESHOLD, thresholdSelector);
    output.addColumn(26, TABLE_POLICY, policySelector);

    for (auto const &violation : violations)
    {
        auto &row            = output[ViolationRowKey(violation)];
        row[TABLE_TIMESTAMP] = FormatTimestamp(violation.timestamp);
        row[TABLE_GPU]       = violation.entity.entityId;
        row[TABLE_FIELD_ID]  = static_cast<unsigned int>(violation.fieldId);
        row[TABLE_VALUE]     = violation.currentValue;
        row[TABLE_THRESHOLD] = violation.policyOperator == DCGM_POLICY_OP_CHANGED
                                   ? std::string("-")
                                   : fmt::format("{}", violation.thresholdValue);
        row[TABLE_POLICY]    = violation.policyName;
    }

    std::cout << output.str();
}

void PrintRecentViolationsText(std::vector<dcgm_policy_violation_t> const &violations)
{
    if (violations.empty())
    {
        return;
    }

    DcgmiOutputColumns output;
    DcgmiOutputFieldSelector timestampSelector = DcgmiOutputFieldSelector().child(TABLE_TIMESTAMP);
    DcgmiOutputFieldSelector gpuSelector       = DcgmiOutputFieldSelector().child(TABLE_GPU);
    DcgmiOutputFieldSelector valueSelector     = DcgmiOutputFieldSelector().child(TABLE_VALUE);

    output.addColumn(21, TABLE_TIMESTAMP, timestampSelector);
    output.addColumn(9, TABLE_GPU, gpuSelector);
    output.addColumn(13, TABLE_VALUE, valueSelector);

    for (auto const &violation : violations)
    {
        auto &row            = output[ViolationRowKey(violation)];
        row[TABLE_TIMESTAMP] = FormatTimestamp(violation.timestamp);
        row[TABLE_GPU]       = fmt::format("GPU {}", violation.entity.entityId);
        row[TABLE_VALUE]     = violation.currentValue;
    }

    std::cout << "\nRecent Violations (Last " << violations.size() << ")\n" << output.str();
}
} // namespace

/*****************************************************************************************/

Policy::Policy(DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : m_dcgmOps(dcgmOps)
{}

Policy::~Policy()
{}

/*******************************************************************************************/
dcgmReturn_t Policy::DisplayCurrentViolationPolicy(dcgmHandle_t mNvcmHandle,
                                                   unsigned int groupId,
                                                   bool verbose,
                                                   bool json)
{
    dcgmReturn_t result;
    dcgmPolicy_t *pPolicy;
    std::unique_ptr<dcgmGroupInfo_t> stNvcmGroupInfo = std::make_unique<dcgmGroupInfo_t>();
    dcgmStatus_t stHandle                            = 0;
    GPUErrorOutputController gpuErrView(m_dcgmOps);

    stNvcmGroupInfo->version = dcgmGroupInfo_version;
    result = m_dcgmOps->dcgmGroupGetInfo(mNvcmHandle, (dcgmGpuGrp_t)(long long)groupId, stNvcmGroupInfo.get());
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Cannot get group info from remote node. Return: " << error << std::endl;
        log_error(
            "Error: could not get information for group: {}. Return: {}", (unsigned int)(uintptr_t)groupId, result);
        return result;
    }

    // from here on must branch to cleanup
    pPolicy = (dcgmPolicy_t *)malloc(sizeof(dcgmPolicy_t) * stNvcmGroupInfo->count);
    if (NULL == pPolicy)
    {
        std::cout << "Error: Cannot malloc space for policy info. Return: " << errorString(result) << std::endl;
        result = DCGM_ST_GENERIC_ERROR;
        goto cleanup;
    }

    for (unsigned int i = 0; i < stNvcmGroupInfo->count; i++)
    {
        pPolicy[i].version = dcgmPolicy_version;
    }

    result = m_dcgmOps->dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot create status handler. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }

    result = m_dcgmOps->dcgmPolicyGet(
        mNvcmHandle, (dcgmGpuGrp_t)(long long)groupId, stNvcmGroupInfo->count, pPolicy, stHandle);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot get policy info from remote node. Return: " << errorString(result) << std::endl;
        std::cout << "Errors for group:" << groupId << std::endl;
        gpuErrView.addError(stHandle);
        gpuErrView.display();
        log_error("Error: could not get policy for group: {}. Return: {}", groupId, result);
        goto cleanup;
    }
    else
    {
        std::stringstream ss;
        std::cout << "Policy information" << std::endl;
        bool allTheSame = true;

        for (unsigned int i = 0; i < (!verbose ? 1 : stNvcmGroupInfo->count); i++)
        {
            DcgmiOutputTree outTree(30, 50);
            DcgmiOutputJson outJson;
            DcgmiOutput &out = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;

            ss.str(""); // clear stream;

            // Display header
            if (!verbose)
            {
                ss << stNvcmGroupInfo->groupName;
            }
            else
            {
                ss << DcgmFieldsGetEntityGroupString(stNvcmGroupInfo->entityList[i].entityGroupId)
                   << " ID: " << stNvcmGroupInfo->entityList[i].entityId;
            }

            out.addHeader(POLICY_HEADER);
            out.addHeader(ss.str());

            // Violation conditions
            if (!verbose)
            {
                allTheSame = true;
                for (unsigned int j = 1; j < stNvcmGroupInfo->count; j++)
                {
                    if (pPolicy[0].condition != pPolicy[j].condition)
                    {
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame)
            {
                out["Violation conditions"] = NOT_APPLICABLE;
            }
            else
            {
                if (pPolicy[i].condition == 0)
                {
                    out["Violation conditions"] = NONE;
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_DBE)
                {
                    out["Violation conditions"].setOrAppend("Double-bit ECC errors");
                }
                if (pPolicy[i].condition & DCGM_POLICY_COND_PCI)
                {
                    out["Violation conditions"].setOrAppend("PCI errors and replays");
                }
                if (pPolicy[i].condition & DCGM_POLICY_COND_MAX_PAGES_RETIRED)
                {
                    ss.str("");
                    ss << "Max retired pages threshold" << " - " << (long long unsigned)pPolicy[i].parms[2].val.llval;

                    out["Violation conditions"].setOrAppend(ss.str());
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_THERMAL)
                {
                    ss.str("");
                    ss << "Max temperature threshold" << " - " << (long long unsigned)pPolicy[i].parms[3].val.llval;
                    out["Violation conditions"].setOrAppend(ss.str());
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_POWER)
                {
                    ss.str("");
                    ss << "Max power threshold" << " - " << (long long unsigned)pPolicy[i].parms[4].val.llval;
                    out["Violation conditions"].setOrAppend(ss.str());
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_NVLINK)
                {
                    out["Violation conditions"].setOrAppend("NVLink Errors");
                }

                if (pPolicy[i].condition & DCGM_POLICY_COND_XID)
                {
                    out["Violation conditions"].setOrAppend("XID error detected.");
                }
            }

            // Isolation Mode
            if (!verbose)
            {
                allTheSame = true;
                for (unsigned int j = 1; j < stNvcmGroupInfo->count; j++)
                {
                    if (pPolicy[0].mode != pPolicy[j].mode)
                    {
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame)
            {
                out["Isolation mode"] = NOT_APPLICABLE;
            }
            else
            {
                ss.str(""); // Reset Stringstream
                if (pPolicy[i].mode == DCGM_POLICY_MODE_AUTOMATED)
                    ss << "Automatic";
                else if (pPolicy[i].mode == DCGM_POLICY_MODE_MANUAL)
                    ss << "Manual";
                out["Isolation mode"] = ss.str();
            }

            // Action on violation
            if (!verbose)
            {
                allTheSame = true;
                for (unsigned int j = 1; j < stNvcmGroupInfo->count; j++)
                {
                    if (pPolicy[0].action != pPolicy[j].action)
                    {
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame)
            {
                out["Action on violation"] = NOT_APPLICABLE;
            }
            else
            {
                ss.str(""); // Reset Stringstream
                if (pPolicy[i].action == DCGM_POLICY_ACTION_NONE)
                    ss << "None";
                else if (pPolicy[i].action == DCGM_POLICY_ACTION_GPURESET)
                    ss << "Reset GPU";

                out["Action on violation"] = ss.str();
            }

            // Action after validation
            if (!verbose)
            {
                allTheSame = true;
                for (unsigned int j = 1; j < stNvcmGroupInfo->count; j++)
                {
                    if (pPolicy[0].validation != pPolicy[j].validation)
                    {
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame)
            {
                out["Validation after action"] = NOT_APPLICABLE;
            }
            else
            {
                ss.str(""); // Reset Stringstream
                if (pPolicy[i].validation == DCGM_POLICY_VALID_NONE)
                    ss << "None";
                if (pPolicy[i].validation == DCGM_POLICY_VALID_SV_SHORT)
                    ss << "System Validation (Short)";
                if (pPolicy[i].validation == DCGM_POLICY_VALID_SV_MED)
                    ss << "System Validation (Medium)";
                if (pPolicy[i].validation == DCGM_POLICY_VALID_SV_LONG)
                    ss << "System Validation (Long)";
                if (pPolicy[i].validation == DCGM_POLICY_VALID_SV_XLONG)
                    ss << "System Validation (Extended)";


                out["Validation after action"] = ss.str();
            }

            // Validation failure action
            if (!verbose)
            {
                allTheSame = true;
                for (unsigned int j = 1; j < stNvcmGroupInfo->count; j++)
                {
                    if (pPolicy[0].response != pPolicy[j].response)
                    {
                        allTheSame = false;
                        break;
                    }
                }
            }

            if (!verbose && !allTheSame)
            {
                out["Validation failure action"] = NOT_APPLICABLE;
            }
            else
            {
                ss.str(""); // Reset Stringstream
                if (pPolicy[i].response == DCGM_POLICY_FAILURE_NONE)
                    ss << "None";
                out["Validation failure action"] = ss.str();
            }

            std::cout << out.str();

            if (!verbose)
            {
                std::cout << "**** Non-homogenous settings across group. Use with -v flag to see details.\n";
            }
        }
    }

cleanup:
    if (stHandle)
    {
        if (DCGM_ST_OK != m_dcgmOps->dcgmStatusDestroy(stHandle))
            std::cout << "Error: Cannot delete status handler. Return: " << errorString(result) << std::endl;
    }

    if (pPolicy)
        free(pPolicy);

    return result;
}

/*******************************************************************************************/
static std::string policy_dcgm_return_to_string(dcgmReturn_t dcgmReturn)
{
    std::string error;

    switch (dcgmReturn)
    {
        case DCGM_ST_NOT_CONFIGURED:
            error = "The Group is not found";
            break;

        case DCGM_ST_NOT_SUPPORTED:
            error
                = "A GPU in the group does not support policy management. Policy management is only supported on Tesla GPUs.";
            break;

        default:
            error = std::string(errorString(dcgmReturn));
            break;
    }

    return error;
}

/*******************************************************************************************/
dcgmReturn_t Policy::SetCurrentViolationPolicy(dcgmHandle_t mNvcmHandle, unsigned int groupId, dcgmPolicy_t &policy)
{
    dcgmReturn_t result;
    dcgmStatus_t stHandle = 0;

    result = m_dcgmOps->dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot create status handler. Return: " << errorString(result) << std::endl;
        return result;
    }

    result = m_dcgmOps->dcgmPolicySet(mNvcmHandle, (dcgmGpuGrp_t)(long long)groupId, &policy, stHandle);
    if (DCGM_ST_OK != result)
    {
        /* Look at status to get individual errors */
        GPUErrorOutputController gpuErrView(m_dcgmOps);
        std::string error = policy_dcgm_return_to_string(result);
        std::cout << "Error: Cannot set policy for remote node. Return: " << error << std::endl;
        std::cout << "Errors for group:" << std::endl;
        gpuErrView.addError(stHandle);
        gpuErrView.display();

        log_error("Error: could not set policy for group: {}. Return: {}", groupId, result);

        goto cleanup;
    }
    else
        std::cout << "Policy successfully set." << std::endl;

cleanup:

    if (stHandle)
    {
        if (DCGM_ST_OK != m_dcgmOps->dcgmStatusDestroy(stHandle))
            std::cout << "Error: Cannot delete status handler. Return: " << errorString(result) << std::endl;
    }

    return result;
}

/*******************************************************************************************/
dcgmReturn_t Policy::RegisterForPolicyUpdates(dcgmHandle_t mNvcmHandle, unsigned int groupId, unsigned int condition)
{
    dcgmReturn_t result = DCGM_ST_OK;

    // Set condition to all available watches
    condition = (DCGM_POLICY_COND_DBE | DCGM_POLICY_COND_PCI | DCGM_POLICY_COND_MAX_PAGES_RETIRED
                 | DCGM_POLICY_COND_THERMAL | DCGM_POLICY_COND_POWER | DCGM_POLICY_COND_NVLINK | DCGM_POLICY_COND_XID);

    result = m_dcgmOps->dcgmPolicyRegister_v2(
        mNvcmHandle, (dcgmGpuGrp_t)(long long)groupId, (dcgmPolicyCondition_t)condition, &ListenForViolations, 0);
    if (DCGM_ST_OK != result)
    {
        std::string error = policy_dcgm_return_to_string(result);

        std::cout << "Error: Cannot register to receive policy violations from the remote node. Return: " << error
                  << std::endl;
        log_error("Error: could not register for policy updates for group: {}. Return: {}", groupId, result);
        return result;
    }
    std::cout << "Listening for violations.\n";
    while (true)
    {
        // go into an infinite loop... this is expected until the user uses ctrl-c to exit
        // where the signal handler will unregister
        sleep(10);
    }

    return result;
}

/*******************************************************************************************/
dcgmReturn_t Policy::UnregisterPolicyUpdates(dcgmHandle_t mNvcmHandle, unsigned int groupId, unsigned int condition)
{
    dcgmReturn_t result = DCGM_ST_OK;

    result = m_dcgmOps->dcgmPolicyUnregister(
        mNvcmHandle, (dcgmGpuGrp_t)(long long)groupId, (dcgmPolicyCondition_t)condition);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot unregister to receive policy violations from the remote node. Return: "
                  << errorString(result) << std::endl;
        log_error("Error: could not unregister for policy updates for group: {}. Return: {}", groupId, result);
        return result;
    }
    return result;
}

/*******************************************************************************************/
int Policy::ListenForViolations(dcgmPolicyCallbackResponse_t *callbackResponse, [[maybe_unused]] uint64_t userData)
{
    switch (callbackResponse->condition)
    {
        case DCGM_POLICY_COND_DBE:
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.pci.timestamp) << std::endl;
            std::cout << "A double-bit ECC error has violated policy manager values." << std::endl;
            std::cout << "DBE error count: " << callbackResponse->val.pci.counter << std::endl;
            break;
        case DCGM_POLICY_COND_PCI:
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.pci.timestamp) << std::endl;
            std::cout << "A PCIe replay event has violated policy manager values." << std::endl;
            std::cout << "PCIe replay count: " << callbackResponse->val.pci.counter << std::endl;
            break;
        case DCGM_POLICY_COND_MAX_PAGES_RETIRED:
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.mpr.timestamp) << std::endl;
            std::cout << "The maximum number of retired pages has violated policy manager values." << std::endl;
            std::cout << "SBE page retirement count: " << callbackResponse->val.mpr.sbepages << std::endl;
            std::cout << "DBE page retirement count: " << callbackResponse->val.mpr.dbepages << std::endl;
            break;
        case DCGM_POLICY_COND_THERMAL:
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.thermal.timestamp) << std::endl;
            std::cout << "The maximum thermal limit has violated policy manager values." << std::endl;
            std::cout << "Temperature: " << callbackResponse->val.thermal.thermalViolation << std::endl;
            break;
        case DCGM_POLICY_COND_POWER:
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.power.timestamp) << std::endl;
            std::cout << "The maximum power limit has violated policy manager values." << std::endl;
            std::cout << "Power: " << callbackResponse->val.power.powerViolation << std::endl;
            break;
        case DCGM_POLICY_COND_NVLINK:
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.nvlink.timestamp) << std::endl;
            std::cout << "NVLink Counter " << callbackResponse->val.nvlink.fieldId
                      << " has violated policy manager values." << std::endl;
            std::cout << "NVLink Counter value: " << callbackResponse->val.nvlink.counter << std::endl;
            break;
        case DCGM_POLICY_COND_XID:
            std::cout << "Timestamp: " << HelperFormatTimestamp(callbackResponse->val.xid.timestamp) << std::endl;
            std::cout << "XID error " << callbackResponse->val.xid.errnum << " detected." << std::endl;
            break;

        default: // unknown
            std::cout << "An unknown error has violated policy manager values." << std::endl;
            break;
    }

    fflush(stdout);

    return 0;
}


/***************************************************************************************/
std::string Policy::HelperFormatTimestamp(long long timestamp)
{
    std::stringstream ss;

    if (DCGM_INT64_IS_BLANK(timestamp))
    {
        switch (timestamp)
        {
            case DCGM_INT64_BLANK:
                return "Not specified";

            case DCGM_INT64_NOT_FOUND:
                return "Not found";

            case DCGM_INT64_NOT_SUPPORTED:
                return "Not supported";

            case DCGM_INT64_NOT_PERMISSIONED:
                return "Insufficient permission";
            default:
                return "Not specified";
        }
    }

    long long temp  = timestamp / 1000000;
    std::string str = ctime((long *)&temp);

    // Remove returned next line character
    str = str.substr(0, str.length() - 1);

    ss << str; //<< ":" << std::setw(4) << std::setfill('0') <<timestamp % 1000000;

    return ss.str();
}

/*****************************************************************************/
dcgmReturn_t Policy::RunFieldPolicyCreate(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options, bool json)
{
    uint64_t policyId = 0;
    auto const ret    = m_dcgmOps->dcgmPolicyCreate(dcgmHandle,
                                                    options.fieldId,
                                                    options.name.c_str(),
                                                    options.threshold,
                                                    options.policyOperator,
                                                    options.entities.empty() ? nullptr : options.entities.data(),
                                                    static_cast<unsigned int>(options.entities.size()),
                                                    options.channelMask,
                                                    options.enabled ? 1 : 0,
                                                    options.rateLimitSec,
                                                    &policyId);
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot create field policy. Return: " << errorString(ret) << std::endl;
        return ret;
    }
    if (json)
    {
        Json::Value output;
        output["status"]           = "created";
        output[JSON_KEY_POLICY_ID] = static_cast<Json::UInt64>(policyId);
        PrintJson(output);
    }
    else
    {
        std::cout << "Policy created successfully\n";
        dcgmPolicyInfo_t created {};
        created.version = dcgmPolicyInfo_version;
        if (m_dcgmOps->dcgmPolicyGetOne(dcgmHandle, options.fieldId, policyId, &created) == DCGM_ST_OK)
        {
            std::cout << "  Field ID:    " << created.fieldId << '\n'
                      << "  Name:        " << created.name << '\n'
                      << "  Operator:    " << PolicyOperatorDisplay(created.policyOperator) << '\n'
                      << "  Threshold:   " << ThresholdToString(created) << '\n'
                      << "  GPUs:        " << EntitiesToString(created) << '\n'
                      << "  Channels:    " << ChannelsToString(created.channelMask) << '\n'
                      << "  Rate Limit:  " << created.rateLimitSec << " seconds\n"
                      << "  Status:      " << PolicyStateToString(created.state) << std::endl;
        }
        else
        {
            std::cout << "  Policy ID: " << policyId << std::endl;
        }
    }
    return DCGM_ST_OK;
}

dcgmReturn_t Policy::RunFieldPolicyList(dcgmHandle_t dcgmHandle, bool json)
{
    std::vector<dcgmPolicyInfo_t> policies;
    auto const ret = GetAllFieldPolicies(dcgmHandle, policies);
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot list field policies. Return: " << errorString(ret) << std::endl;
        return ret;
    }

    PrintPolicies(policies, json, "No field policies configured");
    return DCGM_ST_OK;
}

dcgmReturn_t Policy::GetAllFieldPolicies(dcgmHandle_t dcgmHandle, std::vector<dcgmPolicyInfo_t> &policies)
{
    policies.assign(DCGM_POLICY_MAX_POLICIES, dcgmPolicyInfo_t {});
    for (auto &policy : policies)
    {
        policy.version = dcgmPolicyInfo_version;
    }
    unsigned int count = static_cast<unsigned int>(policies.size());
    auto const ret     = m_dcgmOps->dcgmPolicyGetAll(dcgmHandle, policies.data(), &count);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    policies.resize(count);
    return DCGM_ST_OK;
}

void Policy::PrintFieldPolicyDetails(dcgmHandle_t dcgmHandle, dcgmPolicyInfo_t const &policy)
{
    PrintPolicyText(policy);

    std::vector<dcgm_policy_violation_t> violations(DCGM_POLICY_MAX_VIOLATIONS);
    unsigned int violationCount = static_cast<unsigned int>(violations.size());
    auto const violationsRet    = m_dcgmOps->dcgmPolicyGetViolations(dcgmHandle, 0, violations.data(), &violationCount);
    if (violationsRet != DCGM_ST_OK)
    {
        return;
    }

    violations.resize(violationCount);
    std::erase_if(violations, [&policy](dcgm_policy_violation_t const &violation) {
        return violation.fieldId != policy.fieldId || violation.policyId != policy.policyId;
    });
    std::sort(violations.begin(),
              violations.end(),
              [](dcgm_policy_violation_t const &left, dcgm_policy_violation_t const &right) {
                  return left.timestamp > right.timestamp;
              });

    if (!violations.empty())
    {
        auto const symbol = DcgmPolicyUtils::OperatorToSymbol(policy.policyOperator);
        if (symbol.has_value() && policy.policyOperator != DCGM_POLICY_OP_CHANGED)
        {
            auto const &lastViolation = violations.front();
            std::cout << "                  GPU " << lastViolation.entity.entityId << ": " << lastViolation.currentValue
                      << ' ' << *symbol << ' ' << lastViolation.thresholdValue << '\n';
        }
    }

    if (violations.size() > 5)
    {
        violations.resize(5);
    }
    PrintRecentViolationsText(violations);
}

dcgmReturn_t Policy::RunFieldPolicyGet(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options, bool json)
{
    if (options.hasPolicyId)
    {
        dcgmPolicyInfo_t policy {};
        policy.version = dcgmPolicyInfo_version;
        auto const ret = m_dcgmOps->dcgmPolicyGetOne(dcgmHandle, options.fieldId, options.policyId, &policy);
        if (ret != DCGM_ST_OK)
        {
            std::cout << "Error: Cannot get field policy. Return: " << errorString(ret) << std::endl;
            return ret;
        }
        if (json)
        {
            PrintJson(PolicyToJson(policy));
        }
        else
        {
            PrintFieldPolicyDetails(dcgmHandle, policy);
        }
        return DCGM_ST_OK;
    }

    std::vector<dcgmPolicyInfo_t> policies;
    auto const ret = GetAllFieldPolicies(dcgmHandle, policies);
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot get field policies. Return: " << errorString(ret) << std::endl;
        return ret;
    }

    std::vector<dcgmPolicyInfo_t> matchingPolicies;
    matchingPolicies.reserve(policies.size());
    std::copy_if(policies.begin(),
                 policies.end(),
                 std::back_inserter(matchingPolicies),
                 [&options](dcgmPolicyInfo_t const &policy) { return policy.fieldId == options.fieldId; });
    if (matchingPolicies.empty())
    {
        std::cout << "No field policies configured for field ID " << options.fieldId << std::endl;
        return DCGM_ST_OK;
    }

    if (!json && matchingPolicies.size() == 1)
    {
        PrintFieldPolicyDetails(dcgmHandle, matchingPolicies.front());
    }
    else
    {
        PrintPolicies(matchingPolicies, json, "");
    }
    return DCGM_ST_OK;
}

dcgmReturn_t Policy::RunFieldPolicyModify(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options)
{
    dcgmPolicyInfo_t policy {};
    policy.version = dcgmPolicyInfo_version;
    auto ret       = m_dcgmOps->dcgmPolicyGetOne(dcgmHandle, options.fieldId, options.policyId, &policy);
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot get field policy for modification. Return: " << errorString(ret) << std::endl;
        return ret;
    }

    if (options.hasFieldId && policy.fieldId != options.fieldId)
    {
        std::cout << "Error: Cannot modify field ID for an existing field policy." << std::endl;
        return DCGM_ST_BADPARAM;
    }
    if (options.hasName)
    {
        SafeCopyTo(policy.name, options.name.c_str());
    }
    if (options.hasThreshold)
    {
        policy.threshold = options.threshold;
    }
    if (options.hasOperator)
    {
        policy.policyOperator = options.policyOperator;
    }
    if (options.hasEntities)
    {
        if (options.entities.size() > std::size(policy.entities))
        {
            std::cout << "Error: Too many entities for field policy." << std::endl;
            return DCGM_ST_BADPARAM;
        }
        auto const entityCount = static_cast<unsigned int>(options.entities.size());
        policy.entityCount     = entityCount;
        for (unsigned int i = 0; i < entityCount; ++i)
        {
            policy.entities[i] = options.entities[i];
        }
    }
    if (options.hasChannels)
    {
        policy.channelMask = options.channelMask;
    }
    if (options.hasRateLimit)
    {
        policy.rateLimitSec = options.rateLimitSec;
    }

    ret = m_dcgmOps->dcgmPolicyModify(dcgmHandle, &policy);
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot modify field policy. Return: " << errorString(ret) << std::endl;
        return ret;
    }

    std::cout << TEXT_POLICY_MODIFIED << '\n' << LABEL_FIELD_ID << policy.fieldId << '\n';
    if (options.hasName)
    {
        std::cout << LABEL_POLICY_NAME << policy.name << '\n';
    }
    if (options.hasOperator)
    {
        std::cout << "  Operator:       " << PolicyOperatorDisplay(policy.policyOperator) << '\n';
    }
    if (options.hasThreshold)
    {
        std::cout << LABEL_THRESHOLD << ThresholdToString(policy) << '\n';
    }
    if (options.hasEntities)
    {
        std::cout << LABEL_TARGET_GPUS << EntitiesToString(policy) << '\n';
    }
    if (options.hasChannels)
    {
        std::cout << LABEL_CHANNELS << ChannelsToString(policy.channelMask) << '\n';
    }
    if (options.hasRateLimit)
    {
        std::cout << LABEL_RATE_LIMIT << policy.rateLimitSec << " seconds\n";
    }

    return DCGM_ST_OK;
}

dcgmReturn_t Policy::RunFieldPolicyDelete(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options)
{
    auto const ret
        = m_dcgmOps->dcgmPolicyDelete(dcgmHandle, options.fieldId, options.policyId, options.deleteAll ? 1 : 0);
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot delete field policy. Return: " << errorString(ret) << std::endl;
        return ret;
    }
    std::cout << TEXT_POLICY_DELETED << '\n';
    if (options.deleteAll)
    {
        std::cout << "  Scope:    all field policies\n";
    }
    else
    {
        std::cout << LABEL_FIELD_ID << options.fieldId << '\n';
    }

    return DCGM_ST_OK;
}

dcgmReturn_t Policy::RunFieldPolicyEnableDisable(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options)
{
    auto const ret = options.action == FieldPolicyAction::Enable
                         ? m_dcgmOps->dcgmPolicyEnable(dcgmHandle, options.fieldId, options.policyId)
                         : m_dcgmOps->dcgmPolicyDisable(dcgmHandle, options.fieldId, options.policyId);
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot " << (options.action == FieldPolicyAction::Enable ? "enable" : "disable")
                  << " field policy. Return: " << errorString(ret) << std::endl;
        return ret;
    }
    std::cout << "Policy " << (options.action == FieldPolicyAction::Enable ? "enabled" : "disabled") << "\n"
              << LABEL_FIELD_ID << options.fieldId << std::endl;
    return DCGM_ST_OK;
}

dcgmReturn_t Policy::RunFieldPolicyImport(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options)
{
    auto const ret = m_dcgmOps->dcgmPolicyImport(dcgmHandle, options.importPath.c_str());
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot import field policies. Return: " << errorString(ret) << std::endl;
        return ret;
    }
    std::cout << "Policies imported successfully" << std::endl;
    return DCGM_ST_OK;
}

dcgmReturn_t Policy::RunFieldPolicyWatch(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options, bool json)
{
    timelib64_t sinceTimestamp = options.sinceTimestamp;
    do
    {
        std::vector<dcgm_policy_violation_t> violations(options.maxViolations);
        unsigned int count = static_cast<unsigned int>(violations.size());
        auto const ret     = m_dcgmOps->dcgmPolicyGetViolations(dcgmHandle, sinceTimestamp, violations.data(), &count);
        if (ret != DCGM_ST_OK)
        {
            std::cout << "Error: Cannot get field policy violations. Return: " << errorString(ret) << std::endl;
            return ret;
        }

        violations.resize(count);

        // Track the highest timestamp seen in the raw batch. If the field filter removes every
        // violation, sinceTimestamp would otherwise stay unchanged and the next poll would
        // re-fetch the same batch indefinitely. In that case we advance to batchCursor instead.
        timelib64_t batchCursor = sinceTimestamp;
        for (auto const &violation : violations)
        {
            batchCursor = std::max(batchCursor, violation.timestamp + 1);
        }

        if (options.hasFieldId)
        {
            std::erase_if(violations, [&options](dcgm_policy_violation_t const &violation) {
                return violation.fieldId != options.fieldId;
            });
        }

        if (violations.empty() && !options.watchOnce)
        {
            // No matching violations in this batch; advance past the whole batch to prevent an
            // infinite re-fetch loop, then continue polling.
            sinceTimestamp = batchCursor;
        }
        else
        {
            for (auto const &violation : violations)
            {
                sinceTimestamp = std::max(sinceTimestamp, violation.timestamp + 1);
            }
        }

        count = static_cast<unsigned int>(violations.size());
        if (json)
        {
            if (options.watchOnce)
            {
                Json::Value output(Json::arrayValue);
                for (unsigned int i = 0; i < count; ++i)
                {
                    output.append(ViolationToJson(violations[i]));
                }
                PrintJson(output);
            }
            else
            {
                for (unsigned int i = 0; i < count; ++i)
                {
                    PrintJsonLine(ViolationToJson(violations[i]));
                }
            }
        }
        else
        {
            if (count != 0)
            {
                PrintViolationsText(violations);
            }
            if (count == 0 && options.watchOnce)
            {
                std::cout << "No field policy violations" << std::endl;
            }
        }

        if (!options.watchOnce)
        {
            std::this_thread::sleep_for(std::chrono::seconds(options.watchIntervalSec));
        }
    } while (!options.watchOnce);

    return DCGM_ST_OK;
}

dcgmReturn_t Policy::RunFieldPolicyCommand(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options, bool json)
{
    switch (options.action)
    {
        case FieldPolicyAction::Create:
            return RunFieldPolicyCreate(dcgmHandle, options, json);
        case FieldPolicyAction::List:
            return RunFieldPolicyList(dcgmHandle, json);
        case FieldPolicyAction::Get:
            return RunFieldPolicyGet(dcgmHandle, options, json);
        case FieldPolicyAction::Modify:
            return RunFieldPolicyModify(dcgmHandle, options);
        case FieldPolicyAction::Delete:
            return RunFieldPolicyDelete(dcgmHandle, options);
        case FieldPolicyAction::Enable:
        case FieldPolicyAction::Disable:
            return RunFieldPolicyEnableDisable(dcgmHandle, options);
        case FieldPolicyAction::Import:
            return RunFieldPolicyImport(dcgmHandle, options);
        case FieldPolicyAction::Watch:
            return RunFieldPolicyWatch(dcgmHandle, options, json);
    }

    return DCGM_ST_BADPARAM;
}


/*****************************************************************************
 *****************************************************************************
 *Get Policy Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetPolicy::GetPolicy(std::string hostname, unsigned int groupId, bool verbose, bool json, DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : Command(dcgmOps)
    , policyObj(dcgmOps)
{
    m_hostName    = std::move(hostname);
    m_json        = json;
    this->groupId = groupId;
    this->verbose = verbose;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t GetPolicy::DoExecuteConnected()
{
    return policyObj.DisplayCurrentViolationPolicy(m_dcgmHandle, groupId, verbose, m_json);
}


/*****************************************************************************
 *****************************************************************************
 *Set Policy Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
SetPolicy::SetPolicy(std::string hostname, dcgmPolicy_t &setPolicy, unsigned int groupId, DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : Command(dcgmOps)
    , policyObj(dcgmOps)
{
    m_hostName      = std::move(hostname);
    this->setPolicy = setPolicy;
    this->groupId   = groupId;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t SetPolicy::DoExecuteConnected()
{
    return policyObj.SetCurrentViolationPolicy(m_dcgmHandle, groupId, setPolicy);
}


/*****************************************************************************
 *****************************************************************************
 *Get Policy Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
RegPolicy::RegPolicy(std::string hostname, unsigned int groupId, unsigned int condition, DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : Command(dcgmOps)
    , policyObj(dcgmOps)
{
    m_hostName      = std::move(hostname);
    this->groupId   = groupId;
    this->condition = condition;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
RegPolicy::~RegPolicy()
{
    policyObj.UnregisterPolicyUpdates(m_dcgmHandle, groupId, condition);
}

/*****************************************************************************/
dcgmReturn_t RegPolicy::DoExecuteConnected()
{
    return policyObj.RegisterForPolicyUpdates(m_dcgmHandle, groupId, condition);
}

/*****************************************************************************
 *****************************************************************************
 * Field Policy Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldPolicy::FieldPolicy(std::string hostname, FieldPolicyOptions options, bool json, DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : Command(dcgmOps)
    , policyObj(dcgmOps)
    , m_options(std::move(options))
{
    m_hostName = std::move(hostname);
    m_json     = json;

    /* We want policy watches/configuration to persist once this DCGMI instance exits. */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t FieldPolicy::DoExecuteConnected()
{
    return policyObj.RunFieldPolicyCommand(m_dcgmHandle, m_options, m_json);
}
