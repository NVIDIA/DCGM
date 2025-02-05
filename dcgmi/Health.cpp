/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
/*
 * Health.cpp
 *
 *  Created on: Oct 6, 2015
 *      Author: chris
 */

#include "Health.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <utility>

/***************************************************************************/

/* Get Watches */
#define ENT_GROUP_TAG "<EGRP"
#define ENT_ID_TAG    "<EID"

#define OUTPUT_WIDTH            80
#define OUTPUT_FIELD_NAME_WIDTH 12

#define PCIE_SYSTEMS_TAG              "PCIe"
#define NVLINK_SYSTEMS_TAG            "NVLINK"
#define PMU_SYSTEMS_TAG               "PMU"
#define MCU_SYSTEMS_TAG               "MCU"
#define MEMORY_SYSTEMS_TAG            "Memory"
#define SM_SYSTEMS_TAG                "SM"
#define INFOROM_SYSTEMS_TAG           "InfoROM"
#define THERMAL_SYSTEMS_TAG           "Thermal"
#define POWER_SYSTEMS_TAG             "Power"
#define DRIVER_SYSTEMS_TAG            "Driver"
#define NVSWITCH_NONFATAL_SYSTEMS_TAG "NvSwitch NF"
#define NVSWITCH_FATAL_SYSTEMS_TAG    "NvSwitch F"
#define OVERALL_HEALTH_TAG            "Overall Health"

#define MAX_SIZE_OF_HEALTH_INFO 54 /* Used for overflow (full length of health information tag) */

/*****************************************************************************************/

template <typename T>
std::string to_string(const T &t)
{
    std::ostringstream ss;
    ss << t;
    return ss.str();
}

/*****************************************************************************/
dcgmReturn_t Health::GetWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthSystems_t systems;
    DcgmiOutputTree outTree(18, 70);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;
    std::string on   = "On";
    std::string off  = "Off";

    result = dcgmHealthGet(mDcgmHandle, groupId, &systems);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to get health watches. Return: " << error << std::endl;
        log_error("Error: could not get Health information for group: {}. Return: {}",
                  (unsigned int)(uintptr_t)groupId,
                  result);
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << "Health monitor systems report" << std::endl;
    for (unsigned int index = 0; index < DCGM_HEALTH_WATCH_COUNT_V2; index++)
    {
        unsigned int bit = 1 << index;
        switch (bit)
        {
            case DCGM_HEALTH_WATCH_PCIE:
                out[PCIE_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_NVLINK:
                out[NVLINK_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_PMU:
                // Unimplimented. Do not display
                // out[PMU_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_MCU:
                // Unimplimented. Do not display
                // out[MCU_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_MEM:
                out[MEMORY_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_SM:
                out[SM_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_INFOROM:
                out[INFOROM_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_THERMAL:
                out[THERMAL_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_POWER:
                out[POWER_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_DRIVER:
                out[DRIVER_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_NVSWITCH_NONFATAL:
                out[NVSWITCH_NONFATAL_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            case DCGM_HEALTH_WATCH_NVSWITCH_FATAL:
                out[NVSWITCH_FATAL_SYSTEMS_TAG] = (systems & bit) ? on : off;
                break;
            default:
                std::cout << "Error: DCGM_HEALTH_WATCH_COUNT appears to be incorrect." << std::endl;
                return result;
        }
    }

    std::cout << out.str();
    return result;
}

/*****************************************************************************/
dcgmReturn_t Health::SetWatches(dcgmHandle_t mDcgmHandle,
                                dcgmGpuGrp_t groupId,
                                dcgmHealthSystems_t systems,
                                double updateInterval,
                                double maxKeepAge)
{
    dcgmReturn_t result = DCGM_ST_OK;

    dcgmHealthSetParams_v2 params {};

    params.version        = dcgmHealthSetParams_version2;
    params.groupId        = groupId;
    params.systems        = systems;
    params.updateInterval = (long long)(updateInterval * 1000000.0);
    params.maxKeepAge     = maxKeepAge;

    result = dcgmHealthSet_v2(mDcgmHandle, &params);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to set health watches. Return: " << error << std::endl;
        log_error("Error: could not set Health information for group: {}. Return: {}",
                  (unsigned int)(uintptr_t)groupId,
                  result);
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << "Health monitor systems set successfully." << std::endl;

    return result;
}

/*****************************************************************************/
unsigned int Health::AppendSystemIncidents(const dcgmHealthResponse_t &response,
                                           unsigned int startingIndex,
                                           dcgm_field_eid_t entityId,
                                           dcgm_field_entity_group_t entityGroupId,
                                           dcgmHealthSystems_t system,
                                           std::stringstream &buf,
                                           dcgmHealthWatchResults_t &systemHealth)
{
    unsigned int appendedCount = 0;

    for (unsigned int index = startingIndex; index < response.incidentCount; index++)
    {
        if (response.incidents[index].entityInfo.entityId != entityId
            || response.incidents[index].entityInfo.entityGroupId != entityGroupId
            || response.incidents[index].system != system)
            break;

        appendedCount++;
        buf << ", " << response.incidents[index].error.msg;

        if (response.incidents[index].health > systemHealth)
        {
            systemHealth = response.incidents[index].health;
        }
    }

    return appendedCount;
}

/*****************************************************************************/
void Health::AddErrorMessage(DcgmiOutputBoxer &outErrors, const std::string &inErrorMsg, const std::string &systemStr)
{
    std::string errorMsg(inErrorMsg);
    std::replace(errorMsg.begin(), errorMsg.end(), '\n', ' ');

    unsigned int p     = 0;
    unsigned int start = 0;

    // Add the error message to the stencil; if it is too large to fit into stencil,
    // break it into parts to display.
    if (errorMsg.length() > MAX_SIZE_OF_HEALTH_INFO)
    {
        while (start < errorMsg.length())
        {
            p += MAX_SIZE_OF_HEALTH_INFO;
            if (p >= errorMsg.length())
            {
                p = errorMsg.length() - 1;
            }
            else if (p == errorMsg.length() - 1)
            {
                // NO-OP. Don't change p if we're at the end of the buffer
            }
            else
            {
                // Put pointer to last available word
                while (errorMsg.at(p) != ' ')
                {
                    if (errorMsg.at(p + 1) == ' ' || errorMsg.at(p + 1) == '.' || errorMsg.at(p + 1) == ','
                        || errorMsg.at(p + 1) == '/')
                        break; // check if landed on end of a word
                    p--;
                }

                // Don't print whitespace at the end of this section
                while (errorMsg.at(p) == ' ')
                {
                    p--;
                }
            }

            // Don't print whitespace at the beginning of this section
            for (; errorMsg.at(start) == ' ' && start < errorMsg.length(); start++)
            {
            }

            // Don't print a new line if all that was left was whitespace
            if (start >= errorMsg.length())
            {
                break;
            }

            outErrors[systemStr].addOverflow(errorMsg.substr(start, p - start + 1));

            start = p + 2; // 2 characters till the start of the next word
        }
    }
    else
    {
        outErrors[systemStr].addOverflow(errorMsg);
    }
}

/*****************************************************************************/
unsigned int Health::HandleOneEntity(const dcgmHealthResponse_t &response,
                                     unsigned int startingIndex,
                                     dcgm_field_eid_t entityId,
                                     dcgm_field_entity_group_t entityGroupId,
                                     DcgmiOutput &out)
{
    dcgmHealthWatchResults_t entityHealth = response.incidents[startingIndex].health;
    unsigned int entityCount              = 0;

    DcgmiOutputBoxer &outGroup  = out[std::string(DcgmFieldsGetEntityGroupString(entityGroupId))];
    DcgmiOutputBoxer &outEntity = outGroup[to_string(entityId)];
    DcgmiOutputBoxer &outErrors = outEntity["Errors"];

    for (unsigned int index = startingIndex; index < response.incidentCount; index++)
    {
        if (response.incidents[index].entityInfo.entityId != entityId
            || response.incidents[index].entityInfo.entityGroupId != entityGroupId)
        {
            break;
        }

        dcgmHealthSystems_t system            = response.incidents[index].system;
        dcgmHealthWatchResults_t systemHealth = response.incidents[index].health;
        std::string systemStr                 = Health::HelperSystemToString(system);
        std::stringstream ss;

        // record the initial error
        ss << response.incidents[index].error.msg;

        unsigned int matches
            = AppendSystemIncidents(response, index + 1, entityId, entityGroupId, system, ss, systemHealth);

        // Skip past all of the incidents for that system
        index += matches;
        entityCount += 1 + matches;

        if (systemHealth > entityHealth)
        {
            entityHealth = systemHealth;
        }

        std::string health   = Health::HelperHealthToString(systemHealth);
        outErrors[systemStr] = health;
        AddErrorMessage(outErrors, ss.str(), systemStr);
    }

    outEntity = HelperHealthToString(entityHealth);

    return entityCount;
}

/*****************************************************************************/
std::string Health::GenerateOutputFromResponse(const dcgmHealthResponse_t &response, DcgmiOutput &out)
{
    out.addHeader("Health Monitor Report");

    out[OVERALL_HEALTH_TAG] = HelperHealthToString(response.overallHealth);

    unsigned int index = 0;
    while (index < response.incidentCount)
    {
        dcgm_field_eid_t entityId               = response.incidents[index].entityInfo.entityId;
        dcgm_field_entity_group_t entityGroupId = response.incidents[index].entityInfo.entityGroupId;

        unsigned int entityIncidents = HandleOneEntity(response, index, entityId, entityGroupId, out);
        index += entityIncidents;
    }

    return out.str();
}

/*****************************************************************************/
dcgmReturn_t Health::CheckWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, bool json)
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    dcgmHealthSystems_t systems;
    DcgmiOutputTree outTree(28, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;

    response->version = dcgmHealthResponse_version;

    result = dcgmHealthCheck(mDcgmHandle, groupId, response.get());

    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to check health watches. Return: " << error << std::endl;
        log_error("Error: could not check Health information for group: {}. Return: {}",
                  (unsigned int)(uintptr_t)groupId,
                  result);
        return DCGM_ST_GENERIC_ERROR;
    }


    // Check if watches are enabled
    result = dcgmHealthGet(mDcgmHandle, groupId, &systems);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to check health watches. Return: " << errorString(result) << std::endl;
        return DCGM_ST_GENERIC_ERROR;
    }

    if (!(systems & DCGM_HEALTH_WATCH_ALL))
    {
        std::cout << "Error: Health watches not enabled. Please enable watches. \n";
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << GenerateOutputFromResponse(*(response), out);

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::string Health::HelperHealthToString(dcgmHealthWatchResults_t health)
{
    if (health == DCGM_HEALTH_RESULT_PASS)
        return "Healthy";
    else if (health == DCGM_HEALTH_RESULT_WARN)
        return "Warning";
    else if (health == DCGM_HEALTH_RESULT_FAIL)
        return "Failure";
    else
        return "Internal error";
}

/*****************************************************************************/
std::string Health::HelperSystemToString(dcgmHealthSystems_t system)
{
    switch (system)
    {
        case DCGM_HEALTH_WATCH_PCIE:
            return "PCIe system";
        case DCGM_HEALTH_WATCH_NVLINK:
            return "NVLINK system";
        case DCGM_HEALTH_WATCH_PMU:
            return "PMU system";
        case DCGM_HEALTH_WATCH_MCU:
            return "MCU system";
        case DCGM_HEALTH_WATCH_MEM:
            return "Memory system";
        case DCGM_HEALTH_WATCH_SM:
            return "SM system";
        case DCGM_HEALTH_WATCH_INFOROM:
            return "InfoROM system";
        case DCGM_HEALTH_WATCH_THERMAL:
            return "Thermal system";
        case DCGM_HEALTH_WATCH_POWER:
            return "Power system";
        case DCGM_HEALTH_WATCH_DRIVER:
            return "Driver";
        default:
            return "Internal error";
    }
}

dcgmReturn_t Health::DoExecuteConnected()
{
    return DCGM_ST_OK;
}

/*****************************************************************************
 *****************************************************************************
 *Get Watches Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetHealth::GetHealth(std::string hostname, unsigned int groupId, bool json)
    : Command()
    , groupId(groupId)
{
    m_hostName = std::move(hostname);
    m_json     = json;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t GetHealth::DoExecuteConnected()
{
    return healthObj.GetWatches(m_dcgmHandle, groupId, m_json);
}

/*****************************************************************************
 *****************************************************************************
 *Set Watches Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
SetHealth::SetHealth(std::string hostname,
                     unsigned int groupId,
                     unsigned int system,
                     double updateInterval,
                     double maxKeepAge)
    : Command()
    , mGroupId(groupId)
    , mSystems((dcgmHealthSystems_t)system)
    , mUpdateInterval(updateInterval)
    , mMaxKeepAge(maxKeepAge)
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t SetHealth::DoExecuteConnected()
{
    return healthObj.SetWatches(m_dcgmHandle, mGroupId, mSystems, mUpdateInterval, mMaxKeepAge);
}

/*****************************************************************************
 *****************************************************************************
 *Watch Watches Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
CheckHealth::CheckHealth(std::string hostname, unsigned int groupId, bool json)
    : Command()
    , groupId(groupId)
{
    m_hostName = std::move(hostname);
    m_json     = json;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t CheckHealth::DoExecuteConnected()
{
    return healthObj.CheckWatches(m_dcgmHandle, groupId, m_json);
}
