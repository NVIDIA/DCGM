/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include "CommandLineParser.h"
#include "DcgmLogging.h"
#include "Introspect.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"

static char const INTROSPECT_HEADER[]
    = "+----------------------------------------------------------------------------+\n"
      "| Introspection Information                                                  |\n"
      "+============================================================================+\n";
static char const INTROSPECT_TARGET_HEADER[]
    = "\n"
      "+----------------------------------------------------------------------------+\n"
      "| <TARGET                                                                   >|\n"
      "+============================================================================+\n";

static char const INTROSPECT_SUB_TARGET_HEADER[]
    = "| <TARGET                                                                   >|\n"
      "+-------------------+--------------------------------------------------------+\n";
static char const INTROSPECT_ATTRIBUTE_DATA[]
    = "| <ATTRIBUTE       >| <ATTRIBUTE_DATA                                       >|\n";
static char const INTROSPECT_FIELD_NOT_WATCHED[]
    = "| NOT WATCHED       |                                                        |\n";
static char const INTROSPECT_TARGET_SEPARATOR[]
    = "+-------------------+--------------------------------------------------------+\n";

static char const INTROSPECT_FOOTER[]
    = "+-------------------+--------------------------------------------------------+\n";

static const char ERROR_STRING[] = "Error";
static const char INTROSPECTION_NOT_ENABLED_MSG[]
    = "Error: Introspection is disabled.  Please enable it with \"dcgmi introspect --enable\" before viewing introspection stats again.";

#define TARGET_TAG         "<TARGET"
#define ATTRIBUTE_TAG      "<ATTRIBUTE"
#define ATTRIBUTE_DATA_TAG "<ATTRIBUTE_DATA"

Introspect::Introspect()
{}

Introspect::~Introspect()
{}

dcgmReturn_t Introspect::EnableIntrospect(dcgmHandle_t handle)
{
    dcgmReturn_t result = dcgmIntrospectToggleState(handle, DCGM_INTROSPECT_STATE_ENABLED);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: failed to enable introspection. Return: " << errorString(result) << "." << std::endl;
        PRINT_ERROR("%s", "failed to enable introspection. Return: %s", errorString(result));
    }
    else
    {
        std::cout << "Introspection enabled" << std::endl;
    }

    return result;
}

dcgmReturn_t Introspect::DisableIntrospect(dcgmHandle_t handle)
{
    dcgmReturn_t result = dcgmIntrospectToggleState(handle, DCGM_INTROSPECT_STATE_DISABLED);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: failed to disable introspection. Return: " << errorString(result) << "." << std::endl;
        PRINT_ERROR("%s", "failed to disable introspection. Return: %s", errorString(result));
    }
    else
    {
        std::cout << "Introspection disabled" << std::endl;
    }

    return result;
}

dcgmReturn_t Introspect::DisplayStats(dcgmHandle_t handle,
                                      bool forHostengine,
                                      bool forAllFields,
                                      bool forAllFieldGroups,
                                      std::vector<dcgmFieldGrp_t> forFieldGroups)
{
    // hostengine stats
    dcgmReturn_t heMemReturn = DCGM_ST_GENERIC_ERROR;
    dcgmIntrospectMemory_t heMemInfo;
    heMemInfo.version = dcgmIntrospectMemory_version1;

    dcgmReturn_t heCpuReturn = DCGM_ST_GENERIC_ERROR;
    dcgmIntrospectCpuUtil_t heCpuInfo;
    heCpuInfo.version = dcgmIntrospectCpuUtil_version1;

    // all fields stats
    dcgmReturn_t afMemReturn = DCGM_ST_GENERIC_ERROR;
    dcgmIntrospectFullMemory_t afMemInfo;
    afMemInfo.version = dcgmIntrospectFullMemory_version1;

    dcgmReturn_t afExecReturn = DCGM_ST_GENERIC_ERROR;
    dcgmIntrospectFullFieldsExecTime_t afExecInfo;
    afExecInfo.version = dcgmIntrospectFullFieldsExecTime_version2;

    dcgmAllFieldGroup_t allFieldGroup;
    memset(&allFieldGroup, 0, sizeof(allFieldGroup));

    if (forAllFieldGroups || !forFieldGroups.empty())
    {
        allFieldGroup.version          = dcgmAllFieldGroup_version;
        dcgmReturn_t groupGetAllReturn = dcgmFieldGroupGetAll(handle, &allFieldGroup);
        if (groupGetAllReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "dcgmFieldGroupGetAll returned %d", (int)groupGetAllReturn);
        }

        PRINT_DEBUG("%d %d",
                    "Got %d field groups from the host engine. ret %d",
                    (int)allFieldGroup.numFieldGroups,
                    (int)groupGetAllReturn);

        if (forAllFieldGroups)
        {
            /* Fill our array of field group IDs with the known field group IDs */
            forFieldGroups.clear();
            for (size_t i = 0; i < allFieldGroup.numFieldGroups; i++)
            {
                forFieldGroups.push_back(allFieldGroup.fieldGroups[i].fieldGroupId);
            }
        }
    }

    // field group stats
    std::vector<dcgmReturn_t> fgMemReturns(forFieldGroups.size());
    std::vector<dcgmIntrospectFullMemory_t> fgMemInfos(forFieldGroups.size());
    for (auto &fgMemInfo : fgMemInfos)
    {
        fgMemInfo.version = dcgmIntrospectFullMemory_version1;
    }

    std::vector<dcgmReturn_t> fgExecReturns(forFieldGroups.size());
    std::vector<dcgmIntrospectFullFieldsExecTime_t> fgExecInfos(forFieldGroups.size());
    for (auto &fgExecInfo : fgExecInfos)
    {
        fgExecInfo.version = dcgmIntrospectFullFieldsExecTime_version2;
    }

    // always retrieve hostengine mem usage as a way to check if introspection is enabled
    heMemReturn = dcgmIntrospectGetHostengineMemoryUsage(handle, &heMemInfo, true);
    if (DCGM_ST_NOT_CONFIGURED == heMemReturn)
    {
        std::cout << INTROSPECTION_NOT_ENABLED_MSG << std::endl;
        return heMemReturn;
    }
    else if (DCGM_ST_OK != heMemReturn)
    {
        PRINT_ERROR("%s", "Error retrieving memory usage for hostengine. Return: %s", errorString(heMemReturn));
    }

    if (forHostengine)
    {
        heCpuReturn = dcgmIntrospectGetHostengineCpuUtilization(handle, &heCpuInfo, true);
        if (DCGM_ST_OK != heCpuReturn)
        {
            PRINT_ERROR("%s", "Error retrieving CPU utilization for hostengine. Return: %s", errorString(heCpuReturn));
        }
    }

    if (forAllFields)
    {
        dcgmIntrospectContext_t context;
        context.version       = dcgmIntrospectContext_version;
        context.introspectLvl = DCGM_INTROSPECT_LVL_ALL_FIELDS;

        afMemReturn = dcgmIntrospectGetFieldsMemoryUsage(handle, &context, &afMemInfo, true);
        if (DCGM_ST_OK != afMemReturn)
        {
            PRINT_ERROR("%s", "Error retrieving memory for all fields. Return: %s", errorString(afMemReturn));
        }

        afExecReturn = dcgmIntrospectGetFieldsExecTime(handle, &context, &afExecInfo, true);
        if (DCGM_ST_OK != afExecReturn)
        {
            PRINT_ERROR("%s", "Error retrieving Exec time for all fields. Return: %s", errorString(afExecReturn));
        }
    }

    if (!forFieldGroups.empty())
    {
        for (size_t i = 0; i < forFieldGroups.size(); ++i)
        {
            dcgmFieldGrp_t fgId = forFieldGroups[i];

            dcgmIntrospectContext_t context;
            context.version       = dcgmIntrospectContext_version;
            context.introspectLvl = DCGM_INTROSPECT_LVL_FIELD_GROUP;
            context.fieldGroupId  = fgId;

            fgMemReturns[i] = dcgmIntrospectGetFieldsMemoryUsage(handle, &context, &fgMemInfos[i], (int)true);
            if (DCGM_ST_OK != fgMemReturns[i] && DCGM_ST_NOT_WATCHED != fgMemReturns[i])
            {
                PRINT_ERROR("%llu %s",
                            "Error retrieving memory for field group %llu. Return: %s",
                            (unsigned long long)fgId,
                            errorString(fgMemReturns[i]));
            }

            fgExecReturns[i] = dcgmIntrospectGetFieldsExecTime(handle, &context, &fgExecInfos[i], (int)true);
            if (DCGM_ST_OK != fgExecReturns[i] && DCGM_ST_NOT_WATCHED != fgExecReturns[i])
            {
                PRINT_ERROR("%llu %s",
                            "Error retrieving execution time for field group %llu. Return: %s",
                            (unsigned long long)fgId,
                            errorString(fgExecReturns[i]));
            }
        }
    }

    // Display the stats
    CommandOutputController cmdView;
    cmdView.setDisplayStencil(INTROSPECT_HEADER);
    cmdView.display();

    if (forHostengine)
    {
        // Output target name
        cmdView.setDisplayStencil(INTROSPECT_TARGET_HEADER);
        cmdView.addDisplayParameter(TARGET_TAG, "Hostengine Process");
        cmdView.display();

        // Memory usage
        cmdView.setDisplayStencil(INTROSPECT_ATTRIBUTE_DATA);
        cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Memory");

        if (DCGM_ST_OK == heMemReturn)
        {
            cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, readableMemory(heMemInfo.bytesUsed));
        }
        else
        {
            cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, ERROR_STRING);
        }
        cmdView.display();

        // CPU util
        cmdView.addDisplayParameter(ATTRIBUTE_TAG, "CPU Utilization");

        if (DCGM_ST_OK == heCpuReturn)
        {
            cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, readablePercent(heCpuInfo.total));
        }
        else
        {
            cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, ERROR_STRING);
        }
        cmdView.display();

        cmdView.setDisplayStencil(INTROSPECT_TARGET_SEPARATOR);
        cmdView.display();
    }

    if (forAllFields)
    {
        // Output target name
        cmdView.setDisplayStencil(INTROSPECT_TARGET_HEADER);
        cmdView.addDisplayParameter(TARGET_TAG, "All Field Values");
        cmdView.display();

        if (afMemReturn == DCGM_ST_NOT_WATCHED || afExecReturn == DCGM_ST_NOT_WATCHED)
        {
            cmdView.setDisplayStencil(INTROSPECT_FIELD_NOT_WATCHED);
            cmdView.display();
        }
        else
        {
            // Memory usage
            cmdView.setDisplayStencil(INTROSPECT_ATTRIBUTE_DATA);
            cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Memory");

            if (DCGM_ST_OK == afMemReturn)
            {
                cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, readableMemory(afMemInfo.aggregateInfo.bytesUsed));
            }
            else
            {
                cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, ERROR_STRING);
            }
            cmdView.display();

            // Exec Time
            cmdView.setDisplayStencil(INTROSPECT_ATTRIBUTE_DATA);
            cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Recent Update Time");

            if (DCGM_ST_OK == afExecReturn)
            {
                cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG,
                                            readableTime(afExecInfo.aggregateInfo.recentUpdateUsec));
            }
            else
            {
                cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, ERROR_STRING);
            }
            cmdView.display();
        }

        cmdView.setDisplayStencil(INTROSPECT_TARGET_SEPARATOR);
        cmdView.display();
    }

    if (forFieldGroups.size())
    {
        cmdView.setDisplayStencil(INTROSPECT_TARGET_HEADER);
        cmdView.addDisplayParameter(TARGET_TAG, "Field Groups");
        cmdView.display();

        for (size_t i = 0; i < forFieldGroups.size(); ++i)
        {
            if (i > 0)
            {
                cmdView.setDisplayStencil(INTROSPECT_TARGET_SEPARATOR);
                cmdView.display();
            }

            dcgmFieldGrp_t fgId                           = forFieldGroups[i];
            dcgmReturn_t fgMemReturn                      = fgMemReturns[i];
            dcgmIntrospectFullMemory_t fgMemInfo          = fgMemInfos[i];
            dcgmReturn_t fgExecReturn                     = fgExecReturns[i];
            dcgmIntrospectFullFieldsExecTime_t fcExecInfo = fgExecInfos[i];

            // Output target name
            std::stringstream fcSubHeader;
            fcSubHeader << fgId;

            /* Find the field group name */
            for (unsigned int j = 0; j < allFieldGroup.numFieldGroups; j++)
            {
                if (fgId == allFieldGroup.fieldGroups[j].fieldGroupId)
                {
                    fcSubHeader << " - " << allFieldGroup.fieldGroups[j].fieldGroupName;
                    break;
                }
            }

            cmdView.setDisplayStencil(INTROSPECT_SUB_TARGET_HEADER);
            cmdView.addDisplayParameter(TARGET_TAG, fcSubHeader.str());
            cmdView.display();

            if (fgMemReturn == DCGM_ST_NOT_WATCHED || fgExecReturn == DCGM_ST_NOT_WATCHED)
            {
                cmdView.setDisplayStencil(INTROSPECT_FIELD_NOT_WATCHED);
                cmdView.display();
                continue;
            }

            // Memory usage
            cmdView.setDisplayStencil(INTROSPECT_ATTRIBUTE_DATA);
            cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Memory");

            if (DCGM_ST_OK == fgMemReturn)
            {
                cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, readableMemory(fgMemInfo.aggregateInfo.bytesUsed));
            }
            else
            {
                cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, ERROR_STRING);
            }
            cmdView.display();

            // Recent Exec Time

            cmdView.setDisplayStencil(INTROSPECT_ATTRIBUTE_DATA);
            cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Recent Update Time");

            if (DCGM_ST_OK == fgExecReturn)
            {
                cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG,
                                            readableTime(fcExecInfo.aggregateInfo.recentUpdateUsec));
            }
            else
            {
                cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, ERROR_STRING);
            }
            cmdView.display();
        }

        cmdView.setDisplayStencil(INTROSPECT_TARGET_SEPARATOR);
        cmdView.display();
    }

    return DCGM_ST_OK;
}

template <typename T>
string Introspect::readableTime(T usec)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << usec / 1000.0 << " ms";
    return ss.str();
}

string Introspect::readableMemory(long long bytes)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << bytes / 1024.0 << " KB";
    return ss.str();
}

string Introspect::readablePercent(double p)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << 100 * p << " %";
    return ss.str();
}

ToggleIntrospect::ToggleIntrospect(std::string hostname, bool enabled)
    : Command()
    , enabled(enabled)
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

dcgmReturn_t ToggleIntrospect::DoExecuteConnected()
{
    return enabled ? introspectObj.EnableIntrospect(m_dcgmHandle) : introspectObj.DisableIntrospect(m_dcgmHandle);
}

DisplayIntrospectSummary::DisplayIntrospectSummary(std::string hostname,
                                                   bool forHostengine,
                                                   bool forAllFields,
                                                   bool forAllFieldGroups,
                                                   std::vector<dcgmFieldGrp_t> forFieldGroups)
    : Command()
    , forHostengine(forHostengine)
    , forAllFields(forAllFields)
    , forAllFieldGroups(forAllFieldGroups)
    , forFieldGroups(std::move(forFieldGroups))
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

dcgmReturn_t DisplayIntrospectSummary::DoExecuteConnected()
{
    return introspectObj.DisplayStats(m_dcgmHandle, forHostengine, forAllFields, forAllFieldGroups, forFieldGroups);
}
