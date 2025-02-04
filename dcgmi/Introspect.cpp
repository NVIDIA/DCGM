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

#define TARGET_TAG         "<TARGET"
#define ATTRIBUTE_TAG      "<ATTRIBUTE"
#define ATTRIBUTE_DATA_TAG "<ATTRIBUTE_DATA"

Introspect::Introspect()
{}

Introspect::~Introspect()
{}

dcgmReturn_t Introspect::DisplayStats(dcgmHandle_t handle, bool forHostengine)
{
    // hostengine stats
    dcgmReturn_t heMemReturn = DCGM_ST_GENERIC_ERROR;
    dcgmIntrospectMemory_t heMemInfo;
    heMemInfo.version = dcgmIntrospectMemory_version1;

    dcgmReturn_t heCpuReturn = DCGM_ST_GENERIC_ERROR;
    dcgmIntrospectCpuUtil_t heCpuInfo;
    heCpuInfo.version = dcgmIntrospectCpuUtil_version1;

    // always retrieve hostengine mem usage as a way to check if introspection is enabled
    heMemReturn = dcgmIntrospectGetHostengineMemoryUsage(handle, &heMemInfo, true);
    if (DCGM_ST_OK != heMemReturn)
    {
        log_error("Error retrieving memory usage for hostengine. Return: {}", errorString(heMemReturn));
    }

    if (forHostengine)
    {
        heCpuReturn = dcgmIntrospectGetHostengineCpuUtilization(handle, &heCpuInfo, true);
        if (DCGM_ST_OK != heCpuReturn)
        {
            log_error("Error retrieving CPU utilization for hostengine. Return: {}", errorString(heCpuReturn));
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

DisplayIntrospectSummary::DisplayIntrospectSummary(std::string hostname, bool forHostengine)
    : Command()
    , forHostengine(forHostengine)
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

dcgmReturn_t DisplayIntrospectSummary::DoExecuteConnected()
{
    return introspectObj.DisplayStats(m_dcgmHandle, forHostengine);
}
