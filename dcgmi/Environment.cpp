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
 * File:   Environment.cpp
 */
#include "Environment.h"

#include "dcgmi_common.h"

#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <fmt/format.h>
#include <iostream>
#include <optional>
#include <unordered_set>

/**************************************************************************************/

namespace
{
/* Helper function to get status description */
char const *GetEnvVarStatusDescription(dcgmEnvVarStatus_t status)
{
    switch (status)
    {
        case dcgmEnvVarStatus_t::DCGM_ENV_STATUS_MATCH:
            return "Values are consistent";
        case dcgmEnvVarStatus_t::DCGM_ENV_STATUS_MISMATCH:
            return "Values differ between local process and hostengine";
        case dcgmEnvVarStatus_t::DCGM_ENV_STATUS_ERROR:
            return "Failed to get hostengine environment variable";
        case dcgmEnvVarStatus_t::DCGM_ENV_STATUS_NOT_ALLOWED:
            return "Environment variable is not allowed";
        default:
            return "Unknown status";
    }
}

using namespace fmt::literals;

/* Display constants */
static char const *const ENV_VAR_INFO_HEADER = "Environment Variable Information:";
static char const *const SEPARATOR_LINE      = "=================================";

} //namespace

EnvironmentInfo::EnvironmentInfo(std::string hostname, std::string envVarName)
    : m_envVarName(std::move(envVarName))
{
    m_hostName = std::move(hostname);

    /* suppress connection errors */
    m_silent  = true;
    m_timeout = 100;
}

std::string EnvironmentInfo::GetEnvVarValue() const
{
    return m_hostengineEnvVarValue;
}

void EnvironmentInfo::SetEnvVarName(std::string const &envVarName)
{
    m_envVarName = envVarName;
}

std::optional<std::string> EnvironmentInfo::GetCurrentProcessEnvVarValue() const
{
    char const *envVarValue = std::getenv(m_envVarName.c_str());
    if (envVarValue == nullptr)
    {
        return std::nullopt;
    }

    return std::string(envVarValue);
}

dcgmReturn_t EnvironmentInfo::DoExecuteConnected()
{
    dcgmEnvVarInfo_t envVarInfo = {};

    envVarInfo.version = dcgmEnvVarInfo_version;
    SafeCopyTo(envVarInfo.envVarName, m_envVarName.c_str());

    dcgmReturn_t ret = dcgmHostengineEnvironmentVariableInfo(m_dcgmHandle, &envVarInfo);
    log_debug("hostengine environment variable info ret: {}, envVarInfo.ret: {}", ret, envVarInfo.ret);

    if (ret != DCGM_ST_OK)
    {
        m_status = dcgmEnvVarStatus_t::DCGM_ENV_STATUS_ERROR;
        return ret;
    }

    if (envVarInfo.ret != DCGM_ST_OK)
    {
        if (envVarInfo.ret == DCGM_ST_BADPARAM)
        {
            m_status = dcgmEnvVarStatus_t::DCGM_ENV_STATUS_NOT_ALLOWED;
        }
        else
        {
            m_status = dcgmEnvVarStatus_t::DCGM_ENV_STATUS_ERROR;
        }

        return envVarInfo.ret;
    }

    m_hostengineEnvVarValue = envVarInfo.envVarValue;
    return DCGM_ST_OK;
}

void EnvironmentInfo::ValidateEnvVarValue()
{
    m_dcgmiEnvVarValue = GetCurrentProcessEnvVarValue();

    // Only process if dcgmi has the environment variable set
    if (m_dcgmiEnvVarValue.has_value())
    {
        // Get the env value from hostengine process, if set
        dcgmReturn_t result = Execute();

        std::string displayValue = fmt::format(
            "WARNING: {varName} is set to {varValue} on dcgmi process. This does not affect which GPUs are selected for diagnostics. "
            "Unset {varName} on dcgmi process. Set {varName} on nv-hostengine to select GPUs.\n\n"
            "Variable Name: {varName}\n"
            "Host: {hostName}\n"
            "Hostengine Process Value: {hostValue}\n"
            "Dcgmi Process Value: {varValue}\n",
            "varName"_a   = m_envVarName,
            "varValue"_a  = m_dcgmiEnvVarValue.value(),
            "hostName"_a  = m_hostName,
            "hostValue"_a = (m_hostengineEnvVarValue.empty() ? "(not set)" : m_hostengineEnvVarValue));

        std::string setEnvMessage = "\nPlease set the " + m_envVarName + " env on nv-hostengine\n";
        if (result != DCGM_ST_OK)
        {
            std::string message = (result == DCGM_ST_NOT_CONFIGURED) ? setEnvMessage : "\n";
            displayValue        = displayValue + "Status: " + GetEnvVarStatusDescription(m_status) + message;
            DisplayCliOutput(displayValue);
            return;
        }

        if (m_hostengineEnvVarValue.empty())
        {
            m_status     = dcgmEnvVarStatus_t::DCGM_ENV_STATUS_ERROR;
            displayValue = displayValue + "Status: " + GetEnvVarStatusDescription(m_status) + setEnvMessage;
            DisplayCliOutput(displayValue);
            return;
        }

        if (m_hostengineEnvVarValue != m_dcgmiEnvVarValue.value())
        {
            m_status = dcgmEnvVarStatus_t::DCGM_ENV_STATUS_MISMATCH;
        }
        else
        {
            m_status = dcgmEnvVarStatus_t::DCGM_ENV_STATUS_MATCH;
        }

        displayValue = displayValue + "Status: " + GetEnvVarStatusDescription(m_status) + "\n";
        DisplayCliOutput(displayValue);
    }
}

void EnvironmentInfo::CheckDiagEnabledEnvVar()
{
    dcgmReturn_t result = Execute();
    if (result != DCGM_ST_OK)
    {
        return;
    }

    if (!m_hostengineEnvVarValue.empty())
    {
        std::string displayValue = fmt::format(
            "WARNING: You are using a DCGM installation with limited functionality that does not offer diagnostic capabilities."
            "Please use the appropriate DCGM image or installation to ensure proper operation diagnostics.\n");

        DisplayCliOutput(displayValue);
    }
}

void EnvironmentInfo::DisplayCliOutput(std::string const &displayValue)
{
    std::cerr << SEPARATOR_LINE << std::endl;
    std::cerr << ENV_VAR_INFO_HEADER << std::endl;
    std::cerr << SEPARATOR_LINE << std::endl;

    std::cerr << displayValue;

    std::cerr << std::endl;
}
