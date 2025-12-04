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
 * File:   Environment.h
 */
#pragma once

#include "Command.h"

#include <optional>
#include <string>
#include <string_view>

/*****************************************************************************
 * Environment variable validation status enum
 ****************************************************************************/
enum class dcgmEnvVarStatus_t
{
    DCGM_ENV_STATUS_MATCH       = 0,
    DCGM_ENV_STATUS_MISMATCH    = 1,
    DCGM_ENV_STATUS_ERROR       = 2,
    DCGM_ENV_STATUS_NOT_ALLOWED = 3
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Set Environment Invoker class
 */
class EnvironmentInfo : public Command
{
public:
    EnvironmentInfo(std::string hostname, std::string envVarName);
    std::string GetEnvVarValue() const;
    void SetEnvVarName(std::string const &envVarName);
    std::optional<std::string> GetCurrentProcessEnvVarValue() const;
    void ValidateEnvVarValue();
    void CheckDiagEnabledEnvVar();
    void DisplayCliOutput(std::string const &displayValue);

protected:
    /*****************************************************************************
     * Override the Execute method for getting environment info
     *****************************************************************************/
    dcgmReturn_t DoExecuteConnected() override;

private:
    std::string m_envVarName;
    std::string m_hostengineEnvVarValue;
    std::optional<std::string> m_dcgmiEnvVarValue;
    dcgmEnvVarStatus_t m_status;
};
