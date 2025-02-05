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

#include "LsHw.h"

#include <dcgm_structs.h>

#include <string_view>

#include <json/reader.h>
#include <json/value.h>

namespace
{

bool SupportNonNvidiaCpu()
{
    char const *envVar = "DCGM_SUPPORT_NON_NVIDIA_CPU";
    return !!getenv(envVar);
}

} //namespace

LsHw::LsHw()
    : m_runningUserChecker(std::make_unique<DcgmNs::Utils::RunningUserChecker>())
    , m_runCmdHelper(std::make_unique<DcgmNs::Utils::RunCmdHelper>())
{}

void LsHw::SetChecker(std::unique_ptr<DcgmNs::Utils::RunningUserChecker> checker)
{
    m_runningUserChecker = std::move(checker);
}

void LsHw::SetRunCmdHelper(std::unique_ptr<DcgmNs::Utils::RunCmdHelper> runCmdHelper)
{
    m_runCmdHelper = std::move(runCmdHelper);
}

std::optional<std::vector<std::string>> LsHw::GetCpuSerials() const
{
    if (!m_runningUserChecker->IsRoot())
    {
        return std::nullopt;
    }

    std::optional<std::string> cmdOutput = Exec();
    if (!cmdOutput)
    {
        return std::nullopt;
    }

    return ParseCpuSerials(*cmdOutput);
}

#define LSHW_KEY_CHILDREN      "children"
#define LSHW_KEY_ID            "id"
#define LSHW_VAL_ID_CORE       "core"
#define LSHW_VAL_ID_CPU        "cpu"
#define LSHW_KEY_SERIAL        "serial"
#define LSHW_KEY_VENDOR        "vendor"
#define LSHW_VAL_VENDOR_NVIDIA "NVIDIA"

std::optional<std::vector<std::string>> LsHw::ParseCpuSerials(std::string const &stdout) const
{
    Json::Reader jsonReader;
    Json::Value jsonRoot;

    if (!jsonReader.parse(stdout, jsonRoot))
    {
        log_error("Couldn't parse json: '{}'", stdout);
        return std::nullopt;
    }

    std::vector<std::string> serialNumbers;
    try
    {
        auto childrenL1 = jsonRoot[LSHW_KEY_CHILDREN];
        for (auto &l1Obj : childrenL1)
        {
            if (l1Obj[LSHW_KEY_ID].asString() != LSHW_VAL_ID_CORE)
            {
                continue;
            }
            auto childrenL2 = l1Obj[LSHW_KEY_CHILDREN];
            for (auto &l2Obj : childrenL2)
            {
                if (!l2Obj[LSHW_KEY_ID].asString().starts_with(LSHW_VAL_ID_CPU))
                {
                    continue;
                }
                if (!SupportNonNvidiaCpu())
                {
                    if (l2Obj[LSHW_KEY_VENDOR].asString() != LSHW_VAL_VENDOR_NVIDIA)
                    {
                        continue;
                    }
                }

                unsigned int cpuId;
                if (l2Obj[LSHW_KEY_ID].asString() == LSHW_VAL_ID_CPU)
                {
                    cpuId = 0;
                }
                else
                {
                    if (sscanf(l2Obj[LSHW_KEY_ID].asString().c_str(), "cpu:%u", &cpuId) != 1)
                    {
                        log_error("Unexpected lshw json format, key: [{}].", l2Obj[LSHW_KEY_ID].asString());
                        return std::nullopt;
                    }
                }

                if (serialNumbers.size() <= cpuId)
                {
                    serialNumbers.resize(cpuId + 1);
                }
                serialNumbers[cpuId] = l2Obj[LSHW_KEY_SERIAL].asString();
            }
        }
    }
    catch (const std::exception &e)
    {
        log_error("Unexpected lshw json format: {}", e.what());
        log_debug("lshw json string is: {}", jsonRoot.toStyledString());
        return std::nullopt;
    }

    return serialNumbers;
}

std::optional<std::string> LsHw::Exec() const
{
    static std::string const cmd = "lshw -json";
    std::string cmdOutput;
    static std::array<std::string, 2> const cmdPathPrefix { "/usr/bin/", "/usr/sbin/" };

    dcgmReturn_t result = DCGM_ST_OK;
    for (auto const &prefix : cmdPathPrefix)
    {
        result = m_runCmdHelper->RunCmdAndGetOutput(prefix + cmd, cmdOutput);
        if (result == DCGM_ST_OK)
        {
            break;
        }
    }
    if (result != DCGM_ST_OK)
    {
        return std::nullopt;
    }

    return cmdOutput;
}