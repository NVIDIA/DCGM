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

#include "CpuHelpers.h"
#include "DcgmLogging.h"
#include "DcgmStringHelpers.h"

#include <memory>
#include <regex>
#include <unordered_set>

CpuHelpers::CpuHelpers()
    : CpuHelpers(std::make_unique<FileSystemOperator>(), std::make_unique<LsHw>())
{}

CpuHelpers::CpuHelpers(std::unique_ptr<FileSystemOperator> fileSystemOp, std::unique_ptr<LsHw> lshw)
{
    m_fileSystemOp = std::move(fileSystemOp);
    m_lshw         = std::move(lshw);
    Init();
}

std::string_view CpuHelpers::GetNvidiaVendorName()
{
    return "Nvidia";
}

std::string_view CpuHelpers::GetGraceModelName()
{
    return "Grace";
}

bool CpuHelpers::SupportNonNvidiaCpu()
{
    char const *envVar = "DCGM_SUPPORT_NON_NVIDIA_CPU";
    return !!getenv(envVar);
}

void CpuHelpers::ReadCpuVendorAndModel(std::string_view vendorPath)
{
    /* We expect to see "jep106:036b:0241" in the file:
     * jep106 specifies the standard
     * 036b is the manufacturer's identification code for Nvidia
     * 0241 signifies the chip that has Grace CPUs on it.
     */
    static std::string_view const sGraceChipId           = "0241";
    static std::string_view const sNvidiaManufacturersId = "036b";

    m_cpuVendor = "Unknown";
    m_cpuModel  = "Unknown";

    auto contents = m_fileSystemOp->Read(vendorPath);
    if (!contents.has_value())
    {
        log_debug("fail to read file content of path: {}", vendorPath);
        return;
    }

    auto tokens = DcgmNs::Split(*contents, ':');
    if (tokens.size() == 3)
    {
        if (tokens[1] == sNvidiaManufacturersId)
        {
            m_cpuVendor = GetNvidiaVendorName();
            if (tokens[2] == sGraceChipId)
            {
                m_cpuModel = GetGraceModelName();
            }
            else
            {
                log_debug("Non-Grace chip ID '{}' found", tokens[2]);
            }
        }
        else
        {
            log_debug("Non-Nvidia manufacturer '{}' found", tokens[1]);
        }
    }
    else
    {
        log_error("Couldn't parse soc_id '{}'", *contents);
    }
}

unsigned int CpuHelpers::GetPhysicalCpusNum() const
{
    auto pathsOpt = m_fileSystemOp->Glob(CPU_CORE_SLIBLINGS_GLOB_PATTERN);
    if (!pathsOpt.has_value())
    {
        log_error("failed to glob on pattern [{}].", CPU_CORE_SLIBLINGS_GLOB_PATTERN);
        return 0;
    }

    std::unordered_set<std::string> coreSiblings;
    for (auto const &path : *pathsOpt)
    {
        auto contentOpt = m_fileSystemOp->Read(path);
        if (!contentOpt)
        {
            log_error("failed to read on path [{}].", path);
            return 0;
        }
        coreSiblings.insert(*contentOpt);
    }
    return coreSiblings.size();
}

void CpuHelpers::ReadPhysicalCpuIds()
{
    auto physicalCpusNum = GetPhysicalCpusNum();
    m_cpuIds.reserve(physicalCpusNum);
    for (unsigned int i = 0; i < physicalCpusNum; ++i)
    {
        m_cpuIds.push_back(i);
    }
}

void CpuHelpers::Init()
{
    ReadCpuVendorAndModel(CPU_VENDOR_MODEL_PATH);
    if (GetVendor() == GetNvidiaVendorName() || SupportNonNvidiaCpu())
    {
        ReadPhysicalCpuIds();
    }
}

std::vector<dcgm_field_eid_t> const &CpuHelpers::GetCpuIds() const
{
    return m_cpuIds;
}

std::string const &CpuHelpers::GetVendor() const
{
    return m_cpuVendor;
}

std::string const &CpuHelpers::GetModel() const
{
    return m_cpuModel;
}

std::optional<std::vector<std::string>> CpuHelpers::GetCpuSerials() const
{
    auto const cpuSerials = m_lshw->GetCpuSerials();
    if (!cpuSerials)
    {
        log_debug("failed to get serials from lshw.");
        return std::nullopt;
    }
    if (cpuSerials->size() != m_cpuIds.size())
    {
        log_debug(
            "Number of serials [{}] does not align with number of cpu ids [{}].", cpuSerials->size(), m_cpuIds.size());
        return std::nullopt;
    }
    return cpuSerials;
}