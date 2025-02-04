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

#pragma once

#include "FileSystemOperator.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "LsHw.h"

#include <dcgm_structs.h>

#define CPU_VENDOR_MODEL_PATH           "/sys/devices/soc0/soc_id"
#define CPU_CORE_SLIBLINGS_GLOB_PATTERN "/sys/devices/system/cpu/cpu*/topology/core_siblings"

class CpuHelpers
{
public:
    CpuHelpers();
    explicit CpuHelpers(std::unique_ptr<FileSystemOperator> fileSystemOp, std::unique_ptr<LsHw> lshw);

    void Init();
    std::vector<dcgm_field_eid_t> const &GetCpuIds() const;
    std::string const &GetVendor() const;
    std::string const &GetModel() const;

    virtual std::optional<std::vector<std::string>> GetCpuSerials() const;

    static bool SupportNonNvidiaCpu();

    static std::string_view GetNvidiaVendorName();
    static std::string_view GetGraceModelName();

protected:
    [[nodiscard]] unsigned int GetPhysicalCpusNum() const;

private:
    void ReadPhysicalCpuIds();
    void ReadCpuVendorAndModel(std::string_view vendorPath);

    std::vector<dcgm_field_eid_t> m_cpuIds;
    std::string m_cpuVendor = "Unknown";
    std::string m_cpuModel  = "Unknown";
    std::unique_ptr<FileSystemOperator> m_fileSystemOp;
    std::unique_ptr<LsHw> m_lshw;
};