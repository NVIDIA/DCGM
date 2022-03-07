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

#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>


namespace DcgmNs
{
constexpr std::string_view CutUuidPrefix(std::string_view value) noexcept
{
    if (value.substr(0, 4) == "MIG-")
    {
        value.remove_prefix(4);
    }

    if (value.substr(0, 4) == "GPU-")
    {
        value.remove_prefix(4);
    }

    return value;
}

struct ParsedUnknown
{
    bool operator==(ParsedUnknown const &) const;
};

struct ParsedGpu
{
    std::string gpuUuid;
    explicit ParsedGpu(std::string_view uuid);
    bool operator==(ParsedGpu const &) const;
};

struct ParsedGpuI
{
    std::string gpuUuid;
    std::uint32_t instanceId = -1;
    ParsedGpuI(std::string_view uuid, std::uint32_t instanceId);
    bool operator==(ParsedGpuI const &) const;
};

struct ParsedGpuCi
{
    std::string gpuUuid;
    std::uint32_t instanceId        = -1;
    std::uint32_t computeInstanceId = -1;
    ParsedGpuCi(std::string_view uuid, std::uint32_t instanceId, std::uint32_t computeInstanceId);
    bool operator==(ParsedGpuCi const &) const;
};

using ParseResult = std::variant<ParsedGpu, ParsedGpuI, ParsedGpuCi, ParsedUnknown>;

std::ostream &operator<<(std::ostream &os, ParseResult const &val);

/**
 * Parses possible GPU or Instance or Compute Instance Ids
 * @param value String representation in one of the allowed forms. See notes.
 * @return One of the following possible values:
 *      \retval \c ParsedUnknown Parsing failed
 *      \retval \c ParsedGpu The value was parsed as a GpuId and the returned object will has ParsedGpu::gpuUuid value.
 *      \retval \c ParsedGpuI The value was parsed as a GpuInstanceId and the returned object has ParsedGpuI::gpuUuid
 *                            and ParsedGpuI::instanceId values.
 *      \retval \c ParsedGpuCi The value was parsed as a GpuComputeInstanceId and the
 *                             returned object has ParsedGpuCi::gpuUuid, ParsedGpuCi::instanceId, and
 *                             ParsedGpuCi::computeInstanceId values.
 */
ParseResult ParseInstanceId(std::string_view value);
} // namespace DcgmNs

namespace std
{
template <>
struct hash<DcgmNs::ParsedUnknown>
{
    size_t operator()(DcgmNs::ParsedUnknown const &) const;
};

template <>
struct hash<DcgmNs::ParsedGpu>
{
    size_t operator()(DcgmNs::ParsedGpu const &) const;
};

template <>
struct hash<DcgmNs::ParsedGpuI>
{
    size_t operator()(DcgmNs::ParsedGpuI const &) const;
};

template <>
struct hash<DcgmNs::ParsedGpuCi>
{
    size_t operator()(DcgmNs::ParsedGpuCi const &) const;
};

} // namespace std
