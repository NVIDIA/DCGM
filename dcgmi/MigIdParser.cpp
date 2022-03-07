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

#include "MigIdParser.hpp"

#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <DcgmVariantHelper.hpp>

#include <charconv>


using DcgmNs::Utils::Hash::CompoundHash;
using DcgmNs::Utils::Hash::StdHash;

namespace std
{
size_t hash<DcgmNs::ParsedUnknown>::operator()(DcgmNs::ParsedUnknown const &obj) const
{
    return StdHash(&obj);
}

size_t hash<DcgmNs::ParsedGpu>::operator()(DcgmNs::ParsedGpu const &obj) const
{
    return StdHash(obj.gpuUuid);
}

size_t hash<DcgmNs::ParsedGpuI>::operator()(DcgmNs::ParsedGpuI const &obj) const
{
    return CompoundHash(obj.gpuUuid, obj.instanceId);
}

size_t hash<DcgmNs::ParsedGpuCi>::operator()(DcgmNs::ParsedGpuCi const &obj) const
{
    return CompoundHash(obj.gpuUuid, obj.instanceId, obj.computeInstanceId);
}
} // namespace std

namespace DcgmNs
{
ParseResult ParseInstanceId(std::string_view value)
{
    if (value.empty())
    {
        return ParsedUnknown {};
    }

    auto values = Split(value, '/');

    std::uint32_t instanceId        = -1;
    std::uint32_t computeInstanceId = -1;

    switch (values.size())
    {
        case 1:
            return ParsedGpu { std::string(CutUuidPrefix(values[0])) };
        case 2:
            if (auto [ptr, ec] = std::from_chars(values[1].data(), values[1].data() + values[1].size(), instanceId);
                ec != std::errc())
            {
                return ParsedUnknown {};
            }

            return ParsedGpuI { std::string(CutUuidPrefix(values[0])), instanceId };
        case 3:
            if (auto [ptr, ec] = std::from_chars(values[1].data(), values[1].data() + values[1].size(), instanceId);
                ec != std::errc())
            {
                return ParsedUnknown {};
            }

            if (auto [ptr, ec]
                = std::from_chars(values[2].data(), values[2].data() + values[2].size(), computeInstanceId);
                ec != std::errc())
            {
                return ParsedUnknown {};
            }

            return ParsedGpuCi { std::string(CutUuidPrefix(values[0])), instanceId, computeInstanceId };
        default:
            return ParsedUnknown {};
    }
}


ParsedGpu::ParsedGpu(std::string_view uuid)
    : gpuUuid(uuid)
{}

bool ParsedGpu::operator==(ParsedGpu const &other) const
{
    return gpuUuid == other.gpuUuid;
}


ParsedGpuI::ParsedGpuI(std::string_view uuid, std::uint32_t instanceId)
    : gpuUuid(uuid)
    , instanceId(instanceId)
{}

bool ParsedGpuI::operator==(ParsedGpuI const &other) const
{
    return std::tie(gpuUuid, instanceId) == std::tie(other.gpuUuid, other.instanceId);
}

ParsedGpuCi::ParsedGpuCi(std::string_view uuid, std::uint32_t instanceId, std::uint32_t computeInstanceId)
    : gpuUuid(uuid)
    , instanceId(instanceId)
    , computeInstanceId(computeInstanceId)
{}

bool ParsedGpuCi::operator==(ParsedGpuCi const &other) const
{
    return std::tie(gpuUuid, instanceId, computeInstanceId)
           == std::tie(other.gpuUuid, other.instanceId, other.computeInstanceId);
}

bool ParsedUnknown::operator==(ParsedUnknown const &obj) const
{
    return this == &obj;
}


std::ostream &operator<<(std::ostream &os, ParseResult const &val)
{
    std::visit(
        overloaded([&](ParsedUnknown const &) { os << "ParsedUnknown"; },
                   [&](ParsedGpu const &obj) { os << "ParsedGpu(" << obj.gpuUuid << ")"; },
                   [&](ParsedGpuI const &obj) { os << "ParsedGpuI(" << obj.gpuUuid << ", " << obj.instanceId << ")"; },
                   [&](ParsedGpuCi const &obj) {
                       os << "ParsedGpuCi(" << obj.gpuUuid << ", " << obj.instanceId << ", " << obj.computeInstanceId
                          << ")";
                   }),
        val);

    return os;
}

} // namespace DcgmNs
