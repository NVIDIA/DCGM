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

#include "MigIdParser.hpp"

#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <DcgmVariantHelper.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

#include <charconv>


using DcgmNs::Utils::Hash::CompoundHash;
using DcgmNs::Utils::Hash::StdHash;

size_t std::hash<DcgmNs::ParsedUnknown>::operator()(DcgmNs::ParsedUnknown const &obj) const
{
    return StdHash(&obj);
}

size_t std::hash<DcgmNs::ParsedGpu>::operator()(DcgmNs::ParsedGpu const &obj) const
{
    return StdHash(obj.gpuUuid);
}

size_t std::hash<DcgmNs::ParsedGpuI>::operator()(DcgmNs::ParsedGpuI const &obj) const
{
    return CompoundHash(obj.gpuUuid, obj.instanceId);
}

size_t std::hash<DcgmNs::ParsedGpuCi>::operator()(DcgmNs::ParsedGpuCi const &obj) const
{
    return CompoundHash(obj.gpuUuid, obj.instanceId, obj.computeInstanceId);
}

size_t std::hash<DcgmNs::NotInitialized>::operator()(DcgmNs::NotInitialized const &) const
{
    return 0; // It does not matter which object is actually NotInitialized
}

size_t std::hash<DcgmNs::Wildcarded>::operator()(DcgmNs::Wildcarded const &) const
{
    return 0; // It does not matter for wildcards
} // namespace std

namespace DcgmNs
{
ParseResult ParseInstanceId(std::string_view value)
{
    if (value.empty())
    {
        return ParsedUnknown {};
    }

    if (value.find(":") != std::string_view::npos && value.find("gpu:") == std::string_view::npos)
    {
        // This function mainly focuses on GPU wildcards.
        // Something like cpu:* is not being handled in this context.
        return ParsedUnknown {};
    }

    auto values = Split(value, '/');

    Wildcard<std::string> tmpGpuUuid             = NotInitialized {};
    Wildcard<std::uint32_t> tmpInstanceId        = NotInitialized {};
    Wildcard<std::uint32_t> tmpComputeInstanceId = NotInitialized {};

    if (values.size() >= 1)
    {
        if (values[0].find('*') != std::string_view::npos)
        {
            tmpGpuUuid = Wildcarded {};
        }
        else
        {
            tmpGpuUuid = std::string { CutUuidPrefix(values[0]) };
        }
    }

    if (values.size() >= 2)
    {
        if (values[1].find('*') != std::string_view::npos)
        {
            tmpInstanceId = Wildcarded {};
        }
        else
        {
            std::uint32_t instanceId = -1;
            if (auto [ptr, ec] = std::from_chars(values[1].data(), values[1].data() + values[1].size(), instanceId);
                ec != std::errc())
            {
                return ParsedUnknown {};
            }
            tmpInstanceId = instanceId;
        }
    }

    if (values.size() >= 3)
    {
        if (values[2].find('*') != std::string_view::npos)
        {
            tmpComputeInstanceId = Wildcarded {};
        }
        else
        {
            std::uint32_t computeInstanceId = -1;
            if (auto [ptr, ec]
                = std::from_chars(values[2].data(), values[2].data() + values[2].size(), computeInstanceId);
                ec != std::errc())
            {
                return ParsedUnknown {};
            }

            tmpComputeInstanceId = computeInstanceId;
        }
    }

    switch (values.size())
    {
        case 1:
            return ParsedGpu { tmpGpuUuid };
        case 2:
            return ParsedGpuI { tmpGpuUuid, tmpInstanceId };
        case 3:
            return ParsedGpuCi { tmpGpuUuid, tmpInstanceId, tmpComputeInstanceId };
        default:
            return ParsedUnknown {};
    }
}

ParsedGpu::ParsedGpu(Wildcard<std::string> uuid)
    : gpuUuid(std::move(uuid))
{}

bool ParsedGpu::operator==(ParsedGpu const &other) const
{
    return compare(gpuUuid, other.gpuUuid);
}

ParsedGpuI::ParsedGpuI(Wildcard<std::string> uuid, Wildcard<std::uint32_t> instanceId)
    : gpuUuid(std::move(uuid))
    , instanceId(instanceId)
{}

bool ParsedGpuI::operator==(ParsedGpuI const &other) const
{
    return std::tuple(compare(gpuUuid, other.gpuUuid), compare(instanceId, other.instanceId)) == std::tuple(true, true);
}

ParsedGpuCi::ParsedGpuCi(Wildcard<std::string> uuid,
                         Wildcard<std::uint32_t> instanceId,
                         Wildcard<std::uint32_t> computeInstanceId)
    : gpuUuid(std::move(uuid))
    , instanceId(instanceId)
    , computeInstanceId(computeInstanceId)
{}

bool ParsedGpuCi::operator==(ParsedGpuCi const &other) const
{
    return std::tuple(compare(gpuUuid, other.gpuUuid),
                      compare(instanceId, other.instanceId),
                      compare(computeInstanceId, other.computeInstanceId))
           == std::tuple(true, true, true);
}

bool ParsedUnknown::operator==(ParsedUnknown const &obj) const
{
    return this == &obj;
}


std::ostream &operator<<(std::ostream &os, ParseResult const &val)
{
    std::visit(
        overloaded(
            [&os](ParsedUnknown const &) { os << "ParsedUnknown"; },
            [&os](ParsedGpu const &obj) { os << fmt::format("ParsedGpu({})", obj.gpuUuid); },
            [&os](ParsedGpuI const &obj) { os << fmt::format("ParsedGpuI({},{})", obj.gpuUuid, obj.instanceId); },
            [&os](ParsedGpuCi const &obj) {
                os << fmt::format("ParsedGpuCi({},{},{})", obj.gpuUuid, obj.instanceId, obj.computeInstanceId);
            }),
        val);

    return os;
}

} // namespace DcgmNs
