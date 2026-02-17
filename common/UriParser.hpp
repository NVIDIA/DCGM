// Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>


namespace DcgmNs::Common
{

struct TcpEndpoint
{
    std::string host; // Can be hostname, IPv4, IPv6, "0.0.0.0", or "::"
    std::uint16_t port;

    [[nodiscard]] bool IsBindAny() const;

    bool operator==(const TcpEndpoint &) const = default;
};

struct UnixEndpoint
{
    std::string path; // Path to the Unix domain socket file
    bool operator==(const UnixEndpoint &) const = default;
};

struct VsockEndpoint
{
    std::uint32_t cid; // Context Identifier (CID). VSOCK_CID_ANY (-1) represents "any".
    std::uint16_t port;

    // Helper to check if this represents a "bind to any" CID
    [[nodiscard]] bool IsBindAnyCid() const;

    bool operator==(const VsockEndpoint &) const = default;
};

using EndpointVariant = std::variant<TcpEndpoint, UnixEndpoint, VsockEndpoint>;

std::optional<EndpointVariant> ParseEndpoint(std::string_view uri);

} //namespace DcgmNs::Common
