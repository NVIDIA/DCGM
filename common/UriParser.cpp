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

#include "UriParser.hpp"

//*****
#include <sys/socket.h>
// sys/socket and linux/vm_sockets should be included in strict order
#include <linux/vm_sockets.h>
#include <sys/un.h>
//*****
#include <DcgmLogging.h>
#include <dcgm_structs.h>
#include <netinet/in.h>


static bool StartsWithCaseless(std::string_view haystack, std::string_view needle)
{
    if (haystack.length() < needle.length())
    {
        return false;
    }
    return std::equal(needle.begin(), needle.end(), haystack.begin(), [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
    });
}

namespace DcgmNs::Common
{

bool TcpEndpoint::IsBindAny() const
{
    return host == "0.0.0.0" || host == "::";
}

bool VsockEndpoint::IsBindAnyCid() const
{
    return cid == VMADDR_CID_ANY;
}

std::optional<EndpointVariant> ParseEndpoint(std::string_view uri)
{
    std::string_view constexpr schemeSeparator = "://";

    auto const separatorPos = uri.find(schemeSeparator);
    std::string_view scheme;
    std::string_view rest;
    // If no scheme separator, presume default scheme TCP
    if (separatorPos == std::string_view::npos)
    {
        scheme = "tcp";
        rest   = uri;
    }
    else
    {
        scheme = uri.substr(0, separatorPos);
        rest   = uri.substr(separatorPos + schemeSeparator.length());
    }

    if (rest.empty())
    {
        log_warning("Rejecting empty URI");
        return std::nullopt;
    }

    if (StartsWithCaseless(scheme, "unix"))
    {
        // Check max length of the path
        size_t constexpr SUN_PATH_MAX = sizeof(sockaddr_un::sun_path);
        if (rest.length() > SUN_PATH_MAX)
        {
            log_warning("Rejecting Unix Domain socket path {} with excessive length {}", rest, rest.length());
            return std::nullopt;
        }
        return UnixEndpoint { std::string(rest) };
    }
    else if (StartsWithCaseless(scheme, "vsock"))
    {
        // --- Parse VSOCK: CID_or_-1:port ---
        size_t colonPos = rest.find(':');
        if (colonPos == std::string_view::npos || colonPos == 0 || colonPos == rest.length() - 1)
        {
            log_warning("Rejecting VSOCK URI {} with missing or misplaced colon", rest);
            return std::nullopt; // Missing or misplaced colon
        }

        auto cidPart  = rest.substr(0, colonPos);
        auto portPart = rest.substr(colonPos + 1);

        std::uint32_t cidVal;

        // Check for the special "-1" CID value for VMADDR_CID_ANY
        if (cidPart == "-1")
        {
            cidVal = VMADDR_CID_ANY; // Assign the defined constant (0xFFFFFFFF)
        }
        else
        {
            // Otherwise, parse as a normal unsigned integer
            auto cidResult = std::from_chars(cidPart.data(), cidPart.data() + cidPart.length(), cidVal);
            // Allow hex prefix 0x/0X? std::from_chars default base 10.
            if (cidResult.ec != std::errc {} || cidResult.ptr != cidPart.data() + cidPart.length())
            {
                log_warning("Rejecting VSOCK URI {} with invalid CID {}", rest, cidPart);
                return std::nullopt; // CID parsing failed
            }
            // Ensure parsed CID isn't accidentally the ANY value if "-1" wasn't used.
            // Although technically 0xFFFFFFFF is a valid CID according to vsock.h,
            // disallowing it unless "-1" was specified might prevent confusion.
            // Let's allow it for now, as 0xFFFFFFFF could be a specific target/listen CID.
            // if (cidVal == VSOCK_CID_ANY) return std::nullopt; // Uncomment to disallow direct 0xFFFFFFFF input
        }

        // Parse Port
        std::uint16_t portVal;
        auto portResult = std::from_chars(portPart.data(), portPart.data() + portPart.length(), portVal);
        if (portResult.ec != std::errc {} || portResult.ptr != portPart.data() + portPart.length())
        {
            log_warning("Rejecting VSOCK URI {} with invalid port {}", rest, portPart);
            return std::nullopt; // Port parsing failed
        }

        // Check if port is in valid range. Allowed ports are 1-65535.
        // No need to check for 65536 since we use uint16_t.
        if (portVal == 0)
        {
            log_warning("Rejecting VSOCK URI {} with invalid port {}", rest, portPart);
            return std::nullopt;
        }

        return VsockEndpoint { cidVal, portVal };
    }
    else if (StartsWithCaseless(scheme, "tcp"))
    {
        // --- Parse TCP: hostname_or_ipv4_or_ipv6_or_any:port ---
        // Handles: "host:port", "1.2.3.4:port", "0.0.0.0:port", "[::]:port", "[::1]:port", "host"

        std::string hostPart, portPart;
        size_t const lastColonPos = rest.rfind(':');
        if (lastColonPos == rest.length() - 1)
        {
            // Colon is the last character (missing port)
            // Note: We *don't* reject if lastColonPos == 0 here yet,
            // because that could be a valid IPv6 like "[::]:port" after bracket removal.
            log_warning("Rejecting TCP URI {} with missing or misplaced colon", rest);
            return std::nullopt;
        }
        // If the string starts with '[' and ends with ']', it is a valid IPv6 address in brackets,
        // so use the whole string as the host part.
        // Also use the whole string as the host part if there is no colon.
        else if (lastColonPos == std::string_view::npos || (rest.front() == '[' && rest.back() == ']'))
        {
            hostPart = rest;
        }
        else
        {
            hostPart = rest.substr(0, lastColonPos);
            portPart = rest.substr(lastColonPos + 1);
        }

        // Handle IPv6 address in brackets like "[::]" or "[::1]"
        if (hostPart.starts_with('[') && hostPart.ends_with(']'))
        {
            if (hostPart.length() < 4) // The minimum valid IPv6 address is "[::]"
            {
                log_warning("Rejecting TCP URI {} with invalid IPv6 address", rest);
                return std::nullopt;
            }
            // Check max length of the string accomodating [' ... ']' ':' port
            if ((2 * INET6_ADDRSTRLEN) < rest.length())
            {
                log_warning("Rejecting TCP URI {} with excessive length {}", rest, rest.length());
                return std::nullopt;
            }
            hostPart = hostPart.substr(1, hostPart.length() - 2); // Remove brackets
            // Now hostPart is "::" or "::1" or "2001:db8::a0a" etc.
            // Check if the content inside brackets is empty after stripping.
            if (hostPart.empty())
            {
                log_warning("Rejecting TCP URI {} with invalid IPv6 address", rest);
                return std::nullopt; // Invalid like "[]:port"
            }
        }
        else
        {
            // If not bracketed, it must not contain a colon.
            if (hostPart.find(':') != std::string_view::npos)
            {
                log_warning("Rejecting TCP URI {} with invalid address", rest);
                return std::nullopt; // Invalid format like "host:name:port" or bare "::1:port"
            }
        }

        // After bracket removal (if any), check if host part is empty.
        // This handles the case "tcp://:port" which we decided against supporting.
        // It also handles "tcp://[]:port".
        if (hostPart.empty())
        {
            log_warning("Rejecting TCP URI {} with empty hostname", rest);
            return std::nullopt;
        }

        // Parse port
        if (portPart.empty())
        {
            // No port specified, use default port DCGM_HE_PORT_NUMBER
            return TcpEndpoint { std::string(hostPart), DCGM_HE_PORT_NUMBER };
        }

        std::uint16_t portVal;
        auto result = std::from_chars(portPart.data(), portPart.data() + portPart.length(), portVal);
        if (result.ec != std::errc {} || result.ptr != portPart.data() + portPart.length())
        {
            log_warning("Rejecting TCP URI {} with invalid port {}", rest, portPart);
            return std::nullopt; // Port parsing failed
        }

        // Check if port is in valid range. Allowed ports are 1-65535.
        // No need to check for 65536 since we use uint16_t.
        if (portVal == 0)
        {
            log_warning("Rejecting TCP URI {} with invalid port {}", rest, portPart);
            return std::nullopt;
        }

        return TcpEndpoint { std::string(hostPart), portVal };
    }
    // Unknown scheme
    log_warning("Rejecting unknown scheme {}", scheme);
    return std::nullopt;
}

} //namespace DcgmNs::Common
