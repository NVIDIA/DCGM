// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <catch2/catch_test_macros.hpp>

#include <UriParser.hpp>

//*****
#include <sys/socket.h>
// sys/socket and linux/vm_sockets should be included in strict order
#include <linux/vm_sockets.h>
//*****
#include <dcgm_structs.h>
#include <netinet/in.h>

using namespace DcgmNs::Common;

TEST_CASE("TCP Endpoint Parsing")
{
    SECTION("Valid Hostname/IP")
    {
        auto result = ParseEndpoint("tcp://localhost:8080");
        REQUIRE(result.has_value());
        REQUIRE(std::holds_alternative<TcpEndpoint>(*result));
        auto *ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "localhost");
        CHECK(ep->port == 8080);
        CHECK_FALSE(ep->IsBindAny());

        result = ParseEndpoint("TCP://192.168.1.100:9000"); // Case-insensitive scheme
        REQUIRE(result.has_value());
        ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "192.168.1.100");
        CHECK(ep->port == 9000);
        CHECK_FALSE(ep->IsBindAny());

        result = ParseEndpoint("tcp://my-hostname.domain.com:12345");
        REQUIRE(result.has_value());
        ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "my-hostname.domain.com");
        CHECK(ep->port == 12345);
        CHECK_FALSE(ep->IsBindAny());

        result = ParseEndpoint("127.0.0.1:65535"); // No prefix implies TCP
        REQUIRE(result.has_value());
        ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "127.0.0.1");
        CHECK(ep->port == 65535);
        CHECK_FALSE(ep->IsBindAny());

        // Valid because we do not reject if scheme is missing, nor validate the hostname characters
        result = ParseEndpoint("tcp//localhost:8080");
        REQUIRE(result.has_value());
        ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "tcp//localhost");
        CHECK(ep->port == 8080);
        CHECK_FALSE(ep->IsBindAny());
    }

    SECTION("Valid IPv6")
    {
        auto result = ParseEndpoint("tcp://[::1]:80");
        REQUIRE(result.has_value());
        auto *ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "::1");
        CHECK(ep->port == 80);
        CHECK_FALSE(ep->IsBindAny());

        result = ParseEndpoint("tcp://[2001:db8::a0a]:443");
        REQUIRE(result.has_value());
        ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "2001:db8::a0a");
        CHECK(ep->port == 443);
        CHECK_FALSE(ep->IsBindAny());

        // Max valid IPv6 address is INET6_ADDRSTRLEN with brackets
        auto const &maxValidIpAddr = std::string((2 * INET6_ADDRSTRLEN) - 2, 'A');
        result                     = ParseEndpoint("tcp://[" + maxValidIpAddr + "]");
        REQUIRE(result.has_value());
        ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == maxValidIpAddr);
        CHECK(ep->port == DCGM_HE_PORT_NUMBER);
        CHECK_FALSE(ep->IsBindAny());
    }

    SECTION("Valid Bind Any Addresses")
    {
        auto result = ParseEndpoint("tcp://0.0.0.0:8080");
        REQUIRE(result.has_value());
        auto *ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "0.0.0.0");
        CHECK(ep->port == 8080);
        CHECK(ep->IsBindAny());

        result = ParseEndpoint("tcp://[::]:443");
        REQUIRE(result.has_value());
        ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "::");
        CHECK(ep->port == 443);
        CHECK(ep->IsBindAny());
    }

    SECTION("Valid Default Port")
    {
        auto result = ParseEndpoint("hostnameX"); // No port implies default port DCGM_HE_PORT_NUMBER
        REQUIRE(result.has_value());
        auto *ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "hostnameX");
        CHECK(ep->port == DCGM_HE_PORT_NUMBER);
        CHECK_FALSE(ep->IsBindAny());

        result = ParseEndpoint("[1234:abcd:5678:efab:9012:cdef:3456:abcd]");
        REQUIRE(result.has_value());
        ep = std::get_if<TcpEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->host == "1234:abcd:5678:efab:9012:cdef:3456:abcd");
        CHECK(ep->port == DCGM_HE_PORT_NUMBER);
        CHECK_FALSE(ep->IsBindAny());
    }
}

TEST_CASE("UNIX Endpoint Parsing")
{
    SECTION("Valid Paths")
    {
        auto result = ParseEndpoint("unix:///tmp/my_app.sock");
        REQUIRE(result.has_value());
        REQUIRE(std::holds_alternative<UnixEndpoint>(*result));
        auto *ep = std::get_if<UnixEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->path == "/tmp/my_app.sock");

        result = ParseEndpoint("UNIX:///var/run/docker.sock"); // Case-insensitive scheme
        REQUIRE(result.has_value());
        ep = std::get_if<UnixEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->path == "/var/run/docker.sock");

        result = ParseEndpoint("unix://relative/path.sock");
        REQUIRE(result.has_value());
        ep = std::get_if<UnixEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->path == "relative/path.sock");

        size_t constexpr SUN_PATH_MAX = 108;
        auto maxValidUnixPath         = std::string(SUN_PATH_MAX, 'a');
        result                        = ParseEndpoint("unix://" + maxValidUnixPath);
        REQUIRE(result.has_value());
        ep = std::get_if<UnixEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->path == maxValidUnixPath);
    }
}

TEST_CASE("VSOCK Endpoint Parsing")
{
    SECTION("Valid CID/Port")
    {
        auto result = ParseEndpoint("vsock://2:1024");
        REQUIRE(result.has_value());
        REQUIRE(std::holds_alternative<VsockEndpoint>(*result));
        auto *ep = std::get_if<VsockEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->cid == 2);
        CHECK(ep->port == 1024);
        CHECK_FALSE(ep->IsBindAnyCid());

        result = ParseEndpoint("vsock://1:6000"); // Host CID
        REQUIRE(result.has_value());
        ep = std::get_if<VsockEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->cid == 1);
        CHECK(ep->port == 6000);
        CHECK_FALSE(ep->IsBindAnyCid());

        result = ParseEndpoint("vsock://1234567890:5000"); // Large CID
        REQUIRE(result.has_value());
        ep = std::get_if<VsockEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->cid == 1234567890);
        CHECK(ep->port == 5000);
        CHECK_FALSE(ep->IsBindAnyCid());
    }

    SECTION("Valid Bind Any CID")
    {
        auto result = ParseEndpoint("vsock://-1:5000");
        REQUIRE(result.has_value());
        auto *ep = std::get_if<VsockEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->cid == VMADDR_CID_ANY); // 0xFFFFFFFF
        CHECK(ep->port == 5000);
        CHECK(ep->IsBindAnyCid());

        // Check direct input of the ANY value
        result = ParseEndpoint("vsock://4294967295:5001");
        REQUIRE(result.has_value());
        ep = std::get_if<VsockEndpoint>(&*result);
        REQUIRE(ep != nullptr);
        CHECK(ep->cid == 0xFFFFFFFF); // Should parse correctly
        CHECK(ep->cid == VMADDR_CID_ANY);
        CHECK(ep->port == 5001);
        CHECK(ep->IsBindAnyCid()); // Technically true, as CID matches
    }
}

TEST_CASE("Invalid Endpoint Parsing")
{
    SECTION("Invalid Scheme / Format")
    {
        CHECK_FALSE(ParseEndpoint("https://example.com").has_value()); // Unsupported scheme
        CHECK_FALSE(ParseEndpoint("tcp:localhost:8080").has_value());  // Missing //
        CHECK_FALSE(ParseEndpoint("tcp://").has_value());              // Empty rest
        CHECK_FALSE(ParseEndpoint("vsock://").has_value());            // Empty CID:PORT
        CHECK_FALSE(ParseEndpoint("").has_value());                    // Empty string
    }

    SECTION("TCP Specific Invalid")
    {
        CHECK_FALSE(ParseEndpoint("tcp://localhost:").has_value());      // Missing port number
        CHECK_FALSE(ParseEndpoint("tcp://:8080").has_value());           // Missing host part (rejected form)
        CHECK_FALSE(ParseEndpoint("tcp://[]:8080").has_value());         // Empty brackets
        CHECK_FALSE(ParseEndpoint("tcp://[:]:8080").has_value());        // Malformed IPv6 brackets content
        CHECK_FALSE(ParseEndpoint("tcp://[::1:8080").has_value());       // Missing closing bracket
        CHECK_FALSE(ParseEndpoint("tcp://::1]:8080").has_value());       // Missing opening bracket
        CHECK_FALSE(ParseEndpoint("tcp://host:port_text").has_value());  // Invalid port text
        CHECK_FALSE(ParseEndpoint("tcp://host:99999").has_value());      // Port out of uint16_t range
        CHECK_FALSE(ParseEndpoint("tcp://host:-1").has_value());         // Invalid port text
        CHECK_FALSE(ParseEndpoint("tcp://host:name:8080").has_value());  // Colon in non-bracketed host
        CHECK_FALSE(ParseEndpoint("tcp://::1:8080").has_value());        // Bare IPv6 (requires brackets)
        CHECK_FALSE(ParseEndpoint("tcp://::").has_value());              // Bare IPv6 (requires brackets)
        CHECK_FALSE(ParseEndpoint("tcp://::1").has_value());             // Bare IPv6 (requires brackets)
        CHECK_FALSE(ParseEndpoint("tcp://:::1").has_value());            // Bare IPv6 (requires brackets)
        CHECK_FALSE(ParseEndpoint("tcp://127.0.0.1:0").has_value());     // Out of range port 0
        CHECK_FALSE(ParseEndpoint("tcp://127.0.0.1:65536").has_value()); // Out of range port 65536
        CHECK_FALSE(ParseEndpoint("1234:abcd:5678:efab:9abc:cdef:defc:0123:5566").has_value()); // Missing brackets
        CHECK_FALSE(ParseEndpoint("tcp://2001:db8::a0a").has_value());                          // Missing brackets

        auto const &maxValidIpAddr = std::string((2 * INET6_ADDRSTRLEN) - 2, 'A');
        auto const &invalidIpAddr  = maxValidIpAddr + "Z";
        CHECK_FALSE(ParseEndpoint("tcp://[" + invalidIpAddr + "]").has_value());
    }

    SECTION("VSOCK Specific Invalid")
    {
        CHECK_FALSE(ParseEndpoint("vsock://:1024").has_value());              // Missing CID
        CHECK_FALSE(ParseEndpoint("vsock://2:").has_value());                 // Missing Port
        CHECK_FALSE(ParseEndpoint("vsock://cid_text:1024").has_value());      // Invalid CID text
        CHECK_FALSE(ParseEndpoint("vsock://2:port_text").has_value());        // Invalid Port text
        CHECK_FALSE(ParseEndpoint("vsock://-2:1024").has_value());            // Invalid negative CID
        CHECK_FALSE(ParseEndpoint("vsock://-1:").has_value());                // Missing port after special CID
        CHECK_FALSE(ParseEndpoint("vsock://9999999999999:1024").has_value()); // CID out of uint32_t range
        CHECK_FALSE(ParseEndpoint("vsock://2:99999").has_value());            // Port out of uint16_t range
    }
    SECTION("Unix Specific Invalid")
    {
        CHECK_FALSE(ParseEndpoint("unix://").has_value()); // Empty path

        constexpr size_t SUN_PATH_MAX = 108;
        auto maxValidUnixPath         = std::string(SUN_PATH_MAX, 'a');
        CHECK_FALSE(ParseEndpoint("unix://" + maxValidUnixPath + "z").has_value());
    }
}
