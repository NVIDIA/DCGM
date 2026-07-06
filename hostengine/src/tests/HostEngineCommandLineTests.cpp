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

#include "HostEngineCommandLine.h"
#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <tclap/CmdLine.h>

#include <string>
#include <string_view>
#include <vector>

namespace
{
HostEngineCommandLine Parse(std::vector<std::string> args)
{
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &arg : args)
    {
        argv.push_back(arg.data());
    }
    return ParseCommandLine(static_cast<int>(argv.size()), argv.data());
}
} //namespace

// This test calls ParseBindIp() directly because when ParseCommandLine() fails, TCLAP::ExitException terminates the
// test harness.
TEST_CASE("HostEngineCommandLine::ParseBindIp")
{
    struct
    {
        std::string_view input;
        std::string_view expectedValue;
    } const tests[] = {
        { "", "" },
        { "[", "[" },
        { "]", "]" },
        { "[]", "[]" },
        { "[a", "[a" },
        { "a]", "a]" },
        { "::", "::" },
        { "[ab", "[ab" },
        { "ba]", "ba]" },
        { "all", "" },
        { "ALL", "" },
        { "[abc]", "abc" },
        { "[::]", "::" },
        { "[::1]", "::1" },
        { "127.0.0.1", "127.0.0.1" },
        { "1234:abcd:5678:efab:90ab", "1234:abcd:5678:efab:90ab" },
        { "[1234:abcd:5678:efab:90ab]", "1234:abcd:5678:efab:90ab" },
        { "{invalid}", "{invalid}" },
    };

    HostEngineCommandLineInterface hecl {};

    for (auto const &test : tests)
    {
        DYNAMIC_SECTION("input=" << test.input << ", expected=" << test.expectedValue)
        {
            auto result = hecl.ParseBindIp(test.input.data());
            CHECK(result == test.expectedValue);
        }
    }
}

TEST_CASE("HostEngineCommandLine::ParseCommandLine")
{
    SECTION("Default arguments configure TCP on localhost")
    {
        auto parsed = Parse({ "nv-hostengine" });

        CHECK(parsed.GetConnectionType() == DcgmConnectionTypeTcp);
        CHECK(parsed.GetPort() == 5555);
        CHECK(parsed.GetBindInterface() == "127.0.0.1");
        CHECK(parsed.GetUnixSocketPath() == "/tmp/nv-hostengine");
        CHECK(parsed.GetVsockCid().empty());
        CHECK(parsed.ShouldDaemonize());
        CHECK_FALSE(parsed.ShouldTerminate());
        CHECK_FALSE(parsed.IsLogRotate());
        CHECK(parsed.GetDenylistedModules().empty());
    }

    SECTION("Domain socket without an explicit value uses the default socket path")
    {
        auto parsed = Parse({ "nv-hostengine", "-d" });

        CHECK(parsed.GetConnectionType() == DcgmConnectionTypeDomainSocket);
        CHECK(parsed.GetUnixSocketPath() == "/tmp/nv-hostengine");
        CHECK(parsed.GetBindInterface() == "127.0.0.1");
    }

    SECTION("Domain socket accepts an explicit socket path")
    {
        auto parsed = Parse({ "nv-hostengine", "--domain-socket", "/tmp/custom.sock" });

        CHECK(parsed.GetConnectionType() == DcgmConnectionTypeDomainSocket);
        CHECK(parsed.GetUnixSocketPath() == "/tmp/custom.sock");
    }

    SECTION("Vsock uses the vsock connection type")
    {
        auto parsed = Parse({ "nv-hostengine", "--vsock-cid", "42" });

        CHECK(parsed.GetConnectionType() == DcgmConnectionTypeVsock);
        CHECK(parsed.GetVsockCid() == "42");
        CHECK(parsed.GetUnixSocketPath() == "/tmp/nv-hostengine");
    }

    SECTION("Explicit options populate all corresponding getters")
    {
        auto parsed = Parse({ "nv-hostengine",
                              "--port",
                              "6000",
                              "--bind-interface",
                              "[::1]",
                              "--pid",
                              "/tmp/hostengine.pid",
                              "--log-level",
                              "DEBUG",
                              "--log-filename",
                              "/tmp/hostengine.log",
                              "--log-rotate",
                              "--denylist-modules",
                              "1,2",
                              "--service-account",
                              "nvidia-dcgm",
                              "--home-dir",
                              "/tmp/dcgm",
                              "--no-daemon",
                              "--term" });

        CHECK(parsed.GetConnectionType() == DcgmConnectionTypeTcp);
        CHECK(parsed.GetPort() == 6000);
        CHECK(parsed.GetBindInterface() == "::1");
        CHECK(parsed.GetPidFilePath() == "/tmp/hostengine.pid");
        CHECK(parsed.GetLogLevel() == "DEBUG");
        CHECK(parsed.GetLogFileName() == "/tmp/hostengine.log");
        CHECK(parsed.IsLogRotate());
        CHECK(parsed.GetDenylistedModules().contains(DcgmModuleIdNvSwitch));
        CHECK(parsed.GetDenylistedModules().contains(DcgmModuleIdVGPU));
        CHECK(parsed.GetServiceAccount() == "nvidia-dcgm");
        CHECK(parsed.GetHomeDir() == "/tmp/dcgm");
        CHECK_FALSE(parsed.ShouldDaemonize());
        CHECK(parsed.ShouldTerminate());
    }

    SECTION("Conflicting transport options are rejected")
    {
        REQUIRE_THROWS_MATCHES(Parse({ "nv-hostengine", "--domain-socket", "/tmp/a.sock", "--vsock-cid", "3" }),
                               std::runtime_error,
                               Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                                   "Cannot specify both Unix domain socket and Vsock CID options.")));
        REQUIRE_THROWS_MATCHES(Parse({ "nv-hostengine", "--bind-interface", "127.0.0.1", "--vsock-cid", "3" }),
                               std::runtime_error,
                               Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                                   "Cannot specify both TCP interface and Vsock CID options.")));
        REQUIRE_THROWS_MATCHES(
            Parse({ "nv-hostengine", "--bind-interface", "127.0.0.1", "--domain-socket", "/tmp/a.sock" }),
            std::runtime_error,
            Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                "Cannot specify both TCP interface and Unix domain socket options.")));
    }
}
