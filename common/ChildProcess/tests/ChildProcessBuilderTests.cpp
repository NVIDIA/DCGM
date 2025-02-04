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

#include <catch2/catch_all.hpp>

#include <csignal>
#include <fmt/format.h>

#include <ChildProcess.hpp>
#include <ChildProcessBuilder.hpp>

using namespace DcgmNs::Common::Subprocess;

TEST_CASE("ChildProcessBuilder: Build with StdOut")
{
    auto process = ChildProcessBuilder {}
                       .SetExecutable("./childprocesstesttool")
                       .AddArg("stdout")
                       .AddArg("Hello, World!")
                       .Build();
    process.Run();

    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(stdOutStr == "Hello, World!");
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcessBuilder: Build with StdErr")
{
    auto process
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("stderr").AddArg("Capoo").Build();
    process.Run();

    auto &stdErrLines     = process.StdErr();
    std::string stdErrStr = fmt::to_string(fmt::join(stdErrLines.begin(), stdErrLines.end(), "\n"));
    REQUIRE(stdErrStr == "Capoo");
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcessBuilder: Build with Environment Variables")
{
    std::unordered_map<std::string, std::string> env {
        { "DCGM_TEST_ENV_KEY", "Capoo" },
    };
    auto process = ChildProcessBuilder {}
                       .SetExecutable("./childprocesstesttool")
                       .AddEnvironment(env)
                       .AddArg("env")
                       .AddArg("DCGM_TEST_ENV_KEY")
                       .Build();
    process.Run();

    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(stdOutStr == "Capoo");
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcessBuilder: Build with Channel File Descriptor")
{
    auto process = ChildProcessBuilder {}
                       .SetExecutable("./childprocesstesttool")
                       .SetChannelFd(6)
                       .AddArg("fd-channel")
                       .AddArg("6")
                       .AddArg("Capoo")
                       .Build();
    process.Run();

    auto &channel = process.GetFdChannel();
    for (auto const &frame : channel)
    {
        std::string data(reinterpret_cast<char const *>(frame.data()), frame.size());
        REQUIRE(data == "Capoo");
    }
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcessBuilder: Build with ReceivedSignal")
{
    auto process = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("6").Build();
    process.Run();
    REQUIRE(process.GetPid() != std::nullopt);
    kill(*process.GetPid(), SIGUSR1);
    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(stdOutStr.empty());
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() != 0);
    REQUIRE(process.ReceivedSignal() != std::nullopt);
    REQUIRE(*process.ReceivedSignal() == SIGUSR1);
}
