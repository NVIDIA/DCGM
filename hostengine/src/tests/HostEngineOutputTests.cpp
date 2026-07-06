/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "HostEngineOutput.h"

#include <catch2/catch_all.hpp>

#include <tclap/ArgException.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
class StreamCapture
{
public:
    explicit StreamCapture(std::ostream &stream)
        : m_stream(stream)
        , m_old(stream.rdbuf(m_buffer.rdbuf()))
    {}

    ~StreamCapture()
    {
        m_stream.rdbuf(m_old);
    }

    std::string str() const
    {
        return m_buffer.str();
    }

private:
    std::ostream &m_stream;
    std::ostringstream m_buffer;
    std::streambuf *m_old;
};

struct TestCommandLine
{
    TCLAP::CmdLine cmd { "Hostengine test command line", ' ', "test-version" };
    TCLAP::SwitchArg verbose { "v", "verbose", "Enable verbose output with a deliberately long description.", false };
    TCLAP::ValueArg<int> port { "p", "port", "Listen port for the hostengine process.", false, 5555, "port" };
    TCLAP::SwitchArg fast { "f", "fast", "Start quickly.", false };
    TCLAP::SwitchArg safe { "s", "safe", "Start with extra validation.", false };

    TestCommandLine()
    {
        cmd.add(verbose);
        cmd.add(port);
        std::vector<TCLAP::Arg *> exclusiveArgs { &fast, &safe };
        cmd.xorAdd(exclusiveArgs);
    }
};
} //namespace

TEST_CASE("HostEngineOutput::usage")
{
    GIVEN("a TCLAP command line with exclusive and regular args")
    {
        TestCommandLine commandLine;
        HostEngineOutput output("This prologue wraps around the usage block.",
                                "This epilogue appears after the options.",
                                "hostengine version",
                                54);

        WHEN("usage is printed")
        {
            StreamCapture capture(std::cout);
            output.usage(commandLine.cmd);

            THEN("the output includes wrapped usage and long option details")
            {
                auto text = capture.str();
                CHECK(text.find("This prologue") != std::string::npos);
                CHECK(text.find("USAGE:") != std::string::npos);
                CHECK(text.find("Where:") != std::string::npos);
                CHECK(text.find("--port") != std::string::npos);
                CHECK(text.find("--verbose") != std::string::npos);
                CHECK(text.find("-- OR --") != std::string::npos);
                CHECK(text.find("This epilogue") != std::string::npos);
            }
        }
    }
}

TEST_CASE("HostEngineOutput::version")
{
    GIVEN("a hostengine output formatter")
    {
        TestCommandLine commandLine;
        HostEngineOutput output("", "", "hostengine 1.2.3", 20);

        WHEN("version is printed")
        {
            StreamCapture capture(std::cout);
            output.version(commandLine.cmd);

            THEN("the configured version string is emitted")
            {
                CHECK(capture.str().find("hostengine 1.2.3") != std::string::npos);
            }
        }
    }
}

TEST_CASE("HostEngineOutput::failure")
{
    GIVEN("a TCLAP parse failure")
    {
        TestCommandLine commandLine;
        HostEngineOutput output("prologue", "epilogue", "hostengine version", 64);
        TCLAP::ArgException exception("bad value", "--port");

        WHEN("failure output is requested")
        {
            StreamCapture capture(std::cerr);

            THEN("the formatter prints a parse error and throws TCLAP exit")
            {
                CHECK_THROWS_AS(output.failure(commandLine.cmd, exception), TCLAP::ExitException);
                auto text = capture.str();
                CHECK(text.find("PARSE ERROR") != std::string::npos);
                CHECK(text.find("--port") != std::string::npos);
                CHECK(text.find("Brief USAGE") != std::string::npos);
                CHECK(text.find("--help") != std::string::npos);
            }
        }
    }
}
