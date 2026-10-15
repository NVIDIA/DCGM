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

#include <catch2/catch_test_macros.hpp>

#include <ChildProcess.hpp>
#include <IoContext.hpp>

#include <fmt/format.h>

#include <string>

using namespace DcgmNs::Common::Subprocess;

class ChildProcessTestFixture
{
protected:
    ChildProcessTestFixture()
    {
        process.Create(ioContext, "unused");
    }

    IoContext ioContext;
    ChildProcess process;
};

SCENARIO_METHOD(ChildProcessTestFixture, "ChildProcess: GetStdOutBuffer preserves separators between stdout records")
{
    GIVEN("stdout contains multiple records")
    {
        process.StdOut().Write("Version: 1.1.1");
        process.StdOut().Write("Running on PCI BDF(s): 0002:c1:00.0");
        process.StdOut().Write("Starting RIST");
        process.StdOut().Close();

        WHEN("stdout is copied into a buffer")
        {
            fmt::memory_buffer outputBuffer;
            process.GetStdOutBuffer(outputBuffer, true);
            auto output = fmt::to_string(outputBuffer);

            THEN("the copied buffer keeps separators between records")
            {
                CHECK(output == "Version: 1.1.1\nRunning on PCI BDF(s): 0002:c1:00.0\nStarting RIST");
            }
        }
    }
}

SCENARIO_METHOD(ChildProcessTestFixture, "ChildProcess: GetStdErrBuffer preserves separators between stderr records")
{
    GIVEN("stderr contains multiple records")
    {
        process.StdErr().Write("error one");
        process.StdErr().Write("error two");
        process.StdErr().Write("error three");
        process.StdErr().Close();

        WHEN("stderr is copied into a buffer")
        {
            fmt::memory_buffer outputBuffer;
            process.GetStdErrBuffer(outputBuffer, true);
            auto output = fmt::to_string(outputBuffer);

            THEN("the copied buffer keeps separators between records")
            {
                CHECK(output == "error one\nerror two\nerror three");
            }
        }
    }
}

SCENARIO_METHOD(ChildProcessTestFixture, "ChildProcess: GetStdOutBuffer does not add a trailing stdout separator")
{
    GIVEN("stdout contains a final record")
    {
        process.StdOut().Write("first");
        process.StdOut().Write("partial");
        process.StdOut().Close();

        WHEN("stdout is copied into a buffer")
        {
            fmt::memory_buffer outputBuffer;
            process.GetStdOutBuffer(outputBuffer, true);
            auto output = fmt::to_string(outputBuffer);

            THEN("the final record is present without a synthetic trailing newline")
            {
                CHECK(output == "first\npartial");
            }
        }
    }
}

SCENARIO_METHOD(ChildProcessTestFixture, "ChildProcess: GetStdErrBuffer does not add a trailing stderr separator")
{
    GIVEN("stderr contains a final record")
    {
        process.StdErr().Write("first error");
        process.StdErr().Write("partial error");
        process.StdErr().Close();

        WHEN("stderr is copied into a buffer")
        {
            fmt::memory_buffer outputBuffer;
            process.GetStdErrBuffer(outputBuffer, true);
            auto output = fmt::to_string(outputBuffer);

            THEN("the final record is present without a synthetic trailing newline")
            {
                CHECK(output == "first error\npartial error");
            }
        }
    }
}
