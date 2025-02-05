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

#include <ChildProcess/StdLines.hpp>

#include <cstring>
#include <string>
#include <thread>


TEST_CASE("StdLines: Write")
{
    using namespace DcgmNs::Common::Subprocess;
    auto testFunc = [](std::string const &message) {
        auto stdLines = StdLines {};

        stdLines.Write(message);
        stdLines.Close();

        int count = 0;
        for (auto const &msg : stdLines)
        {
            REQUIRE(std::memcmp(msg.data(), message.data(), message.size()) == 0);
            count += 1;
        }
        REQUIRE(count == 1);
    };

    SECTION("Normal String")
    {
        std::string const message = "Capoo";
        testFunc(message);
    }

    SECTION("Binary Data")
    {
        constexpr unsigned int binaryDataLen = 18;
        char const *message                  = "Capoo\0\1\2\3\4\5\6DogDog";
        testFunc(std::string(message, binaryDataLen));
    }

    SECTION("Large Message")
    {
        constexpr unsigned int buffSize = 65537;
        char buff[buffSize];
        std::memset(buff, 6, buffSize);
        testFunc(std::string(buff, buffSize));
    }
}

TEST_CASE("StdLines: Read & Write at the same time")
{
    using namespace DcgmNs::Common::Subprocess;

    auto stdLines = StdLines {};
    std::vector<std::string> expected {
        "Capoo",
        "Is",
        "Cute!",
    };

    auto handle = std::jthread([&stdLines, &expected] {
        for (auto const &msg : expected)
        {
            stdLines.Write(msg);
        }
        stdLines.Close();
    });

    std::atomic_bool processed = false;

    auto shortCircuit = std::jthread([&processed] {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!processed.load(std::memory_order::acquire))
        {
            fprintf(stderr, "Deadlock detected\n");
            fflush(stderr);
            std::terminate();
        }
    });

    int count = 0;
    for (auto const &msg : stdLines)
    {
        REQUIRE(msg == expected[count]);
        ++count;
    }
    processed.store(true, std::memory_order::release);
    REQUIRE(count == 3);
}

TEST_CASE("StdLines: Read from closed channel")
{
    using namespace DcgmNs::Common::Subprocess;

    auto stdLines = StdLines {};
    std::vector<std::string> expected {
        "Capoo",
        "Is",
        "Cute!",
    };
    for (auto const &msg : expected)
    {
        stdLines.Write(msg);
    }
    stdLines.Close();

    int count = 0;
    for (auto const &msg : stdLines)
    {
        REQUIRE(msg == expected[count]);
        ++count;
    }
    REQUIRE(count == 3);
}
