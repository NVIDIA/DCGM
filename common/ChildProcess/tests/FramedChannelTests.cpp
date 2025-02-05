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

#include <ChildProcess/FramedChannel.hpp>

#include <string_view>
#include <thread>


TEST_CASE("FramedChannel: Write")
{
    using namespace DcgmNs::Common::Subprocess;
    auto testFunc = [](std::string_view message) {
        auto channel = FramedChannel {};

        std::uint32_t size = message.length();
        channel.Write({ reinterpret_cast<std::byte const *>(&size), sizeof(size) });

        while (!message.empty())
        {
            auto to_write = std::min(message.length(), size_t { 2 });
            channel.Write({ reinterpret_cast<std::byte const *>(message.data()), to_write });
            message.remove_prefix(to_write);
        }
        channel.Close();

        int count = 0;
        for (auto const &msg : channel)
        {
            REQUIRE(std::memcmp(msg.data(), message.data(), message.size()) == 0);
            count += 1;
        }
        REQUIRE(count == 1);
    };

    SECTION("Normal String")
    {
        std::string_view message = "Capoo";
        testFunc(message);
    }

    SECTION("Binary Data")
    {
        constexpr unsigned int binaryDataLen = 18;
        char const *message                  = "Capoo\0\1\2\3\4\5\6DogDog";
        testFunc(std::string_view(message, binaryDataLen));
    }

    SECTION("Large Message")
    {
        constexpr unsigned int buffSize = 65537;
        char buff[buffSize];
        std::memset(buff, 6, buffSize);
        testFunc(std::string_view(buff, buffSize));
    }
}

TEST_CASE("FramedChannel: Read & Write at the same time")
{
    using namespace DcgmNs::Common::Subprocess;

    auto channel = FramedChannel {};

    auto handle = std::jthread([&channel] {
        std::string_view message = "Hello, world!";

        std::uint32_t size = message.length();
        channel.Write({ reinterpret_cast<std::byte const *>(&size), sizeof(size) });

        while (!message.empty())
        {
            auto to_write = std::min(message.length(), size_t { 2 });
            channel.Write({ reinterpret_cast<std::byte const *>(message.data()), to_write });
            message.remove_prefix(to_write);
        }
        channel.Close();
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

    for (auto msg : channel)
    {
        auto const rMsg = std::string_view(reinterpret_cast<char const *>(msg.data()),
                                           reinterpret_cast<char const *>(msg.data() + msg.size()));
        REQUIRE(rMsg == "Hello, world!");
        ++count;
    }
    processed.store(true, std::memory_order::release);
    REQUIRE(count > 0);
}

TEST_CASE("FramedChannel: Single consumer")
{
    using namespace DcgmNs::Common::Subprocess;

    auto channel = FramedChannel {};

    std::string_view message = "Hello, world!";
    std::uint32_t length     = message.length();
    channel.Write({ reinterpret_cast<std::byte const *>(&length), sizeof(length) });
    channel.Write({ reinterpret_cast<std::byte const *>(message.data()), message.length() });

    channel.Write({ reinterpret_cast<std::byte const *>(&length), sizeof(length) });
    channel.Write({ reinterpret_cast<std::byte const *>(message.data()), message.length() });

    channel.Write({ reinterpret_cast<std::byte const *>(&length), sizeof(length) });
    channel.Write({ reinterpret_cast<std::byte const *>(message.data()), message.length() });

    {
        volatile auto it1 = channel.begin();
    }
    volatile auto it2 = channel.begin();
    REQUIRE_THROWS_AS(channel.begin(), FramedChannel::ConsumerOccupiedException);
}

TEST_CASE("FramedChannel: Close")
{
    using namespace DcgmNs::Common::Subprocess;

    auto channel = FramedChannel {};
    channel.Close();
    REQUIRE_THROWS_AS(channel.Write({}), FramedChannel::StreamClosedException);
}

TEST_CASE("FramedChannel: Read from closed channel")
{
    using namespace DcgmNs::Common::Subprocess;

    auto channel             = FramedChannel {};
    std::string_view message = "Hello, world!";
    std::uint32_t length     = message.length();
    channel.Write({ reinterpret_cast<std::byte const *>(&length), sizeof(length) });
    channel.Write({ reinterpret_cast<std::byte const *>(message.data()), message.length() });

    channel.Write({ reinterpret_cast<std::byte const *>(&length), sizeof(length) });
    channel.Write({ reinterpret_cast<std::byte const *>(message.data()), message.length() });

    channel.Write({ reinterpret_cast<std::byte const *>(&length), sizeof(length) });
    channel.Write({ reinterpret_cast<std::byte const *>(message.data()), message.length() });
    channel.Close();

    int count = 0;
    for (auto msg : channel)
    {
        auto const rMsg = std::string_view(reinterpret_cast<char const *>(msg.data()),
                                           reinterpret_cast<char const *>(msg.data() + msg.size()));
        REQUIRE(rMsg == "Hello, world!");
        ++count;
    }
    REQUIRE(count == 3);
}