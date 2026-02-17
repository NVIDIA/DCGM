/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "MockFileSystemOperator.h"
#include <catch2/catch_all.hpp>
#include <thread>
#include <vector>

TEST_CASE("MockFileSystemOperator Read")
{
    MockFileSystemOperator mockFs;

    SECTION("Read returns mocked content")
    {
        std::string path    = "/mock/path/file.txt";
        std::string content = "Mock file content";
        mockFs.MockFileContent(path, content);

        auto result = mockFs.Read(path);
        REQUIRE(result.has_value());
        REQUIRE(result.value() == content);
    }

    SECTION("Read returns nullopt for unknown path")
    {
        std::string path = "/mock/path/unknown.txt";
        auto result      = mockFs.Read(path);
        REQUIRE(!result.has_value());
    }
}

TEST_CASE("MockFileSystemOperator Glob")
{
    MockFileSystemOperator mockFs;

    SECTION("Glob returns mocked paths")
    {
        std::string pattern            = "/mock/path/*.txt";
        std::vector<std::string> paths = { "/mock/path/file1.txt", "/mock/path/file2.txt" };
        mockFs.MockGlob(pattern, paths);

        auto result = mockFs.Glob(pattern);
        REQUIRE(result.has_value());
        REQUIRE(result.value() == paths);
    }

    SECTION("Glob returns nullopt for unknown pattern")
    {
        std::string pattern = "/mock/path/*.unknown";
        auto result         = mockFs.Glob(pattern);
        REQUIRE(!result.has_value());
    }
}

TEST_CASE("MockFileSystemOperator Concurrent Access")
{
    MockFileSystemOperator mockFs;
    std::string path    = "/mock/path/file.txt";
    std::string content = "Mock file content";
    mockFs.MockFileContent(path, content);

    constexpr int NUM_THREADS = 10;
    std::vector<std::jthread> threads;
    std::atomic<int> successCount = 0;

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&]() {
            auto result = mockFs.Read(path);
            if (result.has_value() && result.value() == content)
            {
                successCount++;
            }
        });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    REQUIRE(successCount == NUM_THREADS);
}

TEST_CASE("MockFileSystemOperator Edge Cases")
{
    MockFileSystemOperator mockFs;

    SECTION("Empty path")
    {
        auto result = mockFs.Read("");
        REQUIRE(!result.has_value());
    }

    SECTION("Very long path")
    {
        std::string longPath(1000, 'a');
        auto result = mockFs.Read(longPath);
        REQUIRE(!result.has_value());
    }

    SECTION("Special characters in path")
    {
        std::string specialPath = "/mock/path/!@#$%^&*()_+.txt";
        auto result             = mockFs.Read(specialPath);
        REQUIRE(!result.has_value());
    }
}

TEST_CASE("MockFileSystemOperator Invalid Inputs")
{
    MockFileSystemOperator mockFs;

    SECTION("Invalid path")
    {
        std::string invalidPath = "\0";
        auto result             = mockFs.Read(invalidPath);
        REQUIRE(!result.has_value());
    }

    SECTION("Invalid pattern")
    {
        std::string invalidPattern = "\0";
        auto result                = mockFs.Glob(invalidPattern);
        REQUIRE(!result.has_value());
    }
}
