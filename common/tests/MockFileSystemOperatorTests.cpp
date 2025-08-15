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

#include "MockFileSystemOperator.h"
#include <catch2/catch_all.hpp>

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