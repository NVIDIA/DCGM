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

#include <TempDir.hpp>
#include <TempExecutable.hpp>
#include <TempFile.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace
{
std::string ReadFile(std::filesystem::path const &path)
{
    std::ifstream file(path);
    return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
}
} // namespace

TEST_CASE("TempFile::Random creates a random temporary file and removes it")
{
    std::filesystem::path tempPath;

    {
        auto tempFile = TempFile::Random("random contents", "dcgm-temp-file-random-");
        tempPath      = tempFile.Path();

        REQUIRE(std::filesystem::exists(tempPath));
        CHECK(tempPath.filename().string().rfind("dcgm-temp-file-random-", 0) == 0);
        CHECK(ReadFile(tempPath) == "random contents");
    }

    CHECK_FALSE(std::filesystem::exists(tempPath));
}

TEST_CASE("TempFile::Fixed creates a fixed-path temporary file and removes it")
{
    auto tempDir         = TempDir::Random("dcgm-temp-file-fixed-");
    auto const fixedPath = tempDir.Path() / "fixed-file";

    {
        auto tempFile = TempFile::Fixed(fixedPath, "fixed contents");

        CHECK(tempFile.Path() == fixedPath);
        REQUIRE(std::filesystem::exists(fixedPath));
        CHECK(ReadFile(fixedPath) == "fixed contents");
    }

    CHECK_FALSE(std::filesystem::exists(fixedPath));
}

TEST_CASE("TempDir::Random creates a random temporary directory and removes it")
{
    std::filesystem::path tempPath;

    {
        auto tempDir = TempDir::Random("dcgm-temp-dir-random-");
        tempPath     = tempDir.Path();

        REQUIRE(std::filesystem::is_directory(tempPath));
        CHECK(tempPath.filename().string().rfind("dcgm-temp-dir-random-", 0) == 0);
    }

    CHECK_FALSE(std::filesystem::exists(tempPath));
}

TEST_CASE("TempDir::Fixed creates a fixed-path temporary directory and removes it")
{
    auto parentDir       = TempDir::Random("dcgm-temp-dir-fixed-parent-");
    auto const fixedPath = parentDir.Path() / "fixed-dir";

    {
        auto tempDir = TempDir::Fixed(fixedPath);

        CHECK(tempDir.Path() == fixedPath);
        REQUIRE(std::filesystem::is_directory(fixedPath));
        std::ofstream(fixedPath / "entry.txt") << "entry";
    }

    CHECK_FALSE(std::filesystem::exists(fixedPath));
}

TEST_CASE("TempExecutable::Random creates a random executable file and removes it")
{
    std::filesystem::path tempPath;

    {
        auto tempExecutable = TempExecutable::Random("#!/bin/sh\n", "dcgm-temp-executable-random-");
        tempPath            = tempExecutable.Path();

        REQUIRE(std::filesystem::exists(tempPath));
        CHECK(tempPath.filename().string().rfind("dcgm-temp-executable-random-", 0) == 0);
        CHECK(ReadFile(tempPath) == "#!/bin/sh\n");
        CHECK((std::filesystem::status(tempPath).permissions() & std::filesystem::perms::owner_exec)
              != std::filesystem::perms::none);
    }

    CHECK_FALSE(std::filesystem::exists(tempPath));
}

TEST_CASE("TempExecutable::Fixed creates a fixed-path executable file and removes it")
{
    auto tempDir         = TempDir::Random("dcgm-temp-executable-fixed-");
    auto const fixedPath = tempDir.Path() / "fixed-executable";

    {
        auto tempExecutable = TempExecutable::Fixed(fixedPath, "#!/bin/sh\n");

        CHECK(tempExecutable.Path() == fixedPath);
        REQUIRE(std::filesystem::exists(fixedPath));
        CHECK(ReadFile(fixedPath) == "#!/bin/sh\n");
        CHECK((std::filesystem::status(fixedPath).permissions() & std::filesystem::perms::owner_exec)
              != std::filesystem::perms::none);
    }

    CHECK_FALSE(std::filesystem::exists(fixedPath));
}
