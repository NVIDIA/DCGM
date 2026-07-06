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

#include <FileSystemOperator.h>

#include <Defer.hpp>

#include <catch2/catch_all.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace
{
class TempDir
{
public:
    TempDir()
    {
        std::string pattern = "/tmp/dcgm-fs-test-XXXXXX";
        char *path          = mkdtemp(pattern.data());
        REQUIRE(path != nullptr);
        m_path = path;
    }

    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(m_path, ec);
    }

    std::filesystem::path Path() const
    {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};
} // namespace

TEST_CASE("FileSystemOperator reads, globs, and removes files")
{
    FileSystemOperator fs;
    TempDir tempDir;
    auto const filePath = tempDir.Path() / "sample.txt";
    auto const linkPath = tempDir.Path() / "sample.link";

    GIVEN("a temporary file")
    {
        {
            std::ofstream file(filePath);
            file << "hello dcgm\n";
        }

        THEN("Read returns the file contents")
        {
            auto contents = fs.Read(filePath.string());
            REQUIRE(contents.has_value());
            CHECK(contents.value() == "hello dcgm\n");
        }

        THEN("Glob finds matching paths")
        {
            auto paths = fs.Glob((tempDir.Path() / "*.txt").string());
            REQUIRE(paths.has_value());
            CHECK_THAT(paths.value(), Catch::Matchers::VectorContains(filePath.string()));
        }

        THEN("Stat, Access, and IsDirectory reflect filesystem state")
        {
            struct stat st {};
            CHECK(fs.Stat(filePath.c_str(), &st) == 0);
            CHECK(fs.Access(filePath.c_str(), F_OK) == 0);
            CHECK_FALSE(fs.IsDirectory(filePath.string()));
            CHECK(fs.IsDirectory(tempDir.Path().string()));
        }

        THEN("ReadLink returns symlink target bytes")
        {
            std::filesystem::create_symlink(filePath, linkPath);
            char buf[256] = {};

            auto const bytes = fs.ReadLink(linkPath.string(), buf, sizeof(buf));
            REQUIRE(bytes > 0);
            CHECK(std::string(buf, static_cast<size_t>(bytes)) == filePath.string());
        }

        THEN("Unlink removes the file")
        {
            CHECK(fs.Unlink(filePath.string()));
            CHECK_FALSE(fs.Read(filePath.string()).has_value());
        }
    }
}

TEST_CASE("FileSystemOperator handles directories and process cwd")
{
    FileSystemOperator fs;
    TempDir tempDir;

    GIVEN("a temporary directory")
    {
        std::ofstream(tempDir.Path() / "entry.txt") << "entry";

        THEN("ListDirectoryEntries includes created files")
        {
            auto entries = fs.ListDirectoryEntries(tempDir.Path().string());
            REQUIRE(entries.has_value());
            CHECK_THAT(entries.value(), Catch::Matchers::VectorContains(std::string("entry.txt")));
        }

        THEN("GetCurrentWorkingDirectory and ChangeDirectory wrap cwd calls")
        {
            char original[PATH_MAX] = {};
            REQUIRE(fs.GetCurrentWorkingDirectory(original, sizeof(original)) != nullptr);
            std::string const originalPath = original;

            REQUIRE(fs.ChangeDirectory(tempDir.Path().c_str()) == 0);
            DcgmNs::Defer restoreCwd([&fs, originalPath] { CHECK(fs.ChangeDirectory(originalPath.c_str()) == 0); });

            char changed[PATH_MAX] = {};
            REQUIRE(fs.GetCurrentWorkingDirectory(changed, sizeof(changed)) != nullptr);
            CHECK(std::filesystem::equivalent(changed, tempDir.Path()));
        }
    }
}

TEST_CASE("FileSystemOperator returns failures for missing paths")
{
    FileSystemOperator fs;
    TempDir tempDir;
    auto const missing = tempDir.Path() / "missing";

    CHECK_FALSE(fs.Read(missing.string()).has_value());
    CHECK_FALSE(fs.Glob((tempDir.Path() / "*.missing").string()).has_value());
    CHECK_FALSE(fs.Unlink(missing.string()));
    CHECK_FALSE(fs.IsDirectory(missing.string()));
    CHECK_FALSE(fs.ListDirectoryEntries(missing.string()).has_value());
}
