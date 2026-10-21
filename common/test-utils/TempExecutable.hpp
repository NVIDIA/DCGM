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

#pragma once

#include <Defer.hpp>
#include <FilePermissions.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <utility>

/**
 * Creates an executable test file at a requested path and removes it when the test scope exits.
 */
class TempExecutable
{
public:
    /**
     * Creates an executable test file at an exact path.
     *
     * @param path Filesystem path to create.
     * @param contents Contents to write to the executable file.
     * @param executablePermissions Executable permissions to add to the file.
     */
    [[nodiscard]] static TempExecutable Fixed(std::filesystem::path path,
                                              std::string_view contents = "",
                                              std::filesystem::perms executablePermissions
                                              = std::filesystem::perms::owner_exec)
    {
        return TempExecutable(std::move(path), contents, executablePermissions);
    }

    /**
     * Creates an executable test file at a random path under the system temporary directory.
     *
     * @param contents Contents to write to the executable file.
     * @param prefix Filename prefix to use before the random suffix.
     * @param executablePermissions Executable permissions to add to the file.
     */
    [[nodiscard]] static TempExecutable Random(std::string_view contents = "",
                                               std::string_view prefix   = "dcgm-test-executable-",
                                               std::filesystem::perms executablePermissions
                                               = std::filesystem::perms::owner_exec)
    {
        return TempExecutable(CreateRandomPath(prefix), contents, executablePermissions);
    }

    TempExecutable(TempExecutable const &)            = delete;
    TempExecutable(TempExecutable &&)                 = delete;
    TempExecutable &operator=(TempExecutable const &) = delete;
    TempExecutable &operator=(TempExecutable &&)      = delete;

    ~TempExecutable() noexcept
    {
        Cleanup();
    }

    [[nodiscard]] std::filesystem::path const &Path() const
    {
        return m_path;
    }

private:
    explicit TempExecutable(std::filesystem::path path,
                            std::string_view contents                    = "",
                            std::filesystem::perms executablePermissions = std::filesystem::perms::owner_exec)
        : m_path(std::move(path))
    {
        auto cleanup = DcgmNs::Defer([this] { Cleanup(); });

        WriteContents(contents);
        FilePermissions::MakeExecutable(m_path, executablePermissions);
        cleanup.Disarm();
    }

    void WriteContents(std::string_view contents)
    {
        std::ofstream file(m_path);
        REQUIRE(file.is_open());
        file << contents;
        REQUIRE(file);
        file.close();
        REQUIRE(file);
    }

    [[nodiscard]] static std::filesystem::path CreateRandomPath(std::string_view prefix)
    {
        std::error_code error;
        auto const tempDirectory = std::filesystem::temp_directory_path(error);
        REQUIRE_FALSE(error);

        auto path       = tempDirectory / std::string(prefix).append("XXXXXX");
        auto pathString = path.string();

        int fd = mkstemp(pathString.data());
        REQUIRE(fd != -1);

        auto const closeResult = close(fd);
        if (closeResult == -1)
        {
            std::error_code ignoredError;
            std::filesystem::remove(pathString, ignoredError);
        }
        REQUIRE(closeResult != -1);

        return pathString;
    }

    void Cleanup() noexcept
    {
        std::error_code error;
        std::filesystem::remove(m_path, error);
    }

    std::filesystem::path m_path;
};
