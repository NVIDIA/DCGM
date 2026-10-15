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

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <utility>

/**
 * Creates a temporary file and removes it when the test scope exits.
 */
class TempFile
{
public:
    /**
     * Creates a temporary file at an exact path.
     *
     * @param path Filesystem path to create.
     * @param contents Contents to write to the file.
     * @param permissions Optional replacement permissions for the file.
     */
    [[nodiscard]] static TempFile Fixed(std::filesystem::path path,
                                        std::string_view contents,
                                        std::filesystem::perms permissions = std::filesystem::perms::unknown)
    {
        return TempFile(std::move(path), contents, permissions);
    }
    /**
     * Creates a temporary file at an exact path.
     *
     * @param path Filesystem path to create.
     * @param contents Contents to write to the file.
     * @param permissions Optional replacement permissions for the file.
     */
    [[nodiscard]] static std::unique_ptr<TempFile> MakeUniqueFixed(std::filesystem::path path,
                                                                   std::string_view contents,
                                                                   std::filesystem::perms permissions
                                                                   = std::filesystem::perms::unknown)
    {
        return std::unique_ptr<TempFile>(new TempFile(std::move(path), contents, permissions));
    }

    /**
     * Creates a temporary file at a random path under the system temporary directory.
     *
     * @param contents Contents to write to the file.
     * @param prefix Filename prefix to use before the random suffix.
     * @param permissions Optional replacement permissions for the file.
     */
    [[nodiscard]] static TempFile Random(std::string_view contents,
                                         std::string_view prefix            = "dcgm-test-file-",
                                         std::filesystem::perms permissions = std::filesystem::perms::unknown)
    {
        return TempFile(CreateRandomPath(prefix), contents, permissions);
    }

    /**
     * Creates a temporary file at a random path under the system temporary directory.
     *
     * @param contents Contents to write to the file.
     * @param prefix Filename prefix to use before the random suffix.
     * @param permissions Optional replacement permissions for the file.
     */
    [[nodiscard]] static std::unique_ptr<TempFile> MakeUniqueRandom(std::string_view contents,
                                                                    std::string_view prefix = "dcgm-test-file-",
                                                                    std::filesystem::perms permissions
                                                                    = std::filesystem::perms::unknown)
    {
        return std::unique_ptr<TempFile>(new TempFile(CreateRandomPath(prefix), contents, permissions));
    }

    TempFile(TempFile const &)            = delete;
    TempFile(TempFile &&)                 = delete;
    TempFile &operator=(TempFile const &) = delete;
    TempFile &operator=(TempFile &&)      = delete;

    ~TempFile() noexcept
    {
        Cleanup();
    }

    [[nodiscard]] std::filesystem::path const &Path() const
    {
        return m_path;
    }

    /**
     * Sets file permissions.
     *
     * @param permissions Replacement permissions for the temporary file.
     */
    void SetPermissions(std::filesystem::perms permissions) const
    {
        std::error_code error;
        std::filesystem::permissions(m_path, permissions, std::filesystem::perm_options::replace, error);
        REQUIRE_FALSE(error);
    }

    /**
     * Adds file permissions.
     *
     * @param permissions Permissions to add to the temporary file.
     */
    void AddPermissions(std::filesystem::perms permissions) const
    {
        std::error_code error;
        std::filesystem::permissions(m_path, permissions, std::filesystem::perm_options::add, error);
        REQUIRE_FALSE(error);
    }

    /**
     * Adds executable permissions to the temporary file.
     *
     * @param permissions Executable permissions to add.
     */
    void MakeExecutable(std::filesystem::perms permissions = std::filesystem::perms::owner_exec) const
    {
        AddPermissions(permissions);
    }

private:
    explicit TempFile(std::filesystem::path path,
                      std::string_view contents,
                      std::filesystem::perms permissions = std::filesystem::perms::unknown)
        : m_path(std::move(path))
    {
        auto cleanup = DcgmNs::Defer([this] { Cleanup(); });

        WriteContents(contents);
        if (permissions != std::filesystem::perms::unknown)
        {
            SetPermissions(permissions);
        }
        cleanup.Disarm();
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
        if (closeResult != 0)
        {
            RemoveNoThrow(pathString);
        }
        REQUIRE(closeResult == 0);

        return pathString;
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

    static void RemoveNoThrow(std::filesystem::path const &path) noexcept
    {
        if (path.empty())
        {
            return;
        }

        std::error_code error;
        std::filesystem::remove(path, error);
    }

    void Cleanup() noexcept
    {
        RemoveNoThrow(m_path);
    }

    std::filesystem::path m_path;
};
