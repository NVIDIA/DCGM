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

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

/**
 * Creates a temporary directory and removes it recursively when the test scope exits.
 */
class TempDir
{
public:
    /**
     * Creates a temporary directory at an exact path.
     *
     * @param path Filesystem path to create.
     * @param permissions Optional replacement permissions for the directory.
     */
    [[nodiscard]] static TempDir Fixed(std::filesystem::path path,
                                       std::filesystem::perms permissions = std::filesystem::perms::unknown)
    {
        return TempDir(std::move(path), permissions);
    }

    /**
     * Creates a temporary directory at a random path under the system temporary directory.
     *
     * @param prefix Directory name prefix to use before the random suffix.
     * @param permissions Optional replacement permissions for the directory.
     */
    [[nodiscard]] static TempDir Random(std::string_view prefix            = "dcgm-test-dir-",
                                        std::filesystem::perms permissions = std::filesystem::perms::unknown)
    {
        return TempDir(CreateRandomPath(prefix), permissions);
    }

    TempDir(TempDir const &)            = delete;
    TempDir(TempDir &&)                 = delete;
    TempDir &operator=(TempDir const &) = delete;
    TempDir &operator=(TempDir &&)      = delete;

    ~TempDir() noexcept
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    [[nodiscard]] std::filesystem::path const &Path() const
    {
        return m_path;
    }

    /**
     * Sets directory permissions.
     *
     * @param permissions Replacement permissions for the temporary directory.
     */
    void SetPermissions(std::filesystem::perms permissions) const
    {
        std::error_code error;
        std::filesystem::permissions(m_path, permissions, std::filesystem::perm_options::replace, error);
        REQUIRE_FALSE(error);
    }

    /**
     * Adds directory permissions.
     *
     * @param permissions Permissions to add to the temporary directory.
     */
    void AddPermissions(std::filesystem::perms permissions) const
    {
        std::error_code error;
        std::filesystem::permissions(m_path, permissions, std::filesystem::perm_options::add, error);
        REQUIRE_FALSE(error);
    }

private:
    explicit TempDir(std::filesystem::path path, std::filesystem::perms permissions = std::filesystem::perms::unknown)
        : m_path(std::move(path))
    {
        auto cleanup = DcgmNs::Defer([this] { Cleanup(); });

        std::error_code error;
        std::filesystem::create_directories(m_path, error);
        REQUIRE_FALSE(error);
        REQUIRE(std::filesystem::is_directory(m_path, error));
        REQUIRE_FALSE(error);
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

        REQUIRE(mkdtemp(pathString.data()) != nullptr);

        return pathString;
    }

    void Cleanup() noexcept
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    std::filesystem::path m_path;
};
