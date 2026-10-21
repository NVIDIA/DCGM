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

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <system_error>

/**
 * Provides test helpers for common file permission changes.
 */
class FilePermissions
{
public:
    /**
     * Replaces permissions for a filesystem path.
     *
     * @param path Filesystem path to update.
     * @param permissions Replacement permissions.
     */
    static void Set(std::filesystem::path const &path, std::filesystem::perms permissions)
    {
        std::error_code error;
        std::filesystem::permissions(path, permissions, std::filesystem::perm_options::replace, error);
        REQUIRE_FALSE(error);
    }

    /**
     * Adds permissions to a filesystem path.
     *
     * @param path Filesystem path to update.
     * @param permissions Permissions to add.
     */
    static void Add(std::filesystem::path const &path, std::filesystem::perms permissions)
    {
        std::error_code error;
        std::filesystem::permissions(path, permissions, std::filesystem::perm_options::add, error);
        REQUIRE_FALSE(error);
    }

    /**
     * Adds executable permissions to a filesystem path.
     *
     * @param path Filesystem path to update.
     * @param permissions Executable permissions to add.
     */
    static void MakeExecutable(std::filesystem::path const &path,
                               std::filesystem::perms permissions = std::filesystem::perms::owner_exec)
    {
        Add(path, permissions);
    }
};
