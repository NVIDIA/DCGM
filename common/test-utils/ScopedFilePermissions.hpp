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
#include <utility>

/**
 * Applies filesystem permissions and restores the original permissions when a test scope exits.
 */
class ScopedFilePermissions
{
public:
    explicit ScopedFilePermissions(std::filesystem::path path,
                                   std::filesystem::perms permissions,
                                   std::filesystem::perm_options options = std::filesystem::perm_options::replace)
        : m_path(std::move(path))
    {
        std::error_code error;
        auto const status = std::filesystem::status(m_path, error);
        REQUIRE_FALSE(error);
        m_originalPermissions = status.permissions();

        std::filesystem::permissions(m_path, permissions, options, error);
        REQUIRE_FALSE(error);
    }

    ScopedFilePermissions(ScopedFilePermissions const &)            = delete;
    ScopedFilePermissions(ScopedFilePermissions &&)                 = delete;
    ScopedFilePermissions &operator=(ScopedFilePermissions const &) = delete;
    ScopedFilePermissions &operator=(ScopedFilePermissions &&)      = delete;

    ~ScopedFilePermissions() noexcept
    {
        std::error_code error;
        std::filesystem::permissions(m_path, m_originalPermissions, std::filesystem::perm_options::replace, error);
        CHECK_FALSE(error);
    }

private:
    std::filesystem::path m_path;
    std::filesystem::perms m_originalPermissions = std::filesystem::perms::unknown;
};
