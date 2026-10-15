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
 * Changes the process working directory and restores the original directory when a test scope exits.
 */
class ScopedWorkingDirectory
{
public:
    explicit ScopedWorkingDirectory(std::filesystem::path const &path)
    {
        std::error_code error;
        m_originalPath = std::filesystem::current_path(error);
        REQUIRE_FALSE(error);

        std::filesystem::current_path(path, error);
        REQUIRE_FALSE(error);
    }

    ScopedWorkingDirectory(ScopedWorkingDirectory const &)            = delete;
    ScopedWorkingDirectory(ScopedWorkingDirectory &&)                 = delete;
    ScopedWorkingDirectory &operator=(ScopedWorkingDirectory const &) = delete;
    ScopedWorkingDirectory &operator=(ScopedWorkingDirectory &&)      = delete;

    ~ScopedWorkingDirectory() noexcept
    {
        std::error_code error;
        std::filesystem::current_path(m_originalPath, error);
        CHECK_FALSE(error);
    }

private:
    std::filesystem::path m_originalPath;
};
