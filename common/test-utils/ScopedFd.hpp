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

#include <unistd.h>
#include <utility>

/**
 * Owns a POSIX file descriptor and closes it when the test scope exits.
 */
class ScopedFd
{
public:
    ScopedFd() noexcept = default;

    explicit ScopedFd(int fd)
        : m_fd(fd)
    {
        REQUIRE(m_fd >= 0);
    }

    ScopedFd(ScopedFd const &)            = delete;
    ScopedFd &operator=(ScopedFd const &) = delete;

    ScopedFd(ScopedFd &&other) noexcept
        : m_fd(other.Release())
    {}

    ScopedFd &operator=(ScopedFd &&other) noexcept
    {
        if (this != &other)
        {
            Reset(other.Release());
        }
        return *this;
    }

    ~ScopedFd() noexcept
    {
        CloseNoThrow();
    }

    [[nodiscard]] int Get() const noexcept
    {
        return m_fd;
    }

    [[nodiscard]] bool IsValid() const noexcept
    {
        return m_fd >= 0;
    }

    [[nodiscard]] int Release() noexcept
    {
        return std::exchange(m_fd, -1);
    }

    void Reset(int fd = -1) noexcept
    {
        if (m_fd == fd)
        {
            return;
        }

        CloseNoThrow();
        m_fd = fd;
    }

    int Close() noexcept
    {
        if (!IsValid())
        {
            return 0;
        }

        return close(std::exchange(m_fd, -1));
    }

private:
    void CloseNoThrow() noexcept
    {
        if (IsValid())
        {
            static_cast<void>(close(std::exchange(m_fd, -1)));
        }
    }

    int m_fd = -1;
};
