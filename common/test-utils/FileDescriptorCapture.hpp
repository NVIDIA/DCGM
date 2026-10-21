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

#include <array>
#include <cstdio>
#include <string>
#include <unistd.h>

/**
 * Captures writes sent to a C file descriptor and restores the original descriptor on destruction.
 */
class FileDescriptorCapture
{
public:
    /**
     * Redirects a file descriptor into an internal temporary file.
     *
     * @param fd File descriptor to capture.
     * @param stream C stream associated with @p fd, used for flushing buffered writes.
     */
    FileDescriptorCapture(int fd, FILE *stream)
        : m_fd(fd)
        , m_stream(stream)
    {
        m_file = tmpfile();
        REQUIRE(m_file != nullptr);

        auto cleanup = DcgmNs::Defer([this] {
            CloseSavedFdNoThrow();
            CloseFileNoThrow();
        });

        m_savedFd = dup(m_fd);
        REQUIRE(m_savedFd != -1);

        Flush();
        REQUIRE(dup2(fileno(m_file), m_fd) != -1);
        cleanup.Disarm();
    }

    FileDescriptorCapture(FileDescriptorCapture const &)            = delete;
    FileDescriptorCapture(FileDescriptorCapture &&)                 = delete;
    FileDescriptorCapture &operator=(FileDescriptorCapture const &) = delete;
    FileDescriptorCapture &operator=(FileDescriptorCapture &&)      = delete;

    /**
     * Restores the original file descriptor and closes the temporary capture file.
     */
    ~FileDescriptorCapture()
    {
        Restore();
        CHECK(fclose(m_file) == 0);
    }

    /**
     * Restores the original descriptor and returns captured output.
     *
     * @return Captured output.
     */
    [[nodiscard]] std::string GetOutput()
    {
        Restore();
        REQUIRE(fseek(m_file, 0, SEEK_SET) == 0);

        std::string output;
        std::array<char, 256> buffer {};
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), m_file) != nullptr)
        {
            output += buffer.data();
        }
        return output;
    }

private:
    void Flush()
    {
        if (m_stream != nullptr)
        {
            fflush(m_stream);
        }
    }

    void Restore()
    {
        if (m_savedFd != -1)
        {
            Flush();
            CHECK(dup2(m_savedFd, m_fd) != -1);
            CHECK(close(m_savedFd) != -1);
            m_savedFd = -1;
        }
    }

    void CloseSavedFdNoThrow() noexcept
    {
        if (m_savedFd != -1)
        {
            static_cast<void>(close(m_savedFd));
            m_savedFd = -1;
        }
    }

    void CloseFileNoThrow() noexcept
    {
        if (m_file != nullptr)
        {
            static_cast<void>(fclose(m_file));
            m_file = nullptr;
        }
    }

    int m_fd;
    FILE *m_stream;
    int m_savedFd = -1;
    FILE *m_file  = nullptr;
};
