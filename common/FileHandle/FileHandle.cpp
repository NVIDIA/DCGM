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

#include "FileHandle.hpp"

#include <cerrno>
#include <unistd.h>
#include <utility>

namespace DcgmNs::Utils
{

FileHandle::FileHandle() noexcept
    : FileHandle(-1)
{}

FileHandle::FileHandle(int fd) noexcept
    : m_fd(fd)
    , m_errno(0)
{}

FileHandle::~FileHandle() noexcept
{
    if (m_fd != -1)
    {
        close(m_fd);
        m_fd = -1;
    }
}

FileHandle::FileHandle(FileHandle &&other) noexcept
{
    m_fd          = other.m_fd;
    m_errno       = other.m_errno;
    other.m_fd    = -1;
    other.m_errno = 0;
}

FileHandle &FileHandle::operator=(FileHandle &&other) noexcept
{
    if (this != &other)
    {
        if (m_fd != -1)
        {
            close(m_fd);
        }
        m_fd          = other.m_fd;
        m_errno       = other.m_errno;
        other.m_fd    = -1;
        other.m_errno = 0;
    }
    return *this;
}

FileHandle::operator int() const
{
    return Get();
}

int FileHandle::Get() const &
{
    return m_fd;
}

int FileHandle::Release()
{
    int const result = m_fd;
    m_fd             = -1;
    return result;
}

ssize_t FileHandle::ReadExact(std::byte *buffer, size_t size)
{
    if (m_fd == -1)
    {
        m_errno = EBADF;
        return -1;
    }
    size_t totalBytesRead = 0;
    while (totalBytesRead < size)
    {
        ssize_t bytesRead = read(m_fd, buffer + totalBytesRead, size - totalBytesRead);
        if (bytesRead == 0)
        {
            // EOF reached
            return 0;
        }
        if (bytesRead < 0)
        {
            m_errno = errno;
            if (GetErrno() == EINTR)
            {
                continue;
            }
            return -1;
        }
        totalBytesRead += bytesRead;
    }
    return totalBytesRead;
}

int FileHandle::GetErrno() const
{
    return m_errno;
}

ssize_t FileHandle::Write(void const *buffer, size_t size)
{
    if (m_fd == -1)
    {
        m_errno = EBADF;
        return -1;
    }
    auto ret = write(m_fd, buffer, size);
    if (ret < 0)
    {
        m_errno = errno;
    }
    return ret;
}

ssize_t FileHandle::WriteExact(void const *buffer, size_t size)
{
    if (m_fd == -1)
    {
        m_errno = EBADF;
        return -1;
    }
    auto const *bytes        = static_cast<std::byte const *>(buffer);
    size_t totalBytesWritten = 0;
    while (totalBytesWritten < size)
    {
        ssize_t const bytesWritten = write(m_fd, bytes + totalBytesWritten, size - totalBytesWritten);
        if (bytesWritten < 0)
        {
            m_errno = errno;
            if (m_errno == EINTR)
            {
                continue;
            }
            return -1;
        }
        totalBytesWritten += bytesWritten;
    }
    return totalBytesWritten;
}

} // namespace DcgmNs::Utils
