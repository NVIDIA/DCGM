/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "FileHandleMock.h"

ssize_t FileHandleMock::ReadExact(std::byte *buffer, size_t size)
{
    if (m_buffer.size() < size)
    {
        return 0;
    }
    std::copy(m_buffer.begin(), m_buffer.begin() + size, buffer);
    m_buffer.erase(m_buffer.begin(), m_buffer.begin() + size);
    m_errno = 0;
    return size;
}

void FileHandleMock::AppendToBuffer(std::span<std::byte> data)
{
    m_buffer.insert(m_buffer.end(), data.begin(), data.end());
}

ssize_t FileHandleMock::Write(void const *buffer, size_t size)
{
    if (!m_sizeToTriggerEINTR.empty() && *m_sizeToTriggerEINTR.begin() <= size)
    {
        m_sizeToTriggerEINTR.erase(m_sizeToTriggerEINTR.begin());
        m_errno = EINTR;
        return -1;
    }
    m_writeBuffer.insert(m_writeBuffer.end(),
                         reinterpret_cast<std::byte const *>(buffer),
                         reinterpret_cast<std::byte const *>(buffer) + size);
    m_errno = 0;
    return size;
}

int FileHandleMock::GetErrno() const
{
    return m_errno;
}

void FileHandleMock::AddSizeToTriggerEINTR(size_t size)
{
    m_sizeToTriggerEINTR.insert(size);
}

std::vector<std::byte> const &FileHandleMock::GetWriteBuffer() const
{
    return m_writeBuffer;
}
