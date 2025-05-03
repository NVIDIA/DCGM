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

#pragma once

#include <cerrno>
#include <cstdint>
#include <span>
#include <unistd.h>

#include <DcgmLogging.h>
#include <DcgmNvvsResponseWrapper.h>

class FdChannelClient
{
public:
    explicit FdChannelClient(int const channelFd);
    ~FdChannelClient() = default;

    [[nodiscard]] bool IsValid() const;

    [[nodiscard]] bool Write(std::span<char const> buf) const
    {
        if (!IsValid())
        {
            log_error("Process appears to have been launched incorrectly.");
            return false;
        }
        std::uint32_t const len = buf.size();
        if (auto err = WriteAll({ reinterpret_cast<char const *>(&len), sizeof(len) }); err)
        {
            log_error("failed to write frame size to fd {}: ({}) {}", m_channelFd, err.value(), err.message());
            return false;
        }
        if (auto err = WriteAll(buf); err)
        {
            log_error("failed to write frame to fd {}: ({}) {}", m_channelFd, err.value(), err.message());
            return false;
        }
        return true;
    }

private:
    /**
     * @brief Write a buffer to m_channelFd and returns an error_code if an error occurs.
     *
     * This function takes a span and returns a std::error_code containing errno
     * if an error occurs during processing.
     *
     * @param buf A span of char const representing the buffer to be written.
     * @return std::error_code An error code containing errno if an error occurs,
     *         or std::error_code() if the operation is successful.
     */
    std::error_code WriteAll(std::span<char const> buf) const;

    int m_channelFd;
};