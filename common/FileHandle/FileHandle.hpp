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

#include <cstddef>
#include <sys/types.h>

namespace DcgmNs::Utils
{

/**
 * @brief Represents a system file descriptor
 */
class FileHandle
{
public:
    /**
     * @brief Creates invalid (-1) file descriptor
     */
    FileHandle() noexcept;

    /**
     * @brief Acquires ownership of the \a fd file descriptor
     * @param[in] fd    A valid file descriptor. External caller must no call \c close() on this descriptor once
     * it's transferred to the \c FileHandle
     */
    explicit FileHandle(int fd) noexcept;

    virtual ~FileHandle() noexcept;

    FileHandle(FileHandle const &)            = delete;
    FileHandle &operator=(FileHandle const &) = delete;

    /**
     * Takes ownership of other's file descriptor and last errno. Leaves other with an invalid
     * descriptor and errno 0.
     *
     * @param[in,out] other Handle to move from.
     */
    FileHandle(FileHandle &&other) noexcept;

    /**
     * Closes any file descriptor owned by this handle, then takes ownership of other's file descriptor
     * and last errno. Leaves other with an invalid descriptor and errno 0.
     *
     * @param[in,out] other Handle to move from.
     * @return Reference to this handle.
     */
    FileHandle &operator=(FileHandle &&other) noexcept;

    explicit operator int() const;
    /**
     * @brief Returns underlying system file descriptor. Caller must not call \c close() on the descriptor returned
     * from this function.
     * @return \c int  File descriptor.
     */
    [[nodiscard]] int Get() const &;
    /**
     * @brief Returns the underlying system file descriptor and abandons ownership so that it's caller
     * responsibility to call \c close() on the result descriptor
     * @return \c int  File descriptor. Caller must call \c close() on the value later.
     */
    [[nodiscard]] int Release();

    /**
     * @brief Reads data from the file descriptor until it reaches the end of the file or the specified size is
     * read. This function will try its best effort to continue reading the data even if the read is interrupted by
     * a signal.
     * @param[in] buffer  The buffer to read the data into
     * @param[in] size    The number of bytes to read
     * @return The number of bytes read, -1 if an error occurred. 0 if the end of the file is reached. Details of
     * the error can be retrieved by GetErrno(). When 0 or -1 returned, the buffer may be partially filled.
     */
    [[nodiscard]] virtual ssize_t ReadExact(std::byte *buffer, size_t size);

    /**
     * @brief Writes data to the file descriptor
     * @param[in] buffer  The buffer to write the data from
     * @param[in] size    The number of bytes to write
     * @return The number of bytes written, -1 if an error occurred. Details of the error can be retrieved by
     * GetErrno().
     */
    [[nodiscard]] virtual ssize_t Write(void const *buffer, size_t size);

    /**
     * @brief Writes data to the file descriptor until the specified size has been written. Retries on partial
     * writes and EINTR. Body is async-signal-safe: only invokes \c write(2) and stores into a plain int member, so
     * it is safe to call from a post-fork, pre-exec context.
     * @param[in] buffer  The buffer to write the data from
     * @param[in] size    The number of bytes to write
     * @return \a size on success; -1 on error. On error, the buffer may be partially written. Details of the error
     * can be retrieved by GetErrno().
     */
    [[nodiscard]] virtual ssize_t WriteExact(void const *buffer, size_t size);

    /**
     * @brief Returns the errno of the last error that occurred on the file descriptor
     * @return The errno of the last error that occurred on the file descriptor
     */
    [[nodiscard]] virtual int GetErrno() const;

private:
    int m_fd;
    int m_errno;
};

} // namespace DcgmNs::Utils
