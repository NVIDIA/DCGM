/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <optional>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

class FileSystemOperator
{
public:
    virtual ~FileSystemOperator() = default;

    /**
     * @brief Reads the entire contents of a regular file as text.
     *
     * Opens @p path for reading; on failure (e.g. missing file or permission denied) returns @c std::nullopt.
     *
     * @param[in] path  Filesystem path to the file.
     * @return Full file contents on success; @c std::nullopt if the file cannot be read.
     */
    virtual std::optional<std::string> Read(std::string_view path);

    /**
     * @brief Expands a pathname pattern.
     *
     * Same underlying behavior as glob(3) with @c GLOB_NOSORT (paths are not sorted). Pattern syntax is
     * implementation-defined per glob(3).
     *
     * @param[in] pattern  Glob pattern (e.g. shell-style wildcards).
     * @return Matching paths on success; @c std::nullopt if the pattern could not be expanded.
     */
    virtual std::optional<std::vector<std::string>> Glob(std::string_view pattern);

    /**
     * @brief Removes a file.
     *
     * Same semantics as unlink(2): removes the directory entry named by @p path.
     *
     * @param[in] path  Path to the file to remove.
     * @return @c true if unlink succeeded; @c false otherwise.
     */
    virtual bool Unlink(std::string_view path);

    /**
     * @brief Reads the target of a symbolic link.
     *
     * Same semantics as readlink(2): on success, writes up to @p bufsize bytes into @p buf (not NUL-terminated)
     * and returns the number of bytes written; on failure returns -1 and sets errno.
     *
     * @param[in] path     Path to the symlink.
     * @param[out] buf     Buffer to receive the link target.
     * @param[in] bufsize  Size of @p buf in bytes.
     * @return Number of bytes placed in @p buf on success; -1 on error.
     */
    virtual ssize_t ReadLink(std::string_view path, char *buf, size_t bufsize);

    /**
     * @brief Obtains file status.
     *
     * Same semantics as stat(2).
     *
     * @param[in] path  Filesystem path.
     * @param[out] buf  Filled with status information on success.
     * @return 0 on success; -1 on error with errno set.
     */
    virtual int Stat(char const *path, struct stat *buf);

    /**
     * @brief Checks accessibility of a file.
     *
     * Same semantics as access(2).
     *
     * @param[in] path  Path to check.
     * @param[in] mode  Bit mask of R_OK, W_OK, X_OK, or F_OK.
     * @return 0 if the requested access is allowed; -1 otherwise with errno set.
     */
    virtual int Access(char const *path, int mode);

    /**
     * @brief Returns whether @p path refers to an existing directory.
     *
     * @param[in] path  Filesystem path.
     * @return @c true if @p path exists and is a directory; @c false otherwise (including if status cannot be read).
     */
    virtual bool IsDirectory(std::string const &path);

    /**
     * @brief Lists names in a directory.
     *
     * Uses opendir(3)/readdir(3); includes @c "." and @c "..". Order is unspecified.
     *
     * @param[in] path  Directory path.
     * @return Entry names on success; @c std::nullopt if the directory cannot be opened or fully read.
     */
    virtual std::optional<std::vector<std::string>> ListDirectoryEntries(std::string const &path);

    /**
     * @brief Returns the current working directory.
     *
     * Same semantics as getcwd(3).
     *
     * @param[out] buf   Buffer for the NUL-terminated path.
     * @param[in] size   Size of @p buf.
     * @return @p buf on success; @c nullptr on failure with errno set.
     */
    virtual char *GetCurrentWorkingDirectory(char *buf, size_t size);

    /**
     * @brief Changes the current working directory.
     *
     * Same semantics as chdir(2).
     *
     * @param[in] path  Target directory.
     * @return 0 on success; -1 on failure with errno set.
     */
    virtual int ChangeDirectory(char const *path);
};
