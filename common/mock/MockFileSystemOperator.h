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

#include "FileSystemOperator.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <cerrno>
#include <cstring>

/**
 * Mock implementation of FileSystemOperator for testing.
 * If a path or key has not been configured via Mock* helpers, operations fail without
 * delegating to the real filesystem (Read/Glob return nullopt, ReadLink/Stat/Access return -1,
 * IsDirectory return false, Unlink returns true and drops mocked state for that path on success, ListDirectoryEntries
 * returns nullopt, GetCurrentWorkingDirectory returns nullptr, ChangeDirectory returns -1 unless mocked (0 updates
 * the mock current working directory).
 */
class MockFileSystemOperator : public FileSystemOperator
{
public:
    /**
     * Mock reading a file
     *
     * @param path Path to the file
     * @return std::optional<std::string> The mocked content if available
     */
    std::optional<std::string> Read(std::string_view path) override
    {
        auto it = m_pathState.find(std::string(path));
        if (it != m_pathState.end() && it->second.fileContent.has_value())
        {
            return it->second.fileContent;
        }
        return std::nullopt;
    }
    /**
     * Mock glob pattern matching
     *
     * @param pattern The glob pattern to match
     * @return std::optional<std::vector<std::string>> The mocked matching paths
     */
    std::optional<std::vector<std::string>> Glob(std::string_view pattern) override
    {
        auto it = m_mockGlobs.find(std::string(pattern));
        if (it != m_mockGlobs.end())
        {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * Mock unlink
     *
     * On success, removes any mocked state for @p path from the path map (as if the node were deleted).
     *
     * @param path Path to the file
     * @return true for success, false for failure
     */
    bool Unlink(std::string_view path) override
    {
        std::string const p(path);
        bool success  = true;
        auto const it = m_pathState.find(p);
        if (it != m_pathState.end() && it->second.unlinkSuccess.has_value())
        {
            success = *it->second.unlinkSuccess;
        }
        if (success)
        {
            m_pathState.erase(p);
        }
        return success;
    }

    /**
     * Set mock content for a file path
     *
     * @param path The file path
     * @param content The content to return when reading the file
     */
    void MockFileContent(std::string const &path, std::string const &content)
    {
        m_pathState[path].fileContent = content;
    }

    /**
     * Set mock glob results for a pattern
     *
     * @param pattern The glob pattern
     * @param paths The paths to return when matching the pattern
     */
    void MockGlob(std::string const &pattern, std::vector<std::string> const &paths)
    {
        m_mockGlobs[pattern] = paths;
    }

    /**
     * Set mock unlink result for a path
     *
     * @param path The path
     * @param result The result to return when unlinking the path
     */
    void MockUnlink(std::string const &path, bool result)
    {
        m_pathState[path].unlinkSuccess = result;
    }

    ssize_t ReadLink(std::string_view path, char *buf, size_t bufsize) override
    {
        auto it = m_pathState.find(std::string(path));
        if (it != m_pathState.end() && it->second.readLinkTarget.has_value())
        {
            std::string const &s = *it->second.readLinkTarget;
            size_t const n       = std::min(s.size(), bufsize);
            std::memcpy(buf, s.data(), n);
            return static_cast<ssize_t>(n);
        }
        return -1;
    }

    int Stat(char const *path, struct stat *buf) override
    {
        auto it = m_pathState.find(std::string(path));
        if (it != m_pathState.end() && it->second.statResult.has_value())
        {
            *buf = *it->second.statResult;
            return 0;
        }
        return -1;
    }

    int Access(char const *path, int mode) override
    {
        auto it = m_pathState.find(std::string(path));
        if (it != m_pathState.end() && it->second.accessResult.has_value())
        {
            return *it->second.accessResult;
        }
        (void)mode;
        return -1;
    }

    bool IsDirectory(std::string const &path) override
    {
        auto it = m_pathState.find(path);
        if (it != m_pathState.end() && it->second.isDirectory.has_value())
        {
            return *it->second.isDirectory;
        }
        return false;
    }

    std::optional<std::vector<std::string>> ListDirectoryEntries(std::string const &path) override
    {
        auto it = m_pathState.find(path);
        if (it != m_pathState.end() && it->second.listDirEntries.has_value())
        {
            return it->second.listDirEntries;
        }
        return std::nullopt;
    }

    char *GetCurrentWorkingDirectory(char *buf, size_t size) override
    {
        if (!m_mockCwd.has_value())
        {
            return nullptr;
        }
        std::string const &cwd = *m_mockCwd;
        if (cwd.size() + 1 > size)
        {
            errno = ERANGE;
            return nullptr;
        }
        std::memcpy(buf, cwd.data(), cwd.size());
        buf[cwd.size()] = '\0';
        return buf;
    }

    int ChangeDirectory(char const *path) override
    {
        std::string const p(path);
        auto const it = m_pathState.find(p);
        if (it == m_pathState.end() || !it->second.changeDirectoryResult.has_value())
        {
            return -1;
        }
        int const rc = *it->second.changeDirectoryResult;
        if (rc == 0)
        {
            m_mockCwd = p;
        }
        return rc;
    }

    void MockGetCurrentWorkingDirectory(std::string const &path)
    {
        m_mockCwd = path;
    }

    void MockChangeDirectory(std::string const &path, int result)
    {
        m_pathState[path].changeDirectoryResult = result;
    }

    void MockReadLink(std::string const &path, std::string const &target)
    {
        m_pathState[path].readLinkTarget = target;
    }

    void MockStat(std::string const &path, struct stat const &st)
    {
        m_pathState[path].statResult = st;
    }

    void MockAccess(std::string const &path, int result)
    {
        m_pathState[path].accessResult = result;
    }

    void MockIsDirectory(std::string const &path, bool isDir)
    {
        m_pathState[path].isDirectory = isDir;
    }

    void MockListDirectory(std::string const &path, std::vector<std::string> const &entries)
    {
        m_pathState[path].listDirEntries = entries;
    }

private:
    /** Mocked attributes for a single filesystem path; unset optionals are not mocked for that operation. */
    struct MockPathState
    {
        std::optional<std::string> fileContent;
        std::optional<bool> unlinkSuccess;
        std::optional<std::string> readLinkTarget;
        std::optional<struct stat> statResult;
        std::optional<int> accessResult;
        std::optional<bool> isDirectory;
        std::optional<std::vector<std::string>> listDirEntries;
        std::optional<int> changeDirectoryResult;
    };

    std::unordered_map<std::string, MockPathState> m_pathState;
    /** Glob is keyed by pattern string, not a single path. */
    std::unordered_map<std::string, std::vector<std::string>> m_mockGlobs;
    std::optional<std::string> m_mockCwd;
};
