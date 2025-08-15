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

/**
 * Mock implementation of FileSystemOperator for testing
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
        auto it = m_mockContent.find(std::string(path));
        if (it != m_mockContent.end())
        {
            return it->second;
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
     * Set mock content for a file path
     *
     * @param path The file path
     * @param content The content to return when reading the file
     */
    void MockFileContent(std::string const &path, std::string const &content)
    {
        m_mockContent[path] = content;
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

private:
    std::unordered_map<std::string, std::string> m_mockContent;
    std::unordered_map<std::string, std::vector<std::string>> m_mockGlobs;
};