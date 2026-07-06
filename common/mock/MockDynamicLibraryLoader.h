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

#include <string>
#include <unordered_map>

/**
 * Test double with the same interface as DynamicLibraryLoader.
 * Records calls and returns configured handles per path.
 */
class MockDynamicLibraryLoader
{
public:
    /**
     * Mock library open (DynamicLibraryLoader::Open).
     *
     * @param[in] path Library path; empty string is used when @p path is null.
     * @param[in] flags Unused; present for interface compatibility.
     * @return Handle previously set with MockOpenReturns() for this path, or nullptr if none.
     */
    void *Open(char const *path, int /* flags */)
    {
        auto it = m_handleByPath.find(path ? path : "");
        if (it != m_handleByPath.end())
        {
            return it->second;
        }
        return nullptr;
    }

    /**
     * Mock last error string (DynamicLibraryLoader::Error).
     *
     * @return Pointer to the message from SetNextError(), or nullptr if no error was set.
     */
    char const *Error()
    {
        return m_lastError.empty() ? nullptr : m_lastError.c_str();
    }

    /**
     * Mock library close (DynamicLibraryLoader::Close).
     *
     * @param[in] handle Unused; present for interface compatibility.
     * @return Always 0 (success).
     */
    int Close(void * /* handle */)
    {
        return 0;
    }

    /**
     * Configure Open() to return @p handle when called with @p path.
     *
     * @param[in] path Library path key (must match the string passed to Open(), including empty for null path).
     * @param[in] handle Opaque pointer to return from Open().
     */
    void MockOpenReturns(std::string const &path, void *handle)
    {
        m_handleByPath[path] = handle;
    }

    /**
     * Set the message Error() returns until replaced or cleared by another call.
     *
     * @param[in] msg Error text; moved from the argument.
     */
    void SetNextError(std::string msg)
    {
        m_lastError = std::move(msg);
    }

private:
    std::unordered_map<std::string, void *> m_handleByPath;
    std::string m_lastError;
};
