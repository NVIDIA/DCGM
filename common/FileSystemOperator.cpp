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

#include "FileSystemOperator.h"

#include <dirent.h>
#include <glob.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "DcgmLogging.h"

std::optional<std::string> FileSystemOperator::Read(std::string_view path)
{
    std::ifstream file { std::string(path) };
    if (!file.good())
    {
        log_debug("failed to open file [{}]", path);
        return std::nullopt;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::optional<std::vector<std::string>> FileSystemOperator::Glob(std::string_view pattern)
{
    glob_t globResult;
    memset(&globResult, 0, sizeof(globResult));

    int ret = glob(pattern.data(), GLOB_NOSORT, nullptr, &globResult);
    if (ret != 0)
    {
        globfree(&globResult);
        return std::nullopt;
    }

    std::vector<std::string> paths;
    for (size_t i = 0; i < globResult.gl_pathc; ++i)
    {
        paths.emplace_back(globResult.gl_pathv[i]);
    }

    globfree(&globResult);
    return paths;
}

bool FileSystemOperator::Unlink(std::string_view path)
{
    return unlink(std::string(path).c_str()) == 0;
}

ssize_t FileSystemOperator::ReadLink(std::string_view path, char *buf, size_t bufsize)
{
    return readlink(std::string(path).c_str(), buf, bufsize);
}

int FileSystemOperator::Stat(char const *path, struct stat *buf)
{
    return stat(path, buf);
}

int FileSystemOperator::Access(char const *path, int mode)
{
    return access(path, mode);
}

bool FileSystemOperator::IsDirectory(std::string const &path)
{
    struct stat st = {};
    if (Stat(path.c_str(), &st) != 0)
    {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

std::optional<std::vector<std::string>> FileSystemOperator::ListDirectoryEntries(std::string const &path)
{
    DIR *const dir = opendir(path.c_str());
    if (dir == nullptr)
    {
        return std::nullopt;
    }

    std::vector<std::string> names;
    errno = 0;
    while (dirent *const entry = readdir(dir))
    {
        names.emplace_back(entry->d_name);
    }

    int const readdirErrno = errno; // distinguish readdir(3) failure from EOF (see errno after NULL return)
    if (closedir(dir) != 0 || readdirErrno != 0)
    {
        return std::nullopt;
    }
    return names;
}

char *FileSystemOperator::GetCurrentWorkingDirectory(char *buf, size_t size)
{
    return getcwd(buf, size);
}

int FileSystemOperator::ChangeDirectory(char const *path)
{
    return chdir(path);
}
