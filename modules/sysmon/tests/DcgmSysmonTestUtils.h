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

#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

inline int mkdir_wrapper(const std::string &dirpath)
{
    int ret = mkdir(dirpath.c_str(), S_IRWXU);
    if (ret != 0)
    {
        if (errno == EEXIST)
        {
            ret = 0;
        }
    }

    return ret;
}

inline int remove_wrapper(const char *file)
{
    int ret = remove(file);
    if (ret != 0)
    {
        if (errno == ENOENT)
        {
            ret = 0;
        }
    }

    return ret;
}

inline int writeValueToFile(const std::string &filename, const std::string &value)
{
    int ret = remove_wrapper(filename.c_str());
    std::ofstream out(filename);
    out << value;
    out.close();
    return ret;
}

#define WRITE_VALUE_TO_FILE_CHECKED(path, value) \
    do                                           \
    {                                            \
        if (writeValueToFile(path, value) != 0)  \
        {                                        \
            return;                              \
        }                                        \
    } while (0)

#define WRITE_VALUE_TO_FILE_CHECKED_RC(path, value) \
    do                                              \
    {                                               \
        if (writeValueToFile(path, value) != 0)     \
        {                                           \
            return 1;                               \
        }                                           \
    } while (0)

#define REMOVE_CHECKED(file)           \
    do                                 \
    {                                  \
        if (remove_wrapper(file) != 0) \
        {                              \
            return;                    \
        }                              \
    } while (0)

#define REMOVE_CHECKED_RC(file)        \
    do                                 \
    {                                  \
        if (remove_wrapper(file) != 0) \
        {                              \
            return 1;                  \
        }                              \
    } while (0)

#define MKDIR_CHECKED(dirPath)           \
    do                                   \
    {                                    \
        if (mkdir_wrapper(dirPath) != 0) \
        {                                \
            if (errno != EEXIST)         \
            {                            \
                return;                  \
            }                            \
        }                                \
    } while (0)

#define MKDIR_CHECKED_RC(dirPath)        \
    do                                   \
    {                                    \
        if (mkdir_wrapper(dirPath) != 0) \
        {                                \
            if (errno != EEXIST)         \
            {                            \
                return 1;                \
            }                            \
        }                                \
    } while (0)
