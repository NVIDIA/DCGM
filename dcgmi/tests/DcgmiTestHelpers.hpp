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

#include <catch2/catch_all.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>

class StdoutRedirect
{
public:
    StdoutRedirect()
        : m_oldBuffer(std::cout.rdbuf(m_buffer.rdbuf()))
    {}

    ~StdoutRedirect()
    {
        std::cout.rdbuf(m_oldBuffer);
    }

    [[nodiscard]] std::string GetOutput() const
    {
        return m_buffer.str();
    }

private:
    std::stringstream m_buffer;
    std::streambuf *m_oldBuffer;
};

class StderrRedirect
{
public:
    StderrRedirect()
        : m_oldBuffer(std::cerr.rdbuf(m_buffer.rdbuf()))
    {}

    ~StderrRedirect()
    {
        std::cerr.rdbuf(m_oldBuffer);
    }

    [[nodiscard]] std::string GetOutput() const
    {
        return m_buffer.str();
    }

private:
    std::stringstream m_buffer;
    std::streambuf *m_oldBuffer;
};

class TempFile
{
public:
    explicit TempFile(std::string_view contents)
    {
        static unsigned int counter = 0;

        std::stringstream path;
        path << "/tmp/dcgmi_diagtests_" << getpid() << "_" << counter++;
        m_path = path.str();

        std::ofstream file(m_path);
        REQUIRE(file.is_open());
        file << contents;
    }

    ~TempFile()
    {
        std::remove(m_path.c_str());
    }

    [[nodiscard]] std::string const &Path() const
    {
        return m_path;
    }

private:
    std::string m_path;
};
