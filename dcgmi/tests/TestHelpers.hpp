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

#include <iostream>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>

/**
 * Captures writes to a stream and restores the original stream buffer on destruction.
 */
class StreamCapture
{
public:
    /**
     * Redirects the stream into an internal buffer.
     *
     * @param stream Stream to capture until this object is destroyed.
     */
    explicit StreamCapture(std::ostream &stream)
        : m_stream(stream)
        , m_old(m_stream.rdbuf(m_capture.rdbuf()))
    {}

    StreamCapture(StreamCapture const &)            = delete;
    StreamCapture(StreamCapture &&)                 = delete;
    StreamCapture &operator=(StreamCapture const &) = delete;
    StreamCapture &operator=(StreamCapture &&)      = delete;

    /**
     * Restores the stream buffer that was active before construction.
     */
    ~StreamCapture()
    {
        m_stream.rdbuf(m_old);
    }

    /**
     * Returns the text captured from the stream.
     *
     * @return Captured stream contents.
     */
    [[nodiscard]] std::string str() const
    {
        return m_capture.str();
    }

private:
    std::ostream &m_stream;
    std::ostringstream m_capture;
    std::streambuf *m_old;
};

class CoutCapture : public StreamCapture
{
public:
    /**
     * Captures writes to std::cout.
     */
    CoutCapture()
        : StreamCapture(std::cout)
    {}
};

class CerrCapture : public StreamCapture
{
public:
    /**
     * Captures writes to std::cerr.
     */
    CerrCapture()
        : StreamCapture(std::cerr)
    {}
};
