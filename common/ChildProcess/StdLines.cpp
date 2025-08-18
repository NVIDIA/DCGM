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

#include "StdLines.hpp"
#include "ChildProcess.hpp"
#include <unistd.h>

namespace DcgmNs::Common::Subprocess
{

std::string const &StdLines::StdLinesIterator::dereference() const
{
    return m_currentValue;
}

std::string const &StdLines::StdLinesIterator::dereference()
{
    return m_currentValue;
}

bool StdLines::StdLinesIterator::equal(const StdLines::StdLinesIterator &other) const
{
    return (m_stdlines == nullptr) && (other.m_stdlines == nullptr);
}

bool StdLines::StdLinesIterator::equal(StdLines::StdOutSentinel /*sentinel*/) const
{
    return m_stdlines == nullptr;
}

void StdLines::StdLinesIterator::increment()
{
    auto lineOpt = m_stdlines->Read();
    if (!lineOpt)
    {
        m_stdlines = nullptr;
        return;
    }
    m_currentValue = *lineOpt;
}

StdLines::StdLinesIterator::StdLinesIterator(StdLines::StdOutSentinel /*sentinel*/)
{ // DO NOTHING
}

StdLines::StdLinesIterator::StdLinesIterator(StdLines &stdlines)
    : m_stdlines(&stdlines)
{
    increment();
}

StdLines::StdLinesIterator StdLines::begin()
{
    return StdLinesIterator { *this };
}

StdLines::StdLinesIterator StdLines::end()
{
    return StdLines::StdLinesIterator(StdOutSentinel {});
}

bool StdLines::IsEmpty()
{
    std::lock_guard<std::mutex> lg(m_lock);
    return m_container.empty();
}

void StdLines::Write(std::string const &str)
{
    std::unique_lock<std::mutex> lock(m_lock);
    m_container.push(str);
    lock.unlock();
    m_notEmpty.notify_one();
}

void StdLines::Close()
{
    m_closed.store(true, std::memory_order_release);
    m_notEmpty.notify_one();
}

std::optional<std::string> StdLines::Read()
{
    std::unique_lock<std::mutex> lock(m_lock);
    do
    {
        m_notEmpty.wait_for(lock, std::chrono::milliseconds(100), [&] {
            return !m_container.empty() || m_closed.load(std::memory_order_relaxed);
        });
    } while (m_container.empty() && !m_closed.load(std::memory_order_relaxed));

    if (m_closed.load(std::memory_order_relaxed) && m_container.empty())
    {
        return std::nullopt;
    }

    std::string result = m_container.front();
    m_container.pop();

    // coverity[uninit_use_in_call]
    lock.unlock();
    return result;
}

} //namespace DcgmNs::Common::Subprocess
