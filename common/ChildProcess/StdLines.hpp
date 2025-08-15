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

#include <boost/iterator/iterator_facade.hpp>

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <string>


namespace DcgmNs::Common::Subprocess
{

class StdLines
{
public:
    StdLines() = default;

    struct StdOutSentinel
    {};

    class StdLinesIterator
        : public boost::iterator_facade<StdLinesIterator, std::string const, boost::single_pass_traversal_tag>
    {
    public:
        explicit StdLinesIterator(StdOutSentinel /*sentinel*/);
        explicit StdLinesIterator(StdLines &stdlines);

    protected:
        friend class boost::iterator_core_access;

        void increment();
        [[nodiscard]] std::string const &dereference() const;
        [[nodiscard]] std::string const &dereference();
        [[nodiscard]] bool equal(StdLinesIterator const &other) const;
        [[nodiscard]] bool equal(StdOutSentinel) const;

    private:
        std::string m_currentValue;
        StdLines *m_stdlines = nullptr;
    }; // StdLinesIterator

    /**
     * @brief Returns true if the buffer is empty.
     */
    bool IsEmpty();

    /**
     * @brief Writes one line into buffer.
     * @param[in] str message.
     */
    void Write(std::string const &str);

    /**
     * @brief Reads one line from buffer.
     */
    std::optional<std::string> Read();

    /**
     * Closes the channel.
     *
     * All the consumer of the channel will be unblocked and will receive the Sentinel.
     */
    void Close();

    StdLinesIterator begin();
    static StdLinesIterator end();

    std::queue<std::string> m_container;
    std::mutex m_lock;
    std::condition_variable m_notEmpty; //!< Condvar to signal that the buffer has some data to try to parse
    std::atomic_bool m_closed = false;
};

} //namespace DcgmNs::Common::Subprocess