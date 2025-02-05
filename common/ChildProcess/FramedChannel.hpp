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

#include "FastPimpl.hpp"

#include <DcgmUtilities.h>
#include <boost/iterator/iterator_facade.hpp>

#include <cstddef>
#include <span>


namespace DcgmNs::Common::Subprocess
{

/**
 * @brief This is a single-producer, single-consumer channel that allows to write and read framed messages.
 * Each message is prefixed with a 32-bit size of the message.
 */
class FramedChannel
{
public:
    /**
     * @brief Exception thrown when trying to write to a closed channel.
     *
     */
    class StreamClosedException : std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    /**
     * @brief Exception thrown when trying to create a second iterator on the same channel.
     *
     */
    class ConsumerOccupiedException : std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    class Sentinel
    {};
    class FramedChannelIterator
        : public boost::
              iterator_facade<FramedChannelIterator, std::vector<std::byte> const, boost::single_pass_traversal_tag>
    {
    public:
        FramedChannelIterator() = delete;
        explicit FramedChannelIterator(Sentinel);
        explicit FramedChannelIterator(FramedChannel &);
        ~FramedChannelIterator();

    protected:
        friend class boost::iterator_core_access;

        void increment();
        [[nodiscard]] std::vector<std::byte> const &dereference() const;
        [[nodiscard]] std::vector<std::byte> const &dereference();
        [[nodiscard]] bool equal(FramedChannelIterator const &other) const;
        [[nodiscard]] bool equal(Sentinel) const;

        FramedChannel *m_channel       = nullptr;
        std::vector<std::byte> m_value = {};
    };
    /**
     * @brief Writes FramedMessage bytes into the channel.
     * @param[in] buffer message bytes. May be partial of the whole message if the buffer length is already written.
     */
    void Write(std::span<std::byte const> buffer);
    /**
     * Closes the channel.
     *
     * All the consumer of the channel will be unblocked and will receive the Sentinel.
     */
    void Close();

    FramedChannelIterator begin();
    static FramedChannelIterator end()
    {
        return FramedChannelIterator(Sentinel {});
    }

    explicit FramedChannel(size_t bufferSize = 4096);
    ~FramedChannel();

private:
    struct Impl;
#if defined(__x86_64__)
    DcgmNs::Common::FastPimpl<Impl, 136, 8> m_impl;
#elif defined(__aarch64__)
    DcgmNs::Common::FastPimpl<Impl, 144, 8> m_impl;
#endif
};
} //namespace DcgmNs::Common::Subprocess
