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

#include "FramedChannel.hpp"

#include <DcgmLogging.h>

#include <boost/circular_buffer.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>

namespace DcgmNs::Common::Subprocess
{

FramedChannel::FramedChannelIterator::FramedChannelIterator(Sentinel /*sentinel*/)
{}

std::vector<std::byte> const &FramedChannel::FramedChannelIterator::dereference() const
{
    return m_value;
}

std::vector<std::byte> const &FramedChannel::FramedChannelIterator::dereference()
{
    return m_value;
}

bool FramedChannel::FramedChannelIterator::equal(FramedChannelIterator const &other) const
{
    return m_channel == nullptr && other.m_channel == nullptr;
}

bool FramedChannel::FramedChannelIterator::equal(Sentinel) const
{
    return m_channel == nullptr;
}

struct FramedChannel::Impl
{
    friend class FramedChannel::FramedChannelIterator;
    boost::circular_buffer<std::byte> m_buffer;
    std::mutex m_lock;
    std::condition_variable m_notEmpty; //!< Condvar to signal that the buffer has some data to try to parse
    std::atomic_bool m_closed   = false;
    std::atomic_bool m_occupied = false;

    void AddConsumer() noexcept(false)
    {
        bool expected = false;
        if (!m_occupied.compare_exchange_strong(expected, true, std::memory_order_acquire, std::memory_order_relaxed))
        {
            throw ConsumerOccupiedException { "Only one iterator is allowed" };
        }
    }

    void ReleaseConsumer()
    {
        m_occupied.store(false, std::memory_order_release);
    }

    /**
     * @brief Consumes one complete frame from the stream.
     * Will block if there is no complete frame in the buffer yet.
     * @returns One complete frame from the stream. The result will not contain length prefix from the protocol.
     */
    [[nodiscard]] std::optional<std::vector<std::byte>> ReadOneFrame()
    {
        std::unique_lock<std::mutex> lock(m_lock);
        if (m_buffer.size() < sizeof(std::uint32_t))
        {
            do
            {
                m_notEmpty.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return m_buffer.size() >= sizeof(std::uint32_t) || m_closed.load(std::memory_order_relaxed);
                });
            } while (m_buffer.size() < sizeof(std::uint32_t) && !m_closed.load(std::memory_order_relaxed));
        }

        if (m_buffer.size() < sizeof(std::uint32_t))
        {
            assert(m_closed.load(std::memory_order_relaxed));
            return std::nullopt;
        }

        std::uint32_t len = 0;
        if (m_buffer.array_one().second > sizeof(std::uint32_t))
        {
            memcpy(&len, m_buffer.array_one().first, sizeof(std::uint32_t));
        }
        else
        {
            std::byte lenBytes[sizeof(std::uint32_t)] = {};
            std::copy_n(m_buffer.begin(), sizeof(std::uint32_t), lenBytes);
            memcpy(&len, lenBytes, sizeof(std::uint32_t));
        }

        assert(len != 0);
        m_buffer.rerase(m_buffer.begin(), m_buffer.begin() + sizeof(int));

        if ((size_t)len > m_buffer.size())
        {
            do
            {
                m_notEmpty.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return m_buffer.size() >= (size_t)len || m_closed.load(std::memory_order_relaxed);
                });
            } while ((size_t)len > m_buffer.size() && !m_closed.load(std::memory_order_relaxed));
        }

        if (m_buffer.size() < (size_t)len)
        {
            assert(m_closed.load(std::memory_order_relaxed));
            return std::nullopt;
        }

        std::vector<std::byte> result(len);
        std::copy_n(m_buffer.begin(), len, result.begin());
        m_buffer.rerase(m_buffer.begin(), m_buffer.begin() + len);

        return result;
    }

    /**
     * @brief Closes the stream.
     * Will unblock all threads waiting for data to be written or read.
     */
    void Close()
    {
        m_closed.store(true, std::memory_order_relaxed);
        m_notEmpty.notify_one();
    }

    /**
     * @brief Writes a chunk of bytes to the stream.
     * Will block if there is not enough room in the buffer.

     * @param[in] data Bytes to write
     * @throws \c StreamClosedException if the stream is closed
     */
    void Write(std::span<std::byte const> data)
    {
        if (m_closed.load(std::memory_order_relaxed))
        {
            log_error("Attempt to write to closed stream");
            throw StreamClosedException { "Attempt to write to closed stream" };
        }

        if (data.empty())
        {
            return;
        }

        {
            std::unique_lock<std::mutex> lock(m_lock);
            if (data.size() > m_buffer.reserve())
            {
                // Here we could check that the size of the data is greater than m_buffer.capacity() and increase the
                // capacity to the size of the data.
                // Unfortunately, this would lead to a deadlock as the reader waits till the whole frame can be read
                // from the buffer. If a frame cannot be fully written here without overwriting data that has not been
                // consumed by the reader, we would get a deadlock.
                m_buffer.set_capacity(std::max(m_buffer.capacity() + data.size(),
                                               decltype(m_buffer)::capacity_type(m_buffer.capacity() * 1.5)));
            }

            m_buffer.insert(m_buffer.end(), data.begin(), data.end());
        }
        m_notEmpty.notify_one();
    }
};

FramedChannel::FramedChannelIterator FramedChannel::begin()
{
    return FramedChannel::FramedChannelIterator(*this);
}

void FramedChannel::Write(std::span<std::byte const> buffer)
{
    m_impl->Write(buffer);
}

void FramedChannel::Close()
{
    m_impl->Close();
}

FramedChannel::FramedChannel(size_t bufferSize)
{
    m_impl->m_buffer.set_capacity(bufferSize);
}

FramedChannel::~FramedChannel() = default;

void FramedChannel::FramedChannelIterator::increment()
{
    auto frame = m_channel->m_impl->ReadOneFrame();
    if (!frame)
    {
        m_channel = nullptr;
    }
    else
    {
        m_value = *frame;
    }
}

FramedChannel::FramedChannelIterator::FramedChannelIterator(FramedChannel &channel)
    : m_channel(&channel)
{
    m_channel->m_impl->AddConsumer();
    increment();
}

FramedChannel::FramedChannelIterator::~FramedChannelIterator()
{
    if (m_channel != nullptr)
    {
        m_channel->m_impl->ReleaseConsumer();
    }
}
} //namespace DcgmNs::Common::Subprocess