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

#include <FdChannelClient.h>

#include <Defer.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

/**
 * @brief RAII owner of an anonymous pipe() file-descriptor pair used by tests.
 *
 * Wraps a Linux pipe so that the read and write ends are guaranteed to be
 * closed when the object goes out of scope, even if a Catch2 assertion in the
 * surrounding test aborts the scenario mid-flight. Also exposes targeted
 * CloseRead()/CloseWrite() helpers so tests can simulate one-sided shutdowns
 * (e.g., to provoke EPIPE), and a ReadExact() helper so callers don't have to
 * reach for free functions on top of the bare fd.
 *
 * @note This class is non-copyable; the underlying file descriptors have
 *       unique-ownership semantics and must not be duplicated implicitly.
 */
class Pipe
{
public:
    /**
     * @brief Create a fresh anonymous pipe and seed m_readFd / m_writeFd.
     *
     * Calls pipe(2). On the failure path the fds are left at
     * their default sentinel value of -1; downstream read/write calls will then
     * fail with EBADF and the enclosing Catch2 test will surface a normal
     * assertion failure instead of an exception escaping the constructor.
     */
    Pipe()
    {
        std::array<int, 2> fds { -1, -1 };
        (void)pipe(fds.data());
        m_readFd  = fds[0];
        m_writeFd = fds[1];
    }

    /**
     * @brief Close both ends of the pipe (read first, then write) if still open.
     */
    ~Pipe()
    {
        CloseRead();
        CloseWrite();
    }

    Pipe(Pipe const &)            = delete;
    Pipe &operator=(Pipe const &) = delete;

    /**
     * @brief Accessor for the read end of the pipe.
     * @return int The read file descriptor, or -1 if it has already been closed.
     */
    [[nodiscard]] int ReadFd() const
    {
        return m_readFd;
    }

    /**
     * @brief Accessor for the write end of the pipe.
     * @return int The write file descriptor, or -1 if it has already been closed.
     */
    [[nodiscard]] int WriteFd() const
    {
        return m_writeFd;
    }

    /**
     * @brief Close the read end of the pipe and reset m_readFd to -1.
     * @note Idempotent: calling it on an already-closed end is a no-op.
     */
    void CloseRead()
    {
        if (m_readFd != -1)
        {
            close(m_readFd);
            m_readFd = -1;
        }
    }

    /**
     * @brief Close the write end of the pipe and reset m_writeFd to -1.
     * @note Idempotent: calling it on an already-closed end is a no-op.
     */
    void CloseWrite()
    {
        if (m_writeFd != -1)
        {
            close(m_writeFd);
            m_writeFd = -1;
        }
    }

    /**
     * @brief Read up to @p size bytes from the read end into a fresh buffer.
     *
     * Transient read(2) failures with errno set to EINTR or EAGAIN are retried
     * transparently. The loop stops on any other read error or on EOF; in those
     * cases the returned vector is shorter than @p size. Callers should assert
     * on the returned size to detect truncation rather than relying on the
     * helper to abort.
     *
     * @param size Number of bytes the caller would like to read.
     * @return std::vector<char> containing the bytes actually read, in order.
     */
    [[nodiscard]] std::vector<char> ReadExact(std::size_t size) const
    {
        std::vector<char> out;
        out.reserve(size);
        while (out.size() < size)
        {
            std::array<char, 4096> chunk {};
            auto const want = std::min(chunk.size(), size - out.size());
            auto const n    = read(m_readFd, chunk.data(), want);
            if (n < 0 && (errno == EINTR || errno == EAGAIN))
            {
                continue;
            }
            if (n <= 0) // error or EOF
            {
                break;
            }
            out.insert(out.end(), chunk.data(), chunk.data() + n);
        }
        return out;
    }

private:
    int m_readFd  = -1;
    int m_writeFd = -1;
};

std::size_t LargePayloadSize(int fd)
{
    constexpr std::size_t fallbackPipeCapacity = 64 * 1024;
    std::size_t pipeCapacity                   = fallbackPipeCapacity;

#ifdef F_GETPIPE_SZ
    int const actualCapacity = fcntl(fd, F_GETPIPE_SZ);
    if (actualCapacity > 0)
    {
        pipeCapacity = static_cast<std::size_t>(actualCapacity);
    }
#endif

    return pipeCapacity + 4096;
}

class ScopedSigpipeIgnore
{
public:
    ScopedSigpipeIgnore()
        : m_previousHandler(std::signal(SIGPIPE, SIG_IGN))
    {}

    ~ScopedSigpipeIgnore()
    {
        std::signal(SIGPIPE, m_previousHandler);
    }

    ScopedSigpipeIgnore(ScopedSigpipeIgnore const &)            = delete;
    ScopedSigpipeIgnore &operator=(ScopedSigpipeIgnore const &) = delete;

private:
    using SignalHandler = void (*)(int);
    SignalHandler m_previousHandler;
};

} // namespace

SCENARIO("FdChannelClient::IsValid returns true for live pipe endpoints")
{
    GIVEN("an open pipe")
    {
        Pipe pipe;

        WHEN("the client wraps the write end")
        {
            FdChannelClient client(pipe.WriteFd());
            THEN("IsValid() is true")
            {
                REQUIRE(client.IsValid());
            }
        }

        WHEN("the client wraps the read end")
        {
            FdChannelClient client(pipe.ReadFd());
            THEN("IsValid() is true")
            {
                REQUIRE(client.IsValid());
            }
        }
    }
}

SCENARIO("FdChannelClient::IsValid returns false for an fd that was closed before construction")
{
    GIVEN("an open pipe whose write end is closed before the client is built")
    {
        Pipe pipe;
        int const fd = pipe.WriteFd();
        pipe.CloseWrite();

        WHEN("the client wraps the now-stale fd")
        {
            FdChannelClient client(fd);
            THEN("IsValid() is false")
            {
                REQUIRE_FALSE(client.IsValid());
            }
        }
    }
}

SCENARIO("FdChannelClient::IsValid returns false for fds that never refer to an open file")
{
    GIVEN("no real fd")
    {
        WHEN("the client wraps the sentinel value -1")
        {
            FdChannelClient client(-1);
            THEN("IsValid() is false")
            {
                REQUIRE_FALSE(client.IsValid());
            }
        }

        WHEN("the client wraps an fd well outside the process table")
        {
            // 1_000_000 is far above any plausible open fd; fcntl should fail with EBADF.
            FdChannelClient client(1'000'000);
            THEN("IsValid() is false")
            {
                REQUIRE_FALSE(client.IsValid());
            }
        }
    }
}

SCENARIO("FdChannelClient::Write delivers length-prefixed frames to the channel")
{
    GIVEN("a pipe and a client wrapping the write end")
    {
        Pipe pipe;
        FdChannelClient client(pipe.WriteFd());

        WHEN("Write() is called with a non-empty payload")
        {
            std::string const payload = "hello";
            REQUIRE(client.Write(std::span<char const> { payload.data(), payload.size() }));
            pipe.CloseWrite();

            THEN("a uint32_t length header followed by the payload bytes appears on the read end")
            {
                auto header = pipe.ReadExact(sizeof(std::uint32_t));
                REQUIRE(header.size() == sizeof(std::uint32_t));
                std::uint32_t len = 0;
                std::memcpy(&len, header.data(), sizeof(len));
                REQUIRE(len == payload.size());

                auto body = pipe.ReadExact(len);
                REQUIRE(body.size() == len);
                REQUIRE(std::string_view(body.data(), body.size()) == payload);

                AND_THEN("no further data follows the frame")
                {
                    char tail = 0;
                    REQUIRE(read(pipe.ReadFd(), &tail, 1) == 0);
                }
            }
        }

        WHEN("Write() is called with an empty span")
        {
            REQUIRE(client.Write(std::span<char const> {}));
            pipe.CloseWrite();

            THEN("only a zero-length header is delivered")
            {
                auto header = pipe.ReadExact(sizeof(std::uint32_t));
                REQUIRE(header.size() == sizeof(std::uint32_t));
                std::uint32_t len = 0xDEADBEEFu;
                std::memcpy(&len, header.data(), sizeof(len));
                REQUIRE(len == 0u);

                char tail = 0;
                REQUIRE(read(pipe.ReadFd(), &tail, 1) == 0);
            }
        }
    }
}

SCENARIO("FdChannelClient::Write loops through partial writes for large payloads")
{
    GIVEN("a pipe, a client wrapping the write end, and a concurrent reader draining the read end")
    {
        Pipe pipe;
        FdChannelClient client(pipe.WriteFd());

        std::size_t const payloadSize = LargePayloadSize(pipe.WriteFd());
        std::vector<char> payload(payloadSize);
        for (std::size_t i = 0; i < payload.size(); ++i)
        {
            payload[i] = static_cast<char>(i & 0xFF);
        }

        std::vector<char> received;
        received.reserve(sizeof(std::uint32_t) + payloadSize);

        WHEN("Write() is called with a payload larger than the pipe buffer")
        {
            {
                // Concurrent reader drains the pipe to force WriteAll's partial-write loop and
                // exercise concurrent fd access patterns for dynamic analysis (e.g., ThreadSanitizer).
                std::jthread reader([&] {
                    auto header = pipe.ReadExact(sizeof(std::uint32_t));
                    received.insert(received.end(), header.begin(), header.end());
                    auto body = pipe.ReadExact(payloadSize);
                    received.insert(received.end(), body.begin(), body.end());
                });

                auto closeWrite        = DcgmNs::Defer([&] { pipe.CloseWrite(); });
                bool const writeResult = client.Write(std::span<char const> { payload.data(), payload.size() });
                pipe.CloseWrite();
                REQUIRE(writeResult);
            } // jthread destructor joins the reader before THEN inspects `received`.

            THEN("the entire frame is delivered in order via the partial-write loop")
            {
                REQUIRE(received.size() == sizeof(std::uint32_t) + payloadSize);
                std::uint32_t len = 0;
                std::memcpy(&len, received.data(), sizeof(len));
                REQUIRE(len == payloadSize);
                REQUIRE(std::memcmp(received.data() + sizeof(len), payload.data(), payloadSize) == 0);
            }
        }
    }
}

SCENARIO("FdChannelClient::Write returns false when the underlying fd is invalid")
{
    GIVEN("a client wrapping the sentinel value -1")
    {
        FdChannelClient client(-1);

        WHEN("Write() is called with a payload")
        {
            std::string const payload = "noop";
            bool const result         = client.Write(std::span<char const> { payload.data(), payload.size() });

            THEN("it returns false without touching the fd")
            {
                REQUIRE_FALSE(result);
            }
        }
    }
}

SCENARIO("FdChannelClient::Write returns false when the peer has closed the channel")
{
    GIVEN("a pipe whose reader has closed and a client wrapping the write end")
    {
        ScopedSigpipeIgnore ignoreSigpipe;
        Pipe pipe;
        pipe.CloseRead(); // Peer is gone; subsequent writes will fail with EPIPE.
        FdChannelClient client(pipe.WriteFd());

        // Sanity: our end is still open, so IsValid (which only inspects the local fd) is true.
        REQUIRE(client.IsValid());

        WHEN("Write() is called")
        {
            std::string const payload = "broken pipe";
            bool const result         = client.Write(std::span<char const> { payload.data(), payload.size() });

            THEN("it returns false because of EPIPE")
            {
                REQUIRE_FALSE(result);
            }
        }
    }
}
