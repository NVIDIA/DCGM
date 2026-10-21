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

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

bool WriteAll(int fd, void const *buf, size_t count)
{
    size_t totalWritten = 0;
    ssize_t lastWritten = 0;
    while (totalWritten < count)
    {
        lastWritten = write(fd, static_cast<char const *>(buf) + totalWritten, count - totalWritten);
        if (lastWritten < 0)
        {
            if (errno == EINTR || errno == EAGAIN)
            {
                continue;
            }
            return false;
        }

        if (lastWritten == 0)
        {
            errno = EIO;
            return false;
        }

        totalWritten += lastWritten;
    }
    return true;
}

namespace detail
{

/**
 * Parse an integer command-line argument.
 *
 * @param[in] arg   Null-terminated argument text to parse.
 * @param[in] name  Argument name used in the stderr diagnostic.
 *
 * @return Parsed integer, or std::nullopt when the argument is not a complete valid integer.
 */
inline std::optional<int> ParseIntArg(char const *arg, std::string_view name)
{
    int value {};
    auto const *begin = arg;
    auto const *end   = arg + std::strlen(arg);
    auto [ptr, ec]    = std::from_chars(begin, end, value);
    if (ec != std::errc {} || ptr != end)
    {
        std::cerr << "Invalid numeric argument for " << name << ": " << arg << std::endl;
        return std::nullopt;
    }
    return value;
}

/**
 * Parse a non-negative file descriptor argument.
 *
 * @param[in] arg   Null-terminated argument text to parse.
 * @param[in] name  Argument name used in the stderr diagnostic.
 *
 * @return Parsed file descriptor, or std::nullopt for non-numeric, incomplete, or negative values.
 */
inline std::optional<int> ParseNonNegativeFdArg(char const *arg, std::string_view name)
{
    auto value = ParseIntArg(arg, name);
    if (!value.has_value())
    {
        return std::nullopt;
    }
    if (*value < 0)
    {
        std::cerr << "Invalid file descriptor for " << name << ": " << *value << " (must be >= 0)" << std::endl;
        return std::nullopt;
    }
    return value;
}

/**
 * Parse a delay argument in milliseconds and convert it to microseconds.
 *
 * @param[in] arg  Null-terminated delay value in milliseconds.
 *
 * @return Delay in microseconds for usleep(), or std::nullopt for invalid, negative, or overflowing values.
 */
inline std::optional<useconds_t> ParseDelayMicrosArg(char const *arg)
{
    auto delayMs = ParseIntArg(arg, "delayMs");
    if (!delayMs.has_value())
    {
        return std::nullopt;
    }
    if (*delayMs < 0)
    {
        std::cerr << "Invalid delayMs: " << *delayMs << std::endl;
        return std::nullopt;
    }

    auto const micros = static_cast<std::uint64_t>(*delayMs) * 1000ULL;
    if (micros > std::numeric_limits<useconds_t>::max())
    {
        std::cerr << "delayMs exceeds usleep range: " << *delayMs << std::endl;
        return std::nullopt;
    }
    return static_cast<useconds_t>(micros);
}

/**
 * Print a WriteAll failure diagnostic to stderr using the current errno value.
 *
 * @param[in] writeKind  Description of the write operation that failed.
 * @param[in] fd         File descriptor passed to WriteAll.
 * @param[in] size       Number of bytes that WriteAll attempted to write.
 */
inline void PrintWriteFailure(std::string_view writeKind, int fd, size_t size)
{
    std::cerr << "Failed to write " << writeKind << " of " << size << " bytes to fd " << fd << ": "
              << std::strerror(errno) << std::endl;
}

} // namespace detail

int Stdout(int /* argc */, char *argv[])
{
    std::cout << argv[0] << std::endl;
    return 0;
}

int DelayedStdout(int /* argc */, char *argv[])
{
    usleep(20000);
    std::cout << argv[0] << std::endl;
    return 0;
}

int Stderr(int /* argc */, char *argv[])
{
    std::cerr << argv[0] << std::endl;
    return 1;
}

int DelayedStderr(int /* argc */, char *argv[])
{
    usleep(20000);
    std::cerr << argv[0] << std::endl;
    return 1;
}

int Env(int /* argc */, char *argv[])
{
    auto *envVal = std::getenv(argv[0]);
    if (!envVal)
    {
        std::cout << "env not found!" << std::endl;
        return 1;
    }

    std::cout << std::string(envVal) << std::endl;
    return 0;
}

int FdChannel(int /* argc */, char *argv[])
{
    int fdChannel = atoi(argv[0]);

    if (-1 == fcntl(fdChannel, F_GETFD))
    {
        std::cerr << "Unable to get file descriptor for response channel " << argv[0] << std::endl;
        return 1;
    }

    std::string msg = argv[1];
    auto length     = static_cast<std::uint32_t>(msg.size());

    if (!WriteAll(fdChannel, &length, sizeof(length)))
    {
        std::cerr << "Unable to write to fd " << fdChannel << "." << std::endl;
        return 1;
    }

    if (!WriteAll(fdChannel, msg.data(), msg.size()))
    {
        std::cerr << "Unable to write to fd " << fdChannel << "." << std::endl;
        return 1;
    }

    return 0;
}

/**
 * Writes two framed messages to two file descriptors.
 *
 * @param argc Unused argument count.
 * @param argv Arguments where argv[0]=firstFd, argv[1]=firstMsg, argv[2]=secondFd, and argv[3]=secondMsg.
 * @return 0 on success, 1 on invalid input or write failure.
 */
int MultiFdChannel(int /* argc */, char *argv[])
{
    auto firstFd  = detail::ParseNonNegativeFdArg(argv[0], "firstFd");
    auto secondFd = detail::ParseNonNegativeFdArg(argv[2], "secondFd");
    if (!firstFd.has_value() || !secondFd.has_value())
    {
        return 1;
    }

    std::string firstMsg  = argv[1];
    std::string secondMsg = argv[3];
    auto firstLength      = static_cast<std::uint32_t>(firstMsg.size());
    auto secondLength     = static_cast<std::uint32_t>(secondMsg.size());

    if (!WriteAll(*firstFd, &firstLength, sizeof(firstLength)))
    {
        detail::PrintWriteFailure("first length", *firstFd, sizeof(firstLength));
        return 1;
    }
    if (!WriteAll(*firstFd, firstMsg.data(), firstMsg.size()))
    {
        detail::PrintWriteFailure("first data", *firstFd, firstMsg.size());
        return 1;
    }
    if (!WriteAll(*secondFd, &secondLength, sizeof(secondLength)))
    {
        detail::PrintWriteFailure("second length", *secondFd, sizeof(secondLength));
        return 1;
    }
    if (!WriteAll(*secondFd, secondMsg.data(), secondMsg.size()))
    {
        detail::PrintWriteFailure("second data", *secondFd, secondMsg.size());
        return 1;
    }
    return 0;
}

/**
 * Sleeps, then writes one framed message to a file descriptor.
 *
 * @param argc Unused argument count.
 * @param argv Arguments where argv[0]=fdChannel, argv[1]=delayMs in milliseconds, and argv[2]=msg.
 * @return 0 on success, 1 on invalid input or write failure.
 */
int FdChannelDelayed(int /* argc */, char *argv[])
{
    auto fdChannel   = detail::ParseNonNegativeFdArg(argv[0], "fdChannel");
    auto delayMicros = detail::ParseDelayMicrosArg(argv[1]);
    if (!fdChannel.has_value() || !delayMicros.has_value())
    {
        return 1;
    }

    std::string msg   = argv[2];
    auto const length = static_cast<std::uint32_t>(msg.size());

    usleep(*delayMicros);
    if (!WriteAll(*fdChannel, &length, sizeof(length)))
    {
        detail::PrintWriteFailure("length", *fdChannel, sizeof(length));
        return 1;
    }
    if (!WriteAll(*fdChannel, msg.data(), msg.size()))
    {
        detail::PrintWriteFailure("data", *fdChannel, msg.size());
        return 1;
    }
    return 0;
}

int Sleep(int /* argc */, char *argv[])
{
    int seconds = atoi(argv[0]);
    sleep(seconds);
    std::cout << "Sleep [" << seconds << "] seconds completed." << std::endl;
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Too few arguments." << std::endl;
        return 1;
    }

    std::unordered_map<std::string, std::pair<int, std::function<int(int, char *[])>>> handlers {
        { "stdout", { 1, Stdout } },
        { "stderr", { 1, Stderr } },
        { "env", { 1, Env } },
        { "fd-channel", { 2, FdChannel } },
        { "multi-fd-channel", { 4, MultiFdChannel } },
        { "fd-channel-delayed", { 3, FdChannelDelayed } },
        { "sleep", { 1, Sleep } },
        { "delayedStdout", { 1, DelayedStdout } },
        { "delayedStderr", { 1, DelayedStderr } },
    };

    if (!handlers.contains(argv[1]))
    {
        std::cout << "Unknown action." << std::endl;
        return 1;
    }

    auto &[expectedArgc, handler] = handlers.at(argv[1]);
    if (argc < 2 + expectedArgc)
    {
        std::cerr << "Too few arguments." << std::endl;
        return 1;
    }

    return handler(argc - 2, argv + 2);
}
