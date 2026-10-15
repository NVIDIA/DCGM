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

#include <ScopedFd.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <utility>

namespace
{
[[nodiscard]] bool IsOpen(int fd)
{
    errno = 0;
    return fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}
} // namespace

TEST_CASE("ScopedFd closes owned descriptors")
{
    int fds[2] {};
    REQUIRE(pipe(fds) == 0);
    int const readFd  = fds[0];
    int const writeFd = fds[1];

    {
        ScopedFd readEnd(readFd);
        ScopedFd writeEnd(writeFd);

        CHECK(readEnd.Get() == readFd);
        CHECK(writeEnd.Get() == writeFd);
        CHECK(readEnd.IsValid());
        CHECK(writeEnd.IsValid());
    }

    CHECK_FALSE(IsOpen(readFd));
    CHECK_FALSE(IsOpen(writeFd));
}

TEST_CASE("ScopedFd Release abandons ownership")
{
    int fds[2] {};
    REQUIRE(pipe(fds) == 0);
    ScopedFd readEnd(fds[0]);
    ScopedFd writeEnd(fds[1]);

    auto const releasedReadFd = readEnd.Release();

    CHECK(releasedReadFd == fds[0]);
    CHECK_FALSE(readEnd.IsValid());
    CHECK(IsOpen(releasedReadFd));

    ScopedFd releasedReadEnd(releasedReadFd);
}

TEST_CASE("ScopedFd Reset replaces ownership")
{
    int firstPipe[2] {};
    int secondPipe[2] {};
    REQUIRE(pipe(firstPipe) == 0);
    REQUIRE(pipe(secondPipe) == 0);

    ScopedFd firstWriteEnd(firstPipe[1]);
    ScopedFd secondWriteEnd(secondPipe[1]);

    ScopedFd readEnd(firstPipe[0]);
    int const firstReadFd  = firstPipe[0];
    int const secondReadFd = secondPipe[0];

    readEnd.Reset(secondReadFd);

    CHECK_FALSE(IsOpen(firstReadFd));
    CHECK(IsOpen(secondReadFd));
    CHECK(readEnd.Get() == secondReadFd);

    readEnd.Reset();
    CHECK_FALSE(readEnd.IsValid());
    CHECK_FALSE(IsOpen(secondReadFd));
}

TEST_CASE("ScopedFd supports move-only ownership transfer")
{
    int fds[2] {};
    REQUIRE(pipe(fds) == 0);

    ScopedFd readEnd(fds[0]);
    ScopedFd writeEnd(fds[1]);

    auto movedReadEnd = std::move(readEnd);

    CHECK_FALSE(readEnd.IsValid());
    CHECK(movedReadEnd.Get() == fds[0]);

    ScopedFd replacement;
    replacement = std::move(writeEnd);

    CHECK_FALSE(writeEnd.IsValid());
    CHECK(replacement.Get() == fds[1]);
}
