/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <catch2/catch_all.hpp>

#include <Pipe.hpp>

using namespace DcgmNs::Common::Subprocess;

namespace
{

bool IsValidFd(int fd)
{
    return fcntl(fd, F_GETFD) != -1;
}

} //namespace

TEST_CASE("Pipe: Read & Write")
{
    boost::asio::io_context ioCtx;
    Pipe obj(ioCtx);

    std::string const message = "Capoo";
    obj.WriteEnd().write_some(boost::asio::buffer(message));

    std::string buffer(message.size(), '\0');
    obj.ReadEnd().read_some(boost::asio::buffer(buffer));

    REQUIRE(message == buffer);
}

TEST_CASE("Pipe: Read after write end closed")
{
    boost::asio::io_context ioCtx;
    Pipe obj(ioCtx);

    std::string const message = "Capoo";
    obj.WriteEnd().write_some(boost::asio::buffer(message));

    obj.CloseWriteEnd();

    std::string buffer(message.size(), '\0');
    obj.ReadEnd().read_some(boost::asio::buffer(buffer));

    REQUIRE(message == buffer);
}

TEST_CASE("Pipe: Close")
{
    SECTION("CloseReadEnd")
    {
        boost::asio::io_context ioCtx;
        Pipe obj(ioCtx);

        int readFd  = obj.ReadEnd().native_handle();
        int writeFd = obj.WriteEnd().native_handle();

        obj.CloseReadEnd();
        REQUIRE(!IsValidFd(readFd));
        REQUIRE(IsValidFd(writeFd));
    }

    SECTION("CloseWriteEnd")
    {
        boost::asio::io_context ioCtx;
        Pipe obj(ioCtx);

        int readFd  = obj.ReadEnd().native_handle();
        int writeFd = obj.WriteEnd().native_handle();

        obj.CloseWriteEnd();
        REQUIRE(IsValidFd(readFd));
        REQUIRE(!IsValidFd(writeFd));
    }
}

TEST_CASE("Pipe: Destructor")
{
    boost::asio::io_context ioCtx;
    Pipe *obj = new Pipe(ioCtx);

    int readFd  = obj->ReadEnd().native_handle();
    int writeFd = obj->WriteEnd().native_handle();

    delete obj;
    REQUIRE(!IsValidFd(readFd));
    REQUIRE(!IsValidFd(writeFd));
}