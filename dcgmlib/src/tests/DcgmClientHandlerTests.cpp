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

#define DCGMLIB_TESTS
#include <DcgmClientHandler.h>
#include <catch2/catch_all.hpp>
#include <dcgm_structs.h>

TEST_CASE("DcgmClientHandler::splitIdentifierAndPort")
{
    DcgmClientHandler dch {};
    std::vector<char> tmp {};
    unsigned int portNum;

    constexpr size_t SUN_PATH_MAX        = 108;
    constexpr unsigned int PORT_SENTINEL = 222222; // intentionally invalid and unused below

    auto const &maxValidUnixPath = std::string(SUN_PATH_MAX, 'a');
    auto const &invalidUnixPath  = maxValidUnixPath + "z";
    auto const &maxValidIpAddr   = std::string(2 * INET6_ADDRSTRLEN, 'A');
    auto const &invalidIpAddr    = maxValidIpAddr + "Z";

    struct
    {
        char const *identifier;
        bool isUnixDomain;
        dcgmReturn_t expectedRet;
        std::string_view expectedResult;
        unsigned int expectedPort;
    } const tests[] = {
        // Boundary checks
        { "", true, DCGM_ST_NO_DATA, "", PORT_SENTINEL },
        { maxValidUnixPath.c_str(), true, DCGM_ST_OK, maxValidUnixPath.c_str(), DCGM_HE_PORT_NUMBER },
        { invalidUnixPath.c_str(), true, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
        { "", false, DCGM_ST_NO_DATA, "", PORT_SENTINEL },
        { maxValidIpAddr.c_str(), false, DCGM_ST_OK, maxValidIpAddr.c_str(), DCGM_HE_PORT_NUMBER },
        { invalidIpAddr.c_str(), false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },

        // IPv4 Address and Port
        { "127.0.0.1:1", false, DCGM_ST_OK, "127.0.0.1", 1 },
        { "127.0.0.1", false, DCGM_ST_OK, "127.0.0.1", DCGM_HE_PORT_NUMBER },
        { "127.0.0.1:65535", false, DCGM_ST_OK, "127.0.0.1", 65535 },
        { "127.0.0.1:65536", false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
        { "127.0.0.1:0", false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
        { "127.0.0.1:65536", true, DCGM_ST_OK, "127.0.0.1:65536", DCGM_HE_PORT_NUMBER },

        // Bracketed IPv6 Address and Port
        { "[::]", false, DCGM_ST_OK, "[::]", DCGM_HE_PORT_NUMBER },
        { "[::1]", false, DCGM_ST_OK, "[::1]", DCGM_HE_PORT_NUMBER },
        { "[::]:32767", false, DCGM_ST_OK, "[::]", 32767 },
        { "[::1]:32768", false, DCGM_ST_OK, "[::1]", 32768 },
        { "[1234:abcd:5678:efab:9012:cdef:3456:abcd]",
          false,
          DCGM_ST_OK,
          "[1234:abcd:5678:efab:9012:cdef:3456:abcd]",
          DCGM_HE_PORT_NUMBER },
        { "[1234:abcd:5678:efab:9012:cdef:3456:abcd]:1",
          false,
          DCGM_ST_OK,
          "[1234:abcd:5678:efab:9012:cdef:3456:abcd]",
          1 },
        { "[1234:abcd:5678:efab:9012:cdef:3456:abcd]:65535",
          false,
          DCGM_ST_OK,
          "[1234:abcd:5678:efab:9012:cdef:3456:abcd]",
          65535 },
        { "[1234:abcd:5678:efab:9012:cdef:3456:abcd]:0", false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
        { "[1234:abcd:5678:efab:9012:cdef:3456:abcd]:65536", false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
        // The following is invalid, but the current code should process as if valid.
        { "[1234:abcd:5678:efab:9abc:cdef:invalid:invalid:invalid:invalid:invalid:invalid]:12345",
          false,
          DCGM_ST_OK,
          "[1234:abcd:5678:efab:9abc:cdef:invalid:invalid:invalid:invalid:invalid:invalid]",
          12345 },

        // Unbracketed IPv6 Address and Port - invalid
        { "::", false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
        { "::1", false, DCGM_ST_OK, ":", 1 },
        { ":::1", false, DCGM_ST_OK, "::", 1 },
        { "::1:5566", false, DCGM_ST_OK, "::1", 5566 },
        { "1234:abcd:5678:efab:9abc:cdef:defc:0123:5566",
          false,
          DCGM_ST_OK,
          "1234:abcd:5678:efab:9abc:cdef:defc:0123",
          5566 },
        { "ff00::1:5566", false, DCGM_ST_OK, "ff00::1", 5566 },

        // invalid
        { "a", true, DCGM_ST_OK, "a", DCGM_HE_PORT_NUMBER },
        { "ab", true, DCGM_ST_OK, "ab", DCGM_HE_PORT_NUMBER },
        { "a", false, DCGM_ST_OK, "a", DCGM_HE_PORT_NUMBER },
        { "ab", false, DCGM_ST_OK, "ab", DCGM_HE_PORT_NUMBER },
        { ":", false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
        { ":1", false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
        { "[]", false, DCGM_ST_OK, "[]", DCGM_HE_PORT_NUMBER },
        { "]:", false, DCGM_ST_BADPARAM, "", PORT_SENTINEL },
    };

    for (auto const &test : tests)
    {
        DYNAMIC_SECTION("addr=" << test.identifier << ", isUnixDomain=" << test.isUnixDomain)
        {
            tmp.clear();
            portNum         = PORT_SENTINEL;
            dcgmReturn_t st = dch.splitIdentifierAndPort(test.identifier, test.isUnixDomain, tmp, portNum);
            CHECK(st == test.expectedRet);
            CHECK(portNum == test.expectedPort);

            /* Conditional. string_view(empty vector) is invalid and will segfault */
            REQUIRE(test.expectedResult.empty() == tmp.empty());
            if (!test.expectedResult.empty())
            {
                CHECK(std::string_view(tmp.data()) == test.expectedResult);
            }
        }
    }
}