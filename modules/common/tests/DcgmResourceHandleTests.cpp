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

#include "DcgmCoreProxy.h"
#include "DcgmResourceHandle.h"
#include <catch2/catch_test_macros.hpp>

// Mock class that overrides just the methods we need for testing
class MockDcgmCoreProxy : public DcgmCoreProxy
{
public:
    MockDcgmCoreProxy()
        : DcgmCoreProxy(dcgmCoreCallbacks_t {}) // Pass empty callbacks to base
        , m_isReserved(false)
        , m_currentToken(0)
    {}

    // Override the methods we care about
    dcgmReturn_t ReserveResources(unsigned int &token) override
    {
        if (m_isReserved)
        {
            return DCGM_ST_IN_USE;
        }
        m_isReserved   = true;
        m_currentToken = token = 12345; // Fixed token for testing
        return DCGM_ST_OK;
    }

    dcgmReturn_t FreeResources(unsigned int token) override
    {
        if (!m_isReserved || token != m_currentToken)
        {
            return DCGM_ST_GENERIC_ERROR;
        }
        m_isReserved   = false;
        m_currentToken = 0;
        return DCGM_ST_OK;
    }

private:
    bool m_isReserved;
    unsigned int m_currentToken;
};

TEST_CASE("DcgmResourceHandle - Basic functionality")
{
    MockDcgmCoreProxy proxy;

    SECTION("Resource handle creation and cleanup")
    {
        {
            DcgmResourceHandle handle(proxy);

            REQUIRE(handle.GetInitResult() == DCGM_ST_OK);
            REQUIRE(handle.GetToken() == 12345);
        }
    }

    SECTION("Multiple resource handle creation attempts")
    {
        DcgmResourceHandle handle1(proxy);
        DcgmResourceHandle handle2(proxy);

        REQUIRE(handle1.GetInitResult() == DCGM_ST_OK);
        REQUIRE(handle2.GetInitResult() == DCGM_ST_IN_USE);
    }
}

TEST_CASE("DcgmResourceHandle - Move semantics")
{
    MockDcgmCoreProxy proxy;

    SECTION("Move constructor")
    {
        // Move valid 1 to newly created 2
        DcgmResourceHandle handle1(proxy);
        unsigned int token1 = handle1.GetToken();

        DcgmResourceHandle handle2(std::move(handle1));

        REQUIRE(handle1.GetInitResult() == DCGM_ST_GENERIC_ERROR);
        REQUIRE(handle1.GetToken() == 0);
        REQUIRE(handle2.GetInitResult() == DCGM_ST_OK);
        REQUIRE(handle2.GetToken() == token1);
    }

    SECTION("Move assignment")
    {
        // Move invalid 2 to valid 1, valid 1 should be overwritten
        DcgmResourceHandle handle1(proxy);
        DcgmResourceHandle handle2(proxy);
        unsigned int token2 = handle2.GetToken();
        dcgmReturn_t ret    = handle2.GetInitResult();

        handle1 = std::move(handle2);

        REQUIRE(handle1.GetInitResult() == ret);
        REQUIRE(handle1.GetToken() == token2);
        REQUIRE(handle2.GetInitResult() == DCGM_ST_GENERIC_ERROR);
        REQUIRE(handle2.GetToken() == 0);
    }

    SECTION("Resource cleanup in move assignment")
    {
        DcgmResourceHandle handle1(proxy);
        DcgmResourceHandle handle2(proxy);
        unsigned int token1 = handle1.GetToken();
        dcgmReturn_t ret    = handle1.GetInitResult();

        handle2 = std::move(handle1);
        DcgmResourceHandle handle3(proxy);

        REQUIRE(handle1.GetToken() == 0);
        REQUIRE(handle1.GetInitResult() == DCGM_ST_GENERIC_ERROR);
        REQUIRE(handle2.GetToken() == token1);
        REQUIRE(handle2.GetInitResult() == ret);
        REQUIRE(handle3.GetInitResult() == DCGM_ST_IN_USE);
    }

    SECTION("Chained moves")
    {
        DcgmResourceHandle handle1(proxy);
        unsigned int token1 = handle1.GetToken();

        DcgmResourceHandle handle2 = std::move(handle1);
        DcgmResourceHandle handle3 = std::move(handle2);

        REQUIRE(handle3.GetToken() == token1);
        REQUIRE(handle3.GetInitResult() == DCGM_ST_OK);
        REQUIRE(handle1.GetToken() == 0);
        REQUIRE(handle1.GetInitResult() == DCGM_ST_GENERIC_ERROR);
        REQUIRE(handle2.GetToken() == 0);
        REQUIRE(handle2.GetInitResult() == DCGM_ST_GENERIC_ERROR);
    }
}
