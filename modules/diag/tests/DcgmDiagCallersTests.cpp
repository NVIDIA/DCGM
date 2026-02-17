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

#include <catch2/catch_all.hpp>

#include <DcgmDiagCallers.h>

TEST_CASE("DcgmDiagCallers::SetConnectionId")
{
    DcgmDiagCallers callers;

    SECTION("valid case")
    {
        callers.SetConnectionId(5566);
        REQUIRE(callers.Exists(5566));
    }

    SECTION("on existing connection id")
    {
        callers.SetConnectionId(5566);
        callers.SetConnectionId(5566);
        REQUIRE(callers.Exists(5566));
    }
}

TEST_CASE("DcgmDiagCallers::ResetConnectionId")
{
    DcgmDiagCallers callers;

    SECTION("valid case")
    {
        callers.SetConnectionId(5566);
        callers.ResetConnectionId(5566);
        REQUIRE(!callers.Exists(5566));
    }

    SECTION("on non-existent connection id")
    {
        callers.ResetConnectionId(5566);
        REQUIRE(!callers.Exists(5566));
    }
}

TEST_CASE("DcgmDiagCallers::SetAlreadyStopped")
{
    DcgmDiagCallers callers;

    SECTION("valid case")
    {
        callers.SetConnectionId(5566);
        callers.SetAlreadyStopped(5566);
        REQUIRE(callers.IsAlreadyStopped(5566));
    }

    SECTION("on non-existent connection id")
    {
        callers.SetAlreadyStopped(5566);
        REQUIRE(!callers.IsAlreadyStopped(5566));
    }
}

TEST_CASE("DcgmDiagCallers::IsAlreadyStopped")
{
    SECTION("on non-existent connection id")
    {
        DcgmDiagCallers callers;
        REQUIRE(!callers.IsAlreadyStopped(5566));
    }
}

TEST_CASE("DcgmDiagCallers::Exists")
{
    SECTION("on non-existent connection id")
    {
        DcgmDiagCallers callers;
        REQUIRE(!callers.Exists(5566));
    }
}

TEST_CASE("DcgmDiagCallers::SetHeartbeatEnabled")
{
    DcgmDiagCallers callers;

    SECTION("valid case")
    {
        callers.SetConnectionId(5566);
        callers.SetHeartbeatEnabled(5566, true);
        REQUIRE(callers.IsHeartbeatEnabled(5566));
    }

    SECTION("on non-existent connection id")
    {
        callers.SetHeartbeatEnabled(5566, true);
        REQUIRE(!callers.IsHeartbeatEnabled(5566));
    }
}

TEST_CASE("DcgmDiagCallers::IsHeartbeatEnabled")
{
    SECTION("on non-existent connection id")
    {
        DcgmDiagCallers callers;
        REQUIRE(!callers.IsHeartbeatEnabled(5566));
    }
}

TEST_CASE("DcgmDiagCallers::ReceiveHeartbeat")
{
    SECTION("on non-existent connection id")
    {
        DcgmDiagCallers callers;
        callers.ReceiveHeartbeat(5566);
    }

    SECTION("on existing connection id")
    {
        DcgmDiagCallers callers;
        callers.SetConnectionId(5566);
        auto lastHeartbeatTime = callers.GetLastHeartbeatTime(5566);
        callers.ReceiveHeartbeat(5566);
        REQUIRE(callers.GetLastHeartbeatTime(5566) > lastHeartbeatTime);
    }
}
