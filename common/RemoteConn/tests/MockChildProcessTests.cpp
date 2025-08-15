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

#include "MockChildProcess.hpp"

#include <catch2/catch_all.hpp>

using namespace DcgmNs::Common::RemoteConn::Mock;

TEST_CASE("MockStateCache")
{
    SECTION("Set, Get and Clear mock returns")
    {
        // Verify set and get of a mock return structure
        std::string key1             = "test1";
        MockReturns expectedMockRet1 = { true, key1, 221 };
        MockStateCache::SetMockReturns(key1, expectedMockRet1);
        auto retVal1 = MockStateCache::GetMockReturns(key1);
        REQUIRE(retVal1.has_value());
        CHECK(*retVal1 == expectedMockRet1);

        // Verify set and get of a second mock return structure
        std::string key2             = "test2";
        MockReturns expectedMockRet2 = { false, "abc", 222 };
        MockStateCache::SetMockReturns(key2, expectedMockRet2);
        auto retVal2 = MockStateCache::GetMockReturns(key2);
        REQUIRE(retVal2.has_value());
        CHECK(*retVal2 == expectedMockRet2);

        // Verify set mock return structure of an existing key overwrites
        MockReturns expectedMockRet3 = { true, "dabc", 223 };
        MockStateCache::SetMockReturns(key2, expectedMockRet3);
        retVal2 = MockStateCache::GetMockReturns(key2);
        REQUIRE(retVal2.has_value());
        CHECK(*retVal2 == expectedMockRet3);

        // Verify get returns the first mock return structure correctly
        retVal1 = MockStateCache::GetMockReturns(key1);
        REQUIRE(retVal1.has_value());
        CHECK(*retVal1 == expectedMockRet1);

        MockStateCache::ClearMockReturns(key1);

        // Verify the first mock return structure is cleared
        retVal1 = MockStateCache::GetMockReturns(key1);
        REQUIRE(!retVal1.has_value());

        // Verify the second mock return structure remains untouched
        retVal2 = MockStateCache::GetMockReturns(key2);
        REQUIRE(retVal2.has_value());
        CHECK(*retVal2 == expectedMockRet3);

        MockStateCache::ClearAll();

        // Verify the second mock return structure is also cleared
        retVal2 = MockStateCache::GetMockReturns(key2);
        REQUIRE(!retVal2.has_value());
    }

    SECTION("Set, Get and Clear mock call count")
    {
        // Increment call count of only func1, and verify call count returned
        // for func1 is 1, and func2 is 0.
        std::string key1  = "test1";
        std::string func1 = "func1", func2 = "func2";
        MockStateCache::IncrementMockCallCount(key1, func1);
        auto callCountFunc1 = MockStateCache::GetMockCallCount(key1, func1);
        auto callCountFunc2 = MockStateCache::GetMockCallCount(key1, func2);
        CHECK(callCountFunc1 == 1);
        CHECK(callCountFunc2 == 0);

        // Do the same for a different key
        std::string key2 = "test2";
        MockStateCache::IncrementMockCallCount(key2, func1);
        MockStateCache::IncrementMockCallCount(key2, func1);
        MockStateCache::IncrementMockCallCount(key2, func2);
        callCountFunc1 = MockStateCache::GetMockCallCount(key2, func1);
        callCountFunc2 = MockStateCache::GetMockCallCount(key2, func2);
        CHECK(callCountFunc1 == 2);
        CHECK(callCountFunc2 == 1);

        // Verify first key counts remain unchanged
        callCountFunc1 = MockStateCache::GetMockCallCount(key1, func1);
        callCountFunc2 = MockStateCache::GetMockCallCount(key1, func2);
        CHECK(callCountFunc1 == 1);
        CHECK(callCountFunc2 == 0);

        // Clear first key count and verify only first key counts are cleared
        MockStateCache::ClearMockCallCount(key1);
        callCountFunc1 = MockStateCache::GetMockCallCount(key1, func1);
        callCountFunc2 = MockStateCache::GetMockCallCount(key1, func2);
        CHECK(callCountFunc1 == 0);
        CHECK(callCountFunc2 == 0);
        callCountFunc1 = MockStateCache::GetMockCallCount(key2, func1);
        callCountFunc2 = MockStateCache::GetMockCallCount(key2, func2);
        CHECK(callCountFunc1 == 2);
        CHECK(callCountFunc2 == 1);

        // Clear all and verify all keys are cleared
        MockStateCache::ClearAll();
        callCountFunc1 = MockStateCache::GetMockCallCount(key2, func1);
        callCountFunc2 = MockStateCache::GetMockCallCount(key2, func2);
        CHECK(callCountFunc1 == 0);
        CHECK(callCountFunc2 == 0);
    }


    SECTION("Set, Get and Clear mock call wait times")
    {
        // Increment call count of only func1, and verify call count returned
        // for func1 is 1, and func2 is 0.
        std::string key1  = "test1";
        std::string func1 = "func1", func2 = "func2";
        std::chrono::milliseconds waitTime1 { 10 }, zeroTime { 0 };
        MockStateCache::SetMockCallWaitTimes(key1, func1, waitTime1);
        auto callCountWaitTime1 = MockStateCache::GetMockCallWaitTimes(key1, func1);
        auto callCountWaitTime2 = MockStateCache::GetMockCallWaitTimes(key1, func2);
        CHECK(callCountWaitTime1 == waitTime1);
        CHECK(callCountWaitTime2 == zeroTime);

        // Do the same for a different key
        std::string key2 = "test2";
        MockStateCache::SetMockCallWaitTimes(key2, func1, waitTime1);
        MockStateCache::SetMockCallWaitTimes(key2, func2, waitTime1);
        callCountWaitTime1 = MockStateCache::GetMockCallWaitTimes(key2, func1);
        callCountWaitTime2 = MockStateCache::GetMockCallWaitTimes(key2, func2);
        CHECK(callCountWaitTime1 == waitTime1);
        CHECK(callCountWaitTime2 == waitTime1);

        // Verify first key counts remain unchanged
        callCountWaitTime1 = MockStateCache::GetMockCallWaitTimes(key1, func1);
        callCountWaitTime2 = MockStateCache::GetMockCallWaitTimes(key1, func2);
        CHECK(callCountWaitTime1 == waitTime1);
        CHECK(callCountWaitTime2 == zeroTime);

        // Clear first key count and verify only first key counts are cleared
        MockStateCache::ClearMockCallWaitTimes(key1);
        callCountWaitTime1 = MockStateCache::GetMockCallWaitTimes(key1, func1);
        callCountWaitTime2 = MockStateCache::GetMockCallWaitTimes(key1, func2);
        CHECK(callCountWaitTime1 == zeroTime);
        CHECK(callCountWaitTime2 == zeroTime);
        callCountWaitTime1 = MockStateCache::GetMockCallWaitTimes(key2, func1);
        callCountWaitTime2 = MockStateCache::GetMockCallWaitTimes(key2, func2);
        CHECK(callCountWaitTime1 == waitTime1);
        CHECK(callCountWaitTime2 == waitTime1);

        // Clear all and verify all keys are cleared
        MockStateCache::ClearAll();
        callCountWaitTime1 = MockStateCache::GetMockCallWaitTimes(key2, func1);
        callCountWaitTime2 = MockStateCache::GetMockCallWaitTimes(key2, func2);
        CHECK(callCountWaitTime1 == zeroTime);
        CHECK(callCountWaitTime2 == zeroTime);
    }
    MockStateCache::ClearAll();
}

TEST_CASE("MockChildProcess")
{
    SECTION("Create")
    {
        {
            MockChildProcess proc;
            IoContext ioContext;
            REQUIRE(proc.GetAddressFwdSessionKey().empty());

            std::string key = "127.0.0.1:54545:127.0.0.1:32", loopback = "127.0.0.1";
            std::vector<std::string> args = { "-o", "ExitOnForwardFailure=yes", "-v", "-N", "-L", key, loopback };
            proc.Create(ioContext, "", args);
            CHECK(proc.GetAddressFwdSessionKey() == key);

            auto createCallCount = MockStateCache::GetMockCallCount(key, "Create");
            CHECK(createCallCount == 1);
        }
        {
            MockChildProcess proc;
            IoContext ioContext;
            std::vector<std::string> args = { "-L" };
            REQUIRE_THROWS(proc.Create(ioContext, "", args));
            CHECK(proc.GetAddressFwdSessionKey().empty());
        }
        {
            MockChildProcess proc;
            IoContext ioContext;
            std::string key                                  = "127.0.0.1:54545:127.0.0.1:32";
            std::vector<std::string> args                    = { "-L", key };
            std::string username                             = "testuser";
            std::unordered_map<std::string, std::string> env = { { "USER", username } };
            int channelFd                                    = 1;
            proc.Create(ioContext, "", args, env, username, channelFd);
            auto funcArgs = MockStateCache::GetFuncArgs(key, "Create");
            REQUIRE(funcArgs.size() == 5);
            auto envArg = std::any_cast<std::optional<std::unordered_map<std::string, std::string>>>(funcArgs[2]);
            REQUIRE(envArg.has_value());
            CHECK(envArg->at("USER") == username);
            auto usernameArg = std::any_cast<std::optional<std::string>>(funcArgs[3]);
            REQUIRE(usernameArg.has_value());
            CHECK(*usernameArg == username);
            auto channelFdArg = std::any_cast<std::optional<int>>(funcArgs[4]);
            REQUIRE(channelFdArg.has_value());
            CHECK(*channelFdArg == channelFd);

            MockStateCache::ClearFuncArgs(key);
            auto funcArgs2 = MockStateCache::GetFuncArgs(key, "Create");
            REQUIRE(funcArgs2.size() == 0);
        }
    }

    SECTION("Stop")
    {
        MockChildProcess proc;
        IoContext ioContext;
        std::string key = "mock";
        std::chrono::milliseconds waitTime { 20 };
        MockStateCache::SetMockCallWaitTimes(key, "Stop", waitTime);
        std::vector<std::string> args = { "-L", key };
        proc.Create(ioContext, "", args);
        proc.Run();
        auto startTime = std::chrono::high_resolution_clock::now();
        proc.Stop();
        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        CHECK(duration >= waitTime);

        auto stopCallCount = MockStateCache::GetMockCallCount(key, "Stop");
        CHECK(stopCallCount == 1);
    }

    SECTION("GetStdError")
    {
        MockChildProcess proc;
        IoContext ioContext;
        std::string key = "mock", expectedErrorString = "sample error txt";
        std::chrono::milliseconds waitTime { 50 };
        fmt::memory_buffer errorString;
        MockReturns mockReturn        = { false, expectedErrorString, 225 };
        std::vector<std::string> args = { "-L", key };
        proc.Create(ioContext, "", args);
        proc.Run();
        MockStateCache::SetMockReturns(key, mockReturn);
        MockStateCache::SetMockCallWaitTimes(key, "GetStdErrBuffer", waitTime);
        auto startTime = std::chrono::high_resolution_clock::now();
        proc.GetStdErrBuffer(errorString, false);
        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        CHECK(duration >= waitTime);

        CHECK(fmt::to_string(errorString) == expectedErrorString);

        auto getStdErrorCallCount = MockStateCache::GetMockCallCount(key, "GetStdErrBuffer");
        CHECK(getStdErrorCallCount == 1);
    }

    SECTION("IsAlive")
    {
        MockChildProcess proc;
        IoContext ioContext;
        std::string key      = "mock";
        bool expectedIsAlive = true;
        std::chrono::milliseconds waitTime { 50 };
        MockReturns mockReturn        = { expectedIsAlive, "", 226 };
        std::vector<std::string> args = { "-L", key };
        proc.Create(ioContext, "", args);
        proc.Run();
        MockStateCache::SetMockReturns(key, mockReturn);
        MockStateCache::SetMockCallWaitTimes(key, "IsAlive", waitTime);
        auto startTime = std::chrono::high_resolution_clock::now();
        bool isAlive   = proc.IsAlive();
        auto endTime   = std::chrono::high_resolution_clock::now();
        auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        CHECK(duration >= waitTime);

        CHECK(isAlive == expectedIsAlive);

        auto isAliveCallCount = MockStateCache::GetMockCallCount(key, "IsAlive");
        CHECK(isAliveCallCount == 1);
    }

    SECTION("GetPid")
    {
        MockChildProcess proc;
        IoContext ioContext;
        std::string key               = "mock";
        pid_t expectedPid             = 224;
        MockReturns mockReturn        = { true, "", expectedPid };
        std::vector<std::string> args = { "-L", key };
        proc.Create(ioContext, "", args);
        proc.Run();
        MockStateCache::SetMockReturns(key, mockReturn);
        auto pid = proc.GetPid();
        CHECK(pid.has_value());
        CHECK(pid.value() == expectedPid);
    }
}
