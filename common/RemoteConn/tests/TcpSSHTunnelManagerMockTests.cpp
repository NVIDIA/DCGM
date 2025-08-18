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

#define DCGM_SSH_TUNNEL_MANAGER_TEST
#include "MockChildProcess.hpp"
#include "TestHelpers.hpp"

#include <DcgmLogging.h>
#include <SSHTunnelManager.hpp>
#include <catch2/catch_all.hpp>
#include <fmt/format.h>
#include <sys/socket.h>

#include <latch>
#include <unordered_set>

using namespace DcgmNs::Common::RemoteConn::detail;
using namespace DcgmNs::Common::RemoteConn::Mock;

namespace
{

MockReturns mockReturnPortInUse = { .isAlive = false, .stdError = ADDRESS_IN_USE_MSG, .pid = 223 };
MockReturns mockReturnSuccess   = { .isAlive = true, .stdError = CONNECTION_SUCCESS_MSG, .pid = 224 };

class EnvVars
{
public:
    EnvVars() = default;
    int SetEnv(std::string const &key, std::string const &value)
    {
        m_vars.insert(key);
        return setenv(key.c_str(), value.c_str(), 1);
    }
    int UnsetEnv(std::string const &key)
    {
        m_vars.erase(key);
        return unsetenv(key.c_str());
    }
    ~EnvVars()
    {
        for (auto const &key : m_vars)
        {
            unsetenv(key.c_str());
        }
    }

private:
    std::unordered_set<std::string> m_vars;
};

static ChildProcessFactory MockChildProcessFactory = []() {
    return std::make_unique<DcgmNs::Common::RemoteConn::Mock::MockChildProcess>();
};
} //namespace

TEST_CASE("ssh session uses first available default port")
{
    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto expectedLocalPort = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE;
    uint16_t localFwdPort, remotePort = 50000;
    auto mockKey = mgr.GetSSHAddressForwardingString(expectedLocalPort, remotePort);

    MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);
    MockStateCache::ClearAll();
}

TEST_CASE("set ssh path fails when ssh session active")
{
    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto expectedLocalPort = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE;
    uint16_t localFwdPort, remotePort = 50000;
    auto mockKey = mgr.GetSSHAddressForwardingString(expectedLocalPort, remotePort);

    MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);

    auto setPathResult = mgr.SetSshBinaryPath("mock path");
    REQUIRE(!setPathResult);
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);

    setPathResult = mgr.SetSshBinaryPath("mock path");
    REQUIRE(setPathResult);

    MockStateCache::ClearAll();
}

TEST_CASE("ssh session uses port from environment variable")
{
    std::vector<unsigned int> portRange = { 52000, 52001 };
    EnvVars envVars;
    envVars.SetEnv(MIN_PORT_RANGE_ENV, std::to_string(portRange.front()));
    envVars.SetEnv(MAX_PORT_RANGE_ENV, std::to_string(portRange.back()));
    uint16_t localFwdPort, remotePort = 50000, expectedLocalPort = portRange[0];

    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto mockKey = mgr.GetSSHAddressForwardingString(expectedLocalPort, remotePort);
    MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == portRange.front());
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);
    MockStateCache::ClearAll();
}

TEST_CASE("ssh session uses default port when port from environment variable is invalid")
{
    std::vector<unsigned int> portRange = { 52001, 52000 };
    EnvVars envVars;
    envVars.SetEnv(MIN_PORT_RANGE_ENV, std::to_string(portRange.front()));
    envVars.SetEnv(MAX_PORT_RANGE_ENV, std::to_string(portRange.back()));
    auto expectedLocalPortRun1 = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE;
    auto expectedLocalPortRun2 = expectedLocalPortRun1 + 1;
    uint16_t localFwdPort, remotePortRun1 = 50000, remotePortRun2 = 50001;


    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto mockKeyRun1 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun1, remotePortRun1);
    auto mockKeyRun2 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun2, remotePortRun2);
    MockStateCache::SetMockReturns(mockKeyRun1, mockReturnSuccess);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun1);

    MockStateCache::SetMockReturns(mockKeyRun2, mockReturnSuccess);
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun2, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun2);
    MockStateCache::ClearAll();
}

TEST_CASE("Session to same host with same port and same uid returns existing ssh session")
{
    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto expectedLocalPortRun1 = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE;
    auto expectedLocalPortRun2 = expectedLocalPortRun1 + 1;
    uint16_t localFwdPort, remotePortRun1 = 50000, remotePortRun2 = 50001;
    auto mockKeyRun1 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun1, remotePortRun1);
    auto mockKeyRun2 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun2, remotePortRun2);

    MockStateCache::SetMockReturns(mockKeyRun1, mockReturnSuccess);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun1);

    // Verify that the same local port is returned when attempting to start
    // a duplicate session
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun1);
    mgr.EndSession("127.0.0.1", remotePortRun1, std::nullopt, true);

    // Verify that different sessions to the same host with different remote
    // ports can be created
    MockStateCache::SetMockReturns(mockKeyRun2, mockReturnSuccess);
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun2, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun2);
    mgr.EndSession("127.0.0.1", remotePortRun2, std::nullopt, true);
    MockStateCache::ClearAll();
}

class TestTcpSSHTunnelManager : public TcpSSHTunnelManager
{
public:
    std::string GetUsernameForUid(uid_t /* uid */) const override
    {
        return "testuser";
    }
};
TEST_CASE("Session to same host with same port and different uid returns new ssh session")
{
    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TestTcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto expectedLocalPortRun1 = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE;
    auto expectedLocalPortRun2 = expectedLocalPortRun1 + 1;
    auto expectedLocalPortRun3 = expectedLocalPortRun1 + 2;
    uint16_t localFwdPort, remotePortRun1 = 50000;
    auto mockKeyRun1 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun1, remotePortRun1);
    auto mockKeyRun2 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun2, remotePortRun1);
    auto mockKeyRun3 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun3, remotePortRun1);
    uid_t uid1 = 1001, uid2 = 1002;
    MockStateCache::SetMockReturns(mockKeyRun1, mockReturnSuccess);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort, uid1);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun1);

    MockStateCache::SetMockReturns(mockKeyRun2, mockReturnSuccess);
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort, uid2);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun2);
    mgr.EndSession("127.0.0.1", remotePortRun1, uid1, true);

    // Verify that endsession removed the session for uid1
    MockStateCache::SetMockReturns(mockKeyRun3, mockReturnSuccess);
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort, uid1);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun3);

    mgr.EndSession("127.0.0.1", remotePortRun1, uid1, true);
    mgr.EndSession("127.0.0.1", remotePortRun1, uid2, true);
    MockStateCache::ClearAll();
}

TEST_CASE("multithreaded start session to same host, same port; all threads get same port")
{
    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto expectedLocalPortRun1 = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE;
    constexpr int numThreads   = 10;
    uint16_t remotePortRun1    = 50000;
    std::atomic<unsigned int> activeStateCount = 0;
    std::atomic<unsigned int> samePortCount    = 0;
    std::atomic<uint16_t> firstPort { 0 };
    std::vector<std::thread> threads;
    std::latch start_latch(numThreads + 1);

    // Set up mock returns for ssh sessions with local ports for all threads,
    // each of which may try to start a session with a new port due to the port
    // rollover mechanism
    for (int i = 0; i < numThreads; ++i)
    {
        auto mockKeyRun = mgr.GetSSHAddressForwardingString(expectedLocalPortRun1 + i, remotePortRun1);
        MockStateCache::SetMockReturns(mockKeyRun, mockReturnSuccess);
    }
    for (int i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([&, i]() {
            uint16_t localFwdPort;
            start_latch.arrive_and_wait();
            TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort);
            if (tunnelState == TunnelState::Active)
            {
                activeStateCount++;
                // Store first port we see, or compare against it
                uint16_t expected = 0;
                if (firstPort.compare_exchange_strong(expected, localFwdPort))
                {
                    // We're the first to store the port
                    samePortCount++;
                }
                else if (localFwdPort == firstPort)
                {
                    // We got the same port as the first thread
                    samePortCount++;
                }
            }
        });
    }

    start_latch.arrive_and_wait();
    for (auto &thread : threads)
    {
        thread.join();
    }

    CHECK(activeStateCount == numThreads);
    CHECK(samePortCount == numThreads);

    MockStateCache::ClearAll();
}

TEST_CASE("ssh port in use, uses next available port")
{
    uint16_t localFwdPort;
    constexpr int expectedLocalPortRun2 = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE + 1,
                  expectedLocalPortRun3 = expectedLocalPortRun2 + 1, remotePortRun1 = 50000;

    // Ensure the start port is mocked as in use, and the second port is
    // successful

    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto mockKeyRun1 = mgr.GetSSHAddressForwardingString(
        DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE, remotePortRun1);
    auto mockKeyRun2 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun2, remotePortRun1);
    auto mockKeyRun3 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun3, remotePortRun1);
    MockStateCache::SetMockReturns(mockKeyRun1, mockReturnPortInUse);
    MockStateCache::SetMockReturns(mockKeyRun2, mockReturnSuccess);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun2);
    mgr.EndSession("127.0.0.1", remotePortRun1, std::nullopt, true);

    // Even when the ssh session is stopped, verify that the local port assignation
    // is incremental
    MockStateCache::SetMockReturns(mockKeyRun3, mockReturnSuccess);
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun3);
    mgr.EndSession("127.0.0.1", remotePortRun1, std::nullopt, true);
    MockStateCache::ClearAll();
}

TEST_CASE("ssh session error, no port available")
{
    EnvVars envVars;
    uint16_t localFwdPort, remotePort = 50000;
    std::vector<unsigned int> existingPorts = { 52000, 52001 };
    envVars.SetEnv(MIN_PORT_RANGE_ENV, std::to_string(existingPorts.front()));
    envVars.SetEnv(MAX_PORT_RANGE_ENV, std::to_string(existingPorts.back()));

    // Verify that an ssh session can no longer be started because no
    // ports are available

    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto mockKeyRun1 = mgr.GetSSHAddressForwardingString(existingPorts[0], remotePort);
    auto mockKeyRun2 = mgr.GetSSHAddressForwardingString(existingPorts[1], remotePort);
    // Ensure both the ports are mocked as in use
    MockStateCache::SetMockReturns(mockKeyRun1, mockReturnPortInUse);
    MockStateCache::SetMockReturns(mockKeyRun2, mockReturnPortInUse);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::AddressInUse);

    MockStateCache::ClearAll();
}

TEST_CASE("ssh port rollover, finds next available port")
{
    EnvVars envVars;
    uint16_t localFwdPort, remotePort = 50000;
    std::vector<unsigned int> portRange = { 52000, 52001, 52002 };
    envVars.SetEnv(MIN_PORT_RANGE_ENV, std::to_string(portRange.front()));
    envVars.SetEnv(MAX_PORT_RANGE_ENV, std::to_string(portRange.back()));
    // Set up the tunnel manager with a range of three ports, and ensure the first
    // two ports are mocked as in use

    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto mockKeyRun1 = mgr.GetSSHAddressForwardingString(portRange[0], remotePort);
    auto mockKeyRun2 = mgr.GetSSHAddressForwardingString(portRange[1], remotePort);
    auto mockKeyRun3 = mgr.GetSSHAddressForwardingString(portRange[2], remotePort);
    MockStateCache::SetMockReturns(mockKeyRun1, mockReturnPortInUse);
    MockStateCache::SetMockReturns(mockKeyRun2, mockReturnPortInUse);
    MockStateCache::SetMockReturns(mockKeyRun3, mockReturnSuccess);
    // Verify that an ssh session can be started on the third port, the only available
    // port
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == portRange[2]);

    // Verify that no more ssh sessions can be started once all the ports are used
    // up
    auto mockKeyRun4 = mgr.GetSSHAddressForwardingString(portRange[0], remotePort + 1);
    auto mockKeyRun5 = mgr.GetSSHAddressForwardingString(portRange[1], remotePort + 1);
    auto mockKeyRun6 = mgr.GetSSHAddressForwardingString(portRange[2], remotePort + 1);
    MockStateCache::SetMockReturns(mockKeyRun4, mockReturnPortInUse);
    MockStateCache::SetMockReturns(mockKeyRun5, mockReturnPortInUse);
    MockStateCache::SetMockReturns(mockKeyRun6, mockReturnPortInUse);
    tunnelState = mgr.StartSession("127.0.0.1", remotePort + 1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::AddressInUse);

    // Stop one of the TCP servers, and verify that the freed port is assigned to
    // the ssh session
    auto mockKeyRun7 = mgr.GetSSHAddressForwardingString(portRange[1], remotePort + 1);
    MockStateCache::SetMockReturns(mockKeyRun7, mockReturnSuccess);
    tunnelState = mgr.StartSession("127.0.0.1", remotePort + 1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == portRange[1]);

    // Verify that no more ssh sessions can be started once all the ports are used
    // up again
    auto mockKeyRun8  = mgr.GetSSHAddressForwardingString(portRange[0], remotePort + 2);
    auto mockKeyRun9  = mgr.GetSSHAddressForwardingString(portRange[1], remotePort + 2);
    auto mockKeyRun10 = mgr.GetSSHAddressForwardingString(portRange[2], remotePort + 2);
    MockStateCache::SetMockReturns(mockKeyRun8, mockReturnPortInUse);
    MockStateCache::SetMockReturns(mockKeyRun9, mockReturnPortInUse);
    MockStateCache::SetMockReturns(mockKeyRun10, mockReturnPortInUse);
    tunnelState = mgr.StartSession("127.0.0.1", remotePort + 2, localFwdPort);
    REQUIRE(tunnelState == TunnelState::AddressInUse);

    MockStateCache::ClearAll();
}

TEST_CASE("EndSession removes session only when session refcount is 1")
{
    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto expectedLocalPortRun1 = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE;
    uint16_t localFwdPort, remotePortRun = 50000;
    auto mockKeyRun1 = mgr.GetSSHAddressForwardingString(expectedLocalPortRun1, remotePortRun);

    MockStateCache::SetMockReturns(mockKeyRun1, mockReturnSuccess);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePortRun, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun1);

    // Create a duplicate session to the same host and port, and end one session
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);
    mgr.EndSession("127.0.0.1", remotePortRun);

    // Verify that the session is not removed because the refcount is greater than 1
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);

    // Verify that the session is removed when the refcount is 1
    mgr.EndSession("127.0.0.1", remotePortRun);
    mgr.EndSession("127.0.0.1", remotePortRun);

    auto expectedLocalPortRun2 = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE + 1;
    auto mockKeyRun2           = mgr.GetSSHAddressForwardingString(expectedLocalPortRun2, remotePortRun);
    MockStateCache::SetMockReturns(mockKeyRun2, mockReturnSuccess);
    tunnelState = mgr.StartSession("127.0.0.1", remotePortRun, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == expectedLocalPortRun2);

    MockStateCache::ClearAll();
}

TEST_CASE("multithreaded sessions to same host, same port; \
StartSession of all threads gets same port; \
EndSession removes session only when refcount is 1")
{
    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    auto expectedLocalPortRun1 = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE;
    constexpr int numThreads   = 5;
    uint16_t remotePortRun1    = 50000;
    std::atomic<unsigned int> activeStateCount  = 0;
    std::atomic<unsigned int> samePortCount     = 0;
    std::atomic<unsigned int> numActiveSessions = 0;
    std::atomic<uint16_t> firstPort { 0 };
    std::vector<std::thread> threads;
    std::latch start_latch(numThreads + 1);

    // Set up mock returns for ssh sessions with local ports for all threads,
    // each of which may try to start a session with a new port due to the port
    // rollover mechanism
    for (int i = 0; i <= numThreads; ++i)
    {
        auto mockKeyRun = mgr.GetSSHAddressForwardingString(expectedLocalPortRun1 + i, remotePortRun1);
        MockStateCache::SetMockReturns(mockKeyRun, mockReturnSuccess);
    }
    for (int i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([&, i]() {
            uint16_t localFwdPort;
            start_latch.arrive_and_wait();
            TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort);
            if (tunnelState == TunnelState::Active)
            {
                activeStateCount++;
                // Store first port we see, or compare against it
                uint16_t expected = 0;
                if (firstPort.compare_exchange_strong(expected, localFwdPort))
                {
                    // We're the first to store the port
                    samePortCount++;
                }
                else if (localFwdPort == firstPort)
                {
                    // We got the same port as the first thread
                    samePortCount++;
                }
                numActiveSessions++;
            }
            // End a session every other thread once we have 2 active sessions
            if (numActiveSessions > 2 && i % 2 == 0)
            {
                mgr.EndSession("127.0.0.1", remotePortRun1);
                numActiveSessions--;
            }
        });
    }

    start_latch.arrive_and_wait();
    for (auto &thread : threads)
    {
        thread.join();
    }

    // Verify that all threads returned an active session and got the same port
    REQUIRE(activeStateCount == numThreads);
    REQUIRE(samePortCount == numThreads);

    // Verify that EndSession removes a session only when the refcount has
    // dropped to 1 by starting a new session and checking that the port is
    // different from the port returned by all the threads.
    for (int i = numActiveSessions; i > 0; i--)
    {
        mgr.EndSession("127.0.0.1", remotePortRun1);
    }
    uint16_t localFwdPort;
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePortRun1, localFwdPort);
    CHECK(tunnelState == TunnelState::Active);
    CHECK(localFwdPort != firstPort);

    MockStateCache::ClearAll();
}