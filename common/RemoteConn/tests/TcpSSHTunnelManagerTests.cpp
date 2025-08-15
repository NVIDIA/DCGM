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

#include "TestHelpers.hpp"

#include <DcgmChildProcessManager.hpp>
#include <DcgmLogging.h>
#include <SSHTunnelManager.hpp>
#include <catch2/catch_all.hpp>
#include <sys/socket.h>

#include <unordered_set>

using namespace DcgmNs::Common::RemoteConn::detail;
using ChildProcess = DcgmNs::Common::Subprocess::ChildProcess;

namespace
{
std::string portRangeMinEnv = "__DCGM_SSH_PORT_RANGE_MIN__";
std::string portRangeMaxEnv = "__DCGM_SSH_PORT_RANGE_MAX__";

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
} //namespace

TEST_CASE("OpenSSH ssh session created with singleton uses first available default port")
{
    DcgmLoggingInit("test_log.txt", DcgmLoggingSeverityVerbose, DcgmLoggingSeverityVerbose);
    RouteLogToBaseLogger(SYSLOG_LOGGER);
    log_debug("Test 1");

    auto &mgr = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::GetInstance();
    DcgmChildProcessManager childProcessManager;
    SetChildProcessFuncs(mgr, childProcessManager);
    uint16_t localFwdPort, remotePort = 50000;
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);
}

TEST_CASE("OpenSSH ssh session uses first available default port")
{
    DcgmChildProcessManager childProcessManager;
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    uint16_t localFwdPort, remotePort = 50000;
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);
}

TEST_CASE("OpenSSH ssh session uses port from environment variable")
{
    std::vector<unsigned int> portRange = { 52000, 52001 };
    EnvVars envVars;
    envVars.SetEnv(MIN_PORT_RANGE_ENV, std::to_string(portRange.front()));
    envVars.SetEnv(MAX_PORT_RANGE_ENV, std::to_string(portRange.back()));
    TcpSSHTunnelManager mgr;
    DcgmChildProcessManager childProcessManager;
    SetChildProcessFuncs(mgr, childProcessManager);
    uint16_t localFwdPort, remotePort = 50000;
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == portRange.front());
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);
}

TEST_CASE("OpenSSH ssh session uses default port when port from environment variable is invalid")
{
    log_debug("Test 4");
    std::vector<unsigned int> portRange = { 52001, 52000 };
    EnvVars envVars;
    envVars.SetEnv(MIN_PORT_RANGE_ENV, std::to_string(portRange.front()));
    envVars.SetEnv(MAX_PORT_RANGE_ENV, std::to_string(portRange.back()));
    TcpSSHTunnelManager mgr;
    DcgmChildProcessManager childProcessManager;
    SetChildProcessFuncs(mgr, childProcessManager);
    uint16_t localFwdPort, remotePort = 50000;
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);

    tunnelState = mgr.StartSession("127.0.0.1", remotePort + 1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE + 1);
}

TEST_CASE("set ssh path passes only when OpenSSH ssh sessions are not active")
{
    TcpSSHTunnelManager mgr;
    DcgmChildProcessManager childProcessManager;
    SetChildProcessFuncs(mgr, childProcessManager);
    uint16_t localFwdPort, remotePort = 50000;

    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    auto setPathResult = mgr.SetSshBinaryPath("mock path");
    REQUIRE(!setPathResult);

    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);
    setPathResult = mgr.SetSshBinaryPath("mock path");
    CHECK(setPathResult);
}

TEST_CASE("returns existing OpenSSH ssh session, if any")
{
    TcpSSHTunnelManager mgr;
    DcgmChildProcessManager childProcessManager;
    SetChildProcessFuncs(mgr, childProcessManager);
    uint16_t localFwdPort, remotePort = 50000;
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);

    // Verify that the same local port is returned when attempting to start
    // a duplicate session
    tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE);
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);

    // Verify that different sessions to the same host with different remote
    // ports can be created
    tunnelState = mgr.StartSession("127.0.0.1", remotePort + 1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE + 1);
    mgr.EndSession("127.0.0.1", remotePort + 1, std::nullopt, true);
}

TEST_CASE("OpenSSH ssh port in use, uses next available port")
{
    constexpr int localPort  = DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::DEFAULT_START_PORT_RANGE,
                  remotePort = 50000;
    // Start a TCP server on the first port of the default ssh port range
    DcgmChildProcessManager childProcessManager;
    std::vector<ChildProcessHandle_t> tcpServers;
    StartTcpServers(childProcessManager, { localPort }, tcpServers);

    // Verify that the next available port is used for the ssh session
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    uint16_t localFwdPort;
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == localPort + 1);
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);

    // Even when the ssh session is stopped, verify that the local port assignation
    // is incremental
    tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == localPort + 2);
    mgr.EndSession("127.0.0.1", remotePort, std::nullopt, true);

    for (auto tcpServer : tcpServers)
    {
        childProcessManager.Destroy(tcpServer, 1);
    }
}

TEST_CASE("OpenSSH ssh session error, no port available")
{
    std::vector<ChildProcessHandle_t> tcpServers;
    DcgmChildProcessManager childProcessManager;
    EnvVars envVars;
    std::vector<unsigned int> existingPorts = { 52000, 52001 };
    envVars.SetEnv(MIN_PORT_RANGE_ENV, std::to_string(existingPorts.front()));
    envVars.SetEnv(MAX_PORT_RANGE_ENV, std::to_string(existingPorts.back()));

    // Set up the tunnel manager with a range of two ports, and start
    // TCP servers on both ports
    StartTcpServers(childProcessManager, existingPorts, tcpServers);

    // Verify that an ssh session can no longer be started because no
    // ports are available
    uint16_t localFwdPort, remotePort = 50000;
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::AddressInUse);

    for (auto tcpServer : tcpServers)
    {
        childProcessManager.Destroy(tcpServer, 1);
    }
}

TEST_CASE("OpenSSH ssh port rollover, finds next available port")
{
    std::vector<ChildProcessHandle_t> tcpServers;
    DcgmChildProcessManager childProcessManager;
    EnvVars envVars;
    std::vector<unsigned int> existingPorts = { 52000, 52001 };
    envVars.SetEnv(MIN_PORT_RANGE_ENV, std::to_string(existingPorts.front()));
    envVars.SetEnv(MAX_PORT_RANGE_ENV, std::to_string(existingPorts.back() + 1));

    // Set up the tunnel manager with a range of three ports, and start
    // TCP servers on the first two ports
    StartTcpServers(childProcessManager, existingPorts, tcpServers);

    // Verify that an ssh session can be started on the third port, the only available
    // port
    uint16_t localFwdPort, remotePort = 50000;
    TcpSSHTunnelManager mgr;
    SetChildProcessFuncs(mgr, childProcessManager);
    TunnelState tunnelState = mgr.StartSession("127.0.0.1", remotePort, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == existingPorts.back() + 1);

    // Verify that no more ssh sessions can be started once all the ports are used
    // up
    tunnelState = mgr.StartSession("127.0.0.1", remotePort + 1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::AddressInUse);
    CHECK(tunnelState == TunnelState::AddressInUse);

    // Stop one of the TCP servers, and verify that the freed port is assigned to
    // the ssh session
    childProcessManager.Destroy(tcpServers[1], 1);
    tunnelState = mgr.StartSession("127.0.0.1", remotePort + 1, localFwdPort);
    REQUIRE(tunnelState == TunnelState::Active);
    CHECK(localFwdPort == existingPorts.back());

    // Verify that no more ssh sessions can be started once all the ports are used
    // up again
    tunnelState = mgr.StartSession("127.0.0.1", remotePort + 2, localFwdPort);
    REQUIRE(tunnelState == TunnelState::AddressInUse);

    for (auto tcpServer : tcpServers)
    {
        childProcessManager.Destroy(tcpServer, 1);
    }
}
