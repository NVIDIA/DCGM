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
#include <iostream>

using namespace DcgmNs::Common::RemoteConn::detail;
using ChildProcess = DcgmNs::Common::Subprocess::ChildProcess;

// make sure we're in the same namespace as the base class to test private members.
// We have declared the test class as a friend class in the base class and that works within the same namespace.
namespace DcgmNs::Common::RemoteConn::detail
{

class TestUdsSSHTunnelManager : public UdsSSHTunnelManager
{
private:
    bool m_isRunningAsRoot = true;
    bool m_isRootUser      = true;

public:
    std::string_view GetPrimaryPath() const override
    {
        return "./run";
    }
    std::string_view GetSecondaryPath() const override
    {
        return "./tmp";
    }
    bool IsRunningAsRoot() const override
    {
        return m_isRunningAsRoot;
    }
    void SetRunningAsRoot(bool value)
    {
        m_isRunningAsRoot = value;
    }
    bool IsRootUser(uid_t /* uid */) const override
    {
        return m_isRootUser;
    }
    void SetIsRootUser(bool value)
    {
        m_isRootUser = value;
    }

    using UdsSSHTunnelManager::GetSSHAddressForwardingString;
    using UdsSSHTunnelManager::IsRootUser;
    using UdsSSHTunnelManager::ROOT_UID;

    TestUdsSSHTunnelManager()
    {
        try
        {
            boost::filesystem::create_directories(GetPrimaryPath());
            boost::filesystem::create_directories(GetSecondaryPath());
        }
        catch (const boost::filesystem::filesystem_error &e)
        {
            std::cerr << "Failed to create test directories: " << e.what() << '\n';
            throw;
        }
    }

    ~TestUdsSSHTunnelManager()
    {
        try
        {
            boost::filesystem::remove_all(GetPrimaryPath());
            boost::filesystem::remove_all(GetSecondaryPath());
        }
        catch (const boost::filesystem::filesystem_error &e)
        {
            std::cerr << "Failed to cleanup test directories: " << e.what() << '\n';
        }
    }
};

} // namespace DcgmNs::Common::RemoteConn::detail

TEST_CASE("UdsSSHTunnelManager tests")
{
    // Common setup for all sections
    TestUdsSSHTunnelManager manager;
    DcgmChildProcessManager childProcessManager;
    SetChildProcessFuncs(manager, childProcessManager);
    constexpr uid_t ROOT_UID = TestUdsSSHTunnelManager::ROOT_UID;
    auto uid                 = geteuid();
    (uid == ROOT_UID) ? manager.SetIsRootUser(true) : manager.SetIsRootUser(false);
    std::string remoteHostname = "127.0.0.1";
    auto localUnixPathPrefix   = manager.IsRootUser(uid) ? "./run/dcgm" : fmt::format("./tmp/dcgm_{}", uid);
    std::string localUnixPath, remoteUnixPath = "/tmp/testRemoteUnixPath.sock";
    std::string anotherRemoteUnixPath = "/tmp/testAnotherRemoteUnixPath.sock";
    std::string existingLocalUnixPath, returnedLocalUnixPath;
    auto &instance = DcgmNs::Common::RemoteConn::UdsSSHTunnelManager::GetInstance();

    SECTION("Singleton tests")
    {
        auto &anotherInstance = DcgmNs::Common::RemoteConn::UdsSSHTunnelManager::GetInstance();
        REQUIRE(&instance == &anotherInstance);
    }

    SECTION("Start and stop tunnel for single instance")
    {
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);
        // Verify that the local unix path is as expected
        REQUIRE(localUnixPath.find(fmt::format("{}/ssh_", localUnixPathPrefix)) != std::string::npos);
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
    }

    SECTION("Returns existing session if any")
    {
        // Start a session
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, existingLocalUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);
        // Start another session with the same remote host and remote unix path
        auto tunnelState2 = manager.StartSession(remoteHostname, remoteUnixPath, returnedLocalUnixPath, uid);
        REQUIRE(tunnelState2 == TunnelState::Active);
        // Verify that the existing local unix path is returned
        REQUIRE(existingLocalUnixPath == returnedLocalUnixPath);
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
    }

    SECTION("Different sessions to same host")
    {
        // Start a session to remote host with remote unix path
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);
        // Start another session with the same remote host and different remote unix path
        auto tunnelState2 = manager.StartSession(remoteHostname, anotherRemoteUnixPath, localUnixPath, uid);
        // Verify that the second session's tunnel state is active
        REQUIRE(tunnelState2 == TunnelState::Active);
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        manager.EndSession(remoteHostname, anotherRemoteUnixPath, uid);
    }

    SECTION("End session removes local unix domain socket file")
    {
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);
        REQUIRE(boost::filesystem::exists(localUnixPath));
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        // Verify that the local unix path file is removed by EndSession.
        REQUIRE(!boost::filesystem::exists(localUnixPath));
    }

    SECTION("End session does not remove local unix domain socket file if forceEnd is false")
    {
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);
        REQUIRE(boost::filesystem::exists(localUnixPath));
        // Create multiple sessions to the same remote host and remote unix path
        manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        // Verify that the local unix path file is removed by EndSession.
        REQUIRE(boost::filesystem::exists(localUnixPath));
    }

    SECTION("End session removes local unix domain socket file if forceEnd is true")
    {
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);
        REQUIRE(boost::filesystem::exists(localUnixPath));
        // Create multiple sessions to the same remote host and remote unix path
        manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        manager.EndSession(remoteHostname, remoteUnixPath, uid, true);
        // Verify that the local unix path file is removed by EndSession.
        REQUIRE(!boost::filesystem::exists(localUnixPath));
    }

    SECTION("Returns different file with next number")
    {
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);

        std::string anotherLocalUnixPath, anotherRemoteUnixPath = "/tmp/testAnotherRemoteUnixPath.sock";
        auto tunnelState2 = manager.StartSession(remoteHostname, anotherRemoteUnixPath, anotherLocalUnixPath, uid);
        REQUIRE(tunnelState2 == TunnelState::Active);

        // Extract number from localUnixPath and anotherLocalUnixPath and compare
        auto localUnixPathNumber = std::stoi(localUnixPath.substr(localUnixPath.find_last_of('_') + 1));
        auto anotherLocalUnixPathNumber
            = std::stoi(anotherLocalUnixPath.substr(anotherLocalUnixPath.find_last_of('_') + 1));
        REQUIRE(localUnixPathNumber + 1 == anotherLocalUnixPathNumber);

        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        manager.EndSession(remoteHostname, anotherRemoteUnixPath, uid);
    }

    SECTION("Uses next number if file is in use")
    {
        // Start a TCP server on the file /run/dcgm/ssh_0.sock
        std::vector<ChildProcessHandle_t> tcpServers;
        auto portStr = fmt::format("{}/ssh_0.sock", localUnixPathPrefix);

        StartUdsServers(childProcessManager, { portStr }, tcpServers);
        // Verify that the next available file number is used for the ssh session
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        CHECK(tunnelState == TunnelState::Active);
        CHECK(localUnixPath == fmt::format("{}/ssh_1.sock", localUnixPathPrefix));
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        childProcessManager.Destroy(tcpServers[0], 1);

        // Even when the ssh session is stopped, verify that the file number is incremented for the same manager
        // instance.
        auto tunnelState2 = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        CHECK(tunnelState2 == TunnelState::Active);
        CHECK(localUnixPath == fmt::format("{}/ssh_2.sock", localUnixPathPrefix));
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
    }

    SECTION("Returns generic failure for unknown hostnames")
    {
        remoteHostname   = "unknown_hostname";
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        // Verify that for unknown hostnames, the tunnel state returned is generic failure.
        REQUIRE(tunnelState == TunnelState::GenericFailure);
    }
}
