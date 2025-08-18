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
#include "TestHelpers.hpp"

#include <DcgmLogging.h>
#include <SSHTunnelManager.hpp>

#include <catch2/catch_all.hpp>
#include <iostream>

using namespace DcgmNs::Common::RemoteConn::detail;
using namespace DcgmNs::Common::RemoteConn::Mock;

MockReturns mockReturnAddressInUse = { .isAlive = false, .stdError = ADDRESS_IN_USE_MSG, .pid = 223 };
MockReturns mockReturnSuccess      = { .isAlive = true, .stdError = CONNECTION_SUCCESS_MSG, .pid = 224 };

static ChildProcessFactory MockChildProcessFactory = []() {
    return std::make_unique<DcgmNs::Common::RemoteConn::Mock::MockChildProcess>();
};
// make sure we're in the same namespace as the base class to test private members.
// We have declared the test class as a friend class in the base class and that works within the same namespace.
namespace DcgmNs::Common::RemoteConn::detail
{

class TestUdsSSHTunnelManager : public UdsSSHTunnelManager
{
private:
    bool m_isRunningAsRoot = true;
    bool m_isRootUser      = true;
    std::string m_currentUsername;
    bool m_overrideVerifyPathOwnershipAndPermissions = false;

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

    void SetOverrideVerifyPathOwnershipAndPermissions(bool value)
    {
        m_overrideVerifyPathOwnershipAndPermissions = value;
    }

    void SetCurrentUid(uid_t uid)
    {
        m_currentUid = uid;
    }

    void SetCurrentUsername(std::string const &username)
    {
        m_currentUsername = username;
    }

    std::string GetUsernameForUid(uid_t /* uid */) const override
    {
        return m_currentUsername;
    }

    std::expected<void, std::string> VerifyPathOwnershipAndPermissions(std::string_view path,
                                                                       uid_t expectedOwner,
                                                                       mode_t expectedPerms) const override
    {
        if (m_overrideVerifyPathOwnershipAndPermissions)
        {
            return {};
        }
        return UdsSSHTunnelManager::VerifyPathOwnershipAndPermissions(path, expectedOwner, expectedPerms);
    }

    using UdsSSHTunnelManager::GetSSHAddressForwardingString;
    using UdsSSHTunnelManager::IsRootUser;
    using UdsSSHTunnelManager::ROOT_UID;

    TestUdsSSHTunnelManager()
    {
        REQUIRE_NOTHROW(boost::filesystem::create_directories(GetPrimaryPath()));
        REQUIRE_NOTHROW(boost::filesystem::create_directories(GetSecondaryPath()));
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
            // Suppress cleanup errors.
        }
    }
};

} // namespace DcgmNs::Common::RemoteConn::detail


TEST_CASE("UdsSSHTunnelManager tests using mock child process.")
{
    // Common setup
    DcgmChildProcessManager childProcessManager(MockChildProcessFactory);
    TestUdsSSHTunnelManager manager;
    SetChildProcessFuncs(manager, childProcessManager);
    auto uid                 = geteuid();
    constexpr uid_t ROOT_UID = TestUdsSSHTunnelManager::ROOT_UID;
    manager.SetRunningAsRoot(uid == ROOT_UID);
    std::string remoteHostname = "127.0.0.1";
    // The MR pipeline runs the tests as root, while the local build runs the tests as non-root.
    // Accomodate both cases.
    auto localUnixPathPrefix   = manager.IsRunningAsRoot() ? "./run/dcgm" : fmt::format("./tmp/dcgm_{}", uid);
    auto expectedLocalUnixPath = fmt::format("{}/ssh_0.sock", localUnixPathPrefix);
    std::string localUnixPath, remoteUnixPath               = "/tmp/testRemoteUnixPath.sock";
    std::string anotherLocalUnixPath, anotherRemoteUnixPath = "/tmp/testAnotherRemoteUnixPath.sock";
    auto mockKey = manager.GetSSHAddressForwardingString(expectedLocalUnixPath, remoteUnixPath);
    MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);

    SECTION("Basic start and stop test")
    {
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        // Verify that the tunnel state is active and the local unix path is as expected.
        REQUIRE(tunnelState == TunnelState::Active);
        REQUIRE(localUnixPath == expectedLocalUnixPath);
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
    }

    SECTION("No username provided to ChildProcess::Create when current uid is same as uid passed in")
    {
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);
        REQUIRE(localUnixPath == expectedLocalUnixPath);

        auto funcArgs = MockStateCache::GetFuncArgs(mockKey, "Create");
        REQUIRE(funcArgs.size() == 5);
        auto username = std::any_cast<std::optional<std::string>>(funcArgs[3]);
        REQUIRE(!username.has_value());
        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        MockStateCache::ClearFuncArgs(mockKey);
    }

    SECTION("Username provided to ChildProcess::Create when current uid is different from uid passed in")
    {
        std::string currentUsername = "testUser";
        uid_t newUid                = uid + 1;
        manager.SetCurrentUsername(currentUsername);
        // Create the tmp directory for the new uid for the test. The current uid will not be able to create the
        // directory. Override the path ownership and permissions check to allow the test to continue.
        manager.SetIsRootUser(false);
        std::string newUidUnixPathPrefix = manager.IsRunningAsRoot() ? fmt::format("./run/user/{}/dcgm", newUid)
                                                                     : fmt::format("./tmp/dcgm_{}", newUid);
        boost::filesystem::create_directories(newUidUnixPathPrefix);
        manager.SetOverrideVerifyPathOwnershipAndPermissions(true);

        // Set mock expectations for the new uid.
        auto expectedLocalUnixPath = fmt::format("{}/ssh_0.sock", newUidUnixPathPrefix);
        auto mockKey               = manager.GetSSHAddressForwardingString(expectedLocalUnixPath, remoteUnixPath);
        MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);

        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, newUid);
        REQUIRE(tunnelState == TunnelState::Active);
        REQUIRE(localUnixPath == expectedLocalUnixPath);

        auto funcArgs = MockStateCache::GetFuncArgs(mockKey, "Create");
        REQUIRE(funcArgs.size() == 5);
        auto username = std::any_cast<std::optional<std::string>>(funcArgs[3]);
        REQUIRE(username.has_value());
        REQUIRE(!username->empty());
        CHECK(currentUsername == *username);
        manager.EndSession(remoteHostname, remoteUnixPath, newUid);
        MockStateCache::ClearFuncArgs(mockKey);
    }

    SECTION("Uid with no username errors")
    {
        // GetUsernameForUid is called only when uid is different from current uid.
        uid_t newUid = uid + 1;
        manager.SetCurrentUsername("");
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, newUid);
        REQUIRE(tunnelState == TunnelState::GenericFailure);
    }

    SECTION("Session to same host with same remote unix path and same uid returns existing session")
    {
        std::string existingLocalUnixPath, returnedLocalUnixPath;

        // Start a session
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, existingLocalUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);
        // Start another session with the same remote host and unix path as previous session
        auto tunnelState2 = manager.StartSession(remoteHostname, remoteUnixPath, returnedLocalUnixPath, uid);
        REQUIRE(tunnelState2 == TunnelState::Active);
        // Verify that the same local unix path is returned for the second session
        REQUIRE(existingLocalUnixPath == returnedLocalUnixPath);

        manager.EndSession(remoteHostname, remoteUnixPath, uid);
    }

    SECTION("Session to same host with same remote unix path but different uid creates a new tunnel")
    {
        std::string firstLocalUnixPath;
        // Start first session to a host with remoteUnixPath
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, firstLocalUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);

        // Start a session with a different uid
        manager.SetCurrentUsername("randomUser");
        manager.SetIsRootUser(false);
        uid_t secondUid                  = uid + 1;
        std::string newUidUnixPathPrefix = manager.IsRunningAsRoot() ? fmt::format("./run/user/{}/dcgm", secondUid)
                                                                     : fmt::format("./tmp/dcgm_{}", secondUid);
        auto secondLocalUnixPath         = fmt::format("{}/ssh_1.sock", newUidUnixPathPrefix);
        auto mockKey                     = manager.GetSSHAddressForwardingString(secondLocalUnixPath, remoteUnixPath);
        MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);
        // Create the tmp directory for the new uid for the test. The current uid will not be able to create the
        // directory. Override the path ownership and permissions check to allow the test to continue.
        boost::filesystem::create_directories(newUidUnixPathPrefix);
        manager.SetOverrideVerifyPathOwnershipAndPermissions(true);
        auto tunnelState2 = manager.StartSession(remoteHostname, remoteUnixPath, secondLocalUnixPath, secondUid);
        REQUIRE(tunnelState2 == TunnelState::Active);
        // Verify that the same local unix path is returned for the second session
        REQUIRE(firstLocalUnixPath != secondLocalUnixPath);

        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        manager.EndSession(remoteHostname, remoteUnixPath, secondUid);
        boost::filesystem::remove_all(newUidUnixPathPrefix);
    }

    SECTION("Different sessions to same host with different remote unix paths")
    {
        // Start first session to a host with remoteUnixPath
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);

        // Start second session to the same host with a different remote unix path
        expectedLocalUnixPath = fmt::format("{}/ssh_1.sock", localUnixPathPrefix);
        mockKey               = manager.GetSSHAddressForwardingString(expectedLocalUnixPath, anotherRemoteUnixPath);
        MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);
        auto tunnelState2 = manager.StartSession(remoteHostname, anotherRemoteUnixPath, localUnixPath, uid);
        // Verify that second session returns active state.
        REQUIRE(tunnelState2 == TunnelState::Active);

        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        manager.EndSession(remoteHostname, anotherRemoteUnixPath, uid);
    }

    SECTION("For next session, returns different file with next number")
    {
        // Start first session
        auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
        REQUIRE(tunnelState == TunnelState::Active);

        // Setup for second session
        expectedLocalUnixPath = fmt::format("{}/ssh_1.sock", localUnixPathPrefix);
        mockKey               = manager.GetSSHAddressForwardingString(expectedLocalUnixPath, anotherRemoteUnixPath);
        MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);

        // Start second session
        auto tunnelState2 = manager.StartSession(remoteHostname, anotherRemoteUnixPath, anotherLocalUnixPath, uid);
        REQUIRE(tunnelState2 == TunnelState::Active);

        // extract number from both the sessions i.e. localUnixPath and anotherLocalUnixPath and compare
        auto localUnixPathNumber = std::stoi(localUnixPath.substr(localUnixPath.find_last_of('_') + 1));
        auto anotherLocalUnixPathNumber
            = std::stoi(anotherLocalUnixPath.substr(anotherLocalUnixPath.find_last_of('_') + 1));
        REQUIRE(localUnixPathNumber + 1 == anotherLocalUnixPathNumber);

        manager.EndSession(remoteHostname, remoteUnixPath, uid);
        manager.EndSession(remoteHostname, anotherRemoteUnixPath, uid);
    }

    SECTION("Uds file path tests.")
    {
        SECTION("Root user with primary path")
        {
            auto mockKey = manager.GetSSHAddressForwardingString(fmt::format("./run/dcgm/ssh_0.sock"), remoteUnixPath);
            MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);
            manager.SetRunningAsRoot(true); // Run parent process as root to use primary path
            manager.SetIsRootUser(true);    // set the user to root

            auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
            REQUIRE(tunnelState == TunnelState::Active);
            REQUIRE(localUnixPath.starts_with("./run/dcgm"));
            manager.EndSession(remoteHostname, remoteUnixPath, uid);
        }

        SECTION("Non-root user with primary path")
        {
            auto mockKey = manager.GetSSHAddressForwardingString(fmt::format("./run/user/{}/dcgm/ssh_0.sock", uid),
                                                                 remoteUnixPath);
            MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);
            manager.SetRunningAsRoot(true); // Run parent process as root to use primary path
            manager.SetIsRootUser(false);   // set the user to non-root
            // let's create ./run/user/{uid} directory which is required for the test.
            boost::filesystem::create_directories(fmt::format("./run/user/{}", uid));

            auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
            REQUIRE(tunnelState == TunnelState::Active);
            REQUIRE(localUnixPath.starts_with(fmt::format("./run/user/{}/dcgm", uid)));
            manager.EndSession(remoteHostname, remoteUnixPath, uid);
        }

        auto mockKey
            = manager.GetSSHAddressForwardingString(fmt::format("./tmp/dcgm_{}/ssh_0.sock", uid), remoteUnixPath);
        MockStateCache::SetMockReturns(mockKey, mockReturnSuccess);
        SECTION("Root user with secondary path")
        {
            manager.SetRunningAsRoot(false); // Run parent process as non-root to use secondary path
            manager.SetIsRootUser(true);     // set the user to root

            auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
            REQUIRE(tunnelState == TunnelState::Active);
            REQUIRE(localUnixPath.starts_with(fmt::format("./tmp/dcgm_{}", uid)));
            manager.EndSession(remoteHostname, remoteUnixPath, uid);
        }

        SECTION("Non-root user with secondary path")
        {
            manager.SetRunningAsRoot(false); // Run parent process as non-root to use secondary path
            manager.SetIsRootUser(false);    // set the user to non-root

            auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
            REQUIRE(tunnelState == TunnelState::Active);
            REQUIRE(localUnixPath.starts_with(fmt::format("./tmp/dcgm_{}", uid)));
            manager.EndSession(remoteHostname, remoteUnixPath, uid);
        }

        SECTION("No accessible paths")
        {
            // Remove test directories to simulate inaccessible paths
            boost::filesystem::remove_all(manager.GetPrimaryPath());
            boost::filesystem::remove_all(manager.GetSecondaryPath());

            auto tunnelState = manager.StartSession(remoteHostname, remoteUnixPath, localUnixPath, uid);
            REQUIRE(tunnelState == TunnelState::GenericFailure);
            REQUIRE(localUnixPath.empty());
        }
    }

    // Cleanup
    MockStateCache::ClearAll();
}
