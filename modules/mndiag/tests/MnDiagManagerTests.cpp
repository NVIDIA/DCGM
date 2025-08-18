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

#include "DcgmRequest.h"
#include "dcgm_module_structs.h"
#include "dcgm_test_apis.h"
#include <DcgmMnDiagManager.h>
#include <DcgmStringHelpers.h>
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <dcgm_structs.h>
#include <memory>
#include <ranges>

#include "mocks/MockDcgmApi.h"
#include "mocks/MockDcgmCoreProxy.h"
#include "mocks/MockDcgmResourceHandle.h"
#include "mocks/MockMnDiagMpiRunner.h"
#include "mocks/MockMnDiagStateMachine.h"
#include "mocks/MockTcpSSHTunnelManager.h"
#include "mocks/MockUdsSSHTunnelManager.h"

// Mock implementation of processModuleCommandAtHostEngine for tests
dcgmReturn_t processModuleCommandAtHostEngine(dcgmHandle_t,
                                              dcgm_module_command_header_t *,
                                              size_t,
                                              std::unique_ptr<DcgmRequest>,
                                              unsigned int)
{
    return DCGM_ST_OK;
}

// Define minimal callback function for initialization
static dcgmReturn_t dummyPostCallback(dcgm_module_command_header_t *, void *)
{
    return DCGM_ST_OK;
}

/**
 * Fixture class for MnDiagManager tests
 * This provides direct access to the manager for testing
 */
class MnDiagManagerTests
{
protected:
    dcgmCoreCallbacks_t m_callbacks;
    DcgmMnDiagManager m_manager;

public:
    MnDiagManagerTests()
        : m_callbacks(GetCallbacks())
        , m_manager(m_callbacks)
    {
        // Setup is done in the constructor
    }

    // Expose private methods for testing
    dcgmReturn_t ExtractHostnameAndPort(std::string const &host,
                                        std::string &hostname,
                                        uint16_t &port,
                                        std::string &socketPath)
    {
        return m_manager.ExtractHostnameAndPort(host, hostname, port, socketPath);
    }

    dcgmReturn_t ConnectRemoteNodes(std::vector<std::string> const &hostList, uid_t effectiveUid)
    {
        return m_manager.ConnectRemoteNodes(hostList, effectiveUid);
    }

    dcgmReturn_t DisconnectRemoteNodes()
    {
        return m_manager.DisconnectRemoteNodes();
    }

    dcgmReturn_t HandleAuthorizeConnection(size_t connectionId)
    {
        return m_manager.HandleAuthorizeConnection(connectionId);
    }

    dcgmReturn_t HandleRevokeAuthorization(size_t connectionId)
    {
        return m_manager.HandleRevokeAuthorization(connectionId);
    }

    dcgmReturn_t HandleIsConnectionAuthorized(size_t connectionId)
    {
        return m_manager.HandleIsConnectionAuthorized(connectionId);
    }

    dcgmReturn_t HandleReserveResources(dcgm_module_command_header_t *moduleCommand)
    {
        return m_manager.HandleReserveResources(moduleCommand);
    }

    dcgmReturn_t HandleReleaseResources(dcgm_module_command_header_t *moduleCommand)
    {
        return m_manager.HandleReleaseResources(moduleCommand);
    }

    dcgmReturn_t HandleDetectProcess(dcgm_module_command_header_t *moduleCommand)
    {
        return m_manager.HandleDetectProcess(moduleCommand);
    }

    dcgmReturn_t HandleBroadcastRunParameters(dcgm_module_command_header_t *moduleCommand)
    {
        return m_manager.HandleBroadcastRunParameters(moduleCommand);
    }

    dcgmReturn_t BroadcastRunParametersToRemoteNodes(dcgmRunMnDiag_t const &params)
    {
        return m_manager.BroadcastRunParametersToRemoteNodes(params);
    }

    dcgmReturn_t HandleGetNodeInfo(dcgm_module_command_header_t *moduleCommand)
    {
        return m_manager.HandleGetNodeInfo(moduleCommand);
    }

    dcgmReturn_t AcquireResources()
    {
        return m_manager.AcquireResources();
    }

    dcgmReturn_t ReleaseResources()
    {
        return m_manager.ReleaseResources();
    }

    void SetStatus(MnDiagStatus status)
    {
        m_manager.SetStatus(status);
    }

    MnDiagStatus GetStatus()
    {
        return m_manager.GetStatus();
    }

    dcgmReturn_t ReserveRemoteResources()
    {
        return m_manager.ReserveRemoteResources();
    }

    dcgmReturn_t ReleaseRemoteResources()
    {
        return m_manager.ReleaseRemoteResources();
    }

    dcgmReturn_t GetNodeInfo()
    {
        return m_manager.GetNodeInfo();
    }

    bool AreDevicesSupported()
    {
        return m_manager.AreDevicesSupported();
    }

    void LoadSupportedSkusFromEnv()
    {
        return m_manager.LoadSupportedSkusFromEnv();
    }

    void PopulateMpiFailureResponseStruct(MnDiagMpiRunnerBase *mpiRunner,
                                          dcgmMnDiagResponse_v1 &response,
                                          std::optional<pid_t> mpiPID,
                                          DcgmNs::Utils::LogPaths const &logPaths)
    {
        return m_manager.PopulateMpiFailureResponseStruct(mpiRunner, response, mpiPID, logPaths);
    }

    std::unordered_set<std::string> GetSupportedSkus()
    {
        return m_manager.GetSupportedSkus();
    }

    std::string GetMnubergemmPathHeadNode()
    {
        return m_manager.GetMnubergemmPathHeadNode();
    }

    /**
     * @brief Set a custom resource handle factory for testing
     *
     * @param factory The factory implementation to use
     */
    void SetDcgmResourceHandleFactory(std::unique_ptr<DcgmResourceHandleFactoryBase> factory)
    {
        m_manager.m_resourceHandleFactory = std::move(factory);
    }

    /**
     * @brief Set a custom state machine for testing
     *
     * @param stateMachine The state machine implementation to use
     */
    void SetStateMachine(std::unique_ptr<MnDiagStateMachineBase> stateMachine)
    {
        m_manager.m_stateMachine = std::move(stateMachine);
    }

    void SetCoreProxy(std::unique_ptr<DcgmCoreProxyBase> coreProxy)
    {
        m_manager.m_coreProxy = std::move(coreProxy);
    }

    /**
     * @brief Set a new DCGM API instance for testing
     *
     * @param dcgmApi The API implementation to use
     */
    void SetDcgmApi(std::unique_ptr<DcgmApiBase> dcgmApi)
    {
        m_manager.m_dcgmApi = std::move(dcgmApi);
    }

    /**
     * @brief Set the Tcp SSH Tunnel Manager
     *
     * @param tunnelManager The tunnel manager implementation to use
     */
    void SetTcpSSHTunnelManager(std::unique_ptr<TcpSSHTunnelManagerBase> tunnelManager)
    {
        m_manager.m_tcpSSHTunnelManager = std::move(tunnelManager);
    }

    /**
     * @brief Set the Uds SSH Tunnel Manager
     *
     * @param tunnelManager The tunnel manager implementation to use
     */
    void SetUdsSSHTunnelManager(std::unique_ptr<UdsSSHTunnelManagerBase> tunnelManager)
    {
        m_manager.m_udsSSHTunnelManager = std::move(tunnelManager);
    }

    /**
     * @brief Set the MPI Runner Factory
     *
     * @param factory The MPI runner factory implementation to use
     */
    void SetMpiRunnerFactory(std::unique_ptr<MnDiagMpiRunnerFactoryBase> factory)
    {
        m_manager.m_mpiRunnerFactory = std::move(factory);
    }

    void SetCurrentNodeId(unsigned int currentNodeId)
    {
        m_manager.m_currentNodeId = currentNodeId;
    }

    /**
     * @brief Get the Tcp SSH Tunnel Manager
     *
     * @return MockTcpSSHTunnelManager* The tunnel manager instance
     */
    MockTcpSSHTunnelManager *GetTcpSSHTunnelManager()
    {
        return static_cast<MockTcpSSHTunnelManager *>(m_manager.m_tcpSSHTunnelManager.get());
    }

    /**
     * @brief Get the Uds SSH Tunnel Manager
     *
     * @return MockUdsSSHTunnelManager* The tunnel manager instance
     */
    MockUdsSSHTunnelManager *GetUdsSSHTunnelManager()
    {
        return static_cast<MockUdsSSHTunnelManager *>(m_manager.m_udsSSHTunnelManager.get());
    }

    MockDcgmCoreProxy *GetDcgmCoreProxy()
    {
        return static_cast<MockDcgmCoreProxy *>(m_manager.m_coreProxy.get());
    }

    /**
     * @brief Get access to the state machine
     *
     * @return MockMnDiagStateMachine* The state machine instance
     */
    MockMnDiagStateMachine *GetStateMachine()
    {
        return static_cast<MockMnDiagStateMachine *>(m_manager.m_stateMachine.get());
    }

    /**
     * @brief Get the DCGM API instance
     *
     * @return MockDcgmApi* The API instance
     */
    MockDcgmApi *GetDcgmApi()
    {
        return static_cast<MockDcgmApi *>(m_manager.m_dcgmApi.get());
    }

    /**
     * @brief Get the MPI Runner Factory
     *
     * @return MockMnDiagMpiRunnerFactory* The MPI runner factory instance
     */
    MockMnDiagMpiRunnerFactory *GetMpiRunnerFactory()
    {
        return static_cast<MockMnDiagMpiRunnerFactory *>(m_manager.m_mpiRunnerFactory.get());
    }

    /**
     * @brief Check if a resource handle exists
     *
     * @return true If a resource handle exists
     * @return false If no resource handle exists
     */
    bool HasResourceHandle() const
    {
        return m_manager.m_resourceHandle != nullptr;
    }

    /**
     * @brief Set up mock connections for testing remote nodes
     *
     * @param numConnections Number of connections to set up
     */
    void SetupMockConnections(int numConnections = 2)
    {
        // Create mock connections for testing
        for (int i = 0; i < numConnections; i++)
        {
            std::string hostname = "test-node-" + std::to_string(i);
            dcgmHandle_t handle  = i + 10; // Arbitrary handle values

            ConnectionInfo connInfo;
            connInfo.handle           = handle;
            connInfo.remotePort       = 5555 + i;
            connInfo.isLoopback       = false;
            connInfo.uid              = 1000;
            connInfo.remoteSocketPath = "";

            m_manager.m_connections[hostname] = connInfo;
        }
    }

    /**
     * @brief Set up a loopback connection for testing
     */
    void SetupLoopbackConnection()
    {
        ConnectionInfo connInfo;
        connInfo.handle           = 0;
        connInfo.remotePort       = 0;
        connInfo.isLoopback       = true;
        connInfo.uid              = 1000;
        connInfo.remoteSocketPath = "";

        m_manager.m_connections["localhost"] = connInfo;
    }

    /**
     * @brief Run the multi-node diagnostic on the head node
     *
     * @param params Parameters for the diagnostic
     * @param effectiveUid Effective UID of the caller
     * @param response Reference to a response structure to populate
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    dcgmReturn_t RunHeadNode(dcgmRunMnDiag_t const &params, uid_t effectiveUid, dcgmMnDiagResponse_t &response)
    {
        return m_manager.RunHeadNode(params, effectiveUid, response);
    }

private:
    /**
     * @brief Get the callback structure for initializing the manager
     *
     * @return dcgmCoreCallbacks_t Callback structure
     */
    dcgmCoreCallbacks_t GetCallbacks()
    {
        dcgmCoreCallbacks_t callbacks = {};
        callbacks.version             = dcgmCoreCallbacks_version;
        callbacks.postfunc            = dummyPostCallback;
        callbacks.poster              = nullptr;
        callbacks.loggerfunc          = nullptr;
        return callbacks;
    }
};

// Helper to save and restore environment variable
auto saveEnvVar = [](char const *name) -> std::optional<std::string> {
    char const *val = std::getenv(name);
    if (val)
    {
        return std::string(val);
    }
    return std::nullopt;
};

auto restoreEnvVar = [](char const *name, const std::optional<std::string> &val) {
    if (val)
    {
        setenv(name, val->c_str(), 1);
    }
    else
    {
        unsetenv(name);
    }
};

/**
 * Tests for the ExtractHostnameAndPort method
 * Verifies parsing of host strings with various formats
 */
TEST_CASE_METHOD(MnDiagManagerTests, "ExtractHostnameAndPort [mndiag]")
{
    // Common variables for all tests
    std::string host;
    std::string extractedHostname;
    uint16_t extractedPort;
    std::string extractedSocketPath;
    dcgmReturn_t result;

    SECTION("Valid host formats")
    {
        // Common setup for valid format tests
        extractedHostname   = "";
        extractedPort       = DCGM_HE_PORT_NUMBER; // Default port
        extractedSocketPath = DCGM_UNIX_SOCKET_PREFIX + std::string(DCGM_DEFAULT_SOCKET_PATH);

        SECTION("Basic hostname without port")
        {
            host = "hostname";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
            REQUIRE(extractedPort == DCGM_HE_PORT_NUMBER);
        }

        SECTION("Hostname with explicit port")
        {
            host = "hostname:8888";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
            REQUIRE(extractedPort == 8888);
        }

        SECTION("Hostname with port and GPU list")
        {
            host = "hostname:7777=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
            REQUIRE(extractedPort == 7777);
        }

        SECTION("Host with Unix socket path")
        {
            host = "hostname:unix:///tmp/socket=0,1,2";
            std::string extractedSocketPath;

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
            REQUIRE(extractedPort == 0);
            REQUIRE(extractedSocketPath == "/tmp/socket");
        }

        SECTION("Hostname with specific GPU ID")
        {
            host = "hostname=123";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
            REQUIRE(extractedPort == 5555);
        }

        SECTION("Hostname with minimum valid port")
        {
            host = "hostname:1=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
            REQUIRE(extractedPort == 1);
        }

        SECTION("Hostname with maximum valid port")
        {
            host = "hostname:65535=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
            REQUIRE(extractedPort == 65535);
        }
    }

    SECTION("Invalid host formats")
    {
        // Common setup for invalid format tests
        extractedHostname   = "";
        extractedPort       = DCGM_HE_PORT_NUMBER; // Default port
        extractedSocketPath = DCGM_UNIX_SOCKET_PREFIX + std::string(DCGM_DEFAULT_SOCKET_PATH);

        SECTION("Empty hostname")
        {
            host = "";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Missing hostname before GPU list")
        {
            host = "=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Invalid port number")
        {
            host = "hostname:invalid=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Invalid port range (negative)")
        {
            host = "hostname:-1=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Invalid port range (too large)")
        {
            host = "hostname:65536=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Empty port specification")
        {
            host = "hostname:=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Invalid hostname with leading colon")
        {
            host = ":hostname=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Invalid Unix socket format")
        {
            host = "hostname:unix//path=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Missing hostname in Unix socket path")
        {
            host = ":unix:///tmp/socket=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Hostname containing only colons")
        {
            host = ":::=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Alternate invalid Unix socket path format")
        {
            host = "hostname:unix:/path=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Non-numeric GPU ID")
        {
            host = "hostname=a,b,c";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            // Note: We're only testing the hostname/port extraction, not the GPU ID validation
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
        }

        SECTION("Unix socket prefix without path")
        {
            host = "hostname:unix://=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            // This should be valid for the ExtractHostnameAndPort function
            // since it doesn't validate the path part of the Unix socket
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(extractedHostname == "hostname");
            REQUIRE(extractedSocketPath == "");
        }

        SECTION("Very large port number")
        {
            host = "hostname:99999=0,1,2";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        // New test cases for strict port validation
        SECTION("Unix string in port")
        {
            host = "host1:unix=0,1";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Port with trailing garbage")
        {
            host = "host1:12345garbage=0,1";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Alphanumeric port")
        {
            host = "host1:1a2b3c45=0,1";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }

        SECTION("Decimal port")
        {
            host = "host1:1.2345=0,1";

            result = ExtractHostnameAndPort(host, extractedHostname, extractedPort, extractedSocketPath);

            REQUIRE(result == DCGM_ST_BADPARAM);
        }
    }
}

/**
 * Tests for connection authorization functionality
 * Tests the authorization, revocation, and verification methods
 */
TEST_CASE_METHOD(MnDiagManagerTests, "Connection Authorization Tests [mndiag]")
{
    SECTION("HandleAuthorizeConnection Method")
    {
        SECTION("First connection should be authorized")
        {
            size_t connectionId = 1;
            auto result         = HandleAuthorizeConnection(connectionId);
            REQUIRE(result == DCGM_ST_OK);
        }

        SECTION("Same connection can be authorized multiple times")
        {
            size_t connectionId = 2;
            auto result1        = HandleAuthorizeConnection(connectionId);
            auto result2        = HandleAuthorizeConnection(connectionId);

            REQUIRE(result1 == DCGM_ST_OK);
            REQUIRE(result2 == DCGM_ST_OK);
        }

        SECTION("Different connection should be rejected when one is already authorized")
        {
            size_t firstConnectionId  = 3;
            size_t secondConnectionId = 4;

            auto result1 = HandleAuthorizeConnection(firstConnectionId);
            auto result2 = HandleAuthorizeConnection(secondConnectionId);

            REQUIRE(result1 == DCGM_ST_OK);
            REQUIRE(result2 == DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);
        }
    }

    SECTION("HandleRevokeAuthorization Method")
    {
        SECTION("Revoking non-existent authorization should fail")
        {
            size_t connectionId = 5;
            auto result         = HandleRevokeAuthorization(connectionId);
            REQUIRE(result == DCGM_ST_MNDIAG_CONNECTION_NOT_AVAILABLE);
        }

        SECTION("Revoking with wrong connection ID should fail")
        {
            size_t authorizedId   = 6;
            size_t unauthorizedId = 7;

            // Authorize a connection
            auto authResult = HandleAuthorizeConnection(authorizedId);
            REQUIRE(authResult == DCGM_ST_OK);

            // Try to revoke with wrong ID
            auto revokeResult = HandleRevokeAuthorization(unauthorizedId);
            REQUIRE(revokeResult == DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);
        }

        SECTION("Successful revocation")
        {
            size_t connectionId = 8;

            // Authorize a connection
            auto authResult = HandleAuthorizeConnection(connectionId);
            REQUIRE(authResult == DCGM_ST_OK);

            // Revoke authorization
            auto revokeResult = HandleRevokeAuthorization(connectionId);
            REQUIRE(revokeResult == DCGM_ST_OK);

            // Verify it was revoked by trying to authorize another connection
            size_t newConnectionId = 9;
            auto newAuthResult     = HandleAuthorizeConnection(newConnectionId);
            REQUIRE(newAuthResult == DCGM_ST_OK);
        }
    }

    SECTION("HandleIsConnectionAuthorized Method")
    {
        SECTION("Check when no connection is authorized")
        {
            size_t connectionId = 10;
            auto result         = HandleIsConnectionAuthorized(connectionId);
            REQUIRE(result == DCGM_ST_MNDIAG_CONNECTION_NOT_AVAILABLE);
        }

        SECTION("Check with wrong connection ID")
        {
            size_t authorizedId   = 11;
            size_t unauthorizedId = 12;

            // Authorize a connection
            auto authResult = HandleAuthorizeConnection(authorizedId);
            REQUIRE(authResult == DCGM_ST_OK);

            // Check with wrong ID
            auto checkResult = HandleIsConnectionAuthorized(unauthorizedId);
            REQUIRE(checkResult == DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);
        }

        SECTION("Check with correct connection ID")
        {
            size_t connectionId = 13;

            // Authorize a connection
            auto authResult = HandleAuthorizeConnection(connectionId);
            REQUIRE(authResult == DCGM_ST_OK);

            // Check with correct ID
            auto checkResult = HandleIsConnectionAuthorized(connectionId);
            REQUIRE(checkResult == DCGM_ST_OK);
        }
    }

    SECTION("Complete authorization flow")
    {
        size_t connectionId = 14;

        // Initially, no connection is authorized
        auto initialCheck = HandleIsConnectionAuthorized(connectionId);
        REQUIRE(initialCheck == DCGM_ST_MNDIAG_CONNECTION_NOT_AVAILABLE);

        // Authorize the connection
        auto authResult = HandleAuthorizeConnection(connectionId);
        REQUIRE(authResult == DCGM_ST_OK);

        // Verify it's authorized
        auto checkResult = HandleIsConnectionAuthorized(connectionId);
        REQUIRE(checkResult == DCGM_ST_OK);

        // Revoke the authorization
        auto revokeResult = HandleRevokeAuthorization(connectionId);
        REQUIRE(revokeResult == DCGM_ST_OK);

        // Verify it's no longer authorized
        auto finalCheck = HandleIsConnectionAuthorized(connectionId);
        REQUIRE(finalCheck == DCGM_ST_MNDIAG_CONNECTION_NOT_AVAILABLE);
    }
}

/**
 * Tests for resource management functionality in DcgmMnDiagManager
 *
 * This test scenario covers:
 * - Status setting and getting
 * - Resource acquisition
 * - Resource release
 * - State transitions during resource management
 */
TEST_CASE_METHOD(MnDiagManagerTests, "Resource Management Tests [mndiag]")
{
    /**
     * Tests for GetStatus and SetStatus methods
     * Ensures status can be properly set and retrieved
     */
    SECTION("GetStatus and SetStatus Methods")
    {
        SECTION("Initial status should be READY")
        {
            // The default status should be READY
            REQUIRE(GetStatus() == MnDiagStatus::READY);
        }

        SECTION("SetStatus should update status")
        {
            // Set and verify various status values
            SetStatus(MnDiagStatus::RUNNING);
            REQUIRE(GetStatus() == MnDiagStatus::RUNNING);

            SetStatus(MnDiagStatus::COMPLETED);
            REQUIRE(GetStatus() == MnDiagStatus::COMPLETED);

            SetStatus(MnDiagStatus::FAILED);
            REQUIRE(GetStatus() == MnDiagStatus::FAILED);

            SetStatus(MnDiagStatus::READY);
            REQUIRE(GetStatus() == MnDiagStatus::READY);
        }
    }

    /**
     * Tests for AcquireResources method
     * Verifies resources can be acquired only in READY state
     * and handles resource acquisition failures correctly
     */
    SECTION("AcquireResources Method")
    {
        SECTION("Should succeed when status is READY")
        {
            // Ensure status is READY
            SetStatus(MnDiagStatus::READY);

            // Create and configure mock factory with success result
            auto mockFactory = std::make_unique<MockDcgmResourceHandleFactory>(DCGM_ST_OK);

            // Set the mock factory
            SetDcgmResourceHandleFactory(std::move(mockFactory));

            // Acquire resources
            auto result = AcquireResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::RESERVED);
            REQUIRE(HasResourceHandle() == true);
        }

        SECTION("Should fail when status is not READY")
        {
            // Set status to something other than READY
            SetStatus(MnDiagStatus::RUNNING);

            // Create mock factory (should not be used in this test case)
            auto mockFactory = std::make_unique<MockDcgmResourceHandleFactory>(DCGM_ST_OK);

            // Set the mock factory
            SetDcgmResourceHandleFactory(std::move(mockFactory));

            // Acquire resources should fail due to wrong status
            auto result = AcquireResources();

            // Verify results
            REQUIRE(result == DCGM_ST_IN_USE);
            REQUIRE(GetStatus() == MnDiagStatus::RUNNING); // Status should not change
            REQUIRE(HasResourceHandle() == false);
        }

        SECTION("Should fail when resource handle initialization fails")
        {
            // Ensure status is READY
            SetStatus(MnDiagStatus::READY);

            // Create and configure mock factory with failure result
            auto mockFactory = std::make_unique<MockDcgmResourceHandleFactory>(DCGM_ST_IN_USE);

            // Set the mock factory
            SetDcgmResourceHandleFactory(std::move(mockFactory));

            // Acquire resources should fail due to mock returning failure
            auto result = AcquireResources();

            // Verify results
            REQUIRE(result == DCGM_ST_IN_USE);
            REQUIRE(GetStatus() == MnDiagStatus::READY); // Status should not change
            REQUIRE(HasResourceHandle() == false);
        }
    }

    /**
     * Tests for ReleaseResources method
     * Verifies resources can be released from any state
     * and properly cleans up resource handles
     */
    SECTION("ReleaseResources Method")
    {
        SECTION("Should release resources and set status to READY")
        {
            // First acquire resources
            SetStatus(MnDiagStatus::READY);

            // Create and configure mock factory with success result
            auto mockFactory = std::make_unique<MockDcgmResourceHandleFactory>(DCGM_ST_OK);

            // Set the mock factory
            SetDcgmResourceHandleFactory(std::move(mockFactory));

            auto acquireResult = AcquireResources();

            // Verify acquisition worked
            REQUIRE(acquireResult == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::RESERVED);
            REQUIRE(HasResourceHandle() == true);

            // Now release resources
            auto releaseResult = ReleaseResources();

            // Verify release worked
            REQUIRE(releaseResult == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::READY);
            REQUIRE(HasResourceHandle() == false);
        }

        SECTION("Should work even if no resources are currently held")
        {
            // Ensure no resources are held
            SetStatus(MnDiagStatus::READY);
            REQUIRE(HasResourceHandle() == false);

            // Release resources
            auto result = ReleaseResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::READY);
            REQUIRE(HasResourceHandle() == false);
        }

        SECTION("Should work from any state")
        {
            // Set status to RUNNING and simulate holding resources
            SetStatus(MnDiagStatus::RUNNING);

            // Release resources
            auto result = ReleaseResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::READY);
            REQUIRE(HasResourceHandle() == false);
        }
    }

    /**
     * Tests the complete resource management flow
     * Verifies all state transitions and resource handling
     * in a typical usage scenario
     */
    SECTION("Complete Resource Management Flow")
    {
        // Start from a clean state
        SetStatus(MnDiagStatus::READY);
        REQUIRE(HasResourceHandle() == false);

        // 1. Create and configure mock factory with success result
        auto mockFactory = std::make_unique<MockDcgmResourceHandleFactory>(DCGM_ST_OK);

        // Set the mock factory
        SetDcgmResourceHandleFactory(std::move(mockFactory));

        // 2. Acquire resources
        auto acquireResult = AcquireResources();
        REQUIRE(acquireResult == DCGM_ST_OK);
        REQUIRE(GetStatus() == MnDiagStatus::RESERVED);
        REQUIRE(HasResourceHandle() == true);

        // 3. Simulate running by changing status
        SetStatus(MnDiagStatus::RUNNING);
        REQUIRE(GetStatus() == MnDiagStatus::RUNNING);

        // 4. Try to acquire resources again (should fail)
        auto secondAcquireResult = AcquireResources();
        REQUIRE(secondAcquireResult == DCGM_ST_IN_USE);
        REQUIRE(GetStatus() == MnDiagStatus::RUNNING); // Status unchanged

        // 5. Simulate completion by changing status
        SetStatus(MnDiagStatus::COMPLETED);
        REQUIRE(GetStatus() == MnDiagStatus::COMPLETED);

        // 6. Release resources
        auto releaseResult = ReleaseResources();
        REQUIRE(releaseResult == DCGM_ST_OK);
        REQUIRE(GetStatus() == MnDiagStatus::READY);
        REQUIRE(HasResourceHandle() == false);

        // 7. Should be able to acquire resources again
        // Create a new mock factory
        auto newMockFactory = std::make_unique<MockDcgmResourceHandleFactory>(DCGM_ST_OK);
        SetDcgmResourceHandleFactory(std::move(newMockFactory));

        auto finalAcquireResult = AcquireResources();
        REQUIRE(finalAcquireResult == DCGM_ST_OK);
        REQUIRE(GetStatus() == MnDiagStatus::RESERVED);
        REQUIRE(HasResourceHandle() == true);
    }
}

/**
 * Tests for remote command handling in DcgmMnDiagManager
 * Covers the following handler methods:
 * - HandleReserveResources
 * - HandleReleaseResources
 * - HandleDetectProcess
 */
TEST_CASE_METHOD(MnDiagManagerTests, "Mn Diag Handler Tests [mndiag]")
{
    /**
     * Tests for HandleReserveResources method
     * Verifies resources can be reserved through the module command interface
     */
    SECTION("HandleReserveResources Method")
    {
        SECTION("Should reserve resources using the state machine")
        {
            // Create and inject mock state machine
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure state machine with default config (all true)
            MockMnDiagStateMachine::Config config;
            config.shouldReserveReturn = true;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Prepare message structure
            dcgm_mndiag_msg_resource_t resourceMsg {};
            resourceMsg.header.version    = dcgm_mndiag_msg_resource_version1;
            resourceMsg.header.moduleId   = DcgmModuleIdMnDiag;
            resourceMsg.header.subCommand = DCGM_MNDIAG_SR_RESERVE_RESOURCES;
            resourceMsg.resource.response = MnDiagStatus::UNKNOWN;

            // Call the handler method
            dcgmReturn_t result = HandleReserveResources((dcgm_module_command_header_t *)&resourceMsg);

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(resourceMsg.resource.response == MnDiagStatus::RESERVED);
            REQUIRE(GetStatus() == MnDiagStatus::RESERVED);

            // Verify state machine interaction using GetStats
            REQUIRE(GetStateMachine()->GetStats().notifyToReserveCalled == true);
        }

        SECTION("Should fail when state machine fails to reserve")
        {
            // Create and inject mock state machine
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure mock to fail for NotifyToReserve calls using Config struct
            MockMnDiagStateMachine::Config config;
            config.shouldReserveReturn = false;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Prepare message structure
            dcgm_mndiag_msg_resource_t resourceMsg {};
            resourceMsg.header.version    = dcgm_mndiag_msg_resource_version1;
            resourceMsg.header.moduleId   = DcgmModuleIdMnDiag;
            resourceMsg.header.subCommand = DCGM_MNDIAG_SR_RESERVE_RESOURCES;
            resourceMsg.resource.response = MnDiagStatus::UNKNOWN;

            // Call the handler method - it should fail
            dcgmReturn_t result = HandleReserveResources((dcgm_module_command_header_t *)&resourceMsg);

            // Verify results
            REQUIRE(result == DCGM_ST_IN_USE);
            REQUIRE(GetStateMachine()->GetStats().notifyToReserveCalled == true);
        }
    }

    /**
     * Tests for HandleReleaseResources method
     * Verifies resources can be released through the module command interface
     */
    SECTION("HandleReleaseResources Method")
    {
        SECTION("Should call NotifyDiagnosticFinished on the state machine")
        {
            // Create and inject mock state machine
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure state machine with Config struct
            MockMnDiagStateMachine::Config config;
            config.notifyDiagnosticFinishedReturn = true;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Prepare message structure
            dcgm_mndiag_msg_resource_t resourceMsg {};
            resourceMsg.header.version    = dcgm_mndiag_msg_resource_version1;
            resourceMsg.header.moduleId   = DcgmModuleIdMnDiag;
            resourceMsg.header.subCommand = DCGM_MNDIAG_SR_RELEASE_RESOURCES;
            resourceMsg.resource.response = MnDiagStatus::UNKNOWN;

            // Call the handler method
            dcgmReturn_t result = HandleReleaseResources((dcgm_module_command_header_t *)&resourceMsg);

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(resourceMsg.resource.response == MnDiagStatus::READY);
            REQUIRE(GetStatus() == MnDiagStatus::READY);

            // Verify state machine interaction using GetStats
            REQUIRE(GetStateMachine()->GetStats().notifyDiagnosticFinishedCalled == true);
        }

        SECTION("Should handle failure in NotifyDiagnosticFinished")
        {
            // Create and inject mock state machine
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure mock to fail for NotifyDiagnosticFinished calls with Config struct
            MockMnDiagStateMachine::Config config;
            config.notifyDiagnosticFinishedReturn = false;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Prepare message structure
            dcgm_mndiag_msg_resource_t resourceMsg {};
            resourceMsg.header.version    = dcgm_mndiag_msg_resource_version1;
            resourceMsg.header.moduleId   = DcgmModuleIdMnDiag;
            resourceMsg.header.subCommand = DCGM_MNDIAG_SR_RELEASE_RESOURCES;
            resourceMsg.resource.response = MnDiagStatus::UNKNOWN;

            // Set a non-READY status for the test
            SetStatus(MnDiagStatus::RUNNING);

            // Call the handler method
            dcgmReturn_t result = HandleReleaseResources((dcgm_module_command_header_t *)&resourceMsg);

            // Even if the state machine fails, the handler should still succeed
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStateMachine()->GetStats().notifyDiagnosticFinishedCalled == true);
        }
    }

    /**
     * Tests for HandleDetectProcess method
     * Verifies the behavior when detecting MPI processes
     */
    SECTION("HandleDetectProcess Method")
    {
        SECTION("When MPI process is detected")
        {
            // Create and inject mock state machine
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure mock with a detected PID using Config struct
            MockMnDiagStateMachine::Config config;
            config.detectedProcessInfo = { { 12345, "mnubergemm" } };
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Prepare message structure
            dcgm_mndiag_msg_resource_t resourceMsg {};
            resourceMsg.header.version    = dcgm_mndiag_msg_resource_version1;
            resourceMsg.header.moduleId   = DcgmModuleIdMnDiag;
            resourceMsg.header.subCommand = DCGM_MNDIAG_SR_DETECT_PROCESS;
            resourceMsg.resource.response = MnDiagStatus::UNKNOWN;

            // Call the handler method
            dcgmReturn_t result = HandleDetectProcess((dcgm_module_command_header_t *)&resourceMsg);

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(resourceMsg.resource.response == MnDiagStatus::RUNNING);
            REQUIRE(GetStatus() == MnDiagStatus::RUNNING);

            // Verify state machine interaction using GetStats
            REQUIRE(GetStateMachine()->GetStats().tryGetDetectedMpiPidCalled == true);
        }

        SECTION("When no MPI process is detected")
        {
            // Create and inject mock state machine
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure mock to return no PID using Config struct
            MockMnDiagStateMachine::Config config;
            config.detectedProcessInfo = {};
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Set an initial status
            SetStatus(MnDiagStatus::RESERVED);

            // Prepare message structure
            dcgm_mndiag_msg_resource_t resourceMsg {};
            resourceMsg.header.version    = dcgm_mndiag_msg_resource_version1;
            resourceMsg.header.moduleId   = DcgmModuleIdMnDiag;
            resourceMsg.header.subCommand = DCGM_MNDIAG_SR_DETECT_PROCESS;
            resourceMsg.resource.response = MnDiagStatus::UNKNOWN;

            // Call the handler method
            dcgmReturn_t result = HandleDetectProcess((dcgm_module_command_header_t *)&resourceMsg);

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStateMachine()->GetStats().tryGetDetectedMpiPidCalled == true);

            // The response should contain the current status
            REQUIRE(resourceMsg.resource.response == GetStatus());
        }
    }
}

/**
 * Tests for remote resources management with empty remote connections
 */
TEST_CASE_METHOD(MnDiagManagerTests, "Remote Resources Management Tests with empty remote connections [mndiag]")
{
    // Create and inject mock API
    auto dcgmApi = std::make_unique<MockDcgmApi>();
    SetDcgmApi(std::move(dcgmApi));

    /**
     * Tests for ReserveRemoteResources method
     * This method reserves resources on all connected remote nodes
     */
    SECTION("ReserveRemoteResources Method")
    {
        SECTION("Should succeed with no remote connections")
        {
            // Setup the mock state machine for the head node
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure the mock using Config struct
            MockMnDiagStateMachine::Config config;
            config.shouldReserveReturn = true;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Call the method under test
            dcgmReturn_t result = ReserveRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::RESERVED);
            REQUIRE(GetStateMachine()->GetStats().notifyToReserveCalled == true);
        }

        SECTION("Should fail if head node fails to reserve resources")
        {
            // Setup the mock state machine for the head node
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure the mock to fail using Config struct
            MockMnDiagStateMachine::Config config;
            config.shouldReserveReturn = false;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Call the method under test
            dcgmReturn_t result = ReserveRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_IN_USE);
            REQUIRE(GetStateMachine()->GetStats().notifyToReserveCalled == true);
        }
    }

    /**
     * Tests for ReleaseRemoteResources method
     * This method releases resources on all connected remote nodes
     */
    SECTION("ReleaseRemoteResources Method")
    {
        SECTION("Should succeed with no remote connections")
        {
            // Setup the mock state machine for the head node
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure the mock using Config struct
            MockMnDiagStateMachine::Config config;
            config.notifyDiagnosticFinishedReturn = true;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Call the method under test
            dcgmReturn_t result = ReleaseRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::READY);
            REQUIRE(GetStateMachine()->GetStats().notifyDiagnosticFinishedCalled == true);
        }

        SECTION("Should fail if head node fails to release resources")
        {
            // Setup the mock state machine for the head node
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure the mock to fail using Config struct
            MockMnDiagStateMachine::Config config;
            config.notifyDiagnosticFinishedReturn = false;
            mockStateMachine->SetConfig(config);

            // Set status to something other than READY
            SetStatus(MnDiagStatus::RUNNING);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Call the method under test - in this case, we're testing that the release operation
            // still succeeds even if the state machine's notification fails
            dcgmReturn_t result = ReleaseRemoteResources();

            // Verify results - should still return OK as the function is resilient to state machine failures
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStateMachine()->GetStats().notifyDiagnosticFinishedCalled == true);
        }
    }
}

/**
 * Tests for remote resources management with non-empty remote connections
 */
TEST_CASE_METHOD(MnDiagManagerTests, "Remote Resources Management Tests with non-empty remote connections [mndiag]")
{
    // Create and inject mock API
    auto dcgmApi = std::make_unique<MockDcgmApi>();
    SetDcgmApi(std::move(dcgmApi));

    /**
     * Tests for ReserveRemoteResources method with remote connections
     * This tests that requests are sent to all remote nodes
     */
    SECTION("ReserveRemoteResources with Remote Connections")
    {
        SECTION("Should succeed when all remote nodes succeed")
        {
            // Setup the mock state machine for the head node
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure the mock using Config struct
            MockMnDiagStateMachine::Config config;
            config.shouldReserveReturn = true;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Setup mock connections
            SetupMockConnections(3);

            // Configure the mock API to return success for all requests
            GetDcgmApi()->SetSendRequestResult(DCGM_ST_OK);
            GetDcgmApi()->SetReserveResourcesResponse(MnDiagStatus::RESERVED);

            // Call the method under test
            dcgmReturn_t result = ReserveRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::RESERVED);
            REQUIRE(GetStateMachine()->GetStats().notifyToReserveCalled == true);

            // Verify the API was called for each remote connection (3 connections)
            REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 3);
        }

        SECTION("Should fail when any remote node fails to reserve")
        {
            // Setup the mock state machine for the head node
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure the mock using Config struct
            MockMnDiagStateMachine::Config config;
            config.shouldReserveReturn = true;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Setup mock connections
            SetupMockConnections(3);

            // Configure the second request to fail by setting up handle-specific callbacks
            GetDcgmApi()->SetHandleCommandCallback(11,
                                                   dcgmMultinodeRequestType_t::ReserveResources,
                                                   [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_v1 *request) {
                                                       request->requestData.resource.response = MnDiagStatus::FAILED;
                                                       return DCGM_ST_IN_USE;
                                                   });

            // All other requests succeed
            GetDcgmApi()->SetSendRequestResult(DCGM_ST_OK);
            GetDcgmApi()->SetReserveResourcesResponse(MnDiagStatus::RESERVED);

            // Call the method under test
            dcgmReturn_t result = ReserveRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_IN_USE);

            // The API should have been called for only the failing connection before exiting
            // The implementation should stop processing after the first failure
            REQUIRE(GetDcgmApi()->GetSendRequestCallCount() > 0);
        }
    }

    /**
     * Tests for ReleaseRemoteResources method with remote connections
     * This tests that requests are sent to all remote nodes
     */
    SECTION("ReleaseRemoteResources with Remote Connections")
    {
        SECTION("Should succeed when all remote nodes succeed")
        {
            // Setup the mock state machine for the head node
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure the mock using Config struct
            MockMnDiagStateMachine::Config config;
            config.notifyDiagnosticFinishedReturn = true;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Setup mock connections
            SetupMockConnections(3);

            // Configure the mock API to return success for all requests
            GetDcgmApi()->SetSendRequestResult(DCGM_ST_OK);
            GetDcgmApi()->SetReleaseResourcesResponse(MnDiagStatus::READY);

            // Call the method under test
            dcgmReturn_t result = ReleaseRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::READY);
            REQUIRE(GetStateMachine()->GetStats().notifyDiagnosticFinishedCalled == true);

            // Verify the API was called for each remote connection (3 connections)
            REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 3);
        }

        SECTION("Should fail when any remote node fails to release")
        {
            // Setup the mock state machine for the head node
            auto mockStateMachine
                = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

            // Configure the mock using Config struct
            MockMnDiagStateMachine::Config config;
            config.notifyDiagnosticFinishedReturn = true;
            mockStateMachine->SetConfig(config);

            // Inject the mock state machine
            SetStateMachine(std::move(mockStateMachine));

            // Setup mock connections
            SetupMockConnections(3);

            // Configure one request to fail
            GetDcgmApi()->SetHandleCommandCallback(
                11,
                dcgmMultinodeRequestType_t::ReleaseResources,
                [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_v1 * /* request */) {
                    return DCGM_ST_CONNECTION_NOT_VALID;
                });

            // All other requests succeed
            GetDcgmApi()->SetSendRequestResult(DCGM_ST_OK);
            GetDcgmApi()->SetReleaseResourcesResponse(MnDiagStatus::READY);

            // Call the method under test
            dcgmReturn_t result = ReleaseRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_CONNECTION_NOT_VALID);

            // All connections should have been attempted
            REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 3);
        }
    }

    /**
     * Tests for GetNodeInfo method.
     */
    SECTION("GetNodeInfo with Remote Connections")
    {
        SECTION("Should succeed when all remote nodes succeed")
        {
            // Setup mock connections
            SetupMockConnections(3);

            // Configure the mock API to return success for all requests
            GetDcgmApi()->SetSendRequestResult(DCGM_ST_OK);

            // Call the method under test
            dcgmReturn_t result = GetNodeInfo();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);

            // Verify the API was called for each remote connection (3 connections)
            REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 3);
        }

        SECTION("Should fail when any remote node fails to get node info")
        {
            // Setup mock connections
            SetupMockConnections(3);

            // Configure one request to fail
            GetDcgmApi()->SetHandleCommandCallback(
                11,
                dcgmMultinodeRequestType_t::GetNodeInfo,
                [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_v1 * /* request */) {
                    return DCGM_ST_DIAG_BAD_JSON;
                });

            // All other requests succeed
            GetDcgmApi()->SetSendRequestResult(DCGM_ST_OK);

            // Call the method under test
            dcgmReturn_t result = GetNodeInfo();

            // Verify results
            REQUIRE(result == DCGM_ST_DIAG_BAD_JSON);

            // All connections should have been attempted
            REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 3);
        }
    }
}

/**
 * Tests for ConnectRemoteNodes and DisconnectRemoteNodes methods
 */
TEST_CASE_METHOD(MnDiagManagerTests, "ConnectRemoteNodes and DisconnectRemoteNodes Tests [mndiag]")
{
    SECTION("ConnectRemoteNodes handles TCP connections correctly")
    {
        // Set up mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Set default behavior for tunnel manager
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set default behavior for DCGM API
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // Test with a list of hosts using TCP connections
        std::vector<std::string> hostList = { "node1:5555", "node2:6666" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, 1000); // uid 1000

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify TCP tunnel manager was called with correct parameters
        auto calls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(calls.size() == 2);

        // Since threads run in parallel, we can't guarantee call order
        // So we need to check that both expected calls exist, regardless of order
        bool foundNode1 = false;
        bool foundNode2 = false;

        for (const auto &call : calls)
        {
            if (std::get<0>(call) == "node1" && std::get<1>(call) == 5555 && std::get<2>(call).value() == 1000)
            {
                foundNode1 = true;
            }
            else if (std::get<0>(call) == "node2" && std::get<1>(call) == 6666 && std::get<2>(call).value() == 1000)
            {
                foundNode2 = true;
            }
        }

        REQUIRE(foundNode1);
        REQUIRE(foundNode2);

        // Verify DCGM API was called correctly
        REQUIRE(GetDcgmApi()->GetConnectCallCount() == 2);
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 2);

        // Verify that the connection was made using the local port returned by the tunnel manager
        // Examine connections by handle in the DCGM API
        // For TCP connections, the address should include the local port returned by the SSH tunnel manager (12345)
        for (dcgmHandle_t handle = 1; handle <= 2; handle++)
        {
            REQUIRE(GetDcgmApi()->HasConnection(handle));
            auto connInfoOpt = GetDcgmApi()->GetConnectionInfo(handle);
            REQUIRE(connInfoOpt.has_value());
            auto const &connInfo = connInfoOpt.value();
            REQUIRE(connInfo.ipAddress == "127.0.0.1:12345"); // Port is included in the address
            REQUIRE(connInfo.params.addressIsUnixSocket == 0);
        }
    }

    SECTION("ConnectRemoteNodes handles UDS connections correctly")
    {
        // Set up mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockUdsSSHTunnelManager = std::make_unique<MockUdsSSHTunnelManager>();

        // Set default behavior for tunnel manager
        mockUdsSSHTunnelManager->SetDefaultRemoteSocketPath("/tmp/local_socket");
        mockUdsSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set default behavior for DCGM API
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetUdsSSHTunnelManager(std::move(mockUdsSSHTunnelManager));

        // Test with a list of hosts using UDS connections
        std::vector<std::string> hostList = { "node1:unix:///var/run/dcgm.socket", "node2:unix:///tmp/dcgm.socket" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, 1000); // uid 1000

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify UDS tunnel manager was called with correct parameters
        auto calls = GetUdsSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(calls.size() == 2);

        // Since threads run in parallel, we can't guarantee call order
        // So we need to check that both expected calls exist, regardless of order
        bool foundNode1 = false;
        bool foundNode2 = false;

        for (const auto &call : calls)
        {
            if (std::get<0>(call) == "node1" && std::get<1>(call) == "/var/run/dcgm.socket"
                && std::get<2>(call).value() == 1000)
            {
                foundNode1 = true;
            }
            else if (std::get<0>(call) == "node2" && std::get<1>(call) == "/tmp/dcgm.socket"
                     && std::get<2>(call).value() == 1000)
            {
                foundNode2 = true;
            }
        }

        REQUIRE(foundNode1);
        REQUIRE(foundNode2);

        // Verify DCGM API was called correctly
        REQUIRE(GetDcgmApi()->GetConnectCallCount() == 2);
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 2);

        // Verify that the connection was made using the local socket path returned by the tunnel manager
        // Examine connections by handle in the DCGM API
        // The socket path should match the one returned by the UDS SSH tunnel manager ("/tmp/local_socket")
        for (dcgmHandle_t handle = 1; handle <= 2; handle++)
        {
            REQUIRE(GetDcgmApi()->HasConnection(handle));
            auto connInfoOpt = GetDcgmApi()->GetConnectionInfo(handle);
            REQUIRE(connInfoOpt.has_value());
            auto const &connInfo = connInfoOpt.value();
            REQUIRE(connInfo.ipAddress == "/tmp/local_socket");
            REQUIRE(connInfo.params.addressIsUnixSocket == 1);
        }
    }

    SECTION("ConnectRemoteNodes handles mixed connections (TCP and UDS) correctly")
    {
        // Set up mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockUdsSSHTunnelManager = std::make_unique<MockUdsSSHTunnelManager>();

        // Set default behavior for tunnel managers
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);
        mockUdsSSHTunnelManager->SetDefaultRemoteSocketPath("/tmp/local_socket");
        mockUdsSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set default behavior for DCGM API
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetUdsSSHTunnelManager(std::move(mockUdsSSHTunnelManager));

        // Test with a list of hosts using both TCP and UDS connections
        std::vector<std::string> hostList = { "node1:5555", "node2:unix:///tmp/dcgm.socket" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, 1000); // uid 1000

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify TCP tunnel manager was called with correct parameters
        auto tcpCalls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(tcpCalls.size() == 1);

        // Verify the TCP calls contain the expected parameters
        bool foundTcpNode = false;
        for (const auto &call : tcpCalls)
        {
            if (std::get<0>(call) == "node1" && std::get<1>(call) == 5555 && std::get<2>(call).value() == 1000)
            {
                foundTcpNode = true;
                break;
            }
        }
        REQUIRE(foundTcpNode);

        // Verify UDS tunnel manager was called with correct parameters
        auto udsCalls = GetUdsSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(udsCalls.size() == 1);

        // Verify the UDS calls contain the expected parameters
        bool foundUdsNode = false;
        for (const auto &call : udsCalls)
        {
            if (std::get<0>(call) == "node2" && std::get<1>(call) == "/tmp/dcgm.socket"
                && std::get<2>(call).value() == 1000)
            {
                foundUdsNode = true;
                break;
            }
        }
        REQUIRE(foundUdsNode);

        // Verify DCGM API was called correctly
        REQUIRE(GetDcgmApi()->GetConnectCallCount() == 2);
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 2);

        // Verify that the connections were made using the correct local ports/paths returned by the tunnel managers
        // We need to examine each connection handle

        // Since we can't guarantee the order of handles, check both and make sure one is TCP and one is UDS
        REQUIRE(GetDcgmApi()->HasConnection(1));
        REQUIRE(GetDcgmApi()->HasConnection(2));
        auto connInfoOpt1 = GetDcgmApi()->GetConnectionInfo(1);
        auto connInfoOpt2 = GetDcgmApi()->GetConnectionInfo(2);
        REQUIRE(connInfoOpt1.has_value());
        REQUIRE(connInfoOpt2.has_value());
        auto const &connInfo1 = connInfoOpt1.value();
        auto const &connInfo2 = connInfoOpt2.value();

        // Count how many of each type we have
        int tcpCount = 0;
        int udsCount = 0;

        // Check first connection
        if (connInfo1.params.addressIsUnixSocket == 1)
        {
            udsCount++;
            REQUIRE(connInfo1.ipAddress == "/tmp/local_socket");
        }
        else
        {
            tcpCount++;
            REQUIRE(connInfo1.ipAddress == "127.0.0.1:12345"); // Port is included in the address
        }

        // Check second connection
        if (connInfo2.params.addressIsUnixSocket == 1)
        {
            udsCount++;
            REQUIRE(connInfo2.ipAddress == "/tmp/local_socket");
        }
        else
        {
            tcpCount++;
            REQUIRE(connInfo2.ipAddress == "127.0.0.1:12345"); // Port is included in the address
        }

        // Verify we have one of each type
        REQUIRE(tcpCount == 1);
        REQUIRE(udsCount == 1);
    }

    SECTION("ConnectRemoteNodes handles tunnel creation failure")
    {
        // Set up mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Set behavior for tunnel manager to simulate failure
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::GenericFailure);

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // Test with a host that will fail to create a tunnel
        std::vector<std::string> hostList = { "node1:5555" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, 1000); // uid 1000

        // Verify result indicates failure
        REQUIRE(result == DCGM_ST_REMOTE_SSH_CONNECTION_FAILED);

        // Verify tunnel manager was called
        auto startCalls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(startCalls.size() == 1);

        // Verify DCGM API was not called after tunnel failure
        REQUIRE(GetDcgmApi()->GetConnectCallCount() == 0);
    }

    SECTION("ConnectRemoteNodes handles DCGM connection failure")
    {
        // Set up mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Set default behavior for tunnel manager
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set behavior for DCGM API to simulate connection failure
        mockDcgmApi->SetConnectResult(DCGM_ST_CONNECTION_NOT_VALID);

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // Test with a host that will fail to connect to DCGM
        std::vector<std::string> hostList = { "node1:5555" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, 1000); // uid 1000

        // Verify result indicates connection failure
        REQUIRE(result == DCGM_ST_CONNECTION_NOT_VALID);

        // Verify tunnel manager was called
        auto startCalls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(startCalls.size() == 1);

        // Verify DCGM API Connect was called but failed
        REQUIRE(GetDcgmApi()->GetConnectCallCount() == 1);

        // Verify EndSession was called to clean up the tunnel
        auto endCalls = GetTcpSSHTunnelManager()->GetEndSessionCalls();
        REQUIRE(endCalls.size() == 1);
    }

    SECTION("ConnectRemoteNodes handles authorization failure")
    {
        // Set up mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Set default behavior for tunnel manager
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set behavior for DCGM API - connect succeeds but module send fails for authorization
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // Test with a host that will fail authorization
        std::vector<std::string> hostList = { "node1:5555" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, 1000); // uid 1000

        // Verify result indicates authorization failure
        REQUIRE(result == DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // Verify tunnel manager was called
        auto startCalls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(startCalls.size() == 1);

        // Verify DCGM API Connect was called successfully
        REQUIRE(GetDcgmApi()->GetConnectCallCount() == 1);

        // Verify DCGM API ModuleSendBlockingFixedRequest was called but failed
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 1);

        // Verify Disconnect was called to clean up
        REQUIRE(GetDcgmApi()->GetDisconnectCallCount() == 1);

        // Verify EndSession was called to clean up the tunnel
        auto endCalls = GetTcpSSHTunnelManager()->GetEndSessionCalls();
        REQUIRE(endCalls.size() == 1);
    }

    SECTION("DisconnectRemoteNodes properly disconnects all nodes")
    {
        // Set up mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockUdsSSHTunnelManager = std::make_unique<MockUdsSSHTunnelManager>();

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetUdsSSHTunnelManager(std::move(mockUdsSSHTunnelManager));

        // First set up some connections via ConnectRemoteNodes
        // We need to set up the mock behavior for connection
        GetDcgmApi()->SetConnectResult(DCGM_ST_OK);
        GetDcgmApi()->SetSendRequestResult(DCGM_ST_OK);
        GetTcpSSHTunnelManager()->SetDefaultRemotePort(12345);
        GetTcpSSHTunnelManager()->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);
        GetUdsSSHTunnelManager()->SetDefaultRemoteSocketPath("/tmp/local_socket");
        GetUdsSSHTunnelManager()->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Connect to a mix of TCP and UDS nodes
        std::vector<std::string> hostList = { "node1:5555", "node2:unix:///tmp/dcgm.socket" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, 1000); // uid 1000
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 2);

        // Reset the call counters for the disconnect phase
        GetDcgmApi()->Reset();
        GetTcpSSHTunnelManager()->Reset();
        GetUdsSSHTunnelManager()->Reset();

        // Now test DisconnectRemoteNodes
        result = DisconnectRemoteNodes();

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify DCGM API was called for each node (2 remote nodes)
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 2);
        REQUIRE(GetDcgmApi()->GetDisconnectCallCount() == 2);

        // Verify tunnel managers were called to end sessions
        auto tcpEndCalls = GetTcpSSHTunnelManager()->GetEndSessionCalls();
        auto udsEndCalls = GetUdsSSHTunnelManager()->GetEndSessionCalls();
        REQUIRE(tcpEndCalls.size() == 1);
        REQUIRE(udsEndCalls.size() == 1);

        // Verify the TCP and UDS end calls contain the expected parameters
        bool foundTcpEndCall = false;
        for (const auto &call : tcpEndCalls)
        {
            // Check for node1 with port 5555
            if (std::get<0>(call) == "node1" && std::get<1>(call) == 5555)
            {
                foundTcpEndCall = true;
                break;
            }
        }
        REQUIRE(foundTcpEndCall);

        bool foundUdsEndCall = false;
        for (const auto &call : udsEndCalls)
        {
            // Check for node2 with socket path
            if (std::get<0>(call) == "node2" && std::get<1>(call) == "/tmp/dcgm.socket")
            {
                foundUdsEndCall = true;
                break;
            }
        }
        REQUIRE(foundUdsEndCall);
    }

    SECTION("DisconnectRemoteNodes handles revocation failure")
    {
        // Set up mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // First set up a connection via ConnectRemoteNodes
        GetDcgmApi()->SetConnectResult(DCGM_ST_OK);
        GetDcgmApi()->SetSendRequestResult(DCGM_ST_OK);
        GetTcpSSHTunnelManager()->SetDefaultRemotePort(12345);
        GetTcpSSHTunnelManager()->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        std::vector<std::string> hostList = { "node1:5555" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, 1000);
        REQUIRE(result == DCGM_ST_OK);

        // Reset the API and set it to fail on revocation
        GetDcgmApi()->Reset();
        GetDcgmApi()->SetSendRequestResult(DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // Now test DisconnectRemoteNodes
        result = DisconnectRemoteNodes();

        // Verify result is the error from revocation
        REQUIRE(result == DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // Still verify that we tried to disconnect and end the tunnel session
        // Even with the revocation error
        REQUIRE(GetDcgmApi()->GetDisconnectCallCount() == 1);
        auto endCalls = GetTcpSSHTunnelManager()->GetEndSessionCalls();
        REQUIRE(endCalls.size() == 1);
    }
}

/**
 * Tests for the RunHeadNode method
 * Tests the main multi-node diagnostic execution function on the head node
 */
TEST_CASE_METHOD(MnDiagManagerTests, "RunHeadNode Tests [mndiag]")
{
    std::vector<unsigned int> fakeGpuIds;
    std::vector<dcgmcm_gpu_info_cached_t> fakeGpuInfos;
    auto skus = GetSupportedSkus();

    for (auto const &[idx, sku] : std::ranges::views::enumerate(skus))
    {
        unsigned int deviceId = 0;
        try
        {
            deviceId = std::stoul(sku, nullptr, 16); // Handles hex or decimal
        }
        catch (...)
        {
            deviceId = 0;
        }

        fakeGpuIds.push_back(idx);
        dcgmcm_gpu_info_cached_t info = {};
        info.gpuId                    = idx;
        info.pciInfo.pciDeviceId      = (deviceId << 16); // 0x2941 -> 0x29410000
        info.pciInfo.pciSubSystemId   = 0x10de0000;       // Use 8-digit value for ssid as well
        fakeGpuInfos.push_back(info);
    }

    SECTION("Successful execution of RunHeadNode")
    {
        // Create mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();

        // Add path capture for validation
        std::string capturedPath;
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        mockStateMachine->SetMnubergemmPathCallback([&capturedPath](std::string const &path) { capturedPath = path; });

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();

        // Configure StateMachine behavior using Config struct
        MockMnDiagStateMachine::Config config;
        config.shouldReserveReturn         = true;
        config.detectedProcessInfo         = { { 12345, "mnubergemm" } };
        config.notifyProcessDetectedReturn = true;
        mockStateMachine->SetConfig(config);

        // Configure tunnel manager behavior
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Configure DCGM API behavior
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);
        mockDcgmApi->SetReserveResourcesResponse(MnDiagStatus::RESERVED);
        mockDcgmApi->SetDetectProcessResponse(MnDiagStatus::RUNNING);

        // Configure MPI runner factory behavior
        // The mock factory's CreateMpiRunner method will return a MockMnDiagMpiRunner with default behaviors

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        dcgmMnDiagResponse_t response {};
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, 1000, response);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);

        // Verify ConnectRemoteNodes was called
        REQUIRE(GetDcgmApi()->GetConnectCallCount() > 0);

        // Verify ReserveRemoteResources was called
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() > 0);

        // Verify MPI runner methods were called
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().constructMpiCommandCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().launchMpiProcessCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().waitCount == 1);

        // Add new verification for mnubergemm path
        REQUIRE(capturedPath == MnDiagConstants::DEFAULT_MNUBERGEMM_PATH);
    }

    SECTION("Successful execution of RunHeadNode with custom mnubergemm path")
    {
        auto savedMnubergemmPath = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());


        // Create mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();

        // Add path capture for validation
        std::string capturedPath;
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        mockStateMachine->SetMnubergemmPathCallback([&capturedPath](std::string const &path) { capturedPath = path; });

        // Set env to custom path
        std::string customPath = "/bin/true";
        setenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), customPath.c_str(), 1);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();

        // Configure StateMachine behavior using Config struct
        MockMnDiagStateMachine::Config config;
        config.shouldReserveReturn         = true;
        config.detectedProcessInfo         = { { 12345, customPath } };
        config.notifyProcessDetectedReturn = true;
        mockStateMachine->SetConfig(config);

        // Configure tunnel manager behavior
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Configure DCGM API behavior
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);
        mockDcgmApi->SetReserveResourcesResponse(MnDiagStatus::RESERVED);
        mockDcgmApi->SetDetectProcessResponse(MnDiagStatus::RUNNING);

        // Configure MPI runner factory behavior
        // The mock factory's CreateMpiRunner method will return a MockMnDiagMpiRunner with default behaviors

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        dcgmMnDiagResponse_t response {};
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, 1000, response);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);

        // Verify ConnectRemoteNodes was called
        REQUIRE(GetDcgmApi()->GetConnectCallCount() > 0);

        // Verify ReserveRemoteResources was called
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() > 0);

        // Verify MPI runner methods were called
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().constructMpiCommandCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().launchMpiProcessCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().waitCount == 1);

        // Add new verification for mnubergemm path
        REQUIRE(capturedPath == customPath);

        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedMnubergemmPath);
    }

    SECTION("Failed to connect to remote nodes")
    {
        // Create mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();
        auto mockCoreProxy           = std::make_unique<MockDcgmCoreProxy>();

        // Configure tunnel manager to fail to create SSH tunnel
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::GenericFailure);

        // Configure MPI runner factory behavior
        // The mock factory's CreateMpiRunner method will return a MockMnDiagMpiRunner with default behaviors

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        dcgmMnDiagResponse_t response {};
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, 1000, response);

        // Verify results
        REQUIRE(result == DCGM_ST_REMOTE_SSH_CONNECTION_FAILED);

        // Verify MPI runner was never created
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 0);
    }

    SECTION("Failed to reserve resources")
    {
        // Create mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();

        // Configure StateMachine behavior using Config struct
        MockMnDiagStateMachine::Config config;
        config.shouldReserveReturn = false; // Will cause reservation failure
        mockStateMachine->SetConfig(config);

        // Configure tunnel manager behavior
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Configure DCGM API behavior
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);
        mockDcgmApi->SetReserveResourcesResponse(MnDiagStatus::FAILED);

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        dcgmMnDiagResponse_t response {};
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, 1000, response);

        // Verify results
        REQUIRE(result == DCGM_ST_IN_USE);

        // Verify MPI runner was never created
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 0);
    }

    SECTION("Failed to launch MPI process")
    {
        // Create mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();

        // Configure StateMachine behavior using Config struct
        MockMnDiagStateMachine::Config config;
        config.shouldReserveReturn = true; // Resources can be reserved
        mockStateMachine->SetConfig(config);

        // Configure tunnel manager behavior
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Configure DCGM API behavior
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);
        mockDcgmApi->SetReserveResourcesResponse(MnDiagStatus::RESERVED);

        // Configure MPI runner to fail to launch
        // We need to subclass MockMnDiagMpiRunnerFactory to override CreateMpiRunner
        class CustomMockMpiRunnerFactory : public MockMnDiagMpiRunnerFactory
        {
        public:
            std::unique_ptr<MnDiagMpiRunnerBase> CreateMpiRunner(DcgmCoreProxyBase & /* coreProxy */) override
            {
                m_createMpiRunnerCount++;
                auto runner                = std::make_unique<MockMnDiagMpiRunner>();
                runner->m_mockLaunchResult = DCGM_ST_GENERIC_ERROR;
                // Store the runner ID before returning it
                m_lastRunnerId = static_cast<MockMnDiagMpiRunner *>(runner.get())->GetRunnerId();
                return runner;
            }
        };

        auto customMpiRunnerFactory = std::make_unique<CustomMockMpiRunnerFactory>();

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(customMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        dcgmMnDiagResponse_t response {};
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, 1000, response);

        // Verify results
        REQUIRE(result == DCGM_ST_GENERIC_ERROR);

        // Verify MPI runner was created but launch failed
        auto factory = static_cast<CustomMockMpiRunnerFactory *>(GetMpiRunnerFactory());
        REQUIRE(factory->m_createMpiRunnerCount == 1);
        REQUIRE(factory->GetLastRunnerStats().launchMpiProcessCount == 1);
        REQUIRE(factory->GetLastRunnerStats().waitCount == 0); // Wait should not be called if launch fails
    }

    SECTION("Failed to get node info")
    {
        // Create mock objects
        auto mockDcgmApi             = std::make_unique<MockDcgmApi>();
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();

        // Configure StateMachine behavior using Config struct
        MockMnDiagStateMachine::Config config;
        config.shouldReserveReturn         = true;
        config.detectedProcessInfo         = { { 12345, "mnubergemm" } };
        config.notifyProcessDetectedReturn = true;
        mockStateMachine->SetConfig(config);

        // Configure tunnel manager behavior
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Configure DCGM API behavior
        mockDcgmApi->SetConnectResult(DCGM_ST_OK);
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);
        mockDcgmApi->SetReserveResourcesResponse(MnDiagStatus::RESERVED);
        mockDcgmApi->SetDetectProcessResponse(MnDiagStatus::RUNNING);

        // Configure MPI runner factory behavior
        // The mock factory's CreateMpiRunner method will return a MockMnDiagMpiRunner with default behaviors

        // Install mock objects
        SetDcgmApi(std::move(mockDcgmApi));
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        dcgmMnDiagResponse_t response {};
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Set the mock driver version result to NVML_ERROR. This will cause the GetNodeInfo call to fail.
        dcgmReturn_t driverVersionResult = DCGM_ST_NVML_ERROR;
        GetDcgmCoreProxy()->SetMockDriverVersionResult(driverVersionResult);
        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, 1000, response);

        // Verify results. This should be the same as the mock driver version result since that is a
        // fatal error and will halt execution
        REQUIRE(result == driverVersionResult);

        // Verify ConnectRemoteNodes was called
        REQUIRE(GetDcgmApi()->GetConnectCallCount() > 0);

        // Verify only one request was made - authorize connection
        //REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 1);
        REQUIRE(GetDcgmApi()->GetRequestTypeCount(dcgmMultinodeRequestType_t::AuthorizeConnection) == 1);
        REQUIRE(GetDcgmApi()->GetRequestTypeCount(dcgmMultinodeRequestType_t::RevokeAuthorization) == 1);
        REQUIRE(GetDcgmApi()->GetRequestTypeCount(dcgmMultinodeRequestType_t::GetNodeInfo) == 0);
        REQUIRE(GetDcgmApi()->GetRequestTypeCount(dcgmMultinodeRequestType_t::ReserveResources) == 0);
        REQUIRE(GetDcgmApi()->GetRequestTypeCount(dcgmMultinodeRequestType_t::ReleaseResources) == 0);
        REQUIRE(GetDcgmApi()->GetRequestTypeCount(dcgmMultinodeRequestType_t::DetectProcess) == 0);

        // Verify MPI runner methods were not called
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 0);
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "AreDevicesSupported [mndiag]")
{
    // Helper: create a gpu_info struct
    auto make_gpu_info = [](unsigned int gpuId, unsigned int device, unsigned int ssid = 0x0000) {
        dcgmcm_gpu_info_cached_t info {};
        info.gpuId                  = gpuId;
        info.pciInfo.pciDeviceId    = device;
        info.pciInfo.pciSubSystemId = ssid;
        return info;
    };

    SECTION("All devices supported (Same device IDs)")
    {
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));

        std::vector<unsigned int> gpuIds    = { 0, 1 };
        std::vector<unsigned int> deviceIds = { 0x29410000, 0x29410000 };
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;

        for (size_t i = 0; i < gpuIds.size(); ++i)
            gpuInfos.push_back(make_gpu_info(gpuIds[i], deviceIds[i]));

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        REQUIRE(AreDevicesSupported() == true);
    }

    SECTION("Devices not supported (no match)")
    {
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));

        std::vector<unsigned int> gpuIds    = { 0, 1 };
        std::vector<unsigned int> deviceIds = { 0x12340000, 0x12340000 };

        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
        for (size_t i = 0; i < gpuIds.size(); ++i)
            gpuInfos.push_back(make_gpu_info(gpuIds[i], deviceIds[i]));

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        REQUIRE(AreDevicesSupported() == false);
    }

    SECTION("One device supported")
    {
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));

        std::vector<unsigned int> gpuIds    = { 0, 1 };
        std::vector<unsigned int> deviceIds = { 0x12340000, 0x29410000 };

        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
        for (size_t i = 0; i < gpuIds.size(); ++i)
            gpuInfos.push_back(make_gpu_info(gpuIds[i], deviceIds[i]));

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(false);

        REQUIRE(AreDevicesSupported() == false);
    }

    SECTION("No GPUs present")
    {
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));

        std::vector<unsigned int> gpuIds    = {};
        std::vector<unsigned int> deviceIds = {};
        std::vector<unsigned int> ssids     = {};
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        REQUIRE(AreDevicesSupported() == false);
    }


    SECTION("GPU IDs present but no GPU info")
    {
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));

        std::vector<unsigned int> gpuIds    = { 0, 1, 2 };
        std::vector<unsigned int> deviceIds = {};
        std::vector<unsigned int> ssids     = {};
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(false);

        REQUIRE(AreDevicesSupported() == false);
    }

    SECTION("GPU info present but no GPU IDs")
    {
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));

        std::vector<unsigned int> gpuIds    = {};
        std::vector<unsigned int> deviceIds = { 0x29410000, 0x29410000 };
        std::vector<unsigned int> ssids     = { 0x10de0000, 0x10de0000 };
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
        for (size_t i = 0; i < deviceIds.size(); ++i)
            gpuInfos.push_back(make_gpu_info(i, deviceIds[i], ssids[i]));

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        REQUIRE(AreDevicesSupported() == false);
    }

    SECTION("GPU info and GPU Ids present")
    {
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));

        std::vector<unsigned int> gpuIds    = { 0, 1 };
        std::vector<unsigned int> deviceIds = { 0x29410000, 0x29410000 };
        std::vector<unsigned int> ssids     = { 0x10de0000, 0x10de0000 };
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
        for (size_t i = 0; i < deviceIds.size(); ++i)
            gpuInfos.push_back(make_gpu_info(i, deviceIds[i], ssids[i]));

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        REQUIRE(AreDevicesSupported() == true);
    }

    SECTION("Devices supported (different device IDs) - fail")
    {
        // Add additional SKUs
        setenv("DCGM_MNDIAG_SUPPORTED_SKUS", "1240", 1);
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 2);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("1240"));

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));

        std::vector<unsigned int> gpuIds    = { 0, 1 };
        std::vector<unsigned int> deviceIds = { 0x29410000, 0x12400000 };

        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
        for (size_t i = 0; i < gpuIds.size(); ++i)
            gpuInfos.push_back(make_gpu_info(gpuIds[i], deviceIds[i]));

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(false);

        REQUIRE(AreDevicesSupported() == false);
        unsetenv("DCGM_MNDIAG_SUPPORTED_SKUS");
    }

    SECTION("Devices supported from env and default")
    {
        setenv("DCGM_MNDIAG_SUPPORTED_SKUS", "1240", 1);
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 2);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("1240"));

        // Setup mock for both SKUs
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));
        std::vector<unsigned int> gpuIds    = { 0, 1, 2 };
        std::vector<unsigned int> deviceIds = { 0x12400000, 0x12400000, 0x12400000 };
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
        for (size_t i = 0; i < deviceIds.size(); ++i)
            gpuInfos.push_back(make_gpu_info(i, deviceIds[i]));

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        REQUIRE(AreDevicesSupported() == true);
        unsetenv("DCGM_MNDIAG_SUPPORTED_SKUS");
    }

    SECTION("Device not present in SKU (env + default)")
    {
        setenv("DCGM_MNDIAG_SUPPORTED_SKUS", "1240", 1);
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 2);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("1240"));

        // Setup mock for both SKUs
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));
        std::vector<unsigned int> gpuIds    = { 0, 1 };
        std::vector<unsigned int> deviceIds = { 0x11120000, 0x11120000 };
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
        for (size_t i = 0; i < deviceIds.size(); ++i)
            gpuInfos.push_back(make_gpu_info(i, deviceIds[i]));

        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(gpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        REQUIRE(AreDevicesSupported() == false);
        unsetenv("DCGM_MNDIAG_SUPPORTED_SKUS");
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "LoadSupportedSkusFromEnv [mndiag]")
{
    SECTION("Default (no env var set)")
    {
        unsetenv("DCGM_MNDIAG_SUPPORTED_SKUS");
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 1);
        REQUIRE(skus.contains("2941"));
    }

    SECTION("Multiple SKUs in env")
    {
        setenv("DCGM_MNDIAG_SUPPORTED_SKUS", "abcd, 1234, 5678 ", 1);
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 4);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("abcd"));
        REQUIRE(skus.contains("1234"));
        REQUIRE(skus.contains("5678"));
        unsetenv("DCGM_MNDIAG_SUPPORTED_SKUS");
    }

    SECTION("Mixedcase SKUs in env are stored as lowercase")
    {
        setenv("DCGM_MNDIAG_SUPPORTED_SKUS", "24B9,1h34,xY5Z", 1);
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 4);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("24b9"));
        REQUIRE(skus.contains("1h34"));
        REQUIRE(skus.contains("xy5z"));
        unsetenv("DCGM_MNDIAG_SUPPORTED_SKUS");
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "HandleBroadcastRunParameters [mndiag]")
{
    SECTION("Valid parameter broadcast with time_to_run")
    {
        // Setup mock state machine to capture timeout setting
        bool timeoutWasSet                   = false;
        std::chrono::seconds capturedTimeout = std::chrono::seconds(0);
        std::string capturedPath;

        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

        mockStateMachine->SetProcessExecutionTimeoutCallback(
            [&timeoutWasSet, &capturedTimeout](std::chrono::seconds timeout) {
                timeoutWasSet   = true;
                capturedTimeout = timeout;
            });

        mockStateMachine->SetMnubergemmPathCallback([&capturedPath](std::string const &path) { capturedPath = path; });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version1;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with time_to_run
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run=300");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[1], "mnubergemm.other_param=value");

        // Set the mnubergemm path
        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.mnubergemmPath, testPath.c_str());

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(timeoutWasSet == true);
        REQUIRE(capturedTimeout == std::chrono::seconds(300));
        REQUIRE(capturedPath == testPath);
    }

    SECTION("Valid parameter broadcast without time_to_run")
    {
        // Setup mock state machine
        bool timeoutWasSet = false;
        std::string capturedPath;

        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

        mockStateMachine->SetProcessExecutionTimeoutCallback(
            [&timeoutWasSet](std::chrono::seconds /* timeout */) { timeoutWasSet = true; });

        mockStateMachine->SetMnubergemmPathCallback([&capturedPath](std::string const &path) { capturedPath = path; });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version1;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters without time_to_run
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.other_param=value");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[1], "mnubergemm.flag");

        // Set the mnubergemm path
        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.mnubergemmPath, testPath.c_str());

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(timeoutWasSet == false); // No timeout should be set
        REQUIRE(capturedPath == testPath);
    }

    SECTION("Invalid time_to_run parameter format")
    {
        // Setup mock state machine
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version1;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with invalid time_to_run format (missing value)
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run");

        // Set the mnubergemm path
        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.mnubergemmPath, testPath.c_str());

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Multiple time_to_run parameters - uses first one")
    {
        // Setup mock state machine to capture timeout setting
        bool timeoutWasSet                   = false;
        std::chrono::seconds capturedTimeout = std::chrono::seconds(0);
        std::string capturedPath;

        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

        mockStateMachine->SetProcessExecutionTimeoutCallback(
            [&timeoutWasSet, &capturedTimeout](std::chrono::seconds timeout) {
                timeoutWasSet   = true;
                capturedTimeout = timeout;
            });

        mockStateMachine->SetMnubergemmPathCallback([&capturedPath](std::string const &path) { capturedPath = path; });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version1;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with multiple time_to_run values
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run=100");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[1], "mnubergemm.time_to_run=200");

        // Set the mnubergemm path
        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.mnubergemmPath, testPath.c_str());

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results - should use the first value
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(timeoutWasSet == true);
        REQUIRE(capturedTimeout == std::chrono::seconds(100));
        REQUIRE(capturedPath == testPath);
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "BroadcastRunParametersToRemoteNodes [mndiag]")
{
    SECTION("Successful broadcast to multiple remote nodes")
    {
        // Setup mock DCGM API
        auto mockDcgmApi = std::make_unique<MockDcgmApi>();
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);
        SetDcgmApi(std::move(mockDcgmApi));

        // Setup mock connections
        SetupMockConnections(3); // 3 remote connections

        // Create test parameters
        dcgmRunMnDiag_t params {};
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.testParms[0], "mnubergemm.time_to_run=300");
        SafeCopyTo(params.hostList[0], "node1");
        SafeCopyTo(params.hostList[1], "node2");

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);

        // Verify that MultinodeRequest was called for all remote nodes
        // (3 remote nodes, no loopback connections in this test)
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 3);
    }

    SECTION("Broadcast failure on one remote node")
    {
        // Setup mock DCGM API to fail on specific handle
        auto mockDcgmApi = std::make_unique<MockDcgmApi>();

        // Set default success
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);

        // Set failure for handle 11 (second remote node)
        mockDcgmApi->SetHandleCommandCallback(
            11,
            BroadcastRunParameters,
            [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t * /* request */) -> dcgmReturn_t {
                return DCGM_ST_CONNECTION_NOT_VALID;
            });

        SetDcgmApi(std::move(mockDcgmApi));

        // Setup mock connections
        SetupMockConnections(3); // 3 remote connections

        // Create test parameters
        dcgmRunMnDiag_t params {};
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.testParms[0], "mnubergemm.time_to_run=300");
        SafeCopyTo(params.hostList[0], "node1");
        SafeCopyTo(params.hostList[1], "node2");

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results - should fail due to one node failure
        REQUIRE(result == DCGM_ST_CONNECTION_NOT_VALID);

        // Verify that MultinodeRequest was called for all remote nodes
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 3);
    }

    SECTION("Broadcast with no remote connections (loopback only)")
    {
        // Setup mock DCGM API
        auto mockDcgmApi = std::make_unique<MockDcgmApi>();
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);
        SetDcgmApi(std::move(mockDcgmApi));

        // Setup only a loopback connection
        SetupLoopbackConnection();

        // Create test parameters
        dcgmRunMnDiag_t params {};
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.testParms[0], "mnubergemm.time_to_run=300");
        SafeCopyTo(params.hostList[0], "localhost");

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);

        // Verify that no remote MultinodeRequest calls were made (only loopback)
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 0);
    }

    SECTION("Broadcast with empty parameters")
    {
        // Setup mock DCGM API
        auto mockDcgmApi = std::make_unique<MockDcgmApi>();
        mockDcgmApi->SetSendRequestResult(DCGM_ST_OK);
        SetDcgmApi(std::move(mockDcgmApi));

        // Setup mock connections
        SetupMockConnections(2); // 2 remote connections

        // Create empty test parameters
        dcgmRunMnDiag_t params {};
        // Leave all fields empty/default

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results - should still succeed with empty parameters
        REQUIRE(result == DCGM_ST_OK);

        // Verify that MultinodeRequest was called for both remote nodes
        REQUIRE(GetDcgmApi()->GetSendRequestCallCount() == 2);
    }

    SECTION("Verify request structure is populated correctly - custom mnubergemm path")
    {
        // Save current environment state
        auto savedPath         = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());
        std::string customPath = "/bin/true";
        setenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), customPath.c_str(), 1);

        // Setup mock DCGM API with callback to inspect request
        auto mockDcgmApi = std::make_unique<MockDcgmApi>();

        dcgmMultinodeRequest_t capturedRequest {};
        bool requestCaptured = false;

        mockDcgmApi->SetHandleCommandCallback(
            10,
            BroadcastRunParameters,
            [&capturedRequest, &requestCaptured](dcgmHandle_t /* handle */,
                                                 dcgmMultinodeRequest_t *request) -> dcgmReturn_t {
                capturedRequest = *request;
                requestCaptured = true;
                return DCGM_ST_OK;
            });

        SetDcgmApi(std::move(mockDcgmApi));

        // Setup mock connections
        SetupMockConnections(2); // 1 remote + 1 loopback

        // Create test parameters
        dcgmRunMnDiag_t params {};
        SafeCopyTo(params.testName, "test_diagnostic");
        SafeCopyTo(params.testParms[0], "param1=value1");
        SafeCopyTo(params.testParms[1], "param2=value2");
        SafeCopyTo(params.hostList[0], "node1");

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(requestCaptured == true);

        // Verify request structure
        REQUIRE(capturedRequest.version == dcgmMultinodeRequest_version1);
        REQUIRE(capturedRequest.testType == MnDiagTestType::mnubergemm);
        REQUIRE(capturedRequest.requestType == MnDiagRequestType::BroadcastRunParameters);

        // Verify the parameters were copied correctly
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testName) == "test_diagnostic");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testParms[0]) == "param1=value1");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testParms[1]) == "param2=value2");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.hostList[0]) == "node1");

        // Verify custom mnubergemm path is used
        REQUIRE(std::string(capturedRequest.requestData.runParams.mnubergemmPath) == customPath);

        // Restore environment
        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedPath);
    }

    SECTION("Verify request structure is populated correctly - default mnubergemm path")
    {
        // Save current environment state
        auto savedPath = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());
        unsetenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());

        // Setup mock DCGM API with callback to inspect request
        auto mockDcgmApi = std::make_unique<MockDcgmApi>();

        dcgmMultinodeRequest_t capturedRequest {};
        bool requestCaptured = false;

        mockDcgmApi->SetHandleCommandCallback(
            10,
            BroadcastRunParameters,
            [&capturedRequest, &requestCaptured](dcgmHandle_t /* handle */,
                                                 dcgmMultinodeRequest_t *request) -> dcgmReturn_t {
                capturedRequest = *request;
                requestCaptured = true;
                return DCGM_ST_OK;
            });

        SetDcgmApi(std::move(mockDcgmApi));

        // Setup mock connections
        SetupMockConnections(2); // 1 remote + 1 loopback

        // Create test parameters
        dcgmRunMnDiag_t params {};
        SafeCopyTo(params.testName, "test_diagnostic");
        SafeCopyTo(params.testParms[0], "param1=value1");
        SafeCopyTo(params.testParms[1], "param2=value2");
        SafeCopyTo(params.hostList[0], "node1");

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(requestCaptured == true);

        // Verify request structure
        REQUIRE(capturedRequest.version == dcgmMultinodeRequest_version1);
        REQUIRE(capturedRequest.testType == MnDiagTestType::mnubergemm);
        REQUIRE(capturedRequest.requestType == MnDiagRequestType::BroadcastRunParameters);

        // Verify the parameters were copied correctly
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testName) == "test_diagnostic");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testParms[0]) == "param1=value1");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testParms[1]) == "param2=value2");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.hostList[0]) == "node1");

        // Verify default mnubergemm path is used
        REQUIRE(std::string(capturedRequest.requestData.runParams.mnubergemmPath)
                == MnDiagConstants::DEFAULT_MNUBERGEMM_PATH);

        // Restore environment
        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedPath);
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "GetMnubergemmPathHeadNode Tests [mndiag]")
{
    SECTION("Should use default path when no environment variable is set")
    {
        // Save current environment state
        auto savedPath = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());
        unsetenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());

        // Call the method and verify path
        std::string path = GetMnubergemmPathHeadNode();
        REQUIRE(path == MnDiagConstants::DEFAULT_MNUBERGEMM_PATH);

        // Restore environment
        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedPath);
    }

    SECTION("Should use custom path when environment variable points to valid executable")
    {
        // Save current environment state
        auto savedPath = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());

        // Use a known executable that exists
        std::string customPath = "/bin/true";
        setenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), customPath.c_str(), 1);

        // Call the method and verify path
        std::string path = GetMnubergemmPathHeadNode();
        REQUIRE(path == customPath);

        // Restore environment
        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedPath);
    }

    SECTION("Should fallback to default path when environment variable is empty")
    {
        // Save current environment state
        auto savedPath = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());

        setenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), "", 1);

        // Call the method and verify path
        std::string path = GetMnubergemmPathHeadNode();
        REQUIRE(path == MnDiagConstants::DEFAULT_MNUBERGEMM_PATH);

        // Restore environment
        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedPath);
    }

    SECTION("Should fallback to default path when environment variable points to non-existent file")
    {
        // Save current environment state
        auto savedPath = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());

        std::string nonExistentPath = "/path/to/nonexistent/binary";
        setenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), nonExistentPath.c_str(), 1);

        // Call the method and verify path
        std::string path = GetMnubergemmPathHeadNode();
        REQUIRE(path == MnDiagConstants::DEFAULT_MNUBERGEMM_PATH);

        // Restore environment
        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedPath);
    }

    SECTION("Should fallback to default path when environment variable points to non-executable file")
    {
        // Save current environment state
        auto savedPath = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());

        std::string nonExecutablePath = "/dev/null"; // Exists but not executable
        setenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), nonExecutablePath.c_str(), 1);

        // Call the method and verify path
        std::string path = GetMnubergemmPathHeadNode();
        REQUIRE(path == MnDiagConstants::DEFAULT_MNUBERGEMM_PATH);

        // Restore environment
        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedPath);
    }

    SECTION("Should fallback to default path when environment variable points to directory")
    {
        // Save current environment state
        auto savedPath = saveEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());

        std::string directoryPath = "/tmp"; // A directory, not a file
        setenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), directoryPath.c_str(), 1);

        // Call the method and verify path
        std::string path = GetMnubergemmPathHeadNode();
        REQUIRE(path == MnDiagConstants::DEFAULT_MNUBERGEMM_PATH);

        // Restore environment
        restoreEnvVar(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), savedPath);
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "PopulateMpiFailureResponseStruct [mndiag]")
{
    // Prepare a mock MPI runner
    auto mockRunner                = std::make_unique<MockMnDiagMpiRunner>();
    dcgmMnDiagResponse_v1 response = {};
    response.version               = dcgmMnDiagResponse_version1;
    response.numTests              = 1;

    // Prepare log paths
    DcgmNs::Utils::LogPaths logPaths;
    logPaths.stdoutFileName = "/tmp/stdout.log";
    logPaths.stderrFileName = "/tmp/stderr.log";

    SECTION("Sets failure info when exit code is non-zero")
    {
        mockRunner->m_mockPopulateResponseResult = DCGM_ST_OK;
        mockRunner->m_mockExitCode               = 42;

        PopulateMpiFailureResponseStruct(mockRunner.get(), response, 1234, logPaths);

        REQUIRE(response.tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(std::string(response.tests[0].auxData.data).find("MPI exited with code: 42") != std::string::npos);
        REQUIRE(std::string(response.tests[0].auxData.data).find("/tmp/stdout.log") != std::string::npos);
        REQUIRE(std::string(response.tests[0].auxData.data).find("/tmp/stderr.log") != std::string::npos);
        REQUIRE(response.tests[0].auxData.version == dcgmMnDiagTestAuxData_version1);
    }

    SECTION("Does not set failure info when exit code is zero")
    {
        mockRunner->m_mockPopulateResponseResult = DCGM_ST_OK;
        mockRunner->m_mockExitCode               = 0;

        PopulateMpiFailureResponseStruct(mockRunner.get(), response, 1234, logPaths);

        REQUIRE(response.tests[0].result != DCGM_DIAG_RESULT_FAIL);
        REQUIRE(std::string(response.tests[0].auxData.data).empty());
    }

    SECTION("Handles missing exit code gracefully")
    {
        mockRunner->m_mockPopulateResponseResult = DCGM_ST_OK;
        mockRunner->m_mockExitCode.reset();

        PopulateMpiFailureResponseStruct(mockRunner.get(), response, 1234, logPaths);

        REQUIRE(response.tests[0].result != DCGM_DIAG_RESULT_FAIL);
        REQUIRE(std::string(response.tests[0].auxData.data).empty());
    }
}
