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

#include "DcgmRequest.h"
#include "dcgm_module_structs.h"
#include "dcgm_test_apis.h"
#include <DcgmMnDiagManager.h>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <MockDcgmOps.h>
#include <ScopedEnvVars.hpp>
#include <TempDir.hpp>
#include <TempFile.hpp>
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <dcgm_nvml.h>
#include <dcgm_structs.h>
#include <fmt/format.h>
#include <memory>
#include <ranges>
#include <vector>

#include "MockDcgmCoreProxy.h"
#include "MockDcgmResourceHandle.h"
#include "MockMnDiagMpiRunner.h"
#include "MockMnDiagStateMachine.h"
#include "MockTcpSSHTunnelManager.h"
#include "MockUdsSSHTunnelManager.h"
#include "UniquePtrUtil.h"

// Force a UID of 0, corresponding to the root user. This is so that there will
// always be a username associated with the UID
static constexpr uid_t MNDIAG_MANAGER_EFFECTIVE_UID = 0;

std::expected<dcgmMultinodeTestType_t, dcgmReturn_t> GetTestTypeFromName(std::string_view testName);

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

TEST_CASE("DcgmMnDiagManager GetTestTypeFromName")
{
    SECTION("Known tests map to stable multinode test types")
    {
        auto mnubergemmType = GetTestTypeFromName("mnubergemm");
        REQUIRE(mnubergemmType.has_value());
        CHECK(*mnubergemmType == dcgmMultinodeTestType_t::mnubergemm);

        auto mnnvbandwidthType = GetTestTypeFromName("mnnvbandwidth");
        REQUIRE(mnnvbandwidthType.has_value());
        CHECK(*mnnvbandwidthType == dcgmMultinodeTestType_t::mnnvbandwidth);
    }

    SECTION("Unknown tests are rejected")
    {
        auto type = GetTestTypeFromName("unknown");
        REQUIRE_FALSE(type.has_value());
        CHECK(type.error() == DCGM_ST_BADPARAM);
    }
}

/**
 * Fixture class for MnDiagManager tests
 * This provides direct access to the manager for testing
 */
class MnDiagManagerTests
{
protected:
    dcgmCoreCallbacks_t m_callbacks;
    DcgmNs::MockDcgmOps m_mockDcgmOps;
    DcgmMnDiagManager m_manager;

public:
    MnDiagManagerTests()
        : m_callbacks(GetCallbacks())
        , m_manager(m_callbacks, &m_mockDcgmOps)
    {
        m_mockDcgmOps.UseGeneratedConnectHandles();
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
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

    dcgmReturn_t ReserveRemoteResources(dcgmMultinodeTestType_t testType = dcgmMultinodeTestType_t::mnubergemm)
    {
        return m_manager.ReserveRemoteResources(testType);
    }

    dcgmReturn_t ReleaseRemoteResources(dcgmMultinodeTestType_t testType = dcgmMultinodeTestType_t::mnubergemm)
    {
        return m_manager.ReleaseRemoteResources(testType);
    }

    dcgmReturn_t GetNodeInfo(dcgmMultinodeTestType_t testType = dcgmMultinodeTestType_t::mnubergemm)
    {
        return m_manager.GetNodeInfo(testType);
    }

    nodeInfoMap_t const &GetNodeInfoMap() const
    {
        return m_manager.m_nodeInfo;
    }

    dcgmReturn_t AuthorizeRemoteConnections(dcgmMultinodeTestType_t testType = dcgmMultinodeTestType_t::mnubergemm)
    {
        return m_manager.AuthorizeRemoteConnections(testType);
    }

    dcgmReturn_t RevokeRemoteAuthorizations()
    {
        return m_manager.RevokeRemoteAuthorizations();
    }

    dcgmReturn_t CleanupConnections()
    {
        return m_manager.CleanupConnections();
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
                                          dcgmMnDiagResponse_v2 &response,
                                          std::optional<pid_t> mpiPID,
                                          DcgmNs::Utils::LogPaths const &logPaths)
    {
        return m_manager.PopulateMpiFailureResponseStruct(mpiRunner, response, mpiPID, logPaths);
    }

    std::unordered_set<std::string> GetSupportedSkus()
    {
        return m_manager.GetSupportedSkus();
    }

    int GetCudaVersion()
    {
        return m_manager.GetCudaVersion();
    }

    /**
     * @brief Set process detection parameters for testing
     *
     * @param timeout Timeout for process detection
     * @param retryInterval Interval between retry attempts
     */
    void SetProcessDetectionParams(std::chrono::seconds timeout, std::chrono::milliseconds retryInterval)
    {
        m_manager.m_processDetectionTimeout       = timeout;
        m_manager.m_processDetectionRetryInterval = retryInterval;
    }

    std::vector<pid_t> FindSshZombieProcesses(std::filesystem::path const &procPath)
    {
        return m_manager.FindSshZombieProcesses(procPath);
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

    void SetCurrentTestInfo(TestInfo testInfo)
    {
        m_manager.m_currentTestInfo = std::move(testInfo);
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

    void ResetMockOps()
    {
        m_mockDcgmOps.ResetApiState();
        m_mockDcgmOps.UseGeneratedConnectHandles();
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
    }

    void SetResourceResponse(dcgmMultinodeRequestType_t requestType, MnDiagStatus response)
    {
        m_mockDcgmOps.SetMultinodeRequestHandler(
            requestType, [this, response](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t *request) {
                request->requestData.resource.response = response;
                return m_mockDcgmOps.GetMultinodeRequestResult();
            });
    }

    void SetReserveResourcesResponse(MnDiagStatus response)
    {
        SetResourceResponse(dcgmMultinodeRequestType_t::ReserveResources, response);
    }

    void SetReleaseResourcesResponse(MnDiagStatus response)
    {
        SetResourceResponse(dcgmMultinodeRequestType_t::ReleaseResources, response);
    }

    void SetDetectProcessResponse(MnDiagStatus response)
    {
        SetResourceResponse(dcgmMultinodeRequestType_t::DetectProcess, response);
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
            connInfo.uid              = MNDIAG_MANAGER_EFFECTIVE_UID;
            connInfo.remoteSocketPath = "";

            m_manager.m_connections[hostname] = std::move(connInfo);
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
        connInfo.uid              = MNDIAG_MANAGER_EFFECTIVE_UID;
        connInfo.remoteSocketPath = "";

        m_manager.m_connections["localhost"] = std::move(connInfo);
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

    void SetMockCudaVersion(int mockCudaVersion = 12000)
    {
        m_mockCudaVersion   = mockCudaVersion;
        m_systemCudaVersion = NVML_CUDA_DRIVER_VERSION_MAJOR(mockCudaVersion);
        m_mockPath          = fmt::format("cuda{}/mnubergemm", m_systemCudaVersion);
    }

    void InitMockCuda()
    {
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        SetCoreProxy(std::move(mockCoreProxy));
        GetDcgmCoreProxy()->SetMockCudaVersion(m_mockCudaVersion);
    }

    std::string GetMockPath()
    {
        return m_mockPath;
    }

    void SetMockPath(std::string const &path)
    {
        m_mockPath = path;
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

    int m_mockCudaVersion   = 0;
    int m_systemCudaVersion = 0;
    std::string m_mockPath {};
};

/**
 * Simple RAII helper for managing mock /proc directories
 *
 * Automatically creates and cleans up mock /proc directory structures.
 * The directory is cleaned up when the object goes out of scope.
 */
class MockProcDirectory
{
public:
    explicit MockProcDirectory(std::string prefix)
        : m_dir(TempDir::Random(std::move(prefix).append("-")))
    {}

    MockProcDirectory(MockProcDirectory const &)            = delete;
    MockProcDirectory &operator=(MockProcDirectory const &) = delete;
    MockProcDirectory(MockProcDirectory &&)                 = delete;
    MockProcDirectory &operator=(MockProcDirectory &&)      = delete;

    std::filesystem::path const &GetPath() const
    {
        return m_dir.Path();
    }

    void CreateTestDirectory(std::filesystem::path const &path) const
    {
        std::error_code ec;
        if (!std::filesystem::create_directories(path, ec) && ec)
        {
            FAIL("Failed to create test directory '" << path << "': " << ec.message());
        }
    }

    void CreateProcesses(std::vector<std::tuple<pid_t, std::string, char, pid_t>> const &processes)
    {
        for (auto const &[pid, comm, state, ppid] : processes)
        {
            auto pidDir = m_dir.Path() / std::to_string(pid);
            CreateTestDirectory(pidDir);

            auto statFile = pidDir / "stat";
            m_statFiles.emplace_back(TempFile::MakeUniqueFixed(
                statFile, fmt::format("{} {} {} {} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n", pid, comm, state, ppid)));
        }
    }

private:
    TempDir m_dir;
    std::vector<std::unique_ptr<TempFile>> m_statFiles;
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

            // Use short timeout for testing to avoid 60-second wait in retry loop
            SetProcessDetectionParams(std::chrono::seconds(1), std::chrono::milliseconds(100));

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

            // Configure the mock operations to return success for all requests
            m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
            SetReserveResourcesResponse(MnDiagStatus::RESERVED);

            // Call the method under test
            dcgmReturn_t result = ReserveRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::RESERVED);
            REQUIRE(GetStateMachine()->GetStats().notifyToReserveCalled == true);

            // Verify the API was called for each remote connection (3 connections)
            REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 3);
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
            m_mockDcgmOps.SetMultinodeRequestHandler(11,
                                                     dcgmMultinodeRequestType_t::ReserveResources,
                                                     [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t *request) {
                                                         request->requestData.resource.response = MnDiagStatus::FAILED;
                                                         return DCGM_ST_IN_USE;
                                                     });

            // All other requests succeed
            m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
            SetReserveResourcesResponse(MnDiagStatus::RESERVED);

            // Call the method under test
            dcgmReturn_t result = ReserveRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_IN_USE);

            // The API should have been called for only the failing connection before exiting
            // The implementation should stop processing after the first failure
            REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() > 0);
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

            // Configure the mock operations to return success for all requests
            m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
            SetReleaseResourcesResponse(MnDiagStatus::READY);

            // Call the method under test
            dcgmReturn_t result = ReleaseRemoteResources();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);
            REQUIRE(GetStatus() == MnDiagStatus::READY);
            REQUIRE(GetStateMachine()->GetStats().notifyDiagnosticFinishedCalled == true);

            // Verify the API was called for each remote connection (3 connections)
            REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 3);
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
            m_mockDcgmOps.SetMultinodeRequestHandler(
                11,
                dcgmMultinodeRequestType_t::ReleaseResources,
                [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t * /* request */) {
                    return DCGM_ST_CONNECTION_NOT_VALID;
                });

            // All other requests succeed
            m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
            SetReleaseResourcesResponse(MnDiagStatus::READY);
            SetStatus(MnDiagStatus::RUNNING);

            // Call the method under test
            dcgmReturn_t result = ReleaseRemoteResources();

            // Verify results
            CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
            CHECK(GetStatus() == MnDiagStatus::READY);
            CHECK(GetStateMachine()->GetStats().notifyDiagnosticFinishedCalled);

            // All connections should have been attempted
            REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 3);
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

            // Configure the mock operations to return success for all requests
            m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

            // Call the method under test
            dcgmReturn_t result = GetNodeInfo();

            // Verify results
            REQUIRE(result == DCGM_ST_OK);

            // Verify the API was called for each remote connection (3 connections)
            REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 3);
        }

        SECTION("Should fail when any remote node fails to get node info")
        {
            // Setup mock connections
            SetupMockConnections(3);

            // Configure one request to fail
            m_mockDcgmOps.SetMultinodeRequestHandler(
                11,
                dcgmMultinodeRequestType_t::GetNodeInfo,
                [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t * /* request */) {
                    return DCGM_ST_DIAG_BAD_JSON;
                });

            // All other requests succeed
            m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

            // Call the method under test
            dcgmReturn_t result = GetNodeInfo();

            // Verify results
            REQUIRE(result == DCGM_ST_DIAG_BAD_JSON);

            // All connections should have been attempted
            REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 3);
        }
    }

    SECTION("GetNodeInfo includes active GPU inventory")
    {
        InitMockCuda();
        dcgmcm_gpu_info_cached_t gpuInfo {};
        gpuInfo.gpuId = 7;
        SafeCopyTo(gpuInfo.uuid, "GPU-uuid-7");
        GetDcgmCoreProxy()->SetMockGpuIds({ 7 });
        GetDcgmCoreProxy()->SetMockGpuInfo({ gpuInfo });

        dcgm_mndiag_msg_node_info_t message {};
        message.header.version = dcgm_mndiag_msg_node_info_version2;

        REQUIRE(HandleGetNodeInfo(&message.header) == DCGM_ST_OK);
        REQUIRE(message.nodeInfo.numGpus == 1);
        CHECK(message.nodeInfo.gpus[0].gpuId == 7);
        CHECK(std::string(message.nodeInfo.gpus[0].uuid) == "GPU-uuid-7");

        message.header.version = dcgm_mndiag_msg_node_info_version1;
        CHECK(HandleGetNodeInfo(&message.header) == DCGM_ST_VER_MISMATCH);

        dcgm_mndiag_msg_node_info_t incompleteMessage {};
        incompleteMessage.header.version = dcgm_mndiag_msg_node_info_version2;
        GetDcgmCoreProxy()->SetMockGpuIds({ 8 });
        GetDcgmCoreProxy()->SetMockGpuInfo({});
        CHECK(HandleGetNodeInfo(&incompleteMessage.header) == DCGM_ST_NO_DATA);
        CHECK(incompleteMessage.nodeInfo.numGpus == 0);
    }

    SECTION("HandleGetNodeInfo propagates active GPU ID collection failures")
    {
        InitMockCuda();
        GetDcgmCoreProxy()->SetMockGpuIdsResult(DCGM_ST_GENERIC_ERROR);
        dcgm_mndiag_msg_node_info_t message {};
        message.header.version = dcgm_mndiag_msg_node_info_version2;

        CHECK(HandleGetNodeInfo(&message.header) == DCGM_ST_GENERIC_ERROR);
        CHECK(message.nodeInfo.numGpus == 0);
    }

    SECTION("HandleGetNodeInfo propagates GPU metadata collection failures")
    {
        InitMockCuda();
        GetDcgmCoreProxy()->SetMockGpuIds({ 0 });
        GetDcgmCoreProxy()->SetMockGpuInfoResult(DCGM_ST_GENERIC_ERROR);
        dcgm_mndiag_msg_node_info_t message {};
        message.header.version = dcgm_mndiag_msg_node_info_version2;

        CHECK(HandleGetNodeInfo(&message.header) == DCGM_ST_GENERIC_ERROR);
        CHECK(message.nodeInfo.numGpus == 0);
    }

    SECTION("HandleGetNodeInfo rejects inventory exceeding message capacity")
    {
        InitMockCuda();
        dcgm_mndiag_msg_node_info_t message {};
        message.header.version = dcgm_mndiag_msg_node_info_version2;
        std::vector<unsigned int> gpuIds;
        for (size_t gpuIndex = 0; gpuIndex <= std::size(message.nodeInfo.gpus); ++gpuIndex)
        {
            gpuIds.push_back(static_cast<unsigned int>(gpuIndex));
        }
        GetDcgmCoreProxy()->SetMockGpuIds(gpuIds);

        CHECK(HandleGetNodeInfo(&message.header) == DCGM_ST_INSUFFICIENT_SIZE);
        CHECK(message.nodeInfo.numGpus == 0);
    }

    SECTION("GetNodeInfo propagates local and remote GPU inventory")
    {
        InitMockCuda();
        dcgmcm_gpu_info_cached_t localGpu {};
        localGpu.gpuId = 7;
        SafeCopyTo(localGpu.uuid, "GPU-local-7");
        GetDcgmCoreProxy()->SetMockGpuIds({ 7 });
        GetDcgmCoreProxy()->SetMockGpuInfo({ localGpu });
        SetupLoopbackConnection();
        SetupMockConnections(1);
        m_mockDcgmOps.SetMultinodeRequestHandler(dcgmMultinodeRequestType_t::GetNodeInfo,
                                                 [](dcgmHandle_t, dcgmMultinodeRequest_t *request) {
                                                     auto &nodeInfo = request->requestData.nodeInfo;
                                                     SafeCopyTo(nodeInfo.dcgmVersion, "test-dcgm");
                                                     SafeCopyTo(nodeInfo.driverVersion, "test-driver");
                                                     nodeInfo.numGpus       = 1;
                                                     nodeInfo.gpus[0].gpuId = 8;
                                                     SafeCopyTo(nodeInfo.gpus[0].uuid, "GPU-remote-8");
                                                     return DCGM_ST_OK;
                                                 });

        REQUIRE(GetNodeInfo() == DCGM_ST_OK);
        auto const &nodeInfo = GetNodeInfoMap();
        REQUIRE(nodeInfo.at("localhost").gpus.size() == 1);
        CHECK(nodeInfo.at("localhost").gpus[0].gpuId == 7);
        CHECK(nodeInfo.at("localhost").gpus[0].uuid == "GPU-local-7");
        CHECK_FALSE(nodeInfo.at("localhost").mpiNodeName.empty());
        REQUIRE(nodeInfo.at("test-node-0").gpus.size() == 1);
        CHECK(nodeInfo.at("test-node-0").gpus[0].gpuId == 8);
        CHECK(nodeInfo.at("test-node-0").gpus[0].uuid == "GPU-remote-8");
        CHECK(nodeInfo.at("test-node-0").mpiNodeName == "test-node-0");
    }

    SECTION("GetNodeInfo rejects an oversized remote GPU inventory")
    {
        InitMockCuda();
        SetupMockConnections(1);
        m_mockDcgmOps.SetMultinodeRequestHandler(
            dcgmMultinodeRequestType_t::GetNodeInfo, [](dcgmHandle_t, dcgmMultinodeRequest_t *request) {
                request->requestData.nodeInfo.numGpus = std::size(request->requestData.nodeInfo.gpus) + 1;
                return DCGM_ST_OK;
            });

        CHECK(GetNodeInfo() == DCGM_ST_BADPARAM);
        CHECK_FALSE(GetNodeInfoMap().contains("test-node-0"));
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
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Set default behavior for tunnel manager
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set default behavior for DCGM operations (only connection, no authorization)
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // Test with a list of hosts using TCP connections
        std::vector<std::string> hostList = { "node1:5555", "node2:6666" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, MNDIAG_MANAGER_EFFECTIVE_UID);

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify TCP tunnel manager was called with correct parameters (order is non-deterministic)
        auto const &calls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(calls.size() == 2);

        bool foundNode1 = false;
        bool foundNode2 = false;
        for (auto const &call : calls)
        {
            if (std::get<0>(call) == "node1" && std::get<1>(call) == 5555)
            {
                REQUIRE(std::get<2>(call).value() == MNDIAG_MANAGER_EFFECTIVE_UID);
                foundNode1 = true;
            }
            else if (std::get<0>(call) == "node2" && std::get<1>(call) == 6666)
            {
                REQUIRE(std::get<2>(call).value() == MNDIAG_MANAGER_EFFECTIVE_UID);
                foundNode2 = true;
            }
        }
        REQUIRE(foundNode1);
        REQUIRE(foundNode2);

        // Verify DCGM operations was called correctly for connections only
        REQUIRE(m_mockDcgmOps.GetConnectV3CallCount() == 2);

        // Verify NO authorization calls were made (authorization is separate now)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 0);

        // Verify that the connection was made using the local port returned by the tunnel manager
        // Examine connections by handle in the DCGM operations
        // For TCP connections, the address should include the local port returned by the SSH tunnel manager (12345)
        for (dcgmHandle_t handle = 1; handle <= 2; handle++)
        {
            REQUIRE(m_mockDcgmOps.HasConnection(handle));
            auto connInfoOpt = m_mockDcgmOps.GetConnectionInfo(handle);
            REQUIRE(connInfoOpt.has_value());
            auto const &connInfo = connInfoOpt.value();
            REQUIRE(connInfo.connectionString == "tcp://127.0.0.1:12345"); // Port is included in the address
        }
    }

    SECTION("ConnectRemoteNodes handles UDS connections correctly")
    {
        // Set up mock objects
        auto mockUdsSSHTunnelManager = std::make_unique<MockUdsSSHTunnelManager>();

        // Set default behavior for tunnel manager
        mockUdsSSHTunnelManager->SetDefaultRemoteSocketPath("/tmp/local_socket");
        mockUdsSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set default behavior for DCGM operations (only connection, no authorization)
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);

        // Install mock objects
        SetUdsSSHTunnelManager(std::move(mockUdsSSHTunnelManager));

        // Test with a list of hosts using UDS connections
        std::vector<std::string> hostList = { "node1:unix:///var/run/dcgm.socket", "node2:unix:///tmp/dcgm.socket" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, MNDIAG_MANAGER_EFFECTIVE_UID);

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify UDS tunnel manager was called with correct parameters (order is non-deterministic)
        auto calls = GetUdsSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(calls.size() == 2);

        bool foundNode1 = false;
        bool foundNode2 = false;
        for (auto const &call : calls)
        {
            if (std::get<0>(call) == "node1" && std::get<1>(call) == "/var/run/dcgm.socket")
            {
                REQUIRE(std::get<2>(call).value() == MNDIAG_MANAGER_EFFECTIVE_UID);
                foundNode1 = true;
            }
            else if (std::get<0>(call) == "node2" && std::get<1>(call) == "/tmp/dcgm.socket")
            {
                REQUIRE(std::get<2>(call).value() == MNDIAG_MANAGER_EFFECTIVE_UID);
                foundNode2 = true;
            }
        }
        REQUIRE(foundNode1);
        REQUIRE(foundNode2);

        // Verify DCGM operations was called correctly for connections only
        REQUIRE(m_mockDcgmOps.GetConnectV3CallCount() == 2);

        // Verify NO authorization calls were made (authorization is separate now)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 0);

        // Verify that the connection was made using the local socket path returned by the tunnel manager
        // Examine connections by handle in the DCGM operations
        // The socket path should match the one returned by the UDS SSH tunnel manager ("/tmp/local_socket")
        for (dcgmHandle_t handle = 1; handle <= 2; handle++)
        {
            REQUIRE(m_mockDcgmOps.HasConnection(handle));
            auto connInfoOpt = m_mockDcgmOps.GetConnectionInfo(handle);
            REQUIRE(connInfoOpt.has_value());
            auto const &connInfo = connInfoOpt.value();
            REQUIRE(connInfo.connectionString == "unix:///tmp/local_socket");
        }
    }

    SECTION("ConnectRemoteNodes handles mixed connections (TCP and UDS) correctly")
    {
        // Set up mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockUdsSSHTunnelManager = std::make_unique<MockUdsSSHTunnelManager>();

        // Set default behavior for tunnel managers
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);
        mockUdsSSHTunnelManager->SetDefaultRemoteSocketPath("/tmp/local_socket");
        mockUdsSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set default behavior for DCGM operations (only connection, no authorization)
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetUdsSSHTunnelManager(std::move(mockUdsSSHTunnelManager));

        // Test with a list of hosts using both TCP and UDS connections
        std::vector<std::string> hostList = { "node1:5555", "node2:unix:///tmp/dcgm.socket" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, MNDIAG_MANAGER_EFFECTIVE_UID);

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify TCP tunnel manager was called with correct parameters
        auto const &tcpCalls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(tcpCalls.size() == 1);

        // Verify the TCP calls contain the expected parameters
        bool foundTcpNode = false;
        for (const auto &call : tcpCalls)
        {
            if (std::get<0>(call) == "node1" && std::get<1>(call) == 5555
                && std::get<2>(call).value() == MNDIAG_MANAGER_EFFECTIVE_UID)
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
                && std::get<2>(call).value() == MNDIAG_MANAGER_EFFECTIVE_UID)
            {
                foundUdsNode = true;
                break;
            }
        }
        REQUIRE(foundUdsNode);

        // Verify DCGM operations was called correctly for connections only
        REQUIRE(m_mockDcgmOps.GetConnectV3CallCount() == 2);

        // Verify NO authorization calls were made (authorization is separate now)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 0);

        // Verify that the connections were made using the correct local ports/paths returned by the tunnel managers
        // We need to examine each connection handle

        // Since we can't guarantee the order of handles, check both and make sure one is TCP and one is UDS
        REQUIRE(m_mockDcgmOps.HasConnection(1));
        REQUIRE(m_mockDcgmOps.HasConnection(2));
        auto connInfoOpt1 = m_mockDcgmOps.GetConnectionInfo(1);
        auto connInfoOpt2 = m_mockDcgmOps.GetConnectionInfo(2);
        REQUIRE(connInfoOpt1.has_value());
        REQUIRE(connInfoOpt2.has_value());
        auto const &connInfo1 = connInfoOpt1.value();
        auto const &connInfo2 = connInfoOpt2.value();

        // Count how many of each type we have
        int tcpCount = 0;
        int udsCount = 0;

        // Check first connection
        if (connInfo1.connectionString.starts_with("unix://"))
        {
            udsCount++;
            REQUIRE(connInfo1.connectionString == "unix:///tmp/local_socket");
        }
        else
        {
            tcpCount++;
            REQUIRE(connInfo1.connectionString == "tcp://127.0.0.1:12345"); // Port is included in the address
        }

        // Check second connection
        if (connInfo2.connectionString.starts_with("unix://"))
        {
            udsCount++;
            REQUIRE(connInfo2.connectionString == "unix:///tmp/local_socket");
        }
        else
        {
            tcpCount++;
            REQUIRE(connInfo2.connectionString == "tcp://127.0.0.1:12345"); // Port is included in the address
        }

        // Verify we have one of each type
        REQUIRE(tcpCount == 1);
        REQUIRE(udsCount == 1);
    }

    SECTION("ConnectRemoteNodes handles tunnel creation failure")
    {
        // Set up mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Set behavior for tunnel manager to simulate failure
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::GenericFailure);

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // Test with a host that will fail to create a tunnel
        std::vector<std::string> hostList = { "node1:5555" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, MNDIAG_MANAGER_EFFECTIVE_UID);

        // Verify result indicates failure
        REQUIRE(result == DCGM_ST_REMOTE_SSH_CONNECTION_FAILED);

        // Verify tunnel manager was called
        auto const &startCalls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(startCalls.size() == 1);

        // Verify DCGM operations was not called after tunnel failure
        REQUIRE(m_mockDcgmOps.GetConnectV3CallCount() == 0);
    }

    SECTION("ConnectRemoteNodes handles DCGM connection failure")
    {
        // Set up mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Set default behavior for tunnel manager
        mockTcpSSHTunnelManager->SetDefaultRemotePort(12345);
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Set behavior for DCGM operations to simulate connection failure
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_CONNECTION_NOT_VALID);

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // Test with a host that will fail to connect to DCGM
        std::vector<std::string> hostList = { "node1:5555" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, MNDIAG_MANAGER_EFFECTIVE_UID);

        // Verify result indicates connection failure
        REQUIRE(result == DCGM_ST_CONNECTION_NOT_VALID);

        // Verify tunnel manager was called
        auto const &startCalls = GetTcpSSHTunnelManager()->GetStartSessionCalls();
        REQUIRE(startCalls.size() == 1);

        // Verify DCGM operations Connect was called but failed
        REQUIRE(m_mockDcgmOps.GetConnectV3CallCount() == 1);

        // Verify EndSession was called to clean up the tunnel
        auto const &endCalls = GetTcpSSHTunnelManager()->GetEndSessionCalls();
        REQUIRE(endCalls.size() == 1);
    }
}

/**
 * Tests for AuthorizeRemoteConnections method
 */
TEST_CASE_METHOD(MnDiagManagerTests, "AuthorizeRemoteConnections Tests [mndiag]")
{
    SECTION("AuthorizeRemoteConnections handles successful authorization")
    {
        // Set behavior for DCGM operations - authorization succeeds
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        // Setup mock connections first
        SetupMockConnections(2);

        // Test authorization
        dcgmReturn_t result = AuthorizeRemoteConnections();

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify authorization was called for each remote connection (2 connections)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 2);
    }

    SECTION("AuthorizeRemoteConnections handles authorization failure")
    {
        // Set behavior for DCGM operations - ALL remote authorizations fail
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);
        // Setup mock connections first
        SetupMockConnections(2);

        // Test authorization
        dcgmReturn_t result = AuthorizeRemoteConnections();

        // Verify result indicates authorization failure
        REQUIRE(result == DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // All connections in the batch are processed in parallel before failure is checked
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 2);
    }
}

/**
 * Tests for RevokeRemoteAuthorizations method
 */
TEST_CASE_METHOD(MnDiagManagerTests, "RevokeRemoteAuthorizations Tests [mndiag]")
{
    SECTION("RevokeRemoteAuthorizations handles successful revocation")
    {
        // Set behavior for DCGM operations - revocation succeeds
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        // Setup mock connections first
        SetupMockConnections(2);

        // Authorize connections first to simulate an authorized state
        dcgmReturn_t authResult = AuthorizeRemoteConnections();
        REQUIRE(authResult == DCGM_ST_OK);

        // Reset call counter for the revocation test
        ResetMockOps();
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

        // Test revocation
        dcgmReturn_t result = RevokeRemoteAuthorizations();

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify revocation was called for each remote connection (2 connections)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 2);
    }

    SECTION("RevokeRemoteAuthorizations handles revocation failure")
    {
        // First set up for successful authorization
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        // Setup 2 mock connections
        SetupMockConnections(2);

        // Authorize connections first to simulate an authorized state
        dcgmReturn_t authResult = AuthorizeRemoteConnections();
        REQUIRE(authResult == DCGM_ST_OK);

        // Reset and set behavior for DCGM operations - ALL remote revocations fail
        ResetMockOps();
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // Test revocation
        dcgmReturn_t result = RevokeRemoteAuthorizations();

        // Verify result indicates revocation failure
        REQUIRE(result == DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // All connections in the batch are processed in parallel before failure is checked
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 2);
    }
}

/**
 * Tests for CleanupConnections method
 */
TEST_CASE_METHOD(MnDiagManagerTests, "CleanupConnections Tests [mndiag]")
{
    SECTION("CleanupConnections properly cleans up all connections")
    {
        // Set up mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockUdsSSHTunnelManager = std::make_unique<MockUdsSSHTunnelManager>();

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetUdsSSHTunnelManager(std::move(mockUdsSSHTunnelManager));

        // First set up some connections via ConnectRemoteNodes
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);
        GetTcpSSHTunnelManager()->SetDefaultRemotePort(12345);
        GetTcpSSHTunnelManager()->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);
        GetUdsSSHTunnelManager()->SetDefaultRemoteSocketPath("/tmp/local_socket");
        GetUdsSSHTunnelManager()->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Connect to a mix of TCP and UDS nodes
        std::vector<std::string> hostList = { "node1:5555", "node2:unix:///tmp/dcgm.socket" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, MNDIAG_MANAGER_EFFECTIVE_UID);
        REQUIRE(result == DCGM_ST_OK);

        // Reset counters for cleanup phase
        ResetMockOps();
        GetTcpSSHTunnelManager()->Reset();
        GetUdsSSHTunnelManager()->Reset();

        // Now test CleanupConnections
        result = CleanupConnections();

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify DCGM operations was called to disconnect each connection (2 connections)
        REQUIRE(m_mockDcgmOps.GetDisconnectCallCount() == 2);

        // Verify tunnel managers were called to end sessions
        auto tcpEndCalls = GetTcpSSHTunnelManager()->GetEndSessionCalls();
        auto udsEndCalls = GetUdsSSHTunnelManager()->GetEndSessionCalls();
        REQUIRE(tcpEndCalls.size() == 1);
        REQUIRE(udsEndCalls.size() == 1);

        // Verify the TCP and UDS end calls contain the expected parameters
        REQUIRE(std::get<0>(tcpEndCalls[0]) == "node1");
        REQUIRE(std::get<1>(tcpEndCalls[0]) == 5555);

        REQUIRE(std::get<0>(udsEndCalls[0]) == "node2");
        REQUIRE(std::get<1>(udsEndCalls[0]) == "/tmp/dcgm.socket");
    }
}

/**
 * Tests for DisconnectRemoteNodes method
 */
TEST_CASE_METHOD(MnDiagManagerTests, "DisconnectRemoteNodes Tests [mndiag]")
{
    SECTION("DisconnectRemoteNodes properly disconnects all nodes")
    {
        // Set up mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockUdsSSHTunnelManager = std::make_unique<MockUdsSSHTunnelManager>();

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetUdsSSHTunnelManager(std::move(mockUdsSSHTunnelManager));

        // First set up some connections via ConnectRemoteNodes
        // We need to set up the mock behavior for connection (no authorization)
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);
        GetTcpSSHTunnelManager()->SetDefaultRemotePort(12345);
        GetTcpSSHTunnelManager()->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);
        GetUdsSSHTunnelManager()->SetDefaultRemoteSocketPath("/tmp/local_socket");
        GetUdsSSHTunnelManager()->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        // Connect to a mix of TCP and UDS nodes
        std::vector<std::string> hostList = { "node1:5555", "node2:unix:///tmp/dcgm.socket" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, MNDIAG_MANAGER_EFFECTIVE_UID);
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 0); // No authorization in ConnectRemoteNodes

        // Authorize the connections before disconnecting them
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        result = AuthorizeRemoteConnections();
        REQUIRE(result == DCGM_ST_OK);

        // Reset the call counters for the disconnect phase
        ResetMockOps();
        GetTcpSSHTunnelManager()->Reset();
        GetUdsSSHTunnelManager()->Reset();

        // Set up mock behavior for revocation during disconnect
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

        // Now test DisconnectRemoteNodes
        result = DisconnectRemoteNodes();

        // Verify result
        REQUIRE(result == DCGM_ST_OK);

        // Verify DCGM operations was called for revocation (2 remote nodes) + disconnect (2 remote nodes)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 2); // Revocation calls
        REQUIRE(m_mockDcgmOps.GetDisconnectCallCount() == 2);       // Disconnect calls

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
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));

        // First set up a connection via ConnectRemoteNodes
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        GetTcpSSHTunnelManager()->SetDefaultRemotePort(12345);
        GetTcpSSHTunnelManager()->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::Active);

        std::vector<std::string> hostList = { "node1:5555" };
        dcgmReturn_t result               = ConnectRemoteNodes(hostList, MNDIAG_MANAGER_EFFECTIVE_UID);
        REQUIRE(result == DCGM_ST_OK);

        // Authorize the connection first
        result = AuthorizeRemoteConnections();
        REQUIRE(result == DCGM_ST_OK);

        // Reset the API and set it to fail on revocation
        ResetMockOps();
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // Now test DisconnectRemoteNodes
        result = DisconnectRemoteNodes();

        // Verify result is the error from revocation
        REQUIRE(result == DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED);

        // Still verify that we tried to disconnect and end the tunnel session
        // Even with the revocation error
        REQUIRE(m_mockDcgmOps.GetDisconnectCallCount() == 1);
        const auto &endCalls = GetTcpSSHTunnelManager()->GetEndSessionCalls();
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
    SetMockCudaVersion(12000);

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

    SECTION("RunHeadNode dispatches mnnvbandwidth GPU pair mode directly")
    {
        int const executionBehavior = GENERATE(0, 1, 2);
        CAPTURE(executionBehavior);

        // Create mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();

        // Add path capture for validation from StateMachine
        std::string capturedStateMachinePath;
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        mockStateMachine->SetExpectedBinaryPathCallback(
            [&capturedStateMachinePath](std::string const &path) { capturedStateMachinePath = path; });

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

        // Configure DCGM operations behavior
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        SetReserveResourcesResponse(MnDiagStatus::RESERVED);
        SetReleaseResourcesResponse(executionBehavior == 2 ? MnDiagStatus::FAILED : MnDiagStatus::READY);
        SetDetectProcessResponse(MnDiagStatus::RUNNING);

        // Configure MPI runner factory behavior
        // The mock factory's CreateMpiRunner method will return a MockMnDiagMpiRunner with default behaviors
        mockMpiRunnerFactory->SetMockTestBinaryPath(GetMockPath());
        if (executionBehavior == 2)
        {
            mockMpiRunnerFactory->SetMockBatchResult(std::unexpected(DCGM_ST_TIMEOUT));
        }

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.testName, executionBehavior == 0 ? "mnubergemm" : "mnnvbandwidth");
        if (executionBehavior != 0)
        {
            SafeCopyTo(params.testParms[0], "mnnvbandwidth.gpuPairs=all");
        }
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        auto responsePtr = MakeUniqueZero<dcgmMnDiagResponse_t>();
        auto &response   = *responsePtr;
        response.version = dcgmMnDiagResponse_version;
        if (executionBehavior == 2)
        {
            response.numErrors = 1;
        }

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);
        GetDcgmCoreProxy()->SetMockCudaVersion(12000);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, MNDIAG_MANAGER_EFFECTIVE_UID, response);

        // Verify results
        dcgmReturn_t const expectedResult = executionBehavior == 2 ? DCGM_ST_TIMEOUT : DCGM_ST_OK;
        REQUIRE(result == expectedResult);
        if (executionBehavior == 2)
        {
            CHECK(response.version == dcgmMnDiagResponse_version);
            CHECK(response.numErrors == 0);
        }

        // Verify ConnectRemoteNodes was called
        REQUIRE(m_mockDcgmOps.GetConnectV3CallCount() > 0);

        // Verify ReserveRemoteResources was called
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() > 0);

        // Verify MPI runner methods were called
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().runBatchCount
                == (executionBehavior == 1 || executionBehavior == 2 ? 1 : 0));
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().constructMpiCommandCount
                == (executionBehavior == 0 ? 1 : 0));
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().launchMpiProcessCount == (executionBehavior == 0 ? 1 : 0));
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().waitCount == (executionBehavior == 0 ? 1 : 0));
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::ReleaseResources) == 1);
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::DetectProcess)
                == (executionBehavior == 0 ? 1 : 0));

        // Verify state machine received the binary path
        std::string mockPath = GetMockPath();
        REQUIRE(capturedStateMachinePath.find(mockPath) != std::string::npos);
    }

    SECTION("Successful execution of RunHeadNode with custom mnubergemm path")
    {
        // Create mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();

        // Add path capture for validation from StateMachine
        std::string capturedStateMachinePath;
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        mockStateMachine->SetExpectedBinaryPathCallback(
            [&capturedStateMachinePath](std::string const &path) { capturedStateMachinePath = path; });

        // Set env to custom path
        std::string customPath = "/bin/true";
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), customPath);

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

        // Configure DCGM operations behavior
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        SetReserveResourcesResponse(MnDiagStatus::RESERVED);
        SetReleaseResourcesResponse(MnDiagStatus::READY);
        SetDetectProcessResponse(MnDiagStatus::RUNNING);

        // Configure MPI runner factory behavior
        // The mock factory's CreateMpiRunner method will return a MockMnDiagMpiRunner with default behaviors
        mockMpiRunnerFactory->SetMockTestBinaryPath(customPath);

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        auto responsePtr = MakeUniqueZero<dcgmMnDiagResponse_t>();
        auto &response   = *responsePtr;
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);
        GetDcgmCoreProxy()->SetMockCudaVersion(12000);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, MNDIAG_MANAGER_EFFECTIVE_UID, response);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);

        // Verify ConnectRemoteNodes was called
        REQUIRE(m_mockDcgmOps.GetConnectV3CallCount() > 0);

        // Verify ReserveRemoteResources was called
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() > 0);

        // Verify MPI runner methods were called
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().constructMpiCommandCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().launchMpiProcessCount == 1);
        REQUIRE(GetMpiRunnerFactory()->GetLastRunnerStats().waitCount == 1);

        // Verify StateMachine received the custom path
        REQUIRE(capturedStateMachinePath == customPath);
    }

    SECTION("Failed to connect to remote nodes")
    {
        // Create mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();
        auto mockCoreProxy           = std::make_unique<MockDcgmCoreProxy>();

        // Configure tunnel manager to fail to create SSH tunnel
        mockTcpSSHTunnelManager->SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState::GenericFailure);

        // Configure MPI runner factory behavior
        // The mock factory's CreateMpiRunner method will return a MockMnDiagMpiRunner with default behaviors

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        auto responsePtr = MakeUniqueZero<dcgmMnDiagResponse_t>();
        auto &response   = *responsePtr;
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, MNDIAG_MANAGER_EFFECTIVE_UID, response);

        // Verify results
        REQUIRE(result == DCGM_ST_REMOTE_SSH_CONNECTION_FAILED);

        // Verify MPI runner was never created
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 0);
    }

    SECTION("Failed to reserve resources")
    {
        // Create mock objects
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

        // Configure DCGM operations behavior
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        SetReserveResourcesResponse(MnDiagStatus::FAILED);

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        auto responsePtr = MakeUniqueZero<dcgmMnDiagResponse_t>();
        auto &response   = *responsePtr;
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, MNDIAG_MANAGER_EFFECTIVE_UID, response);

        // Verify results
        REQUIRE(result == DCGM_ST_IN_USE);

        // Verify MPI runner was never created
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 0);
    }

    SECTION("Failed to launch MPI process")
    {
        // Create mock objects
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

        // Configure DCGM operations behavior
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        SetReserveResourcesResponse(MnDiagStatus::RESERVED);
        SetReleaseResourcesResponse(MnDiagStatus::READY);

        // Configure MPI runner to fail to launch
        // We need to subclass MockMnDiagMpiRunnerFactory to override CreateMpiRunner
        class CustomMockMpiRunnerFactory : public MockMnDiagMpiRunnerFactory
        {
        public:
            std::unique_ptr<MnDiagMpiRunnerBase> CreateMpiRunner(DcgmCoreProxyBase & /* coreProxy */,
                                                                 dcgmMultinodeTestType_t /* testType */,
                                                                 uid_t /* effectiveUid */) override
            {
                m_createMpiRunnerCount++;
                auto runner                = std::make_unique<MockMnDiagMpiRunner>();
                runner->m_mockLaunchResult = DCGM_ST_GENERIC_ERROR;
                runner->SetMockTestBinaryPath("/fake/mnubergemm");
                // Store the runner ID before returning it
                m_lastRunnerId = static_cast<MockMnDiagMpiRunner *>(runner.get())->GetRunnerId();
                return runner;
            }
        };

        auto customMpiRunnerFactory = std::make_unique<CustomMockMpiRunnerFactory>();

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(customMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        auto responsePtr = MakeUniqueZero<dcgmMnDiagResponse_t>();
        auto &response   = *responsePtr;
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, MNDIAG_MANAGER_EFFECTIVE_UID, response);

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

        // Configure DCGM operations behavior
        m_mockDcgmOps.SetConnectV3Result(DCGM_ST_OK);
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
        SetReserveResourcesResponse(MnDiagStatus::RESERVED);
        SetDetectProcessResponse(MnDiagStatus::RUNNING);

        // Configure MPI runner factory behavior
        // The mock factory's CreateMpiRunner method will return a MockMnDiagMpiRunner with default behaviors

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.hostList[1], "test-node-1:5555");

        // Create response structure to be populated
        auto responsePtr = MakeUniqueZero<dcgmMnDiagResponse_t>();
        auto &response   = *responsePtr;
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Set the mock driver version result to NVML_ERROR. This will cause the GetNodeInfo call to fail.
        dcgmReturn_t driverVersionResult = DCGM_ST_NVML_ERROR;
        GetDcgmCoreProxy()->SetMockDriverVersionResult(driverVersionResult);
        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, MNDIAG_MANAGER_EFFECTIVE_UID, response);

        // Verify results. This should be the same as the mock driver version result since that is a
        // fatal error and will halt execution
        REQUIRE(result == driverVersionResult);

        // Verify ConnectRemoteNodes was called
        REQUIRE(m_mockDcgmOps.GetConnectV3CallCount() > 0);

        // Verify only one request was made - authorize connection
        //REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 1);
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::AuthorizeConnection) == 1);
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::RevokeAuthorization) == 1);
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::GetNodeInfo) == 0);
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::ReserveResources) == 0);
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::ReleaseResources) == 0);
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::DetectProcess) == 0);

        // Verify MPI runner methods were not called
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 0);
    }

    SECTION("RunHeadNode fails when factory returns null runner")
    {
        // Create mock objects
        auto mockTcpSSHTunnelManager = std::make_unique<MockTcpSSHTunnelManager>();
        auto mockMpiRunnerFactory    = std::make_unique<MockMnDiagMpiRunnerFactory>();
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();

        // Configure factory to return nullptr (simulates unknown test type)
        mockMpiRunnerFactory->SetReturnNullNext();

        // Install mock objects
        SetTcpSSHTunnelManager(std::move(mockTcpSSHTunnelManager));
        SetMpiRunnerFactory(std::move(mockMpiRunnerFactory));
        SetStateMachine(std::move(mockStateMachine));
        SetCoreProxy(std::move(mockCoreProxy));

        // Create test input parameters
        dcgmRunMnDiag_t params {};
        params.version = dcgmRunMnDiag_version;
        SafeCopyTo(params.hostList[0], "localhost");
        SafeCopyTo(params.testName, "mnubergemm");

        // Create response structure to be populated
        auto responsePtr = MakeUniqueZero<dcgmMnDiagResponse_t>();
        auto &response   = *responsePtr;
        response.version = dcgmMnDiagResponse_version;

        // Set the mock GPU IDs and GPU Info
        GetDcgmCoreProxy()->SetMockGpuIds(fakeGpuIds);
        GetDcgmCoreProxy()->SetMockGpuInfo(fakeGpuInfos);
        GetDcgmCoreProxy()->SetMockGpuIdsSameSku(true);

        // Call method under test
        dcgmReturn_t result = m_manager.RunHeadNode(params, MNDIAG_MANAGER_EFFECTIVE_UID, response);

        // Verify results - should fail because factory returned null
        REQUIRE(result == DCGM_ST_BADPARAM);

        // Verify factory was called once
        REQUIRE(GetMpiRunnerFactory()->m_createMpiRunnerCount == 1);
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
        ScopedEnvVars envVars("DCGM_MNDIAG_SUPPORTED_SKUS", "1240");
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 3);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("31c2"));
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
    }

    SECTION("Devices supported from env and default")
    {
        ScopedEnvVars envVars("DCGM_MNDIAG_SUPPORTED_SKUS", "1240");
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 3);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("31c2"));
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
    }

    SECTION("Device not present in SKU (env + default)")
    {
        ScopedEnvVars envVars("DCGM_MNDIAG_SUPPORTED_SKUS", "1240");
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 3);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("31c2"));
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
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "LoadSupportedSkusFromEnv [mndiag]")
{
    SECTION("Default (no env var set)")
    {
        ScopedEnvVars envVars("DCGM_MNDIAG_SUPPORTED_SKUS", std::nullopt);
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 2);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("31c2"));
    }

    SECTION("Multiple SKUs in env")
    {
        ScopedEnvVars envVars("DCGM_MNDIAG_SUPPORTED_SKUS", "abcd, 1234, 5678 ");
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 5);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("31c2"));
        REQUIRE(skus.contains("abcd"));
        REQUIRE(skus.contains("1234"));
        REQUIRE(skus.contains("5678"));
    }

    SECTION("Mixedcase SKUs in env are stored as lowercase")
    {
        ScopedEnvVars envVars("DCGM_MNDIAG_SUPPORTED_SKUS", "24B9,1h34,xY5Z");
        LoadSupportedSkusFromEnv();
        auto skus = GetSupportedSkus();
        REQUIRE(skus.size() == 5);
        REQUIRE(skus.contains("2941"));
        REQUIRE(skus.contains("31c2"));
        REQUIRE(skus.contains("24b9"));
        REQUIRE(skus.contains("1h34"));
        REQUIRE(skus.contains("xy5z"));
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

        mockStateMachine->SetExpectedBinaryPathCallback(
            [&capturedPath](std::string const &path) { capturedPath = path; });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with time_to_run
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run=300");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[1], "mnubergemm.other_param=value");

        // Set the mnubergemm path
        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.testBinaryPath, testPath.c_str());
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnubergemm.");

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

        mockStateMachine->SetExpectedBinaryPathCallback(
            [&capturedPath](std::string const &path) { capturedPath = path; });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters without time_to_run
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.other_param=value");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[1], "mnubergemm.flag");

        // Set the mnubergemm path
        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.testBinaryPath, testPath.c_str());
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnubergemm.");

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(timeoutWasSet == true); // Timeout should always be set (to default when not specified)
        REQUIRE(capturedTimeout == std::chrono::seconds(3600)); // Default timeout is 3600 seconds
        REQUIRE(capturedPath == testPath);
        CHECK_FALSE(GetStateMachine()->GetStats().lastKeepReserved);
    }

    SECTION("GPU pair mode extends the reservation timeout for the full run")
    {
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });
        SetStateMachine(std::move(mockStateMachine));

        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId       = 12345;
        paramMsg.runParams.timeToRunSeconds = 15;
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnnvbandwidth");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnnvbandwidth.gpuPairs=all");
        SafeCopyTo(paramMsg.runParams.testBinaryPath, "/test/path/mnnvbandwidth");
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnnvbandwidth.");

        REQUIRE(HandleBroadcastRunParameters(reinterpret_cast<dcgm_module_command_header_t *>(&paramMsg))
                == DCGM_ST_OK);
        CHECK(GetStateMachine()->GetStats().setProcessExecutionTimeoutCalled);
        CHECK(GetStateMachine()->GetStats().lastTimeoutInSeconds == std::chrono::seconds(15));
        CHECK(GetStateMachine()->GetStats().lastKeepReserved);
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
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with invalid time_to_run format (missing value)
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run");

        // Set the mnubergemm path
        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.testBinaryPath, testPath.c_str());
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnubergemm.");

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

        mockStateMachine->SetExpectedBinaryPathCallback(
            [&capturedPath](std::string const &path) { capturedPath = path; });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with multiple time_to_run values
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run=100");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[1], "mnubergemm.time_to_run=200");

        // Set the mnubergemm path
        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.testBinaryPath, testPath.c_str());
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnubergemm.");

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results - should use the first value
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(timeoutWasSet == true);
        REQUIRE(capturedTimeout == std::chrono::seconds(100));
        REQUIRE(capturedPath == testPath);
    }

    SECTION("Negative time_to_run value rejected")
    {
        // Setup mock state machine
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with negative time_to_run
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run=-100");

        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.testBinaryPath, testPath.c_str());
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnubergemm.");

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results - negative value should be rejected
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Zero time_to_run value rejected")
    {
        // Setup mock state machine
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with zero time_to_run
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run=0");

        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.testBinaryPath, testPath.c_str());
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnubergemm.");

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results - zero value should be rejected
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Non-numeric time_to_run value rejected")
    {
        // Setup mock state machine
        auto mockStateMachine
            = std::make_unique<MockMnDiagStateMachine>([this](MnDiagStatus status) { SetStatus(status); });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with non-numeric time_to_run
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run=invalid");

        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.testBinaryPath, testPath.c_str());
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnubergemm.");

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results - non-numeric value should be rejected
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("time_to_run with semicolon delimiter parsed correctly")
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

        mockStateMachine->SetExpectedBinaryPathCallback(
            [&capturedPath](std::string const &path) { capturedPath = path; });

        SetStateMachine(std::move(mockStateMachine));

        // Create a parameter broadcast message
        dcgm_mndiag_msg_run_params_t paramMsg {};
        paramMsg.header.length     = sizeof(paramMsg);
        paramMsg.header.version    = dcgm_mndiag_msg_run_params_version2;
        paramMsg.header.moduleId   = DcgmModuleIdMnDiag;
        paramMsg.header.subCommand = DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS;

        paramMsg.runParams.headNodeId = 12345;

        // Set up test parameters with time_to_run followed by semicolon and other params
        SafeCopyTo(paramMsg.runParams.runMnDiag.testName, "mnubergemm");
        SafeCopyTo(paramMsg.runParams.runMnDiag.testParms[0], "mnubergemm.time_to_run=500;mnubergemm.other=123");

        std::string testPath = "/test/path/mnubergemm";
        SafeCopyTo(paramMsg.runParams.testBinaryPath, testPath.c_str());
        SafeCopyTo(paramMsg.runParams.testPrefix, "mnubergemm.");

        // Call the method under test
        dcgmReturn_t result = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&paramMsg);

        // Verify results - should parse 500, not "500;mnubergemm.other=123"
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(timeoutWasSet == true);
        REQUIRE(capturedTimeout == std::chrono::seconds(500));
        REQUIRE(capturedPath == testPath);
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "BroadcastRunParametersToRemoteNodes [mndiag]")
{
    SetMockCudaVersion(12000);
    InitMockCuda();

    SECTION("Successful broadcast to multiple remote nodes")
    {
        // Setup mock DCGM operations
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

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
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 3);
    }

    SECTION("Broadcast failure on one remote node")
    {
        // Setup mock DCGM operations to fail on specific handle

        // Set default success
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

        // Set failure for handle 11 (second remote node)
        m_mockDcgmOps.SetMultinodeRequestHandler(
            11,
            BroadcastRunParameters,
            [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t * /* request */) -> dcgmReturn_t {
                return DCGM_ST_CONNECTION_NOT_VALID;
            });

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
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 3);
    }

    SECTION("Broadcast with no remote connections (loopback only)")
    {
        // Setup mock DCGM operations
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

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
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 0);
    }

    SECTION("Broadcast with empty parameters")
    {
        // Setup mock DCGM operations
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

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
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 2);
    }

    SECTION("Verify request structure is populated correctly - custom mnubergemm path")
    {
        std::string customPath = "/bin/true";
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), customPath);

        // Setup mock DCGM operations with callback to inspect request

        dcgmMultinodeRequest_t capturedRequest {};
        bool requestCaptured = false;

        m_mockDcgmOps.SetMultinodeRequestHandler(
            10,
            BroadcastRunParameters,
            [&capturedRequest, &requestCaptured](dcgmHandle_t /* handle */,
                                                 dcgmMultinodeRequest_t *request) -> dcgmReturn_t {
                capturedRequest = *request;
                requestCaptured = true;
                return DCGM_ST_OK;
            });

        // Setup mock connections
        SetupMockConnections(2); // 1 remote + 1 loopback

        // Set up test info as HandleRunHeadNode would before calling BroadcastRunParametersToRemoteNodes
        SetCurrentTestInfo({ mnubergemm, "test_diagnostic", customPath, "test_diagnostic." });

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
        REQUIRE(capturedRequest.version == dcgmMultinodeRequest_version);
        REQUIRE(capturedRequest.testType == MnDiagTestType::mnubergemm);
        REQUIRE(capturedRequest.requestType == MnDiagRequestType::BroadcastRunParameters);

        // Verify the parameters were copied correctly
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testName) == "test_diagnostic");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testParms[0]) == "param1=value1");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testParms[1]) == "param2=value2");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.hostList[0]) == "node1");

        // Verify custom mnubergemm path is used
        REQUIRE(std::string(capturedRequest.requestData.runParams.testBinaryPath) == customPath);
    }

    SECTION("Verify request structure is populated correctly - default mnubergemm path")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), std::nullopt);

        // Setup mock DCGM operations with callback to inspect request

        dcgmMultinodeRequest_t capturedRequest {};
        bool requestCaptured = false;

        m_mockDcgmOps.SetMultinodeRequestHandler(
            10,
            BroadcastRunParameters,
            [&capturedRequest, &requestCaptured](dcgmHandle_t /* handle */,
                                                 dcgmMultinodeRequest_t *request) -> dcgmReturn_t {
                capturedRequest = *request;
                requestCaptured = true;
                return DCGM_ST_OK;
            });

        // Setup mock connections
        SetupMockConnections(2); // 1 remote + 1 loopback

        // Get default path and set up test info as HandleRunHeadNode would
        std::string mockPath = GetMockPath();
        SetCurrentTestInfo({ mnubergemm, "test_diagnostic", mockPath, "test_diagnostic." });

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
        REQUIRE(capturedRequest.version == dcgmMultinodeRequest_version);
        REQUIRE(capturedRequest.testType == MnDiagTestType::mnubergemm);
        REQUIRE(capturedRequest.requestType == MnDiagRequestType::BroadcastRunParameters);

        // Verify the parameters were copied correctly
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testName) == "test_diagnostic");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testParms[0]) == "param1=value1");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.testParms[1]) == "param2=value2");
        REQUIRE(std::string(capturedRequest.requestData.runParams.runMnDiag.hostList[0]) == "node1");

        // Verify default mnubergemm path is used
        REQUIRE(std::string(capturedRequest.requestData.runParams.testBinaryPath).find(mockPath) != std::string::npos);
    }

    SECTION("Malformed time_to_run on head node returns early without broadcasting")
    {
        // Setup mock DCGM operations
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

        // Setup mock connections
        SetupMockConnections(3); // 3 remote connections

        SetCurrentTestInfo({ mnubergemm, "mnubergemm", "", "mnubergemm." });

        // Create test parameters with malformed time_to_run value
        dcgmRunMnDiag_t params {};
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.testParms[0], "mnubergemm.time_to_run=-100"); // Negative value - invalid
        SafeCopyTo(params.hostList[0], "node1");
        SafeCopyTo(params.hostList[1], "node2");

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results - should fail with BADPARAM
        REQUIRE(result == DCGM_ST_BADPARAM);

        // Verify that no remote MultinodeRequest calls were made (early return)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 0);
    }

    SECTION("Zero time_to_run on head node returns early without broadcasting")
    {
        // Setup mock DCGM operations
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

        // Setup mock connections
        SetupMockConnections(3); // 3 remote connections

        SetCurrentTestInfo({ mnubergemm, "mnubergemm", "", "mnubergemm." });

        // Create test parameters with zero time_to_run value
        dcgmRunMnDiag_t params {};
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.testParms[0], "mnubergemm.time_to_run=0"); // Zero - invalid
        SafeCopyTo(params.hostList[0], "node1");
        SafeCopyTo(params.hostList[1], "node2");

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results - should fail with BADPARAM
        REQUIRE(result == DCGM_ST_BADPARAM);

        // Verify that no remote MultinodeRequest calls were made (early return)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 0);
    }

    SECTION("Non-numeric time_to_run on head node returns early without broadcasting")
    {
        // Setup mock DCGM operations
        m_mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

        // Setup mock connections
        SetupMockConnections(3); // 3 remote connections

        SetCurrentTestInfo({ mnubergemm, "mnubergemm", "", "mnubergemm." });

        // Create test parameters with non-numeric time_to_run value
        dcgmRunMnDiag_t params {};
        SafeCopyTo(params.testName, "mnubergemm");
        SafeCopyTo(params.testParms[0], "mnubergemm.time_to_run=abc"); // Non-numeric - invalid
        SafeCopyTo(params.hostList[0], "node1");
        SafeCopyTo(params.hostList[1], "node2");

        // Call the method under test
        dcgmReturn_t result = BroadcastRunParametersToRemoteNodes(params);

        // Verify results - should fail with BADPARAM
        REQUIRE(result == DCGM_ST_BADPARAM);

        // Verify that no remote MultinodeRequest calls were made (early return)
        REQUIRE(m_mockDcgmOps.GetMultinodeRequestCallCount() == 0);
    }
}

TEST_CASE_METHOD(MnDiagManagerTests, "FindSshZombieProcesses Tests [mndiag]")
{
    pid_t ourPid = getpid();

    SECTION("Should find ssh zombie processes and filter out all other processes (zombie, not zombie, etc.)")
    {
        MockProcDirectory mockProc("test-proc-ssh-zombies");

        mockProc.CreateProcesses({ { 1234, "(ssh)", 'Z', ourPid },
                                   { 1235, "(sshd)", 'Z', ourPid },
                                   { 1236, "(ssh-agent)", 'Z', ourPid },
                                   { 1237, "(bash)", 'Z', ourPid },
                                   { 1238, "(ssh)", 'S', ourPid } });

        auto result = FindSshZombieProcesses(mockProc.GetPath());

        REQUIRE(result.size() == 3);
        CHECK(std::find(result.begin(), result.end(), 1234) != result.end());
        CHECK(std::find(result.begin(), result.end(), 1235) != result.end());
        CHECK(std::find(result.begin(), result.end(), 1236) != result.end());
    }

    SECTION("Should filter out zombie processes with wrong parent PID")
    {
        MockProcDirectory mockProc("test-proc-wrong-ppid");

        pid_t wrongPpid = ourPid + 1000;

        mockProc.CreateProcesses(
            { { 1234, "(ssh)", 'Z', wrongPpid }, { 1235, "(ssh)", 'Z', 1 }, { 1236, "(ssh)", 'Z', ourPid } });

        auto result = FindSshZombieProcesses(mockProc.GetPath());

        REQUIRE(result.size() == 1);
        CHECK(result[0] == 1236);
    }

    SECTION("Should handle empty proc directory")
    {
        MockProcDirectory mockProc("test-proc-empty");

        auto result = FindSshZombieProcesses(mockProc.GetPath());

        CHECK(result.empty());
    }

    SECTION("Should handle malformed, missing, and empty stat files")
    {
        MockProcDirectory mockProc("test-proc-stat-errors");

        // Create a directory with malformed stat file (invalid format)
        auto pidDir1 = mockProc.GetPath() / "1234";
        mockProc.CreateTestDirectory(pidDir1);
        [[maybe_unused]] auto malformedStatFile = TempFile::Fixed(pidDir1 / "stat", "malformed data\n");

        // Create a PID directory without stat file (missing)
        mockProc.CreateTestDirectory(mockProc.GetPath() / "1236");

        // Create a directory with empty stat file
        auto pidDir3 = mockProc.GetPath() / "9999";
        mockProc.CreateTestDirectory(pidDir3);
        [[maybe_unused]] auto emptyStatFile = TempFile::Fixed(pidDir3 / "stat", "");

        // Create valid SSH zombie processes
        mockProc.CreateProcesses({ { 1235, "(ssh)", 'Z', ourPid }, { 8888, "(sshd)", 'Z', ourPid } });

        auto result = FindSshZombieProcesses(mockProc.GetPath());

        // Should skip malformed, missing, and empty stat files, return only valid ones
        REQUIRE(result.size() == 2);
        CHECK(std::find(result.begin(), result.end(), 1235) != result.end());
        CHECK(std::find(result.begin(), result.end(), 8888) != result.end());
    }

    SECTION("Should skip non-directory entries and directories with non-numeric names")
    {
        MockProcDirectory mockProc("test-proc-non-dirs");

        // Create a valid SSH zombie process
        mockProc.CreateProcesses({ { 5678, "(ssh)", 'Z', ourPid } });

        // Create non-directory entries (files)
        [[maybe_unused]] auto cmdlineFile = TempFile::Fixed(mockProc.GetPath() / "cmdline", "dummy\n");
        [[maybe_unused]] auto statFile    = TempFile::Fixed(mockProc.GetPath() / "stat", "dummy\n");

        // Create directories (common in /proc)
        mockProc.CreateTestDirectory(mockProc.GetPath() / "self");
        mockProc.CreateTestDirectory(mockProc.GetPath() / "thread-self");
        mockProc.CreateTestDirectory(mockProc.GetPath() / "net");
        mockProc.CreateTestDirectory(mockProc.GetPath() / "sys");
        mockProc.CreateTestDirectory(mockProc.GetPath() / "abc123"); // Partial digits

        auto result = FindSshZombieProcesses(mockProc.GetPath());

        // Should only find the valid SSH zombie, skip all directories
        REQUIRE(result.size() == 1);
        CHECK(result[0] == 5678);
    }

    SECTION("Should skip PIDs 0 and 1 (init and kernel)")
    {
        MockProcDirectory mockProc("test-proc-special-pids");

        // Try to create SSH zombies for special PIDs (should be filtered out by implementation)
        mockProc.CreateProcesses(
            { { 0, "(ssh)", 'Z', ourPid }, { 1, "(ssh)", 'Z', ourPid }, { 2, "(ssh)", 'Z', ourPid } });

        auto result = FindSshZombieProcesses(mockProc.GetPath());

        // Should skip PID 0 and 1, but find PID 2
        REQUIRE(result.size() == 1);
        CHECK(result[0] == 2);
    }

    SECTION("Should gracefully handle exceptions from filesystem operations")
    {
        auto removedDir      = TempDir::Random("non-existent-proc-dir-");
        auto nonExistentPath = removedDir.Path();

        // Ensure the directory doesn't exist
        std::error_code ec;
        std::filesystem::remove_all(nonExistentPath, ec);

        auto result = FindSshZombieProcesses(nonExistentPath);

        CHECK(result.empty());
    }
}
