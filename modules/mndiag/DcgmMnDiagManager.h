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

#ifndef DCGM_MNDIAG_MANAGER_H
#define DCGM_MNDIAG_MANAGER_H

#include <fmt/format.h>
#include <json/json.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "DcgmMutex.h"
#include "DcgmUtilities.h"
#include "MnDiagMpiRunner.h"
#include "dcgm_agent.h"
#include "dcgm_mndiag_structs.hpp"
#include "dcgm_structs.h"
#include <DcgmCoreProxy.h>
#include <DcgmResourceHandle.h>
#include <SSHTunnelManager.hpp>

#include "MnDiagCommon.h"
#include "MnDiagProcessUtils.h"

#include "DcgmApiBase.h"
#include "DcgmCoreProxyBase.h"
#include "DcgmResourceHandleBase.h"
#include "DcgmResourceHandleFactoryBase.h"
#include "MnDiagMpiRunnerFactoryBase.h"
#include "MnDiagStateMachineBase.h"
#include "TcpSSHTunnelManagerBase.h"
#include "UdsSSHTunnelManagerBase.h"

// Define StringHash for heterogeneous lookups with string_view
struct StringHash : std::hash<std::string_view>
{
    using is_transparent = void;
};

struct ConnectionInfo
{
    dcgmHandle_t handle;
    uint16_t remotePort;
    bool isLoopback;
    uid_t uid;
    std::string remoteSocketPath;
};

struct HostInfo
{
    std::string hostname;
    std::unordered_set<std::string> ipv4Addresses; //!< All local IPv4 addresses (excluding loopback)
    std::string shortHostname;
};

/**
 * Connection parameters for establishing connections (before they're established)
 */
struct ConnectionParams
{
    std::string hostname;                              //!< Remote hostname
    std::variant<uint16_t, std::string> remoteAddress; //!< Remote port or socket path
    std::variant<uint16_t, std::string> localAddress;  //!< Local port or socket path (populated during tunnel creation)
    bool isUnixSocket;                                 //!< True for Unix socket, false for TCP
    uid_t effectiveUid;                                //!< Effective user ID
};

/**
 * Information about the current test
 */
struct TestInfo
{
    dcgmMultinodeTestType_t testType; //!< Type of the test
    std::string testName;             //!< Name of the test
    std::string testBinaryPath;       //!< Path to the test binary
    std::string testPrefix;           //!< Test parameter prefix (e.g. "mnubergemm.")
};

class DcgmMnDiagManager
{
public:
    explicit DcgmMnDiagManager(dcgmCoreCallbacks_t &dcc);
    ~DcgmMnDiagManager();

    /**
     * @brief Handles the run head node command
     * @param params Parameters including host list and diagnostic settings
     * @param effectiveUid Effective UID of the caller
     * @param response Reference to a response structure to be populated with the results
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    dcgmReturn_t HandleRunHeadNode(dcgmRunMnDiag_t const &params, uid_t effectiveUid, dcgmMnDiagResponse_t &response);

    /**
     * @brief Run multi-node diagnostic
     *
     * This method:
     * 1. Parses host list from parameters
     * 2. Reserves resources if available
     * 3. Launches the diagnostic
     * 4. Populates the response structure with the results

     * @param params Parameters including host list and diagnostic settings
     * @param effectiveUid Effective UID of the caller
     * @param response Reference to a response structure to be populated with the results
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    dcgmReturn_t RunHeadNode(dcgmRunMnDiag_t const &params, uid_t effectiveUid, dcgmMnDiagResponse_t &response);

    /*
     * Stops a running multi-node diagnostic if any. Does not stop multi-node diagnostics that are not launched by
     * nv-hostengine .
     *
     * Returns: DCGM_ST_OK on success or if no multi-node diagnostic is currently running.
     *          DCGM_ST_* on failure. Currently there are no failure conditions.
     */
    dcgmReturn_t StopHeadNode();

    // Methods for handling remote node requests
    dcgmReturn_t HandleReserveResources(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t HandleReleaseResources(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t HandleDetectProcess(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t HandleBroadcastRunParameters(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t HandleGetNodeInfo(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t HandleBroadcastEnvVariables(dcgm_module_command_header_t *moduleCommand);

    // Methods to manage authorization
    dcgmReturn_t HandleAuthorizeConnection(size_t connectionId);
    dcgmReturn_t HandleRevokeAuthorization(size_t connectionId);
    dcgmReturn_t HandleIsConnectionAuthorized(size_t connectionId);

    std::unordered_set<std::string> GetSupportedSkus();

private:
    // Methods for head node to communicate with remote nodes
    dcgmReturn_t ConnectRemoteNodes(std::vector<std::string> const &hostList, uid_t effectiveUid);
    dcgmReturn_t ReserveRemoteResources(dcgmMultinodeTestType_t testType);
    dcgmReturn_t ReleaseRemoteResources(dcgmMultinodeTestType_t testType);
    dcgmReturn_t DisconnectRemoteNodes();
    dcgmReturn_t BroadcastRunParametersToRemoteNodes(dcgmRunMnDiag_t const &params, unsigned int timeToRunSeconds = 0);
    dcgmReturn_t BroadcastEnvVariablesToRemoteNodes();

    /**
     * Collects DCGM and driver version info from all nodes in the cluster.
     *
     * @param[in] testType The type of test to get node info for
     * @return DCGM_ST_OK if successful, DCGM_ST_* on failure
     * @note Thread-safe: All accesses to m_nodeInfo are protected by a local mutex
     */
    dcgmReturn_t GetNodeInfo(dcgmMultinodeTestType_t testType);
    // Generate a unique ID for the current node
    size_t GenerateCurrentNodeId() const;

    /**
     * @brief Attempts to acquire resources for diagnostic execution
     *        Callback used by the MnDiagStateMachine
     * @returns DCGM_ST_OK if resources were successfully acquired
     *          DCGM_ST_IN_USE if resources are already in use or state is invalid
     */
    dcgmReturn_t AcquireResources();

    /**
     * @brief Releases resources for diagnostic execution
     *        Callback used by the MnDiagStateMachine
     * @returns DCGM_ST_OK if resources were successfully released
     */
    dcgmReturn_t ReleaseResources();

    /**
     * @brief Sets the status of the diagnostic
     *        Callback used by the MnDiagStateMachine
     * @param status The new status of the diagnostic
     */
    void SetStatus(MnDiagStatus status);

    /**
     * @brief Gets the status of the diagnostic
     * @returns The current status of the diagnostic
     */
    MnDiagStatus GetStatus();

    /**
     * @brief Extracts the hostname and port from a host string
     * @param host The host string to extract from
     * @param hostname The extracted hostname
     * @param port The extracted port
     * @param socketPath The extracted socket path (for Unix socket connections)
     * @return DCGM_ST_OK if successful, DCGM_ST_BADPARAM if the host string is invalid
     */
    dcgmReturn_t ExtractHostnameAndPort(std::string const &host,
                                        std::string &hostname,
                                        uint16_t &port,
                                        std::string &socketPath);

    /**
     * @brief Checks if the devices are supported
     * @return true if the devices are supported, false otherwise
     */
    bool AreDevicesSupported();

    /**
     * @brief Loads supported SKUs from environment variable
     */
    void LoadSupportedSkusFromEnv();

    /**
     * @brief Checks if a host is a loopback address
     * @param host The host string to check
     * @return true if the host is a loopback address, false otherwise
     */
    bool IsLoopback(std::string_view host);

    /**
     * @brief Reaps orphaned SSH processes that were left behind by mpirun
     *
     * Since nv-hostengine is a subreaper, when mpirun exits, its SSH child processes
     * become orphaned and get reparented to nv-hostengine. This function actively
     * reaps these orphaned processes to prevent zombie accumulation.
     */
    void ReapOrphanedSshProcesses();

    /**
     * @brief Finds SSH zombie processes that are children of nv-hostengine
     *
     * Scans /proc to identify zombie processes with "ssh" in their command name
     * that are direct children of the current nv-hostengine process.
     *
     * @param[in] procPath Path to the proc filesystem (defaults to "/proc")
     *
     * @return Vector of PIDs of SSH zombie processes
     */
    std::vector<pid_t> FindSshZombieProcesses(std::filesystem::path const &procPath = "/proc");

    /**
     * @brief Initializes the SSH tunnel managers
     */
    void InitializeSSHTunnelManagers();

    /**
     * @brief Populates the response structure for a failed MPI process
     * @param mpiRunner The MPI runner instance
     * @param response The response structure to populate
     * @param mpiPID The PID of the MPI process
     * @param logPaths The log paths for the MPI process
     */
    void PopulateMpiFailureResponseStruct(MnDiagMpiRunnerBase *mpiRunner,
                                          dcgmMnDiagResponse_v1 &response,
                                          std::optional<pid_t> mpiPID,
                                          DcgmNs::Utils::LogPaths const &logPaths);

    /**
     * @brief Gets the installed CUDA version
     * @return The installed CUDA version
     */
    int GetCudaVersion();

    /**
     * @brief Authorizes remote connections
     * @return DCGM_ST_OK if successful, DCGM_ST_* on failure
     */
    dcgmReturn_t AuthorizeRemoteConnections(dcgmMultinodeTestType_t testType);

    /**
     * @brief Revokes remote authorizations
     * @return DCGM_ST_OK if successful, DCGM_ST_* on failure
     */
    dcgmReturn_t RevokeRemoteAuthorizations();

    /**
     * @brief Starts a tunnel session for the given connection parameters
     * @param params Connection parameters including hostname and address
     * @return TunnelState indicating success or failure
     */
    DcgmNs::Common::RemoteConn::detail::TunnelState StartTunnelSession(ConnectionParams &params);

    /**
     * @brief Ends a tunnel session (opposite of StartTunnelSession)
     * @param params Connection parameters
     */
    void EndTunnelSession(ConnectionParams const &params);

    /**
     * @brief Parses connection parameters from a host string
     * @param host The host string to parse
     * @param effectiveUid The effective user ID
     * @return ConnectionParams structure with parsed information
     */
    ConnectionParams ParseConnectionParams(std::string const &host, uid_t effectiveUid);

    /**
     * @brief Creates a DCGM connection through the established tunnel
     * @param params Connection parameters with local address populated
     * @param handle Output parameter for the DCGM handle
     * @return dcgmReturn_t indicating success or failure
     */
    dcgmReturn_t CreateDcgmConnection(ConnectionParams const &params, dcgmHandle_t &handle);

    /**
     * @brief Creates and stores ConnectionInfo from successful connection
     * @param params Connection parameters used to establish the connection
     * @param handle DCGM connection handle
     * @return ConnectionInfo structure for storage in m_connections
     */
    ConnectionInfo CreateConnectionInfo(ConnectionParams const &params, dcgmHandle_t handle);

    /**
     * @brief Connects to a single remote node using the unified connection logic (without authorization)
     * @param params Connection parameters
     * @return dcgmReturn_t indicating success or failure
     */
    dcgmReturn_t ConnectSingleNode(ConnectionParams params);

    /**
     * @brief Cleans up connections
     * @return DCGM_ST_OK if successful, DCGM_ST_* on failure
     */
    dcgmReturn_t CleanupConnections();

    // Member variables
    HostInfo m_localHostInfo;                     //!< Local host information including hostname and IP addresses
    std::optional<size_t> m_authorizedConnection; //!< Optional ID of the authorized connection
    DcgmMutex m_authMutex { 0 };                  //!< Mutex to protect authorization state
    DcgmMutex m_resourceMutex { 0 };              //!< Mutex to protect m_status

    // Condition variable for status change notifications
    std::condition_variable m_statusCV;

    /*!< hostname -> ConnectionInfo mapping
         TODO: Consider changing to {hostname, gpuid} -> ConnectionInfo */
    std::unordered_map<std::string, ConnectionInfo, StringHash, std::equal_to<>> m_connections;

    MnDiagStatus m_status;                          //!< Current status of the diagnostic
    std::unique_ptr<DcgmCoreProxyBase> m_coreProxy; //!< Proxy for interacting with DCGM core
    size_t m_currentNodeId;                         //!< Unique identifier for the current node
    TestInfo m_currentTestInfo;                     //!< Information about the current test

    mutable pid_t m_mpiPID { -1 }; //!< PID of the MPI process

    std::unique_ptr<MnDiagStateMachineBase> m_stateMachine; //!< State machine for managing diagnostic lifecycle

    std::unique_ptr<DcgmResourceHandleBase> m_resourceHandle; //!< Resource handle for interacting with the hostengine
    std::unique_ptr<DcgmResourceHandleFactoryBase> m_resourceHandleFactory; //!< Factory for creating resource handles

    std::unique_ptr<DcgmApiBase> m_dcgmApi; //!< DCGM API for making module requests

    std::unique_ptr<TcpSSHTunnelManagerBase> m_tcpSSHTunnelManager; //!< Manager for TCP SSH tunnels
    std::unique_ptr<UdsSSHTunnelManagerBase> m_udsSSHTunnelManager; //!< Manager for Unix domain socket SSH tunnels

    std::unique_ptr<MnDiagMpiRunnerFactoryBase> m_mpiRunnerFactory; //!< Factory for creating MPI runners

    std::unordered_set<std::string> m_supportedSkus; //!< Set of supported GPU SKUs

    nodeInfoMap_t m_nodeInfo; //!< Map of node information for all nodes in the cluster

    std::chrono::seconds m_processDetectionTimeout {
        MnDiagConstants::PROCESS_DETECTION_TIMEOUT
    }; //!< Timeout for detecting MPI processes (configurable for testing)
    std::chrono::milliseconds m_processDetectionRetryInterval {
        MnDiagConstants::PROCESS_DETECTION_RETRY_INTERVAL
    }; //!< Retry interval for process detection (configurable for testing)

    friend class MnDiagManagerTests;
};

#endif // DCGM_MNDIAG_MANAGER_H
