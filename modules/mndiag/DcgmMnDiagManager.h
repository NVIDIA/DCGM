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

#ifndef DCGM_MNDIAG_MANAGER_H
#define DCGM_MNDIAG_MANAGER_H

#include <fmt/format.h>
#include <json/json.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
    dcgmReturn_t ReserveRemoteResources();
    dcgmReturn_t ReleaseRemoteResources();
    dcgmReturn_t DisconnectRemoteNodes();
    dcgmReturn_t BroadcastRunParametersToRemoteNodes(dcgmRunMnDiag_t const &params);
    dcgmReturn_t BroadcastEnvVariablesToRemoteNodes();

    dcgmReturn_t GetNodeInfo();
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
     * @return Vector of PIDs of SSH zombie processes
     */
    std::vector<pid_t> FindSshZombieProcesses();

    /**
     * @brief Initializes the SSH tunnel managers
     */
    void InitializeSSHTunnelManagers();

    /**
     * @brief Gets the path to the mnubergemm binary
     * @return The path to the mnubergemm binary
     */
    std::string GetMnubergemmPathHeadNode();

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

    // Member variables
    HostInfo m_localHostInfo;
    std::optional<size_t> m_authorizedConnection; // Optional ID of the authorized connection
    DcgmMutex m_authMutex { 0 };
    DcgmMutex m_resourceMutex { 0 }; // Mutex to protect m_status

    // hostname -> ConnectionInfo mapping, maybe should be {hostname, gpuid} -> ConnectionInfo
    std::unordered_map<std::string, ConnectionInfo, StringHash, std::equal_to<>> m_connections;

    MnDiagStatus m_status;
    std::unique_ptr<DcgmCoreProxyBase> m_coreProxy;
    size_t m_currentNodeId; // Unique identifier for the current node

    mutable pid_t m_mpiPID { -1 };

    // State machine for managing diagnostic lifecycle
    std::unique_ptr<MnDiagStateMachineBase> m_stateMachine;

    // Resource handle for interacting with the hostengine
    std::unique_ptr<DcgmResourceHandleBase> m_resourceHandle;
    std::unique_ptr<DcgmResourceHandleFactoryBase> m_resourceHandleFactory;

    // DCGM API for making module requests
    std::unique_ptr<DcgmApiBase> m_dcgmApi;

    std::unique_ptr<TcpSSHTunnelManagerBase> m_tcpSSHTunnelManager;
    std::unique_ptr<UdsSSHTunnelManagerBase> m_udsSSHTunnelManager;

    // Factory for creating MPI runners
    std::unique_ptr<MnDiagMpiRunnerFactoryBase> m_mpiRunnerFactory;

    // Set of supported SKUs
    std::unordered_set<std::string> m_supportedSkus;

    nodeInfoMap_t m_nodeInfo;

    friend class MnDiagManagerTests;
};

#endif // DCGM_MNDIAG_MANAGER_H
