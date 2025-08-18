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

#include "DcgmMnDiagManager.h"
#include <DcgmBuildInfo.hpp>
#include <DcgmLogging.h>
#include <DcgmModuleApi.h>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <optional>
#include <pwd.h>
#include <ranges>
#include <sstream>
#include <unordered_set>

#include <arpa/inet.h>
#include <expected>
#include <ifaddrs.h>
#include <netinet/in.h>

#include "DcgmApiAdapter.h"
#include "DcgmCoreProxyAdapter.h"
#include "DcgmResourceHandleAdapter.h"
#include "DcgmResourceHandleFactory.h"
#include "MnDiagMpiRunnerAdapter.h"
#include "MnDiagMpiRunnerFactory.h"
#include "MnDiagStateMachineAdapter.h"
#include "TcpSSHTunnelManagerAdapter.h"
#include "UdsSSHTunnelManagerAdapter.h"

namespace
{

std::unordered_set<std::string> getLocalIPv4Addresses()
{
    struct ifaddrs *ifaddr = nullptr;
    struct ifaddrs *ifa    = nullptr;
    char ip[INET_ADDRSTRLEN];
    std::unordered_set<std::string> addresses;

    if (getifaddrs(&ifaddr) == -1)
    {
        log_error("getifaddrs failed: {}", strerror(errno));
        return addresses; // Return empty set on error
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next)
    {
        if (ifa->ifa_addr == nullptr)
            continue;

        // Check if the interface is IPv4 and is UP
        if (ifa->ifa_addr->sa_family == AF_INET && (ifa->ifa_flags & IFF_UP) && (ifa->ifa_flags & IFF_RUNNING)
            && std::string(ifa->ifa_name) != "lo")
        {
            // Exclude loopback
            struct sockaddr_in *addr = (struct sockaddr_in *)ifa->ifa_addr;
            inet_ntop(AF_INET, &addr->sin_addr, ip, INET_ADDRSTRLEN);
            addresses.insert(ip);
            log_debug("Found local IPv4 address: {} on interface {}", ip, ifa->ifa_name);
        }
    }

    freeifaddrs(ifaddr);
    log_debug("Collected {} unique local IPv4 addresses", addresses.size());
    return addresses;
}

std::expected<HostInfo, dcgmReturn_t> GetLocalHostInfo()
{
    HostInfo hostInfo;

    std::array<char, HOST_NAME_MAX + 1> buffer {}; // guarantees NIL termination
    if (gethostname(buffer.data(), buffer.size() - 1) == 0)
    {
        hostInfo.hostname = std::string(buffer.data());
        std::string_view localHostName(buffer.data());
        auto dotPos = localHostName.find('.');
        if (dotPos != std::string_view::npos)
        {
            hostInfo.shortHostname = localHostName.substr(0, dotPos);
        }
    }
    else
    {
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }

    std::unordered_set<std::string> localIPv4Addresses = getLocalIPv4Addresses();
    if (localIPv4Addresses.empty())
    {
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }

    hostInfo.ipv4Addresses = std::move(localIPv4Addresses);
    return hostInfo;
}

std::optional<std::string> GetUsernameFromUid(uid_t uid)
{
    auto *pwStruct = getpwuid(uid);
    if (pwStruct)
    {
        return pwStruct->pw_name;
    }
    return std::nullopt;
}

} // namespace

DcgmMnDiagManager::DcgmMnDiagManager(dcgmCoreCallbacks_t &dcc)
    : m_status(MnDiagStatus::READY)
    , m_currentNodeId(GenerateCurrentNodeId())
{
    auto localHostInfo = GetLocalHostInfo();
    if (!localHostInfo.has_value())
    {
        log_error("Failed to get local host info");
        m_localHostInfo = HostInfo {
            .hostname      = "unknown-host",
            .ipv4Addresses = { "unknown-ip" },
            .shortHostname = "unknown-host",
        };
    }
    else
    {
        m_localHostInfo = localHostInfo.value();
    }

    m_coreProxy             = std::make_unique<DcgmCoreProxyAdapter>(dcc);
    m_resourceHandleFactory = std::make_unique<DcgmResourceHandleFactory>();
    m_dcgmApi               = std::make_unique<DcgmApiAdapter>();
    m_mpiRunnerFactory      = std::make_unique<MnDiagMpiRunnerFactory>();

    // Create and initialize the state machine with our callback methods
    m_stateMachine = std::make_unique<MnDiagStateMachineAdapter>(
        // Is process running callback
        [this](pid_t pid) { return DcgmNs::Common::ProcessUtils::IsProcessRunning(pid); },
        // Stop process callback
        [this](pid_t pid) { return DcgmNs::Common::ProcessUtils::StopProcess(pid); },
        // Acquire resources callback
        [this]() { return this->AcquireResources(); },
        // Release resources callback
        [this]() { return this->ReleaseResources(); },
        // Set status callback
        [this](MnDiagStatus status) { this->SetStatus(status); });

    // Start the state machine
    m_stateMachine->Start();

    // Supported SKUs - hardcoded - always lowercase
    m_supportedSkus = {
        "2941", // GB200 NVL
    };

    LoadSupportedSkusFromEnv();
}

DcgmMnDiagManager::~DcgmMnDiagManager()
{
    // Stop the state machine
    if (m_stateMachine)
    {
        m_stateMachine->Stop();
    }
}

dcgmReturn_t DcgmMnDiagManager::DisconnectRemoteNodes()
{
    DcgmMutex disconnectMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;

    {
        // Process remote connections in parallel
        for (auto const &[hostname, connInfo] : m_connections)
        {
            if (connInfo.isLoopback)
            {
                continue; // handle the head node last
            }

            auto disconnectNodeFunc = [this, hostname, connInfo, &disconnectMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version                              = dcgmMultinodeRequest_version1;
                request.testType                             = MnDiagTestType::mnubergemm;
                request.requestType                          = MnDiagRequestType::RevokeAuthorization;
                request.requestData.authorization.headNodeId = m_currentNodeId;

                dcgmReturn_t threadResult = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);

                if (threadResult != DCGM_ST_OK)
                {
                    log_error("Failed to revoke authorization for connection: {}. Return: ({}): {}",
                              hostname,
                              std::to_underlying(threadResult),
                              errorString(threadResult));
                    DcgmLockGuard lg(&disconnectMutex);
                    results.emplace_back(hostname, threadResult);
                    anyFailure = true;
                }

                m_dcgmApi->Disconnect(connInfo.handle);
                // End the SSH tunnel session based on socket path or port
                if (!connInfo.remoteSocketPath.empty())
                {
                    // Use UdsSSHTunnelManager for Unix socket connections
                    log_debug(
                        "Ending SSH tunnel with Unix socket path {} for host {}", connInfo.remoteSocketPath, hostname);
                    m_udsSSHTunnelManager->EndSession(hostname, connInfo.remoteSocketPath, connInfo.uid);
                }
                else
                {
                    // Use TcpSSHTunnelManager for TCP connections
                    log_debug("Ending SSH tunnel with port {} for host {}", connInfo.remotePort, hostname);
                    m_tcpSSHTunnelManager->EndSession(hostname, connInfo.remotePort, connInfo.uid);
                }
            };
            disconnectNodeFunc();
        }

        // threads will automatically join when it goes out of scope at the end of this block
    }

    // Clear all connections
    m_connections.clear();
    m_nodeInfo.clear();

    // Handle the head node last
    dcgmReturn_t dcgmResult = HandleRevokeAuthorization(m_currentNodeId);
    if (dcgmResult != DCGM_ST_OK)
    {
        log_error("Failed to revoke authorization for head node");
        return dcgmResult;
    }

    // Check if any disconnections failed
    if (anyFailure)
    {
        for (auto const &[hostname, result] : results)
        {
            if (result != DCGM_ST_OK)
            {
                return result;
            }
        }
        return DCGM_ST_GENERIC_ERROR; // Default error if specific error code not found
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::HandleRunHeadNode(dcgmRunMnDiag_t const &params,
                                                  uid_t effectiveUid,
                                                  dcgmMnDiagResponse_t &response)
{
    InitializeSSHTunnelManagers();
    log_debug("SSH tunnel managers initialized");

    return RunHeadNode(params, effectiveUid, response);
}

dcgmReturn_t DcgmMnDiagManager::RunHeadNode(dcgmRunMnDiag_t const &params,
                                            uid_t effectiveUid,
                                            dcgmMnDiagResponse_t &response)
{
    log_debug("Checking if GPU devices are supported");
    if (!AreDevicesSupported())
    {
        log_error("MnDiag is not supported on the current GPU devices");
        return DCGM_ST_NOT_SUPPORTED;
    }

    std::vector<std::string> hostList;
    for (int i = 0; i < DCGM_MAX_NUM_HOSTS && params.hostList[i][0] != '\0'; i++)
    {
        hostList.emplace_back(params.hostList[i]);
    }

    log_debug("Starting multi-node diagnostic on {} hosts", hostList.size());

    // First connect to all remote nodes, save connections in m_connections
    dcgmReturn_t result = ConnectRemoteNodes(hostList, effectiveUid);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to connect to remote nodes");
        return result;
    }

    // Get dcgm and driver versions from all remote nodes
    result = GetNodeInfo();
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to get node info from remote nodes");
        DisconnectRemoteNodes();
        return result;
    }

    // If all resource checks passed, reserve resources
    result = ReserveRemoteResources();
    if (result != DCGM_ST_OK)
    {
        log_error("Resource reservation failed");
        DisconnectRemoteNodes();
        return result;
    }

    result = BroadcastRunParametersToRemoteNodes(params);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to broadcast parameters to remote nodes");
        DisconnectRemoteNodes();
        return result;
    }

    // Create a new MpiRunner for this run
    auto mpiRunner = m_mpiRunnerFactory->CreateMpiRunner(*m_coreProxy);

    // Set the user info
    {
        auto userName = GetUsernameFromUid(effectiveUid);
        if (userName.has_value())
        {
            mpiRunner->SetUserInfo(std::make_pair(*userName, effectiveUid));
        }
    }

    // Redirect MPI output to files
    DcgmNs::Utils::LogPaths logPaths = DcgmNs::Utils::GetLogFilePath("mndiag_mnubergemm");
    mpiRunner->SetLogFileNames(std::make_pair(logPaths.stdoutFileName.string(), logPaths.stderrFileName.string()));

    // Set the mnubergemm path for the MPI runner on head node
    // This makes sure that MPI runner gets the same path which is broadcasted across nodes
    std::string mnubergemmPath = GetMnubergemmPathHeadNode();
    mpiRunner->SetMnubergemmPath(mnubergemmPath);

    // Generate the MPI command directly from the dcgmRunMnDiag_t struct
    mpiRunner->ConstructMpiCommand(&params);
    log_debug("Generated MPI command: {}", mpiRunner->GetLastCommand());

    // Set the output callback without the response struct - we'll pass it directly in ProcessAndGetMpiOutput
    mpiRunner->SetOutputCallback(
        [&mpiRunner](std::istream &dataStream, void *responseStruct, nodeInfoMap_t const &nodeInfo) {
            return mpiRunner->MnDiagOutputCallback(dataStream, responseStruct, nodeInfo);
        });

    // Launch the MPI process
    result = mpiRunner->LaunchMpiProcess();
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to launch MPI process");
        StopHeadNode();
        return result;
    }

    // Get the PID of the MPI process
    std::optional<pid_t> mpiPID = mpiRunner->GetMpiProcessPid();
    if (!mpiPID.has_value())
    {
        log_error("Failed to get valid MPI process PID");
        StopHeadNode();
        return DCGM_ST_CHILD_SPAWN_FAILED;
    }

    log_debug("Started MPI diagnostic process with PID: {}", *mpiPID);

    // Look up the stdout log file and make sure all processes are launched on worker nodes
    auto allLaunched = mpiRunner->HasMpiLaunchedEnoughProcesses();
    if (!allLaunched.has_value())
    {
        log_error("Not all MPI processes were launched on worker nodes");
        PopulateMpiFailureResponseStruct(mpiRunner.get(), response, *mpiPID, logPaths);
        StopHeadNode();
        return allLaunched.error();
    }

    // Wait for remote nodes to report RUNNING status
    // Confirm the MPI process is running on all remote nodes, otherwise some node either failed or timed out
    DcgmMutex resultMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;

    // Handle the head node first
    // Directly call the detect process function on the head node
    dcgm_mndiag_msg_resource_t dummyMsgOnHeadNode {};
    HandleDetectProcess((dcgm_module_command_header_t *)&dummyMsgOnHeadNode);
    if (dummyMsgOnHeadNode.resource.response != MnDiagStatus::RUNNING)
    {
        log_error("No MPI process running on head node");

        PopulateMpiFailureResponseStruct(mpiRunner.get(), response, *mpiPID, logPaths);
        StopHeadNode();
        return DCGM_ST_CHILD_SPAWN_FAILED;
    }

    {
        std::vector<std::jthread> threads;
        // Launch threads for remote nodes
        for (auto const &[hostname, connInfo] : m_connections)
        {
            if (connInfo.isLoopback)
            {
                continue; // Already handled above
            }

            // Launch a thread for each remote node
            threads.emplace_back([this, hostname, connInfo, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version                         = dcgmMultinodeRequest_version1;
                request.testType                        = MnDiagTestType::mnubergemm;
                request.requestType                     = MnDiagRequestType::DetectProcess;
                request.requestData.resource.headNodeId = m_currentNodeId;
                request.requestData.resource.response   = MnDiagStatus::UNKNOWN;

                dcgmReturn_t threadResult = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);

                {
                    DcgmLockGuard lg(&resultMutex);
                    results.emplace_back(hostname, threadResult);
                    if (threadResult != DCGM_ST_OK || request.requestData.resource.response != MnDiagStatus::RUNNING)
                    {
                        anyFailure = true;
                    }
                }
                // Log errors for failed messages
                if (threadResult != DCGM_ST_OK)
                {
                    log_error("Failed to send detect process message to remote node {}. Return: ({}): {}",
                              hostname,
                              std::to_underlying(threadResult),
                              errorString(threadResult));
                }
                else if (request.requestData.resource.response != MnDiagStatus::RUNNING)
                {
                    log_error("No MPI process running on remote node {}", hostname);
                }
            });
        }

        // threads will automatically join when it goes out of scope at the end of this block
    }

    // Check if any failures occurred
    if (anyFailure)
    {
        StopHeadNode();
        for (auto const &[hostname, result] : results)
        {
            if (result != DCGM_ST_OK)
            {
                return result;
            }
        }
        return DCGM_ST_GENERIC_ERROR; // Default error if specific error code not found
    }
    // Block and wait for the MPI process to complete
    log_debug("Waiting for MPI process with PID {} to complete...", *mpiPID);

    dcgmReturn_t waitResult = mpiRunner->Wait(); // using default timeoutSec -1, wait till the process finishes

    // No matter what is returned form Wait, we need to populate the response structure
    PopulateMpiFailureResponseStruct(mpiRunner.get(), response, *mpiPID, logPaths);

    if (waitResult != DCGM_ST_OK)
    {
        log_error("Failed to wait for MPI process with PID {}: {}", *mpiPID, errorString(waitResult));
        StopHeadNode();
        return waitResult;
    }

    // Release resources and clean up
    StopHeadNode();

    return DCGM_ST_OK;
}

/****************************************************************************/
dcgmReturn_t DcgmMnDiagManager::StopHeadNode()
{
    dcgmReturn_t finalResult = DCGM_ST_OK;

    // Continue cleanup even if individual steps fail, but track errors
    dcgmReturn_t releaseResult = ReleaseRemoteResources();
    if (releaseResult != DCGM_ST_OK)
    {
        log_error("Failed to release remote resources: {}", errorString(releaseResult));
        finalResult = releaseResult; // Remember first error
    }

    dcgmReturn_t disconnectResult = DisconnectRemoteNodes();
    if (disconnectResult != DCGM_ST_OK)
    {
        log_error("Failed to disconnect remote nodes: {}", errorString(disconnectResult));
        if (finalResult == DCGM_ST_OK)
            finalResult = disconnectResult;
    }

    // Reset SSH tunnel managers
    log_debug("Before resetting SSH tunnel managers");
    m_tcpSSHTunnelManager.reset();
    m_udsSSHTunnelManager.reset();
    log_debug("SSH tunnel managers reset");

    // Reap any orphaned SSH processes
    ReapOrphanedSshProcesses();

    // Reset ChildProcessManager - critical for system stability
    dcgmReturn_t resetResult = m_coreProxy->ChildProcessManagerReset();
    if (resetResult != DCGM_ST_OK)
    {
        log_error("Failed to reset ChildProcessManager: {}", errorString(resetResult));
        if (finalResult == DCGM_ST_OK)
            finalResult = resetResult;
    }
    else
    {
        log_debug("Successfully reset ChildProcessManager");
    }

    return finalResult; // Let caller decide how to handle partial cleanup failures
}

dcgmReturn_t DcgmMnDiagManager::ConnectRemoteNodes(std::vector<std::string> const &hostList, uid_t effectiveUid)
{
    DcgmMutex connectionMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;
    // Get the username from the effective UID
    {
        std::optional<std::string> username;
        username = GetUsernameFromUid(effectiveUid);
        if (!username.has_value())
        {
            log_error("Failed to get username for uid: {}, error: {}", effectiveUid, strerror(errno));
            return DCGM_ST_NO_PERMISSION;
        }
    }
    // Handle duplicated hosts in the host list
    {
        std::unordered_set<std::string> uniqueHosts;
        for (auto const &host : hostList)
        {
            // Parse host string to extract hostname and port
            std::string hostname;
            uint16_t remotePort;
            std::string socketPath;

            dcgmReturn_t res = ExtractHostnameAndPort(host, hostname, remotePort, socketPath);
            if (res != DCGM_ST_OK)
            {
                log_error("Failed to extract hostname and port from host: {}", host);
                return res;
            }

            if (!uniqueHosts.insert(hostname).second)
            {
                log_error("Duplicate host found in input list: {}", host);
                return DCGM_ST_BADPARAM;
            }
        }
    }
    // First handle loopback connections
    // Authorize itself on the head node
    dcgmReturn_t dcgmResult = HandleAuthorizeConnection(m_currentNodeId);
    if (dcgmResult != DCGM_ST_OK)
    {
        log_error("Failed to authorize itself on the head node");
        DisconnectRemoteNodes();
        return dcgmResult;
    }
    // Now handle remote connections asynchronously
    {
        for (auto const &host : hostList)
        {
            // Parse host string to extract hostname and port
            std::string hostname;
            uint16_t remotePort;
            std::string socketPath;

            dcgmReturn_t res = ExtractHostnameAndPort(host, hostname, remotePort, socketPath);
            if (res != DCGM_ST_OK)
            {
                log_error("Failed to extract hostname and port from host: {}", host);
                DisconnectRemoteNodes();
                return res;
            }

            if (IsLoopback(hostname))
            {
                m_connections[hostname] = ConnectionInfo {
                    .handle = 0, .remotePort = 0, .isLoopback = true, .uid = effectiveUid, .remoteSocketPath = ""
                };
                continue; // Already handled above
            }
            auto connectNodeFunc = [this,
                                    hostname,
                                    remotePort,
                                    socketPath,
                                    effectiveUid,
                                    &connectionMutex,
                                    &results,
                                    &anyFailure]() {
                // Create SSH tunnel to remote host
                DcgmNs::Common::RemoteConn::detail::TunnelState tunnelState;
                dcgmHandle_t handle;
                if (!socketPath.empty())
                {
                    // Connect using Unix socket path with UdsSSHTunnelManager
                    std::string localSocketPath;

                    log_debug("Starting SSH tunnel with Unix socket path {} for host {}", socketPath, hostname);
                    tunnelState
                        = m_udsSSHTunnelManager->StartSession(hostname, socketPath, localSocketPath, effectiveUid);
                    if (tunnelState != DcgmNs::Common::RemoteConn::detail::TunnelState::Active)
                    {
                        {
                            DcgmLockGuard lg(&connectionMutex);
                            results.emplace_back(hostname, DCGM_ST_REMOTE_SSH_CONNECTION_FAILED);
                            anyFailure = true;
                        }
                        log_error("Failed to create SSH tunnel to {} with socket path {} (TunnelState: {})",
                                  hostname,
                                  socketPath,
                                  static_cast<std::underlying_type_t<DcgmNs::Common::RemoteConn::detail::TunnelState>>(
                                      tunnelState));
                        return;
                    }
                    // Initialize V2 connection parameters
                    dcgmConnectV2Params_v2 connectParams;
                    memset(&connectParams, 0, sizeof(connectParams));
                    connectParams.version             = dcgmConnectV2Params_version2;
                    connectParams.addressIsUnixSocket = 1;
                    dcgmReturn_t dcgmResult = m_dcgmApi->Connect_v2(localSocketPath.c_str(), &connectParams, &handle);
                    if (dcgmResult != DCGM_ST_OK)
                    {
                        log_error(
                            "Failed to connect to remote DCGM through UDS {} for host {}", localSocketPath, hostname);
                        m_udsSSHTunnelManager->EndSession(hostname, socketPath, effectiveUid);
                        return;
                    }

                    dcgmMultinodeRequest_t request {};
                    request.version                              = dcgmMultinodeRequest_version1;
                    request.testType                             = MnDiagTestType::mnubergemm;
                    request.requestType                          = MnDiagRequestType::AuthorizeConnection;
                    request.requestData.authorization.headNodeId = m_currentNodeId;

                    dcgmResult = m_dcgmApi->MultinodeRequest(handle, &request);
                    if (dcgmResult != DCGM_ST_OK)
                    {
                        {
                            DcgmLockGuard lg(&connectionMutex);
                            results.emplace_back(hostname, dcgmResult);
                            anyFailure = true;
                        }
                        log_error("Failed to authorize connection with {}: {}", hostname, errorString(dcgmResult));
                        m_dcgmApi->Disconnect(handle);
                        m_udsSSHTunnelManager->EndSession(hostname, socketPath, effectiveUid);
                        return;
                    }
                    // Store connection info
                    {
                        DcgmLockGuard lg(&connectionMutex);
                        m_connections[hostname] = ConnectionInfo { .handle           = handle,
                                                                   .remotePort       = 0,
                                                                   .isLoopback       = false,
                                                                   .uid              = effectiveUid,
                                                                   .remoteSocketPath = socketPath };
                        results.emplace_back(hostname, DCGM_ST_OK);
                    }
                }
                else
                {
                    // Connect using TCP port
                    uint16_t localPort;

                    log_debug("Starting SSH tunnel with port {} for host {}", remotePort, hostname);
                    tunnelState = m_tcpSSHTunnelManager->StartSession(hostname, remotePort, localPort, effectiveUid);
                    if (tunnelState != DcgmNs::Common::RemoteConn::detail::TunnelState::Active)
                    {
                        {
                            DcgmLockGuard lg(&connectionMutex);
                            results.emplace_back(hostname, DCGM_ST_REMOTE_SSH_CONNECTION_FAILED);
                            anyFailure = true;
                        }

                        log_error("Failed to create SSH tunnel to {}:{} (TunnelState: {})",
                                  hostname,
                                  remotePort,
                                  static_cast<std::underlying_type_t<DcgmNs::Common::RemoteConn::detail::TunnelState>>(
                                      tunnelState));
                        return;
                    }
                    // Create connection to remote DCGM through tunnel
                    std::string address = fmt::format("127.0.0.1:{}", localPort);

                    // Initialize V2 connection parameters
                    dcgmConnectV2Params_v2 connectParams;
                    memset(&connectParams, 0, sizeof(connectParams));
                    connectParams.version             = dcgmConnectV2Params_version2;
                    connectParams.addressIsUnixSocket = 0;
                    dcgmReturn_t dcgmResult           = m_dcgmApi->Connect_v2(address.c_str(), &connectParams, &handle);

                    if (dcgmResult != DCGM_ST_OK)
                    {
                        {
                            DcgmLockGuard lg(&connectionMutex);
                            results.emplace_back(hostname, dcgmResult);
                            anyFailure = true;
                        }
                        log_error("Failed to connect to remote DCGM {} from {}", address, hostname);
                        m_tcpSSHTunnelManager->EndSession(hostname, remotePort, effectiveUid);
                        return;
                    }
                    dcgmMultinodeRequest_t request {};
                    request.version                              = dcgmMultinodeRequest_version1;
                    request.testType                             = MnDiagTestType::mnubergemm;
                    request.requestType                          = MnDiagRequestType::AuthorizeConnection;
                    request.requestData.authorization.headNodeId = m_currentNodeId;

                    dcgmResult = m_dcgmApi->MultinodeRequest(handle, &request);
                    if (dcgmResult != DCGM_ST_OK)
                    {
                        {
                            DcgmLockGuard lg(&connectionMutex);
                            results.emplace_back(hostname, dcgmResult);
                            anyFailure = true;
                        }

                        log_error("Failed to authorize connection with {}: {}", hostname, errorString(dcgmResult));
                        m_dcgmApi->Disconnect(handle);
                        m_tcpSSHTunnelManager->EndSession(hostname, remotePort, effectiveUid);
                        return;
                    }
                    // Store connection info
                    {
                        DcgmLockGuard lg(&connectionMutex);
                        m_connections[hostname] = ConnectionInfo { .handle           = handle,
                                                                   .remotePort       = remotePort,
                                                                   .isLoopback       = false,
                                                                   .uid              = effectiveUid,
                                                                   .remoteSocketPath = "" };
                        results.emplace_back(hostname, DCGM_ST_OK);
                    }
                }
            };
            connectNodeFunc();
        }

        // threads will automatically join when it goes out of scope at the end of this block
    }

    // Check if any connections failed
    if (anyFailure)
    {
        DisconnectRemoteNodes();
        for (auto const &[hostname, result] : results)
        {
            if (result != DCGM_ST_OK)
            {
                return result;
            }
        }
        return DCGM_ST_GENERIC_ERROR; // Default error if specific error code not found
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::ReserveRemoteResources()
{
    // First set all remote nodes to RESERVED status
    DcgmMutex resultMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;

    // Handle the head node first
    // Directly call the reserve resources function on the head node
    dcgm_mndiag_msg_resource_t dummyMsgOnHeadNode {};
    HandleReserveResources((dcgm_module_command_header_t *)&dummyMsgOnHeadNode);
    if (dummyMsgOnHeadNode.resource.response != MnDiagStatus::RESERVED)
    {
        log_error("Failed to reserve resources on head node");
        return DCGM_ST_IN_USE;
    }


    {
        std::vector<std::jthread> threads;

        // Launch threads for remote nodes
        for (auto const &[hostname, connInfo] : m_connections)
        {
            if (connInfo.isLoopback)
            {
                continue; // Already handled above
            }

            // Launch a thread for each remote node
            threads.emplace_back([this, hostname, connInfo, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version                         = dcgmMultinodeRequest_version1;
                request.testType                        = MnDiagTestType::mnubergemm;
                request.requestType                     = MnDiagRequestType::ReserveResources;
                request.requestData.resource.headNodeId = m_currentNodeId;
                request.requestData.resource.response   = MnDiagStatus::UNKNOWN;

                dcgmReturn_t threadResult = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);
                {
                    DcgmLockGuard lg(&resultMutex);
                    results.emplace_back(hostname, threadResult);
                    if (threadResult != DCGM_ST_OK || request.requestData.resource.response != MnDiagStatus::RESERVED)
                    {
                        anyFailure = true;
                    }
                }
                // Log errors for failed messages
                if (threadResult != DCGM_ST_OK)
                {
                    log_error("Failed to send reserve resources message to remote node {}. Return: ({}): {}",
                              hostname,
                              std::to_underlying(threadResult),
                              errorString(threadResult));
                }
                else if (request.requestData.resource.response != MnDiagStatus::RESERVED)
                {
                    log_error("Failed to set remote node {} status to RESERVED", hostname);
                }
            });
        }

        // threads will automatically join when it goes out of scope at the end of this block
    }

    // Check if any failures occurred
    if (anyFailure)
    {
        ReleaseRemoteResources();
        for (auto const &[hostname, result] : results)
        {
            if (result != DCGM_ST_OK)
            {
                return result;
            }
        }
        return DCGM_ST_IN_USE; // Default error if specific error code not found
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::ReleaseRemoteResources()
{
    DcgmMutex resultMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;

    {
        std::vector<std::jthread> threads;

        // Launch threads for remote nodes
        for (auto const &[hostname, connInfo] : m_connections)
        {
            if (connInfo.isLoopback)
            {
                continue; // handle the head node last
            }

            // Launch a thread for each remote node
            threads.emplace_back([this, hostname, connInfo, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version                         = dcgmMultinodeRequest_version1;
                request.testType                        = MnDiagTestType::mnubergemm;
                request.requestType                     = MnDiagRequestType::ReleaseResources;
                request.requestData.resource.headNodeId = m_currentNodeId;
                request.requestData.resource.response   = MnDiagStatus::UNKNOWN;

                dcgmReturn_t threadResult = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);

                {
                    DcgmLockGuard lg(&resultMutex);
                    results.emplace_back(hostname, threadResult);
                    if (threadResult != DCGM_ST_OK || request.requestData.resource.response != MnDiagStatus::READY)
                    {
                        anyFailure = true;
                    }
                }
                // Log errors for failed messages
                if (threadResult != DCGM_ST_OK)
                {
                    log_error("Failed to send release resources message to remote node {}. Return: ({}): {}",
                              hostname,
                              std::to_underlying(threadResult),
                              errorString(threadResult));
                }
                else if (request.requestData.resource.response != MnDiagStatus::READY)
                {
                    log_error("Failed to release resources on remote node {}", hostname);
                }
            });
        }

        // threads will automatically join when it goes out of scope at the end of this block
    }

    // Check if any failures occurred
    if (anyFailure)
    {
        for (auto const &[hostname, result] : results)
        {
            if (result != DCGM_ST_OK)
            {
                return result;
            }
        }
        return DCGM_ST_GENERIC_ERROR; // Default error if specific error code not found
    }

    // Handle the head node last
    // Directly call the release resources function on the head node
    dcgm_mndiag_msg_resource_t dummyMsgOnHeadNode {};
    HandleReleaseResources((dcgm_module_command_header_t *)&dummyMsgOnHeadNode);
    if (dummyMsgOnHeadNode.resource.response != MnDiagStatus::READY)
    {
        log_error("Failed to release resources on head node");
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::GetNodeInfo()
{
    DcgmMutex resultMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;
    // Handle the head node first
    // Directly call the reserve resources function on the head node
    dcgm_mndiag_msg_node_info_t dummyMsgOnHeadNode {};
    if (auto ret = HandleGetNodeInfo((dcgm_module_command_header_t *)&dummyMsgOnHeadNode); ret != DCGM_ST_OK)
    {
        log_error("Failed to get node info on head node: {}", errorString(ret));
        return ret;
    }
    std::string localDcgmVersion   = dummyMsgOnHeadNode.nodeInfo.dcgmVersion;
    std::string localDriverVersion = dummyMsgOnHeadNode.nodeInfo.driverVersion;
    {
        std::vector<std::jthread> threads;

        // Launch threads for remote nodes
        for (auto const &[hostname, connInfo] : m_connections)
        {
            if (connInfo.isLoopback)
            {
                log_debug("Adding version info for host {}: dcgmVersion - {}, driverVersion - {}",
                          hostname,
                          localDcgmVersion,
                          localDriverVersion);
                m_nodeInfo[hostname].dcgmVersion   = localDcgmVersion;
                m_nodeInfo[hostname].driverVersion = localDriverVersion;
                continue; // Already handled above
            }

            // Launch a thread for each remote node
            threads.emplace_back([this, hostname, connInfo, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version     = dcgmMultinodeRequest_version1;
                request.testType    = MnDiagTestType::mnubergemm;
                request.requestType = MnDiagRequestType::GetNodeInfo;

                dcgmReturn_t threadResult = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);
                {
                    DcgmLockGuard lg(&resultMutex);
                    results.emplace_back(hostname, threadResult);
                    if (threadResult != DCGM_ST_OK)
                    {
                        anyFailure = true;
                    }
                }
                // Log errors for failed messages
                if (threadResult != DCGM_ST_OK)
                {
                    log_error("Failed to send get node info message to remote node {}. Return: ({}): {}",
                              hostname,
                              std::to_underlying(threadResult),
                              errorString(threadResult));
                }
                else
                {
                    log_debug("Adding version info for host {}: dcgmVersion - {}, driverVersion - {}",
                              hostname,
                              request.requestData.nodeInfo.dcgmVersion,
                              request.requestData.nodeInfo.driverVersion);
                    m_nodeInfo[hostname].dcgmVersion   = request.requestData.nodeInfo.dcgmVersion;
                    m_nodeInfo[hostname].driverVersion = request.requestData.nodeInfo.driverVersion;
                }
            });
        }

        // threads will automatically join when it goes out of scope at the end of this block
    }

    // Check if any failures occurred
    if (anyFailure)
    {
        for (auto const &[hostname, result] : results)
        {
            if (result != DCGM_ST_OK)
            {
                return result;
            }
        }
        return DCGM_ST_GENERIC_ERROR; // Default error if specific error code not found
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::HandleGetNodeInfo(dcgm_module_command_header_t *moduleCommand)
{
    // Cast to our new struct type
    dcgm_mndiag_msg_node_info_t *nodeInfoMsg = std::bit_cast<dcgm_mndiag_msg_node_info_t *>(moduleCommand);

    auto dcgmVersion = std::string(DcgmNs::DcgmBuildInfo().GetVersion());
    SafeCopyTo(nodeInfoMsg->nodeInfo.dcgmVersion, dcgmVersion.c_str());
    std::string driverVersion;
    if (auto ret = m_coreProxy->GetDriverVersion(driverVersion); ret != DCGM_ST_OK)
    {
        log_error("Failed to get driver version: {}", errorString(ret));
        return ret;
    }
    SafeCopyTo(nodeInfoMsg->nodeInfo.driverVersion, driverVersion.c_str());
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::HandleReserveResources(dcgm_module_command_header_t *moduleCommand)
{
    // Cast to our new struct type
    dcgm_mndiag_msg_resource_t *resourceMsg = std::bit_cast<dcgm_mndiag_msg_resource_t *>(moduleCommand);

    // Notify the state machine to reserve resources
    if (!m_stateMachine->NotifyToReserve())
    {
        log_error("State machine fail to reserve resources");
        return DCGM_ST_IN_USE;
    }

    // m_status is set in the state machine using the callback function
    resourceMsg->resource.response = GetStatus();

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::HandleReleaseResources(dcgm_module_command_header_t *moduleCommand)
{
    // Cast to our new struct type
    dcgm_mndiag_msg_resource_t *resourceMsg = std::bit_cast<dcgm_mndiag_msg_resource_t *>(moduleCommand);

    log_debug("Handling release resources request");

    // Notify the state machine that the diagnostic is finished
    m_stateMachine->NotifyDiagnosticFinished();

    // Set the response to the status
    resourceMsg->resource.response = MnDiagStatus::READY;

    log_debug("Resources released successfully");
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::HandleDetectProcess(dcgm_module_command_header_t *moduleCommand)
{
    // Cast to our new struct type
    dcgm_mndiag_msg_resource_t *resourceMsg = std::bit_cast<dcgm_mndiag_msg_resource_t *>(moduleCommand);

    // Try to get the detected MPI process PID from the state machine
    // Head node has noticed processes running on worker nodes,
    // so worker nodes need to verify the processes are occupying GPUs
    if (auto ret = m_stateMachine->TryGetDetectedMpiPid(); ret == DCGM_ST_OK)
    {
        log_debug("MPI process detected");
    }
    else
    {
        log_error("No MPI process related to this run has been detected on compute nodes");
        // Let the state machine handle the resource release
        m_stateMachine->NotifyDiagnosticFinished();
    }

    // m_status is set in the state machine using the callback function
    resourceMsg->resource.response = GetStatus();

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::HandleBroadcastRunParameters(dcgm_module_command_header_t *moduleCommand)
{
    // Cast to our parameter broadcast struct type
    dcgm_mndiag_msg_run_params_t *paramMsg = std::bit_cast<dcgm_mndiag_msg_run_params_t *>(moduleCommand);

    log_debug("Handling broadcast parameters request from head node {}, current node ID - {}",
              paramMsg->runParams.headNodeId,
              static_cast<unsigned int>(m_currentNodeId));

    // Process the time_to_run parameter
    std::optional<unsigned int> time_to_run;
    size_t testParmsLen
        = std::min(static_cast<size_t>(DCGM_MAX_TEST_PARMS), std::size(paramMsg->runParams.runMnDiag.testParms));
    std::span<std::remove_reference_t<decltype(paramMsg->runParams.runMnDiag.testParms[0])>> testParmsSpan(
        paramMsg->runParams.runMnDiag.testParms, testParmsLen);
    for (auto const &param_cstr : testParmsSpan)
    {
        if (param_cstr[0] == '\0')
        {
            break; // Stop at first empty parameter
        }

        std::string_view param(param_cstr);
        // param is in the format "mnubergemm.param=value;mnubergemm.flag;mnubergemm.time_to_run=value;"
        if (auto posKey = param.find("mnubergemm.time_to_run"); posKey != std::string::npos)
        {
            // we've found mnubergemm.time_to_run, now we need to find the value
            if (auto posValue = param.find("=", posKey); posValue != std::string::npos)
            {
                time_to_run = std::stoi(std::string(param.substr(posValue + 1)));
                log_debug("Received time_to_run parameter: {}", *time_to_run);
            }
            else
            {
                log_error("Invalid time_to_run parameter format: {}", param);
                return DCGM_ST_BADPARAM;
            }
            break;
        }
    }

    if (time_to_run.has_value())
    {
        m_stateMachine->SetProcessExecutionTimeout(std::chrono::seconds(*time_to_run));
    }

    // Set the mnubergemm path in the state machine
    if (paramMsg->runParams.mnubergemmPath[0] == '\0')
    {
        log_warning("Received empty mnubergemm path on node {}, not setting path.",
                    static_cast<unsigned int>(m_currentNodeId));
        m_stateMachine->SetMnubergemmPath("");
    }
    else
    {
        log_debug("Setting mnubergemm path to {} on node {}",
                  paramMsg->runParams.mnubergemmPath,
                  static_cast<unsigned int>(m_currentNodeId));
        m_stateMachine->SetMnubergemmPath(paramMsg->runParams.mnubergemmPath);
    }

    log_debug("Successfully processed broadcast parameters");
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::HandleAuthorizeConnection(size_t connectionId)
{
    dcgmReturn_t result = DCGM_ST_OK;

    {
        DcgmLockGuard lg(&m_authMutex);

        if (m_authorizedConnection.has_value() && *m_authorizedConnection != connectionId)
        {
            result = DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED;
        }
        else
        {
            m_authorizedConnection = connectionId;
        }
    }

    if (result != DCGM_ST_OK)
    {
        log_warning("Rejecting authorization request from connection {} - already authorized to {}",
                    connectionId,
                    m_authorizedConnection.value());
    }
    else
    {
        log_info("Authorized connection: {}", connectionId);
    }

    return result;
}

dcgmReturn_t DcgmMnDiagManager::HandleRevokeAuthorization(size_t connectionId)
{
    std::optional<size_t> authorizedConnection;

    {
        DcgmLockGuard lg(&m_authMutex);
        authorizedConnection = m_authorizedConnection;
        if (authorizedConnection.has_value() && authorizedConnection.value() == connectionId)
        {
            m_authorizedConnection.reset();
        }
    }

    // Handle error cases outside the critical section
    if (!authorizedConnection.has_value())
    {
        log_warning("Rejecting revocation request from connection {}: no connection is currently authorized",
                    connectionId);
        return DCGM_ST_MNDIAG_CONNECTION_NOT_AVAILABLE;
    }

    if (authorizedConnection.value() != connectionId)
    {
        log_warning("Rejecting revocation request from connection {}: authorized connection is {}",
                    connectionId,
                    authorizedConnection.value());
        return DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED;
    }

    // Success case - we know the connection was revoked in the critical section
    log_info("Revoked authorization for connection: {}", connectionId);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::HandleIsConnectionAuthorized(size_t connectionId)
{
    DcgmLockGuard lg(&m_authMutex);

    if (!m_authorizedConnection.has_value())
    {
        return DCGM_ST_MNDIAG_CONNECTION_NOT_AVAILABLE;
    }

    if (m_authorizedConnection.value() != connectionId)
    {
        return DCGM_ST_MNDIAG_CONNECTION_UNAUTHORIZED;
    }

    return DCGM_ST_OK;
}

// Generate a unique ID for this head node based on hostname
size_t DcgmMnDiagManager::GenerateCurrentNodeId() const
{
    char hostname[256];
    // Initialize the buffer to ensure it's not used uninitialized in case of error
    memset(hostname, 0, sizeof(hostname));

    if (gethostname(hostname, sizeof(hostname)) != 0)
    {
        // Handle error case
        log_error("Failed to get hostname: {}", strerror(errno));

        // Use a fallback value based on current time and process ID
        std::string fallback = fmt::format("fallback_host_{}_{}", time(nullptr), getpid());
        log_info("Using fallback hostname: {}", fallback);

        // Copy fallback to hostname buffer
        SafeCopyTo(hostname, fallback.c_str());
    }
    else if (hostname[sizeof(hostname) - 1] != '\0')
    {
        // In case hostname was truncated and not null-terminated
        hostname[sizeof(hostname) - 1] = '\0';
        log_warning("Hostname may have been truncated: {}", hostname);
    }

    // Create a hash of the hostname
    std::hash<std::string> hasher;
    size_t hash = static_cast<size_t>(hasher(hostname));

    log_info("Generated head node ID: {} from hostname: {}", hash, hostname);
    return hash;
}

dcgmReturn_t DcgmMnDiagManager::AcquireResources()
{
    // This is the callback used by the StateMachine
    if (GetStatus() == MnDiagStatus::READY)
    {
        auto resourceHandle
            = m_resourceHandleFactory->CreateDcgmResourceHandle(m_coreProxy->GetUnderlyingDcgmCoreProxy());
        dcgmReturn_t resRet = resourceHandle->GetInitResult();
        if (resRet != DCGM_ST_OK)
        {
            log_error("Cannot reserve resources - {} already in use", errorString(resRet));
            return resRet;
        }
        m_resourceHandle = std::move(resourceHandle);
        SetStatus(MnDiagStatus::RESERVED);
        return DCGM_ST_OK;
    }

    log_error("Cannot reserve resources - not in READY state");
    return DCGM_ST_IN_USE;
}

dcgmReturn_t DcgmMnDiagManager::ReleaseResources()
{
    // This is the callback used by the StateMachine
    m_resourceHandle.reset();
    SetStatus(MnDiagStatus::READY);
    return DCGM_ST_OK;
}

void DcgmMnDiagManager::SetStatus(MnDiagStatus status)
{
    // This is the callback used by the StateMachine
    DcgmLockGuard lg(&m_resourceMutex);
    m_status = status;
}

MnDiagStatus DcgmMnDiagManager::GetStatus()
{
    DcgmLockGuard lg(&m_resourceMutex);
    return m_status;
}

dcgmReturn_t DcgmMnDiagManager::ExtractHostnameAndPort(std::string const &host,
                                                       std::string &hostname,
                                                       uint16_t &port,
                                                       std::string &socketPath)
{
    hostname   = "";
    port       = 0;
    socketPath = "";

    if (host.empty())
    {
        log_error("Empty host string provided");
        return DCGM_ST_BADPARAM;
    }

    auto equalPos = host.find('=');
    std::string hostPart;

    if (equalPos != std::string::npos)
    {
        hostPart = host.substr(0, equalPos);

        if (hostPart.empty())
        {
            log_error("Missing hostname before GPU list in: {}", host);
            return DCGM_ST_BADPARAM;
        }
    }
    else
    {
        hostPart = host;
    }

    auto colonPos = hostPart.find(':');
    if (colonPos != std::string::npos)
    {
        hostname = hostPart.substr(0, colonPos);

        if (hostname.empty())
        {
            log_error("Empty hostname before colon in: {}", host);
            return DCGM_ST_BADPARAM;
        }

        std::string remainder = hostPart.substr(colonPos + 1);

        if (remainder.starts_with(DCGM_UNIX_SOCKET_PREFIX))
        {
            // Store only the path part after the unix:// prefix
            socketPath = remainder.substr(std::string(DCGM_UNIX_SOCKET_PREFIX).length());
            return DCGM_ST_OK;
        }
        else
        {
            try
            {
                int portNum = DcgmNs::strictStrToInt(remainder);
                if (portNum < 0 || portNum > 65535)
                {
                    log_error("Port number out of range (0-65535) in host: {}", host);
                    return DCGM_ST_BADPARAM;
                }
                port = static_cast<uint16_t>(portNum);
            }
            catch (std::exception const &ex)
            {
                log_error("Invalid port number in host string: {}, error: {}", host, ex.what());
                return DCGM_ST_BADPARAM;
            }
        }
    }
    else
    {
        hostname = hostPart;
        port     = DCGM_HE_PORT_NUMBER; // Use default port
    }

    return DCGM_ST_OK;
}

bool DcgmMnDiagManager::AreDevicesSupported()
{
    std::vector<unsigned int> gpuIds;
    std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;

    // Get the GPU IDs and info
    dcgmReturn_t ret = m_coreProxy->GetGpuIds(1, gpuIds); //Active gpus only
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to get GPU IDs");
        return false;
    }

    ret = m_coreProxy->GetAllGpuInfo(gpuInfo);
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to get GPU info");
        return false;
    }

    if (gpuIds.empty() || gpuInfo.empty())
    {
        log_error("No GPUs found or no GPU info found");
        return false;
    }

    bool areAllSameSku = m_coreProxy->AreAllGpuIdsSameSku(gpuIds);
    if (!areAllSameSku)
    {
        log_error("All GPUs must be the same SKU");
        return false;
    }

    // Check if first GPU is supported by looking into the SKU list, all other gpus are of same SKU
    unsigned int gpuId = gpuIds[0];
    auto it            = std::find_if(
        gpuInfo.begin(), gpuInfo.end(), [gpuId](const dcgmcm_gpu_info_cached_t &info) { return info.gpuId == gpuId; });

    if (it == gpuInfo.end())
    {
        log_error("No GPU info found for GPU ID {}", gpuId);
        return false;
    }

    std::stringstream ss;
    unsigned int pciDeviceId = it->pciInfo.pciDeviceId >> 16;
    ss << std::hex << std::setw(4) << std::setfill('0') << pciDeviceId;
    std::string deviceId = ss.str();
    std::ranges::transform(deviceId, deviceId.begin(), ::tolower);

    ss.str("");
    ss.clear();
    unsigned int pciSubSystemId = it->pciInfo.pciSubSystemId >> 16;
    ss << std::hex << std::setw(4) << std::setfill('0') << pciSubSystemId;
    std::string ssid = ss.str();
    std::ranges::transform(ssid, ssid.begin(), ::tolower);

    if (m_supportedSkus.contains(deviceId + ssid))
    {
        log_debug("Device {} ({}{}) is supported", gpuId, deviceId, ssid);
    }
    else if (m_supportedSkus.contains(deviceId))
    {
        log_debug("Device {} ({}) is supported", gpuId, deviceId);
    }
    else
    {
        log_error("Device {} ({}{}) is not supported", gpuId, deviceId, ssid);
        return false;
    }

    return true;
}

void DcgmMnDiagManager::LoadSupportedSkusFromEnv()
{
    const char *envSkus = std::getenv(MnDiagConstants::ENV_SUPPORTED_SKUS.data());
    if (envSkus && *envSkus)
    {
        std::stringstream ss(envSkus);
        std::string sku;
        while (std::getline(ss, sku, ','))
        {
            // Remove whitespace
            sku.erase(std::remove_if(sku.begin(), sku.end(), ::isspace), sku.end());
            if (!sku.empty())
            {
                std::ranges::transform(sku, sku.begin(), ::tolower);
                m_supportedSkus.insert(sku);
            }
        }
        log_info("Overriding supported SKUs from environment: {}", envSkus);
    }
}

std::unordered_set<std::string> DcgmMnDiagManager::GetSupportedSkus()
{
    return m_supportedSkus;
}

bool DcgmMnDiagManager::IsLoopback(std::string_view host)
{
    // Basic string matches for common loopback representations
    if (host.starts_with("localhost") || host.starts_with("127.") || // Covers all 127.0.0.0/8 IPv4 loopback range
        host == "::1" || host.starts_with("0:0:0:0:0:0:0:1"))
    {
        return true;
    }

    // Check hostname and short hostname
    if (host == m_localHostInfo.hostname || host == m_localHostInfo.shortHostname)
    {
        log_debug("Found loopback match for host {} against hostname {} or shortHostname {}",
                  host,
                  m_localHostInfo.hostname,
                  m_localHostInfo.shortHostname);
        return true;
    }

    // Check all local IPv4 addresses using efficient contains() lookup
    if (m_localHostInfo.ipv4Addresses.contains(std::string(host)))
    {
        log_debug("Found loopback match for host {} in local IPv4 addresses", host);
        return true;
    }

    return false;
}

dcgmReturn_t DcgmMnDiagManager::BroadcastRunParametersToRemoteNodes(dcgmRunMnDiag_t const &params)
{
    DcgmMutex resultMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;

    // Handle the head node first (local processing)
    // Create a dummy message for local processing
    dcgm_mndiag_msg_run_params_t dummyMsgOnHeadNode {};
    dummyMsgOnHeadNode.runParams.headNodeId = m_currentNodeId;
    memcpy(&dummyMsgOnHeadNode.runParams.runMnDiag, &params, sizeof(dcgmRunMnDiag_v1));
    SafeCopyTo(dummyMsgOnHeadNode.runParams.mnubergemmPath, GetMnubergemmPathHeadNode().c_str());

    HandleBroadcastRunParameters((dcgm_module_command_header_t *)&dummyMsgOnHeadNode);

    {
        std::vector<std::jthread> threads;

        // Launch threads for remote nodes
        for (auto const &[hostname, connInfo] : m_connections)
        {
            if (connInfo.isLoopback)
            {
                continue; // Already handled above
            }

            // Launch a thread for each remote node
            threads.emplace_back([this, hostname, connInfo, &params, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version                          = dcgmMultinodeRequest_version1;
                request.testType                         = MnDiagTestType::mnubergemm;
                request.requestType                      = MnDiagRequestType::BroadcastRunParameters;
                request.requestData.runParams.headNodeId = m_currentNodeId;
                memcpy(&request.requestData.runParams.runMnDiag, &params, sizeof(dcgmRunMnDiag_v1));
                SafeCopyTo(request.requestData.runParams.mnubergemmPath, GetMnubergemmPathHeadNode().c_str());

                dcgmReturn_t threadResult = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);

                {
                    DcgmLockGuard lg(&resultMutex);
                    results.emplace_back(hostname, threadResult);
                    if (threadResult != DCGM_ST_OK)
                    {
                        anyFailure = true;
                    }
                }

                // Log errors for failed messages
                if (threadResult != DCGM_ST_OK)
                {
                    log_error("Failed to broadcast parameters to remote node {}. Return: ({}): {}",
                              hostname,
                              std::to_underlying(threadResult),
                              errorString(threadResult));
                }
                else
                {
                    log_debug("Successfully broadcast parameters to remote node {}", hostname);
                }
            });
        }

        // threads will automatically join when it goes out of scope at the end of this block
    }

    // Check if any failures occurred
    if (anyFailure)
    {
        for (auto const &[hostname, result] : results)
        {
            if (result != DCGM_ST_OK)
            {
                return result;
            }
        }
        return DCGM_ST_GENERIC_ERROR; // Default error if specific error code not found
    }

    log_info("Successfully broadcast parameters to all remote nodes");
    return DCGM_ST_OK;
}

/****************************************************************************/
void DcgmMnDiagManager::ReapOrphanedSshProcesses()
{
    log_debug("Checking for orphaned SSH processes...");

    // First, identify potential SSH zombie processes by examining /proc
    std::vector<pid_t> sshZombies = FindSshZombieProcesses();

    if (sshZombies.empty())
    {
        log_debug("No SSH zombie processes found");
        return;
    }

    log_debug("Found {} potential SSH zombie processes", sshZombies.size());

    int status;
    int reaped = 0;

    // Try to reap each identified SSH zombie process
    for (pid_t pid : sshZombies)
    {
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid)
        {
            reaped++;
            if (WIFEXITED(status))
            {
                log_debug("Reaped SSH zombie process PID {} (exit code: {})", pid, WEXITSTATUS(status));
            }
            else if (WIFSIGNALED(status))
            {
                log_debug("Reaped SSH zombie process PID {} (killed by signal: {})", pid, WTERMSIG(status));
            }
            else
            {
                log_debug("Reaped SSH zombie process PID {} (unknown exit reason)", pid);
            }
        }
        else if (result == 0)
        {
            log_debug("SSH process PID {} is still running (not a zombie)", pid);
        }
        else if (result == -1)
        {
            if (errno == ECHILD)
            {
                log_debug("SSH process PID {} already reaped by another process", pid);
            }
            else
            {
                log_warning("Error trying to reap SSH process PID {}: {} ({})", pid, strerror(errno), errno);
            }
        }
    }

    if (reaped > 0)
    {
        log_info("Successfully reaped {} orphaned SSH processes", reaped);
    }
    else
    {
        log_debug("No SSH zombie processes were reaped");
    }
}

/****************************************************************************/
std::vector<pid_t> DcgmMnDiagManager::FindSshZombieProcesses()
{
    std::vector<pid_t> sshZombies;

    try
    {
        // Look for zombie processes with "ssh" in their command line
        // and whose parent is our process (nv-hostengine)
        pid_t ourPid = getpid();

        for (const auto &entry : std::filesystem::directory_iterator("/proc"))
        {
            if (!entry.is_directory())
                continue;

            std::string pidStr = entry.path().filename().string();
            if (!std::all_of(pidStr.begin(), pidStr.end(), ::isdigit))
                continue;

            pid_t pid = std::stoi(pidStr);
            if (pid <= 1)
                continue; // Skip init and kernel threads

            // Check if this process is a zombie
            std::string statPath = entry.path() / "stat";
            std::ifstream statFile(statPath);
            if (!statFile.is_open())
                continue;

            std::string line;
            if (!std::getline(statFile, line))
                continue;

            // Parse /proc/PID/stat format: pid (comm) state ppid ...
            std::istringstream iss(line);
            std::string pidField, comm, state, ppidField;

            if (!(iss >> pidField >> comm >> state >> ppidField))
                continue;

            // Check if it's a zombie (state 'Z') and parent is us
            if (state != "Z")
                continue;

            pid_t ppid = std::stoi(ppidField);
            if (ppid != ourPid)
                continue;

            // Check if the command contains "ssh"
            if (comm.contains("ssh"))
            {
                log_debug("Found SSH zombie process: PID {} ({})", pid, comm);
                sshZombies.push_back(pid);
            }
        }
    }
    catch (const std::exception &e)
    {
        log_warning("Error scanning for SSH zombie processes: {}", e.what());
    }

    return sshZombies;
}

void DcgmMnDiagManager::InitializeSSHTunnelManagers()
{
    m_tcpSSHTunnelManager = std::make_unique<TcpSSHTunnelManagerAdapter>();
    m_udsSSHTunnelManager = std::make_unique<UdsSSHTunnelManagerAdapter>();
    DcgmNs::Common::RemoteConn::detail::ChildProcessFuncs const childProcessFuncs = {
        .Spawn
        = [this](auto &&...args) { return m_coreProxy->ChildProcessSpawn(std::forward<decltype(args)>(args)...); },
        .GetStatus
        = [this](auto &&...args) { return m_coreProxy->ChildProcessGetStatus(std::forward<decltype(args)>(args)...); },
        .GetStdErrHandle =
            [this](auto &&...args) {
                return m_coreProxy->ChildProcessGetStdErrHandle(std::forward<decltype(args)>(args)...);
            },
        .GetStdOutHandle =
            [this](auto &&...args) {
                return m_coreProxy->ChildProcessGetStdOutHandle(std::forward<decltype(args)>(args)...);
            },
        .GetDataChannelHandle =
            [this](auto &&...args) {
                return m_coreProxy->ChildProcessGetDataChannelHandle(std::forward<decltype(args)>(args)...);
            },
        .Stop                                                                     = [this](auto &&...args) { return m_coreProxy->ChildProcessStop(std::forward<decltype(args)>(args)...); },
        .Wait                                                                     = [this](auto &&...args) { return m_coreProxy->ChildProcessWait(std::forward<decltype(args)>(args)...); },
        .Destroy
        = [this](auto &&...args) { return m_coreProxy->ChildProcessDestroy(std::forward<decltype(args)>(args)...); },
    };
    m_tcpSSHTunnelManager->SetChildProcessFuncs(&childProcessFuncs);
    m_udsSSHTunnelManager->SetChildProcessFuncs(&childProcessFuncs);
}


void DcgmMnDiagManager::PopulateMpiFailureResponseStruct(MnDiagMpiRunnerBase *mpiRunner,
                                                         dcgmMnDiagResponse_v1 &response,
                                                         std::optional<pid_t> mpiPID,
                                                         DcgmNs::Utils::LogPaths const &logPaths)
{
    mpiRunner->PopulateResponse(&response, m_nodeInfo);

    if (auto exitCode = mpiRunner->GetMpiProcessExitCode(); exitCode.has_value() && *exitCode != 0)
    {
        log_error("MPI process with PID {} exited with non-zero exit code: {}", *mpiPID, *exitCode);
        // Set the response status to FAILED
        response.tests[0].result = DCGM_DIAG_RESULT_FAIL;
        SafeCopyTo(response.tests[0].auxData.data,
                   fmt::format("MPI exited with code: {}; Check logs: {}, {} for more details",
                               *exitCode,
                               logPaths.stdoutFileName.string(),
                               logPaths.stderrFileName.string())
                       .c_str());
        response.tests[0].auxData.version = dcgmMnDiagTestAuxData_version1;
    }
}

std::string DcgmMnDiagManager::GetMnubergemmPathHeadNode()
{
    char const *customBinPath = std::getenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());
    std::string defaultBinPath(MnDiagConstants::DEFAULT_MNUBERGEMM_PATH);

    std::string binPath;
    if (customBinPath && *customBinPath != '\0')
    {
        binPath = customBinPath;
        try
        {
            if (!std::filesystem::exists(binPath) || !std::filesystem::is_regular_file(binPath))
            {
                log_error(
                    "Custom binary path '{}' is invalid (not a readable, executable, regular file). Falling back to default '{}'.",
                    binPath,
                    defaultBinPath);
                binPath = defaultBinPath;
            }
            else if (access(binPath.c_str(), R_OK | X_OK) != 0)
            {
                log_error("Custom binary path '{}' is not accessible (errno {}: {}). Falling back to default '{}'.",
                          binPath,
                          errno,
                          strerror(errno),
                          defaultBinPath);
                binPath = defaultBinPath;
            }
        }
        catch (const std::exception &e)
        {
            log_error("Exception while validating custom binary path '{}': {}. Falling back to default '{}'.",
                      binPath,
                      e.what(),
                      defaultBinPath);
            binPath = defaultBinPath;
        }
    }
    else
    {
        binPath = defaultBinPath;
    }

    log_debug("Mnubergemm path for head node: {}", binPath);
    return binPath;
}
