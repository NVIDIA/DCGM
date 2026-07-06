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
#include <fstream>
#include <optional>
#include <pwd.h>
#include <ranges>
#include <sstream>
#include <thread>
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
        [this](MnDiagStatus status) { this->SetStatus(status); },
        // Get MPI process info callback
        []() { return DcgmNs::Common::ProcessUtils::GetMpiProcessInfo(); });

    // Start the state machine
    m_stateMachine->Start();

    // Supported SKUs - hardcoded - always lowercase
    m_supportedSkus = {
        "2941", // GB200 NVL
        "31c2", // GB300 NVL-Bianca
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
    log_debug("Disconnecting from all remote nodes");

    // First revoke all authorizations
    dcgmReturn_t revokeResult = RevokeRemoteAuthorizations();
    if (revokeResult != DCGM_ST_OK)
    {
        log_error("Failed to revoke remote authorizations: {}", errorString(revokeResult));
        // Continue with cleanup even if revocation fails
    }

    // Then clean up all connections
    dcgmReturn_t cleanupResult = CleanupConnections();
    if (cleanupResult != DCGM_ST_OK)
    {
        log_error("Failed to cleanup connections: {}", errorString(cleanupResult));
        return cleanupResult;
    }

    // Return revocation error if cleanup succeeded but revocation failed
    if (revokeResult != DCGM_ST_OK)
    {
        return revokeResult;
    }

    log_info("Successfully disconnected from all remote nodes");
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

std::expected<dcgmMultinodeTestType_t, dcgmReturn_t> GetTestTypeFromName(std::string_view testName)
{
    if (testName == "mnubergemm")
    {
        return dcgmMultinodeTestType_t::mnubergemm;
    }
    return std::unexpected(DCGM_ST_BADPARAM);
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

    auto testTypeResult = GetTestTypeFromName(params.testName);
    if (!testTypeResult.has_value())
    {
        log_error("Unknown test name '{}': {}", params.testName, errorString(testTypeResult.error()));
        return testTypeResult.error();
    }
    dcgmMultinodeTestType_t const testType = testTypeResult.value();

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

    // Then authorize all connections
    result = AuthorizeRemoteConnections(testType);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to authorize remote connections");
        DisconnectRemoteNodes();
        return result;
    }

    // Get dcgm and driver versions from all remote nodes
    result = GetNodeInfo(testType);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to get node info from remote nodes");
        DisconnectRemoteNodes();
        return result;
    }

    // If all resource checks passed, reserve resources
    result = ReserveRemoteResources(testType);
    if (result != DCGM_ST_OK)
    {
        log_error("Resource reservation failed");
        DisconnectRemoteNodes();
        return result;
    }

    // Create a runner for the requested test type
    auto mpiRunner = m_mpiRunnerFactory->CreateMpiRunner(*m_coreProxy, testType, effectiveUid);
    if (!mpiRunner)
    {
        log_error("Factory could not create a runner for test type {}", static_cast<int>(testType));
        StopHeadNode();
        return DCGM_ST_BADPARAM;
    }

    std::string testBinaryPath;
    result = mpiRunner->GetTestBinaryPath(testBinaryPath);
    if (testBinaryPath.empty() || result != DCGM_ST_OK)
    {
        log_error("Failed to get test binary path");
        StopHeadNode();
        return DCGM_ST_INIT_ERROR;
    }

    m_currentTestInfo = {
        .testType       = testType,
        .testName       = params.testName,
        .testBinaryPath = testBinaryPath,
        .testPrefix     = std::string(mpiRunner->GetTestPrefix()),
    };

    // Generate the MPI command directly from the dcgmRunMnDiag_t struct
    mpiRunner->ConstructMpiCommand(&params);
    log_debug("Generated MPI command: {}", mpiRunner->GetLastCommand());

    // Get the test run time from the mpi runner
    auto testRunTimeResult = mpiRunner->GetTestRunTime(params);
    if (!testRunTimeResult.has_value())
    {
        log_error("Invalid time_to_run in run params: {}", errorString(testRunTimeResult.error()));
        StopHeadNode();
        return testRunTimeResult.error();
    }
    unsigned int const timeToRunSeconds
        = static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::seconds>(*testRunTimeResult).count());

    result = BroadcastRunParametersToRemoteNodes(params, timeToRunSeconds);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to broadcast parameters to remote nodes");
        DisconnectRemoteNodes();
        return result;
    }

    m_stateMachine->SetProcessExecutionTimeout(std::chrono::seconds(timeToRunSeconds));

    // Redirect MPI output to files
    DcgmNs::Utils::LogPaths logPaths = DcgmNs::Utils::GetLogFilePath(mpiRunner->GetLogFilePrefix());
    mpiRunner->SetLogFileNames(std::make_pair(logPaths.stdoutFileName.string(), logPaths.stderrFileName.string()));

    // Set the output callback without the response struct - we'll pass it directly in ProcessAndGetMpiOutput
    mpiRunner->SetOutputCallback([&mpiRunner](int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) {
        return mpiRunner->MnDiagOutputCallback(fd, responseStruct, nodeInfo);
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

    // Wait for all nodes (head + workers) to detect MPI processes running on GPUs
    // Each node uses nvidia-smi detection with retry logic to handle the startup delay
    // All nodes are checked in parallel to minimize total detection time
    DcgmMutex resultMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;

    // Head node detection result storage
    dcgm_mndiag_msg_resource_t headNodeMsg {};

    {
        std::vector<std::jthread> threads;

        // Launch head node detection in parallel with worker nodes
        threads.emplace_back([this, &headNodeMsg, &resultMutex, &results, &anyFailure]() {
            HandleDetectProcess((dcgm_module_command_header_t *)&headNodeMsg);

            bool const success      = (headNodeMsg.resource.response == MnDiagStatus::RUNNING);
            dcgmReturn_t const code = success ? DCGM_ST_OK : DCGM_ST_CHILD_SPAWN_FAILED;

            {
                DcgmLockGuard lg(&resultMutex);
                results.emplace_back(m_localHostInfo.hostname, code);
                anyFailure = anyFailure || !success;
            }

            if (!success)
            {
                log_error("No MPI process running on head node");
            }
        });

        // Launch threads for remote nodes
        for (auto const &[hostname, connInfo] : m_connections)
        {
            if (connInfo.isLoopback)
            {
                continue; // Head node is handled by the thread above
            }

            // Launch a thread for each remote node
            threads.emplace_back([this, hostname, connInfo, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version                         = dcgmMultinodeRequest_version;
                request.testType                        = m_currentTestInfo.testType;
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

    // Check if any failures occurred (head or worker nodes)
    if (anyFailure)
    {
        PopulateMpiFailureResponseStruct(mpiRunner.get(), response, *mpiPID, logPaths);
        StopHeadNode();

        // Return first error found in results (head node will be first if it failed)
        for (auto const &[hostname, result] : results)
        {
            if (result != DCGM_ST_OK)
            {
                return result;
            }
        }
        return DCGM_ST_CHILD_SPAWN_FAILED; // Default error for detection failure
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

    // Release resources first - this calls NotifyDiagnosticFinished()
    dcgmReturn_t releaseResult = ReleaseRemoteResources(m_currentTestInfo.testType);
    if (releaseResult != DCGM_ST_OK)
    {
        log_error("Failed to release remote resources: {}", errorString(releaseResult));
        finalResult = releaseResult; // Remember first error
    }

    // IMPORTANT: Wait for state machine to fully transition to WAITING/READY state
    // This prevents race conditions with the next run
    log_debug("Waiting for state machine to stabilize to READY state");

    // Use condition variable to wait for READY status
    auto startTime = std::chrono::steady_clock::now();

    bool stateStabilized = false;
    {
        // Acquire lock for CondWait
        DcgmLockGuard lg(&m_resourceMutex);

        dcgmMutexReturn_t waitResult = m_resourceMutex.CondWait(
            m_statusCV, MnDiagConstants::MAX_WAIT_MS.count(), [this]() { return m_status == MnDiagStatus::READY; });

        auto elapsed   = std::chrono::steady_clock::now() - startTime;
        auto elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

        if (waitResult == DCGM_MUTEX_ST_OK)
        {
            stateStabilized = true;
            log_debug("State machine stabilized to READY state after {}ms", elapsedMS);
        }
        else if (waitResult == DCGM_MUTEX_ST_TIMEOUT)
        {
            log_error("State machine did not stabilize to WAITING/READY within {}ms timeout",
                      MnDiagConstants::MAX_WAIT_MS.count());
            if (finalResult == DCGM_ST_OK)
            {
                finalResult = DCGM_ST_TIMEOUT;
            }
        }
        else
        {
            log_error("Error waiting for state machine to stabilize: {}", waitResult);
            if (finalResult == DCGM_ST_OK)
            {
                finalResult = DCGM_ST_GENERIC_ERROR;
            }
        }
    } // Release lock after CondWait

    // Now proceed with connection cleanup
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
        {
            finalResult = resetResult;
        }
    }
    else
    {
        log_debug("Successfully reset ChildProcessManager");
    }

    // Additional safety: Small delay to ensure kernel-level process cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    log_debug("StopHeadNode completed, state machine stabilized: {}", stateStabilized);

    return finalResult; // Let caller decide how to handle partial cleanup failures
}

dcgmReturn_t DcgmMnDiagManager::ConnectRemoteNodes(std::vector<std::string> const &hostList, uid_t effectiveUid)
{
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
    // Parse all connection parameters once and validate for duplicates
    std::vector<ConnectionParams> allConnectionParams;
    std::unordered_set<std::string> uniqueHosts;

    for (auto const &host : hostList)
    {
        try
        {
            ConnectionParams params = ParseConnectionParams(host, effectiveUid);

            if (!uniqueHosts.insert(params.hostname).second)
            {
                log_error("Duplicate host found in input list: {}", host);
                return DCGM_ST_BADPARAM;
            }

            allConnectionParams.push_back(std::move(params));
        }
        catch (std::exception const &e)
        {
            log_error("Failed to parse connection parameters for host {}: {}", host, e.what());
            return DCGM_ST_BADPARAM;
        }
    }
    // Handle connections using pre-parsed parameters
    for (auto const &params : allConnectionParams)
    {
        if (IsLoopback(params.hostname))
        {
            m_connections[params.hostname] = ConnectionInfo {
                .handle = 0, .remotePort = 0, .isLoopback = true, .uid = effectiveUid, .remoteSocketPath = ""
            };
            continue; // Loopback connection stored
        }

        // Connect to remote node directly (synchronous)
        dcgmReturn_t result = ConnectSingleNode(params);
        if (result != DCGM_ST_OK)
        {
            log_error("Failed to connect to remote node {}: {}", params.hostname, errorString(result));
            CleanupConnections();
            return result;
        }
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::ReserveRemoteResources(dcgmMultinodeTestType_t testType)
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
            threads.emplace_back([this, hostname, connInfo, testType, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version                         = dcgmMultinodeRequest_version;
                request.testType                        = testType;
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
        ReleaseRemoteResources(testType);
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

dcgmReturn_t DcgmMnDiagManager::ReleaseRemoteResources(dcgmMultinodeTestType_t testType)
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
            threads.emplace_back([this, hostname, connInfo, testType, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version                         = dcgmMultinodeRequest_version;
                request.testType                        = testType;
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

dcgmReturn_t DcgmMnDiagManager::GetNodeInfo(dcgmMultinodeTestType_t testType)
{
    DcgmMutex resultMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    results.reserve(m_connections.size());
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
                DcgmLockGuard lg(&resultMutex);
                m_nodeInfo[hostname].dcgmVersion   = localDcgmVersion;
                m_nodeInfo[hostname].driverVersion = localDriverVersion;
                continue; // Already handled above
            }

            // Launch a thread for each remote node
            threads.emplace_back([this, hostname, connInfo, testType, &resultMutex, &results, &anyFailure]() {
                dcgmMultinodeRequest_t request {};
                request.version     = dcgmMultinodeRequest_version;
                request.testType    = testType;
                request.requestType = MnDiagRequestType::GetNodeInfo;

                dcgmReturn_t threadResult = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);
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
                }
                auto resultEntry = std::make_pair(hostname, threadResult);
                {
                    DcgmLockGuard lg(&resultMutex);
                    results.emplace_back(std::move(resultEntry));
                    if (threadResult != DCGM_ST_OK)
                    {
                        anyFailure = true;
                    }
                    else
                    {
                        m_nodeInfo[hostname].dcgmVersion   = request.requestData.nodeInfo.dcgmVersion;
                        m_nodeInfo[hostname].driverVersion = request.requestData.nodeInfo.driverVersion;
                    }
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

    // Retry logic: MPI processes may take time to start and attach to GPUs
    // We retry detection at intervals until timeout
    auto const timeout       = m_processDetectionTimeout;
    auto const retryInterval = m_processDetectionRetryInterval;
    auto const startTime     = std::chrono::steady_clock::now();

    bool detected = false;
    int attempt   = 0;

    while (std::chrono::steady_clock::now() - startTime < timeout)
    {
        attempt++;
        // Try to get the detected MPI process PID from the state machine
        // This uses nvidia-smi to detect processes running on GPUs
        if (auto ret = m_stateMachine->TryGetDetectedMpiPid(); ret == DCGM_ST_OK)
        {
            log_debug("MPI process detected on attempt {}", attempt);
            detected = true;
            break;
        }

        // Wait before next retry
        std::this_thread::sleep_for(retryInterval);
    }

    if (!detected)
    {
        log_error("No MPI process related to this run detected after {} attempts ({} seconds)",
                  attempt,
                  std::chrono::duration_cast<std::chrono::seconds>(timeout).count());
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

    // Version incompatibility is not supported between the head node and the remote nodes
    if (paramMsg->header.version != dcgm_mndiag_msg_run_params_version2)
    {
        log_error("Unsupported version of broadcast run parameters message: {}", paramMsg->header.version);
        return DCGM_ST_VER_MISMATCH;
    }

    log_debug("Handling broadcast parameters request from head node {}, current node ID - {}",
              paramMsg->runParams.headNodeId,
              static_cast<unsigned int>(m_currentNodeId));

    // Resolve the run time to use for the process execution timeout.
    // timeToRunSeconds > 0: pre-validated by the head node (normal production path).
    // timeToRunSeconds == 0: fall back to testParms parsing (unit tests / legacy callers).
    std::optional<unsigned int> timeToRun;
    if (paramMsg->runParams.timeToRunSeconds > 0)
    {
        timeToRun = paramMsg->runParams.timeToRunSeconds;
        log_debug("Using pre-validated time_to_run from message: {} seconds", *timeToRun);
    }
    else
    {
        std::string const timeToRunKey = std::string(paramMsg->runParams.testPrefix) + "time_to_run";
        auto result                    = ParseTimeToRunSeconds(paramMsg->runParams.runMnDiag, timeToRunKey);
        if (!result.has_value())
        {
            log_error("Failed to parse time_to_run from testParms: {}", errorString(result.error()));
            return result.error();
        }
        if (*result > 0)
        {
            timeToRun = static_cast<unsigned int>(*result);
            log_debug("Extracted time_to_run from testParms: {} seconds", *timeToRun);
        }
    }

    // Always set the process execution timeout, defaulting to 3600 seconds if not specified.
    // This ensures the timeout is reset between runs.
    if (timeToRun.has_value())
    {
        m_stateMachine->SetProcessExecutionTimeout(std::chrono::seconds(*timeToRun));
    }
    else
    {
        auto timeout = MnDiagConstants::DEFAULT_TIME_TO_RUN_SECONDS;
        log_debug("No time_to_run parameter specified, using default: {} seconds", timeout.count());
        m_stateMachine->SetProcessExecutionTimeout(timeout);
    }

    // Set the expected test binary path in the state machine
    if (paramMsg->runParams.testBinaryPath[0] == '\0')
    {
        log_warning("Received empty test binary path on node {}, not setting path.",
                    static_cast<unsigned int>(m_currentNodeId));
        m_stateMachine->SetExpectedBinaryPath("");
    }
    else
    {
        log_debug("Setting expected test binary path to: {} on node: {}",
                  paramMsg->runParams.testBinaryPath,
                  static_cast<unsigned int>(m_currentNodeId));
        m_stateMachine->SetExpectedBinaryPath(paramMsg->runParams.testBinaryPath);
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

    // Notify any threads waiting for status changes
    m_statusCV.notify_all();
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
        hostname = std::move(hostPart);
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

dcgmReturn_t DcgmMnDiagManager::BroadcastRunParametersToRemoteNodes(dcgmRunMnDiag_t const &params,
                                                                    unsigned int timeToRunSeconds)
{
    DcgmMutex resultMutex(0);
    std::vector<std::pair<std::string, dcgmReturn_t>> results;
    bool anyFailure = false;

    // Handle the head node first (local processing)
    // Create a dummy message for local processing
    dcgm_mndiag_msg_run_params_t dummyMsgOnHeadNode {};
    dummyMsgOnHeadNode.header.version             = dcgm_mndiag_msg_run_params_version2;
    dummyMsgOnHeadNode.header.length              = sizeof(dummyMsgOnHeadNode);
    dummyMsgOnHeadNode.runParams.headNodeId       = m_currentNodeId;
    dummyMsgOnHeadNode.runParams.timeToRunSeconds = timeToRunSeconds;
    memcpy(&dummyMsgOnHeadNode.runParams.runMnDiag, &params, sizeof(dcgmRunMnDiag_v1));

    SafeCopyTo(dummyMsgOnHeadNode.runParams.testBinaryPath, m_currentTestInfo.testBinaryPath.c_str());
    SafeCopyTo(dummyMsgOnHeadNode.runParams.testPrefix, m_currentTestInfo.testPrefix.c_str());

    auto headResult = HandleBroadcastRunParameters((dcgm_module_command_header_t *)&dummyMsgOnHeadNode);
    if (headResult != DCGM_ST_OK)
    {
        log_error("Failed to apply broadcast run parameters on head node: {}", errorString(headResult));
        return headResult;
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
            threads.emplace_back(
                [this, hostname, connInfo, &params, timeToRunSeconds, &resultMutex, &results, &anyFailure]() {
                    dcgmMultinodeRequest_t request {};
                    request.version                                = dcgmMultinodeRequest_version;
                    request.testType                               = m_currentTestInfo.testType;
                    request.requestType                            = MnDiagRequestType::BroadcastRunParameters;
                    request.requestData.runParams.headNodeId       = m_currentNodeId;
                    request.requestData.runParams.timeToRunSeconds = timeToRunSeconds;
                    memcpy(&request.requestData.runParams.runMnDiag, &params, sizeof(dcgmRunMnDiag_v1));
                    SafeCopyTo(request.requestData.runParams.testBinaryPath, m_currentTestInfo.testBinaryPath.c_str());
                    SafeCopyTo(request.requestData.runParams.testPrefix, m_currentTestInfo.testPrefix.c_str());

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
std::vector<pid_t> DcgmMnDiagManager::FindSshZombieProcesses(std::filesystem::path const &procPath)
{
    std::vector<pid_t> sshZombies;

    try
    {
        // Look for zombie processes with "ssh" in their command line
        // and whose parent is our process (nv-hostengine)
        pid_t ourPid = getpid();

        for (auto const &entry : std::filesystem::directory_iterator(procPath))
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

int DcgmMnDiagManager::GetCudaVersion()
{
    int cudaVersion = 0;
    m_coreProxy->GetCudaVersion(cudaVersion);
    return cudaVersion;
}

dcgmReturn_t DcgmMnDiagManager::AuthorizeRemoteConnections(dcgmMultinodeTestType_t testType)
{
    log_debug("Starting authorization of remote connections");

    // First authorize the head node
    dcgmReturn_t dcgmResult = HandleAuthorizeConnection(m_currentNodeId);
    if (dcgmResult != DCGM_ST_OK)
    {
        log_error("Failed to authorize head node connection");
        return dcgmResult;
    }

    // Now authorize all remote connections (synchronous)
    for (auto const &[hostname, connInfo] : m_connections)
    {
        if (connInfo.isLoopback)
        {
            continue; // Already handled above
        }

        // Send authorization request to remote node
        dcgmMultinodeRequest_t request {};
        request.version                              = dcgmMultinodeRequest_version;
        request.testType                             = testType;
        request.requestType                          = MnDiagRequestType::AuthorizeConnection;
        request.requestData.authorization.headNodeId = m_currentNodeId;

        dcgmReturn_t result = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);
        if (result != DCGM_ST_OK)
        {
            log_error("Failed to authorize connection with {}: {}", hostname, errorString(result));
            return result;
        }

        log_debug("Successfully authorized connection with {}", hostname);
    }

    log_info("Successfully authorized all remote connections");
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::RevokeRemoteAuthorizations()
{
    log_debug("Starting revocation of remote authorizations");

    // First revoke authorization on the head node
    dcgmReturn_t dcgmResult = HandleRevokeAuthorization(m_currentNodeId);
    if (dcgmResult != DCGM_ST_OK)
    {
        log_error("Failed to revoke authorization for head node");
        return dcgmResult;
    }

    // Now revoke all remote authorizations (synchronous)
    for (auto const &[hostname, connInfo] : m_connections)
    {
        if (connInfo.isLoopback)
        {
            continue; // Already handled above
        }

        // Send revocation request to remote node
        dcgmMultinodeRequest_t request {};
        request.version                              = dcgmMultinodeRequest_version;
        request.testType                             = m_currentTestInfo.testType;
        request.requestType                          = MnDiagRequestType::RevokeAuthorization;
        request.requestData.authorization.headNodeId = m_currentNodeId;

        dcgmReturn_t result = m_dcgmApi->MultinodeRequest(connInfo.handle, &request);
        if (result != DCGM_ST_OK)
        {
            log_error("Failed to revoke authorization for connection: {}. Return: ({}): {}",
                      hostname,
                      std::to_underlying(result),
                      errorString(result));
            return result;
        }

        log_debug("Successfully revoked authorization for connection with {}", hostname);
    }

    log_info("Successfully revoked all remote authorizations");
    return DCGM_ST_OK;
}

DcgmNs::Common::RemoteConn::detail::TunnelState DcgmMnDiagManager::StartTunnelSession(ConnectionParams &params)
{
    DcgmNs::Common::RemoteConn::detail::TunnelState tunnelState;

    if (params.isUnixSocket)
    {
        std::string const &socketPath = std::get<std::string>(params.remoteAddress);
        std::string localSocketPath;

        log_debug("Starting SSH tunnel with Unix socket path {} for host {}", socketPath, params.hostname);
        tunnelState
            = m_udsSSHTunnelManager->StartSession(params.hostname, socketPath, localSocketPath, params.effectiveUid);

        if (tunnelState == DcgmNs::Common::RemoteConn::detail::TunnelState::Active)
        {
            params.localAddress = localSocketPath;
        }
        else
        {
            log_error(
                "Failed to create SSH tunnel to {} with socket path {} (TunnelState: {})",
                params.hostname,
                socketPath,
                static_cast<std::underlying_type_t<DcgmNs::Common::RemoteConn::detail::TunnelState>>(tunnelState));
        }
    }
    else
    {
        uint16_t const &remotePort = std::get<uint16_t>(params.remoteAddress);
        uint16_t localPort;

        log_debug("Starting SSH tunnel with port {} for host {}", remotePort, params.hostname);
        tunnelState = m_tcpSSHTunnelManager->StartSession(params.hostname, remotePort, localPort, params.effectiveUid);

        if (tunnelState == DcgmNs::Common::RemoteConn::detail::TunnelState::Active)
        {
            params.localAddress = localPort;
        }
        else
        {
            log_error(
                "Failed to create SSH tunnel to {}:{} (TunnelState: {})",
                params.hostname,
                remotePort,
                static_cast<std::underlying_type_t<DcgmNs::Common::RemoteConn::detail::TunnelState>>(tunnelState));
        }
    }

    return tunnelState;
}

void DcgmMnDiagManager::EndTunnelSession(ConnectionParams const &params)
{
    if (params.isUnixSocket)
    {
        std::string const &socketPath = std::get<std::string>(params.remoteAddress);
        m_udsSSHTunnelManager->EndSession(params.hostname, socketPath, params.effectiveUid);
    }
    else
    {
        uint16_t const &remotePort = std::get<uint16_t>(params.remoteAddress);
        m_tcpSSHTunnelManager->EndSession(params.hostname, remotePort, params.effectiveUid);
    }
}

ConnectionParams DcgmMnDiagManager::ParseConnectionParams(std::string const &host, uid_t effectiveUid)
{
    ConnectionParams params {};
    params.effectiveUid = effectiveUid;

    std::string socketPath;
    uint16_t remotePort;

    dcgmReturn_t res = ExtractHostnameAndPort(host, params.hostname, remotePort, socketPath);
    if (res != DCGM_ST_OK)
    {
        throw std::runtime_error(fmt::format("Failed to extract hostname and port from host: {}", host));
    }

    if (!socketPath.empty())
    {
        params.isUnixSocket  = true;
        params.remoteAddress = socketPath;
        // Local address will be populated by StartTunnelSession
        params.localAddress = std::string {};
    }
    else
    {
        params.isUnixSocket  = false;
        params.remoteAddress = remotePort;
        // Local address will be populated by StartTunnelSession
        params.localAddress = uint16_t { 0 };
    }

    return params;
}

dcgmReturn_t DcgmMnDiagManager::CreateDcgmConnection(ConnectionParams const &params, dcgmHandle_t &handle)
{
    std::string connectionString;
    dcgmConnectV3Params_t connectParams;
    memset(&connectParams, 0, sizeof(connectParams));
    connectParams.version = dcgmConnectV3Params_version;

    if (params.isUnixSocket)
    {
        connectionString = fmt::format("unix://{}", std::get<std::string>(params.localAddress));
    }
    else
    {
        connectionString = fmt::format("tcp://127.0.0.1:{}", std::get<uint16_t>(params.localAddress));
    }

    dcgmReturn_t dcgmResult = m_dcgmApi->Connect_v3(connectionString.c_str(), &connectParams, &handle);

    if (dcgmResult != DCGM_ST_OK)
    {
        log_error("Failed to connect to remote DCGM {} from {}", connectionString, params.hostname);
    }

    return dcgmResult;
}

ConnectionInfo DcgmMnDiagManager::CreateConnectionInfo(ConnectionParams const &params, dcgmHandle_t handle)
{
    if (params.isUnixSocket)
    {
        std::string const &socketPath = std::get<std::string>(params.remoteAddress);
        return ConnectionInfo { .handle           = handle,
                                .remotePort       = 0,
                                .isLoopback       = false,
                                .uid              = params.effectiveUid,
                                .remoteSocketPath = socketPath };
    }
    else
    {
        uint16_t const &remotePort = std::get<uint16_t>(params.remoteAddress);
        return ConnectionInfo { .handle           = handle,
                                .remotePort       = remotePort,
                                .isLoopback       = false,
                                .uid              = params.effectiveUid,
                                .remoteSocketPath = "" };
    }
}

dcgmReturn_t DcgmMnDiagManager::ConnectSingleNode(ConnectionParams params)
{
    // 1. Start tunnel session
    auto tunnelState = StartTunnelSession(params);
    if (tunnelState != DcgmNs::Common::RemoteConn::detail::TunnelState::Active)
    {
        return DCGM_ST_REMOTE_SSH_CONNECTION_FAILED;
    }

    // 2. Create DCGM connection
    dcgmHandle_t handle;
    auto dcgmResult = CreateDcgmConnection(params, handle);
    if (dcgmResult != DCGM_ST_OK)
    {
        EndTunnelSession(params);
        return dcgmResult;
    }

    // 3. Store connection info (without authorization)
    ConnectionInfo connInfo        = CreateConnectionInfo(params, handle);
    m_connections[params.hostname] = connInfo;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMnDiagManager::CleanupConnections()
{
    log_debug("Cleaning up connections");

    // Clean up remote connections
    for (auto const &[hostname, connInfo] : m_connections)
    {
        if (connInfo.isLoopback)
        {
            continue; // Skip loopback connections
        }

        // Disconnect DCGM connection
        if (connInfo.handle != 0)
        {
            m_dcgmApi->Disconnect(connInfo.handle);
        }

        // Create ConnectionParams for tunnel cleanup
        ConnectionParams params {};
        params.hostname     = hostname;
        params.effectiveUid = connInfo.uid;

        if (!connInfo.remoteSocketPath.empty())
        {
            params.isUnixSocket  = true;
            params.remoteAddress = connInfo.remoteSocketPath;
        }
        else
        {
            params.isUnixSocket  = false;
            params.remoteAddress = connInfo.remotePort;
        }

        // End tunnel session
        EndTunnelSession(params);
    }

    // Clear all connections and node info
    m_connections.clear();
    m_nodeInfo.clear();

    log_debug("Connection cleanup completed");
    return DCGM_ST_OK;
}
