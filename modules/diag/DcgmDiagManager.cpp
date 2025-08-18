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

#include "DcgmDiagManager.h"

#include <ChildProcess/ChildProcess.hpp>
#include <ChildProcess/ChildProcessBuilder.hpp>
#include <ChildProcess/IoContext.hpp>
#include <DcgmError.h>
#include <DcgmResourceHandle.h>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <Defer.hpp>
#include <EntityListHelpers.h>
#include <NvvsExitCode.h>
#include <UniquePtrUtil.h>
#include <dcgm_config_structs.h>
#include <dcgm_core_structs.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <iterator>
#include <ranges>
#include <sstream>
#include <sys/epoll.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


#define NVVS_CHANNEL_FD 3

namespace
{
std::optional<std::string> GetServiceAccount(DcgmCoreProxy const &proxy)
{
    std::string serviceAccount;
    if (auto ret = proxy.GetServiceAccount(serviceAccount); ret == DCGM_ST_OK)
    {
        if (serviceAccount.empty())
        {
            return std::nullopt;
        }
        DCGM_LOG_DEBUG << fmt::format("GetServiceAccount result: {}", serviceAccount);
        return serviceAccount;
    }
    else
    {
        DCGM_LOG_DEBUG << fmt::format("GetServiceAccount error: ({}) {}.", ret, errorString(ret));
    }

    return std::nullopt;
}

std::string SanitizedString(std::string const &str)
{
    std::string sanitized;
    sanitized.reserve(str.size());
    std::transform(str.begin(), str.end(), std::back_inserter(sanitized), [](char c) {
        if (c == '\n')
        {
            return '\t';
        }
        return c;
    });
    return sanitized;
}
} // namespace

/*****************************************************************************/
DcgmDiagManager::DcgmDiagManager(dcgmCoreCallbacks_t &dcc)
    : m_nvvsPath(GetNvvsBinPath())
    , m_mutex(0)
    , m_nvvsPID(-1)
    , m_ticket(0)
    , m_coreProxy(dcc)
    , m_amShuttingDown(false)
{}

DcgmDiagManager::~DcgmDiagManager()
{
    DcgmLockGuard lock(&m_mutex);
    m_amShuttingDown = true;

    // Clean up any running child process
    if (m_childProcessHandle != 0)
    {
        log_debug("Cleaning up leftover nvvs process with pid {}", m_nvvsPID);

        if (m_nvvsPID >= 0)
        {
            KillActiveNvvs((unsigned int)-1); // don't stop until it's dead
        }

        m_childProcessHandle = 0;
        m_nvvsPID            = -1;
    }

    // The lock guard above will obey RAII. m_mutex will be destructed with this instance.
    // coverity[missing_unlock: FALSE]
}

dcgmReturn_t DcgmDiagManager::KillActiveNvvs(unsigned int maxRetries)
{
    static const std::chrono::seconds SIGTERM_RETRY_DELAY = std::chrono::seconds(4);
    static const unsigned int MAX_SIGTERM_ATTEMPTS        = 4;

    pid_t childPid                   = m_nvvsPID;
    unsigned int kill_count          = 0;
    bool sigkilled                   = false;
    ChildProcessHandle_t childHandle = m_childProcessHandle;
    bool processExists               = true;
    dcgmReturn_t ret                 = DCGM_ST_OK;

    assert(m_mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);

    // First check if we have a valid child process handle
    if (childHandle == 0 || childPid < 0)
    {
        log_debug("No active NVVS process to kill (handle={}; pid={})", childHandle, childPid);
        return DCGM_ST_OK;
    }

    while (kill_count <= maxRetries && processExists)
    {
        // Try to get process status
        dcgmChildProcessStatus_v1 status = {};
        status.version                   = dcgmChildProcessStatus_version1;

        // First check if the process is still running
        dcgm_mutex_unlock(&m_mutex);
        ret = m_coreProxy.ChildProcessGetStatus(childHandle, status);
        dcgm_mutex_lock(&m_mutex);

        if (ret != DCGM_ST_OK || !status.running)
        {
            log_debug("NVVS process {} is no longer running", childPid);
            processExists = false;
            break;
        }

        // As long as the process exists, keep killing it
        if (kill_count < MAX_SIGTERM_ATTEMPTS)
        {
            dcgm_mutex_unlock(&m_mutex);
            ret = m_coreProxy.ChildProcessStop(childHandle, false); // false = use SIGTERM
            dcgm_mutex_lock(&m_mutex);

            if (ret != DCGM_ST_OK)
            {
                log_error("Failed to send SIGTERM to process {}: {}", childPid, errorString(ret));
            }
        }
        else
        {
            // Escalate to SIGKILL if SIGTERM is not working
            if (sigkilled == false)
            {
                log_error("Unable to kill NVVS pid {} with {} SIGTERM attempts, escalating to SIGKILL",
                          childPid,
                          MAX_SIGTERM_ATTEMPTS);
            }

            dcgm_mutex_unlock(&m_mutex);
            ret = m_coreProxy.ChildProcessStop(childHandle, true); // true = use SIGKILL
            dcgm_mutex_lock(&m_mutex);

            if (ret != DCGM_ST_OK)
            {
                log_error("Failed to send SIGKILL to process {}: {}", childPid, errorString(ret));
            }

            sigkilled = true;
        }

        dcgm_mutex_unlock(&m_mutex);
        if (kill_count == 0)
        {
            // Only yield on first attempt, perhaps the process exits quickly
            std::this_thread::yield();
        }
        else
        {
            // Sleep on subsequent retry attempts
            // The lock was just released.
            // coverity[sleep: FALSE]
            std::this_thread::sleep_for(SIGTERM_RETRY_DELAY);
        }
        dcgm_mutex_lock(&m_mutex);
        kill_count++;
    }

    if (kill_count >= maxRetries && processExists)
    {
        // Child process died and may not resume hostengine (EUD only?)
        // Resume request may block, temporary release of lock.
        dcgm_mutex_unlock(&m_mutex);
        dcgmReturn_t dcgmReturn = PauseResumeHostEngine(false); // Resume hostengine
        dcgm_mutex_lock(&m_mutex);

        if (dcgmReturn != DCGM_ST_OK)
        {
            log_error("Unable to resume engine, error: {}", dcgmReturn, errorString(dcgmReturn));
            // m_mutex is locked
            // coverity[missing_unlock: FALSE]
            return dcgmReturn;
        }

        log_error("Giving up attempting to kill NVVS process {} after {} retries", childPid, maxRetries);
        // m_mutex is (still) locked
        // coverity[missing_unlock: FALSE]
        return DCGM_ST_CHILD_NOT_KILLED;
    }

    // Process successfully terminated
    // Clean up the handle
    dcgm_mutex_unlock(&m_mutex);
    if (childHandle != 0)
    {
        ret = m_coreProxy.ChildProcessDestroy(childHandle, 0);
    }
    dcgm_mutex_lock(&m_mutex);

    // m_mutex is (still) locked
    // coverity[missing_unlock: FALSE]
    return DCGM_ST_OK;
}

/*****************************************************************************/
std::string DcgmDiagManager::GetNvvsBinPath()
{
    std::string nvvsBinPath, cmd;
    const char *value;
    int result;

    // Default NVVS binary path
    cmd = "/usr/libexec/datacenter-gpu-manager-4/nvvs";

    // Check for NVVS binary path enviroment variable
    value = std::getenv("NVVS_BIN_PATH");
    if (value != NULL)
    {
        nvvsBinPath = std::string(value) + "/nvvs";
        // Check if file exists
        result = access(nvvsBinPath.c_str(), F_OK);
        if (result == 0)
        {
            cmd = std::move(nvvsBinPath);
            DCGM_LOG_DEBUG << "The new NVVS binary path is: " << cmd;
            return cmd;
        }
        else
        {
            DCGM_LOG_WARNING << "Ignoring specified NVVS binary path " << value
                             << " because the file cannot be accessed.";
        }
    }
    return cmd;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::EnforceGPUConfiguration(unsigned int gpuId, dcgm_connection_id_t connectionId)
{
    dcgm_config_msg_enforce_gpu_v1 msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length       = sizeof(msg);
    msg.header.moduleId     = DcgmModuleIdConfig;
    msg.header.subCommand   = DCGM_CONFIG_SR_ENFORCE_GPU;
    msg.header.connectionId = connectionId;
    msg.header.version      = dcgm_config_msg_enforce_gpu_version;
    msg.gpuId               = gpuId;

    dcgmReturn = m_coreProxy.SendModuleCommand(&msg);

    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("ProcessModuleCommand returned {}.", (int)dcgmReturn);
        for (unsigned int i = 0; i < msg.numStatuses; i++)
        {
            log_error("Error in Enforcing Configuration. API Err Code: {} "
                      "GPU ID: {} Field ID: {} Additional Error Code: {}",
                      dcgmReturn,
                      msg.statuses[i].gpuId,
                      msg.statuses[i].fieldId,
                      msg.statuses[i].errorCode);
        }
    }
    else
    {
        /* Log that enforcing of configuration is successful */
        log_info("After safe reset, configuration enforced successfully for GPU ID {}", gpuId);
        return dcgmReturn;
    }

    log_info("Configuration enforced successfully for GPU ID {}", gpuId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::AddRunOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_v10 *drd) const
{
    std::string testParms;
    std::string testNames;

    // Tests to run
    for (unsigned int i = 0; i < DCGM_MAX_TEST_NAMES && drd->testNames[i][0] != '\0'; i++)
    {
        if (testNames.size() > 0)
            testNames += ",";
        testNames += drd->testNames[i];
    }

    cmdArgs.push_back("--specifiedtest");
    if (testNames.size() > 0)
    {
        cmdArgs.push_back(std::move(testNames));
    }
    else
    {
        switch (drd->validate)
        {
            case DCGM_POLICY_VALID_SV_SHORT:
                cmdArgs.push_back("short");
                break;
            case DCGM_POLICY_VALID_SV_MED:
                cmdArgs.push_back("medium");
                break;
            case DCGM_POLICY_VALID_SV_LONG:
                cmdArgs.push_back("long");
                break;
            case DCGM_POLICY_VALID_SV_XLONG:
                cmdArgs.push_back("xlong");
                break;
            default:
                log_error("Bad drd->validate {}", drd->validate);
                return DCGM_ST_BADPARAM;
        }
    }

    // Test parameters
    for (unsigned int i = 0; i < DCGM_MAX_TEST_PARMS && drd->testParms[i][0] != '\0'; i++)
    {
        if (testParms.size() > 0)
            testParms += ";";
        testParms += drd->testParms[i];
    }
    if (testParms.size() > 0)
    {
        cmdArgs.push_back("--parameters");
        cmdArgs.push_back(std::move(testParms));
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagManager::AddConfigFile(dcgmRunDiag_v10 *drd, std::vector<std::string> &cmdArgs) const
{
    static const unsigned int MAX_RETRIES = 3;

    size_t configFileContentsSize = strnlen(drd->configFileContents, sizeof(drd->configFileContents));

    if (configFileContentsSize > 0)
    {
        char fileName[] = "/tmp/tmp-dcgm-XXXXXX";
        int fd          = -1;

        for (unsigned int retries = 0; retries < MAX_RETRIES && fd == -1; retries++)
        {
            // According to man 2 umask, there is no way to read the current
            // umask through the API without also setting it. It can be read
            // through /proc/[pid]/status, but that is likely excessive and is
            // only supported on Linux 4.7+. So we silence the warning here, and
            // we chmod the file to 600 below as we don't want to set the umask
            // here without being able to revert it to its original value
            // coverity[secure_temp]
            fd = mkstemp(fileName);
        }

        if (fd == -1)
        {
            DCGM_LOG_ERROR << "Couldn't create a temporary configuration file for NVVS: " << std::strerror(errno);
            return DCGM_ST_GENERIC_ERROR;
        }

        int ret = chmod(fileName, 0600);
        if (ret == -1)
        {
            close(fd);
            DCGM_LOG_ERROR << "Couldn't chmod a temporary configuration file for NVVS: " << std::strerror(errno);
            return DCGM_ST_GENERIC_ERROR;
        }
        size_t written = write(fd, drd->configFileContents, configFileContentsSize);
        close(fd);
        // Adjust file permissions
        if (auto serviceAccount = GetServiceAccount(m_coreProxy); serviceAccount.has_value())
        {
            if (auto cred = GetUserCredentials((*serviceAccount).c_str()); cred.has_value())
            {
                if (chown(fileName, (*cred).uid, (*cred).gid) != 0)
                {
                    auto const err = errno;

                    DCGM_LOG_ERROR << fmt::format(
                        "The service account {} is specified, but it's impossible to set permissions for the file '{}'"
                        ". chown returned ({}) {}",
                        *serviceAccount,
                        fileName,
                        err,
                        strerror(err));

                    throw std::system_error(
                        std::error_code(err, std::generic_category()),
                        fmt::format("Unable to change permissions for the configuration file {}", fileName));
                }
            }
            else
            {
                DCGM_LOG_ERROR << fmt::format(
                    "The service account {} is specified, but there is not such user in the system."
                    " Unable to set permissions for the file {}",
                    (*serviceAccount),
                    fileName);

                return DCGM_ST_GENERIC_ERROR;
            }
        }
        else
        {
            DCGM_LOG_DEBUG << fmt::format(
                "Service account is not specified. Skipping permissions adjustments for the config file {}", fileName);
        }

        if (written == configFileContentsSize)
        {
            cmdArgs.push_back("--config");
            cmdArgs.push_back(fileName);
        }
        else
        {
            DCGM_LOG_ERROR << "Failed to write the temporary file for NVVS: " << std::strerror(errno);
            return DCGM_ST_GENERIC_ERROR;
        }
    }
    else
    {
        cmdArgs.push_back("--configless");
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmDiagManager::AddMiscellaneousNvvsOptions(std::vector<std::string> &cmdArgs,
                                                  dcgmRunDiag_v10 *drd,
                                                  std::string const &fakeGpuIds,
                                                  std::string const &entityIds) const
{
    if (drd->flags & DCGM_RUN_FLAGS_STATSONFAIL)
    {
        cmdArgs.push_back("--statsonfail");
    }

    if (drd->flags & DCGM_RUN_FLAGS_VERBOSE)
    {
        cmdArgs.push_back("-v");
    }

    if (strlen(drd->debugLogFile) > 0)
    {
        cmdArgs.push_back("-l");
        std::string debugArg(drd->debugLogFile);
        cmdArgs.push_back(std::move(debugArg));
    }

    if (strlen(drd->statsPath) > 0)
    {
        cmdArgs.push_back("--statspath");
        std::string statsPathArg(drd->statsPath);
        cmdArgs.push_back(std::move(statsPathArg));
    }

    // Gpu ids
    if (!fakeGpuIds.empty())
    {
        cmdArgs.push_back("-f");
        cmdArgs.push_back(fakeGpuIds);
    }
    else if (entityIds.size())
    {
        cmdArgs.push_back("--entity-id");
        cmdArgs.push_back(entityIds);
    }

    // Logging severity
    if (drd->debugLevel != DCGM_INT32_BLANK)
    {
        cmdArgs.push_back("-d");
        cmdArgs.push_back(std::string(LoggingSeverityToString(drd->debugLevel, DCGM_LOGGING_DEFAULT_NVVS_SEVERITY)));
    }

    // Plugin path
    const char *pluginDir = getenv(NVVS_PLUGIN_DIR);
    if (pluginDir)
    {
        cmdArgs.push_back("-p");
        cmdArgs.push_back(std::string(pluginDir));
    }

    if (drd->clocksEventMask[0] != '\0')
    {
        cmdArgs.push_back("--clocksevent-mask");
        cmdArgs.push_back(std::string(drd->clocksEventMask));
    }

    if (drd->flags & DCGM_RUN_FLAGS_FAIL_EARLY)
    {
        cmdArgs.push_back("--fail-early");
        if (drd->failCheckInterval)
        {
            cmdArgs.push_back("--check-interval");
            char buf[30];
            snprintf(buf, sizeof(buf), "%u", drd->failCheckInterval);
            cmdArgs.push_back(std::string(buf));
        }
    }

    if (drd->totalIterations > 1)
    {
        cmdArgs.push_back("--current-iteration");
        cmdArgs.push_back(fmt::format("{}", drd->currentIteration));
        cmdArgs.push_back("--total-iterations");
        cmdArgs.push_back(fmt::format("{}", drd->totalIterations));
    }

    unsigned int constexpr DEFAULT_WATCH_FREQUENCY_IN_MICROSECONDS { 5000000 };
    if (drd->watchFrequency > 0 && drd->watchFrequency != DEFAULT_WATCH_FREQUENCY_IN_MICROSECONDS)
    {
        cmdArgs.push_back("--watch-frequency");
        cmdArgs.push_back(fmt::format("{}", drd->watchFrequency));
    }

    if (strlen(drd->ignoreErrorCodes) > 0)
    {
        cmdArgs.push_back("--ignoreErrorCodes");
        cmdArgs.push_back(fmt::format("{}", drd->ignoreErrorCodes));
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::CreateNvvsCommand(std::vector<std::string> &cmdArgs,
                                                dcgmRunDiag_v10 *drd,
                                                unsigned int diagResponseVersion,
                                                std::string const &fakeGpuIds,
                                                std::string const &entityIds,
                                                ExecuteWithServiceAccount useServiceAccount)
{
    dcgmReturn_t ret;

    // Ensure given vector is empty
    if (!cmdArgs.empty())
    {
        return DCGM_ST_BADPARAM;
    }
    // Reserve enough space for args
    cmdArgs.reserve(25);

    cmdArgs.push_back(m_nvvsPath);
    cmdArgs.push_back("--channel-fd");
    cmdArgs.push_back(std::to_string(NVVS_CHANNEL_FD));
    cmdArgs.push_back("--response-version");
    cmdArgs.push_back(std::to_string(diagResponseVersion));

    if (useServiceAccount == ExecuteWithServiceAccount::No)
    {
        cmdArgs.push_back("--rerun-as-root");
    }

    ret = AddRunOptions(cmdArgs, drd);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    if ((ret = AddConfigFile(drd, cmdArgs)) != DCGM_ST_OK)
    {
        // Failure logged in AddConfigFile()
        return ret;
    }

    AddMiscellaneousNvvsOptions(cmdArgs, drd, fakeGpuIds, entityIds);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformNVVSExecute(std::string *stdoutStr,
                                                 std::string *stderrStr,
                                                 dcgmRunDiag_v10 *drd,
                                                 DcgmDiagResponseWrapper &response,
                                                 std::string const &fakeGpuIds,
                                                 std::string const &entityIds,
                                                 ExecuteWithServiceAccount useServiceAccount)
{
    std::vector<std::string> args;

    if (auto const ret = CreateNvvsCommand(args, drd, response.GetVersion(), fakeGpuIds, entityIds, useServiceAccount);
        ret != DCGM_ST_OK)
    {
        return ret;
    }

    return PerformExternalCommand(args, response, stdoutStr, stderrStr, useServiceAccount);
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformDummyTestExecute(std::string *stdoutStr, std::string *stderrStr)
{
    std::vector<std::string> args;
    std::unique_ptr<dcgmDiagResponse_v12> responseUptr = std::make_unique<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper response;

    response.SetVersion(responseUptr.get());
    args.emplace_back("dummy");
    return PerformExternalCommand(args, response, stdoutStr, stderrStr);
}

/****************************************************************************/
uint64_t DcgmDiagManager::GetTicket()
{
    // It is assumed that the calling thread has locked m_mutex so we can safely modify shared variables
    m_ticket += 1;
    return m_ticket;
}

/****************************************************************************/
void DcgmDiagManager::UpdateChildPID(pid_t value, uint64_t myTicket)
{
    DcgmLockGuard lock(&m_mutex);
    // Check to see if another thread has updated the pid
    if (myTicket != m_ticket)
    {
        // If another thread has updated the PID, do not update the pid
        return;
    }
    m_nvvsPID = value;
}

static unsigned int GetStructVersion(std::span<std::byte> data)
{
    // Get the version from the first four bytes of data. If data
    // is smaller than expected, return a blank value.
    if (data.size() < sizeof(unsigned int))
    {
        return DCGM_INT32_BLANK;
    }

    unsigned int version;
    std::memcpy(&version, data.data(), sizeof(version));
    return version;
}

static void PrintDiagStatus(dcgmDiagStatus_v1 const &diagStatus)
{
    std::stringstream diagStatusStr;
    diagStatusStr << "version " << diagStatus.version << "\ttotalTests " << diagStatus.totalTests << "\tcompletedTests "
                  << diagStatus.completedTests << "\tplugin name " << diagStatus.testName << "\terror code "
                  << diagStatus.errorCode;
    log_debug("Diag status : {}", diagStatusStr.str());
}

void DcgmDiagManager::UpdateDiagStatus(std::span<std::byte> data) const
{
    dcgmDiagStatus_v1 diagStatus {};
    DcgmFvBuffer buf;
    if (sizeof(diagStatus) != data.size())
    {
        log_debug("Data size {} does not match dcgmDiagStatus_v1 size {}", data.size(), sizeof(diagStatus));
        return;
    }

    memcpy(&diagStatus, data.data(), data.size());
    PrintDiagStatus(diagStatus);

    auto mapIt = m_testNameResultFieldId.find(diagStatus.testName);
    if (mapIt == m_testNameResultFieldId.end())
    {
        log_debug("Test name '{}' cannot be mapped to a result field id. Skipping updating test result field id.",
                  diagStatus.testName);
    }
    else
    {
        auto fieldId = mapIt->second;
        auto now     = timelib_usecSince1970();
        buf.AddInt64Value(DCGM_FE_NONE, 0, fieldId, diagStatus.errorCode, now, DCGM_ST_OK);
    }

    auto now = timelib_usecSince1970();
    buf.AddBlobValue(DCGM_FE_NONE, 0, DCGM_FI_DEV_DIAG_STATUS, &diagStatus, sizeof(diagStatus), now, DCGM_ST_OK);
    auto ret = m_coreProxy.AppendSamples(&buf);
    if (ret != DCGM_ST_OK)
    {
        log_error("Unable to append DCGM_FI_DEV_DIAG_STATUS to the cache manager");
    }
}

/****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformExternalCommand(std::vector<std::string> &args,
                                                     DcgmDiagResponseWrapper &response,
                                                     std::string *const stdoutStr,
                                                     std::string *const stderrStr,
                                                     ExecuteWithServiceAccount useServiceAccount)
{
    if (args.empty() || stdoutStr == nullptr || stderrStr == nullptr)
    {
        log_error("PerformExternalCommand: args is empty or NULL stdoutStr or stderrStr");
        return DCGM_ST_BADPARAM;
    }

    std::string filename;
    fmt::memory_buffer stdoutStream;
    fmt::memory_buffer stderrStream;
    struct stat fileStat = {};
    int statSt;
    pid_t pid = -1;
    uint64_t myTicket;

    AppendDummyArgs(args);
    std::string nvvsPath = args[0];

    // Validate executable exists and is accessible
    filename = nvvsPath;
    if (statSt = stat(filename.c_str(), &fileStat); statSt != 0)
    {
        log_error("stat of {} failed. errno {}: {}", filename, errno, strerror(errno));
        return DCGM_ST_NVVS_BINARY_NOT_FOUND;
    }

    // Check for previous run of nvvs and launch new one if previous one is no longer running
    {
        DcgmLockGuard lock(&m_mutex);
        if (auto const ret = CanRunNewNvvsInstance(); ret != DCGM_ST_OK)
        {
            return ret;
        }

        auto serviceAccount
            = useServiceAccount == ExecuteWithServiceAccount::Yes ? GetServiceAccount(m_coreProxy) : std::nullopt;

        if (serviceAccount.has_value())
        {
            // traverse all parts of the filename path and check if the serviceAccount has read and execute permissions
            auto serviceAccountCredentials = GetUserCredentials((*serviceAccount).c_str());
            if (!serviceAccountCredentials.has_value())
            {
                DCGM_LOG_ERROR << "Failed to get user credentials for service account " << (*serviceAccount);
                return DCGM_ST_GENERIC_ERROR;
            }
            std::set<gid_t> accountGroups { (*serviceAccountCredentials).groups.begin(),
                                            (*serviceAccountCredentials).groups.end() };
            std::filesystem::path path(filename);
            std::filesystem::path traverseSoFar;
            for (auto const &pathPart : path)
            {
                traverseSoFar /= pathPart;
                if (statSt = stat(traverseSoFar.c_str(), &fileStat); statSt != 0)
                {
                    auto const msg = fmt::format(
                        "Unable to validate if the service account {} has read and execute permissions to access the file {}."
                        " Stat failed for {} with errno {}: {}",
                        (*serviceAccount),
                        filename,
                        traverseSoFar.string(),
                        errno,
                        strerror(errno));
                    DCGM_LOG_ERROR << msg;
                    *stderrStr = std::move(msg);
                    return DCGM_ST_GENERIC_ERROR;
                }

                bool const canBeAccessedByUser = fileStat.st_uid == (*serviceAccountCredentials).uid
                                                 && (fileStat.st_mode & S_IRUSR) == S_IRUSR
                                                 && (fileStat.st_mode & S_IXUSR) == S_IXUSR;

                bool const canBeAccessedByGroup = (fileStat.st_gid == (*serviceAccountCredentials).gid
                                                   || accountGroups.find(fileStat.st_gid) != accountGroups.end())
                                                  && (fileStat.st_mode & S_IRGRP) == S_IRGRP
                                                  && (fileStat.st_mode & S_IXGRP) == S_IXGRP;

                bool const canBeAccessedByAll
                    = (fileStat.st_mode & S_IROTH) == S_IROTH && (fileStat.st_mode & S_IXOTH) == S_IXOTH;

                if (canBeAccessedByUser || canBeAccessedByGroup || canBeAccessedByAll)
                {
                    continue;
                }

                auto const msg = fmt::format(
                    "The service account {} does not have read and execute permissions on the file/directory {}",
                    (*serviceAccount),
                    traverseSoFar.string());
                DCGM_LOG_ERROR << msg;
                *stderrStr = std::move(msg);
                return DCGM_ST_GENERIC_ERROR;
            }
        }

        // Setup child process parameters
        dcgmChildProcessParams_v1 childParams = {};
        childParams.version                   = dcgmChildProcessParams_version1;
        childParams.executable                = filename.c_str();
        childParams.dataChannelFd             = NVVS_CHANNEL_FD;
        if (serviceAccount.has_value())
        {
            childParams.userName = serviceAccount->c_str();
        }

        // skip args[0] as it is executable path
        std::vector<const char *> argPtrs;
        argPtrs.reserve(args.size());
        for (size_t i = 1; i < args.size(); i++)
        {
            argPtrs.push_back(args[i].c_str());
        }
        argPtrs.push_back(nullptr);

        childParams.args    = argPtrs.data();
        childParams.numArgs = argPtrs.size() - 1;

        // Get ticket for PID tracking
        myTicket = GetTicket();

        // Spawn process using proxy
        ChildProcessHandle_t handle;
        dcgmReturn_t ret = m_coreProxy.ChildProcessSpawn(childParams, handle, pid);
        if (ret != DCGM_ST_OK || pid < 0)
        {
            log_error("Unable to run external command '{}'. Error: {}", nvvsPath, errorString(ret));
            if (handle != 0)
            {
                m_coreProxy.ChildProcessDestroy(handle);
            }
            return DCGM_ST_DIAG_BAD_LAUNCH;
        }

        m_childProcessHandle = handle;
        UpdateChildPID(pid, myTicket);
    }

    /* Do not return DCGM_ST_DIAG_BAD_LAUNCH for errors after this point since the child has been launched - use
       DCGM_ST_NVVS_ERROR or DCGM_ST_GENERIC_ERROR instead */
    log_debug("Launched external command '{}' (PID: {})", fmt::to_string(fmt::join(args, " ")), pid);


    // Get file descriptors for stdout/stderr/data
    int dataFd   = -1;
    int stdoutFd = -1;
    int stderrFd = -1;

    if (auto ret = m_coreProxy.ChildProcessGetStdOutHandle(m_childProcessHandle, stdoutFd); ret != DCGM_ST_OK)
    {
        log_error("Failed to get stdout handle: {}", errorString(ret));
    }

    if (auto ret = m_coreProxy.ChildProcessGetStdErrHandle(m_childProcessHandle, stderrFd); ret != DCGM_ST_OK)
    {
        log_error("Failed to get stderr handle: {}", errorString(ret));
    }

    if (auto ret = m_coreProxy.ChildProcessGetDataChannelHandle(m_childProcessHandle, dataFd); ret != DCGM_ST_OK)
    {
        log_error("Failed to get data channel handle: {}", errorString(ret));
    }

    // Read from stdout pipe
    if (stdoutFd >= 0)
    {
        fmt::memory_buffer buffer;
        ReadProcessOutput(buffer, DcgmNs::Utils::FileHandle(stdoutFd));
        *stdoutStr = fmt::to_string(buffer);
    }

    // Read from stderr pipe
    if (stderrFd >= 0)
    {
        fmt::memory_buffer buffer;
        ReadProcessOutput(buffer, DcgmNs::Utils::FileHandle(stderrFd));
        *stderrStr = fmt::to_string(buffer);
    }

    // Read the FD channel
    if (dataFd >= 0)
    {
        DcgmNs::Utils::FileHandle dataHandle(dataFd);
        if (auto ret = ReadDataFromFd(dataHandle, response); ret != DCGM_ST_OK)
        {
            log_error("Failed to read data from FD: {}", errorString(ret));
            if (m_childProcessHandle != 0)
            {
                m_coreProxy.ChildProcessDestroy(m_childProcessHandle);
                m_childProcessHandle = 0;
            }
            return ret;
        }
    }

    log_debug("External command stdout: {}", SanitizedString(*stdoutStr));
    log_debug("External command stderr: {}", SanitizedString(*stderrStr));

    if (auto ret = m_coreProxy.ChildProcessWait(m_childProcessHandle); ret != DCGM_ST_OK)
    {
        log_error("Error waiting for child process: {}", errorString(ret));
        if (m_childProcessHandle != 0)
        {
            m_coreProxy.ChildProcessDestroy(m_childProcessHandle);
            m_childProcessHandle = 0;
        }
        return DCGM_ST_NVVS_ERROR;
    }

    // Reset pid so that future runs know that nvvs is no longer running
    UpdateChildPID(-1, myTicket);

    // Get process status
    dcgmChildProcessStatus_v1 status = {};
    status.version                   = dcgmChildProcessStatus_version1;

    if (auto ret = m_coreProxy.ChildProcessGetStatus(m_childProcessHandle, status); ret != DCGM_ST_OK)
    {
        log_error("Error getting child process status: {}", errorString(ret));
        if (m_childProcessHandle != 0)
        {
            m_coreProxy.ChildProcessDestroy(m_childProcessHandle);
            m_childProcessHandle = 0;
        }
        return DCGM_ST_NVVS_ERROR;
    }

    // Handle process exit status
    if (status.running)
    {
        log_error("The external command '{}' may still be running.", args[0]);
        return DCGM_ST_NVVS_ERROR;
    }

    if (status.exitCode != 0)
    {
        if (status.receivedSignal)
        {
            log_error("The external command '{}' returned a non-zero exit code: {}, received signal: {}.",
                      args[0],
                      status.exitCode,
                      status.receivedSignalNumber);
            stderrStr->insert(0,
                              fmt::format("The DCGM diagnostic subprocess was terminated due to signal {}"
                                          "\n**************\n",
                                          status.receivedSignalNumber));
            if (m_childProcessHandle != 0)
            {
                m_coreProxy.ChildProcessDestroy(m_childProcessHandle);
                m_childProcessHandle = 0;
            }
            return DCGM_ST_NVVS_KILLED;
        }

        if (status.exitCode == NVVS_ST_TEST_NOT_FOUND)
        {
            if (m_childProcessHandle != 0)
            {
                m_coreProxy.ChildProcessDestroy(m_childProcessHandle);
                m_childProcessHandle = 0;
            }
            return DCGM_ST_NVVS_NO_AVAILABLE_TEST;
        }

        log_error("The external command '{}' returned a non-zero exit code: {}", args[0], status.exitCode);
    }

    // Cleanup
    if (auto ret = m_coreProxy.ChildProcessDestroy(m_childProcessHandle); ret != DCGM_ST_OK)
    {
        log_error("Error destroying child process: {}", errorString(ret));
    }

    return DCGM_ST_OK;
}

void DcgmDiagManager::AppendDummyArgs(std::vector<std::string> &args)
{
    if (args[0] == "dummy") // for unittests
    {
        args[0] = "./dummy_script";
        args.emplace_back("arg1");
        args.emplace_back("arg2");
        args.emplace_back("arg3");
    }
}

/****************************************************************************/
dcgmReturn_t DcgmDiagManager::StopRunningDiag()
{
    DcgmLockGuard lock(&m_mutex);
    if (m_nvvsPID < 0)
    {
        DCGM_LOG_DEBUG << "No diagnostic is running";
        return DCGM_ST_OK;
    }
    // Stop the running diagnostic
    DCGM_LOG_DEBUG << "Stopping diagnostic with PID " << m_nvvsPID;
    return KillActiveNvvs(5);
    /* Do not wait for child - let the thread that originally launched the diagnostic manage the child and reset
       pid. We do not reset the PID here because it can result in multiple nvvs processes running (e.g. previous
       nvvs process has not stopped yet, and a new one is launched because we've reset the pid).
    */
}

dcgmReturn_t DcgmDiagManager::ResetGpuAndEnforceConfig(unsigned int gpuId,
                                                       dcgmPolicyAction_t action,
                                                       dcgm_connection_id_t connectionId)
{
    if (action == DCGM_POLICY_ACTION_GPURESET)
    {
        EnforceGPUConfiguration(gpuId, connectionId);
        return DCGM_ST_OK;
    }

    if (action == DCGM_POLICY_ACTION_NONE)
    {
        return DCGM_ST_OK;
    }

    DCGM_LOG_ERROR << "Invalid action given to execute: " << action;
    return DCGM_ST_GENERIC_ERROR;
}

namespace
{
/**
 * @brief Result type of the \c ExecuteAndParseNvvsOutput function
 */
struct ExecuteAndParseNvvsResult
{
    dcgmReturn_t ret = DCGM_ST_GENERIC_ERROR; /*!< Return code of the nvvs execution. If value is DCGM_ST_OK, the
                                               * \c results field is guaranteed to be populated. */
};

/**
 * @brief Executes nvvs and parses the JSON output to the DiagnosticResults structure
 * @param self DiagManager instance
 * @param response Response
 * @param drd Diagnostic request data
 * @param gpuIds GPU IDs
 * @param useServiceAccount Whether to use the service account
 * @return \c ExecuteAndParseNvvsResult structure.
 *         If the ret field is \c DCGM_ST_OK, the results field is guaranteed to be populated.
 * @note This function fills in the response system error information in case of an error.
 */
auto ExecuteAndParseNvvs(DcgmDiagManager &self,
                         DcgmDiagResponseWrapper &response,
                         dcgmRunDiag_v10 *drd,
                         std::string const &fakeGpuIds,
                         std::string const &entityIds,
                         DcgmDiagManager::ExecuteWithServiceAccount useServiceAccount
                         = DcgmDiagManager::ExecuteWithServiceAccount::Yes) -> ExecuteAndParseNvvsResult
{
    ExecuteAndParseNvvsResult result {};
    std::string stdoutStr;
    std::string stderrStr;

    result.ret
        = self.PerformNVVSExecute(&stdoutStr, &stderrStr, drd, response, fakeGpuIds, entityIds, useServiceAccount);
    if (result.ret != DCGM_ST_OK)
    {
        auto const msg = fmt::format("Error when executing the diagnostic: {}\n"
                                     "Nvvs stderr: \n{}\n",
                                     errorString(result.ret),
                                     stderrStr);
        if (result.ret != DCGM_ST_NVVS_NO_AVAILABLE_TEST)
        {
            log_error(msg);
            response.RecordSystemError({ msg.data(), msg.size() });
        }
        else
        {
            log_debug(msg);
        }

        return result;
    }

    if (!response.GetSystemErr().empty())
    {
        // If original return was OK, change return code to indicate NVVS error
        result.ret = (result.ret == DCGM_ST_OK) ? DCGM_ST_NVVS_ERROR : result.ret;
    }

    return result;
}

/**
 * @brief Parses a comma-separated list of GPU IDs and filters out invalid IDs.
 * @param[in] input A string containing comma-separated GPU IDs.
 * @return A vector of valid GPU IDs parsed from the input string.
 */
std::vector<std::uint32_t> ParseAndFilterGpuList(std::string_view input)
{
    std::vector<std::uint32_t> gpuList;
    gpuList.reserve(DCGM_MAX_NUM_DEVICES);
    for (auto const &str : DcgmNs::Split(input, ','))
    {
        std::uint32_t parsedInt {};
        auto [p, ecc] = std::from_chars(str.data(), str.data() + str.size(), parsedInt);
        if (ecc == std::errc())
        {
            gpuList.push_back(parsedInt);
        }
        else
        {
            log_error("Failed to parse GPU ID from string '{}'. Will be skipped", str);
        }
    }
    return gpuList;
}

/**
 * @brief Parses an entity ids string and filters out GPU with valid IDs.
 * @param[in] input A string containing entity ids (e.g. gpu:0,1).
 * @return A vector of valid GPU IDs parsed from the input string.
 */
std::vector<std::uint32_t> ParseEntityIdsAndFilterGpu(DcgmCoreProxy &coreProxy, std::string_view entityIds)
{
    dcgmReturn_t ret;

    std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
    ret = coreProxy.GetAllGpuInfo(gpuInfos);
    if (ret != DCGM_ST_OK)
    {
        log_error("failed to get gpu info from core proxy: {}", errorString(ret));
        return {};
    }
    std::vector<std::pair<unsigned, std::string>> gpuIdUuids;
    for (auto const &gpuInfo : gpuInfos)
    {
        // Skip GPUs that are lost to the CM
        if (gpuInfo.status == DcgmEntityStatusLost)
        {
            continue;
        }
        gpuIdUuids.emplace_back(gpuInfo.gpuId, gpuInfo.uuid);
    }

    dcgmMigHierarchy_v2 migHierarchy;
    ret = coreProxy.GetGpuInstanceHierarchy(migHierarchy);
    if (ret != DCGM_ST_OK)
    {
        log_error("failed to get gpu instance hierachy from core proxy: {}", errorString(ret));
        return {};
    }
    return DcgmNs::ParseEntityIdsAndFilterGpu(migHierarchy, gpuIdUuids, entityIds);
}

/**
 * @brief Executes EUD with root permissions if the service account does not have root permissions
 * @param[in] diagManager DiagManager instance
 * @param[in] drd Diagnostic request data
 * @param[in] response Response wrapper
 * @param[in] serviceAccount Service account name
 * @param[in] fakeGpuIds Comma-separated list of fake GPU IDs
 * @param[in,out] entityIds Entity-id list
 * @param[in] testNames List of EUD tests to run
 * @return true if the EUD was executed, false otherwise
 */
bool ExecuteEudPluginsAsRoot(DcgmDiagManager &diagManager,
                             dcgmRunDiag_v10 const *drd,
                             DcgmDiagResponseWrapper &eudResponse,
                             char const *serviceAccount,
                             std::string const &fakeGpuIds,
                             std::string const &entityIds,
                             std::vector<std::string> const &testNames)
{
    if (auto serviceAccountCredentials = GetUserCredentials(serviceAccount);
        !serviceAccountCredentials.has_value() || (*serviceAccountCredentials).gid == 0
        || (*serviceAccountCredentials).uid == 0 || std::ranges::count((*serviceAccountCredentials).groups, 0) > 0)
    {
        log_debug("Skip second EUD plugin's run if the service account has root permissions");
        return false;
    }

    log_debug("Run EUD plugin with root permissions");
    auto eudDrd     = *drd;
    eudDrd.validate = DCGM_POLICY_VALID_NONE;
    memset(eudDrd.testNames, 0, sizeof(eudDrd.testNames));
    for (unsigned int index = 0; auto const &test : testNames)
    {
        SafeCopyTo(eudDrd.testNames[index++], test.c_str());
    }

    if (auto ret = ExecuteAndParseNvvs(
            diagManager, eudResponse, &eudDrd, fakeGpuIds, entityIds, DcgmDiagManager::ExecuteWithServiceAccount::No);
        ret.ret != DCGM_ST_OK)
    {
        log_debug("failed to rerun NVVS with root, ret: [{}].", ret.ret);
        return false;
    }

    for (auto const &test : testNames)
    {
        if (eudResponse.HasTest(test) == false)
        {
            log_warning("{} test is missing from results", test);
        }
    }

    return true;
}

} // namespace

std::tuple<WasExecuted_t, dcgmReturn_t> DcgmDiagManager::RerunEudDiagAsRoot(
    dcgmRunDiag_v10 *drd,
    std::string const &fakeGpuIds,
    std::string const &entityIds,
    std::unordered_set<std::string_view> const &testNames,
    dcgmReturn_t lastRunRet,
    DcgmDiagResponseWrapper &response)
{
    std::vector<std::string> eudRerunTestNames;
    static const std::string EUD_TEST     = "eud";
    static const std::string CPU_EUD_TEST = "cpu_eud";
    WasExecuted_t wasExecuted { WasExecuted_t::WAS_NOT_EXECUTED };

    if (drd->validate == DCGM_POLICY_VALID_SV_LONG || drd->validate == DCGM_POLICY_VALID_SV_XLONG
        || testNames.contains("production_testing"))
    {
        eudRerunTestNames.push_back(EUD_TEST);
        eudRerunTestNames.push_back(CPU_EUD_TEST);
    }
    else
    {
        if (testNames.contains(EUD_TEST))
        {
            eudRerunTestNames.push_back(EUD_TEST);
        }
        if (testNames.contains(CPU_EUD_TEST))
        {
            eudRerunTestNames.push_back(CPU_EUD_TEST);
        }
    }

    if (eudRerunTestNames.empty())
    {
        return { wasExecuted, DCGM_ST_OK };
    }

    std::unique_ptr<dcgmDiagResponse_v12> v12;
    std::unique_ptr<dcgmDiagResponse_v11> v11;
    std::unique_ptr<dcgmDiagResponse_v10> v10;
    std::unique_ptr<dcgmDiagResponse_v9> v9;
    std::unique_ptr<dcgmDiagResponse_v8> v8;
    DcgmDiagResponseWrapper eudResponse;

    switch (response.GetVersion())
    {
        case dcgmDiagResponse_version12:
            v12          = MakeUniqueZero<dcgmDiagResponse_v12>();
            v12->version = dcgmDiagResponse_version12;
            eudResponse.SetVersion(v12.get());
            break;
        case dcgmDiagResponse_version11:
            v11          = MakeUniqueZero<dcgmDiagResponse_v11>();
            v11->version = dcgmDiagResponse_version11;
            eudResponse.SetVersion(v11.get());
            break;
        case dcgmDiagResponse_version10:
            v10          = MakeUniqueZero<dcgmDiagResponse_v10>();
            v10->version = dcgmDiagResponse_version10;
            eudResponse.SetVersion(v10.get());
            break;
        case dcgmDiagResponse_version9:
            v9          = MakeUniqueZero<dcgmDiagResponse_v9>();
            v9->version = dcgmDiagResponse_version9;
            eudResponse.SetVersion(v9.get());
            break;
        case dcgmDiagResponse_version8:
            v8          = MakeUniqueZero<dcgmDiagResponse_v8>();
            v8->version = dcgmDiagResponse_version8;
            eudResponse.SetVersion(v8.get());
            break;
        case dcgmDiagResponse_version7:
            log_error("dcgmDiagResponse_version7 does not support eud.");
            return { wasExecuted, DCGM_ST_OK };
        default:
            log_error("unknown version: [{}].", response.GetVersion());
            return { wasExecuted, DCGM_ST_VER_MISMATCH };
    }

    auto serviceAccount = GetServiceAccount(m_coreProxy);
    if (!serviceAccount.has_value())
    {
        return { wasExecuted, DCGM_ST_OK };
    }

    if (!ExecuteEudPluginsAsRoot(
            *this, drd, eudResponse, (*serviceAccount).c_str(), fakeGpuIds, entityIds, eudRerunTestNames))
    {
        return { wasExecuted, DCGM_ST_OK };
    }

    wasExecuted = WasExecuted_t::WAS_EXECUTED;

    dcgmReturn_t eudRet = DCGM_ST_OK;
    if (lastRunRet == DCGM_ST_OK)
    {
        eudRet = response.MergeEudResponse(eudResponse);
    }
    else if (lastRunRet == DCGM_ST_NVVS_NO_AVAILABLE_TEST)
    {
        // If the last execution was DCGM_ST_NVVS_NO_AVAILABLE_TEST, most fields of the response will be
        // empty. In this case, we're relying on the EUD response instead.
        eudRet = response.AdoptEudResponse(eudResponse);
    }

    if (eudRet != DCGM_ST_OK)
    {
        log_error("failed to apply eud response, ret: [{}]", eudRet);
        return { wasExecuted, eudRet };
    }

    return { wasExecuted, DCGM_ST_OK };
}

dcgmReturn_t DcgmDiagManager::RunDiag(dcgmRunDiag_v10 *drd, DcgmDiagResponseWrapper &response)
{
    bool areAllSameSku = true;
    std::vector<unsigned int> gpuIds;
    std::string fakeGpuIds;
    std::string entityIds;

    /* NVVS is only allowed to run on a single SKU at a time. Returning this error gives
     * users a deterministic response. See bug 1714115 and its related bugs for details
     */
    if (strlen(drd->fakeGpuList) != 0U)
    {
        log_debug("Parsing fake GPU list: {}", drd->fakeGpuList);
        gpuIds        = ParseAndFilterGpuList(drd->fakeGpuList);
        areAllSameSku = m_coreProxy.AreAllGpuIdsSameSku(gpuIds);
        fakeGpuIds    = fmt::format("{}", fmt::join(gpuIds, ","));
    }
    else if (drd->groupId != DCGM_GROUP_NULL)
    {
        dcgmReturn_t ret = DCGM_ST_OK;
        // Check the group
        ret = m_coreProxy.AreAllTheSameSku(0, static_cast<unsigned int>(drd->groupId), &areAllSameSku);
        if (ret != DCGM_ST_OK)
        {
            log_error("Got st {} from AreAllTheSameSku()", (int)ret);
            return ret;
        }

        ret = m_coreProxy.GetGroupGpuIds(0, static_cast<unsigned int>(drd->groupId), gpuIds);
        if (ret != DCGM_ST_OK)
        {
            log_error("Got st {} from GetGroupGpuIds()", (int)ret);
            return ret;
        }
        entityIds = fmt::format("{}", fmt::join(gpuIds, ","));
    }
    else
    {
        // Check entity ids
        log_debug("Parsing entity ids: {}", drd->entityIds);
        gpuIds        = ParseEntityIdsAndFilterGpu(m_coreProxy, drd->entityIds);
        areAllSameSku = m_coreProxy.AreAllGpuIdsSameSku(gpuIds);
        entityIds     = drd->entityIds;
    }

    if (fakeGpuIds.empty() && entityIds.empty())
    {
        log_debug("Cannot perform diag: {}", errorString(DCGM_ST_GROUP_IS_EMPTY));
        return DCGM_ST_GROUP_IS_EMPTY;
    }

    if (!areAllSameSku)
    {
        DCGM_LOG_DEBUG << "GPUs are incompatible for Validation";
        return DCGM_ST_GROUP_INCOMPATIBLE;
    }

    unsigned int expectedNumGpus = 0;
    auto err                     = DcgmNs::ParseExpectedNumEntitiesForGpus(drd->expectedNumEntities, expectedNumGpus);
    if (!err.empty())
    {
        log_error(err);
        return DCGM_ST_BADPARAM;
    }

    if (expectedNumGpus > 0 && expectedNumGpus != gpuIds.size())
    {
        log_error("expectedNumGpus [{}] does not match the number of GPUs [{}] listed/discovered in the system.",
                  expectedNumGpus,
                  gpuIds.size());
        return DCGM_ST_BADPARAM;
    }

    if (!fakeGpuIds.empty())
    {
        log_debug("Running diag on fake gpu ids: {}", fakeGpuIds);
    }
    else
    {
        log_debug("Running diag on entities: {}", entityIds);
    }

    if ((drd->validate == DCGM_POLICY_VALID_NONE) && (strlen(drd->testNames[0]) == 0))
    {
        return DCGM_ST_OK;
    }

    auto nvvsResults = ExecuteAndParseNvvs(*this, response, drd, fakeGpuIds, entityIds);
    // Some tests require root access. If no tests are found to run, consider re-running them
    // with root privileges to attempt execution again. Need to check again below for NO_AVAILABLE_TEST.
    if (nvvsResults.ret != DCGM_ST_OK && nvvsResults.ret != DCGM_ST_NVVS_NO_AVAILABLE_TEST)
    {
        // Do not overwrite the response system error here as it should already have more specific information
        return nvvsResults.ret;
    }

    /*
     * EUD plugins requires root permissions to run even if the service account is specified.
     * At this moment, the EUD is included into long/xlong/production_testing and if specified directly via -r eud.
     * The nvvs binary skips EUD tests if run as non-root.
     * So we need to run EUD separately if the service account is specified, and it does not have root permissions.
     *
     * The production_testing is not a validation level and rather a "meta" test name. If individual tests are
     * specified, the validate is equal to none, and we need to check the test names manually.
     *
     * It should be forbidden to have validation level other than none if the test names are specified.
     */
    std::unordered_set<std::string_view> testNames;
    if (drd->validate == DCGM_POLICY_VALID_NONE)
    {
        for (auto const &testName : drd->testNames)
        {
            if (testName[0] != '\0')
            {
                testNames.insert(testName);
            }
        }
    }

    auto [eudWasRerun, eudRet] = RerunEudDiagAsRoot(drd, fakeGpuIds, entityIds, testNames, nvvsResults.ret, response);
    if (eudRet != DCGM_ST_OK)
    {
        log_error("failed to rerun eud as root, err: [{}].", eudRet);
        return eudRet;
    }

    // NVVS may lack of root permission, we add cpu serials in host engine here.
    if (!response.AddCpuSerials())
    {
        log_debug("failed to add cpu serials to response.");
    }

    // Return any results previously skipped for RerunEudDiagAsRoot()
    if (eudWasRerun == WasExecuted_t::WAS_NOT_EXECUTED && nvvsResults.ret != DCGM_ST_OK)
    {
        return nvvsResults.ret;
    }

    return DCGM_ST_OK;
}

std::string DcgmDiagManager::GetCompareTestName(const std::string &testname)
{
    std::string compareName(testname);
    std::transform(compareName.begin(), compareName.end(), compareName.begin(), [](unsigned char c) {
        if (c == ' ')
        {
            return '_';
        }
        return (char)std::tolower(c);
    });

    return compareName;
}

unsigned int DcgmDiagManager::GetTestIndex(const std::string &testName)
{
    std::string compareName = GetCompareTestName(testName);
    unsigned int index      = DCGM_PER_GPU_TEST_COUNT_V8;
    if (compareName == "diagnostic")
        index = DCGM_DIAGNOSTIC_INDEX;
    else if (compareName == "pcie")
        index = DCGM_PCI_INDEX;
    else if (compareName == "sm_stress")
        index = DCGM_SM_STRESS_INDEX;
    else if (compareName == "targeted_stress")
        index = DCGM_TARGETED_STRESS_INDEX;
    else if (compareName == "targeted_power")
        index = DCGM_TARGETED_POWER_INDEX;
    else if (compareName == "memory_bandwidth")
        index = DCGM_MEMORY_BANDWIDTH_INDEX;
    else if (compareName.find("memory") != std::string::npos)
        index = DCGM_MEMORY_INDEX;
    else if (compareName == "memtest")
        index = DCGM_MEMTEST_INDEX;
    else if (compareName == "context_create")
        index = DCGM_CONTEXT_CREATE_INDEX;
    else if (compareName == "pulse_test")
    {
        index = DCGM_PULSE_TEST_INDEX;
    }
    else if (compareName == "eud")
    {
        index = DCGM_EUD_TEST_INDEX;
    }
    else if (compareName == "nvbandwidth")
    {
        index = DCGM_NVBANDWIDTH_INDEX;
    }

    return index;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::RunDiagAndAction(dcgmRunDiag_v10 *drd,
                                               dcgmPolicyAction_t action,
                                               DcgmDiagResponseWrapper &response,
                                               dcgm_connection_id_t connectionId)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK; /* Return value from sub-calls */
    dcgmReturn_t retVal     = DCGM_ST_OK; /* Return value from this function */
    dcgmReturn_t retValidation;
    dcgmReturn_t retAction;
    std::vector<dcgmGroupEntityPair_t> entities;

    log_debug("performing action {} on group {} with validation {}", action, drd->groupId, drd->validate);

    // RAII resource management - Not moving to member variable, need to release as soon as it goes out of this method
    DcgmResourceHandle lock(m_coreProxy);
    dcgmReturn_t resRet = lock.GetInitResult();
    if (resRet != DCGM_ST_OK)
    {
        return resRet;
    }

    if (drd->groupId != DCGM_GROUP_NULL)
    {
        // Verify group id is valid and update it if no GPU list was specified
        unsigned int gId = (uintptr_t)drd->groupId;

        dcgmReturn = m_coreProxy.VerifyAndUpdateGroupId(&gId);
        if (DCGM_ST_OK != dcgmReturn)
        {
            DCGM_LOG_ERROR << "Error: Bad group id " << gId << ": " << errorString(dcgmReturn);
            return dcgmReturn;
        }

        dcgmReturn = m_coreProxy.GetGroupEntities(gId, entities);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Error: Failed to get group entities for group " << gId << ": "
                           << errorString(dcgmReturn);
            return dcgmReturn;
        }

        drd->groupId = (dcgmGpuGrp_t)(long long)gId; // save the updated value
    }

    if (action != DCGM_POLICY_ACTION_NONE)
    {
        for (unsigned int i = 0; i < entities.size(); i++)
        {
            /* Policies only work on GPUs for now */
            if (entities[i].entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }

            unsigned int gpuId = entities[i].entityId;
            retAction          = ResetGpuAndEnforceConfig(gpuId, action, connectionId);

            if (retAction != DCGM_ST_OK)
            {
                retVal = retAction; /* Tell caller the error */
            }
        }
    }

    if ((drd->validate != DCGM_POLICY_VALID_NONE) || (strlen(drd->testNames[0]) > 0))
    {
        retValidation = RunDiag(drd, response);

        if (retValidation != DCGM_ST_OK)
        {
            retVal = retValidation; /* Tell caller the error */
        }
    }

    return retVal;
}

dcgmReturn_t DcgmDiagManager::CanRunNewNvvsInstance() const
{
    if (m_amShuttingDown)
    {
        DCGM_LOG_WARNING << "Not running diag due to DCGM shutting down.";
        return DCGM_ST_DIAG_ALREADY_RUNNING; // Not perfect but seems to be the sanest return that already exists
    }

    if (m_nvvsPID > 0)
    {
        // nvvs instance already running - do not launch a new one
        log_warning("Previous instance of nvvs is still running. PID: {}", m_nvvsPID);
        return DCGM_ST_DIAG_ALREADY_RUNNING;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::PauseResumeHostEngine(bool pause = false)
{
    dcgm_core_msg_pause_resume_v1 msg {};

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_PAUSE_RESUME;
    msg.header.version    = dcgm_core_msg_pause_resume_version1;
    msg.pause             = pause;

    dcgmReturn_t ret = m_coreProxy.SendModuleCommand(&msg);
    log_debug("PauseResumeHostEngine({}) returns {} ({})", pause, ret, errorString(ret));
    return ret;
}


dcgmReturn_t DcgmDiagManager::ReadDataFromFd(DcgmNs::Utils::FileHandle &dataFd, DcgmDiagResponseWrapper &response)
{
    int bytesRead;
    unsigned int receivedResultFrameNum = 0;
    constexpr size_t buffCapacity       = std::max({ sizeof(dcgmDiagStatus_v1),
                                                     sizeof(dcgmDiagResponse_v12),
                                                     sizeof(dcgmDiagResponse_v11),
                                                     sizeof(dcgmDiagResponse_v10),
                                                     sizeof(dcgmDiagResponse_v9),
                                                     sizeof(dcgmDiagResponse_v8),
                                                     sizeof(dcgmDiagResponse_v7) });
    static_assert(buffCapacity >= sizeof(unsigned int),
                  "buffCapacity must be greater than or equal to sizeof(unsigned int)");
    std::vector<std::byte> buffer(buffCapacity);

    while (true)
    {
        bytesRead = dataFd.ReadExact(buffer.data(), sizeof(unsigned int));
        if (bytesRead == 0)
        {
            // EOF
            break;
        }
        if (bytesRead < 0)
        {
            log_error("Error reading from data channel: {}", strerror(dataFd.GetErrno()));
            break;
        }

        auto version        = GetStructVersion(buffer);
        size_t expectedSize = 0;
        switch (version)
        {
            case dcgmDiagStatus_version1:
                expectedSize = sizeof(dcgmDiagStatus_v1);
                break;
            case dcgmDiagResponse_version12:
                expectedSize = sizeof(dcgmDiagResponse_v12);
                break;
            case dcgmDiagResponse_version11:
                expectedSize = sizeof(dcgmDiagResponse_v11);
                break;
            case dcgmDiagResponse_version10:
                expectedSize = sizeof(dcgmDiagResponse_v10);
                break;
            case dcgmDiagResponse_version9:
                expectedSize = sizeof(dcgmDiagResponse_v9);
                break;
            case dcgmDiagResponse_version8:
                expectedSize = sizeof(dcgmDiagResponse_v8);
                break;
            case dcgmDiagResponse_version7:
                expectedSize = sizeof(dcgmDiagResponse_v7);
                break;
            default:
                log_error("Unexpected struct with version {} from nvvs channel", version);
                return DCGM_ST_NVVS_ERROR;
        }

        if (expectedSize > buffCapacity)
        {
            log_error("Expected size {} is greater than buffCapacity {}", expectedSize, buffCapacity);
            return DCGM_ST_NVVS_ERROR;
        }

        bytesRead = dataFd.ReadExact(buffer.data() + sizeof(unsigned int), expectedSize - sizeof(unsigned int));
        if (bytesRead == 0)
        {
            // EOF
            break;
        }
        if (bytesRead < 0)
        {
            log_error("Error reading from data channel: {}", strerror(dataFd.GetErrno()));
            break;
        }

        if (version == dcgmDiagStatus_version1)
        {
            UpdateDiagStatus({ buffer.data(), expectedSize });
        }
        else
        {
            auto const &ret = response.SetResult({ buffer.data(), expectedSize });
            if (ret != DCGM_ST_OK)
            {
                log_error("failed to set results, err: [{}]", ret);
                return ret;
            }
            receivedResultFrameNum++;
        }
    }

    if (receivedResultFrameNum == 0)
    {
        log_error("Diag result struct not received from NVVS.");
    }
    return DCGM_ST_OK;
}
