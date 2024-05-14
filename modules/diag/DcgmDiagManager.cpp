/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmError.h"
#include "DcgmStringHelpers.h"
#include "DcgmUtilities.h"
#include "Defer.hpp"
#include "NvvsJsonStrings.h"
#include "dcgm_config_structs.h"
#include "dcgm_core_structs.h"
#include "dcgm_structs.h"
#include "serialize/DcgmJsonSerialize.hpp"

#include <JsonResult.hpp>

#include <algorithm>
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
    if (m_nvvsPID >= 0)
    {
        DCGM_LOG_DEBUG << "Cleaning up leftover nvvs process with pid " << m_nvvsPID;
        KillActiveNvvs((unsigned int)-1); // don't stop until it's dead
    }
}

dcgmReturn_t DcgmDiagManager::KillActiveNvvs(unsigned int maxRetries)
{
    static const std::chrono::seconds SIGTERM_RETRY_DELAY = std::chrono::seconds(4);
    static const unsigned int MAX_SIGTERM_ATTEMPTS        = 4;

    pid_t childPid          = m_nvvsPID;
    unsigned int kill_count = 0;
    bool sigkilled          = false;

    assert(m_mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);

    while (kill_count <= maxRetries && kill(childPid, 0) == 0)
    {
        // As long as the process exists, keep killing it
        if (kill_count < MAX_SIGTERM_ATTEMPTS)
        {
            kill(childPid, SIGTERM);
        }
        else
        {
            if (sigkilled == false)
            {
                DCGM_LOG_ERROR << "Unable to kill nvvs pid with 3 SIGTERM attempts, escalating to SIGKILL. pid: "
                               << childPid;
            }

            kill(childPid, SIGKILL);
            sigkilled = true;
        }

        // sleep on subsequent retry attempts
        if (kill_count > 0)
        {
            dcgm_mutex_unlock(&m_mutex);
            // Yield, then wait for the process to exit.
            std::this_thread::yield();
            // The lock was just released.
            // coverity[sleep: FALSE]
            std::this_thread::sleep_for(SIGTERM_RETRY_DELAY);
            dcgm_mutex_lock(&m_mutex);
        }
        kill_count++;
    }

    if (kill_count > maxRetries)
    {
        // Child process died and may not resume hostengine (EUD only?)
        // Resume request may block, temporary release of lock.
        dcgm_mutex_unlock(&m_mutex);
        dcgmReturn_t dcgmReturn = PauseResumeHostEngine(false);
        dcgm_mutex_lock(&m_mutex);

        if (dcgmReturn != DCGM_ST_OK)
        {
            log_error("Unable to resume engine, error: {}", dcgmReturn, errorString(dcgmReturn));
            // m_mutex is locked
            // coverity[missing_unlock: FALSE]
            return dcgmReturn;
        }

        DCGM_LOG_ERROR << "Giving up attempting to kill NVVS process " << childPid << " after " << maxRetries
                       << " retries.";
        // m_mutex is (still) locked
        // coverity[missing_unlock: FALSE]
        return DCGM_ST_CHILD_NOT_KILLED;
    }

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
    cmd = "/usr/share/nvidia-validation-suite/nvvs";

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
dcgmReturn_t DcgmDiagManager::AddRunOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd) const
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
        cmdArgs.push_back(testNames);
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
        cmdArgs.push_back(testParms);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagManager::AddConfigFile(dcgmRunDiag_t *drd, std::vector<std::string> &cmdArgs) const
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
                                                  dcgmRunDiag_t *drd,
                                                  const std::string &gpuIds) const
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
        cmdArgs.push_back(debugArg);
    }

    if (strlen(drd->statsPath) > 0)
    {
        cmdArgs.push_back("--statspath");
        std::string statsPathArg(drd->statsPath);
        cmdArgs.push_back(statsPathArg);
    }

    // Gpu ids
    if (strlen(drd->fakeGpuList) > 0)
    {
        cmdArgs.push_back("-f");
        cmdArgs.push_back(gpuIds);
    }
    else if (gpuIds.length())
    {
        cmdArgs.push_back("--indexes");
        cmdArgs.push_back(gpuIds);
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

    if (drd->throttleMask[0] != '\0')
    {
        cmdArgs.push_back("--throttle-mask");
        cmdArgs.push_back(std::string(drd->throttleMask));
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
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::CreateNvvsCommand(std::vector<std::string> &cmdArgs,
                                                dcgmRunDiag_t *drd,
                                                std::string const &gpuIds) const
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

    // Request json output and say we're DCGM
    cmdArgs.push_back("-j");
    cmdArgs.push_back("-z");

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

    AddMiscellaneousNvvsOptions(cmdArgs, drd, gpuIds);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformNVVSExecute(std::string *stdoutStr,
                                                 std::string *stderrStr,
                                                 dcgmRunDiag_t *drd,
                                                 std::string const &gpuIds,
                                                 ExecuteWithServiceAccount useServiceAccount) const
{
    std::vector<std::string> args;

    if (auto const ret = CreateNvvsCommand(args, drd, gpuIds); ret != DCGM_ST_OK)
    {
        return ret;
    }

    return PerformExternalCommand(args, stdoutStr, stderrStr, useServiceAccount);
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformDummyTestExecute(std::string *stdoutStr, std::string *stderrStr) const
{
    std::vector<std::string> args;
    args.emplace_back("dummy");
    return PerformExternalCommand(args, stdoutStr, stderrStr);
}

/****************************************************************************/
uint64_t DcgmDiagManager::GetTicket() const
{
    // It is assumed that the calling thread has locked m_mutex so we can safely modify shared variables
    m_ticket += 1;
    return m_ticket;
}

/****************************************************************************/
void DcgmDiagManager::UpdateChildPID(pid_t value, uint64_t myTicket) const
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

/****************************************************************************/
dcgmReturn_t DcgmDiagManager::PerformExternalCommand(std::vector<std::string> &args,
                                                     std::string *const stdoutStr,
                                                     std::string *const stderrStr,
                                                     ExecuteWithServiceAccount useServiceAccount) const
{
    if (stdoutStr == nullptr || stderrStr == nullptr)
    {
        DCGM_LOG_ERROR << "PerformExternalCommand: NULL stdoutStr or stderrStr";
        return DCGM_ST_BADPARAM;
    }

    std::string filename;
    fmt::memory_buffer stdoutStream;
    fmt::memory_buffer stderrStream;
    struct stat fileStat = {};
    int statSt;
    int childStatus;
    DcgmNs::Utils::FileHandle stdoutFd;
    DcgmNs::Utils::FileHandle stderrFd;
    pid_t pid = -1;
    uint64_t myTicket;
    int errno_cached; /* Cached value of errno for logging */

    AppendDummyArgs(args);

    /* See if the program we're planning to run even exists. We have to do this because popen() will still
     * succeed, even if the program isn't found.
     */
    filename = args[0];

    if (statSt = stat(filename.c_str(), &fileStat); statSt != 0)
    {
        DCGM_LOG_ERROR << "stat of " << filename << " failed. errno " << errno << ": " << strerror(errno);
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
                    *stderrStr = msg;
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
                *stderrStr = msg;
                return DCGM_ST_GENERIC_ERROR;
            }
        }

        // Run command
        pid = DcgmNs::Utils::ForkAndExecCommand(args,
                                                nullptr,
                                                &stdoutFd,
                                                &stderrFd,
                                                false,
                                                serviceAccount.has_value() ? (*serviceAccount).c_str() : nullptr,
                                                nullptr);
        // Update the nvvs pid
        myTicket = GetTicket();
        UpdateChildPID(pid, myTicket);
    }

    if (pid < 0)
    {
        DCGM_LOG_ERROR << fmt::format("Unable to run external command '{}'.", args[0]);
        return DCGM_ST_DIAG_BAD_LAUNCH;
    }
    /* Do not return DCGM_ST_DIAG_BAD_LAUNCH for errors after this point since the child has been launched - use
       DCGM_ST_NVVS_ERROR or DCGM_ST_GENERIC_ERROR instead */
    DCGM_LOG_DEBUG << fmt::format("Launched external command '{}' (PID: {})", args[0], pid);

    auto killPidGuard = DcgmNs::Defer { [&]() {
        if (pid > 0)
        {
            DCGM_LOG_ERROR << "Killing DCGM diagnostic subprocess due to a communication error. External command: "
                           << args[0];
            kill(pid, SIGKILL);
            // Prevent zombie child
            if (waitpid(pid, nullptr, 0) == -1)
            {
                log_error("waitpid returned -1 for pid {}", pid);
            }
            DCGM_LOG_DEBUG << "The child process has been killed.";
            UpdateChildPID(-1, myTicket);
            pid = -1;
        }
    } };

    if (auto const ret = ReadProcessOutput(stdoutStream, stderrStream, std::move(stdoutFd), std::move(stderrFd));
        ret != DCGM_ST_OK)
    {
        *stdoutStr = fmt::to_string(stdoutStream);
        *stderrStr = fmt::to_string(stderrStream);
        DCGM_LOG_DEBUG << "External command stdout (partial): " << SanitizedString(*stdoutStr);
        DCGM_LOG_DEBUG << "External command stderr (partial): " << SanitizedString(*stderrStr);
        return ret;
    }

    // Disarming the killPidGuard as at this point the child process exit status will be handled explicitly
    killPidGuard.Disarm();

    // Set output string in caller's context
    // Do this before the error check so that if there are errors, we have more useful error messages
    *stdoutStr = fmt::to_string(stdoutStream);
    *stderrStr = fmt::to_string(stderrStream);
    DCGM_LOG_DEBUG << "External command stdout: " << SanitizedString(*stdoutStr);
    DCGM_LOG_DEBUG << "External command stderr: " << SanitizedString(*stderrStr);

    // Get exit status of child
    if (waitpid(pid, &childStatus, 0) == -1)
    {
        errno_cached = errno;
        DCGM_LOG_ERROR << fmt::format("There was an error waiting for external command '{}' (PID: {}) to exit: {}",
                                      args[0],
                                      pid,
                                      strerror(errno_cached));

        UpdateChildPID(-1, myTicket);
        return DCGM_ST_NVVS_ERROR;
    }

    // Reset pid so that future runs know that nvvs is no longer running
    UpdateChildPID(-1, myTicket);

    // Check exit status
    if (WIFEXITED(childStatus))
    {
        // Exited normally - check for non-zero exit code
        childStatus = WEXITSTATUS(childStatus);
        if (childStatus != EXIT_SUCCESS)
        {
            /* If the nvvs has a non-zero exit code, that may mean some handled exception is properly wrapped into a
             * json object and printed to stdout. The nvvs command itself was successful from our point of view. Now
             * it's up to the upper caller logic to decide if the stdout is valid */
            DCGM_LOG_DEBUG << fmt::format(
                "The external command '{}' returned a non-zero exit code: {}", args[0], childStatus);
            return DCGM_ST_OK;
        }
    }
    else if (WIFSIGNALED(childStatus))
    {
        // Child terminated due to signal
        childStatus = WTERMSIG(childStatus);
        DCGM_LOG_ERROR << "The external command '" << args[0] << "' was terminated due to signal " << childStatus;
        stderrStr->insert(0,
                          fmt::format("The DCGM diagnostic subprocess was terminated due to signal {}"
                                      "\n**************\n",
                                      childStatus));
        return DCGM_ST_NVVS_KILLED;
    }
    else
    {
        // We should never hit this in practice, but it is possible if the child process is being traced via ptrace
        DCGM_LOG_DEBUG << "The external command '" << args[0] << "' is being traced";
        return DCGM_ST_NVVS_ERROR;
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
    std::optional<DcgmNs::Nvvs::Json::DiagnosticResults> results {}; /*!< Parsed results of the nvvs execution.
                                                                      * This field is populated only if \c ret is
                                                                      * DCGM_ST_OK.*/
};

static std::string_view SanitizeNvvsJson(std::string_view stdoutStr)
{
    /*
     *  Sometimes during the tests NVML outputs warnings like
     *  WARNING: Failed to acquire log file lock. File is in use by a different instance
     *  right to the stdout where we expect only JSON objects.
     */
    std::string_view jsonStr = stdoutStr;
    jsonStr.remove_prefix(std::min(jsonStr.find_first_of('{'), jsonStr.size()));
    if (jsonStr.length() != stdoutStr.length())
    {
        log_warning("NVVS JSON output has some non-JSON prefix");
        log_debug("NVVS non-JSON prefix: {}", stdoutStr.substr(0, stdoutStr.length() - jsonStr.length()));
    }
    return jsonStr;
}

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
auto ExecuteAndParseNvvs(DcgmDiagManager const &self,
                         DcgmDiagResponseWrapper const &response,
                         dcgmRunDiag_t *drd,
                         std::string const &gpuIds,
                         DcgmDiagManager::ExecuteWithServiceAccount useServiceAccount
                         = DcgmDiagManager::ExecuteWithServiceAccount::Yes) -> ExecuteAndParseNvvsResult
{
    ExecuteAndParseNvvsResult result {};
    std::string stdoutStr;
    std::string stderrStr;

    result.ret = self.PerformNVVSExecute(&stdoutStr, &stderrStr, drd, gpuIds, useServiceAccount);
    if (result.ret != DCGM_ST_OK)
    {
        auto const msg = fmt::format("Error when executing the diagnostic: {}\n"
                                     "Nvvs stderr: \n{}\n",
                                     errorString(result.ret),
                                     stderrStr);
        log_error(msg);
        response.RecordSystemError({ msg.data(), msg.size() });

        return result;
    }

    auto tmpResults
        = DcgmNs::JsonSerialize::TryDeserialize<DcgmNs::Nvvs::Json::DiagnosticResults>(SanitizeNvvsJson(stdoutStr));
    if (!tmpResults.has_value())
    {
        auto const msg = fmt::format("Failed to parse the NVVS JSON output\n"
                                     "Nvvs stderr: \n{}\n",
                                     SanitizedString(stderrStr));
        log_error(msg);
        response.RecordSystemError({ msg.data(), msg.size() });
        log_debug("Sanitized raw NVVS output: {}", SanitizedString(stdoutStr));

        return { DCGM_ST_NVVS_ERROR, std::nullopt };
    }

    log_debug("NVVS parsed JSON: {}", SanitizeNvvsJson(stdoutStr));

    result.results = std::move(*tmpResults);

    return result;
}

/**
 * @brief Checks if the plugin is missing from the NVVS JSON results
 * @param results Parsed NVVS results
 * @return true if the error code indicates that the plugin is missing (NVVS_ST_TEST_NOT_FOUND), false otherwise
 */
bool IsPluginMissing(DcgmNs::Nvvs::Json::DiagnosticResults const &results)
{
    if (results.errorCode.has_value() && results.errorCode.value() == NVVS_ST_TEST_NOT_FOUND)
    {
        return true;
    }

    return false;
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
 * @brief Executes EUD with root permissions if the service account does not have root permissions
 * @param[in] diagManager DiagManager instance
 * @param[in] drd Diagnostic request data
 * @param[in] response Response wrapper
 * @param[in] serviceAccount Service account name
 * @param[in] indexList Comma-separated list of GPU IDs
 * @param[in,out] nvvsResults NVVS results of previously run tests. The EUD results will be merged into this structure
 * @return DCGM_ST_OK if the EUD was executed successfully, an error code otherwise
 */
dcgmReturn_t ExecuteEudAsRoot(DcgmDiagManager const &diagManager,
                              dcgmRunDiag_t const *drd,
                              DcgmDiagResponseWrapper const &response,
                              char const *serviceAccount,
                              std::string const &indexList,
                              std::optional<DcgmNs::Nvvs::Json::DiagnosticResults> &nvvsResults)
{
    if (auto serviceAccountCredentials = GetUserCredentials(serviceAccount);
        !serviceAccountCredentials.has_value() || (*serviceAccountCredentials).gid == 0
        || (*serviceAccountCredentials).uid == 0 || std::ranges::count((*serviceAccountCredentials).groups, 0) > 0)
    {
        log_debug("Skip second EUD if the service account has root permissions");
        return DCGM_ST_OK;
    }

    log_debug("Run EUD with root permissions");
    auto eudDrd     = *drd;
    eudDrd.validate = DCGM_POLICY_VALID_NONE;
    memset(eudDrd.testNames, 0, sizeof(eudDrd.testNames));
    SafeCopyTo(eudDrd.testNames[0], (char const *)"eud");

    auto eudResults = ExecuteAndParseNvvs(
        diagManager, response, &eudDrd, indexList, DcgmDiagManager::ExecuteWithServiceAccount::No);

    if (eudResults.ret != DCGM_ST_OK)
    {
        // Do not overwrite the response system error here
        return eudResults.ret;
    }

    if (!eudResults.results.has_value())
    {
        log_warning("EUD results are missing (empty optional)");
        return DCGM_ST_OK;
    }

    if (IsPluginMissing(*eudResults.results))
    {
        log_warning("EUD plugin is missing");
        return DCGM_ST_OK;
    }

    auto tmpResult = nvvsResults.value_or(DcgmNs::Nvvs::Json::DiagnosticResults {});

    if (!DcgmNs::Nvvs::Json::MergeResults(tmpResult, std::move(*eudResults.results)))
    {
        log_error("Failed to merge the NVVS JSON output");
        nvvsResults = std::move(tmpResult);
        response.RecordSystemError("Unable to merge JSON results for regular Diag and EUD tests");
        return DCGM_ST_NVVS_ERROR;
    }

    nvvsResults = std::move(tmpResult);

    return DCGM_ST_OK;
}

} // namespace

dcgmReturn_t DcgmDiagManager::RunDiag(dcgmRunDiag_t *drd, DcgmDiagResponseWrapper &response)
{
    dcgmReturn_t ret   = DCGM_ST_OK;
    bool areAllSameSku = true;
    std::vector<unsigned int> gpuIds;

    /* NVVS is only allowed to run on a single SKU at a time. Returning this error gives
     * users a deterministic response. See bug 1714115 and its related bugs for details
     */
    if (strlen(drd->fakeGpuList) != 0U)
    {
        log_debug("Parsing fake GPU list: {}", drd->fakeGpuList);
        gpuIds        = ParseAndFilterGpuList(drd->fakeGpuList);
        areAllSameSku = m_coreProxy.AreAllGpuIdsSameSku(gpuIds);
    }
    else if (strlen(drd->gpuList) != 0U)
    {
        log_debug("Parsing GPU list: {}", drd->gpuList);
        gpuIds        = ParseAndFilterGpuList(drd->gpuList);
        areAllSameSku = m_coreProxy.AreAllGpuIdsSameSku(gpuIds);
    }
    else
    {
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
    }

    if (gpuIds.empty())
    {
        log_debug("Cannot perform diag: {}", errorString(DCGM_ST_GROUP_IS_EMPTY));
        return DCGM_ST_GROUP_IS_EMPTY;
    }

    if (!areAllSameSku)
    {
        DCGM_LOG_DEBUG << "GPUs are incompatible for Validation";
        return DCGM_ST_GROUP_INCOMPATIBLE;
    }

    std::string indexList = fmt::format("{}", fmt::join(gpuIds, ","));
    log_debug("Running diag on GPUs: {}", indexList);

    if ((drd->validate == DCGM_POLICY_VALID_NONE) && (strlen(drd->testNames[0]) == 0))
    {
        return DCGM_ST_OK;
    }

    auto nvvsResults = ExecuteAndParseNvvs(*this, response, drd, indexList);
    if (nvvsResults.ret != DCGM_ST_OK)
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

    if (drd->validate == DCGM_POLICY_VALID_SV_LONG || drd->validate == DCGM_POLICY_VALID_SV_XLONG
        || testNames.contains("eud") || testNames.contains("production_testing"))
    {
        if (auto serviceAccount = GetServiceAccount(m_coreProxy); serviceAccount.has_value())
        {
            if (auto eudRet
                = ExecuteEudAsRoot(*this, drd, response, (*serviceAccount).c_str(), indexList, nvvsResults.results);
                eudRet != DCGM_ST_OK)
            {
                return eudRet;
            }
        }
    }

    /*
     * The FillResponseStructure function may return DCGM_ST_NVVS_ERROR if the results contain runtime errors.
     * Otherwise, the oldRet (ret) is returned.
     * In this execution branch the ret is never touched and is always DCGM_ST_OK as the previous non-OK results
     * are returned and logged immediately.
     *
     * If we see the ret == DCGM_ST_NVVS_ERROR here, that means we got some runtime errors in the nvvs execution
     * and we are reporting such errors back to the user.
     */
    ret = FillResponseStructure(
        nvvsResults.results.value_or(DcgmNs::Nvvs::Json::DiagnosticResults {}), response, drd->groupId, ret);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << fmt::format("Error happened during JSON parsing of NVVS output: {}", errorString(ret));
        // Do not overwrite the response system error here as there may be a more specific one already
    }

    return ret;
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

    return index;
}

static void PopulateErrorDetail(DcgmNs::Nvvs::Json::Warning const &warning, dcgmDiagErrorDetail_v2 &ed)
{
    if (warning.error_code.has_value())
    {
        ed.code = *warning.error_code;
    }

    if (warning.error_severity.has_value())
    {
        ed.severity = *warning.error_severity;
    }

    if (warning.error_category.has_value())
    {
        ed.category = *warning.error_category;
    }

    snprintf(ed.msg, sizeof(ed.msg), "%s", warning.message.c_str());
}

static void PopulateErrorDetail(DcgmNs::Nvvs::Json::Info const &info, dcgmDiagErrorDetail_v2 &ed)
{
    int errCode = DCGM_FR_OK;
    fmt::format_to_n(ed.msg, sizeof(ed.msg), "{}", fmt::join(info.messages, ","));

    ed.code = errCode;
}

dcgmDiagResult_t NvvsPluginResultToDiagResult(nvvsPluginResult_enum nvvsResult)
{
    switch (nvvsResult)
    {
        case NVVS_RESULT_PASS:
            return DCGM_DIAG_RESULT_PASS;
        case NVVS_RESULT_WARN:
            return DCGM_DIAG_RESULT_WARN;
        case NVVS_RESULT_FAIL:
            return DCGM_DIAG_RESULT_FAIL;
        case NVVS_RESULT_SKIP:
            return DCGM_DIAG_RESULT_SKIP;
        default:
        {
            log_error("Unknown NVVS result {}", static_cast<int>(nvvsResult));
            return DCGM_DIAG_RESULT_FAIL;
        }
    }
}

static std::string InfoToCsvString(DcgmNs::Nvvs::Json::Info const &info)
{
    return fmt::format("{}", fmt::join(info.messages, ", "));
}

void DcgmDiagManager::FillTestResult(DcgmNs::Nvvs::Json::Test const &test,
                                     DcgmDiagResponseWrapper &response,
                                     std::unordered_set<unsigned int> &gpuIdSet)
{
    if (test.name.empty() || test.results.empty())
    {
        return;
    }

    for (auto const &result : test.results)
    {
        auto testIndex = GetTestIndex(test.name);
        if (testIndex == DCGM_CONTEXT_CREATE_INDEX)
        {
            testIndex = 0; // Context create is only ever run by itself, and its result is stored in index 0.
        }

        dcgmDiagResult_t const ret = NvvsPluginResultToDiagResult(result.status.result);

        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            // Software test
            for (auto const gpuId : result.gpuIds.ids)
            {
                dcgmDiagErrorDetail_v2 id = { { 0 }, 0, 0, 0, 0 };
                response.SetGpuIndex(gpuId);
                gpuIdSet.insert(gpuId);

                if (result.info.has_value())
                {
                    ::PopulateErrorDetail(*result.info, id);
                }

                response.AddInfoDetail(0, testIndex, test.name, id, ret);

                if (!result.warnings.has_value())
                {
                    /* Populate empty error details */
                    dcgmDiagErrorDetail_v2 ed = { { 0 }, 0, 0, 0, 0 };
                    response.AddErrorDetail(0, testIndex, test.name, ed, 0, ret);
                    continue;
                }

                unsigned int edIndex = 0;

                for (auto const &warning : *result.warnings)
                {
                    dcgmDiagErrorDetail_v2 ed = { { 0 }, 0, 0, 0, 0 };

                    ::PopulateErrorDetail(warning, ed);
                    response.AddErrorDetail(0, testIndex, test.name, ed, edIndex, ret);
                    edIndex++;
                }
            }

            continue; // To next test.results item
        }

        for (auto const gpuId : result.gpuIds.ids)
        {
            unsigned int edIndex = 0;
            response.SetGpuIndex(gpuId);
            gpuIdSet.insert(gpuId);
            response.SetPerGpuResponseState(testIndex, ret, gpuId);

            if (result.info.has_value() && !(*result.info).messages.empty())
            {
                std::string info = InfoToCsvString(*result.info);
                response.AddPerGpuMessage(testIndex, info, gpuId, false);
            }

            if (!result.warnings.has_value())
            {
                continue;
            }

            for (auto const &warning : *result.warnings)
            {
                dcgmDiagErrorDetail_v2 ed = { { 0 }, 0, 0, 0, 0 };
                ::PopulateErrorDetail(warning, ed);
                response.AddErrorDetail(gpuId, testIndex, "", ed, edIndex, ret);

                edIndex++;
            }
        }
    }
}


/**
 * @brief Fills the response structure from the parsed NVVS results
 *
 * @param[in] results Parsed NVVS results
 * @param[out] response Diag response
 * @param[in] groupId Group ID to get the GPU count from
 * @param[in] oldRet Previous return code
 *
 * @return \a oldRet if the results do not contain runtime errors
 *         \ref DCGM_ST_NVVS_ERROR otherwise
 */
dcgmReturn_t DcgmDiagManager::FillResponseStructure(DcgmNs::Nvvs::Json::DiagnosticResults const &results,
                                                    DcgmDiagResponseWrapper &response,
                                                    int groupId,
                                                    dcgmReturn_t oldRet)
{
    unsigned int const numGpus = m_coreProxy.GetGpuCount(groupId);
    std::unordered_set<unsigned int> gpuIdSet;
    response.InitializeResponseStruct(numGpus);

    if (results.version.has_value())
    {
        response.RecordDcgmVersion(*results.version);
    }

    if (results.devIds.has_value())
    {
        response.RecordDevIds(*results.devIds);
    }

    if (results.devSerials.has_value())
    {
        response.RecordGpuSerials(*results.devSerials);
    }

    if (results.driverVersion.has_value())
    {
        response.RecordDriverVersion(*results.driverVersion);
    }

    if (results.runtimeError.has_value())
    {
        response.RecordSystemError((*results.runtimeError));
        // If oldRet was OK, change return code to indicate NVVS error
        return oldRet == DCGM_ST_OK ? DCGM_ST_NVVS_ERROR : oldRet;
    }
    else if (results.categories.has_value())
    {
        for (auto const &category : (*results.categories))
        {
            for (auto const &test : category.tests)
            {
                FillTestResult(test, response, gpuIdSet);
            }
        }

        response.SetGpuCount(gpuIdSet.size());
    }

    // Do not mask old errors if there are any
    return oldRet;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagManager::RunDiagAndAction(dcgmRunDiag_t *drd,
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

    if (drd->gpuList[0] == '\0')
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
dcgmReturn_t DcgmDiagManager::ReadProcessOutput(fmt::memory_buffer &stdoutStream,
                                                fmt::memory_buffer &stderrStream,
                                                DcgmNs::Utils::FileHandle stdoutFd,
                                                DcgmNs::Utils::FileHandle stderrFd) const
{
    std::array<char, 1024> buff = {};

    int epollFd = epoll_create1(EPOLL_CLOEXEC);
    if (epollFd < 0)
    {
        auto const err = errno;
        DCGM_LOG_ERROR << "epoll_create1 failed. errno " << err;
        return DCGM_ST_GENERIC_ERROR;
    }
    auto cleanupEpollFd = DcgmNs::Defer { [&]() {
        close(epollFd);
    } };

    struct epoll_event event = {};

    event.events  = EPOLLIN | EPOLLHUP;
    event.data.fd = stdoutFd.Get();
    if (epoll_ctl(epollFd, EPOLL_CTL_ADD, stdoutFd.Get(), &event) < 0)
    {
        auto const err = errno;
        DCGM_LOG_ERROR << "epoll_ctl failed. errno " << err;
        return DCGM_ST_GENERIC_ERROR;
    }

    event.data.fd = stderrFd.Get();
    if (epoll_ctl(epollFd, EPOLL_CTL_ADD, stderrFd.Get(), &event) < 0)
    {
        log_error("epoll_ctl failed. errno {}", errno);
        return DCGM_ST_GENERIC_ERROR;
    }

    int pipesLeft = 2;
    while (pipesLeft > 0)
    {
        int numEvents = epoll_wait(epollFd, &event, 1, -1);
        if (numEvents < 0)
        {
            auto const err = errno;
            DCGM_LOG_ERROR << "epoll_wait failed. errno " << err;
            return DCGM_ST_GENERIC_ERROR;
        }

        if (numEvents == 0)
        {
            // Timeout
            continue;
        }

        if (event.data.fd == stdoutFd.Get())
        {
            auto const bytesRead = read(stdoutFd.Get(), buff.data(), buff.size());
            if (bytesRead < 0)
            {
                auto const err = errno;
                if (err == EAGAIN || err == EINTR)
                {
                    continue;
                }
                DCGM_LOG_ERROR << "read from stdout failed. errno " << err;
                return DCGM_ST_GENERIC_ERROR;
            }
            if (bytesRead == 0)
            {
                if (epoll_ctl(epollFd, EPOLL_CTL_DEL, stdoutFd.Get(), nullptr) < 0)
                {
                    auto const err = errno;
                    DCGM_LOG_ERROR << "epoll_ctl to remove stdoutFd failed. errno " << err;
                    return DCGM_ST_GENERIC_ERROR;
                }
                pipesLeft -= 1;
            }
            else
            {
                stdoutStream.append(buff.data(), buff.data() + bytesRead);
            }
        }

        if (event.data.fd == stderrFd.Get())
        {
            auto const bytesRead = read(stderrFd.Get(), buff.data(), buff.size());
            if (bytesRead < 0)
            {
                auto const err = errno;
                if (err == EAGAIN || err == EINTR)
                {
                    continue;
                }
                DCGM_LOG_ERROR << "read from stderr failed. errno " << err;
                return DCGM_ST_GENERIC_ERROR;
            }
            if (bytesRead == 0)
            {
                if (epoll_ctl(epollFd, EPOLL_CTL_DEL, stderrFd.Get(), nullptr) < 0)
                {
                    auto const err = errno;
                    DCGM_LOG_ERROR << "epoll_ctl to remove stderrFd failed. errno " << err;
                    return DCGM_ST_GENERIC_ERROR;
                }
                pipesLeft -= 1;
            }
            else
            {
                stderrStream.append(buff.data(), buff.data() + bytesRead);
            }
        }
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
