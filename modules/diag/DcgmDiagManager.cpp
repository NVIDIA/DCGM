/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "dcgm_structs.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <iterator>
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
    static const unsigned int MAX_SIGTERM_ATTEMPTS = 3;

    unsigned int kill_count = 0;
    bool sigkilled          = false;

    while (kill(m_nvvsPID, 0) == 0 && kill_count <= maxRetries)
    {
        // As long as the process exists, keep killing it
        if (kill_count < MAX_SIGTERM_ATTEMPTS)
        {
            kill(m_nvvsPID, SIGTERM);
        }
        else
        {
            if (sigkilled == false)
            {
                DCGM_LOG_ERROR << "Unable to kill nvvs with 3 SIGTERM attempts, escalating to SIGKILL. pid: "
                               << m_nvvsPID;
            }

            kill(m_nvvsPID, SIGKILL);
            sigkilled = true;
        }
        kill_count++;
        std::this_thread::yield();
    }

    if (kill_count > maxRetries)
    {
        DCGM_LOG_ERROR << "Giving up attempting to kill NVVS process " << m_nvvsPID << " after " << maxRetries
                       << " retries.";
        return DCGM_ST_CHILD_NOT_KILLED;
    }

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
            cmd = nvvsBinPath;
            std::cout << "The new NVVS binary path is: " << cmd << std::endl;
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
        PRINT_ERROR("%d", "ProcessModuleCommand returned %d.", (int)dcgmReturn);
        for (unsigned int i = 0; i < msg.numStatuses; i++)
        {
            PRINT_ERROR("%d %d %d %d",
                        "Error in Enforcing Configuration. API Err Code: %d"
                        " GPU ID: %d Field ID: %d Additional Error Code: %d",
                        dcgmReturn,
                        msg.statuses[i].gpuId,
                        msg.statuses[i].fieldId,
                        msg.statuses[i].errorCode);
        }
    }
    else
    {
        /* Log that enforcing of configuration is successful */
        PRINT_INFO("%d", "After safe reset, configuration enforced successfully for GPU ID %d", gpuId);
        return dcgmReturn;
    }

    PRINT_INFO("%d", "Configuration enforced successfully for GPU ID %d", gpuId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmDiagManager::AddTrainingOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd) const
{
    // Training (golden values)
    if (drd->flags & DCGM_RUN_FLAGS_TRAIN)
    {
        cmdArgs.push_back("--train");
        if (drd->flags & DCGM_RUN_FLAGS_FORCE_TRAIN)
        {
            cmdArgs.push_back("--force");
        }

        if (drd->trainingIterations != 0)
        {
            std::stringstream buf;
            buf << drd->trainingIterations;
            cmdArgs.push_back("--training-iterations");
            cmdArgs.push_back(buf.str());
        }

        if (drd->trainingVariance != 0)
        {
            std::stringstream buf;
            buf << drd->trainingVariance;
            cmdArgs.push_back("--training-variance");
            cmdArgs.push_back(buf.str());
        }

        if (drd->trainingTolerance != 0)
        {
            std::stringstream buf;
            buf << drd->trainingTolerance;
            cmdArgs.push_back("--training-tolerance");
            cmdArgs.push_back(buf.str());
        }

        if (drd->goldenValuesFile[0] != '\0')
        {
            std::string path("/tmp/");
            path += drd->goldenValuesFile;
            cmdArgs.push_back("--golden-values-filename");
            cmdArgs.push_back(path);
        }

        return true;
    }

    return false;
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
                PRINT_ERROR("%u", "Bad drd->validate %u", drd->validate);
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
        cmdArgs.push_back(
            std::string(DcgmLogging::severityToString(drd->debugLevel, DCGM_LOGGING_DEFAULT_NVVS_SEVERITY)));
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

    if (AddTrainingOptions(cmdArgs, drd) == false)
    {
        ret = AddRunOptions(cmdArgs, drd);
        if (ret != DCGM_ST_OK)
        {
            return ret;
        }
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
                                                 std::string const &gpuIds) const
{
    std::vector<std::string> args;

    if (auto const ret = CreateNvvsCommand(args, drd, gpuIds); ret != DCGM_ST_OK)
    {
        return ret;
    }

    return PerformExternalCommand(args, stdoutStr, stderrStr);
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
                                                     std::string *const stderrStr) const
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
        DCGM_LOG_ERROR << "stat of " << filename << " failed. errno " << statSt << ": " << strerror(statSt);
        return DCGM_ST_NVVS_BINARY_NOT_FOUND;
    }

    // Check for previous run of nvvs and launch new one if previous one is no longer running
    {
        DcgmLockGuard lock(&m_mutex);
        if (auto const ret = CanRunNewNvvsInstance(); ret != DCGM_ST_OK)
        {
            return ret;
        }

        auto serviceAccount = GetServiceAccount(m_coreProxy);
        // Run command
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
                        statSt,
                        strerror(statSt));
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
                                                serviceAccount.has_value() ? (*serviceAccount).c_str() : nullptr);
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
            waitpid(pid, nullptr, 0);
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

dcgmReturn_t DcgmDiagManager::RunDiag(dcgmRunDiag_t *drd, DcgmDiagResponseWrapper &response)
{
    std::string stdoutStr;
    std::string stderrStr;
    dcgmReturn_t ret = DCGM_ST_OK;
    std::stringstream indexList;
    int areAllSameSku = 1;
    std::vector<unsigned int> gpuIds;

    /* NVVS is only allowed to run on a single SKU at a time. Returning this error gives
     * users a deterministic response. See bug 1714115 and its related bugs for details
     */
    if (strlen(drd->fakeGpuList) != 0u)
    {
        // Check just the supplied list
        std::vector<std::string> gpuIdStrs;
        dcgmTokenizeString(drd->fakeGpuList, ",", gpuIdStrs);

        for (size_t i = 0; i < gpuIdStrs.size(); i++)
        {
            gpuIds.push_back(strtol(gpuIdStrs[i].c_str(), NULL, 10));
        }

        areAllSameSku = m_coreProxy.AreAllGpuIdsSameSku(gpuIds);

        indexList << drd->fakeGpuList;
    }
    else if (strlen(drd->gpuList) != 0u)
    {
        // Check just the supplied list
        std::vector<std::string> gpuIdStrs;
        dcgmTokenizeString(drd->gpuList, ",", gpuIdStrs);

        for (size_t i = 0; i < gpuIdStrs.size(); i++)
        {
            gpuIds.push_back(strtol(gpuIdStrs[i].c_str(), NULL, 10));
        }

        areAllSameSku = m_coreProxy.AreAllGpuIdsSameSku(gpuIds);

        indexList << drd->gpuList;
    }
    else
    {
        bool foundGpus = false;
        // Check the group
        ret = m_coreProxy.AreAllTheSameSku(0, (unsigned long long)drd->groupId, &areAllSameSku);

        if (ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "Got st %d from AreAllTheSameSku()", (int)ret);
            return ret;
        }

        ret = m_coreProxy.GetGroupGpuIds(0, (unsigned long long)drd->groupId, gpuIds);
        if (ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "Got st %d from GetGroupGpuIds()", (int)ret);
            return ret;
        }

        for (unsigned int i = 0; i < gpuIds.size(); i++)
        {
            unsigned int gpuId = gpuIds[i];
            indexList << gpuId;
            if (i != gpuIds.size() - 1)
                indexList << ",";

            foundGpus = true;
        }

        if (foundGpus == false)
        {
            PRINT_DEBUG("%s", "Cannot perform diag: %s", errorString(DCGM_ST_GROUP_IS_EMPTY));
            return DCGM_ST_GROUP_IS_EMPTY;
        }
    }

    if (!areAllSameSku)
    {
        DCGM_LOG_DEBUG << "GPUs are incompatible for Validation";
        return DCGM_ST_GROUP_INCOMPATIBLE;
    }

    if ((drd->validate == DCGM_POLICY_VALID_NONE) && (strlen(drd->testNames[0]) == 0)
        && ((drd->flags & DCGM_RUN_FLAGS_TRAIN) == 0))
    {
        return DCGM_ST_OK;
    }
    else
    {
        ret = PerformNVVSExecute(&stdoutStr, &stderrStr, drd, indexList.str());
        if (ret != DCGM_ST_OK)
        {
            fmt::memory_buffer msg;
            fmt::format_to(std::back_inserter(msg), "Error when executing the diagnostic: {}\n", errorString(ret));
            fmt::format_to(std::back_inserter(msg), "Nvvs stderr:\n{}\n", stderrStr);
            // Record a system error here even though it may be overwritten with a more specific one later
            response.RecordSystemError({ msg.data(), msg.size() });

            return ret;
        }

        // FillResponseStructure will return DCGM_ST_OK if it can parse the json, passing through
        // better information than just DCGM_ST_NOT_CONFIGURED if NVVS had an error.
        ret = FillResponseStructure(stdoutStr, response, drd->groupId, ret);
        if (ret != DCGM_ST_OK && !stderrStr.empty())
        {
            DCGM_LOG_ERROR << fmt::format("Error happened during JSON parsing of NVVS output: {}", errorString(ret));
            DCGM_LOG_ERROR << fmt::format("NVVS stderr:\n{}", SanitizedString(stderrStr));
            // Do not overwrite the response system error here as there may be a more specific one already
        }
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
    unsigned int index      = DCGM_PER_GPU_TEST_COUNT_V7;
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

    return index;
}

bool DcgmDiagManager::IsMsgForThisTest(unsigned int testIndex, const std::string &msg, const std::string &gpuMsg)

{
    // Memory tests are run individually per GPU
    if (testIndex == DCGM_MEMORY_INDEX)
        return true;

    size_t pos = msg.find(gpuMsg);

    if (pos == std::string::npos)
    {
        if (msg.find("GPU ") == std::string::npos)
            return true; // If a GPU isn't specified, then assume it's a message for all GPUs
        else
            return false; // If a different GPU is specified, then it definitly isn't for us
    }

    // Make sure GPU 1 doesn't match GPU 12, etc.
    if (msg[pos + gpuMsg.size()] == '\0' || msg[pos + gpuMsg.size()] == ' ')
        return true;

    return false;
}

std::string DcgmDiagManager::JsonStringArrayToCsvString(Json::Value &array,
                                                        unsigned int testIndex,
                                                        const std::string &gpuMsg)
{
    std::stringstream message;
    bool first         = true;
    bool perGpuResults = gpuMsg.empty();
    for (Json::ArrayIndex index = 0; index < array.size(); index++)
    {
        if (perGpuResults || IsMsgForThisTest(testIndex, array[index].asString(), gpuMsg))
        {
            if (!first)
            {
                message << ", ";
            }
            message << array[index].asString();
            first = false;
        }
    }
    return message.str();
}

void DcgmDiagManager::PopulateErrorDetail(Json::Value &jsonResult, dcgmDiagErrorDetail_t &ed, double nvvsVersion)
{
    std::stringstream tmp;

    ed.code = DCGM_FR_OK;

    for (Json::ArrayIndex warnIndex = 0; warnIndex < jsonResult[NVVS_WARNINGS].size(); warnIndex++)
    {
        if (nvvsVersion >= 1.7)
        {
            if (warnIndex != 0)
            {
                tmp << ", " << jsonResult[NVVS_WARNINGS][warnIndex][NVVS_WARNING].asString();
            }
            else
            {
                tmp << jsonResult[NVVS_WARNINGS][warnIndex][NVVS_WARNING].asString();
                ed.code = static_cast<dcgmError_t>(jsonResult[NVVS_WARNINGS][warnIndex][NVVS_ERROR_ID].asInt());
            }
        }
        else
        {
            if (warnIndex != 0)
            {
                tmp << ", " << jsonResult[NVVS_WARNINGS][warnIndex].asString();
            }
            else
            {
                tmp << jsonResult[NVVS_WARNINGS][warnIndex].asString();
                // Since we've run a legacy NVVS, we don't know the error code
                ed.code = DCGM_FR_UNKNOWN;
            }
        }
    }

    if (!tmp.str().empty())
    {
        SafeCopyTo(ed.msg, tmp.str().c_str());
    }
}

void DcgmDiagManager::FillTestResult(Json::Value &test,
                                     DcgmDiagResponseWrapper &response,
                                     std::set<unsigned int> &gpuIdSet,
                                     double nvvsVersion)
{
    // It's only relevant if it has a test name, results, a status, and gpu ids
    if (test[NVVS_TEST_NAME].empty() || test[NVVS_RESULTS].empty())
        return;

    const double PER_GPU_NVVS_VERSION = 1.7;
    bool perGpuResults                = (nvvsVersion >= PER_GPU_NVVS_VERSION);

    DCGM_LOG_ERROR << test;
    for (Json::ArrayIndex resultIndex = 0; resultIndex < test[NVVS_RESULTS].size(); resultIndex++)
    {
        Json::Value &result = test[NVVS_RESULTS][resultIndex];

        if (result[NVVS_STATUS].empty() || result[NVVS_GPU_IDS].empty())
        {
            continue;
        }

        unsigned int testIndex = GetTestIndex(test[NVVS_TEST_NAME].asString());
        std::string gpuIds     = result[NVVS_GPU_IDS].asString();
        std::vector<std::string> gpuIdVec;

        dcgmTokenizeString(gpuIds, ",", gpuIdVec);
        if (testIndex == DCGM_CONTEXT_CREATE_INDEX)
        {
            testIndex = 0; // Context create is only ever run by itself, and its result is stored in index 0.
        }

        dcgmDiagResult_t ret = StringToDiagResponse(result[NVVS_STATUS].asString());

        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V7)
        {
            // Software test
            dcgmDiagErrorDetail_t ed = { { 0 }, 0 };
            PopulateErrorDetail(result, ed, nvvsVersion);
            response.AddErrorDetail(0, testIndex, test[NVVS_TEST_NAME].asString(), ed, ret);

            for (size_t gpuIdIt = 0; gpuIdIt < gpuIdVec.size() && gpuIdSet.size() < gpuIdVec.size(); gpuIdIt++)
            {
                unsigned int gpuIndex = strtol(gpuIdVec[gpuIdIt].c_str(), NULL, 10);
                response.SetGpuIndex(gpuIndex);
                gpuIdSet.insert(gpuIndex);
            }

            continue;
        }

        for (size_t gpuIdIt = 0; gpuIdIt < gpuIdVec.size(); gpuIdIt++)
        {
            char gpuMsg[128];
            unsigned int gpuIndex = strtol(gpuIdVec[gpuIdIt].c_str(), NULL, 10);

            response.SetGpuIndex(gpuIndex);
            gpuIdSet.insert(gpuIndex);
            response.SetPerGpuResponseState(testIndex, ret, gpuIndex);

            if (perGpuResults)
            {
                gpuMsg[0] = '\0';
            }
            else
            {
                snprintf(gpuMsg, sizeof(gpuMsg), "GPU %u", gpuIndex);
            }

            if (result[NVVS_WARNINGS].empty() == false)
            {
                dcgmDiagErrorDetail_t ed = { { 0 }, 0 };
                PopulateErrorDetail(result, ed, nvvsVersion);
                response.AddErrorDetail(gpuIndex, testIndex, "", ed, ret);
            }

            if (result[NVVS_INFO].empty() == false)
            {
                std::string info = JsonStringArrayToCsvString(result[NVVS_INFO], testIndex, gpuMsg);
                response.AddPerGpuMessage(testIndex, info, gpuIndex, false);
            }
        }
    }
}

dcgmReturn_t DcgmDiagManager::ValidateNvvsOutput(const std::string &output,
                                                 size_t &jsonStart,
                                                 Json::Value &jv,
                                                 DcgmDiagResponseWrapper &response)
{
    Json::CharReaderBuilder rBuilder;
    std::stringstream jsonStream(output);

    if (output.empty() || !Json::parseFromStream(rBuilder, jsonStream, &jv, nullptr))
    {
        // logging.c prepends error messages and we can't change it right now. Attempt to move to JSON beginning
        // and parse.
        bool failed = true;
        jsonStart   = output.find('{');

        if (jsonStart != std::string::npos)
        {
            std::string justJson = output.substr(jsonStart);

            std::string jsonError;
            jsonStream.str(justJson);
            if (!justJson.empty() && Json::parseFromStream(rBuilder, jsonStream, &jv, &jsonError))
            {
                // We recovered. log the warning and move on with life
                DCGM_LOG_DEBUG << "Found non JSON data in the NVVS stdout: " << output.substr(0, jsonStart);
                failed = false; // flag to know we can proceed
            }
            else if (!jsonError.empty())
            {
                DCGM_LOG_ERROR << "JSON parse error: " << jsonError;
            }
        }

        if (failed)
        {
            DCGM_LOG_ERROR << "Failed to parse NVVS output: " << SanitizedString(output);
            response.RecordSystemError(fmt::format("Couldn't parse json: '{}'", output));
            return DCGM_ST_DIAG_BAD_JSON;
        }
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagManager::FillResponseStructure(const std::string &output,
                                                    DcgmDiagResponseWrapper &response,
                                                    int groupId,
                                                    dcgmReturn_t oldRet)
{
    unsigned int numGpus = m_coreProxy.GetGpuCount(groupId);
    Json::Value jValue;
    std::set<unsigned int> gpuIdSet;
    size_t jsonStart;

    try
    {
        if (dcgmReturn_t ret = ValidateNvvsOutput(output, jsonStart, jValue, response); ret != DCGM_ST_OK)
        {
            return oldRet == DCGM_ST_OK ? ret : oldRet;
        }

        response.InitializeResponseStruct(numGpus);

        if (jValue[NVVS_NAME].empty())
        {
            fmt::basic_memory_buffer<char, 1024> buf;
            fmt::format_to(std::back_inserter(buf),
                           "Received json that doesn't include required data {}. Full output: '{}'",
                           NVVS_NAME,
                           jsonStart == 0 ? output : output.substr(jsonStart));

            auto msg = fmt::to_string(buf);
            DCGM_LOG_ERROR << SanitizedString(msg);
            response.RecordSystemError(msg);

            // If oldRet was OK, change return code to indicate JSON error
            if (oldRet == DCGM_ST_OK)
            {
                return DCGM_ST_DIAG_BAD_JSON;
            }
        }
        else if (!jValue[NVVS_NAME][NVVS_RUNTIME_ERROR].empty())
        {
            response.RecordSystemError(jValue[NVVS_NAME][NVVS_RUNTIME_ERROR].asString());
            // If oldRet was OK, change return code to indicate NVVS error
            if (oldRet == DCGM_ST_OK)
            {
                return DCGM_ST_NVVS_ERROR;
            }
        }
        else
        {
            // Good output
            if (!jValue[NVVS_NAME][NVVS_TRAINING_MSG].empty())
            {
                return response.RecordTrainingMessage(jValue[NVVS_NAME][NVVS_TRAINING_MSG].asString());
            }

            if (!jValue[NVVS_NAME][NVVS_HEADERS].empty())
            {
                // Get nvvs version
                double nvvsVersion = 0.0;
                try
                {
                    nvvsVersion = std::stod(jValue[NVVS_NAME][NVVS_VERSION_STR].asString());
                }
                catch (...)
                {
                    DCGM_LOG_ERROR << "Failed to parse NVVS version string: "
                                   << jValue[NVVS_NAME][NVVS_VERSION_STR].asString();
                }
                if (nvvsVersion < 0.1 || nvvsVersion > 10)
                {
                    // Default to version 1.3
                    nvvsVersion = 1.3;
                }

                for (Json::ArrayIndex i = 0; i < jValue[NVVS_NAME][NVVS_HEADERS].size(); i++)
                {
                    Json::Value &category = jValue[NVVS_NAME][NVVS_HEADERS][i];

                    if (category[NVVS_TESTS].empty())
                    {
                        break;
                    }

                    for (Json::ArrayIndex testIndex = 0; testIndex < category[NVVS_TESTS].size(); testIndex++)
                    {
                        FillTestResult(category[NVVS_TESTS][testIndex], response, gpuIdSet, nvvsVersion);
                    }
                }

                response.SetGpuCount(gpuIdSet.size());
            }
        }
    }
    catch (Json::Exception const &err)
    {
        DCGM_LOG_ERROR << "Could not parse JSON received from NVVS: " << err.what();
        response.RecordSystemError(fmt::format("Couldn't parse json: '{}'", output));
        return DCGM_ST_DIAG_BAD_JSON;
    }

    // Do not mask old errors if there are any
    return oldRet;
}

dcgmDiagResult_t DcgmDiagManager::StringToDiagResponse(std::string_view result)
{
    return result == "PASS"   ? DCGM_DIAG_RESULT_PASS
           : result == "SKIP" ? DCGM_DIAG_RESULT_SKIP
           : result == "WARN" ? DCGM_DIAG_RESULT_WARN
                              : DCGM_DIAG_RESULT_FAIL;
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

    PRINT_DEBUG("%d %p %d", "performing action %d on group %p with validation %d", action, drd->groupId, drd->validate);

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

    if ((drd->validate != DCGM_POLICY_VALID_NONE) || (strlen(drd->testNames[0]) > 0)
        || (drd->flags & DCGM_RUN_FLAGS_TRAIN) == DCGM_RUN_FLAGS_TRAIN)
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
        PRINT_WARNING("%d", "Previous instance of nvvs is still running. PID: %d", m_nvvsPID);
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
        PRINT_ERROR("%s %d", "epoll_ctl failed. errno %d", errno);
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
