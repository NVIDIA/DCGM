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

#ifndef MPI_RUNNER_H
#define MPI_RUNNER_H

#include <DcgmUtilities.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <thread>
#include <vector>

#include "DcgmCoreProxyBase.h"
#include "MnDiagCommon.h"
#include "dcgm_mndiag_structs.hpp"

/**
 * @brief Base class for managing MPI processes
 */
class MpiRunner
{
public:
    /**
     * @brief Constructor
     *
     * @param coreProxy The core proxy to use for the ChildProcess management
     * @param effectiveUid The effective UID of the caller; used to determine the user to run the MPI process as
     */
    MpiRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid);

    /**
     * @brief Destructor
     */
    virtual ~MpiRunner();

    /**
     * @brief Set the output callback function
     *
     * @param callback The function to process output from the MPI process
     */
    void SetOutputCallback(std::function<dcgmReturn_t(int, void *, nodeInfoMap_t const &)> callback)
    {
        m_outputCallback = std::move(callback);
    }

    /**
     * @brief Construct an MPI command from input parameters and store it internally
     *
     * @param params Pointer to parameters, type depends on the derived class
     */
    virtual void ConstructMpiCommand(void const *params) = 0;

    /**
     * @brief Launch the MPI process with the stored command
     *
     * Uses the command previously constructed via ConstructMpiCommand.
     * Uses the output callback set via SetOutputCallback or the default callback.
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t LaunchMpiProcess();

    /**
     * @brief Default output callback handler
     *
     * @param fd File descriptor to read MPI process output from
     * @param responseStruct Pointer to a structure that can be updated with parsed data (not used in default
     * implementation)
     * @param nodeInfo The node info map (not used in default implementation)
     */
    dcgmReturn_t DefaultOutputCallback(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo);

    /**
     * @brief Get the exit code of the MPI process if it has exited
     *
     * @return std::optional<int> The exit code if the process has exited, std::nullopt if still running or not started
     */
    [[nodiscard]] std::optional<int> GetMpiProcessExitCode();

    /**
     * @brief Get the PID of the running MPI process
     *
     * @return pid_t The process ID, or std::nullopt if not running
     */
    [[nodiscard]] std::optional<pid_t> GetMpiProcessPid() const;

    /**
     * @brief Stop the MPI process if it's running
     *
     * Attempts to stop the process gracefully with SIGTERM first,
     * then escalates to SIGKILL if needed.
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t StopMpiProcess();

    /**
     * @brief Wait for the MPI process to complete
     *
     * This method blocks until the MPI process has completed execution.
     * If no process is running, the method returns immediately.
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, or error code otherwise
     */
    dcgmReturn_t Wait(int timeoutSec = -1);

    /**
     * @brief Get the last command that was used to launch the MPI process
     *
     * @return std::string The last command as a space-delimited string for logging
     */
    std::string GetLastCommand() const
    {
        return m_lastCommand.empty() ? "" : fmt::format("{} {}", GetMpiBinPath(), fmt::join(m_lastCommand, " "));
    }

    /**
     * @brief Set the log file names
     *
     * @param logFileNames The log file names for stdout and stderr
     */
    void SetLogFileNames(std::pair<std::string, std::string> const &logFileNames)
    {
        m_logFileNames = logFileNames;
    }

    /**
     * @brief Populate the response structure with the MPI process output
     *
     * @param responseStruct The response structure to populate
     * @param nodeInfo The node info map used to populate the response structure
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo);

protected:
    /**
     * @brief Get the path to the mpirun binary
     *
     * @return std::string The path to the mpirun binary, defaults to "/usr/bin/mpirun"
     */
    virtual std::string GetMpiBinPath() const
    {
        return std::string(MnDiagConstants::DEFAULT_MPIRUN_PATH);
    }

    /**
     * @brief Open or create the output files (user-specified or temp).
     *
     * Pure filesystem work with no child-process dependency, so it is
     * called before ChildProcessSpawn.  Failures here (bad path,
     * permissions, disk full) abort the launch without ever spawning
     * a child process.
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t PrepareOutputFiles();

    /**
     * @brief Wire up pipe fds from the child process and launch splice threads
     *
     * Must be called after ChildProcessSpawn since it retrieves pipe fds
     * from the child process handle.
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t RedirectMpiOutput();

    /**
     * @brief Start threads to splice output from MPI pipes to output files
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t StartRedirectThreads();

    /**
     * @brief Stop redirect threads and close pipe file descriptors.
     *
     * Closing the pipe read-ends causes splice() to return EBADF, which
     * unblocks the threads regardless of whether the child is still alive.
     * The stop token lets the threads distinguish an intentional teardown
     * from an unexpected fd closure.
     */
    void StopRedirectThreads();

    /**
     * @brief Set the MPI output pipe file descriptor
     *
     * @param isStderr True if stderr, false if stdout
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t SetMpiOutputPipeFd(bool isStderr);

    /**
     * @brief Clean up temporary files created for output redirection
     */
    void CleanupTempFiles();

    std::optional<std::pair<std::string, uid_t>> m_userInfo;
    std::vector<std::string> m_lastCommand;
    std::function<dcgmReturn_t(int, void *, nodeInfoMap_t const &)> m_outputCallback;
    int m_pid { -1 };
    ChildProcessHandle_t m_childProcessHandle { INVALID_CHILD_PROCESS_HANDLE };
    DcgmCoreProxyBase &m_coreProxy;

    DcgmNs::Utils::FileHandle m_stdoutFd;
    DcgmNs::Utils::FileHandle m_stderrFd;

    DcgmNs::Utils::FileHandle m_stdoutFileFd;
    DcgmNs::Utils::FileHandle m_stderrFileFd;
    std::string m_stdoutFilePath;
    std::string m_stderrFilePath;
    std::optional<std::pair<std::string, std::string>> m_logFileNames;
    bool m_ownsTempFiles { false };

    std::jthread m_stdoutRedirectThread;
    std::jthread m_stderrRedirectThread;
};

#endif // MPI_RUNNER_H
