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

#ifndef MPI_RUNNER_H
#define MPI_RUNNER_H

#include <fmt/format.h>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
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
     */
    MpiRunner(DcgmCoreProxyBase &coreProxy);

    /**
     * @brief Destructor
     */
    virtual ~MpiRunner();

    /**
     * @brief Set the output callback function
     *
     * @param callback The function to process output from the MPI process
     */
    void SetOutputCallback(std::function<dcgmReturn_t(std::istream &, void *, nodeInfoMap_t const &)> callback)
    {
        m_outputCallback = callback;
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
     * @param dataStream The stream to parse
     * @param responseStruct Pointer to a structure that can be updated with parsed data (not used in default
     * implementation)
     */
    dcgmReturn_t DefaultOutputCallback(std::istream &dataStream, void *responseStruct, nodeInfoMap_t const &nodeInfo);

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
     * @brief Set the user name and uid to run the MPI process as
     *
     * @param userInfo The user name and uid to run the MPI process as, or nullopt to run as the user running
     * nv-hostengine
     */
    void SetUserInfo(std::pair<std::string, uid_t> userInfo)
    {
        m_userInfo = userInfo;
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
     * @brief Redirect the MPI process output to files or memory streams
     *
     * Redirects the MPI process output to files or memory streams based on the log file names set via SetLogFileNames.
     * If no log file names are set, redirects to memory streams.
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t RedirectMpiOutput();

    /**
     * @brief Start threads to redirect output from MPI pipes to the target streams
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t StartRedirectThreads();

    /**
     * @brief Stop redirecting the MPI process output
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
     * @brief Set the MPI output file stream
     *
     * @param isStderr True if stderr, false if stdout
     * @param filename The filename to redirect to
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t SetDiskOutputFileStream(bool isStderr, std::string const &filename);

    std::optional<std::pair<std::string, uid_t>> m_userInfo;
    std::vector<std::string> m_lastCommand;
    std::function<dcgmReturn_t(std::istream &, void *, nodeInfoMap_t const &)> m_outputCallback;
    int m_pid { -1 };
    ChildProcessHandle_t m_childProcessHandle { INVALID_CHILD_PROCESS_HANDLE };
    DcgmCoreProxyBase &m_coreProxy; // Holding a reference to the core proxy which is created by the MnDiagManager

    // File handles for stdout and stderr from the child process's pipe
    std::optional<int> m_stdoutFd;
    std::optional<int> m_stderrFd;

    // File streams for capturing output to disk
    std::ofstream m_stdoutDiskFileStream;
    std::ofstream m_stderrDiskFileStream;
    std::optional<std::pair<std::string, std::string>> m_logFileNames;

    // Memory streams for capturing output when not redirecting to disk
    std::ostringstream m_stdoutMemoryStream;
    std::ostringstream m_stderrMemoryStream;

    // Pointers to the active output streams (either disk or memory)
    std::ostream *m_stdoutStream { nullptr };
    std::ostream *m_stderrStream { nullptr };

    std::atomic<bool> m_stopRedirectFlag { false };
    std::thread m_stdoutRedirectThread;
    std::thread m_stderrRedirectThread;
};

#endif // MPI_RUNNER_H