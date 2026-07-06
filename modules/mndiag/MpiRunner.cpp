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

#include "MpiRunner.h"
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <algorithm>
#include <fcntl.h>
#include <filesystem>
#include <fmt/format.h>

#include <functional>
#include <pwd.h>
#include <signal.h>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

MpiRunner::MpiRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid)
    : m_coreProxy(coreProxy)
{
    std::string userName;
    if (auto const *pw = getpwuid(effectiveUid); pw != nullptr)
    {
        userName = pw->pw_name;
    }
    m_userInfo = std::make_pair(std::move(userName), effectiveUid);
}

MpiRunner::~MpiRunner()
{
    StopMpiProcess();
    StopRedirectThreads();
    CleanupTempFiles();
}

dcgmReturn_t MpiRunner::LaunchMpiProcess()
{
    if (m_lastCommand.empty())
    {
        log_error("Cannot launch MPI process: No command has been constructed");
        return DCGM_ST_BADPARAM;
    }

    if (!m_outputCallback)
    {
        SetOutputCallback([this](int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) {
            return this->DefaultOutputCallback(fd, responseStruct, nodeInfo);
        });
    }

    dcgmReturn_t result = PrepareOutputFiles();
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    log_debug("Launching: {} ", GetLastCommand());

    std::vector<char const *> args;
    for (auto const &arg : m_lastCommand)
    {
        args.push_back(arg.c_str());
    }

    std::string executablePath = GetMpiBinPath();

    dcgmChildProcessParams_v1 params {};
    params.version    = dcgmChildProcessParams_version1;
    params.executable = executablePath.c_str();
    params.args       = args.data();
    params.numArgs    = args.size();

    // Only set the user name if it's different from the current effective user id
    if (m_userInfo.has_value() && !m_userInfo->first.empty() && m_userInfo->second != geteuid())
    {
        params.userName = m_userInfo->first.c_str();
    }

    result = m_coreProxy.ChildProcessSpawn(params, m_childProcessHandle, m_pid);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to create MPI process instance");
        return result;
    }

    log_debug("Successfully launched MPI process with PID: {}", m_pid);

    result = RedirectMpiOutput();
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to redirect MPI output, stopping MPI process");
        StopMpiProcess();
        return result;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t MpiRunner::DefaultOutputCallback(int fd, void * /* responseStruct */, nodeInfoMap_t const & /* nodeInfo */)
{
    int dupFd = dup(fd);
    if (dupFd < 0)
    {
        log_error("DefaultOutputCallback: dup failed: {}", strerror(errno));
        return DCGM_ST_FILE_IO_ERROR;
    }

    auto fileCloser = [](FILE *f) {
        fclose(f);
    };
    std::unique_ptr<FILE, decltype(fileCloser)> fp(fdopen(dupFd, "r"), fileCloser);
    if (!fp)
    {
        close(dupFd);
        log_error("DefaultOutputCallback: fdopen failed: {}", strerror(errno));
        return DCGM_ST_FILE_IO_ERROR;
    }

    char buf[4096];
    while (fgets(buf, sizeof(buf), fp.get()))
    {
        log_debug("MPI stdout: {}", buf);
    }

    return DCGM_ST_OK;
}

/**
 * Gets the exit code of the MPI process if it has exited
 *
 * @returns the exit code if the process has exited, std::nullopt if still running or not started
 */
[[nodiscard]] std::optional<int> MpiRunner::GetMpiProcessExitCode()
{
    if (m_childProcessHandle == INVALID_CHILD_PROCESS_HANDLE || m_pid == -1)
    {
        // No process running
        return std::nullopt;
    }

    dcgmChildProcessStatus_v1 status {};
    status.version      = dcgmChildProcessStatus_version1;
    dcgmReturn_t result = m_coreProxy.ChildProcessGetStatus(m_childProcessHandle, status);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to get MPI process status: {}", result);
        return std::nullopt;
    }

    if (status.running == 1)
    {
        // Process is still running
        return std::nullopt;
    }

    return status.exitCode;
}

/**
 * Gets the PID of the running MPI process
 *
 * @return pid_t The process ID if process is running, or std::nullopt if not started or already exited
 */
[[nodiscard]] std::optional<pid_t> MpiRunner::GetMpiProcessPid() const
{
    if (m_childProcessHandle == INVALID_CHILD_PROCESS_HANDLE || m_pid == -1)
    {
        // No process running
        return std::nullopt;
    }

    dcgmChildProcessStatus_v1 status {};
    status.version      = dcgmChildProcessStatus_version1;
    dcgmReturn_t result = m_coreProxy.ChildProcessGetStatus(m_childProcessHandle, status);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to get MPI process status: {}", result);
        return std::nullopt;
    }

    if (status.running == 1)
    {
        return m_pid;
    }

    return std::nullopt;
}

dcgmReturn_t MpiRunner::StopMpiProcess()
{
    // If there's no process or it's not running, nothing to stop
    if (m_childProcessHandle == INVALID_CHILD_PROCESS_HANDLE || m_pid == -1)
    {
        // No process to stop
        return DCGM_ST_OK;
    }

    // Try to stop the process gracefully first (force=false)
    dcgmReturn_t result = m_coreProxy.ChildProcessDestroy(m_childProcessHandle);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to stop MPI process: {}", result);
    }

    // Process was stopped, clean up our state, process handle is invalidated by destroy
    m_pid = -1;

    return result;
}

dcgmReturn_t MpiRunner::Wait(int timeoutSec)
{
    // If there's no process nothing to wait for
    if (m_childProcessHandle == INVALID_CHILD_PROCESS_HANDLE || m_pid == -1)
    {
        log_debug("No MPI process to wait for");
        return DCGM_ST_UNINITIALIZED;
    }

    log_debug("Waiting for MPI process {} to complete (timeout: {}s)", m_pid, timeoutSec);

    // Call through to the core proxy
    dcgmReturn_t result = m_coreProxy.ChildProcessWait(m_childProcessHandle, timeoutSec);

    if (result != DCGM_ST_OK)
    {
        log_error("Error waiting for MPI process {}: {}", m_pid, errorString(result));
        return result;
    }

    // Check if the process is still running after wait
    dcgmChildProcessStatus_v1 status = {};
    status.version                   = dcgmChildProcessStatus_version1;

    result = m_coreProxy.ChildProcessGetStatus(m_childProcessHandle, status);
    if (result != DCGM_ST_OK)
    {
        log_error("Error getting status for MPI process {}: {}", m_pid, errorString(result));
        return result;
    }

    if (status.receivedSignal)
    {
        log_warning("MPI process {} received signal {}", m_pid, status.receivedSignalNumber);
        return DCGM_ST_CHILD_SIGNAL_RECEIVED;
    }

    if (status.running)
    {
        log_warning("Timeout waiting for MPI process {} to complete after {}s", m_pid, timeoutSec);
        return DCGM_ST_CHILD_NOT_KILLED; // Process still running after timeout
    }

    log_info("MPI process {} completed with exit code {}", m_pid, status.exitCode);
    return DCGM_ST_OK;
}

dcgmReturn_t MpiRunner::SetMpiOutputPipeFd(bool isStderr)
{
    auto &pipeFd = isStderr ? m_stderrFd : m_stdoutFd;
    if (pipeFd.Get() >= 0)
    {
        return DCGM_ST_OK;
    }

    log_debug("Getting file handle for {} stream", isStderr ? "stderr" : "stdout");
    int fd = -1;
    dcgmReturn_t result { DCGM_ST_OK };
    if (isStderr)
    {
        result = m_coreProxy.ChildProcessGetStdErrHandle(m_childProcessHandle, fd);
    }
    else
    {
        result = m_coreProxy.ChildProcessGetStdOutHandle(m_childProcessHandle, fd);
    }
    if (result != DCGM_ST_OK || fd < 0)
    {
        log_error("Failed to get file descriptor for {} stream: {}", isStderr ? "stderr" : "stdout", result);
        return result;
    }
    constexpr int DESIRED_PIPE_BUF_SIZE = 1048576;
    int actual                          = fcntl(fd, F_SETPIPE_SZ, DESIRED_PIPE_BUF_SIZE);
    if (actual < 0)
    {
        log_warning("Failed to increase pipe buffer size: {}", strerror(errno));
    }
    else
    {
        log_debug("Pipe buffer size set to {} bytes for {} stream", actual, isStderr ? "stderr" : "stdout");
    }

    pipeFd = DcgmNs::Utils::FileHandle(fd);
    return DCGM_ST_OK;
}

void MpiRunner::CleanupTempFiles()
{
    if (!m_ownsTempFiles)
    {
        return;
    }

    if (!m_stdoutFilePath.empty())
    {
        std::error_code ec;
        std::filesystem::remove(m_stdoutFilePath, ec);
        if (ec)
        {
            log_warning("Failed to remove temp file '{}': {}", m_stdoutFilePath, ec.message());
        }
        m_stdoutFilePath.clear();
    }
    if (!m_stderrFilePath.empty())
    {
        std::error_code ec;
        std::filesystem::remove(m_stderrFilePath, ec);
        if (ec)
        {
            log_warning("Failed to remove temp file '{}': {}", m_stderrFilePath, ec.message());
        }
        m_stderrFilePath.clear();
    }
    m_ownsTempFiles = false;
}

dcgmReturn_t MpiRunner::PrepareOutputFiles()
{
    if (m_logFileNames.has_value())
    {
        m_stdoutFilePath = m_logFileNames->first;
        m_stderrFilePath = m_logFileNames->second;
        m_ownsTempFiles  = false;

        for (auto const &path : { m_stdoutFilePath, m_stderrFilePath })
        {
            std::filesystem::path dirPath = std::filesystem::path(path).parent_path();
            if (!dirPath.empty())
            {
                try
                {
                    std::filesystem::create_directories(dirPath);
                }
                catch (std::filesystem::filesystem_error const &e)
                {
                    log_error("Failed to create directory '{}': {}", dirPath.string(), e.what());
                    return DCGM_ST_FILE_IO_ERROR;
                }
            }
        }

        int stdoutFd = open(m_stdoutFilePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (stdoutFd < 0)
        {
            log_error("Failed to open '{}': {}", m_stdoutFilePath, strerror(errno));
            return DCGM_ST_FILE_IO_ERROR;
        }
        m_stdoutFileFd = DcgmNs::Utils::FileHandle(stdoutFd);

        int stderrFd = open(m_stderrFilePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (stderrFd < 0)
        {
            log_error("Failed to open '{}': {}", m_stderrFilePath, strerror(errno));
            m_stdoutFileFd = {};
            return DCGM_ST_FILE_IO_ERROR;
        }
        m_stderrFileFd = DcgmNs::Utils::FileHandle(stderrFd);
    }
    else
    {
        char stdoutTmpl[] = "/tmp/dcgm_mpirunner_stdout_XXXXXX";
        int stdoutFd      = mkstemp(stdoutTmpl);
        if (stdoutFd < 0)
        {
            log_error("Failed to create temp file: {}", strerror(errno));
            return DCGM_ST_FILE_IO_ERROR;
        }
        m_stdoutFilePath = stdoutTmpl;
        m_stdoutFileFd   = DcgmNs::Utils::FileHandle(stdoutFd);

        char stderrTmpl[] = "/tmp/dcgm_mpirunner_stderr_XXXXXX";
        int stderrFd      = mkstemp(stderrTmpl);
        if (stderrFd < 0)
        {
            log_error("Failed to create temp file: {}", strerror(errno));
            m_stdoutFileFd = {};
            unlink(stdoutTmpl);
            m_stdoutFilePath.clear();
            return DCGM_ST_FILE_IO_ERROR;
        }
        m_stderrFilePath = stderrTmpl;
        m_stderrFileFd   = DcgmNs::Utils::FileHandle(stderrFd);
        m_ownsTempFiles  = true;
    }

    log_debug("Output files prepared: {} and {}", m_stdoutFilePath, m_stderrFilePath);
    return DCGM_ST_OK;
}

dcgmReturn_t MpiRunner::RedirectMpiOutput()
{
    dcgmReturn_t result = SetMpiOutputPipeFd(false);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = SetMpiOutputPipeFd(true);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    return StartRedirectThreads();
}

void MpiRunner::StopRedirectThreads()
{
    // Request stop so the threads know the upcoming EBADF is intentional.
    m_stdoutRedirectThread.request_stop();
    m_stderrRedirectThread.request_stop();

    // Close the pipe read-ends to unblock splice().  If the child already
    // exited, splice() has already returned 0 (EOF) and this is a no-op.
    m_stdoutFd = {};
    m_stderrFd = {};

    // jthread destructor would join automatically, but we join explicitly here
    // so callers like PopulateResponse can rely on output being fully flushed.
    if (m_stdoutRedirectThread.joinable())
    {
        m_stdoutRedirectThread.join();
    }
    if (m_stderrRedirectThread.joinable())
    {
        m_stderrRedirectThread.join();
    }
}

dcgmReturn_t MpiRunner::StartRedirectThreads()
{
    auto splicePipeToFile = [](std::stop_token stopToken, int pipeFd, int fileFd) {
        log_debug("Starting splice redirection thread");
        constexpr size_t SPLICE_CHUNK_SIZE = 1048576;
        while (true)
        {
            ssize_t bytes = splice(pipeFd, nullptr, fileFd, nullptr, SPLICE_CHUNK_SIZE, SPLICE_F_MOVE);
            if (bytes > 0)
            {
                continue;
            }
            if (bytes == 0)
            {
                log_debug("Pipe writer closed, splice redirection complete");
                break;
            }
            if (errno == EINTR)
            {
                continue;
            }
            if (errno == EBADF)
            {
                if (stopToken.stop_requested())
                {
                    log_debug("Pipe fd closed during intentional shutdown");
                }
                else
                {
                    log_warning("Pipe fd closed unexpectedly during splice redirection");
                }
                break;
            }
            log_error("splice error: {}", strerror(errno));
            break;
        }
        log_debug("Splice redirection thread completed");
    };

    log_debug("Starting splice redirection threads");
    m_stdoutRedirectThread = std::jthread(splicePipeToFile, m_stdoutFd.Get(), m_stdoutFileFd.Get());
    m_stderrRedirectThread = std::jthread(splicePipeToFile, m_stderrFd.Get(), m_stderrFileFd.Get());

    return DCGM_ST_OK;
}

dcgmReturn_t MpiRunner::PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo)
{
    StopRedirectThreads();

    dcgmReturn_t result { DCGM_ST_OK };

    if (m_stdoutFileFd.Get() >= 0)
    {
        if (lseek(m_stdoutFileFd.Get(), 0, SEEK_SET) == -1)
        {
            log_error("Failed to lseek stdout file fd: {}", strerror(errno));
            return DCGM_ST_FILE_IO_ERROR;
        }
        result = m_outputCallback(m_stdoutFileFd.Get(), responseStruct, nodeInfo);
    }

    return result;
}
