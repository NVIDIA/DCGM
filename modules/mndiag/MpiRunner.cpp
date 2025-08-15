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

#include "MpiRunner.h"
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <algorithm>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <functional>
#include <signal.h>
#include <string>
#include <string_view>
#include <vector>

MpiRunner::MpiRunner(DcgmCoreProxyBase &coreProxy)
    : m_coreProxy(coreProxy)
{}

MpiRunner::~MpiRunner()
{
    // First stop the redirection (which will close streams and join threads)
    StopRedirectThreads();

    // Then ensure we clean up any running process
    StopMpiProcess();
}

dcgmReturn_t MpiRunner::LaunchMpiProcess()
{
    // Check if we have a valid command to execute
    if (m_lastCommand.empty())
    {
        log_error("Cannot launch MPI process: No command has been constructed");
        return DCGM_ST_BADPARAM;
    }

    // If no output callback is set, use the default
    if (!m_outputCallback)
    {
        SetOutputCallback([this](std::istream &dataStream, void *responseStruct, nodeInfoMap_t const &nodeInfo) {
            return this->DefaultOutputCallback(dataStream, responseStruct, nodeInfo);
        });
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

    if (m_userInfo.has_value() && !m_userInfo->first.empty())
    {
        params.userName = m_userInfo->first.c_str();
    }

    dcgmReturn_t result = m_coreProxy.ChildProcessSpawn(params, m_childProcessHandle, m_pid);

    if (result != DCGM_ST_OK)
    {
        log_error("Failed to create MPI process instance");
        return result;
    }

    log_debug("Successfully launched MPI process with PID: {}", m_pid);

    // Redirect MPI output to either files or memory streams
    result = RedirectMpiOutput();
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to redirect MPI output. Stopping MPI process.");
        return result;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t MpiRunner::DefaultOutputCallback(std::istream &dataStream,
                                              void * /* responseStruct */,
                                              nodeInfoMap_t const & /* nodeInfo */)
{
    std::string output;
    dataStream >> output;
    log_debug("MPI stdout: {}", output);
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
    std::optional<int> &pipeFd = isStderr ? m_stderrFd : m_stdoutFd;
    if (pipeFd.has_value())
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
    // Set non-blocking mode for reading
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1)
    {
        log_error("Failed to set pipe file descriptor to non-blocking mode: {}", strerror(errno));
        return DCGM_ST_FILE_IO_ERROR;
    }
    pipeFd = fd;
    return DCGM_ST_OK;
}

dcgmReturn_t MpiRunner::SetDiskOutputFileStream(bool isStderr, std::string const &filename)
{
    std::ofstream &stream = isStderr ? m_stderrDiskFileStream : m_stdoutDiskFileStream;
    if (stream.is_open())
    {
        return DCGM_ST_OK;
    }
    // Create directory if it doesn't exist
    std::filesystem::path filePath(filename);
    std::filesystem::path dirPath = filePath.parent_path();
    if (!dirPath.empty())
    {
        try
        {
            std::filesystem::create_directories(dirPath);
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            log_error("Failed to create directory '{}': {}", dirPath.string(), e.what());
            return DCGM_ST_FILE_IO_ERROR;
        }
    }
    stream.open(filename, std::ios::out | std::ios::trunc);
    if (!stream.is_open())
    {
        log_error("Failed to open file: {}", filename);
        return DCGM_ST_FILE_IO_ERROR;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MpiRunner::RedirectMpiOutput()
{
    // Set up file descriptors for stdout and stderr
    dcgmReturn_t result = SetMpiOutputPipeFd(false); // stdout
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = SetMpiOutputPipeFd(true); // stderr
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Set up output streams based on whether log file names are provided
    if (m_logFileNames.has_value())
    {
        // Redirect to files
        result = SetDiskOutputFileStream(false, m_logFileNames.value().first);
        if (result != DCGM_ST_OK)
        {
            log_error("Failed to redirect stdout to file: {}", m_logFileNames.value().first);
            return result;
        }

        result = SetDiskOutputFileStream(true, m_logFileNames.value().second);
        if (result != DCGM_ST_OK)
        {
            log_error("Failed to redirect stderr to file: {}", m_logFileNames.value().second);
            return result;
        }

        m_stdoutStream = &m_stdoutDiskFileStream;
        m_stderrStream = &m_stderrDiskFileStream;

        log_debug(
            "Redirecting MPI output to files: {} and {}", m_logFileNames.value().first, m_logFileNames.value().second);
    }
    else
    {
        // Redirect to memory streams
        m_stdoutMemoryStream.str("");
        m_stderrMemoryStream.str("");

        m_stdoutStream = &m_stdoutMemoryStream;
        m_stderrStream = &m_stderrMemoryStream;

        log_debug("Redirecting MPI output to memory streams");
    }

    return StartRedirectThreads();
}

void MpiRunner::StopRedirectThreads()
{
    // Signal threads to stop
    m_stopRedirectFlag.store(true, std::memory_order_relaxed);

    // Wait for threads to complete
    if (m_stdoutRedirectThread.joinable())
    {
        m_stdoutRedirectThread.join();
    }
    if (m_stderrRedirectThread.joinable())
    {
        m_stderrRedirectThread.join();
    }

    // Close pipe file descriptors
    if (m_stdoutFd.has_value())
    {
        close(*m_stdoutFd);
        m_stdoutFd = std::nullopt;
    }

    if (m_stderrFd.has_value())
    {
        close(*m_stderrFd);
        m_stderrFd = std::nullopt;
    }

    // Close file streams if they're open
    if (m_stdoutDiskFileStream.is_open())
    {
        m_stdoutDiskFileStream.close();
    }

    if (m_stderrDiskFileStream.is_open())
    {
        m_stderrDiskFileStream.close();
    }

    // Reset stream pointers
    m_stdoutStream = nullptr;
    m_stderrStream = nullptr;
}

dcgmReturn_t MpiRunner::StartRedirectThreads()
{
    // Reset the stop flag
    m_stopRedirectFlag.store(false, std::memory_order_relaxed);

    // Create two threads to read from the pipe and write to the appropriate streams
    auto RedirectPipeToStreamFunc
        = [](int pipeFd, std::ostream &stream, std::atomic<bool> &stopFlag, bool addTimestamps = false) {
              log_debug("Starting pipe redirection thread");
              std::array<char, 4096> buffer;
              std::string lineBuffer; // Buffer for partial lines

              // Set a reasonable upper limit for line buffer to prevent unbounded growth
              constexpr size_t MAX_LINE_BUFFER_SIZE = 2 * 1024 * 1024; // 2MB limit

              // Helper lambda to read from pipe to stream while a condition is met
              auto redirectPipeToStreamWhile
                  = [&buffer, &stream, &pipeFd, &lineBuffer, addTimestamps](std::function<bool()> shouldContinue) {
                        auto helperAddTimestamps = [](std::string_view line) -> std::string {
                            // Add timestamp to the line
                            auto now = std::chrono::system_clock::now();
                            auto ms
                                = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
                            auto time_t_val = std::chrono::system_clock::to_time_t(now);
                            std::tm tm_info = *std::localtime(&time_t_val);
                            return fmt::format("[{:%Y-%m-%d %H:%M:%S}.{:03d}] {}\n", tm_info, ms.count(), line);
                        };
                        while (shouldContinue())
                        {
                            ssize_t bytesRead = read(pipeFd, buffer.data(), buffer.size());
                            if (bytesRead > 0)
                            {
                                if (addTimestamps)
                                {
                                    // Process data line by line with timestamps
                                    std::string_view data(buffer.data(), bytesRead);
                                    // Check if adding this data would exceed the buffer limit
                                    if (lineBuffer.size() + data.size() > MAX_LINE_BUFFER_SIZE)
                                    {
                                        // Force output the current buffer content as a truncated line
                                        if (!lineBuffer.empty())
                                        {
                                            std::string truncatedLine = lineBuffer + " [TRUNCATED - line too long]";
                                            stream << helperAddTimestamps(truncatedLine);
                                            lineBuffer.clear();
                                        }

                                        // If even the new data alone exceeds the limit, truncate it
                                        if (data.size() > MAX_LINE_BUFFER_SIZE)
                                        {
                                            std::string truncatedData(data.substr(0, MAX_LINE_BUFFER_SIZE - 50));
                                            truncatedData += " [TRUNCATED - data too large]";
                                            stream << helperAddTimestamps(truncatedData);
                                            continue;
                                        }
                                    }
                                    lineBuffer += data;

                                    // Process complete lines
                                    size_t pos = 0;
                                    while ((pos = lineBuffer.find('\n')) != std::string::npos)
                                    {
                                        std::string line = lineBuffer.substr(0, pos);
                                        stream << helperAddTimestamps(line);
                                        lineBuffer.erase(0, pos + 1);
                                    }
                                }
                                else
                                {
                                    // Original behavior - write raw data
                                    stream.write(buffer.data(), bytesRead);
                                }
                                stream.flush();
                            }
                            else if (bytesRead == 0)
                            {
                                log_debug("The writer has closed the pipe, stop redirecting");
                                // If we have a partial line in the buffer, write it with timestamp
                                if (addTimestamps && !lineBuffer.empty())
                                {
                                    stream << helperAddTimestamps(lineBuffer);
                                    stream.flush();
                                }
                                break;
                            }
                            else if (bytesRead == -1 && (errno == EAGAIN || errno == EINTR))
                            {
                                // No data available right now, continue
                                // or the thread was interrupted, continue
                            }
                            else if (bytesRead == -1)
                            {
                                log_error("Error reading from pipe: {}", strerror(errno));
                                break;
                            }

                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        }
                    };

              // Main loop - read while not stopped
              redirectPipeToStreamWhile([&stopFlag]() { return !stopFlag.load(std::memory_order_relaxed); });

              // Drain the pipe after stopFlag is set
              redirectPipeToStreamWhile([]() { return true; });

              log_debug("Pipe redirection thread completed");
          };

    bool useTimestamps = m_logFileNames.has_value();

    // std::ref is required to pass the reference to the streams and stop flag to the thread
    log_debug("Starting pipe redirection threads");
    m_stdoutRedirectThread = std::thread(RedirectPipeToStreamFunc,
                                         m_stdoutFd.value(),
                                         std::ref(*m_stdoutStream),
                                         std::ref(m_stopRedirectFlag),
                                         useTimestamps);
    m_stderrRedirectThread = std::thread(RedirectPipeToStreamFunc,
                                         m_stderrFd.value(),
                                         std::ref(*m_stderrStream),
                                         std::ref(m_stopRedirectFlag),
                                         useTimestamps);

    return DCGM_ST_OK;
}

dcgmReturn_t MpiRunner::PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo)
{
    // Stop the redirection threads
    StopRedirectThreads();

    dcgmReturn_t result { DCGM_ST_OK };

    // Choose the appropriate input stream for the callback based on whether we're using files or memory
    if (m_logFileNames.has_value())
    {
        // We were redirecting to files, so use the file as input
        std::ifstream stdoutFileStream(m_logFileNames.value().first);
        if (!stdoutFileStream.is_open())
        {
            log_error("Failed to open file: {}", m_logFileNames.value().first);
            return DCGM_ST_FILE_IO_ERROR;
        }
        result = m_outputCallback(stdoutFileStream, responseStruct, nodeInfo);
    }
    else
    {
        // We were redirecting to memory, so use the string stream as input
        std::istringstream stdoutMemStream(m_stdoutMemoryStream.str());
        result = m_outputCallback(stdoutMemStream, responseStruct, nodeInfo);
    }

    return result;
}