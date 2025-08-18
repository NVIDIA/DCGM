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

#include "DcgmChildProcessManager.hpp"

#include <ChildProcess/ChildProcessBuilder.hpp>
#include <DcgmStringHelpers.h>

using namespace DcgmNs::Common::Subprocess;

dcgmReturn_t WriteToFd(std::byte const *data, size_t size, DcgmNs::Utils::FileHandle &fd, std::stop_token &stopToken)
{
    // Loop until all bytes are written
    ssize_t bytesWritten     = 0;
    size_t totalBytesWritten = 0;
    while (totalBytesWritten < size)
    {
        if (stopToken.stop_requested())
        {
            log_verbose("Stop requested, exiting write loop");
            return DCGM_ST_OK;
        }
        bytesWritten = fd.Write(data + totalBytesWritten, size - totalBytesWritten);
        if (bytesWritten == -1)
        {
            // If the pipe is full or interrupted, wait for it to be writable
            if (fd.GetErrno() == EAGAIN || fd.GetErrno() == EINTR)
            {
                continue;
            }
            else
            {
                log_error("Failed to write to pipe: {}", strerror(fd.GetErrno()));
                return DCGM_ST_FILE_IO_ERROR;
            }
        }
        totalBytesWritten += bytesWritten;
    }
    log_verbose("Wrote {} bytes to fd {}", totalBytesWritten, fd.Get());
    return DCGM_ST_OK;
}

static dcgmReturn_t WriteStdLinesToFd(StdLines &stdLines, int fd, std::stop_token &stop_token)
{
    DcgmNs::Utils::FileHandle dataFd(fd);
    for (auto &line : stdLines)
    {
        if (auto ret = WriteToFd(reinterpret_cast<std::byte const *>(line.data()), line.size(), dataFd, stop_token);
            ret != DCGM_ST_OK)
        {
            return ret;
        }
        // StdLines iterates over lines. Restore newline to the end of the current line.
        if (auto ret = WriteToFd(reinterpret_cast<std::byte const *>("\n"), 1, dataFd, stop_token); ret != DCGM_ST_OK)
        {
            return ret;
        }
    }
    // Caller is responsible for closing the file descriptor
    fd = dataFd.Release();
    return DCGM_ST_OK;
}

static dcgmReturn_t WriteFramedChannelToFd(FramedChannel &channel, int fd, std::stop_token &stop_token)
{
    DcgmNs::Utils::FileHandle dataFd(fd);
    for (auto &frame : channel)
    {
        if (auto ret = WriteToFd(reinterpret_cast<std::byte const *>(frame.data()), frame.size(), dataFd, stop_token);
            ret != DCGM_ST_OK)
        {
            return ret;
        }
    }
    // Caller is responsible for closing the file descriptor
    fd = dataFd.Release();
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmChildProcessManager::CreateChannelPipe(auto &channelSource, auto writeFunction, PipeInfo &outPipe)
{
    int pipeFds[2];
    if (pipe(pipeFds) != 0)
    {
        log_error("Failed to create pipe: {}", strerror(errno));
        return DCGM_ST_FILE_IO_ERROR;
    }
    log_verbose("Created pipe with fds {} and {}", pipeFds[0], pipeFds[1]);
    std::jthread channelThread([writeFd = pipeFds[1], &channelSource, writeFunction](std::stop_token stop_token) {
        writeFunction(channelSource, writeFd, stop_token);
        close(writeFd);
    });

    outPipe = PipeInfo { pipeFds[0], pipeFds[1], std::move(channelThread) };
    return DCGM_ST_OK;
}

DcgmChildProcessManager::DcgmChildProcessManager(ChildProcessFactory processFactory)
    : m_processFactory(processFactory)
    , m_ioContext(std::make_unique<IoContext>())
{}

dcgmReturn_t DcgmChildProcessManager::Spawn(dcgmChildProcessParams_t const &params,
                                            ChildProcessHandle_t &handle,
                                            int &pid)
{
    handle = INVALID_CHILD_PROCESS_HANDLE;
    pid    = 0;

    std::vector<std::string> args;
    std::unordered_map<std::string, std::string> envMap;
    std::optional<std::string> userName;
    std::optional<int> dataChannelFd;

    // Validate params version
    if (params.version != dcgmChildProcessParams_version1)
    {
        log_error("Invalid params version provided: {}, expected: {}", params.version, dcgmChildProcessParams_version1);
        return DCGM_ST_VER_MISMATCH;
    }

    // Validate executable
    if (!params.executable)
    {
        log_error("Invalid executable.");
        return DCGM_ST_BADPARAM;
    }

    // Validate arguments
    if (params.args && params.numArgs > 0)
    {
        args = std::move(std::vector<std::string>(params.args, params.args + params.numArgs));
    }

    // Validate environment variables
    if (params.env)
    {
        for (auto env : std::span<char const *const>(params.env, params.numEnv))
        {
            auto keyValue = DcgmNs::Split(env, '=');
            if (keyValue.size() != 2)
            {
                log_error("Invalid environment variable: {}", env);
                return DCGM_ST_BADPARAM;
            }
            envMap.insert(std::make_pair(keyValue[0], keyValue[1]));
        }
    }

    // Get user name if specified
    if (params.userName)
    {
        userName = params.userName;
    }

    // Get data channel file descriptor if specified
    if (params.dataChannelFd > 2)
    {
        dataChannelFd = params.dataChannelFd;
    }

    auto &&process = m_processFactory();
    process->Create(*m_ioContext,
                    params.executable,
                    std::move(args),
                    std::move(envMap),
                    std::move(userName),
                    std::move(dataChannelFd));
    process->Run();

    // If the process was launched successfully, a valid pid will be returned
    auto getPid = process->GetPid();
    if (getPid.has_value())
    {
        pid = getPid.value();
    }
    else
    {
        log_error("Failed to get pid for child process {}", params.executable);
        return DCGM_ST_CHILD_SPAWN_FAILED;
    }

    // Create pipes for stdout, stderr, and data channel
    PipeInfo stdErrPipe, stdOutPipe, dataChannelPipe;

    // Create stderr pipe
    dcgmReturn_t result = CreateChannelPipe(process->StdErr(), WriteStdLinesToFd, stdErrPipe);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Create stdout pipe
    result = CreateChannelPipe(process->StdOut(), WriteStdLinesToFd, stdOutPipe);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Handle data channel if needed
    auto dataChannel = process->GetFdChannel();
    if (params.dataChannelFd > 2)
    {
        if (!dataChannel.has_value())
        {
            log_error("Child process does not have a data channel");
            return DCGM_ST_UNINITIALIZED;
        }
        // dereference the reference_wrapper to get the FramedChannel, since CreateChannelPipe will create a thread to
        // access the FramedChannel. If we pass the reference_wrapper to CreateChannelPipe, the thread has a possibility
        // to access a stack variable (dataChannel) that is already destroyed.
        auto &channel = dataChannel.value().get();
        result        = CreateChannelPipe(channel, WriteFramedChannelToFd, dataChannelPipe);
        if (result != DCGM_ST_OK)
        {
            return result;
        }
    }

    handle = GetNextHandle();
    AddProcessInfo(handle,
                   ProcessInfo { .process         = std::move(process),
                                 .stdErrPipe      = std::move(stdErrPipe),
                                 .stdOutPipe      = std::move(stdOutPipe),
                                 .dataChannelPipe = std::move(dataChannelPipe) });

    return DCGM_ST_OK;
}


void DcgmChildProcessManager::AddProcessInfo(ChildProcessHandle_t handle, ProcessInfo &&processInfo)
{
    // The following creates a temporary map to allocate and extract a node that is used
    // to insert into m_processes. This ensures that the mutex is not held during map
    // node allocation.
    auto newNode = [&] {
        std::unordered_map<ChildProcessHandle_t, ProcessInfo> temp;
        auto [it, success] = temp.emplace(std::move(handle), std::move(processInfo));
        static_cast<void>(success); // We know it was successful. Suppress unused variable warnings
        return temp.extract(it);
    }();
    {
        std::lock_guard<std::mutex> lg(m_processesMutex);
        m_processes.insert(std::move(newNode));
    }
}

dcgmReturn_t DcgmChildProcessManager::Stop(ChildProcessHandle_t handle, bool force)
{
    auto processInfoRef = IsValidHandle(handle);
    if (!processInfoRef.has_value())
    {
        log_error("Invalid handle provided: {}", handle);
        return DCGM_ST_BADPARAM;
    }

    auto &processInfo = (*processInfoRef).get();
    processInfo.process->Stop(force);
    if (processInfo.process->IsAlive())
    {
        if (force)
        {
            log_error("Process {} has not terminated yet after SIGKILL.", processInfo.process->GetPid().value());
        }
        else
        {
            log_info("Process {} has not terminated yet after SIGTERM.", processInfo.process->GetPid().value());
        }
    }
    return DCGM_ST_OK;
}


dcgmReturn_t DcgmChildProcessManager::GetStatus(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status)
{
    auto processInfoRef = IsValidHandle(handle);
    if (!processInfoRef.has_value())
    {
        log_error("Invalid handle provided: {}", handle);
        return DCGM_ST_BADPARAM;
    }
    if (status.version != dcgmChildProcessStatus_version1)
    {
        log_error("Invalid status version provided: {}, expected: {}", status.version, dcgmChildProcessStatus_version1);
        return DCGM_ST_VER_MISMATCH;
    }
    memset(&status, 0, sizeof(dcgmChildProcessStatus_v1));
    status.version = dcgmChildProcessStatus_version1;

    auto &processInfo = (*processInfoRef).get();
    if (processInfo.process->IsAlive())
    {
        status.running = true;
    }
    else
    {
        status.running = false;
        if (!processInfo.process->GetExitCode().has_value())
        {
            // This is unexpected. This value should be available once the process has exited
            log_error("Process {} exit code not available.", processInfo.process->GetPid().value());
            return DCGM_ST_GENERIC_ERROR;
        }
        status.exitCode = processInfo.process->GetExitCode().value();
        if (!processInfo.process->ReceivedSignal().has_value())
        {
            status.receivedSignal       = false;
            status.receivedSignalNumber = 0;
        }
        else
        {
            status.receivedSignal       = true;
            status.receivedSignalNumber = *processInfo.process->ReceivedSignal();
        }
    }
    return DCGM_ST_OK;
}


dcgmReturn_t DcgmChildProcessManager::Wait(ChildProcessHandle_t handle, int timeoutSec)
{
    auto processInfoRef = IsValidHandle(handle);
    if (!processInfoRef.has_value())
    {
        log_error("Invalid handle provided: {}", handle);
        return DCGM_ST_BADPARAM;
    }

    if (timeoutSec == -1)
    {
        processInfoRef->get().process->Wait();
    }
    else
    {
        processInfoRef->get().process->Wait(std::chrono::seconds(timeoutSec));
    }
    return DCGM_ST_OK;
}


dcgmReturn_t DcgmChildProcessManager::Destroy(ChildProcessHandle_t handle, int sigTermTimeoutSec)
{
    // Remove the processInfo from the map holding the lock, and
    // cleanup the processInfo after the lock is released.
    auto const processInfoNode = [&] {
        std::lock_guard<std::mutex> lg(m_processesMutex);
        auto const it = m_processes.find(handle);
        if (it != m_processes.end())
        {
            return m_processes.extract(handle);
        }
        return typename std::unordered_map<ChildProcessHandle_t, ProcessInfo>::node_type();
    }();
    if (processInfoNode.empty())
    {
        return DCGM_ST_BADPARAM;
    }
    processInfoNode.mapped().process->Kill(sigTermTimeoutSec);
    if (processInfoNode.mapped().process->IsAlive())
    {
        log_error("Process {} did not terminate in time after SIGKILL. Destroying ChildProcess instance anyway.",
                  processInfoNode.mapped().process->GetPid().value());
        return DCGM_ST_CHILD_NOT_KILLED;
    }
    return DCGM_ST_OK;
}


dcgmReturn_t DcgmChildProcessManager::GetStdErrHandle(ChildProcessHandle_t handle, int &fd)
{
    auto processInfoRef = IsValidHandle(handle);
    if (!processInfoRef.has_value())
    {
        log_error("Invalid handle provided: {}", handle);
        return DCGM_ST_BADPARAM;
    }
    auto &processInfo = (*processInfoRef).get();
    fd                = processInfo.stdErrPipe.readFd;
    return DCGM_ST_OK;
}


dcgmReturn_t DcgmChildProcessManager::GetStdOutHandle(ChildProcessHandle_t handle, int &fd)
{
    auto processInfoRef = IsValidHandle(handle);
    if (!processInfoRef.has_value())
    {
        log_error("Invalid handle provided: {}", handle);
        return DCGM_ST_BADPARAM;
    }
    auto &processInfo = (*processInfoRef).get();
    fd                = processInfo.stdOutPipe.readFd;
    return DCGM_ST_OK;
}


dcgmReturn_t DcgmChildProcessManager::GetDataChannelHandle(ChildProcessHandle_t handle, int &fd)
{
    auto processInfoRef = IsValidHandle(handle);
    if (!processInfoRef.has_value())
    {
        log_error("Invalid handle provided: {}", handle);
        return DCGM_ST_BADPARAM;
    }

    auto &processInfo = (*processInfoRef).get();

    // Check if the process has a data channel
    auto dataChannel = processInfo.process->GetFdChannel();
    if (!dataChannel.has_value())
    {
        log_error("Child process does not have a data channel");
        return DCGM_ST_BADPARAM;
    }
    fd = processInfo.dataChannelPipe.readFd;
    return DCGM_ST_OK;
}

ChildProcessHandle_t DcgmChildProcessManager::GetNextHandle()
{
    ChildProcessHandle_t invalidHandle = INVALID_CHILD_PROCESS_HANDLE;
    // This will happen in the unlikely event that 2^32 processes are spawned
    // at which point we will start over from 1
    m_nextHandle.compare_exchange_strong(invalidHandle, 1);

    return m_nextHandle.fetch_add(1);
}

std::optional<std::reference_wrapper<DcgmChildProcessManager::ProcessInfo>> DcgmChildProcessManager::IsValidHandle(
    ChildProcessHandle_t handle)
{
    if (handle == INVALID_CHILD_PROCESS_HANDLE)
    {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(m_processesMutex);
    auto it = m_processes.find(handle);
    if (it == m_processes.end())
    {
        return std::nullopt;
    }
    return it->second;
}

dcgmReturn_t DcgmChildProcessManager::Reset()
{
    log_debug("Resetting ChildProcessManager - destroying all child processes");

    size_t processCount      = 0;
    dcgmReturn_t worstResult = DCGM_ST_OK;

    // The declaration order of ioContextToDestroy and processesToDestroy is important.
    // processInfo's member PipeInfo threads use the IoContext, so it must be destroyed before the IoContext is
    // destroyed.
    std::unique_ptr<IoContext> ioContextToDestroy = std::make_unique<IoContext>();
    std::unordered_map<ChildProcessHandle_t, ProcessInfo> processesToDestroy;
    {
        std::lock_guard<std::mutex> lock(m_processesMutex);
        std::swap(processesToDestroy, m_processes);
        processCount = processesToDestroy.size();
        m_nextHandle.store(INVALID_CHILD_PROCESS_HANDLE);
        std::swap(ioContextToDestroy, m_ioContext);
    }

    // Kill processes
    for (auto &[handle, processInfo] : processesToDestroy)
    {
        try
        {
            processInfo.process->Kill(3);
            if (processInfo.process->IsAlive())
            {
                if (worstResult == DCGM_ST_OK)
                {
                    worstResult = DCGM_ST_CHILD_NOT_KILLED;
                }
            }
        }
        catch (const std::exception &e)
        {
            log_error("Exception while killing process during reset: {}", e.what());
            if (worstResult == DCGM_ST_OK)
            {
                worstResult = DCGM_ST_CHILD_NOT_KILLED;
            }
        }
    }

    log_debug("ChildProcessManager reset completed. Destroyed {} processes", processCount);
    return worstResult;
    // 1. processesToDestroy destructor runs
    // 2. ProcessInfo destructors run
    // 3. PipeInfo thread.join() - threads using IoContext!
    // 4. IoContext destructor runs
}

DcgmChildProcessManager::~DcgmChildProcessManager()
{
    for (auto &[handle, processInfo] : m_processes)
    {
        // Kill the process with a short sigterm timeout
        processInfo.process->Kill(1);
        // Join the threads before destroying the ChildProcess instance
        if (processInfo.stdErrPipe.thread.joinable())
        {
            processInfo.stdErrPipe.thread.request_stop();
            processInfo.stdErrPipe.thread.join();
        }
        if (processInfo.stdOutPipe.thread.joinable())
        {
            processInfo.stdOutPipe.thread.request_stop();
            processInfo.stdOutPipe.thread.join();
        }
        if (processInfo.dataChannelPipe.thread.joinable())
        {
            processInfo.dataChannelPipe.thread.request_stop();
            processInfo.dataChannelPipe.thread.join();
        }
    }
}