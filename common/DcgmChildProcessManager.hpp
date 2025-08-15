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

#pragma once

#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <ChildProcess/ChildProcess.hpp>
#include <ChildProcess/IoContext.hpp>
#include <DcgmUtilities.h>

#include <memory>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

#define INVALID_CHILD_PROCESS_HANDLE ((ChildProcessHandle_t)0)

using ChildProcessFactory = std::function<std::unique_ptr<DcgmNs::Common::Subprocess::ChildProcessBase>()>;

static ChildProcessFactory DefaultChildProcessFactory = []() {
    return std::make_unique<DcgmNs::Common::Subprocess::ChildProcess>();
};

dcgmReturn_t WriteToFd(std::byte const *data, size_t size, DcgmNs::Utils::FileHandle &fd, std::stop_token &stopToken);

class DcgmChildProcessManager
{
public:
    explicit DcgmChildProcessManager(ChildProcessFactory processFactory = DefaultChildProcessFactory);
    ~DcgmChildProcessManager();
    /**
     * Spawn a new child process.
     *
     * @param[in] params       Parameters required to spawn the child process
     * @param[out] handle      Handle to the ChildProcess instance
     * @param[out] pid         Process ID of the spawned process
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t Spawn(dcgmChildProcessParams_t const &params, ChildProcessHandle_t &handle, int &pid);
    /**
     * Stop the child process (issues a SIGTERM, unless specified otherwise).
     * Verify that the process was stopped with IsAlive().
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[in] force       Whether to force the process to stop with SIGKILL
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t Stop(ChildProcessHandle_t handle, bool force = false);
    /**
     * Get the status of the child process.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[out] status     Status of the child process
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t GetStatus(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status);
    /**
     * Wait for the child process to exit. Default is -1 (no timeout).
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[in] timeoutSec  Timeout in seconds
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t Wait(ChildProcessHandle_t handle, int timeoutSec = -1);
    /**
     * Destroy the child process.
     * Thread safety is not guaranteed when accessing a child process using
     * one of the accessor functions and destroying the same child process at
     * the same time. Destroy is expected to be called after all accessor
     * functions for the same child process have returned.
     *
     * @param[in] handle             Handle to the ChildProcess instance
     * @param[in] sigTermTimeoutSec  Timeout in seconds for SIGTERM
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t Destroy(ChildProcessHandle_t handle, int sigTermTimeoutSec = 10);
    /**
     * Get the standard error handle of the child process.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[out] fd         File descriptor of the standard error
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t GetStdErrHandle(ChildProcessHandle_t handle, int &fd);
    /**
     * Get the standard output handle of the child process.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[out] fd         File descriptor of the standard output
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t GetStdOutHandle(ChildProcessHandle_t handle, int &fd);
    /**
     * Get the data channel handle of the child process.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[out] fd         File descriptor of the data channel
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t GetDataChannelHandle(ChildProcessHandle_t handle, int &fd);

    /**
     * Reset the ChildProcessManager to clean state.
     * Cleans up all processes and resets internal state to prevent
     * signal handling corruption between runs.
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t Reset();

private:
    struct PipeInfo
    {
        int readFd;          //!< Read end of the pipe
        int writeFd;         //!< Write end of the pipe
        std::jthread thread; //!< Thread to read from stdout/stderr/data channel and write to writeFd
    };
    ChildProcessFactory m_processFactory;

    struct ProcessInfo
    {
        std::unique_ptr<DcgmNs::Common::Subprocess::ChildProcessBase> process;
        PipeInfo stdErrPipe;
        PipeInfo stdOutPipe;
        PipeInfo dataChannelPipe;
    };

    std::unique_ptr<IoContext> m_ioContext;
    std::atomic<ChildProcessHandle_t> m_nextHandle = INVALID_CHILD_PROCESS_HANDLE;
    std::unordered_map<ChildProcessHandle_t, ProcessInfo> m_processes;
    std::mutex m_processesMutex;

    std::optional<std::reference_wrapper<ProcessInfo>> IsValidHandle(ChildProcessHandle_t handle);
    ChildProcessHandle_t GetNextHandle();

    template <typename ChannelType, typename WriteFunction>
    dcgmReturn_t CreateChannelPipe(ChannelType &channelSource, WriteFunction writeFunction, PipeInfo &outPipe);

    void AddProcessInfo(ChildProcessHandle_t handle, ProcessInfo &&processInfo);
};
