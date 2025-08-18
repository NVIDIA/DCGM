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

#include "FramedChannel.hpp"
#include "IoContext.hpp"
#include "StdLines.hpp"

#include <FastPimpl.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>


namespace DcgmNs::Common::Subprocess
{

class ChildProcessBase
{
public:
    virtual void Create(IoContext &ioContext,
                        boost::filesystem::path const &executable,
                        std::optional<std::vector<std::string>> const &args                    = std::nullopt,
                        std::optional<std::unordered_map<std::string, std::string>> const &env = std::nullopt,
                        std::optional<std::string> const &userName                             = std::nullopt,
                        std::optional<int> channelFd                                           = std::nullopt)
        = 0;
    virtual void Run()                                                                 = 0;
    virtual void Stop(bool force = false) noexcept                                     = 0;
    virtual void Kill(int sigTermTimeoutSec = 10) noexcept                             = 0;
    virtual void GetStdErrBuffer(fmt::memory_buffer &errorStrings, bool block)         = 0;
    virtual void GetStdOutBuffer(fmt::memory_buffer &errorStrings, bool block)         = 0;
    virtual StdLines &StdOut()                                                         = 0;
    virtual StdLines &StdErr()                                                         = 0;
    virtual std::optional<std::reference_wrapper<FramedChannel>> GetFdChannel()        = 0;
    virtual std::optional<pid_t> GetPid() const                                        = 0;
    virtual bool IsAlive() const                                                       = 0;
    virtual std::optional<int> GetExitCode() const                                     = 0;
    virtual std::optional<int> ReceivedSignal() const                                  = 0;
    virtual void Wait(std::optional<std::chrono::milliseconds> timeout = std::nullopt) = 0;

    virtual ~ChildProcessBase() = default;
};

class ChildProcess final : public ChildProcessBase
{
public:
    friend class ChildProcessBuilder;
    StdLines &StdOut() override;
    StdLines &StdErr() override;
    void GetStdErrBuffer(fmt::memory_buffer &errString, bool block) override;
    void GetStdOutBuffer(fmt::memory_buffer &outString, bool block) override;
    std::optional<std::reference_wrapper<FramedChannel>> GetFdChannel() override;
    void Run() override;
    std::optional<pid_t> GetPid() const override;
    bool IsAlive() const override;
    std::optional<int> GetExitCode() const override;
    std::optional<int> ReceivedSignal() const override;

    // Wait for the process to exit. Waiting on multiple ChildProcesses at the same time
    // is not supported.
    void Wait(std::optional<std::chrono::milliseconds> timeout = std::nullopt) override;
    /**
     * Stop the child process (issues a SIGTERM, unless specified otherwise).
     * Verify that the process was stopped with IsAlive().
     *
     * @param[in] force       Whether to force the process to stop with SIGKILL
     */
    void Stop(bool force = false) noexcept override;
    /**
     * Kill the child process (issues a SIGTERM, then SIGKILL if the process does not exit
     * within sigTermTimeoutSec seconds).
     *
     * @param[in] sigTermTimeoutSec  Timeout in seconds for SIGTERM
     */
    void Kill(int sigTermTimeoutSec = 10) noexcept override;
    void Create(IoContext &ioContext,
                boost::filesystem::path const &executable,
                std::optional<std::vector<std::string>> const &args                    = std::nullopt,
                std::optional<std::unordered_map<std::string, std::string>> const &env = std::nullopt,
                std::optional<std::string> const &userName                             = std::nullopt,
                std::optional<int> channelFd                                           = std::nullopt) override;
    ChildProcess();
    ~ChildProcess();
    ChildProcess(ChildProcess const &)            = delete;
    ChildProcess &operator=(ChildProcess const &) = delete;
    ChildProcess(ChildProcess &&) noexcept;
    ChildProcess &operator=(ChildProcess &&) noexcept;

private:
    void Validate() const noexcept(false);
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} //namespace DcgmNs::Common::Subprocess