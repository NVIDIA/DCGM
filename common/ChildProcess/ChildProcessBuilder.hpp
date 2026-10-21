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

#pragma once

#include "ChildProcess.hpp"

#include <boost/asio/io_context.hpp>
#include <boost/filesystem/path.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>


namespace DcgmNs::Common::Subprocess
{

class ChildProcessBuilder
{
public:
    ChildProcessBuilder &SetExecutable(boost::filesystem::path executable);
    ChildProcessBuilder &SetRunningUser(std::string user);
    ChildProcessBuilder &AddArg(std::string arg);
    ChildProcessBuilder &AddArgs(std::vector<std::string> args);
    ChildProcessBuilder &AddEnvironment(std::unordered_map<std::string, std::string> &&env);
    ChildProcessBuilder &ReplaceEnvironment(ChildProcessEnvironment &&env);
    [[nodiscard]] ChildProcessEnvironment BuildEnvironment(ChildProcessEnvironment baseEnvironment) const;
    /**
     * Register the legacy single framed data channel.
     *
     * This is a compatibility wrapper for existing callers. New multi-channel
     * callers should prefer AddFramedChannel(fd) for each child FD they need.
     *
     * @param[in] fd  Child file descriptor to bind as the legacy framed channel.
     */
    ChildProcessBuilder &SetChannelFd(int const fd);
    /**
     * Register one framed child output channel.
     *
     * The child's fd is bound during exec setup. The parent can read framed
     * data through ChildProcess::GetFdChannel(fd). Register channels before
     * Build(); duplicate registrations are ignored by ChildProcess.
     *
     * @param[in] fd  Child file descriptor to bind as a framed output channel.
     */
    ChildProcessBuilder &AddFramedChannel(int fd);
    /**
     * Configure child launch to close all file descriptors except those listed
     * here and any framed channel descriptors registered on this builder.
     *
     * @param[in] allowedFds  Child file descriptors that may remain open across exec.
     *
     * @return This builder.
     */
    ChildProcessBuilder &CloseFdsExcept(std::set<int> allowedFds);
    /**
     * Create and start a child process using this builder configuration.
     *
     * @param[in] ioContext  I/O context used to manage the child process.
     *
     * @return The created child process.
     */
    ChildProcess Build(IoContext &ioContext) const;
    /**
     * Configure and start an existing child process instance.
     *
     * @param[in]     ioContext  I/O context used to manage the child process.
     * @param[in,out] process    Child process instance to configure and start.
     *
     * @return The configured child process.
     */
    ChildProcessBase &Build(IoContext &ioContext, ChildProcessBase &process) const;

private:
    boost::filesystem::path m_executable;
    std::vector<std::string> m_args;
    ChildProcessEnvironment m_environment;
    ChildProcessEnvironmentMode m_environmentMode = ChildProcessEnvironmentMode::Inherit;
    std::string m_user;
    std::optional<int> m_channelFd;
    std::vector<int> m_framedChannelFds;
    std::optional<std::set<int>> m_allowedFds;
};

} //namespace DcgmNs::Common::Subprocess
