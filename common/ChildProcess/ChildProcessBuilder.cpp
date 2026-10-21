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

#include "ChildProcessBuilder.hpp"

#include <algorithm>


namespace DcgmNs::Common::Subprocess
{

ChildProcessBuilder &ChildProcessBuilder::SetExecutable(boost::filesystem::path executable)
{
    m_executable = std::move(executable);
    return *this;
}

ChildProcessBuilder &ChildProcessBuilder::SetRunningUser(std::string user)
{
    m_user = std::move(user);
    return *this;
}

ChildProcessBuilder &ChildProcessBuilder::AddArg(std::string arg)
{
    m_args.emplace_back(std::move(arg));
    return *this;
}

ChildProcessBuilder &ChildProcessBuilder::AddArgs(std::vector<std::string> args)
{
    m_args.reserve(m_args.capacity() + args.size());
    for (auto &arg : args)
    {
        m_args.emplace_back(std::move(arg));
    }
    return *this;
}

ChildProcess ChildProcessBuilder::Build(IoContext &ioContext) const
{
    ChildProcess cp;
    Build(ioContext, cp);
    return cp;
}

ChildProcessBase &ChildProcessBuilder::Build(IoContext &ioContext, ChildProcessBase &process) const
{
    auto channelFd = m_channelFd;
    if (!channelFd.has_value() && m_framedChannelFds.size() == 1)
    {
        channelFd = m_framedChannelFds.front();
    }
    std::optional<ChildProcessFds> allowedFds;
    if (m_allowedFds.has_value())
    {
        auto builderFds = ChildProcessFds { m_framedChannelFds.begin(), m_framedChannelFds.end() };
        if (channelFd.has_value())
        {
            builderFds.insert(*channelFd);
        }
        allowedFds = BuildChildProcessAllowedFds(*m_allowedFds, builderFds);
    }
    process.Create(ioContext,
                   m_executable,
                   m_args,
                   BuildEnvironment(CurrentChildProcessEnvironment()),
                   m_user.empty() ? std::nullopt : std::optional { m_user },
                   channelFd,
                   ChildProcessEnvironmentMode::Replace,
                   allowedFds);
    for (auto const fd : m_framedChannelFds)
    {
        if (channelFd.has_value() && fd == *channelFd)
        {
            continue;
        }
        process.AddFramedChannel(fd);
    }
    return process;
}

ChildProcessBuilder &ChildProcessBuilder::AddEnvironment(std::unordered_map<std::string, std::string> &&env)
{
    for (auto &&pair : env)
    {
        m_environment.insert(std::move(pair));
    }
    return *this;
}

ChildProcessBuilder &ChildProcessBuilder::ReplaceEnvironment(ChildProcessEnvironment &&env)
{
    m_environment     = std::move(env);
    m_environmentMode = ChildProcessEnvironmentMode::Replace;
    return *this;
}

ChildProcessEnvironment ChildProcessBuilder::BuildEnvironment(ChildProcessEnvironment baseEnvironment) const
{
    return BuildChildProcessEnvironment(std::move(baseEnvironment), m_environment, m_environmentMode);
}

ChildProcessBuilder &ChildProcessBuilder::SetChannelFd(int const fd)
{
    m_channelFd = fd;
    std::erase(m_framedChannelFds, fd);
    return *this;
}

ChildProcessBuilder &ChildProcessBuilder::AddFramedChannel(int fd)
{
    if ((!m_channelFd.has_value() || *m_channelFd != fd)
        && std::find(m_framedChannelFds.begin(), m_framedChannelFds.end(), fd) == m_framedChannelFds.end())
    {
        m_framedChannelFds.emplace_back(fd);
    }
    return *this;
}

ChildProcessBuilder &ChildProcessBuilder::CloseFdsExcept(std::set<int> allowedFds)
{
    m_allowedFds = std::move(allowedFds);
    return *this;
}

} //namespace DcgmNs::Common::Subprocess
