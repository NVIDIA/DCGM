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

#include "ChildProcessBuilder.hpp"


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

ChildProcess ChildProcessBuilder::Build(IoContext &ioContext)
{
    ChildProcess cp;
    cp.Create(ioContext,
              m_executable,
              m_args,
              m_environment,
              m_user.empty() ? std::nullopt : std::optional { m_user },
              m_channelFd);
    return cp;
}

ChildProcessBuilder &ChildProcessBuilder::AddEnvironment(std::unordered_map<std::string, std::string> &&env)
{
    for (auto &&pair : env)
    {
        m_environment.insert(std::move(pair));
    }
    return *this;
}

ChildProcessBuilder &ChildProcessBuilder::SetChannelFd(int const fd)
{
    m_channelFd = fd;
    return *this;
}

} //namespace DcgmNs::Common::Subprocess