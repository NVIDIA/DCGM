/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <boost/filesystem/path.hpp>
#include <string>
#include <unordered_map>
#include <vector>


namespace DcgmNs::Common::Subprocess
{

constexpr int DEFAULT_CHANNEL_FD = 3;

class ChildProcessBuilder
{
public:
    ChildProcessBuilder &SetExecutable(boost::filesystem::path executable);
    ChildProcessBuilder &SetRunningUser(std::string user);
    ChildProcessBuilder &AddArg(std::string arg);
    ChildProcessBuilder &AddArgs(std::vector<std::string> args);
    ChildProcessBuilder &AddEnvironment(std::unordered_map<std::string, std::string> env);
    ChildProcessBuilder &SetChannelFd(int const fd);
    ChildProcess Build();

private:
    boost::filesystem::path m_executable;
    std::vector<std::string> m_args;
    std::unordered_map<std::string, std::string> m_environment;
    std::string m_user;
    int m_channelFd = DEFAULT_CHANNEL_FD;
};

} //namespace DcgmNs::Common::Subprocess