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
#include "StdLines.hpp"

#include <FastPimpl.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>


namespace DcgmNs::Common::Subprocess
{

class ChildProcess
{
public:
    friend class ChildProcessBuilder;
    StdLines &StdOut();
    StdLines &StdErr();
    FramedChannel &GetFdChannel();
    void Run();
    ChildProcess();
    ~ChildProcess();

    ChildProcess(ChildProcess const &)            = delete;
    ChildProcess &operator=(ChildProcess const &) = delete;

    ChildProcess(ChildProcess &&) noexcept;
    ChildProcess &operator=(ChildProcess &&) noexcept;
    std::optional<pid_t> GetPid() const;
    bool IsAlive() const;
    std::optional<int> GetExitCode() const;
    std::optional<int> ReceivedSignal() const;
    void Wait();

private:
    void Validate() const noexcept(false);
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    friend ChildProcess Create(boost::filesystem::path const &executable,
                               std::vector<std::string> const &args,
                               std::unordered_map<std::string, std::string> const &env,
                               std::optional<std::string> const &userName,
                               int channelFd);
};

ChildProcess Create(boost::filesystem::path const &executable,
                    std::vector<std::string> const &args,
                    std::unordered_map<std::string, std::string> const &env,
                    std::optional<std::string> const &userName,
                    int channelFd);

} //namespace DcgmNs::Common::Subprocess