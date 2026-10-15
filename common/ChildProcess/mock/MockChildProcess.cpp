/**
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

#include "MockChildProcess.hpp"

#include <DcgmLogging.h>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

// Static member variable definitions
std::unique_ptr<DcgmNs::Common::Subprocess::StdLines> MockChildProcess::m_stdOut;
std::unique_ptr<DcgmNs::Common::Subprocess::StdLines> MockChildProcess::m_stdErr;
std::unordered_map<int, std::unique_ptr<DcgmNs::Common::Subprocess::FramedChannel>>
    MockChildProcess::m_framedDataChannels;
std::unordered_map<int, MockChildProcess::ChannelTimeoutRegistration> MockChildProcess::m_channelTimeouts;
bool MockChildProcess::m_isAlive;
std::optional<int> MockChildProcess::m_exitCode;
std::optional<int> MockChildProcess::m_receivedSignal;
std::optional<pid_t> MockChildProcess::m_pid;
MockChildProcess::SpawnParams MockChildProcess::m_spawnParams;
std::optional<std::chrono::milliseconds> MockChildProcess::m_waitTimeout;
void MockChildProcess::Create(IoContext &,
                              boost::filesystem::path const &executable,
                              std::optional<std::vector<std::string>> const &args,
                              std::optional<std::unordered_map<std::string, std::string>> const &env,
                              std::optional<std::string> const &userName,
                              std::optional<int> dataChannelFd,
                              DcgmNs::Common::Subprocess::ChildProcessEnvironmentMode,
                              std::optional<DcgmNs::Common::Subprocess::ChildProcessFds> const &allowedFds)
{
    m_spawnParams.executable = executable;

    if (args)
    {
        m_spawnParams.args = *args;
    }

    if (env)
    {
        m_spawnParams.env = *env;
    }

    if (userName)
    {
        m_spawnParams.userName = userName;
    }

    if (dataChannelFd)
    {
        m_spawnParams.dataChannelFd = dataChannelFd;
        AddFramedChannel(*dataChannelFd);
    }
    m_spawnParams.allowedFds = allowedFds;
}

void MockChildProcess::Run()
{
    m_isAlive = true;
}

void MockChildProcess::Stop(bool force)
{
    if (m_isAlive)
    {
        m_isAlive        = false;
        m_exitCode       = 1;
        m_receivedSignal = force ? SIGKILL : SIGTERM;
    }
}

void MockChildProcess::Kill(int)
{
    if (m_isAlive)
    {
        m_isAlive        = false;
        m_exitCode       = 1;
        m_receivedSignal = SIGKILL;
    }
}

void MockChildProcess::GetStdErrBuffer(fmt::memory_buffer &errorStrings, bool)
{
    // If stderr has content, add it to the provided buffer
    while (!m_stdErr->IsEmpty())
    {
        auto line = m_stdErr->Read();
        if (line)
        {
            fmt::format_to(std::back_inserter(errorStrings), "{}\n", *line);
        }
        else
        {
            break;
        }
    }
}

void MockChildProcess::GetStdOutBuffer(fmt::memory_buffer &outStrings, bool)
{
    // If stdout has content, add it to the provided buffer
    while (!m_stdOut->IsEmpty())
    {
        auto line = m_stdOut->Read();
        if (line)
        {
            fmt::format_to(std::back_inserter(outStrings), "{}\n", *line);
        }
        else
        {
            break;
        }
    }
}

DcgmNs::Common::Subprocess::StdLines &MockChildProcess::StdOut()
{
    return *m_stdOut;
}

DcgmNs::Common::Subprocess::StdLines &MockChildProcess::StdErr()
{
    return *m_stdErr;
}

std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> MockChildProcess::GetFdChannel()
{
    if (m_spawnParams.dataChannelFd.has_value())
    {
        return GetFdChannel(*m_spawnParams.dataChannelFd);
    }
    return std::nullopt;
}

void MockChildProcess::AddFramedChannel(int fd)
{
    m_framedDataChannels.insert_or_assign(fd, std::make_unique<DcgmNs::Common::Subprocess::FramedChannel>());
}

std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> MockChildProcess::GetFdChannel(int fd)
{
    auto it = m_framedDataChannels.find(fd);
    if (it == m_framedDataChannels.end())
    {
        return std::nullopt;
    }
    return std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>(*it->second);
}

std::expected<void, DcgmNs::Common::Subprocess::ChannelTimeoutError> MockChildProcess::OnChannelTimeout(
    int fd,
    std::chrono::milliseconds duration,
    std::function<void()> handler)
{
    if (duration <= std::chrono::milliseconds { 0 })
    {
        return std::unexpected(DcgmNs::Common::Subprocess::ChannelTimeoutError::InvalidDuration);
    }
    if (!m_framedDataChannels.contains(fd))
    {
        return std::unexpected(DcgmNs::Common::Subprocess::ChannelTimeoutError::NotRegistered);
    }
    m_channelTimeouts.insert_or_assign(fd,
                                       ChannelTimeoutRegistration {
                                           .duration = duration,
                                           .handler  = std::move(handler),
                                       });
    return {};
}

std::optional<pid_t> MockChildProcess::GetPid() const
{
    return m_pid;
}

bool MockChildProcess::IsAlive() const
{
    return m_isAlive;
}

bool MockChildProcess::FailedToLaunch() const
{
    return false;
}

std::optional<int> MockChildProcess::GetExitCode() const
{
    return m_exitCode;
}

std::optional<int> MockChildProcess::ReceivedSignal() const
{
    return m_receivedSignal;
}

void MockChildProcess::Wait(std::optional<std::chrono::milliseconds> timeout)
{
    // If timeout is specified
    if (timeout.has_value())
    {
        m_waitTimeout = *timeout;
    }
}

std::optional<std::chrono::milliseconds> MockChildProcess::GetWaitTimeout()
{
    return m_waitTimeout;
}

void MockChildProcess::SimulateExit(int exitCode, std::optional<int> signal)
{
    m_isAlive  = false;
    m_exitCode = exitCode;
    if (signal.has_value())
    {
        m_receivedSignal = *signal;
    }
}

void MockChildProcess::WriteToStdout(std::string const &line)
{
    m_stdOut->Write(line);
}

void MockChildProcess::WriteToStderr(std::string const &line)
{
    m_stdErr->Write(line);
}

void MockChildProcess::WriteToDataChannel(std::string const &data)
{
    if (m_spawnParams.dataChannelFd.has_value())
    {
        auto it = m_framedDataChannels.find(*m_spawnParams.dataChannelFd);
        if (it == m_framedDataChannels.end())
        {
            return;
        }
        // Convert string to a byte span and write to the channel. Prefix the
        // data with a 4 byte length of the data to be written. This constitutes
        // a single frame.
        std::uint32_t length = data.length();
        it->second->Write({ reinterpret_cast<std::byte const *>(&length), sizeof(length) });
        it->second->Write({ reinterpret_cast<std::byte const *>(data.data()), data.length() });
    }
}

std::optional<std::chrono::milliseconds> MockChildProcess::GetChannelTimeoutDuration(int fd)
{
    auto it = m_channelTimeouts.find(fd);
    if (it == m_channelTimeouts.end())
    {
        return std::nullopt;
    }
    return it->second.duration;
}

bool MockChildProcess::TriggerChannelTimeout(int fd)
{
    auto it = m_channelTimeouts.find(fd);
    if (it == m_channelTimeouts.end() || !it->second.handler)
    {
        return false;
    }
    it->second.handler();
    return true;
}

MockChildProcess::SpawnParams MockChildProcess::GetSpawnParams()
{
    return m_spawnParams;
}

void MockChildProcess::CloseAllPipes()
{
    m_stdOut->Close();
    m_stdErr->Close();
    for (auto &[_, channel] : m_framedDataChannels)
    {
        channel->Close();
    }
}

void MockChildProcess::Reset()
{
    m_stdOut.reset(new DcgmNs::Common::Subprocess::StdLines());
    m_stdErr.reset(new DcgmNs::Common::Subprocess::StdLines());
    m_framedDataChannels.clear();
    m_channelTimeouts.clear();
    m_isAlive        = false;
    m_exitCode       = std::nullopt;
    m_receivedSignal = std::nullopt;
    m_pid            = std::nullopt;
    m_waitTimeout    = std::nullopt;
    m_spawnParams    = SpawnParams();
}

void MockChildProcess::SetPid(pid_t pid)
{
    m_pid = pid;
}

MockChildProcess::~MockChildProcess()
{
    log_info("MockChildProcess destructor called");
}
