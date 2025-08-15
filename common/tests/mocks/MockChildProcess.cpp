/**
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

#include "MockChildProcess.hpp"

#include <chrono>
#include <thread>

// Static member variable definitions
std::unique_ptr<DcgmNs::Common::Subprocess::StdLines> MockChildProcess::m_stdOut;
std::unique_ptr<DcgmNs::Common::Subprocess::StdLines> MockChildProcess::m_stdErr;
std::optional<std::unique_ptr<DcgmNs::Common::Subprocess::FramedChannel>> MockChildProcess::m_framedDataChannel;
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
                              std::optional<int> dataChannelFd)
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
        m_framedDataChannel         = std::make_unique<DcgmNs::Common::Subprocess::FramedChannel>();
    }
}

void MockChildProcess::Run()
{
    m_isAlive = true;
}

void MockChildProcess::Stop(bool force) noexcept
{
    if (m_isAlive)
    {
        m_isAlive        = false;
        m_exitCode       = 1;
        m_receivedSignal = force ? SIGKILL : SIGTERM;
    }
}

void MockChildProcess::Kill(int) noexcept
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
    if (m_framedDataChannel.has_value())
    {
        return std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>(*(*m_framedDataChannel));
    }
    return std::nullopt;
}

std::optional<pid_t> MockChildProcess::GetPid() const
{
    return m_pid;
}

bool MockChildProcess::IsAlive() const
{
    return m_isAlive;
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
    if (m_framedDataChannel.has_value())
    {
        // Convert string to a byte span and write to the channel. Prefix the
        // data with a 4 byte length of the data to be written. This constitutes
        // a single frame.
        std::uint32_t length = data.length();
        (*m_framedDataChannel)->Write({ reinterpret_cast<std::byte const *>(&length), sizeof(length) });
        (*m_framedDataChannel)->Write({ reinterpret_cast<std::byte const *>(data.data()), data.length() });
    }
}

MockChildProcess::SpawnParams MockChildProcess::GetSpawnParams()
{
    return m_spawnParams;
}

void MockChildProcess::CloseAllPipes()
{
    m_stdOut->Close();
    m_stdErr->Close();
    if (m_framedDataChannel.has_value())
    {
        (*m_framedDataChannel)->Close();
    }
}

void MockChildProcess::Reset()
{
    m_stdOut.reset(new DcgmNs::Common::Subprocess::StdLines());
    m_stdErr.reset(new DcgmNs::Common::Subprocess::StdLines());
    m_framedDataChannel = std::nullopt;
    m_isAlive           = false;
    m_exitCode          = std::nullopt;
    m_receivedSignal    = std::nullopt;
    m_pid               = std::nullopt;
    m_waitTimeout       = std::nullopt;
    m_spawnParams       = SpawnParams();
}

void MockChildProcess::SetPid(pid_t pid)
{
    m_pid = pid;
}

MockChildProcess::~MockChildProcess()
{
    log_info("MockChildProcess destructor called");
}