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

#include "MockChildProcess.hpp"

namespace DcgmNs::Common::RemoteConn::Mock
{

void MockChildProcess::Create(IoContext &,
                              boost::filesystem::path const &path,
                              std::optional<std::vector<std::string>> const &args,
                              std::optional<std::unordered_map<std::string, std::string>> const &env,
                              std::optional<std::string> const &username,
                              std::optional<int> channelFd)
{
    // This mock cannot be used without passing in the args.
    if (!args.has_value())
    {
        throw std::invalid_argument("args must have a value");
    }

    // The argument that follows the -L argument is needed here. It should be of the
    // form "127.0.0.1:34567:127.0.0.1:65001" for port forwarding or
    // "/tmp/localUnixPath.sock:/tmp/remoteUnixPath.sock" for unix domain socket forwarding.
    // Note this is very specific to the ssh arguments passed in. This logic should be
    // made more generic to apply to other commands.
    auto it = std::find(args->begin(), args->end(), "-L");
    if (it == args->end() || ++it == args->end())
    {
        throw std::invalid_argument("Required argument after -L is missing");
    }

    m_addressFwdSession = *it;
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);

    // Store the function arguments. IoContext is skipped because it's not a copyable type.
    MockStateCache::StoreFuncArgs(m_addressFwdSession, __func__, path, args, env, username, channelFd);
}

void MockChildProcess::Run()
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
}

std::string MockChildProcess::GetAddressFwdSessionKey()
{
    return m_addressFwdSession;
}

void MockChildProcess::Stop(bool force) noexcept
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    MockStateCache::StoreFuncArgs(m_addressFwdSession, __func__, force);
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
    m_stdOut.Close();
    m_stdErr.Close();
}

void MockChildProcess::GetStdErrBuffer(fmt::memory_buffer &errorString, bool block)
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    MockStateCache::StoreFuncArgs(m_addressFwdSession, __func__, std::ref(errorString), block);
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
    auto mockReturns = MockStateCache::GetMockReturns(m_addressFwdSession);
    if (mockReturns.has_value())
    {
        errorString.append(mockReturns->stdError);
    }
}

bool MockChildProcess::IsAlive() const
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    // Since this is a const method, we can't modify state, so we don't store args here
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
    auto mockReturns = MockStateCache::GetMockReturns(m_addressFwdSession);
    if (mockReturns.has_value())
    {
        return mockReturns->isAlive;
    }
    return false;
}

std::optional<pid_t> MockChildProcess::GetPid() const
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    // Since this is a const method, we can't modify state, so we don't store args here
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
    auto mockReturns = MockStateCache::GetMockReturns(m_addressFwdSession);
    if (mockReturns.has_value())
    {
        return mockReturns->pid;
    }
    return std::nullopt;
}

void MockChildProcess::Kill(int timeoutSec) noexcept
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    MockStateCache::StoreFuncArgs(m_addressFwdSession, __func__, timeoutSec);
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
    m_stdOut.Close();
    m_stdErr.Close();
}

DcgmNs::Common::Subprocess::StdLines &MockChildProcess::StdOut()
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    // Since this is a const method, we can't modify state, so we don't store args here
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
    return m_stdOut;
}

DcgmNs::Common::Subprocess::StdLines &MockChildProcess::StdErr()
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    // Since this is a const method, we can't modify state, so we don't store args here
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
    auto mockReturns = MockStateCache::GetMockReturns(m_addressFwdSession);
    if (mockReturns.has_value())
    {
        m_stdErr.Write(mockReturns->stdError);
    }
    m_stdErr.Close();
    return m_stdErr;
}

std::optional<int> MockChildProcess::GetExitCode() const
{
    MockStateCache::IncrementMockCallCount(m_addressFwdSession, __func__);
    // Since this is a const method, we can't modify state, so we don't store args here
    auto waitTime = MockStateCache::GetMockCallWaitTimes(m_addressFwdSession, __func__);
    std::this_thread::sleep_for(waitTime);
    auto mockReturns = MockStateCache::GetMockReturns(m_addressFwdSession);
    if (mockReturns.has_value())
    {
        return mockReturns->isAlive;
    }
    return std::nullopt;
}

std::unordered_map<std::string, MockReturns> MockStateCache::m_mockReturns;
std::unordered_map<std::string, std::unordered_map<std::string, unsigned int>> MockStateCache::m_mockCallCount;
std::unordered_map<std::string, std::unordered_map<std::string, std::chrono::milliseconds>>
    MockStateCache::m_mockCallWaitTimes;
std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::any>>> MockStateCache::m_mockFuncArgs;
std::mutex MockStateCache::m_stateMutex;

void MockStateCache::SetMockReturns(std::string const &addressFwdSession, MockReturns const &mockReturns)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    m_mockReturns[addressFwdSession] = mockReturns;
}

std::optional<MockReturns> MockStateCache::GetMockReturns(std::string const &addressFwdSession)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    if (m_mockReturns.contains(addressFwdSession))
    {
        return m_mockReturns[addressFwdSession];
    }
    return std::nullopt;
}

void MockStateCache::ClearMockReturns(std::string const &addressFwdSession)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    m_mockReturns.erase(addressFwdSession);
}

unsigned int MockStateCache::GetMockCallCount(std::string const &addressFwdSession, std::string const &function)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    if (m_mockCallCount.contains(addressFwdSession))
    {
        if (m_mockCallCount[addressFwdSession].contains(function))
        {
            return m_mockCallCount[addressFwdSession][function];
        }
    }
    return 0;
}
void MockStateCache::IncrementMockCallCount(std::string const &addressFwdSession, std::string const &function)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    m_mockCallCount[addressFwdSession][function]++;
}
void MockStateCache::ClearMockCallCount(std::string const &addressFwdSession)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    m_mockCallCount.erase(addressFwdSession);
}

std::vector<std::any> MockStateCache::GetFuncArgs(std::string const &addressFwdSession, std::string const &function)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    if (m_mockFuncArgs.contains(addressFwdSession))
    {
        if (m_mockFuncArgs[addressFwdSession].contains(function))
        {
            return m_mockFuncArgs[addressFwdSession][function];
        }
    }
    return {};
}

void MockStateCache::ClearFuncArgs(std::string const &addressFwdSession)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    m_mockFuncArgs.erase(addressFwdSession);
}

void MockStateCache::ClearAll()
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    m_mockReturns.clear();
    m_mockCallCount.clear();
    m_mockCallWaitTimes.clear();
    m_mockFuncArgs.clear();
}

std::chrono::milliseconds MockStateCache::GetMockCallWaitTimes(std::string const &addressFwdSession,
                                                               std::string const &function)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    if (m_mockCallWaitTimes.contains(addressFwdSession))
    {
        if (m_mockCallWaitTimes[addressFwdSession].contains(function))
        {
            return m_mockCallWaitTimes[addressFwdSession][function];
        }
    }
    return std::chrono::milliseconds { 0 };
}

void MockStateCache::SetMockCallWaitTimes(std::string const &addressFwdSession,
                                          std::string const &function,
                                          std::chrono::milliseconds waitTime)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    m_mockCallWaitTimes[addressFwdSession][function] = waitTime;
}

void MockStateCache::ClearMockCallWaitTimes(std::string const &addressFwdSession)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    m_mockCallWaitTimes.erase(addressFwdSession);
}

// The following functions are not required by the tests and are implemented as no-ops

void MockChildProcess::GetStdOutBuffer(fmt::memory_buffer &, bool)
{}

std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> MockChildProcess::GetFdChannel()
{
    return std::nullopt;
}

std::optional<int> MockChildProcess::ReceivedSignal() const
{
    return std::nullopt;
}

void MockChildProcess::Wait(std::optional<std::chrono::milliseconds>)
{}

} //namespace DcgmNs::Common::RemoteConn::Mock