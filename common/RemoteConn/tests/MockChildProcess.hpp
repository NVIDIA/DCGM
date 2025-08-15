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

#include <ChildProcess/ChildProcess.hpp>
#include <any>
#include <vector>

namespace DcgmNs::Common::RemoteConn::Mock
{

class MockChildProcess final : public DcgmNs::Common::Subprocess::ChildProcessBase
{
public:
    void Create(IoContext &,
                boost::filesystem::path const &,
                std::optional<std::vector<std::string>> const &args                    = std::nullopt,
                std::optional<std::unordered_map<std::string, std::string>> const &env = std::nullopt,
                std::optional<std::string> const &userName                             = std::nullopt,
                std::optional<int> channelFd                                           = std::nullopt) override;
    void Run() override;
    void Stop(bool force = false) noexcept override;
    void Kill(int sigTermTimeoutSec = 10) noexcept override;
    void GetStdErrBuffer(fmt::memory_buffer &errorString, bool block) override;
    void GetStdOutBuffer(fmt::memory_buffer &errorString, bool block) override;
    DcgmNs::Common::Subprocess::StdLines &StdOut() override;
    DcgmNs::Common::Subprocess::StdLines &StdErr() override;
    std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> GetFdChannel() override;
    std::optional<pid_t> GetPid() const override;
    bool IsAlive() const override;
    std::optional<int> GetExitCode() const override;
    std::optional<int> ReceivedSignal() const override;
    void Wait(std::optional<std::chrono::milliseconds> timeout = std::nullopt) override;

    std::string GetAddressFwdSessionKey();

    MockChildProcess()           = default;
    ~MockChildProcess() noexcept = default;

private:
    std::string m_addressFwdSession;
    DcgmNs::Common::Subprocess::StdLines m_stdOut, m_stdErr;
};

struct MockReturns
{
    bool isAlive;
    std::string stdError;
    pid_t pid;
    bool operator==(MockReturns const &other) const = default;
};

class MockStateCache
{
public:
    static void SetMockReturns(std::string const &addressFwdSession, MockReturns const &mockReturns);
    static std::optional<MockReturns> GetMockReturns(std::string const &addressFwdSession);
    static void ClearMockReturns(std::string const &addressFwdSession);

    static unsigned int GetMockCallCount(std::string const &addressFwdSession, std::string const &function);
    static void IncrementMockCallCount(std::string const &addressFwdSession, std::string const &function);
    static void ClearMockCallCount(std::string const &addressFwdSession);

    static std::chrono::milliseconds GetMockCallWaitTimes(std::string const &addressFwdSession,
                                                          std::string const &function);
    static void SetMockCallWaitTimes(std::string const &addressFwdSession,
                                     std::string const &function,
                                     std::chrono::milliseconds wait);
    static void ClearMockCallWaitTimes(std::string const &addressFwdSession);

    template <typename... Args>
    static void StoreFuncArgs(std::string const &addressFwdSession, std::string const &function, Args &&...args);

    static std::vector<std::any> GetFuncArgs(std::string const &addressFwdSession, std::string const &function);
    static void ClearFuncArgs(std::string const &addressFwdSession);

    static void ClearAll();

private:
    static std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::any>>> m_mockFuncArgs;
    static std::unordered_map<std::string, MockReturns> m_mockReturns;
    static std::unordered_map<std::string, std::unordered_map<std::string, unsigned int>> m_mockCallCount;
    static std::unordered_map<std::string, std::unordered_map<std::string, std::chrono::milliseconds>>
        m_mockCallWaitTimes;
    static std::mutex m_stateMutex;
};

// Template implementation
template <typename... Args>
void MockStateCache::StoreFuncArgs(std::string const &addressFwdSession, std::string const &function, Args &&...args)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    std::vector<std::any> argsVec;
    (argsVec.push_back(std::forward<Args>(args)), ...);
    m_mockFuncArgs[addressFwdSession][function] = std::move(argsVec);
}

} //namespace DcgmNs::Common::RemoteConn::Mock