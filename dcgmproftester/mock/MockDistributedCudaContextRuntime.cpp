/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <MockDistributedCudaContextRuntime.hpp>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace DcgmNs::ProfTester::Test
{
struct MockDistributedCudaContextRuntimePipes
{
    int commandReadFd   = -1;
    int responseWriteFd = -1;
    std::string commands;
};

namespace
{
    void CloseFd(int &fd)
    {
        if (fd >= 0)
        {
            close(fd);
            fd = -1;
        }
    }
} // namespace

MockDistributedCudaContextRuntimeState::MockDistributedCudaContextRuntimeState() = default;

MockDistributedCudaContextRuntimeState::~MockDistributedCudaContextRuntimeState()
{
    for (auto &worker : workers)
    {
        CloseFd(worker->commandReadFd);
        CloseFd(worker->responseWriteFd);
    }
}

MockDistributedCudaContextRuntime::MockDistributedCudaContextRuntime(MockDistributedCudaContextRuntimeState *state)
    : m_state(state)
{
    REQUIRE(m_state != nullptr);
}

std::vector<MockDistributedCudaContextRuntime::WorkerHandle> MockDistributedCudaContextRuntime::GetWorkers(
    std::size_t count) const
{
    if (m_state->workers.size() < count
        || !std::all_of(m_state->workers.begin(),
                        m_state->workers.begin() + static_cast<std::ptrdiff_t>(count),
                        [](auto const &worker) { return worker->responseWriteFd >= 0; }))
    {
        return {};
    }

    std::vector<WorkerHandle> result;
    result.reserve(count);
    std::transform(m_state->workers.begin(),
                   m_state->workers.begin() + static_cast<std::ptrdiff_t>(count),
                   std::back_inserter(result),
                   [](auto const &worker) { return WorkerHandle(worker.get()); });
    return result;
}

bool MockDistributedCudaContextRuntime::WriteResponse(WorkerHandle const &worker, std::string_view response) const
{
    char const *cursor    = response.data();
    std::size_t remaining = response.size();
    while (remaining > 0)
    {
        ssize_t const written = write(worker.m_pipes->responseWriteFd, cursor, remaining);
        if (written <= 0)
        {
            return false;
        }
        cursor += written;
        remaining -= static_cast<std::size_t>(written);
    }
    return true;
}

std::string MockDistributedCudaContextRuntime::DrainCommands(WorkerHandle const &worker) const
{
    char buffer[256];
    for (;;)
    {
        ssize_t const bytes = read(worker.m_pipes->commandReadFd, buffer, sizeof(buffer));
        if (bytes > 0)
        {
            worker.m_pipes->commands.append(buffer, buffer + bytes);
            continue;
        }
        if (bytes == 0 || errno == EAGAIN || errno == EWOULDBLOCK)
        {
            break;
        }
        break;
    }
    return worker.m_pipes->commands;
}

bool MockDistributedCudaContextRuntime::HasCommand(WorkerHandle const &worker, std::string_view token) const
{
    return DrainCommands(worker).find(token) != std::string::npos;
}

MockDistributedCudaContextRuntime::CallCounts MockDistributedCudaContextRuntime::GetCallCounts() const
{
    return {
        .pipe    = m_state->pipeCalls,
        .fork    = m_state->forkCalls,
        .waitPid = m_state->waitPidCalls,
        .kill    = m_state->killCalls,
    };
}

int MockDistributedCudaContextRuntime::Pipe(int pipeFds[2])
{
    REQUIRE(::pipe(pipeFds) == 0);

    if ((m_state->pipeCalls % 2) == 0)
    {
        auto worker           = std::make_unique<MockDistributedCudaContextRuntimePipes>();
        worker->commandReadFd = dup(pipeFds[0]);
        REQUIRE(worker->commandReadFd >= 0);
        REQUIRE(fcntl(worker->commandReadFd, F_SETFL, O_NONBLOCK) != -1);
        m_state->workers.push_back(std::move(worker));
    }
    else if (!m_state->workers.empty())
    {
        auto &worker            = m_state->workers.back();
        worker->responseWriteFd = dup(pipeFds[1]);
        REQUIRE(worker->responseWriteFd >= 0);
    }
    m_state->pipeCalls++;
    return 0;
}

pid_t MockDistributedCudaContextRuntime::Fork()
{
    m_state->forkCalls++;
    if (m_state->failOnForkCall.has_value() && m_state->forkCalls == *m_state->failOnForkCall)
    {
        return -1;
    }
    return m_state->nextPid++;
}

pid_t MockDistributedCudaContextRuntime::WaitPid(pid_t pid, int *, int)
{
    m_state->waitPidCalls++;
    return pid;
}

int MockDistributedCudaContextRuntime::Kill(pid_t, int)
{
    m_state->killCalls++;
    return 0;
}
} // namespace DcgmNs::ProfTester::Test
