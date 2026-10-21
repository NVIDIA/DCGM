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
#pragma once

#include <DistributedCudaContextRuntimeAnyUniquePtr.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace DcgmNs::ProfTester::Test
{
struct MockDistributedCudaContextRuntimePipes;

struct MockDistributedCudaContextRuntimeState
{
    MockDistributedCudaContextRuntimeState();
    ~MockDistributedCudaContextRuntimeState();

    std::vector<std::unique_ptr<MockDistributedCudaContextRuntimePipes>> workers;
    unsigned int pipeCalls {};
    unsigned int forkCalls {};
    unsigned int waitPidCalls {};
    unsigned int killCalls {};
    pid_t nextPid { 4200 };
    /**
     * When set, Fork() returns -1 on that 1-based call number and still
     * increments forkCalls so callers can observe the failed attempt.
     */
    std::optional<unsigned int> failOnForkCall {};
};

class MockDistributedCudaContextRuntime
{
public:
    struct CallCounts
    {
        unsigned int pipe {};
        unsigned int fork {};
        unsigned int waitPid {};
        unsigned int kill {};
    };

    class WorkerHandle
    {
    public:
        WorkerHandle() = default;

    private:
        explicit WorkerHandle(MockDistributedCudaContextRuntimePipes *pipes)
            : m_pipes(pipes)
        {}

        MockDistributedCudaContextRuntimePipes *m_pipes {};
        friend class MockDistributedCudaContextRuntime;
    };

    explicit MockDistributedCudaContextRuntime(MockDistributedCudaContextRuntimeState *state);

    std::vector<WorkerHandle> GetWorkers(std::size_t count) const;
    bool WriteResponse(WorkerHandle const &worker, std::string_view response) const;
    bool HasCommand(WorkerHandle const &worker, std::string_view token) const;
    CallCounts GetCallCounts() const;

    int Pipe(int pipeFds[2]);
    pid_t Fork();
    pid_t WaitPid(pid_t pid, int *status, int options);
    int Kill(pid_t pid, int signal);

private:
    std::string DrainCommands(WorkerHandle const &worker) const;

    MockDistributedCudaContextRuntimeState *m_state;
};
} // namespace DcgmNs::ProfTester::Test
