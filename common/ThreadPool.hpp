/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmThread.h>
#include <TaskRunner.hpp>

#include <atomic>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <vector>

#include <pthread.h>

namespace DcgmNs
{
class ThreadPool
{
public:
    explicit ThreadPool(std::size_t numOfWorkers)
        : m_shouldStop(false)
        , m_numOfWorkers(numOfWorkers)
    {
        std::stringstream ss;
        ss << "Worker of a ThreadPool at 0x" << std::hex << this;
        const std::string threadName = ss.str();

        m_threads.reserve(numOfWorkers);
        for (std::size_t i = 0; i < numOfWorkers; ++i)
        {
            m_threads.emplace_back(threadName, *this);
        }
    }

    void Stop()
    {
        m_runner.Stop();
        for (auto &&th : m_threads)
        {
            th.Stop();
        }
    }

    void StopAndWait()
    {
        Stop();

        for (auto &&th : m_threads)
        {
            th.Wait();
        }
    }

    ~ThreadPool()
    {
        try
        {
            StopAndWait();
        }
        catch (std::exception &e)
        {
            std::cerr << "Caught exception in ~ThreadPool. Swallowing " << e.what() << std::endl;
        }
    }

    [[nodiscard]] std::size_t GetNumWorkers() const
    {
        return m_numOfWorkers;
    }

    template <class Func>
    auto Enqueue(Func func)
    {
        return m_runner.Enqueue(DcgmNs::make_task(
            [func = std::move(func)]() mutable -> std::invoke_result_t<Func> { return std::invoke(func); }));
    }

private:
    class WorkingThread
    {
    public:
        explicit WorkingThread(std::string const &name, ThreadPool &owner)
            : m_owner(owner)
            , m_stop(false)
            , m_joinedAlready(false)
        {
            m_thread = std::thread([this]() mutable { Run(); });
            pthread_setname_np(m_thread.native_handle(), name.c_str());
        }

        WorkingThread(WorkingThread &&r) noexcept
            : m_owner(r.m_owner)
            , m_thread(std::move(r.m_thread))
            , m_stop(r.m_stop.load(std::memory_order_relaxed))
            , m_joinedAlready(r.m_joinedAlready.load(std::memory_order_relaxed))
        {}

        WorkingThread &operator=(WorkingThread &&r) = delete;

        void Run()
        {
            while (!ShouldStop())
            {
                if (m_owner.m_runner.Run() != TaskRunner::RunResult::Ok)
                {
                    break;
                }
            }
        }

        void Stop()
        {
            m_stop.store(true, std::memory_order_relaxed);
            m_owner.m_runner.Stop();
        }

        void Wait()
        {
            if (!m_joinedAlready)
            {
                m_thread.join();
                m_joinedAlready = true;
            }
        }

    private:
        ThreadPool &m_owner;
        std::thread m_thread;
        std::atomic_bool m_stop;
        std::atomic_bool m_joinedAlready; /* have we already .join()'d this thread */

        bool ShouldStop()
        {
            return m_stop.load(std::memory_order_relaxed);
        }
    };

    DcgmNs::TaskRunner m_runner;

    std::vector<WorkingThread> m_threads;
    std::atomic_bool m_shouldStop;
    std::size_t m_numOfWorkers;
};

} // namespace DcgmNs
