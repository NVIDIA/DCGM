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

#include "IoContext.hpp"

#include "SubreaperGuard.hpp"
#include <DcgmLogging.h>

#include <fmt/format.h>

#include <thread>

struct IoContext::Impl
{
    boost::asio::io_context ioContext;
    std::vector<std::jthread> ioThreads;
    std::mutex processCreationMutex;
    DcgmNs::Common::Subprocess::Detail::SubreaperGuard subreaperGuard {};
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> workGuard;

    Impl()
        : workGuard(boost::asio::make_work_guard(ioContext))
    {
        // Allocate the threads upfront to avoid the need to reallocate them later.
        ioThreads.reserve(1);

        // main I/O thread. All above coroutines are run here
        //NOLINTNEXTLINE(*-unnecessary-value-param)
        ioThreads.emplace_back([this](std::stop_token /* do not use reference here */ cancellationToken) {
            char const *threadName = "MainIOContext\0";
            pthread_setname_np(pthread_self(), threadName);
            try
            {
                while (!cancellationToken.stop_requested())
                {
                    ioContext.run_for(boost::asio::chrono::milliseconds(500));
                    ioContext.reset();
                }
            }
            catch (std::exception const &e)
            {
                log_error("{} thread error: {}", threadName, e.what());
            }
            catch (...)
            {
                log_error("{} thread error: unknown", threadName);
            }
            log_debug("{} thread finished", threadName);
        });
    }

    ~Impl()
    {
        for (auto &thread : ioThreads)
        {
            thread.request_stop();
        }

        for (auto &thread : ioThreads)
        {
            // As threads are communicating with each other, we need to wait for them to finish before cleaning
            // the ioThreads vector. Calling ioThreads.clear() before joining the threads would cause a crash.
            if (thread.joinable())
            {
                thread.join();
            }
        }
        ioContext.stop();
        ioThreads.clear();
    }

    boost::asio::io_context &Get()
    {
        return ioContext;
    }

    std::mutex &GetProcessCreationMutex()
    {
        return processCreationMutex;
    }
};

IoContext::IoContext()
    : m_impl(std::make_unique<Impl>())
{}

IoContext::~IoContext() = default;

boost::asio::io_context &IoContext::Get() const
{
    return m_impl->Get();
}

std::mutex &IoContext::GetProcessCreationMutex()
{
    return m_impl->GetProcessCreationMutex();
}

void IoContext::Post(std::function<void()> func)
{
    m_impl->Get().post(func);
}