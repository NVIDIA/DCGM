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

#include <DcgmLogging.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <sys/types.h>
#include <thread>
#include <tuple>

class HangDetectHandler
{
public:
    using HangDetectedEvent = std::tuple<pid_t, std::optional<pid_t>, std::chrono::seconds>;

    /**
     * Constructor
     *
     * @note The caller must ensure the handler remains valid while monitoring is active. Instantiate the
     *       handler before instantiating the monitor to ensure this.
     */
    HangDetectHandler();

    HangDetectHandler(HangDetectHandler const &)            = delete;
    HangDetectHandler &operator=(HangDetectHandler const &) = delete;
    HangDetectHandler(HangDetectHandler &&)                 = delete;
    HangDetectHandler &operator=(HangDetectHandler &&)      = delete;

    /**
     * Add a hang event to the handler
     *
     * @param pid The process ID of the hang event
     * @param tid The thread ID of the hang event
     * @param duration The duration of the hang event
     *
     * @note Currently, only one hang event is supported. The most recently added hang event will be reported.
     *       We can replace this with a queue if this becomes a limitation.
     */
    template <typename Rep, typename Period>
    void AddHangEvent(pid_t pid, std::optional<pid_t> tid, std::chrono::duration<Rep, Period> duration)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_hangEvent = { pid, tid, std::chrono::duration_cast<std::chrono::seconds>(duration) };
        }
        m_cv.notify_one();
    }

    /**
     * Disable the default handler
     *
     * @param disable Whether to disable the default handler. If true, the default handler will not be called.
     *                If false, the default handler will be called. The default handler will only be called if
     *                HandleHangDetectedEvent is not overridden.
     */
    void DisableDefaultHandler(bool disable = true)
    {
        m_disableDefaultHandler = disable;
    }

    /**
     * Handle a hang detected event
     *
     * @param hangEvent The hang event to handle
     * @note Override this method to handle hang detected events. The default handler will log a warning.
     */
    virtual void HandleHangDetectedEvent(HangDetectedEvent const &hangEvent);

    virtual ~HangDetectHandler();

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::jthread m_thread;
    HangDetectedEvent m_hangEvent {};
    std::atomic_bool m_running { false };
    bool m_disableDefaultHandler { false };

    void Run(std::stop_token stoken);

    friend class HangDetectHandlerTest;
};
