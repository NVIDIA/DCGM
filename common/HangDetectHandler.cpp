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

#include "HangDetectHandler.h"
#include "HangDetectMonitor.h"

#include <DcgmLogging.h>
#include <fmt/format.h>

namespace
{
std::chrono::milliseconds constexpr waitTime(10);
} // namespace

HangDetectHandler::HangDetectHandler()
{
    m_thread = std::jthread([this](std::stop_token stoken) { this->Run(stoken); });
    m_running.store(true);
}

void HangDetectHandler::Run(std::stop_token stoken)
{
    while (!stoken.stop_requested())
    {
        std::optional<HangDetectedEvent> hangEvent;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_cv.wait_for(lock, waitTime, [this, &stoken] {
                    return stoken.stop_requested() || std::get<0>(m_hangEvent) != 0;
                }))
            {
                if (stoken.stop_requested())
                {
                    break;
                }
                hangEvent   = m_hangEvent;
                m_hangEvent = {};
            }
        }

        if (hangEvent)
        {
            /* Derived classes must catch exceptions that should not propagate to the application. */
            HandleHangDetectedEvent(*hangEvent);
        }
    }
    m_running.store(false);
}

void HangDetectHandler::HandleHangDetectedEvent(HangDetectedEvent const &hangEvent)
{
    if (m_disableDefaultHandler)
    {
        return;
    }

    auto const [pid, tid, duration] = hangEvent;
    auto const entityType           = tid.has_value() ? "thread" : "process";
    auto const entityId             = tid.value_or(pid);
    auto const idType               = tid.has_value() ? "tid" : "pid";

    auto const msg
        = fmt::format("A {} ({}: {}) has been unresponsive for {} seconds. "
                      "If the process does not exit, it may need to be killed or the system may need to be restarted. "
                      "Collect problem report logs before restarting.",
                      entityType,
                      idType,
                      entityId,
                      duration.count());
    log_error(msg);
}

HangDetectHandler::~HangDetectHandler()
{
    m_thread.request_stop();
    m_cv.notify_one();
}
