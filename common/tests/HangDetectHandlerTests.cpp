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

#include "DcgmUtilities.h"
#include "HangDetect.h"
#include "HangDetectHandler.h"
#include "HangDetectMonitor.h"

#include <catch2/catch_all.hpp>
#include <memory>

namespace
{
constexpr std::chrono::milliseconds checkInterval(20);
constexpr std::chrono::milliseconds expiryTime(250);
constexpr std::chrono::milliseconds waitTime(10);
} //namespace

class HangDetectHandlerTest : public HangDetectHandler
{
public:
    HangDetectHandlerTest()
        : HangDetectHandler()
    {}

    void HandleHangDetectedEvent(HangDetectedEvent const & /* hangEvent */) override
    {
        m_handlerCalls++;
    }

    bool IsRunning() const
    {
        return m_running.load();
    }

    void Stop()
    {
        m_thread.request_stop();
        m_cv.notify_one();
    }

    std::atomic<size_t> m_handlerCalls = 0;
};

#if 0 // Test temporarily disabled due to intermittent failures.
TEST_CASE("HangDetectHandler")
{
    auto handler  = HangDetectHandlerTest();
    auto detector = std::make_unique<HangDetect>();
    auto monitor  = std::make_unique<HangDetectMonitor>(*detector, checkInterval, expiryTime);

    monitor->SetHangDetectedHandler(&handler);

    // Verify handler start
    REQUIRE(DcgmNs::Utils::WaitFor([&handler]() { return handler.IsRunning(); }, 2 * waitTime));

    handler.AddHangEvent(1, 2, std::chrono::seconds(3));
    REQUIRE(handler.m_handlerCalls == 0);

    // Observe exactly one handler call
    REQUIRE(DcgmNs::Utils::WaitFor([&handler]() { return handler.m_handlerCalls != 0; }, 2 * waitTime));
    REQUIRE(handler.m_handlerCalls == 1);
    REQUIRE(DcgmNs::Utils::WaitFor([&handler]() { return handler.m_handlerCalls == 1; }, 2 * waitTime));

    // Clear handler and verify no handler is called
    monitor->SetHangDetectedHandler(nullptr);
    handler.AddHangEvent(1, 2, std::chrono::seconds(3));
    REQUIRE(handler.m_handlerCalls == 1);
    REQUIRE(DcgmNs::Utils::WaitFor([&handler]() { return handler.m_handlerCalls == 1; }, 2 * waitTime));

    monitor->StopMonitoring();

    // Verify handler stop
    handler.Stop();
    REQUIRE(DcgmNs::Utils::WaitFor([&handler]() { return !handler.IsRunning(); }, 2 * waitTime));
}
#endif