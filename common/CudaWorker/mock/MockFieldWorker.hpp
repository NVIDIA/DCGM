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

#include <FieldWorkers.hpp>

#include <condition_variable>
#include <mutex>

struct MockFieldWorkerDutyCycleSync
{
    std::mutex mutex;
    std::condition_variable cv;
    unsigned int dutyCycleCalls {};
    double lastLoadTarget {};
};

class MockFieldWorker : public FieldWorkerBase
{
public:
    MockFieldWorker(CudaWorkerDevice_t cudaDevice,
                    CudaOpsAnyPtr cudaOps,
                    unsigned int fieldId,
                    MockFieldWorkerDutyCycleSync *dutyCycleSync)
        : FieldWorkerBase(cudaDevice, cudaOps, fieldId)
        , m_dutyCycleSync(dutyCycleSync)
    {}

    void DoOneDutyCycle(double loadTarget, std::chrono::milliseconds dutyCycleLength) override
    {
        m_loadTarget      = loadTarget;
        m_dutyCycleLength = dutyCycleLength;
        m_achievedLoad    = loadTarget;
        ++m_dutyCycleCalls;

        if (m_dutyCycleSync != nullptr)
        {
            {
                std::lock_guard const lock(m_dutyCycleSync->mutex);
                ++m_dutyCycleSync->dutyCycleCalls;
                m_dutyCycleSync->lastLoadTarget = loadTarget;
            }
            m_dutyCycleSync->cv.notify_all();
        }
    }

    std::chrono::milliseconds m_dutyCycleLength {};
    unsigned int m_dutyCycleCalls {};
    double m_loadTarget {};
    MockFieldWorkerDutyCycleSync *m_dutyCycleSync;
};
