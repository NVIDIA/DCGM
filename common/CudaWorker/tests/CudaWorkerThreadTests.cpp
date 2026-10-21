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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <CudaWorkerThread.hpp>
#include <MockCudaOps.h>
#include <MockFieldWorkerFactory.hpp>

#include <dcgm_fields.h>

#include <chrono>
#include <memory>

using namespace Catch::Matchers;

namespace
{
bool LoadMatches(double actual, double expected)
{
    return WithinAbs(expected, 0.000001).match(actual);
}

/**
 * Waits until the mock field worker has run at least two duty cycles at
 * expectedLoad. The second cycle guarantees CudaWorkerThread has published the
 * first cycle's achieved load through GetCurrentAchievedLoad().
 */
bool WaitForPublishedLoad(MockFieldWorkerDutyCycleSync &sync, CudaWorkerThread &worker, double expectedLoad)
{
    std::unique_lock lock(sync.mutex);
    if (!sync.cv.wait_for(
            lock, std::chrono::milliseconds(500), [&] { return LoadMatches(sync.lastLoadTarget, expectedLoad); }))
    {
        return false;
    }

    auto const callsAtFirstMatch = sync.dutyCycleCalls;
    if (!sync.cv.wait_for(lock, std::chrono::milliseconds(500), [&] {
            return sync.dutyCycleCalls >= callsAtFirstMatch + 1 && LoadMatches(sync.lastLoadTarget, expectedLoad);
        }))
    {
        return false;
    }

    lock.unlock();
    return LoadMatches(worker.GetCurrentAchievedLoad(), expectedLoad);
}

struct CudaWorkerThreadFixture
{
    MockCudaOps cudaOps;
    std::unique_ptr<MockFieldWorkerFactory> mockFactory = std::make_unique<MockFieldWorkerFactory>(&cudaOps);
    MockFieldWorkerFactory *factoryPtr { mockFactory.get() };
    std::unique_ptr<CudaWorkerThread> worker = std::make_unique<CudaWorkerThread>(&cudaOps, std::move(mockFactory));
};
} // namespace

SCENARIO_METHOD(CudaWorkerThreadFixture, "CudaWorkerThread manages active workload state through public APIs")
{
    GIVEN("an initialized CUDA worker thread with a recording field-worker factory")
    {
        // Concurrent Init/SetWorkload/GetCurrentAchievedLoad access is exercised for ThreadSanitizer.
        REQUIRE(worker->Init(0) == DCGM_ST_OK);

        WHEN("no workload has been selected")
        {
            THEN("achieved load stays idle and the factory has not allocated a worker")
            {
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.0, 0.000001));
                CHECK(factoryPtr->m_createCalls.empty());
            }
        }

        WHEN("a workload is selected")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_PROF_SM_UTIL_RATIO, 0.5, true);

            THEN("the factory allocates the requested field and duty cycles report the target load")
            {
                REQUIRE(factoryPtr->m_createCalls.size() == 1);
                CHECK(factoryPtr->m_createCalls.back().fieldId == DCGM_FI_PROF_SM_UTIL_RATIO);
                CHECK_FALSE(factoryPtr->m_createCalls.back().preferCublas);
                CAPTURE(worker->GetCurrentAchievedLoad());
                REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.5));
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.5, 0.000001));
            }
        }

        WHEN("a datatype workload prefers CUBLAS")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_PROF_FP32_UTIL_RATIO, 0.4, true, true);

            THEN("the factory forwards the CUBLAS preference for the requested field")
            {
                REQUIRE(factoryPtr->m_createCalls.size() == 1);
                CHECK(factoryPtr->m_createCalls.back().fieldId == DCGM_FI_PROF_FP32_UTIL_RATIO);
                CHECK(factoryPtr->m_createCalls.back().preferCublas);
                CAPTURE(worker->GetCurrentAchievedLoad());
                REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.4));
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.4, 0.000001));
            }
        }

        WHEN("the same workload target is updated")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_PROF_SM_UTIL_RATIO, 0.25, true);
            REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.25));
            REQUIRE(factoryPtr->m_createCalls.size() == 1);

            worker->SetWorkloadAndTarget(DCGM_FI_PROF_SM_UTIL_RATIO, 0.75, true);

            THEN("the existing worker remains active and the new target is observed")
            {
                CHECK(factoryPtr->m_createCalls.size() == 1);
                CAPTURE(worker->GetCurrentAchievedLoad());
                REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.75));
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.75, 0.000001));
            }
        }

        WHEN("the same field is retargeted with a different CUBLAS preference")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_PROF_FP32_UTIL_RATIO, 0.40, true, false);
            REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.40));
            REQUIRE(factoryPtr->m_createCalls.size() == 1);
            CHECK_FALSE(factoryPtr->m_createCalls.back().preferCublas);

            worker->SetWorkloadAndTarget(DCGM_FI_PROF_FP32_UTIL_RATIO, 0.80, true, true);

            THEN("the existing field worker is reused and the new load target is observed")
            {
                CHECK(factoryPtr->m_createCalls.size() == 1);
                CAPTURE(worker->GetCurrentAchievedLoad());
                REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.80));
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.80, 0.000001));
            }
        }

        WHEN("the worker is returned to idle")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_PROF_SM_UTIL_RATIO, 0.5, true);
            REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.5));
            auto const createsBeforeIdle = factoryPtr->m_createCalls.size();

            // Idle clears achieved load on the worker thread before
            // blockOnCompletion returns; WaitForPublishedLoad does not apply
            // because no field worker runs after fieldId 0.
            worker->SetWorkloadAndTarget(0, 0.0, true);

            THEN("achieved load clears and no additional worker is allocated")
            {
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.0, 0.000001));
                CHECK(factoryPtr->m_createCalls.size() == createsBeforeIdle);
            }
        }
    }
}

SCENARIO_METHOD(CudaWorkerThreadFixture, "CudaWorkerThread duty cycles update achieved load for active workers")
{
    GIVEN("an initialized CUDA worker thread with a selected SM workload")
    {
        // Concurrent workload replacement and achieved-load reads are exercised for ThreadSanitizer.
        REQUIRE(worker->Init(0) == DCGM_ST_OK);
        worker->SetWorkloadAndTarget(DCGM_FI_PROF_SM_UTIL_RATIO, 0.5, true);
        REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.5));
        REQUIRE(factoryPtr->m_createCalls.size() == 1);

        WHEN("the active field changes to a datatype workload")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_PROF_FP32_UTIL_RATIO, 0.25, true);

            THEN("the active worker is replaced and its achieved load is observed")
            {
                REQUIRE(factoryPtr->m_createCalls.size() == 2);
                CHECK(factoryPtr->m_createCalls.back().fieldId == DCGM_FI_PROF_FP32_UTIL_RATIO);
                CAPTURE(worker->GetCurrentAchievedLoad());
                REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.25));
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.25, 0.000001));
            }
        }

        WHEN("an unsupported active field is selected")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_DEV_GPU_NAME, 0.5, true);

            THEN("no active worker remains and achieved load falls back to idle")
            {
                REQUIRE(factoryPtr->m_createCalls.size() == 2);
                CHECK(factoryPtr->m_createCalls.back().fieldId == DCGM_FI_DEV_GPU_NAME);
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.0, 0.000001));
            }
        }
    }
}

SCENARIO_METHOD(CudaWorkerThreadFixture, "CudaWorkerThread exposes CUDA device metadata from Init")
{
    GIVEN("mocked CUDA device attributes for a known ordinal")
    {
        constexpr int deviceOrdinal               = 7;
        constexpr int multiProcessorCount         = 8;
        constexpr int maxThreadsPerMultiprocessor = 2048;
        constexpr int sharedMemPerMultiprocessor  = 65536;
        constexpr int computeCapabilityMajor      = 8;
        constexpr int computeCapabilityMinor      = 9;
        constexpr int memoryBusWidth              = 5120;
        constexpr int memoryClockRateKhz          = 1215000; // becomes 1215 MHz after Init
        constexpr int eccSupport                  = 1;

        cudaOps.cuDeviceAttributeValues = {
            { CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, maxThreadsPerMultiprocessor },
            { CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, multiProcessorCount },
            { CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, sharedMemPerMultiprocessor },
            { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, computeCapabilityMajor },
            { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, computeCapabilityMinor },
            { CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, memoryBusWidth },
            { CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, memoryClockRateKhz },
            { CU_DEVICE_ATTRIBUTE_ECC_ENABLED, eccSupport },
        };

        WHEN("the worker is initialized against that device")
        {
            REQUIRE(worker->Init(deviceOrdinal) == DCGM_ST_OK);
            auto const &observed = worker->GetCudaWorkerDevice();

            THEN("GetCudaWorkerDevice returns the derived metadata snapshot")
            {
                CHECK(observed.m_device == deviceOrdinal);
                CHECK(observed.m_multiProcessorCount == multiProcessorCount);
                CHECK(observed.m_maxThreadsPerMultiProcessor == maxThreadsPerMultiprocessor);
                CHECK(observed.m_computeCapabilityMajor == computeCapabilityMajor);
                CHECK(observed.m_computeCapabilityMinor == computeCapabilityMinor);
                // Combined capability is stored as int, so 8.9 truncates to 8.
                CHECK(observed.m_computeCapability == 8);
                CHECK(observed.m_maxMemoryClockMhz == 1215);
                CHECK(observed.m_memoryBusWidth == memoryBusWidth);
                CHECK_THAT(observed.m_maxMemBandwidth, WithinAbs(1555200000000.0, 1.0));
                CHECK(observed.m_sharedMemPerMultiprocessor == sharedMemPerMultiprocessor);
                CHECK(observed.m_eccSupport == eccSupport);
                CHECK(observed.m_module != nullptr);
                CHECK(observed.m_cuFuncWaitNs != nullptr);
                CHECK(observed.m_cuFuncDoWorkFP64 != nullptr);
                CHECK(observed.m_cuFuncDoWorkFP32 != nullptr);
                CHECK(observed.m_cuFuncDoWorkFP16 != nullptr);
            }
        }
    }
}

SCENARIO_METHOD(CudaWorkerThreadFixture, "CudaWorkerThread handles unsupported workload replacement deterministically")
{
    GIVEN("an initialized CUDA worker thread")
    {
        // Concurrent workload replacement is exercised for ThreadSanitizer.
        REQUIRE(worker->Init(0) == DCGM_ST_OK);

        WHEN("an unsupported workload is selected after a valid active workload")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_PROF_FP32_UTIL_RATIO, 0.60, true, false);
            REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.60));
            REQUIRE(factoryPtr->m_createCalls.size() == 1);

            worker->SetWorkloadAndTarget(DCGM_FI_DEV_FB_FREE, 0.10, true, false);

            THEN("the factory records the unsupported request and achieved load returns to idle")
            {
                REQUIRE(factoryPtr->m_createCalls.size() == 2);
                CHECK(factoryPtr->m_createCalls.back().fieldId == DCGM_FI_DEV_FB_FREE);
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.0, 0.000001));
            }
        }

        WHEN("a tensor-preferred datatype workload is selected after an unsupported field")
        {
            worker->SetWorkloadAndTarget(DCGM_FI_DEV_FB_FREE, 0.10, true, false);
            REQUIRE(factoryPtr->m_createCalls.size() == 1);
            CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.0, 0.000001));

            worker->SetWorkloadAndTarget(DCGM_FI_PROF_FP64_UTIL_RATIO, 0.35, true, true);

            THEN("a new worker replaces the empty state and publishes achieved load")
            {
                REQUIRE(factoryPtr->m_createCalls.size() == 2);
                CHECK(factoryPtr->m_createCalls.back().fieldId == DCGM_FI_PROF_FP64_UTIL_RATIO);
                CHECK(factoryPtr->m_createCalls.back().preferCublas);
                CAPTURE(worker->GetCurrentAchievedLoad());
                REQUIRE(WaitForPublishedLoad(factoryPtr->m_dutyCycleSync, *worker, 0.35));
                CHECK_THAT(worker->GetCurrentAchievedLoad(), WithinAbs(0.35, 0.000001));
            }
        }
    }
}

SCENARIO_METHOD(CudaWorkerThreadFixture, "CudaWorkerThread Init propagates CUDA attach and module-load failures")
{
    GIVEN("a CUDA worker thread that has not been initialized")
    {
        WHEN("CUDA reports a module function lookup failure")
        {
            cudaOps.cuModuleGetFunctionResult = CUDA_ERROR_INVALID_VALUE;

            THEN("Init fails after device attach but before function handles are populated")
            {
                CHECK(worker->Init(3) == DCGM_ST_GENERIC_ERROR);
                auto const &device = worker->GetCudaWorkerDevice();
                CHECK(device.m_device == 3);
                CHECK(device.m_module != nullptr);
                CHECK(device.m_cuFuncWaitNs == nullptr);
                CHECK(device.m_cuFuncDoWorkFP64 == nullptr);
                CHECK(device.m_cuFuncDoWorkFP32 == nullptr);
                CHECK(device.m_cuFuncDoWorkFP16 == nullptr);
            }
        }

        WHEN("CUDA initialization fails during attach")
        {
            cudaOps.cuInitResult          = CUDA_ERROR_INVALID_VALUE;
            cudaOps.cuGetErrorStringValue = "invalid CUDA init";

            THEN("Init fails before mutating device identity")
            {
                CHECK(worker->Init(4) == DCGM_ST_GENERIC_ERROR);
                CHECK(worker->GetCudaWorkerDevice().m_device == 0);
            }
        }
    }
}
