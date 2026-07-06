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

#include <catch2/catch_all.hpp>

#define CUDA_VERSION_USED 12
#include <FieldWorkers.hpp>

namespace
{
class TestFieldWorker : public FieldWorkerBase
{
public:
    TestFieldWorker(CudaWorkerDevice_t cudaDevice, unsigned int fieldId)
        : FieldWorkerBase(cudaDevice, fieldId)
    {}

    void DoOneDutyCycle(double, std::chrono::milliseconds) override
    {}
};

CudaWorkerDevice_t MakeCudaDevice()
{
    CudaWorkerDevice_t device {};
    device.m_multiProcessorCount         = 8;
    device.m_maxThreadsPerMultiProcessor = 2048;
    return device;
}
} // namespace

TEST_CASE("FieldWorkerBase computes CUDA dimensions")
{
    GIVEN("a test worker with a large per-SM thread capacity")
    {
        TestFieldWorker worker(MakeCudaDevice(), DCGM_FI_PROF_FP32_UTIL_RATIO);

        SECTION("thread counts within one CUDA block are unchanged")
        {
            auto dims = worker.ComputeProperCudaDimensions(7, 512);

            CHECK(dims.gridDim.x == 7);
            CHECK(dims.blockDim.x == 512);
        }

        SECTION("thread counts above CUDA block size are rebalanced into more blocks")
        {
            auto dims = worker.ComputeProperCudaDimensions(3, 1536);

            CHECK(dims.gridDim.x == 48);
            CHECK(dims.blockDim.x == 96);
        }
    }
}

TEST_CASE("FieldWorkerBase validates kernel launch inputs before CUDA calls")
{
    SECTION("sleep kernel rejects zero SMs")
    {
        TestFieldWorker worker(MakeCudaDevice(), DCGM_FI_PROF_SM_UTIL_RATIO);

        CHECK(worker.RunSleepKernel(0, 1, 10) == DCGM_ST_BADPARAM);
    }

    SECTION("work kernel rejects zero SMs")
    {
        TestFieldWorker worker(MakeCudaDevice(), DCGM_FI_PROF_FP32_UTIL_RATIO);

        CHECK(worker.RunDoWorkKernel(0, 1, 10) == DCGM_ST_BADPARAM);
    }

    SECTION("work kernel rejects SM counts above the device limit")
    {
        auto device = MakeCudaDevice();
        TestFieldWorker worker(device, DCGM_FI_PROF_FP32_UTIL_RATIO);

        CHECK(worker.RunDoWorkKernel(device.m_multiProcessorCount + 1, 1, 10) == DCGM_ST_BADPARAM);
    }

    SECTION("work kernel rejects unsupported field ids")
    {
        TestFieldWorker worker(MakeCudaDevice(), DCGM_FI_PROF_SM_UTIL_RATIO);

        CHECK(worker.RunDoWorkKernel(1, 1, 10) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("FieldWorkerBase launches supported work kernels with CUDA stubs")
{
    GIVEN("a test worker with stubbed CUDA functions")
    {
        for (auto fieldId : { static_cast<unsigned int>(DCGM_FI_PROF_FP16_UTIL_RATIO),
                              static_cast<unsigned int>(DCGM_FI_PROF_FP32_UTIL_RATIO),
                              static_cast<unsigned int>(DCGM_FI_PROF_FP64_UTIL_RATIO) })
        {
            CAPTURE(fieldId);
            TestFieldWorker worker(MakeCudaDevice(), fieldId);

            WHEN("a valid work kernel is launched")
            {
                THEN("the helper reports success")
                {
                    CHECK(worker.GetFieldId() == fieldId);
                    CHECK(worker.GetAchievedLoad() == 0.0);
                    CHECK(worker.RunDoWorkKernel(1, 1, 10) == DCGM_ST_OK);
                }
            }
        }
    }
}

TEST_CASE("FieldWorker duty cycles update achieved load")
{
    GIVEN("activity workers with a stubbed CUDA device")
    {
        auto device = MakeCudaDevice();

        SECTION("graphics activity handles idle and active loads")
        {
            FieldWorkerGrActivity worker(device);

            WHEN("the target load is zero")
            {
                worker.DoOneDutyCycle(0.0, std::chrono::milliseconds(0));

                THEN("the worker records the requested load")
                {
                    CHECK(worker.GetFieldId() == DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);
                    CHECK(worker.GetAchievedLoad() == 0.0);
                }
            }

            WHEN("the target load is positive")
            {
                worker.DoOneDutyCycle(0.25, std::chrono::milliseconds(1));

                THEN("the worker records the requested load after the stubbed launch")
                {
                    CHECK(worker.GetAchievedLoad() == 0.25);
                }
            }
        }

        SECTION("SM activity sleeps for tiny loads and clamps oversized loads")
        {
            FieldWorkerSmActivity worker(device);

            WHEN("the target load rounds to zero SMs")
            {
                worker.DoOneDutyCycle(0.0, std::chrono::milliseconds(0));

                THEN("no achieved load is recorded")
                {
                    CHECK(worker.GetFieldId() == DCGM_FI_PROF_SM_UTIL_RATIO);
                    CHECK(worker.GetAchievedLoad() == 0.0);
                }
            }

            WHEN("the target load exceeds the device SM count")
            {
                worker.DoOneDutyCycle(2.0, std::chrono::milliseconds(1));

                THEN("the achieved load is clamped to the full device")
                {
                    CHECK(worker.GetAchievedLoad() == 1.0);
                }
            }
        }

        SECTION("SM occupancy sleeps for tiny loads and records active loads")
        {
            FieldWorkerSmOccupancy worker(device);

            WHEN("the target load rounds to zero threads")
            {
                worker.DoOneDutyCycle(0.0, std::chrono::milliseconds(0));

                THEN("no achieved load is recorded")
                {
                    CHECK(worker.GetFieldId() == DCGM_FI_PROF_SM_OCCUPANCY_RATIO);
                    CHECK(worker.GetAchievedLoad() == 0.0);
                }
            }

            WHEN("the target requires more than one block per SM")
            {
                worker.DoOneDutyCycle(0.75, std::chrono::milliseconds(1));

                THEN("the requested load is recorded")
                {
                    CHECK(worker.GetAchievedLoad() == 0.75);
                }
            }
        }
    }
}
