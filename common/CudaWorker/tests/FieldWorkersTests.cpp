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
#include <catch2/generators/catch_generators.hpp>

#include <FieldWorkerFactory.hpp>
#include <FieldWorkers.hpp>
#include <MockCublasOps.h>
#include <MockCudaOps.h>

#include <cmath>
#include <limits>

namespace
{
class TestFieldWorker : public FieldWorkerBase
{
public:
    TestFieldWorker(CudaWorkerDevice_t cudaDevice, CudaOpsAnyPtr cudaOps, unsigned int fieldId)
        : FieldWorkerBase(cudaDevice, cudaOps, fieldId)
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

class FieldWorkersTestsFixture
{
public:
    MockCublasOps cublasOps;
    MockCudaOps cudaOps;
    CudaWorkerDevice_t cudaDevice { MakeCudaDevice() };
};
} // namespace

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorkerFactory allocates field workers without starting CUDA threads")
{
    GIVEN("a field-worker factory with mocked CUDA and CUBLAS ops")
    {
        FieldWorkerFactory factory(&cudaOps, &cublasOps);
        auto const &device = cudaDevice;

        // Avoid FieldWorkerTensorActivity / PCIe / DRAM / NvLink Create paths:
        // their ctors allocate ~100MB+ and fill large host matrices, which
        // exceeds the ASan CTest timeout.
        WHEN("a lightweight profiling field is allocated")
        {
            auto const fieldId = GENERATE(as<unsigned int> {},
                                          DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO,
                                          DCGM_FI_PROF_SM_UTIL_RATIO,
                                          DCGM_FI_PROF_SM_OCCUPANCY_RATIO,
                                          DCGM_FI_PROF_FP16_UTIL_RATIO,
                                          DCGM_FI_PROF_FP32_UTIL_RATIO,
                                          DCGM_FI_PROF_FP64_UTIL_RATIO);
            CAPTURE(fieldId);
            auto fieldWorker = factory.Create(device, {}, fieldId, false);

            THEN("a concrete worker is created for the requested field")
            {
                REQUIRE(fieldWorker != nullptr);
                CHECK(fieldWorker->GetFieldId() == fieldId);
                CHECK(cudaOps.cudaSetDeviceCallCount == 0);
                switch (fieldId)
                {
                    case DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO:
                        CHECK(dynamic_cast<FieldWorkerGrActivity *>(fieldWorker.get()) != nullptr);
                        break;
                    case DCGM_FI_PROF_SM_UTIL_RATIO:
                        CHECK(dynamic_cast<FieldWorkerSmActivity *>(fieldWorker.get()) != nullptr);
                        break;
                    case DCGM_FI_PROF_SM_OCCUPANCY_RATIO:
                        CHECK(dynamic_cast<FieldWorkerSmOccupancy *>(fieldWorker.get()) != nullptr);
                        break;
                    case DCGM_FI_PROF_FP16_UTIL_RATIO:
                    case DCGM_FI_PROF_FP32_UTIL_RATIO:
                    case DCGM_FI_PROF_FP64_UTIL_RATIO:
                        CHECK(dynamic_cast<FieldWorkerDataTypeActivity *>(fieldWorker.get()) != nullptr);
                        break;
                    default:
                        FAIL("unexpected field id in lightweight profiling GENERATE");
                        break;
                }
            }
        }

        WHEN("an unsupported field is requested")
        {
            auto const unsupportedWorker = factory.Create(device, {}, DCGM_FI_DEV_GPU_NAME, false);
            auto const invalidWorker     = factory.Create(device, {}, std::numeric_limits<unsigned int>::max(), false);
            auto const idleWorker        = factory.Create(device, {}, 0, false);

            THEN("no worker is created")
            {
                CHECK(unsupportedWorker == nullptr);
                CHECK(invalidWorker == nullptr);
                CHECK(idleWorker == nullptr);
            }
        }

        WHEN("a datatype field is requested without CUBLAS preference")
        {
            auto const fieldId = GENERATE(as<unsigned int> {},
                                          DCGM_FI_PROF_FP16_UTIL_RATIO,
                                          DCGM_FI_PROF_FP32_UTIL_RATIO,
                                          DCGM_FI_PROF_FP64_UTIL_RATIO);
            CAPTURE(fieldId);
            auto fieldWorker = factory.Create(device, {}, fieldId, false);

            THEN("a lightweight datatype activity worker is created")
            {
                REQUIRE(fieldWorker != nullptr);
                CHECK(fieldWorker->GetFieldId() == fieldId);
                CHECK(dynamic_cast<FieldWorkerDataTypeActivity *>(fieldWorker.get()) != nullptr);
                CHECK(cudaOps.cudaSetDeviceCallCount == 0);
            }
        }
    }
}


TEST_CASE_METHOD(FieldWorkersTestsFixture, "FieldWorkerBase computes CUDA dimensions")
{
    GIVEN("a test worker with a large per-SM thread capacity")
    {
        TestFieldWorker worker(cudaDevice, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

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


TEST_CASE_METHOD(FieldWorkersTestsFixture, "FieldWorkerBase validates kernel launch inputs before CUDA calls")
{
    SECTION("sleep kernel rejects zero SMs")
    {
        TestFieldWorker worker(cudaDevice, &cudaOps, DCGM_FI_PROF_SM_UTIL_RATIO);

        CHECK(worker.RunSleepKernel(0, 1, 10) == DCGM_ST_BADPARAM);
    }

    SECTION("work kernel rejects zero SMs")
    {
        TestFieldWorker worker(cudaDevice, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

        CHECK(worker.RunDoWorkKernel(0, 1, 10) == DCGM_ST_BADPARAM);
    }

    SECTION("work kernel rejects SM counts above the device limit")
    {
        auto const &device = cudaDevice;
        TestFieldWorker worker(device, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

        CHECK(worker.RunDoWorkKernel(device.m_multiProcessorCount + 1, 1, 10) == DCGM_ST_BADPARAM);
    }

    SECTION("work kernel rejects unsupported field ids")
    {
        TestFieldWorker worker(cudaDevice, &cudaOps, DCGM_FI_PROF_SM_UTIL_RATIO);

        CHECK(worker.RunDoWorkKernel(1, 1, 10) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE_METHOD(FieldWorkersTestsFixture, "FieldWorkerBase launches supported work kernels with mocked CUDA")
{
    GIVEN("a test worker with mocked CUDA functions")
    {
        for (auto fieldId : { static_cast<unsigned int>(DCGM_FI_PROF_FP16_UTIL_RATIO),
                              static_cast<unsigned int>(DCGM_FI_PROF_FP32_UTIL_RATIO),
                              static_cast<unsigned int>(DCGM_FI_PROF_FP64_UTIL_RATIO) })
        {
            CAPTURE(fieldId);
            TestFieldWorker worker(cudaDevice, &cudaOps, fieldId);

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

TEST_CASE_METHOD(FieldWorkersTestsFixture, "FieldWorker duty cycles update achieved load")
{
    GIVEN("activity workers with a mocked CUDA device")
    {
        auto const &device = cudaDevice;

        SECTION("graphics activity handles idle and active loads")
        {
            FieldWorkerGrActivity worker(device, &cudaOps);

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

                THEN("the worker records the requested load after the mocked launch")
                {
                    CHECK(worker.GetAchievedLoad() == 0.25);
                }
            }
        }

        SECTION("SM activity sleeps for tiny loads and clamps oversized loads")
        {
            FieldWorkerSmActivity worker(device, &cudaOps);

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
            FieldWorkerSmOccupancy worker(device, &cudaOps);

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

TEST_CASE_METHOD(FieldWorkersTestsFixture, "FieldWorkerBase covers boundary dimensions and sleep-kernel overflow")
{
    GIVEN("test workers with representative CUDA device limits")
    {
        WHEN("the requested threads exactly match the CUDA block limit")
        {
            auto device                          = MakeCudaDevice();
            device.m_maxThreadsPerMultiProcessor = 1024;
            TestFieldWorker worker(device, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

            auto const dims = worker.ComputeProperCudaDimensions(5, 1024);

            THEN("no rebalancing is needed")
            {
                CHECK(dims.gridDim.x == 5);
                CHECK(dims.blockDim.x == 1024);
            }
        }

        WHEN("the requested threads are just over the CUDA block limit")
        {
            auto device                          = MakeCudaDevice();
            device.m_maxThreadsPerMultiProcessor = 2048;
            TestFieldWorker worker(device, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

            auto const dims = worker.ComputeProperCudaDimensions(2, 1025);

            THEN("the helper preserves the load ratio by increasing grid blocks")
            {
                CHECK(dims.gridDim.x == 32);
                CHECK(dims.blockDim.x == 64);
            }
        }

        WHEN("the sleep kernel duration overflows the 32-bit nanosecond parameter")
        {
            TestFieldWorker worker(MakeCudaDevice(), &cudaOps, DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);

            auto const result = worker.RunSleepKernel(1, 1, std::numeric_limits<unsigned int>::max());

            THEN("the launch is clamped internally and still reported through the CUDA result")
            {
                CHECK(result == DCGM_ST_OK);
            }
        }
    }
}

TEST_CASE_METHOD(FieldWorkersTestsFixture, "FieldWorkerDataTypeActivity duty cycles cover supported datatype fields")
{
    GIVEN("datatype workers backed by CUDA launch stubs")
    {
        auto device = MakeCudaDevice();

        for (auto fieldId : { static_cast<unsigned int>(DCGM_FI_PROF_FP16_UTIL_RATIO),
                              static_cast<unsigned int>(DCGM_FI_PROF_FP32_UTIL_RATIO),
                              static_cast<unsigned int>(DCGM_FI_PROF_FP64_UTIL_RATIO) })
        {
            CAPTURE(fieldId);

            WHEN("the target load rounds to zero threads")
            {
                FieldWorkerDataTypeActivity worker(device, &cudaOps, fieldId);

                worker.DoOneDutyCycle(0.0, std::chrono::milliseconds(0));

                THEN("the worker remains idle and keeps its field identity")
                {
                    CHECK(worker.GetFieldId() == fieldId);
                    CHECK(worker.GetAchievedLoad() == 0.0);
                }
            }

            WHEN("the target load is active")
            {
                FieldWorkerDataTypeActivity worker(device, &cudaOps, fieldId);

                worker.DoOneDutyCycle(0.5, std::chrono::milliseconds(1));

                THEN("the worker records the requested load after the mocked launch")
                {
                    CHECK(worker.GetFieldId() == fieldId);
                    CHECK(worker.GetAchievedLoad() == 0.5);
                }
            }

            WHEN("the target load exceeds the device thread capacity")
            {
                FieldWorkerDataTypeActivity worker(device, &cudaOps, fieldId);

                worker.DoOneDutyCycle(2.0, std::chrono::milliseconds(1));

                THEN("the public achieved load still reflects the requested contract input")
                {
                    CHECK(worker.GetAchievedLoad() == 2.0);
                }
            }
        }
    }
}

TEST_CASE_METHOD(FieldWorkersTestsFixture, "FieldWorker base getters expose immutable construction state")
{
    GIVEN("a worker constructed for a profiling field")
    {
        auto device                     = MakeCudaDevice();
        device.m_computeCapabilityMajor = 8;
        device.m_computeCapabilityMinor = 9;
        TestFieldWorker worker(device, &cudaOps, DCGM_FI_PROF_FP64_UTIL_RATIO);

        WHEN("the worker is queried before any duty cycle")
        {
            THEN("the field id and initial achieved load are observable")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_FP64_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.0);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorkerBase covers launch boundary combinations")
{
    GIVEN("a test worker for each supported datatype workload")
    {
        auto device = MakeCudaDevice();

        for (auto fieldId : { static_cast<unsigned int>(DCGM_FI_PROF_FP16_UTIL_RATIO),
                              static_cast<unsigned int>(DCGM_FI_PROF_FP32_UTIL_RATIO),
                              static_cast<unsigned int>(DCGM_FI_PROF_FP64_UTIL_RATIO) })
        {
            CAPTURE(fieldId);
            TestFieldWorker worker(device, &cudaOps, fieldId);

            WHEN("the launch uses every SM at the device thread limit")
            {
                auto const result
                    = worker.RunDoWorkKernel(device.m_multiProcessorCount, device.m_maxThreadsPerMultiProcessor, 1);

                THEN("the CUDA stubbed launch succeeds")
                {
                    CHECK(result == DCGM_ST_OK);
                    CHECK(worker.GetAchievedLoad() == 0.0);
                }
            }

            WHEN("the launch has zero threads per SM")
            {
                auto const result = worker.RunDoWorkKernel(1, 0, 1);

                THEN("the public helper forwards the request to the CUDA layer")
                {
                    CHECK(result == DCGM_ST_OK);
                }
            }
        }
    }

    GIVEN("a sleep worker with legal and illegal SM counts")
    {
        auto device = MakeCudaDevice();
        TestFieldWorker worker(device, &cudaOps, DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);

        WHEN("the sleep kernel spans the full device")
        {
            auto const result = worker.RunSleepKernel(device.m_multiProcessorCount, 1, 1);

            THEN("the launch succeeds through the CUDA stubs")
            {
                CHECK(result == DCGM_ST_OK);
            }
        }

        WHEN("the sleep kernel requests multiple blocks per SM")
        {
            auto const result = worker.RunSleepKernel(1, device.m_maxThreadsPerMultiProcessor, 1);

            THEN("dimension rebalancing still produces a successful launch")
            {
                CHECK(result == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorker duty-cycle edge loads are observable")
{
    GIVEN("SM occupancy workers on devices with different thread capacities")
    {
        auto twoBlockDevice                          = MakeCudaDevice();
        twoBlockDevice.m_maxThreadsPerMultiProcessor = 2048;

        WHEN("occupancy asks for a value above full occupancy")
        {
            FieldWorkerSmOccupancy worker(twoBlockDevice, &cudaOps);

            worker.DoOneDutyCycle(1.5, std::chrono::milliseconds(1));

            THEN("the thread launch is clamped but the requested target remains observable")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_SM_OCCUPANCY_RATIO);
                CHECK(worker.GetAchievedLoad() == 1.5);
            }
        }

        WHEN("occupancy uses the exact CUDA block limit")
        {
            auto singleBlockDevice                          = MakeCudaDevice();
            singleBlockDevice.m_maxThreadsPerMultiProcessor = 1024;
            FieldWorkerSmOccupancy worker(singleBlockDevice, &cudaOps);

            worker.DoOneDutyCycle(1.0, std::chrono::milliseconds(1));

            THEN("the target is recorded after a one-block-per-SM launch")
            {
                CHECK(worker.GetAchievedLoad() == 1.0);
            }
        }
    }

    GIVEN("datatype activity workers on a stubbed CUDA device")
    {
        auto device = MakeCudaDevice();

        WHEN("a fractional FP32 load truncates to an integer thread count")
        {
            FieldWorkerDataTypeActivity worker(device, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

            worker.DoOneDutyCycle(0.333, std::chrono::milliseconds(1));

            THEN("the requested load remains the public achieved-load contract")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_FP32_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.333);
            }
        }

        WHEN("a tiny non-zero FP64 load rounds below one thread")
        {
            FieldWorkerDataTypeActivity worker(device, &cudaOps, DCGM_FI_PROF_FP64_UTIL_RATIO);

            worker.DoOneDutyCycle(0.0001, std::chrono::milliseconds(0));

            THEN("the worker stays idle")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_FP64_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.0);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorkerBase dimension generation covers common GPU thread capacities")
{
    GIVEN("a test worker on devices with representative thread-per-SM limits")
    {
        struct DimensionCase
        {
            unsigned int maxThreadsPerSm;
            unsigned int desiredSms;
            unsigned int desiredThreadsPerSm;
            unsigned int expectedGrid;
            unsigned int expectedBlock;
        };

        auto const testCase = GENERATE(values({ DimensionCase { 1024, 3, 1024, 3, 1024 },
                                                DimensionCase { 1536, 3, 1536, 36, 128 },
                                                DimensionCase { 2048, 3, 2048, 48, 128 },
                                                DimensionCase { 2048, 5, 512, 5, 512 } }));
        CAPTURE(testCase.maxThreadsPerSm);
        CAPTURE(testCase.desiredSms);
        CAPTURE(testCase.desiredThreadsPerSm);

        auto device                          = MakeCudaDevice();
        device.m_maxThreadsPerMultiProcessor = testCase.maxThreadsPerSm;
        TestFieldWorker worker(device, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

        WHEN("CUDA dimensions are computed for the workload")
        {
            auto const dims = worker.ComputeProperCudaDimensions(testCase.desiredSms, testCase.desiredThreadsPerSm);

            THEN("the block limit is respected while preserving the requested load ratio")
            {
                CHECK(dims.gridDim.x == testCase.expectedGrid);
                CHECK(dims.blockDim.x == testCase.expectedBlock);
                CHECK(dims.blockDim.x <= 1024);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture,
                "FieldWorker activity implementations preserve observable load contracts at boundaries")
{
    GIVEN("workers on a stubbed CUDA device")
    {
        auto device = MakeCudaDevice();

        WHEN("graphics activity is asked to run a full duty cycle")
        {
            FieldWorkerGrActivity worker(device, &cudaOps);

            worker.DoOneDutyCycle(1.0, std::chrono::milliseconds(1));

            THEN("the requested graphics load is recorded")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 1.0);
            }
        }

        WHEN("SM activity receives a load just below one whole SM")
        {
            FieldWorkerSmActivity worker(device, &cudaOps);

            worker.DoOneDutyCycle(0.1249, std::chrono::milliseconds(0));

            THEN("the integer scheduler keeps the worker idle")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_SM_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.0);
            }
        }

        WHEN("SM occupancy requires splitting work across two blocks per SM")
        {
            FieldWorkerSmOccupancy worker(device, &cudaOps);
            double const loadTarget = 1025.0 / static_cast<double>(device.m_maxThreadsPerMultiProcessor);

            worker.DoOneDutyCycle(loadTarget, std::chrono::milliseconds(1));

            THEN("the requested occupancy target remains observable after launch adjustment")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_SM_OCCUPANCY_RATIO);
                CHECK(worker.GetAchievedLoad() == loadTarget);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorkerDataTypeActivity covers unsupported and tiny datatype loads")
{
    GIVEN("datatype activity workers with stubbed CUDA launches")
    {
        auto device = MakeCudaDevice();

        WHEN("an unsupported field id receives a positive datatype load")
        {
            FieldWorkerDataTypeActivity worker(device, &cudaOps, DCGM_FI_DEV_GPU_NAME);

            worker.DoOneDutyCycle(0.5, std::chrono::milliseconds(1));

            THEN("the worker still records the requested public load target")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_DEV_GPU_NAME);
                CHECK(worker.GetAchievedLoad() == 0.5);
            }
        }

        WHEN("the smallest non-zero FP16 target still rounds below one thread")
        {
            FieldWorkerDataTypeActivity worker(device, &cudaOps, DCGM_FI_PROF_FP16_UTIL_RATIO);

            worker.DoOneDutyCycle(1.0 / 20000.0, std::chrono::milliseconds(0));

            THEN("no achieved load is reported")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_FP16_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.0);
            }
        }
    }
}


SCENARIO_METHOD(FieldWorkersTestsFixture,
                "FieldWorkerBase computes dimensions for degenerate but accepted launch inputs")
{
    GIVEN("a base worker on representative CUDA devices")
    {
        struct DimensionCase
        {
            unsigned int maxThreadsPerSm;
            unsigned int desiredSms;
            unsigned int desiredThreadsPerSm;
            unsigned int expectedGrid;
            unsigned int expectedBlock;
        };

        auto const testCase = GENERATE(values({ DimensionCase { 1024, 0, 512, 0, 512 },
                                                DimensionCase { 1536, 0, 1536, 0, 128 },
                                                DimensionCase { 2048, 1, 1024, 1, 1024 },
                                                DimensionCase { 2048, 1, 1025, 16, 64 },
                                                DimensionCase { 2048, 8, 2047, 128, 127 } }));
        CAPTURE(testCase.maxThreadsPerSm);
        CAPTURE(testCase.desiredSms);
        CAPTURE(testCase.desiredThreadsPerSm);

        auto device                          = MakeCudaDevice();
        device.m_maxThreadsPerMultiProcessor = testCase.maxThreadsPerSm;
        TestFieldWorker worker(device, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

        WHEN("CUDA dimensions are computed")
        {
            auto const dims = worker.ComputeProperCudaDimensions(testCase.desiredSms, testCase.desiredThreadsPerSm);

            THEN("the helper returns the documented grid and block mapping")
            {
                CHECK(dims.gridDim.x == testCase.expectedGrid);
                CHECK(dims.blockDim.x == testCase.expectedBlock);
                CHECK(dims.blockDim.x <= 1024);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture,
                "FieldWorker launch helpers keep CUDA-stub errors observable through return codes")
{
    GIVEN("workers for supported and unsupported field ids")
    {
        auto device = MakeCudaDevice();

        WHEN("sleep launches use zero threads but at least one block")
        {
            TestFieldWorker worker(device, &cudaOps, DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);
            auto const result = worker.RunSleepKernel(1, 0, 1);

            THEN("the request reaches the CUDA stub and succeeds")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(worker.GetAchievedLoad() == 0.0);
            }
        }

        WHEN("datatype launches cover each supported kernel function at one thread")
        {
            THEN("each field id returns success from the CUDA stub")
            {
                for (auto fieldId : { static_cast<unsigned int>(DCGM_FI_PROF_FP16_UTIL_RATIO),
                                      static_cast<unsigned int>(DCGM_FI_PROF_FP32_UTIL_RATIO),
                                      static_cast<unsigned int>(DCGM_FI_PROF_FP64_UTIL_RATIO) })
                {
                    CAPTURE(fieldId);
                    TestFieldWorker worker(device, &cudaOps, fieldId);
                    CHECK(worker.RunDoWorkKernel(1, 1, 0) == DCGM_ST_OK);
                    CHECK(worker.GetFieldId() == fieldId);
                }
            }
        }

        WHEN("an unsupported field id is launched with an otherwise valid work request")
        {
            TestFieldWorker worker(device, &cudaOps, DCGM_FI_DEV_BOARD_POWER_WATTS);
            auto const result = worker.RunDoWorkKernel(1, 1, 1);

            THEN("the field dispatch rejects the request before reporting success")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(worker.GetFieldId() == DCGM_FI_DEV_BOARD_POWER_WATTS);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorker duty cycles cover exact integer scheduling boundaries")
{
    GIVEN("SM and datatype workers on an eight-SM device")
    {
        auto device = MakeCudaDevice();

        WHEN("SM activity maps exactly one SM")
        {
            FieldWorkerSmActivity worker(device, &cudaOps);
            worker.DoOneDutyCycle(0.125, std::chrono::milliseconds(1));

            THEN("the achieved load reflects one scheduled SM")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_SM_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.125);
            }
        }

        WHEN("SM activity maps exactly all SMs")
        {
            FieldWorkerSmActivity worker(device, &cudaOps);
            worker.DoOneDutyCycle(1.0, std::chrono::milliseconds(1));

            THEN("the achieved load reaches full device utilization")
            {
                CHECK(worker.GetAchievedLoad() == 1.0);
            }
        }

        WHEN("SM occupancy maps exactly one thread per SM")
        {
            FieldWorkerSmOccupancy worker(device, &cudaOps);
            auto const loadTarget = 1.0 / static_cast<double>(device.m_maxThreadsPerMultiProcessor);
            worker.DoOneDutyCycle(loadTarget, std::chrono::milliseconds(1));

            THEN("the tiny non-zero occupancy target is recorded")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_SM_OCCUPANCY_RATIO);
                CHECK(worker.GetAchievedLoad() == loadTarget);
            }
        }

        WHEN("datatype activity maps exactly one thread per SM")
        {
            FieldWorkerDataTypeActivity worker(device, &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);
            auto const loadTarget = 1.0 / static_cast<double>(device.m_maxThreadsPerMultiProcessor);
            worker.DoOneDutyCycle(loadTarget, std::chrono::milliseconds(1));

            THEN("the requested datatype target is recorded")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_FP32_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == loadTarget);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorker launch helpers propagate CUDA stub launch failures")
{
    GIVEN("the CUDA mock is configured to fail kernel launches")
    {
        cudaOps.cuLaunchKernelResult = CUDA_ERROR_INVALID_VALUE;

        WHEN("a sleep workload is launched")
        {
            TestFieldWorker worker(MakeCudaDevice(), &cudaOps, DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);

            THEN("the CUDA failure is surfaced as a generic DCGM error")
            {
                CHECK(worker.RunSleepKernel(1, 1, 1) == DCGM_ST_GENERIC_ERROR);
                CHECK(worker.GetAchievedLoad() == 0.0);
            }
        }

        WHEN("a datatype workload is launched")
        {
            TestFieldWorker worker(MakeCudaDevice(), &cudaOps, DCGM_FI_PROF_FP32_UTIL_RATIO);

            THEN("the CUDA failure is surfaced as a generic DCGM error")
            {
                CHECK(worker.RunDoWorkKernel(1, 1, 1) == DCGM_ST_GENERIC_ERROR);
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_FP32_UTIL_RATIO);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorker concrete duty cycles keep state stable when CUDA launch fails")
{
    GIVEN("the CUDA mock is configured to fail kernel launches")
    {
        cudaOps.cuLaunchKernelResult = CUDA_ERROR_INVALID_VALUE;
        auto const device            = MakeCudaDevice();

        WHEN("graphics activity attempts an active duty cycle")
        {
            FieldWorkerGrActivity worker(device, &cudaOps);
            worker.DoOneDutyCycle(0.5, std::chrono::milliseconds(1));

            THEN("the worker keeps the requested public load even though the launch failed")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.5);
            }
        }

        WHEN("SM activity attempts an active duty cycle")
        {
            FieldWorkerSmActivity worker(device, &cudaOps);
            worker.DoOneDutyCycle(0.5, std::chrono::milliseconds(1));

            THEN("the failed launch does not corrupt the computed load bookkeeping")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_SM_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.5);
            }
        }

        WHEN("datatype activity attempts an active duty cycle")
        {
            FieldWorkerDataTypeActivity worker(device, &cudaOps, DCGM_FI_PROF_FP16_UTIL_RATIO);
            worker.DoOneDutyCycle(0.25, std::chrono::milliseconds(1));

            THEN("the selected field and requested load remain observable")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_FP16_UTIL_RATIO);
                CHECK(worker.GetAchievedLoad() == 0.25);
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture,
                "FieldWorker memory-copy workers construct and handle short duty cycles with CUDA stubs")
{
    GIVEN("a CUDA mock that backs host allocations")
    {
        cudaOps.cudaStubAllocateMemory = true;
        auto device                    = MakeCudaDevice();
        device.m_context               = reinterpret_cast<CUcontext>(0x7400);

        WHEN("PCIe RX and TX workers run short duty cycles")
        {
            FieldWorkerPciRxTxBytes rxWorker(device, &cudaOps, DCGM_FI_PROF_PCIE_RX_BYTES);
            FieldWorkerPciRxTxBytes txWorker(device, &cudaOps, DCGM_FI_PROF_PCIE_TX_BYTES);

            rxWorker.DoOneDutyCycle(1.0, std::chrono::milliseconds(1));
            txWorker.DoOneDutyCycle(1.0, std::chrono::milliseconds(1));

            THEN("the workers remain valid and publish finite zero-transfer loads")
            {
                CHECK(rxWorker.GetFieldId() == DCGM_FI_PROF_PCIE_RX_BYTES);
                CHECK(txWorker.GetFieldId() == DCGM_FI_PROF_PCIE_TX_BYTES);
                CHECK(std::isfinite(rxWorker.GetAchievedLoad()));
                CHECK(std::isfinite(txWorker.GetAchievedLoad()));
            }
        }

        WHEN("DRAM utilization runs a short duty cycle")
        {
            FieldWorkerDramUtil worker(device, &cudaOps);

            worker.DoOneDutyCycle(1.0, std::chrono::milliseconds(1));

            THEN("the worker keeps its field identity and returns without real CUDA work")
            {
                CHECK(worker.GetFieldId() == DCGM_FI_PROF_DRAM_UTIL_RATIO);
                CHECK(std::isfinite(worker.GetAchievedLoad()));
            }
        }

        WHEN("NVLink RX and TX workers run short duty cycles")
        {
            FieldWorkerNvLinkRwBytes rxWorker(device, &cudaOps, DCGM_FI_PROF_NVLINK_RX_BYTES, "0000:02:00.0");
            FieldWorkerNvLinkRwBytes txWorker(device, &cudaOps, DCGM_FI_PROF_NVLINK_TX_BYTES, "0000:03:00.0");

            rxWorker.DoOneDutyCycle(1.0, std::chrono::milliseconds(1));
            txWorker.DoOneDutyCycle(1.0, std::chrono::milliseconds(1));

            THEN("the peer-copy workers stay observable through their field ids")
            {
                CHECK(rxWorker.GetFieldId() == DCGM_FI_PROF_NVLINK_RX_BYTES);
                CHECK(txWorker.GetFieldId() == DCGM_FI_PROF_NVLINK_TX_BYTES);
                CHECK(std::isfinite(rxWorker.GetAchievedLoad()));
                CHECK(std::isfinite(txWorker.GetAchievedLoad()));
            }
        }
    }
}

SCENARIO_METHOD(FieldWorkersTestsFixture, "FieldWorker memory-copy constructors surface CUDA setup failures")
{
    GIVEN("a CUDA mock configured to fail allocation or peer-context setup")
    {
        auto device      = MakeCudaDevice();
        device.m_context = reinterpret_cast<CUcontext>(0x7400);

        WHEN("PCIe worker host allocation fails")
        {
            cudaOps.cuMemAllocResult = CUDA_ERROR_OUT_OF_MEMORY;

            THEN("construction reports the CUDA setup failure")
            {
                CHECK_THROWS_AS(FieldWorkerPciRxTxBytes(device, &cudaOps, DCGM_FI_PROF_PCIE_RX_BYTES),
                                std::runtime_error);
            }
        }

        WHEN("DRAM worker device allocation fails")
        {
            cudaOps.cuMemAllocResult = CUDA_ERROR_OUT_OF_MEMORY;

            THEN("construction reports the CUDA setup failure")
            {
                CHECK_THROWS_AS(FieldWorkerDramUtil(device, &cudaOps), std::runtime_error);
            }
        }

        WHEN("NVLink worker cannot resolve the peer bus id")
        {
            cudaOps.cuDeviceGetByPCIBusIdResult = CUDA_ERROR_INVALID_VALUE;

            THEN("construction reports the peer lookup failure")
            {
                CHECK_THROWS_AS(FieldWorkerNvLinkRwBytes(device, &cudaOps, DCGM_FI_PROF_NVLINK_RX_BYTES, "bad-bus"),
                                std::runtime_error);
            }
        }

        WHEN("NVLink worker cannot create the peer context")
        {
            cudaOps.cuCtxCreateResult = CUDA_ERROR_INVALID_VALUE;

            THEN("construction reports the context setup failure")
            {
                CHECK_THROWS_AS(
                    FieldWorkerNvLinkRwBytes(device, &cudaOps, DCGM_FI_PROF_NVLINK_RX_BYTES, "0000:02:00.0"),
                    std::runtime_error);
            }
        }
    }
}
