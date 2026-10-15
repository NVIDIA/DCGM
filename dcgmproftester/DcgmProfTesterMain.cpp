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
#include "DcgmProfTester.h"

#include <CublasOps.hpp>
#include <CudaOps.hpp>
#include <DcgmBuildInfo.hpp>
#include <DcgmLogging.h>
#include <cuda.h>
#include <fmt/ranges.h>

#include <atomic>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace DcgmNs::ProfTester
{
extern std::atomic_bool g_signalCaught;
}

namespace
{
void SignalHandler(int)
{
    DcgmNs::ProfTester::g_signalCaught = true;
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        CudaOps cudaOps;
        CublasOps cublasOps;
        auto cudaWorkerFactory = std::make_unique<CudaWorkerThreadFactory>(&cudaOps, &cublasOps);

        DcgmProfTester dpt(std::make_unique<DcgmNs::ProfTester::DistributedCudaContextRuntimeFactory>(),
                           std::move(cudaWorkerFactory),
                           std::make_unique<DcgmNs::DcgmOps>(),
                           &cudaOps);

        dcgmReturn_t dcgmReturn;

        // Ensure clean termination for code coverage tests.
        signal(SIGPIPE, SignalHandler);
        signal(SIGTERM, SignalHandler);
        signal(SIGHUP, SignalHandler);

        // We do this to avoid zombies. We don't care about worker exit codes.
        struct sigaction sa;
        sa.sa_handler = SIG_IGN; // handle signal by ignoring
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        if (sigaction(SIGCHLD, &sa, 0) == -1)
        {
            std::cerr << "Could not ignore SIGCHLD from worker threads." << std::endl;

            return DCGM_ST_GENERIC_ERROR;
        }

        dcgmReturn = dpt.Init(argc, argv);
        if (dcgmReturn)
        {
            std::cerr << "Error " << std::to_underlying(dcgmReturn) << " from Init(). Exiting." << std::endl;

            return dcgmReturn;
        }

        auto maxwellPascalVoltaGpus = dpt.GetMaxwellPascalVoltaArchGpus();
        if (!maxwellPascalVoltaGpus.has_value())
        {
            std::cerr << "Error [" << maxwellPascalVoltaGpus.error()
                      << "] from taking GPUs with Maxwell, Pascal, or Volta architecture. Exiting." << std::endl;
            return DCGM_ST_GENERIC_ERROR;
        }

        int cudaLoaded = 0;
        auto cuResult  = cudaOps.cuDriverGetVersion(&cudaLoaded);
        if (cuResult != CUDA_SUCCESS)
        {
            std::cerr << "Unable to validate Cuda version. cuDriverGetVersion returned " << std::to_underlying(cuResult)
                      << "." << std::endl;
            return DCGM_ST_GENERIC_ERROR;
        }

        // CUDA_VERSION_USED is defined in CMakeLists.txt file
        int majorCudaVersionLoaded = cudaLoaded / 1000;
        int minorCudaVersionLoaded = (cudaLoaded - majorCudaVersionLoaded * 1000) / 10;
        if (maxwellPascalVoltaGpus->size() > 0 && majorCudaVersionLoaded == 13 && minorCudaVersionLoaded == 0)
        {
            if (CUDA_VERSION_USED != 12)
            {
                std::string const gpus = fmt::format("{}", fmt::join(maxwellPascalVoltaGpus.value(), ", "));
                std::string const msg  = fmt::format(
                    "CUDA 13.0 drops support for everything that is < 7.5 SM Cuda Compatibility. These older SKUs need to run a 12.9-linked application. "
                    "The following GPUs are affected: {}",
                    gpus);
                std::cerr << msg << std::endl;
                std::cerr << "Please use dcgmproftester12 instead." << std::endl;
                return DCGM_ST_VER_MISMATCH;
            }
        }
        else if (majorCudaVersionLoaded > MAX_CUDA_MAJOR_VERSION && CUDA_VERSION_USED == MAX_CUDA_MAJOR_VERSION)
        {
            std::cout << "Running version " << CUDA_VERSION_USED << " binary with CUDA driver version "
                      << majorCudaVersionLoaded << "." << std::endl;
        }
        else
        {
            if (majorCudaVersionLoaded != CUDA_VERSION_USED)
            {
                std::cerr << "Wrong version of dcgmproftester is used. Expected Cuda version is " << CUDA_VERSION_USED
                          << ". Installed Cuda version is " << majorCudaVersionLoaded << "." << std::endl;
                return DCGM_ST_VER_MISMATCH;
            }
        }

        dcgmReturn = dpt.Process([&dpt](DcgmNs::ProfTester::Arguments_t const &arguments) -> dcgmReturn_t {
            /**
             * This is the normal logging initialization.
             */
            dpt.InitializeLogging(arguments.m_parameters.m_logFile.c_str(), arguments.m_parameters.m_logLevel);

            dcgmReturn_t dcgmReturn = dpt.InitializeGpus(arguments);

            if (dcgmReturn)
            {
                DCGM_LOG_ERROR << "Error " << dcgmReturn << " from InitializeGpus(). Exiting.";

                return dcgmReturn;
            }

            dcgmReturn_t st = dpt.RunTests(arguments.m_parameters.m_reportInterval,
                                           arguments.m_parameters.m_duration,
                                           arguments.m_parameters.m_fieldId,
                                           arguments.m_parameters.m_maxGpusInParallel);

            if (st != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "Error " << st << " from RunTests(). Exiting.";
            }

            dcgmReturn = dpt.ShutdownGpus();

            if (dcgmReturn)
            {
                DCGM_LOG_ERROR << "Error " << dcgmReturn << " from ShutdownGpus(). Exiting.";

                return dcgmReturn;
            }

            if (st == DCGM_ST_NOT_SUPPORTED)
            {
                st = DCGM_ST_OK;
            }

            return st;
        });

        if (dcgmReturn == DCGM_ST_OK)
        {
            std::cout << "All Tests Passed." << std::endl;
        }

        return dcgmReturn;
    }
    catch (std::runtime_error const &ex)
    {
        DCGM_LOG_ERROR << "Uncaught runtime exception occured: " << ex.what();

        return DCGM_ST_GENERIC_ERROR;
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Uncaught unexpected exception occured.";

        return DCGM_ST_GENERIC_ERROR;
    }

    return 0;
}
