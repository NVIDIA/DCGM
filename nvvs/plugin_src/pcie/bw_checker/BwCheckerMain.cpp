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

#include "BwCheckerMain.h"

#ifndef __BW_CHECKER_TESTS__
#endif
#include <DcgmStringHelpers.h>
#include <NvcmTCLAP.h>
#include <json/json.h>

#include <fmt/format.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <vector>

#ifndef __BW_CHECKER_TESTS__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define D2H_NAME    "bwchecker"
#define D2H_VERSION "1.0"

/*
 * This macro is used to remove CUDA code and the main function since they aren't meaningfully
 * covered in catch2 unit tests which are often run without GPUs and can cause issues.
 */
#ifndef __BW_CHECKER_TESTS__
bool CudaCallLogIfError(const std::string &callName, cudaError_t cuSt, unsigned int gpuId, Json::Value &root)
{
    if (cuSt == cudaSuccess)
    {
        return false;
    }

    if (gpuId == std::numeric_limits<unsigned int>::max())
    {
        root[BWC_JSON_ERRORS][0] = fmt::format("CUDA call {} failed: '{}'", callName, cudaGetErrorString(cuSt));
    }
    else
    {
        root[BWC_JSON_ERRORS][0]
            = fmt::format("CUDA call {} for GPU {} failed: '{}'", callName, gpuId, cudaGetErrorString(cuSt));
    }

    return true;
}

void TestHostDeviceBandwidth(std::vector<dcgmGpuPciIdPair_t> &pairs,
                             unsigned int intsPerCopy,
                             unsigned int iterations,
                             bool pinned,
                             Json::Value &root)
{
    // Needs:
    // GPU list
    // ints per copy
    // iterations
    // CUDA identifiers for each device
    unsigned int gpuCount = pairs.size();
    std::vector<int *> d_buffers(gpuCount);
    int *h_buffer = 0;
    std::vector<cudaEvent_t> start(gpuCount);
    std::vector<cudaEvent_t> stop(gpuCount);
    std::vector<cudaStream_t> stream1(gpuCount);
    std::vector<cudaStream_t> stream2(gpuCount);
    float time_ms;
    double time_s;
    double gb;
    std::string key;

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < pairs.size(); i++)
    {
        d_buffers[i] = 0;
        stream1[i]   = 0;
        stream2[i]   = 0;
    }

    std::map<unsigned int, CUdevice> gpuIdToCudaHandle;
    std::vector<bwTestResult_t> results(gpuCount);

    for (size_t d = 0; d < gpuCount; d++)
    {
        CUdevice cudaDevice;
        cudaError_t cuSt = cudaDeviceGetByPCIBusId(&cudaDevice, pairs[d].pciBusId.c_str());
        if (cuSt != cudaSuccess)
        {
            root[BWC_JSON_ERRORS][0]
                = fmt::format("Couldn't select the CUDA device with PCI bus id {} for GPU {}: '{}'",
                              pairs[d].pciBusId,
                              pairs[d].gpuId,
                              cudaGetErrorString(cuSt));
            return;
        }

        cudaSetDevice(cudaDevice);
        if (CudaCallLogIfError("cudaMalloc", cudaMalloc(&d_buffers[d], intsPerCopy * sizeof(int)), pairs[d].gpuId, root)
            || CudaCallLogIfError("cudaEventCreate", cudaEventCreate(&start[d]), pairs[d].gpuId, root)
            || CudaCallLogIfError("cudaEventCreate", cudaEventCreate(&stop[d]), pairs[d].gpuId, root)
            || CudaCallLogIfError("cudaStreamCreate", cudaStreamCreate(&stream1[d]), pairs[d].gpuId, root)
            || CudaCallLogIfError("cudaStreamCreate", cudaStreamCreate(&stream2[d]), pairs[d].gpuId, root))
        {
            return;
        }

        // Save the handle
        gpuIdToCudaHandle[pairs[d].gpuId] = cudaDevice;
    }

    if (pinned)
    {
        if (CudaCallLogIfError("cudaMallocHost",
                               cudaMallocHost(&h_buffer, intsPerCopy * sizeof(int)),
                               std::numeric_limits<unsigned int>::max(),
                               root))
        {
            return;
        }
    }
    else
    {
        h_buffer = (int *)calloc(1, intsPerCopy * sizeof(int));
    }

    std::vector<double> bandwidthMatrix(6 * gpuCount);

    for (size_t i = 0; i < gpuCount; i++)
    {
        results[i].dcgmGpuId = pairs[i].gpuId;
        cudaSetDevice(gpuIdToCudaHandle[pairs[i].gpuId]);

        // D2H bandwidth test
        // coverity[leaked_storage] this macro can exit the function without freeing h_buffer
        if (CudaCallLogIfError("cudaDeviceSynchronize", cudaDeviceSynchronize(), pairs[i].gpuId, root))
        {
            goto cleanup;
        }
        cudaEventRecord(start[i]);

        for (unsigned int r = 0; r < iterations; r++)
        {
            cudaMemcpyAsync(h_buffer, d_buffers[i], sizeof(int) * intsPerCopy, cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(stop[i]);
        if (CudaCallLogIfError("cudaDeviceSynchronize", cudaDeviceSynchronize(), pairs[i].gpuId, root))
        {
            goto cleanup;
        }

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                         = intsPerCopy * sizeof(int) * iterations / (double)1e9;
        results[i].bandwidths[D2H] = gb / time_s;

        // H2D bandwidth test
        if (CudaCallLogIfError("cudaDeviceSynchronize", cudaDeviceSynchronize(), pairs[i].gpuId, root))
        {
            goto cleanup;
        }
        cudaEventRecord(start[i]);

        for (unsigned int r = 0; r < iterations; r++)
        {
            cudaMemcpyAsync(d_buffers[i], h_buffer, sizeof(int) * intsPerCopy, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop[i]);
        if (CudaCallLogIfError("cudaDeviceSynchronize", cudaDeviceSynchronize(), pairs[i].gpuId, root))
        {
            goto cleanup;
        }

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                         = intsPerCopy * sizeof(int) * iterations / (double)1e9;
        results[i].bandwidths[H2D] = gb / time_s;

        // Bidirectional
        if (CudaCallLogIfError("cudaDeviceSynchronize", cudaDeviceSynchronize(), pairs[i].gpuId, root))
        {
            goto cleanup;
        }
        cudaEventRecord(start[i]);

        for (unsigned int r = 0; r < iterations; r++)
        {
            cudaMemcpyAsync(d_buffers[i], h_buffer, sizeof(int) * intsPerCopy, cudaMemcpyHostToDevice, stream1[i]);
            cudaMemcpyAsync(h_buffer, d_buffers[i], sizeof(int) * intsPerCopy, cudaMemcpyDeviceToHost, stream2[i]);
        }

        cudaEventRecord(stop[i]);
        if (CudaCallLogIfError("cudaDeviceSynchronize", cudaDeviceSynchronize(), pairs[i].gpuId, root))
        {
            goto cleanup;
        }

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                           = 2 * intsPerCopy * sizeof(int) * iterations / (double)1e9;
        results[i].bandwidths[BIDIR] = gb / time_s;
    }

    AppendResultsToJson(results, root);

cleanup:
    // Done, cleaning up
    if (pinned)
    {
        cudaFreeHost(h_buffer);
    }
    else
    {
        free(h_buffer);
    }

    for (size_t d = 0; d < gpuCount; d++)
    {
        cudaSetDevice(gpuIdToCudaHandle[pairs[d].gpuId]);
        if (CudaCallLogIfError("cudaFree", cudaFree(d_buffers[d]), pairs[d].gpuId, root)
            || CudaCallLogIfError("cudaEventDestroy", cudaEventDestroy(start[d]), pairs[d].gpuId, root)
            || CudaCallLogIfError("cudaEventDestroy", cudaEventDestroy(stop[d]), pairs[d].gpuId, root)
            || CudaCallLogIfError("cudaStreamDestroy", cudaStreamDestroy(stream1[d]), pairs[d].gpuId, root)
            || CudaCallLogIfError("cudaStreamDestroy", cudaStreamDestroy(stream2[d]), pairs[d].gpuId, root))
        {
            return;
        }
    }

    return;
}


#endif

void AppendResultsToJson(std::vector<bwTestResult_t> &results, Json::Value &root)
{
    for (unsigned int i = 0; i < results.size(); i++)
    {
        Json::Value gpu;
        if (results[i].error.empty() == false)
        {
            gpu[BWC_JSON_ERROR] = results[i].error;
        }
        gpu[BWC_JSON_MAX_RX_BW]    = results[i].bandwidths[H2D];
        gpu[BWC_JSON_MAX_TX_BW]    = results[i].bandwidths[D2H];
        gpu[BWC_JSON_MAX_BIDIR_BW] = results[i].bandwidths[BIDIR];
        gpu[BWC_JSON_GPU_ID]       = results[i].dcgmGpuId;
        root[BWC_JSON_GPUS][i]     = gpu;
    }
}

#ifndef __BW_CHECKER_TESTS__

int main(int argc, char *argv[])
{
    try
    {
        TCLAP::CmdLine cmd(D2H_NAME, ' ', D2H_VERSION);
        TCLAP::ValueArg<std::string> gpuIdsArg(
            "", "gpuIds", "The list of GPU ids whose bandwidth should be checked", true, "", "GPU list", cmd);

        TCLAP::ValueArg<std::string> pciBusIds("",
                                               "pciBusIds",
                                               "The list of CUDA device ids whose bandwidth should be checked",
                                               true,
                                               "",
                                               "CUDA device id list",
                                               cmd);

        TCLAP::ValueArg<unsigned int> iterations(
            "", "iterations", "The number of times to perform the copy.", false, 1, "iteration count", cmd);

        TCLAP::ValueArg<unsigned int> intsPerCopy(
            "", "ints-per-copy", "The number of integers to copy per test.", false, 10000000, "ints per copy", cmd);

        TCLAP::ValueArg<bool> pinned("p",
                                     "pinned",
                                     "Use pinned memory (1 to Enable, 0 to Disable)",
                                     false,
                                     0,
                                     "0/1",
                                     cmd); // Option 1 for set command-line


        cmd.parse(argc, argv);

        std::vector<std::string> gpuIdArray    = dcgmTokenizeString(gpuIdsArg.getValue(), ",");
        std::vector<std::string> pciBusIdArray = dcgmTokenizeString(pciBusIds.getValue(), ",");
        std::vector<unsigned int> gpuIds;

        Json::Value root;
        int errorIndex = 0;

        if (gpuIdArray.size() != pciBusIdArray.size())
        {
            root[BWC_JSON_ERRORS][errorIndex] = fmt::format(
                "We must have the same number of GPU ids and CUDA device ids, but we have {} and {} respectively",
                gpuIdArray.size(),
                pciBusIdArray.size());
        }
        else
        {
            std::vector<dcgmGpuPciIdPair_t> pairs;
            dcgmGpuPciIdPair_t pair;
            for (size_t i = 0; i < pciBusIdArray.size(); i++)
            {
                pair.gpuId    = std::stoi(gpuIdArray[i]);
                pair.pciBusId = pciBusIdArray[i];
                pairs.push_back(pair);
            }

            TestHostDeviceBandwidth(pairs, intsPerCopy.getValue(), iterations.getValue(), pinned.getValue(), root);
        }

        // Write Json output to stdout
        std::cout << root.toStyledString() << std::endl;
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "Argument parsing exception: " << e.error() << std::endl;
        return 1;
    }
}
#endif // __BW_CHECKER_TESTS__
