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

// This file stubs the methods from nvvs/plugin_src/common/CudaCommon.cpp

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include <Plugin.h>

std::string AppendCudaDriverError(const std::string & /* error */, CUresult /* cuRes */)
{
    return "";
}

std::string AddAPIError(Plugin * /* p */,
                        std::string const & /* testName */,
                        const char * /* callName */,
                        const char * /* errorText */,
                        unsigned int /* gpuId */,
                        size_t /* bytes */,
                        bool /* isGpuSpecific */)
{
    return "";
}

std::string AddCudaError(Plugin * /* p */,
                         std::string const & /* testName */,
                         const char * /* callName */,
                         cudaError_t /* cuSt */,
                         unsigned int /* gpuId */,
                         size_t /* bytes */,
                         bool /* isGpuSpecific */)
{
    return "";
}

std::string AddCudaError(Plugin * /* p */,
                         std::string const & /* testName */,
                         const char * /* callName */,
                         CUresult /* cuSt */,
                         unsigned int /* gpuId */,
                         size_t /* bytes */,
                         bool /* isGpuSpecific */)
{
    return "";
}

std::string AddCublasError(Plugin * /* p */,
                           std::string const & /* testName */,
                           const char * /* callName */,
                           cublasStatus_t /* cubSt */,
                           unsigned int /* gpuId */,
                           size_t /* bytes */,
                           bool /* isGpuSpecific */)
{
    return "";
}

const char *GetAdditionalCuInitDetail(CUresult /* cuSt */)
{
    return "";
}
