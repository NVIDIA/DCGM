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

#include <CudaCommon.h>

class MockCudaCommon
{
public:
    std::string AppendCudaDriverError(const std::string &error, CUresult cuRes) const;
    const char *CublasGetErrorString(cublasStatus_t status) const;
    std::string AddAPIError(Plugin *p,
                            std::string const &testName,
                            const char *callName,
                            const char *errorText,
                            unsigned int gpuId,
                            size_t bytes,
                            bool isGpuSpecific) const;

    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             cudaError_t cuSt,
                             unsigned int gpuId) const;
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             cudaError_t cuSt,
                             unsigned int gpuId,
                             size_t bytes) const;
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             cudaError_t cuSt,
                             unsigned int gpuId,
                             size_t bytes,
                             bool isGpuSpecific) const;

    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             CUresult cuSt,
                             unsigned int gpuId) const;
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             CUresult cuSt,
                             unsigned int gpuId,
                             size_t bytes) const;
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             CUresult cuSt,
                             unsigned int gpuId,
                             size_t bytes,
                             bool isGpuSpecific) const;

    std::string AddCublasError(Plugin *p,
                               std::string const &testName,
                               const char *callName,
                               cublasStatus_t cubSt,
                               unsigned int gpuId) const;
    std::string AddCublasError(Plugin *p,
                               std::string const &testName,
                               const char *callName,
                               cublasStatus_t cubSt,
                               unsigned int gpuId,
                               size_t bytes) const;
    std::string AddCublasError(Plugin *p,
                               std::string const &testName,
                               const char *callName,
                               cublasStatus_t cubSt,
                               unsigned int gpuId,
                               size_t bytes,
                               bool isGpuSpecific) const;

    const char *GetAdditionalCuInitDetail(CUresult cuSt) const;
};
