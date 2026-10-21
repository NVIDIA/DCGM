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
#include "MockCudaCommon.h"

std::string MockCudaCommon::AppendCudaDriverError(const std::string &, CUresult) const
{
    return "";
}

const char *MockCudaCommon::CublasGetErrorString(cublasStatus_t) const
{
    return "";
}

std::string MockCudaCommon::AddAPIError(Plugin *,
                                        std::string const &,
                                        const char *,
                                        const char *,
                                        unsigned int,
                                        size_t,
                                        bool) const
{
    return "";
}

std::string MockCudaCommon::AddCudaError(Plugin *p,
                                         std::string const &testName,
                                         const char *callName,
                                         cudaError_t cuSt,
                                         unsigned int gpuId) const
{
    return AddCudaError(p, testName, callName, cuSt, gpuId, 0, true);
}

std::string MockCudaCommon::AddCudaError(Plugin *p,
                                         std::string const &testName,
                                         const char *callName,
                                         cudaError_t cuSt,
                                         unsigned int gpuId,
                                         size_t bytes) const
{
    return AddCudaError(p, testName, callName, cuSt, gpuId, bytes, true);
}

std::string MockCudaCommon::AddCudaError(Plugin *,
                                         std::string const &,
                                         const char *,
                                         cudaError_t,
                                         unsigned int,
                                         size_t,
                                         bool) const
{
    return "";
}

std::string MockCudaCommon::AddCudaError(Plugin *p,
                                         std::string const &testName,
                                         const char *callName,
                                         CUresult cuSt,
                                         unsigned int gpuId) const
{
    return AddCudaError(p, testName, callName, cuSt, gpuId, 0, true);
}

std::string MockCudaCommon::AddCudaError(Plugin *p,
                                         std::string const &testName,
                                         const char *callName,
                                         CUresult cuSt,
                                         unsigned int gpuId,
                                         size_t bytes) const
{
    return AddCudaError(p, testName, callName, cuSt, gpuId, bytes, true);
}

std::string MockCudaCommon::AddCudaError(Plugin *,
                                         std::string const &,
                                         const char *,
                                         CUresult,
                                         unsigned int,
                                         size_t,
                                         bool) const
{
    return "";
}

std::string MockCudaCommon::AddCublasError(Plugin *p,
                                           std::string const &testName,
                                           const char *callName,
                                           cublasStatus_t cubSt,
                                           unsigned int gpuId) const
{
    return AddCublasError(p, testName, callName, cubSt, gpuId, 0, true);
}

std::string MockCudaCommon::AddCublasError(Plugin *p,
                                           std::string const &testName,
                                           const char *callName,
                                           cublasStatus_t cubSt,
                                           unsigned int gpuId,
                                           size_t bytes) const
{
    return AddCublasError(p, testName, callName, cubSt, gpuId, bytes, true);
}

std::string MockCudaCommon::AddCublasError(Plugin *,
                                           std::string const &,
                                           const char *,
                                           cublasStatus_t,
                                           unsigned int,
                                           size_t,
                                           bool) const
{
    return "";
}

const char *MockCudaCommon::GetAdditionalCuInitDetail(CUresult) const
{
    return "";
}
