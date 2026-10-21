/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef _NVVS_NVVS_Cuda_common_H_
#define _NVVS_NVVS_Cuda_common_H_

#include <CudaOpsAnyPtr.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <string>
#include <utility>

#include "Plugin.h"

// This comment leads with '; ' so that it can be appended directly onto other error messages where applicable
#define CHECK_FABRIC_MANAGER_MSG "; verify that the fabric-manager has been started if applicable"

/********************************************************************/
class CudaCommon final
{
public:
    /**
     * Creates a CUDA common helper that uses the supplied CUDA facade for error string lookup.
     *
     * @param[in] cudaOps CUDA facade. The caller must keep it alive for the lifetime of this object.
     */
    explicit CudaCommon(CudaOpsAnyPtr cudaOps);

    /**
     * Appends CUDA driver error text to an existing message.
     *
     * @param[in] error Existing error text.
     * @param[in] cuRes CUDA driver result.
     * @return The error text with CUDA driver detail appended when available.
     */
    std::string AppendCudaDriverError(const std::string &error, CUresult cuRes) const;

    /**
     * Returns a cublasStatus_t name.
     *
     * @param[in] status CUBLAS status.
     * @return The status name, or "Unknown error" for unknown values.
     */
    const char *CublasGetErrorString(cublasStatus_t status) const;

    /**
     * Adds a warning for an API call error and returns the added message for logging.
     *
     * @param[in,out] p Plugin that receives the error.
     * @param[in] testName Plugin test name.
     * @param[in] callName API call name.
     * @param[in] errorText API error text.
     * @param[in] gpuId GPU id associated with the error.
     * @param[in] bytes Optional byte count associated with the call.
     * @param[in] isGpuSpecific Whether the error is specific to gpuId.
     * @return The formatted error message.
     */
    std::string AddAPIError(Plugin *p,
                            std::string const &testName,
                            const char *callName,
                            const char *errorText,
                            unsigned int gpuId,
                            size_t bytes,
                            bool isGpuSpecific) const;

    /** Adds a warning for a CUDA error and returns the added message for logging. */
    template <typename CudaStatus, typename... Args>
    std::string LogCudaErrorForPlugin(Plugin *p,
                                      std::string const &testName,
                                      const char *callName,
                                      CudaStatus cuSt,
                                      unsigned int gpuId,
                                      Args &&...args) const
    {
        return AddCudaError(p, testName, callName, cuSt, gpuId, std::forward<Args>(args)...);
    }

    /** Adds a warning for a CUBLAS error and returns the added message for logging. */
    template <typename... Args>
    std::string LogCublasErrorForPlugin(Plugin *p,
                                        std::string const &testName,
                                        const char *callName,
                                        cublasStatus_t cubSt,
                                        unsigned int gpuId,
                                        Args &&...args) const
    {
        return AddCublasError(p, testName, callName, cubSt, gpuId, std::forward<Args>(args)...);
    }

    /**
     * Adds a warning for a CUDA runtime error and returns the added message for logging.
     */
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             cudaError_t cuSt,
                             unsigned int gpuId) const;
    /**
     * Adds a warning for a CUDA runtime error and returns the added message for logging.
     */
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             cudaError_t cuSt,
                             unsigned int gpuId,
                             size_t bytes) const;
    /**
     * Adds a warning for a CUDA runtime error and returns the added message for logging.
     */
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             cudaError_t cuSt,
                             unsigned int gpuId,
                             size_t bytes,
                             bool isGpuSpecific) const;

    /**
     * Adds a warning for a CUDA driver error and returns the added message for logging.
     */
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             CUresult cuSt,
                             unsigned int gpuId) const;
    /**
     * Adds a warning for a CUDA driver error and returns the added message for logging.
     */
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             CUresult cuSt,
                             unsigned int gpuId,
                             size_t bytes) const;
    /**
     * Adds a warning for a CUDA driver error and returns the added message for logging.
     */
    std::string AddCudaError(Plugin *p,
                             std::string const &testName,
                             const char *callName,
                             CUresult cuSt,
                             unsigned int gpuId,
                             size_t bytes,
                             bool isGpuSpecific) const;

    /**
     * Adds a warning for a CUBLAS error and returns the added message for logging.
     */
    std::string AddCublasError(Plugin *p,
                               std::string const &testName,
                               const char *callName,
                               cublasStatus_t cubSt,
                               unsigned int gpuId) const;
    /**
     * Adds a warning for a CUBLAS error and returns the added message for logging.
     */
    std::string AddCublasError(Plugin *p,
                               std::string const &testName,
                               const char *callName,
                               cublasStatus_t cubSt,
                               unsigned int gpuId,
                               size_t bytes) const;
    /**
     * Adds a warning for a CUBLAS error and returns the added message for logging.
     */
    std::string AddCublasError(Plugin *p,
                               std::string const &testName,
                               const char *callName,
                               cublasStatus_t cubSt,
                               unsigned int gpuId,
                               size_t bytes,
                               bool isGpuSpecific) const;

    /**
     * Returns additional detail for cuInit failures.
     *
     * @param[in] cuSt CUDA driver result.
     * @return Additional detail for known cuInit failures, otherwise an empty string.
     */
    const char *GetAdditionalCuInitDetail(CUresult cuSt) const;

private:
    mutable CudaOpsAnyPtr m_cudaOps;
};

/*
 * The purpose of these macros is to ensure that the logged message includes accurate line numbers and
 * file name corresponding to the location where the macro is called. If the error message was logged inside the
 * AddCudaError method, the line number and file name would have been obscured.
 */
#define LOG_CUDA_ERROR_FOR_PLUGIN(cudaCommon, ...)                                   \
    {                                                                                \
        std::string pluginCommonCudaError = (cudaCommon)->AddCudaError(__VA_ARGS__); \
        log_error(pluginCommonCudaError);                                            \
    }                                                                                \
    (void)0

// Only for use by the Plugin subclasses
#define LOG_CUDA_ERROR(cudaCommon, testName, callName, cuSt, gpuId, ...)                        \
    {                                                                                           \
        std::string pluginCommonCudaError                                                       \
            = (cudaCommon)->AddCudaError(this, testName, callName, cuSt, gpuId, ##__VA_ARGS__); \
        log_error(pluginCommonCudaError);                                                       \
    }                                                                                           \
    (void)0

/*
 * The purpose of these macros is to ensure that the logged message includes accurate line numbers and
 * file name corresponding to the location where the macro is called. If the error message was logged inside the
 * AddCublasError method, the line number and file name would have been obscured.
 */
#define LOG_CUBLAS_ERROR_FOR_PLUGIN(cudaCommon, ...)                                     \
    {                                                                                    \
        std::string pluginCommonCublasError = (cudaCommon)->AddCublasError(__VA_ARGS__); \
        log_error(pluginCommonCublasError);                                              \
    }                                                                                    \
    (void)0

// Only for use by the Plugin subclasses
#define LOG_CUBLAS_ERROR(cudaCommon, testName, callName, cubSt, gpuId, ...)                        \
    {                                                                                              \
        std::string pluginCommonCublasError                                                        \
            = (cudaCommon)->AddCublasError(this, testName, callName, cubSt, gpuId, ##__VA_ARGS__); \
        log_error(pluginCommonCublasError);                                                        \
    }                                                                                              \
    (void)0

#endif // _NVVS_NVVS_Cuda_common_H_
