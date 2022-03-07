/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cublas_v2.h>
#include <cuda.h>
#include <string>

#include "Plugin.h"

// This comment leads with '; ' so that it can be appended directly onto other error messages where applicable
#define CHECK_FABRIC_MANAGER_MSG "; verify that the fabric-manager has been started if applicable"

/********************************************************************/
/*
 * Translates cuRes to a string and concatenates it. If we cannot get an
 * error from cuGetErrorString(), then just error is returned.
 */
std::string AppendCudaDriverError(const std::string &error, CUresult cuRes);

/********************************************************************/
/*
 * cudaGetErrorString equivalent for cublasStatus_t.
 * Returns the cublas status name as a string.
 */
const char *cublasGetErrorString(cublasStatus_t status);

/*************************************************************************/
/*
 * Adds a warning for the relevant API call error and returns the added string for logging to a log file.
 * Thread-safe method.
 */
std::string AddAPIError(Plugin *p,
                        const char *callName,
                        const char *errorText,
                        unsigned int gpuId,
                        size_t bytes       = 0,
                        bool isGpuSpecific = true);

/*************************************************************************/
/*
 * Adds a warning for the relevant cuda error and returns the added string for logging to a log file.
 *
 * For simplicity when adding an API error to warnings and logging it, use the LOG_CUDA_ERROR* macro instead.
 * Thread-safe method.
 */
std::string AddCudaError(Plugin *p,
                         const char *callName,
                         cudaError_t cuSt,
                         unsigned int gpuId,
                         size_t bytes       = 0,
                         bool isGpuSpecific = true);

/* Overloaded version for cuda driver errors (i.e. CUResult) */
std::string AddCudaError(Plugin *p,
                         const char *callName,
                         CUresult cuSt,
                         unsigned int gpuId,
                         size_t bytes       = 0,
                         bool isGpuSpecific = true);

/*
 * The purpose of these macros is to ensure that the logged message includes accurate line numbers and
 * file name corresponding to the location where the macro is called. If the error message was logged inside the
 * AddCudaError method, the line number and file name would have been obscured.
 */
#define LOG_CUDA_ERROR_FOR_PLUGIN(plugin, callName, cuSt, gpuId, ...)                                   \
    {                                                                                                   \
        std::string pluginCommonCudaError = AddCudaError(plugin, callName, cuSt, gpuId, ##__VA_ARGS__); \
        PRINT_ERROR("%s", "%s", pluginCommonCudaError.c_str());                                         \
    }                                                                                                   \
    (void)0

// Only for use by the Plugin subclasses
#define LOG_CUDA_ERROR(callName, cuSt, gpuId, ...)                                                    \
    {                                                                                                 \
        std::string pluginCommonCudaError = AddCudaError(this, callName, cuSt, gpuId, ##__VA_ARGS__); \
        PRINT_ERROR("%s", "%s", pluginCommonCudaError.c_str());                                       \
    }                                                                                                 \
    (void)0

/*************************************************************************/
/*
 * Adds a warning for the relevant cublas error and returns the added string for logging to a log file.
 *
 * For simplicity when adding an API error to warnings and logging it, use the LOG_CUBLAS_ERROR* macro instead.
 * Thread-safe method.
 */
std::string AddCublasError(Plugin *p,
                           const char *callName,
                           cublasStatus_t cubSt,
                           unsigned int gpuId,
                           size_t bytes       = 0,
                           bool isGpuSpecific = true);

/*************************************************************************/
/*
 * Adds an additional error message to cuInit return codes if the Adds an additional error message to cuInit if
 * the error code was that it isn't initialized, as this usually means the fabric manager isn't running yet.
 *
 * Returns the additional detail if cuSt indicates it wasn't initialized, and an emptry string otherwise
 */
const char *GetAdditionalCuInitDetail(CUresult cuSt);

/*
 * The purpose of these macros is to ensure that the logged message includes accurate line numbers and
 * file name corresponding to the location where the macro is called. If the error message was logged inside the
 * AddCublasError method, the line number and file name would have been obscured.
 */
#define LOG_CUBLAS_ERROR_FOR_PLUGIN(plugin, callName, cubSt, gpuId, ...)                                     \
    {                                                                                                        \
        std::string pluginCommonCublasError = AddCublasError(plugin, callName, cubSt, gpuId, ##__VA_ARGS__); \
        PRINT_ERROR("%s", "%s", pluginCommonCublasError.c_str());                                            \
    }                                                                                                        \
    (void)0

// Only for use by the Plugin subclasses
#define LOG_CUBLAS_ERROR(callName, cubSt, gpuId, ...)                                                      \
    {                                                                                                      \
        std::string pluginCommonCublasError = AddCublasError(this, callName, cubSt, gpuId, ##__VA_ARGS__); \
        PRINT_ERROR("%s", "%s", pluginCommonCublasError.c_str());                                          \
    }                                                                                                      \
    (void)0

#endif // _NVVS_NVVS_Cuda_common_H_
