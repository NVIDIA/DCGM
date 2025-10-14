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
#ifndef _NVVS_NVVS_Plugin_common_H_
#define _NVVS_NVVS_Plugin_common_H_

#include "Plugin.h"
#include <expected>

#define RUN_CUDA_KERNEL_REC \
    "Please check if a CUDA sample program can be run successfully on this host. Refer to https://github.com/nvidia/cuda-samples"

/********************************************************************/
/*
 * Sets the result for the GPU based on the value of 'passed'. Logs warnings from 'errorList' if the result is fail.
 * Sets 'allPassed' to false if the result is failed for this GPU.
 *
 * 'i' is the index of the GPU in 'gpuList'; the GPU ID is the value in the vector (i.e. gpuList[i] is GPU ID).
 * If 'dcgmCommError' is true, sets result for all GPUs starting at index 'i' to fail and adds a warning to
 * 'errorListAllGpus'.
 */
void CheckAndSetResult(Plugin *p,
                       std::string const &testName,
                       const std::vector<unsigned int> &gpuList,
                       size_t i,
                       bool passed,
                       const std::vector<DcgmError> &errorList,
                       bool &allPassed,
                       bool dcgmCommError);

/********************************************************************/
/*
 * Returns whether the environmental variable for Small FrameBuffer mode
 * is set. This is a hint that we're not testing the plugins in their entirety
 * but just want them to run minimally to test something else.
 *
 * Startup time for plugins scales linearly with the number of GPUs we have and
 * how much FB (FrameBuffer) we allocate per GPU. Setting this will cause plugins
 * that allocate a lot of FB to start faster. This should only be used from the
 * DCGM test framework.
 */
bool IsSmallFrameBufferModeSet(void);

/********************************************************************/
/**
 * @brief Gets the CUDA driver major and minor versions from DCGM
 *
 * @param dcgmRecorder Reference to the DCGM recorder
 * @param gpuId GPU ID to query
 * @return std::expected<std::pair<unsigned int, unsigned int>, dcgmReturn_t>
 *         Pair of (major, minor) versions on success, or dcgmReturn_t error code on failure.
 *         The minor version is the human-readable value (e.g., 3 for CUDA 12.3), not scaled by 10.
 */
std::expected<std::pair<unsigned int, unsigned int>, dcgmReturn_t> GetCudaDriverVersions(DcgmRecorder &dcgmRecorder,
                                                                                         unsigned int gpuId);

/********************************************************************/
/**
 * @brief Sets the CUDA driver versions from DCGM field value
 *
 * @param dcgmRecorder Reference to the DCGM recorder
 * @param gpuId GPU ID to query
 * @param defaultMajorVersion Default major version to use if query fails
 * @param defaultMinorVersion Default minor version to use if query fails
 * @param[out] cudaDriverMajorVersion The major version of the CUDA driver
 * @param[out] cudaDriverMinorVersion The minor version of the CUDA driver
 * @return dcgmReturn_t DCGM_ST_OK if successful
 */
dcgmReturn_t SetCudaDriverVersions(DcgmRecorder &dcgmRecorder,
                                   unsigned int gpuId,
                                   unsigned int defaultMajorVersion,
                                   unsigned int defaultMinorVersion,
                                   unsigned int &cudaDriverMajorVersion,
                                   unsigned int &cudaDriverMinorVersion);

/********************************************************************/
/**
 * @brief Sets the CUDA driver major version from DCGM field value
 *
 * @param dcgmRecorder Reference to the DCGM recorder
 * @param gpuId GPU ID to query
 * @param defaultMajorVersion Default major version to use if query fails
 * @param[out] cudaDriverMajorVersion The major version of the CUDA driver
 * @return dcgmReturn_t DCGM_ST_OK if successful
 */
dcgmReturn_t SetCudaDriverMajorVersion(DcgmRecorder &dcgmRecorder,
                                       unsigned int gpuId,
                                       unsigned int defaultMajorVersion,
                                       unsigned int &cudaDriverMajorVersion);

/********************************************************************/
/**
 * Find executable in search paths
 *
 * @param executableName Name of executable to find
 * @param searchPaths Vector of paths to search in
 * @param[out] executableDir Path to executable directory if found
 * @return std::expected<std::string, dcgmReturn_t> Path to executable if found, error code otherwise
 */
std::expected<std::string, dcgmReturn_t> FindExecutable(std::string_view executableName,
                                                        std::vector<std::string> const &searchPaths,
                                                        std::string &executableDir);

/********************************************************************/
/**
 * Get default search paths
 *
 * @param cudaDriverMajorVersion CUDA driver major version
 * @param useUpdatedPath Whether to use updated path
 * @return std::vector<std::string> Default search paths
 */
std::vector<std::string> GetDefaultSearchPaths(unsigned int cudaDriverMajorVersion, bool useUpdatedPath);

/********************************************************************/
/*
 * Initializes the plugin to be able to write logs to the NVVS log file.
 * This is a macro that allows each plugin to initialize its own locally visible instance of the logger singleton.
 * If this were a function, each plugin would initialize the logger in the common plugin address space.
 * However, when actual log_* functions are called, the logger from the local plugin address space would be used.
 */
#define InitializeLoggingCallbacks(severity, callback, pluginName) \
    do                                                             \
    {                                                              \
        InitLogToHostengine(severity);                             \
        LoggingSetHostEngineCallback(callback);                    \
        LoggingSetHostEngineComponentName(pluginName);             \
    } while (0)

#endif // _NVVS_NVVS_Plugin_common_H_
