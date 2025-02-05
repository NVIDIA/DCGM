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
