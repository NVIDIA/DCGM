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
#ifndef MEMORY_H
#define MEMORY_H

#include "DcgmRecorder.h"
#include "Memory_wrapper.h"
#include "NvvsDeviceList.h"
#include "Plugin.h"
#include "PluginStrings.h"
#include "TestParameters.h"
#include <CudaCommonAnyPtr.hpp>
#include <CudaOpsAnyUniquePtr.hpp>
#include <DcgmOpsAnyPtr.hpp>
#include <NvvsStructs.h>
#include <cuda.h>

#include <memory>

/*****************************************************************************/
/* String constants */

/* Public parameters - we expect users to change these */

/* Public sub-test parameters. These apply to some sub tests and not others */

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */

/*****************************************************************************/
/* Constants */

/* Default maximum free memory to use when small FB mode is enabled (256MB) */
#define MEMORY_DEFAULT_MAX_FREE_MEMORY_BYTES ((size_t)256 * 1024 * 1024)

/*****************************************************************************/
/* Sub tests */

struct mem_globals_t
{
    mem_globals_t(Memory &memory,
                  TestParameters *testParameters,
                  dcgmHandle_t handle,
                  DcgmNs::DcgmOpsAnyPtr dcgmOps,
                  unsigned int dcgmGpuIndex,
                  CudaOpsAnyPtr cudaOps,
                  CudaCommonAnyPtr cudaCommon,
                  NvvsCommon *nvvsCommon);
    mem_globals_t(mem_globals_t const &)            = delete;
    mem_globals_t &operator=(mem_globals_t const &) = delete;
    mem_globals_t(mem_globals_t &&)                 = delete;
    mem_globals_t &operator=(mem_globals_t &&)      = delete;
    ~mem_globals_t()                                = default;

    TestParameters *testParameters; //!< Parameters passed in from the framework.

    Memory *memory;                //!< Plugin handle for setting status.
    dcgmHandle_t handle;           //!< DCGM connection handle.
    DcgmNs::DcgmOpsAnyPtr dcgmOps; //!< DCGM API interface.
    CudaOpsAnyPtr cudaOps;         //!< CUDA API facade.
    CudaCommonAnyPtr cudaCommon;   //!< CUDA error/reporting helper.
    NvvsCommon *nvvsCommon;        //!< Common NVVS state owned by the host.

    int cudaInitialized = 0;   //!< Has cuInit been called yet?
    unsigned int dcgmGpuIndex; //!< DCGM GPU index for the GPU.
    CUdevice cuDevice = 0;     //!< CUDA device handle to dispatch work to.

    CUcontext cuCtx  = nullptr; //!< CUDA context to dispatch work to.
    int cuCtxCreated = 0;       //!< Does cuCtx need to be freed?

    std::unique_ptr<NvvsDevice> nvvsDevice; //!< NVVS device object for controlling/querying this device.

    std::unique_ptr<DcgmRecorder> m_dcgmRecorder; //!< DCGM stats recording interface object.
};

using mem_globals_p = mem_globals_t *;

/*****************************************************************************/
int main_entry(const dcgmDiagPluginEntityInfo_v1 &entity,
               Memory *memObj,
               TestParameters *testParameters,
               dcgmHandle_t handle,
               DcgmNs::DcgmOpsAnyPtr dcgmOps,
               CudaOpsAnyPtr cudaOps,
               CudaCommonAnyPtr cudaCommon,
               NvvsCommon *nvvsCommon);

/*****************************************************************************/
void mem_cleanup(mem_globals_p memGlobals);
int mem_init(mem_globals_p memGlobals, const dcgmDiagPluginEntityInfo_v1 &entityInfo);
nvvsPluginResult_t runTestDeviceMemory(mem_globals_p memGlobals);

/*****************************************************************************/

#endif // MEMORY_H
