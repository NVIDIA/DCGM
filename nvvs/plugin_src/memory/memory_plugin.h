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
#ifndef MEMORY_H
#define MEMORY_H

#include "DcgmRecorder.h"
#include "Memory_wrapper.h"
#include "NvvsDeviceList.h"
#include "Plugin.h"
#include "PluginStrings.h"
#include "TestParameters.h"
#include <NvvsStructs.h>
#include <cuda.h>

/*****************************************************************************/
/* String constants */

/* Public parameters - we expect users to change these */

/* Public sub-test parameters. These apply to some sub tests and not others */

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */

/*****************************************************************************/
/* Sub tests */

/*****************************************************************************/
/* Struct to hold global information for this test */
typedef struct mem_globals_t
{
    TestParameters *testParameters; /* Parameters passed in from the framework */

    Memory *memory; /* Plugin handle for setting status */

    int cudaInitialized;       /* Has cuInit been called yet? */
    unsigned int dcgmGpuIndex; /* DCGM gpu index for the GPU */
    CUdevice cuDevice;         /* Cuda device handle to dispatch work to */

    CUcontext cuCtx;  /* Cuda context to dispatch work to */
    int cuCtxCreated; /* Does cuCtx need to be freed? */

    NvvsDevice *nvvsDevice; /* NVVS device object for controlling/querying this device */

    DcgmRecorder *m_dcgmRecorder;

} mem_globals_t, *mem_globals_p;

/*****************************************************************************/
int main_entry(const dcgmDiagPluginGpuInfo_t &gpu, Memory *memObj, TestParameters *testParameters);

/*****************************************************************************/

#endif // MEMORY_H
