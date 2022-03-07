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
#ifndef _NVVS_NVVS_Memory_H_
#define _NVVS_NVVS_Memory_H_

#include <iostream>
#include <string>
#include <vector>

#include "CudaCommon.h"
#include "Plugin.h"
#include "PluginCommon.h"
#include "TestParameters.h"

#include <NvvsStructs.h>
#include <PluginInterface.h>
#include <dcgm_structs.h>

class Memory : public Plugin
{
public:
    Memory(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo);
    ~Memory()
    {}

    void Go(TestParameters *testParameters, const dcgmDiagGpuInfo_t &info);
    void Go(TestParameters * /*testParameters*/, const dcgmDiagGpuList_t & /*gpuList*/)
    {
        throw std::runtime_error("Not implemented in this test.");
    }
    void Go(TestParameters * /*testParameters*/)
    {
        throw std::runtime_error("Not implemented in this test.");
    }
    void Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters);

    dcgmHandle_t GetHandle();

private:
    dcgmHandle_t m_handle;
    dcgmDiagPluginGpuList_t m_gpuInfo;
};


#endif // _NVVS_NVVS_Memory_H_
