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
    Memory(dcgmHandle_t handle);
    ~Memory()
    {}

    void Go(std::string const &testName,
            dcgmDiagPluginEntityList_v1 const *entityInfo,
            unsigned int numParameters,
            dcgmDiagPluginTestParameter_t const *testParameters) override;

    std::string GetMemoryTestName() const;
    dcgmHandle_t GetHandle();

private:
    dcgmHandle_t m_handle;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> m_entityInfo;
};


#endif // _NVVS_NVVS_Memory_H_
