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
#ifndef CONTEXTCREATEPLUGIN_H
#define CONTEXTCREATEPLUGIN_H

#include "CudaCommon.h"
#include "Plugin.h"
#include <NvvsStructs.h>

#define CONTEXT_CREATE_PASS 0
#define CONTEXT_CREATE_FAIL -1
#define CONTEXT_CREATE_SKIP -2

class ContextCreatePlugin : public Plugin
{
public:
    ContextCreatePlugin(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo);
    ~ContextCreatePlugin()
    {}

    void Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters);

    dcgmHandle_t GetHandle();

private:
    dcgmHandle_t m_handle;
    dcgmDiagPluginGpuList_t m_gpuInfo;
};

#endif
