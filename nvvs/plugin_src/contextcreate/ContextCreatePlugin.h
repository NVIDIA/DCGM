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
    ContextCreatePlugin(dcgmHandle_t handle);
    ~ContextCreatePlugin()
    {}

    void Go(std::string const &testName,
            dcgmDiagPluginEntityList_v1 const *entityInfo,
            unsigned int numParameters,
            dcgmDiagPluginTestParameter_t const *testParameters) override;

    dcgmHandle_t GetHandle();

    std::string GetCtxCreateTestName() const;

private:
    dcgmHandle_t m_handle;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> m_entityInfo;
};

#endif
