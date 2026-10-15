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
#ifndef CONTEXTCREATEPLUGIN_H
#define CONTEXTCREATEPLUGIN_H

#include "CudaCommon.h"
#include "CudaCommonAnyUniquePtr.hpp"
#include "Plugin.h"
#include <CudaOpsAnyUniquePtr.hpp>
#include <DcgmOpsAnyPtr.hpp>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <NvvsStructs.h>

#include <memory>

#define CONTEXT_CREATE_PASS 0
#define CONTEXT_CREATE_FAIL -1
#define CONTEXT_CREATE_SKIP -2

class ContextCreatePlugin : public Plugin
{
public:
    ContextCreatePlugin(dcgmHandle_t handle,
                        DcgmNs::DcgmOpsAnyUniquePtr dcgmOps,
                        CudaOpsAnyUniquePtr cudaOps,
                        CudaCommonAnyUniquePtr cudaCommon);
    ~ContextCreatePlugin()
    {}

    void Go(std::string const &testName,
            dcgmDiagPluginEntityList_v1 const *entityInfo,
            unsigned int numParameters,
            dcgmDiagPluginTestParameter_t const *testParameters) override;

    std::string GetCtxCreateTestName() const;

private:
    DcgmNs::DcgmOpsAnyUniquePtr m_dcgmOpsUnique; //!< Owned DCGM API interface.
    CudaOpsAnyUniquePtr m_cudaOps;               //!< CUDA API interface.
    CudaCommonAnyUniquePtr m_cudaCommon;
    dcgmHandle_t m_handle;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> m_entityInfo;
    std::unique_ptr<TestParameters> m_testParameters;
};

#endif
