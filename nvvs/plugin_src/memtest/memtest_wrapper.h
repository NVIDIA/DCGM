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
#ifndef _NVVS_NVVS_MEMTEST_H_
#define _NVVS_NVVS_MEMTEST_H_

#include "Plugin.h"

#include <CudaCommonAnyUniquePtr.hpp>
#include <CudaOpsAnyPtr.hpp>
#include <CudaOpsAnyUniquePtr.hpp>
#include <DcgmOpsAnyPtr.hpp>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <NvvsStructs.h>
#include <PluginInterface.h>
#include <dcgm_structs.h>

#include <memory>
#include <string>

class MemtestPlugin : public Plugin
{
public:
    MemtestPlugin(dcgmHandle_t handle,
                  DcgmNs::DcgmOpsAnyUniquePtr dcgmOps,
                  CudaOpsAnyUniquePtr cudaOps,
                  CudaCommonAnyUniquePtr cudaCommon,
                  NvvsCommon *nvvsCommon);

    void Go(std::string const &testName,
            dcgmDiagPluginEntityList_v1 const *entityInfo,
            unsigned int numParameters,
            dcgmDiagPluginTestParameter_t const *testParameters) override;

    std::string GetMmeTestTestName() const;

private:
    DcgmNs::DcgmOpsAnyUniquePtr m_dcgmOpsUnique; //!< Owned DCGM API interface.
    CudaOpsAnyUniquePtr m_cudaOpsUnique;         //!< Owned CUDA API interface.
    CudaCommonAnyUniquePtr m_cudaCommonUnique;   //!< Owned CUDA error/reporting helper.
                                                 //! NVVS common state
    NvvsCommon *m_nvvsCommon;
    dcgmHandle_t m_handle;
    std::unique_ptr<TestParameters> m_defaultTestParameters;
};


#endif // _NVVS_NVVS_MEMTEST_H_
