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

#pragma once

#include "MnDiagMpiMnubergemmRunner.h"
#include "MnDiagMpiRunnerAdapter.h"
#include "MnDiagMpiRunnerFactoryBase.h"

/**
 * @brief Factory that instantiates the correct MnDiagMpiRunner subclass for a given test type.
 *        Add a new case here when introducing a new MPI test type.
 */
class MnDiagMpiRunnerFactory : public MnDiagMpiRunnerFactoryBase
{
public:
    /**
     * Create a MnDiagMpiRunnerAdapter wrapping the runner for the requested test type.
     *
     * @param[in] coreProxy  Reference to the DCGM core proxy for GPU interactions.
     * @param[in] testType   The multinode test type to create a runner for.
     *
     * @return Owning pointer to the adapter for known test types, or nullptr if testType is unrecognized.
     */
    [[nodiscard]] std::unique_ptr<MnDiagMpiRunnerBase> CreateMpiRunner(DcgmCoreProxyBase &coreProxy,
                                                                       dcgmMultinodeTestType_t testType,
                                                                       uid_t effectiveUid) override
    {
        switch (testType)
        {
            case dcgmMultinodeTestType_t::mnubergemm:
                return std::make_unique<MnDiagMpiRunnerAdapter>(
                    std::make_unique<MnDiagMpiMnubergemmRunner>(coreProxy, effectiveUid));
            default:
                return nullptr;
        }
    }
};
