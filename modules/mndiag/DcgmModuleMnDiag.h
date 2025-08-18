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

#ifndef DCGMMODULEMNDIAG_H
#define DCGMMODULEMNDIAG_H


#include "DcgmMnDiagManager.h"
#include "dcgm_mndiag_structs.hpp"
#include <DcgmModule.h>

class DcgmModuleMnDiag : public DcgmModuleWithCoreProxy<DcgmModuleIdMnDiag>
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    explicit DcgmModuleMnDiag(dcgmCoreCallbacks_t &dcc);
    virtual ~DcgmModuleMnDiag();

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) override;

private:
    /*************************************************************************/
    // Handle commands when running as head node
    dcgmReturn_t ProcessHeadNodeMsg(dcgm_module_command_header_t *moduleCommand);
    // Handle commands when running as compute node
    dcgmReturn_t ProcessComputeNodeMsg(dcgm_module_command_header_t *moduleCommand);
    // Handle core DCGM messages
    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /* Private member variables */
    std::unique_ptr<DcgmMnDiagManager> m_mnDiagManager;
    std::atomic_bool m_isPaused { false };
};

#endif // DCGMMODULEMNDIAG_H
