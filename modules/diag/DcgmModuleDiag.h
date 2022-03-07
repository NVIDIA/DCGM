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
#ifndef DCGMMODULEDIAG_H
#define DCGMMODULEDIAG_H

#include "DcgmDiagManager.h"
#include "DcgmModule.h"
#include "dcgm_diag_structs.h"

class DcgmModuleDiag

    : public DcgmModuleWithCoreProxy<DcgmModuleIdDiag>
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    explicit DcgmModuleDiag(dcgmCoreCallbacks_t &dcc);
    virtual ~DcgmModuleDiag(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
private:
    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessRun(dcgm_diag_msg_run_t *msg);
    dcgmReturn_t ProcessStop(dcgm_diag_msg_stop_t *msg);
    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /* Private member variables */
    std::unique_ptr<DcgmDiagManager> mpDiagManager; /* Pointer to the worker class for this module */
};


#endif // DCGMMODULEDIAG_H
