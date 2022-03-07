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
#ifndef DCGMMODULEHEALTH_H
#define DCGMMODULEHEALTH_H

#include "DcgmHealthWatch.h"
#include "DcgmModule.h"
#include "dcgm_health_structs.h"

class DcgmModuleHealth

    : public DcgmModuleWithCoreProxy<DcgmModuleIdHealth>
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleHealth(dcgmCoreCallbacks_t &dcc);
    virtual ~DcgmModuleHealth(); /* Virtual because of ancient C++ library */

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
    dcgmReturn_t ProcessGetSystems(dcgm_health_msg_get_systems_t *msg);
    dcgmReturn_t ProcessSetSystems(dcgm_health_msg_set_systems_t *msg);
    dcgmReturn_t ProcessCheckV4(dcgm_health_msg_check_v4 *msg);
    dcgmReturn_t ProcessCheckGpus(dcgm_health_msg_check_gpus_t *msg);
    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessFieldValuesUpdated(dcgm_core_msg_field_values_updated_t *msg);
    dcgmReturn_t ProcessGroupRemoved(dcgm_core_msg_group_removed_t *msg);

    /*************************************************************************/
    /* Private member variables */
    std::unique_ptr<DcgmHealthWatch> mpHealthWatch; /* Pointer to the worker class for this module */
};


#endif // DCGMMODULEHEALTH_H
