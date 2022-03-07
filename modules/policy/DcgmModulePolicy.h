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
#ifndef DCGMMODULEPOLICY_H
#define DCGMMODULEPOLICY_H

#include "DcgmModule.h"
#include "DcgmPolicyManager.h"
#include "dcgm_policy_structs.h"

class DcgmModulePolicy

    : public DcgmModuleWithCoreProxy<DcgmModuleIdPolicy>
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModulePolicy(dcgmCoreCallbacks_t &dcc);
    virtual ~DcgmModulePolicy(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /*
     * Process a client disconnecting (inherited from DcgmModule)
     */
    void OnClientDisconnect(dcgm_connection_id_t connectionId);

    /*************************************************************************/
private:
    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessGetPolicies(dcgm_policy_msg_get_policies_t *msg);
    dcgmReturn_t ProcessSetPolicy(dcgm_policy_msg_set_policy_t *msg);
    dcgmReturn_t ProcessRegister(dcgm_policy_msg_register_t *msg);
    dcgmReturn_t ProcessUnregister(dcgm_policy_msg_unregister_t *msg);
    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessFieldValuesUpdated(dcgm_core_msg_field_values_updated_t *msg);
    dcgmReturn_t ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg);

    /*************************************************************************/
    /* Private member variables */
    std::unique_ptr<DcgmPolicyManager> mpPolicyManager; /* Pointer to the worker class for this module */
};


#endif // DCGMMODULEPOLICY_H
