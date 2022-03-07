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
#ifndef DCGMMODULECONFIG_H
#define DCGMMODULECONFIG_H

#include "DcgmConfigManager.h"
#include "DcgmModule.h"
#include "dcgm_config_structs.h"

class DcgmModuleConfig

    : public DcgmModuleWithCoreProxy<DcgmModuleIdConfig>
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleConfig(dcgmCoreCallbacks_t &dcc);
    virtual ~DcgmModuleConfig(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /*
     * Virtual method for this module to handle when a client disconnects from
     * DCGM.
     *
     */
    void OnClientDisconnect(dcgm_connection_id_t connectionId);

    /*************************************************************************/
private:
    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessGetConfig(dcgm_config_msg_get_v1 *msg);
    dcgmReturn_t ProcessSetConfig(dcgm_config_msg_set_v1 *msg);
    dcgmReturn_t ProcessEnforceConfigGroup(dcgm_config_msg_enforce_group_v1 *msg);
    dcgmReturn_t ProcessEnforceConfigGpu(dcgm_config_msg_enforce_gpu_v1 *msg);
    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /* Private member variables */
    std::unique_ptr<DcgmConfigManager> mpConfigManager; /* Pointer to the worker class for this module */
};


#endif // DCGMMODULECONFIG_H
