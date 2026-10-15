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
#ifndef DCGMMODULEPOLICY_H
#define DCGMMODULEPOLICY_H

#include "DcgmModule.h"
#include "DcgmPolicyManager.h"
#include "dcgm_policy_structs.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>

class DcgmTaskRunner;

class DcgmModulePolicy : public DcgmModuleWithCoreProxy<DcgmModuleIdPolicy>
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModulePolicy(dcgmCoreCallbacks_t &dcc, std::chrono::milliseconds taskTimeout = std::chrono::seconds(30));
    virtual ~DcgmModulePolicy(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) override;

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

    /**
     * Validate and enqueue a field-policy create request.
     *
     * @param[in,out] msg Create request and created policy readback.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessCreatePolicy(dcgm_policy_msg_create_t *msg);

    /**
     * Validate and enqueue a field-policy modify request.
     *
     * @param[in,out] msg Modify request and modified policy readback.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessModifyPolicy(dcgm_policy_msg_modify_t *msg);

    /**
     * Validate and enqueue a field-policy delete request.
     *
     * @param[in] msg Delete request.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessDeletePolicy(dcgm_policy_msg_delete_t *msg);

    /**
     * Validate and enqueue a field-policy enable request.
     *
     * @param[in] msg Enable request.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessEnablePolicy(dcgm_policy_msg_set_state_t *msg);

    /**
     * Validate and enqueue a field-policy disable request.
     *
     * @param[in] msg Disable request.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessDisablePolicy(dcgm_policy_msg_set_state_t *msg);

    /**
     * Validate and enqueue a field-policy get-one request.
     *
     * @param[in,out] msg Get-one request and policy output.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessGetOnePolicy(dcgm_policy_msg_get_one_t *msg);

    /**
     * Validate and enqueue a field-policy get-all request.
     *
     * @param[in,out] msg Get-all request and policy output array.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessGetAllPolicies(dcgm_policy_msg_get_all_t *msg);

    /**
     * Validate and enqueue a field-policy import request.
     *
     * @param[in] msg Import request with pathLength bytes in path.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessImportPolicies(dcgm_policy_msg_import_t *msg);

    /**
     * Validate and enqueue a field-policy get-violations request.
     *
     * @param[in,out] msg Get-violations request and violation output array.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessGetViolations(dcgm_policy_msg_get_violations_t *msg);

    /**
     * Validate and enqueue a field-policy callback registration request.
     *
     * @param[in] msg Callback registration request.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessRegisterCallback(dcgm_policy_msg_register_callback_t *msg);

    /**
     * Validate and enqueue a field-policy callback unregistration request.
     *
     * @param[in] msg Callback unregistration request.
     * @return DCGM_ST_OK on success, or a validation/manager error code.
     */
    [[nodiscard]] dcgmReturn_t ProcessUnregisterCallback(dcgm_policy_msg_unregister_callback_t *msg);

    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessFieldValuesUpdated(dcgm_core_msg_field_values_updated_t *msg);
    dcgmReturn_t ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg);
    dcgmReturn_t ProcessAttachGpus(dcgm_core_msg_attach_gpus_t *msg);
    dcgmReturn_t ProcessDetachGpus(dcgm_core_msg_detach_gpus_t *msg);
    dcgmReturn_t EnqueueSync(std::string taskName,
                             std::function<dcgmReturn_t()> work,
                             std::function<void()> onComplete = {},
                             bool isMutating                  = false);
    dcgmReturn_t EnqueueAsync(std::string taskName, std::function<dcgmReturn_t()> work);

    /*************************************************************************/
    /* Private member variables */
    std::unique_ptr<DcgmPolicyManager> mpPolicyManager; /* Pointer to the worker class for this module */
    std::unique_ptr<DcgmTaskRunner> m_taskRunner;
    std::chrono::milliseconds m_taskTimeout;
};


#endif // DCGMMODULEPOLICY_H
