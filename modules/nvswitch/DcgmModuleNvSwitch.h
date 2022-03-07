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
#pragma once

#include <DcgmModule.h>
#include <DcgmTaskRunner.h>
#include <dcgm_core_structs.h>

#include "DcgmNvSwitchManager.h"
#include "dcgm_nvswitch_structs.h"

namespace DcgmNs
{
class DcgmModuleNvSwitch

    : public DcgmModuleWithCoreProxy<DcgmModuleIdNvSwitch>
    , DcgmTaskRunner

{
public:
    explicit DcgmModuleNvSwitch(dcgmCoreCallbacks_t &dcc);

    ~DcgmModuleNvSwitch();
    /*************************************************************************/
    /**
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) override;

    /*************************************************************************/
    /*
     * Process a DCGM module message from our taskrunner thread.
     */
    dcgmReturn_t ProcessMessageFromTaskRunner(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /*
     * This is the main background worker function of the module.
     *
     * Returns: Minimum ms before we should call this function again. This will
     *          be how long we block on QueueTask() being called again.
     *          Returning 0 = Don't care when we get called back.
     */
    unsigned int RunOnce();

private:
    DcgmNvSwitchManager m_switchMgr;
    std::chrono::system_clock::time_point m_nextWakeup
        = std::chrono::system_clock::time_point::min(); /*!< Next time when RunOnce should be called. */
    std::chrono::milliseconds m_runInterval {}; /*!< Last result of the latest successful RunOnce function call made in
                                                     the TryRunOnce method.  */
    timelib64_t m_lastLinkStatusUpdateUsec;     /* When we last rescanned link statuses in usec since 1970 */

    /*************************************************************************/
    dcgmReturn_t ProcessGetSwitchIds(dcgm_nvswitch_msg_get_switches_v1 *moduleCommand);
    dcgmReturn_t ProcessWatchField(const dcgm_nvswitch_msg_watch_field_t *const msg);
    dcgmReturn_t ProcessUnwatchField(const dcgm_nvswitch_msg_unwatch_field_t *const msg);
    dcgmReturn_t ProcessSetEntityNvLinkLinkState(dcgm_nvswitch_msg_set_link_state_t *msg);
    dcgmReturn_t ProcessGetAllLinkStates(dcgm_nvswitch_msg_get_all_link_states_t *msg);
    dcgmReturn_t ProcessGetLinkStates(dcgm_nvswitch_msg_get_link_states_t *msg);
    dcgmReturn_t SetEntityNvLinkLinkState(dcgm_nvswitch_msg_set_link_state_t *msg);
    dcgmReturn_t ProcessGetEntityStatus(dcgm_nvswitch_msg_get_entity_status_t *msg);
    dcgmReturn_t ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg);
    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);
    std::chrono::system_clock::time_point TryRunOnce(bool forceRun);

    /*
     */
    void run() override;
};

} // namespace DcgmNs
