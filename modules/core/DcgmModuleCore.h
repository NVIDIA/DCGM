/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
// #pragma once
#pragma once

#include "DcgmModule.h"
#include "dcgm_core_structs.h"
#include <DcgmCacheManager.h>
#include <DcgmGroupManager.h>
#include <vector>

class DcgmModuleCore : public DcgmModule
{
public:
    DcgmModuleCore();
    ~DcgmModuleCore();

    void Initialize(DcgmCacheManager *cm);

    void SetGroupManager(DcgmGroupManager *gm);

    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessSetLoggingSeverity(dcgm_core_msg_set_severity_t &moduleCommand);
    dcgmReturn_t ProcessCreateMigEntity(dcgm_core_msg_create_mig_entity_t &msg);
    dcgmReturn_t ProcessDeleteMigEntity(dcgm_core_msg_delete_mig_entity_t &msg);
    dcgmReturn_t ProcessGetGpuStatus(dcgm_core_msg_get_gpu_status_t &msg);
    dcgmReturn_t ProcessHostengineVersion(dcgm_core_msg_hostengine_version_t &msg);
    dcgmReturn_t ProcessCreateGroup(dcgm_core_msg_create_group_t &msg);
    dcgmReturn_t ProcessRemoveEntity(dcgm_core_msg_remove_entity_t &msg);
    dcgmReturn_t ProcessGroupDestroy(dcgm_core_msg_group_destroy_t &msg);
    dcgmReturn_t ProcessGetEntityGroupEntities(dcgm_core_msg_get_entity_group_entities_t &msg);
    dcgmReturn_t ProcessGroupGetAllIds(dcgm_core_msg_group_get_all_ids_t &msg);
    dcgmReturn_t ProcessGroupGetInfo(dcgm_core_msg_group_get_info_t &msg);

    dcgmModuleProcessMessage_f GetMessageProcessingCallback() const;

private:
    DcgmCacheManager *m_cacheManager;
    DcgmGroupManager *m_groupManager;
    dcgmModuleProcessMessage_f m_processMsgCB;
};
