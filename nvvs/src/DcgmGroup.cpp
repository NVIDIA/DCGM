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
#include <string.h>

#include "DcgmGroup.h"
#include "DcgmLogging.h"
#include "dcgm_agent.h"

DcgmGroup::DcgmGroup()
    : m_groupId(0)
    , m_handle(0)
    , m_fieldGroup(nullptr)
{
    memset(&m_info, 0, sizeof(m_info));
}

DcgmGroup::DcgmGroup(DcgmGroup &&other) noexcept
    : m_groupId(other.m_groupId)
    , m_handle(other.m_handle)
    , m_fieldGroup(other.m_fieldGroup)
{
    other.m_handle     = 0;
    other.m_groupId    = 0;
    other.m_fieldGroup = nullptr;
    memset(&m_info, 0, sizeof(m_info));
}

DcgmGroup::~DcgmGroup()
{
    Cleanup();
}

dcgmReturn_t DcgmGroup::Init(dcgmHandle_t handle, const std::string &groupNameStr)
{
    char groupName[128];

    if (handle == 0)
    {
        PRINT_ERROR("", "Cannot initialize the DCGM group with an invalid DCGM handle");
        return DCGM_ST_BADPARAM;
    }

    if (m_groupId != 0)
    {
        PRINT_ERROR("", "Cannot initialize the DCGM group - it has already been initialized");
        return DCGM_ST_BADPARAM;
    }

    m_handle = handle;

    snprintf(groupName, sizeof(groupName), "%s", groupNameStr.c_str());

    return dcgmGroupCreate(m_handle, DCGM_GROUP_EMPTY, groupName, &m_groupId);
}

dcgmReturn_t DcgmGroup::Init(dcgmHandle_t handle, const std::string &groupName, const std::vector<unsigned int> &gpuIds)
{
    if (handle == 0)
    {
        PRINT_ERROR("", "Cannot initialize the DCGM group with an invalid DCGM handle");
        return DCGM_ST_BADPARAM;
    }

    if (m_groupId != 0)
    {
        PRINT_ERROR("", "Cannot initialize the DCGM group - it has already been initialized");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = Init(handle, groupName);

    if (ret == DCGM_ST_OK)
    {
        for (size_t i = 0; i < gpuIds.size(); i++)
        {
            if ((ret = AddGpu(gpuIds[i])) != DCGM_ST_OK)
            {
                // Don't return with a partially created group
                dcgmGroupDestroy(m_handle, m_groupId);
                break;
            }
        }
    }

    return ret;
}


dcgmReturn_t DcgmGroup::AddGpu(unsigned int gpuId)
{
    if (m_handle == 0)
    {
        PRINT_ERROR("", "Cannot add a GPU to a group that does not have a valid DCGM handle");
        return DCGM_ST_BADPARAM;
    }

    if (m_groupId == 0)
    {
        PRINT_ERROR("", "Cannot add a GPU to a group that does not exist yet");
        return DCGM_ST_BADPARAM;
    }

    // Mark that we need to refresh group information
    m_info.version = 0;

    return dcgmGroupAddDevice(m_handle, m_groupId, gpuId);
}

dcgmReturn_t DcgmGroup::Cleanup()
{
    if (m_handle == 0 || m_groupId == 0)
        return DCGM_ST_OK;

    dcgmReturn_t ret = FieldGroupDestroy();
    dcgmReturn_t tmp;

    tmp = dcgmGroupDestroy(m_handle, m_groupId);
    if (tmp == DCGM_ST_OK)
    {
        m_groupId      = 0;
        m_info.version = 0;
    }
    else if (ret == DCGM_ST_OK)
        ret = tmp;

    return ret;
}

dcgmReturn_t DcgmGroup::RefreshGroupInfo()
{
    // If the version is set then we have good info
    if (m_info.version != 0)
        return DCGM_ST_OK;
    else if (m_handle == 0)
    {
        PRINT_ERROR("", "Cannot refresh group information because we do not have a valid handle to DCGM");
        return DCGM_ST_BADPARAM;
    }

    m_info.version   = dcgmGroupInfo_version2;
    dcgmReturn_t ret = dcgmGroupGetInfo(m_handle, m_groupId, &m_info);
    if (ret != DCGM_ST_OK)
        m_info.version = 0;

    return ret;
}

dcgmReturn_t DcgmGroup::GetConfig(dcgmConfig_t current[], unsigned int maxSize, unsigned int &actualSize)
{
    dcgmReturn_t ret      = RefreshGroupInfo();
    dcgmStatus_t stHandle = 0;

    if (ret != DCGM_ST_OK)
        return ret;

    // Return an error if there aren't enough slots for the status
    if (m_info.count > maxSize)
    {
        PRINT_INFO("%u %u",
                   "We cannot save the config status because we received %u and have a max count of %u",
                   m_info.count,
                   maxSize);
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    ret = dcgmStatusCreate(&stHandle);

    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%s", "Failed to create the status in DCGM: '%s'", errorString(ret));
        return ret;
    }

    actualSize = m_info.count;
    for (unsigned int i = 0; i < actualSize; i++)
        current[i].version = dcgmConfig_version1;

    ret = dcgmConfigGet(m_handle, m_groupId, DCGM_CONFIG_CURRENT_STATE, actualSize, current, stHandle);

    // Ignore the return
    dcgmStatusDestroy(stHandle);

    return ret;
}

dcgmGpuGrp_t DcgmGroup::GetGroupId()
{
    return m_groupId;
}


dcgmReturn_t DcgmGroup::FieldGroupCreate(const std::vector<unsigned short> &fieldIds, const std::string &fieldGroupName)
{
    DcgmGdFieldGroup *dfg = new DcgmGdFieldGroup();
    dcgmReturn_t ret      = dfg->Init(m_handle, fieldIds, fieldGroupName);

    if (ret == DCGM_ST_OK)
        m_fieldGroup = dfg;
    else
        delete dfg;

    return ret;
}

dcgmReturn_t DcgmGroup::FieldGroupDestroy()
{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (m_fieldGroup != nullptr)
    {
        ret = m_fieldGroup->Cleanup();
        delete m_fieldGroup;
        m_fieldGroup = 0;
    }

    return ret;
}


dcgmReturn_t DcgmGroup::WatchFields(long long frequency, double keepAge)
{
    if (m_handle == 0)
    {
        PRINT_INFO("", "Cannot watch fields with an invalid DCGM handle.");
        return DCGM_ST_BADPARAM;
    }

    if (m_fieldGroup == nullptr)
    {
        PRINT_INFO("", "Cannot watch fields without an initialized field group");
        return DCGM_ST_BADPARAM;
    }

    return dcgmWatchFields(m_handle, m_groupId, m_fieldGroup->GetFieldGroupId(), frequency, keepAge, 0);
}

dcgmReturn_t DcgmGroup::GetValuesSince(long long timestamp,
                                       dcgmFieldValueEntityEnumeration_f checker,
                                       void *userData,
                                       long long *nextTs)
{
    long long dummyTs = 0;

    if (m_handle == 0 || m_fieldGroup == nullptr)
        return DCGM_ST_BADPARAM;

    if (nextTs == NULL)
        nextTs = &dummyTs;

    return dcgmGetValuesSince_v2(
        m_handle, m_groupId, m_fieldGroup->GetFieldGroupId(), timestamp, nextTs, checker, userData);
}
