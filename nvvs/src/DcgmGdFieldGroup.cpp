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
#include "DcgmGdFieldGroup.h"
#include "DcgmLogging.h"
#include "dcgm_agent.h"

DcgmGdFieldGroup::DcgmGdFieldGroup()
    : m_fieldGroupId((dcgmFieldGrp_t) nullptr)
    , m_handle((dcgmHandle_t) nullptr)
{}

DcgmGdFieldGroup::~DcgmGdFieldGroup()
{
    Cleanup();
}

dcgmFieldGrp_t DcgmGdFieldGroup::GetFieldGroupId()
{
    return m_fieldGroupId;
}

dcgmReturn_t DcgmGdFieldGroup::Init(dcgmHandle_t handle,
                                    const std::vector<unsigned short> &fieldIds,
                                    const std::string &fieldGroupName)
{
    unsigned short fieldIdArray[DCGM_FI_MAX_FIELDS];
    int numFieldIds = fieldIds.size();
    char nameBuf[128];

    if (handle == (dcgmHandle_t) nullptr)
    {
        PRINT_ERROR("", "Cannot initialize with an invalid DCGM handle");
        return DCGM_ST_BADPARAM;
    }

    if (fieldIds.empty())
    {
        PRINT_ERROR("", "Cannot initialize the field group with an empty list of field ids");
        return DCGM_ST_BADPARAM;
    }

    m_handle = handle;

    for (size_t i = 0; i < fieldIds.size(); i++)
    {
        fieldIdArray[i] = fieldIds[i];
    }

    snprintf(nameBuf, sizeof(nameBuf), "%s", fieldGroupName.c_str());

    return dcgmFieldGroupCreate(m_handle, numFieldIds, fieldIdArray, nameBuf, &m_fieldGroupId);
}

dcgmReturn_t DcgmGdFieldGroup::Cleanup()
{
    if (m_handle == (dcgmHandle_t) nullptr)
    {
        return DCGM_ST_OK;
    }

    dcgmReturn_t ret = dcgmFieldGroupDestroy(m_handle, m_fieldGroupId);

    if (ret == DCGM_ST_OK)
    {
        m_fieldGroupId = 0;
    }

    return ret;
}
