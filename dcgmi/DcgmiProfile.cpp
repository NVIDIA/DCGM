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
#include "DcgmiProfile.h"

#include "MigIdParser.hpp"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgmi_common.h"
#include <DcgmStringHelpers.h>
#include <EntityListHelpers.h>

#include <sstream>


#define COLUMN_MG_ID     "Group.Subgroup"
#define COLUMN_FIELD_ID  "Field ID"
#define COLUMN_FIELD_TAG "Field Tag"

/*****************************************************************************/
dcgmReturn_t DcgmiProfile::RunProfileList(dcgmHandle_t dcgmHandle, unsigned int gpuId, bool outputAsJson)
{
    dcgmProfGetMetricGroups_t gmg;
    DcgmiOutputColumns outColumns;
    DcgmiOutputJson outJson;
    DcgmiOutput &out = outputAsJson ? (DcgmiOutput &)outJson : (DcgmiOutput &)outColumns;

    memset(&gmg, 0, sizeof(gmg));
    gmg.version = dcgmProfGetMetricGroups_version;
    gmg.gpuId   = gpuId;

    dcgmReturn_t dcgmReturn = dcgmProfGetSupportedMetricGroups(dcgmHandle, &gmg);
    if (dcgmReturn != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to Get supported metric groups: " << errorString(dcgmReturn) << "." << std::endl;
        return dcgmReturn;
    }

    DcgmiOutputFieldSelector idSelector       = DcgmiOutputFieldSelector().child(COLUMN_MG_ID);
    DcgmiOutputFieldSelector fieldIdSelector  = DcgmiOutputFieldSelector().child(COLUMN_FIELD_ID);
    DcgmiOutputFieldSelector fieldTagSelector = DcgmiOutputFieldSelector().child(COLUMN_FIELD_TAG);

    out.addColumn(16, COLUMN_MG_ID, idSelector);
    out.addColumn(10, COLUMN_FIELD_ID, fieldIdSelector);
    out.addColumn(54, COLUMN_FIELD_TAG, fieldTagSelector);

    for (unsigned int i = 0; i < gmg.numMetricGroups; i++)
    {
        dcgmProfMetricGroupInfo_v2 *mgInfo = &gmg.metricGroups[i];
        for (unsigned int j = 0; j < mgInfo->numFieldIds; j++)
        {
            std::string fieldTag;
            unsigned short fieldId      = mgInfo->fieldIds[j];
            dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
                fieldTag = "<<Unknown>>";
            else
                fieldTag = fieldMeta->tag;

            std::stringstream ss, ss1, ss2;
            ss << i << "_" << j;
            std::string uniqueTag = ss.str();

            char mgChar = 'A' + mgInfo->majorId; // A,B,C..etc. We only have 10 metric groups so far.
            ss1 << mgChar << "." << mgInfo->minorId;
            out[uniqueTag][COLUMN_MG_ID] = ss1.str();

            ss2 << fieldId;
            out[uniqueTag][COLUMN_FIELD_ID] = ss2.str();

            out[uniqueTag][COLUMN_FIELD_TAG] = fieldTag;
        }
    }

    std::cout << out.str();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmiProfile::RunProfileSetPause(dcgmHandle_t dcgmHandle, bool pause)
{
    dcgmReturn_t dcgmReturn;
    std::string action;
    if (pause)
    {
        action     = "pause";
        dcgmReturn = dcgmProfPause(dcgmHandle);
    }
    else
    {
        action     = "resume";
        dcgmReturn = dcgmProfResume(dcgmHandle);
    }

    if (dcgmReturn != DCGM_ST_OK)
    {
        std::cout << "Error: unable to " << action << " profiling metrics: " << errorString(dcgmReturn) << "."
                  << std::endl;
    }
    else
    {
        std::cout << "Successfully " << action << "d profiling." << std::endl;
    }
    return dcgmReturn;
}

/*****************************************************************************/
DcgmiProfileList::DcgmiProfileList(std::string hostname,
                                   std::string gpuIdsStr,
                                   std::string groupIdStr,
                                   bool outputAsJson)
    : mGpuIdsStr(std::move(gpuIdsStr))
    , mGroupIdStr(std::move(groupIdStr))
    , mGpuId(0)
{
    m_hostName = std::move(hostname);
    m_json     = outputAsJson;
}

/*****************************************************************************/
dcgmReturn_t DcgmiProfileList::SetGpuIdFromEntityList()
{
    std::vector<dcgmGroupEntityPair_t> entityList;
    auto err = DcgmNs::EntityListWithMigAndUuidParser(m_dcgmHandle, mGpuIdsStr, entityList);
    if (!err.empty())
    {
        SHOW_AND_LOG_ERROR << err;
        return DCGM_ST_BADPARAM;
    }

    /* Find the first GPU ID in our entity list */
    for (auto &entity : entityList)
    {
        if (entity.entityGroupId == DCGM_FE_GPU)
        {
            DCGM_LOG_DEBUG << "Using gpuId " << entity.entityId;
            mGpuId = entity.entityId;
            return DCGM_ST_OK;
        }
    }

    std::cout << "Error: No GPUs found in the provided entity list." << std::endl;
    return DCGM_ST_BADPARAM;
}

/*****************************************************************************/
dcgmReturn_t DcgmiProfileList::SetGpuId(void)
{
    dcgmReturn_t dcgmReturn;
    /**
     * Check if m_requestedEntityIds is set or not. If set, we use the first GPU ID we
     * find in the entity list.
     */
    if (mGpuIdsStr != "-1")
    {
        dcgmReturn = SetGpuIdFromEntityList();
        return dcgmReturn;
    }

    /* If no group ID or entity list was specified, Get the first GPU in the system to act like nvidia-smi dmon */
    if (mGroupIdStr == "-1")
    {
        unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES];
        int count  = 0;
        dcgmReturn = dcgmGetAllSupportedDevices(m_dcgmHandle, gpuIdList, &count);

        if (dcgmReturn != DCGM_ST_OK)
        {
            std::cout << "Error: dcgmGetAllSupportedDevices() returned " << errorString(dcgmReturn) << std::endl;
            return dcgmReturn;
        }
        else if (count < 1)
        {
            std::cout << "Error: dcgmGetAllSupportedDevices() returned 0 devices" << std::endl;
            return DCGM_ST_NOT_SUPPORTED;
        }
        mGpuId = gpuIdList[0];
        return DCGM_ST_OK;
    }

    /* Get the first GPU ID of the provided groupId */

    dcgmGroupType_t groupType = DCGM_GROUP_DEFAULT;
    dcgmGpuGrp_t groupId      = 0;

    /* Parse any special group ID strings and populate groupId */
    bool groupIdIsSpecial = dcgmi_entity_group_id_is_special(mGroupIdStr, &groupType, &groupId);
    if (!groupIdIsSpecial)
    {
        int groupIdAsInt = atoi(mGroupIdStr.c_str());
        if (!groupIdAsInt && mGroupIdStr.at(0) != '0')
        {
            std::cout << "Error: Expected a numerical groupId. Instead got '" << mGroupIdStr << "'" << std::endl;
            return DCGM_ST_BADPARAM;
        }

        groupId = (dcgmGpuGrp_t)(intptr_t)groupIdAsInt;
    }

    /* Try to get a handle to the group the user specified so we can get a GPU ID from it */
    std::unique_ptr<dcgmGroupInfo_t> groupInfo = std::make_unique<dcgmGroupInfo_t>();
    groupInfo->version                         = dcgmGroupInfo_version;

    dcgmReturn = dcgmGroupGetInfo(m_dcgmHandle, groupId, groupInfo.get());
    if (DCGM_ST_OK != dcgmReturn)
    {
        std::string error = (dcgmReturn == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(dcgmReturn);
        std::cout << "Error: Unable to retrieve information about group " << groupId << ". Return: " << error << "."
                  << std::endl;
        DCGM_LOG_ERROR << "Error! retrieving info on group. Return: " << dcgmReturn;
        return dcgmReturn;
    }

    for (size_t i = 0; i < groupInfo->count; i++)
    {
        if (groupInfo->entityList[i].entityGroupId == DCGM_FE_GPU)
        {
            mGpuId = groupInfo->entityList[i].entityId;
            DCGM_LOG_DEBUG << "Using gpuId " << mGpuId;
            return DCGM_ST_OK;
        }
    }

    std::cout << "There were no GPUs in group " << mGroupIdStr << std::endl;
    return DCGM_ST_BADPARAM;
}

/*****************************************************************************/
dcgmReturn_t DcgmiProfileList::DoExecuteConnected()
{
    /* Set mGpuId */
    auto const dcgmReturn = SetGpuId();
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    return mProfileObj.RunProfileList(m_dcgmHandle, mGpuId, m_json);
}

/*****************************************************************************/
DcgmiProfileSetPause::DcgmiProfileSetPause(std::string hostname, bool pause)
    : m_pause(pause)
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t DcgmiProfileSetPause::DoExecuteConnected()
{
    return mProfileObj.RunProfileSetPause(m_dcgmHandle, m_pause);
}

/*****************************************************************************/
