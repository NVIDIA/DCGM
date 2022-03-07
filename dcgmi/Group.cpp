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
/*
 * Group.cpp
 *
 */

#include "Group.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgmi_common.h"

#include <iostream>
#include <sstream>
#include <stdexcept>


/***************************************************************************/

static char const GROUP_HEADER[]      = "GROUPS";
static char const GROUP_INFO_HEADER[] = "GROUP INFO";

#define GROUP_ID_TAG      "Group ID"
#define GROUP_NAME_TAG    "Group Name"
#define GROUP_DEVICES_TAG "Entities"

#define MAX_SIZE_OF_DEVICE_STRING 62 /* Used for overflow (full length of group devices tag) */


/*****************************************************************************/

dcgmReturn_t Group::RunGroupList(dcgmHandle_t mDcgmHandle, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGpuGrp_t groupIdList[DCGM_MAX_NUM_GROUPS];
    unsigned int count = 0;
    DcgmiOutputTree outTree(20, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;
    std::ostringstream ss;

    groupId = 0;

    result = dcgmGroupGetAllIds(mDcgmHandle, groupIdList, &count);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot retrieve group list. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d", "Error: could not retrieve group lists. Return: %d", result);
        return result;
    }

    if (count == 0)
    {
        std::cout << "No groups found. Please create one. \n";
    }
    else
    {
        ss << count << " group" << ((count == 1) ? " " : "s ") << "found.";

        out.addHeader(GROUP_HEADER);
        out.addHeader(ss.str());

        for (unsigned int i = 0; i < count; i++)
        {
            groupId = groupIdList[i];
            ss.str("");
            ss << (unsigned int)(uintptr_t)groupId;

            // Group info handles the display of each group (by appending to out)
            result = RunGroupInfo(mDcgmHandle, out["Groups"][ss.str()]);
            if (DCGM_ST_OK != result)
            {
                PRINT_ERROR("%u %d",
                            "Error in displaying group info with group ID: %u. Return: %d",
                            (unsigned int)(uintptr_t)groupId,
                            result);
                return result;
            }
        }
    }

    std::cout << out.str();

    return result;
}

/*******************************************************************************/
dcgmReturn_t Group::RunGroupCreate(dcgmHandle_t mDcgmHandle, dcgmGroupType_t groupType)
{
    dcgmReturn_t result     = DCGM_ST_OK;
    dcgmGpuGrp_t newGroupId = 0;

    result = dcgmGroupCreate(mDcgmHandle, groupType, (char *)groupName.c_str(), &newGroupId);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot create group " << groupName << ". Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%u %d", "Error: could not create group with ID: %u. Return: %d", (unsigned int)newGroupId, result);
        return result;
    }

    std::cout << "Successfully created group \"" << groupName << "\" with a group ID of " << (unsigned int)newGroupId
              << std::endl;

    // Add Devices to new group if specified
    if (!groupInfo.empty() && groupType == DCGM_GROUP_EMPTY)
    {
        groupId = newGroupId;
        result  = RunGroupManageDevice(mDcgmHandle, true);
    }

    return result;
}

/*******************************************************************************/
dcgmReturn_t Group::RunGroupDestroy(dcgmHandle_t mDcgmHandle)
{
    dcgmReturn_t result = DCGM_ST_OK;

    result = dcgmGroupDestroy(mDcgmHandle, groupId);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Cannot destroy group " << (unsigned int)(uintptr_t)groupId << ". Return: " << error << "."
                  << std::endl;
        PRINT_ERROR(
            "%u %d", "Error in destroying group with ID: %u. Return: %d", (unsigned int)(uintptr_t)groupId, result);
        return result;
    }

    std::cout << "Successfully removed group " << (unsigned int)(uintptr_t)groupId << std::endl;

    return DCGM_ST_OK;
}

/*******************************************************************************/
dcgmReturn_t Group::RunGroupInfo(dcgmHandle_t dcgmHandle, bool outputJson)
{
    DcgmiOutputTree outTree(20, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = outputJson ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;
    std::ostringstream ss;
    ss << (unsigned int)(uintptr_t)groupId;
    DcgmiOutputBoxer &outGroup = out[ss.str()];

    out.addHeader(GROUP_INFO_HEADER);

    dcgmReturn_t result = RunGroupInfo(dcgmHandle, outGroup);

    if (result == DCGM_ST_OK)
    {
        std::cout << out.str();
    }

    return result;
}

dcgmReturn_t Group::RunGroupInfo(dcgmHandle_t dcgmHandle, DcgmiOutputBoxer &outGroup)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupInfo_t stDcgmGroupInfo;
    std::stringstream ss;

    stDcgmGroupInfo.version = dcgmGroupInfo_version;
    result                  = dcgmGroupGetInfo(dcgmHandle, groupId, &stDcgmGroupInfo);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to retrieve information about group " << (unsigned int)(uintptr_t)groupId
                  << ". Return: " << error << "." << std::endl;
        PRINT_ERROR("%u %d",
                    "Error retrieving info on group with ID: %u. Return: %d",
                    (unsigned int)(uintptr_t)groupId,
                    result);
        return result;
    }

    outGroup[GROUP_ID_TAG] = (unsigned int)(uintptr_t)groupId;

    outGroup[GROUP_NAME_TAG] = std::string(stDcgmGroupInfo.groupName);

    // Create GPU List string to display
    if (stDcgmGroupInfo.count == 0)
        ss << "None";
    for (unsigned int i = 0; i < stDcgmGroupInfo.count; i++)
    {
        ss << DcgmFieldsGetEntityGroupString(stDcgmGroupInfo.entityList[i].entityGroupId) << " ";
        ss << stDcgmGroupInfo.entityList[i].entityId;
        if (i < stDcgmGroupInfo.count - 1)
            ss << ", ";
    }
#if 0
    // Code can be used if support is issued for more than 16 GPUs to display properly.
    // If it is too large to fit into stencil, break it into parts to display
    std::string strHold = ss.str();
    dcgmDisplayParameter_t deviceOverflow;
    unsigned int p = 0;
    unsigned int start = 0;

    if (strHold.length() > MAX_SIZE_OF_DEVICE_STRING){
        while (start < strHold.length()){
            p += MAX_SIZE_OF_DEVICE_STRING;
            if (p >= strHold.length()) p = strHold.length() - 1;

            else { // Put pointer to last available digit
                while (isdigit(strHold.at(p))){
                    if (p + 1 < strHold.length() && !isdigit(strHold.at(p + 1))) break; //check if landed on end of a digit
                    p--;
                }
                while (!isdigit(strHold.at(p))){
                    p--;
                }
            }
            // p is now the index of a the last digit of a GPU ID
            ss.str(strHold.substr(start, p - start + 1));
            if (start == 0){
                outGroup[GROUP_DEVICES_TAG] = ss.str();
            } else {
                outGroup[GROUP_DEVICES_TAG].addOverflow(ss.str());
            }
            start = p + 3; // 3 characters till the start of the next GPU ID
        }
    }
    else {
        outGroup[GROUP_DEVICES_TAG] = ss.str();
    }
#endif

    outGroup[GROUP_DEVICES_TAG] = ss.str();

    return result;
}

/*******************************************************************************/
dcgmReturn_t Group::RunGroupManageDevice(dcgmHandle_t dcgmHandle, bool add)
{
    auto [entityList, rejectedIds] = DcgmNs::TryParseEntityList(dcgmHandle, groupInfo);

    // Fallback to old method

    std::vector<dcgmGroupEntityPair_t> oldEntityList;
    dcgmReturn_t result = DCGM_ST_OK;

    // Assume that GroupStringParse will print error to user, no output needed here.
    result = dcgmi_parse_entity_list_string(rejectedIds, oldEntityList);
    if (DCGM_ST_OK != result)
    {
        PRINT_ERROR("%d", "Error: parsing for GPUs failed. Return: %d", result);
        return result;
    }

    std::move(begin(oldEntityList), end(oldEntityList), std::back_inserter(entityList));

    for (unsigned int i = 0; i < entityList.size(); i++)
    {
        if (add)
        {
            result = dcgmGroupAddEntity(dcgmHandle, groupId, entityList[i].entityGroupId, entityList[i].entityId);
        }
        else
        {
            result = dcgmGroupRemoveEntity(dcgmHandle, groupId, entityList[i].entityGroupId, entityList[i].entityId);
        }
        if (DCGM_ST_OK != result)
        {
            std::string error;
            if (result == DCGM_ST_NOT_CONFIGURED)
            {
                error = "The Group is not found";
            }
            else
            {
                error = errorString(result);
                if (result == DCGM_ST_BADPARAM)
                {
                    error += ".\nThe GPU was not found or is ";
                    error += (add ? "already in the group" : "not part of the group");
                }
            }
            std::cout << (i > 0 ? "Operation partially successful." : "") << std::endl;
            std::cout << "Error: Unable to perform " << (add ? "add " : "remove ") << "of "
                      << DcgmFieldsGetEntityGroupString(entityList[i].entityGroupId) << " " << entityList[i].entityId;
            std::cout << " in" << (add ? "to " : " ") << "group " << (unsigned int)(uintptr_t)groupId
                      << ". Return: " << error << "." << std::endl;
            return result;
        }
    }

    std::cout << (add ? "Add to " : "Remove from ") << "group operation successful." << std::endl;
    return result;
}

// Getters and Setters
void Group::SetGroupId(unsigned int id)
{
    this->groupId = (dcgmGpuGrp_t)(long long)id;
}
unsigned int Group::GetGroupId() const
{
    return (unsigned int)groupId;
}
void Group::SetGroupName(std::string name)
{
    groupName = std::move(name);
}
std::string Group::getGroupName()
{
    return groupName;
}
void Group::SetGroupInfo(std::string info)
{
    groupInfo = std::move(info);
}
std::string Group::getGroupInfo()
{
    return groupInfo;
}

/*****************************************************************************
 *****************************************************************************
 * Group List Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupList::GroupList(std::string hostname, bool json)
    : Command()
{
    m_hostName = std::move(hostname);
    m_json     = json;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t GroupList::DoExecuteConnected()
{
    return groupObj.RunGroupList(m_dcgmHandle, m_json);
}

/*****************************************************************************
 *****************************************************************************
 * Group Create Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupCreate::GroupCreate(std::string hostname, Group &obj, dcgmGroupType_t groupType)
    : Command()
    , groupObj(obj)
    , groupType(groupType)
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t GroupCreate::DoExecuteConnected()
{
    return groupObj.RunGroupCreate(m_dcgmHandle, groupType);
}


/*****************************************************************************
 *****************************************************************************
 * Group Destroy Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupDestroy::GroupDestroy(std::string hostname, Group &obj)
    : Command()
    , groupObj(obj)
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t GroupDestroy::DoExecuteConnected()
{
    return groupObj.RunGroupDestroy(m_dcgmHandle);
}

/*****************************************************************************
 *****************************************************************************
 * Group Info Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupInfo::GroupInfo(std::string hostname, Group &obj, bool json)
    : Command()
    , groupObj(obj)
{
    m_hostName = std::move(hostname);
    m_json     = json;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}


/*****************************************************************************/
dcgmReturn_t GroupInfo::DoExecuteConnected()
{
    return groupObj.RunGroupInfo(m_dcgmHandle, m_json);
}

/*****************************************************************************
 *****************************************************************************
 * Add to Group Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupAddTo::GroupAddTo(std::string hostname, Group &obj)
    : Command()
    , groupObj(obj)
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t GroupAddTo::DoExecuteConnected()
{
    return groupObj.RunGroupManageDevice(m_dcgmHandle, true);
}

/*****************************************************************************
 *****************************************************************************
 * Delete from Group Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GroupDeleteFrom::GroupDeleteFrom(std::string hostname, Group &obj)
    : Command()
    , groupObj(obj)
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t GroupDeleteFrom::DoExecuteConnected()
{
    return groupObj.RunGroupManageDevice(m_dcgmHandle, false);
}
