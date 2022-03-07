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

#include "FieldGroup.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include <ctype.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>


/***************************************************************************/

/* List Field Group */
static const std::string FG_HEADER = "FIELD GROUPS";

static const std::string FG_ID_TAG        = "ID";
static const std::string FG_NAME_TAG      = "Name";
static const std::string FG_FIELD_IDS_TAG = "Field IDs";

// Used for overflow (full length of group devices tag)
#define MAX_FIELDS_PER_LINE 9


/*****************************************************************************/

dcgmReturn_t FieldGroup::RunGroupListAll(dcgmHandle_t dcgmHandle, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    m_fieldGroupId      = 0;
    dcgmAllFieldGroup_t allGroupInfo;

    memset(&allGroupInfo, 0, sizeof(allGroupInfo));
    allGroupInfo.version = dcgmAllFieldGroup_version;
    result               = dcgmFieldGroupGetAll(dcgmHandle, &allGroupInfo);

    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot retrieve field group list. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d", "Error: could not retrieve field group lists. Return: %d", result);
        return result;
    }

    if (allGroupInfo.numFieldGroups == 0)
    {
        std::cout << "No field groups found. Please create one.\n";
    }
    else
    {
        std::cout << allGroupInfo.numFieldGroups << " field group" << ((allGroupInfo.numFieldGroups == 1) ? " " : "s ")
                  << "found." << std::endl;
        for (unsigned int i = 0; i < allGroupInfo.numFieldGroups; i++)
        {
            m_fieldGroupId = (unsigned int)(intptr_t)allGroupInfo.fieldGroups[i].fieldGroupId;

            // Group info handles the display of each group
            result = RunGroupInfo(dcgmHandle, json);
            if (DCGM_ST_OK != result)
            {
                PRINT_ERROR("%u %d",
                            "Error in displaying field group info with fielg group ID: %u. Return: %d",
                            m_fieldGroupId,
                            (int)result);
                return result;
            }
        }
    }

    return result;
}

/*******************************************************************************/
dcgmReturn_t FieldGroup::RunGroupCreate(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmFieldGrp_t newFieldGroupId;
    std::vector<unsigned short> fieldIds;

    result = HelperFieldIdStringParse(m_fieldIdsStr, fieldIds);
    if (DCGM_ST_OK != result)
    {
        PRINT_ERROR("%d", "Error: parsing for field IDs failed. Return: %d", result);
        return result;
    }

    result = dcgmFieldGroupCreate(
        dcgmHandle, fieldIds.size(), &fieldIds[0], (char *)m_fieldGroupName.c_str(), &newFieldGroupId);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot create field group " << m_fieldGroupName << ". Return: " << errorString(result)
                  << std::endl;
        PRINT_ERROR("%s %d",
                    "Error: could not create group with name: %s. Return: %d",
                    (char *)m_fieldGroupName.c_str(),
                    result);
        return result;
    }

    std::cout << "Successfully created field group \"" << m_fieldGroupName << "\" with a field group ID of "
              << (unsigned int)(uintptr_t)newFieldGroupId << std::endl;
    return result;
}

/*******************************************************************************/
dcgmReturn_t FieldGroup::RunGroupDestroy(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t result           = DCGM_ST_OK;
    dcgmFieldGrp_t dcgmFieldGrpId = (dcgmFieldGrp_t)(intptr_t)m_fieldGroupId;

    result = dcgmFieldGroupDestroy(dcgmHandle, dcgmFieldGrpId);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Cannot destroy field group " << m_fieldGroupId << ". Return: " << error << "."
                  << std::endl;
        PRINT_ERROR("%u %d", "Error in destroying field group with ID: %d. Return: %d", m_fieldGroupId, result);
        return result;
    }

    std::cout << "Successfully removed field group " << m_fieldGroupId << std::endl;
    return DCGM_ST_OK;
}

/*******************************************************************************/
dcgmReturn_t FieldGroup::RunGroupInfo(dcgmHandle_t dcgmHandle, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmFieldGroupInfo_t fieldGroupInfo;
    DcgmiOutputTree outTree(20, 60);
    DcgmiOutputJson outJson;
    DcgmiOutput &out        = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;
    unsigned int fieldCount = 0;
    std::stringstream ss;

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version      = dcgmFieldGroupInfo_version;
    fieldGroupInfo.fieldGroupId = (dcgmFieldGrp_t)(intptr_t)m_fieldGroupId;

    result = dcgmFieldGroupGetInfo(dcgmHandle, &fieldGroupInfo);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Field Group is not found" : errorString(result);
        std::cout << "Error: Unable to retrieve information about field group " << m_fieldGroupId
                  << ". Return: " << error << "." << std::endl;
        PRINT_ERROR(
            "%u %d", "Error retrieving info on field group with ID: %u. Return: %d", (int)m_fieldGroupId, result);
        return result;
    }

    out.addHeader(FG_HEADER);

    out[FG_ID_TAG] = m_fieldGroupId;

    out[FG_NAME_TAG] = fieldGroupInfo.fieldGroupName;

    // Create field ID list string to display
    if (fieldGroupInfo.numFieldIds == 0)
        ss << "None";
    for (unsigned int i = 0; i < fieldGroupInfo.numFieldIds; i++)
    {
        ss << fieldGroupInfo.fieldIds[i];
        if (i < fieldGroupInfo.numFieldIds - 1)
            ss << ", ";
        if (fieldCount > MAX_FIELDS_PER_LINE)
        {
            out[FG_FIELD_IDS_TAG].setOrAppend(ss.str());
            ss.str("");
            fieldCount = 0;
        }
        fieldCount++;
    }

    // If there are fields we haven't printed
    if (ss.str().length() > 0)
        out[FG_FIELD_IDS_TAG].setOrAppend(ss.str());

    std::cout << out.str();

    return result;
}

/*******************************************************************************/
dcgmReturn_t FieldGroup::HelperFieldIdStringParse(std::string input, std::vector<unsigned short> &fieldIds)
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::stringstream ss(input);
    unsigned int buff;

    // Check for valid input
    if (!isdigit(input.at(0)))
    {
        std::cout << "Error: Invalid first character detected: \"" << input.at(0) << "\" while parsing for field IDs."
                  << std::endl;
        return DCGM_ST_BADPARAM;
    }

    for (unsigned int i = 0; i < input.length(); i++)
    {
        if ((input.at(i) != ',') && !(isdigit(input.at(i))))
        {
            std::cout << "Error: Invalid character detected: \"" << input.at(i) << "\" while parsing for field IDs."
                      << std::endl;
            return DCGM_ST_BADPARAM;
        }
    }

    // Add GPU IDs to vector
    while (ss >> buff)
    {
        fieldIds.push_back(buff);

        if (ss.peek() == ',')
        {
            ss.ignore();
        }
    }

    return result;
}

// Getters and Setters
void FieldGroup::SetFieldGroupId(unsigned int id)
{
    this->m_fieldGroupId = id;
}
unsigned int FieldGroup::GetFieldGroupId()
{
    return m_fieldGroupId;
}
void FieldGroup::SetFieldGroupName(std::string name)
{
    m_fieldGroupName = std::move(name);
}
std::string FieldGroup::GetFieldGroupName()
{
    return m_fieldGroupName;
}

void FieldGroup::SetFieldIdsString(std::string fieldIdsString)
{
    m_fieldIdsStr = fieldIdsString;
}

/*****************************************************************************
 *****************************************************************************
 * Field Group Create Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldGroupCreate::FieldGroupCreate(std::string hostname, FieldGroup &obj)
{
    m_hostName    = std::move(hostname);
    fieldGroupObj = obj;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t FieldGroupCreate::DoExecuteConnected()
{
    return fieldGroupObj.RunGroupCreate(m_dcgmHandle);
}


/*****************************************************************************
 *****************************************************************************
 * Group Destroy Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldGroupDestroy::FieldGroupDestroy(std::string hostname, FieldGroup &obj)
{
    m_hostName    = std::move(hostname);
    fieldGroupObj = obj;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t FieldGroupDestroy::DoExecuteConnected()
{
    return fieldGroupObj.RunGroupDestroy(m_dcgmHandle);
}

/*****************************************************************************
 *****************************************************************************
 * Field Group List All Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldGroupListAll::FieldGroupListAll(std::string hostname, bool json)
{
    m_hostName = hostname;
    m_json     = json;
}

/*****************************************************************************/
dcgmReturn_t FieldGroupListAll::DoExecuteConnected()
{
    return fieldGroupObj.RunGroupListAll(m_dcgmHandle, m_json);
}


/*****************************************************************************
 *****************************************************************************
 * Field Group Info Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
FieldGroupInfo::FieldGroupInfo(std::string hostname, FieldGroup &obj, bool json)
{
    m_hostName    = hostname;
    fieldGroupObj = obj;
    m_json        = json;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t FieldGroupInfo::DoExecuteConnected()
{
    return fieldGroupObj.RunGroupInfo(m_dcgmHandle, m_json);
}
