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

#include <DcgmVariantHelper.hpp>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <unistd.h>

#include "DcgmStringHelpers.h"
#include "MigIdParser.hpp"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgmi_common.h"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <sys/ioctl.h>

/*******************************************************************************/
dcgmReturn_t dcgmi_create_entity_group(dcgmHandle_t dcgmHandle,
                                       dcgmGroupType_t groupType,
                                       dcgmGpuGrp_t *groupId,
                                       std::vector<dcgmGroupEntityPair_t> &entityList)
{
    dcgmReturn_t dcgmReturn;
    static int numGroupsCreated = 0;
    unsigned int myPid          = (unsigned int)getpid();
    char groupName[32]          = { 0 };
    unsigned int i;

    snprintf(groupName, sizeof(groupName) - 1, "dcgmi_%u_%d", myPid, ++numGroupsCreated);

    dcgmReturn = dcgmGroupCreate(dcgmHandle, groupType, groupName, groupId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << "Got error while creating a GPU group: " << dcgmReturn << " " << errorString(dcgmReturn);
        return dcgmReturn;
    }

    for (i = 0; i < entityList.size(); i++)
    {
        dcgmReturn = dcgmGroupAddEntity(dcgmHandle, *groupId, entityList[i].entityGroupId, entityList[i].entityId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            SHOW_AND_LOG_ERROR << "Error: Got error " << dcgmReturn << " " << errorString(dcgmReturn) << " while "
                               << "adding " << DcgmFieldsGetEntityGroupString(entityList[i].entityGroupId) << " "
                               << entityList[i].entityId << " to our entity group.";
            return dcgmReturn;
        }
    }

    return DCGM_ST_OK;
}

/*******************************************************************************/
bool dcgmi_entity_group_id_is_special(std::string &groupIdStr, dcgmGroupType_t *groupType, dcgmGpuGrp_t *groupId)
{
    if (groupIdStr == "all_gpus")
    {
        *groupType = DCGM_GROUP_DEFAULT;
        *groupId   = (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS;
        return true;
    }
    else if (groupIdStr == "all_nvswitches")
    {
        *groupType = DCGM_GROUP_DEFAULT_NVSWITCHES;
        *groupId   = (dcgmGpuGrp_t)DCGM_GROUP_ALL_NVSWITCHES;
        return true;
    }
    else
    {
        *groupType = DCGM_GROUP_EMPTY;
        return false;
    }
}

/*******************************************************************************/
dcgmReturn_t dcgmi_parse_field_id_list_string(std::string input, std::vector<unsigned short> &fieldIds, bool validate)
{
    std::vector<std::string> tokens;
    unsigned int i;
    unsigned short fieldId;
    dcgm_field_meta_p fieldMeta;

    /* Divide the string into a vector of substrings by comma */
    std::string delimStr = ",";
    dcgmTokenizeString(input, delimStr, tokens);

    /* Convert each token into an unsigned integer */
    for (i = 0; i < tokens.size(); i++)
    {
        fieldId = atoi(tokens[i].c_str());
        if (!fieldId && (!tokens[i].size() || tokens[i].at(0) != '0'))
        {
            SHOW_AND_LOG_ERROR << "Error: Expected numerical fieldId. Got '" << tokens[i] << "' instead.";
            return DCGM_ST_BADPARAM;
        }

        if (validate)
        {
            fieldMeta = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                SHOW_AND_LOG_ERROR << "Error: Got invalid field ID. '" << fieldId
                                   << "'. See dcgm_fields.h for a list of valid field IDs.";
                return DCGM_ST_BADPARAM;
            }
        }

        fieldIds.push_back(fieldId);
    }

    return DCGM_ST_OK;
}

/*******************************************************************************/
dcgmReturn_t dcgmi_create_field_group(dcgmHandle_t dcgmHandle,
                                      dcgmFieldGrp_t *groupId,
                                      std::vector<unsigned short> &fieldIds)
{
    dcgmReturn_t dcgmReturn;
    static int numGroupsCreated = 0;
    unsigned int myPid          = (unsigned int)getpid();
    char groupName[32]          = { 0 };

    snprintf(groupName, sizeof(groupName) - 1, "dcgmi_%u_%d", myPid, ++numGroupsCreated);

    dcgmReturn = dcgmFieldGroupCreate(dcgmHandle, fieldIds.size(), &fieldIds[0], groupName, groupId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << "Got error while creating a Field Group: " << errorString(dcgmReturn);
    }

    return dcgmReturn;
}

/*******************************************************************************/
const char *dcgmi_parse_hostname_string(const char *hostName, bool *isUnixSocketAddress, bool logOnError)
{
    if (std::string_view(hostName).starts_with(DCGM_UNIX_SOCKET_PREFIX))
    {
        *isUnixSocketAddress = true;
        /* Looks like a unix socket. Do some validation */
        if (strlen(hostName) < 8)
        {
            if (logOnError)
            {
                SHOW_AND_LOG_ERROR << "Missing hostname after \"" << DCGM_UNIX_SOCKET_PREFIX << "\".";
            }

            return nullptr;
        }
        else
            return &hostName[strlen(DCGM_UNIX_SOCKET_PREFIX)];
    }

    /* No "unix://". Treat like a regular hostname */
    *isUnixSocketAddress = false;
    return hostName;
}

/*******************************************************************************/
namespace DcgmNs
{

namespace Terminal
{
    bool IsTTY()
    {
        return !!isatty(fileno(stdout));
    }

    std::optional<TermDimensions> GetTermDimensions()
    {
        if (!IsTTY())
        {
            return std::nullopt;
        }
        winsize w = {};
        ioctl(fileno(stdout), TIOCGWINSZ, &w);
        if (w.ws_col == 0 || w.ws_row == 0)
        {
            return std::nullopt;
        }
        return TermDimensions { w.ws_row, w.ws_col };
    }
} // namespace Terminal

} // namespace DcgmNs
