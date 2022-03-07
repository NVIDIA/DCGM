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
/******************************************************************************
 *                                Device Monitor                              *
 *                                                                            *
 *                                                                            *
 * Purpose: Device Monitor class is the heart of dmon. The Dmon is a tool that*
 * creates a group of GPUs and runs the watch over them based on the fields   *
 * mentioned in the command line options. Dmon provides flexibility to use    *
 * many options like mentioning field IDs, Gpu IDs, FieldGroup Ids, Gpu group *
 * Ids, delay and count. A small description of the command line options is as*
 * below:                                                                     *
 *                                                                            *
 * Option    Long Form          Default Value                                 *
 *                                                                            *
 * -i         --gpuIds          ALL IDs (GPU/GPU-I/GPU-CI)                    *
 * -g         --group-id        All GPUs group                                *
 * -f         --field-group-id  Null                                          *
 * -e         --field-ids      Null                                           *
 * -h         --help                                                          *
 * -d         --delay          1sec                                           *
 * -c         --count          0(infinite)                                    *
 * -l         --list                                                          *
 *                                                                            *
 * When field Ids are mentioned, the dmon creates a group of FieldIds and     *
 * similarly when the Gpu Ids are mentioned, the dmon creates a group of Gpus *
 * and then runs the dmon against the Gpu group for the field Ids mentioned.  *
 *                                                                            *
 ******************************************************************************/

#include "DeviceMonitor.h"
#include "dcgm_structs_internal.h"
#include "dcgmi_common.h"

#include <DcgmLogging.h>

#include <iomanip>


#define MAX_PRINT_HEADER_COUNT 14
#define MILLI_SEC              1000
#define FLOAT_VAL_PREC         3
#define MAX_KEEP_SAMPLES       2
#define MAX_KEEP_AGE           0.0 /* Enforce quota via MAX_KEEP_SAMPLES */
#define DEFAULT_COUNT          0
#define WIDTH_1                1
#define PADDING                2
#define WIDTH_3                3
#define WIDTH_10               10
#define WIDTH_15               15
#define WIDTH_20               20
#define WIDTH_30               30
#define WIDTH_40               40
#define NA                     "N/A"
#define FORMAT_LINE            "___________________________________________________________________________________ "

/*****************************************************************************/
/* This is used by the signal handler to let the device monitor know that
   we've received a signal from a ctrl-c..etc that we should stop */
static std::atomic<bool> deviceMonitorShouldStop = false;

static std::string_view HelperGetGroupIdPrefix(dcgm_field_entity_group_t const groupId)
{
    switch (groupId)
    {
        case DCGM_FE_GPU:
            return "GPU";
        case DCGM_FE_VGPU:
            return "vGPU";
        case DCGM_FE_SWITCH:
            return "Switch";
        case DCGM_FE_GPU_I:
            return "GPU-I";
        case DCGM_FE_GPU_CI:
            return "GPU-CI";
        case DCGM_FE_NONE:
        case DCGM_FE_COUNT:
            break;
    }
    return "";
}

/**
 * Constructor : Initialize the members of class Device Info
 */
DeviceInfo::DeviceInfo(std::string hostname,
                       std::string requestedEntityIds,
                       std::string grpIdStr,
                       std::string fldIds,
                       int fieldGroupId,
                       int updateDelay,
                       int numOfIterations,
                       bool listOptionMentioned)
    : m_requestedEntityIds(std::move(requestedEntityIds))
    , m_groupIdStr(std::move(grpIdStr))
    , m_fieldIdsStr(std::move(fldIds))
    , m_fieldGrpId_int(fieldGroupId)
    , m_delay(updateDelay)
    , m_count(numOfIterations)
    , m_myGroupId(0)
    , m_fieldGroupId(0)
    , m_list(listOptionMentioned)
    , m_widthArray()
{
    deviceMonitorShouldStop = false;

    m_hostName = std::move(hostname);
}

/**
 * This function populates the Field Details structure
 * which is used in listing the details of Long name,
 * short name and the field ids.
 */
void DeviceInfo::PopulateFieldDetails()
{
    for (unsigned short fieldId = 0; fieldId < DCGM_FI_MAX_FIELDS; fieldId++)
    {
        auto const *dcgmFormat_s = DcgmFieldGetById(fieldId);
        if (dcgmFormat_s == nullptr)
        {
            continue;
        }
        m_fieldDetails.emplace_back(dcgmFormat_s->tag, dcgmFormat_s->valueFormat->shortName, fieldId);
    }
}

/**
 * This function formats and displays the field details for -l option.
 */
void DeviceInfo::DisplayFieldDetails()
{
    using std::left;
    using std::right;
    using std::setw;

    std::cout << FORMAT_LINE << "\n";
    std::cout << setw(WIDTH_40) << "Long Name" << setw(WIDTH_20) << "Short Name" << setw(WIDTH_15) << "Field Id"
              << "\n";
    std::cout << FORMAT_LINE << "\n";
    for (auto const &fieldDetail : m_fieldDetails)
    {
        std::cout << setw(WIDTH_40) << left << fieldDetail.m_longName << setw(WIDTH_20) << right
                  << fieldDetail.m_shortName << setw(WIDTH_15) << right << fieldDetail.m_fieldId << "\n";
    }
    std::cout.flush();
}

dcgmReturn_t DeviceInfo::DoExecuteConnected()
{
    if (m_list)
    {
        PopulateFieldDetails();
        DisplayFieldDetails();
        return DCGM_ST_OK;
    }

    return CreateFieldGroupAndGpuGroup();
}

/**
 * killHandler      Handler to handle SIGINT -> Ctrl+c
 * @param sig       Integer type of the signal fired.
 */
void killHandler(int sig)
{
    signal(sig, SIG_IGN);
    deviceMonitorShouldStop = true;
}

/**
 * In this function we simply print out the information in the callback of
 * the  watched field. This ia called as a callback to dcgmGetLatestValues_v2
 * function.
 *
 * @param entityGroupId entityGroup of the entity this field value set belongs to
 * @param entityId      Entity this field value set belongs to
 * @param values        The values under watch.
 * @param numValues     Total number of values under watch.
 * @param userdata      The generic data which needs to be passed to function. The void
 *                      pointer at the end allows a pointer to be passed to this function. Here we know
 *                      that we are passing in a map, so we can cast it as such.
 *                      This pointer is also useful if we need a reference to something else inside your function.
 * @return              0 on success; 1 on failure
 */
static int ListFieldValues(dcgm_field_entity_group_t entityGroupId,
                           dcgm_field_eid_t entityId,
                           dcgmFieldValue_v1 *values,
                           int numValues,
                           void *userdata)
{
    std::string st;
    std::stringstream strs;
    dcgmi_entity_pair_t entityKey;

    // Note: this is a pointer to a map and we cast it to map below.
    std::map<dcgmi_entity_pair_t, std::vector<std::string>> &entityStats
        = *(static_cast<std::map<dcgmi_entity_pair_t, std::vector<std::string>> *>(userdata));

    entityKey.entityGroupId = entityGroupId;
    entityKey.entityId      = entityId;

    for (int i = 0; i < numValues; i++)
    {
        strs.clear(); // clear any bits set
        strs.str(std::string());
        strs << std::fixed;
        strs << std::setprecision(
            FLOAT_VAL_PREC); // fixing the number of values after decimal for consistency in display.

        // Including a switch statement here for handling different
        // types of values (except binary blobs).
        dcgm_field_meta_p field = DcgmFieldGetById(values[i].fieldId);
        if (field == nullptr)
        {
            return 1;
        }

        switch (field->fieldType)
        {
            case DCGM_FT_BINARY:
                strs << NA;
                entityStats[entityKey].push_back(strs.str());
                break;
            case DCGM_FT_DOUBLE:
                if (DCGM_FP64_IS_BLANK(values[i].value.dbl))
                {
                    entityStats[entityKey].push_back(NA);
                }
                else
                {
                    strs << values[i].value.dbl;
                    st = strs.str();
                    entityStats[entityKey].push_back(st);
                }
                break;
            case DCGM_FT_INT64:
                if (DCGM_INT64_IS_BLANK(values[i].value.i64))
                {
                    entityStats[entityKey].push_back(NA);
                }
                else
                {
                    strs << values[i].value.i64;
                    st = strs.str();
                    entityStats[entityKey].push_back(st);
                }
                break;
            case DCGM_FT_STRING:
                entityStats[entityKey].push_back(values[i].value.str);
                break;
            case DCGM_FT_TIMESTAMP:
                if (DCGM_INT64_IS_BLANK(values[i].value.i64))
                {
                    entityStats[entityKey].push_back(NA);
                }
                else
                {
                    strs << values[i].value.i64;
                    entityStats[entityKey].push_back(strs.str());
                }
                break;
            default:
                std::cout << "Error in field types. " << values[i].fieldType << " Exiting." << std::endl;
                // Error, return > 0 error code.
                return 1;
        }
    }
    // Program executed correctly. Return 0 to notify DCGM (callee) that it
    // was successful.
    return 0;
}

/**
 * lockWatchAndUpdate : The function sets the watches on the field group
 *                      for the mentioned Gpu group.
 * @return dcgmReturn_t : result - success or error
 */

dcgmReturn_t DeviceInfo::LockWatchAndUpdate()
{
    int running          = 0;
    int decrement        = 0;
    int printHeaderCount = 0;

    /* Install a signal handler to catch ctrl-c */
    signal(SIGINT, &killHandler);

    // set the watch on field group id for Group of gpus.
    dcgmReturn_t result = dcgmWatchFields(
        m_dcgmHandle, m_myGroupId, m_fieldGroupId, m_delay * MILLI_SEC, MAX_KEEP_AGE, MAX_KEEP_SAMPLES);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Error! Error setting watches. Return: " << result;
        const char *e = errorString(result);

        switch (result)
        {
            case DCGM_ST_REQUIRES_ROOT:
                e = "Unable to watch one or more of the requested fields because doing so requires the host engine to be running as root.";
                break;

            case DCGM_ST_INSUFFICIENT_DRIVER_VERSION:
                e = "Unable to watch one or more of the requested fields because doing so requires a newer driver version than what is currently installed.";
                break;

            default:
                break; /* Intentional fall-through. e is initialized before this switch statement */
        }

        std::cout << "Error setting watches. Result: " << e << std::endl;
        return result;
    }

    dcgmUpdateAllFields(m_dcgmHandle, 1);

    // Set the loop to running forever and set decrement check to false as there will
    // no decrement in m_count when no m_count value is mentioned. Loop runs forever in default
    // condition.
    if (m_count == DEFAULT_COUNT)
    {
        running   = 1;
        decrement = false;
    }
    // The running variable is initialized with mentioned value in command and decrement flag is set to
    // true as the running variable is decremented with each iteration.
    else
    {
        running   = m_count;
        decrement = true;
    }

    while (running && !deviceMonitorShouldStop)
    {
        m_entityStats.clear();
        result = dcgmGetLatestValues_v2(m_dcgmHandle, m_myGroupId, m_fieldGroupId, &ListFieldValues, &m_entityStats);
        // Check the result to see if our DCGM operation was successful.
        if (result != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "Error! Error getting values information. Return: %d", result);
            std::cout << "Error getting values information."
                         "Return: "
                      << errorString(result) << std::endl;
            return result;
        }

        if (m_entityStats.empty())
        {
            std::cout << "Error: The field group or entity group is empty." << std::endl;
            return DCGM_ST_NO_DATA; /* Propagate this to any callers */
        }

        // print the map that we populated in callback of dcgmGetLatestValues
        for (auto const &[entityPair, statsVec] : m_entityStats)
        {
            auto const groupPrefix = HelperGetGroupIdPrefix(entityPair.entityGroupId);

            std::cout << std::setw(m_widthArray[0]);
            std::cout << groupPrefix << " " << entityPair.entityId;

            for (size_t i = 0; i < statsVec.size(); i++)
            {
                auto const width = m_widthArray[i + 1];
                /* Check if we are overflowing the width we are assigned */
                if (statsVec[i].length() >= width)
                {
                    std::cout << std::setw(0);
                    std::cout << ' ';
                }
                std::cout << std::setw(width);
                std::cout << statsVec[i];
            }
            std::cout << "\n";
        }

        std::cout.flush();

        // We decrement only when count value is not default and the loop is not supposed to run forever.
        if (decrement)
        {
            running--;
            if (running <= 0)
            {
                break;
            }
        }

        usleep(m_delay * MILLI_SEC);
        if (printHeaderCount == MAX_PRINT_HEADER_COUNT)
        {
            std::cout << "" << std::endl;
            PrintHeaderForOutput();
            printHeaderCount = 0;
        }
        printHeaderCount++;
    }

    if (deviceMonitorShouldStop)
    {
        std::cout << std::endl << "dmon was stopped due to receiving a signal." << std::endl;
    }

    return result;
}

/**
 * printHeaderForOutput - Directs the header to Console output stream.
 */
void DeviceInfo::PrintHeaderForOutput() const
{
    std::cout << m_header << "\n";
    std::cout << m_headerUnit << "\n";
    std::cout.flush();
}

/**
 * The functions initializes the header for tabular output with alignment by
 * setting width and using the field Ids to create header.
 *
 * @param fieldIds      The vector containing field ids.
 * @param numFields     Number of fields int the vector.
 */
void DeviceInfo::SetHeaderForOutput(unsigned short fieldIds[], unsigned int const numFields)
{
    int const entityWidth = 7;
    std::stringstream ss;
    std::stringstream ss_unit;
    ss << "#";
    ss << std::setw(entityWidth);
    ss << "Entity";

    ss_unit << std::setw(entityWidth + 1);
    ss_unit << "Id";
    m_widthArray[0] = entityWidth + 1;
    for (unsigned int i = 0; i < numFields; i++)
    {
        dcgm_field_meta_p meta_p   = DcgmFieldGetById(fieldIds[i]);
        unsigned short const width = meta_p == nullptr ? 0 : meta_p->valueFormat->width;
        m_widthArray[i + 1]        = (width + PADDING);
        ss << std::setw(width + (PADDING));
        ss << ((meta_p == nullptr) ? "" : meta_p->valueFormat->shortName);

        ss_unit << std::setw(width + PADDING);
        ss_unit << ((meta_p == nullptr) ? "" : meta_p->valueFormat->unit);
    }
    m_header     = ss.str();
    m_headerUnit = ss_unit.str();
}

dcgmReturn_t DeviceInfo::CreateEntityGroupFromEntityList(void)
{
    auto [entityList, rejectedIds] = DcgmNs::TryParseEntityList(m_dcgmHandle, m_requestedEntityIds);

    // Fallback to old method

    std::vector<dcgmGroupEntityPair_t> oldEntityList;

    /* Convert the string to a list of entities */
    auto dcgmReturn = dcgmi_parse_entity_list_string(rejectedIds, oldEntityList);
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    std::move(begin(oldEntityList), end(oldEntityList), std::back_inserter(entityList));

    /* Create a group based on this list of entities */
    dcgmReturn = dcgmi_create_entity_group(m_dcgmHandle, DCGM_GROUP_EMPTY, &m_myGroupId, entityList);
    return dcgmReturn;
}

dcgmReturn_t DeviceInfo::ValidateOrCreateEntityGroup(void)
{
    dcgmReturn_t dcgmReturn;
    dcgmGroupType_t groupType = DCGM_GROUP_EMPTY;
    /**
     * Check if m_requestedEntityIds is set or not. If set, we create
     * a group including the devices mentioned with flag.
     */
    if (m_requestedEntityIds != "-1")
    {
        dcgmReturn = CreateEntityGroupFromEntityList();
        return dcgmReturn;
    }

    /* If no group ID or entity list was specified, assume all GPUs to act like nvidia-smi dmon */
    if (m_groupIdStr == "-1")
    {
        std::vector<dcgmGroupEntityPair_t> entityList; /* Empty List */
        dcgmReturn = dcgmi_create_entity_group(m_dcgmHandle, DCGM_GROUP_DEFAULT, &m_myGroupId, entityList);
        return dcgmReturn;
    }

    bool const groupIdIsSpecial = dcgmi_entity_group_id_is_special(m_groupIdStr, &groupType, &m_myGroupId);
    if (groupIdIsSpecial)
    {
        /* m_myGroupId was already set to the correct group ID of the special group */
        return DCGM_ST_OK;
    }

    int groupIdAsInt = 0;
    try
    {
        groupIdAsInt = std::stoi(m_groupIdStr);
    }
    catch (std::exception const & /*ex*/)
    {
        std::cout << "Error: Expected a numerical groupId. Instead got '" << m_groupIdStr << "'" << std::endl;
        return DCGM_ST_BADPARAM;
    }

    m_myGroupId = static_cast<dcgmGpuGrp_t>(groupIdAsInt);

    /* Try to get a handle to the group the user specified */
    dcgmGroupInfo_t dcgmGroupInfo {};
    dcgmGroupInfo.version = dcgmGroupInfo_version;

    dcgmReturn = dcgmGroupGetInfo(m_dcgmHandle, m_myGroupId, &dcgmGroupInfo);
    if (DCGM_ST_OK != dcgmReturn)
    {
        std::stringstream ss;
        ss << "Error: Unable to retrieve information about group " << groupIdAsInt << ". Return: " << dcgmReturn << " "
           << (dcgmReturn == DCGM_ST_NOT_CONFIGURED ? "The Group is not found" : errorString(dcgmReturn));

        auto const errorString = ss.str();
        std::cout << errorString << std::endl;
        DCGM_LOG_ERROR << errorString;

        return dcgmReturn;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DeviceInfo::ValidateOrCreateFieldGroup(void)
{
    dcgmReturn_t dcgmReturn;

    dcgmFieldGroupInfo_t fieldGroupInfo;

    if (m_fieldIdsStr != "-1")
    {
        dcgmReturn = dcgmi_parse_field_id_list_string(m_fieldIdsStr, m_fieldIds, true);
        if (dcgmReturn != DCGM_ST_OK)
            return dcgmReturn;

        dcgmReturn = dcgmi_create_field_group(m_dcgmHandle, &m_fieldGroupId, m_fieldIds);
        if (dcgmReturn != DCGM_ST_OK)
            return dcgmReturn;

        SetHeaderForOutput(&m_fieldIds[0], m_fieldIds.size());
        return DCGM_ST_OK;
    }

    if (m_fieldGrpId_int == -1)
    {
        std::cout << "Error: Either field group ID or a list of field IDs must be provided. "
                  << "See dcgmi dmon --help for usage." << std::endl;
    }

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version = dcgmFieldGroupInfo_version;

    fieldGroupInfo.fieldGroupId = static_cast<dcgmFieldGrp_t>(m_fieldGrpId_int);

    dcgmReturn = dcgmFieldGroupGetInfo(m_dcgmHandle, &fieldGroupInfo);
    if (DCGM_ST_OK != dcgmReturn)
    {
        std::stringstream ss;
        ss << "Error: Unable to retrieve information about field group " << m_fieldGrpId_int
           << ". Return: " << dcgmReturn << " "
           << (dcgmReturn == DCGM_ST_NOT_CONFIGURED ? "The Field Group is not found" : errorString(dcgmReturn));

        auto const errorString = ss.str();
        std::cout << errorString << std::endl;
        DCGM_LOG_ERROR << errorString;

        return DCGM_ST_BADPARAM;
    }

    m_fieldIds.clear();
    m_fieldIds.reserve(fieldGroupInfo.numFieldIds);
    for (unsigned int i = 0; i < fieldGroupInfo.numFieldIds; i++)
    {
        m_fieldIds.push_back(fieldGroupInfo.fieldIds[i]);
    }

    SetHeaderForOutput(fieldGroupInfo.fieldIds, fieldGroupInfo.numFieldIds);
    m_fieldGroupId = static_cast<dcgmFieldGrp_t>(m_fieldGrpId_int);
    return DCGM_ST_OK;
}


dcgmReturn_t DeviceInfo::CreateFieldGroupAndGpuGroup(void)
{
    dcgmReturn_t dcgmReturn = ValidateOrCreateEntityGroup();
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    /**
     * The Gpu group has been created and Group ID has been set now. We create a
     * field group either along with fieldIDs mentioned or the fields mentioned.
     */
    dcgmReturn = ValidateOrCreateFieldGroup();
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    PrintHeaderForOutput();

    dcgmReturn = LockWatchAndUpdate();
    return dcgmReturn;
}
