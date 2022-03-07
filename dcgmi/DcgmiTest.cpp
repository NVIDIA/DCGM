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
 * DcgmiTest.cpp
 *
 */

#include "DcgmiTest.h"
#include "CommandOutputController.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgm_test_apis.h"
#include <DcgmStringHelpers.h>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>


/***************************************************************************************/

static char const TEST_DATA[] = " <DATA_NAME              > : <DATA_INFO                                   > \n";

#define DATA_NAME_TAG "<DATA_NAME"
#define DATA_INFO_TAG "<DATA_INFO"


dcgmReturn_t DcgmiTest::IntrospectCache(dcgmHandle_t mDcgmHandle,
                                        unsigned int gpuId,
                                        std::string const &fieldId,
                                        bool isGroup)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmCacheManagerFieldInfo_t fieldInfo;
    dcgmGroupInfo_t stNvcmGroupInfo;
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES];
    unsigned int numGpus = 0;

    // fetch gpus for group
    if (isGroup)
    {
        stNvcmGroupInfo.version = dcgmGroupInfo_version;
        result                  = dcgmGroupGetInfo(mDcgmHandle, (dcgmGpuGrp_t)(long long)gpuId, &stNvcmGroupInfo);
        if (DCGM_ST_OK != result)
        {
            std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
            std::cout << "Error: Unable to retrieve information about group " << gpuId << ". Return: " << error << "."
                      << std::endl;
            return result;
        }

        for (unsigned int i = 0; i < stNvcmGroupInfo.count; i++)
        {
            if (stNvcmGroupInfo.entityList[i].entityGroupId == DCGM_FE_GPU)
            {
                gpuIds[numGpus] = stNvcmGroupInfo.entityList[i].entityId;
                numGpus++;
            }
        }
    }
    else
    {
        gpuIds[0] = gpuId;
        numGpus++;
    }

    // get field info
    DcgmFieldsInit();
    memset(&fieldInfo, 0, sizeof(dcgmCacheManagerFieldInfo_t));
    fieldInfo.version = dcgmCacheManagerFieldInfo_version;
    result            = HelperParseForFieldId(fieldId, fieldInfo.fieldId, mDcgmHandle);

    if (result != DCGM_ST_OK)
    {
        std::cout << "Bad parameter passed to function. ";
        return result;
    }

    for (unsigned int i = 0; i < numGpus; i++)
    {
        fieldInfo.gpuId = (unsigned int)(uintptr_t)gpuIds[i];

        result = dcgmGetCacheManagerFieldInfo(mDcgmHandle, &fieldInfo);

        if (DCGM_ST_OK != result)
        {
            std::cout << "Error: Unable to get field info for GPU ID: " << fieldInfo.gpuId
                      << ". Return: " << errorString(result) << std::endl;
            return result;
        }

        HelperDisplayField(fieldInfo);
    }

    // std::cout << "Successfully retrieved cache field info." << std::endl;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmiTest::InjectCache(dcgmHandle_t mDcgmHandle,
                                    unsigned int gpuId,
                                    std::string const &fieldId,
                                    unsigned int pTime,
                                    std::string &injectValue)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmInjectFieldValue_t injectFieldValue {};

    DcgmFieldsInit();

    // get current time
    time_t timev;
    time(&timev);

    injectFieldValue.ts = (pTime + timev) * 1000000 + 4000000; /// adding default of 4 microseconds into the future
    HelperParseForFieldId(fieldId, injectFieldValue.fieldId, mDcgmHandle);
    HelperInitFieldValue(injectFieldValue, injectValue);

    // inject field value
    result = dcgmInjectFieldValue(mDcgmHandle, gpuId, &injectFieldValue);

    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to inject info. Return: " << errorString(result) << std::endl;
        return result;
    }

    std::cout << "Successfully injected field info." << std::endl;

    return DCGM_ST_OK;
}

void DcgmiTest::HelperDisplayField(dcgmCacheManagerFieldInfo_t &fieldInfo)
{
    CommandOutputController cmdView = CommandOutputController();
    dcgm_field_meta_p fieldMeta;

    fieldMeta = DcgmFieldGetById(fieldInfo.fieldId);

    cmdView.setDisplayStencil(TEST_DATA);

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Field ID");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldMeta == nullptr ? "Unknown" : fieldMeta->tag);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Flags");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.flags);
    cmdView.display();

    // buffer = fieldInfo.lastStatus;
    cmdView.addDisplayParameter(DATA_NAME_TAG, "Last Status");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.lastStatus);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Number of Samples");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.numSamples);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Newest Timestamp");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimestamp(fieldInfo.newestTimestamp));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Oldest Timestamp");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimestamp(fieldInfo.oldestTimestamp));
    cmdView.display();

    // buffer = fieldInfo.monitorFrequencyUsec / 1000000;
    cmdView.addDisplayParameter(DATA_NAME_TAG, "Monitor Frequency");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.monitorFrequencyUsec / 1000000);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Max Age (sec)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.maxAgeUsec / 1000000);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Fetch Count");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.fetchCount);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Total Fetch Usec");
    cmdView.addDisplayParameter(DATA_INFO_TAG, fieldInfo.execTimeUsec);
    cmdView.display();

    double usecPerFetch = 0.0;
    if (fieldInfo.fetchCount != 0)
    {
        usecPerFetch = (double)fieldInfo.execTimeUsec / (double)fieldInfo.fetchCount;
    }
    cmdView.addDisplayParameter(DATA_NAME_TAG, "Usec Per Fetch");
    cmdView.addDisplayParameter(DATA_INFO_TAG, usecPerFetch);
    cmdView.display();

    std::cout << std::endl;
}

dcgmReturn_t DcgmiTest::HelperInitFieldValue(dcgmInjectFieldValue_t &injectFieldValue, std::string &injectValue)
{
    dcgm_field_meta_p fieldMeta;

    // get meta data
    fieldMeta = DcgmFieldGetById(injectFieldValue.fieldId);

    injectFieldValue.version   = dcgmInjectFieldValue_version;
    injectFieldValue.fieldType = fieldMeta == nullptr ? DCGM_FI_UNKNOWN : fieldMeta->fieldType;
    injectFieldValue.status    = DCGM_ST_OK;

    // wrap in try catch
    switch (injectFieldValue.fieldType)
    {
        case DCGM_FT_TIMESTAMP:
        case DCGM_FT_INT64:
            injectFieldValue.value.i64 = atol(injectValue.c_str());
            break;
        case DCGM_FT_STRING:
            dcgmStrncpy(injectFieldValue.value.str, injectValue.c_str(), sizeof(injectFieldValue.value.str));
            break;
        case DCGM_FT_DOUBLE:
            injectFieldValue.value.dbl = atof(injectValue.c_str());
            break;
        case DCGM_FT_BINARY:
            // Not Supported
            // injectFieldValue.value.blob
            injectFieldValue.value.i64 = 0;
            break;
        default:
            break;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmiTest::HelperParseForFieldId(std::string const &str, unsigned short &fieldId, dcgmHandle_t dcgmHandle)
{
    // find , or :

    dcgmFieldGrp_t fieldGroupId;
    dcgmReturn_t dcgmReturn;
    int index;
    dcgmFieldGroupInfo_t fieldGroupInfo;

    if (str.find(',') != std::string::npos)
    {
        index = str.find(',');
    }
    else
    {
        fieldId = atoi(str.c_str());
        return DCGM_ST_OK;
    }

    fieldGroupId = (dcgmFieldGrp_t)(intptr_t)atoi(str.substr(0, 1).c_str());

    index = atoi(str.substr(index + 1).c_str());

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version      = dcgmFieldGroupInfo_version;
    fieldGroupInfo.fieldGroupId = fieldGroupId;

    dcgmReturn = dcgmFieldGroupGetInfo(dcgmHandle, &fieldGroupInfo);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %llu",
                    "dcgmFieldGroupGetInfo returned %d for fieldGrpId %llu",
                    (int)dcgmReturn,
                    (unsigned long long)fieldGroupId);
        return dcgmReturn;
    }

    if (index >= (int)fieldGroupInfo.numFieldIds)
    {
        return DCGM_ST_BADPARAM;
    }

    fieldId = fieldGroupInfo.fieldIds[index];
    return DCGM_ST_OK;
}

std::string DcgmiTest::HelperFormatTimestamp(long long timestamp)
{
    std::stringstream ss;
    long long temp  = timestamp / 1000000;
    std::string str = ctime((long *)&temp);

    // Remove returned next line character
    str = str.substr(0, str.length() - 1);

    ss << str; //<< ":" << std::setw(4) << std::setfill('0') <<timestamp % 1000000;

    return ss.str();
}

dcgmReturn_t DcgmiTest::DoExecuteConnected()
{
    return DCGM_ST_OK;
}

/*****************************************************************************
 *****************************************************************************
 *Cache Introspect Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
IntrospectCache::IntrospectCache(std::string hostname, unsigned int groupId, std::string fieldId, bool isGroup)
    : Command()
    , mGpuId(groupId)
    , mFieldId(std::move(fieldId))
    , mIDisGroup(isGroup)
{
    m_hostName = std::move(hostname);
}


/*****************************************************************************/
dcgmReturn_t IntrospectCache::DoExecuteConnected()
{
    return adminObj.IntrospectCache(m_dcgmHandle, mGpuId, mFieldId, mIDisGroup);
}

/*****************************************************************************
 *****************************************************************************
 *Cache Inject Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
InjectCache::InjectCache(std::string hostname,
                         unsigned int gpuId,
                         std::string fieldId,
                         unsigned int pTime,
                         std::string injectValue)
    : Command()
    , mGId(gpuId)
    , mFieldId(std::move(fieldId))
    , mTime(pTime)
    , mInjectValue(std::move(injectValue))
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t InjectCache::DoExecuteConnected()
{
    return adminObj.InjectCache(m_dcgmHandle, mGId, mFieldId, mTime, mInjectValue);
}
