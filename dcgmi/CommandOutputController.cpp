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
 * CommandOutputController.cpp
 *
 *  Created on: Oct 23, 2015
 *      Author: chris
 */

#include "CommandOutputController.h"
#include "dcgm_agent.h"
#include <DcgmStringHelpers.h>
#include <algorithm>
#include <cstdarg>
#include <iostream>
#include <string>


/***************************************************************/
void CommandOutputController::clearDisplayParameters()
{
    displayParams.clear();
}

/***************************************************************/
void CommandOutputController::setDisplayStencil(char const stencil[])
{
    displayStencil = stencil;
}

/***************************************************************/
void CommandOutputController::display()
{
    std::string display_tmp = displayStencil;

    // Fill in tags
    for (auto &displayParam : displayParams)
    {
        ReplaceTag(display_tmp, displayParam.tag, "%s", displayParam.val.c_str());
    }

    std::cout << display_tmp;

    clearDisplayParameters();
}

/*****************************************************************************/
int CommandOutputController::OnlySpacesBetween(char *start, char *end)
{
    while (start < end)
    {
        if (' ' != *start)
            return 0;
        start++;
    }
    return 1;
}

/*****************************************************************************/
void CommandOutputController::ReplaceTag(std::string &buff, std::string_view tag, char const *fmt, ...)
{
    auto const it = std::search(begin(buff), end(buff), std::boyer_moore_horspool_searcher(begin(tag), end(tag)));
    if (it == end(buff))
    {
        std::cout << "Debug Error: Parser is unable to find a tag start. Tag: " << tag << ". Is stencil set?"
                  << std::endl;
        return;
    }
    auto const tagBeginPos = std::distance(buff.begin(), it);
    auto tagEndPos         = buff.find_first_of('>', tagBeginPos + tag.length());
    if (tagEndPos == std::string::npos)
    {
        std::cout << "Debug Error: Parser unable to find tag end. (2)" << std::endl;
        return;
    }
    ++tagEndPos;

    std::string val(255, ' ');
    static std::string const filler(255, ' ');

    va_list args;
    va_start(args, fmt);
    int valLen = vsnprintf(val.data(), val.size() + 1, fmt, args);
    va_end(args);

    if (valLen < 0)
    {
        std::cout << "Debug Error: Unable to format tag replacement value. Tag: " << tag << std::endl;
        return;
    }

    if ((size_t)valLen < val.size())
    {
        val.resize(valLen);
    }

    if (val.size() < tagEndPos - tagBeginPos)
    {
        val.append(filler, 0, tagEndPos - tagBeginPos - val.size());
    }
    else if (val.length() > tagEndPos - tagBeginPos)
    {
        val.resize(tagEndPos - tagBeginPos);
        val.replace(val.size() - 3, 3, "...");
    }

    buff.replace(tagBeginPos, tagEndPos - tagBeginPos, val);
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val))
    {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss << "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss << "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss << "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss << "Insf. Permission";
                break;

            default:
                ss << "Unknown";
                break;
        }
    }
    else
    {
        ss << val;
    }

    return ss.str();
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(unsigned int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val))
    {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss << "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss << "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss << "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss << "Insf. Permission";
                break;

            default:
                ss << "Unknown";
                break;
        }
    }
    else
    {
        ss << val;
    }

    return ss.str();
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(long long val)
{
    std::stringstream ss;

    if (DCGM_INT64_IS_BLANK(val))
    {
        switch (val)
        {
            case DCGM_INT64_BLANK:
                ss << "Not Specified";
                break;

            case DCGM_INT64_NOT_FOUND:
                ss << "Not Found";
                break;

            case DCGM_INT64_NOT_SUPPORTED:
                ss << "Not Supported";
                break;

            case DCGM_INT64_NOT_PERMISSIONED:
                ss << "Insf. Permission";
                break;

            default:
                ss << "Unknown";
                break;
        }
    }
    else
    {
        ss << val;
    }

    return ss.str();
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(double val)
{
    std::stringstream ss;

    if (DCGM_FP64_IS_BLANK(val))
    {
        if (val == DCGM_FP64_BLANK)
            ss << "Not Specified";
        else if (val == DCGM_FP64_NOT_FOUND)
            ss << "Not Found";
        else if (val == DCGM_FP64_NOT_SUPPORTED)
            ss << "Not Supported";
        else if (val == DCGM_FP64_NOT_PERMISSIONED)
            ss << "Insf. Permission";
        else
            ss << "Unknown";
    }
    else
    {
        ss << val;
    }

    return ss.str();
}

void CommandOutputController::RemoveTabsAndNewlines(std::string &str)
{
    std::replace(str.begin(), str.end(), '\t', ' ');
    std::replace(str.begin(), str.end(), '\n', ' ');
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(std::string val)
{
    std::string str;

    if (DCGM_STR_IS_BLANK(val.c_str()))
    {
        if (!val.compare(DCGM_STR_BLANK))
        {
            str = "Not Specified";
        }
        else if (!val.compare(DCGM_STR_NOT_FOUND))
        {
            str = "Not Found";
        }
        else if (!val.compare(DCGM_STR_NOT_SUPPORTED))
        {
            str = "Not Supported";
        }
        else if (!val.compare(DCGM_STR_NOT_PERMISSIONED))
        {
            str = "Insf. Permission";
        }
        else
        {
            str = "Unknown";
        }
    }
    else
    {
        str = val;
        RemoveTabsAndNewlines(str);
    }

    return str;
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(char const *val)
{
    std::string str;

    if (DCGM_STR_IS_BLANK(val))
    {
        if (!strcmp(val, DCGM_STR_BLANK))
        {
            str = "Not Specified";
        }
        else if (!strcmp(val, DCGM_STR_NOT_FOUND))
        {
            str = "Not Found";
        }
        else if (!strcmp(val, DCGM_STR_NOT_SUPPORTED))
        {
            str = "Not Supported";
        }
        else if (!strcmp(val, DCGM_STR_NOT_PERMISSIONED))
        {
            str = "Insf. Permission";
        }
        else
        {
            str = "Unknown";
        }
    }
    else
    {
        str = val;
        RemoveTabsAndNewlines(str);
    }

    return str;
}

/***************************************************************************************
 **
 **    GPU Error     *******************************************************************/

static char const ERROR_HEADER[] = "+---------+------------------------------------------------------------------+\n"
                                   "| GPU ID  | Error Message                                                    |\n"
                                   "+=========+==================================================================+\n";

static char const ERROR_DISPLAY[] = "| <GPUID >| <ERROR_MESSAGE                                                  >|\n";

static char const ERROR_FOOTER[] = "+---------+------------------------------------------------------------------+\n";

#define GPU_ID_TAG        "<GPUID"
#define ERROR_MESSAGE_TAG "<ERROR_MESSAGE"


/************************************************************************************/
GPUErrorOutputController::GPUErrorOutputController()
{}
GPUErrorOutputController::~GPUErrorOutputController()
{}

/************************************************************************************/
void GPUErrorOutputController::display()
{
    std::stringstream ss;
    dcgmErrorInfo_t currentError;
    dcgmReturn_t result;
    dcgm_field_meta_p errorID;
    bool hadData      = false;
    bool isOverridden = false;

    DcgmFieldsInit();

    /* Look at status to get individual errors */
    result = dcgmStatusPopError(mErrHandle, &currentError);

    if (result != DCGM_ST_NO_DATA)
    {
        std::cout << ERROR_HEADER;
        hadData = true;
    }

    this->setDisplayStencil(ERROR_DISPLAY);

    while (result != DCGM_ST_NO_DATA)
    {
        // Fill in tags
        if (currentError.gpuId > 512)
        {
            this->addDisplayParameter(GPU_ID_TAG, "N/A");
        }
        else
        {
            this->addDisplayParameter(GPU_ID_TAG, currentError.gpuId);
        }

        // Create error message
        errorID = DcgmFieldGetById(currentError.fieldId);
        ss.str("");

        // Check if error message has been overridden.
        isOverridden = false;
        for (unsigned int i = 0; i < mStringOverriders.size(); i++)
        {
            if ((mStringOverriders[i].fieldId == currentError.fieldId)
                && (mStringOverriders[i].errorCode == currentError.status))
            {
                ss << mStringOverriders[i].overrideString;
                isOverridden = true;
                break;
            }
        }

        if (!isOverridden)
        {
            if (errorID)
                ss << errorID->tag << " - " << errorString((dcgmReturn_t)currentError.status);
            else
                ss << "No Field ID - " << errorString((dcgmReturn_t)currentError.status);
        }

        // Display Error
        this->addDisplayParameter(ERROR_MESSAGE_TAG, ss.str());
        CommandOutputController::display();

        // Get next error
        result = dcgmStatusPopError(mErrHandle, &currentError);
    }

    if (hadData)
    {
        std::cout << ERROR_FOOTER;
    }
}

/************************************************************************************/
void GPUErrorOutputController::addError(dcgmStatus_t errHandle)
{
    mErrHandle = errHandle;
}

/************************************************************************************/
void GPUErrorOutputController::addErrorStringOverride(short fieldId, dcgmReturn_t errorCode, std::string replacement)
{
    dcgmErrorStringOverride_t temp = { replacement, fieldId, errorCode };
    mStringOverriders.push_back(temp);
}
