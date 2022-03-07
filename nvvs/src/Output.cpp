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
#include "Output.h"
#include "NvidiaValidationSuite.h"
#include "NvvsCommon.h"
#include <sstream>
#include <stdexcept>

#define MAX_LINE_LENGTH 50

/*****************************************************************************/
void Output::header(const std::string &headerString)
{
    // This output is only for regular stdout, not training or 'parse' mode (which is deprecated)
    if (!nvvsCommon.parse && !nvvsCommon.training)
        m_out << "\t" << headerString << std::endl;
}

/*****************************************************************************/
std::string Output::fill(fillType_enum type)
{
    std::string ret;
    if (nvvsCommon.parse)
    {
        switch (type)
        {
            case NVVS_FILL_PREFACE:
            case NVVS_FILL_DOT:
            case NVVS_FILL_DELIMITER2:
            default:
                ret = "";
                break;
            case NVVS_FILL_DELIMITER1:
                ret = ":";
                break;
        }
    }
    else
    {
        switch (type)
        {
            case NVVS_FILL_PREFACE:
                ret = "\t\t";
                break;
            case NVVS_FILL_DOT:
                ret = ".";
                break;
            case NVVS_FILL_DELIMITER1:
            case NVVS_FILL_DELIMITER2:
            default:
                ret = " ";
                break;
        }
    }
    return ret;
}

/*****************************************************************************/
void Output::prep(const std::string &testString)
{
    if (nvvsCommon.training)
        return; // Training mode doesn't want this output

    std::stringstream ss;

    ss << fill(NVVS_FILL_PREFACE) << testString << fill(NVVS_FILL_DELIMITER1);
    for (unsigned int i = 0; i < (unsigned int)(MAX_LINE_LENGTH - testString.length()); i++)
    {
        ss << fill(NVVS_FILL_DOT);
    }
    m_out << ss.str() << fill(NVVS_FILL_DELIMITER2);
    m_out.flush();
}

/*****************************************************************************/
void Output::WriteInfo(const std::vector<dcgmDiagEvent_t> &info)
{
    for (unsigned int i = 0; i < info.size(); i++)
    {
        m_out << "\t\t    info (GPU " << info[i].gpuId << "): " << info[i].msg << std::endl;
    }
}

/*****************************************************************************/
void Output::Result(nvvsPluginResult_t overallResult,
                    const std::vector<dcgmDiagSimpleResult_t> &perGpuResults,
                    const std::vector<dcgmDiagEvent_t> &errors,
                    const std::vector<dcgmDiagEvent_t> &info)
{
    if (nvvsCommon.training)
    {
        return; // Training mode doesn't want this output
    }

    std::string resultString = resultEnumToString(overallResult);

    m_out << resultString << std::endl;

    if (overallResult == NVVS_RESULT_PASS && nvvsCommon.verbose)
    {
        // Verbose mode should output info
        WriteInfo(info);
    }
    else
    {
        for (unsigned int i = 0; i < errors.size(); i++)
        {
            m_out << "\t\t   *** (GPU " << errors[i].gpuId << ") " << errors[i].msg << std::endl;
        }
        WriteInfo(info);
    }
}

/*****************************************************************************/
std::string Output::resultEnumToString(nvvsPluginResult_t resultEnum)
{
    std::string result;
    switch (resultEnum)
    {
        case NVVS_RESULT_PASS:
            result = "PASS";
            break;
        case NVVS_RESULT_WARN:
            result = "WARN";
            break;
        case NVVS_RESULT_SKIP:
            result = "SKIP";
            break;
        case NVVS_RESULT_FAIL:
        default:
            result = "FAIL";
            break;
    }
    return result;
}

void Output::updatePluginProgress(unsigned int progress, bool clear)
{
    static bool display                   = false;
    static unsigned int previousStrLength = 0;

    // This output is only for regular stdout, not training or 'parse' mode (which is deprecated)
    if (!nvvsCommon.parse && !nvvsCommon.training)
    {
        std::stringstream ss;
        ss << progress;
        if (display || clear)
        {
            for (unsigned int j = 0; j < previousStrLength; j++)
                m_out << "\b";
        }

        if (!clear)
        {
            m_out << ss.str() << "%";
            m_out.flush();

            // set up info for next progress call.
            previousStrLength = ss.str().length() + 1;
            display           = true;
        }
        else // reset
        {
            previousStrLength = 0;
            display           = false;
        }
    }

    return;
}

void Output::print()
{
    // This output is only for regular stdout, not training or 'parse' mode (which is deprecated)
    if (nvvsCommon.parse || nvvsCommon.training)
        return;

    if (globalInfo.size() > 0)
    {
        m_out << std::endl << std::endl;
    }

    for (size_t i = 0; i < globalInfo.size(); i++)
    {
        if (globalInfo[i].find("***") == std::string::npos)
        {
            m_out << " *** ";
        }
        m_out << globalInfo[i] << std::endl;
    }

    if (nvvsCommon.fromDcgm == false)
    {
        m_out << DEPRECATION_WARNING << std::endl;
    }
}

std::string Output::RemoveNewlines(const std::string &str)
{
    std::string altered(str);
    size_t pos;
    // Remove newlines
    while ((pos = altered.find('\n')) != std::string::npos)
    {
        altered[pos] = ' ';
    }
    while ((pos = altered.find('\r')) != std::string::npos)
    {
        altered[pos] = ' ';
    }
    while ((pos = altered.find('\f')) != std::string::npos)
    {
        altered[pos] = ' ';
    }
    return altered;
}

void Output::addInfoStatement(const std::string &info)
{
    this->globalInfo.push_back(RemoveNewlines(info));
}

void Output::AddTrainingResult(const std::string &trainingOut)
{
    m_out << trainingOut;
}
