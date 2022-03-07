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

#include "JsonOutput.h"
#include "NvidiaValidationSuite.h"
#include "NvvsCommon.h"
#include <DcgmBuildInfo.hpp>

/* This class fills in a json object in the format:
 * {
 *   "DCGM GPU Diagnostic" : {
 *     "test_categories" : [
 *       {
 *         "category" : "<header>",    # One of Deployment|Hardware|Integration|Performance|Custom
 *         "tests" : [
 *           {
 *             "name" : <name>,
 *             "results" : [              # There is one results entry per GPU for all tests except Software/Deployment.
 * Software test has one results entry which represents all GPUs.
 *               {
 *                 "gpu_ids" : <gpu_ids>, # GPU ID (as string) (name is "gpu_ids" for backwards compatibility). For
 * deployment test, this is a CSV string of GPU ids "status : "<status>",  # One of PASS|FAIL|WARN|SKIPPED "warnings" :
 * [         # Optional, depends on test output and result
 *                   "<warning_text>", ...
 *                 ],
 *                 "info" : [             # Optional, depends on test output and result
 *                    "<info_text>", ...
 *                 ]
 *               }, ...
 *             ]
 *           }, ...
 *         ]
 *       }, ...
 *     ],
 *     "version" : "<version_str>" # 1.7
 *   }
 * }
 */

// Forward declarations
void AddStringVectorToJson(Json::Value &value, const std::vector<std::string> &strings, const char *prefix = "");

void JsonOutput::header(const std::string &headerString)
{
    if (nvvsCommon.training)
    {
        return; // Training mode doesn't want this output
    }

    if (m_testIndex != 0)
    {
        headerIndex++;
        m_testIndex = 0;
        m_gpuId     = -1;
    }
    else
    {
        jv[NVVS_VERSION_STR] = std::string(DcgmNs::DcgmBuildInfo().GetVersion());
    }
    jv[NVVS_HEADERS][headerIndex][NVVS_HEADER] = headerString;
}

bool isSoftwareTest(const std::string &testName)
{
    if (testName == "Blacklist" || testName == "NVML Library" || testName == "CUDA Main Library"
        || testName == "CUDA Toolkit Libraries" || testName == "Permissions and OS-related Blocks"
        || testName == "Persistence Mode" || testName == "Environmental Variables"
        || testName == "Page Retirement/Row Remap" || testName == "Graphics Processes" || testName == "Inforom")
    {
        return true;
    }
    return false;
}

void JsonOutput::prep(const std::string &testString)
{
    if (nvvsCommon.training)
    {
        return; // Training mode doesn't want this output
    }

    softwareTest    = isSoftwareTest(testString);
    std::size_t pos = testString.find(" GPU");
    if (pos == std::string::npos)
    {
        if (m_gpuId != -1)
        {
            m_testIndex++; // Move past the individually reported tests
            m_gpuId = -1;
        }

        jv[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_TEST_NAME] = testString;
    }
    else
    {
        int nextGpuId;

        // testString is in the format "Test Name GPU<index>", so we need to make the gpu results a member of this
        // test.
        std::string testName(testString.substr(0, pos));                  // Capture just the name
        nextGpuId = strtol(testString.substr(pos + 4).c_str(), NULL, 10); // Capture just the <index>

        if (nextGpuId <= m_gpuId)
        {
            m_testIndex++;
        }
        m_gpuId = nextGpuId;

        jv[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_TEST_NAME] = testName;
    }
}

void AddStringVectorToJson(Json::Value &errorArray, const std::vector<std::string> &strings, const char *prefix)
{
    for (size_t i = 0; i < strings.size(); i++)
    {
        errorArray.append(prefix + strings[i]);
    }
}

void AddErrorVectorToJson(Json::Value &errorArray, const std::vector<DcgmError> &errors, const char *prefix = "")
{
    for (size_t i = 0; i < errors.size(); i++)
    {
        Json::Value entry;
        entry[NVVS_WARNING]  = prefix + errors[i].GetMessage();
        entry[NVVS_ERROR_ID] = errors[i].GetCode();

        errorArray.append(entry);
    }
}

/*****************************************************************************/
void JsonOutput::AppendError(const dcgmDiagEvent_t &error, Json::Value &resultField, const std::string &prefix)
{
    Json::Value entry;
    entry[NVVS_WARNING]  = prefix + error.msg;
    entry[NVVS_ERROR_ID] = error.errorCode;
    resultField[NVVS_WARNINGS].append(entry);
}

/*****************************************************************************/
void JsonOutput::AppendInfo(const dcgmDiagEvent_t &info, Json::Value &resultField, const std::string &prefix)
{
    resultField[NVVS_INFO].append(prefix + info.msg);
}

/*****************************************************************************/
void JsonOutput::Result(nvvsPluginResult_t overallResult,
                        const std::vector<dcgmDiagSimpleResult_t> &perGpuResults,
                        const std::vector<dcgmDiagEvent_t> &errors,
                        const std::vector<dcgmDiagEvent_t> &info)
{
    if (nvvsCommon.training)
    {
        return; // Training mode doesn't want this output
    }

    char buf[26];
    std::string resultStr = resultEnumToString(overallResult);

    if (overallResult == NVVS_RESULT_SKIP)
    {
        for (size_t i = 0; i < m_gpuIndices.size(); i++)
        {
            Json::Value resultField;
            resultField[NVVS_STATUS] = resultStr;
            snprintf(buf, sizeof(buf), "%d", m_gpuIndices[i]);
            resultField[NVVS_GPU_IDS]                                                             = buf;
            jv[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_RESULTS][m_gpuIndices[i]] = resultField;
        }
        m_testIndex++;
    }
    // Software tests are independent of GPUs and have the same results for all GPUs
    else if (softwareTest)
    {
        Json::Value resultField;

        resultField[NVVS_STATUS]  = resultStr;
        resultField[NVVS_GPU_IDS] = gpuList;

        for (unsigned int i = 0; i < errors.size(); i++)
        {
            AppendError(errors[i], resultField);
        }

        for (unsigned int i = 0; i < info.size(); i++)
        {
            AppendInfo(info[i], resultField);
        }

        jv[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_RESULTS][0] = resultField;
        m_testIndex++;
    }
    else
    {
        // Add Json for each GPU
        for (size_t i = 0; i < m_gpuIndices.size(); i++)
        {
            unsigned int gpuId           = m_gpuIndices[i];
            nvvsPluginResult_t gpuResult = overallResult;

            for (auto &&entry : perGpuResults)
            {
                if ((unsigned int)entry.gpuId == gpuId)
                {
                    gpuResult = entry.result;
                    break;
                }
            }

            Json::Value resultField;
            // GPU %u: Prefix for general warnings/info messages
            snprintf(buf, sizeof(buf), "GPU %u: ", gpuId);

            for (auto &&error : errors)
            {
                if (error.gpuId == gpuId || error.gpuId == DCGM_DIAG_ALL_GPUS)
                {
                    AppendError(error, resultField, buf);
                    gpuResult = NVVS_RESULT_FAIL;
                }
            }

            for (auto &&singleInfo : info)
            {
                if (singleInfo.gpuId == gpuId || singleInfo.gpuId == DCGM_DIAG_ALL_GPUS)
                {
                    AppendInfo(singleInfo, resultField, buf);
                }
            }

            /* if any errors are detected then the test fails, but individual gpus may pass */
            resultField[NVVS_STATUS]                                                    = resultEnumToString(gpuResult);
            resultField[NVVS_GPU_IDS]                                                   = gpuId;
            jv[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_RESULTS][gpuId] = resultField;
        }
    }

    if (m_gpuId == -1)
    {
        m_testIndex++;
    }
}

void JsonOutput::updatePluginProgress(unsigned int progress, bool clear)
{
    // NO-OP for Json Output
}

void JsonOutput::print()
{
    Json::Value complete;
    complete[NVVS_NAME] = jv;
    if (nvvsCommon.fromDcgm == false)
    {
        complete[NVVS_GLOBAL_WARN] = DEPRECATION_WARNING;
    }
    m_out << complete.toStyledString();
    m_out.flush();
}

void JsonOutput::addInfoStatement(const std::string &info)
{
    if (jv[NVVS_INFO].empty() == true)
    {
        Json::Value infoArray;
        infoArray[globalInfoCount] = RemoveNewlines(info);
        jv[NVVS_INFO]              = infoArray;
    }
    else
    {
        jv[NVVS_INFO][globalInfoCount] = RemoveNewlines(info);
    }

    globalInfoCount++;
}

void JsonOutput::AddTrainingResult(const std::string &trainingOut)
{
    if (jv[NVVS_VERSION_STR].empty())
    {
        jv[NVVS_VERSION_STR] = std::string(DcgmNs::DcgmBuildInfo().GetVersion());
    }

    jv[NVVS_TRAINING_MSG] = trainingOut;
}
