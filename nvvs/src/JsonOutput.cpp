/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "NvvsCommon.h"
#include <DcgmBuildInfo.hpp>
#include <fmt/format.h>
#include <ranges>

/* This class fills in a json object in the format:
 * {
 *   "DCGM GPU Diagnostic" : {
 *     "test_categories" : [
 *       {
 *         "category" : "<header>",    # One of Deployment|Hardware|Integration|Performance|Custom
 *         "tests" : [
 *           {
 *             "name" : <name>,
 *             # There is one results entry per GPU for all tests except Software/Deployment.
 *             # Software test has one results entry which represents all GPUs.
 *             "results" : [
 *               {
 *                 # GPU ID (as string) (name is "gpu_ids" for backwards compatibility).
 *                 # For deployment test, this is a CSV string of GPU ids
 *                 "gpu_ids" : <gpu_ids>,
 *                 "status : "<status>",  # One of PASS|FAIL|WARN|SKIPPED
 *                 "warnings" : [         # Optional, depends on test output and result
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

void JsonOutput::header(const std::string &headerString)
{
    if (m_testIndex != 0)
    {
        headerIndex++;
        m_testIndex = 0;
        m_gpuId     = -1;
    }
    else
    {
        m_root[NVVS_VERSION_STR] = std::string(DcgmNs::DcgmBuildInfo().GetVersion());
    }
    m_root[NVVS_HEADERS][headerIndex][NVVS_HEADER] = headerString;
}

bool isSoftwareTest(const std::string &testName)
{
    if (testName == "Denylist" || testName == "NVML Library" || testName == "CUDA Main Library"
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
    softwareTest    = isSoftwareTest(testString);
    std::size_t pos = testString.find(" GPU");
    if (pos == std::string::npos)
    {
        if (m_gpuId != -1)
        {
            m_testIndex++; // Move past the individually reported tests
            m_gpuId = -1;
        }

        m_root[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_TEST_NAME] = testString;
    }
    else
    {
        int nextGpuId = -1;

        // testString is in the format "Test Name GPU<index>", so we need to make the gpu results a member of this
        // test.
        std::string testName(testString.substr(0, pos)); // Capture just the name
        try
        {
            nextGpuId = std::stoi(testString.substr(pos + 4)); // Capture just the index
        }
        catch (std::exception const &)
        {
            nextGpuId = -1;
        }

        if (nextGpuId != -1 && nextGpuId <= m_gpuId)
        {
            m_testIndex++;
        }
        m_gpuId = nextGpuId;

        m_root[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_TEST_NAME] = testName;
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

[[maybe_unused]] static DcgmNs::Nvvs::Json::Result MakeSkipResult(
    nvvsPluginResult_t overallResult,
    const std::vector<dcgmDiagSimpleResult_t> &perGpuResults,
    const std::vector<dcgmDiagEvent_t> &info)
{
    using namespace DcgmNs::Nvvs::Json;
    Result result {};
    if (overallResult == nvvsPluginResult_t::NVVS_RESULT_SKIP)
    {
        for (auto const &gpuResult : perGpuResults)
        {
            result.gpuIds.ids.insert(gpuResult.gpuId);
            result.status.result = overallResult;
            for (auto const &msg : info)
            {
                if (msg.gpuId == gpuResult.gpuId || msg.gpuId == DCGM_DIAG_ALL_GPUS)
                {
                    if (!result.info.has_value())
                    {
                        result.info = Info {};
                    }
                    (*result.info).messages.emplace_back(msg.msg);
                }
            }
        }
    }

    return result;
}

[[maybe_unused]] static DcgmNs::Nvvs::Json::Result MakeSoftwareResult(nvvsPluginResult_t overallResult,
                                                                      std::string const &gpuIdsStr,
                                                                      const std::vector<dcgmDiagEvent_t> &errors,
                                                                      const std::vector<dcgmDiagEvent_t> &info)
{
    using namespace DcgmNs::Nvvs::Json;
    Result result {};
    for (auto const &id : DcgmNs::Split(gpuIdsStr, ','))
    {
        int gpuId      = -1;
        auto [ptr, ec] = std::from_chars(id.data(), id.data() + id.size(), gpuId);
        if (ec != std::errc {})
        {
            log_error("Failed to parse GPU ID: {}", id);
        }
        else
        {
            result.gpuIds.ids.insert(gpuId);
        }
    }
    result.status.result = overallResult;

    if (!errors.empty())
    {
        result.warnings = std::vector<Warning> {};
    }
    for (const auto &error : errors)
    {
        (*result.warnings).emplace_back(Warning { .message = error.msg, .error_code = error.errorCode });
    }

    if (!info.empty())
    {
        result.info = Info {};
    }
    for (const auto &i : info)
    {
        (*result.info).messages.emplace_back(i.msg);
    }

    return result;
}

[[maybe_unused]] static std::vector<DcgmNs::Nvvs::Json::Result> MakeHardwareResults(
    nvvsPluginResult_t overallResult,
    std::vector<unsigned int> const &gpuIndices,
    const std::vector<dcgmDiagSimpleResult_t> &perGpuResults,
    const std::vector<dcgmDiagEvent_t> &errors,
    const std::vector<dcgmDiagEvent_t> &info)
{
    using namespace DcgmNs::Nvvs::Json;
    std::vector<Result> result {};

    // Add Json for each GPU
    for (unsigned int gpuId : gpuIndices)
    {
        Result gpuResult {};
        gpuResult.gpuIds.ids.insert((int)gpuId);
        gpuResult.status.result = overallResult;

        for (auto &&entry : perGpuResults)
        {
            if ((unsigned int)entry.gpuId == gpuId)
            {
                gpuResult.status.result = entry.result;
                break;
            }
        }

        // GPU %u: Prefix for general warnings/info messages
        for (auto &&error : errors)
        {
            if (error.gpuId == DCGM_DIAG_ALL_GPUS || (error.gpuId >= 0 && (unsigned)error.gpuId == gpuId))
            {
                if (!gpuResult.warnings.has_value())
                {
                    gpuResult.warnings = std::vector<Warning> {};
                }
                (*gpuResult.warnings)
                    .emplace_back(Warning { .message    = fmt::format("GPU {}: {}", gpuId, error.msg),
                                            .error_code = error.errorCode });
            }
            gpuResult.status.result = nvvsPluginResult_t::NVVS_RESULT_FAIL;
        }

        for (auto &&singleInfo : info)
        {
            if (singleInfo.gpuId == DCGM_DIAG_ALL_GPUS
                || (singleInfo.gpuId >= 0 && (unsigned)singleInfo.gpuId == gpuId))
            {
                if (!gpuResult.info.has_value())
                {
                    gpuResult.info = Info {};
                }
                (*gpuResult.info).messages.emplace_back(fmt::format("GPU {}: {}", gpuId, singleInfo.msg));
            }
        }

        result.push_back(std::move(gpuResult));
    }

    return result;
}

/*****************************************************************************/
void JsonOutput::Result(nvvsPluginResult_t overallResult,
                        const std::vector<dcgmDiagSimpleResult_t> &perGpuResults,
                        const std::vector<dcgmDiagEvent_t> &errors,
                        const std::vector<dcgmDiagEvent_t> &info,
                        const std::optional<std::any> &pluginSpecificData)
{
    std::string resultStr = resultEnumToString(overallResult);

    if (overallResult == NVVS_RESULT_SKIP)
    {
        for (size_t i = 0; i < m_gpuIndices.size(); i++)
        {
            Json::Value resultField;
            resultField[NVVS_STATUS]  = resultStr;
            resultField[NVVS_GPU_IDS] = fmt::to_string(m_gpuIndices[i]);
            unsigned int gpuId        = m_gpuIndices[i];

            for (auto &&singleInfo : info)
            {
                if (singleInfo.gpuId == DCGM_DIAG_ALL_GPUS
                    || (singleInfo.gpuId >= 0 && (unsigned)singleInfo.gpuId == gpuId))
                {
                    AppendInfo(singleInfo, resultField, "");
                }
            }

            m_root[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_RESULTS][m_gpuIndices[i]] = resultField;
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

        m_root[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_RESULTS][0] = resultField;

        if (pluginSpecificData)
        {
            try
            {
                auto auxData = std::any_cast<Json::Value>(*pluginSpecificData);
                m_root[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_AUX_DATA] = auxData;
            }
            catch (std::bad_any_cast const &e)
            {
                log_debug("Failed to cast plugin specific data to json: {}", e.what());
            }
        }

        m_testIndex++;
    }
    else
    {
        // Add Json for each GPU
        for (size_t i = 0; i < m_gpuIndices.size(); i++)
        {
            unsigned int gpuId           = m_gpuIndices[i];
            nvvsPluginResult_t gpuResult = NVVS_RESULT_SKIP;

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

            for (auto &&error : errors)
            {
                if (error.gpuId == DCGM_DIAG_ALL_GPUS || (error.gpuId >= 0 && (unsigned)error.gpuId == gpuId))
                {
                    AppendError(error, resultField, fmt::format("GPU {} ", gpuId));
                    gpuResult = NVVS_RESULT_FAIL;
                }
            }

            for (auto &&singleInfo : info)
            {
                if (singleInfo.gpuId == DCGM_DIAG_ALL_GPUS
                    || (singleInfo.gpuId >= 0 && (unsigned)singleInfo.gpuId == gpuId))
                {
                    AppendInfo(singleInfo, resultField, fmt::format("GPU {} ", gpuId));
                }
            }

            /* if any errors are detected then the test fails, but individual gpus may pass */
            resultField[NVVS_STATUS]  = resultEnumToString(gpuResult);
            resultField[NVVS_GPU_IDS] = gpuId;

            m_root[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_RESULTS][gpuId] = resultField;
        }

        if (pluginSpecificData)
        {
            try
            {
                auto auxData = std::any_cast<Json::Value>(*pluginSpecificData);
                m_root[NVVS_HEADERS][headerIndex][NVVS_TESTS][m_testIndex][NVVS_AUX_DATA] = auxData;
            }
            catch (std::bad_any_cast const &e)
            {
                log_debug("Failed to cast plugin specific data to json: {}", e.what());
            }
        }
    }

    if (m_gpuId == -1)
    {
        m_testIndex++;
    }
}

void JsonOutput::updatePluginProgress(unsigned int /*progress*/, bool /*clear*/)
{
    // NO-OP for Json Output
}

void JsonOutput::print()
{
    Json::Value complete;
    complete[NVVS_NAME] = m_root;
    if (!nvvsCommon.fromDcgm)
    {
        complete[NVVS_GLOBAL_WARN] = DEPRECATION_WARNING;
    }
    m_out << complete.toStyledString();
    m_out.flush();
}

void JsonOutput::addInfoStatement(const std::string &info)
{
    if (m_root[NVVS_INFO].empty())
    {
        Json::Value infoArray;
        infoArray[m_globalInfoCount] = RemoveNewlines(info);

        m_root[NVVS_INFO] = infoArray;
    }
    else
    {
        m_root[NVVS_INFO][m_globalInfoCount] = RemoveNewlines(info);
    }

    m_globalInfoCount++;
}

JsonOutput::JsonOutput(std::vector<unsigned int> gpuIndices)
    : gpuList(fmt::to_string(fmt::join(gpuIndices, ",")))
    , m_gpuIndices(std::move(gpuIndices))
{}

void JsonOutput::AddGpusAndDriverVersion(std::vector<Gpu *> &gpuList)
{
    Json::Value gpuDevIds;
    int index = 0;

    for (auto gpu : gpuList)
    {
        if (index == 0)
        {
            m_root[NVVS_DRIVER_VERSION] = gpu->GetDriverVersion();
        }

        std::string devid = gpu->getDevicePciDeviceId();
        gpuDevIds[index]  = devid;
        index++;
    }

    m_root[NVVS_GPU_DEV_IDS] = gpuDevIds;
}
