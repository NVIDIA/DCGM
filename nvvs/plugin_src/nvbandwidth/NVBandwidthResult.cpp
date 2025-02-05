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
#include "NVBandwidthResult.h"

#include <DcgmLogging.h>

#include <boost/algorithm/string.hpp>
#include <limits>


namespace DcgmNs::Nvvs::Plugins::NVBandwidth
{
std::optional<TestCaseStatus> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<TestCaseStatus>)
{
    if (root.empty() || !root.isString())
    {
        return std::nullopt;
    }

    auto const &status = boost::algorithm::to_lower_copy(root.asString());
    if (status == "error")
    {
        return TestCaseStatus::ERROR;
    }
    if (status == "waived")
    {
        return TestCaseStatus::WAIVED;
    }
    if (status == "not found")
    {
        return TestCaseStatus::NOTFOUND;
    }
    if (status == "passed")
    {
        return TestCaseStatus::PASSED;
    }

    return std::nullopt;
}


std::optional<Matrix> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<Matrix>)
{
    if (root.empty() || !root.isArray())
    {
        log_error("bandwidth_matrix is either empty or not an array.");
        return std::nullopt;
    }

    Matrix matrix;

    for (auto const &row : root)
    {
        std::vector<float> rowData;
        for (auto const &value : row)
        {
            if (value.isString() && value.asString() == "N/A")
            {
                // If the value is "N/A", convert it to NaN
                rowData.emplace_back(std::numeric_limits<float>::quiet_NaN());
            }
            else
            {
                try
                {
                    // Convert the value to a floating-point number
                    rowData.emplace_back(std::stof(value.asString()));
                }
                catch (std::invalid_argument const &e)
                {
                    log_error("Invalid argument exception thrown: {}. Happened at row: {}",
                              e.what(),
                              Json::FastWriter {}.write(row));
                    return std::nullopt;
                }
                catch (std::out_of_range const &e)
                {
                    log_error("Out of range exception thrown: {}. Happened at row: {}",
                              e.what(),
                              Json::FastWriter {}.write(row));
                    return std::nullopt;
                }
            }
        }
        // Got [] but no data elements inside the brackets
        if (rowData.empty())
        {
            return std::nullopt;
        }
        else
        {
            matrix.data.emplace_back(std::move(rowData));
        }
    }
    return matrix;
}

std::optional<TestCase> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<TestCase>)
{
    if (!root.isMember("name") || !root["name"].isString() || !root.isMember("status") || !root["status"].isString())
    {
        log_error("Missing name/status entries in the \"testcases\".");
        return std::nullopt;
    }
    TestCase testCase;
    testCase.name  = root["name"].asString();
    auto optStatus = DcgmNs::JsonSerialize::TryDeserialize<TestCaseStatus>(root["status"]);
    if (optStatus.has_value())
    {
        testCase.status = optStatus.value();
    }
    else
    {
        return std::nullopt;
    }

    if (root.isMember("bandwidth_description") && root["bandwidth_description"].isString())
    {
        testCase.bandwidthDescription = root["bandwidth_description"].asString();
    }

    if (root.isMember("sum") && root["sum"].isNumeric())
    {
        testCase.sum = root["sum"].asFloat();
    }

    if (root.isMember("min") && root["min"].isNumeric())
    {
        testCase.min = root["min"].asFloat();
    }

    if (root.isMember("max") && root["max"].isNumeric())
    {
        testCase.max = root["max"].asFloat();
    }
    if (root.isMember("average") && root["average"].isNumeric())
    {
        testCase.avg = root["average"].asFloat();
    }

    if (root.isMember("bandwidth_matrix") && root["bandwidth_matrix"].isArray())
    {
        auto optBandwidthMatrix = DcgmNs::JsonSerialize::TryDeserialize<Matrix>(root["bandwidth_matrix"]);
        if (optBandwidthMatrix.has_value())
        {
            testCase.bandwidthMatrix = std::move(optBandwidthMatrix.value());
        }
        else
        {
            return std::nullopt;
        }
    }
    return testCase;
}

std::optional<NVBandwidthResult> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<NVBandwidthResult>)
{
    if (!root.isMember("nvbandwidth"))
    {
        log_error("Missing member \"nvbandwidth\" in the json file.");
        return std::nullopt;
    }
    Json::Value const &nvb = root["nvbandwidth"];
    if (!nvb.isMember("CUDA Runtime Version") || !nvb["CUDA Runtime Version"].isInt() || !nvb.isMember("Driver Version")
        || !nvb["Driver Version"].isString() || !nvb.isMember("git_version") || !nvb["git_version"].isString()
        || !nvb.isMember("testcases") || !nvb["testcases"].isArray())
    {
        log_error("Missing entries in the json file.");
        return std::nullopt;
    }
    NVBandwidthResult result;
    result.cudaRuntimeVersion = nvb["CUDA Runtime Version"].asInt();
    result.driverVersion      = nvb["Driver Version"].asString();
    result.gitVersion         = nvb["git_version"].asString();
    if (nvb.isMember("error") && nvb["error"].isString())
    {
        result.overallError = nvb["error"].asString();
    }
    if (nvb.isMember("warning") && nvb["warning"].isString())
    {
        result.warning = nvb["warning"].asString();
    }
    // If testcases entry exist, make sure each test case is not empty, otherwise the whole result is invalid
    for (auto const &testCase : nvb["testcases"])
    {
        auto optTestCase = DcgmNs::JsonSerialize::TryDeserialize<TestCase>(testCase);
        if (optTestCase.has_value())
        {
            result.testCases.emplace_back(optTestCase.value());
        }
        else
        {
            return std::nullopt;
        }
    }
    return result;
}

} //namespace DcgmNs::Nvvs::Plugins::NVBandwidth
