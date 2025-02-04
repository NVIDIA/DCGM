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
#pragma once

#include "NVBandwidthPlugin.h"

#include <DcgmJsonSerialize.hpp>

#include <fmt/format.h>
#include <json/json.h>

#include <cmath>
#include <optional>
#include <string>
#include <variant>
#include <vector>


namespace DcgmNs::Nvvs::Plugins::NVBandwidth
{

enum class TestCaseStatus
{
    ERROR,
    WAIVED,
    NOTFOUND,
    PASSED,
};

std::optional<TestCaseStatus> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<TestCaseStatus>);


struct Matrix
{
    std::vector<std::vector<float>> data;
};

std::optional<Matrix> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<Matrix>);


struct TestCase
{
    std::string name {};                 /*!< Name of the test case */
    TestCaseStatus status {};            /*!< error|waived|not found|passed */
    std::string bandwidthDescription {}; /*!< # memcpy CE GPU(row) -> GPU(column) bandwidth GB/s, for example */
    float sum;
    float min;
    float max;
    float avg;
    Matrix bandwidthMatrix;
};

std::optional<TestCase> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<TestCase>);


struct NVBandwidthResult
{
    int cudaRuntimeVersion { 12070 }; /*!< CUDA Runtime Version */
    std::string driverVersion {};     /*!< Driver Version */
    std::string gitVersion {};        /*!< Git version */
    std::optional<std::string> overallError {
        std::nullopt
    }; /*!< "invalid arguments, ...", # if this is present, the tests probably didn't run */
    std::string warning; /*!< smaller buffer size warning */
    std::vector<TestCase> testCases;
};

std::optional<NVBandwidthResult> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<NVBandwidthResult>);

} //namespace DcgmNs::Nvvs::Plugins::NVBandwidth
