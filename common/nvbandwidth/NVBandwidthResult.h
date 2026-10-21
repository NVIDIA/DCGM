/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmJsonSerialize.hpp>

#include <json/json.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace DcgmNs::NvBandwidth
{

/**
 * NVBandwidth testcase status values emitted in JSON output.
 */
enum class TestCaseStatus
{
    ERROR,
    WAIVED,
    NOTFOUND,
    PASSED,
};

/**
 * Convert an NVBandwidth testcase status to its JSON string representation.
 *
 * @param[in] status Status value to convert.
 * @return String representation used by NVBandwidth JSON output.
 */
std::string_view ToString(TestCaseStatus status);

/**
 * Parse an NVBandwidth testcase status from JSON.
 *
 * @param[in] root JSON value containing the status string.
 * @return Parsed status, or std::nullopt when the value is not a recognized status.
 */
std::optional<TestCaseStatus> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<TestCaseStatus>);

/**
 * Parsed NVBandwidth matrix data.
 */
struct Matrix
{
    std::vector<std::vector<float>> data; //!< Parsed rows of bandwidth or latency values.
};

/**
 * Parse an NVBandwidth matrix from JSON.
 *
 * @param[in] root JSON array containing matrix row arrays.
 * @return Parsed matrix, or std::nullopt when any matrix element is invalid.
 */
std::optional<Matrix> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<Matrix>);

/**
 * Parsed NVBandwidth GPU device-list entry.
 */
struct GpuDevice
{
    unsigned int gpuId {}; //!< GPU id reported by NVBandwidth.
    std::string hostname;  //!< Hostname reported in the GPU device-list entry.
    std::string raw;       //!< Original device-list entry.
};

/**
 * Parse an NVBandwidth GPU device-list entry from JSON.
 *
 * @param[in] root JSON string containing one GPU device-list entry.
 * @return Parsed GPU device, or std::nullopt when the entry cannot be parsed.
 */
std::optional<GpuDevice> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<GpuDevice>);

/**
 * Parsed NVBandwidth testcase result.
 */
struct TestCase
{
    std::string name {};                 //!< Name of the test case.
    TestCaseStatus status {};            //!< Testcase status.
    std::string bandwidthDescription {}; //!< Bandwidth matrix description.
    float sum { 0.0F };                  //!< Sum reported by NVBandwidth.
    float min { 0.0F };                  //!< Minimum value reported by NVBandwidth.
    float max { 0.0F };                  //!< Maximum value reported by NVBandwidth.
    float avg { 0.0F };                  //!< Average value reported by NVBandwidth.
    Matrix bandwidthMatrix {};           //!< Parsed bandwidth matrix.
};

/**
 * Parse an NVBandwidth testcase from JSON.
 *
 * @param[in] root JSON object containing a testcase result.
 * @return Parsed testcase, or std::nullopt when required fields are missing or invalid.
 */
std::optional<TestCase> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<TestCase>);

/**
 * Parsed NVBandwidth JSON output.
 */
struct NVBandwidthResult
{
    int cudaRuntimeVersion { 12070 };           //!< CUDA runtime version.
    std::string driverVersion {};               //!< Driver version.
    std::string gitVersion {};                  //!< NVBandwidth git version.
    std::optional<std::string> overallError {}; //!< Top-level NVBandwidth error, when present.
    std::string warning {};                     //!< Top-level NVBandwidth warning, when present.
    std::vector<TestCase> testCases {};         //!< Parsed testcase results.
    std::vector<GpuDevice> gpuDevices {};       //!< Parsed GPU device-list entries.
};

/**
 * Parse NVBandwidth JSON output.
 *
 * @param[in] root JSON object containing the top-level nvbandwidth member.
 * @return Parsed NVBandwidth result, or std::nullopt when required fields are missing or invalid.
 */
std::optional<NVBandwidthResult> ParseJson(Json::Value const &root, DcgmNs::JsonSerialize::To<NVBandwidthResult>);

} // namespace DcgmNs::NvBandwidth
