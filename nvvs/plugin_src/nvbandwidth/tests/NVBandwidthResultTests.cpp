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
#include <NVBandwidthPlugin.h>

#include <NVBandwidthResult.h>

#include <PluginInterface.h>
#include <catch2/catch_all.hpp>

TEST_CASE("NVBandwidthResult: Deserialize TestCaseStatus")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    Json::Value root;
    Json::Reader reader;

    SECTION("testcases:status: error|waived|not found|passed")
    {
        std::string jsonStr
            = R"({ "nvbandwidth" : { "testcases": [ {"status" : "error"}, {"status" : "waived"}, {"status" : "not found"}, {"status" : "passed"} ] } })";
        bool parsingSuccessful = reader.parse(jsonStr, root);
        REQUIRE(parsingSuccessful == true);
        auto const resultError
            = DcgmNs::JsonSerialize::Deserialize<TestCaseStatus>(root["nvbandwidth"]["testcases"][0]["status"]);
        REQUIRE(resultError == TestCaseStatus::ERROR);
        auto const resultWaived
            = DcgmNs::JsonSerialize::Deserialize<TestCaseStatus>(root["nvbandwidth"]["testcases"][1]["status"]);
        REQUIRE(resultWaived == TestCaseStatus::WAIVED);
        auto const resultNotFound
            = DcgmNs::JsonSerialize::Deserialize<TestCaseStatus>(root["nvbandwidth"]["testcases"][2]["status"]);
        REQUIRE(resultNotFound == TestCaseStatus::NOTFOUND);
        auto const resultPassed
            = DcgmNs::JsonSerialize::Deserialize<TestCaseStatus>(root["nvbandwidth"]["testcases"][3]["status"]);
        REQUIRE(resultPassed == TestCaseStatus::PASSED);
    }

    SECTION("testcases:status: invalid strings")
    {
        std::string jsonStr    = R"({ "nvbandwidth" : { "testcases": [ {"status" : "invalid string"}] } })";
        bool parsingSuccessful = reader.parse(jsonStr, root);
        REQUIRE(parsingSuccessful == true);
        REQUIRE_THROWS(
            DcgmNs::JsonSerialize::Deserialize<TestCaseStatus>(root["nvbandwidth"]["testcases"][0]["status"]));
    }

    SECTION("testcases:status: empty")
    {
        std::string jsonStr    = R"({ "nvbandwidth" : { "testcases": [ {}] } })";
        bool parsingSuccessful = reader.parse(jsonStr, root);
        REQUIRE(parsingSuccessful == true);
        REQUIRE_THROWS(
            DcgmNs::JsonSerialize::Deserialize<TestCaseStatus>(root["nvbandwidth"]["testcases"][0]["status"]));
    }
}

TEST_CASE("NVBandwidthResult: Deserialize brandwidth_matrix but with invalid data inside")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    Json::Value root;
    Json::Reader reader;

    std::string jsonStr = R"(
                            {
                                "nvbandwidth": {
                                    "testcases" : [
                                        {
                                            "bandwidth_matrix" : [
                                                [ "N/A", "this", "is", "invalid", "374.30", "373.09", "372.05", "372.96" ]
                                            ]
                                        },
                                        {
                                            "bandwidth_matrix" : [
                                                []
                                            ]
                                        }
                                    ]
                                }
                            }
                            )";

    bool parsingSuccessful = reader.parse(jsonStr, root);
    REQUIRE(parsingSuccessful == true);
    auto const result0
        = DcgmNs::JsonSerialize::TryDeserialize<Matrix>(root["nvbandwidth"]["testcases"][0]["bandwidth_matrix"]);
    REQUIRE(result0.has_value() == false);
    REQUIRE_THROWS(DcgmNs::JsonSerialize::Deserialize<Matrix>(root["nvbandwidth"]["testcases"][0]["bandwidth_matrix"]));
    auto const result1
        = DcgmNs::JsonSerialize::TryDeserialize<Matrix>(root["nvbandwidth"]["testcases"][1]["bandwidth_matrix"]);
    REQUIRE(result1.has_value() == false);
    REQUIRE_THROWS(DcgmNs::JsonSerialize::Deserialize<Matrix>(root["nvbandwidth"]["testcases"][1]["bandwidth_matrix"]));
}


TEST_CASE("NVBandwidthResult: Deserialize brandwidth_matrix")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    Json::Value root;
    Json::Reader reader;

    std::string jsonStr = R"(
                            {
                                "nvbandwidth": {
                                    "testcases" : [
                                        {
                                            "bandwidth_matrix" : [
                                                [ "N/A", "373.22", "373.22", "374.39", "374.30", "373.09", "372.05", "372.96" ],
                                                [ "372.05", "N/A", "373.22", "370.90", "372.05", "373.22", "372.05", "371.92" ]
                                            ]
                                        }
                                    ]
                                }
                            }
                            )";

    bool parsingSuccessful = reader.parse(jsonStr, root);
    REQUIRE(parsingSuccessful == true);
    auto const result
        = DcgmNs::JsonSerialize::Deserialize<Matrix>(root["nvbandwidth"]["testcases"][0]["bandwidth_matrix"]);
    auto const &resultData = result.data;

    std::vector<std::vector<float>> groundTruth = { { NAN, 373.22, 373.22, 374.39, 374.30, 373.09, 372.05, 372.96 },
                                                    { 372.05, NAN, 373.22, 370.90, 372.05, 373.22, 372.05, 371.92 } };

    REQUIRE(resultData.size() == groundTruth.size());
    constexpr float FLT_EPSILON { 0.000001 };
    for (size_t jj = 0; jj < groundTruth.size(); jj++)
    {
        REQUIRE(resultData[jj].size() == groundTruth[jj].size());
        for (size_t ii = 0; ii < groundTruth[jj].size(); ii++)
        {
            if (isnanf(groundTruth[jj][ii]))
            {
                REQUIRE(isnanf(resultData[jj][ii]));
            }
            else
            {
                REQUIRE(fabsf(resultData[jj][ii] - groundTruth[jj][ii]) < FLT_EPSILON * groundTruth[jj][ii]);
            }
        }
    }
}

TEST_CASE("NVBandwidthResult: Deserialize TestCase with empty status")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    Json::Value root;
    Json::Reader reader;

    std::string jsonStr = R"(
                            {
                                "nvbandwidth": {
                                    "testcases" : [
                                        {
                                            "name" : "testname",
                                            "status" : "",
                                            "bandwidth_description" : "memcpy CE GPU(row) -> GPU(column) bandwidth GB/s",
                                            "bandwidth_matrix" : [
                                                [ "N/A", "373.22", "373.22", "374.39", "374.30", "373.09", "372.05", "372.96" ], 
                                                [ "372.05", "N/A", "373.22", "370.90", "372.05", "373.22", "372.05", "371.92" ]
                                            ],
                                            "sum" : 20841.13,
                                            "min" : 370.90,
                                            "max" : 373.22,
                                            "average" : 371.92
                                        }
                                    ]
                                }
                            }
                        )";

    bool parsingSuccessful = reader.parse(jsonStr, root);
    REQUIRE(parsingSuccessful == true);
    // TestCase's status is empty or not valid, the whole TestCase is marked as invalid, thus returning std::nullopt
    auto const result = DcgmNs::JsonSerialize::TryDeserialize<TestCase>(root["nvbandwidth"]["testcases"][0]);
    REQUIRE(result.has_value() == false);
    // Invalid TestCase parsing throws exceptions with Deserialize(...)
    REQUIRE_THROWS(DcgmNs::JsonSerialize::Deserialize<TestCase>(root["nvbandwidth"]["testcases"][0]));
}

TEST_CASE("NVBandwidthResult: Deserialize TestCase with invalid bandwidth_matrix elements")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    Json::Value root;
    Json::Reader reader;

    std::string jsonStr = R"(
                            {
                                "nvbandwidth": {
                                    "testcases" : [
                                        {
                                            "name" : "testname1",
                                            "status" : "passed",
                                            "bandwidth_description" : "memcpy CE GPU(row) -> GPU(column) bandwidth GB/s",
                                            "bandwidth_matrix" : [
                                                [ "N/A", "373.22", "373.22", "foo", "374.30", "373.09", "372.05", "372.96" ], 
                                                [ "372.05", "N/A", "373.22", "370.90", "372.05", "373.22", "372.05", "371.92" ]
                                            ],
                                            "sum" : 20841.13,
                                            "min" : 370.90,
                                            "max" : 373.22,
                                            "average" : 371.92
                                        },
                                        {
                                            "name" : "testname2",
                                            "status" : "passed",
                                            "bandwidth_description" : "memcpy CE GPU(row) -> GPU(column) bandwidth GB/s",
                                            "bandwidth_matrix" : [
                                                [ "N/A", "373.22", "373.22", "374.39", "374.30", "373.09", "372.05", "372.96" ], 
                                                [ ]
                                            ],
                                            "sum" : 20841.13,
                                            "min" : 370.90,
                                            "max" : 373.22,
                                            "average" : 371.92
                                        }
                                    ]
                                }
                            }
                        )";

    bool parsingSuccessful = reader.parse(jsonStr, root);
    REQUIRE(parsingSuccessful == true);
    // TestCase's status is empty or not valid, the whole TestCase is marked as invalid, thus returning std::nullopt
    auto const result0 = DcgmNs::JsonSerialize::TryDeserialize<TestCase>(root["nvbandwidth"]["testcases"][0]);
    REQUIRE(result0.has_value() == false);
    // Invalid TestCase parsing throws exceptions with Deserialize(...)
    REQUIRE_THROWS(DcgmNs::JsonSerialize::Deserialize<TestCase>(root["nvbandwidth"]["testcases"][0]));

    auto const result1 = DcgmNs::JsonSerialize::TryDeserialize<TestCase>(root["nvbandwidth"]["testcases"][1]);
    REQUIRE(result1.has_value() == false);
    // Invalid TestCase parsing throws exceptions with Deserialize(...)
    REQUIRE_THROWS(DcgmNs::JsonSerialize::Deserialize<TestCase>(root["nvbandwidth"]["testcases"][1]));
}

TEST_CASE("NVBandwidthResult: Deserialize TestCase")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    Json::Value root;
    Json::Reader reader;

    std::string jsonStr = R"(
                        {
                            "nvbandwidth": {
                                "testcases" : [
                                    {
                                        "name" : "testname",
                                        "status" : "passed",
                                        "bandwidth_description" : "memcpy CE GPU(row) -> GPU(column) bandwidth GB/s",
                                        "bandwidth_matrix" : [
                                            [ "N/A", "373.22", "373.22", "374.39", "374.30", "373.09", "372.05", "372.96" ], 
                                            [ "372.05", "N/A", "373.22", "370.90", "372.05", "373.22", "372.05", "371.92" ]
                                        ],
                                        "sum" : 20841.13,
                                        "min" : 370.90,
                                        "max" : 373.22,
                                        "average" : 371.92
                                    }
                                ]
                            }
                        }
                    )";

    bool parsingSuccessful = reader.parse(jsonStr, root);
    REQUIRE(parsingSuccessful == true);
    auto const result = DcgmNs::JsonSerialize::Deserialize<TestCase>(root["nvbandwidth"]["testcases"][0]);
    constexpr float FLT_EPSILON { 0.000001 };
    REQUIRE(result.name == "testname");
    REQUIRE(result.status == TestCaseStatus::PASSED);
    REQUIRE(result.bandwidthDescription == "memcpy CE GPU(row) -> GPU(column) bandwidth GB/s");
    REQUIRE(fabsf(result.sum - 20841.13) < FLT_EPSILON * 20841.13);
    REQUIRE(fabsf(result.min - 370.90) < FLT_EPSILON * 370.90);
    REQUIRE(fabsf(result.max - 373.22) < FLT_EPSILON * 373.22);
    REQUIRE(fabsf(result.avg - 371.92) < FLT_EPSILON * 371.92);

    std::vector<std::vector<float>> groundTruth = { { NAN, 373.22, 373.22, 374.39, 374.30, 373.09, 372.05, 372.96 },
                                                    { 372.05, NAN, 373.22, 370.90, 372.05, 373.22, 372.05, 371.92 } };

    auto const &matrixData = result.bandwidthMatrix.data;
    REQUIRE(matrixData.size() == groundTruth.size());

    for (size_t jj = 0; jj < groundTruth.size(); jj++)
    {
        REQUIRE(matrixData[jj].size() == groundTruth[jj].size());
        for (size_t ii = 0; ii < groundTruth[jj].size(); ii++)
        {
            if (isnanf(groundTruth[jj][ii]))
            {
                REQUIRE(isnanf(matrixData[jj][ii]));
            }
            else
            {
                REQUIRE(fabsf(matrixData[jj][ii] - groundTruth[jj][ii]) < FLT_EPSILON * groundTruth[jj][ii]);
            }
        }
    }
}

TEST_CASE("NVBandwidthResult: Deserialize NVBandwidthResult with invalid testcase")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    const char *ss = R"(
                        {
                            "nvbandwidth": {
                                "CUDA Runtime Version" : 12030,
                                "Driver Version" : "535.40",
                                "git_version" : "2.34.1",
                                "error" : "invalid arguments",
                                "warning" : "smaller buffer size warning",
                                "testcases" : [
                                    {
                                        "name" : "invalid test",
                                        "status" : "not a valid status",
                                        "bandwidth_description" : "memcpy CE GPU(row) -> GPU(column) bandwidth GB/s",
                                        "bandwidth_matrix" : [
                                            []
                                        ],
                                        "sum" : 20841.13,
                                        "min" : 370.90,
                                        "max" : 373.22,
                                        "average" : 371.92
                                    }
                                ]
                            }
                        }
                    )";

    REQUIRE_THROWS(DcgmNs::JsonSerialize::Deserialize<NVBandwidthResult>(std::string_view { ss, ss + strlen(ss) }));
}


TEST_CASE("NVBandwidthResult: Deserialize NVBandwidthResult with valid testcase")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    const char *ss = R"(
                        {
                            "nvbandwidth": {
                                "CUDA Runtime Version" : 12030,
                                "Driver Version" : "535.40",
                                "git_version" : "2.34.1",
                                "error" : "invalid arguments",
                                "warning" : "smaller buffer size warning",
                                "testcases" : [
                                    {
                                        "name" : "valid test",
                                        "status" : "passed",
                                        "bandwidth_description" : "memcpy CE GPU(row) -> GPU(column) bandwidth GB/s",
                                        "bandwidth_matrix" : [
                                            [ "N/A", "373.22", "373.22", "374.39", "374.30", "373.09", "372.05", "372.96" ], 
                                            [ "372.05", "N/A", "373.22", "370.90", "372.05", "373.22", "372.05", "371.92" ]
                                        ],
                                        "sum" : 20841.13,
                                        "min" : 370.90,
                                        "max" : 373.22,
                                        "average" : 371.92
                                    }
                                ]
                            }
                        }
                    )";

    auto const result = DcgmNs::JsonSerialize::Deserialize<NVBandwidthResult>(std::string_view { ss, ss + strlen(ss) });
    constexpr float FLT_EPSILON { 0.000001 };

    REQUIRE(result.cudaRuntimeVersion == 12030);
    REQUIRE(result.driverVersion == "535.40");
    REQUIRE(result.gitVersion == "2.34.1");
    REQUIRE(result.overallError == "invalid arguments");
    REQUIRE(result.warning == "smaller buffer size warning");
    auto const &testcase0 = result.testCases[0];
    REQUIRE(testcase0.name == "valid test");
    REQUIRE(testcase0.status == TestCaseStatus::PASSED);
    REQUIRE(testcase0.bandwidthDescription == "memcpy CE GPU(row) -> GPU(column) bandwidth GB/s");
    REQUIRE(fabsf(testcase0.sum - 20841.13) < FLT_EPSILON * 20841.13);
    REQUIRE(fabsf(testcase0.min - 370.90) < FLT_EPSILON * 370.90);
    REQUIRE(fabsf(testcase0.max - 373.22) < FLT_EPSILON * 373.22);
    REQUIRE(fabsf(testcase0.avg - 371.92) < FLT_EPSILON * 371.92);

    std::vector<std::vector<float>> groundTruth = { { NAN, 373.22, 373.22, 374.39, 374.30, 373.09, 372.05, 372.96 },
                                                    { 372.05, NAN, 373.22, 370.90, 372.05, 373.22, 372.05, 371.92 } };

    auto const &matrixData = testcase0.bandwidthMatrix.data;
    REQUIRE(matrixData.size() == groundTruth.size());

    for (size_t jj = 0; jj < groundTruth.size(); jj++)
    {
        REQUIRE(matrixData[jj].size() == groundTruth[jj].size());
        for (size_t ii = 0; ii < groundTruth[jj].size(); ii++)
        {
            if (isnanf(groundTruth[jj][ii]))
            {
                REQUIRE(isnanf(matrixData[jj][ii]));
            }
            else
            {
                REQUIRE(fabsf(matrixData[jj][ii] - groundTruth[jj][ii]) < FLT_EPSILON * groundTruth[jj][ii]);
            }
        }
    }
}