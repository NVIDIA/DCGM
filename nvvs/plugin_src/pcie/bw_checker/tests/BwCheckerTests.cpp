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
#include <catch2/catch_all.hpp>
#include <json/json.h>

#include <BwCheckerMain.h>

TEST_CASE("BwChecker: AppendResultsToJson")
{
    std::string error1("Really bad error Scoob");
    std::string error2("Just kind of bad but still an error Scoob");

    const unsigned int TEST_GPU_COUNT = 4;

    double maxRxBw[TEST_GPU_COUNT];
    double maxTxBw[TEST_GPU_COUNT];
    double maxBidirBw[TEST_GPU_COUNT];
    std::vector<bwTestResult_t> results(TEST_GPU_COUNT);

    for (unsigned int gpuId = 0; gpuId < TEST_GPU_COUNT; gpuId++)
    {
        maxRxBw[gpuId]    = 400 + gpuId;
        maxTxBw[gpuId]    = 300 + gpuId;
        maxBidirBw[gpuId] = 150 + gpuId;

        results[gpuId].dcgmGpuId         = gpuId;
        results[gpuId].bandwidths[H2D]   = maxRxBw[gpuId];
        results[gpuId].bandwidths[D2H]   = maxTxBw[gpuId];
        results[gpuId].bandwidths[BIDIR] = maxBidirBw[gpuId];
    }

    Json::Value root0;
    AppendResultsToJson(results, root0);
    REQUIRE(root0[BWC_JSON_GPUS].isArray());
    for (unsigned int gpuId = 0; gpuId < TEST_GPU_COUNT; gpuId++)
    {
        CHECK(root0[BWC_JSON_GPUS][gpuId][BWC_JSON_GPU_ID].asUInt() == gpuId);
        CHECK(root0[BWC_JSON_GPUS][gpuId][BWC_JSON_MAX_RX_BW].asDouble() == maxRxBw[gpuId]);
        CHECK(root0[BWC_JSON_GPUS][gpuId][BWC_JSON_MAX_TX_BW].asDouble() == maxTxBw[gpuId]);
        CHECK(root0[BWC_JSON_GPUS][gpuId][BWC_JSON_MAX_BIDIR_BW].asDouble() == maxBidirBw[gpuId]);
        CHECK(root0[BWC_JSON_GPUS][gpuId][BWC_JSON_ERROR].asString() == "");
    }

    results[0].error = error1;
    results[3].error = error2;
    Json::Value root1;
    AppendResultsToJson(results, root1);
    REQUIRE(root1[BWC_JSON_GPUS].isArray());
    for (unsigned int gpuId = 0; gpuId < TEST_GPU_COUNT; gpuId++)
    {
        CHECK(root1[BWC_JSON_GPUS][gpuId][BWC_JSON_GPU_ID].asUInt() == gpuId);
        CHECK(root1[BWC_JSON_GPUS][gpuId][BWC_JSON_MAX_TX_BW].asDouble() == maxTxBw[gpuId]);
        CHECK(root1[BWC_JSON_GPUS][gpuId][BWC_JSON_MAX_RX_BW].asDouble() == maxRxBw[gpuId]);
        CHECK(root1[BWC_JSON_GPUS][gpuId][BWC_JSON_MAX_BIDIR_BW].asDouble() == maxBidirBw[gpuId]);
    }

    CHECK(root1[BWC_JSON_GPUS][0][BWC_JSON_ERROR].asString() == error1);
    CHECK(root1[BWC_JSON_GPUS][1][BWC_JSON_ERROR].asString().empty());
    CHECK(root1[BWC_JSON_GPUS][2][BWC_JSON_ERROR].asString().empty());
    CHECK(root1[BWC_JSON_GPUS][3][BWC_JSON_ERROR].asString() == error2);
}
