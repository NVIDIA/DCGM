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

#include <json/json.h>
#include <string>
#include <vector>

#define BWC_JSON_GPUS         "GPUs"
#define BWC_JSON_GPU_ID       "gpuId"
#define BWC_JSON_MAX_RX_BW    "maxRxBw"
#define BWC_JSON_MAX_TX_BW    "maxTxBw"
#define BWC_JSON_MAX_BIDIR_BW "maxBidirBw"
#define BWC_JSON_ERROR        "error"
#define BWC_JSON_ERRORS       "errors"

/*
 * Output is written in JSON in this format:
 *
 * {
 *      "errors" : [ <global error 1>, <global error 2> ... ], # optional, will not exist if not present
 *      "GPUs" : [
 *          {
 *              "gpuId" : <gpuId0>,
 *              "maxRxBw" : <maximum bandwidth1>,
 *              "maxTxBw" : <maximum bandwidth1>,
 *              "maxBidirBw" : <maximum bandwidth1>,
 *              "error" : "error string", # This will not exist if no error occurred.
 *          },
 *          {
 *              "gpuId" : <gpuId1>,
 *              "maxRxBw" : <maximum bandwidth1>,
 *              "maxTxBw" : <maximum bandwidth1>,
 *              "maxBidirBw" : <maximum bandwidth1>,
 *              "error" : "error string", # This will not exist if no error occurred.
 *          }
 *          ...
 *      ]
 * }
 *
 */

typedef enum dcgmBandwidth_enum
{
    H2D             = 0,
    D2H             = 1,
    BIDIR           = 2,
    BANDWIDTH_COUNT = 3
} dcgmBandwidth_t;


typedef struct
{
    unsigned int dcgmGpuId;
    std::string pciBusId;
    double bandwidths[BANDWIDTH_COUNT];
    std::string error = "";
} bwTestResult_t;

typedef struct
{
    unsigned int gpuId;
    std::string pciBusId;
} dcgmGpuPciIdPair_t;

void AppendResultsToJson(std::vector<bwTestResult_t> &results, Json::Value &root);
