/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "BwCheckerMain.h"

void AppendResultsToJson(std::vector<bwTestResult_t> &results, Json::Value &root)
{
    for (unsigned int i = 0; i < results.size(); i++)
    {
        Json::Value gpu;
        if (results[i].error.empty() == false)
        {
            gpu[BWC_JSON_ERROR] = results[i].error;
        }
        gpu[BWC_JSON_MAX_RX_BW]    = results[i].bandwidths[H2D];
        gpu[BWC_JSON_MAX_TX_BW]    = results[i].bandwidths[D2H];
        gpu[BWC_JSON_MAX_BIDIR_BW] = results[i].bandwidths[BIDIR];
        gpu[BWC_JSON_GPU_ID]       = results[i].dcgmGpuId;
        root[BWC_JSON_GPUS][i]     = gpu;
    }
}
