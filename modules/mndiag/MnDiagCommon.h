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


/*
Common helper functions and classes relating to DCGM Multi Node GPU Diagnostics
*/

#include <dcgm_structs.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

typedef struct
{
    std::string dcgmVersion;
    std::string driverVersion;
} nodeInfo_t;

typedef std::unordered_map<std::string, nodeInfo_t> nodeInfoMap_t;

/*****************************************************************************/
void validate_parameters(std::vector<std::string> const &hostListVec, std::string_view runValue);

/*****************************************************************************/
dcgmReturn_t dcgm_mn_diag_common_populate_run_mndiag(dcgmRunMnDiag_v1 &mndrd,
                                                     std::vector<std::string> const &hostList,
                                                     std::string_view parameters,
                                                     std::string_view runValue);

/*****************************************************************************/
