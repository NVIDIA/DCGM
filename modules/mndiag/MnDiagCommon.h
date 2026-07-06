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

#include "dcgm_mndiag_structs.hpp"
#include <dcgm_structs.h>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
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
dcgmReturn_t infer_mnubergemm_default_path(std::string &mnubergemm_path, int cudaVersion);

/*****************************************************************************/
std::expected<std::filesystem::path, std::system_error> get_executable_path();

/*****************************************************************************/
std::expected<std::string, std::system_error> get_mnubergemm_binary_path(int cudaVersion,
                                                                         auto const &supportedCudaVersions);

/**
 * @brief Parse the time_to_run value from a testParms array.
 *
 * Scans params.testParms for an entry whose text is exactly `key=<value>`,
 * anchored at the start, and returns the validated integer value in seconds.
 * Semicolons are treated as delimiters: only the text between '=' and the
 * first ';' (if any) is interpreted as the value.
 *
 * @param params  The run parameters struct whose testParms array is scanned.
 * @param key     Full parameter key to match (e.g. "mnubergemm.time_to_run").
 * @return On success: 0 if the key is absent, or a positive integer (seconds) if found and valid.
 *         On error: std::unexpected(DCGM_ST_BADPARAM) if the entry is
 *         present but malformed or has a non-positive value.
 */
std::expected<int, dcgmReturn_t> ParseTimeToRunSeconds(dcgmRunMnDiag_t const &params, std::string_view key);
