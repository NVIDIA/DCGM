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

#include "DcgmGroupEntityPairHelpers.h"
#include "DcgmStringHelpers.h"

#include <exception>
#include <map>
#include <string>
#include <unordered_set>

using gpuIgnoreErrorCodeMap_t = std::map<dcgmGroupEntityPair_t, std::unordered_set<unsigned int>>;

/**
 * Verify the syntax of the ignoreErrorCodes input string, and populate the GPU error code map with
 * the error codes to be ignored on each GPU. The format of the input string is as follows:
 *   "28,140" (ignore error codes 28 and 140 on all GPUs)
 *   "gpu0:28;gpu1:140,92" (ignore error 28 on GPU 0, and errors 140 and 92 on GPU 1)
 *   "*:*" (ignore all errors that can be ignored on all GPUs)
 *
 * @param[in] inputString - validate this string
 * @param[out] gpuErrorCodeMap - map of gpu and error codes set to be ignored on each gpu
 * @param[in] validGpuIndices - only this list of indices are allowed in the input string
 * @param[in] validErrorCodes - if provided, only these error codes are allowed in the input string,
 *                              a default set of error codes is used otherwise
 * @param[in] entitySeparator - if provided, this character is used to split the input string by gpu
 *                              id; by default, ';' is used as the gpu separator
 *
 * @return   "" on success
 *           an error string on failure
 */
std::string ParseIgnoreErrorCodesString(std::string_view inputString,
                                        gpuIgnoreErrorCodeMap_t &gpuErrorCodeMap,
                                        std::vector<unsigned int> const &validGpuIndices,
                                        std::optional<std::unordered_set<unsigned int>> const &validErrorCodes
                                        = std::nullopt,
                                        std::optional<char> entitySeparator = std::nullopt);
