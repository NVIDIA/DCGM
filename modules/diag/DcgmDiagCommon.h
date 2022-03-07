/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
Common helper functions and classes relating to DCGM GPU Diagnostics
*/

#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <string>

/*****************************************************************************/
dcgmReturn_t dcgm_diag_common_populate_run_diag(dcgmRunDiag_t &drd,
                                                const std::string &testNames,
                                                const std::string &parms,
                                                const std::string &configFileContents,
                                                const std::string &fakeGpuList,
                                                const std::string &gpuList,
                                                bool verbose,
                                                bool statsOnFail,
                                                const std::string &debugLogFile,
                                                const std::string &statsPath,
                                                unsigned int debugLevel,
                                                const std::string &throttleMask,
                                                bool training,
                                                bool forceTrain,
                                                unsigned int trainingIterations,
                                                unsigned int trainingVariance,
                                                unsigned int trainingTolerance,
                                                const std::string &goldenValuesFile,
                                                unsigned int groupId,
                                                bool failEarly,
                                                unsigned int failCheckInterval,
                                                std::string &error);

/*****************************************************************************/
void dcgm_diag_common_set_config_file_contents(const std::string &configFileContents, dcgmRunDiag_t &drd);

/*****************************************************************************/
