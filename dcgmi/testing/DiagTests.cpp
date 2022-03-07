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
#include <catch2/catch.hpp>

#include <Diag.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

SCENARIO("Diag::GetFailureResult")
{
    Diag d;
    dcgmDiagResponse_t result {};

    // Initialized to all zeros should pass
    CHECK(d.GetFailureResult(result) == DCGM_ST_OK);

    result.levelOneTestCount             = 5;
    result.levelOneResults[4].status     = DCGM_DIAG_RESULT_FAIL;
    result.levelOneResults[4].error.code = DCGM_FR_VOLATILE_SBE_DETECTED;
    CHECK(d.GetFailureResult(result) == DCGM_ST_NVVS_ERROR);

    // Make sure that a subsequent error will return the worse failure
    result.perGpuResponses[0].results[0].status     = DCGM_DIAG_RESULT_FAIL;
    result.perGpuResponses[0].results[0].error.code = DCGM_FR_VOLATILE_DBE_DETECTED;
    CHECK(d.GetFailureResult(result) == DCGM_ST_NVVS_ISOLATE_ERROR);
}
