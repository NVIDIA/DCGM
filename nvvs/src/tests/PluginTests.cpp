/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <stdlib.h>

#include <DcgmError.h>
#include <Plugin.h>
#include <dcgm_structs.h>

class UnitTestPlugin : public Plugin
{
    void Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters) override
    {}
};

TEST_CASE("Plugin Results Reporting")
{
    UnitTestPlugin p;
    dcgmDiagResults_t results {};

    DcgmError d { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVLINK_DOWN, d, 0, 1);
    p.AddError(d);
    memset(&results, 0, sizeof(results));

    CHECK(p.GetResults(nullptr) == DCGM_ST_BADPARAM);
    CHECK(p.GetResults(&results) == DCGM_ST_OK);
    CHECK(results.numErrors == 1);
    CHECK(results.errors[0].gpuId == -1);

    DcgmError d1 { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d1, 10000, 0, 95);
    const unsigned int GPU_ID = 0;
    p.AddErrorForGpu(GPU_ID, d1);
    memset(&results, 0, sizeof(results));

    CHECK(p.GetResults(&results) == DCGM_ST_OK);
    CHECK(results.numErrors == 2); // it will still have the first error
    CHECK(results.errors[1].gpuId == GPU_ID);

    DcgmError d2 { 0 };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d2, 10000, 0, 95);
    p.AddErrorForGpu(0, d2);
    memset(&results, 0, sizeof(results));

    CHECK(p.GetResults(&results) == DCGM_ST_OK);
    CHECK(results.numErrors == 3); // it'll still have the first two errors
    CHECK(results.errors[2].gpuId == GPU_ID);
}
