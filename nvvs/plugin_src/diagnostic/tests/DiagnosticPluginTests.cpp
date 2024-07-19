/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <DiagnosticPlugin.h>
#include <PluginInterface.h>

TEST_CASE("Diagnostic: SetPrecisionFromString")
{
    dcgmDiagPluginGpuList_t gpuList = {};

    gpuList.numGpus        = 1;
    gpuList.gpus[0].gpuId  = 0;
    gpuList.gpus[0].status = DcgmEntityStatusOk;


    GpuBurnPlugin gbp((dcgmHandle_t)0, &gpuList); // we don't need real values for these
    // Make sure the correct defaults are set - it has no test parameters so far
    int32_t precision = gbp.SetPrecisionFromString(true);
    CHECK(USE_HALF_PRECISION(precision) != 0);
    CHECK(USE_SINGLE_PRECISION(precision) != 0);
    CHECK(USE_DOUBLE_PRECISION(precision) != 0);

    precision = gbp.SetPrecisionFromString(false);
    CHECK(USE_HALF_PRECISION(precision) != 0);
    CHECK(USE_SINGLE_PRECISION(precision) != 0);
    CHECK(USE_DOUBLE_PRECISION(precision) == 0);

    dcgmDiagPluginTestParameter_t params[2];
    snprintf(params[0].parameterName, sizeof(params[0].parameterName), "%s", DIAGNOSTIC_STR_PRECISION);
    snprintf(params[0].parameterValue, sizeof(params[0].parameterValue), "h");
    params[0].type = DcgmPluginParamString;
    snprintf(params[1].parameterName, sizeof(params[1].parameterName), "%s", DIAGNOSTIC_STR_IS_ALLOWED);
    snprintf(params[1].parameterValue, sizeof(params[1].parameterValue), "false");
    params[1].type = DcgmPluginParamBool;
    gbp.Go("test", 2, params); // Set the parameters

    // Make sure we get only half precision
    precision = gbp.SetPrecisionFromString(false);
    CHECK(USE_HALF_PRECISION(precision) != 0);
    CHECK(USE_SINGLE_PRECISION(precision) == 0);
    CHECK(USE_DOUBLE_PRECISION(precision) == 0);

    precision = gbp.SetPrecisionFromString(true); // this input is ignored when the parameter is set
    CHECK(USE_HALF_PRECISION(precision) != 0);
    CHECK(USE_SINGLE_PRECISION(precision) == 0);
    CHECK(USE_DOUBLE_PRECISION(precision) == 0);

    // Make sure we get the default value if it's an empty string
    memset(params[0].parameterValue, 0, sizeof(params[0].parameterValue));
    gbp.Go("test", 2, params); // Set the parameters
    precision = gbp.SetPrecisionFromString(true);
    CHECK(USE_HALF_PRECISION(precision) != 0);
    CHECK(USE_SINGLE_PRECISION(precision) != 0);
    CHECK(USE_DOUBLE_PRECISION(precision) != 0);

    // Make sure an invalid string also results in the default settings
    snprintf(params[0].parameterValue, sizeof(params[0].parameterValue), "bobby");
    gbp.Go("test", 2, params); // Set the parameters
    precision = gbp.SetPrecisionFromString(true);
    CHECK(USE_HALF_PRECISION(precision) != 0);
    CHECK(USE_SINGLE_PRECISION(precision) != 0);
    CHECK(USE_DOUBLE_PRECISION(precision) != 0);
}

TEST_CASE("Diagnostic: GFLOPS Threshold Violation")
{
    SECTION("Only one GPU")
    {
        // Test with one GPU ...
        dcgmDiagPluginGpuList_t gpuList = {};
        gpuList.numGpus                 = 1;
        gpuList.gpus[0].gpuId           = 0;
        gpuList.gpus[0].status          = DcgmEntityStatusOk;
        GpuBurnPlugin gbp((dcgmHandle_t)0, &gpuList); // we don't need real values for these

        std::vector<double> gflops = { 600.0 };
        double minThresh           = GpuBurnPluginTester::GetGflopsMinThreshold(gbp, gflops, 0.05);
        auto indicesBelowThreshold = GpuBurnPluginTester::GetGflopsBelowMinThreshold(gbp, gflops, minThresh);
        CHECK(indicesBelowThreshold.size() == 0);
    }

    // Populate the gpuList to be passed
    dcgmDiagPluginGpuList_t gpuList = {};
    gpuList.numGpus                 = 4;

    for (size_t i = 0; i < gpuList.numGpus; i++)
    {
        gpuList.gpus[i].gpuId  = i;
        gpuList.gpus[i].status = DcgmEntityStatusOk;
    }
    GpuBurnPlugin gbp((dcgmHandle_t)0, &gpuList); // we don't need real values for these

    SECTION("Value below threshold")
    {
        std::vector<double> gflops = { 600.0, 598.0, 600.0, 550.0 };
        double minThresh           = GpuBurnPluginTester::GetGflopsMinThreshold(gbp, gflops, 0.05);
        auto indicesBelowThreshold = GpuBurnPluginTester::GetGflopsBelowMinThreshold(gbp, gflops, minThresh);
        CHECK(indicesBelowThreshold.size() > 0);
        CHECK(indicesBelowThreshold == std::vector<size_t> { 3 });
    }

    SECTION("All values within threshold")
    {
        std::vector<double> gflops = { 600.0, 598.0, 600.0, 550.0 };
        double minThresh           = GpuBurnPluginTester::GetGflopsMinThreshold(gbp, gflops, 0.1);
        auto indicesBelowThreshold = GpuBurnPluginTester::GetGflopsBelowMinThreshold(gbp, gflops, minThresh);
        CHECK(indicesBelowThreshold.size() == 0);
    }

    SECTION("Multiple values below threshold")
    {
        std::vector<double> gflops = { 600.0, 1.0, 300.0, 550.0 };
        double minThresh           = GpuBurnPluginTester::GetGflopsMinThreshold(gbp, gflops, 0.01);
        auto indicesBelowThreshold = GpuBurnPluginTester::GetGflopsBelowMinThreshold(gbp, gflops, minThresh);
        CHECK(indicesBelowThreshold.size() > 0);
        CHECK(indicesBelowThreshold == std::vector<size_t> { 1, 2 });
    }
}

TEST_CASE("Diagnostic: GpuBurnDevice parameters")
{
    // Test with one GPU ...
    dcgmDiagPluginGpuList_t gpuList = {};
    gpuList.numGpus                 = 1;
    gpuList.gpus[0].gpuId           = 0;
    gpuList.gpus[0].status          = DcgmEntityStatusOk;
    GpuBurnPlugin gbp((dcgmHandle_t)0, &gpuList); // we don't need real values for these
    GpuBurnDevice gbd;
    DcgmRecorder dr;
    GpuBurnWorker gbw(&gbd, gbp, DIAG_SINGLE_PRECISION, 15.0, 2048, dr, false, 1001);
    CHECK(GpuBurnWorkerTester::GetNElemsPerIter(gbw) == (2048 * 2048));
}