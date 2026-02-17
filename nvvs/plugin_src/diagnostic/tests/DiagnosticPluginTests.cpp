/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "dcgm_fields.h"
#include <catch2/catch_all.hpp>

#include <DiagnosticPlugin.h>
#include <PluginInterface.h>

char DcgmFieldGetType(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_POWER_USAGE:
            return DCGM_FT_DOUBLE;
        case DCGM_FI_DEV_SM_CLOCK:
            return DCGM_FT_INT64;
        default:
            return ' ';
    }

    return ' ';
}

std::unique_ptr<dcgmDiagPluginEntityList_v1> GetEntityList(unsigned int gpuCount, DcgmEntityStatus_t status)
{
    auto pEntityInfo                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityInfo = *pEntityInfo;

    entityInfo.numEntities = gpuCount;
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        entityInfo.entities[i].entity.entityId      = i;
        entityInfo.entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityInfo.entities[i].auxField.gpu.status  = status;
    }

    return pEntityInfo;
}

void InitializeDiagPlugin(GpuBurnPlugin &gbp, unsigned int gpuCount, DcgmEntityStatus_t status)
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = GetEntityList(gpuCount, status);
    gbp.InitializeForEntityList(gbp.GetDiagnosticTestName(), *entityList);
}

TEST_CASE("Diagnostic: SetPrecisionFromString")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = GetEntityList(1, DcgmEntityStatusOk);

    GpuBurnPlugin gbp((dcgmHandle_t)0); // we don't need real values for these
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
    gbp.Go(gbp.GetDiagnosticTestName(), entityList.get(), 2, params); // Set the parameters

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
    gbp.Go(gbp.GetDiagnosticTestName(), entityList.get(), 2, params); // Set the parameters
    precision = gbp.SetPrecisionFromString(true);
    CHECK(USE_HALF_PRECISION(precision) != 0);
    CHECK(USE_SINGLE_PRECISION(precision) != 0);
    CHECK(USE_DOUBLE_PRECISION(precision) != 0);

    // Make sure an invalid string also results in the default settings
    snprintf(params[0].parameterValue, sizeof(params[0].parameterValue), "bobby");
    gbp.Go(gbp.GetDiagnosticTestName(), entityList.get(), 2, params); // Set the parameters
    precision = gbp.SetPrecisionFromString(true);
    CHECK(USE_HALF_PRECISION(precision) != 0);
    CHECK(USE_SINGLE_PRECISION(precision) != 0);
    CHECK(USE_DOUBLE_PRECISION(precision) != 0);
}

TEST_CASE("Diagnostic: GFLOPS Threshold Violation")
{
    SECTION("Only one GPU")
    {
        GpuBurnPlugin gbp((dcgmHandle_t)0); // we don't need real values for these

        std::vector<double> gflops = { 600.0 };
        double minThresh           = GpuBurnPluginTester::GetMinThreshold(gbp, gflops, 0.05);
        auto indicesBelowThreshold = GpuBurnPluginTester::GetIndicesBelowMinThreshold(gbp, gflops, minThresh);
        CHECK(indicesBelowThreshold.size() == 0);
    }

    GpuBurnPlugin gbp((dcgmHandle_t)0); // we don't need real values for these

    SECTION("Value below threshold")
    {
        std::vector<double> gflops = { 600.0, 598.0, 600.0, 550.0 };
        double minThresh           = GpuBurnPluginTester::GetMinThreshold(gbp, gflops, 0.05);
        auto indicesBelowThreshold = GpuBurnPluginTester::GetIndicesBelowMinThreshold(gbp, gflops, minThresh);
        CHECK(indicesBelowThreshold.size() > 0);
        CHECK(indicesBelowThreshold == std::vector<size_t> { 3 });
    }

    SECTION("All values within threshold")
    {
        std::vector<double> gflops = { 600.0, 598.0, 600.0, 550.0 };
        double minThresh           = GpuBurnPluginTester::GetMinThreshold(gbp, gflops, 0.1);
        auto indicesBelowThreshold = GpuBurnPluginTester::GetIndicesBelowMinThreshold(gbp, gflops, minThresh);
        CHECK(indicesBelowThreshold.size() == 0);
    }

    SECTION("Multiple values below threshold")
    {
        std::vector<double> gflops = { 600.0, 1.0, 300.0, 550.0 };
        double minThresh           = GpuBurnPluginTester::GetMinThreshold(gbp, gflops, 0.01);
        auto indicesBelowThreshold = GpuBurnPluginTester::GetIndicesBelowMinThreshold(gbp, gflops, minThresh);
        CHECK(indicesBelowThreshold.size() > 0);
        CHECK(indicesBelowThreshold == std::vector<size_t> { 1, 2 });
    }
}

TEST_CASE("Diagnostic: GpuBurnDevice parameters")
{
    GpuBurnPlugin gbp((dcgmHandle_t)0); // we don't need real values for these
    InitializeDiagPlugin(gbp, 1, DcgmEntityStatusOk);

    GpuBurnDevice gbd;
    DcgmRecorder dr;
    GpuBurnWorker gbw(&gbd, gbp, DIAG_SINGLE_PRECISION, 15.0, 2048, dr, false, 1001);
    CHECK(GpuBurnWorkerTester::GetNElemsPerIter(gbw) == (2048 * 2048));
}

TEST_CASE("Diagnostic: Calculate GFlops Multiplier")
{
    int matrixSizes[]     = { 2048, 4096, 6144, 8192, 0 };
    double baseMultiplier = CalculateGFlopsMultiplier(matrixSizes[0]);
    for (unsigned int i = 0; matrixSizes[i] != 0; i++)
    {
        double multiplier = CalculateGFlopsMultiplier(matrixSizes[i]);
        CHECK(abs(multiplier - (baseMultiplier * pow(matrixSizes[i] / 2048.0, 3))) < 1.0);
    }
}

TEST_CASE("Diagnostic: CheckAveragesForExcessiveVariation")
{
    GpuBurnPlugin gbp((dcgmHandle_t)0); // we don't need real values for these
    InitializeDiagPlugin(gbp, 4, DcgmEntityStatusOk);

    std::vector<unsigned long> clockAvgsGood = { 900, 900, 900, 900 };
    std::vector<unsigned long> clockAvgsBad  = { 900, 500, 900, 900 };
    double tolerance                         = 0.1;
    double minThresh                         = GpuBurnPluginTester::GetMinThreshold(gbp, clockAvgsGood, tolerance);
    auto indices = GpuBurnPluginTester::GetIndicesBelowMinThreshold(gbp, clockAvgsGood, minThresh);
    CHECK(indices.size() == 0);
    minThresh = GpuBurnPluginTester::GetMinThreshold(gbp, clockAvgsBad, tolerance);
    indices   = GpuBurnPluginTester::GetIndicesBelowMinThreshold(gbp, clockAvgsBad, minThresh);
    REQUIRE(indices.size() == 1);
    CHECK(indices[0] == 1);

    std::vector<double> powerAvgsGood = { 1350.0, 1340.0, 1400.0, 1370.0 };
    std::vector<double> powerAvgsBad  = { 1350.0, 1340.0, 1400.0, 870.0 };

    minThresh = GpuBurnPluginTester::GetMinThreshold(gbp, powerAvgsGood, tolerance);
    indices   = GpuBurnPluginTester::GetIndicesBelowMinThreshold(gbp, powerAvgsGood, minThresh);
    REQUIRE(indices.size() == 0);
    minThresh = GpuBurnPluginTester::GetMinThreshold(gbp, powerAvgsBad, tolerance);
    indices   = GpuBurnPluginTester::GetIndicesBelowMinThreshold(gbp, powerAvgsBad, minThresh);
    REQUIRE(indices.size() == 1);
    CHECK(indices[0] == 3);
}
