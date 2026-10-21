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

#include <DcgmOpsAnyUniquePtr.hpp>
#include <DcgmStringHelpers.h>
#include <Defer.hpp>
#include <DiagnosticPlugin.h>
#include <MockCublasOps.h>
#include <MockCudaCommon.h>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <ScopedAtomic.hpp>
#include <UniquePtrUtil.h>

#include <stdexcept>
#include <string_view>
#include <utility>

class TestGpuBurnPlugin : public GpuBurnPlugin
{
public:
    using GpuBurnPlugin::GpuBurnPlugin;

    using GpuBurnPlugin::CheckAveragesForExcessiveVariation;
    using GpuBurnPlugin::CheckToleranceForField;
    using GpuBurnPlugin::CheckVariationFailures;
    using GpuBurnPlugin::CorrectTolerancePercentageIfWrong;
    using GpuBurnPlugin::GetIndicesBelowMinThreshold;
    using GpuBurnPlugin::GetMinThreshold;
    using GpuBurnPlugin::Init;
    using GpuBurnPlugin::SetToleranceIfUnset;
    using GpuBurnPlugin::UpdateForDGemmSupport;
    using GpuBurnPlugin::UpdateForHGemmSupport;

    bool CheckPassFail(std::vector<long long> const &errorCount, std::vector<long long> const &nanCount)
    {
        return GpuBurnPlugin::CheckPassFail(errorCount, nanCount);
    }

    std::vector<GpuBurnDevice *> &GetDeviceList()
    {
        return m_device;
    }

    int32_t GetPrecision() const
    {
        return m_precision;
    }

    void SetPrecision(int32_t precision)
    {
        m_precision = precision;
    }

    void SetGflopsTolerance(double tolerancePcnt)
    {
        m_gflopsTolerancePcnt = tolerancePcnt;
    }

    void SetExplicitTests(bool explicitTests)
    {
        m_explicitTests = explicitTests;
    }
};

namespace
{
struct DiagnosticPluginTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    std::unique_ptr<MockCudaOps> mockCudaOpsUnique;
    std::unique_ptr<MockCublasOps> mockCublasOpsUnique;
    MockCudaOps *mockCudaOps;
    MockCublasOps *mockCublasOps;
    MockCudaCommon mockCudaCommon;
    CudaCommonAnyPtr mockCudaCommonPtr { &mockCudaCommon };
    NvvsCommon nvvsCommon;
    TestGpuBurnPlugin gbp;

    DiagnosticPluginTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , mockCudaOpsUnique(std::make_unique<MockCudaOps>())
        , mockCublasOpsUnique(std::make_unique<MockCublasOps>())
        , mockCudaOps(mockCudaOpsUnique.get())
        , mockCublasOps(mockCublasOpsUnique.get())
        , gbp((dcgmHandle_t)0,
              std::move(mockDcgmOpsUnique),
              std::move(mockCudaOpsUnique),
              std::move(mockCublasOpsUnique),
              std::make_unique<MockCudaCommon>(),
              &nvvsCommon)
    {}
};
} // namespace


char DcgmFieldGetType(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_BOARD_POWER_WATTS:
            return DCGM_FT_DOUBLE;
        case DCGM_FI_DEV_SM_CLOCK:
            return DCGM_FT_INT64;
        default:
            return ' ';
    }

    return ' ';
}


dcgmDiagPluginTestParameter_t MakeDiagnosticStringParameter(std::string_view name,
                                                            std::string_view value,
                                                            dcgmPluginValue_t type = DcgmPluginParamString)
{
    dcgmDiagPluginTestParameter_t param {};
    snprintf(param.parameterName, sizeof(param.parameterName), "%.*s", static_cast<int>(name.size()), name.data());
    snprintf(param.parameterValue, sizeof(param.parameterValue), "%.*s", static_cast<int>(value.size()), value.data());
    param.type = type;
    return param;
}

std::unique_ptr<dcgmDiagPluginEntityList_v1> GetEntityList(unsigned int gpuCount, DcgmEntityStatus_t status)
{
    auto pEntityInfo                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityInfo = *pEntityInfo;

    entityInfo.numEntities = gpuCount;
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        entityInfo.entities[i].entity.entityId                                 = i;
        entityInfo.entities[i].entity.entityGroupId                            = DCGM_FE_GPU;
        entityInfo.entities[i].auxField.gpu.status                             = status;
        entityInfo.entities[i].auxField.gpu.attributes.identifiers.pciDeviceId = (status == DcgmEntityStatusOk) ? 1 : 0;
    }

    return pEntityInfo;
}

void InitializeDiagPlugin(TestGpuBurnPlugin &gbp, unsigned int gpuCount, DcgmEntityStatus_t status)
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = GetEntityList(gpuCount, status);
    gbp.InitializeForEntityList(gbp.GetDiagnosticTestName(), *entityList);
}


bool CheckPassFailWithDevices(TestGpuBurnPlugin &gbp,
                              DcgmNs::DcgmOpsAnyPtr dcgmOps,
                              CudaOpsAnyPtr cudaOps,
                              CublasOpsAnyPtr cublasOps,
                              std::vector<unsigned int> const &gpuIds,
                              std::vector<long long> const &errorCount,
                              std::vector<long long> const &nanCount)
{
    auto &devices = gbp.GetDeviceList();
    std::vector<std::unique_ptr<GpuBurnDevice>> burnDevices;
    burnDevices.reserve(gpuIds.size());
    for (auto gpuId : gpuIds)
    {
        burnDevices.push_back(std::make_unique<GpuBurnDevice>(dcgmOps, cudaOps, cublasOps));
        devices.push_back(burnDevices.back().get());
        burnDevices.back()->gpuId = gpuId;
    }

    DcgmNs::Defer cleanup([&devices] { devices.clear(); });
    bool result = gbp.CheckPassFail(errorCount, nanCount);
    return result;
}


SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic Init handles entity filtering and CUDA runtime failures")
{
    GIVEN("a diagnostic plugin and injected CUDA runtime operations")
    {
        WHEN("Init receives only a non-GPU entity")
        {
            dcgmDiagPluginEntityList_v1 entityList {};
            entityList.numEntities                      = 1;
            entityList.entities[0].entity.entityGroupId = DCGM_FE_SWITCH;
            entityList.entities[0].entity.entityId      = 7;
            bool const initialized                      = gbp.Init(entityList);

            THEN("the plugin initializes without creating diagnostic devices")
            {
                CHECK(initialized);
                CHECK(gbp.GetDeviceList().empty());
            }
        }

        WHEN("CUDA attribute probing fails while pruning half precision support")
        {
            gbp.SetPrecision(DIAG_HALF_PRECISION | DIAG_SINGLE_PRECISION);
            mockCudaOps->cudaDeviceGetAttributeResult = cudaErrorInvalidValue;

            gbp.UpdateForHGemmSupport(0);

            THEN("the plugin preserves the requested precision rather than pruning on missing CUDA data")
            {
                int32_t const precision = gbp.GetPrecision();
                CHECK(USE_HALF_PRECISION(precision));
                CHECK(USE_SINGLE_PRECISION(precision));
            }
        }

        WHEN("CUDA attribute probing fails while checking double precision support")
        {
            gbp.SetPrecision(DIAG_SINGLE_PRECISION | DIAG_DOUBLE_PRECISION);
            gbp.SetExplicitTests(false);
            mockCudaOps->cudaDeviceGetAttributeResult = cudaErrorInvalidValue;

            gbp.UpdateForDGemmSupport(0);

            THEN("the plugin leaves the existing precision mask unchanged")
            {
                int32_t const precision = gbp.GetPrecision();
                CHECK(USE_SINGLE_PRECISION(precision));
                CHECK(USE_DOUBLE_PRECISION(precision));
            }
        }
    }
}

SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic: Go handles unknown name, null entities, and non-GPU lists")
{
    GIVEN("a diagnostic plugin")
    {
        WHEN("Go is called with an unknown test name")
        {
            auto entityList = GetEntityList(1, DcgmEntityStatusOk);
            gbp.Go("not_diagnostic", entityList.get(), 0, nullptr);

            THEN("the diagnostic test is not initialized")
            {
                CHECK_THROWS_AS(gbp.GetGpuResults(gbp.GetDiagnosticTestName()), std::out_of_range);
            }
        }

        WHEN("Go is called without entity info")
        {
            gbp.Go(gbp.GetDiagnosticTestName(), nullptr, 0, nullptr);

            THEN("the diagnostic test is not initialized")
            {
                CHECK_THROWS_AS(gbp.GetGpuResults(gbp.GetDiagnosticTestName()), std::out_of_range);
            }
        }

        WHEN("Go is called with only non-GPU entities")
        {
            dcgmDiagPluginEntityList_v1 entityList {};
            entityList.numEntities                      = 2;
            entityList.entities[0].entity.entityGroupId = DCGM_FE_SWITCH;
            entityList.entities[0].entity.entityId      = 11;
            entityList.entities[1].entity.entityGroupId = DCGM_FE_SWITCH;
            entityList.entities[1].entity.entityId      = 12;
            dcgmDiagPluginTestParameter_t params[]      = {
                MakeDiagnosticStringParameter(DIAGNOSTIC_STR_IS_ALLOWED, "true", DcgmPluginParamBool),
                MakeDiagnosticStringParameter(DIAGNOSTIC_STR_TEST_DURATION, "0", DcgmPluginParamInt),
            };

            gbp.Go(gbp.GetDiagnosticTestName(), &entityList, std::size(params), params);

            THEN("the diagnostic initializes without GPU results or entity errors")
            {
                CHECK(gbp.GetGpuResults(gbp.GetDiagnosticTestName()).empty());
                CHECK(gbp.GetResult(gbp.GetDiagnosticTestName()) == NVVS_RESULT_SKIP);

                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic precision parameter parses explicit masks")
{
    GIVEN("a diagnostic plugin and one OK GPU entity")
    {
        auto entityList = GetEntityList(1, DcgmEntityStatusOk);

        auto loadPrecisionParameter = [&](std::string_view precisionValue) {
            dcgmDiagPluginTestParameter_t params[] = {
                MakeDiagnosticStringParameter(DIAGNOSTIC_STR_PRECISION, precisionValue),
                MakeDiagnosticStringParameter(DIAGNOSTIC_STR_IS_ALLOWED, "false", DcgmPluginParamBool),
            };
            gbp.Go(gbp.GetDiagnosticTestName(), entityList.get(), std::size(params), params);
        };

        WHEN("the precision mask requests single only")
        {
            loadPrecisionParameter("s");
            int32_t const precision = gbp.SetPrecisionFromString(false);

            THEN("only single precision is selected")
            {
                CHECK_FALSE(USE_HALF_PRECISION(precision));
                CHECK(USE_SINGLE_PRECISION(precision));
                CHECK_FALSE(USE_DOUBLE_PRECISION(precision));
            }
        }

        WHEN("the precision mask requests double only")
        {
            loadPrecisionParameter("d");
            int32_t const precision = gbp.SetPrecisionFromString(false);

            THEN("only double precision is selected")
            {
                CHECK_FALSE(USE_HALF_PRECISION(precision));
                CHECK_FALSE(USE_SINGLE_PRECISION(precision));
                CHECK(USE_DOUBLE_PRECISION(precision));
            }
        }

        WHEN("the precision mask requests half and single")
        {
            loadPrecisionParameter("hs");
            int32_t const precision = gbp.SetPrecisionFromString(false);

            THEN("half and single precision are selected")
            {
                CHECK(USE_HALF_PRECISION(precision));
                CHECK(USE_SINGLE_PRECISION(precision));
                CHECK_FALSE(USE_DOUBLE_PRECISION(precision));
            }
        }

        WHEN("the precision mask requests single and double")
        {
            loadPrecisionParameter("sd");
            int32_t const precision = gbp.SetPrecisionFromString(false);

            THEN("single and double precision are selected")
            {
                CHECK_FALSE(USE_HALF_PRECISION(precision));
                CHECK(USE_SINGLE_PRECISION(precision));
                CHECK(USE_DOUBLE_PRECISION(precision));
            }
        }

        WHEN("the precision mask uses a comma-separated half and double request")
        {
            loadPrecisionParameter("h,d");
            int32_t const precision = gbp.SetPrecisionFromString(false);

            THEN("half and double precision are selected")
            {
                CHECK(USE_HALF_PRECISION(precision));
                CHECK_FALSE(USE_SINGLE_PRECISION(precision));
                CHECK(USE_DOUBLE_PRECISION(precision));
            }
        }

        WHEN("the precision mask is an uppercase combined request")
        {
            loadPrecisionParameter("HSD");
            int32_t const precision = gbp.SetPrecisionFromString(false);

            THEN("half, single, and double precision are selected")
            {
                CHECK(USE_HALF_PRECISION(precision));
                CHECK(USE_SINGLE_PRECISION(precision));
                CHECK(USE_DOUBLE_PRECISION(precision));
            }
        }

        WHEN("the precision mask is invalid")
        {
            loadPrecisionParameter("xyz");
            int32_t const noDoubleDefault = gbp.SetPrecisionFromString(false);
            int32_t const doubleDefault   = gbp.SetPrecisionFromString(true);

            THEN("defaults follow the supportsDoubles contract")
            {
                CHECK(USE_HALF_PRECISION(noDoubleDefault));
                CHECK(USE_SINGLE_PRECISION(noDoubleDefault));
                CHECK_FALSE(USE_DOUBLE_PRECISION(noDoubleDefault));

                CHECK(USE_HALF_PRECISION(doubleDefault));
                CHECK(USE_SINGLE_PRECISION(doubleDefault));
                CHECK(USE_DOUBLE_PRECISION(doubleDefault));
            }
        }
    }
}

SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic helper paths classify variation and compute capability")
{
    GIVEN("a diagnostic plugin with tolerance parameters")
    {
        auto entityList                        = GetEntityList(2, DcgmEntityStatusOk);
        dcgmDiagPluginTestParameter_t params[] = {
            MakeDiagnosticStringParameter(DIAGNOSTIC_STR_IS_ALLOWED, "false", DcgmPluginParamBool),
            MakeDiagnosticStringParameter(DIAGNOSTIC_STR_GFLOPS_TOLERANCE_PCNT, "0.10", DcgmPluginParamFloat),
            MakeDiagnosticStringParameter(DIAGNOSTIC_STR_TOLERANCE_PCNT, "0.00", DcgmPluginParamFloat),
            MakeDiagnosticStringParameter(DIAGNOSTIC_STR_CLOCKS_TOLERANCE_PCNT, "0.00", DcgmPluginParamFloat),
            MakeDiagnosticStringParameter(DIAGNOSTIC_STR_POWER_TOLERANCE_PCNT, "0.00", DcgmPluginParamFloat),
        };
        gbp.Go(gbp.GetDiagnosticTestName(), entityList.get(), std::size(params), params);
        std::vector<std::unique_ptr<GpuBurnDevice>> burnDevices;
        auto &devices = gbp.GetDeviceList();
        devices.clear();
        for (unsigned int gpuId : { 0U, 1U })
        {
            burnDevices.push_back(std::make_unique<GpuBurnDevice>(mockDcgmOps, mockCudaOps, mockCublasOps));
            burnDevices.back()->gpuId = gpuId;
            devices.push_back(burnDevices.back().get());
        }
        DcgmNs::Defer cleanupDevices([&devices] { devices.clear(); });

        WHEN("GFLOPS variation is above the configured tolerance")
        {
            gbp.SetGflopsTolerance(0.10);
            std::vector<double> gflops { 100.0, 40.0 };
            bool const failed = gbp.CheckVariationFailures(1, gflops);

            THEN("the variation helper reports a failure without querying recorder-backed fields")
            {
                CHECK(failed);
                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors >= 1);
            }
        }

        WHEN("field tolerance is disabled or field metadata is unsupported")
        {
            bool const disabled = gbp.CheckToleranceForField(DCGM_FI_DEV_SM_CLOCK, 0.0, 1, 2);
            bool const unknown = gbp.CheckToleranceForField(static_cast<unsigned short>(DCGM_FI_MAX_FIELDS), 0.2, 1, 2);

            THEN("the helper returns no variation failure for both non-actionable inputs")
            {
                CHECK_FALSE(disabled);
                CHECK_FALSE(unknown);
            }
        }
    }

    GIVEN("a diagnostic plugin precision mask and CUDA attribute stubs")
    {
        WHEN("half precision is unsupported by the reported compute capability")
        {
            gbp.SetPrecision(DIAG_HALF_PRECISION | DIAG_SINGLE_PRECISION | DIAG_DOUBLE_PRECISION);
            mockCudaOps->cudaAttributeValue = 4;
            gbp.UpdateForHGemmSupport(0);

            THEN("half precision is pruned from the mask")
            {
                CHECK_FALSE(USE_HALF_PRECISION(gbp.GetPrecision()));
                CHECK(USE_SINGLE_PRECISION(gbp.GetPrecision()));
                CHECK(USE_DOUBLE_PRECISION(gbp.GetPrecision()));
            }
        }

        WHEN("double precision support is inferred from the single-to-double perf ratio")
        {
            gbp.SetPrecision(DIAG_SINGLE_PRECISION);
            gbp.SetExplicitTests(false);
            mockCudaOps->cudaAttributeValue = 4;
            gbp.UpdateForDGemmSupport(0);

            THEN("double precision is added for low ratios")
            {
                CHECK(USE_DOUBLE_PRECISION(gbp.GetPrecision()));
            }
        }

        WHEN("explicit precision was requested")
        {
            gbp.SetPrecision(DIAG_SINGLE_PRECISION);
            gbp.SetExplicitTests(true);
            mockCudaOps->cudaAttributeValue = 4;
            gbp.UpdateForDGemmSupport(0);

            THEN("automatic double precision updates are skipped")
            {
                CHECK_FALSE(USE_DOUBLE_PRECISION(gbp.GetPrecision()));
            }
        }
    }
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: SetPrecisionFromString")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = GetEntityList(1, DcgmEntityStatusOk);

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

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: GFLOPS Threshold Violation")
{
    SECTION("Only one GPU")
    {
        std::vector<double> gflops = { 600.0 };
        double minThresh           = gbp.GetMinThreshold(gflops, 0.05);
        auto indicesBelowThreshold = gbp.GetIndicesBelowMinThreshold(gflops, minThresh);
        CHECK(indicesBelowThreshold.size() == 0);
    }


    SECTION("Value below threshold")
    {
        std::vector<double> gflops = { 600.0, 598.0, 600.0, 550.0 };
        double minThresh           = gbp.GetMinThreshold(gflops, 0.05);
        auto indicesBelowThreshold = gbp.GetIndicesBelowMinThreshold(gflops, minThresh);
        CHECK(indicesBelowThreshold.size() > 0);
        CHECK(indicesBelowThreshold == std::vector<size_t> { 3 });
    }

    SECTION("All values within threshold")
    {
        std::vector<double> gflops = { 600.0, 598.0, 600.0, 550.0 };
        double minThresh           = gbp.GetMinThreshold(gflops, 0.1);
        auto indicesBelowThreshold = gbp.GetIndicesBelowMinThreshold(gflops, minThresh);
        CHECK(indicesBelowThreshold.size() == 0);
    }

    SECTION("Multiple values below threshold")
    {
        std::vector<double> gflops = { 600.0, 1.0, 300.0, 550.0 };
        double minThresh           = gbp.GetMinThreshold(gflops, 0.01);
        auto indicesBelowThreshold = gbp.GetIndicesBelowMinThreshold(gflops, minThresh);
        CHECK(indicesBelowThreshold.size() > 0);
        CHECK(indicesBelowThreshold == std::vector<size_t> { 1, 2 });
    }
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: GpuBurnDevice parameters")
{
    InitializeDiagPlugin(gbp, 1, DcgmEntityStatusOk);

    GpuBurnDevice gbd(mockDcgmOps, mockCudaOps, mockCublasOps);
    DcgmNs::MockDcgmOps recorderMockDcgmOps;
    DcgmRecorder dr(recorderMockDcgmOps.m_handleId, &recorderMockDcgmOps, &nvvsCommon);
    GpuBurnWorker gbw(&gbd, gbp, DIAG_SINGLE_PRECISION, 15.0, 2048, dr, mockCudaCommonPtr, false, 1001);
    CHECK(GpuBurnWorkerTester::GetNElemsPerIter(gbw) == (2048 * 2048));
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: Calculate GFlops Multiplier")
{
    int matrixSizes[]     = { 2048, 4096, 6144, 8192, 0 };
    double baseMultiplier = CalculateGFlopsMultiplier(matrixSizes[0]);
    for (unsigned int i = 0; matrixSizes[i] != 0; i++)
    {
        double multiplier = CalculateGFlopsMultiplier(matrixSizes[i]);
        CHECK(abs(multiplier - (baseMultiplier * pow(matrixSizes[i] / 2048.0, 3))) < 1.0);
    }
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: CheckAveragesForExcessiveVariation")
{
    InitializeDiagPlugin(gbp, 4, DcgmEntityStatusOk);

    std::vector<unsigned long> clockAvgsGood = { 900, 900, 900, 900 };
    std::vector<unsigned long> clockAvgsBad  = { 900, 500, 900, 900 };
    double tolerance                         = 0.1;
    double minThresh                         = gbp.GetMinThreshold(clockAvgsGood, tolerance);
    auto indices                             = gbp.GetIndicesBelowMinThreshold(clockAvgsGood, minThresh);
    CHECK(indices.size() == 0);
    minThresh = gbp.GetMinThreshold(clockAvgsBad, tolerance);
    indices   = gbp.GetIndicesBelowMinThreshold(clockAvgsBad, minThresh);
    REQUIRE(indices.size() == 1);
    CHECK(indices[0] == 1);

    std::vector<double> powerAvgsGood = { 1350.0, 1340.0, 1400.0, 1370.0 };
    std::vector<double> powerAvgsBad  = { 1350.0, 1340.0, 1400.0, 870.0 };

    minThresh = gbp.GetMinThreshold(powerAvgsGood, tolerance);
    indices   = gbp.GetIndicesBelowMinThreshold(powerAvgsGood, minThresh);
    REQUIRE(indices.size() == 0);
    minThresh = gbp.GetMinThreshold(powerAvgsBad, tolerance);
    indices   = gbp.GetIndicesBelowMinThreshold(powerAvgsBad, minThresh);
    REQUIRE(indices.size() == 1);
    CHECK(indices[0] == 3);
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: negative test for DCGM_FR_CUDA_UNBOUND")
{
    unsigned int gpuCount { 2 };
    unsigned int unboundGpuId { 0 };
    int unboundCudaDeviceIdx { 0 };

    mockCudaOps->cuInitResult = CUDA_SUCCESS;

    InitializeDiagPlugin(gbp, gpuCount, DcgmEntityStatusOk);

    // A synthetic GpuBurnDevice leaves cuContext as nullptr, triggering the unbound error path.
    GpuBurnDevice gbd(mockDcgmOps, mockCudaOps, mockCublasOps);
    gbd.gpuId         = unboundGpuId;
    gbd.cudaDeviceIdx = unboundCudaDeviceIdx;

    DcgmNs::MockDcgmOps recorderMockDcgmOps;
    DcgmRecorder dr(recorderMockDcgmOps.m_handleId, &recorderMockDcgmOps, &nvvsCommon);
    GpuBurnWorker gbw(&gbd, gbp, DIAG_SINGLE_PRECISION, 15.0, 2048, dr, mockCudaCommonPtr, false, 1001);

    int result = gbw.Bind();
    REQUIRE(result == -1);

    // Verify the error is reported for the correct entity with the expected error code
    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = gbp.GetResults(gbp.GetDiagnosticTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 1);
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == unboundGpuId);
    CHECK(entityResults.errors[0].code == DCGM_FR_CUDA_UNBOUND);

    // Confirm only the affected GPU received the error, all other entities should have none
    nvvsPluginEntityErrors_t errorsPerEntity = gbp.GetEntityErrors(gbp.GetDiagnosticTestName());

    unsigned int count { 0 };
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        if (entityPair.entityGroupId == DCGM_FE_GPU && entityPair.entityId == unboundGpuId)
        {
            REQUIRE(diagErrors.size() == 1);
            count++;
        }
        else
        {
            REQUIRE(diagErrors.size() == 0);
        }
    }
    REQUIRE(count == 1);
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: negative test for DCGM_FR_FAULTY_MEMORY")
{
    unsigned int gpuCount { 2 };
    unsigned int faultyGpuId { 1 };

    InitializeDiagPlugin(gbp, gpuCount, DcgmEntityStatusOk);

    std::vector<unsigned int> gpuIds { 0, 1 };
    std::vector<long long> errorCount { 0, 3 };
    std::vector<long long> nanCount { 0, 0 };

    // GPU 1 has non-zero error count, which triggers the faulty memory error path
    bool allPassed
        = CheckPassFailWithDevices(gbp, mockDcgmOps, mockCudaOps, mockCublasOps, gpuIds, errorCount, nanCount);
    REQUIRE(allPassed == false);

    // Verify the error is reported for the correct entity with the expected error code
    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();

    dcgmReturn_t ret = gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get());
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(pEntityResults->numErrors == 1);
    CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(pEntityResults->errors[0].entity.entityId == faultyGpuId);
    CHECK(pEntityResults->errors[0].code == DCGM_FR_FAULTY_MEMORY);

    // Confirm only the affected GPU received the error, all other entities should have none
    nvvsPluginEntityErrors_t errorsPerEntity = gbp.GetEntityErrors(gbp.GetDiagnosticTestName());

    unsigned int count { 0 };
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        if (entityPair.entityGroupId == DCGM_FE_GPU && entityPair.entityId == faultyGpuId)
        {
            CHECK(diagErrors.size() == 1);
            count++;
        }
        else
        {
            CHECK(diagErrors.size() == 0);
        }
    }
    REQUIRE(count == 1);
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: negative test for DCGM_FR_TEST_DISABLED")
{
    auto entityList                                                         = GetEntityList(1, DcgmEntityStatusOk);
    entityList->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;

    dcgmDiagPluginTestParameter_t param {};
    param.type = DcgmPluginParamBool;
    SafeCopyTo(param.parameterName, DIAGNOSTIC_STR_IS_ALLOWED);
    SafeCopyTo(param.parameterValue, "False");

    gbp.Go(gbp.GetDiagnosticTestName(), entityList.get(), 1, &param);
    CHECK(gbp.GetResult(gbp.GetDiagnosticTestName()) == NVVS_RESULT_SKIP);

    auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), entityResults.get()) == DCGM_ST_OK);
    CHECK(entityResults->numErrors == 0);
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic: negative test for DCGM_FR_ABORTED")
{
    auto entityList                                                         = GetEntityList(1, DcgmEntityStatusOk);
    entityList->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
    snprintf(entityList->entities[0].auxField.gpu.attributes.identifiers.pciBusId,
             sizeof(entityList->entities[0].auxField.gpu.attributes.identifiers.pciBusId),
             "0000:01:00.0");

    dcgmDiagPluginTestParameter_t param {};
    snprintf(param.parameterName, sizeof(param.parameterName), "%s", DIAGNOSTIC_STR_IS_ALLOWED);
    snprintf(param.parameterValue, sizeof(param.parameterValue), "True");
    param.type = DcgmPluginParamBool;


    ScopedAtomic stopGuard(main_should_stop, 1);
    gbp.Go(gbp.GetDiagnosticTestName(), entityList.get(), 1, &param);

    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);

    // The abort path may also record incidental setup/cleanup errors from non-injected dependencies,
    // so we only assert that DCGM_FR_ABORTED is present and correctly attributed.
    REQUIRE(pEntityResults->numErrors >= 1);
    bool sawAborted = false;
    for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
    {
        if (pEntityResults->errors[i].code == DCGM_FR_ABORTED)
        {
            sawAborted = true;
            CHECK(pEntityResults->errors[i].entity.entityGroupId == 0);
            CHECK(pEntityResults->errors[i].entity.entityId == 0);
        }
    }
    CHECK(sawAborted);
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic tolerance percentages are normalized")
{
    GIVEN("valid tolerance percentages")
    {
        auto validZero = 0.0;
        auto validMid  = 0.25;

        WHEN("the values are checked")
        {
            gbp.CorrectTolerancePercentageIfWrong(validZero, "zero tolerance");
            gbp.CorrectTolerancePercentageIfWrong(validMid, "mid tolerance");

            THEN("they are preserved")
            {
                CHECK(validZero == 0.0);
                CHECK(validMid == 0.25);
            }
        }
    }

    GIVEN("out-of-range tolerance percentages")
    {
        auto negative = -0.1;
        auto one      = 1.0;
        auto aboveOne = 1.01;

        WHEN("the values are checked")
        {
            gbp.CorrectTolerancePercentageIfWrong(negative, "negative tolerance");
            gbp.CorrectTolerancePercentageIfWrong(one, "one tolerance");
            gbp.CorrectTolerancePercentageIfWrong(aboveOne, "above-one tolerance");

            THEN("they are disabled")
            {
                CHECK(negative == 0.0);
                CHECK(one == 0.0);
                CHECK(aboveOne == 0.0);
            }
        }
    }
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic global tolerance fills unset field tolerances")
{
    GIVEN("an unset field-specific tolerance and a positive global tolerance")
    {
        double fieldTolerance = 0.0;

        WHEN("the defaulting helper is applied")
        {
            gbp.SetToleranceIfUnset(DIAGNOSTIC_STR_CLOCKS_TOLERANCE_PCNT, fieldTolerance, 0.2);

            THEN("the field-specific value receives the global tolerance")
            {
                CHECK(fieldTolerance == 0.2);
            }
        }
    }

    GIVEN("an already configured field-specific tolerance")
    {
        double fieldTolerance = 0.05;

        WHEN("the defaulting helper is applied")
        {
            gbp.SetToleranceIfUnset(DIAGNOSTIC_STR_POWER_TOLERANCE_PCNT, fieldTolerance, 0.2);

            THEN("the field-specific value is preserved")
            {
                CHECK(fieldTolerance == 0.05);
            }
        }
    }

    GIVEN("a non-positive global tolerance")
    {
        double zeroFieldTolerance     = 0.0;
        double negativeFieldTolerance = 0.0;

        WHEN("the defaulting helper is applied")
        {
            gbp.SetToleranceIfUnset(DIAGNOSTIC_STR_CLOCKS_TOLERANCE_PCNT, zeroFieldTolerance, 0.0);
            gbp.SetToleranceIfUnset(DIAGNOSTIC_STR_POWER_TOLERANCE_PCNT, negativeFieldTolerance, -0.1);

            THEN("the field-specific values remain unset")
            {
                CHECK(zeroFieldTolerance == 0.0);
                CHECK(negativeFieldTolerance == 0.0);
            }
        }
    }
}

SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic CheckPassFail records all-pass and empty-device outcomes")
{
    GIVEN("a diagnostic plugin initialized for two GPUs")
    {
        InitializeDiagPlugin(gbp, 2, DcgmEntityStatusOk);

        WHEN("both synthetic workers report clean error and NaN counts")
        {
            bool const allPassed
                = CheckPassFailWithDevices(gbp, mockDcgmOps, mockCudaOps, mockCublasOps, { 0, 1 }, { 0, 0 }, { 0, 0 });

            THEN("each GPU is marked passing and no entity errors are emitted")
            {
                CHECK(allPassed);
                auto const &gpuResults = gbp.GetGpuResults(gbp.GetDiagnosticTestName());
                REQUIRE(gpuResults.size() == 2);
                CHECK(gpuResults.at(0) == NVVS_RESULT_PASS);
                CHECK(gpuResults.at(1) == NVVS_RESULT_PASS);

                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("pass/fail aggregation is asked to evaluate no runtime devices")
        {
            bool const allPassed = CheckPassFailWithDevices(gbp, mockDcgmOps, mockCudaOps, mockCublasOps, {}, {}, {});

            THEN("the empty worker set is a successful no-op")
            {
                CHECK(allPassed);
                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic variation helpers treat no-op domains as non-failing")
{
    GIVEN("a diagnostic plugin without runtime worker devices")
    {
        WHEN("the tolerance is positive but there is only one sample")
        {
            std::vector<double> oneSample { 42.0 };
            bool const errorDetected = gbp.CheckAveragesForExcessiveVariation(oneSample, 0.5, "single sample");

            THEN("no threshold error is recorded")
            {
                CHECK_FALSE(errorDetected);
            }
        }

        WHEN("the sample set has variation but tolerance checking is disabled")
        {
            std::vector<double> values { 100.0, 1.0, 100.0 };
            bool const errorDetected = gbp.CheckAveragesForExcessiveVariation(values, 0.0, "disabled tolerance");

            THEN("the helper exits before touching plugin entity state")
            {
                CHECK_FALSE(errorDetected);
                CHECK_THROWS_AS(gbp.GetGpuResults(gbp.GetDiagnosticTestName()), std::out_of_range);
            }
        }
    }
}

TEST_CASE_METHOD(DiagnosticPluginTestFixture, "Diagnostic CheckPassFail reports NaN and combined failures")
{
    InitializeDiagPlugin(gbp, 3, DcgmEntityStatusOk);

    GIVEN("three burn devices with clean, NaN, and faulty-memory outcomes")
    {
        std::vector<unsigned int> gpuIds { 0, 1, 2 };
        std::vector<long long> errorCount { 0, 0, 4 };
        std::vector<long long> nanCount { 0, 2, 5 };

        WHEN("pass/fail is evaluated")
        {
            bool const allPassed
                = CheckPassFailWithDevices(gbp, mockDcgmOps, mockCudaOps, mockCublasOps, gpuIds, errorCount, nanCount);

            THEN("only the clean GPU passes and each failure is reported on its GPU")
            {
                CHECK_FALSE(allPassed);
                auto const &gpuResults = gbp.GetGpuResults(gbp.GetDiagnosticTestName());
                REQUIRE(gpuResults.size() == 3);
                CHECK(gpuResults.at(0) == NVVS_RESULT_PASS);
                CHECK(gpuResults.at(1) == NVVS_RESULT_FAIL);
                CHECK(gpuResults.at(2) == NVVS_RESULT_FAIL);

                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 3);
                CHECK(pEntityResults->errors[0].entity.entityId == 1);
                CHECK(pEntityResults->errors[0].code == DCGM_FR_NAN_VALUE);
                CHECK(pEntityResults->errors[1].entity.entityId == 2);
                CHECK(pEntityResults->errors[1].code == DCGM_FR_FAULTY_MEMORY);
                CHECK(pEntityResults->errors[2].entity.entityId == 2);
                CHECK(pEntityResults->errors[2].code == DCGM_FR_NAN_VALUE);
            }
        }
    }
}


SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic variation reporting records every low unsigned average")
{
    GIVEN("a diagnostic plugin initialized for four synthetic GPUs")
    {
        InitializeDiagPlugin(gbp, 4, DcgmEntityStatusOk);
        auto &devices = gbp.GetDeviceList();
        std::vector<std::unique_ptr<GpuBurnDevice>> burnDevices;
        burnDevices.reserve(4);
        for (unsigned int gpuId = 0; gpuId < 4; ++gpuId)
        {
            burnDevices.push_back(std::make_unique<GpuBurnDevice>(mockDcgmOps, mockCudaOps, mockCublasOps));
            burnDevices.back()->gpuId = gpuId;
            devices.push_back(burnDevices.back().get());
        }
        DcgmNs::Defer cleanup([&devices] { devices.clear(); });

        WHEN("two clock averages fall below the computed tolerance threshold")
        {
            std::vector<unsigned long> clockAverages { 1000, 200, 1000, 300 };
            bool const errorDetected = gbp.CheckAveragesForExcessiveVariation(clockAverages, 0.25, "Clock Speed");

            THEN("each low GPU receives a threshold-violation error")
            {
                CHECK(errorDetected);
                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 2);
                CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
                CHECK(pEntityResults->errors[0].entity.entityId == 1);
                CHECK(pEntityResults->errors[0].code == DCGM_FR_GFLOPS_THRESHOLD_VIOLATION);
                CHECK(pEntityResults->errors[1].entity.entityGroupId == DCGM_FE_GPU);
                CHECK(pEntityResults->errors[1].entity.entityId == 3);
                CHECK(pEntityResults->errors[1].code == DCGM_FR_GFLOPS_THRESHOLD_VIOLATION);
            }
        }
    }
}

SCENARIO_METHOD(DiagnosticPluginTestFixture,
                "Diagnostic variation checks report threshold failures through plugin results")
{
    GIVEN("a diagnostic plugin initialized for three GPUs")
    {
        InitializeDiagPlugin(gbp, 3, DcgmEntityStatusOk);
        auto &devices = gbp.GetDeviceList();
        std::vector<std::unique_ptr<GpuBurnDevice>> burnDevices;
        burnDevices.reserve(3);
        for (unsigned int gpuId = 0; gpuId < 3; ++gpuId)
        {
            burnDevices.push_back(std::make_unique<GpuBurnDevice>(mockDcgmOps, mockCudaOps, mockCublasOps));
            burnDevices.back()->gpuId = gpuId;
            devices.push_back(burnDevices.back().get());
        }
        DcgmNs::Defer cleanup([&devices] { devices.clear(); });

        WHEN("one GPU average falls below the tolerated range")
        {
            std::vector<double> gflops { 100.0, 40.0, 100.0 };
            bool const errorDetected = gbp.CheckAveragesForExcessiveVariation(gflops, 0.10, "diagnostic gflops");

            THEN("the low GPU is reported with a threshold violation")
            {
                CHECK(errorDetected);
                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 1);
                CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
                CHECK(pEntityResults->errors[0].entity.entityId == 1);
                CHECK(pEntityResults->errors[0].code == DCGM_FR_GFLOPS_THRESHOLD_VIOLATION);
            }
        }

        WHEN("the tolerance is disabled")
        {
            std::vector<double> gflops { 100.0, 40.0, 100.0 };
            bool const errorDetected = gbp.CheckAveragesForExcessiveVariation(gflops, 0.0, "diagnostic gflops");

            THEN("no threshold error is recorded")
            {
                CHECK_FALSE(errorDetected);
                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(DiagnosticPluginTestFixture,
                "Diagnostic threshold helpers cover exact-boundary and empty-index domains")
{
    GIVEN("a diagnostic plugin")
    {
        WHEN("values sit exactly on the computed minimum threshold")
        {
            std::vector<double> values { 100.0, 80.0, 120.0 };
            double const threshold = gbp.GetMinThreshold(values, 0.20);
            auto below             = gbp.GetIndicesBelowMinThreshold(values, threshold);

            THEN("the strict below-threshold comparison does not flag boundary values")
            {
                CHECK(threshold == 80.0);
                CHECK(below.empty());
            }
        }

        WHEN("unsigned clock averages contain one value just below the threshold")
        {
            std::vector<unsigned long> values { 100, 79, 121 };
            double const threshold = gbp.GetMinThreshold(values, 0.20);
            auto below             = gbp.GetIndicesBelowMinThreshold(values, threshold);

            THEN("only the below-boundary index is returned")
            {
                CHECK(threshold == 80.0);
                REQUIRE(below.size() == 1);
                CHECK(below[0] == 1);
            }
        }
    }
}


SCENARIO_METHOD(DiagnosticPluginTestFixture,
                "Diagnostic GFLOPS multiplier covers degenerate and sub-base matrix dimensions")
{
    GIVEN("matrix dimensions at zero, below base, and base sizes")
    {
        unsigned int const zeroDim = 0;
        unsigned int const halfDim = 1024;
        unsigned int const baseDim = 2048;

        WHEN("the diagnostic multiplier is calculated")
        {
            double const zeroMultiplier = CalculateGFlopsMultiplier(zeroDim);
            double const halfMultiplier = CalculateGFlopsMultiplier(halfDim);
            double const baseMultiplier = CalculateGFlopsMultiplier(baseDim);

            THEN("the cubic scaling remains deterministic")
            {
                CHECK(zeroMultiplier == 0.0);
                REQUIRE(baseMultiplier > 0.0);
                CHECK_THAT(halfMultiplier, Catch::Matchers::WithinRel(baseMultiplier / 8.0, 1e-12));
            }
        }
    }
}


SCENARIO_METHOD(DiagnosticPluginTestFixture,
                "Diagnostic worker CUDA-stub paths cover binding, buffer setup, and fail-early compare")
{
    GIVEN("a diagnostic worker with a synthetic CUDA context")
    {
        auto entityList = GetEntityList(1, DcgmEntityStatusOk);
        gbp.InitializeForEntityList(gbp.GetDiagnosticTestName(), *entityList);

        GpuBurnDevice device(mockDcgmOps, mockCudaOps, mockCublasOps);
        device.gpuId                      = 0;
        device.cudaDeviceIdx              = 0;
        device.cuContext                  = reinterpret_cast<CUcontext>(0x1);
        mockCudaOps->cuCtxGetCurrentValue = device.cuContext;
        DcgmNs::MockDcgmOps recorderMockDcgmOps;
        DcgmRecorder recorder(recorderMockDcgmOps.m_handleId, &recorderMockDcgmOps, &nvvsCommon);

        WHEN("the worker binds and initializes small matrix buffers")
        {
            mockCudaOps->cuMemGetInfoFree  = 4096;
            mockCudaOps->cuMemGetInfoTotal = 8192;
            mockCudaOps->cuMemAllocResult  = CUDA_SUCCESS;

            GpuBurnWorker worker(&device,
                                 gbp,
                                 DIAG_HALF_PRECISION | DIAG_SINGLE_PRECISION | DIAG_DOUBLE_PRECISION,
                                 0.0,
                                 2,
                                 recorder,
                                 mockCudaCommonPtr,
                                 false,
                                 1);

            int bindStatus         = worker.Bind();
            int memoryStatus       = 0;
            size_t const available = worker.AvailMemory(memoryStatus);
            int const initStatus   = worker.InitBuffers();

            THEN("the stubbed driver path succeeds and derives matrix metadata")
            {
                CHECK(bindStatus == 0);
                CHECK(memoryStatus == 0);
                CHECK(available == 4096);
                CHECK(initStatus == 0);
                CHECK(GpuBurnWorkerTester::GetNElemsPerIter(worker) == 4);
                CHECK(worker.GetGFlopsMultiplier() == CalculateGFlopsMultiplier(2));
            }
        }

        WHEN("device allocation fails during buffer initialization")
        {
            mockCudaOps->cuMemGetInfoFree  = 4096;
            mockCudaOps->cuMemGetInfoTotal = 8192;
            mockCudaOps->cuMemAllocResult  = CUDA_ERROR_OUT_OF_MEMORY;

            GpuBurnWorker worker(&device, gbp, DIAG_SINGLE_PRECISION, 0.0, 2, recorder, mockCudaCommonPtr, false, 1);
            int const initStatus = worker.InitBuffers();

            THEN("the CUDA allocation failure is returned to the caller")
            {
                CHECK(initStatus == -1);
            }
        }

        WHEN("fail-early compare observes synthetic faulty and NaN counts")
        {
            mockCudaOps->cuMemcpyDtoHValue = 1;
            GpuBurnWorker worker(&device, gbp, DIAG_SINGLE_PRECISION, 0.0, 2, recorder, mockCudaCommonPtr, true, 1);
            int const compareStatus = worker.Compare(DIAG_SINGLE_PRECISION);

            THEN("the comparison requests early termination")
            {
                CHECK(compareStatus == 1);
            }
        }
    }

    GIVEN("a diagnostic worker whose runtime context is not bound")
    {
        InitializeDiagPlugin(gbp, 1, DcgmEntityStatusOk);
        GpuBurnDevice device(mockDcgmOps, mockCudaOps, mockCublasOps);
        device.gpuId         = 3;
        device.cudaDeviceIdx = 3;
        device.cuContext     = nullptr;
        DcgmNs::MockDcgmOps recorderMockDcgmOps;
        DcgmRecorder recorder(recorderMockDcgmOps.m_handleId, &recorderMockDcgmOps, &nvvsCommon);

        WHEN("Bind is called")
        {
            GpuBurnWorker worker(&device, gbp, DIAG_SINGLE_PRECISION, 0.0, 2, recorder, mockCudaCommonPtr, false, 1);
            int const bindStatus = worker.Bind();

            THEN("the unbound CUDA context is reported as a worker initialization failure")
            {
                CHECK(bindStatus == -1);
            }
        }
    }
}

SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic worker reports CUDA compare-kernel setup failures")
{
    GIVEN("a diagnostic worker with a bound synthetic CUDA context")
    {
        auto entityList = GetEntityList(1, DcgmEntityStatusOk);
        gbp.InitializeForEntityList(gbp.GetDiagnosticTestName(), *entityList);

        GpuBurnDevice device(mockDcgmOps, mockCudaOps, mockCublasOps);
        device.gpuId                      = 0;
        device.cudaDeviceIdx              = 0;
        device.cuContext                  = reinterpret_cast<CUcontext>(0x1);
        mockCudaOps->cuCtxGetCurrentValue = device.cuContext;
        DcgmNs::MockDcgmOps recorderMockDcgmOps;
        DcgmRecorder recorder(recorderMockDcgmOps.m_handleId, &recorderMockDcgmOps, &nvvsCommon);

        WHEN("the CUDA module cannot be loaded")
        {
            mockCudaOps->cuModuleLoadDataResult = CUDA_ERROR_OUT_OF_MEMORY;
            GpuBurnWorker worker(&device, gbp, DIAG_SINGLE_PRECISION, 0.0, 2, recorder, mockCudaCommonPtr, false, 1);

            THEN("InitCompareKernel reports a worker setup failure")
            {
                CHECK(worker.InitCompareKernel() == -1);
            }
        }

        WHEN("a compare function cannot be looked up")
        {
            mockCudaOps->cuModuleGetFunctionResult = CUDA_ERROR_OUT_OF_MEMORY;
            GpuBurnWorker worker(&device, gbp, DIAG_SINGLE_PRECISION, 0.0, 2, recorder, mockCudaCommonPtr, false, 1);

            THEN("InitCompareKernel reports a worker setup failure")
            {
                CHECK(worker.InitCompareKernel() == -1);
            }
        }

        WHEN("kernel launch fails during compare")
        {
            mockCudaOps->cuLaunchKernelResult = CUDA_ERROR_OUT_OF_MEMORY;
            GpuBurnWorker worker(&device, gbp, DIAG_SINGLE_PRECISION, 0.0, 2, recorder, mockCudaCommonPtr, false, 1);

            THEN("Compare returns the CUDA failure")
            {
                CHECK(worker.Compare(DIAG_SINGLE_PRECISION) == -1);
            }
        }
    }
}


SCENARIO_METHOD(DiagnosticPluginTestFixture,
                "Diagnostic device construction reports CUDA driver initialization failures")
{
    GIVEN("CUDA driver initialization is configured to fail through the test stub")
    {
        InitializeDiagPlugin(gbp, 9, DcgmEntityStatusOk);
        DcgmNs::MockDcgmOps deviceDcgmOps;
        deviceDcgmOps.m_entitiesGetLatestValuesState.handler = [](dcgmHandle_t,
                                                                  dcgmGroupEntityPair_t *,
                                                                  unsigned int entityCount,
                                                                  unsigned short *,
                                                                  unsigned int fieldCount,
                                                                  unsigned int,
                                                                  dcgmFieldValue_v2 *values) {
            for (unsigned int i = 0; i < entityCount * fieldCount; ++i)
            {
                values[i].status = DCGM_ST_NOT_SUPPORTED;
            }
            return DCGM_ST_OK;
        };
        mockCudaOps->cuInitResult = CUDA_ERROR_INVALID_VALUE;

        WHEN("the driver provides a diagnostic string")
        {
            mockCudaOps->cuGetErrorStringValue = "driver initialization failed";

            THEN("GpuBurnDevice construction throws a DcgmError")
            {
                CHECK_THROWS_AS(GpuBurnDevice("diagnostic",
                                              7U,
                                              "0000:07:00.0",
                                              &gbp,
                                              &deviceDcgmOps,
                                              deviceDcgmOps.m_handleId,
                                              mockCudaOps,
                                              mockCublasOps,
                                              mockCudaCommonPtr),
                                DcgmError);
            }
        }

        WHEN("the driver provides no diagnostic string")
        {
            mockCudaOps->cuGetErrorStringValue = nullptr;

            THEN("GpuBurnDevice construction still throws a DcgmError")
            {
                CHECK_THROWS_AS(GpuBurnDevice("diagnostic",
                                              8U,
                                              "0000:08:00.0",
                                              &gbp,
                                              &deviceDcgmOps,
                                              deviceDcgmOps.m_handleId,
                                              mockCudaOps,
                                              mockCublasOps,
                                              mockCudaCommonPtr),
                                DcgmError);
            }
        }
    }
}


SCENARIO_METHOD(DiagnosticPluginTestFixture,
                "Diagnostic precision support helpers update only the advertised precision bits")
{
    GIVEN("a diagnostic plugin with synthetic CUDA attribute values")
    {
        WHEN("double precision support is explicitly controlled by the caller")
        {
            gbp.SetPrecision(DIAG_SINGLE_PRECISION);
            gbp.SetExplicitTests(true);
            mockCudaOps->cudaAttributeValue = 1;

            gbp.UpdateForDGemmSupport(0);

            THEN("automatic double-precision pruning is bypassed")
            {
                CHECK(gbp.GetPrecision() == DIAG_SINGLE_PRECISION);
            }
        }

        WHEN("the device reports fast double precision")
        {
            gbp.SetPrecision(DIAG_SINGLE_PRECISION);
            gbp.SetExplicitTests(false);
            mockCudaOps->cudaAttributeValue = 4;

            gbp.UpdateForDGemmSupport(0);

            THEN("double precision is enabled in addition to single precision")
            {
                int32_t const precision = gbp.GetPrecision();
                CHECK(USE_SINGLE_PRECISION(precision));
                CHECK(USE_DOUBLE_PRECISION(precision));
            }
        }

        WHEN("the device reports slow double precision")
        {
            gbp.SetPrecision(DIAG_SINGLE_PRECISION | DIAG_DOUBLE_PRECISION);
            gbp.SetExplicitTests(false);
            mockCudaOps->cudaAttributeValue = 8;

            gbp.UpdateForDGemmSupport(0);

            THEN("double precision is removed while single precision remains")
            {
                int32_t const precision = gbp.GetPrecision();
                CHECK(USE_SINGLE_PRECISION(precision));
                CHECK_FALSE(USE_DOUBLE_PRECISION(precision));
            }
        }

        WHEN("half precision is checked for a supported compute capability")
        {
            gbp.SetPrecision(DIAG_HALF_PRECISION | DIAG_SINGLE_PRECISION);
            mockCudaOps->cudaAttributeValue = 7;

            gbp.UpdateForHGemmSupport(0);

            THEN("half precision remains enabled")
            {
                int32_t const precision = gbp.GetPrecision();
                CHECK(USE_HALF_PRECISION(precision));
                CHECK(USE_SINGLE_PRECISION(precision));
            }
        }
    }
}


SCENARIO_METHOD(DiagnosticPluginTestFixture, "Diagnostic field tolerance checks handle recorder-backed error paths")
{
    GIVEN("a diagnostic plugin initialized for two synthetic GPUs")
    {
        InitializeDiagPlugin(gbp, 2, DcgmEntityStatusOk);
        std::vector<std::unique_ptr<GpuBurnDevice>> burnDevices;
        auto &devices = gbp.GetDeviceList();
        devices.clear();
        for (unsigned int gpuId = 0; gpuId < 2; ++gpuId)
        {
            burnDevices.push_back(std::make_unique<GpuBurnDevice>(mockDcgmOps, mockCudaOps, mockCublasOps));
            burnDevices.back()->gpuId = gpuId;
            devices.push_back(burnDevices.back().get());
        }
        DcgmNs::Defer cleanup([&devices] { devices.clear(); });

        WHEN("clock and power tolerances require recorder field summaries")
        {
            bool const clockFailed = gbp.CheckToleranceForField(DCGM_FI_DEV_SM_CLOCK, 0.15, 1, 2);
            bool const powerFailed = gbp.CheckToleranceForField(DCGM_FI_DEV_BOARD_POWER_WATTS, 0.15, 1, 2);

            THEN("missing recorder samples are treated as non-threshold failures")
            {
                CHECK_FALSE(clockFailed);
                CHECK_FALSE(powerFailed);

                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(gbp.GetResults(gbp.GetDiagnosticTestName(), pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}
