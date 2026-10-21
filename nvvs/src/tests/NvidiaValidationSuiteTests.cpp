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
#include "PluginInterface.h"
#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <dcgm_fields.h>
#include <dcgm_fields_internal.hpp>

#include <Allowlist.h>
#include <ConfigFileParser_v2.h>
#include <CoutCapture.hpp>
#include <DcgmSystem.h>
#include <Defer.hpp>
#include <Gpu.h>
#include <GpuSet.h>
#include <MockDcgmOps.h>
#include <NvidiaValidationSuite.h>
#include <NvvsCommon.h>
#include <NvvsException.hpp>
#include <NvvsExitCode.h>
#include <PluginStrings.h>
#include <ScopedEnvVars.hpp>
#include <TempDir.hpp>
#include <TempFile.hpp>
#include <Test.h>

#include <array>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

bool incompatibleGpus       = false;
bool migEnabled             = false;
unsigned int hierarchyCount = 4;
/** When true, @ref dcgmEntitiesGetLatestValues mock reports CUDA driver 12.8 for @ref DCGM_FI_CUDA_DRIVER_VERSION. */
bool g_mockCudaDriverVersion128 = false;

using namespace DcgmNs::Nvvs;

class TestNvidiaValidationSuite : protected NvidiaValidationSuite
{
public:
    using NvidiaValidationSuite::NvidiaValidationSuite;

    void WrapperProcessCommandLine(int argc, char *argv[]);
    bool WrapperGetListGpus(void);
    bool WrapperGetListTests(void);
    void WrapperEnumerateAllVisibleGpus();
    void WrapperDecipherProperties(GpuSet *set);
    void WrapperEnumerateAllVisibleTests();
    std::vector<Test *>::iterator WrapperFindTestName(std::string test);
    void WrapperOverrideParameters(TestParameters *tp, const std::string &lowerCaseTestName);
    void WrapperInitializeParameters(const std::string &params, const ParameterValidator &pv);
    bool WrapperIsGpuIncluded(unsigned int gpuIndex, std::vector<unsigned int> &gpuIndices) const;
    bool WrapperHasGenericSupport(std::string const &gpuBrand, uint64_t gpuArch);
    void WrapperSetCudaVisibleDevicesForTesting(std::vector<unsigned int> const &gpuIndices);
    void WrapperAddGpuForTesting(std::unique_ptr<Gpu> gpu);
    void WrapperEnsureTestFramework();
    void WrapperSetTestVector(std::vector<Test *> tests);
    void WrapperValidateSubtestParameters();
    void WrapperBanner();
    void WrapperThrowTestNotFoundException() const;
    std::string const &WrapperGetConfigFile() const;
    std::string const &WrapperGetDebugFile() const;
    std::string const &WrapperGetDebugLogLevel() const;
    std::string const &WrapperGetHwDiagLogFile() const;
    using NvidiaValidationSuite::BuildCommonGpusList;
};

struct NvidiaValidationSuiteTestFixture
{
    NvidiaValidationSuiteTestFixture()
        : dcgmOps()
        , nvvsCommon()
        , handle(&dcgmOps)
        , system(&dcgmOps, dcgmOps.m_handleId)
    {
        handle = dcgmOps.m_handleId;
    }

    DcgmNs::MockDcgmOps dcgmOps;
    NvvsCommon nvvsCommon;
    DcgmHandle handle;
    DcgmSystem system;

    TestNvidiaValidationSuite MakeTestNvidiaValidationSuite()
    {
        DcgmHandle dcgmHandle(&dcgmOps);
        dcgmHandle = dcgmOps.m_handleId;
        return TestNvidiaValidationSuite(&nvvsCommon, std::move(dcgmHandle), DcgmSystem(&dcgmOps, dcgmOps.m_handleId));
    }

    void ProcessCommandLine(TestNvidiaValidationSuite &nvvs, std::initializer_list<std::string> arguments)
    {
        argumentStorage.assign(arguments.begin(), arguments.end());
        argv.clear();
        argv.reserve(argumentStorage.size());
        for (std::string &argument : argumentStorage)
        {
            argv.push_back(argument.data());
        }

        nvvs.WrapperProcessCommandLine(static_cast<int>(argv.size()), argv.data());
    }

    std::unique_ptr<Gpu> CreateGpu(unsigned int gpuId, std::string const &uuid)
    {
        auto gpu             = std::make_unique<Gpu>(gpuId, &dcgmOps, &handle, &system);
        m_uuidMap[gpu.get()] = uuid;
        return gpu;
    }

    std::string GetUuid(Gpu *gpu) const
    {
        auto it = m_uuidMap.find(gpu);
        if (it != m_uuidMap.end())
        {
            return it->second;
        }
        return "";
    }

private:
    std::map<Gpu *, std::string> m_uuidMap;
    std::vector<std::string> argumentStorage;
    std::vector<char *> argv;
};

TEST_CASE_METHOD(NvidiaValidationSuiteTestFixture, "NvvsCommon: value helpers and construction defaults")
{
    SECTION("construction initializes public fields")
    {
        static_assert(!std::is_copy_constructible_v<NvvsCommon>);
        static_assert(!std::is_copy_assignable_v<NvvsCommon>);
        static_assert(!std::is_move_constructible_v<NvvsCommon>);
        static_assert(!std::is_move_assignable_v<NvvsCommon>);

        NvvsCommon nvvsCommon;
        REQUIRE(nvvsCommon.logFile.empty());
        REQUIRE(nvvsCommon.m_statsPath == "./");
        REQUIRE(nvvsCommon.logFileType == NVVS_LOGFILE_TYPE_JSON);
        REQUIRE(nvvsCommon.requirePersistenceMode == true);
        REQUIRE(nvvsCommon.desiredTest.empty());
        REQUIRE(nvvsCommon.parms.empty());
        REQUIRE(nvvsCommon.parsedIgnoreErrorCodes.empty());
        REQUIRE(nvvsCommon.channelFd == -1);
        REQUIRE(nvvsCommon.diagResponseVersion == dcgmDiagResponse_version12);
        REQUIRE(nvvsCommon.rerunAsRoot == false);
        REQUIRE(nvvsCommon.watchFrequency == 5000000);
        REQUIRE(nvvsCommon.hangDetectExpirySec == 600);
        REQUIRE(nvvsCommon.genericMode == false);
    }

    SECTION("SetStatsPath accepts empty and existing directories but rejects bad paths")
    {
        NvvsCommon nvvsCommon;
        nvvsCommon.SetStatsPath("");
        REQUIRE(nvvsCommon.m_statsPath == "./");

        nvvsCommon.SetStatsPath(".");
        REQUIRE(nvvsCommon.m_statsPath == ".");

        auto badStatsPath = TempFile::Random("", "nvvs-stats-path-file-");

        REQUIRE_THROWS_MATCHES(
            nvvsCommon.SetStatsPath(badStatsPath.Path().string()),
            std::runtime_error,
            Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                std::string("Error: statspath '") + badStatsPath.Path().string() + "' is not a directory.")));
        REQUIRE_THROWS_MATCHES(nvvsCommon.SetStatsPath("/definitely/not/a/dcgm/stats/path"),
                               std::runtime_error,
                               Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                                   "Error: cannot access statspath '/definitely/not/a/dcgm/stats/path'")));
    }
}

TEST_CASE_METHOD(NvidiaValidationSuiteTestFixture, "NvvsCommon: test and result mappings")
{
    SECTION("display names and indices round-trip for known plugin tests")
    {
        constexpr auto testMappings = std::to_array<std::pair<dcgmPerGpuTestIndices_t, std::string_view>>({
            { DCGM_MEMORY_INDEX, MEMORY_PLUGIN_NAME },
            { DCGM_DIAGNOSTIC_INDEX, DIAGNOSTIC_PLUGIN_NAME },
            { DCGM_PCI_INDEX, PCIE_PLUGIN_NAME },
            { DCGM_SM_STRESS_INDEX, SMSTRESS_PLUGIN_NAME },
            { DCGM_TARGETED_STRESS_INDEX, TS_PLUGIN_NAME },
            { DCGM_TARGETED_POWER_INDEX, TP_PLUGIN_NAME },
            { DCGM_MEMORY_BANDWIDTH_INDEX, MEMBW_PLUGIN_NAME },
            { DCGM_SOFTWARE_INDEX, SW_PLUGIN_NAME },
            { DCGM_CONTEXT_CREATE_INDEX, CTXCREATE_PLUGIN_NAME },
            { DCGM_MEMTEST_INDEX, MEMTEST_PLUGIN_NAME },
            { DCGM_PULSE_TEST_INDEX, PULSE_TEST_PLUGIN_NAME },
            { DCGM_EUD_TEST_INDEX, EUD_PLUGIN_NAME },
            { DCGM_NVBANDWIDTH_INDEX, NVBANDWIDTH_PLUGIN_NAME },
            { DCGM_NCCL_TESTS_INDEX, NCCL_TESTS_PLUGIN_NAME },
            { DCGM_RIST_TEST_INDEX, RIST_PLUGIN_NAME },
        });

        for (auto const &[index, name] : testMappings)
        {
            INFO("name: " << name);
            REQUIRE(GetTestDisplayName(index) == name);
            if (index == DCGM_NCCL_TESTS_INDEX)
            {
                REQUIRE(GetTestIndex(std::string(name)) == DCGM_UNKNOWN_INDEX);
            }
            else
            {
                REQUIRE(GetTestIndex(std::string(name)) == index);
            }
        }

        REQUIRE(GetTestDisplayName(DCGM_UNKNOWN_INDEX) == "Unknown");
        REQUIRE(GetTestIndex("not-a-plugin") == DCGM_UNKNOWN_INDEX);
        REQUIRE(GetTestIndex("PCIE") == DCGM_PCI_INDEX);
    }

    SECTION("NVVS and diag result conversions cover all public states")
    {
        REQUIRE(NvvsPluginResultToDiagResult(NVVS_RESULT_PASS) == DCGM_DIAG_RESULT_PASS);
        REQUIRE(NvvsPluginResultToDiagResult(NVVS_RESULT_WARN) == DCGM_DIAG_RESULT_WARN);
        REQUIRE(NvvsPluginResultToDiagResult(NVVS_RESULT_FAIL) == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(NvvsPluginResultToDiagResult(NVVS_RESULT_SKIP) == DCGM_DIAG_RESULT_SKIP);
        REQUIRE(NvvsPluginResultToDiagResult(static_cast<nvvsPluginResult_enum>(999)) == DCGM_DIAG_RESULT_FAIL);

        REQUIRE(DcgmResultToNvvsResult(DCGM_DIAG_RESULT_PASS) == NVVS_RESULT_PASS);
        REQUIRE(DcgmResultToNvvsResult(DCGM_DIAG_RESULT_WARN) == NVVS_RESULT_WARN);
        REQUIRE(DcgmResultToNvvsResult(DCGM_DIAG_RESULT_FAIL) == NVVS_RESULT_FAIL);
        REQUIRE(DcgmResultToNvvsResult(DCGM_DIAG_RESULT_SKIP) == NVVS_RESULT_SKIP);
        REQUIRE(DcgmResultToNvvsResult(DCGM_DIAG_RESULT_NOT_RUN) == NVVS_RESULT_SKIP);
        REQUIRE(DcgmResultToNvvsResult(static_cast<dcgmDiagResult_t>(999)) == NVVS_RESULT_SKIP);
    }
}

TEST_CASE_METHOD(NvidiaValidationSuiteTestFixture, "Gpu: local attributes and getters")
{
    SECTION("constructor initializes defaults and NVML index")
    {
        Gpu gpu(7, &dcgmOps, &handle, &system);

        REQUIRE(gpu.GetGpuId() == 7);
        REQUIRE(gpu.getDeviceIndex() == 7);
        REQUIRE(gpu.getDeviceIsSupported() == false);
        REQUIRE(gpu.getDeviceArchitecture() == 0);
        REQUIRE(gpu.GetMaxOperatingTemperature() == 0);
        REQUIRE(gpu.getDeviceSerial().empty());
        REQUIRE(gpu.getDeviceId().empty());
        REQUIRE_THROWS_MATCHES(gpu.getDeviceIndex(Gpu::NVVS_GPUENUM_LAST),
                               std::runtime_error,
                               Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                                   "Illegal enumeration method given to getDeviceIndex")));
    }

    SECTION("SetAttributes populates PCI ids and serial-backed getters")
    {
        Gpu gpu(3, &dcgmOps, &handle, &system);
        dcgmDeviceAttributes_v3 attrs {};
        attrs.version                         = dcgmDeviceAttributes_version3;
        attrs.identifiers.pciDeviceId         = 0x23350000;
        attrs.identifiers.pciSubSystemId      = 0x14590000;
        attrs.powerLimits.enforcedPowerLimit  = 350;
        attrs.settings.persistenceModeEnabled = 1;
        attrs.thermalSettings.shutdownTemp    = 95;
        attrs.thermalSettings.slowdownTemp    = 90;
        SafeCopyTo(attrs.identifiers.serial, "serial-3");
        SafeCopyTo(attrs.identifiers.brandName, "brand");
        SafeCopyTo(attrs.identifiers.uuid, "GPU-uuid");
        SafeCopyTo(attrs.identifiers.pciBusId, "0000:01:00.0");
        SafeCopyTo(attrs.identifiers.deviceName, "device");
        SafeCopyTo(attrs.identifiers.driverVersion, "555.42");

        gpu.SetAttributes(attrs);
        gpu.setDeviceIsSupported(true);
        gpu.setDeviceEntityStatus(DcgmEntityStatusDetached);

        REQUIRE(gpu.getDevicePciDeviceId() == "2335");
        REQUIRE(gpu.getDevicePciSubsystemId().empty());
        REQUIRE(gpu.getDeviceId() == "2335");
        gpu.setUseSsid(true);
        REQUIRE(gpu.getDeviceId() == "2335");
        REQUIRE(gpu.getDeviceSerial() == "serial-3");
        REQUIRE(gpu.getDeviceBrandAsString() == "brand");
        REQUIRE(gpu.getDeviceGpuUuid() == "GPU-uuid");
        REQUIRE(gpu.getDevicePciBusId() == "0000:01:00.0");
        REQUIRE(gpu.getDeviceName() == "device");
        REQUIRE(gpu.GetDriverVersion() == "555.42");
        REQUIRE(gpu.GetEnforcedPowerLimit() == 350);
        REQUIRE(gpu.PersistenceModeEnabled() == true);
        REQUIRE(gpu.getDeviceIsSupported() == true);
        REQUIRE(gpu.GetDeviceEntityStatus() == DcgmEntityStatusDetached);
        REQUIRE(gpu.GetDeviceThermals().shutdownTemp == 95);
        REQUIRE(gpu.GetDeviceThermals().slowdownTemp == 90);
        REQUIRE(gpu.GetAttributes().identifiers.pciDeviceId == attrs.identifiers.pciDeviceId);
    }
}

dcgmMigHierarchy_v2 MakeMockMigHierarchy()
{
    dcgmMigHierarchy_v2 hierarchy {};
    hierarchy.version = dcgmMigHierarchy_version2;

    if (incompatibleGpus)
    {
        hierarchy.count                                = 2;
        hierarchy.entityList[0].parent.entityId        = 0;
        hierarchy.entityList[0].parent.entityGroupId   = DCGM_FE_GPU;
        hierarchy.entityList[0].entity.entityId        = 0;
        hierarchy.entityList[0].entity.entityGroupId   = DCGM_FE_GPU_I;
        hierarchy.entityList[0].info.nvmlProfileSlices = DCGM_MAX_INSTANCES_PER_GPU / 2;
        hierarchy.entityList[1].parent.entityId        = 0;
        hierarchy.entityList[1].parent.entityGroupId   = DCGM_FE_GPU;
        hierarchy.entityList[1].entity.entityId        = 0;
        hierarchy.entityList[1].entity.entityGroupId   = DCGM_FE_GPU_I;
        hierarchy.entityList[1].info.nvmlProfileSlices = DCGM_MAX_INSTANCES_PER_GPU / 2;
    }
    else if (migEnabled)
    {
        /* Gpu::IsMigModeDiagCompatible expects compute instances (DCGM_FE_GPU_CI) under each GPU instance. */
        unsigned int const n = hierarchyCount;
        hierarchy.count      = n * 2;
        for (unsigned int i = 0; i < n; i++)
        {
            hierarchy.entityList[i].parent.entityId        = i;
            hierarchy.entityList[i].parent.entityGroupId   = DCGM_FE_GPU;
            hierarchy.entityList[i].entity.entityId        = i * DCGM_MAX_INSTANCES_PER_GPU;
            hierarchy.entityList[i].entity.entityGroupId   = DCGM_FE_GPU_I;
            hierarchy.entityList[i].info.nvmlProfileSlices = DCGM_MAX_INSTANCES_PER_GPU;
        }
        for (unsigned int i = 0; i < n; i++)
        {
            unsigned int const ci                           = n + i;
            hierarchy.entityList[ci].parent.entityId        = i * DCGM_MAX_INSTANCES_PER_GPU;
            hierarchy.entityList[ci].parent.entityGroupId   = DCGM_FE_GPU_I;
            hierarchy.entityList[ci].entity.entityId        = i * DCGM_MAX_INSTANCES_PER_GPU + 1U;
            hierarchy.entityList[ci].entity.entityGroupId   = DCGM_FE_GPU_CI;
            hierarchy.entityList[ci].info.nvmlProfileSlices = DCGM_MAX_INSTANCES_PER_GPU;
        }
    }

    return hierarchy;
}

unsigned int fieldIntVal                     = DCGM_MAX_INSTANCES_PER_GPU;
long long g_unrepairableMemoryFlagVal        = 0;
char g_graphicsPidsBlobVal                   = 0;
long long g_sramExceededVal                  = 0;
dcgmReturn_t g_sramExceededFieldStatus       = DCGM_ST_OK;
dcgmReturn_t g_graphicsPidsFieldStatus       = DCGM_ST_OK;
long long g_pendingRetirementsVal            = 0;
dcgmReturn_t g_pendingRetirementsFieldStatus = DCGM_ST_OK;
long long g_dbeVolVal                        = 0;
long long g_dbeTotalVal                      = 0;
long long g_sbeTotalVal                      = 0;
long long g_rowRemapFailedVal                = 0;
long long g_rowRemapPendingVal               = 0;
long long g_rowRemapUncorrectableTotalVal    = 0;
dcgmReturn_t g_memoryUnrepairableFieldStatus = DCGM_ST_OK;
long long g_inforomValidVal                  = 0;
dcgmReturn_t g_inforomValidFieldStatus       = DCGM_ST_OK;
long long g_gpuRecoveryActionVal             = 0;
dcgmReturn_t g_gpuRecoveryActionFieldStatus  = DCGM_ST_OK;
long long g_fmStatusVal                      = 0;
long long g_fmErrorVal                       = 0;
std::string g_fmUuidStr                      = "test-uuid";
long long g_fmCliqueIdVal                    = 0;
long long g_fmHealthMaskVal                  = 0;

void PopulateMockLatestValue(dcgmGroupEntityPair_t const &entity, unsigned short fieldId, dcgmFieldValue_v2 &value)
{
    memset(&value, 0, sizeof(value));
    value.entityGroupId = entity.entityGroupId;
    value.entityId      = entity.entityId;
    value.fieldId       = fieldId;
    value.status        = DCGM_ST_OK;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
    value.fieldType             = fieldMeta == nullptr ? DCGM_FT_INT64 : fieldMeta->fieldType;

    switch (fieldId)
    {
        case DCGM_FI_DEV_MIG_MODE:
            if (migEnabled)
            {
                if (hierarchyCount > 0U)
                {
                    value.value.i64 = (entity.entityGroupId == DCGM_FE_GPU && entity.entityId < hierarchyCount) ? 1 : 0;
                }
                else
                {
                    value.value.i64 = 1;
                }
            }
            else
            {
                value.value.i64 = 0;
            }
            break;
        case DCGM_FI_DEV_MIG_MAX_SLICES:
            value.value.i64 = fieldIntVal;
            break;
        case DCGM_FI_DEV_MEMORY_UNREPAIRABLE:
            value.status    = g_memoryUnrepairableFieldStatus;
            value.value.i64 = g_unrepairableMemoryFlagVal;
            break;
        case DCGM_FI_CUDA_DRIVER_VERSION:
            if (g_mockCudaDriverVersion128)
            {
                value.fieldType = DCGM_FT_INT64;
                value.status    = DCGM_ST_OK;
                /* Encoding matches @ref DcgmSystem::GetCudaMajorVersion (e.g. 12080 -> 12.8). */
                value.value.i64 = 12080;
            }
            break;
        case DCGM_FI_DEV_GRAPHICS_PIDS:
            value.status        = g_graphicsPidsFieldStatus;
            value.value.blob[0] = g_graphicsPidsBlobVal;
            break;
        case DCGM_FI_DEV_SRAM_EXCEEDED:
            value.status    = g_sramExceededFieldStatus;
            value.value.i64 = g_sramExceededVal;
            break;
        case DCGM_FI_DEV_PAGE_RETIRED_PENDING:
            value.status    = g_pendingRetirementsFieldStatus;
            value.value.i64 = g_pendingRetirementsVal;
            break;
        case DCGM_FI_DEV_ECC_DBE_VOL_TOTAL:
            value.value.i64 = g_dbeVolVal;
            break;
        case DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL:
            value.value.i64 = g_dbeTotalVal;
            break;
        case DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL:
            value.value.i64 = g_sbeTotalVal;
            break;
        case DCGM_FI_DEV_ROW_REMAP_FAILED:
            value.value.i64 = g_rowRemapFailedVal;
            break;
        case DCGM_FI_DEV_ROW_REMAP_PENDING:
            value.value.i64 = g_rowRemapPendingVal;
            break;
        case DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL:
            value.value.i64 = g_rowRemapUncorrectableTotalVal;
            break;
        case DCGM_FI_DEV_INFOROM_VALID:
            value.status    = g_inforomValidFieldStatus;
            value.value.i64 = g_inforomValidVal;
            break;
        case DCGM_FI_DEV_GPU_RECOVERY_ACTION:
            value.status    = g_gpuRecoveryActionFieldStatus;
            value.value.i64 = g_gpuRecoveryActionVal;
            break;
        case DCGM_FI_DEV_FABRIC_MANAGER_STATUS:
            value.value.i64 = g_fmStatusVal;
            break;
        case DCGM_FI_DEV_FABRIC_MANAGER_ERROR:
            value.value.i64 = g_fmErrorVal;
            break;
        case DCGM_FI_DEV_FABRIC_CLUSTER_UUID:
            SafeCopyTo(value.value.str, g_fmUuidStr.c_str());
            break;
        case DCGM_FI_DEV_FABRIC_CLIQUE_ID:
            value.value.i64 = g_fmCliqueIdVal;
            break;
        case DCGM_FI_DEV_FABRIC_HEALTH_MASK:
            value.value.i64 = g_fmHealthMaskVal;
            break;
    }
}

void ConfigureMockDcgmOpsForNvvsTests(DcgmNs::MockDcgmOps &mockDcgmOps)
{
    mockDcgmOps.m_migHierarchyState.handler = [](dcgmHandle_t, dcgmMigHierarchy_v2 *hierarchy) {
        *hierarchy = MakeMockMigHierarchy();
        return DCGM_ST_OK;
    };

    mockDcgmOps.m_entitiesGetLatestValuesState.handler = [&mockDcgmOps](dcgmHandle_t,
                                                                        dcgmGroupEntityPair_t entities[],
                                                                        unsigned int entityCount,
                                                                        unsigned short fieldIds[],
                                                                        unsigned int fieldCount,
                                                                        unsigned int,
                                                                        dcgmFieldValue_v2 values[]) {
        if (entities == nullptr || fieldIds == nullptr || values == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        std::optional<dcgmReturn_t> fieldReturn;
        for (unsigned int entityIndex = 0; entityIndex < entityCount; ++entityIndex)
        {
            for (unsigned int fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex)
            {
                unsigned int const valueIndex = entityIndex * fieldCount + fieldIndex;
                PopulateMockLatestValue(entities[entityIndex], fieldIds[fieldIndex], values[valueIndex]);
                if (!fieldReturn.has_value())
                {
                    if (auto it = mockDcgmOps.m_entitiesGetLatestValuesState.returnByFieldId.find(fieldIds[fieldIndex]);
                        it != mockDcgmOps.m_entitiesGetLatestValuesState.returnByFieldId.end())
                    {
                        fieldReturn = it->second;
                    }
                }
            }
        }

        return fieldReturn.value_or(mockDcgmOps.m_entitiesGetLatestValuesState.returnCode);
    };
}

TEST_CASE_METHOD(NvidiaValidationSuiteTestFixture, "Gpu: MIG helpers use DCGM field mocks")
{
    ConfigureMockDcgmOpsForNvvsTests(dcgmOps);
    Gpu gpu0(0, &dcgmOps, &handle, &system);

    SECTION("GetMigMode follows the mocked MIG mode field")
    {
        bool enabled = true;
        migEnabled   = false;
        REQUIRE(gpu0.GetMigMode(enabled) == DCGM_ST_OK);
        REQUIRE(enabled == false);

        migEnabled = true;
        REQUIRE(gpu0.GetMigMode(enabled) == DCGM_ST_OK);
        REQUIRE(enabled == true);
    }

    SECTION("IsMigModeDiagCompatible accepts full-GPU MIG layout")
    {
        incompatibleGpus = false;
        migEnabled       = true;
        hierarchyCount   = 1;
        fieldIntVal      = DCGM_MAX_INSTANCES_PER_GPU;

        auto validity = gpu0.IsMigModeDiagCompatible();

        REQUIRE(validity.migEnabled == true);
        REQUIRE(validity.migInvalidConfiguration == false);
        REQUIRE(validity.gpuInstanceCount == 1);
    }

    SECTION("IsMigModeDiagCompatible rejects partial GPU instances")
    {
        incompatibleGpus = true;
        migEnabled       = false;
        hierarchyCount   = 4;
        fieldIntVal      = DCGM_MAX_INSTANCES_PER_GPU;

        auto validity = gpu0.IsMigModeDiagCompatible();

        REQUIRE(validity.migInvalidConfiguration == true);
        REQUIRE(validity.gpuInstanceCount == 1);
    }

    incompatibleGpus = false;
    migEnabled       = false;
    hierarchyCount   = 4;
    fieldIntVal      = DCGM_MAX_INSTANCES_PER_GPU;
}

SCENARIO("NvidiaValidationSuite: RIST level-run inclusion follows configured path presence")
{
    GIVEN("a configured DCGM_RIST_BIN_PATH value")
    {
        auto const configuredPath = GENERATE(std::optional<std::string> {},
                                             std::make_optional(std::string {}),
                                             std::make_optional(std::string { " " }),
                                             std::make_optional(std::string { "relative/ristapp" }),
                                             std::make_optional(std::string { "/missing/ristapp" }),
                                             std::make_optional(std::string { "/opt/nvidia/ristapp" }));

        WHEN("level-run inclusion is evaluated")
        {
            auto const includeRist = NvidiaValidationSuite::ShouldIncludeRistInLevelRun(configuredPath);

            THEN("only non-empty configured paths are included")
            {
                CAPTURE(configuredPath, includeRist);
                CHECK(includeRist == (configuredPath.has_value() && !configuredPath->empty()));
            }
        }
    }
}

TEST_CASE_METHOD(NvidiaValidationSuiteTestFixture, "NvidiaValidationSuite: build common gpu list")
{
    ConfigureMockDcgmOpsForNvvsTests(dcgmOps);
    TestNvidiaValidationSuite nvs = MakeTestNvidiaValidationSuite();
    std::vector<unsigned int> gpuIndices;
    std::vector<Gpu *> visibleGpus;
    Gpu gpu0(0, &dcgmOps, &handle, &system);
    Gpu gpu1(1, &dcgmOps, &handle, &system);
    Gpu gpu2(2, &dcgmOps, &handle, &system);
    Gpu gpu3(3, &dcgmOps, &handle, &system);
    visibleGpus.push_back(&gpu0);
    visibleGpus.push_back(&gpu1);
    visibleGpus.push_back(&gpu2);
    visibleGpus.push_back(&gpu3);

    incompatibleGpus = false;
    migEnabled       = false;
    hierarchyCount   = 4;
    fieldIntVal      = DCGM_MAX_INSTANCES_PER_GPU;

    std::string errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(errorString.empty());
    for (unsigned int i = 0; i < 4; i++)
    {
        REQUIRE(gpuIndices[i] == i);
    }

    // Set up gpuIndices for the next test
    gpuIndices.clear();
    gpuIndices.push_back(3);
    errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(errorString.empty());
    gpuIndices.clear();

    gpuIndices.push_back(1);
    gpuIndices.push_back(2);
    errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(errorString.empty());

    // Make sure we pass with 1 MIG GPU
    migEnabled = true;
    gpuIndices.clear();
    gpuIndices.push_back(0);
    incompatibleGpus = false;
    errorString      = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(errorString.empty());

    // Make sure we fail if the slice configuration is wrong with 1 GPU
    incompatibleGpus = true;
    gpuIndices.clear();
    gpuIndices.push_back(0);
    errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(!errorString.empty());
    REQUIRE(errorString.find("MIG configuration is incompatible with the diagnostic because it prevents")
            != std::string::npos);

    // Make sure we fail if the slice configuration is good, but we have more than 1 GPU
    migEnabled       = true;
    incompatibleGpus = false;
    gpuIndices.clear();
    gpuIndices.push_back(0);
    gpuIndices.push_back(1);
    errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(!errorString.empty());
    REQUIRE(errorString.find("CUDA before 12.8 does not support enumerating GPUs with MIG mode enabled")
            != std::string::npos);

    // Make sure we fail if we have a GPU in MIG mode are we are running on more than 1 GPU even if it isn't in the list
    migEnabled       = true;
    incompatibleGpus = false;
    gpuIndices.clear();
    gpuIndices.push_back(1);
    gpuIndices.push_back(2);
    errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(!errorString.empty());
    REQUIRE(errorString.find("CUDA before 12.8 does not support enumerating GPUs with MIG mode enabled")
            != std::string::npos);

    // CUDA 12.8+ (driver-reported): multi-GPU + MIG is allowed when instances cover the full GPU
    {
        g_mockCudaDriverVersion128          = true;
        unsigned int const savedFieldIntVal = fieldIntVal;
        DcgmNs::Defer resetCudaDriverMock([&] {
            fieldIntVal                = savedFieldIntVal;
            g_mockCudaDriverVersion128 = false;
        });
        fieldIntVal                          = DCGM_MAX_INSTANCES_PER_GPU;
        TestNvidiaValidationSuite cuda128Nvs = MakeTestNvidiaValidationSuite();

        migEnabled       = true;
        incompatibleGpus = false;
        hierarchyCount   = 4;
        gpuIndices.clear();
        gpuIndices.push_back(0);
        gpuIndices.push_back(1);
        errorString = cuda128Nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
        REQUIRE(errorString.empty());
    }

    // Make sure we fail if there are GPUs on the system that aren't MIG enabled with GPUs that are MIG enabled
    migEnabled       = true;
    incompatibleGpus = false;
    hierarchyCount   = 2; // this makes 2 GPUs that are MIG enabled and two that aren't
    gpuIndices.clear();
    gpuIndices.push_back(0);
    errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(!errorString.empty());
    REQUIRE(
        errorString
        == "Cannot run diagnostic: CUDA does not support enumerating GPUs when one more GPUs has MIG mode enabled and one or more GPUs has MIG mode disabled.");

    // Fail when MIG is enabled but no compute instances are present
    migEnabled       = true;
    incompatibleGpus = false;
    hierarchyCount   = 0;
    gpuIndices.clear();
    gpuIndices.push_back(0);
    gpuIndices.push_back(1);
    errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(!errorString.empty());

    const char *err = "Cannot run diagnostic: MIG is enabled, but no compute instances are configured."
                      " CUDA needs to execute on a compute instance if MIG is enabled.";
    REQUIRE(errorString == err);
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture, "NVVS correctly processes command line arguments")
{
    GIVEN("default values")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();
        ProcessCommandLine(nvvs, { "nvvs" });

        CHECK(nvvs.WrapperGetListGpus() == false);
        CHECK(nvvs.WrapperGetListTests() == false);
        CHECK(nvvsCommon.verbose == false);
        CHECK(nvvsCommon.pluginPath == "");
        CHECK(nvvsCommon.parse == false);
        CHECK(nvvsCommon.quietMode == false);
        CHECK(nvvsCommon.configless == false);
        CHECK(nvvsCommon.statsOnlyOnFail == false);
        CHECK(nvvsCommon.indexString == "");
        CHECK(nvvsCommon.dcgmHostname == "");
        CHECK(nvvsCommon.m_statsPath == "./");
        CHECK(nvvsCommon.watchFrequency == 5000000);
        CHECK(nvvsCommon.channelFd == -1);
    }

    GIVEN("Non-default paramters")
    {
        auto statsDir = TempDir::Random("dcgm-diag-test-stats-dir-");

        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        ProcessCommandLine(nvvs,
                           { "nvvs",
                             "-g",
                             "-v",
                             "-t",
                             "--statsonfail",
                             "--hwdiaglogfile",
                             "hwlogfile",
                             "--specifiedtest",
                             "-s",
                             "--quiet",
                             "--pluginpath",
                             "plugins",
                             "-i",
                             "1",
                             "-d",
                             "2",
                             "--configless",
                             "-c",
                             "-a",
                             "--statspath",
                             statsDir.Path().string(),
                             "--parameters",
                             "sm stress.test_duration=5",
                             "--dcgmHostname",
                             "host",
                             "--channel-fd",
                             "3",
                             "--clocksevent-mask",
                             "--throttle-mask",
                             "--fail-early",
                             "--check-interval",
                             "3",
                             "-l",
                             "logfile",
                             "--watch-frequency",
                             "1000000" });

        CHECK(nvvs.WrapperGetListGpus() == true);
        CHECK(nvvs.WrapperGetListTests() == true);
        CHECK(nvvsCommon.verbose == true);
        CHECK(nvvsCommon.pluginPath == "plugins");
        CHECK(nvvsCommon.quietMode == true);
        CHECK(nvvsCommon.configless == true);
        CHECK(nvvsCommon.statsOnlyOnFail == true);
        CHECK(nvvsCommon.indexString == "1");
        CHECK(nvvsCommon.dcgmHostname == "host");
        CHECK(nvvsCommon.m_statsPath == statsDir.Path().string());
        CHECK(nvvsCommon.watchFrequency == 1000000);
        CHECK(nvvsCommon.channelFd == 3);
    }

    GIVEN("Conflicting throttle and clocks events")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        CHECK_THROWS((ProcessCommandLine(
            nvvs, { "nvvs", "--throttle-mask", "hw_power_brake", "--clocksevent-mask", "sw_thermal" })));
    }

    GIVEN("Invalid throttle and clock events")
    {
        {
            TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

            ProcessCommandLine(nvvs, { "nvvs", "--throttle-mask", "invalid" });
            CHECK(nvvsCommon.clocksEventIgnoreMask == DCGM_INT64_BLANK);
        }

        {
            TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

            ProcessCommandLine(nvvs, { "nvvs", "--clocksevent-mask", "sonotvalid" });
            CHECK(nvvsCommon.clocksEventIgnoreMask == DCGM_INT64_BLANK);
        }
    }

    GIVEN("Revert to default value for invalid user-set watch frequencies")
    {
        {
            TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

            ProcessCommandLine(nvvs, { "nvvs", "--watch-frequency", "99999" });
            CHECK(nvvsCommon.watchFrequency == 5000000);
        }
        {
            TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

            ProcessCommandLine(nvvs, { "nvvs", "--watch-frequency", "60000001" });
            CHECK(nvvsCommon.watchFrequency == 5000000);
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture, "findTestName finds the test if it exists", "[.]")
{
    TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();
    nvvs.WrapperEnumerateAllVisibleTests();
    // We need to load the plugins for this to work. So we will only test the negative cases
    // std::vector<Test *>::iterator sm_stress = nvvs.WrapperFindTestName("SM Stress");
    std::vector<Test *>::iterator skip        = nvvs.WrapperFindTestName("Skip");
    std::vector<Test *>::iterator nonexistent = nvvs.WrapperFindTestName("Nonexistent");
    CHECK(skip == nonexistent);
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture, "overrideparameters overrides the correct parameters", "[.]")
{
    std::string paramString = "sm stress.test_duration=5";
    // Mock the parameters being returned from a plugin
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> params;
    dcgmDiagPluginParameterInfo_t info;
    snprintf(info.parameterName, sizeof(info.parameterName), "test_duration");
    info.parameterType = DcgmPluginParamInt;
    params["sm stress"].push_back(info);
    // Create the ParameterValidator from the mocked paramaters
    ParameterValidator pv = ParameterValidator(params);
    TestParameters tp;
    tp.AddString("test_duration", "0");


    TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();
    nvvs.WrapperInitializeParameters(paramString, pv);

    nvvs.WrapperOverrideParameters(&tp, "sm_stress");
    CHECK(tp.GetString("test_duration") == "5");
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture, "InitializeParameters accepts only valid global parameters")
{
    std::string paramString = "generic_mode=true";
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> params;
    ParameterValidator pv = ParameterValidator(params);
    TestParameters tp;
    tp.AddString(PS_USE_GENERIC_MODE, "false");


    TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();
    nvvs.WrapperInitializeParameters(paramString, pv);
    REQUIRE(nvvsCommon.genericMode == true);

    // Test with an invalid global parameter
    paramString = "planet=roshar";
    REQUIRE_THROWS_MATCHES(nvvs.WrapperInitializeParameters(paramString, pv),
                           std::runtime_error,
                           Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                               "unable to parse test, parameter name, and value from string 'planet=roshar'")));

    paramString = "cpu_sdc.is_allowed=false";
    try
    {
        nvvs.WrapperInitializeParameters(paramString, pv);
        FAIL("Expected cpu_sdc parameter validation to throw.");
    }
    catch (std::runtime_error const &ex)
    {
        std::string const errorMessage = ex.what();
        CHECK(errorMessage.find("dependent package is installed") != std::string::npos);
        CHECK(errorMessage.find("DCGM proprietary package") == std::string::npos);
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture,
                "ThrowTestNotFoundException reports cpu_sdc as a missing dependent package")
{
    TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();
    nvvs.WrapperEnumerateAllVisibleTests();
    nvvsCommon.desiredTest.insert(CPU_SDC_PLUGIN_NAME);

    try
    {
        nvvs.WrapperThrowTestNotFoundException();
        FAIL("Expected missing cpu_sdc to throw.");
    }
    catch (std::runtime_error const &ex)
    {
        std::string const errorMessage = ex.what();
        CHECK(errorMessage.find("[cpu_sdc] cannot find dependent package") != std::string::npos);
        CHECK(errorMessage.find("dependent package is installed") != std::string::npos);
        CHECK(errorMessage.find("DCGM proprietary package") == std::string::npos);
    }
}

void TestNvidiaValidationSuite::WrapperProcessCommandLine(int argc, char *argv[])
{
    processCommandLine(argc, argv);
}

bool TestNvidiaValidationSuite::WrapperGetListGpus()
{
    return listGpus;
}

bool TestNvidiaValidationSuite::WrapperGetListTests()
{
    return listTests;
}

void TestNvidiaValidationSuite::WrapperInitializeParameters(const std::string &parms, const ParameterValidator &pv)
{
    InitializeParameters(parms, pv);
}

void TestNvidiaValidationSuite::WrapperThrowTestNotFoundException() const
{
    ThrowTestNotFoundExecption();
}

void TestNvidiaValidationSuite::WrapperEnumerateAllVisibleTests()
{
    std::vector<std::unique_ptr<EntitySet>> entitySet;
    m_tf = new TestFramework(entitySet, &m_dcgmOps, m_nvvsCommon, &m_dcgmHandle, &m_dcgmSystem);
    enumerateAllVisibleTests();
}

std::vector<Test *>::iterator TestNvidiaValidationSuite::WrapperFindTestName(std::string test)
{
    return FindTestName(std::move(test));
}

void TestNvidiaValidationSuite::WrapperOverrideParameters(TestParameters *tp, const std::string &lowerCaseTestName)
{
    overrideParameters(tp, lowerCaseTestName);
}

bool TestNvidiaValidationSuite::WrapperIsGpuIncluded(unsigned int gpuIndex, std::vector<unsigned int> &gpuIndices) const
{
    return IsGpuIncluded(gpuIndex, gpuIndices);
}

bool TestNvidiaValidationSuite::WrapperHasGenericSupport(std::string const &gpuBrand, uint64_t gpuArch)
{
    return HasGenericSupport(gpuBrand, gpuArch);
}

void TestNvidiaValidationSuite::WrapperSetCudaVisibleDevicesForTesting(std::vector<unsigned int> const &gpuIndices)
{
    SetCudaVisibleDevicesForTesting(gpuIndices);
}

void TestNvidiaValidationSuite::WrapperAddGpuForTesting(std::unique_ptr<Gpu> gpu)
{
    m_gpuVect.push_back(gpu.release());
}

void TestNvidiaValidationSuite::WrapperEnsureTestFramework()
{
    if (m_tf != nullptr)
    {
        return;
    }

    std::vector<std::unique_ptr<EntitySet>> entitySet;
    m_tf = new TestFramework(entitySet, &m_dcgmOps, m_nvvsCommon, &m_dcgmHandle, &m_dcgmSystem);
}

void TestNvidiaValidationSuite::WrapperSetTestVector(std::vector<Test *> tests)
{
    WrapperEnsureTestFramework();
    testVect = std::move(tests);
}

void TestNvidiaValidationSuite::WrapperValidateSubtestParameters()
{
    ValidateSubtestParameters();
}

void TestNvidiaValidationSuite::WrapperBanner()
{
    banner();
}

std::string const &TestNvidiaValidationSuite::WrapperGetConfigFile() const
{
    return configFile;
}

std::string const &TestNvidiaValidationSuite::WrapperGetDebugFile() const
{
    return debugFile;
}

std::string const &TestNvidiaValidationSuite::WrapperGetDebugLogLevel() const
{
    return debugLogLevel;
}

std::string const &TestNvidiaValidationSuite::WrapperGetHwDiagLogFile() const
{
    return hwDiagLogFile;
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture,
                "NvidiaValidationSuite: helper methods preserve GPU selection behavior")
{
    TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

    GIVEN("explicit and implicit GPU inclusion lists")
    {
        std::vector<unsigned int> emptySelection;
        std::vector<unsigned int> selectedGpus { 2, 4 };

        WHEN("inclusion is evaluated for empty and populated selections")
        {
            auto const emptyIncludesNine     = nvvs.WrapperIsGpuIncluded(9, emptySelection);
            auto const selectedIncludesTwo   = nvvs.WrapperIsGpuIncluded(2, selectedGpus);
            auto const selectedIncludesThree = nvvs.WrapperIsGpuIncluded(3, selectedGpus);

            THEN("an empty selection includes every GPU and a populated selection is exact")
            {
                CHECK(emptyIncludesNine);
                CHECK(selectedIncludesTwo);
                CHECK_FALSE(selectedIncludesThree);
            }
        }
    }

    GIVEN("GPU brand and encoded architecture combinations")
    {
        WHEN("generic support is evaluated")
        {
            auto const teslaKepler   = nvvs.WrapperHasGenericSupport("Tesla", 3ULL << 16);
            auto const quadroMaxwell = nvvs.WrapperHasGenericSupport("Quadro", 5ULL << 16);
            auto const geForceNone   = nvvs.WrapperHasGenericSupport("GeForce", 0);
            auto const teslaUnknown  = nvvs.WrapperHasGenericSupport("Tesla", 0);

            THEN("Maxwell-or-newer and Tesla Kepler-or-newer are accepted; unavailable architecture is rejected")
            {
                CHECK(teslaKepler);
                CHECK(quadroMaxwell);
                CHECK_FALSE(geForceNone);
                CHECK_FALSE(teslaUnknown);
            }
        }
    }

    GIVEN("GPU UUIDs available to CUDA_VISIBLE_DEVICES setup")
    {
        ScopedEnvVars env({ { "CUDA_VISIBLE_DEVICES", "existing" } });
        auto attributes0 = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        SafeCopyTo(attributes0.identifiers.uuid, "GPU-wrapper-0");
        auto gpu0 = std::make_unique<Gpu>(0, &dcgmOps, &handle, &system);
        gpu0->SetAttributes(attributes0);
        nvvs.WrapperAddGpuForTesting(std::move(gpu0));

        auto attributes1 = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        SafeCopyTo(attributes1.identifiers.uuid, "GPU-wrapper-1");
        auto gpu1 = std::make_unique<Gpu>(1, &dcgmOps, &handle, &system);
        gpu1->SetAttributes(attributes1);
        nvvs.WrapperAddGpuForTesting(std::move(gpu1));

        WHEN("a matching GPU selection is applied")
        {
            nvvs.WrapperSetCudaVisibleDevicesForTesting({ 1, 0 });

            THEN("CUDA_VISIBLE_DEVICES is rewritten in the requested order")
            {
                REQUIRE(std::getenv("CUDA_VISIBLE_DEVICES") != nullptr);
                CHECK(std::string(std::getenv("CUDA_VISIBLE_DEVICES")) == "GPU-wrapper-1,GPU-wrapper-0");
            }
        }

        WHEN("the requested GPU does not exist")
        {
            nvvs.WrapperSetCudaVisibleDevicesForTesting({ 8 });

            THEN("the existing CUDA_VISIBLE_DEVICES value is preserved")
            {
                REQUIRE(std::getenv("CUDA_VISIBLE_DEVICES") != nullptr);
                CHECK(std::string(std::getenv("CUDA_VISIBLE_DEVICES")) == "existing");
            }
        }

        WHEN("the requested GPU list is empty")
        {
            nvvs.WrapperSetCudaVisibleDevicesForTesting({});

            THEN("the existing CUDA_VISIBLE_DEVICES value is preserved")
            {
                REQUIRE(std::getenv("CUDA_VISIBLE_DEVICES") != nullptr);
                CHECK(std::string(std::getenv("CUDA_VISIBLE_DEVICES")) == "existing");
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture, "NvidiaValidationSuite: initializes multiple parameter overrides")
{
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> params;
    dcgmDiagPluginParameterInfo_t duration {};
    snprintf(duration.parameterName, sizeof(duration.parameterName), "test_duration");
    duration.parameterType = DcgmPluginParamInt;
    dcgmDiagPluginParameterInfo_t boolParam {};
    snprintf(boolParam.parameterName, sizeof(boolParam.parameterName), "use_tensor_cores");
    boolParam.parameterType = DcgmPluginParamBool;
    params["diagnostic"].push_back(duration);
    params["diagnostic"].push_back(boolParam);
    ParameterValidator pv(params);

    TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

    GIVEN("two plugin parameters and one global parameter")
    {
        std::string paramString = "diagnostic.test_duration=12;diagnostic.use_tensor_cores=true;generic_mode=true";

        WHEN("the parameters are initialized and applied")
        {
            nvvs.WrapperEnsureTestFramework();
            nvvs.WrapperInitializeParameters(paramString, pv);
            TestParameters tp;
            tp.AddString("test_duration", "0");
            tp.AddString("use_tensor_cores", "false");
            nvvs.WrapperOverrideParameters(&tp, "diagnostic");

            THEN("plugin overrides and global mode are both visible")
            {
                CHECK(tp.GetString("test_duration") == "12");
                CHECK(tp.GetString("use_tensor_cores") == "true");
                CHECK(nvvsCommon.genericMode == true);
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture, "NVVS: command line parsing records diagnostic orchestration options")
{
    GIVEN("a command line with iteration, hang-detect, and ignore-code options")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs,
                               { "nvvs",
                                 "--specifiedtest",
                                 "diagnostic,pcie",
                                 "--ignoreErrorCodes",
                                 "gpu0:28",
                                 "--current-iteration",
                                 "2",
                                 "--total-iterations",
                                 "4",
                                 "--entity-id",
                                 "gpu:0,cpu:1",
                                 "--response-version",
                                 "11",
                                 "--rerun-as-root",
                                 "--noHangDetect",
                                 "--hangDetectExpirySec",
                                 "42",
                                 "--fail-early" });

            THEN("the global NVVS state reflects those options")
            {
                CHECK(nvvsCommon.desiredTest.contains("diagnostic"));
                CHECK(nvvsCommon.desiredTest.contains("pcie"));
                CHECK(nvvsCommon.ignoreErrorCodesString == "gpu0:28");
                CHECK(nvvsCommon.currentIteration == 2);
                CHECK(nvvsCommon.totalIterations == 4);
                CHECK(nvvsCommon.entityIds == "gpu:0,cpu:1");
                CHECK(nvvsCommon.diagResponseVersion == 11);
                CHECK(nvvsCommon.rerunAsRoot);
                CHECK(nvvsCommon.hangDetectDisable);
                CHECK(nvvsCommon.hangDetectExpirySec == 42);
                CHECK(nvvsCommon.failEarly);
                CHECK(nvvsCommon.failCheckInterval == 5);
            }
        }
    }

    GIVEN("list-only and watch-frequency inclusive boundary options")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the lower watch-frequency boundary is requested with --listTests")
        {
            ProcessCommandLine(nvvs, { "nvvs", "--listTests", "--watch-frequency", "100000" });

            THEN("listing forces configless mode and accepts the inclusive lower bound")
            {
                CHECK(nvvs.WrapperGetListTests());
                CHECK(nvvsCommon.configless);
                CHECK(nvvsCommon.watchFrequency == 100000);
            }
        }

        WHEN("the upper watch-frequency boundary is requested with --listTests")
        {
            ProcessCommandLine(nvvs, { "nvvs", "--listTests", "--watch-frequency", "60000000" });

            THEN("listing forces configless mode and accepts the inclusive upper bound")
            {
                CHECK(nvvs.WrapperGetListTests());
                CHECK(nvvsCommon.configless);
                CHECK(nvvsCommon.watchFrequency == 60000000);
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture,
                "NVVS: command-line parsing handles logging environment and clock masks")
{
    GIVEN("NVVS logging environment variables without command-line overrides")
    {
        ScopedEnvVars env({ { "__NVVS_DBG_FILE", "env-nvvs.log" }, { "__NVVS_DBG_LVL", "INFO" } });
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs, { "nvvs" });

            THEN("the debug log settings come from the guarded environment")
            {
                CHECK(nvvs.WrapperGetDebugFile() == "env-nvvs.log");
                CHECK(nvvs.WrapperGetDebugLogLevel() == "INFO");
            }
        }
    }

    GIVEN("NVVS logging environment variables and explicit command-line overrides")
    {
        ScopedEnvVars env({ { "__NVVS_DBG_FILE", "env-nvvs.log" }, { "__NVVS_DBG_LVL", "INFO" } });
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs, { "nvvs", "--debugLogFile", "arg-nvvs.log", "--debugLevel", "WARN" });

            THEN("the explicit debug arguments win over the environment")
            {
                CHECK(nvvs.WrapperGetDebugFile() == "arg-nvvs.log");
                CHECK(nvvs.WrapperGetDebugLogLevel() == "WARN");
            }
        }
    }

    GIVEN("the deprecated throttle mask is used by itself")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs, { "nvvs", "--throttle-mask", "hw_power_brake" });

            THEN("it populates the same clocks-event ignore mask used by the replacement option")
            {
                CHECK(nvvsCommon.clocksEventIgnoreMask == DCGM_CLOCKS_EVENT_REASON_HW_POWER_BRAKE);
            }
        }
    }

    GIVEN("a numeric clocks-event mask")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs, { "nvvs", "--clocksevent-mask", "40" });

            THEN("the numeric mask is preserved")
            {
                CHECK(nvvsCommon.clocksEventIgnoreMask == 40);
            }
        }
    }

    GIVEN("GPU listing mode")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs, { "nvvs", "--listGpus" });

            THEN("the suite switches to configless listing without enabling test listing")
            {
                CHECK(nvvs.WrapperGetListGpus());
                CHECK_FALSE(nvvs.WrapperGetListTests());
                CHECK(nvvsCommon.configless);
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture,
                "NVVS: command-line parsing captures file, fake-GPU, and related options")
{
    GIVEN("explicit config, log, and fake GPU options")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs,
                               { "nvvs",
                                 "--config",
                                 "suite.yaml",
                                 "--debugLogFile",
                                 "debug.log",
                                 "--debugLevel",
                                 "debug",
                                 "--hwdiaglogfile",
                                 "hwdiag.log",
                                 "-f",
                                 "8,9",
                                 "--scriptable" });

            THEN("file targets and orchestration settings are stored on the suite and NvvsCommon")
            {
                CHECK(nvvs.WrapperGetConfigFile() == "suite.yaml");
                CHECK(nvvs.WrapperGetDebugFile() == "debug.log");
                CHECK(nvvs.WrapperGetDebugLogLevel() == "debug");
                CHECK(nvvs.WrapperGetHwDiagLogFile() == "hwdiag.log");
                CHECK(nvvsCommon.fakegpusString == "8,9");
                CHECK(nvvsCommon.parse);
            }
        }
    }

    GIVEN("fail-early with an explicit check interval")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs, { "nvvs", "--fail-early", "--check-interval", "300" });

            THEN("early checking uses the requested interval")
            {
                CHECK(nvvsCommon.failEarly);
                CHECK(nvvsCommon.failCheckInterval == 300);
            }
        }
    }

    GIVEN("a check interval without fail-early")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

        WHEN("the command line is processed")
        {
            ProcessCommandLine(nvvs, { "nvvs", "--check-interval", "44" });

            THEN("the interval is ignored because early checking is disabled")
            {
                CHECK_FALSE(nvvsCommon.failEarly);
                CHECK(nvvsCommon.failCheckInterval == 5);
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture, "NVVS: parameter initialization reports missing EUD dependencies")
{
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> params;
    ParameterValidator pv(params);
    TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

    WHEN("an EUD parameter is initialized without the plugin registered")
    {
        THEN("the EUD-specific diagnostic hints at the dependent packages")
        {
            REQUIRE_THROWS_MATCHES(nvvs.WrapperInitializeParameters("eud.test_duration=1", pv),
                                   NvvsException,
                                   Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                                       "dependent package and the DCGM proprietary package are installed")));
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture,
                "NVVS: parameter initialization validates subtest domains and global booleans")
{
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> params;
    dcgmDiagPluginParameterInfo_t info {};
    snprintf(info.parameterName, sizeof(info.parameterName), "test_pinned");
    info.parameterType = DcgmPluginParamBool;
    params["pcie"].push_back(info);
    snprintf(info.parameterName, sizeof(info.parameterName), "test_unpinned");
    info.parameterType = DcgmPluginParamBool;
    params["pcie"].push_back(info);
    ParameterValidator pv(params);
    TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();

    GIVEN("a valid subtest parameter and a false generic-mode flag")
    {
        WHEN("the parameters are initialized")
        {
            nvvs.WrapperEnsureTestFramework();
            nvvs.WrapperInitializeParameters("pcie.test_pinned.test_unpinned=true;generic_mode=false", pv);

            THEN("the plugin override is retained and generic mode remains disabled")
            {
                REQUIRE(nvvsCommon.parms.contains("pcie"));
                CHECK(nvvsCommon.parms["pcie"]["test_pinned.test_unpinned"] == "true");
                CHECK_FALSE(nvvsCommon.genericMode);
            }
        }
    }

    GIVEN("an invalid subtest name")
    {
        WHEN("initialization is attempted")
        {
            THEN("initialization reports the missing subtest precisely")
            {
                REQUIRE_THROWS_MATCHES(nvvs.WrapperInitializeParameters("pcie.missing.param=true", pv),
                                       std::runtime_error,
                                       Catch::Matchers::MessageMatches(
                                           Catch::Matchers::ContainsSubstring("test 'pcie' has no subtest 'missing'")));
            }
        }
    }

    GIVEN("an invalid subtest parameter")
    {
        WHEN("initialization is attempted")
        {
            THEN("initialization reports the invalid parameter precisely")
            {
                REQUIRE_THROWS_MATCHES(nvvs.WrapperInitializeParameters("pcie.test_pinned.missing=true", pv),
                                       std::runtime_error,
                                       Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                                           "subtest 'test_pinned' has no parameter 'missing'")));
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture, "NvidiaValidationSuite: banner emits the suite identity")
{
    GIVEN("stdout is captured")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();
        CoutCapture capture;

        WHEN("the banner is printed")
        {
            nvvs.WrapperBanner();
            auto const output = capture.GetOutput();

            THEN("the public suite name and version label are observable")
            {
                CHECK_THAT(output, Catch::Matchers::ContainsSubstring("DCGM Diagnostic"));
                CHECK_THAT(output, Catch::Matchers::ContainsSubstring("(version "));
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture,
                "NvidiaValidationSuite: reports unresolved requested tests through public diagnostics")
{
    GIVEN("an NVVS wrapper with a loaded pcie test and unresolved desired tests")
    {
        Test pcie(0, "pcie", PCIE_PLUGIN_NAME, "pcie", DCGM_FE_GPU, "integration");
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();
        nvvs.WrapperSetTestVector({ &pcie });

        WHEN("the requested test exists but no supported entity was assigned")
        {
            nvvsCommon.desiredTest.insert(PCIE_PLUGIN_NAME);

            THEN("the diagnostic reports an unsupported entity rather than an unknown test")
            {
                REQUIRE_THROWS_MATCHES(nvvs.WrapperThrowTestNotFoundException(),
                                       NvvsException,
                                       Catch::Matchers::MessageMatches(
                                           Catch::Matchers::ContainsSubstring("cannot find supported entity to test")));
            }
        }

        WHEN("the requested test is not part of the loaded test catalog")
        {
            nvvsCommon.desiredTest.insert("missing_test");

            THEN("the diagnostic reports the missing test name")
            {
                REQUIRE_THROWS_MATCHES(nvvs.WrapperThrowTestNotFoundException(),
                                       NvvsException,
                                       Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                                           "were not found among possible test choices")));
            }
        }

        WHEN("the requested test is EUD and the plugin is unavailable")
        {
            nvvsCommon.desiredTest.insert(EUD_PLUGIN_NAME);

            THEN("the diagnostic reports the missing dependent packages")
            {
                REQUIRE_THROWS_MATCHES(nvvs.WrapperThrowTestNotFoundException(),
                                       NvvsException,
                                       Catch::Matchers::MessageMatches(
                                           Catch::Matchers::ContainsSubstring("cannot find dependent package")));
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture,
                "NvidiaValidationSuite: validates subtest parameters through the framework catalog")
{
    GIVEN("an NVVS wrapper with an empty in-memory framework")
    {
        TestNvidiaValidationSuite nvvs = MakeTestNvidiaValidationSuite();
        nvvs.WrapperSetTestVector({});

        WHEN("only a global parameter is requested")
        {
            nvvsCommon.parmsString = "generic_mode=true";
            nvvs.WrapperValidateSubtestParameters();

            THEN("the global value is applied through the validation path")
            {
                CHECK(nvvsCommon.genericMode);
            }
        }

        WHEN("a catalogued software plugin parameter is requested")
        {
            nvvsCommon.parmsString = "software.require_persistence_mode=true";
            nvvs.WrapperValidateSubtestParameters();

            THEN("the plugin parameter is retained in NvvsCommon")
            {
                REQUIRE(nvvsCommon.parms.contains("software"));
                CHECK(nvvsCommon.parms["software"]["require_persistence_mode"] == "true");
            }
        }

        WHEN("a plugin parameter is requested without a catalog entry")
        {
            nvvsCommon.parmsString = "pcie.test_pinned=true";

            THEN("the validator reports the unknown plugin parameter")
            {
                REQUIRE_THROWS_MATCHES(nvvs.WrapperValidateSubtestParameters(),
                                       std::runtime_error,
                                       Catch::Matchers::MessageMatches(
                                           Catch::Matchers::ContainsSubstring("does not match any loaded tests")));
            }
        }
    }
}

SCENARIO_METHOD(NvidiaValidationSuiteTestFixture,
                "BuildCudaVisibleDevicesString creates correct CUDA_VISIBLE_DEVICES strings")
{
    GIVEN("Empty GPU indices")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        std::vector<unsigned int> gpuIndices;
        std::vector<Gpu *> gpuVect;

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It returns an empty string")
            {
                CHECK(result.empty());
            }
        }
    }

    GIVEN("Single GPU")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        auto gpu0 = CreateGpu(0, "GPU-00000000-0000-0000-0000-000000000000");
        std::vector<Gpu *> gpuVect { gpu0.get() };
        std::vector<unsigned int> gpuIndices { 0 };

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It returns the single GPU UUID")
            {
                CHECK(result == "GPU-00000000-0000-0000-0000-000000000000");
            }
        }
    }

    GIVEN("Multiple GPUs in sequential order")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        auto gpu0 = CreateGpu(0, "GPU-00000000-0000-0000-0000-000000000000");
        auto gpu1 = CreateGpu(1, "GPU-11111111-1111-1111-1111-111111111111");
        auto gpu2 = CreateGpu(2, "GPU-22222222-2222-2222-2222-222222222222");
        auto gpu3 = CreateGpu(3, "GPU-33333333-3333-3333-3333-333333333333");

        std::vector<Gpu *> gpuVect { gpu0.get(), gpu1.get(), gpu2.get(), gpu3.get() };
        std::vector<unsigned int> gpuIndices { 0, 1, 2, 3 };

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It returns comma-separated UUIDs in order")
            {
                CHECK(
                    result
                    == "GPU-00000000-0000-0000-0000-000000000000,GPU-11111111-1111-1111-1111-111111111111,GPU-22222222-2222-2222-2222-222222222222,GPU-33333333-3333-3333-3333-333333333333");
            }
        }
    }

    GIVEN("Multiple GPUs in non-sequential order")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        auto gpu0 = CreateGpu(0, "GPU-00000000-0000-0000-0000-000000000000");
        auto gpu1 = CreateGpu(1, "GPU-11111111-1111-1111-1111-111111111111");
        auto gpu2 = CreateGpu(2, "GPU-22222222-2222-2222-2222-222222222222");
        auto gpu3 = CreateGpu(3, "GPU-33333333-3333-3333-3333-333333333333");

        std::vector<Gpu *> gpuVect { gpu0.get(), gpu1.get(), gpu2.get(), gpu3.get() };
        std::vector<unsigned int> gpuIndices { 3, 1 };

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It returns UUIDs in the order of gpuIndices")
            {
                CHECK(result == "GPU-33333333-3333-3333-3333-333333333333,GPU-11111111-1111-1111-1111-111111111111");
            }
        }
    }

    GIVEN("GPU index not found in gpuVect")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        auto gpu0 = CreateGpu(0, "GPU-00000000-0000-0000-0000-000000000000");
        auto gpu1 = CreateGpu(1, "GPU-11111111-1111-1111-1111-111111111111");

        std::vector<Gpu *> gpuVect { gpu0.get(), gpu1.get() };
        std::vector<unsigned int> gpuIndices { 0, 5, 1 }; // GPU 5 doesn't exist

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It skips the missing GPU and includes valid ones")
            {
                CHECK(result == "GPU-00000000-0000-0000-0000-000000000000,GPU-11111111-1111-1111-1111-111111111111");
            }
        }
    }

    GIVEN("All GPU indices not found in gpuVect")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        auto gpu0 = CreateGpu(0, "GPU-00000000-0000-0000-0000-000000000000");
        auto gpu1 = CreateGpu(1, "GPU-11111111-1111-1111-1111-111111111111");

        std::vector<Gpu *> gpuVect { gpu0.get(), gpu1.get() };
        std::vector<unsigned int> gpuIndices { 5, 6, 7 }; // None exist

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It returns an empty string")
            {
                CHECK(result.empty());
            }
        }
    }

    GIVEN("gpuVect with nullptr entries")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        auto gpu0 = CreateGpu(0, "GPU-00000000-0000-0000-0000-000000000000");
        auto gpu2 = CreateGpu(2, "GPU-22222222-2222-2222-2222-222222222222");

        std::vector<Gpu *> gpuVect { gpu0.get(), nullptr, gpu2.get() };
        std::vector<unsigned int> gpuIndices { 0, 2 };

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It handles nullptr gracefully and returns valid UUIDs")
            {
                CHECK(result == "GPU-00000000-0000-0000-0000-000000000000,GPU-22222222-2222-2222-2222-222222222222");
            }
        }
    }

    GIVEN("Testing GPU 3 only on a 4-GPU system")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        auto gpu0 = CreateGpu(0, "GPU-00000000-0000-0000-0000-000000000000");
        auto gpu1 = CreateGpu(1, "GPU-11111111-1111-1111-1111-111111111111");
        auto gpu2 = CreateGpu(2, "GPU-22222222-2222-2222-2222-222222222222");
        auto gpu3 = CreateGpu(3, "GPU-33333333-3333-3333-3333-333333333333");

        std::vector<Gpu *> gpuVect { gpu0.get(), gpu1.get(), gpu2.get(), gpu3.get() };
        std::vector<unsigned int> gpuIndices { 3 };

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It returns only GPU 3's UUID")
            {
                CHECK(result == "GPU-33333333-3333-3333-3333-333333333333");
            }
        }
    }

    GIVEN("Non-contiguous GPU selection (GPUs 1 and 3)")
    {
        auto testUuidGetter = [this](Gpu *gpu) {
            return GetUuid(gpu);
        };

        auto gpu0 = CreateGpu(0, "GPU-00000000-0000-0000-0000-000000000000");
        auto gpu1 = CreateGpu(1, "GPU-11111111-1111-1111-1111-111111111111");
        auto gpu2 = CreateGpu(2, "GPU-22222222-2222-2222-2222-222222222222");
        auto gpu3 = CreateGpu(3, "GPU-33333333-3333-3333-3333-333333333333");

        std::vector<Gpu *> gpuVect { gpu0.get(), gpu1.get(), gpu2.get(), gpu3.get() };
        std::vector<unsigned int> gpuIndices { 1, 3 };

        WHEN("BuildCudaVisibleDevicesString is called")
        {
            std::string result
                = NvidiaValidationSuite::BuildCudaVisibleDevicesString(gpuIndices, gpuVect, testUuidGetter);

            THEN("It returns UUIDs for GPUs 1 and 3 in that order")
            {
                CHECK(result == "GPU-11111111-1111-1111-1111-111111111111,GPU-33333333-3333-3333-3333-333333333333");
            }
        }
    }
}
