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

#include <catch2/catch_all.hpp>

#include "DcgmNvvsResponseWrapper.h"

#include <CpuSet.h>
#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>
#include <DiagResponseUtils.h>
#include <GpuSet.h>
#include <NvvsCommon.h>
#include <PluginStrings.h>
#include <UniquePtrUtil.h>
#include <dcgm_errors.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::SetVersion",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11,
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType = TestType;
    // Get the version directly from the trait
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.IsVersionSet());
        auto const &rawResponse = wrapper.ConstResponse<ResponseType>();
        REQUIRE(sizeof(rawResponse) == sizeof(ResponseType));
        REQUIRE(rawResponse.version == version);
    }
}

TEST_CASE("DcgmNvvsResponseWrapper::IsVersionSet")
{
    SECTION("Default")
    {
        DcgmNvvsResponseWrapper wrapper;
        REQUIRE(!wrapper.IsVersionSet());
    }

    SECTION("Set")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version12);
        REQUIRE(wrapper.IsVersionSet());
    }
}

namespace
{
std::vector<std::unique_ptr<EntitySet>> CreateFakeEntitySets(std::string const &driverVersion,
                                                             unsigned int const gpuCount,
                                                             unsigned int const cpuCount)
{
    std::vector<std::unique_ptr<EntitySet>> entitySets;
    std::vector<Gpu *> gpuObjs;

    auto gpuSet = std::make_unique<GpuSet>();
    for (unsigned int gpuId = 0; gpuId < gpuCount; ++gpuId)
    {
        gpuSet->AddEntityId(gpuId);
        gpuObjs.emplace_back(new Gpu(gpuId));

        dcgmDeviceAttributes_v3 attr {};
        attr.identifiers.pciDeviceId = 1234;
        SafeCopyTo(attr.identifiers.uuid, fmt::format("GPU-12345678-0000-0000-0000-00000000000{}", gpuId).c_str());
        SafeCopyTo(attr.identifiers.serial, fmt::format("GPU-Serial-{}", gpuId).c_str());
        SafeCopyTo(attr.identifiers.driverVersion, driverVersion.c_str());
        gpuObjs.back()->SetAttributes(attr);
    }
    gpuSet->SetGpuObjs(std::move(gpuObjs));
    entitySets.emplace_back(std::move(gpuSet));

    auto cpuSet = std::make_unique<CpuSet>();
    for (unsigned int cpuId = 0; cpuId < cpuCount; ++cpuId)
    {
        cpuSet->AddEntityId(cpuId);
    }
    entitySets.emplace_back(std::move(cpuSet));
    return entitySets;
}

void ReleaseEntitySets(std::vector<std::unique_ptr<EntitySet>> &entitySets)
{
    for (auto &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        auto *gpuSet  = ToGpuSet(entitySet.get());
        auto &gpuObjs = gpuSet->GetGpuObjs();
        for (auto &gpu : gpuObjs)
        {
            delete gpu;
        }
    }
}
} // namespace

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::PopulateDefault for newer versions",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);

        REQUIRE(wrapper.PopulateDefault(entitySets));
        auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

        REQUIRE(std::string_view(rawResponse.dcgmVersion) == DcgmNs::DcgmBuildInfo().GetVersion());
        REQUIRE(std::string_view(rawResponse.driverVersion) == "545.29.06");

        REQUIRE(rawResponse.numTests == 0);
        REQUIRE(rawResponse.numErrors == 0);
        REQUIRE(rawResponse.numInfo == 0);
        REQUIRE(rawResponse.numResults == 0);
        REQUIRE(rawResponse.numCategories == 0);

        for (unsigned int i = 0; i < DCGM_DIAG_RESPONSE_TESTS_MAX; ++i)
        {
            REQUIRE(rawResponse.tests[i].result == DCGM_DIAG_RESULT_NOT_RUN);
        }

        std::vector<unsigned int> const expectedCpuIds { 0, 1 };
        std::vector<unsigned int> const expectedGpuIds { 0, 1 };
        std::vector<std::string> const expectedGpuSerials { "GPU-Serial-0", "GPU-Serial-1" };
        std::vector<std::string> const expectedGpuDevIds { "0000", "0000" };

        std::vector<unsigned int> foundCpuIds;
        std::vector<unsigned int> foundGpuIds;
        std::vector<unsigned int> foundOtherIds;
        std::vector<std::string> foundGpuSerials;
        std::vector<std::string> foundGpuDevIds;

        REQUIRE(rawResponse.numEntities == 4);
        for (unsigned int i = 0; i < rawResponse.numEntities; ++i)
        {
            auto const &entity = rawResponse.entities[i];
            if (entity.entity.entityGroupId == DCGM_FE_GPU)
            {
                foundGpuIds.push_back(entity.entity.entityId);
                foundGpuSerials.emplace_back(entity.serialNum);
                foundGpuDevIds.emplace_back(entity.skuDeviceId);
            }
            else if (entity.entity.entityGroupId == DCGM_FE_CPU)
            {
                foundCpuIds.push_back(entity.entity.entityId);
                REQUIRE(std::string_view(entity.serialNum) == DCGM_STR_BLANK);
                REQUIRE(std::string_view(entity.skuDeviceId).empty());
            }
            else
            {
                foundOtherIds.push_back(entity.entity.entityId);
            }
        }

        REQUIRE(expectedCpuIds == foundCpuIds);
        REQUIRE(expectedGpuIds == foundGpuIds);
        REQUIRE(expectedGpuSerials == foundGpuSerials);
        REQUIRE(expectedGpuDevIds == foundGpuDevIds);
        REQUIRE(foundOtherIds.empty());

        for (auto const &result : std::span(rawResponse.results,
                                            std::min(static_cast<unsigned int>(DCGM_DIAG_RESPONSE_RESULTS_MAX),
                                                     static_cast<unsigned int>(std::size(rawResponse.results)))))
        {
            CHECK(result.result == DCGM_DIAG_RESULT_PASS);
        }

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::PopulateDefault::Deprecated Versions",
                   "",
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);

        REQUIRE(wrapper.PopulateDefault(entitySets));
        auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

        // Common checks for all versions
        REQUIRE(rawResponse.gpuCount == 2);
        REQUIRE(rawResponse.levelOneTestCount == 0);
        REQUIRE(rawResponse.systemError.msg[0] == '\0');

        // Check level one results for all versions
        for (unsigned int i = 0; i < LEVEL_ONE_MAX_RESULTS; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_NOT_RUN);
        }

        // Version-specific checks for per-GPU test counts
        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v7>)
        {
            for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
            {
                for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V7; ++j)
                {
                    REQUIRE(rawResponse.perGpuResponses[i].results[j].status == DCGM_DIAG_RESULT_NOT_RUN);
                }
            }
        }
        else
        {
            for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
            {
                for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; ++j)
                {
                    REQUIRE(rawResponse.perGpuResponses[i].results[j].status == DCGM_DIAG_RESULT_NOT_RUN);
                }
            }
        }

        // Version-specific checks for v8, v9, and v10
        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v8>
                      || std::is_same_v<ResponseType, dcgmDiagResponse_v9>
                      || std::is_same_v<ResponseType, dcgmDiagResponse_v10>)
        {
            REQUIRE(std::string_view(rawResponse.dcgmVersion) == DcgmNs::DcgmBuildInfo().GetVersion());
            REQUIRE(std::string_view(rawResponse.driverVersion) == "545.29.06");

            std::vector<std::string> const expectedGpuDevIds { "0000", "0000" };
            std::vector<std::string> foundGpuDevIds;

            for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
            {
                if (rawResponse.devIds[i][0] == '\0')
                {
                    break;
                }
                foundGpuDevIds.emplace_back(rawResponse.devIds[i]);
            }

            // Additional checks for v9 and v10 (serial numbers)
            if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v9>
                          || std::is_same_v<ResponseType, dcgmDiagResponse_v10>)
            {
                std::vector<std::string> const expectedGpuSerials { "GPU-Serial-0", "GPU-Serial-1" };
                std::vector<std::string> foundGpuSerials;

                for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
                {
                    if (rawResponse.devSerials[i][0] == '\0')
                    {
                        break;
                    }
                    foundGpuSerials.emplace_back(rawResponse.devSerials[i]);
                }

                // Additional check for v10 only (auxData)
                if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v10>)
                {
                    REQUIRE(rawResponse.auxDataPerTest[0].version == 0);
                }
            }
        }

        ReleaseEntitySets(entitySets);
    }
}

TEST_CASE("DcgmNvvsResponseWrapper::RawBinaryBlob")
{
    struct VersionSizePair
    {
        unsigned int version;
        size_t size;
    };

    auto [version, expectedSize]
        = GENERATE(VersionSizePair { dcgmDiagResponse_version12, sizeof(dcgmDiagResponse_v12) },
                   VersionSizePair { dcgmDiagResponse_version11, sizeof(dcgmDiagResponse_v11) },
                   VersionSizePair { dcgmDiagResponse_version10, sizeof(dcgmDiagResponse_v10) },
                   VersionSizePair { dcgmDiagResponse_version9, sizeof(dcgmDiagResponse_v9) },
                   VersionSizePair { dcgmDiagResponse_version8, sizeof(dcgmDiagResponse_v8) },
                   VersionSizePair { dcgmDiagResponse_version7, sizeof(dcgmDiagResponse_v7) });

    DYNAMIC_SECTION("Version " << version)
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);

        auto buf = wrapper.RawBinaryBlob();
        REQUIRE(buf.size() == expectedSize);
    }
}

namespace
{
std::unique_ptr<dcgmDiagEntityResults_v2> CreateFakeEntityResults(unsigned int const numErr,
                                                                  unsigned int const numInfo,
                                                                  unsigned int const numResults)
{
    auto entityResultsPtr = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    auto &entityResults   = *entityResultsPtr;

    assert(numErr < std::size(entityResults.errors));
    assert(numInfo < std::size(entityResults.info));
    assert(numResults < std::size(entityResults.results));

    for (unsigned int i = 0; i < numErr; ++i)
    {
        entityResults.errors[i].entity   = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
        entityResults.errors[i].category = DCGM_FR_EC_INTERNAL_OTHER;
        entityResults.errors[i].code     = 0xc8763;
        entityResults.errors[i].severity = DCGM_ERROR_ISOLATE;
        entityResults.errors[i].testId   = 0;
        std::string msg                  = fmt::format("error_{}", i);
        SafeCopyTo(entityResults.errors[i].msg, msg.c_str());
        entityResults.numErrors += 1;
    }

    for (unsigned int i = 0; i < numInfo; ++i)
    {
        entityResults.info[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
        entityResults.info[i].testId = 0;
        std::string msg              = fmt::format("info_{}", i);
        SafeCopyTo(entityResults.info[i].msg, msg.c_str());
        entityResults.numInfo += 1;
    }

    for (unsigned int i = 0; i < numResults; ++i)
    {
        entityResults.results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
        entityResults.results[i].testId = 0;
        entityResults.results[i].result = (i < numErr) ? DCGM_DIAG_RESULT_FAIL : DCGM_DIAG_RESULT_PASS;
        entityResults.numResults += 1;
    }

    return entityResultsPtr;
}
} // namespace

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::SetSoftwareTestResult", "", dcgmDiagResponse_v12, dcgmDiagResponse_v11)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        auto entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

        // SetSoftwareTestResult should not increase numTests as all subtests of software are aggregated.
        // The test framework will increase numTests in the end of the software test.
        REQUIRE(rawResponse.numTests == 0);

        REQUIRE(std::string_view(rawResponse.tests[0].name) == "software");
        REQUIRE(rawResponse.tests[0].numErrors == 2);
        REQUIRE(rawResponse.tests[0].errorIndices[0] == 0);
        REQUIRE(rawResponse.tests[0].errorIndices[1] == 1);
        REQUIRE(rawResponse.tests[0].numInfo == 4);
        for (unsigned int i = 0; i < 4; ++i)
        {
            REQUIRE(rawResponse.tests[0].infoIndices[i] == i);
        }
        REQUIRE(rawResponse.tests[0].numResults == 2);
        REQUIRE(rawResponse.tests[0].resultIndices[0] == 0);
        REQUIRE(rawResponse.tests[0].resultIndices[1] == 1);
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_FAIL);
        // 1 error for each subtest
        REQUIRE(rawResponse.numErrors == 2);
        dcgmDiagError_v1 expectedError;
        expectedError.entity   = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        expectedError.code     = 0xc8763;
        expectedError.category = DCGM_FR_EC_INTERNAL_OTHER;
        expectedError.severity = DCGM_ERROR_ISOLATE;
        SafeCopyTo(expectedError.msg, "error_0");
        expectedError.testId = 0;

        REQUIRE(rawResponse.errors[0] == expectedError);
        REQUIRE(rawResponse.errors[1] == expectedError);

        // 2 info for each subtest
        REQUIRE(rawResponse.numInfo == 4);
        dcgmDiagInfo_v1 expectedInfo;
        expectedInfo.entity = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        SafeCopyTo(expectedInfo.msg, "info_0");
        expectedInfo.testId = 0;
        REQUIRE(rawResponse.info[0] == expectedInfo);
        REQUIRE(rawResponse.info[2] == expectedInfo);
        expectedInfo.entity.entityId = 1;
        SafeCopyTo(expectedInfo.msg, "info_1");
        REQUIRE(rawResponse.info[1] == expectedInfo);
        REQUIRE(rawResponse.info[3] == expectedInfo);

        // Each entity will only have one result in the same test.
        REQUIRE(rawResponse.numResults == 2);
        REQUIRE(rawResponse.results[0].entity == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 });
        REQUIRE(rawResponse.results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.results[1].entity == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 });
        REQUIRE(rawResponse.results[1].result == DCGM_DIAG_RESULT_PASS);

        REQUIRE(std::string_view(rawResponse.tests[1].name).empty());
        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::SetSoftwareTestResult::Deprecated Versions",
                   "",
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);

        REQUIRE(wrapper.PopulateDefault(entitySets));

        auto entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

        REQUIRE(rawResponse.levelOneTestCount == 2);
        for (unsigned int i = 0; i < 2; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_FAIL);

            if constexpr (version >= dcgmDiagResponse_version9)
            {
                REQUIRE(rawResponse.levelOneResults[i].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
                REQUIRE(rawResponse.levelOneResults[i].error[0].severity == DCGM_ERROR_ISOLATE);
                REQUIRE(rawResponse.levelOneResults[i].error[0].gpuId == 0);
                REQUIRE(rawResponse.levelOneResults[i].error[0].code == 0xc8763);
                REQUIRE(std::string_view(rawResponse.levelOneResults[i].error[0].msg) == "error_0");
            }
            else
            {
                REQUIRE(rawResponse.levelOneResults[i].error.code == 0xc8763);
                REQUIRE(std::string_view(rawResponse.levelOneResults[i].error.msg) == "error_0");
            }

            REQUIRE(std::string_view(rawResponse.levelOneResults[i].info) == "info_0, info_1");
        }

        ReleaseEntitySets(entitySets);
    }
}

TEST_CASE("DcgmNvvsResponseWrapper::SetSoftwareTestResult::v12::Edge Cases")
{
    SECTION("Overwrite SW test result of entities")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version12);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        auto twoEntitiesResultsPtr = MakeUniqueZero<dcgmDiagEntityResults_v2>();

        for (unsigned int i = 0; i < 2; ++i)
        {
            twoEntitiesResultsPtr->results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
            twoEntitiesResultsPtr->results[i].testId = 0;
            twoEntitiesResultsPtr->numResults += 1;
        }
        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_SKIP;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_NOT_RUN;
        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_PASS, *twoEntitiesResultsPtr) == DCGM_ST_OK);
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v12>();

        REQUIRE(rawResponse.tests[0].numResults == 2);
        REQUIRE(rawResponse.tests[0].resultIndices[0] == 0);
        REQUIRE(rawResponse.tests[0].resultIndices[1] == 1);
        REQUIRE(rawResponse.results[0].result == DCGM_DIAG_RESULT_SKIP);
        REQUIRE(rawResponse.results[1].result == DCGM_DIAG_RESULT_NOT_RUN);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_PASS;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_PASS;
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_PASS, *twoEntitiesResultsPtr) == DCGM_ST_OK);

        REQUIRE(rawResponse.tests[0].numResults == 2);
        REQUIRE(rawResponse.tests[0].resultIndices[0] == 0);
        REQUIRE(rawResponse.tests[0].resultIndices[1] == 1);
        // PASS should overwrite SKIP or NOT_RUN.
        REQUIRE(rawResponse.results[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.results[1].result == DCGM_DIAG_RESULT_PASS);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_FAIL;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_WARN;
        REQUIRE(wrapper.SetSoftwareTestResult("CUDA Main Library", NVVS_RESULT_FAIL, *twoEntitiesResultsPtr)
                == DCGM_ST_OK);
        REQUIRE(rawResponse.tests[0].numResults == 2);
        REQUIRE(rawResponse.tests[0].resultIndices[0] == 0);
        REQUIRE(rawResponse.tests[0].resultIndices[1] == 1);
        // FAIL, WARN should overwrite PASS.
        REQUIRE(rawResponse.results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.results[1].result == DCGM_DIAG_RESULT_WARN);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_PASS;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_PASS;
        REQUIRE(wrapper.SetSoftwareTestResult("CUDA Toolkit Libraries", NVVS_RESULT_PASS, *twoEntitiesResultsPtr)
                == DCGM_ST_OK);
        REQUIRE(rawResponse.tests[0].numResults == 2);
        REQUIRE(rawResponse.tests[0].resultIndices[0] == 0);
        REQUIRE(rawResponse.tests[0].resultIndices[1] == 1);
        // PASS cannot overwrite FAIL or WARN.
        REQUIRE(rawResponse.results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.results[1].result == DCGM_DIAG_RESULT_WARN);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_SKIP;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_NOT_RUN;
        REQUIRE(
            wrapper.SetSoftwareTestResult("Permissions and OS-related Blocks", NVVS_RESULT_PASS, *twoEntitiesResultsPtr)
            == DCGM_ST_OK);
        REQUIRE(rawResponse.tests[0].numResults == 2);
        REQUIRE(rawResponse.tests[0].resultIndices[0] == 0);
        REQUIRE(rawResponse.tests[0].resultIndices[1] == 1);
        // SKIP, NOT_RUN cannot overwrite FAIL or WARN.
        REQUIRE(rawResponse.results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.results[1].result == DCGM_DIAG_RESULT_WARN);

        ReleaseEntitySets(entitySets);
    }

    SECTION("Overwrite whole SW test result")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version12);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        auto twoEntitiesResultsPtr = MakeUniqueZero<dcgmDiagEntityResults_v2>();

        for (unsigned int i = 0; i < 2; ++i)
        {
            twoEntitiesResultsPtr->results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
            twoEntitiesResultsPtr->results[i].testId = 0;
            twoEntitiesResultsPtr->numResults += 1;
        }
        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_NOT_RUN;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_NOT_RUN;

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_SKIP, *twoEntitiesResultsPtr) == DCGM_ST_OK);
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v12>();
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_NOT_RUN);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_SKIP;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_NOT_RUN;
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_SKIP, *twoEntitiesResultsPtr) == DCGM_ST_OK);
        // SKIP can overwrite NOT_RUN
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_SKIP);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_SKIP;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_PASS;
        REQUIRE(wrapper.SetSoftwareTestResult("CUDA Main Library", NVVS_RESULT_PASS, *twoEntitiesResultsPtr)
                == DCGM_ST_OK);
        // PASS can overwrite SKIP
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_PASS);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_WARN;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_PASS;
        REQUIRE(wrapper.SetSoftwareTestResult("CUDA Main Library", NVVS_RESULT_WARN, *twoEntitiesResultsPtr)
                == DCGM_ST_OK);
        // WARN can overwrite SKIP
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_WARN);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_WARN;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_FAIL;
        REQUIRE(wrapper.SetSoftwareTestResult("CUDA Main Library", NVVS_RESULT_FAIL, *twoEntitiesResultsPtr)
                == DCGM_ST_OK);
        // FAIL can overwrite WARN
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_FAIL);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_SKIP;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_PASS;
        REQUIRE(wrapper.SetSoftwareTestResult("CUDA Toolkit Libraries", NVVS_RESULT_PASS, *twoEntitiesResultsPtr)
                == DCGM_ST_OK);
        // PASS cannot overwrite FAIL
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_FAIL);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_WARN;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_PASS;
        REQUIRE(
            wrapper.SetSoftwareTestResult("Permissions and OS-related Blocks", NVVS_RESULT_WARN, *twoEntitiesResultsPtr)
            == DCGM_ST_OK);
        // WARN cannot overwrite FAIL
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_FAIL);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_SKIP;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_NOT_RUN;
        REQUIRE(wrapper.SetSoftwareTestResult("Persistence Mode", NVVS_RESULT_SKIP, *twoEntitiesResultsPtr)
                == DCGM_ST_OK);
        // SKIP cannot overwrite FAIL
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_FAIL);

        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_NOT_RUN;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_NOT_RUN;
        REQUIRE(wrapper.SetSoftwareTestResult("Environmental Variables", NVVS_RESULT_SKIP, *twoEntitiesResultsPtr)
                == DCGM_ST_OK);
        // NOT_RUN cannot overwrite FAIL
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_FAIL);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::IncreaseNumTests",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11,
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType = TestType;
    // Get the version directly from the trait
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);

        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v12>
                      || std::is_same_v<ResponseType, dcgmDiagResponse_v11>)
        {
            auto const &rawResponse = wrapper.ConstResponse<ResponseType>();
            wrapper.IncreaseNumTests();
            REQUIRE(rawResponse.numTests == 1);
            wrapper.IncreaseNumTests();
            REQUIRE(rawResponse.numTests == 2);
        }
        else
        {
            REQUIRE_NOTHROW(wrapper.IncreaseNumTests());
        }
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::SetTestResult", "", dcgmDiagResponse_v12, dcgmDiagResponse_v11)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        SECTION("Basic Test Results")
        {
            auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

            DcgmNvvsResponseWrapper wrapper;
            wrapper.SetVersion(version);
            REQUIRE(wrapper.PopulateDefault(entitySets));
            ::Json::Value auxData;

            auto entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

            auxData["hello"] = "eud";
            REQUIRE(wrapper.SetTestResult("eud", "eud", *entityResultsPtr, auxData) == DCGM_ST_OK);
            auxData["hello"] = "memory";
            REQUIRE(wrapper.SetTestResult("memory", "memory", *entityResultsPtr, auxData) == DCGM_ST_OK);

            auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

            REQUIRE(rawResponse.numTests == 2);
            REQUIRE(std::string_view(rawResponse.tests[0].name) == "eud");
            REQUIRE(rawResponse.tests[0].numErrors == 1);
            REQUIRE(rawResponse.tests[0].errorIndices[0] == 0);
            REQUIRE(rawResponse.tests[0].numInfo == 2);
            for (unsigned int i = 0; i < 2; ++i)
            {
                REQUIRE(rawResponse.tests[0].infoIndices[i] == i);
            }
            REQUIRE(rawResponse.tests[0].numResults == 2);
            REQUIRE(rawResponse.tests[0].resultIndices[0] == 0);
            REQUIRE(rawResponse.tests[0].resultIndices[1] == 1);
            REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.tests[0].auxData.version == dcgmDiagTestAuxData_version1);
            REQUIRE(std::string_view(rawResponse.tests[0].auxData.data) == "{\"hello\":\"eud\"}");

            REQUIRE(std::string_view(rawResponse.tests[1].name) == "memory");
            REQUIRE(rawResponse.tests[1].numErrors == 1);
            REQUIRE(rawResponse.tests[1].errorIndices[0] == 1);
            REQUIRE(rawResponse.tests[1].numInfo == 2);
            for (unsigned int i = 0; i < 2; ++i)
            {
                REQUIRE(rawResponse.tests[1].infoIndices[i] == i + 2);
            }
            REQUIRE(rawResponse.tests[1].numResults == 2);
            REQUIRE(rawResponse.tests[1].resultIndices[0] == 2);
            REQUIRE(rawResponse.tests[1].resultIndices[1] == 3);
            REQUIRE(rawResponse.tests[1].result == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.tests[1].auxData.version == dcgmDiagTestAuxData_version1);
            REQUIRE(std::string_view(rawResponse.tests[1].auxData.data) == "{\"hello\":\"memory\"}");

            REQUIRE(rawResponse.numErrors == 2);
            dcgmDiagError_v1 expectedError;
            expectedError.entity   = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
            expectedError.code     = 0xc8763;
            expectedError.category = DCGM_FR_EC_INTERNAL_OTHER;
            expectedError.severity = DCGM_ERROR_ISOLATE;
            SafeCopyTo(expectedError.msg, "error_0");
            expectedError.testId = 0;

            REQUIRE(rawResponse.errors[0] == expectedError);
            expectedError.testId = 1;
            REQUIRE(rawResponse.errors[1] == expectedError);

            // 2 info for each subtest
            REQUIRE(rawResponse.numInfo == 4);
            dcgmDiagInfo_v1 expectedInfo;
            expectedInfo.entity = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
            expectedInfo.testId = 0;
            SafeCopyTo(expectedInfo.msg, "info_0");
            REQUIRE(rawResponse.info[0] == expectedInfo);
            expectedInfo.entity.entityId = 1;
            SafeCopyTo(expectedInfo.msg, "info_1");
            REQUIRE(rawResponse.info[1] == expectedInfo);
            expectedInfo.testId          = 1;
            expectedInfo.entity.entityId = 0;
            SafeCopyTo(expectedInfo.msg, "info_0");
            REQUIRE(rawResponse.info[2] == expectedInfo);
            expectedInfo.entity.entityId = 1;
            SafeCopyTo(expectedInfo.msg, "info_1");
            REQUIRE(rawResponse.info[3] == expectedInfo);

            // Each entity will only have one result in the same test.
            REQUIRE(rawResponse.numResults == 4);
            REQUIRE(rawResponse.results[0].entity
                    == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 });
            REQUIRE(rawResponse.results[0].result == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.results[1].entity
                    == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 });
            REQUIRE(rawResponse.results[1].result == DCGM_DIAG_RESULT_PASS);
            REQUIRE(rawResponse.results[2].entity
                    == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 });
            REQUIRE(rawResponse.results[2].result == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.results[3].entity
                    == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 });
            REQUIRE(rawResponse.results[3].result == DCGM_DIAG_RESULT_PASS);

            REQUIRE(std::string_view(rawResponse.tests[2].name).empty());

            ReleaseEntitySets(entitySets);
        }

        SECTION("Overall Result")
        {
            auto entitySets = CreateFakeEntitySets("545.29.06", 5, 1);

            DcgmNvvsResponseWrapper wrapper;
            wrapper.SetVersion(version);
            REQUIRE(wrapper.PopulateDefault(entitySets));
            ::Json::Value auxData;

            auto entitiesResultsPtr = MakeUniqueZero<dcgmDiagEntityResults_v2>();

            for (unsigned int i = 0; i < 5; ++i)
            {
                entitiesResultsPtr->results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
                entitiesResultsPtr->results[i].testId = 0;
                entitiesResultsPtr->numResults += 1;
                entitiesResultsPtr->results[i].result = DCGM_DIAG_RESULT_NOT_RUN;
            }
            auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

            REQUIRE(wrapper.SetTestResult("eud", "eud", *entitiesResultsPtr, auxData) == DCGM_ST_OK);
            REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_NOT_RUN);

            entitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_SKIP;
            REQUIRE(wrapper.SetTestResult("memory", "memory", *entitiesResultsPtr, auxData) == DCGM_ST_OK);
            REQUIRE(rawResponse.tests[1].result == DCGM_DIAG_RESULT_SKIP);

            entitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_PASS;
            REQUIRE(wrapper.SetTestResult("pcie", "pcie", *entitiesResultsPtr, auxData) == DCGM_ST_OK);
            REQUIRE(rawResponse.tests[2].result == DCGM_DIAG_RESULT_PASS);

            entitiesResultsPtr->results[2].result = DCGM_DIAG_RESULT_WARN;
            REQUIRE(wrapper.SetTestResult("eud", "cpu_eud", *entitiesResultsPtr, auxData) == DCGM_ST_OK);
            REQUIRE(rawResponse.tests[3].result == DCGM_DIAG_RESULT_WARN);

            entitiesResultsPtr->results[3].result = DCGM_DIAG_RESULT_FAIL;
            REQUIRE(wrapper.SetTestResult("targeted_power", "targeted_power", *entitiesResultsPtr, auxData)
                    == DCGM_ST_OK);
            REQUIRE(rawResponse.tests[4].result == DCGM_DIAG_RESULT_FAIL);

            ReleaseEntitySets(entitySets);
        }
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::SetTestResult::OlderVersions",
                   "",
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        auto entityResultsPtr = CreateFakeEntityResults(1, 2, 2);
        ::Json::Value auxData;

        // Choose test names based on version
        std::string test1Name = "eud";
        std::string test2Name = "memory";

        // For v7, use "pcie" instead of "eud"
        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v7>)
            test1Name = "pcie";

        auxData["hello"] = test1Name;
        REQUIRE(wrapper.SetTestResult(test1Name, test1Name, *entityResultsPtr, auxData) == DCGM_ST_OK);
        auxData["hello"] = test2Name;
        REQUIRE(wrapper.SetTestResult(test2Name, test2Name, *entityResultsPtr, auxData) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

        // Get test indices
        unsigned int const test1Idx = GetTestIndex(test1Name);
        unsigned int const test2Idx = GetTestIndex(test2Name);

        // Common checks for all versions
        REQUIRE(rawResponse.perGpuResponses[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[test1Idx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[1].gpuId == 1);
        REQUIRE(rawResponse.perGpuResponses[1].results[test1Idx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[test2Idx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[test1Idx].info) == "info_1");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[test2Idx].info) == "info_1");

        // Version-specific checks
        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v10>
                      || std::is_same_v<ResponseType, dcgmDiagResponse_v9>)
        {
            // Checks for v9 and v10
            REQUIRE(rawResponse.perGpuResponses[0].results[test1Idx].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
            REQUIRE(rawResponse.perGpuResponses[0].results[test1Idx].error[0].code == 0xc8763);
            REQUIRE(rawResponse.perGpuResponses[0].results[test1Idx].error[0].severity == DCGM_ERROR_ISOLATE);
            REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[test1Idx].error[0].msg) == "error_0");
            REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[test1Idx].info) == "info_0");

            REQUIRE(rawResponse.perGpuResponses[0].results[test2Idx].status == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.perGpuResponses[0].results[test2Idx].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
            REQUIRE(rawResponse.perGpuResponses[0].results[test2Idx].error[0].code == 0xc8763);
            REQUIRE(rawResponse.perGpuResponses[0].results[test2Idx].error[0].severity == DCGM_ERROR_ISOLATE);
            REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[test2Idx].error[0].msg) == "error_0");
            REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[test2Idx].info) == "info_0");

            // Additional checks for v10 only
            if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v10>)
            {
                REQUIRE(rawResponse.auxDataPerTest[test1Idx].version == dcgmDiagTestAuxData_version1);
                REQUIRE(std::string_view(rawResponse.auxDataPerTest[test1Idx].data)
                        == fmt::format("{{\"hello\":\"{}\"}}", test1Name));
                REQUIRE(rawResponse.auxDataPerTest[test2Idx].version == dcgmDiagTestAuxData_version1);
                REQUIRE(std::string_view(rawResponse.auxDataPerTest[test2Idx].data)
                        == fmt::format("{{\"hello\":\"{}\"}}", test2Name));
            }
        }
        else
        {
            // Checks for v7 and v8
            REQUIRE(rawResponse.perGpuResponses[0].results[test1Idx].error.code == 0xc8763);
            REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[test1Idx].error.msg) == "error_0");
            REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[test1Idx].info) == "info_0");

            REQUIRE(rawResponse.perGpuResponses[0].results[test2Idx].status == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.perGpuResponses[0].results[test2Idx].error.code == 0xc8763);
            REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[test2Idx].error.msg) == "error_0");
            REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[test2Idx].info) == "info_0");

            REQUIRE(rawResponse.perGpuResponses[1].results[test1Idx].error.code == 0);
            REQUIRE(rawResponse.perGpuResponses[1].results[test2Idx].error.code == 0);
        }

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::SetTestSkipped",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11,
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v11>
                      || std::is_same_v<ResponseType, dcgmDiagResponse_v12>)
        {
            // For newer response versions (11+)
            REQUIRE(wrapper.SetTestSkipped(EUD_PLUGIN_NAME, EUD_PLUGIN_NAME) == DCGM_ST_OK);
            auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

            REQUIRE(rawResponse.numTests == 1);
            REQUIRE(std::string_view(rawResponse.tests[0].name) == EUD_PLUGIN_NAME);
            REQUIRE(std::string_view(rawResponse.tests[0].pluginName) == EUD_PLUGIN_NAME);
            REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_SKIP);
            REQUIRE(rawResponse.tests[0].numResults == 0);
        }
        else
        {
            // For older response versions (<11)
            REQUIRE(wrapper.SetTestSkipped(PCIE_PLUGIN_NAME, PCIE_PLUGIN_NAME) == DCGM_ST_OK);
            auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

            auto const pcieIdx = GetTestIndex(PCIE_PLUGIN_NAME);
            REQUIRE(pcieIdx < std::size(rawResponse.perGpuResponses[0].results));

            // coverity[overrun] - the previous REQUIRE should prevent this condition
            REQUIRE(rawResponse.perGpuResponses[0].gpuId == 0);
            REQUIRE(rawResponse.perGpuResponses[0].results[pcieIdx].status == DCGM_DIAG_RESULT_SKIP);
            REQUIRE(rawResponse.perGpuResponses[1].gpuId == 1);
            REQUIRE(rawResponse.perGpuResponses[1].results[pcieIdx].status == DCGM_DIAG_RESULT_SKIP);
        }

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::SetSystemError",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11,
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        wrapper.SetSystemError("Capoo", 0xc8763);

        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v11>
                      || std::is_same_v<ResponseType, dcgmDiagResponse_v12>)
        {
            auto const &rawResponse = wrapper.ConstResponse<ResponseType>();
            CHECK(rawResponse.numErrors == 1);
            CHECK(rawResponse.errors[0].testId == DCGM_DIAG_RESPONSE_SYSTEM_ERROR);
            CHECK(rawResponse.errors[0].code == 0xc8763);
            CHECK(std::string_view(rawResponse.errors[0].msg) == "Capoo");
        }
        else if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v10>
                           || std::is_same_v<ResponseType, dcgmDiagResponse_v9>
                           || std::is_same_v<ResponseType, dcgmDiagResponse_v8>
                           || std::is_same_v<ResponseType, dcgmDiagResponse_v7>)
        {
            auto const &rawResponse = wrapper.ConstResponse<ResponseType>();
            REQUIRE(rawResponse.systemError.code == 0xc8763);
            REQUIRE(std::string_view(rawResponse.systemError.msg) == "Capoo");
        }

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::TestSlotsFull",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11,
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));
        REQUIRE(!wrapper.TestSlotsFull());

        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v11>
                      || std::is_same_v<ResponseType, dcgmDiagResponse_v12>)
        {
            for (unsigned int i = 0; i < DCGM_DIAG_RESPONSE_TESTS_MAX; ++i)
            {
                REQUIRE(wrapper.SetTestSkipped("eud", "cpu_eud") == DCGM_ST_OK);
            }
            REQUIRE(wrapper.TestSlotsFull());
        }
        else
        {
            // For older versions, just verify TestSlotsFull() doesn't crash
            REQUIRE_NOTHROW(wrapper.TestSlotsFull());
        }

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::AddTestCategory::Basic", "", dcgmDiagResponse_v12, dcgmDiagResponse_v11)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        auto entityResultsPtr   = CreateFakeEntityResults(1, 2, 2);
        auto const &rawResponse = wrapper.ConstResponse<ResponseType>();

        REQUIRE(wrapper.SetTestResult("eud", "eud", *entityResultsPtr, std::nullopt) == DCGM_ST_OK);
        REQUIRE(wrapper.SetTestResult("memory", "memory", *entityResultsPtr, std::nullopt) == DCGM_ST_OK);
        REQUIRE(wrapper.SetTestResult("pcie", "pcie", *entityResultsPtr, std::nullopt) == DCGM_ST_OK);

        REQUIRE(wrapper.AddTestCategory("eud", "stress") == DCGM_ST_OK);
        REQUIRE(rawResponse.numCategories == 1);
        REQUIRE(std::string_view(rawResponse.categories[0]) == "stress");
        REQUIRE(rawResponse.tests[0].categoryIndex == 0);
        REQUIRE(wrapper.AddTestCategory("memory", "stress") == DCGM_ST_OK);
        REQUIRE(rawResponse.numCategories == 1);
        REQUIRE(std::string_view(rawResponse.categories[0]) == "stress");
        REQUIRE(rawResponse.tests[1].categoryIndex == 0);
        REQUIRE(wrapper.AddTestCategory("pcie", "hardware") == DCGM_ST_OK);
        REQUIRE(rawResponse.numCategories == 2);
        REQUIRE(std::string_view(rawResponse.categories[0]) == "stress");
        REQUIRE(std::string_view(rawResponse.categories[1]) == "hardware");
        REQUIRE(rawResponse.tests[2].categoryIndex == 1);

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::AddTestCategory::Full", "", dcgmDiagResponse_v12, dcgmDiagResponse_v11)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        auto entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetTestResult("eud", "eud", *entityResultsPtr, std::nullopt) == DCGM_ST_OK);

        for (unsigned int i = 0; i < DCGM_DIAG_RESPONSE_CATEGORIES_MAX; ++i)
        {
            REQUIRE(wrapper.AddTestCategory("eud", fmt::format("{}", i)) == DCGM_ST_OK);
        }
        REQUIRE(wrapper.AddTestCategory("eud", "Capoo") == DCGM_ST_INSUFFICIENT_RESOURCES);

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::AddTestCategory::UnknownName",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        REQUIRE(wrapper.AddTestCategory("name", "category") == DCGM_ST_GENERIC_ERROR);

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("AddTestCategory::Deprecated Versions",
                   "",
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        REQUIRE(wrapper.AddTestCategory("name", "category") == DCGM_ST_VER_MISMATCH);

        ReleaseEntitySets(entitySets);
    }
}

TEMPLATE_TEST_CASE("DcgmNvvsResponseWrapper::MaxInfoMessages", "", dcgmDiagResponse_v12, dcgmDiagResponse_v11)
{
    using ResponseType             = TestType;
    constexpr unsigned int version = DcgmNs::ResponseVersionTrait<ResponseType>::version;

    DYNAMIC_SECTION("Testing version " << version)
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(version);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        auto entityResultsPtr = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        auto &entityResults   = *entityResultsPtr;

        auto const &rawResponse    = wrapper.ConstResponse<ResponseType>();
        auto const maxInfoMessages = std::size(rawResponse.info);

        // Create more than the maximum number of info messages
        entityResults.numInfo = maxInfoMessages + 2;
        for (unsigned int i = 0; i < entityResults.numInfo; ++i)
        {
            entityResults.info[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
            std::string msg              = fmt::format("Info message #{}", i);
            SafeCopyTo(entityResults.info[i].msg, msg.c_str());
            entityResults.info[i].testId = 0;
        }

        REQUIRE(wrapper.SetTestResult("memory", "memory", entityResults, std::nullopt) == DCGM_ST_OK);

        // Verify test was added correctly
        REQUIRE(rawResponse.numTests == 1);
        REQUIRE(std::string_view(rawResponse.tests[0].name) == "memory");

        // Verify only up to the maximum info messages were stored
        REQUIRE(rawResponse.numInfo == maxInfoMessages);
        REQUIRE(rawResponse.tests[0].numInfo == maxInfoMessages);

        // Verify the content of stored messages
        for (unsigned int i = 0; i < rawResponse.numInfo; ++i)
        {
            std::string expectedMsg = fmt::format("Info message #{}", i);
            REQUIRE(std::string_view(rawResponse.info[i].msg) == expectedMsg);
            REQUIRE(rawResponse.info[i].entity.entityGroupId == DCGM_FE_GPU);
            REQUIRE(rawResponse.info[i].entity.entityId == 0);
            REQUIRE(rawResponse.info[i].testId == 0);
        }

        ReleaseEntitySets(entitySets);
    }
}
