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

#include <DcgmNvvsResponseWrapper.h>

#include <CpuSet.h>
#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>
#include <GpuSet.h>
#include <NvvsCommon.h>
#include <PluginStrings.h>
#include <dcgm_errors.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

template <typename T>
void TestSetVersion(unsigned int version)
{
    DcgmNvvsResponseWrapper wrapper;

    wrapper.SetVersion(version);
    REQUIRE(wrapper.IsVersionSet());
    auto const &rawResponse = wrapper.ConstResponse<T>();
    REQUIRE(sizeof(rawResponse) == sizeof(T));
    REQUIRE(rawResponse.version == version);
}

TEST_CASE("DcgmNvvsResponseWrapper::SetVersion")
{
    SECTION("Version 11")
    {
        TestSetVersion<dcgmDiagResponse_v11>(dcgmDiagResponse_version11);
    }

    SECTION("Version 10")
    {
        TestSetVersion<dcgmDiagResponse_v10>(dcgmDiagResponse_version10);
    }

    SECTION("Version 9")
    {
        TestSetVersion<dcgmDiagResponse_v9>(dcgmDiagResponse_version9);
    }

    SECTION("Version 8")
    {
        TestSetVersion<dcgmDiagResponse_v8>(dcgmDiagResponse_version8);
    }

    SECTION("Version 7")
    {
        TestSetVersion<dcgmDiagResponse_v7>(dcgmDiagResponse_version7);
    }
}

TEST_CASE("DcgmNvvsResponseWrapper::GetVersion")
{
    DcgmNvvsResponseWrapper wrapper;
    wrapper.SetVersion(dcgmDiagResponse_version11);
    REQUIRE(wrapper.GetVersion() == dcgmDiagResponse_version11);
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
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.IsVersionSet());
    }
}

std::vector<std::unique_ptr<EntitySet>> CreateFakeEntitySets(std::string const &driverVersion,
                                                             unsigned int gpuCount,
                                                             unsigned int cpuCount)
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

TEST_CASE("DcgmNvvsResponseWrapper::PopulateDefault")
{
    SECTION("Version 11")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);

        REQUIRE(wrapper.PopulateDefault(entitySets));
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();

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

    SECTION("Version 10")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version10);

        REQUIRE(wrapper.PopulateDefault(entitySets));
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v10>();

        REQUIRE(std::string_view(rawResponse.dcgmVersion) == DcgmNs::DcgmBuildInfo().GetVersion());
        REQUIRE(std::string_view(rawResponse.driverVersion) == "545.29.06");
        REQUIRE(rawResponse.gpuCount == 2);
        REQUIRE(rawResponse.levelOneTestCount == 0);
        REQUIRE(rawResponse.systemError.msg[0] == '\0');

        std::vector<std::string> const expectedGpuSerials { "GPU-Serial-0", "GPU-Serial-1" };
        std::vector<std::string> const expectedGpuDevIds { "0000", "0000" };
        std::vector<std::string> foundGpuSerials;
        std::vector<std::string> foundGpuDevIds;

        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
        {
            if (rawResponse.devIds[i][0] == '\0')
            {
                break;
            }
            foundGpuDevIds.emplace_back(rawResponse.devIds[i]);
        }

        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
        {
            if (rawResponse.devSerials[i][0] == '\0')
            {
                break;
            }
            foundGpuSerials.emplace_back(rawResponse.devSerials[i]);
        }

        REQUIRE(rawResponse.auxDataPerTest[0].version == 0);

        for (unsigned int i = 0; i < LEVEL_ONE_MAX_RESULTS; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_NOT_RUN);
        }

        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; ++j)
            {
                REQUIRE(rawResponse.perGpuResponses[i].results[j].status == DCGM_DIAG_RESULT_NOT_RUN);
            }
        }

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 9")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version9);

        REQUIRE(wrapper.PopulateDefault(entitySets));
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v9>();

        REQUIRE(std::string_view(rawResponse.dcgmVersion) == DcgmNs::DcgmBuildInfo().GetVersion());
        REQUIRE(std::string_view(rawResponse.driverVersion) == "545.29.06");
        REQUIRE(rawResponse.gpuCount == 2);
        REQUIRE(rawResponse.levelOneTestCount == 0);
        REQUIRE(rawResponse.systemError.msg[0] == '\0');

        std::vector<std::string> const expectedGpuSerials { "GPU-Serial-0", "GPU-Serial-1" };
        std::vector<std::string> const expectedGpuDevIds { "0000", "0000" };
        std::vector<std::string> foundGpuSerials;
        std::vector<std::string> foundGpuDevIds;

        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
        {
            if (rawResponse.devIds[i][0] == '\0')
            {
                break;
            }
            foundGpuDevIds.emplace_back(rawResponse.devIds[i]);
        }

        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
        {
            if (rawResponse.devSerials[i][0] == '\0')
            {
                break;
            }
            foundGpuSerials.emplace_back(rawResponse.devSerials[i]);
        }

        for (unsigned int i = 0; i < LEVEL_ONE_MAX_RESULTS; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_NOT_RUN);
        }

        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; ++j)
            {
                REQUIRE(rawResponse.perGpuResponses[i].results[j].status == DCGM_DIAG_RESULT_NOT_RUN);
            }
        }

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 8")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version8);

        REQUIRE(wrapper.PopulateDefault(entitySets));
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v8>();

        REQUIRE(std::string_view(rawResponse.dcgmVersion) == DcgmNs::DcgmBuildInfo().GetVersion());
        REQUIRE(std::string_view(rawResponse.driverVersion) == "545.29.06");
        REQUIRE(rawResponse.gpuCount == 2);
        REQUIRE(rawResponse.levelOneTestCount == 0);
        REQUIRE(rawResponse.systemError.msg[0] == '\0');

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

        for (unsigned int i = 0; i < LEVEL_ONE_MAX_RESULTS; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_NOT_RUN);
        }

        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; ++j)
            {
                REQUIRE(rawResponse.perGpuResponses[i].results[j].status == DCGM_DIAG_RESULT_NOT_RUN);
            }
        }

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 7")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version7);

        REQUIRE(wrapper.PopulateDefault(entitySets));
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v7>();

        REQUIRE(rawResponse.gpuCount == 2);
        REQUIRE(rawResponse.levelOneTestCount == 0);
        REQUIRE(rawResponse.systemError.msg[0] == '\0');

        for (unsigned int i = 0; i < LEVEL_ONE_MAX_RESULTS; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_NOT_RUN);
        }

        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; ++i)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V7; ++j)
            {
                REQUIRE(rawResponse.perGpuResponses[i].results[j].status == DCGM_DIAG_RESULT_NOT_RUN);
            }
        }

        ReleaseEntitySets(entitySets);
    }
}

TEST_CASE("DcgmNvvsResponseWrapper::RawBinaryBlob")
{
    SECTION("Version 11")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);

        auto buf = wrapper.RawBinaryBlob();
        REQUIRE(buf.size() == sizeof(dcgmDiagResponse_v11));
    }

    SECTION("Version 10")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version10);

        auto buf = wrapper.RawBinaryBlob();
        REQUIRE(buf.size() == sizeof(dcgmDiagResponse_v10));
    }

    SECTION("Version 9")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version9);

        auto buf = wrapper.RawBinaryBlob();
        REQUIRE(buf.size() == sizeof(dcgmDiagResponse_v9));
    }

    SECTION("Version 8")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version8);

        auto buf = wrapper.RawBinaryBlob();
        REQUIRE(buf.size() == sizeof(dcgmDiagResponse_v8));
    }

    SECTION("Version 7")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version7);

        auto buf = wrapper.RawBinaryBlob();
        REQUIRE(buf.size() == sizeof(dcgmDiagResponse_v7));
    }
}

std::unique_ptr<dcgmDiagEntityResults_v1> CreateFakeEntityResults(unsigned int numErr,
                                                                  unsigned int numInfo,
                                                                  unsigned int numResults)
{
    std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = std::make_unique<dcgmDiagEntityResults_v1>();

    assert(numErr < DCGM_DIAG_RESPONSE_ERRORS_MAX);
    assert(numInfo < DCGM_DIAG_RESPONSE_INFO_MAX);
    assert(numResults < DCGM_DIAG_TEST_RUN_RESULTS_MAX);

    for (unsigned int i = 0; i < numErr; ++i)
    {
        entityResultsPtr->errors[i].entity   = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
        entityResultsPtr->errors[i].category = DCGM_FR_EC_INTERNAL_OTHER;
        entityResultsPtr->errors[i].code     = 0xc8763;
        entityResultsPtr->errors[i].severity = DCGM_ERROR_ISOLATE;
        entityResultsPtr->errors[i].testId   = 0;
        std::string msg                      = fmt::format("error_{}", i);
        SafeCopyTo(entityResultsPtr->errors[i].msg, msg.c_str());
        entityResultsPtr->numErrors += 1;
    }

    for (unsigned int i = 0; i < numInfo; ++i)
    {
        entityResultsPtr->info[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
        entityResultsPtr->info[i].testId = 0;
        std::string msg                  = fmt::format("info_{}", i);
        SafeCopyTo(entityResultsPtr->info[i].msg, msg.c_str());
        entityResultsPtr->numInfo += 1;
    }

    for (unsigned int i = 0; i < numResults; ++i)
    {
        entityResultsPtr->results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
        entityResultsPtr->results[i].testId = 0;
        entityResultsPtr->results[i].result = (i < numErr) ? DCGM_DIAG_RESULT_FAIL : DCGM_DIAG_RESULT_PASS;
        entityResultsPtr->numResults += 1;
    }

    return entityResultsPtr;
}

TEST_CASE("DcgmNvvsResponseWrapper::SetSoftwareTestResult")
{
    SECTION("Version 11")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();

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

    SECTION("Version 10")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version10);

        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v10>();

        REQUIRE(rawResponse.levelOneTestCount == 2);
        for (unsigned int i = 0; i < 2; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.levelOneResults[i].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
            REQUIRE(rawResponse.levelOneResults[i].error[0].code == 0xc8763);
            REQUIRE(rawResponse.levelOneResults[i].error[0].severity == DCGM_ERROR_ISOLATE);
            REQUIRE(rawResponse.levelOneResults[i].error[0].gpuId == 0);
            REQUIRE(std::string_view(rawResponse.levelOneResults[i].error[0].msg) == "error_0");
            REQUIRE(std::string_view(rawResponse.levelOneResults[i].info) == "info_0, info_1");
        }

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 9")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version9);

        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v9>();

        REQUIRE(rawResponse.levelOneTestCount == 2);
        for (unsigned int i = 0; i < 2; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.levelOneResults[i].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
            REQUIRE(rawResponse.levelOneResults[i].error[0].code == 0xc8763);
            REQUIRE(rawResponse.levelOneResults[i].error[0].severity == DCGM_ERROR_ISOLATE);
            REQUIRE(rawResponse.levelOneResults[i].error[0].gpuId == 0);
            REQUIRE(std::string_view(rawResponse.levelOneResults[i].error[0].msg) == "error_0");
            REQUIRE(std::string_view(rawResponse.levelOneResults[i].info) == "info_0, info_1");
        }

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 8")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version8);

        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v8>();

        REQUIRE(rawResponse.levelOneTestCount == 2);
        for (unsigned int i = 0; i < 2; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.levelOneResults[i].error.code == 0xc8763);
            REQUIRE(std::string_view(rawResponse.levelOneResults[i].error.msg) == "error_0");
            REQUIRE(std::string_view(rawResponse.levelOneResults[i].info) == "info_0, info_1");
        }

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 7")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version7);

        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);
        REQUIRE(wrapper.SetSoftwareTestResult("NVML Library", NVVS_RESULT_FAIL, *entityResultsPtr) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v7>();

        REQUIRE(rawResponse.levelOneTestCount == 2);
        for (unsigned int i = 0; i < 2; ++i)
        {
            REQUIRE(rawResponse.levelOneResults[i].status == DCGM_DIAG_RESULT_FAIL);
            REQUIRE(rawResponse.levelOneResults[i].error.code == 0xc8763);
            REQUIRE(std::string_view(rawResponse.levelOneResults[i].error.msg) == "error_0");
            REQUIRE(std::string_view(rawResponse.levelOneResults[i].info) == "info_0, info_1");
        }

        ReleaseEntitySets(entitySets);
    }

    SECTION("Overwrite SW test result of entities")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> twoEntitiesResultsPtr = std::make_unique<dcgmDiagEntityResults_v1>();

        for (unsigned int i = 0; i < 2; ++i)
        {
            twoEntitiesResultsPtr->results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
            twoEntitiesResultsPtr->results[i].testId = 0;
            twoEntitiesResultsPtr->numResults += 1;
        }
        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_SKIP;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_NOT_RUN;
        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_PASS, *twoEntitiesResultsPtr) == DCGM_ST_OK);
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();

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
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> twoEntitiesResultsPtr = std::make_unique<dcgmDiagEntityResults_v1>();

        for (unsigned int i = 0; i < 2; ++i)
        {
            twoEntitiesResultsPtr->results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
            twoEntitiesResultsPtr->results[i].testId = 0;
            twoEntitiesResultsPtr->numResults += 1;
        }
        twoEntitiesResultsPtr->results[0].result = DCGM_DIAG_RESULT_NOT_RUN;
        twoEntitiesResultsPtr->results[1].result = DCGM_DIAG_RESULT_NOT_RUN;

        REQUIRE(wrapper.SetSoftwareTestResult("Denylist", NVVS_RESULT_SKIP, *twoEntitiesResultsPtr) == DCGM_ST_OK);
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();
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

TEST_CASE("DcgmNvvsResponseWrapper::IncreaseNumTests")
{
    SECTION("Version 11")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();
        wrapper.IncreaseNumTests();
        REQUIRE(rawResponse.numTests == 1);
        wrapper.IncreaseNumTests();
        REQUIRE(rawResponse.numTests == 2);
    }

    SECTION("Version 10")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version10);
        REQUIRE_NOTHROW(wrapper.IncreaseNumTests());
    }

    SECTION("Version 9")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version9);
        REQUIRE_NOTHROW(wrapper.IncreaseNumTests());
    }

    SECTION("Version 8")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version8);
        REQUIRE_NOTHROW(wrapper.IncreaseNumTests());
    }

    SECTION("Version 7")
    {
        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version7);
        REQUIRE_NOTHROW(wrapper.IncreaseNumTests());
    }
}

TEST_CASE("DcgmNvvsResponseWrapper::SetTestResult")
{
    SECTION("Version 11")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));
        ::Json::Value auxData;

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        auxData["hello"] = "eud";
        REQUIRE(wrapper.SetTestResult("eud", "eud", *entityResultsPtr, auxData) == DCGM_ST_OK);
        auxData["hello"] = "memory";
        REQUIRE(wrapper.SetTestResult("memory", "memory", *entityResultsPtr, auxData) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();

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

        REQUIRE(rawResponse.numResults == 4);
        REQUIRE(rawResponse.results[0].entity == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 });
        REQUIRE(rawResponse.results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.results[1].entity == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 });
        REQUIRE(rawResponse.results[1].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.results[2].entity == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 });
        REQUIRE(rawResponse.results[2].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.results[3].entity == dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 });
        REQUIRE(rawResponse.results[3].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(std::string_view(rawResponse.tests[2].name).empty());

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 11: Overall Result")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 5, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));
        ::Json::Value auxData;

        std::unique_ptr<dcgmDiagEntityResults_v1> entitiesResultsPtr = std::make_unique<dcgmDiagEntityResults_v1>();

        for (unsigned int i = 0; i < 5; ++i)
        {
            entitiesResultsPtr->results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = i };
            entitiesResultsPtr->results[i].testId = 0;
            entitiesResultsPtr->numResults += 1;
            entitiesResultsPtr->results[i].result = DCGM_DIAG_RESULT_NOT_RUN;
        }
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();

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
        REQUIRE(wrapper.SetTestResult("targeted_power", "targeted_power", *entitiesResultsPtr, auxData) == DCGM_ST_OK);
        REQUIRE(rawResponse.tests[4].result == DCGM_DIAG_RESULT_FAIL);

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 10")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version10);

        REQUIRE(wrapper.PopulateDefault(entitySets));
        ::Json::Value auxData;

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        auxData["hello"] = "eud";
        REQUIRE(wrapper.SetTestResult("eud", "eud", *entityResultsPtr, auxData) == DCGM_ST_OK);
        auxData["hello"] = "memory";
        REQUIRE(wrapper.SetTestResult("memory", "memory", *entityResultsPtr, auxData) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v10>();

        unsigned int const eudIdx    = GetTestIndex("eud");
        unsigned int const memoryIdx = GetTestIndex("memory");
        REQUIRE(rawResponse.perGpuResponses[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error[0].code == 0xc8763);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error[0].severity == DCGM_ERROR_ISOLATE);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[eudIdx].error[0].msg) == "error_0");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[eudIdx].info) == "info_0");
        REQUIRE(rawResponse.auxDataPerTest[eudIdx].version == dcgmDiagTestAuxData_version1);
        REQUIRE(std::string_view(rawResponse.auxDataPerTest[eudIdx].data) == "{\"hello\":\"eud\"}");

        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].code == 0xc8763);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].severity == DCGM_ERROR_ISOLATE);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].msg) == "error_0");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[memoryIdx].info) == "info_0");
        REQUIRE(rawResponse.auxDataPerTest[memoryIdx].version == dcgmDiagTestAuxData_version1);
        REQUIRE(std::string_view(rawResponse.auxDataPerTest[memoryIdx].data) == "{\"hello\":\"memory\"}");

        REQUIRE(rawResponse.perGpuResponses[1].gpuId == 1);
        REQUIRE(rawResponse.perGpuResponses[1].results[eudIdx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[eudIdx].error[0].code == 0);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[eudIdx].info) == "info_1");
        REQUIRE(rawResponse.perGpuResponses[1].results[memoryIdx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[memoryIdx].error[0].code == 0);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[memoryIdx].info) == "info_1");

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 9")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version9);

        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);
        ::Json::Value auxData;

        auxData["hello"] = "eud";
        REQUIRE(wrapper.SetTestResult("eud", "eud", *entityResultsPtr, auxData) == DCGM_ST_OK);
        auxData["hello"] = "memory";
        REQUIRE(wrapper.SetTestResult("memory", "memory", *entityResultsPtr, auxData) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v9>();

        unsigned int const eudIdx    = GetTestIndex("eud");
        unsigned int const memoryIdx = GetTestIndex("memory");
        REQUIRE(rawResponse.perGpuResponses[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error[0].code == 0xc8763);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error[0].severity == DCGM_ERROR_ISOLATE);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[eudIdx].error[0].msg) == "error_0");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[eudIdx].info) == "info_0");

        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].category == DCGM_FR_EC_INTERNAL_OTHER);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].code == 0xc8763);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].severity == DCGM_ERROR_ISOLATE);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[memoryIdx].error[0].msg) == "error_0");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[memoryIdx].info) == "info_0");

        REQUIRE(rawResponse.perGpuResponses[1].gpuId == 1);
        REQUIRE(rawResponse.perGpuResponses[1].results[eudIdx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[eudIdx].error[0].code == 0);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[eudIdx].info) == "info_1");
        REQUIRE(rawResponse.perGpuResponses[1].results[memoryIdx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[memoryIdx].error[0].code == 0);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[memoryIdx].info) == "info_1");

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 8")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version8);

        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);
        ::Json::Value auxData;

        auxData["hello"] = "eud";
        REQUIRE(wrapper.SetTestResult("eud", "eud", *entityResultsPtr, auxData) == DCGM_ST_OK);
        auxData["hello"] = "memory";
        REQUIRE(wrapper.SetTestResult("memory", "memory", *entityResultsPtr, auxData) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v8>();

        unsigned int const eudIdx    = GetTestIndex("eud");
        unsigned int const memoryIdx = GetTestIndex("memory");
        REQUIRE(rawResponse.perGpuResponses[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[0].results[eudIdx].error.code == 0xc8763);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[eudIdx].error.msg) == "error_0");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[eudIdx].info) == "info_0");

        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error.code == 0xc8763);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[memoryIdx].error.msg) == "error_0");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[memoryIdx].info) == "info_0");

        REQUIRE(rawResponse.perGpuResponses[1].gpuId == 1);
        REQUIRE(rawResponse.perGpuResponses[1].results[eudIdx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[eudIdx].error.code == 0);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[eudIdx].info) == "info_1");
        REQUIRE(rawResponse.perGpuResponses[1].results[memoryIdx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[memoryIdx].error.code == 0);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[memoryIdx].info) == "info_1");

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 7")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 1);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version7);

        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);
        ::Json::Value auxData;

        auxData["hello"] = "pcie";
        REQUIRE(wrapper.SetTestResult("pcie", "pcie", *entityResultsPtr, auxData) == DCGM_ST_OK);
        auxData["hello"] = "memory";
        REQUIRE(wrapper.SetTestResult("memory", "memory", *entityResultsPtr, auxData) == DCGM_ST_OK);

        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v7>();

        unsigned int const pcieIdx   = GetTestIndex("pcie");
        unsigned int const memoryIdx = GetTestIndex("memory");
        REQUIRE(rawResponse.perGpuResponses[0].gpuId == 0);
        REQUIRE(rawResponse.perGpuResponses[0].results[pcieIdx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[0].results[pcieIdx].error.code == 0xc8763);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[pcieIdx].error.msg) == "error_0");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[pcieIdx].info) == "info_0");

        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(rawResponse.perGpuResponses[0].results[memoryIdx].error.code == 0xc8763);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[memoryIdx].error.msg) == "error_0");
        REQUIRE(std::string_view(rawResponse.perGpuResponses[0].results[memoryIdx].info) == "info_0");

        REQUIRE(rawResponse.perGpuResponses[1].gpuId == 1);
        REQUIRE(rawResponse.perGpuResponses[1].results[pcieIdx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[pcieIdx].error.code == 0);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[pcieIdx].info) == "info_1");
        REQUIRE(rawResponse.perGpuResponses[1].results[memoryIdx].status == DCGM_DIAG_RESULT_PASS);
        REQUIRE(rawResponse.perGpuResponses[1].results[memoryIdx].error.code == 0);
        REQUIRE(std::string_view(rawResponse.perGpuResponses[1].results[memoryIdx].info) == "info_1");

        ReleaseEntitySets(entitySets);
    }
}

template <typename T>
void TestSetTestSkipped(unsigned int version)
{
    auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

    DcgmNvvsResponseWrapper wrapper;
    wrapper.SetVersion(version);
    REQUIRE(wrapper.PopulateDefault(entitySets));

    REQUIRE(wrapper.SetTestSkipped(PCIE_PLUGIN_NAME, PCIE_PLUGIN_NAME) == DCGM_ST_OK);
    auto const &rawResponse = wrapper.ConstResponse<T>();

    auto const pcieIdx = GetTestIndex(PCIE_PLUGIN_NAME);
    REQUIRE(pcieIdx < std::size(rawResponse.perGpuResponses[0].results));

    // coverity[overrun] - the previous REQUIRE should prevent this condition
    REQUIRE(rawResponse.perGpuResponses[0].gpuId == 0);
    REQUIRE(rawResponse.perGpuResponses[0].results[pcieIdx].status == DCGM_DIAG_RESULT_SKIP);
    REQUIRE(rawResponse.perGpuResponses[1].gpuId == 1);
    REQUIRE(rawResponse.perGpuResponses[1].results[pcieIdx].status == DCGM_DIAG_RESULT_SKIP);

    ReleaseEntitySets(entitySets);
}

TEST_CASE("DcgmNvvsResponseWrapper::SetTestSkipped")
{
    SECTION("Version 11")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        REQUIRE(wrapper.SetTestSkipped("eud_plugin", "eud") == DCGM_ST_OK);
        auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();

        REQUIRE(rawResponse.numTests == 1);
        REQUIRE(std::string_view(rawResponse.tests[0].name) == "eud");
        REQUIRE(std::string_view(rawResponse.tests[0].pluginName) == "eud_plugin");
        REQUIRE(rawResponse.tests[0].result == DCGM_DIAG_RESULT_SKIP);
        REQUIRE(rawResponse.tests[0].numResults == 0);

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 10")
    {
        TestSetTestSkipped<dcgmDiagResponse_v10>(dcgmDiagResponse_version10);
    }

    SECTION("Version 9")
    {
        TestSetTestSkipped<dcgmDiagResponse_v9>(dcgmDiagResponse_version9);
    }

    SECTION("Version 8")
    {
        TestSetTestSkipped<dcgmDiagResponse_v8>(dcgmDiagResponse_version8);
    }

    SECTION("Version 7")
    {
        TestSetTestSkipped<dcgmDiagResponse_v7>(dcgmDiagResponse_version7);
    }
}

template <typename T>
void TestSetSystemError(unsigned int version)
{
    auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

    DcgmNvvsResponseWrapper wrapper;
    wrapper.SetVersion(version);
    REQUIRE(wrapper.PopulateDefault(entitySets));

    wrapper.SetSystemError("Capoo", 0xc8763);
    auto const &rawResponse = wrapper.ConstResponse<T>();
    REQUIRE(rawResponse.systemError.code == 0xc8763);
    REQUIRE(std::string_view(rawResponse.systemError.msg) == "Capoo");

    ReleaseEntitySets(entitySets);
}

template <>
void TestSetSystemError<dcgmDiagResponse_v11>(unsigned int version)
{
    auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

    DcgmNvvsResponseWrapper wrapper;
    wrapper.SetVersion(version);
    REQUIRE(wrapper.PopulateDefault(entitySets));

    wrapper.SetSystemError("Capoo", 0xc8763);

    auto const &rawResponse = wrapper.ConstResponse<dcgmDiagResponse_v11>();
    CHECK(rawResponse.numErrors == 1);
    CHECK(rawResponse.errors[0].testId == DCGM_DIAG_RESPONSE_SYSTEM_ERROR);
    CHECK(rawResponse.errors[0].code == 0xc8763);
    CHECK(std::string_view(rawResponse.errors[0].msg) == "Capoo");

    ReleaseEntitySets(entitySets);
}

TEST_CASE("DcgmNvvsResponseWrapper::SetSystemError")
{
    SECTION("Version 11")
    {
        TestSetSystemError<dcgmDiagResponse_v11>(dcgmDiagResponse_version11);
    }

    SECTION("Version 10")
    {
        TestSetSystemError<dcgmDiagResponse_v10>(dcgmDiagResponse_version10);
    }

    SECTION("Version 9")
    {
        TestSetSystemError<dcgmDiagResponse_v9>(dcgmDiagResponse_version9);
    }

    SECTION("Version 8")
    {
        TestSetSystemError<dcgmDiagResponse_v8>(dcgmDiagResponse_version8);
    }

    SECTION("Version 7")
    {
        TestSetSystemError<dcgmDiagResponse_v7>(dcgmDiagResponse_version7);
    }
}

void TestTestSlotsFull(unsigned int version)
{
    DcgmNvvsResponseWrapper wrapper;
    wrapper.SetVersion(version);
    REQUIRE(!wrapper.TestSlotsFull());
}

TEST_CASE("DcgmNvvsResponseWrapper::TestSlotsFull")
{
    SECTION("Version 11")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));
        REQUIRE(!wrapper.TestSlotsFull());

        for (unsigned int i = 0; i < DCGM_DIAG_RESPONSE_TESTS_MAX; ++i)
        {
            REQUIRE(wrapper.SetTestSkipped("eud", "cpu_eud") == DCGM_ST_OK);
        }
        REQUIRE(wrapper.TestSlotsFull());

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 10")
    {
        TestTestSlotsFull(dcgmDiagResponse_version10);
    }

    SECTION("Version 9")
    {
        TestTestSlotsFull(dcgmDiagResponse_version9);
    }

    SECTION("Version 8")
    {
        TestTestSlotsFull(dcgmDiagResponse_version8);
    }

    SECTION("Version 7")
    {
        TestTestSlotsFull(dcgmDiagResponse_version7);
    }
}

void TestAddTestCategory(unsigned int version)
{
    auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

    DcgmNvvsResponseWrapper wrapper;
    wrapper.SetVersion(version);
    REQUIRE(wrapper.PopulateDefault(entitySets));

    REQUIRE(wrapper.AddTestCategory("name", "category") == DCGM_ST_VER_MISMATCH);

    ReleaseEntitySets(entitySets);
}

TEST_CASE("DcgmNvvsResponseWrapper::AddTestCategory")
{
    SECTION("Version 11: Basic")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);
        auto const &rawResponse                                    = wrapper.ConstResponse<dcgmDiagResponse_v11>();

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

    SECTION("Version 11: Full")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        std::unique_ptr<dcgmDiagEntityResults_v1> entityResultsPtr = CreateFakeEntityResults(1, 2, 2);

        REQUIRE(wrapper.SetTestResult("eud", "eud", *entityResultsPtr, std::nullopt) == DCGM_ST_OK);

        for (unsigned int i = 0; i < DCGM_DIAG_RESPONSE_CATEGORIES_MAX; ++i)
        {
            REQUIRE(wrapper.AddTestCategory("eud", fmt::format("{}", i)) == DCGM_ST_OK);
        }
        REQUIRE(wrapper.AddTestCategory("eud", "Capoo") == DCGM_ST_INSUFFICIENT_RESOURCES);

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 11: Unknown Test Name")
    {
        auto entitySets = CreateFakeEntitySets("545.29.06", 2, 2);

        DcgmNvvsResponseWrapper wrapper;
        wrapper.SetVersion(dcgmDiagResponse_version11);
        REQUIRE(wrapper.PopulateDefault(entitySets));

        REQUIRE(wrapper.AddTestCategory("name", "category") == DCGM_ST_GENERIC_ERROR);

        ReleaseEntitySets(entitySets);
    }

    SECTION("Version 10")
    {
        TestAddTestCategory(dcgmDiagResponse_version10);
    }

    SECTION("Version 9")
    {
        TestAddTestCategory(dcgmDiagResponse_version9);
    }

    SECTION("Version 8")
    {
        TestAddTestCategory(dcgmDiagResponse_version8);
    }

    SECTION("Version 7")
    {
        TestAddTestCategory(dcgmDiagResponse_version7);
    }
}
