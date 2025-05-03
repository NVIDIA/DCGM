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

#include <DcgmStringHelpers.h>
#include <DiagResponseUtils.h>
#include <PluginStrings.h>
#include <UniquePtrUtil.h>
#include <dcgm_errors.h>

#define __DIAG_UNIT_TESTING__
#include <DcgmDiagResponseWrapper.h>

TEST_CASE("DcgmDiagResponseWrapper: HasTest")
{
    SECTION("Unsupported version")
    {
        DcgmDiagResponseWrapper ddr;
        auto dr12Uptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        auto &dr12    = *dr12Uptr;

        dr12.numTests           = 1;
        std::string const capoo = "capoo";
        SafeCopyTo(dr12.tests[0].name, capoo.c_str());

        REQUIRE(ddr.SetVersion(&dr12) == DCGM_ST_OK);
        ddr.m_version = dcgmDiagResponse_version9;
        CHECK(ddr.HasTest(capoo) == false);
        CHECK(ddr.HasTest("dogdog") == false);
    }

    SECTION("Basic case")
    {
        DcgmDiagResponseWrapper ddr;
        auto dr12Uptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        auto &dr12    = *dr12Uptr;

        dr12.numTests            = 2;
        std::string const capoo  = "capoo";
        std::string const dogdog = "dogdog";
        SafeCopyTo(dr12.tests[0].name, capoo.c_str());
        SafeCopyTo(dr12.tests[1].name, dogdog.c_str());

        REQUIRE(ddr.SetVersion(&dr12) == DCGM_ST_OK);

        CHECK(ddr.HasTest(capoo) == true);
        CHECK(ddr.HasTest("dogdog") == true);
        CHECK(ddr.HasTest("rabbit") == false);
    }
}

TEST_CASE("DcgmDiagResponseWrapper::MergeEudResponse::Version Mismatch")
{
    SECTION("v12/v11 mismatch")
    {
        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        dest.SetVersion(destResponse.get());

        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v11>();
        src.SetVersion(srcResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_VER_MISMATCH);
    }

    SECTION("v11/v10 mismatch")
    {
        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v11>();
        dest.SetVersion(destResponse.get());

        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v10>();
        src.SetVersion(srcResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_VER_MISMATCH);
    }
}

TEST_CASE("DcgmDiagResponseWrapper::MergeEudResponse::v12")
{
    SECTION("Version 12: Destination includes eud results with non-DCGM_FR_EUD_NON_ROOT_USER code")
    {
        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(srcResponse->tests[0].name, EUD_PLUGIN_NAME);
        srcResponse->tests[0].result = DCGM_DIAG_RESULT_PASS;
        srcResponse->numTests        = 1;
        src.SetVersion(srcResponse.get());

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(destResponse->tests[0].name, EUD_PLUGIN_NAME);
        destResponse->tests[0].result          = DCGM_DIAG_RESULT_FAIL;
        destResponse->tests[0].numErrors       = 1;
        destResponse->tests[0].errorIndices[0] = 0;
        destResponse->errors[0].code           = DCGM_FR_EUD_TEST_FAILED;
        destResponse->numTests                 = 1;
        dest.SetVersion(destResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(destResponse->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    }

    SECTION("Version 12: Category found")
    {
        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(srcResponse->tests[0].name, EUD_PLUGIN_NAME);
        srcResponse->tests[0].result = DCGM_DIAG_RESULT_PASS;
        srcResponse->numTests        = 1;
        srcResponse->numCategories   = 1;
        SafeCopyTo(srcResponse->categories[0], PLUGIN_CATEGORY_HW);
        src.SetVersion(srcResponse.get());

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(destResponse->tests[1].name, EUD_PLUGIN_NAME);
        destResponse->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        destResponse->tests[1].numErrors       = 1;
        destResponse->tests[1].errorIndices[0] = 0;
        destResponse->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        destResponse->numErrors                = 1;
        destResponse->numTests                 = 2;
        destResponse->numCategories            = 2;
        SafeCopyTo(destResponse->categories[0], SW_PLUGIN_NAME);
        SafeCopyTo(destResponse->categories[1], PLUGIN_CATEGORY_HW);
        dest.SetVersion(destResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(destResponse->tests[1].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(destResponse->tests[1].categoryIndex == 1);
        REQUIRE(destResponse->tests[1].numErrors == 0);
    }

    SECTION("Version 12: Category not found")
    {
        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(srcResponse->tests[0].name, EUD_PLUGIN_NAME);
        srcResponse->tests[0].result = DCGM_DIAG_RESULT_PASS;
        srcResponse->numTests        = 1;
        srcResponse->numCategories   = 1;
        SafeCopyTo(srcResponse->categories[0], PLUGIN_CATEGORY_HW);
        src.SetVersion(srcResponse.get());

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(destResponse->tests[0].name, EUD_PLUGIN_NAME);
        destResponse->tests[0].result          = DCGM_DIAG_RESULT_FAIL;
        destResponse->tests[0].numErrors       = 1;
        destResponse->tests[0].errorIndices[0] = 0;
        destResponse->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        destResponse->numErrors                = 1;
        destResponse->numTests                 = 1;
        dest.SetVersion(destResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destResponse->tests[0].name) == EUD_PLUGIN_NAME);
        REQUIRE(destResponse->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(destResponse->tests[0].categoryIndex == 0);
        REQUIRE(destResponse->tests[0].numErrors == 0);
    }

    SECTION("Version 12: Merge error")
    {
        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(srcResponse->tests[0].name, EUD_PLUGIN_NAME);
        srcResponse->tests[0].result          = DCGM_DIAG_RESULT_FAIL;
        srcResponse->tests[0].errorIndices[0] = 0;
        srcResponse->tests[0].errorIndices[1] = 1;
        srcResponse->numTests                 = 1;
        srcResponse->numErrors                = 2;
        srcResponse->errors[0].code           = 0xc8763;
        srcResponse->errors[1].code           = 3345678;
        srcResponse->numCategories            = 1;
        SafeCopyTo(srcResponse->categories[0], PLUGIN_CATEGORY_HW);
        src.SetVersion(srcResponse.get());

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(destResponse->tests[0].name, MEMORY_PLUGIN_NAME);
        SafeCopyTo(destResponse->tests[1].name, EUD_PLUGIN_NAME);
        destResponse->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        destResponse->tests[1].numErrors       = 1;
        destResponse->tests[1].errorIndices[0] = 0;
        destResponse->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        destResponse->numErrors                = 1;
        destResponse->numTests                 = 2;
        dest.SetVersion(destResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destResponse->tests[0].name) == MEMORY_PLUGIN_NAME);
        REQUIRE(std::string_view(destResponse->tests[1].name) == EUD_PLUGIN_NAME);
        REQUIRE(destResponse->numTests == 2);
        REQUIRE(destResponse->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(destResponse->tests[1].numErrors == 2);
        REQUIRE(destResponse->tests[1].errorIndices[0] == 0);
        REQUIRE(destResponse->tests[1].errorIndices[1] == 1);
        REQUIRE(destResponse->numErrors == 2);
        REQUIRE(destResponse->errors[0].code == 0xc8763);
        REQUIRE(destResponse->errors[0].testId == 1);
        REQUIRE(destResponse->errors[1].code == 3345678);
        REQUIRE(destResponse->errors[1].testId == 1);
    }

    SECTION("Version 12: Merge info")
    {
        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(srcResponse->tests[0].name, EUD_PLUGIN_NAME);
        srcResponse->tests[0].result         = DCGM_DIAG_RESULT_FAIL;
        srcResponse->tests[0].infoIndices[0] = 0;
        srcResponse->tests[0].infoIndices[1] = 1;
        srcResponse->numTests                = 1;
        srcResponse->numInfo                 = 2;
        SafeCopyTo(srcResponse->info[0].msg, "Hello Capoo!");
        SafeCopyTo(srcResponse->info[1].msg, "Hello Dogdog!");
        srcResponse->numCategories = 1;
        SafeCopyTo(srcResponse->categories[0], PLUGIN_CATEGORY_HW);
        src.SetVersion(srcResponse.get());

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(destResponse->tests[0].name, MEMORY_PLUGIN_NAME);
        SafeCopyTo(destResponse->tests[1].name, EUD_PLUGIN_NAME);
        destResponse->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        destResponse->tests[1].numErrors       = 1;
        destResponse->tests[1].errorIndices[0] = 0;
        destResponse->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        destResponse->numErrors                = 1;
        destResponse->numTests                 = 2;
        dest.SetVersion(destResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destResponse->tests[0].name) == MEMORY_PLUGIN_NAME);
        REQUIRE(std::string_view(destResponse->tests[1].name) == EUD_PLUGIN_NAME);
        REQUIRE(destResponse->numTests == 2);
        REQUIRE(destResponse->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(destResponse->tests[1].numInfo == 2);
        REQUIRE(destResponse->tests[1].infoIndices[0] == 0);
        REQUIRE(destResponse->tests[1].infoIndices[1] == 1);
        REQUIRE(destResponse->numInfo == 2);
        REQUIRE(std::string_view(destResponse->info[0].msg) == "Hello Capoo!");
        REQUIRE(destResponse->info[0].testId == 1);
        REQUIRE(std::string_view(destResponse->info[1].msg) == "Hello Dogdog!");
        REQUIRE(destResponse->info[1].testId == 1);
    }

    SECTION("Version 12: Merge empty results")
    {
        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(srcResponse->tests[0].name, EUD_PLUGIN_NAME);
        srcResponse->tests[0].result           = DCGM_DIAG_RESULT_FAIL;
        srcResponse->tests[0].resultIndices[0] = 0;
        srcResponse->tests[0].resultIndices[1] = 1;
        srcResponse->numTests                  = 1;
        srcResponse->numResults                = 2;
        srcResponse->results[0].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        srcResponse->results[0].result         = DCGM_DIAG_RESULT_FAIL;
        srcResponse->results[1].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
        srcResponse->results[1].result         = DCGM_DIAG_RESULT_PASS;
        srcResponse->numCategories             = 1;
        SafeCopyTo(srcResponse->categories[0], PLUGIN_CATEGORY_HW);
        src.SetVersion(srcResponse.get());

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(destResponse->tests[0].name, MEMORY_PLUGIN_NAME);
        SafeCopyTo(destResponse->tests[1].name, EUD_PLUGIN_NAME);
        destResponse->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        destResponse->tests[1].numErrors       = 1;
        destResponse->tests[1].errorIndices[0] = 0;
        destResponse->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        destResponse->numErrors                = 1;
        destResponse->numTests                 = 2;
        dest.SetVersion(destResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destResponse->tests[0].name) == MEMORY_PLUGIN_NAME);
        REQUIRE(std::string_view(destResponse->tests[1].name) == EUD_PLUGIN_NAME);
        REQUIRE(destResponse->numTests == 2);
        REQUIRE(destResponse->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(destResponse->tests[1].numResults == 2);
        REQUIRE(destResponse->tests[1].resultIndices[0] == 0);
        REQUIRE(destResponse->tests[1].resultIndices[1] == 1);
        REQUIRE(destResponse->numResults == 2);
        REQUIRE(destResponse->results[0].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(destResponse->results[0].entity.entityId == 0);
        REQUIRE(destResponse->results[0].testId == 1);
        REQUIRE(destResponse->results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(destResponse->results[1].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(destResponse->results[1].entity.entityId == 1);
        REQUIRE(destResponse->results[1].testId == 1);
        REQUIRE(destResponse->results[1].result == DCGM_DIAG_RESULT_PASS);
    }

    SECTION("Version 12: Merge non-empty results")
    {
        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(srcResponse->tests[0].name, EUD_PLUGIN_NAME);
        srcResponse->tests[0].result           = DCGM_DIAG_RESULT_FAIL;
        srcResponse->tests[0].resultIndices[0] = 0;
        srcResponse->tests[0].resultIndices[1] = 1;
        srcResponse->tests[0].numResults       = 2;
        srcResponse->numTests                  = 1;
        srcResponse->numResults                = 2;
        srcResponse->results[0].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        srcResponse->results[0].result         = DCGM_DIAG_RESULT_FAIL;
        srcResponse->results[0].testId         = 0;
        srcResponse->results[1].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
        srcResponse->results[1].result         = DCGM_DIAG_RESULT_PASS;
        srcResponse->results[1].testId         = 0;
        srcResponse->numCategories             = 1;
        SafeCopyTo(srcResponse->categories[0], PLUGIN_CATEGORY_HW);
        src.SetVersion(srcResponse.get());

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(destResponse->tests[0].name, MEMORY_PLUGIN_NAME);
        SafeCopyTo(destResponse->tests[1].name, EUD_PLUGIN_NAME);
        destResponse->tests[1].result           = DCGM_DIAG_RESULT_FAIL;
        destResponse->tests[1].numErrors        = 1;
        destResponse->tests[1].errorIndices[0]  = 0;
        destResponse->errors[0].code            = DCGM_FR_EUD_NON_ROOT_USER;
        destResponse->numErrors                 = 1;
        destResponse->tests[1].resultIndices[0] = 0;
        destResponse->tests[1].resultIndices[1] = 1;
        destResponse->tests[1].numResults       = 2;
        destResponse->results[0].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        destResponse->results[0].result         = DCGM_DIAG_RESULT_FAIL;
        destResponse->results[0].testId         = 1;
        destResponse->results[1].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
        destResponse->results[1].result         = DCGM_DIAG_RESULT_FAIL;
        destResponse->results[1].testId         = 1;
        destResponse->numResults                = 2;
        destResponse->numTests                  = 2;
        dest.SetVersion12(destResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destResponse->tests[0].name) == MEMORY_PLUGIN_NAME);
        REQUIRE(std::string_view(destResponse->tests[1].name) == EUD_PLUGIN_NAME);
        REQUIRE(destResponse->numTests == 2);
        REQUIRE(destResponse->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(destResponse->tests[1].numResults == 2);
        REQUIRE(destResponse->tests[1].resultIndices[0] == 0);
        REQUIRE(destResponse->tests[1].resultIndices[1] == 1);
        REQUIRE(destResponse->numResults == 2);
        REQUIRE(destResponse->results[0].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(destResponse->results[0].entity.entityId == 0);
        REQUIRE(destResponse->results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(destResponse->results[0].testId == 1);
        REQUIRE(destResponse->results[1].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(destResponse->results[1].entity.entityId == 1);
        REQUIRE(destResponse->results[1].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(destResponse->results[1].testId == 1);
    }

    SECTION("Version 12: Merge aux")
    {
        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        SafeCopyTo(srcResponse->tests[0].name, EUD_PLUGIN_NAME);
        srcResponse->tests[0].result = DCGM_DIAG_RESULT_PASS;
        SafeCopyTo(srcResponse->tests[0].auxData.data, "Hello Capoo!");
        srcResponse->tests[0].auxData.version = dcgmDiagTestAuxData_version1;
        srcResponse->numTests                 = 1;
        srcResponse->numErrors                = 0;
        srcResponse->numCategories            = 1;
        SafeCopyTo(srcResponse->categories[0], PLUGIN_CATEGORY_HW);
        src.SetVersion(srcResponse.get());

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        memset(destResponse.get(), 0, sizeof(dcgmDiagResponse_v12));
        SafeCopyTo(destResponse->tests[0].name, MEMORY_PLUGIN_NAME);
        SafeCopyTo(destResponse->tests[1].name, EUD_PLUGIN_NAME);
        destResponse->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        destResponse->tests[1].numErrors       = 1;
        destResponse->tests[1].errorIndices[0] = 0;
        destResponse->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        destResponse->numErrors                = 1;
        destResponse->numTests                 = 2;
        dest.SetVersion(destResponse.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(destResponse->numTests == 2);
        REQUIRE(std::string_view(destResponse->tests[0].name) == MEMORY_PLUGIN_NAME);
        REQUIRE(std::string_view(destResponse->tests[1].name) == EUD_PLUGIN_NAME);
        REQUIRE(destResponse->tests[1].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(std::string_view(destResponse->tests[1].auxData.data) == "Hello Capoo!");
        REQUIRE(destResponse->tests[1].auxData.version == dcgmDiagTestAuxData_version1);
        REQUIRE(destResponse->tests[1].numErrors == 0);
    }
}

TEST_CASE("DcgmDiagResponseWrapper::MergeEudResponse::v11")
{
    SECTION("Version 11: Destination includes eud results with non-DCGM_FR_EUD_NON_ROOT_USER code")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Src = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Src.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Src->tests[0].name, "eud");
        dr11Src->tests[0].result = DCGM_DIAG_RESULT_PASS;
        dr11Src->numTests        = 1;
        src.SetVersion11(dr11Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Dest = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Dest.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Dest->tests[0].name, "eud");
        dr11Dest->tests[0].result          = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->tests[0].numErrors       = 1;
        dr11Dest->tests[0].errorIndices[0] = 0;
        dr11Dest->errors[0].code           = DCGM_FR_EUD_TEST_FAILED;
        dr11Dest->numTests                 = 1;
        dest.SetVersion11(dr11Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(dr11Dest->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    }

    SECTION("Version 11: Category found")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Src = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Src.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Src->tests[0].name, "eud");
        dr11Src->tests[0].result = DCGM_DIAG_RESULT_PASS;
        dr11Src->numTests        = 1;
        dr11Src->numCategories   = 1;
        SafeCopyTo(dr11Src->categories[0], "hardware");
        src.SetVersion11(dr11Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Dest = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Dest.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Dest->tests[1].name, "eud");
        dr11Dest->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->tests[1].numErrors       = 1;
        dr11Dest->tests[1].errorIndices[0] = 0;
        dr11Dest->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        dr11Dest->numErrors                = 1;
        dr11Dest->numTests                 = 2;
        dr11Dest->numCategories            = 2;
        SafeCopyTo(dr11Dest->categories[0], "software");
        SafeCopyTo(dr11Dest->categories[1], "hardware");
        dest.SetVersion11(dr11Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(dr11Dest->tests[1].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(dr11Dest->tests[1].categoryIndex == 1);
        REQUIRE(dr11Dest->tests[1].numErrors == 0);
    }

    SECTION("Version 11: Category not found")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Src = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Src.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Src->tests[0].name, "eud");
        dr11Src->tests[0].result = DCGM_DIAG_RESULT_PASS;
        dr11Src->numTests        = 1;
        dr11Src->numCategories   = 1;
        SafeCopyTo(dr11Src->categories[0], "hardware");
        src.SetVersion11(dr11Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Dest = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Dest.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Dest->tests[0].name, "eud");
        dr11Dest->tests[0].result          = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->tests[0].numErrors       = 1;
        dr11Dest->tests[0].errorIndices[0] = 0;
        dr11Dest->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        dr11Dest->numErrors                = 1;
        dr11Dest->numTests                 = 1;
        dest.SetVersion11(dr11Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(dr11Dest->tests[0].name) == "eud");
        REQUIRE(dr11Dest->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(dr11Dest->tests[0].categoryIndex == 0);
        REQUIRE(dr11Dest->tests[0].numErrors == 0);
    }

    SECTION("Version 11: Merge error")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Src = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Src.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Src->tests[0].name, "eud");
        dr11Src->tests[0].result          = DCGM_DIAG_RESULT_FAIL;
        dr11Src->tests[0].errorIndices[0] = 0;
        dr11Src->tests[0].errorIndices[1] = 1;
        dr11Src->numTests                 = 1;
        dr11Src->numErrors                = 2;
        dr11Src->errors[0].code           = 0xc8763;
        dr11Src->errors[1].code           = 3345678;
        dr11Src->numCategories            = 1;
        SafeCopyTo(dr11Src->categories[0], "hardware");
        src.SetVersion11(dr11Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Dest = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Dest.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Dest->tests[0].name, "memory");
        SafeCopyTo(dr11Dest->tests[1].name, "eud");
        dr11Dest->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->tests[1].numErrors       = 1;
        dr11Dest->tests[1].errorIndices[0] = 0;
        dr11Dest->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        dr11Dest->numErrors                = 1;
        dr11Dest->numTests                 = 2;
        dest.SetVersion11(dr11Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(dr11Dest->tests[0].name) == "memory");
        REQUIRE(std::string_view(dr11Dest->tests[1].name) == "eud");
        REQUIRE(dr11Dest->numTests == 2);
        REQUIRE(dr11Dest->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(dr11Dest->tests[1].numErrors == 2);
        REQUIRE(dr11Dest->tests[1].errorIndices[0] == 0);
        REQUIRE(dr11Dest->tests[1].errorIndices[1] == 1);
        REQUIRE(dr11Dest->numErrors == 2);
        REQUIRE(dr11Dest->errors[0].code == 0xc8763);
        REQUIRE(dr11Dest->errors[0].testId == 1);
        REQUIRE(dr11Dest->errors[1].code == 3345678);
        REQUIRE(dr11Dest->errors[1].testId == 1);
    }

    SECTION("Version 11: Merge info")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Src = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Src.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Src->tests[0].name, "eud");
        dr11Src->tests[0].result         = DCGM_DIAG_RESULT_FAIL;
        dr11Src->tests[0].infoIndices[0] = 0;
        dr11Src->tests[0].infoIndices[1] = 1;
        dr11Src->numTests                = 1;
        dr11Src->numInfo                 = 2;
        SafeCopyTo(dr11Src->info[0].msg, "Hello Capoo!");
        SafeCopyTo(dr11Src->info[1].msg, "Hello Dogdog!");
        dr11Src->numCategories = 1;
        SafeCopyTo(dr11Src->categories[0], "hardware");
        src.SetVersion11(dr11Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Dest = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Dest.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Dest->tests[0].name, "memory");
        SafeCopyTo(dr11Dest->tests[1].name, "eud");
        dr11Dest->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->tests[1].numErrors       = 1;
        dr11Dest->tests[1].errorIndices[0] = 0;
        dr11Dest->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        dr11Dest->numErrors                = 1;
        dr11Dest->numTests                 = 2;
        dest.SetVersion11(dr11Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(dr11Dest->tests[0].name) == "memory");
        REQUIRE(std::string_view(dr11Dest->tests[1].name) == "eud");
        REQUIRE(dr11Dest->numTests == 2);
        REQUIRE(dr11Dest->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(dr11Dest->tests[1].numInfo == 2);
        REQUIRE(dr11Dest->tests[1].infoIndices[0] == 0);
        REQUIRE(dr11Dest->tests[1].infoIndices[1] == 1);
        REQUIRE(dr11Dest->numInfo == 2);
        REQUIRE(std::string_view(dr11Dest->info[0].msg) == "Hello Capoo!");
        REQUIRE(dr11Dest->info[0].testId == 1);
        REQUIRE(std::string_view(dr11Dest->info[1].msg) == "Hello Dogdog!");
        REQUIRE(dr11Dest->info[1].testId == 1);
    }

    SECTION("Version 11: Merge empty results")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Src = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Src.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Src->tests[0].name, "eud");
        dr11Src->tests[0].result           = DCGM_DIAG_RESULT_FAIL;
        dr11Src->tests[0].resultIndices[0] = 0;
        dr11Src->tests[0].resultIndices[1] = 1;
        dr11Src->numTests                  = 1;
        dr11Src->numResults                = 2;
        dr11Src->results[0].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        dr11Src->results[0].result         = DCGM_DIAG_RESULT_FAIL;
        dr11Src->results[1].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
        dr11Src->results[1].result         = DCGM_DIAG_RESULT_PASS;
        dr11Src->numCategories             = 1;
        SafeCopyTo(dr11Src->categories[0], "hardware");
        src.SetVersion11(dr11Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Dest = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Dest.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Dest->tests[0].name, "memory");
        SafeCopyTo(dr11Dest->tests[1].name, "eud");
        dr11Dest->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->tests[1].numErrors       = 1;
        dr11Dest->tests[1].errorIndices[0] = 0;
        dr11Dest->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        dr11Dest->numErrors                = 1;
        dr11Dest->numTests                 = 2;
        dest.SetVersion11(dr11Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(dr11Dest->tests[0].name) == "memory");
        REQUIRE(std::string_view(dr11Dest->tests[1].name) == "eud");
        REQUIRE(dr11Dest->numTests == 2);
        REQUIRE(dr11Dest->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(dr11Dest->tests[1].numResults == 2);
        REQUIRE(dr11Dest->tests[1].resultIndices[0] == 0);
        REQUIRE(dr11Dest->tests[1].resultIndices[1] == 1);
        REQUIRE(dr11Dest->numResults == 2);
        REQUIRE(dr11Dest->results[0].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(dr11Dest->results[0].entity.entityId == 0);
        REQUIRE(dr11Dest->results[0].testId == 1);
        REQUIRE(dr11Dest->results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(dr11Dest->results[1].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(dr11Dest->results[1].entity.entityId == 1);
        REQUIRE(dr11Dest->results[1].testId == 1);
        REQUIRE(dr11Dest->results[1].result == DCGM_DIAG_RESULT_PASS);
    }

    SECTION("Version 11: Merge non-empty results")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Src = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Src.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Src->tests[0].name, "eud");
        dr11Src->tests[0].result           = DCGM_DIAG_RESULT_FAIL;
        dr11Src->tests[0].resultIndices[0] = 0;
        dr11Src->tests[0].resultIndices[1] = 1;
        dr11Src->tests[0].numResults       = 2;
        dr11Src->numTests                  = 1;
        dr11Src->numResults                = 2;
        dr11Src->results[0].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        dr11Src->results[0].result         = DCGM_DIAG_RESULT_FAIL;
        dr11Src->results[0].testId         = 0;
        dr11Src->results[1].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
        dr11Src->results[1].result         = DCGM_DIAG_RESULT_PASS;
        dr11Src->results[1].testId         = 0;
        dr11Src->numCategories             = 1;
        SafeCopyTo(dr11Src->categories[0], "hardware");
        src.SetVersion11(dr11Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Dest = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Dest.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Dest->tests[0].name, "memory");
        SafeCopyTo(dr11Dest->tests[1].name, "eud");
        dr11Dest->tests[1].result           = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->tests[1].numErrors        = 1;
        dr11Dest->tests[1].errorIndices[0]  = 0;
        dr11Dest->errors[0].code            = DCGM_FR_EUD_NON_ROOT_USER;
        dr11Dest->numErrors                 = 1;
        dr11Dest->tests[1].resultIndices[0] = 0;
        dr11Dest->tests[1].resultIndices[1] = 1;
        dr11Dest->tests[1].numResults       = 2;
        dr11Dest->results[0].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        dr11Dest->results[0].result         = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->results[0].testId         = 1;
        dr11Dest->results[1].entity         = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
        dr11Dest->results[1].result         = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->results[1].testId         = 1;
        dr11Dest->numResults                = 2;
        dr11Dest->numTests                  = 2;
        dest.SetVersion11(dr11Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(dr11Dest->tests[0].name) == "memory");
        REQUIRE(std::string_view(dr11Dest->tests[1].name) == "eud");
        REQUIRE(dr11Dest->numTests == 2);
        REQUIRE(dr11Dest->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(dr11Dest->tests[1].numResults == 2);
        REQUIRE(dr11Dest->tests[1].resultIndices[0] == 0);
        REQUIRE(dr11Dest->tests[1].resultIndices[1] == 1);
        REQUIRE(dr11Dest->numResults == 2);
        REQUIRE(dr11Dest->results[0].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(dr11Dest->results[0].entity.entityId == 0);
        REQUIRE(dr11Dest->results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(dr11Dest->results[0].testId == 1);
        REQUIRE(dr11Dest->results[1].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(dr11Dest->results[1].entity.entityId == 1);
        REQUIRE(dr11Dest->results[1].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(dr11Dest->results[1].testId == 1);
    }

    SECTION("Version 11: Merge aux")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Src = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Src.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Src->tests[0].name, "eud");
        dr11Src->tests[0].result = DCGM_DIAG_RESULT_PASS;
        SafeCopyTo(dr11Src->tests[0].auxData.data, "Hello Capoo!");
        dr11Src->tests[0].auxData.version = dcgmDiagTestAuxData_version1;
        dr11Src->numTests                 = 1;
        dr11Src->numErrors                = 0;
        dr11Src->numCategories            = 1;
        SafeCopyTo(dr11Src->categories[0], "hardware");
        src.SetVersion11(dr11Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11Dest = std::make_unique<dcgmDiagResponse_v11>();
        memset(dr11Dest.get(), 0, sizeof(dcgmDiagResponse_v11));
        SafeCopyTo(dr11Dest->tests[0].name, "memory");
        SafeCopyTo(dr11Dest->tests[1].name, "eud");
        dr11Dest->tests[1].result          = DCGM_DIAG_RESULT_FAIL;
        dr11Dest->tests[1].numErrors       = 1;
        dr11Dest->tests[1].errorIndices[0] = 0;
        dr11Dest->errors[0].code           = DCGM_FR_EUD_NON_ROOT_USER;
        dr11Dest->numErrors                = 1;
        dr11Dest->numTests                 = 2;
        dest.SetVersion11(dr11Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(dr11Dest->numTests == 2);
        REQUIRE(std::string_view(dr11Dest->tests[0].name) == "memory");
        REQUIRE(std::string_view(dr11Dest->tests[1].name) == "eud");
        REQUIRE(dr11Dest->tests[1].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(std::string_view(dr11Dest->tests[1].auxData.data) == "Hello Capoo!");
        REQUIRE(dr11Dest->tests[1].auxData.version == dcgmDiagTestAuxData_version1);
        REQUIRE(dr11Dest->tests[1].numErrors == 0);
    }
}


TEST_CASE("DcgmDiagResponseWrapper::MergeEudResponse::v10")
{
    SECTION("Version 10: Merge aux")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v10> dr10Src = std::make_unique<dcgmDiagResponse_v10>();
        memset(dr10Src.get(), 0, sizeof(dcgmDiagResponse_v10));
        SafeCopyTo(dr10Src->auxDataPerTest[DCGM_EUD_TEST_INDEX].data, "Hello Capoo!");
        dr10Src->gpuCount                                               = 1;
        dr10Src->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status = DCGM_DIAG_RESULT_PASS;
        dr10Src->auxDataPerTest[DCGM_EUD_TEST_INDEX].version            = dcgmDiagTestAuxData_version1;
        src.SetVersion10(dr10Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v10> dr10Dest = std::make_unique<dcgmDiagResponse_v10>();
        memset(dr10Dest.get(), 0, sizeof(dcgmDiagResponse_v10));
        dr10Dest->gpuCount                                                      = 1;
        dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status        = DCGM_DIAG_RESULT_FAIL;
        dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].error[0].code = DCGM_FR_EUD_NON_ROOT_USER;
        dest.SetVersion10(dr10Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(dr10Dest->auxDataPerTest[DCGM_EUD_TEST_INDEX].data) == "Hello Capoo!");
        REQUIRE(dr10Dest->auxDataPerTest[DCGM_EUD_TEST_INDEX].version == dcgmDiagTestAuxData_version1);
    }

    SECTION("Version 10: Skip merge")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v10> dr10Src = std::make_unique<dcgmDiagResponse_v10>();
        memset(dr10Src.get(), 0, sizeof(dcgmDiagResponse_v10));
        SafeCopyTo(dr10Src->auxDataPerTest[DCGM_EUD_TEST_INDEX].data, "Hello Capoo!");
        dr10Src->gpuCount                                               = 1;
        dr10Src->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status = DCGM_DIAG_RESULT_PASS;
        dr10Src->auxDataPerTest[DCGM_EUD_TEST_INDEX].version            = dcgmDiagTestAuxData_version1;
        src.SetVersion10(dr10Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v10> dr10Dest = std::make_unique<dcgmDiagResponse_v10>();
        memset(dr10Dest.get(), 0, sizeof(dcgmDiagResponse_v10));
        dr10Dest->gpuCount                                               = 1;
        dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status = DCGM_DIAG_RESULT_PASS;
        dest.SetVersion10(dr10Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(dr10Dest->auxDataPerTest[DCGM_EUD_TEST_INDEX].data).empty());
        REQUIRE(dr10Dest->auxDataPerTest[DCGM_EUD_TEST_INDEX].version == 0);
    }

    SECTION("Version 10: Merge - Fail with NON_ROOT_USER case")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v10> dr10Src = std::make_unique<dcgmDiagResponse_v10>();
        memset(dr10Src.get(), 0, sizeof(dcgmDiagResponse_v10));
        SafeCopyTo(dr10Src->auxDataPerTest[DCGM_EUD_TEST_INDEX].data, "Hello Capoo!");
        dr10Src->gpuCount                                                      = 1;
        dr10Src->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status        = DCGM_DIAG_RESULT_FAIL;
        dr10Src->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].error[0].code = 0xc8763;
        src.SetVersion10(dr10Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v10> dr10Dest = std::make_unique<dcgmDiagResponse_v10>();
        memset(dr10Dest.get(), 0, sizeof(dcgmDiagResponse_v10));
        dr10Dest->gpuCount                                                      = 1;
        dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status        = DCGM_DIAG_RESULT_FAIL;
        dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].error[0].code = DCGM_FR_EUD_NON_ROOT_USER;
        dest.SetVersion10(dr10Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].error[0].code == 0xc8763);
    }

    SECTION("Version 10: Merge - NOT_RUN case")
    {
        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v10> dr10Src = std::make_unique<dcgmDiagResponse_v10>();
        memset(dr10Src.get(), 0, sizeof(dcgmDiagResponse_v10));
        SafeCopyTo(dr10Src->auxDataPerTest[DCGM_EUD_TEST_INDEX].data, "Hello Capoo!");
        dr10Src->gpuCount                                                      = 1;
        dr10Src->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status        = DCGM_DIAG_RESULT_FAIL;
        dr10Src->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].error[0].code = 0xc8763;
        src.SetVersion10(dr10Src.get());

        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v10> dr10Dest = std::make_unique<dcgmDiagResponse_v10>();
        memset(dr10Dest.get(), 0, sizeof(dcgmDiagResponse_v10));
        dr10Dest->gpuCount                                                      = 1;
        dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status        = DCGM_DIAG_RESULT_NOT_RUN;
        dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].error[0].code = DCGM_FR_OK;
        dest.SetVersion10(dr10Dest.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_OK);
        REQUIRE(dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].status == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(dr10Dest->perGpuResponses[0].results[DCGM_EUD_TEST_INDEX].error[0].code == 0xc8763);
    }
}

TEMPLATE_TEST_CASE("DcgmDiagResponseWrapper::RecordSystemError",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11,
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType = TestType;

    DYNAMIC_SECTION("Testing version " << DcgmNs::ResponseVersionTrait<ResponseType>::version)
    {
        DcgmDiagResponseWrapper wrapper;

        auto response = MakeUniqueZero<ResponseType>();
        wrapper.SetVersion(response.get());

        static const std::string horrible("You've Moash'ed things horribly");

        wrapper.RecordSystemError(horrible);

        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v12>
                      || std::is_same_v<ResponseType, dcgmDiagResponse_v11>)
        {
            REQUIRE(wrapper.GetSystemErr() == horrible);
        }
        else
        {
            REQUIRE(response->systemError.msg == horrible);
        }
    }
}

TEMPLATE_TEST_CASE("DcgmDiagResponseWrapper::AdoptEudResponse",
                   "",
                   dcgmDiagResponse_v12,
                   dcgmDiagResponse_v11,
                   dcgmDiagResponse_v10,
                   dcgmDiagResponse_v9,
                   dcgmDiagResponse_v8,
                   dcgmDiagResponse_v7)
{
    using ResponseType = TestType;

    DYNAMIC_SECTION("Testing version " << DcgmNs::ResponseVersionTrait<ResponseType>::version)
    {
        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<ResponseType>();

        // Set up destination response
        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v7>)
        {
            destResponse->levelOneTestCount = 123;
        }
        else
        {
            SafeCopyTo(destResponse->dcgmVersion, "4.2.0");
        }
        dest.SetVersion(destResponse.get());

        DcgmDiagResponseWrapper src;
        auto srcResponse = MakeUniqueZero<ResponseType>();

        // Set up source response
        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v7>)
        {
            srcResponse->levelOneTestCount = 456;
        }
        else
        {
            SafeCopyTo(srcResponse->dcgmVersion, "4.2.0");
        }

        src.SetVersion(srcResponse.get());

        REQUIRE(dest.AdoptEudResponse(src) == DCGM_ST_OK);

        // Verify results
        if constexpr (std::is_same_v<ResponseType, dcgmDiagResponse_v7>)
        {
            REQUIRE(destResponse->levelOneTestCount == 123);
        }
        else
        {
            REQUIRE(std::string_view(destResponse->dcgmVersion) == "4.2.0");
        }
    }
}

TEST_CASE("DcgmDiagResponseWrapper::AdoptEudResponse::Version mismatch")
{
    DcgmDiagResponseWrapper dest;
    auto destRv7 = MakeUniqueZero<dcgmDiagResponse_v7>();
    dest.SetVersion(destRv7.get());

    DcgmDiagResponseWrapper src;
    auto srcRv8 = MakeUniqueZero<dcgmDiagResponse_v8>();
    src.SetVersion(srcRv8.get());

    REQUIRE(dest.AdoptEudResponse(src) == DCGM_ST_VER_MISMATCH);
}
