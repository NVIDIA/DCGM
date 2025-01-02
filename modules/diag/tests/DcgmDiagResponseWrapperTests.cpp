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

#include <catch2/catch_all.hpp>

#include <DcgmStringHelpers.h>
#include <UniquePtrUtil.h>
#include <dcgm_errors.h>

#define __DIAG_UNIT_TESTING__
#include <DcgmDiagResponseWrapper.h>

TEST_CASE("DcgmDiagResponseWrapper: HasTest")
{
    SECTION("Unsupported version")
    {
        DcgmDiagResponseWrapper ddr;
        dcgmDiagResponse_v11 dr11 {};

        dr11.numTests           = 1;
        std::string const capoo = "capoo";
        SafeCopyTo(dr11.tests[0].name, capoo.c_str());

        REQUIRE(ddr.SetVersion11(&dr11) == DCGM_ST_OK);

        ddr.m_version = dcgmDiagResponse_version9;
        CHECK(ddr.HasTest(capoo) == false);
        CHECK(ddr.HasTest("dogdog") == false);
    }

    SECTION("Basic case")
    {
        DcgmDiagResponseWrapper ddr;
        dcgmDiagResponse_v11 dr11 {};

        dr11.numTests            = 2;
        std::string const capoo  = "capoo";
        std::string const dogdog = "dogdog";
        SafeCopyTo(dr11.tests[0].name, capoo.c_str());
        SafeCopyTo(dr11.tests[1].name, dogdog.c_str());

        REQUIRE(ddr.SetVersion11(&dr11) == DCGM_ST_OK);

        CHECK(ddr.HasTest(capoo) == true);
        CHECK(ddr.HasTest("dogdog") == true);
        CHECK(ddr.HasTest("rabbit") == false);
    }
}

TEST_CASE("DcgmDiagResponseWrapper::MergeEudResponse")
{
    SECTION("Version Mismatch")
    {
        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> dr11 = std::make_unique<dcgmDiagResponse_v11>();
        dest.SetVersion11(dr11.get());

        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v10> dr10 = std::make_unique<dcgmDiagResponse_v10>();
        src.SetVersion10(dr10.get());

        REQUIRE(dest.MergeEudResponse(src) == DCGM_ST_VER_MISMATCH);
    }

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

TEST_CASE("DcgmDiagResponseWrapper: RecordSystemError")
{
    SECTION("Version 11")
    {
        DcgmDiagResponseWrapper wrapper;

        std::unique_ptr<dcgmDiagResponse_v11> rv11 = std::make_unique<dcgmDiagResponse_v11>();
        memset(rv11.get(), 0, sizeof(*rv11));

        wrapper.SetVersion11(rv11.get());

        static const std::string horrible("You've Moash'ed things horribly");

        wrapper.RecordSystemError(horrible);
        REQUIRE(wrapper.GetSystemErr() == horrible);
    }

    SECTION("Version 10")
    {
        DcgmDiagResponseWrapper wrapper;

        std::unique_ptr<dcgmDiagResponse_v10> rv10 = std::make_unique<dcgmDiagResponse_v10>();
        memset(rv10.get(), 0, sizeof(*rv10));

        wrapper.SetVersion10(rv10.get());

        static const std::string horrible("You've Moash'ed things horribly");

        wrapper.RecordSystemError(horrible);
        REQUIRE(rv10->systemError.msg == horrible);
    }

    SECTION("Version 9")
    {
        DcgmDiagResponseWrapper wrapper;

        std::unique_ptr<dcgmDiagResponse_v9> rv9 = std::make_unique<dcgmDiagResponse_v9>();
        memset(rv9.get(), 0, sizeof(*rv9));

        wrapper.SetVersion9(rv9.get());

        static const std::string horrible("You've Moash'ed things horribly");

        wrapper.RecordSystemError(horrible);
        REQUIRE(rv9->systemError.msg == horrible);
    }

    SECTION("Version 8")
    {
        DcgmDiagResponseWrapper wrapper;

        std::unique_ptr<dcgmDiagResponse_v8> rv8 = std::make_unique<dcgmDiagResponse_v8>();
        memset(rv8.get(), 0, sizeof(*rv8));

        wrapper.SetVersion8(rv8.get());

        static const std::string horrible("You've Moash'ed things horribly");

        wrapper.RecordSystemError(horrible);
        REQUIRE(rv8->systemError.msg == horrible);
    }

    SECTION("Version 7")
    {
        DcgmDiagResponseWrapper wrapper;

        std::unique_ptr<dcgmDiagResponse_v7> rv7 = std::make_unique<dcgmDiagResponse_v7>();
        memset(rv7.get(), 0, sizeof(*rv7));

        wrapper.SetVersion7(rv7.get());

        static const std::string horrible("You've Moash'ed things horribly");

        wrapper.RecordSystemError(horrible);
        REQUIRE(rv7->systemError.msg == horrible);
    }
}

TEST_CASE("DcgmDiagResponseWrapper: AdoptEudResponse")
{
    SECTION("Version 11")
    {
        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v11> destRv11 = MakeUniqueZero<dcgmDiagResponse_v11>();
        SafeCopyTo(destRv11->dcgmVersion, "6.6.6");
        dest.SetVersion11(destRv11.get());

        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v11> srcRv11 = MakeUniqueZero<dcgmDiagResponse_v11>();
        SafeCopyTo(srcRv11->dcgmVersion, "4.0.0");
        src.SetVersion11(srcRv11.get());

        REQUIRE(dest.AdoptEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destRv11->dcgmVersion) == "4.0.0");
    }

    SECTION("Version 10")
    {
        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v10> destRv10 = MakeUniqueZero<dcgmDiagResponse_v10>();
        SafeCopyTo(destRv10->dcgmVersion, "6.6.6");
        dest.SetVersion10(destRv10.get());

        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v10> srcRv10 = MakeUniqueZero<dcgmDiagResponse_v10>();
        SafeCopyTo(srcRv10->dcgmVersion, "4.0.0");
        src.SetVersion10(srcRv10.get());

        REQUIRE(dest.AdoptEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destRv10->dcgmVersion) == "4.0.0");
    }

    SECTION("Version 9")
    {
        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v9> destRv9 = MakeUniqueZero<dcgmDiagResponse_v9>();
        SafeCopyTo(destRv9->dcgmVersion, "6.6.6");
        dest.SetVersion9(destRv9.get());

        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v9> srcRv9 = MakeUniqueZero<dcgmDiagResponse_v9>();
        SafeCopyTo(srcRv9->dcgmVersion, "4.0.0");
        src.SetVersion9(srcRv9.get());

        REQUIRE(dest.AdoptEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destRv9->dcgmVersion) == "4.0.0");
    }

    SECTION("Version 8")
    {
        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v8> destRv8 = MakeUniqueZero<dcgmDiagResponse_v8>();
        SafeCopyTo(destRv8->dcgmVersion, "6.6.6");
        dest.SetVersion8(destRv8.get());

        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v8> srcRv8 = MakeUniqueZero<dcgmDiagResponse_v8>();
        SafeCopyTo(srcRv8->dcgmVersion, "4.0.0");
        src.SetVersion8(srcRv8.get());

        REQUIRE(dest.AdoptEudResponse(src) == DCGM_ST_OK);
        REQUIRE(std::string_view(destRv8->dcgmVersion) == "4.0.0");
    }

    SECTION("Version 7")
    {
        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v7> destRv7 = MakeUniqueZero<dcgmDiagResponse_v7>();
        dest.SetVersion7(destRv7.get());
        destRv7->levelOneTestCount = 123;

        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v7> srcRv7 = MakeUniqueZero<dcgmDiagResponse_v7>();
        src.SetVersion7(srcRv7.get());
        srcRv7->levelOneTestCount = 456;

        REQUIRE(dest.AdoptEudResponse(src) == DCGM_ST_OK);
        REQUIRE(destRv7->levelOneTestCount == 123);
    }

    SECTION("Version mismatch")
    {
        DcgmDiagResponseWrapper dest;
        std::unique_ptr<dcgmDiagResponse_v7> destRv7 = MakeUniqueZero<dcgmDiagResponse_v7>();
        dest.SetVersion7(destRv7.get());

        DcgmDiagResponseWrapper src;
        std::unique_ptr<dcgmDiagResponse_v8> srcRv8 = MakeUniqueZero<dcgmDiagResponse_v8>();
        src.SetVersion8(srcRv8.get());

        REQUIRE(dest.AdoptEudResponse(src) == DCGM_ST_VER_MISMATCH);
    }
}