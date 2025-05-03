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

#include <PluginInterface.h>
#include <PluginStrings.h>
#define TEST_SOFTWARE_PLUGIN
#include <Software.h>
#include <dcgm_structs.h>

TEST_CASE("Software: CountDevEntry")
{
    dcgmHandle_t handle = (dcgmHandle_t)0;
    Software s(handle);

    bool valid = s.CountDevEntry("nvidia0");
    CHECK(valid == true);
    valid = s.CountDevEntry("nvidia16");
    CHECK(valid == true);
    valid = s.CountDevEntry("nvidia6");
    CHECK(valid == true);

    valid = s.CountDevEntry("nvidiatom");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidiactl");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidia-uvm");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidia-modeset");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidia-nvswitch");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidia-caps");
    CHECK(valid == false);
}

TEST_CASE("Software Subtest Context")
{
    dcgmHandle_t handle = (dcgmHandle_t)0;
    Software s(handle);

    std::string const subtestName { "Subtest" };

    std::unique_ptr<dcgmDiagPluginEntityList_v1> pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    std::unique_ptr<dcgmDiagEntityResults_v2> pEntityResults = std::make_unique<dcgmDiagEntityResults_v2>();

    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    SECTION("Software Subtest Errors")
    {
        memset(pEntityResults.get(), 0, sizeof(*pEntityResults));

        s.setSubtestName("");
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        d.SetMessage("Error");
        s.addError(d);

        s.setSubtestName(subtestName);
        d.SetMessage("Error with subtext context");
        s.addError(d);

        d.SetGpuId(42);
        d.SetMessage("Error with subtext and gpu context");
        s.addError(d);

        dcgmReturn_t ret = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
        CHECK(ret == DCGM_ST_OK);
        dcgmDiagEntityResults_v2 const &entityResults = *pEntityResults;

        REQUIRE(entityResults.numErrors == 3);
        CHECK(entityResults.numInfo == 0);

        CHECK(entityResults.errors[0].entity == dcgmGroupEntityPair_t({ DCGM_FE_NONE, 0 }));
        CHECK(std::string_view(entityResults.errors[1].msg).starts_with(subtestName));
        CHECK(std::string_view(entityResults.errors[2].msg).starts_with(subtestName));
        CHECK(entityResults.errors[2].entity == dcgmGroupEntityPair_t({ DCGM_FE_GPU, 42 }));
    }

    // DCGM-4346: Software: Fix repeating text in error messages
    SECTION("addError(): DcgmDiagError from DcgmError")
    {
        memset(pEntityResults.get(), 0, sizeof(*pEntityResults));

        DcgmError d { DcgmError::GpuIdTag::Unknown };
        d.AddDetail("Detail.");
        d.SetNextSteps("Next Steps.");
        d.SetMessage("Message.");

        s.setSubtestName("Subtest");
        s.addError(d);

        dcgmReturn_t ret = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
        CHECK(ret == DCGM_ST_OK);
        dcgmDiagEntityResults_v2 const &entityResults = *pEntityResults;

        REQUIRE(entityResults.numErrors == 1);
        CHECK(entityResults.numInfo == 0);

        std::string_view expected = "Subtest: Message. Next Steps. Detail.";
        CHECK(entityResults.errors[0].msg == expected);
    }
}