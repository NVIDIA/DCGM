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
#include "PluginInterface.h"
#include "dcgm_fields.h"
#include <catch2/catch_all.hpp>

#include <stdlib.h>

#include <PluginLib.h>
#include <PluginStrings.h>
#include <dcgm_structs.h>

TEST_CASE("PluginLib: General")
{
    /* Initialize logging or the plugin will crash when it tries to log to us */
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    PluginLib pl;
    dcgmReturn_t ret = pl.LoadPlugin("./libtestplugin.so.4", "software");
    CHECK(ret == DCGM_ST_OK);

    ret = pl.GetPluginInfo();
    CHECK(ret == DCGM_ST_OK);

    std::vector<dcgmDiagPluginEntityInfo_v1> entityInfo;
    dcgmDiagPluginEntityInfo_v1 ei = {};
    dcgmHandle_t handle            = {};
    ei.entity.entityId             = 0;
    ei.entity.entityGroupId        = DCGM_FE_GPU;
    entityInfo.push_back(ei);
    ei.entity.entityId = 1;
    entityInfo.push_back(ei);
    constexpr int pluginIdx = 0xc8763;

    ret = pl.InitializePlugin(handle, pluginIdx);
    CHECK(ret == DCGM_ST_OK);

    TestParameters tp;
    tp.AddDouble(PS_LOGFILE_TYPE, 0.0);
    setenv("result", "pass", 1);
    pl.RunTest("software", entityInfo, 10, &tp);
    pl.SetTestRunningState("software", TestRuningState::Done);

    nvvsPluginResult_t result = pl.GetResult("software");
    CHECK(result == NVVS_RESULT_PASS);

    setenv("result", "fail", 1);
    pl.RunTest("software", entityInfo, 10, &tp);
    pl.SetTestRunningState("software", TestRuningState::Done);
    result = pl.GetResult("software");
    CHECK(result == NVVS_RESULT_FAIL);

    // Need to wait for DCGM-3857: support RetrieveResults getting dcgmDiagError_v1 () for testing the correctness of
    // plugin index
}
