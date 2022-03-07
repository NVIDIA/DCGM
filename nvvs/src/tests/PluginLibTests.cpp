/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <stdlib.h>

#include <PluginLib.h>
#include <PluginStrings.h>
#include <dcgm_structs.h>

TEST_CASE("PluginLib: General")
{
    PluginLib pl;
    dcgmReturn_t ret = pl.LoadPlugin("./libtestplugin.so", "software");
    CHECK(ret == DCGM_ST_OK);

    ret = pl.GetPluginInfo();
    CHECK(ret == DCGM_ST_OK);

    std::vector<dcgmDiagPluginGpuInfo_t> gpuInfo;
    dcgmDiagPluginGpuInfo_t gi = {};
    dcgmHandle_t handle        = {};
    gi.gpuId                   = 0;
    gpuInfo.push_back(gi);
    gi.gpuId = 1;
    gpuInfo.push_back(gi);

    ret = pl.InitializePlugin(handle, gpuInfo);
    CHECK(ret == DCGM_ST_OK);

    TestParameters tp;
    tp.AddDouble(PS_LOGFILE_TYPE, 0.0);
    setenv("result", "pass", 1);
    pl.RunTest(10, &tp);

    nvvsPluginResult_t result = pl.GetResult();
    CHECK(result == NVVS_RESULT_PASS);

    setenv("result", "fail", 1);
    pl.RunTest(10, &tp);
    result = pl.GetResult();
    CHECK(result == NVVS_RESULT_FAIL);
}
