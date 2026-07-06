/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmHostEngineHandler.h>
#include <UnitTestHelpers.h>

namespace
{
uint32_t GetNvmlFuncCallCount(std::string_view funcName)
{
    injectNvmlFuncCallCounts_t funcCallCounts = {};
    REQUIRE(nvmlGetFuncCallCount(&funcCallCounts) == NVML_SUCCESS);
    for (unsigned int i = 0; i < funcCallCounts.numFuncs; ++i)
    {
        if (funcName == funcCallCounts.funcCallInfo[i].funcName)
        {
            return funcCallCounts.funcCallInfo[i].funcCallCount;
        }
    }
    return 0;
}
} // namespace

class TestDcgmHostEngineHandler : public DcgmHostEngineHandler
{
public:
    TestDcgmHostEngineHandler()
        : DcgmHostEngineHandler()
    {} // minimal DcgmHostEngineHandler to pass to DcgmNvmlSystemEventThread constructor
};

TEST_CASE("DcgmNvmlSystemEventThread: nvmlInitWithFlags logic")
{
    auto [yamlFile, setEnvVar, expectedCallCount] = GENERATE(table<std::string, bool, uint32_t>({
        // driver <=610 without env var falls back to NVML_INIT_FLAG_NO_GPUS only (2 init calls)
        { "H200.yaml", false, 2 },
        // driver <=610 with env var keeps NVML_INIT_FLAG_NO_ATTACH (1 init call)
        { "H200.yaml", true, 1 },
        // driver >610 keeps NVML_INIT_FLAG_NO_ATTACH by default (1 init call)
        { "Driver615.yaml", false, 1 },
        // driver >610 with env var keeps NVML_INIT_FLAG_NO_ATTACH (1 init call)
        { "Driver615.yaml", true, 1 },
    }));

    auto guard = WithNvmlInjectionSkuFile(yamlFile);
    if (!guard)
    {
        SKIP("YAML file " << yamlFile << " not found");
    }

    unsetenv("DCGM_NVML_INIT_FLAG_NO_ATTACH");
    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer nvmlDefer([] { nvmlShutdown(); });

    if (setEnvVar)
    {
        setenv("DCGM_NVML_INIT_FLAG_NO_ATTACH", "1", 1);
    }
    DcgmNs::Defer envCleanup([] { unsetenv("DCGM_NVML_INIT_FLAG_NO_ATTACH"); });

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    TestDcgmHostEngineHandler handler;
    DcgmNvmlSystemEventThread sysevThread(handler); // Constructor calls Init() which calls nvmlInitWithFlags
    REQUIRE(GetNvmlFuncCallCount("nvmlInitWithFlags") == expectedCallCount);
}
