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
#include "PluginInterface.h"
#include <catch2/catch_all.hpp>

#include <Gpu.h>
#include <NvidiaValidationSuite.h>
#include <NvvsCommon.h>

#define ARG_LENGTH 50

#define SETUP()   \
    int argc = 0; \
    std::vector<char *> argv;

#define CLEANUP()                      \
    do                                 \
    {                                  \
        nvvsCommon = NvvsCommon();     \
        for (int i = 0; i < argc; i++) \
        {                              \
            delete[] argv[i];          \
        }                              \
        argc = 0;                      \
    } while (0)

#define add_argument(arg)                   \
    do                                      \
    {                                       \
        char *c = new char[ARG_LENGTH];     \
        snprintf(c, ARG_LENGTH, "%s", arg); \
        argv.push_back(c);                  \
        argc++;                             \
    } while (0)

bool incompatibleGpus       = false;
bool migEnabled             = false;
unsigned int hierarchyCount = 4;

using namespace DcgmNs::Nvvs;

class WrapperNvidiaValidationSuite : protected NvidiaValidationSuite
{
public:
    void WrapperProcessCommandLine(int argc, char *argv[]);
    bool WrapperGetListGpus(void);
    bool WrapperGetListTests(void);
    void WrapperEnumerateAllVisibleGpus();
    void WrapperDecipherProperties(GpuSet *set);
    void WrapperEnumerateAllVisibleTests();
    std::vector<Test *>::iterator WrapperFindTestName(std::string test);
    void WrapperOverrideParameters(TestParameters *tp, const std::string &lowerCaseTestName);
    void WrapperInitializeParameters(const std::string &params, const ParameterValidator &pv);
};

/* define this here so we can control when it fails */
dcgmReturn_t dcgmGetGpuInstanceHierarchy(dcgmHandle_t /* handle */, dcgmMigHierarchy_v2 *hierarchy)
{
    if (incompatibleGpus && hierarchy != nullptr)
    {
        hierarchy->count                                = 2;
        hierarchy->entityList[0].parent.entityId        = 0;
        hierarchy->entityList[0].parent.entityGroupId   = DCGM_FE_GPU;
        hierarchy->entityList[0].entity.entityId        = 0;
        hierarchy->entityList[0].entity.entityGroupId   = DCGM_FE_GPU_I;
        hierarchy->entityList[0].info.nvmlProfileSlices = DCGM_MAX_INSTANCES_PER_GPU / 2;
        hierarchy->entityList[1].parent.entityId        = 0;
        hierarchy->entityList[1].parent.entityGroupId   = DCGM_FE_GPU;
        hierarchy->entityList[1].entity.entityId        = 0;
        hierarchy->entityList[1].entity.entityGroupId   = DCGM_FE_GPU_I;
        hierarchy->entityList[1].info.nvmlProfileSlices = DCGM_MAX_INSTANCES_PER_GPU / 2;
    }
    else if (migEnabled && hierarchy != nullptr)
    {
        hierarchy->count = hierarchyCount;
        for (unsigned int i = 0; i < hierarchy->count; i++)
        {
            hierarchy->entityList[i].parent.entityId        = i;
            hierarchy->entityList[i].parent.entityGroupId   = DCGM_FE_GPU;
            hierarchy->entityList[i].entity.entityId        = i * DCGM_MAX_INSTANCES_PER_GPU;
            hierarchy->entityList[i].entity.entityGroupId   = DCGM_FE_GPU_I;
            hierarchy->entityList[i].info.nvmlProfileSlices = DCGM_MAX_INSTANCES_PER_GPU;
        }
    }

    return DCGM_ST_OK;
}

unsigned int fieldIntVal = 0;
dcgmReturn_t dcgmEntitiesGetLatestValues(dcgmHandle_t /* handle */,
                                         dcgmGroupEntityPair_t & /* entity */,
                                         unsigned int /* entityCount */,
                                         unsigned short fieldIds[],
                                         unsigned int /* fieldCount */,
                                         unsigned int /* flags */,
                                         dcgmFieldValue_v2 values[])
{
    switch (fieldIds[0])
    {
        case DCGM_FI_DEV_MIG_MODE:
            if (migEnabled)
            {
                values[0].value.i64 = 1;
            }
            else
            {
                values[0].value.i64 = 0;
            }
            break;
        case DCGM_FI_DEV_MIG_MAX_SLICES:
            values[0].value.i64 = fieldIntVal;
            break;
    }

    return DCGM_ST_OK;
}

TEST_CASE("NvidiaValidationSuite: build common gpu list", "[.]")
{
    NvidiaValidationSuite nvs;
    std::vector<unsigned int> gpuIndices;
    std::vector<Gpu *> visibleGpus;
    Gpu gpu0(0);
    Gpu gpu1(1);
    Gpu gpu2(2);
    Gpu gpu3(3);
    visibleGpus.push_back(&gpu0);
    visibleGpus.push_back(&gpu1);
    visibleGpus.push_back(&gpu2);
    visibleGpus.push_back(&gpu3);

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
    REQUIRE(errorString.find("Cannot run diagnostic: CUDA does not support enumerating GPUs") == 0);

    // Make sure we fail if we have a GPU in MIG mode are we are running on more than 1 GPU even if it isn't in the list
    migEnabled       = true;
    incompatibleGpus = false;
    gpuIndices.clear();
    gpuIndices.push_back(1);
    gpuIndices.push_back(2);
    errorString = nvs.BuildCommonGpusList(gpuIndices, visibleGpus);
    REQUIRE(!errorString.empty());
    REQUIRE(errorString.find("Cannot run diagnostic: CUDA does not support enumerating GPUs") == 0);

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

SCENARIO("NVVS correctly processes command line arguments")
{
    GIVEN("default values")
    {
        SETUP();
        WrapperNvidiaValidationSuite nvvs;
        add_argument("nvvs");
        nvvs.WrapperProcessCommandLine(argc, &argv[0]);

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
        CLEANUP();
    }

    GIVEN("Non-default paramters")
    {
        SETUP();
        char statsDir[] = "/tmp/dcgm-diag-test-stats-dir-XXXXXX";
        mkdtemp(statsDir);

        WrapperNvidiaValidationSuite nvvs;

        add_argument("nvvs");
        add_argument("-g");
        add_argument("-v");
        add_argument("-t");
        add_argument("--statsonfail");
        add_argument("--hwdiaglogfile");
        add_argument("hwlogfile");
        add_argument("--specifiedtest");
        add_argument("-s");
        add_argument("--quiet");
        add_argument("--pluginpath");
        add_argument("plugins");
        add_argument("-i");
        add_argument("1");
        add_argument("-d");
        add_argument("2");
        add_argument("--configless");
        add_argument("-c");
        add_argument("-a");
        add_argument("--statspath");
        add_argument(statsDir);
        add_argument("--parameters");
        add_argument("sm stress.test_duration=5");
        add_argument("-w");
        add_argument("3");
        add_argument("--dcgmHostname");
        add_argument("host");
        add_argument("--channel-fd");
        add_argument("3");
        add_argument("--clocksevent-mask");
        add_argument("--throttle-mask");
        add_argument("--fail-early");
        add_argument("--check-interval");
        add_argument("3");
        add_argument("-l");
        add_argument("logfile");
        add_argument("--watch-frequency");
        add_argument("1000000");

        nvvs.WrapperProcessCommandLine(argc, &argv[0]);

        CHECK(nvvs.WrapperGetListGpus() == true);
        CHECK(nvvs.WrapperGetListTests() == true);
        CHECK(nvvsCommon.verbose == true);
        CHECK(nvvsCommon.pluginPath == "plugins");
        CHECK(nvvsCommon.quietMode == true);
        CHECK(nvvsCommon.configless == true);
        CHECK(nvvsCommon.statsOnlyOnFail == true);
        CHECK(nvvsCommon.indexString == "1");
        CHECK(nvvsCommon.dcgmHostname == "host");
        CHECK(nvvsCommon.m_statsPath == statsDir);
        CHECK(nvvsCommon.watchFrequency == 1000000);
        CHECK(nvvsCommon.channelFd == 3);

        rmdir(statsDir);
        CLEANUP();
    }

    GIVEN("Conflicting throttle and clocks events")
    {
        SETUP();
        char statsDir[] = "/tmp/dcgm-diag-test-stats-dir-XXXXXX";
        mkdtemp(statsDir);

        WrapperNvidiaValidationSuite nvvs;

        add_argument("nvvs");
        add_argument("--throttle-mask hw_power_brake");
        add_argument("--clocksevent-mask sw_thermal");

        CHECK_THROWS(nvvs.WrapperProcessCommandLine(argc, &argv[0]));

        rmdir(statsDir);
        CLEANUP();
    }

    GIVEN("Invalid throttle and clock events")
    {
        {
            SETUP();
            char statsDir[] = "/tmp/dcgm-diag-test-stats-dir-XXXXXX";
            mkdtemp(statsDir);

            WrapperNvidiaValidationSuite nvvs;

            add_argument("nvvs");
            add_argument("--throttle-mask invalid");

            nvvs.WrapperProcessCommandLine(argc, &argv[0]);
            CHECK(nvvsCommon.clocksEventIgnoreMask == DCGM_INT64_BLANK);

            rmdir(statsDir);
            CLEANUP();
        }

        {
            SETUP();
            char statsDir[] = "/tmp/dcgm-diag-test-stats-dir-XXXXXX";
            mkdtemp(statsDir);

            WrapperNvidiaValidationSuite nvvs;

            add_argument("nvvs");
            add_argument("--clocksevent-mask sonotvalid");

            nvvs.WrapperProcessCommandLine(argc, &argv[0]);
            CHECK(nvvsCommon.clocksEventIgnoreMask == DCGM_INT64_BLANK);

            rmdir(statsDir);
            CLEANUP();
        }
    }

    GIVEN("Revert to default value for invalid user-set watch frequencies")
    {
        {
            SETUP();
            char statsDir[] = "/tmp/dcgm-diag-test-stats-dir-XXXXXX";
            mkdtemp(statsDir);

            WrapperNvidiaValidationSuite nvvs;

            add_argument("nvvs");
            add_argument("--watch-frequency");
            add_argument("99999");

            nvvs.WrapperProcessCommandLine(argc, &argv[0]);
            CHECK(nvvsCommon.watchFrequency == 5000000);

            rmdir(statsDir);
            CLEANUP();
        }
        {
            SETUP();
            char statsDir[] = "/tmp/dcgm-diag-test-stats-dir-XXXXXX";
            mkdtemp(statsDir);

            WrapperNvidiaValidationSuite nvvs;

            add_argument("nvvs");
            add_argument("--watch-frequency");
            add_argument("60000001");

            nvvs.WrapperProcessCommandLine(argc, &argv[0]);
            CHECK(nvvsCommon.watchFrequency == 5000000);

            rmdir(statsDir);
            CLEANUP();
        }
    }
}

SCENARIO("findTestName finds the test if it exists", "[.]")
{
    WrapperNvidiaValidationSuite nvvs;
    nvvs.WrapperEnumerateAllVisibleTests();
    // We need to load the plugins for this to work. So we will only test the negative cases
    // std::vector<Test *>::iterator sm_stress = nvvs.WrapperFindTestName("SM Stress");
    std::vector<Test *>::iterator skip        = nvvs.WrapperFindTestName("Skip");
    std::vector<Test *>::iterator nonexistent = nvvs.WrapperFindTestName("Nonexistent");
    CHECK(skip == nonexistent);
}

SCENARIO("overrideparameters overrides the correct parameters", "[.]")
{
    SETUP();
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

    WrapperNvidiaValidationSuite nvvs;
    nvvs.WrapperInitializeParameters(paramString, pv);

    nvvs.WrapperOverrideParameters(&tp, "sm_stress");
    CHECK(tp.GetString("test_duration") == "5");
    CLEANUP();
}

void WrapperNvidiaValidationSuite::WrapperProcessCommandLine(int argc, char *argv[])
{
    processCommandLine(argc, argv);
}

bool WrapperNvidiaValidationSuite::WrapperGetListGpus()
{
    return listGpus;
}

bool WrapperNvidiaValidationSuite::WrapperGetListTests()
{
    return listTests;
}

void WrapperNvidiaValidationSuite::WrapperInitializeParameters(const std::string &parms, const ParameterValidator &pv)
{
    InitializeParameters(parms, pv);
}

void WrapperNvidiaValidationSuite::WrapperEnumerateAllVisibleTests()
{
    std::vector<std::unique_ptr<EntitySet>> entitySet;
    m_tf = new TestFramework(entitySet);
    enumerateAllVisibleTests();
}

std::vector<Test *>::iterator WrapperNvidiaValidationSuite::WrapperFindTestName(std::string test)
{
    return FindTestName(std::move(test));
}

void WrapperNvidiaValidationSuite::WrapperOverrideParameters(TestParameters *tp, const std::string &lowerCaseTestName)
{
    overrideParameters(tp, lowerCaseTestName);
}
