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

#include <DcgmStringHelpers.h>
#include <Defer.hpp>
#include <MockDynamicLibraryLoader.h>
#include <PluginInterface.h>

#include "MockPluginLib.h"
#include "MockSoftwarePluginFramework.h"

#include <MockFileSystemOperator.h>

#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>

#include <TestFramework.h>
#include <TestFramework.inl>

namespace impl
{

template class TestFramework<MockPluginLib,
                             MockDynamicLibraryLoader,
                             MockSoftwarePluginFramework,
                             MockFileSystemOperator>;

} // namespace impl

#include <catch2/catch_all.hpp>
#include <cstring>
#include <dcgm_errors.h>
#include <string_view>
#include <sys/stat.h>

#include <CpuSet.h>
#include <DcgmLogging.h>
#include <EntitySet.h>
#include <Gpu.h>
#include <GpuSet.h>
#include <NvvsCommon.h>
#include <PluginStrings.h>
#include <Test.h>
#include <TestParameters.h>

#include <memory>

class MockTestFramework
    : public impl::
          TestFramework<MockPluginLib, MockDynamicLibraryLoader, MockSoftwarePluginFramework, MockFileSystemOperator>
{
public:
    using Base = impl::
        TestFramework<MockPluginLib, MockDynamicLibraryLoader, MockSoftwarePluginFramework, MockFileSystemOperator>;

    using Base::Base;
    using Base::m_diagResponse;
    using Base::m_dynamicLibrary;
    using Base::m_fileSystem;
    using Base::m_softwarePluginFrameworks;
    using Base::PushPlugin;

    using Base::GetPluginBaseDir;
    using Base::GetPluginCudalessDir;
};

namespace
{
std::unique_ptr<GpuSet> MakeGpuSetWithFakeGpus(std::vector<unsigned int> const &gpuIds,
                                               std::string const &driverVersion = "545.29.06")
{
    auto gpuSet = std::make_unique<GpuSet>();
    std::vector<Gpu *> gpuObjs;
    gpuObjs.reserve(gpuIds.size());

    for (auto const gpuId : gpuIds)
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
    return gpuSet;
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
        gpuSet->SetGpuObjs({});
    }
}
} // namespace

SCENARIO("GetPluginBaseDir returns plugin directory relative to current process's location")
{
    std::string const savedPluginPath = nvvsCommon.pluginPath;
    DcgmNs::Defer restorePluginPath([&] { nvvsCommon.pluginPath = savedPluginPath; });
    nvvsCommon.pluginPath.clear();

    // Mock filesystem: GetPluginBaseDir uses m_fileSystem.ReadLink/Stat/Access, not the real
    // /proc/<pid>/exe or plugin directories (which are unavailable or racy in some CI setups).
    std::string const fakeBinary     = "/fake/nvvs_exe";
    std::string const pluginDir      = "/fake/plugins";
    std::string const libexecPlugins = std::string("/fake/../libexec/datacenter-gpu-manager-4/plugins");

    INFO("pluginDir: " << pluginDir);

    std::vector<std::unique_ptr<EntitySet>> entitySet;
    MockTestFramework tf(entitySet);

    char procExe[64];
    snprintf(procExe, sizeof(procExe), "/proc/%d/exe", getpid());
    tf.m_fileSystem.MockReadLink(procExe, fakeBinary);
    struct stat binSt = {};
    binSt.st_uid      = 0;
    binSt.st_gid      = 0;
    binSt.st_mode     = S_IFREG | 0755;
    tf.m_fileSystem.MockStat(fakeBinary, binSt);

    tf.m_fileSystem.MockAccess(pluginDir, -1);
    tf.m_fileSystem.MockAccess(libexecPlugins, -1);
    CHECK_THROWS(tf.GetPluginBaseDir());

    struct stat dirSt = {};
    dirSt.st_mode     = S_IFDIR | 0755;
    tf.m_fileSystem.MockAccess(pluginDir, 0);
    tf.m_fileSystem.MockStat(pluginDir, dirSt);

    CHECK(tf.GetPluginBaseDir() == pluginDir);
}

SCENARIO("GetPluginCudalessDir returns cudaless directory in plugin directory")
{
    std::string const savedPluginPath = nvvsCommon.pluginPath;
    DcgmNs::Defer restorePluginPath([&] { nvvsCommon.pluginPath = savedPluginPath; });
    nvvsCommon.pluginPath.clear();

    std::string const fakeBinary = "/fake/nvvs_exe";
    std::string const pluginDir  = "/fake/plugins";
    INFO("pluginDir: " << pluginDir);

    std::vector<std::unique_ptr<EntitySet>> entitySet;
    MockTestFramework tf(entitySet);

    char procExe[64];
    snprintf(procExe, sizeof(procExe), "/proc/%d/exe", getpid());
    tf.m_fileSystem.MockReadLink(procExe, fakeBinary);
    struct stat binSt = {};
    binSt.st_uid      = 0;
    binSt.st_gid      = 0;
    binSt.st_mode     = S_IFREG | 0755;
    tf.m_fileSystem.MockStat(fakeBinary, binSt);
    struct stat dirSt = {};
    dirSt.st_mode     = S_IFDIR | 0755;
    tf.m_fileSystem.MockAccess(pluginDir, 0);
    tf.m_fileSystem.MockStat(pluginDir, dirSt);

    std::string const reference = pluginDir + "/cudaless/";

    CHECK(tf.GetPluginCudalessDir() == reference);
}

TEST_CASE("TestFramework::Go skips RunTest when plugin index is invalid")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0 });

    Test test(0, "plugin_a", "hardware_test", "d", DCGM_FE_GPU, "category");
    TestParameters tp;
    tp.AddString(PS_PLUGIN_NAME, "plugin_a");
    test.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tp);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &test) == 0);

    entitySets.push_back(std::move(gpu));

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    auto const &diag = tf.m_diagResponse.ConstResponse<dcgmDiagResponse_v12>();
    REQUIRE(diag.numErrors == 1);
    CHECK(diag.errors[0].testId == DCGM_DIAG_RESPONSE_SYSTEM_ERROR);
    CHECK(diag.errors[0].code == static_cast<unsigned int>(DCGM_ST_GENERIC_ERROR));
    CHECK(std::string_view(diag.errors[0].msg) == "Invalid index 0 or too many tests");
}

TEST_CASE("TestFramework::Go invokes plugin RunTest when plugin index is valid")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0 });

    Test test(0, "plugin_a", "hardware_test", "d", DCGM_FE_GPU, "category");
    TestParameters tp;
    tp.AddString(PS_PLUGIN_NAME, "plugin_a");
    test.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tp);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &test) == 0);

    entitySets.push_back(std::move(gpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetPassResult("hardware_test");
    MockPluginLib *const pluginLib = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(pluginLib->RunCount("hardware_test") == 1);
}

TEST_CASE("TestFramework::Go omits skipped entities from later RunTest entity lists")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer restoreChannelFd([&] { nvvsCommon.channelFd = savedChannelFd; });
    bool const savedFailEarly = nvvsCommon.failEarly;
    DcgmNs::Defer restoreFailEarly([&] { nvvsCommon.failEarly = savedFailEarly; });
    nvvsCommon.channelFd       = -1;
    nvvsCommon.failEarly       = false;
    bool const savedConfigless = nvvsCommon.configless;
    DcgmNs::Defer restoreConfigless([&] { nvvsCommon.configless = savedConfigless; });
    nvvsCommon.configless = true;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0, 1 });

    Test testFirst(0, "plugin_a", "skip_seed_test", "d", DCGM_FE_GPU, "category");
    TestParameters tpFirst;
    tpFirst.AddString(PS_PLUGIN_NAME, "plugin_a");
    testFirst.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpFirst);

    Test testSecond(0, "plugin_a", "after_skip_test", "d", DCGM_FE_GPU, "category");
    TestParameters tpSecond;
    tpSecond.AddString(PS_PLUGIN_NAME, "plugin_a");
    testSecond.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpSecond);

    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testFirst) == 0);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testSecond) == 0);

    GpuSet *gpuPtr = gpu.get();
    entitySets.push_back(std::move(gpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetFailWithSkipsFromRowRemap("skip_seed_test", { 0 });
    mock->SetPassResult("after_skip_test");
    MockPluginLib *mockPtr = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    REQUIRE(mockPtr->GetRunEntityIdSnapshots("skip_seed_test").size() == 1);
    CHECK(mockPtr->GetRunEntityIdSnapshots("skip_seed_test")[0] == std::vector<dcgm_field_eid_t> { 0, 1 });

    REQUIRE(mockPtr->GetRunEntityIdSnapshots("after_skip_test").size() == 1);
    CHECK(mockPtr->GetRunEntityIdSnapshots("after_skip_test")[0] == std::vector<dcgm_field_eid_t> { 1 });

    REQUIRE(gpuPtr->GetSkippedEntities().contains(0));
    CHECK_FALSE(gpuPtr->GetSkippedEntities().contains(1));
}

TEST_CASE("TestFramework::Go does not call RunTest when all entities were skipped by a prior test")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer restoreChannelFd([&] { nvvsCommon.channelFd = savedChannelFd; });
    bool const savedFailEarly = nvvsCommon.failEarly;
    DcgmNs::Defer restoreFailEarly([&] { nvvsCommon.failEarly = savedFailEarly; });
    nvvsCommon.channelFd       = -1;
    nvvsCommon.failEarly       = false;
    bool const savedConfigless = nvvsCommon.configless;
    DcgmNs::Defer restoreConfigless([&] { nvvsCommon.configless = savedConfigless; });
    nvvsCommon.configless = true;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0, 1 });

    Test testFirst(0, "plugin_a", "skip_all_test", "d", DCGM_FE_GPU, "category");
    TestParameters tpFirst;
    tpFirst.AddString(PS_PLUGIN_NAME, "plugin_a");
    testFirst.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpFirst);

    Test testSecond(0, "plugin_a", "no_entities_left_test", "d", DCGM_FE_GPU, "category");
    TestParameters tpSecond;
    tpSecond.AddString(PS_PLUGIN_NAME, "plugin_a");
    testSecond.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpSecond);

    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testFirst) == 0);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testSecond) == 0);

    GpuSet *gpuPtr = gpu.get();
    entitySets.push_back(std::move(gpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetFailWithSkipsFromRowRemap("skip_all_test", { 0, 1 });
    mock->SetPassResult("no_entities_left_test");
    MockPluginLib *mockPtr = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(mockPtr->RunCount("skip_all_test") == 1);
    CHECK(mockPtr->RunCount("no_entities_left_test") == 0);
    CHECK(mockPtr->GetRunEntityIdSnapshots("no_entities_left_test").empty());
    CHECK(gpuPtr->GetSkippedEntities().size() == 2);
}

TEST_CASE("TestFramework::Go runs eud when row remap failure and EUD-only is true")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer restoreChannelFd([&] { nvvsCommon.channelFd = savedChannelFd; });
    bool const savedFailEarly = nvvsCommon.failEarly;
    DcgmNs::Defer restoreFailEarly([&] { nvvsCommon.failEarly = savedFailEarly; });
    nvvsCommon.channelFd       = -1;
    nvvsCommon.failEarly       = false;
    bool const savedConfigless = nvvsCommon.configless;
    DcgmNs::Defer restoreConfigless([&] { nvvsCommon.configless = savedConfigless; });
    nvvsCommon.configless     = true;
    bool const savedIsEudOnly = nvvsCommon.isEudOnly;
    DcgmNs::Defer restoreIsEudOnly([&] { nvvsCommon.isEudOnly = savedIsEudOnly; });
    nvvsCommon.isEudOnly = true;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0, 1 });

    Test testFirst(0, "plugin_a", "row_remap_failure_seed", "d", DCGM_FE_GPU, "category");
    TestParameters tpFirst;
    tpFirst.AddString(PS_PLUGIN_NAME, "plugin_a");
    testFirst.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpFirst);

    Test testEud(0, EUD_PLUGIN_NAME, EUD_PLUGIN_NAME, "d", DCGM_FE_GPU, "category");
    TestParameters tpEud;
    tpEud.AddString(PS_PLUGIN_NAME, EUD_PLUGIN_NAME);
    testEud.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpEud);

    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testFirst) == 0);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testEud) == 0);

    entitySets.push_back(std::move(gpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetFailWithSkipsFromRowRemap("row_remap_failure_seed", { 0, 1 }, DCGM_FE_GPU, DCGM_FR_ROW_REMAP_FAILURE);
    mock->SetPassResult(EUD_PLUGIN_NAME);
    MockPluginLib *mockPtr = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(mockPtr->RunCount("row_remap_failure_seed") == 1);
    CHECK(mockPtr->RunCount(EUD_PLUGIN_NAME) == 1);
    REQUIRE(mockPtr->GetRunEntityIdSnapshots(EUD_PLUGIN_NAME).size() == 1);
    CHECK(mockPtr->GetRunEntityIdSnapshots(EUD_PLUGIN_NAME)[0] == std::vector<dcgm_field_eid_t> { 0, 1 });
}

TEST_CASE("TestFramework::Go does not run eud when row remap failure and EUD-only is false")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer restoreChannelFd([&] { nvvsCommon.channelFd = savedChannelFd; });
    bool const savedFailEarly = nvvsCommon.failEarly;
    DcgmNs::Defer restoreFailEarly([&] { nvvsCommon.failEarly = savedFailEarly; });
    nvvsCommon.channelFd       = -1;
    nvvsCommon.failEarly       = false;
    bool const savedConfigless = nvvsCommon.configless;
    DcgmNs::Defer restoreConfigless([&] { nvvsCommon.configless = savedConfigless; });
    nvvsCommon.configless     = true;
    bool const savedIsEudOnly = nvvsCommon.isEudOnly;
    DcgmNs::Defer restoreIsEudOnly([&] { nvvsCommon.isEudOnly = savedIsEudOnly; });
    nvvsCommon.isEudOnly = false;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0, 1 });

    Test testFirst(0, "plugin_a", "row_remap_failure_seed", "d", DCGM_FE_GPU, "category");
    TestParameters tpFirst;
    tpFirst.AddString(PS_PLUGIN_NAME, "plugin_a");
    testFirst.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpFirst);

    Test testEud(0, EUD_PLUGIN_NAME, EUD_PLUGIN_NAME, "d", DCGM_FE_GPU, "category");
    TestParameters tpEud;
    tpEud.AddString(PS_PLUGIN_NAME, EUD_PLUGIN_NAME);
    testEud.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpEud);

    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testFirst) == 0);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testEud) == 0);

    entitySets.push_back(std::move(gpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetFailWithSkipsFromRowRemap("row_remap_failure_seed", { 0, 1 }, DCGM_FE_GPU, DCGM_FR_ROW_REMAP_FAILURE);
    mock->SetPassResult(EUD_PLUGIN_NAME);
    MockPluginLib *mockPtr = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(mockPtr->RunCount("row_remap_failure_seed") == 1);
    CHECK(mockPtr->RunCount(EUD_PLUGIN_NAME) == 0);
    CHECK(mockPtr->GetRunEntityIdSnapshots(EUD_PLUGIN_NAME).empty());
}

TEST_CASE("TestFramework::getTests and GetTestCategories are empty before loadPlugins")
{
    MockTestFramework tf;
    CHECK(tf.getTests().empty());
    CHECK(tf.GetTestCategories().empty());
}

TEST_CASE("TestFramework::loadPlugins registers tests from PushPlugin with mocked filesystem and dl loader")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    std::string const savedPluginPath = nvvsCommon.pluginPath;
    DcgmNs::Defer restorePluginPath([&] { nvvsCommon.pluginPath = savedPluginPath; });
    nvvsCommon.pluginPath = "/mock/plugins";

    MockTestFramework tf;

    char procExe[64];
    snprintf(procExe, sizeof(procExe), "/proc/%d/exe", getpid());
    tf.m_fileSystem.MockReadLink(procExe, "/fake/nvvs");
    struct stat binSt = {};
    binSt.st_uid      = 0;
    binSt.st_gid      = 0;
    binSt.st_mode     = S_IFREG | 0755;
    tf.m_fileSystem.MockStat("/fake/nvvs", binSt);
    tf.m_fileSystem.MockIsDirectory("/mock/plugins", true);
    tf.m_fileSystem.MockListDirectory("/mock/plugins/cudaless/", {});
    tf.m_fileSystem.MockListDirectory("/mock/plugins/cuda11/", {});
    tf.m_fileSystem.MockListDirectory("/mock/plugins/cuda12/", {});
    tf.m_fileSystem.MockListDirectory("/mock/plugins/cuda13/", {});

    std::string const cudalessDir = std::string("/mock/plugins") + "/cudaless/";
    tf.m_fileSystem.MockGetCurrentWorkingDirectory("/tmp/original");
    tf.m_fileSystem.MockChangeDirectory("/tmp/original", 0);
    tf.m_fileSystem.MockChangeDirectory(cudalessDir, 0);

    std::string const pluginCommonPath = cudalessDir + "/libpluginCommon.so.4";
    static int dummyCommonHandle;
    tf.m_dynamicLibrary.MockOpenReturns(pluginCommonPath, &dummyCommonHandle);

    dcgmDiagPluginTest_t pt {};
    SafeCopyTo(pt.testName, "my_subtest");
    SafeCopyTo(pt.description, "desc");
    SafeCopyTo(pt.testCategory, "cat_a");
    pt.targetEntityGroup  = DCGM_FE_GPU;
    pt.numValidParameters = 0;

    auto mock = std::make_unique<MockPluginLib>();
    mock->AddSupportedTest(pt);
    tf.PushPlugin(std::move(mock));

    tf.loadPlugins(nullptr);

    auto const tests = tf.getTests();
    REQUIRE(tests.size() == 1);
    CHECK(tests[0]->GetTestName() == "my_subtest");
    CHECK(tests[0]->GetPluginName() == "mock_plugin");

    auto const cats = tf.GetTestCategories();
    REQUIRE(cats.count("cat_a") == 1);
    CHECK(cats.at("cat_a").size() == 1);
}

TEST_CASE("TestFramework::loadPlugins registers tests from multiple PushPlugin mocks")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    std::string const savedPluginPath = nvvsCommon.pluginPath;
    DcgmNs::Defer restorePluginPath([&] { nvvsCommon.pluginPath = savedPluginPath; });
    nvvsCommon.pluginPath = "/mock/plugins";

    MockTestFramework tf;

    char procExe[64];
    snprintf(procExe, sizeof(procExe), "/proc/%d/exe", getpid());
    tf.m_fileSystem.MockReadLink(procExe, "/fake/nvvs");
    struct stat binSt = {};
    binSt.st_uid      = 0;
    binSt.st_gid      = 0;
    binSt.st_mode     = S_IFREG | 0755;
    tf.m_fileSystem.MockStat("/fake/nvvs", binSt);
    tf.m_fileSystem.MockIsDirectory("/mock/plugins", true);
    tf.m_fileSystem.MockListDirectory("/mock/plugins/cudaless/", {});
    tf.m_fileSystem.MockListDirectory("/mock/plugins/cuda11/", {});
    tf.m_fileSystem.MockListDirectory("/mock/plugins/cuda12/", {});
    tf.m_fileSystem.MockListDirectory("/mock/plugins/cuda13/", {});

    std::string const cudalessDir = std::string("/mock/plugins") + "/cudaless/";
    tf.m_fileSystem.MockGetCurrentWorkingDirectory("/tmp/original");
    tf.m_fileSystem.MockChangeDirectory("/tmp/original", 0);
    tf.m_fileSystem.MockChangeDirectory(cudalessDir, 0);

    std::string const pluginCommonPath = cudalessDir + "/libpluginCommon.so.4";
    static int dummyCommonHandle;
    tf.m_dynamicLibrary.MockOpenReturns(pluginCommonPath, &dummyCommonHandle);

    dcgmDiagPluginTest_t ptA {};
    SafeCopyTo(ptA.testName, "alpha_subtest");
    SafeCopyTo(ptA.description, "a");
    SafeCopyTo(ptA.testCategory, "cat_a");
    ptA.targetEntityGroup  = DCGM_FE_GPU;
    ptA.numValidParameters = 0;

    dcgmDiagPluginTest_t ptB {};
    SafeCopyTo(ptB.testName, "beta_subtest");
    SafeCopyTo(ptB.description, "b");
    SafeCopyTo(ptB.testCategory, "cat_b");
    ptB.targetEntityGroup  = DCGM_FE_GPU;
    ptB.numValidParameters = 0;

    auto mockA = std::make_unique<MockPluginLib>();
    mockA->SetPluginName("alpha_plugin");
    mockA->AddSupportedTest(ptA);

    auto mockB = std::make_unique<MockPluginLib>();
    mockB->SetPluginName("beta_plugin");
    mockB->AddSupportedTest(ptB);

    tf.PushPlugin(std::move(mockA));
    tf.PushPlugin(std::move(mockB));

    tf.loadPlugins(nullptr);

    auto const tests = tf.getTests();
    REQUIRE(tests.size() == 2);

    Test *alphaTest = nullptr;
    Test *betaTest  = nullptr;
    for (auto *t : tests)
    {
        if (t->GetTestName() == "alpha_subtest")
        {
            alphaTest = t;
        }
        if (t->GetTestName() == "beta_subtest")
        {
            betaTest = t;
        }
    }
    REQUIRE(alphaTest != nullptr);
    REQUIRE(betaTest != nullptr);
    CHECK(alphaTest->GetPluginIndex() == 0);
    CHECK(betaTest->GetPluginIndex() == 1);
    CHECK(alphaTest->GetPluginName() == "alpha_plugin");
    CHECK(betaTest->GetPluginName() == "beta_plugin");

    auto const cats = tf.GetTestCategories();
    REQUIRE(cats.count("cat_a") == 1);
    REQUIRE(cats.count("cat_b") == 1);
    CHECK(cats.at("cat_a").size() == 1);
    CHECK(cats.at("cat_b").size() == 1);
}

TEST_CASE("TestFramework::Go invokes the correct mock plugin by plugin index")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0 });

    Test test0(0, "alpha_plugin", "run_on_first", "d", DCGM_FE_GPU, "category");
    TestParameters tp0;
    tp0.AddString(PS_PLUGIN_NAME, "alpha_plugin");
    test0.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tp0);

    Test test1(1, "beta_plugin", "run_on_second", "d", DCGM_FE_GPU, "category");
    TestParameters tp1;
    tp1.AddString(PS_PLUGIN_NAME, "beta_plugin");
    test1.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tp1);

    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &test0) == 0);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &test1) == 0);

    entitySets.push_back(std::move(gpu));

    auto mockA = std::make_unique<MockPluginLib>();
    mockA->SetPluginName("alpha_plugin");
    mockA->SetPassResult("run_on_first");
    MockPluginLib *const ptrA = mockA.get();

    auto mockB = std::make_unique<MockPluginLib>();
    mockB->SetPluginName("beta_plugin");
    mockB->SetPassResult("run_on_second");
    MockPluginLib *const ptrB = mockB.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mockA));
    tf.PushPlugin(std::move(mockB));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(ptrA->RunCount("run_on_first") == 1);
    CHECK(ptrA->RunCount("run_on_second") == 0);
    CHECK(ptrB->RunCount("run_on_second") == 1);
    CHECK(ptrB->RunCount("run_on_first") == 0);
}

TEST_CASE("TestFramework::Go rejects plugin index past the last PushPlugin mock")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0 });

    Test test(2, "beta_plugin", "out_of_range", "d", DCGM_FE_GPU, "category");
    TestParameters tp;
    tp.AddString(PS_PLUGIN_NAME, "beta_plugin");
    test.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tp);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &test) == 0);

    entitySets.push_back(std::move(gpu));

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);

    auto mockA = std::make_unique<MockPluginLib>();
    auto mockB = std::make_unique<MockPluginLib>();
    mockA->SetPluginName("alpha_plugin");
    mockB->SetPluginName("beta_plugin");
    tf.PushPlugin(std::move(mockA));
    tf.PushPlugin(std::move(mockB));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    auto const &diag = tf.m_diagResponse.ConstResponse<dcgmDiagResponse_v12>();
    REQUIRE(diag.numErrors == 1);
    CHECK(diag.errors[0].testId == DCGM_DIAG_RESPONSE_SYSTEM_ERROR);
    CHECK(diag.errors[0].code == static_cast<unsigned int>(DCGM_ST_GENERIC_ERROR));
}

TEST_CASE("TestFramework::GetCompareName normalizes spaces and case")
{
    MockTestFramework tf;
    CHECK(tf.GetCompareName("My Test_Name", false) == "my_test_name");
    CHECK(tf.GetCompareName("MY_TEST_NAME", true) == "my test name");
    CHECK(tf.GetCompareName("AlreadyLower", false) == "alreadylower");
}

TEST_CASE("TestFramework::GetSubtestParameters includes software entry with common parameters")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);

    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> const parms = tf.GetSubtestParameters();
    REQUIRE(parms.count("software") == 1);

    auto const &sw = parms.at("software");
    REQUIRE_FALSE(sw.empty());

    bool foundPersistence = false;
    bool foundSuiteLevel  = false;
    for (auto const &p : sw)
    {
        if (std::string_view(p.parameterName) == SW_STR_REQUIRE_PERSISTENCE)
        {
            foundPersistence = true;
        }
        if (std::string_view(p.parameterName) == PS_SUITE_LEVEL)
        {
            foundSuiteLevel = true;
        }
    }
    CHECK(foundPersistence);
    CHECK(foundSuiteLevel);
}

TEST_CASE("TestFramework::SetDiagResponseVersion rejects unknown version")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    MockTestFramework tf;
    CHECK(tf.SetDiagResponseVersion(0x55665566u) == DCGM_ST_GENERIC_ERROR);
}

TEST_CASE("TestFramework::SetDiagResponseVersion cannot be set twice")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    CHECK(tf.SetDiagResponseVersion(dcgmDiagResponse_version11) == DCGM_ST_GENERIC_ERROR);
}

TEST_CASE("TestFramework::GetCompatibleCudaMajorVersion returns zero when driver major is zero")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    CHECK(MockTestFramework::GetCompatibleCudaMajorVersion(0, 0, 0) == 0);
}

TEST_CASE("TestFramework::Go with empty entity sets completes without invoking hardware plugins")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    auto mock                      = std::make_unique<MockPluginLib>();
    MockPluginLib *const pluginLib = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    tf.CreateSoftwarePluginFrameworks(entitySets);
    CHECK_NOTHROW(tf.Go(entitySets));

    CHECK(pluginLib->TotalRunCalls() == 0);
}

TEST_CASE("TestFramework::Go runs hardware tests on CpuSet without GPU software plugin path")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    auto cpu = std::make_unique<CpuSet>();
    cpu->AddEntityId(0);

    Test test(0, "plugin_a", "cpu_hw_test", "d", DCGM_FE_CPU, "category");
    TestParameters tp;
    tp.AddString(PS_PLUGIN_NAME, "plugin_a");
    test.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tp);
    REQUIRE(cpu->AddTestObject(HARDWARE_TEST_OBJS, &test) == 0);

    entitySets.push_back(std::move(cpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetPassResult("cpu_hw_test");
    MockPluginLib *const pluginLib = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(pluginLib->RunCount("cpu_hw_test") == 1);
    CHECK(tf.m_softwarePluginFrameworks.empty());

    auto const &diag = tf.m_diagResponse.ConstResponse<dcgmDiagResponse_v12>();
    for (unsigned int i = 0; i < diag.numTests; ++i)
    {
        CHECK(std::string_view(diag.tests[i].name) != SW_PLUGIN_NAME);
    }
}

TEST_CASE("TestFramework::Go runs software stack and hardware on GpuSet")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0 });

    Test test(0, "plugin_a", "gpu_hw_only", "d", DCGM_FE_GPU, "category");
    TestParameters tp;
    tp.AddString(PS_PLUGIN_NAME, "plugin_a");
    test.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tp);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &test) == 0);

    entitySets.push_back(std::move(gpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetPassResult("gpu_hw_only");
    MockPluginLib *const pluginLib = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(pluginLib->RunCount("gpu_hw_only") == 1);
    REQUIRE(tf.m_softwarePluginFrameworks.size() == 1);
    CHECK(tf.m_softwarePluginFrameworks[0]->GetRunCount() == 1);
}

TEST_CASE("TestFramework::Go runs hardware and integration tests on GpuSet")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0 });

    TestParameters tp;
    tp.AddString(PS_PLUGIN_NAME, "plugin_a");

    Test testHw(0, "plugin_a", "gpu_hw_multi", "d", DCGM_FE_GPU, "category");
    testHw.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tp);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testHw) == 0);

    Test testInt(0, "plugin_a", "gpu_integ_multi", "d", DCGM_FE_GPU, "category");
    testInt.pushArgVectorElement(Test::NVVS_CLASS_INTEGRATION, &tp);
    REQUIRE(gpu->AddTestObject(INTEGRATION_TEST_OBJS, &testInt) == 0);

    entitySets.push_back(std::move(gpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetPassResult("gpu_hw_multi");
    mock->SetPassResult("gpu_integ_multi");
    MockPluginLib *const pluginLib = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(pluginLib->RunCount("gpu_hw_multi") == 1);
    CHECK(pluginLib->RunCount("gpu_integ_multi") == 1);
    CHECK(pluginLib->TotalRunCalls() == 2);
    REQUIRE(tf.m_softwarePluginFrameworks.size() == 1);
    CHECK(tf.m_softwarePluginFrameworks[0]->GetRunCount() == 1);
}

TEST_CASE("TestFramework::Go runs software on GpuSet without hardware tests")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0 });
    entitySets.push_back(std::move(gpu));

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    REQUIRE(tf.m_softwarePluginFrameworks.size() == 1);
    CHECK(tf.m_softwarePluginFrameworks[0]->GetRunCount() == 1);
}

TEST_CASE("TestFramework::Go aggregates tests across CpuSet and GpuSet entity sets")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });

    auto cpu = std::make_unique<CpuSet>();
    cpu->AddEntityId(0);
    Test testCpu(0, "plugin_a", "cpu_agg", "d", DCGM_FE_CPU, "category");
    TestParameters tpCpu;
    tpCpu.AddString(PS_PLUGIN_NAME, "plugin_a");
    testCpu.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpCpu);
    REQUIRE(cpu->AddTestObject(HARDWARE_TEST_OBJS, &testCpu) == 0);
    entitySets.push_back(std::move(cpu));

    auto gpu = MakeGpuSetWithFakeGpus({ 1 });
    Test testGpu(0, "plugin_a", "gpu_agg", "d", DCGM_FE_GPU, "category");
    TestParameters tpGpu;
    tpGpu.AddString(PS_PLUGIN_NAME, "plugin_a");
    testGpu.pushArgVectorElement(Test::NVVS_CLASS_HARDWARE, &tpGpu);
    REQUIRE(gpu->AddTestObject(HARDWARE_TEST_OBJS, &testGpu) == 0);
    entitySets.push_back(std::move(gpu));

    auto mock = std::make_unique<MockPluginLib>();
    mock->SetPassResult("cpu_agg");
    mock->SetPassResult("gpu_agg");
    MockPluginLib *const pluginLib = mock.get();

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);
    tf.PushPlugin(std::move(mock));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.Go(entitySets);

    CHECK(pluginLib->RunCount("cpu_agg") == 1);
    CHECK(pluginLib->RunCount("gpu_agg") == 1);
    CHECK(pluginLib->TotalRunCalls() == 2);
    REQUIRE(tf.m_softwarePluginFrameworks.size() == 1);
    CHECK(tf.m_softwarePluginFrameworks[0]->GetRunCount() == 1);
}

TEST_CASE("TestFramework::RunSoftwarePlugin does not invoke SoftwarePluginFramework when no GPU entity sets")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    dcgmDiagPluginAttr_v1 const pluginAttr { .pluginId = 7 };

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.RunSoftwarePlugin(entitySets, &pluginAttr);

    CHECK(tf.m_softwarePluginFrameworks.empty());
}

TEST_CASE("TestFramework::RunSoftwarePlugin skips non-GPU entity sets")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    entitySets.push_back(std::make_unique<CpuSet>());

    dcgmDiagPluginAttr_v1 const pluginAttr { .pluginId = 8 };
    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.RunSoftwarePlugin(entitySets, &pluginAttr);

    CHECK(tf.m_softwarePluginFrameworks.empty());
}

TEST_CASE("TestFramework::RunSoftwarePlugin runs software stack once per GPU entity set")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpuA = MakeGpuSetWithFakeGpus({ 0 });
    entitySets.push_back(std::move(gpuA));

    auto gpuB = MakeGpuSetWithFakeGpus({ 1 });
    entitySets.push_back(std::move(gpuB));

    dcgmDiagPluginAttr_v1 const pluginAttr { .pluginId = 99 };
    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.RunSoftwarePlugin(entitySets, &pluginAttr);

    auto const &created = tf.m_softwarePluginFrameworks;
    REQUIRE(created.size() == 2);
    CHECK(created[0]->GetRunCount() == 1);
    CHECK(created[1]->GetRunCount() == 1);
    REQUIRE(created[1]->GetLastPluginId().has_value());
    CHECK(created[1]->GetLastPluginId().value() == 99);
}

TEST_CASE("TestFramework::RunSoftwarePlugin completes when mock software stack reports errors")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    int const savedChannelFd = nvvsCommon.channelFd;
    DcgmNs::Defer defer([&] { nvvsCommon.channelFd = savedChannelFd; });
    nvvsCommon.channelFd = -1;

    dcgmDiagError_v1 err {};
    err.code = 42;

    MockTestFramework tf;
    REQUIRE(tf.SetDiagResponseVersion(dcgmDiagResponse_version12) == DCGM_ST_OK);

    std::vector<std::unique_ptr<EntitySet>> entitySets;
    DcgmNs::Defer release([&] { ReleaseEntitySets(entitySets); });
    auto gpu = MakeGpuSetWithFakeGpus({ 0 });
    entitySets.push_back(std::move(gpu));

    tf.CreateSoftwarePluginFrameworks(entitySets);
    tf.m_softwarePluginFrameworks[0]->SetInjectedErrors({ err });

    dcgmDiagPluginAttr_v1 const pluginAttr { .pluginId = 0 };
    tf.RunSoftwarePlugin(entitySets, &pluginAttr);

    auto const &created = tf.m_softwarePluginFrameworks;
    REQUIRE(created.size() == 1);
    CHECK(created[0]->GetRunCount() == 1);
    REQUIRE(created[0]->GetErrors().size() == 1);
    CHECK(created[0]->GetErrors()[0].code == 42);
}
