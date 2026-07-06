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
#include <fstream>

#include <PluginInterface.h>
#include <PluginStrings.h>
#include <string_view>
#define TEST_SOFTWARE_PLUGIN
#include <DcgmLib.h>
#include <Defer.hpp>
#include <MockDcgmLib.h>
#include <MockDcgmSystem.h>
#include <Software.h>
#include <UniquePtrUtil.h>
#include <dcgm_structs.h>

extern long long g_unrepairableMemoryFlagVal;
extern char g_graphicsPidsBlobVal;

namespace
{

dcgmHandle_t GetDcgmHandle(DcgmNs::MockDcgmLib &dcgmLibMock)
{
    dcgmHandle_t dcgmHandle;
    REQUIRE(dcgmLibMock.dcgmInit() == DCGM_ST_OK);
    REQUIRE(dcgmLibMock.dcgmConnect_v2("localhost", nullptr, &dcgmHandle) == DCGM_ST_OK);
    return dcgmHandle;
}

void AddEntitiesToMockDcgmSystem(MockDcgmSystem &dcgmSystemMock)
{
    std::array<dcgmGroupEntityPair_t, 2> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
    };
    for (auto const &entity : entities)
    {
        DcgmNs::MockDcgmEntity mockedEntity(entity);
        dcgmSystemMock.AddMockedEntity(mockedEntity);
    }
}

class TestSoftware : public Software
{
public:
    // Constructor, destructor, etc.
    using Software::Software;
    ~TestSoftware()
    {
        if (m_testDeviceDir.empty())
        {
            return;
        }
        std::error_code ec;
        for (auto const &deviceFile : GetDeviceFiles())
        {
            std::string const deviceFilePath = m_testDeviceDir + deviceFile;
            if (!std::filesystem::remove(deviceFilePath, ec) || ec)
            {
                WARN("Error while removing test device file: " << deviceFilePath << " - Error: " << ec.message());
            }
        }
        if (!std::filesystem::remove(m_testDeviceDir, ec) || ec)
        {
            WARN("Error while removing test device directory: " << m_testDeviceDir << " - Error: " << ec.message());
        }
    }

    // Override the virtual method to return custom directory
    std::string GetDeviceDirectory() const override
    {
        return m_testDeviceDir;
    }

    // Override the virtual method to find library
    bool findLib(std::string library, std::string &error) override
    {
        if (m_forceFindLibFailure)
        {
            error = "fake dlerror: cannot open shared object file";
            return false;
        }
        return Software::findLib(library, error);
    }

    // Test object methods
    void SetDeviceDirectory(std::string const &deviceDir)
    {
        m_testDeviceDir = deviceDir;
    }

    void SetForceFindLibFailure()
    {
        m_forceFindLibFailure = true;
    }

    void CreateFakeNvidiaDevices()
    {
        std::error_code ec;
        if (!std::filesystem::create_directories(m_testDeviceDir, ec) || ec)
        {
            SKIP("Failed to create test device directory: " << m_testDeviceDir << " - Error: " << ec.message());
        }

        for (auto const &deviceFile : GetDeviceFiles())
        {
            std::string const deviceFilePath = m_testDeviceDir + deviceFile;
            std::ofstream file(deviceFilePath);
            REQUIRE(file);
            file.close();

            // Remove permissions to simulate access failure
            if (chmod(deviceFilePath.c_str(), 0000) != 0)
            {
                SKIP("Failed to remove permissions from test device file: " << deviceFilePath);
            }
        }
    }

    static std::vector<std::string> const &GetDeviceFiles()
    {
        static std::vector<std::string> const files = { "/nvidia0", "/nvidia1", "/nvidia2" };
        return files;
    }

private:
    std::string m_testDeviceDir;
    bool m_forceFindLibFailure { false };
};

} //namespace

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

TEST_CASE("Negative test for DCGM_FR_NO_ACCESS_TO_FILE")
{
    // If we run as root, access will never fail, which is used in Software::checkPermissions() before error is added.
    if (DcgmNs::Utils::IsRunningAsRoot())
    {
        SKIP("This test requires running as a non-root user.");
    }

    // Setup dcgm lib and dcgm system mock
    std::unique_ptr<DcgmNs::MockDcgmLib> dcgmLibMock = std::make_unique<DcgmNs::MockDcgmLib>();
    dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);
    std::unique_ptr<MockDcgmSystem> dcgmSystemMock   = std::make_unique<MockDcgmSystem>();
    AddEntitiesToMockDcgmSystem(*dcgmSystemMock);

    // Setup test object with custom device directory and fake devices
    TestSoftware s(dcgmHandle, std::move(dcgmSystemMock));
    s.SetDeviceDirectory("./test_dev_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    s.CreateFakeNvidiaDevices();
    std::unique_ptr<dcgmDiagPluginEntityList_v1> pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    std::unique_ptr<dcgmDiagEntityResults_v2> pEntityResults = std::make_unique<dcgmDiagEntityResults_v2>();
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    // Invoke the function to hit the code path that generates the DCGM_FR_NO_ACCESS_TO_FILE error
    s.checkPermissions(true, false);
    dcgmReturn_t ret = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
    REQUIRE(ret == DCGM_ST_OK);

    // Verify that DCGM_FR_NO_ACCESS_TO_FILE error is generated.
    dcgmDiagEntityResults_v2 const &entityResults = *pEntityResults;
    unsigned int noAccessErrorCount               = 0;
    for (unsigned int i = 0;
         i < std::min(static_cast<std::size_t>(entityResults.numErrors), std::size(entityResults.errors));
         i++)
    {
        if (entityResults.errors[i].code == DCGM_FR_NO_ACCESS_TO_FILE)
        {
            noAccessErrorCount++;
            std::string_view errorMsg = entityResults.errors[i].msg;
            REQUIRE(errorMsg.size() > 0);
            REQUIRE(errorMsg.find(s.GetDeviceDirectory()) != std::string_view::npos);
        }
    }
    REQUIRE(noAccessErrorCount == s.GetDeviceFiles().size());
}

TEST_CASE("Software: negative test for DCGM_FR_DEVICE_COUNT_MISMATCH")
{
    if (DcgmNs::Utils::IsRunningAsRoot())
    {
        SKIP("This test requires running as a non-root user.");
    }

    std::unique_ptr<DcgmNs::MockDcgmLib> dcgmLibMock = std::make_unique<DcgmNs::MockDcgmLib>();
    dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);
    std::unique_ptr<MockDcgmSystem> dcgmSystemMock   = std::make_unique<MockDcgmSystem>();
    AddEntitiesToMockDcgmSystem(*dcgmSystemMock);

    TestSoftware s(dcgmHandle, std::move(dcgmSystemMock));
    s.SetDeviceDirectory("./test_dev_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    s.CreateFakeNvidiaDevices();

    std::unique_ptr<dcgmDiagPluginEntityList_v1> pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    std::unique_ptr<dcgmDiagEntityResults_v2> pEntityResults = std::make_unique<dcgmDiagEntityResults_v2>();
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    s.checkPermissions(false, false);
    dcgmReturn_t ret = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
    REQUIRE(ret == DCGM_ST_OK);

    dcgmDiagEntityResults_v2 const &entityResults = *pEntityResults;
    REQUIRE(entityResults.numErrors == 4);

    // Verify the primary error under test.
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(entityResults.errors[0].entity.entityId == 0);
    CHECK(entityResults.errors[0].code == DCGM_FR_DEVICE_COUNT_MISMATCH);

    // The mismatch is caused by inaccessible device files, which also produce access errors.
    for (unsigned int i = 1; i < entityResults.numErrors; i++)
    {
        CHECK(entityResults.errors[i].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(entityResults.errors[i].entity.entityId == 0);
        CHECK(entityResults.errors[i].code == DCGM_FR_NO_ACCESS_TO_FILE);
    }
}

TEST_CASE("Software: negative test for DCGM_FR_FAULTY_MEMORY")
{
    unsigned int faultyGpuId { 0 };

    std::unique_ptr<DcgmNs::MockDcgmLib> dcgmLibMock = std::make_unique<DcgmNs::MockDcgmLib>();
    dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);

    std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = faultyGpuId };
    DcgmNs::MockDcgmEntity gpu(entity);
    dcgmSystemMock->AddMockedEntity(gpu);

    Software s(dcgmHandle, std::move(dcgmSystemMock));

    auto pEntityList                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    pEntityList->numEntities                      = 1;
    pEntityList->entities[0].entity.entityId      = faultyGpuId;
    pEntityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    // Unrepairable memory flag set, triggering the faulty memory error path
    g_unrepairableMemoryFlagVal = 1;
    DcgmNs::Defer resetFlag([&] { g_unrepairableMemoryFlagVal = 0; });
    s.checkUnrepairableMemory();

    // Verify the error is reported for the correct entity with the expected error code
    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmReturn_t ret    = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
    REQUIRE(ret == DCGM_ST_OK);

    REQUIRE(pEntityResults->numErrors == 1);
    CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(pEntityResults->errors[0].entity.entityId == faultyGpuId);
    CHECK(pEntityResults->errors[0].code == DCGM_FR_FAULTY_MEMORY);
}

TEST_CASE("Software: negative test for DCGM_FR_GRAPHICS_PROCESSES")
{
    unsigned int faultyGpuId { 0 };

    std::unique_ptr<DcgmNs::MockDcgmLib> dcgmLibMock = std::make_unique<DcgmNs::MockDcgmLib>();
    dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);

    std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = faultyGpuId };
    DcgmNs::MockDcgmEntity gpu(entity);

    // Entity must be registered in MockDcgmSystem for the GPU list
    dcgmSystemMock->AddMockedEntity(gpu);

    Software s(dcgmHandle, std::move(dcgmSystemMock));

    auto pEntityList                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    pEntityList->numEntities                      = 1;
    pEntityList->entities[0].entity.entityId      = faultyGpuId;
    pEntityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    g_graphicsPidsBlobVal = 1;
    DcgmNs::Defer resetFlag([&] { g_graphicsPidsBlobVal = 0; });
    CHECK(s.checkForGraphicsProcesses() == 0);

    // Verify the warning is reported for the correct entity with the expected error code
    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmReturn_t ret    = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
    REQUIRE(ret == DCGM_ST_OK);
    CHECK(s.GetResult(s.GetSoftwareTestName()) == NVVS_RESULT_WARN);

    REQUIRE(pEntityResults->numErrors == 1);
    CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(pEntityResults->errors[0].entity.entityId == faultyGpuId);
    CHECK(pEntityResults->errors[0].code == DCGM_FR_GRAPHICS_PROCESSES);
}

TEST_CASE("Software: negative test for DCGM_FR_CANNOT_OPEN_LIB")
{
    std::unique_ptr<DcgmNs::MockDcgmLib> dcgmLibMock = std::make_unique<DcgmNs::MockDcgmLib>();
    dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);

    TestSoftware s(dcgmHandle);
    s.SetForceFindLibFailure();

    auto pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    Software::libraryCheck_t checkType {};
    unsigned int expectedNumErrors { 0 };

    SECTION("CHECK_NVML produces DCGM_FR_CANNOT_OPEN_LIB")
    {
        checkType         = Software::CHECK_NVML;
        expectedNumErrors = 1; // libraries[] = { DCGM_NVML_SONAME }
    }
    SECTION("CHECK_CUDA produces DCGM_FR_CANNOT_OPEN_LIB")
    {
        checkType         = Software::CHECK_CUDA;
        expectedNumErrors = 1; // libraries[] = { DCGM_CUDA_SONAME }
    }
    SECTION("CHECK_CUDATK produces DCGM_FR_CANNOT_OPEN_LIB")
    {
        checkType         = Software::CHECK_CUDATK;
        expectedNumErrors = 2; // libraries[] = { DCGM_CUDART_SONAME, DCGM_CUBLAS_SONAME }
    }

    // Should return true when at least one library failed
    CHECK(s.checkLibraries(checkType) == true);

    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmReturn_t ret    = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
    REQUIRE(ret == DCGM_ST_OK);

    REQUIRE(pEntityResults->numErrors == expectedNumErrors);
    for (unsigned int i = 0; i < pEntityResults->numErrors; i++)
    {
        CHECK(pEntityResults->errors[i].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(pEntityResults->errors[i].entity.entityId == 0);
        CHECK(pEntityResults->errors[i].code == DCGM_FR_CANNOT_OPEN_LIB);
    }
}

TEST_CASE("Software: negative test for DCGM_FR_BAD_CUDA_ENV")
{
    std::unique_ptr<DcgmNs::MockDcgmLib> dcgmLibMock = std::make_unique<DcgmNs::MockDcgmLib>();
    dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);
    constexpr const char *envVariable                = "CUDA_INJECTION64_PATH";

    Software s(dcgmHandle);

    auto pEntityList                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    pEntityList->numEntities                      = 1;
    pEntityList->entities[0].entity.entityId      = 0;
    pEntityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    setenv(envVariable, "/fake/path", 1);
    DcgmNs::Defer resetEnv([&] { unsetenv(envVariable); });

    s.checkForBadEnvVaribles();

    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmReturn_t ret    = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
    REQUIRE(ret == DCGM_ST_OK);
    CHECK(s.GetResult(s.GetSoftwareTestName()) == NVVS_RESULT_WARN);

    REQUIRE(pEntityResults->numErrors == 1);
    CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(pEntityResults->errors[0].entity.entityId == 0);
    CHECK(pEntityResults->errors[0].code == DCGM_FR_BAD_CUDA_ENV);
    CHECK(std::string(pEntityResults->errors[0].msg).find(envVariable) != std::string::npos);
}

TEST_CASE("Software: negative test for DCGM_FR_PERSISTENCE_MODE")
{
    constexpr unsigned int gpuCount = 2;

    struct TestCase
    {
        std::string name;
        std::vector<unsigned int> disabledGpuIds;
        unsigned int expectedNumErrors;
    };

    std::vector<TestCase> const cases = {
        { "single entity affected", { 1 }, 1 },
        { "all entities affected", { 0, 1 }, gpuCount },
        { "no entities affected", {}, 0 },
    };

    for (auto const &tc : cases)
    {
        DYNAMIC_SECTION(tc.name)
        {
            std::unique_ptr<DcgmNs::MockDcgmLib> dcgmLibMock = std::make_unique<DcgmNs::MockDcgmLib>();
            dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);

            // Register every GPU so checkPersistenceMode's gpuList matches the entity list size
            std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();
            for (unsigned int i = 0; i < gpuCount; i++)
            {
                dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = i };
                DcgmNs::MockDcgmEntity mockedEntity(entity);
                dcgmSystemMock->AddMockedEntity(mockedEntity);
            }

            Software s(dcgmHandle, std::move(dcgmSystemMock));

            auto pEntityList         = std::make_unique<dcgmDiagPluginEntityList_v1>();
            pEntityList->numEntities = gpuCount;
            for (unsigned int i = 0; i < gpuCount; i++)
            {
                pEntityList->entities[i].entity.entityId                                         = i;
                pEntityList->entities[i].entity.entityGroupId                                    = DCGM_FE_GPU;
                pEntityList->entities[i].auxField.gpu.attributes.settings.persistenceModeEnabled = true;
            }
            // Flip persistenceModeEnabled to false only on the target GPUs
            for (auto const gpuId : tc.disabledGpuIds)
            {
                pEntityList->entities[gpuId].auxField.gpu.attributes.settings.persistenceModeEnabled = false;
            }

            s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

            CHECK(s.checkPersistenceMode(*pEntityList) == 0);

            auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
            REQUIRE(s.GetResults(s.GetSoftwareTestName(), pEntityResults.get()) == DCGM_ST_OK);

            REQUIRE(pEntityResults->numErrors == tc.expectedNumErrors);

            // Each error must be attributed to the correct disabled GPU
            for (unsigned int i = 0; i < pEntityResults->numErrors; i++)
            {
                CHECK(pEntityResults->errors[i].entity.entityGroupId == DCGM_FE_GPU);
                CHECK(pEntityResults->errors[i].entity.entityId == tc.disabledGpuIds[i]);
                CHECK(pEntityResults->errors[i].code == DCGM_FR_PERSISTENCE_MODE);
            }

            if (tc.expectedNumErrors > 0)
            {
                CHECK(s.GetResult(s.GetSoftwareTestName()) == NVVS_RESULT_WARN);
            }
        }
    }
}

TEST_CASE("Software: negative test for DCGM_FR_DENYLISTED_DRIVER")
{
    std::unique_ptr<DcgmNs::MockDcgmLib> dcgmLibMock = std::make_unique<DcgmNs::MockDcgmLib>();
    dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);
    std::string const denylistedDriver               = "nouveau";

    Software s(dcgmHandle);
    auto pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    // Atomically create a unique tmp dir for the symlink fixture
    char tmpl[] = "/tmp/dcgm_denylist_XXXXXX";
    REQUIRE(::mkdtemp(tmpl) != nullptr);
    std::filesystem::path const tmpDir = tmpl;
    std::error_code ec;
    DcgmNs::Defer cleanup([&] {
        std::filesystem::remove_all(tmpDir, ec);
        if (ec)
        {
            WARN("Failed to remove tmp dir: " << tmpDir << " - Error: " << ec.message());
        }
    });

    // Mimic sysfs, PCI device "driver" entries are symlinks whose basename is the driver name
    std::filesystem::path const driverPath = tmpDir / "driver";
    REQUIRE(::symlink(("/sys/bus/pci/drivers/" + denylistedDriver).c_str(), driverPath.c_str()) == 0);

    std::vector<std::string> const denyList = { denylistedDriver };
    CHECK(s.checkDriverPathDenylist(driverPath.string(), denyList) == 1);

    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(s.GetResults(s.GetSoftwareTestName(), pEntityResults.get()) == DCGM_ST_OK);

    REQUIRE(pEntityResults->numErrors == 1);
    CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(pEntityResults->errors[0].entity.entityId == 0);
    CHECK(pEntityResults->errors[0].code == DCGM_FR_DENYLISTED_DRIVER);
    // Verify the matched driver name is in the user facing message
    CHECK(std::string_view(pEntityResults->errors[0].msg).find(denylistedDriver) != std::string_view::npos);
}
