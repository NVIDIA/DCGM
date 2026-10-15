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
#include <catch2/catch_all.hpp>

#include <DcgmGPUHardwareLimits.h>
#include <DcgmOps.h>
#include <Defer.hpp>
#include <MockDcgmOps.h>
#include <MockDcgmSystem.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <ScopedEnvVars.hpp>
#include <ScopedFilePermissions.hpp>
#include <ScopedWorkingDirectory.hpp>
#include <Software.h>
#include <TempDir.hpp>
#include <TempFile.hpp>
#include <UniquePtrUtil.h>
#include <dcgm_fields.h>
#include <dcgm_fields_internal.hpp>
#include <dcgm_structs.h>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern long long g_unrepairableMemoryFlagVal;
extern char g_graphicsPidsBlobVal;
extern long long g_sramExceededVal;
extern dcgmReturn_t g_sramExceededFieldStatus;
extern dcgmReturn_t g_graphicsPidsFieldStatus;
extern long long g_pendingRetirementsVal;
extern dcgmReturn_t g_pendingRetirementsFieldStatus;
extern long long g_dbeVolVal;
extern long long g_dbeTotalVal;
extern long long g_sbeTotalVal;
extern long long g_rowRemapFailedVal;
extern long long g_rowRemapPendingVal;
extern long long g_rowRemapUncorrectableTotalVal;
extern dcgmReturn_t g_memoryUnrepairableFieldStatus;
extern long long g_inforomValidVal;
extern dcgmReturn_t g_inforomValidFieldStatus;
extern long long g_gpuRecoveryActionVal;
extern dcgmReturn_t g_gpuRecoveryActionFieldStatus;
extern long long g_fmStatusVal;
extern long long g_fmErrorVal;
extern std::string g_fmUuidStr;
extern long long g_fmCliqueIdVal;
extern long long g_fmHealthMaskVal;

void ConfigureMockDcgmOpsForNvvsTests(DcgmNs::MockDcgmOps &mockDcgmOps);

namespace
{

dcgmHandle_t GetDcgmHandle(DcgmNs::MockDcgmOps &mockDcgmOps)
{
    ConfigureMockDcgmOpsForNvvsTests(mockDcgmOps);
    dcgmHandle_t dcgmHandle;
    REQUIRE(mockDcgmOps.dcgmInit() == DCGM_ST_OK);
    REQUIRE(mockDcgmOps.dcgmConnect_v2("localhost", nullptr, &dcgmHandle) == DCGM_ST_OK);
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

void SetRealGpu(dcgmDiagPluginEntityList_v1 &entityList, unsigned int idx, unsigned int gpuId)
{
    entityList.entities[idx].entity.entityId                                 = gpuId;
    entityList.entities[idx].entity.entityGroupId                            = DCGM_FE_GPU;
    entityList.entities[idx].auxField.gpu.status                             = DcgmEntityStatusOk;
    entityList.entities[idx].auxField.gpu.attributes.identifiers.pciDeviceId = 0xDEAD0000 + gpuId;
}

class TestSoftware : public Software
{
public:
    // Constructor, destructor, etc.
    using Software::Software;
    using LibraryCheck = Software::libraryCheck_t;

    using Software::addError;
    using Software::checkDriverPathDenylist;
    using Software::checkForBadEnvVaribles;
    using Software::checkForGraphicsProcesses;
    using Software::checkLibraries;
    using Software::checkPermissions;
    using Software::checkPersistenceMode;
    using Software::checkUnrepairableMemory;
    using Software::setSubtestName;

    static constexpr LibraryCheck CHECK_NVML   = Software::CHECK_NVML;
    static constexpr LibraryCheck CHECK_CUDA   = Software::CHECK_CUDA;
    static constexpr LibraryCheck CHECK_CUDATK = Software::CHECK_CUDATK;

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
        if (m_forceFindLibSuccess)
        {
            return true;
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

    void SetForceFindLibSuccess()
    {
        m_forceFindLibSuccess = true;
    }

    void CreateFakeNvidiaDevices()
    {
        std::error_code ec;
        std::filesystem::create_directories(m_testDeviceDir, ec);
        if (ec || !std::filesystem::exists(m_testDeviceDir))
        {
            SKIP("Failed to create test device directory: " << m_testDeviceDir << " - Error: " << ec.message());
        }

        m_deviceFiles.clear();
        for (auto const &deviceFile : GetDeviceFiles())
        {
            std::string const deviceFilePath = m_testDeviceDir + deviceFile;
            m_deviceFiles.emplace_back(TempFile::MakeUniqueFixed(deviceFilePath, "", std::filesystem::perms::none));
        }
    }

    static std::vector<std::string> const &GetDeviceFiles()
    {
        static std::vector<std::string> const files = { "/nvidia0", "/nvidia1", "/nvidia2" };
        return files;
    }

private:
    std::string m_testDeviceDir;
    std::vector<std::unique_ptr<TempFile>> m_deviceFiles;
    bool m_forceFindLibFailure { false };
    bool m_forceFindLibSuccess { false };
};

std::unique_ptr<MockDcgmSystem> MakeDcgmSystemMockWithEntities()
{
    auto dcgmSystemMock = std::make_unique<MockDcgmSystem>();
    AddEntitiesToMockDcgmSystem(*dcgmSystemMock);
    return dcgmSystemMock;
}

std::unique_ptr<dcgmDiagPluginEntityList_v1> MakeRealGpuEntityList(unsigned int gpuCount)
{
    auto pEntityList         = std::make_unique<dcgmDiagPluginEntityList_v1>();
    pEntityList->numEntities = gpuCount;
    for (unsigned int i = 0; i < gpuCount; ++i)
    {
        SetRealGpu(*pEntityList, i, i);
    }
    return pEntityList;
}

struct SoftwareTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    NvvsCommon nvvsCommon;
    dcgmHandle_t dcgmHandle;
    std::unique_ptr<MockDcgmSystem> dcgmSystemMock;

    SoftwareTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , dcgmHandle(GetDcgmHandle(*mockDcgmOps))
        , dcgmSystemMock(MakeDcgmSystemMockWithEntities())
    {}
};

struct SoftwareGoFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    NvvsCommon nvvsCommon;
    dcgmHandle_t dcgmHandle;
    std::unique_ptr<MockDcgmSystem> dcgmSystemMock;
    TestSoftware s;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> pEntityList;
    std::unique_ptr<dcgmDiagEntityResults_v2> pEntityResults;

    SoftwareGoFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , dcgmHandle(GetDcgmHandle(*mockDcgmOps))
        , dcgmSystemMock(MakeDcgmSystemMockWithEntities())
        , s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get())
        , pEntityList(MakeRealGpuEntityList(2))
        , pEntityResults(MakeUniqueZero<dcgmDiagEntityResults_v2>())
    {}

    static dcgmDiagPluginTestParameter_t MakeParam(char const *name, char const *value)
    {
        dcgmDiagPluginTestParameter_t param {};
        SafeCopyTo(param.parameterName, name);
        SafeCopyTo(param.parameterValue, value);
        param.type = DcgmPluginParamString;
        return param;
    }
};

} //namespace

TEST_CASE_METHOD(SoftwareTestFixture, "Software: CountDevEntry")
{
    Software s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

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

TEST_CASE_METHOD(SoftwareTestFixture, "Software Subtest Context")
{
    TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

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

TEST_CASE_METHOD(SoftwareTestFixture, "Negative test for DCGM_FR_NO_ACCESS_TO_FILE")
{
    // If we run as root, access will never fail, which is used in Software::checkPermissions() before error is added.
    if (DcgmNs::Utils::IsRunningAsRoot())
    {
        SKIP("This test requires running as a non-root user.");
    }

    // Setup dcgm system mock
    std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();
    AddEntitiesToMockDcgmSystem(*dcgmSystemMock);

    // Setup test object with custom device directory and fake devices
    TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());
    auto testDeviceDir = TempDir::Random("dcgm-software-dev-");
    s.SetDeviceDirectory(testDeviceDir.Path().string());
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

TEST_CASE_METHOD(SoftwareTestFixture, "Software: negative test for DCGM_FR_DEVICE_COUNT_MISMATCH")
{
    if (DcgmNs::Utils::IsRunningAsRoot())
    {
        SKIP("This test requires running as a non-root user.");
    }

    std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();
    AddEntitiesToMockDcgmSystem(*dcgmSystemMock);

    TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());
    auto testDeviceDir = TempDir::Random("dcgm-software-dev-");
    s.SetDeviceDirectory(testDeviceDir.Path().string());
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

TEST_CASE_METHOD(SoftwareTestFixture, "Software: negative test for DCGM_FR_FAULTY_MEMORY")
{
    unsigned int faultyGpuId { 0 };

    std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = faultyGpuId };
    DcgmNs::MockDcgmEntity gpu(entity);
    dcgmSystemMock->AddMockedEntity(gpu);

    TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

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

TEST_CASE_METHOD(SoftwareTestFixture, "Software: negative test for DCGM_FR_GRAPHICS_PROCESSES")
{
    unsigned int faultyGpuId { 0 };

    std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = faultyGpuId };
    DcgmNs::MockDcgmEntity gpu(entity);

    // Entity must be registered in MockDcgmSystem for the GPU list
    dcgmSystemMock->AddMockedEntity(gpu);

    TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

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

TEST_CASE_METHOD(SoftwareTestFixture, "Software: negative test for DCGM_FR_CANNOT_OPEN_LIB")
{
    TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());
    s.SetForceFindLibFailure();

    auto pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    TestSoftware::LibraryCheck checkType {};
    unsigned int expectedNumErrors { 0 };

    SECTION("CHECK_NVML produces DCGM_FR_CANNOT_OPEN_LIB")
    {
        checkType         = TestSoftware::CHECK_NVML;
        expectedNumErrors = 1; // libraries[] = { DCGM_NVML_SONAME }
    }
    SECTION("CHECK_CUDA produces DCGM_FR_CANNOT_OPEN_LIB")
    {
        checkType         = TestSoftware::CHECK_CUDA;
        expectedNumErrors = 1; // libraries[] = { DCGM_CUDA_SONAME }
    }
    SECTION("CHECK_CUDATK produces DCGM_FR_CANNOT_OPEN_LIB")
    {
        checkType         = TestSoftware::CHECK_CUDATK;
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

TEST_CASE_METHOD(SoftwareTestFixture, "Software: negative test for DCGM_FR_BAD_CUDA_ENV")
{
    constexpr const char *envVariable = "CUDA_INJECTION64_PATH";

    TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

    auto pEntityList                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    pEntityList->numEntities                      = 1;
    pEntityList->entities[0].entity.entityId      = 0;
    pEntityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    ScopedEnvVars envVars(envVariable, "/fake/path");

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

TEST_CASE_METHOD(SoftwareTestFixture, "Software: negative test for DCGM_FR_PERSISTENCE_MODE")
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
            // Register every GPU so checkPersistenceMode's gpuList matches the entity list size
            std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();
            for (unsigned int i = 0; i < gpuCount; i++)
            {
                dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = i };
                DcgmNs::MockDcgmEntity mockedEntity(entity);
                dcgmSystemMock->AddMockedEntity(mockedEntity);
            }

            TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

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

TEST_CASE_METHOD(SoftwareTestFixture, "Software: negative test for DCGM_FR_DENYLISTED_DRIVER")
{
    std::string const denylistedDriver = "nouveau";

    TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());
    auto pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    s.InitializeForEntityList(SW_PLUGIN_NAME, *pEntityList);

    auto tmpDir = TempDir::Random("dcgm-denylist-");

    // Mimic sysfs, PCI device "driver" entries are symlinks whose basename is the driver name
    std::filesystem::path const driverPath = tmpDir.Path() / "driver";
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

TEST_CASE_METHOD(SoftwareTestFixture, "Software: negative test for DCGM_FR_CORRUPT_INFOROM")
{
    constexpr unsigned int faultyGpuId { 0 };

    std::unique_ptr<MockDcgmSystem> dcgmSystemMock = std::make_unique<MockDcgmSystem>();

    dcgmGroupEntityPair_t faultyEntity { .entityGroupId = DCGM_FE_GPU, .entityId = faultyGpuId };
    DcgmNs::MockDcgmEntity faultyGpu(faultyEntity);
    dcgmSystemMock->AddMockedEntity(faultyGpu);

    g_inforomValidVal                                                                      = 0;
    g_inforomValidFieldStatus                                                              = DCGM_ST_OK;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_INFOROM_VALID] = DCGM_ST_OK;
    DcgmNs::Defer reset([mockDcgmOps = mockDcgmOps] {
        g_inforomValidVal         = 0;
        g_inforomValidFieldStatus = DCGM_ST_OK;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.erase(DCGM_FI_DEV_INFOROM_VALID);
    });

    Software s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

    auto pEntityList                             = std::make_unique<dcgmDiagPluginEntityList_v1>();
    pEntityList->numEntities                     = 1;
    pEntityList->entities[0].entity              = faultyEntity;
    pEntityList->entities[0].auxField.gpu.status = DcgmEntityStatusOk;
    pEntityList->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;

    TestParameters testParameters;
    REQUIRE(testParameters.AddString(SW_STR_DO_TEST, "inforom") == TP_ST_OK);
    std::vector<dcgmDiagPluginTestParameter_t> params = testParameters.GetParametersAsStruct();

    s.Go(SW_PLUGIN_NAME, pEntityList.get(), static_cast<unsigned int>(params.size()), params.data());

    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmReturn_t ret    = s.GetResults(s.GetSoftwareTestName(), pEntityResults.get());
    REQUIRE(ret == DCGM_ST_OK);
    CHECK(s.GetResult(s.GetSoftwareTestName()) == NVVS_RESULT_FAIL);

    REQUIRE(pEntityResults->numErrors == 1);
    CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(pEntityResults->errors[0].entity.entityId == faultyGpuId);
    CHECK(pEntityResults->errors[0].code == DCGM_FR_CORRUPT_INFOROM);
}

SCENARIO_METHOD(SoftwareGoFixture, "Software::Go dispatch validates inputs and short-circuits for fake GPUs")
{
    GIVEN("a Software plugin and a real-GPU entity list")
    {
        WHEN("Go is called with an unrecognized test name")
        {
            s.Go("not_software", pEntityList.get(), 0, nullptr);

            THEN("the software test was never initialized so GetResults throws")
            {
                CHECK_THROWS_AS(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()), std::out_of_range);
            }
        }

        WHEN("Go is called with a null entityInfo pointer")
        {
            s.Go(SW_PLUGIN_NAME, nullptr, 0, nullptr);

            THEN("the software test was never initialized so GetResults throws")
            {
                CHECK_THROWS_AS(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()), std::out_of_range);
            }
        }

        WHEN("Go is called with an unrecognized SW_STR_DO_TEST value")
        {
            std::array params { MakeParam(SW_STR_DO_TEST, "no_such_subtest") };

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the test is initialized but no subtest dispatches and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }

    GIVEN("an entity list containing a fake GPU (pciDeviceId == 0)")
    {
        pEntityList->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 0;

        WHEN("Go is called with the software test name")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), 0, nullptr);

            THEN("the result is set to PASS via the fake-GPU short-circuit and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_PASS);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software persistence-mode subtest is skipped when persistence check is disabled")
{
    for (unsigned int i = 0; i < pEntityList->numEntities; ++i)
    {
        pEntityList->entities[i].auxField.gpu.attributes.settings.persistenceModeEnabled = true;
    }

    GIVEN("a Software plugin invoked with the persistence-mode subtest")
    {
        std::array<dcgmDiagPluginTestParameter_t, 2> params {
            MakeParam(SW_STR_DO_TEST, "persistence_mode"),
            MakeParam(SW_STR_REQUIRE_PERSISTENCE, ""),
        };

        WHEN("SW_STR_REQUIRE_PERSISTENCE is False")
        {
            SafeCopyTo(params[1].parameterValue, "False");
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the result is SKIP and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_SKIP);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(SoftwareTestFixture,
                "Software permissions subtest via Go covers early-out, skip-device, and file-create paths")
{
    if (DcgmNs::Utils::IsRunningAsRoot())
    {
        SKIP("This test requires running as a non-root user.");
    }

    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();

    GIVEN("MockDcgmSystem::GetAllDevices is configured to fail")
    {
        auto dcgmSystemMock = std::make_unique<MockDcgmSystem>();
        dcgmSystemMock->SetGetAllDevicesReturn(DCGM_ST_GENERIC_ERROR);

        TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

        auto pEntityList         = std::make_unique<dcgmDiagPluginEntityList_v1>();
        pEntityList->numEntities = 1;
        SetRealGpu(*pEntityList, 0, 0);

        std::array<dcgmDiagPluginTestParameter_t, 1> params {};
        SafeCopyTo(params[0].parameterName, SW_STR_DO_TEST);
        SafeCopyTo(params[0].parameterValue, "permissions");
        params[0].type = DcgmPluginParamString;

        WHEN("the permissions subtest runs")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("checkPermissions exits before SetResult and emits no errors")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }

    GIVEN("a Software plugin with SW_STR_SKIP_DEVICE_TEST=True")
    {
        auto dcgmSystemMock = std::make_unique<MockDcgmSystem>();
        AddEntitiesToMockDcgmSystem(*dcgmSystemMock);

        TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

        auto pEntityList         = std::make_unique<dcgmDiagPluginEntityList_v1>();
        pEntityList->numEntities = 2;
        SetRealGpu(*pEntityList, 0, 0);
        SetRealGpu(*pEntityList, 1, 1);

        std::array<dcgmDiagPluginTestParameter_t, 2> params {};
        SafeCopyTo(params[0].parameterName, SW_STR_DO_TEST);
        SafeCopyTo(params[0].parameterValue, "permissions");
        SafeCopyTo(params[1].parameterName, SW_STR_SKIP_DEVICE_TEST);
        SafeCopyTo(params[1].parameterValue, "True");
        params[0].type = DcgmPluginParamString;
        params[1].type = DcgmPluginParamString;

        WHEN("the permissions subtest runs")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the device-count check is bypassed, the result is PASS, and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_PASS);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }

    GIVEN("the file-creation check runs from a non-writable working directory")
    {
        auto dcgmSystemMock = std::make_unique<MockDcgmSystem>();
        AddEntitiesToMockDcgmSystem(*dcgmSystemMock);

        TestSoftware s(dcgmHandle, mockDcgmOps, &nvvsCommon, dcgmSystemMock.get());

        auto tmpDir = TempDir::Random("dcgm-perm-");
        ScopedFilePermissions tmpDirPermissions(
            tmpDir.Path(), std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
        ScopedWorkingDirectory workingDirectory(tmpDir.Path());

        auto pEntityList         = std::make_unique<dcgmDiagPluginEntityList_v1>();
        pEntityList->numEntities = 2;
        SetRealGpu(*pEntityList, 0, 0);
        SetRealGpu(*pEntityList, 1, 1);

        std::array<dcgmDiagPluginTestParameter_t, 3> params {};
        SafeCopyTo(params[0].parameterName, SW_STR_DO_TEST);
        SafeCopyTo(params[0].parameterValue, "permissions");
        SafeCopyTo(params[1].parameterName, SW_STR_SKIP_DEVICE_TEST);
        SafeCopyTo(params[1].parameterValue, "True");
        SafeCopyTo(params[2].parameterName, SW_STR_CHECK_FILE_CREATION);
        SafeCopyTo(params[2].parameterValue, "True");
        params[0].type = DcgmPluginParamString;
        params[1].type = DcgmPluginParamString;
        params[2].type = DcgmPluginParamString;

        WHEN("the permissions subtest runs")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FILE_CREATE_PERMISSIONS is reported")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors >= 1);
                bool found = false;
                for (unsigned int i = 0; i < pEntityResults->numErrors; i++)
                {
                    if (pEntityResults->errors[i].code == DCGM_FR_FILE_CREATE_PERMISSIONS)
                    {
                        found = true;
                        break;
                    }
                }
                CHECK(found);
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software library-check subtest via Go covers all three positive paths")
{
    s.SetForceFindLibSuccess();

    GIVEN("a Software plugin where findLib is forced to succeed")
    {
        std::array<dcgmDiagPluginTestParameter_t, 1> params {
            MakeParam(SW_STR_DO_TEST, ""),
        };

        WHEN("the libraries_nvml subtest runs")
        {
            SafeCopyTo(params[0].parameterValue, "libraries_nvml");
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no DCGM_FR_CANNOT_OPEN_LIB errors are emitted")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("the libraries_cuda subtest runs")
        {
            SafeCopyTo(params[0].parameterValue, "libraries_cuda");
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no DCGM_FR_CANNOT_OPEN_LIB errors are emitted")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("the libraries_cudatk subtest runs")
        {
            SafeCopyTo(params[0].parameterValue, "libraries_cudatk");
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no DCGM_FR_CANNOT_OPEN_LIB errors are emitted")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software env-variables subtest via Go covers clean and multi-var states")
{
    static constexpr char const *checkKeys[] = {
        "NSIGHT_CUDA_DEBUGGER",
        "CUDA_INJECTION32_PATH",
        "CUDA_INJECTION64_PATH",
        "CUDA_AUTO_BOOST",
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
        "CUDA_COREDUMP_FILE",
        "CUDA_DEVICE_WAITS_ON_EXCEPTION",
        "CUDA_PROFILE",
        "COMPUTE_PROFILE",
        "OPENCL_PROFILE",
    };

    ScopedEnvVars envVars;
    for (auto const *checkKey : checkKeys)
    {
        envVars.Add(checkKey, std::nullopt);
    }

    std::array<dcgmDiagPluginTestParameter_t, 1> params {
        MakeParam(SW_STR_DO_TEST, "env_variables"),
    };

    GIVEN("the env-variables subtest is invoked via Go")
    {
        WHEN("no flagged environment variables are set")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are emitted")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("multiple flagged environment variables are set")
        {
            ScopedEnvVars envVars(
                { { "NSIGHT_CUDA_DEBUGGER", "1" }, { "CUDA_PROFILE", "1" }, { "OPENCL_PROFILE", "1" } });

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("one DCGM_FR_BAD_CUDA_ENV error is emitted per set variable and the result is WARN")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_WARN);
                REQUIRE(pEntityResults->numErrors == 3);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_BAD_CUDA_ENV);
                }
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software SRAM-threshold subtest via Go covers all four field-read outcomes")
{
    std::array<dcgmDiagPluginTestParameter_t, 1> params {
        MakeParam(SW_STR_DO_TEST, "sram_threshold"),
    };

    g_sramExceededVal                                      = 0;
    g_sramExceededFieldStatus                              = DCGM_ST_OK;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnCode = DCGM_ST_OK;
    DcgmNs::Defer resetStub([mockDcgmOps = mockDcgmOps] {
        g_sramExceededVal                                      = 0;
        g_sramExceededFieldStatus                              = DCGM_ST_OK;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnCode = DCGM_ST_OK;
    });

    GIVEN("the SRAM-threshold subtest is invoked via Go")
    {
        WHEN("the SRAM-exceeded field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnCode = DCGM_ST_GENERIC_ERROR;
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("the SRAM-exceeded field returns a non-OK status")
        {
            g_sramExceededFieldStatus = DCGM_ST_NOT_SUPPORTED;
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the check is skipped with informational output and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
                CHECK(pEntityResults->numInfo >= 1);
            }
        }

        WHEN("the SRAM-exceeded value is zero")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("the SRAM-exceeded value is non-zero")
        {
            g_sramExceededVal = 5;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_SRAM_THRESHOLD is reported per GPU and the result is WARN")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_WARN);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_SRAM_THRESHOLD);
                }
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software page-retirement subtest via Go covers pending-page outcomes")
{
    std::array<dcgmDiagPluginTestParameter_t, 1> params {
        MakeParam(SW_STR_DO_TEST, "page_retirement"),
    };

    g_pendingRetirementsVal         = 0;
    g_pendingRetirementsFieldStatus = DCGM_ST_OK;
    g_dbeVolVal                     = 0;
    g_dbeTotalVal                   = 0;
    g_sbeTotalVal                   = 0;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    DcgmNs::Defer resetStub([mockDcgmOps = mockDcgmOps] {
        g_pendingRetirementsVal         = 0;
        g_pendingRetirementsFieldStatus = DCGM_ST_OK;
        g_dbeVolVal                     = 0;
        g_dbeTotalVal                   = 0;
        g_sbeTotalVal                   = 0;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    });

    GIVEN("the page-retirement subtest is invoked via Go")
    {
        WHEN("the pending-retirement field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_PAGE_RETIRED_PENDING]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("the pending-retirement field returns a non-OK status")
        {
            g_pendingRetirementsFieldStatus = DCGM_ST_NOT_SUPPORTED;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the pending-retirement check is silently skipped and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("pending retirements are present and volatile DBE is non-zero")
        {
            g_pendingRetirementsVal = 5;
            g_dbeVolVal             = 1;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_DBE_PENDING_PAGE_RETIREMENTS is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_DBE_PENDING_PAGE_RETIREMENTS);
                }
            }
        }

        WHEN("pending retirements are present and volatile DBE is zero")
        {
            g_pendingRetirementsVal = 5;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_PENDING_PAGE_RETIREMENTS is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_PENDING_PAGE_RETIREMENTS);
                }
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software page-retirement subtest via Go covers total-retired-pages outcomes")
{
    std::array<dcgmDiagPluginTestParameter_t, 1> params {
        MakeParam(SW_STR_DO_TEST, "page_retirement"),
    };

    g_pendingRetirementsVal = 0;
    g_dbeTotalVal           = 0;
    g_sbeTotalVal           = 0;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    DcgmNs::Defer resetStub([mockDcgmOps = mockDcgmOps] {
        g_pendingRetirementsVal = 0;
        g_dbeTotalVal           = 0;
        g_sbeTotalVal           = 0;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    });

    GIVEN("the page-retirement subtest is invoked via Go with no pending retirements")
    {
        WHEN("the DBE-total field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("the SBE-total field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("DBE+SBE retired-page totals stay below the limit")
        {
            g_dbeTotalVal = 1;
            g_sbeTotalVal = 1;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("DBE+SBE retired-page totals reach the limit")
        {
            g_dbeTotalVal = DCGM_LIMIT_MAX_RETIRED_PAGES;
            g_sbeTotalVal = 0;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_RETIRED_PAGES_LIMIT is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_RETIRED_PAGES_LIMIT);
                }
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software row-remap subtest via Go covers all field-read outcomes")
{
    g_rowRemapFailedVal             = 0;
    g_rowRemapPendingVal            = 0;
    g_rowRemapUncorrectableTotalVal = 0;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    DcgmNs::Defer resetStub([mockDcgmOps = mockDcgmOps] {
        g_rowRemapFailedVal             = 0;
        g_rowRemapPendingVal            = 0;
        g_rowRemapUncorrectableTotalVal = 0;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    });

    GIVEN("the page-retirement subtest is invoked via Go (covers checkRowRemapping)")
    {
        std::array<dcgmDiagPluginTestParameter_t, 2> params {
            MakeParam(SW_STR_DO_TEST, "page_retirement"),
            MakeParam(PS_IGNORE_ERROR_CODES, ""),
        };

        WHEN("the row-remap-failed field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_ROW_REMAP_FAILED]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("row-remap-failed is positive and the error code is not ignored")
        {
            g_rowRemapFailedVal = 1;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_ROW_REMAP_FAILURE is reported per GPU")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_ROW_REMAP_FAILURE);
                }
            }
        }

        WHEN("row-remap-failed is positive and the error code is in PS_IGNORE_ERROR_CODES")
        {
            g_rowRemapFailedVal = 1;

            std::string ignoreList = "*:" + std::to_string(static_cast<int>(DCGM_FR_ROW_REMAP_FAILURE));
            SafeCopyTo(params[1].parameterValue, ignoreList.c_str());

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the error is suppressed and informational output is recorded instead")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
                CHECK(pEntityResults->numInfo >= 1);
            }
        }

        WHEN("the row-remap-pending field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_ROW_REMAP_PENDING]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("the uncorrectable-remap total reaches the limit")
        {
            long long const count           = DCGM_LIMIT_MAX_ROW_REMAP_UNCORRECTABLE + 7;
            g_rowRemapUncorrectableTotalVal = count;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_UNCORRECTABLE_ROW_REMAP_LIMIT is reported per GPU")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    auto const &error = pEntityResults->errors[i];
                    CHECK(error.code == DCGM_FR_UNCORRECTABLE_ROW_REMAP_LIMIT);
                    std::string const expectedMessage = "GPU " + std::to_string(error.entity.entityId)
                                                        + " exceeded uncorrectable row remap limit with "
                                                        + std::to_string(static_cast<unsigned int>(count))
                                                        + " remapped rows";
                    CHECK(std::string_view(error.msg).find(expectedMessage) != std::string_view::npos);
                }
            }
        }

        WHEN("pending row-remap is positive with uncorrectable-remap also positive (below limit)")
        {
            long long const count           = 37;
            g_rowRemapPendingVal            = 1;
            g_rowRemapUncorrectableTotalVal = count;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_UNCORRECTABLE_ROW_REMAP is reported per GPU")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    auto const &error = pEntityResults->errors[i];
                    CHECK(error.code == DCGM_FR_UNCORRECTABLE_ROW_REMAP);
                    std::string const expectedMessage
                        = "GPU " + std::to_string(error.entity.entityId) + " had uncorrectable memory errors and "
                          + std::to_string(static_cast<unsigned int>(count)) + " rows were remapped";
                    CHECK(std::string_view(error.msg).find(expectedMessage) != std::string_view::npos);
                }
            }
        }

        WHEN("pending row-remap is positive with uncorrectable-remap zero")
        {
            g_rowRemapPendingVal = 1;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_PENDING_ROW_REMAP is reported per GPU")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_PENDING_ROW_REMAP);
                }
            }
        }

        WHEN("no row-remap activity is present")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software memory-health subtest via Go covers field-query and clean-state outcomes")
{
    std::array<dcgmDiagPluginTestParameter_t, 1> params {
        MakeParam(SW_STR_DO_TEST, "memory_health"),
    };

    g_unrepairableMemoryFlagVal     = 0;
    g_memoryUnrepairableFieldStatus = DCGM_ST_OK;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    DcgmNs::Defer resetStub([mockDcgmOps = mockDcgmOps] {
        g_unrepairableMemoryFlagVal     = 0;
        g_memoryUnrepairableFieldStatus = DCGM_ST_OK;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    });

    GIVEN("the memory-health subtest is invoked via Go")
    {
        WHEN("the unrepairable-memory field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_MEMORY_UNREPAIRABLE]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("the unrepairable-memory field returns a non-OK status")
        {
            g_memoryUnrepairableFieldStatus = DCGM_ST_NOT_SUPPORTED;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the check is silently skipped and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("the unrepairable-memory flag is zero")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software inforom subtest via Go covers all field-read outcomes")
{
    std::array<dcgmDiagPluginTestParameter_t, 1> params {
        MakeParam(SW_STR_DO_TEST, "inforom"),
    };

    g_inforomValidVal         = 1;
    g_inforomValidFieldStatus = DCGM_ST_OK;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    DcgmNs::Defer resetStub([mockDcgmOps = mockDcgmOps] {
        g_inforomValidVal         = 1;
        g_inforomValidFieldStatus = DCGM_ST_OK;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    });

    GIVEN("the inforom subtest is invoked via Go")
    {
        WHEN("the inforom-valid field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_INFOROM_VALID]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("the inforom-valid status is DCGM_ST_NOT_SUPPORTED")
        {
            g_inforomValidFieldStatus = DCGM_ST_NOT_SUPPORTED;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the result is SKIP, informational output is recorded, and no errors are emitted")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_SKIP);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("the inforom-valid status is non-OK but not DCGM_ST_NOT_SUPPORTED")
        {
            g_inforomValidFieldStatus = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("informational output is recorded and no errors are emitted")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("the inforom-valid value is zero")
        {
            g_inforomValidVal = 0;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_CORRUPT_INFOROM is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_CORRUPT_INFOROM);
                }
            }
        }

        WHEN("the inforom-valid value is one")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software gpu-recovery-state subtest via Go covers all field-read outcomes")
{
    std::array<dcgmDiagPluginTestParameter_t, 1> params {
        MakeParam(SW_STR_DO_TEST, "gpu_recovery_state"),
    };

    g_gpuRecoveryActionVal         = 0;
    g_gpuRecoveryActionFieldStatus = DCGM_ST_OK;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    DcgmNs::Defer resetStub([mockDcgmOps = mockDcgmOps] {
        g_gpuRecoveryActionVal         = 0;
        g_gpuRecoveryActionFieldStatus = DCGM_ST_OK;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    });

    GIVEN("the gpu-recovery-state subtest is invoked via Go")
    {
        WHEN("the gpu-recovery-action field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_GPU_RECOVERY_ACTION]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FIELD_QUERY is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_FIELD_QUERY);
                }
            }
        }

        WHEN("the gpu-recovery-action field returns a non-OK status")
        {
            g_gpuRecoveryActionFieldStatus = DCGM_ST_NOT_SUPPORTED;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the check is silently skipped and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("the gpu-recovery-action value is zero")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("the gpu-recovery-action value is GPU_RESET")
        {
            g_gpuRecoveryActionVal = NVML_GPU_RECOVERY_ACTION_GPU_RESET;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_GPU_RECOVERY_RESET is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_GPU_RECOVERY_RESET);
                }
            }
        }

        WHEN("the gpu-recovery-action value is NODE_REBOOT")
        {
            g_gpuRecoveryActionVal = NVML_GPU_RECOVERY_ACTION_NODE_REBOOT;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_GPU_RECOVERY_REBOOT is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_GPU_RECOVERY_REBOOT);
                }
            }
        }

        WHEN("the gpu-recovery-action value is DRAIN_P2P")
        {
            g_gpuRecoveryActionVal = NVML_GPU_RECOVERY_ACTION_DRAIN_P2P;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_GPU_RECOVERY_DRAIN_P2P is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_GPU_RECOVERY_DRAIN_P2P);
                }
            }
        }

        WHEN("the gpu-recovery-action value is DRAIN_AND_RESET")
        {
            g_gpuRecoveryActionVal = NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_GPU_RECOVERY_DRAIN_RESET is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_GPU_RECOVERY_DRAIN_RESET);
                }
            }
        }

        WHEN("the gpu-recovery-action value is RECOVER_IMEX_DOMAIN")
        {
            g_gpuRecoveryActionVal = NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_GPU_RECOVERY_IMEX_DOMAIN is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_GPU_RECOVERY_IMEX_DOMAIN);
                }
            }
        }

        WHEN("the gpu-recovery-action value is unknown to the switch")
        {
            g_gpuRecoveryActionVal = 99;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_UNKNOWN is reported per GPU and the result is FAIL")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 2);
                for (unsigned int i = 0; i < pEntityResults->numErrors; ++i)
                {
                    CHECK(pEntityResults->errors[i].code == DCGM_FR_UNKNOWN);
                }
            }
        }
    }
}

SCENARIO_METHOD(SoftwareGoFixture, "Software fabric-manager subtest via Go covers all status outcomes")
{
    std::array<dcgmDiagPluginTestParameter_t, 1> params {
        MakeParam(SW_STR_DO_TEST, "fabric_manager"),
    };

    g_fmStatusVal     = DcgmFMStatusSuccess;
    g_fmErrorVal      = 0;
    g_fmUuidStr       = "test-uuid";
    g_fmCliqueIdVal   = 42;
    g_fmHealthMaskVal = 0;
    mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    DcgmNs::Defer resetStub([mockDcgmOps = mockDcgmOps] {
        g_fmStatusVal     = DcgmFMStatusSuccess;
        g_fmErrorVal      = 0;
        g_fmUuidStr       = "test-uuid";
        g_fmCliqueIdVal   = 42;
        g_fmHealthMaskVal = 0;
        mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId.clear();
    });

    GIVEN("the fabric-manager subtest is invoked via Go")
    {
        WHEN("the FM-status field query fails")
        {
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_FABRIC_MANAGER_STATUS]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("informational output is recorded per GPU and no errors are emitted")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("FM status is DcgmFMStatusNotSupported")
        {
            g_fmStatusVal = DcgmFMStatusNotSupported;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("FM status is DcgmFMStatusNvmlTooOld")
        {
            g_fmStatusVal = DcgmFMStatusNvmlTooOld;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the result is SKIP, the loop short-circuits, and no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_SKIP);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("FM status is DcgmFMStatusSuccess")
        {
            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("no errors are recorded")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }

        WHEN("FM status is DcgmFMStatusNotStarted")
        {
            g_fmStatusVal = DcgmFMStatusNotStarted;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR is reported with 'training not started'")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 1);
                CHECK(pEntityResults->errors[0].code == DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR);
                CHECK(std::string_view(pEntityResults->errors[0].msg).find("training not started")
                      != std::string_view::npos);
            }
        }

        WHEN("FM status is DcgmFMStatusInProgress")
        {
            g_fmStatusVal = DcgmFMStatusInProgress;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR is reported with 'training in progress'")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 1);
                CHECK(pEntityResults->errors[0].code == DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR);
                CHECK(std::string_view(pEntityResults->errors[0].msg).find("training in progress")
                      != std::string_view::npos);
            }
        }

        WHEN("FM status is DcgmFMStatusFailure and the FM-error field is readable")
        {
            g_fmStatusVal = DcgmFMStatusFailure;
            g_fmErrorVal  = DCGM_ST_BADPARAM;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the error message embeds 'Training completed with an error'")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 1);
                CHECK(pEntityResults->errors[0].code == DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR);
                std::string_view msg = pEntityResults->errors[0].msg;
                CHECK(msg.find("Training completed with an error") != std::string_view::npos);
                CHECK(msg.find("failed to read the error code") == std::string_view::npos);
            }
        }

        WHEN("FM status is DcgmFMStatusFailure and the FM-error field query fails")
        {
            g_fmStatusVal = DcgmFMStatusFailure;
            mockDcgmOps->m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_FABRIC_MANAGER_ERROR]
                = DCGM_ST_GENERIC_ERROR;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the error message embeds 'failed to read the error code'")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 1);
                CHECK(pEntityResults->errors[0].code == DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR);
                CHECK(std::string_view(pEntityResults->errors[0].msg).find("failed to read the error code")
                      != std::string_view::npos);
            }
        }

        WHEN("FM status is an unknown value")
        {
            g_fmStatusVal = 99;

            s.Go(SW_PLUGIN_NAME, pEntityList.get(), params.size(), params.data());

            THEN("the error message indicates an unknown fabric manager state")
            {
                REQUIRE(s.GetResults(SW_PLUGIN_NAME, pEntityResults.get()) == DCGM_ST_OK);
                CHECK(s.GetResult(SW_PLUGIN_NAME) == NVVS_RESULT_FAIL);
                REQUIRE(pEntityResults->numErrors == 1);
                CHECK(pEntityResults->errors[0].code == DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR);
                CHECK(std::string_view(pEntityResults->errors[0].msg).find("unknown fabric manager state")
                      != std::string_view::npos);
            }
        }
    }
}
