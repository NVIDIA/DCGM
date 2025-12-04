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
#define TEST_SOFTWARE_PLUGIN
#include <DcgmLib.h>
#include <DcgmLibMock.h>
#include <DcgmSystemMock.h>
#include <Software.h>
#include <dcgm_structs.h>

namespace
{

dcgmHandle_t GetDcgmHandle(DcgmNs::DcgmLibMock &dcgmLibMock)
{
    dcgmHandle_t dcgmHandle;
    REQUIRE(dcgmLibMock.dcgmInit() == DCGM_ST_OK);
    REQUIRE(dcgmLibMock.dcgmConnect_v2("localhost", nullptr, &dcgmHandle) == DCGM_ST_OK);
    return dcgmHandle;
}

void AddEntitiesToDcgmSystemMock(DcgmSystemMock &dcgmSystemMock)
{
    std::array<dcgmGroupEntityPair_t, 2> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
    };
    for (auto const &entity : entities)
    {
        DcgmNs::DcgmMockEntity mockedEntity(entity);
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

    // Test object methods
    void SetDeviceDirectory(std::string const &deviceDir)
    {
        m_testDeviceDir = deviceDir;
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
    std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmNs::DcgmLibMock>();
    dcgmHandle_t dcgmHandle                          = GetDcgmHandle(*dcgmLibMock);
    std::unique_ptr<DcgmSystemMock> dcgmSystemMock   = std::make_unique<DcgmSystemMock>();
    AddEntitiesToDcgmSystemMock(*dcgmSystemMock);

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
