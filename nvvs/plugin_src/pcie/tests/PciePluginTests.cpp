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
#include <Pcie.h>
#include <PcieMain.h>

#include "dcgm_fields.h"
#include <BwCheckerMain.h>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <DcgmStringHelpers.h>
#include <MockCublasOps.h>
#include <MockCudaCommon.h>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginStrings.h>
#include <ScopedAtomic.hpp>
#include <UniquePtrUtil.h>
#include <algorithm>
#include <catch2/catch_all.hpp>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>


class TestBusGrind : public BusGrind
{
public:
    TestBusGrind(dcgmHandle_t handle, DcgmNs::DcgmOpsAnyUniquePtr dcgmOps, NvvsCommon *nvvsCommon)
        : BusGrind(handle,
                   std::move(dcgmOps),
                   std::make_unique<MockCudaOps>(),
                   std::make_unique<MockCublasOps>(),
                   std::make_unique<MockCudaCommon>(),
                   nvvsCommon)
    {}

    void SetCopySizes()
    {
        BusGrind::SetCopySizes();
    }

    void SetCudaCapabilityInfo(unsigned int const val)
    {
        m_cudaCompat = val;
    }

    dcgmReturn_t SetCudaCapabilityInfoWrapper()
    {
        return BusGrind::SetCudaCapabilityInfo();
    }

    TestParameters &GetTestParams()
    {
        return *m_testParameters;
    }

    bool GetPrintedConcurrentGpuErrorMessage() const
    {
        return m_printedConcurrentGpuErrorMessage;
    }

    void ResetPrintedConcurrentGpuErrorMessage()
    {
        m_printedConcurrentGpuErrorMessage = false;
    }
};

namespace
{
struct PciePluginTestFixture
{
    MockCudaOps mockCudaOps;
    MockCublasOps mockCublasOps;
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique2;
    DcgmNs::MockDcgmOps *mockDcgmOps2;
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique3;
    DcgmNs::MockDcgmOps *mockDcgmOps3;
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique4;
    DcgmNs::MockDcgmOps *mockDcgmOps4;
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique5;
    DcgmNs::MockDcgmOps *mockDcgmOps5;
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique6;
    DcgmNs::MockDcgmOps *mockDcgmOps6;
    NvvsCommon nvvsCommon;

    PciePluginTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , mockDcgmOpsUnique2(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps2(mockDcgmOpsUnique2.get())
        , mockDcgmOpsUnique3(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps3(mockDcgmOpsUnique3.get())
        , mockDcgmOpsUnique4(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps4(mockDcgmOpsUnique4.get())
        , mockDcgmOpsUnique5(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps5(mockDcgmOpsUnique5.get())
        , mockDcgmOpsUnique6(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps6(mockDcgmOpsUnique6.get())
    {}
};
} // namespace

namespace
{

dcgmNvLinkStatus_v5 MakeNvLinkStatus()
{
    dcgmNvLinkStatus_v5 linkStatus {};
    linkStatus.version                    = dcgmNvLinkStatus_version5;
    linkStatus.numGpus                    = 2;
    linkStatus.gpus[0].entityId           = 0;
    linkStatus.gpus[0].linkState[0]       = DcgmNvLinkLinkStateUp;
    linkStatus.gpus[1].entityId           = 1;
    linkStatus.gpus[1].linkState[0]       = DcgmNvLinkLinkStateUp;
    linkStatus.numNvSwitches              = 1;
    linkStatus.nvSwitches[0].entityId     = 0;
    linkStatus.nvSwitches[0].linkState[0] = DcgmNvLinkLinkStateUp;
    return linkStatus;
}
} // namespace

TEST_CASE_METHOD(PciePluginTestFixture, "Pcie: pcie_gpu_id_in_list")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                                 = 2;
    for (unsigned idx = 0; idx < entityInfo->numEntities; ++idx)
    {
        entityInfo->entities[idx].entity.entityId      = idx + 1;
        entityInfo->entities[idx].entity.entityGroupId = DCGM_FE_GPU;
    }

    CHECK(pcie_gpu_id_in_list(0, *(entityInfo)) == false);
    for (unsigned int i = 3; i < 10; i++)
    {
        CHECK(pcie_gpu_id_in_list(i, *(entityInfo)) == false);
    }

    for (unsigned int i = 0; i < entityInfo->numEntities; i++)
    {
        CHECK(pcie_gpu_id_in_list(entityInfo->entities[i].entity.entityId, *(entityInfo)) == true);
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "Pcie: pcie_check_nvlink_status_expected")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    mockDcgmOps->m_discoveryState.nvLinkStatus              = MakeNvLinkStatus();
    dcgmHandle_t handle                                     = mockDcgmOps->m_handleId;
    entityInfo->numEntities                                 = 2;
    const unsigned int ourGpuId                             = 1;
    entityInfo->entities[0].entity.entityId                 = ourGpuId;
    entityInfo->entities[0].entity.entityGroupId            = DCGM_FE_GPU;
    entityInfo->entities[1].entity.entityId                 = 2;
    entityInfo->entities[1].entity.entityGroupId            = DCGM_FE_GPU;
    auto entityResults                                      = MakeUniqueZero<dcgmDiagEntityResults_v2>();

    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);
    bg.m_gpuNvlinksExpectedUp = 1;

    SECTION("Base Case No Errors")
    {
        pcie_check_nvlink_status(&bg, *entityInfo, mockDcgmOps, handle);
        bg.GetResults(bg.GetPcieTestName(), entityResults.get());
        CHECK(entityResults->numErrors == 0);
    }

    SECTION("Expected Up Mismatch")
    {
        bg.m_gpuNvlinksExpectedUp = 2;
        pcie_check_nvlink_status(&bg, *entityInfo, mockDcgmOps, handle);
        bg.GetResults(bg.GetPcieTestName(), entityResults.get());
        CHECK(entityResults->numErrors == 1);
    }

    bg.m_nvSwitchNvlinksExpectedUp = 1;
    SECTION("No Errors Again")
    {
        pcie_check_nvlink_status(&bg, *entityInfo, mockDcgmOps, handle);
        bg.GetResults(bg.GetPcieTestName(), entityResults.get());
        CHECK(entityResults->numErrors == 0);
    }

    SECTION("Expected Up Switches Mismatch")
    {
        bg.m_nvSwitchNvlinksExpectedUp = 2;
        pcie_check_nvlink_status(&bg, *entityInfo, mockDcgmOps, handle);
        bg.GetResults(bg.GetPcieTestName(), entityResults.get());
        CHECK(entityResults->numErrors == 1);
    }
}

std::string startDelimiter = BWC_JSON_OUTPUT_START;
std::string endDelimiter   = BWC_JSON_OUTPUT_END;

std::string testJson1 = startDelimiter + R""(
{
    "GPUs" : [
        {
            "gpuId" : 7,
            "maxRxBw" : 2.98,
            "maxTxBw" : 39502.8,
            "maxBidirBw" : 3502.8
        }
    ]
}
)"" + endDelimiter;

std::string testJson2 = startDelimiter + R""(
{
    "GPUs" : [
        {
            "gpuId" : 2,
            "maxRxBw" : 395002000.8,
            "maxTxBw" : 49502000.8,
            "maxBidirBw" : 4902.8
        }
    ]
}
)"" + endDelimiter;

std::string testJson3 = startDelimiter + R""(
{
    "GPUs" : [
        {
            "gpuId" : 4,
            "error" : "Got rocked and couldn't launch stuff"
        }
    ]
}
)"" + endDelimiter;

TEST_CASE_METHOD(PciePluginTestFixture, "Pcie: ProcessChildrenOutputs")
{
    dcgmHandle_t handle = {};

    auto entityResults                                      = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    // This test keeps six MockDcgmOps-backed BusGrind objects alive at once.
    // The fixture owns the heap-backed mocks because stacking all six can exceed the test thread stack.
    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
    entityInfo->numEntities                      = 1;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;

    std::vector<dcgmChildInfo_t> childrenInfo;
    dcgmChildInfo_t childInfo;
    childInfo.stdoutStr     = testJson3;
    childInfo.stderrStr     = "";
    childInfo.pid           = 23981;
    childInfo.readOutputRet = 12;
    childInfo.readErrorRet  = 0;
    childInfo.outputFdIndex = 0;
    childrenInfo.push_back(childInfo);

    double minBw = 10.0;
    std::string groupName("bobby");
    auto tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);
    bg.m_testParameters.reset(tp);
    unsigned int ret = ProcessChildrenOutputs(childrenInfo, bg, groupName);
    CHECK(ret == 1); // we should have one failed test
    bg.GetResults(bg.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 1);
    CHECK(entityResults->errors[0].entity.entityId == 4);
    CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);

    TestBusGrind bg2(handle, std::move(mockDcgmOpsUnique2), &nvvsCommon);
    bg2.InitializeForEntityList(bg2.GetPcieTestName(), *entityInfo);
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg2.m_testParameters.reset(tp);
    memset(entityResults.get(), 0, sizeof(*entityResults));
    childrenInfo.clear();
    childInfo.stdoutStr     = testJson1;
    childInfo.readOutputRet = 0;
    childrenInfo.push_back(childInfo);
    ret = ProcessChildrenOutputs(childrenInfo, bg2, groupName);
    CHECK(ret == 1); // we should have one failed test due to low bandwidth
    bg2.GetResults(bg2.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 1);
    CHECK(entityResults->errors[0].entity.entityId == 7);
    CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);

    TestBusGrind bg3(handle, std::move(mockDcgmOpsUnique3), &nvvsCommon);
    bg3.InitializeForEntityList(bg3.GetPcieTestName(), *entityInfo);
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg3.m_testParameters.reset(tp);
    memset(entityResults.get(), 0, sizeof(*entityResults));
    childrenInfo.clear();
    childInfo.stdoutStr = testJson2;
    childrenInfo.push_back(childInfo);
    ret = ProcessChildrenOutputs(childrenInfo, bg3, groupName);
    CHECK(ret == 0); // we shouldn't have any failures
    bg3.GetResults(bg3.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 0);

    TestBusGrind bg4(handle, std::move(mockDcgmOpsUnique4), &nvvsCommon);
    bg4.InitializeForEntityList(bg4.GetPcieTestName(), *entityInfo);
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg4.m_testParameters.reset(tp);
    memset(entityResults.get(), 0, sizeof(*entityResults));
    childrenInfo.clear();
    childInfo.stdoutStr    = "This isn't json";
    childInfo.stderrStr    = "";
    childInfo.readErrorRet = 0;
    childrenInfo.push_back(std::move(childInfo));
    ret = ProcessChildrenOutputs(childrenInfo, bg4, groupName);
    CHECK(ret == 1);
    bg4.GetResults(bg4.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 1);                 // we get one failure for bad json
    CHECK(entityResults->errors[0].entity.entityId == 0); // this error isn't tied to a GPU id
    CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);

    // Test with stderr output (libnuma warnings on stderr, clean JSON on stdout)
    TestBusGrind bg5(handle, std::move(mockDcgmOpsUnique5), &nvvsCommon);
    bg5.InitializeForEntityList(bg5.GetPcieTestName(), *entityInfo);
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg5.m_testParameters.reset(tp);
    memset(entityResults.get(), 0, sizeof(*entityResults));
    childrenInfo.clear();
    // libnuma warning goes to stderr (properly separated now)
    childInfo.stderrStr = "libnuma: Warning: node 3 not allowed";
    // Clean JSON with delimiters on stdout
    childInfo.stdoutStr     = startDelimiter
                              + "\n"
                                "{\n"
                                "    \"GPUs\" : [\n"
                                "        {\n"
                                "            \"gpuId\" : 2,\n"
                                "            \"maxRxBw\" : 395002000.8,\n"
                                "            \"maxTxBw\" : 49502000.8,\n"
                                "            \"maxBidirBw\" : 4902.8\n"
                                "        }\n"
                                "    ]\n"
                                "}\n"
                              + endDelimiter;
    childInfo.readOutputRet = 0;
    childInfo.readErrorRet  = 0;
    childrenInfo.push_back(childInfo);
    ret = ProcessChildrenOutputs(childrenInfo, bg5, groupName);
    CHECK(ret == 0); // Should successfully parse
    bg5.GetResults(bg5.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 0); // No errors

    // Test with stderr read failure and JSON parsing failure
    TestBusGrind bg6(handle, std::move(mockDcgmOpsUnique6), &nvvsCommon);
    bg6.InitializeForEntityList(bg6.GetPcieTestName(), *entityInfo);
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg6.m_testParameters.reset(tp);
    memset(entityResults.get(), 0, sizeof(*entityResults));
    childrenInfo.clear();
    // Malformed stdout (no delimiters) - will cause JSON parsing to fail
    childInfo.stdoutStr = "{\"GPUs\": []}";
    // stderr has libnuma warning, but read failed so it can't be processed
    childInfo.stderrStr     = "libnuma: Warning: node 3 not allowed\nlibnuma: Warning: cpu 0 not allowed";
    childInfo.readOutputRet = 0;
    childInfo.readErrorRet  = EIO; // stderr read failed
    childrenInfo.push_back(childInfo);
    ret = ProcessChildrenOutputs(childrenInfo, bg6, groupName);
    CHECK(ret == 1); // Should fail
    bg6.GetResults(bg6.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 1);                 // Error reported
    CHECK(entityResults->errors[0].entity.entityId == 0); // Not tied to specific GPU
    CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);
}

SCENARIO_METHOD(PciePluginTestFixture, "Pcie: ProcessChildrenOutputs records child-output failure variants")
{
    GIVEN("an initialized PCIe plugin and a bandwidth subtest")
    {
        dcgmHandle_t handle                                     = {};
        auto entityResults                                      = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
        TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
        entityInfo->numEntities                      = 1;
        entityInfo->entities[0].entity.entityId      = 0;
        entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;

        std::string groupName("bobby");
        auto tp = new TestParameters();
        tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, 10.0);
        bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);
        bg.m_testParameters.reset(tp);

        WHEN("a child reports top-level JSON errors")
        {
            std::vector<dcgmChildInfo_t> childrenInfo;
            dcgmChildInfo_t childInfo {};
            childInfo.stdoutStr = startDelimiter + R""(
{
    "errors" : [
        "first failure",
        "second failure"
    ]
}
)"" + endDelimiter;
            childInfo.pid       = 101;
            childrenInfo.push_back(childInfo);

            unsigned int ret = ProcessChildrenOutputs(childrenInfo, bg, groupName);
            bg.GetResults(bg.GetPcieTestName(), entityResults.get());

            THEN("the global child error is recorded and processing stops for that child")
            {
                REQUIRE(ret == 1);
                REQUIRE(entityResults->numErrors == 1);
                CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);
                CHECK(std::string_view(entityResults->errors[0].msg).find("first failure,second failure")
                      != std::string_view::npos);
            }
        }

        WHEN("the child output stream cannot be read")
        {
            std::vector<dcgmChildInfo_t> childrenInfo;
            dcgmChildInfo_t childInfo {};
            childInfo.stdoutStr     = "";
            childInfo.stderrStr     = "";
            childInfo.pid           = 102;
            childInfo.readOutputRet = EIO;
            childrenInfo.push_back(childInfo);

            unsigned int ret = ProcessChildrenOutputs(childrenInfo, bg, groupName);
            bg.GetResults(bg.GetPcieTestName(), entityResults.get());

            THEN("the read failure is reported as a global internal error")
            {
                REQUIRE(ret == 1);
                REQUIRE(entityResults->numErrors == 1);
                CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);
                CHECK(std::string_view(entityResults->errors[0].msg).find("couldn't be read")
                      != std::string_view::npos);
            }
        }

        WHEN("the child emits malformed JSON inside valid delimiters")
        {
            std::vector<dcgmChildInfo_t> childrenInfo;
            dcgmChildInfo_t childInfo {};
            childInfo.stdoutStr = startDelimiter + "\n{\"GPUs\": [\n" + endDelimiter;
            childInfo.pid       = 103;
            childrenInfo.push_back(childInfo);

            unsigned int ret = ProcessChildrenOutputs(childrenInfo, bg, groupName);
            bg.GetResults(bg.GetPcieTestName(), entityResults.get());

            THEN("the parse failure is recorded separately from empty-output failures")
            {
                REQUIRE(ret == 1);
                REQUIRE(entityResults->numErrors == 1);
                CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);
                CHECK(std::string_view(entityResults->errors[0].msg).find("couldn't be parsed")
                      != std::string_view::npos);
            }
        }
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "Pcie: StartDcgmGroupWatch")
{
    mockDcgmOps->m_groupManagementState.addReturn = DCGM_ST_OK;
    auto *dcgmOps                                 = mockDcgmOps;
    dcgmHandle_t handle                           = mockDcgmOps->m_handleId;
    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
    std::vector<unsigned short> fieldIds = { DCGM_FI_PROF_NVLINK_TX_BYTES };
    std::vector<unsigned int> gpuIds     = { 0, 1, 2 };

    SECTION("Happy path returns valid DcgmGroup ptr")
    {
        std::unique_ptr<DcgmGroup> dcgmGroupPtr = StartDcgmGroupWatch(&bg, fieldIds, gpuIds, dcgmOps, handle);
        REQUIRE(dcgmGroupPtr.get() != nullptr);
    }
    SECTION("Field group creation error returns nullptr")
    {
        dcgmOps->m_fieldGroupState.createReturn = DCGM_ST_BADPARAM;
        std::unique_ptr<DcgmGroup> dcgmGroupPtr = StartDcgmGroupWatch(&bg, fieldIds, gpuIds, dcgmOps, handle);
        REQUIRE(dcgmGroupPtr.get() == nullptr);
    }
    SECTION("Group creation error returns nullptr")
    {
        dcgmOps->m_groupManagementState.createReturn = DCGM_ST_BADPARAM;
        std::unique_ptr<DcgmGroup> dcgmGroupPtr      = StartDcgmGroupWatch(&bg, fieldIds, gpuIds, dcgmOps, handle);
        REQUIRE(dcgmGroupPtr.get() == nullptr);
    }
    SECTION("Watch field error returns nullptr")
    {
        dcgmOps->m_fieldGroupState.watchReturn  = DCGM_ST_BADPARAM;
        std::unique_ptr<DcgmGroup> dcgmGroupPtr = StartDcgmGroupWatch(&bg, fieldIds, gpuIds, dcgmOps, handle);
        REQUIRE(dcgmGroupPtr.get() == nullptr);
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "Pcie: ExtractJson with delimiters")
{
    std::string startDelimiter(BWC_JSON_OUTPUT_START);
    std::string endDelimiter(BWC_JSON_OUTPUT_END);

    SECTION("JSON with proper delimiters")
    {
        std::string jsonValue = "{\"test\": \"value\"}";
        std::string output    = startDelimiter + "\n" + jsonValue + "\n" + endDelimiter;
        auto result           = ExtractJson(output);
        CHECK(result == jsonValue + "\n"); // Trailing newline is included
    }

    SECTION("JSON with delimiters and prefix contamination")
    {
        std::string jsonValue = "{\"GPUs\": []}";
        std::string output
            = "libnuma: Warning: node 3 not allowed\n" + startDelimiter + "\n" + jsonValue + "\n" + endDelimiter;
        auto result = ExtractJson(output);
        CHECK(result == jsonValue + "\n"); // Trailing newline is included
    }

    SECTION("JSON with only start delimiter")
    {
        std::string jsonValue = "{\"GPUs\": []}";
        std::string output    = startDelimiter + "\n" + jsonValue + "\n";
        auto result           = ExtractJson(output);
        CHECK(result.empty());
    }

    SECTION("JSON without delimiters")
    {
        std::string output = "{\"GPUs\": []}";
        auto result        = ExtractJson(output);
        CHECK(result.empty());
    }

    SECTION("Empty output")
    {
        std::string output = "";
        auto result        = ExtractJson(output);
        CHECK(result.empty());
    }

    SECTION("Multiple lines of warnings before JSON")
    {
        std::string jsonValue = "{\n"
                                "    \"GPUs\" : [\n"
                                "        {\n"
                                "            \"gpuId\" : 0\n"
                                "        }\n"
                                "    ]\n"
                                "}\n";
        std::string output    = "libnuma: Warning: node 3 not allowed\n"
                                "libnuma: Warning: node 4 not allowed\n"
                                "libnuma: Warning: node 5 not allowed\n"
                                + startDelimiter + "\n" + jsonValue + endDelimiter;
        auto result           = ExtractJson(output);
        CHECK(result == jsonValue);
    }

    SECTION("No newline after start delimiter - does not parse")
    {
        std::string output = startDelimiter + "{\"test\": 1}" + endDelimiter;
        auto result        = ExtractJson(output);
        CHECK(result.empty());
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "Pcie: SetCopySize")
{
    constexpr auto BEFORE_BLACKWELL = 9;
    constexpr auto COMPAT_BLACKWELL = 10;

    static_assert(PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY != PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
    static_assert(PCIE_HOPPER_AND_BEFORE_DEFAULT_BROKEN_P2P_SIZE != PCIE_BLACKWELL_DEFAULT_BROKEN_P2P_SIZE);

    dcgmHandle_t handle = 42;
    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);

    bg.SetCudaCapabilityInfo(BEFORE_BLACKWELL);
    bg.SetCopySizes(); // expected to be a no-op in this case
    auto &tp = bg.GetTestParams();

    SECTION("Hopper And Before")
    {
        std::vector<std::tuple<std::string, std::string, double>> const testData = {
            { PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_P2P_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_P2P_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_BROKEN_P2P,
              PCIE_SUBTEST_BROKEN_P2P_SIZE_IN_KB,
              PCIE_HOPPER_AND_BEFORE_DEFAULT_BROKEN_P2P_SIZE }
        };

        for (auto [subtest, param, expectedVal] : testData)
        {
            double actual = tp.GetSubTestDouble(std::move(subtest), std::move(param));
            CHECK(actual == expectedVal);
        }
    }

    bg.SetCudaCapabilityInfo(COMPAT_BLACKWELL);
    bg.SetCopySizes(); // should apply GPU-specific defaults
    tp = bg.GetTestParams();

    SECTION("Blackwell")
    {
        std::vector<std::tuple<std::string, std::string, double>> const testData = {
            { PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_P2P_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_P2P_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED,
              PCIE_STR_INTS_PER_COPY,
              PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY },
            { PCIE_SUBTEST_BROKEN_P2P, PCIE_SUBTEST_BROKEN_P2P_SIZE_IN_KB, PCIE_BLACKWELL_DEFAULT_BROKEN_P2P_SIZE }
        };

        for (auto [subtest, param, expectedVal] : testData)
        {
            double actual = tp.GetSubTestDouble(std::move(subtest), std::move(param));
            CHECK(actual == expectedVal);
        }
    }
}


class DcgmRecorderMock : public DcgmRecorderBase
{
public:
    std::map<std::pair<unsigned int, unsigned short>, std::pair<dcgmReturn_t, dcgmFieldValue_v2>> fieldValues;

    dcgmReturn_t GetCurrentFieldValue(unsigned int gpuId,
                                      unsigned short fieldId,
                                      dcgmFieldValue_v2 &val,
                                      unsigned int /* flags */) override
    {
        auto it = fieldValues.find({ gpuId, fieldId });
        if (it != fieldValues.end())
        {
            val = it->second.second;
            return it->second.first;
        }

        // Fallback to match the default behavior of DcgmRecorderBase
        return DCGM_ST_NOT_SUPPORTED;
    }
};

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind: SetCudaCapabilityInfo, negative test for DCGM_FR_DCGM_API")
{
    dcgmHandle_t handle                                     = 42;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
    entityInfo->numEntities                      = 2;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo->entities[1].entity.entityId      = 1;
    entityInfo->entities[1].entity.entityGroupId = DCGM_FE_GPU;
    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    auto mockRecorder                                                     = std::make_unique<DcgmRecorderMock>();
    mockRecorder->fieldValues[{ 0, DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY }] = { DCGM_ST_UNKNOWN_FIELD, {} };
    mockRecorder->fieldValues[{ 1, DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY }] = { DCGM_ST_UNKNOWN_FIELD, {} };
    bg.SetDcgmRecorder(std::move(mockRecorder));
    auto ret = bg.SetCudaCapabilityInfoWrapper();

    REQUIRE(ret == DCGM_ST_UNKNOWN_FIELD);

    DcgmError expectedError { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DCGM_API, expectedError, "GetCurrentFieldValue");
    expectedError.AddDcgmError(DCGM_ST_UNKNOWN_FIELD);

    // The error is not entity specific
    nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());

    unsigned int count { 0 };
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        // Make sure the error is not entity specific
        if (entityPair.entityGroupId == DCGM_FE_NONE)
        {
            REQUIRE(diagErrors.size() == 1);
            REQUIRE(diagErrors[0].entity.entityGroupId == DCGM_FE_NONE);
            REQUIRE(std::string(diagErrors[0].msg) == expectedError.GetMessage());
            count++;
        }
        else
        {
            REQUIRE(diagErrors.size() == 0);
        }
    }
    // Make sure the map has only one entity with entityGroupId == DCGM_FE_NONE
    REQUIRE(count == 1);
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind: negative test for DCGM_FR_CONCURRENT_GPUS")
{
    unsigned int numGpus = 1;

    dcgmHandle_t handle                                     = 42;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
    entityInfo->numEntities                      = numGpus;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    SECTION("outputConcurrentPairsP2PBandwidthMatrix")
    {
        bg.ResetPrintedConcurrentGpuErrorMessage();
        int ret = outputConcurrentPairsP2PBandwidthMatrix(&bg, &mockCudaOps, false);
        REQUIRE(ret == 0);
        REQUIRE(bg.GetPrintedConcurrentGpuErrorMessage() == true);
    }
    SECTION("outputConcurrent1DExchangeBandwidthMatrix")
    {
        bg.ResetPrintedConcurrentGpuErrorMessage();
        int ret = outputConcurrent1DExchangeBandwidthMatrix(&bg, &mockCudaOps, false);
        REQUIRE(ret == 0);
        REQUIRE(bg.GetPrintedConcurrentGpuErrorMessage() == true);
    }

    nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        REQUIRE(diagErrors.size() == 0);
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind: negative test for DCGM_FR_PCIE_WIDTH")
{
    unsigned int gpuCount                                   = 4;
    unsigned int targetGpuId                                = 1;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                                 = gpuCount;
    TestBusGrind bg(42, std::move(mockDcgmOpsUnique), &nvvsCommon);

    for (unsigned int i = 0; i < gpuCount; i++)
    {
        entityInfo->entities[i].entity.entityId      = i;
        entityInfo->entities[i].entity.entityGroupId = DCGM_FE_GPU;
    }

    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    // Populate bg.gpu with minimal PluginDevice objects, no CUDA needed
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        auto *pd  = new PluginDevice(mockDcgmOps, &mockCudaOps);
        pd->gpuId = i;
        bg.gpu.push_back(pd);
    }

    // bg_check_pci_link exits early if m_test_links is false
    bg.m_test_links = true;

    std::string subTest = PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED;
    auto &tp            = bg.GetTestParams();
    tp.SetSubTestDouble(subTest, PCIE_STR_MIN_PCI_GEN, 3.0);
    tp.SetSubTestDouble(subTest, PCIE_STR_MIN_PCI_WIDTH, 16.0);

    // Mock field values to pass all gates except the width check for targetGpuId
    auto mockRecorder             = std::make_unique<DcgmRecorderMock>();
    dcgmFieldValue_v2 valPstate   = {};
    valPstate.value.i64           = NVML_PSTATE_0;
    dcgmFieldValue_v2 valGen      = {};
    valGen.value.i64              = 4;
    dcgmFieldValue_v2 valWidthOk  = {};
    valWidthOk.value.i64          = 16;
    dcgmFieldValue_v2 valWidthLow = {};
    valWidthLow.value.i64         = 1;

    for (unsigned int i = 0; i < gpuCount; i++)
    {
        mockRecorder->fieldValues[{ i, DCGM_FI_DEV_GPU_PSTATE }]    = { DCGM_ST_OK, valPstate };
        mockRecorder->fieldValues[{ i, DCGM_FI_DEV_PCIE_LINK_GEN }] = { DCGM_ST_OK, valGen };
        mockRecorder->fieldValues[{ i, DCGM_FI_DEV_PCIE_LINK_WIDTH }]
            = { DCGM_ST_OK, (i == targetGpuId) ? valWidthLow : valWidthOk };
    }

    bg.SetDcgmRecorder(std::move(mockRecorder));

    int failedTests = bg_check_pci_link(bg, subTest);
    REQUIRE(failedTests == 1);

    // Verify only the target GPU received the DCGM_FR_PCIE_WIDTH error
    nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        if (i == targetGpuId)
        {
            REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
            CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_PCIE_WIDTH);
        }
        else
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind: negative test for DCGM_FR_PCIE_GENERATION")
{
    unsigned int gpuCount                                   = 4;
    unsigned int targetGpuId                                = 1;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                                 = gpuCount;
    TestBusGrind bg(42, std::move(mockDcgmOpsUnique), &nvvsCommon);

    for (unsigned int i = 0; i < gpuCount; i++)
    {
        entityInfo->entities[i].entity.entityId      = i;
        entityInfo->entities[i].entity.entityGroupId = DCGM_FE_GPU;
    }

    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    for (unsigned int i = 0; i < gpuCount; i++)
    {
        auto *pd  = new PluginDevice(mockDcgmOps, &mockCudaOps);
        pd->gpuId = i;
        bg.gpu.push_back(pd);
    }

    bg.m_test_links = true;

    std::string subTest = PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED;
    auto &tp            = bg.GetTestParams();
    tp.SetSubTestDouble(subTest, PCIE_STR_MIN_PCI_GEN, 3.0);
    tp.SetSubTestDouble(subTest, PCIE_STR_MIN_PCI_WIDTH, 16.0);

    // Mock field values: targetGpuId gets a low link gen, all others pass
    auto mockRecorder           = std::make_unique<DcgmRecorderMock>();
    dcgmFieldValue_v2 valPstate = {};
    valPstate.value.i64         = NVML_PSTATE_0;
    dcgmFieldValue_v2 valGenOk  = {};
    valGenOk.value.i64          = 4;
    dcgmFieldValue_v2 valGenLow = {};
    valGenLow.value.i64         = 1;
    dcgmFieldValue_v2 valWidth  = {};
    valWidth.value.i64          = 16;

    for (unsigned int i = 0; i < gpuCount; i++)
    {
        mockRecorder->fieldValues[{ i, DCGM_FI_DEV_GPU_PSTATE }]      = { DCGM_ST_OK, valPstate };
        mockRecorder->fieldValues[{ i, DCGM_FI_DEV_PCIE_LINK_WIDTH }] = { DCGM_ST_OK, valWidth };
        // Only targetGpuId gets a link gen below the threshold
        mockRecorder->fieldValues[{ i, DCGM_FI_DEV_PCIE_LINK_GEN }]
            = { DCGM_ST_OK, (i == targetGpuId) ? valGenLow : valGenOk };
    }

    bg.SetDcgmRecorder(std::move(mockRecorder));

    int failedTests = bg_check_pci_link(bg, subTest);
    REQUIRE(failedTests == 1);

    // Verify only the target GPU received the DCGM_FR_PCIE_GENERATION error
    nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        if (i == targetGpuId)
        {
            REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
            CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_PCIE_GENERATION);
        }
        else
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind: negative test for DCGM_FR_LOW_BANDWIDTH (host-device)")
{
    unsigned int gpuCount                                   = 4;
    unsigned int targetGpuId                                = 1;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                                 = gpuCount;
    TestBusGrind bg(42, std::move(mockDcgmOpsUnique), &nvvsCommon);

    for (unsigned int i = 0; i < gpuCount; i++)
    {
        entityInfo->entities[i].entity.entityId      = i;
        entityInfo->entities[i].entity.entityGroupId = DCGM_FE_GPU;
    }

    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    // Populate bg.gpu with minimal PluginDevice objects, no CUDA needed
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        auto *pd  = new PluginDevice(mockDcgmOps, &mockCudaOps);
        pd->gpuId = i;
        bg.gpu.push_back(pd);
    }

    std::string subTest       = PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED;
    double const minBandwidth = 10.0;
    bg.GetTestParams().SetSubTestDouble(subTest, PCIE_STR_MIN_BANDWIDTH, minBandwidth);

    // bandwidthMatrix layout [labelIndex * gpuCount + gpuIndex]; entries start at the threshold
    // (passing, since the check is strict '<') unless an entry is dropped below it.
    std::vector<double> bandwidthMatrix(3 * gpuCount, minBandwidth);

    SECTION("bandwidth below the minimum fails only the affected GPU")
    {
        bandwidthMatrix[0 * gpuCount + targetGpuId] = minBandwidth - 1.0;

        int failures = pcie_check_bandwidth(bg, subTest, bandwidthMatrix, /* pinned */ false);
        REQUIRE(failures == 1);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            if (i == targetGpuId)
            {
                REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
                CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_LOW_BANDWIDTH);
            }
            else
            {
                CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
            }
        }
    }

    SECTION("all GPUs below the minimum each report an error")
    {
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            bandwidthMatrix[0 * gpuCount + i] = minBandwidth - 1.0;
        }

        int failures = pcie_check_bandwidth(bg, subTest, bandwidthMatrix, /* pinned */ false);
        REQUIRE(failures == static_cast<int>(gpuCount));

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
            CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_LOW_BANDWIDTH);
        }
    }

    SECTION("bandwidth exactly at the minimum does not fail")
    {
        int failures = pcie_check_bandwidth(bg, subTest, bandwidthMatrix, /* pinned */ false);
        REQUIRE(failures == 0);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }

    SECTION("all bandwidths above the minimum pass")
    {
        std::fill(bandwidthMatrix.begin(), bandwidthMatrix.end(), minBandwidth + 1.0);

        int failures = pcie_check_bandwidth(bg, subTest, bandwidthMatrix, /* pinned */ false);
        REQUIRE(failures == 0);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind: negative test for DCGM_FR_LOW_BANDWIDTH (child output)")
{
    dcgmHandle_t handle       = {};
    unsigned int gpuCount     = 4;
    unsigned int targetGpuId  = 1;
    double const minBandwidth = 10.0;
    double const lowBw        = minBandwidth - 1.0;
    double const highBw       = minBandwidth + 1.0;
    std::string groupName("bandwidth");

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                                 = gpuCount;
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        entityInfo->entities[i].entity.entityId      = i;
        entityInfo->entities[i].entity.entityGroupId = DCGM_FE_GPU;
    }
    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);
    auto tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBandwidth);
    bg.m_testParameters.reset(tp);

    // Build delimiter-wrapped bw-checker JSON from {gpuId, maxTxBw, maxRxBw} tuples. ProcessChildOutput
    // emits one DCGM_FR_LOW_BANDWIDTH per direction below the minimum (so both-directions-low yields two).
    auto makeChildJson = [](std::vector<std::tuple<unsigned int, double, double>> const &gpus) {
        std::stringstream js;
        js << startDelimiter << "\n{\n    \"GPUs\" : [\n";
        for (size_t k = 0; k < gpus.size(); k++)
        {
            auto const &[gpuId, maxTxBw, maxRxBw] = gpus[k];
            js << "        {\n"
               << "            \"gpuId\" : " << gpuId << ",\n"
               << "            \"maxRxBw\" : " << maxRxBw << ",\n"
               << "            \"maxTxBw\" : " << maxTxBw << ",\n"
               << "            \"maxBidirBw\" : 1000.0\n"
               << "        }" << (k + 1 < gpus.size() ? ",\n" : "\n");
        }
        js << "    ]\n}\n" << endDelimiter;
        return js.str();
    };

    auto runChild = [&](std::string const &json) {
        std::vector<dcgmChildInfo_t> childrenInfo;
        dcgmChildInfo_t childInfo;
        childInfo.stdoutStr     = json;
        childInfo.stderrStr     = "";
        childInfo.pid           = 1234;
        childInfo.readOutputRet = 0;
        childInfo.readErrorRet  = 0;
        childInfo.outputFdIndex = 0;
        childrenInfo.push_back(childInfo);
        return ProcessChildrenOutputs(childrenInfo, bg, groupName);
    };

    SECTION("one GPU below the minimum fails only that GPU")
    {
        std::vector<std::tuple<unsigned int, double, double>> gpus;
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            // only targetGpuId is below the minimum (in the maxRxBw direction)
            gpus.emplace_back(i, highBw, (i == targetGpuId) ? lowBw : highBw);
        }

        unsigned int failures = runChild(makeChildJson(gpus));
        REQUIRE(failures == 1);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            if (i == targetGpuId)
            {
                REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
                CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_LOW_BANDWIDTH);
            }
            else
            {
                CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
            }
        }
    }

    SECTION("a GPU below in both directions reports two low-bandwidth errors")
    {
        std::vector<std::tuple<unsigned int, double, double>> gpus;
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            // targetGpuId is below the minimum in both directions -> both branches fire
            gpus.emplace_back(i, (i == targetGpuId) ? lowBw : highBw, (i == targetGpuId) ? lowBw : highBw);
        }

        unsigned int failures = runChild(makeChildJson(gpus));
        REQUIRE(failures == 2);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        REQUIRE(errorsPerEntity[entityInfo->entities[targetGpuId].entity].size() == 2);
        CHECK(errorsPerEntity[entityInfo->entities[targetGpuId].entity][0].code == DCGM_FR_LOW_BANDWIDTH);
        CHECK(errorsPerEntity[entityInfo->entities[targetGpuId].entity][1].code == DCGM_FR_LOW_BANDWIDTH);
    }

    SECTION("all GPUs below the minimum each report an error")
    {
        std::vector<std::tuple<unsigned int, double, double>> gpus;
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            gpus.emplace_back(i, highBw, lowBw); // each GPU below in one direction
        }

        unsigned int failures = runChild(makeChildJson(gpus));
        REQUIRE(failures == gpuCount);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
            CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_LOW_BANDWIDTH);
        }
    }

    SECTION("all GPUs above the minimum pass")
    {
        std::vector<std::tuple<unsigned int, double, double>> gpus;
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            gpus.emplace_back(i, highBw, highBw);
        }

        unsigned int failures = runChild(makeChildJson(gpus));
        REQUIRE(failures == 0);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind: negative test for DCGM_FR_LOW_BANDWIDTH (concurrent host-device)")
{
    unsigned int gpuCount                                   = 4;
    unsigned int targetGpuId                                = 1;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                                 = gpuCount;
    TestBusGrind bg(42, std::move(mockDcgmOpsUnique), &nvvsCommon);

    for (unsigned int i = 0; i < gpuCount; i++)
    {
        entityInfo->entities[i].entity.entityId      = i;
        entityInfo->entities[i].entity.entityGroupId = DCGM_FE_GPU;
    }

    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    // Populate bg.gpu with minimal PluginDevice objects, no CUDA needed
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        auto *pd  = new PluginDevice(mockDcgmOps, &mockCudaOps);
        pd->gpuId = i;
        bg.gpu.push_back(pd);
    }

    std::string subTest       = PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED;
    double const minBandwidth = 10.0;
    bg.GetTestParams().SetSubTestDouble(subTest, PCIE_STR_MIN_BANDWIDTH, minBandwidth);

    // Layout [labelIndex * gpuCount + gpuIndex] over labels {h2d=0, d2h=1, bidir=2}.
    // Entries start at the threshold (passing, since the check is strict '<').
    constexpr int kNumLabels = 3;
    std::vector<double> bandwidthMatrix(kNumLabels * gpuCount, minBandwidth);

    SECTION("one direction below the minimum fails only the affected GPU once")
    {
        // targetGpuId below the minimum in the h2d direction only
        bandwidthMatrix[0 * gpuCount + targetGpuId] = minBandwidth - 1.0;

        int failures = pcie_check_concurrent_bandwidth(bg, subTest, bandwidthMatrix);
        REQUIRE(failures == 1);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            if (i == targetGpuId)
            {
                REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
                CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_LOW_BANDWIDTH);
            }
            else
            {
                CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
            }
        }
    }

    SECTION("a GPU below in multiple directions reports one error per direction")
    {
        // targetGpuId below the minimum in h2d and d2h -> two errors for that GPU
        bandwidthMatrix[0 * gpuCount + targetGpuId] = minBandwidth - 1.0;
        bandwidthMatrix[1 * gpuCount + targetGpuId] = minBandwidth - 1.0;

        int failures = pcie_check_concurrent_bandwidth(bg, subTest, bandwidthMatrix);
        REQUIRE(failures == 2);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        REQUIRE(errorsPerEntity[entityInfo->entities[targetGpuId].entity].size() == 2);
        CHECK(errorsPerEntity[entityInfo->entities[targetGpuId].entity][0].code == DCGM_FR_LOW_BANDWIDTH);
        CHECK(errorsPerEntity[entityInfo->entities[targetGpuId].entity][1].code == DCGM_FR_LOW_BANDWIDTH);
    }

    SECTION("all GPUs below the minimum in one direction each report an error")
    {
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            bandwidthMatrix[2 * gpuCount + i] = minBandwidth - 1.0; // bidir row
        }

        int failures = pcie_check_concurrent_bandwidth(bg, subTest, bandwidthMatrix);
        REQUIRE(failures == static_cast<int>(gpuCount));

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
            CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_LOW_BANDWIDTH);
        }
    }

    SECTION("bandwidth exactly at the minimum does not fail (strict '<')")
    {
        int failures = pcie_check_concurrent_bandwidth(bg, subTest, bandwidthMatrix);
        REQUIRE(failures == 0);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }

    SECTION("all bandwidths above the minimum pass")
    {
        std::fill(bandwidthMatrix.begin(), bandwidthMatrix.end(), minBandwidth + 1.0);

        int failures = pcie_check_concurrent_bandwidth(bg, subTest, bandwidthMatrix);
        REQUIRE(failures == 0);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind: negative test for DCGM_FR_HIGH_LATENCY")
{
    unsigned int gpuCount                                   = 4;
    unsigned int targetGpuId                                = 1;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                                 = gpuCount;
    TestBusGrind bg(42, std::move(mockDcgmOpsUnique), &nvvsCommon);

    for (unsigned int i = 0; i < gpuCount; i++)
    {
        entityInfo->entities[i].entity.entityId      = i;
        entityInfo->entities[i].entity.entityGroupId = DCGM_FE_GPU;
    }

    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    // Populate bg.gpu with minimal PluginDevice objects, no CUDA needed
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        auto *pd  = new PluginDevice(mockDcgmOps, &mockCudaOps);
        pd->gpuId = i;
        bg.gpu.push_back(pd);
    }

    std::string subTest     = PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED;
    double const maxLatency = 10.0;
    bg.GetTestParams().SetSubTestDouble(subTest, PCIE_STR_MAX_LATENCY, maxLatency);

    // latencyMatrix layout: [labelIndex * gpuCount + gpuIndex], 3 labels (d2h, h2d, bidir)
    // All entries start at the threshold, the check is strict '>', so nothing fails until bumped above
    std::vector<double> latencyMatrix(3 * gpuCount, maxLatency);

    SECTION("latency above the maximum fails only the affected GPU")
    {
        latencyMatrix[0 * gpuCount + targetGpuId] = maxLatency + 1.0;

        int failures = pcie_check_latency(bg, subTest, latencyMatrix, /* pinned */ false);
        REQUIRE(failures == 1);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            if (i == targetGpuId)
            {
                REQUIRE(errorsPerEntity[entityInfo->entities[i].entity].size() == 1);
                CHECK(errorsPerEntity[entityInfo->entities[i].entity][0].code == DCGM_FR_HIGH_LATENCY);
            }
            else
            {
                CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
            }
        }
    }

    SECTION("latency exactly at the maximum does not fail")
    {
        int failures = pcie_check_latency(bg, subTest, latencyMatrix, /* pinned */ false);
        REQUIRE(failures == 0);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }

    SECTION("all latencies below the maximum pass")
    {
        std::fill(latencyMatrix.begin(), latencyMatrix.end(), maxLatency - 1.0);

        int failures = pcie_check_latency(bg, subTest, latencyMatrix, /* pinned */ false);
        REQUIRE(failures == 0);

        nvvsPluginEntityErrors_t errorsPerEntity = bg.GetEntityErrors(bg.GetPcieTestName());
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            CHECK(errorsPerEntity[entityInfo->entities[i].entity].size() == 0);
        }
    }
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind::Go: negative test for DCGM_FR_TEST_DISABLED")
{
    auto entityInfo                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                      = 1;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo->entities[0].auxField.gpu.status  = DcgmEntityStatusOk;
    entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
    SafeCopyTo(entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciBusId, "0000:01:00.0");
    mockDcgmOps->m_groupManagementState.addReturn = DCGM_ST_OK;
    dcgmHandle_t handle                           = mockDcgmOps->m_handleId;

    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);

    dcgmFieldValue_v2 cudaComputeVal {};
    cudaComputeVal.value.i64 = 0;

    auto mockRecorder                                                     = std::make_unique<DcgmRecorderMock>();
    mockRecorder->fieldValues[{ 0, DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY }] = { DCGM_ST_OK, cudaComputeVal };
    bg.SetDcgmRecorder(std::move(mockRecorder));

    dcgmDiagPluginTestParameter_t param {};
    SafeCopyTo(param.parameterName, PCIE_STR_IS_ALLOWED);
    SafeCopyTo(param.parameterValue, "False");
    param.type = DcgmPluginParamBool;

    bg.Go(bg.GetPcieTestName(), entityInfo.get(), 1, &param);
    CHECK(bg.GetResult(bg.GetPcieTestName()) == NVVS_RESULT_SKIP);

    auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(bg.GetResults(bg.GetPcieTestName(), entityResults.get()) == DCGM_ST_OK);

    CHECK(entityResults->numErrors == 1);
    CHECK(entityResults->errors[0].code == DCGM_FR_DCGM_API);
    CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults->errors[0].entity.entityId == 0);
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind main_entry: negative test for DCGM_FR_ABORTED")
{
    dcgmHandle_t handle = mockDcgmOps->m_handleId;
    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);

    auto entityInfo                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                      = 1;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo->entities[0].entity.entityId      = 0;

    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    bg.test_broken_p2p = false;
    bg.test_1          = true;
    bg.test_2 = bg.test_3 = bg.test_4 = bg.test_5 = bg.test_6 = bg.test_7 = false;
    bg.test_8 = bg.test_9 = bg.test_10 = bg.test_11 = bg.test_12 = bg.test_13 = bg.test_14 = false;
    bg.test_pinned                                                                         = true;
    bg.check_errors                                                                        = false;

    ScopedAtomic stopGuard(main_should_stop, 1);

    CHECK(main_entry(&bg, *entityInfo, mockDcgmOps, handle, &mockCudaOps) == 0);
    CHECK(bg.GetResult(bg.GetPcieTestName()) == NVVS_RESULT_SKIP);

    auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(bg.GetResults(bg.GetPcieTestName(), entityResults.get()) == DCGM_ST_OK);

    // The abort path may also record incidental setup/cleanup errors from non-injected dependencies,
    // so we only assert that DCGM_FR_ABORTED is present and correctly attributed.
    REQUIRE(entityResults->numErrors >= 1);
    bool sawAborted = false;
    for (unsigned int i = 0; i < entityResults->numErrors; ++i)
    {
        if (entityResults->errors[i].code == DCGM_FR_ABORTED)
        {
            sawAborted = true;
            CHECK(entityResults->errors[i].entity.entityGroupId == 0);
            CHECK(entityResults->errors[i].entity.entityId == 0);
        }
    }
    CHECK(sawAborted);
}

TEST_CASE_METHOD(PciePluginTestFixture, "BusGrind::Go: negative test for DCGM_FR_ABORTED")
{
    auto entityInfo                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfo->numEntities                      = 1;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo->entities[0].auxField.gpu.status  = DcgmEntityStatusOk;
    entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
    snprintf(entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciBusId,
             sizeof(entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciBusId),
             "0000:01:00.0");
    dcgmHandle_t handle = mockDcgmOps->m_handleId;
    TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);

    dcgmFieldValue_v2 cudaComputeVal {};
    cudaComputeVal.value.i64 = 0;

    auto mockRecorder                                                     = std::make_unique<DcgmRecorderMock>();
    mockRecorder->fieldValues[{ 0, DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY }] = { DCGM_ST_OK, cudaComputeVal };
    bg.SetDcgmRecorder(std::move(mockRecorder));

    dcgmDiagPluginTestParameter_t param {};
    snprintf(param.parameterName, sizeof(param.parameterName), "%s", PCIE_STR_IS_ALLOWED);
    snprintf(param.parameterValue, sizeof(param.parameterValue), "True");
    param.type = DcgmPluginParamBool;

    ScopedAtomic stopGuard(main_should_stop, 1);

    bg.Go(bg.GetPcieTestName(), entityInfo.get(), 1, &param);

    CHECK(bg.GetResult(bg.GetPcieTestName()) == NVVS_RESULT_SKIP);

    auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(bg.GetResults(bg.GetPcieTestName(), entityResults.get()) == DCGM_ST_OK);

    // The abort path currently also records setup/telemetry errors from non-injected dependencies.
    // Isolating only DCGM_FR_ABORTED would require a broad production refactor, so we assert its presence.
    REQUIRE(entityResults->numErrors >= 1);
    bool sawAborted = false;
    for (unsigned int i = 0; i < entityResults->numErrors; ++i)
    {
        if (entityResults->errors[i].code == DCGM_FR_ABORTED)
        {
            sawAborted = true;
            CHECK(entityResults->errors[i].entity.entityGroupId == 0);
            CHECK(entityResults->errors[i].entity.entityId == 0);
        }
    }
    CHECK(sawAborted);
}

SCENARIO_METHOD(PciePluginTestFixture, "Pcie: GPU list membership ignores non-GPU entities")
{
    GIVEN("an entity list with a switch sharing an ID with the searched GPU")
    {
        dcgmDiagPluginEntityList_v1 entityInfo {};
        entityInfo.numEntities                      = 2;
        entityInfo.entities[0].entity.entityId      = 7;
        entityInfo.entities[0].entity.entityGroupId = DCGM_FE_SWITCH;
        entityInfo.entities[1].entity.entityId      = 8;
        entityInfo.entities[1].entity.entityGroupId = DCGM_FE_GPU;

        WHEN("membership is checked")
        {
            bool const switchMatch = pcie_gpu_id_in_list(7, entityInfo);
            bool const gpuMatch    = pcie_gpu_id_in_list(8, entityInfo);

            THEN("only GPU entities are considered members")
            {
                CHECK_FALSE(switchMatch);
                CHECK(gpuMatch);
            }
        }
    }
}

SCENARIO_METHOD(PciePluginTestFixture, "Pcie: dispatch and initialization paths avoid CUDA work when skipped")
{
    GIVEN("a PCIe plugin")
    {
        dcgmHandle_t handle = mockDcgmOps->m_handleId;
        TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);

        WHEN("Go is called with an unknown test name")
        {
            auto entityInfo                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
            entityInfo->numEntities                      = 1;
            entityInfo->entities[0].entity.entityId      = 0;
            entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
            entityInfo->entities[0].auxField.gpu.status  = DcgmEntityStatusFake;
            entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
            SafeCopyTo(entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciBusId, "0000:01:00.0");

            bg.Go("not_pcie", entityInfo.get(), 0, nullptr);

            THEN("the PCIe test is not initialized")
            {
                CHECK_THROWS_AS(bg.GetGpuResults(bg.GetPcieTestName()), std::out_of_range);
            }
        }

        WHEN("Go is called with null entity info")
        {
            bg.Go(bg.GetPcieTestName(), nullptr, 0, nullptr);

            THEN("the PCIe test is not initialized")
            {
                CHECK_THROWS_AS(bg.GetGpuResults(bg.GetPcieTestName()), std::out_of_range);
            }
        }

        WHEN("Go is called with fake GPUs")
        {
            auto entityInfo                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
            entityInfo->numEntities                      = 1;
            entityInfo->entities[0].entity.entityId      = 0;
            entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
            entityInfo->entities[0].auxField.gpu.status  = DcgmEntityStatusFake;
            entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
            SafeCopyTo(entityInfo->entities[0].auxField.gpu.attributes.identifiers.pciBusId, "0000:01:00.0");

            bg.Go(bg.GetPcieTestName(), entityInfo.get(), 0, nullptr);

            THEN("the fake-GPU short-circuit marks the test as passing")
            {
                REQUIRE(bg.GetGpuResults(bg.GetPcieTestName()).at(0) == NVVS_RESULT_PASS);
                CHECK(bg.GetResult(bg.GetPcieTestName()) == NVVS_RESULT_PASS);
            }
        }
    }
}

SCENARIO_METHOD(PciePluginTestFixture, "Pcie: Init rejects null input and skips fake GPUs")
{
    GIVEN("a PCIe plugin")
    {
        dcgmHandle_t handle = mockDcgmOps->m_handleId;
        TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);

        WHEN("Init is called without entity info")
        {
            bool initialized = bg.Init(nullptr);

            THEN("initialization fails without creating devices")
            {
                CHECK_FALSE(initialized);
                CHECK(bg.gpu.empty());
            }
        }

        WHEN("Init is called with fake GPUs")
        {
            auto entityInfo         = std::make_unique<dcgmDiagPluginEntityList_v1>();
            entityInfo->numEntities = 2;
            for (unsigned int i = 0; i < entityInfo->numEntities; ++i)
            {
                entityInfo->entities[i].entity.entityId                                 = i;
                entityInfo->entities[i].entity.entityGroupId                            = DCGM_FE_GPU;
                entityInfo->entities[i].auxField.gpu.status                             = DcgmEntityStatusFake;
                entityInfo->entities[i].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
                snprintf(entityInfo->entities[i].auxField.gpu.attributes.identifiers.pciBusId,
                         sizeof(entityInfo->entities[i].auxField.gpu.attributes.identifiers.pciBusId),
                         "0000:%02x:00.0",
                         i + 1);
            }

            bool initialized = bg.Init(entityInfo.get());

            THEN("initialization succeeds through the fake-GPU path without creating CUDA devices")
            {
                CHECK(initialized);
                CHECK(bg.gpu.empty());
                CHECK(bg.GetGpuResults(bg.GetPcieTestName()).size() == 2);
            }
        }
    }
}

SCENARIO_METHOD(PciePluginTestFixture, "PcieMain: bg_should_stop converts the global stop flag into a skipped test")
{
    GIVEN("an initialized PCIe plugin")
    {
        auto entityInfo                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
        entityInfo->numEntities                      = 1;
        entityInfo->entities[0].entity.entityId      = 0;
        entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;

        dcgmHandle_t handle = mockDcgmOps->m_handleId;
        TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
        bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

        WHEN("the global stop flag is clear")
        {
            ScopedAtomic stopGuard(main_should_stop, 0);

            int const result = bg_should_stop(&bg);

            THEN("no abort result is recorded")
            {
                CHECK(result == 0);

                auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(bg.GetResults(bg.GetPcieTestName(), entityResults.get()) == DCGM_ST_OK);
                CHECK(entityResults->numErrors == 0);
            }
        }

        WHEN("the global stop flag is set")
        {
            ScopedAtomic stopGuard(main_should_stop, 1);

            int const result = bg_should_stop(&bg);

            THEN("an aborted error is recorded and the test is skipped")
            {
                CHECK(result == 1);
                CHECK(bg.GetResult(bg.GetPcieTestName()) == NVVS_RESULT_SKIP);

                auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(bg.GetResults(bg.GetPcieTestName(), entityResults.get()) == DCGM_ST_OK);
                REQUIRE(entityResults->numErrors == 1);
                CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);
                CHECK(entityResults->errors[0].code == DCGM_FR_ABORTED);
            }
        }
    }
}

SCENARIO_METHOD(PciePluginTestFixture, "PcieMain: main_entry validates required inputs")
{
    GIVEN("a GPU entity list")
    {
        dcgmDiagPluginEntityList_v1 entityInfo {};
        entityInfo.numEntities                      = 1;
        entityInfo.entities[0].entity.entityId      = 0;
        entityInfo.entities[0].entity.entityGroupId = DCGM_FE_GPU;

        dcgmHandle_t handle = mockDcgmOps->m_handleId;

        WHEN("main_entry is called without a BusGrind instance")
        {
            int ret = main_entry(nullptr, entityInfo, mockDcgmOps, handle, &mockCudaOps);

            THEN("the call fails before dereferencing the plugin")
            {
                CHECK(ret == 1);
            }
        }
    }
}

SCENARIO_METHOD(PciePluginTestFixture, "PcieMain verbose metric helpers format GPU-local latency and bandwidth")
{
    GIVEN("an initialized PCIe plugin with two GPU result slots")
    {
        auto entityInfo                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
        entityInfo->numEntities                      = 2;
        entityInfo->entities[0].entity.entityId      = 0;
        entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
        entityInfo->entities[1].entity.entityId      = 1;
        entityInfo->entities[1].entity.entityGroupId = DCGM_FE_GPU;

        dcgmHandle_t handle = mockDcgmOps->m_handleId;
        TestBusGrind bg(handle, std::move(mockDcgmOpsUnique), &nvvsCommon);
        bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

        WHEN("latency values are reported for every supported direction")
        {
            addLatencyInfo(&bg, 1, "bidir", 12.34567);
            addLatencyInfo(&bg, 1, "d2h", 0.12549);
            addLatencyInfo(&bg, 1, "h2d", 999.9999);

            THEN("the GPU verbose messages use stable labels and three decimal places")
            {
                auto const &verboseInfo = bg.GetGpuVerboseInfo(bg.GetPcieTestName()).at(1);
                REQUIRE(verboseInfo.size() == 3);
                CHECK(verboseInfo[0] == "bidirectional latency:\t\t12.346 us");
                CHECK(verboseInfo[1] == "GPU to Host latency:\t\t0.125 us");
                CHECK(verboseInfo[2] == "Host to GPU latency:\t\t1000.000 us");
                CHECK(bg.GetGpuVerboseInfo(bg.GetPcieTestName()).at(0).empty());
            }
        }

        WHEN("bandwidth values are reported for every supported direction")
        {
            addBandwidthInfo(bg, 0, "bidir", 48.125);
            addBandwidthInfo(bg, 0, "d2h", 7.999);
            addBandwidthInfo(bg, 0, "h2d", 101.004);

            THEN("the GPU verbose messages use stable labels and two decimal places")
            {
                auto const &verboseInfo = bg.GetGpuVerboseInfo(bg.GetPcieTestName()).at(0);
                REQUIRE(verboseInfo.size() == 3);
                CHECK(verboseInfo[0] == "bidirectional bandwidth:\t48.12 GB/s");
                CHECK(verboseInfo[1] == "GPU to Host bandwidth:\t\t8.00 GB/s");
                CHECK(verboseInfo[2] == "Host to GPU bandwidth:\t\t101.00 GB/s");
                CHECK(bg.GetGpuVerboseInfo(bg.GetPcieTestName()).at(1).empty());
            }
        }
    }
}

TEST_CASE("BusGrind::ElapsedMinutes subtracts before truncating to whole minutes")
{
    using std::chrono::microseconds;
    using std::chrono::seconds;

    struct TestCase
    {
        std::string name;
        microseconds start;
        microseconds end;
        unsigned int expected;
    };

    auto testCase = GENERATE(
        TestCase { "sub-minute interval straddling a boundary yields 0, not 1",
                   microseconds { 119'900'000 },
                   microseconds { 121'000'000 },
                   0 },
        TestCase {
            "exactly one minute apart yields 1", microseconds { seconds { 30 } }, microseconds { seconds { 90 } }, 1 },
        TestCase { "several whole minutes apart yields the full count",
                   microseconds { seconds { 0 } },
                   microseconds { seconds { 300 } },
                   5 },
        TestCase {
            "identical timestamps yield 0", microseconds { seconds { 42 } }, microseconds { seconds { 42 } }, 0 },
        TestCase { "reversed interval clamps to 0 rather than wrapping",
                   microseconds { seconds { 300 } },
                   microseconds { seconds { 60 } },
                   0 },
        TestCase { "interval just under two minutes truncates down to 1",
                   microseconds { 0 },
                   microseconds { 119'000'000 },
                   1 });

    DYNAMIC_SECTION(testCase.name)
    {
        CHECK(BusGrind::ElapsedMinutes(testCase.start, testCase.end) == testCase.expected);
    }
}
