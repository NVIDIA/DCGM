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
#include <Pcie.h>
#include <PcieMain.h>

#include "dcgm_fields.h"
#include <UniquePtrUtil.h>
#include <catch2/catch_all.hpp>
#include <sstream>
#include <string_view>
#include <tuple>
#include <vector>

unsigned int failingGpuId    = DCGM_MAX_NUM_DEVICES;
unsigned int failingSwitchId = DCGM_MAX_NUM_SWITCHES;

TEST_CASE("Pcie: pcie_gpu_id_in_list")
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

/*
 * Spoof this dcgmlib function so we can control program execution
 */
dcgmReturn_t dcgmGetNvLinkLinkStatus(dcgmHandle_t /* handle */, dcgmNvLinkStatus_v4 *linkStatus)
{
    memset(linkStatus, 0, sizeof(*linkStatus));
    linkStatus->numGpus                    = 2;
    linkStatus->gpus[0].entityId           = 0;
    linkStatus->gpus[0].linkState[0]       = DcgmNvLinkLinkStateUp;
    linkStatus->gpus[1].entityId           = 1;
    linkStatus->gpus[1].linkState[0]       = DcgmNvLinkLinkStateUp;
    linkStatus->numNvSwitches              = 1;
    linkStatus->nvSwitches[0].entityId     = 0;
    linkStatus->nvSwitches[0].linkState[0] = DcgmNvLinkLinkStateUp;

    if (failingGpuId < DCGM_MAX_NUM_DEVICES)
    {
        linkStatus->gpus[0].entityId     = failingGpuId;
        linkStatus->gpus[0].linkState[0] = DcgmNvLinkLinkStateDown;
    }

    if (failingSwitchId < DCGM_MAX_NUM_SWITCHES)
    {
        linkStatus->nvSwitches[0].entityId     = failingSwitchId;
        linkStatus->nvSwitches[0].linkState[0] = DcgmNvLinkLinkStateDown;
    }

    return DCGM_ST_OK;
}

TEST_CASE("Pcie: pcie_check_nvlink_status_expected")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmHandle_t handle                                     = {};
    entityInfo->numEntities                                 = 2;
    const unsigned int ourGpuId                             = 1;
    entityInfo->entities[0].entity.entityId                 = ourGpuId;
    entityInfo->entities[0].entity.entityGroupId            = DCGM_FE_GPU;
    entityInfo->entities[1].entity.entityId                 = 2;
    entityInfo->entities[1].entity.entityGroupId            = DCGM_FE_GPU;
    auto entityResults                                      = MakeUniqueZero<dcgmDiagEntityResults_v2>();

    BusGrind bg1 = BusGrind(handle);
    bg1.InitializeForEntityList(bg1.GetPcieTestName(), *entityInfo);
    bg1.m_gpuNvlinksExpectedUp = 1;
    pcie_check_nvlink_status(&bg1, *entityInfo);
    bg1.GetResults(bg1.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 0);

    memset(entityResults.get(), 0, sizeof(*entityResults));
    BusGrind bg2 = BusGrind(handle);
    bg2.InitializeForEntityList(bg2.GetPcieTestName(), *entityInfo);
    bg2.m_gpuNvlinksExpectedUp = 2;
    pcie_check_nvlink_status(&bg2, *entityInfo);
    bg2.GetResults(bg2.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 1);

    memset(entityResults.get(), 0, sizeof(*entityResults));
    BusGrind bg3 = BusGrind(handle);
    bg3.InitializeForEntityList(bg3.GetPcieTestName(), *entityInfo);
    bg3.m_nvSwitchNvlinksExpectedUp = 1;
    pcie_check_nvlink_status(&bg3, *entityInfo);
    bg3.GetResults(bg3.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 0);

    memset(entityResults.get(), 0, sizeof(*entityResults));
    BusGrind bg4 = BusGrind(handle);
    bg4.InitializeForEntityList(bg4.GetPcieTestName(), *entityInfo);
    bg4.m_nvSwitchNvlinksExpectedUp = 2;
    pcie_check_nvlink_status(&bg4, *entityInfo);
    bg4.GetResults(bg4.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 1);
}

#define BWC_JSON_GPUS   "GPUs"
#define BWC_JSON_GPU_ID "gpuId"
#define BWC_JSON_MAX_BW "maxBw"
#define BWC_JSON_BW     "bandwidths"
#define BWC_JSON_ERROR  "error"
#define BWC_JSON_ERRORS "errors"

std::string testJson1 = R""(
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
)"";
std::string testJson2 = R""(
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
)"";
std::string testJson3 = R""(
{
    "GPUs" : [
        {
            "gpuId" : 4,
            "error" : "Got rocked and couldn't launch stuff"
        }
    ]
}
)"";

TEST_CASE("Pcie: ProcessChildrenOutputs")
{
    dcgmHandle_t handle = {};

    auto entityResults                                      = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    BusGrind bg(handle);
    entityInfo->numEntities                      = 1;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;

    std::vector<dcgmChildInfo_t> childrenInfo;
    dcgmChildInfo_t childInfo;
    childInfo.stdoutStr     = testJson3;
    childInfo.pid           = 23981;
    childInfo.readOutputRet = 12;
    childInfo.outputFdIndex = 0;
    childrenInfo.push_back(childInfo);

    double minBw = 10.0;
    std::string groupName("bobby");
    auto tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);
    bg.m_testParameters = tp;
    unsigned int ret    = ProcessChildrenOutputs(childrenInfo, bg, groupName);
    CHECK(ret == 1); // we should have one failed test
    bg.GetResults(bg.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 1);
    CHECK(entityResults->errors[0].entity.entityId == 4);
    CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);

    BusGrind bg2(handle);
    bg2.InitializeForEntityList(bg2.GetPcieTestName(), *entityInfo);
    // We can discard the old pointer as its ownership was passed to bg
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg2.m_testParameters = tp;
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

    BusGrind bg3(handle);
    bg3.InitializeForEntityList(bg3.GetPcieTestName(), *entityInfo);
    // We can discard the old pointer as its ownership was passed to bg2
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg3.m_testParameters = tp;
    memset(entityResults.get(), 0, sizeof(*entityResults));
    childrenInfo.clear();
    childInfo.stdoutStr = testJson2;
    childrenInfo.push_back(childInfo);
    ret = ProcessChildrenOutputs(childrenInfo, bg3, groupName);
    CHECK(ret == 0); // we shouldn't have any failures
    bg3.GetResults(bg3.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 0);

    BusGrind bg4(handle);
    bg4.InitializeForEntityList(bg4.GetPcieTestName(), *entityInfo);
    // We can discard the old pointer as its ownership was passed to bg2
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg4.m_testParameters = tp;
    memset(entityResults.get(), 0, sizeof(*entityResults));
    childrenInfo.clear();
    childInfo.stdoutStr = "This isn't json";
    childrenInfo.push_back(std::move(childInfo));
    ret = ProcessChildrenOutputs(childrenInfo, bg4, groupName);
    CHECK(ret == 1);
    bg4.GetResults(bg4.GetPcieTestName(), entityResults.get());
    CHECK(entityResults->numErrors == 1);                 // we get one failure for bad json
    CHECK(entityResults->errors[0].entity.entityId == 0); // this error isn't tied to a GPU id
    CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_NONE);
}


dcgmReturn_t globalWatchError;
dcgmReturn_t globalGroupCreateError;
dcgmReturn_t globalFieldGroupCreateError;

// Mock the dcgm functions used by DcgmGroup
dcgmReturn_t dcgmGroupCreate(dcgmHandle_t, dcgmGroupType_t, const char *, dcgmGpuGrp_t *groupId)
{
    *groupId = 1;
    return globalGroupCreateError;
}

dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t, dcgmGpuGrp_t)
{
    return DCGM_ST_OK;
}

dcgmReturn_t dcgmGroupAddDevice(dcgmHandle_t, dcgmGpuGrp_t, unsigned int)
{
    return DCGM_ST_OK;
}

dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t, int, const unsigned short *, const char *, dcgmFieldGrp_t *groupId)
{
    *groupId = 1;
    return globalFieldGroupCreateError;
}

dcgmReturn_t dcgmFieldGroupDestroy(dcgmHandle_t, dcgmFieldGrp_t)
{
    return DCGM_ST_OK;
}

dcgmReturn_t dcgmWatchFields(dcgmHandle_t, dcgmGpuGrp_t, dcgmFieldGrp_t, long long, double, int)
{
    return globalWatchError;
}

dcgmReturn_t dcgmUnwatchFields(dcgmHandle_t, dcgmGpuGrp_t, dcgmFieldGrp_t)
{
    return DCGM_ST_OK;
}

static void SetUp()
{
    globalWatchError            = DCGM_ST_OK;
    globalGroupCreateError      = DCGM_ST_OK;
    globalFieldGroupCreateError = DCGM_ST_OK;
}

TEST_CASE("Pcie: StartDcgmGroupWatch")
{
    dcgmHandle_t handle = 2;
    BusGrind bg(handle);
    std::vector<unsigned short> fieldIds = { DCGM_FI_PROF_NVLINK_TX_BYTES };
    std::vector<unsigned int> gpuIds     = { 0, 1, 2 };

    SECTION("Happy path returns valid DcgmGroup ptr")
    {
        SetUp();
        std::unique_ptr<DcgmGroup> dcgmGroupPtr = StartDcgmGroupWatch(&bg, fieldIds, gpuIds);
        REQUIRE(dcgmGroupPtr.get() != nullptr);
    }
    SECTION("Field group creation error returns nullptr")
    {
        SetUp();
        globalFieldGroupCreateError             = DCGM_ST_BADPARAM;
        std::unique_ptr<DcgmGroup> dcgmGroupPtr = StartDcgmGroupWatch(&bg, fieldIds, gpuIds);
        REQUIRE(dcgmGroupPtr.get() == nullptr);
    }
    SECTION("Group creation error returns nullptr")
    {
        SetUp();
        globalGroupCreateError                  = DCGM_ST_BADPARAM;
        std::unique_ptr<DcgmGroup> dcgmGroupPtr = StartDcgmGroupWatch(&bg, fieldIds, gpuIds);
        REQUIRE(dcgmGroupPtr.get() == nullptr);
    }
    SECTION("Watch field error returns nullptr")
    {
        SetUp();
        globalWatchError                        = DCGM_ST_BADPARAM;
        std::unique_ptr<DcgmGroup> dcgmGroupPtr = StartDcgmGroupWatch(&bg, fieldIds, gpuIds);
        REQUIRE(dcgmGroupPtr.get() == nullptr);
    }
}

/**
 * @brief BusGrind test interface
 * friend of @ref BusGrind
 */
class BusGrindTest : public BusGrind
{
public:
    BusGrindTest(dcgmHandle_t h)
        : BusGrind(h)
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
};

TEST_CASE("Pcie: SetCopySize")
{
    constexpr auto BEFORE_BLACKWELL = 9;
    constexpr auto COMPAT_BLACKWELL = 10;

    static_assert(PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY != PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
    static_assert(PCIE_HOPPER_AND_BEFORE_DEFAULT_BROKEN_P2P_SIZE != PCIE_BLACKWELL_DEFAULT_BROKEN_P2P_SIZE);

    dcgmHandle_t handle = 42;
    BusGrindTest bg(handle);

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
    dcgmReturn_t GetCurrentFieldValue(unsigned int, unsigned short, dcgmFieldValue_v2 &, unsigned int) override
    {
        return DCGM_ST_UNKNOWN_FIELD;
    }
};

TEST_CASE("BusGrind: SetCudaCapabilityInfo, negative test for DCGM_FR_DCGM_API")
{
    dcgmHandle_t handle                                     = 42;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();
    BusGrindTest bg(handle);
    entityInfo->numEntities                      = 2;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo->entities[1].entity.entityId      = 1;
    entityInfo->entities[1].entity.entityGroupId = DCGM_FE_GPU;
    bg.InitializeForEntityList(bg.GetPcieTestName(), *entityInfo);

    auto mockRecorder = std::make_unique<DcgmRecorderMock>();
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
