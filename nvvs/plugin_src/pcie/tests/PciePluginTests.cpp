/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <Pcie.h>
#include <PcieMain.h>
#include <sstream>

unsigned int failingGpuId    = DCGM_MAX_NUM_DEVICES;
unsigned int failingSwitchId = DCGM_MAX_NUM_SWITCHES;

TEST_CASE("Pcie: pcie_gpu_id_in_list")
{
    dcgmDiagPluginGpuList_t gpuInfo = {};
    gpuInfo.numGpus                 = 2;
    gpuInfo.gpus[0].gpuId           = 1;
    gpuInfo.gpus[1].gpuId           = 2;

    CHECK(pcie_gpu_id_in_list(0, gpuInfo) == false);
    for (unsigned int i = 3; i < 10; i++)
    {
        CHECK(pcie_gpu_id_in_list(i, gpuInfo) == false);
    }

    for (unsigned int i = 0; i < gpuInfo.numGpus; i++)
    {
        CHECK(pcie_gpu_id_in_list(gpuInfo.gpus[i].gpuId, gpuInfo) == true);
    }
}

/*
 * Spoof this dcgmlib function so we can control program execution
 */
dcgmReturn_t dcgmGetNvLinkLinkStatus(dcgmHandle_t handle, dcgmNvLinkStatus_v3 *linkStatus)
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
    dcgmHandle_t handle             = {};
    dcgmDiagPluginGpuList_t gpuInfo = {};
    gpuInfo.numGpus                 = 2;
    const unsigned int ourGpuId     = 1;
    gpuInfo.gpus[0].gpuId           = ourGpuId;
    gpuInfo.gpus[1].gpuId           = 2;

    BusGrind bg1               = BusGrind(handle, &gpuInfo);
    dcgmDiagResults_t results  = {};
    bg1.m_gpuNvlinksExpectedUp = 1;
    pcie_check_nvlink_status(&bg1, gpuInfo);
    bg1.GetResults(&results);
    CHECK(results.numErrors == 0);

    memset(&results, 0, sizeof(results));
    BusGrind bg2               = BusGrind(handle, &gpuInfo);
    bg2.m_gpuNvlinksExpectedUp = 2;
    pcie_check_nvlink_status(&bg2, gpuInfo);
    bg2.GetResults(&results);
    CHECK(results.numErrors == 1);

    memset(&results, 0, sizeof(results));
    BusGrind bg3                    = BusGrind(handle, &gpuInfo);
    bg3.m_nvSwitchNvlinksExpectedUp = 1;
    pcie_check_nvlink_status(&bg3, gpuInfo);
    bg3.GetResults(&results);
    CHECK(results.numErrors == 0);

    memset(&results, 0, sizeof(results));
    BusGrind bg4                    = BusGrind(handle, &gpuInfo);
    bg4.m_nvSwitchNvlinksExpectedUp = 2;
    pcie_check_nvlink_status(&bg4, gpuInfo);
    bg4.GetResults(&results);
    CHECK(results.numErrors == 1);
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
    dcgmHandle_t handle             = {};
    dcgmDiagPluginGpuList_t gpuInfo = {};
    gpuInfo.numGpus                 = 1;
    const unsigned int ourGpuId     = 2;
    gpuInfo.gpus[0].gpuId           = ourGpuId;

    BusGrind bg(handle, &gpuInfo);
    dcgmDiagResults_t results = {};

    std::vector<dcgmChildInfo_t> childrenInfo;
    dcgmChildInfo_t childInfo;
    childInfo.stdoutStr     = testJson3;
    childInfo.pid           = 23981;
    childInfo.readOutputRet = 12;
    childrenInfo.push_back(childInfo);

    double minBw = 10.0;
    std::string groupName("bobby");
    auto tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg.m_testParameters = tp;
    unsigned int ret    = ProcessChildrenOutputs(childrenInfo, bg, groupName);
    CHECK(ret == 1); // we should have one failed test
    bg.GetResults(&results);
    CHECK(results.numErrors == 1);
    CHECK(results.errors[0].gpuId == 4);

    BusGrind bg2(handle, &gpuInfo);
    // We can discard the old pointer as its ownership was passed to bg
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg2.m_testParameters = tp;
    memset(&results, 0, sizeof(results));
    childrenInfo.clear();
    childInfo.stdoutStr     = testJson1;
    childInfo.readOutputRet = 0;
    childrenInfo.push_back(childInfo);
    ret = ProcessChildrenOutputs(childrenInfo, bg2, groupName);
    CHECK(ret == 1); // we should have one failed test due to low bandwidth
    bg2.GetResults(&results);
    CHECK(results.numErrors == 1);
    CHECK(results.errors[0].gpuId == 7);

    BusGrind bg3(handle, &gpuInfo);
    // We can discard the old pointer as its ownership was passed to bg2
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg3.m_testParameters = tp;
    memset(&results, 0, sizeof(results));
    childrenInfo.clear();
    childInfo.stdoutStr = testJson2;
    childrenInfo.push_back(childInfo);
    ret = ProcessChildrenOutputs(childrenInfo, bg3, groupName);
    CHECK(ret == 0); // we shouldn't have any failures
    bg3.GetResults(&results);
    CHECK(results.numErrors == 0);

    BusGrind bg4(handle, &gpuInfo);
    // We can discard the old pointer as its ownership was passed to bg2
    tp = new TestParameters();
    tp->AddSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH, minBw);
    bg4.m_testParameters = tp;
    memset(&results, 0, sizeof(results));
    childrenInfo.clear();
    childInfo.stdoutStr = "This isn't json";
    childrenInfo.push_back(childInfo);
    ret = ProcessChildrenOutputs(childrenInfo, bg4, groupName);
    CHECK(ret == 1);
    bg4.GetResults(&results);
    CHECK(results.numErrors == 1);        // we get one failure for bad json
    CHECK(results.errors[0].gpuId == -1); // this error isn't tied to a GPU id
}
