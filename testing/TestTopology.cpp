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
#include "TestTopology.h"
#include <ctime>
#include <iostream>
#include <stddef.h>

TestTopology::TestTopology()
{}
TestTopology::~TestTopology()
{}

int TestTopology::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    m_gpus = gpus;
    return 0;
}

int TestTopology::Run()
{
    int st;
    int Nfailed = 0;
    st          = TestTopologyDevice();

    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestTopology::Test topology device FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestTopology::Test topology device PASSED\n");

    st = TestTopologyGroup();

    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestTopology::Test topology group FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestTopology::Test topology group PASSED\n");

    if (Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    return 0;
}

int TestTopology::Cleanup()
{
    return 0;
}

std::string TestTopology::GetTag()
{
    return std::string("topology");
}

int TestTopology::TestTopologyDevice()
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmDeviceTopology_t topologyInfo;
    dcgmGroupInfo_t groupIdInfo = {};

    groupIdInfo.version = dcgmGroupInfo_version;
    result              = dcgmGroupGetInfo(m_dcgmHandle, (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS, &groupIdInfo);
    if (result != DCGM_ST_OK)
        return -1;

    result = dcgmGetDeviceTopology(m_dcgmHandle, m_gpus[0].gpuId, &topologyInfo);
    if (result == DCGM_ST_NOT_SUPPORTED && groupIdInfo.count < 2)
    {
        printf("Ignoring NOT_SUPPORTED with only %u GPU(s)", groupIdInfo.count);
        return 0;
    }
    else if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGetDeviceTopology failed with %d\n", (int)result);
        return -1;
    }

    // the GPU should have *some* affinity
    if (topologyInfo.cpuAffinityMask[0] == 0)
    {
        fprintf(stderr, "CPU affinity mask of gpuId %u == 0\n", m_gpus[0].gpuId);
        return 1;
    }

    // we should have path information for all GPUs in the system not including 0
    if (topologyInfo.numGpus != m_gpus.size() - 1)
    {
        fprintf(stderr,
                "topologyInfo.numGpus %u != m_gpus.size() - 1 %u for gpuId %u\n",
                topologyInfo.numGpus,
                (unsigned int)m_gpus.size() - 1,
                m_gpus[0].gpuId);
        return 1;
    }

    return 0;
}

int TestTopology::TestTopologyGroup()
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupTopology_t groupInfo;
    dcgmGpuGrp_t groupId        = 0;
    dcgmGroupInfo_t groupIdInfo = {};

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
        return -1;

    groupIdInfo.version = dcgmGroupInfo_version;
    result              = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupIdInfo);
    if (result != DCGM_ST_OK)
        return -1;

    // not a lot to be tested here without a priori knowledge of the system so
    // just make sure the command runs for now.
    result = dcgmGetGroupTopology(m_dcgmHandle, groupId, &groupInfo);
    if (result == DCGM_ST_NOT_SUPPORTED && groupIdInfo.count < 2)
    {
        printf("Ignoring NOT_SUPPORTED with only %u GPU(s)", groupIdInfo.count);
        return 0;
    }
    else if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGetGroupTopology failed with %d\n", (int)result);
        return -1;
    }

    return 0;
}
