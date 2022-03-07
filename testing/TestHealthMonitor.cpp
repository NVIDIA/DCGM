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
#include "TestHealthMonitor.h"
#include "dcgm_test_apis.h"
#include <Defer.hpp>
#include <TimeLib.hpp>

#include <ctime>
#include <fmt/core.h>
#include <iostream>
#include <stddef.h>
#include <string.h>

using DcgmNs::Defer;
using DcgmNs::Timelib::Now;
using DcgmNs::Timelib::ToLegacyTimestamp;
using namespace std::chrono_literals;

TestHealthMonitor::TestHealthMonitor()
{}
TestHealthMonitor::~TestHealthMonitor()
{}

int TestHealthMonitor::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    m_gpus = gpus;
    return 0;
}

int TestHealthMonitor::Run()
{
    int st;
    int Nfailed = 0;

    st = TestHMSet();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM set FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM set PASSED\n");

    st = TestHMCheckPCIe();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (PCIe) FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (PCIe) PASSED\n");

    st = TestHMCheckMemSbe();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (Mem,Sbe) FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (Mem,Sbe) PASSED\n");

    st = TestHMCheckMemDbe();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (Mem,Dbe) FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (Mem,Dbe) PASSED\n");

    st = TestHMCheckInforom();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (InfoROM) FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (InfoROM) PASSED\n");
    st = TestHMCheckThermal();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (Thermal) FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (Thermal) PASSED\n");
    st = TestHMCheckPower();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (Power) FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (Power) PASSED\n");

    st = TestHMCheckNVLink();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (NVLink) FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (NVLink) PASSED\n");


    if (Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    return 0;
}

int TestHealthMonitor::Cleanup()
{
    return 0;
}

std::string TestHealthMonitor::GetTag()
{
    return std::string("healthmonitor");
}

int TestHealthMonitor::TestHMSet()
{
    dcgmGpuGrp_t groupId           = 0;
    dcgmReturn_t result            = DCGM_ST_OK;
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_MEM);
    dcgmHealthSystems_t oldSystems;

    Defer defer([&, this] {
        if (groupId)
            dcgmGroupDestroy(m_dcgmHandle, groupId);
    });

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, dcgmHealthSystems_t(0));
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthGet(m_dcgmHandle, groupId, &oldSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (oldSystems != (dcgmHealthSystems_t)0)
    {
        result = DCGM_ST_GENERIC_ERROR;
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthGet(m_dcgmHandle, groupId, &oldSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (oldSystems != newSystems)
    {
        result = DCGM_ST_GENERIC_ERROR;
    }

    return result;
}

int TestHealthMonitor::TestHMCheckMemDbe()
{
    dcgmGpuGrp_t groupId          = 0;
    dcgmReturn_t result           = DCGM_ST_OK;
    dcgmHealthResponse_t response = {};
    dcgmInjectFieldValue_t fv;
    dcgmGroupInfo_t groupInfo;

    Defer defer([&groupId, this] {
        if (groupId)
            dcgmGroupDestroy(m_dcgmHandle, groupId);
    });

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_MEM);
    response.version               = dcgmHealthResponse_version;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupCreate failed with %d\n", (int)result);
        return result;
    }

    memset(&groupInfo, 0, sizeof(groupInfo));
    groupInfo.version = dcgmGroupInfo_version;
    result            = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    if (groupInfo.count < 1)
    {
        printf("Skipping TestHMCheckMemDbe due to no GPUs being present");
        result = DCGM_ST_OK; /* Don't fail */
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineHealthSet failed with %d\n", (int)result);
        return result;
    }

    auto now = Now();

    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldId   = DCGM_FI_DEV_ECC_DBE_VOL_TOTAL;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now - 50s);

    result = dcgmInjectFieldValue(m_dcgmHandle, groupInfo.entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        return result;
    }

    fv.fieldId   = DCGM_FI_DEV_ECC_DBE_VOL_TOTAL;
    fv.value.i64 = 5;
    fv.ts        = ToLegacyTimestamp(now);

    result = dcgmInjectFieldValue(m_dcgmHandle, groupInfo.entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        return result;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_FAIL)
    {
        fprintf(stderr, "response.overallHealth %d != DCGM_HEALTH_RESULT_FAIL\n", (int)response.overallHealth);
        result = DCGM_ST_GENERIC_ERROR;
    }

    if (response.incidentCount < 1)
    {
        fmt::print(stderr, "response.incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response.incidents[0].error.msg << std::endl;

    return result;
}


int TestHealthMonitor::TestHMCheckMemSbe()
{
    dcgmGpuGrp_t groupId          = 0;
    dcgmReturn_t result           = DCGM_ST_OK;
    dcgmHealthResponse_t response = {};
    dcgmInjectFieldValue_t fv;
    dcgmGroupInfo_t groupInfo;

    Defer defer([&, this] {
        if (groupId)
            dcgmGroupDestroy(m_dcgmHandle, groupId);
    });

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_MEM);
    response.version               = dcgmHealthResponse_version;

    auto now = Now();

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupCreate failed with %d\n", (int)result);
        return result;
    }

    memset(&groupInfo, 0, sizeof(groupInfo));
    groupInfo.version = dcgmGroupInfo_version;
    result            = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    if (groupInfo.count < 1)
    {
        printf("Skipping TestHMCheckMemSbe due to no GPUs being present");
        result = DCGM_ST_OK; /* Don't fail */
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineHealthSet failed with %d\n", (int)result);
        return result;
    }

    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.ts        = ToLegacyTimestamp(now - 50s);
    fv.fieldId   = DCGM_FI_DEV_ECC_SBE_VOL_TOTAL;
    fv.value.i64 = 0;

    result = dcgmInjectFieldValue(m_dcgmHandle, groupInfo.entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        return result;
    }

    fv.fieldId   = DCGM_FI_DEV_ECC_SBE_VOL_TOTAL;
    fv.value.i64 = 20;
    fv.ts        = ToLegacyTimestamp(now);

    result = dcgmInjectFieldValue(m_dcgmHandle, groupInfo.entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        return result;
    }


    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        return result;
    }

    /* Health checks no longer look for SBEs. We should not fail */
    if (response.overallHealth != DCGM_HEALTH_RESULT_PASS)
    {
        fprintf(stderr, "response.overallHealth %d != DCGM_HEALTH_RESULT_PASS\n", (int)response.overallHealth);
        result = DCGM_ST_GENERIC_ERROR;
    }

    if (response.incidentCount < 1)
    {
        fmt::print(stderr, "response.incidentCount < 1\n");
        /*
         * There will be no incidnets if we do not consider SBE as an issue.
         */
        return DCGM_ST_OK;
    }

    std::cout << response.incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckPCIe()
{
    dcgmGpuGrp_t groupId          = 0;
    dcgmReturn_t result           = DCGM_ST_OK;
    dcgmHealthResponse_t response = {};
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0].gpuId;

    Defer defer([&, this] {
        if (groupId)
            dcgmGroupDestroy(m_dcgmHandle, groupId);
    });

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_PCIE);
    response.version               = dcgmHealthResponse_version;

    auto now = Now();

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldId   = DCGM_FI_DEV_PCIE_REPLAY_COUNTER;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now - 50s);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        return result;
    }

    fv.value.i64 = 100;
    fv.ts        = ToLegacyTimestamp(now);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    if (response.incidentCount < 1)
    {
        fmt::print(stderr, "response.incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response.incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckInforom()
{
    dcgmGpuGrp_t groupId          = 0;
    dcgmReturn_t result           = DCGM_ST_OK;
    dcgmHealthResponse_t response = {};
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0].gpuId;

    Defer defer([&, this] {
        if (groupId)
            dcgmGroupDestroy(m_dcgmHandle, groupId);
    });

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_INFOROM);
    response.version               = dcgmHealthResponse_version;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldId   = DCGM_FI_DEV_INFOROM_CONFIG_VALID;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = 0; // inject that it is invalid
    fv.ts        = DcgmNs::Timelib::ToLegacyTimestamp(DcgmNs::Timelib::Now());

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        return result;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    if (response.incidentCount < 1)
    {
        fmt::print(stderr, "response.incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response.incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckThermal()
{
    dcgmGpuGrp_t groupId          = 0;
    dcgmReturn_t result           = DCGM_ST_OK;
    dcgmHealthResponse_t response = {};
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0].gpuId;

    Defer defer([&, this] {
        if (groupId)
            dcgmGroupDestroy(m_dcgmHandle, groupId);
    });

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_THERMAL);
    response.version               = dcgmHealthResponse_version;

    auto now = Now();

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldId   = DCGM_FI_DEV_THERMAL_VIOLATION;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now - 50s);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA && result != DCGM_ST_STALE_DATA)
    {
        return result;
    }

    fv.value.i64 = 1000;
    fv.ts        = ToLegacyTimestamp(now);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    if (response.incidentCount < 1)
    {
        fmt::print(stderr, "response.incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response.incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckPower()
{
    dcgmGpuGrp_t groupId          = 0;
    dcgmReturn_t result           = DCGM_ST_OK;
    dcgmHealthResponse_t response = {};
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0].gpuId;

    Defer defer([&, this] {
        if (groupId)
            dcgmGroupDestroy(m_dcgmHandle, groupId);
    });

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_POWER);
    response.version               = dcgmHealthResponse_version;

    auto now = Now();

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldId   = DCGM_FI_DEV_POWER_VIOLATION;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now - 50s);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA && result != DCGM_ST_STALE_DATA)
    {
        return result;
    }

    fv.value.i64 = 1000;
    fv.ts        = ToLegacyTimestamp(now);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    if (response.incidentCount < 1)
    {
        fmt::print(stderr, "response.incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response.incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckNVLink()
{
    dcgmGpuGrp_t groupId          = 0;
    dcgmReturn_t result           = DCGM_ST_OK;
    dcgmHealthResponse_t response = {};
    dcgmInjectFieldValue_t fv;
    dcgmGroupInfo_t groupInfo;
    unsigned int gpuId;
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_NVLINK);
    response.version               = dcgmHealthResponse_version;

    Defer defer([&, this] {
        if (groupId)
            dcgmGroupDestroy(m_dcgmHandle, groupId);
    });

    auto now = Now();

    // Create a group consisting of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Cannot create group 'TEST1': '%s'\n", errorString(result));
        return result;
    }

    // Get the group Info
    memset(&groupInfo, 0, sizeof(groupInfo));
    groupInfo.version = dcgmGroupInfo_version;
    result            = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    // Skip the test if no GPU is found
    if (groupInfo.count < 1)
    {
        printf("Skipping TestHMCheckNVLink due to no GPUs being present\n");
        result = DCGM_ST_OK; /* Don't fail */
        return result;
    }

    // Save the first GPU Id in the list
    gpuId = groupInfo.entityList[0].entityId;

    // Destroy the group and create a new group with the first GPU in the list
    if (groupId)
    {
        result = dcgmGroupDestroy(m_dcgmHandle, groupId);
        if (result != DCGM_ST_OK)
        {
            fprintf(stderr, "dcgmEngineGroupDestroy  failed with %d\n", (int)result);
            return result;
        }
        groupId = 0;
    }

    // Create an empty group
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Cannot create group 'TEST1': '%s'\n", errorString(result));
        return result;
    }

    // Add saved gpudId to the empty group
    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Cannot add device %u to group '%s'\n", gpuId, errorString(result));
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to set NVLINK health watch: '%s'\n", errorString(result));
        return result;
    }

    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldId   = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now - 50s);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to inject a 0 value for an NVLINK field: '%s'\n", errorString(result));
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "Unable to check the health watches for this system: '%s'\n", errorString(result));
        return result;
    }

    // Ensure the initial nvlink health is good otherwise report and skip test
    if (response.overallHealth != DCGM_HEALTH_RESULT_PASS)
    {
        printf("Skipping TestHealthMonitor::Test HM check (NVLink). "
               "Test cannot run since NVLink health check did not pass.\n");
        result = DCGM_ST_OK;
        return result;
    }

    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now - 50s);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to inject an error to trigger the NVLINK health failure: '%s'\n", errorString(result));
        return result;
    }

    fv.value.i64 = 1;
    fv.ts        = ToLegacyTimestamp(now);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr,
                "Unable to inject a second error to trigger the NVLINK health failure: '%s'\n",
                errorString(result));
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK)
    {
        fprintf(
            stderr, "Unable to check the NVLINK health watches after injecting a failure: '%s'\n", errorString(result));
        return result;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
    {
        result = DCGM_ST_GENERIC_ERROR;
        fprintf(stderr, "Did not get a health watch warning even though we injected errors.\n");
    }

    if (response.incidentCount < 1)
    {
        fmt::print(stderr, "response.incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response.incidents[0].error.msg << std::endl;

    return result;
}
