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
#include "TestHealthMonitor.h"
#include "dcgm_test_apis.h"
#include <TimeLib.hpp>

#include <ctime>
#include <fmt/core.h>
#include <iostream>
#include <memory>
#include <stddef.h>
#include <string.h>

using DcgmNs::Timelib::Now;
using DcgmNs::Timelib::ToLegacyTimestamp;
using namespace std::chrono_literals;

TestHealthMonitor::TestHealthMonitor()
{
    m_gpuGroup = 0;
}

TestHealthMonitor::~TestHealthMonitor()
{}

int TestHealthMonitor::Init(const TestDcgmModuleInitParams &initParams)
{
    m_gpus = initParams.fakeGpuIds;

    dcgmReturn_t dcgmReturn = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, "healthgroup", &m_gpuGroup);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupCreate returned %d", (int)dcgmReturn);
        return -1;
    }

    for (auto &gpuId : m_gpus)
    {
        dcgmReturn = dcgmGroupAddEntity(m_dcgmHandle, m_gpuGroup, DCGM_FE_GPU, gpuId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "dcgmGroupAddEntity returned %d for gpuId %u", (int)dcgmReturn, gpuId);
            return -1;
        }
    }

    return 0;
}

int TestHealthMonitor::Run()
{
    size_t nFailed = 0;

    constexpr struct
    {
        std::string_view name;
        int (TestHealthMonitor::*method)();
    } testCases[] = {
        { "Test HM set", &TestHealthMonitor::TestHMSet },
        { "Test HM check (PCIe)", &TestHealthMonitor::TestHMCheckPCIe },
        { "Test HM check (Mem,Sbe)", &TestHealthMonitor::TestHMCheckMemSbe },
        { "Test HM check (Mem,Dbe)", &TestHealthMonitor::TestHMCheckMemDbe },
        { "Test HM check (InfoROM)", &TestHealthMonitor::TestHMCheckInforom },
        { "Test HM check (Thermal)", &TestHealthMonitor::TestHMCheckThermal },
        { "Test HM check (Power)", &TestHealthMonitor::TestHMCheckPower },
        { "Test HM check (NVLink)", &TestHealthMonitor::TestHMCheckNVLink },
    };

    for (auto const &test : testCases)
    {
        int st = (this->*(test.method))();
        if (st != 0)
        {
            nFailed++;
            fmt::print(stderr, "TestHealthMonitor::{} FAILED with {}\n", test.name, st);
        }
        else
        {
            fmt::print("TestHealthMonitor::{} PASSED\n", test.name);
        }
    }

    if (nFailed > 0)
    {
        fmt::print(stderr, "TestHealthMonitor: {} tests FAILED.\n", nFailed);
        return -1;
    }
    else
    {
        fmt::print("TestHealthMonitor: All tests PASSED.\n");
    }

    return 0;
}

int TestHealthMonitor::Cleanup()
{
    if (m_gpuGroup != 0)
    {
        dcgmGroupDestroy(m_dcgmHandle, m_gpuGroup);
        m_gpuGroup = 0;
    }

    return 0;
}

std::string TestHealthMonitor::GetTag()
{
    return std::string("healthmonitor");
}

int TestHealthMonitor::TestHMSet()
{
    dcgmReturn_t result            = DCGM_ST_OK;
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_MEM);
    dcgmHealthSystems_t oldSystems;

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, dcgmHealthSystems_t(0));
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthGet(m_dcgmHandle, m_gpuGroup, &oldSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (oldSystems != (dcgmHealthSystems_t)0)
    {
        result = DCGM_ST_GENERIC_ERROR;
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthGet(m_dcgmHandle, m_gpuGroup, &oldSystems);
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
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmInjectFieldValue_t fv;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo     = std::make_unique<dcgmGroupInfo_t>();
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_MEM);
    response->version              = dcgmHealthResponse_version;

    memset(groupInfo.get(), 0, sizeof(*groupInfo));
    groupInfo->version = dcgmGroupInfo_version;
    result             = dcgmGroupGetInfo(m_dcgmHandle, m_gpuGroup, groupInfo.get());
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    if (groupInfo->count < 1)
    {
        printf("Skipping TestHMCheckMemDbe due to no GPUs being present");
        result = DCGM_ST_OK; /* Don't fail */
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
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

    result = dcgmInjectFieldValue(m_dcgmHandle, groupInfo->entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        return result;
    }

    fv.fieldId   = DCGM_FI_DEV_ECC_DBE_VOL_TOTAL;
    fv.value.i64 = 5;
    fv.ts        = ToLegacyTimestamp(now);

    result = dcgmInjectFieldValue(m_dcgmHandle, groupInfo->entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        return result;
    }

    if (response->overallHealth != DCGM_HEALTH_RESULT_FAIL)
    {
        fprintf(stderr, "response->overallHealth %d != DCGM_HEALTH_RESULT_FAIL\n", (int)response->overallHealth);
        result = DCGM_ST_GENERIC_ERROR;
    }

    if (response->incidentCount < 1)
    {
        fmt::print(stderr, "response->incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response->incidents[0].error.msg << std::endl;

    return result;
}


int TestHealthMonitor::TestHMCheckMemSbe()
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmInjectFieldValue_t fv;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo     = std::make_unique<dcgmGroupInfo_t>();
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_MEM);
    response->version              = dcgmHealthResponse_version;

    auto now = Now();

    memset(groupInfo.get(), 0, sizeof(*groupInfo));
    groupInfo->version = dcgmGroupInfo_version;
    result             = dcgmGroupGetInfo(m_dcgmHandle, m_gpuGroup, groupInfo.get());
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    if (groupInfo->count < 1)
    {
        printf("Skipping TestHMCheckMemSbe due to no GPUs being present");
        result = DCGM_ST_OK; /* Don't fail */
        return result;
    }

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
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

    result = dcgmInjectFieldValue(m_dcgmHandle, groupInfo->entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        return result;
    }

    fv.fieldId   = DCGM_FI_DEV_ECC_SBE_VOL_TOTAL;
    fv.value.i64 = 20;
    fv.ts        = ToLegacyTimestamp(now);

    result = dcgmInjectFieldValue(m_dcgmHandle, groupInfo->entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        return result;
    }


    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        return result;
    }

    /* Health checks no longer look for SBEs. We should not fail */
    if (response->overallHealth != DCGM_HEALTH_RESULT_PASS)
    {
        fprintf(stderr, "response->overallHealth %d != DCGM_HEALTH_RESULT_PASS\n", (int)response->overallHealth);
        result = DCGM_ST_GENERIC_ERROR;
    }

    if (response->incidentCount < 1)
    {
        fmt::print(stderr, "response->incidentCount < 1\n");
        /*
         * There will be no incidnets if we do not consider SBE as an issue.
         */
        return DCGM_ST_OK;
    }

    std::cout << response->incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckPCIe()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0];

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_PCIE);
    response->version              = dcgmHealthResponse_version;

    auto now = Now();

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    /* Inject PCIe generation and width/lanes. */
    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = 4;
    fv.fieldId   = DCGM_FI_DEV_PCIE_LINK_GEN;
    fv.ts        = ToLegacyTimestamp(now);
    result       = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    fv.value.i64 = 16;
    fv.fieldId   = DCGM_FI_DEV_PCIE_LINK_WIDTH;
    result       = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    fv.fieldId   = DCGM_FI_DEV_PCIE_REPLAY_COUNTER;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now - 50s);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
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

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (response->overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    if (response->incidentCount < 1)
    {
        fmt::print(stderr, "response->incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response->incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckInforom()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0];

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_INFOROM);
    response->version              = dcgmHealthResponse_version;

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
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

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        return result;
    }

    if (response->overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    if (response->incidentCount < 1)
    {
        fmt::print(stderr, "response->incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response->incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckThermal()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0];

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_THERMAL);
    response->version              = dcgmHealthResponse_version;

    auto now = Now();

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
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

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
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

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (response->overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    if (response->incidentCount < 1)
    {
        fmt::print(stderr, "response->incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response->incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckPower()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0];

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_POWER);
    response->version              = dcgmHealthResponse_version;

    auto now = Now();

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
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

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
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

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    if (response->overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    if (response->incidentCount < 1)
    {
        fmt::print(stderr, "response->incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response->incidents[0].error.msg << std::endl;

    return result;
}

int TestHealthMonitor::TestHMCheckNVLink()
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmInjectFieldValue_t fv;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo     = std::make_unique<dcgmGroupInfo_t>();
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));
    unsigned int gpuId;
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_NVLINK);
    response->version              = dcgmHealthResponse_version;

    auto now = Now();

    // Get the group Info
    memset(groupInfo.get(), 0, sizeof(*groupInfo));
    groupInfo->version = dcgmGroupInfo_version;
    result             = dcgmGroupGetInfo(m_dcgmHandle, m_gpuGroup, groupInfo.get());
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    // Skip the test if no GPU is found
    if (groupInfo->count < 1)
    {
        printf("Skipping TestHMCheckNVLink due to no GPUs being present\n");
        result = DCGM_ST_OK; /* Don't fail */
        return result;
    }

    // Save the first GPU Id in the list
    gpuId = groupInfo->entityList[0].entityId;

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
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

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "Unable to check the health watches for this system: '%s'\n", errorString(result));
        return result;
    }

    // Ensure the initial nvlink health is good otherwise report and skip test
    if (response->overallHealth != DCGM_HEALTH_RESULT_PASS)
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
    fv.ts        = ToLegacyTimestamp(now + 2s);

    result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr,
                "Unable to inject a second error to trigger the NVLINK health failure: '%s'\n",
                errorString(result));
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK)
    {
        fprintf(
            stderr, "Unable to check the NVLINK health watches after injecting a failure: '%s'\n", errorString(result));
        return result;
    }

    if (response->overallHealth != DCGM_HEALTH_RESULT_WARN)
    {
        result = DCGM_ST_GENERIC_ERROR;
        fprintf(stderr, "Did not get a health watch warning even though we injected errors.\n");
    }

    if (response->incidentCount < 1)
    {
        fmt::print(stderr, "response->incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response->incidents[0].error.msg << std::endl;

    return result;
}
