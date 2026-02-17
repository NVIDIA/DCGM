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
#include "TestHealthMonitor.h"
#include "dcgm_structs.h"
#include "dcgm_test_apis.h"
#include <DcgmStringHelpers.h>
#include <TimeLib.hpp>
#include <UniquePtrUtil.h>

#include <algorithm>
#include <ctime>
#include <fmt/core.h>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <stddef.h>
#include <string.h>

using DcgmNs::Timelib::Now;
using DcgmNs::Timelib::ToLegacyTimestamp;
using namespace std::chrono_literals;

TestHealthMonitor::TestHealthMonitor()
{
    m_gpuGroup = 0;

    // Initialize devastating XIDs
    m_devastatingXids = {
        { 48, "Double Bit ECC Error", DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_ALL, DCGM_FR_XID_ERROR },
        { 74, "NVLink Critical Error", DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_ALL, DCGM_FR_XID_ERROR },
        { 79, DCGM_FR_FALLEN_OFF_BUS_MSG, DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_ALL, DCGM_FR_FALLEN_OFF_BUS },
        { 94, "Contained Error", DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_ALL, DCGM_FR_XID_ERROR },
        { 95, DCGM_FR_FALLEN_OFF_BUS_MSG, DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_ALL, DCGM_FR_UNCONTAINED_ERROR },
        { 119, "GSP RPC Timeout", DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_ALL, DCGM_FR_XID_ERROR },
        { 120, "GSP Error", DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_ALL, DCGM_FR_XID_ERROR },
        { 140, "ECC unrecovered error", DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_ALL, DCGM_FR_XID_ERROR }
    };

    // Initialize subsystem XIDs
    m_subsystemXids = {
        // Memory subsystem XIDs
        { 31, "MMU Error", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_MEM, DCGM_FR_XID_ERROR },
        { 32, "PBDMA Error", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_MEM, DCGM_FR_XID_ERROR },
        { 43, "Reset Channel Verif Error", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_MEM, DCGM_FR_XID_ERROR },
        { 63,
          DCGM_FR_PENDING_PAGE_RETIREMENTS_MSG,
          DCGM_HEALTH_RESULT_WARN,
          DCGM_HEALTH_WATCH_MEM,
          DCGM_FR_PENDING_PAGE_RETIREMENTS },
        { 64,
          DCGM_FR_ROW_REMAP_FAILURE_MSG,
          DCGM_HEALTH_RESULT_WARN,
          DCGM_HEALTH_WATCH_MEM,
          DCGM_FR_ROW_REMAP_FAILURE },

        // PCIe subsystem XIDs
        { 38, "PCIe Bus Error", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_PCIE, DCGM_FR_XID_ERROR },
        { 39, "PCIe Fabric Error", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_PCIE, DCGM_FR_XID_ERROR },
        { 42, DCGM_FR_PCI_REPLAY_RATE_MSG, DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_PCIE, DCGM_FR_PCI_REPLAY_RATE },

        // Thermal subsystem XIDs
        { 60,
          DCGM_FR_CLOCKS_EVENT_THERMAL_MSG,
          DCGM_HEALTH_RESULT_WARN,
          DCGM_HEALTH_WATCH_THERMAL,
          DCGM_FR_CLOCKS_EVENT_THERMAL },
        { 61, "EDPP Power Brake Thermal limit", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_THERMAL, DCGM_FR_XID_ERROR },
        { 62,
          DCGM_FR_THERMAL_VIOLATIONS_MSG,
          DCGM_HEALTH_RESULT_WARN,
          DCGM_HEALTH_WATCH_THERMAL,
          DCGM_FR_THERMAL_VIOLATIONS },
        { 63, "Thermal diode detects short", DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_THERMAL, DCGM_FR_XID_ERROR },

        // Power subsystem XIDs
        { 54, "Power state change", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_POWER, DCGM_FR_XID_ERROR },
        { 56, "Clock change", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_POWER, DCGM_FR_XID_ERROR },
        { 57,
          DCGM_FR_CLOCKS_EVENT_POWER_MSG,
          DCGM_HEALTH_RESULT_WARN,
          DCGM_HEALTH_WATCH_POWER,
          DCGM_FR_CLOCKS_EVENT_POWER },
        { 58, "Clock change due to thermal", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_POWER, DCGM_FR_XID_ERROR },
        { 78, "Power state forced change", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_POWER, DCGM_FR_XID_ERROR },

        // NVLink subsystem XIDs
        { 67,
          DCGM_FR_NVLINK_ERROR_THRESHOLD_MSG,
          DCGM_HEALTH_RESULT_WARN,
          DCGM_HEALTH_WATCH_NVLINK,
          DCGM_FR_NVLINK_ERROR_THRESHOLD },
        { 73, "NVLink Flow Control Error", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_NVLINK, DCGM_FR_XID_ERROR },
        { 121, "C2C Link corrected error", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_NVLINK, DCGM_FR_XID_ERROR },
        { 137, "NVLink FLA privilege error", DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_NVLINK, DCGM_FR_XID_ERROR },

        // InfoROM subsystem XIDs
        { 93, DCGM_FR_CORRUPT_INFOROM_MSG, DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_INFOROM, DCGM_FR_CORRUPT_INFOROM }
    };
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
        { "Test HM check (Mem,UnrepairableFlag)", &TestHealthMonitor::TestHMCheckMemUnrepairableFlag },
        { "Test HM check (IMEX)", &TestHealthMonitor::TestHMCheckImex },
        { "Test HM check (InfoROM)", &TestHealthMonitor::TestHMCheckInforom },
        { "Test HM check (Thermal)", &TestHealthMonitor::TestHMCheckThermal },
        { "Test HM check (Power)", &TestHealthMonitor::TestHMCheckPower },
        { "Test HM check (NVLink)", &TestHealthMonitor::TestHMCheckNVLink },
        { "Test HM check (NVLink States)", &TestHealthMonitor::TestHMCheckNVLinkStates },
        { "Test HM check (Fabric Manager Status)", &TestHealthMonitor::TestHMCheckFabricManagerStatus },
        { "Test HM check (XIDs)", &TestHealthMonitor::TestHMCheckXids },
        { "Test HM check (residual - final)", &TestHealthMonitor::TestHMCheckResidual },
    };

    int result = TestHMCheckResidual();
    if (result != DCGM_ST_OK)
    {
        fmt::print(
            stderr, "TestHealthMonitor::TestHMCheckResidual (initial) FAILED with {}, skipping other tests\n", result);
        return 0;
    }

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

    result = TestHMCheckResidual();
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "TestHealthMonitor::TestHMCheckResidual (final) FAILED with {}\n", result);
        nFailed++;
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

    // Cleanup: Inject healthy value to avoid impacting other tests
    fv.fieldId   = DCGM_FI_DEV_ECC_DBE_VOL_TOTAL;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now + 1s);
    result       = dcgmInjectFieldValue(m_dcgmHandle, groupInfo->entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject cleanup value: {}\n", (int)result);
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

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

    // Cleanup: Inject healthy value to avoid impacting other tests
    fv.fieldId   = DCGM_FI_DEV_ECC_SBE_VOL_TOTAL;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now + 1s);
    result       = dcgmInjectFieldValue(m_dcgmHandle, groupInfo->entityList[0].entityId, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject cleanup value: {}\n", (int)result);
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

    return result;
}

int TestHealthMonitor::TestHMCheckMemUnrepairableFlag()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0];

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_MEM);
    response->version              = dcgmHealthResponse_version;

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldId   = DCGM_FI_DEV_MEMORY_UNREPAIRABLE_FLAG;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = 1; // inject that unrepairable memory is detected
    fv.ts        = ToLegacyTimestamp(Now());

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

    if (response->overallHealth != DCGM_HEALTH_RESULT_FAIL)
        result = DCGM_ST_GENERIC_ERROR;

    if (response->incidentCount < 1)
    {
        fmt::print(stderr, "response->incidentCount < 1\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response->incidents[0].error.msg << std::endl;

    // Cleanup: Inject healthy value to avoid impacting other tests
    fv.fieldId   = DCGM_FI_DEV_MEMORY_UNREPAIRABLE_FLAG;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(Now() + 1s);
    result       = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject cleanup value: {}\n", (int)result);
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

    return result;
}

namespace
{
// Test data structure for IMEX health tests
struct ImexHealthTestCase
{
    std::string_view domainStatus;
    int64_t daemonStatus;
    dcgmHealthWatchResults_t expectedHealth;
    bool checkIncidents;
    std::string_view description;
};
} // namespace

dcgmReturn_t TestHealthMonitor::ValidateImexHealth([[maybe_unused]] unsigned int gpuId,
                                                   std::string_view domainStatus,
                                                   int64_t daemonStatus,
                                                   dcgmHealthWatchResults_t expectedHealth,
                                                   bool checkIncidents,
                                                   DcgmNs::Timelib::TimePoint const timestamp,
                                                   std::string_view description)
{
    dcgmInjectFieldValue_t fv;
    fv.version = dcgmInjectFieldValue_version;
    fv.status  = 0;

    // Inject domain status
    // IMEX fields are global scope (DCGM_FE_NONE), inject with entity 0
    fv.fieldId   = DCGM_FI_IMEX_DOMAIN_STATUS;
    fv.fieldType = DCGM_FT_STRING;
    SafeCopyTo(fv.value.str, domainStatus.data());
    fv.ts = ToLegacyTimestamp(timestamp);

    auto result = dcgmInjectFieldValue(m_dcgmHandle, 0, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject domain status '{}': {}\n", domainStatus, (int)result);
        return result;
    }

    // Inject daemon status
    fv.fieldId   = DCGM_FI_IMEX_DAEMON_STATUS;
    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = daemonStatus;
    fv.ts        = ToLegacyTimestamp(timestamp);

    result = dcgmInjectFieldValue(m_dcgmHandle, 0, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject daemon status {}: {}\n", daemonStatus, (int)result);
        return result;
    }

    // Check health
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    response->version                              = dcgmHealthResponse_version;

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fmt::print(stderr, "dcgmHealthCheck failed for '{}': {}\n", description, (int)result);
        return result;
    }

    // Verify expected health
    if (response->overallHealth != expectedHealth)
    {
        fmt::print(
            stderr, "{}: Expected health {}, got {}\n", description, (int)expectedHealth, (int)response->overallHealth);
        return DCGM_ST_GENERIC_ERROR;
    }

    // Optionally check incidents
    if (checkIncidents && response->incidentCount < 1)
    {
        fmt::print(stderr, "{}: Expected at least 1 incident\n", description);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

int TestHealthMonitor::TestHMCheckImex()
{
    unsigned int gpuId = m_gpus[0];

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_NVLINK);
    auto result                    = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    auto now = Now();

    // Define test cases
    static constexpr std::array<ImexHealthTestCase, 5> testCases = { {
        { "UP", 5, DCGM_HEALTH_RESULT_PASS, false, "Healthy state (UP/READY)" },
        { "DOWN", 5, DCGM_HEALTH_RESULT_FAIL, true, "Domain DOWN" },
        { "DEGRADED", 5, DCGM_HEALTH_RESULT_FAIL, true, "Domain DEGRADED" },
        { "UP", 0, DCGM_HEALTH_RESULT_FAIL, true, "Daemon INITIALIZING" },
        { "NOT_INSTALLED", -1, DCGM_HEALTH_RESULT_PASS, false, "NOT_INSTALLED" }, // Clean up
    } };

    // Run all test cases
    size_t timestampOffset = 0;
    for (auto const &tc : testCases)
    {
        result = ValidateImexHealth(gpuId,
                                    tc.domainStatus,
                                    tc.daemonStatus,
                                    tc.expectedHealth,
                                    tc.checkIncidents,
                                    now + std::chrono::seconds(timestampOffset),
                                    tc.description);

        if (result != DCGM_ST_OK)
        {
            return result;
        }

        timestampOffset++;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

    return DCGM_ST_OK;
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

    // Cleanup: Inject healthy value to avoid impacting other tests
    fv.fieldId   = DCGM_FI_DEV_PCIE_REPLAY_COUNTER;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now + 1s);
    result       = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject cleanup value: {}\n", (int)result);
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

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

    // Cleanup: Inject healthy value to avoid impacting other tests
    fv.value.i64 = 1; // 1 = valid
    fv.ts        = ToLegacyTimestamp(Now() + 1s);
    result       = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject cleanup value: {}\n", (int)result);
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

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

    // Cleanup: Inject healthy value to avoid impacting other tests
    fv.fieldId   = DCGM_FI_DEV_THERMAL_VIOLATION;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now + 1s);
    result       = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject cleanup value: {}\n", (int)result);
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

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

    // Cleanup: Inject healthy value to avoid impacting other tests
    fv.fieldId   = DCGM_FI_DEV_POWER_VIOLATION;
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now + 1s);
    result       = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "Failed to inject cleanup value: {}\n", (int)result);
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

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

    // Cleanup: Inject healthy value to avoid impacting other tests
    // If these tests are expanded, we'll want to wrap this in a call to DcgmNs::Defer
    fv.value.i64 = 0;
    fv.ts        = ToLegacyTimestamp(now + 3s);
    result       = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to inject a healthy value: '%s'\n", errorString(result));
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

    return result;
}

struct TestHealthMonitor::NVLinkTestContext
{
    dcgmHandle_t dcgmHandle;      //!< DCGM handle
    dcgmGpuGrp_t gpuGroup;        //!< GPU group to test
    unsigned int gpuId;           //!< GPU ID to test
    size_t baselineIncidentCount; //!< Baseline incident count
};

namespace
{
/**
 * Helper to set all NVLink links to a specific state
 *
 * @param dcgmHandle DCGM handle
 * @param gpuId GPU ID to set links for
 * @param linkState State to set all links to
 * @return DCGM_ST_OK on success, error code on failure
 */
dcgmReturn_t SetAllNvLinkLinksToState(dcgmHandle_t dcgmHandle, unsigned int gpuId, dcgmNvLinkLinkState_t linkState)
{
    for (unsigned int linkId = 0; linkId < DCGM_NVLINK_MAX_LINKS_PER_GPU; linkId++)
    {
        dcgmSetNvLinkLinkState_v1 linkStateMsg = {};
        linkStateMsg.version                   = dcgmSetNvLinkLinkState_version1;
        linkStateMsg.entityGroupId             = DCGM_FE_GPU;
        linkStateMsg.entityId                  = gpuId;
        linkStateMsg.linkId                    = linkId;
        linkStateMsg.linkState                 = linkState;
        linkStateMsg.unused                    = 0;

        dcgmReturn_t result = dcgmSetEntityNvLinkLinkState(dcgmHandle, &linkStateMsg);
        if (result != DCGM_ST_OK)
        {
            fprintf(stderr,
                    "Unable to set NVLink link %u to state %d: '%s'\n",
                    linkId,
                    static_cast<int>(linkState),
                    errorString(result));
            return result;
        }
    }
    return DCGM_ST_OK;
}
} // namespace

int TestHealthMonitor::SubtestHMCheckNVLinkStatesAllLinksUp(NVLinkTestContext &context)
{
    dcgmReturn_t result = SetAllNvLinkLinksToState(context.dcgmHandle, context.gpuId, DcgmNvLinkLinkStateUp);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    auto response     = MakeUniqueZero<dcgmHealthResponse_t>();
    response->version = dcgmHealthResponse_version;

    result = dcgmHealthCheck(context.dcgmHandle, context.gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "Unable to check NVLink health with all links Up: '%s'\n", errorString(result));
        return result;
    }

    if (response->overallHealth == DCGM_HEALTH_RESULT_FAIL)
    {
        auto incidents
            = std::span(response->incidents,
                        std::min(std::size(response->incidents), static_cast<size_t>(response->incidentCount)))
                  .subspan(context.baselineIncidentCount);

        fprintf(stderr, "Expected PASS with all links Up, got health result %d\n", response->overallHealth);
        fprintf(stderr, "Total incidents: %d, New incidents: %lu\n", response->incidentCount, incidents.size());

        // Print details about new incidents
        size_t i = 0;
        for (auto incident : incidents)
        {
            fprintf(stderr,
                    "  New Incident %lu: system=%d, health=%d, error.code=%d, msg='%s'\n",
                    i,
                    incident.system,
                    incident.health,
                    incident.error.code,
                    incident.error.msg);
            i++;
        }
        return DCGM_ST_GENERIC_ERROR;
    }
    return DCGM_ST_OK;
}

int TestHealthMonitor::SubtestHMCheckNVLinkStatesLinkDown(NVLinkTestContext &context)
{
    // Reset all links to Up state before testing Down state
    dcgmReturn_t result = SetAllNvLinkLinksToState(context.dcgmHandle, context.gpuId, DcgmNvLinkLinkStateUp);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    unsigned int downLinkId             = 2;
    dcgmSetNvLinkLinkState_v1 linkState = {};
    linkState.version                   = dcgmSetNvLinkLinkState_version1;
    linkState.entityGroupId             = DCGM_FE_GPU;
    linkState.entityId                  = context.gpuId;
    linkState.linkId                    = downLinkId;
    linkState.linkState                 = DcgmNvLinkLinkStateDown;
    linkState.unused                    = 0;

    result = dcgmSetEntityNvLinkLinkState(context.dcgmHandle, &linkState);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to set NVLink link %u to Down state: '%s'\n", downLinkId, errorString(result));
        return result;
    }

    auto response     = MakeUniqueZero<dcgmHealthResponse_t>();
    response->version = dcgmHealthResponse_version;

    result = dcgmHealthCheck(context.dcgmHandle, context.gpuGroup, response.get());
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to check NVLink health with one link Down: '%s'\n", errorString(result));
        return result;
    }

    // Check for new NVLink incidents compared to baseline
    auto incidents = std::span(response->incidents,
                               std::min(std::size(response->incidents), static_cast<size_t>(response->incidentCount)))
                         .subspan(context.baselineIncidentCount);
    auto nvLinkIncident = std::find_if(incidents.begin(), incidents.end(), [](auto const &incident) {
        return incident.system == DCGM_HEALTH_WATCH_NVLINK;
    });

    if (nvLinkIncident == incidents.end())
    {
        fprintf(stderr, "Expected new NVLink incident with one link Down, but found none\n");
        fprintf(stderr, "Total incidents: %d, New incidents: %lu\n", response->incidentCount, incidents.size());
        return DCGM_ST_GENERIC_ERROR;
    }

    // Check that the NVLink incident has the correct error code
    if (nvLinkIncident->error.code != DCGM_FR_NVLINK_DOWN)
    {
        fprintf(stderr,
                "Expected new NVLink incident with DCGM_FR_NVLINK_DOWN, but got error code %d\n",
                nvLinkIncident->error.code);
        fprintf(stderr, "Incident msg: '%s'\n", nvLinkIncident->error.msg);
        return DCGM_ST_GENERIC_ERROR;
    }

    // Check that the error message contains the link ID
    std::string errorMsg(nvLinkIncident->error.msg);
    if (errorMsg.find(std::to_string(downLinkId)) == std::string::npos)
    {
        fprintf(stderr, "Expected error message to contain link ID %u, got: %s\n", downLinkId, errorMsg.c_str());
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

int TestHealthMonitor::SubtestHMCheckNVLinkStatesOther(NVLinkTestContext &context)
{
    // Reset all links to Up state before testing NotSupported/Disabled
    dcgmReturn_t result = SetAllNvLinkLinksToState(context.dcgmHandle, context.gpuId, DcgmNvLinkLinkStateUp);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Set some links to NotSupported
    for (unsigned int linkId = 0; linkId < 2; linkId++)
    {
        dcgmSetNvLinkLinkState_v1 linkState = {};
        linkState.version                   = dcgmSetNvLinkLinkState_version1;
        linkState.entityGroupId             = DCGM_FE_GPU;
        linkState.entityId                  = context.gpuId;
        linkState.linkId                    = linkId;
        linkState.linkState                 = DcgmNvLinkLinkStateNotSupported;
        linkState.unused                    = 0;

        result = dcgmSetEntityNvLinkLinkState(context.dcgmHandle, &linkState);
        if (result != DCGM_ST_OK)
        {
            fprintf(stderr, "Unable to set NVLink link %u to NotSupported state: '%s'\n", linkId, errorString(result));
            return result;
        }
    }

    // Set some links to Disabled
    for (unsigned int linkId = 3; linkId < 5; linkId++)
    {
        dcgmSetNvLinkLinkState_v1 linkState = {};
        linkState.version                   = dcgmSetNvLinkLinkState_version1;
        linkState.entityGroupId             = DCGM_FE_GPU;
        linkState.entityId                  = context.gpuId;
        linkState.linkId                    = linkId;
        linkState.linkState                 = DcgmNvLinkLinkStateDisabled;
        linkState.unused                    = 0;

        result = dcgmSetEntityNvLinkLinkState(context.dcgmHandle, &linkState);
        if (result != DCGM_ST_OK)
        {
            fprintf(stderr, "Unable to set NVLink link %u to Disabled state: '%s'\n", linkId, errorString(result));
            return result;
        }
    }

    auto response     = MakeUniqueZero<dcgmHealthResponse_t>();
    response->version = dcgmHealthResponse_version;

    result = dcgmHealthCheck(context.dcgmHandle, context.gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "Unable to check NVLink health with NotSupported/Disabled links: '%s'\n", errorString(result));
        return result;
    }

    // Check for new NVLink-specific incidents
    auto incidents = std::span(response->incidents,
                               std::min(std::size(response->incidents), static_cast<size_t>(response->incidentCount)))
                         .subspan(context.baselineIncidentCount);
    auto nvLinkIncident = std::find_if(incidents.begin(), incidents.end(), [](auto const &incident) {
        return incident.system == DCGM_HEALTH_WATCH_NVLINK;
    });

    if (nvLinkIncident != incidents.end())
    {
        fprintf(stderr, "Unexpected new NVLink incident with NotSupported/Disabled links\n");
        fprintf(stderr, "Total incidents: %d, New incidents: %lu\n", response->incidentCount, incidents.size());

        // Print details about new incidents
        size_t i = 0;
        for (auto incident : incidents)
        {
            fprintf(stderr,
                    "  New Incident %lu: system=%d, health=%d, error.code=%d, msg='%s'\n",
                    i,
                    incident.system,
                    incident.health,
                    incident.error.code,
                    incident.error.msg);
            i++;
        }
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

int TestHealthMonitor::TestHMCheckNVLinkStates()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo     = std::make_unique<dcgmGroupInfo_t>();
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));
    unsigned int gpuId;
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_NVLINK);
    response->version              = dcgmHealthResponse_version;

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "Unable to get baseline health state: '%s'\n", errorString(result));
        return result;
    }

    size_t baselineIncidentCount = response->incidentCount;

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
        printf("Skipping TestHMCheckNVLinkStates due to no GPUs being present\n");
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

    // Create context for subtest methods
    NVLinkTestContext context = { .dcgmHandle            = m_dcgmHandle,
                                  .gpuGroup              = m_gpuGroup,
                                  .gpuId                 = gpuId,
                                  .baselineIncidentCount = baselineIncidentCount };

    int (TestHealthMonitor::*const subtests[])(NVLinkTestContext &)
        = { &TestHealthMonitor::SubtestHMCheckNVLinkStatesAllLinksUp,
            &TestHealthMonitor::SubtestHMCheckNVLinkStatesLinkDown,
            &TestHealthMonitor::SubtestHMCheckNVLinkStatesOther };

    for (auto &subtest : subtests)
    {
        if (auto ret = (this->*subtest)(context); ret != DCGM_ST_OK)
        {
            return ret;
        }
    }

    return DCGM_ST_OK;
}

int TestHealthMonitor::TestHMCheckXids()
{
    // Test all devastating XIDs
    TestDevastatingXids();

    // Test subsystem-specific XIDs
    TestSubsystemXids();

    // Test XID severity levels
    TestXidSeverityLevels();

    return 0;
}

dcgmReturn_t TestHealthMonitor::TestSingleXid(unsigned int const gpuId,
                                              uint64_t const xid,
                                              char const *const xidDesc,
                                              dcgmHealthWatchResults_t const expectedStatus,
                                              dcgmError_t const expectedError,
                                              auto const timestamp,
                                              std::unique_ptr<dcgmHealthResponse_t> &response,
                                              dcgmHealthSystems_t const currentSubsystem) const
{
    dcgmInjectFieldValue_t fv {};
    fv.version   = dcgmInjectFieldValue_version;
    fv.fieldId   = DCGM_FI_DEV_XID_ERRORS;
    fv.fieldType = DCGM_FT_INT64;
    fv.status    = 0;
    fv.value.i64 = xid;
    fv.ts        = ToLegacyTimestamp(timestamp);

    dcgmReturn_t result = dcgmInjectFieldValue(m_dcgmHandle, gpuId, &fv);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr,
                "dcgmInjectFieldValue failed with %d for XID %llu (%s)\n",
                static_cast<int>(result),
                static_cast<unsigned long long>(xid),
                xidDesc);
        return result;
    }

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr,
                "dcgmHealthCheck failed with %d for XID %llu (%s)\n",
                static_cast<int>(result),
                static_cast<unsigned long long>(xid),
                xidDesc);
        return result;
    }

    if (response->overallHealth != expectedStatus)
    {
        fprintf(stderr,
                "XID %llu (%s) did not trigger expected health status %d, actual health status %d\n",
                static_cast<unsigned long long>(xid),
                xidDesc,
                static_cast<int>(expectedStatus),
                static_cast<int>(response->overallHealth));
        return DCGM_ST_GENERIC_ERROR;
    }

    bool foundMatchingSubsystem = false;
    for (unsigned int i = 0; i < response->incidentCount; i++)
    {
        if (response->incidents[i].system == currentSubsystem)
        {
            // Verify the error code matches what we expect
            if (response->incidents[i].error.code != expectedError)
            {
                fprintf(stderr,
                        "XID %llu (%s) triggered error code %d but expected %d for subsystem %d\n",
                        static_cast<unsigned long long>(xid),
                        xidDesc,
                        response->incidents[i].error.code,
                        expectedError,
                        static_cast<int>(currentSubsystem));
                return DCGM_ST_GENERIC_ERROR;
            }
            foundMatchingSubsystem = true;
            break;
        }
    }

    if (!foundMatchingSubsystem)
    {
        fprintf(stderr,
                "XID %llu (%s) did not trigger any incident for expected subsystem %d\n",
                static_cast<unsigned long long>(xid),
                xidDesc,
                static_cast<int>(currentSubsystem));
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

int TestHealthMonitor::TestDevastatingXids()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo     = std::make_unique<dcgmGroupInfo_t>();
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));

    // Enable all health monitoring since devastating XIDs are critical hardware errors
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_ALL);
    response->version              = dcgmHealthResponse_version;

    auto now = Now();

    memset(groupInfo.get(), 0, sizeof(*groupInfo));
    groupInfo->version = dcgmGroupInfo_version;
    result             = dcgmGroupGetInfo(m_dcgmHandle, m_gpuGroup, groupInfo.get());
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    if (groupInfo->count < 1)
    {
        printf("Skipping TestDevastatingXids due to no GPUs being present");
        return DCGM_ST_OK;
    }

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmHealthSet failed with %d\n", (int)result);
        return result;
    }

    for (size_t i = 0; i < m_devastatingXids.size(); i++)
    {
        result = TestSingleXid(groupInfo->entityList[0].entityId,
                               m_devastatingXids[i].xid,
                               m_devastatingXids[i].desc,
                               m_devastatingXids[i].expectedStatus,
                               m_devastatingXids[i].expectedError,
                               now + std::chrono::seconds(i),
                               response,
                               m_devastatingXids[i].subsystem);
        if (result != DCGM_ST_OK)
        {
            return result;
        }
    }

    // NOTE: XID history will persist across other tests
    return DCGM_ST_OK;
}

int TestHealthMonitor::TestSubsystemXids()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo     = std::make_unique<dcgmGroupInfo_t>();
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));

    // Enable all subsystems for testing
    dcgmHealthSystems_t newSystems
        = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_MEM | DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_THERMAL
                              | DCGM_HEALTH_WATCH_POWER | DCGM_HEALTH_WATCH_NVLINK | DCGM_HEALTH_WATCH_INFOROM);
    response->version = dcgmHealthResponse_version;

    auto now = Now();

    memset(groupInfo.get(), 0, sizeof(*groupInfo));
    groupInfo->version = dcgmGroupInfo_version;
    result             = dcgmGroupGetInfo(m_dcgmHandle, m_gpuGroup, groupInfo.get());
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    if (groupInfo->count < 1)
    {
        printf("Skipping TestSubsystemXids due to no GPUs being present");
        return DCGM_ST_OK;
    }

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmHealthSet failed with %d\n", (int)result);
        return result;
    }

    for (size_t i = 0; i < m_subsystemXids.size(); i++)
    {
        result = TestSingleXid(groupInfo->entityList[0].entityId,
                               m_subsystemXids[i].xid,
                               m_subsystemXids[i].desc,
                               m_subsystemXids[i].expectedStatus,
                               m_subsystemXids[i].expectedError,
                               now + std::chrono::seconds(i),
                               response,
                               m_subsystemXids[i].subsystem);
        if (result != DCGM_ST_OK)
        {
            return result;
        }
    }

    // NOTE: XID history will persist across other tests
    return DCGM_ST_OK;
}

int TestHealthMonitor::TestXidSeverityLevels()
{
    dcgmReturn_t result                            = DCGM_ST_OK;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo     = std::make_unique<dcgmGroupInfo_t>();
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));

    // Enable all subsystems for testing
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(
        DCGM_HEALTH_WATCH_ALL | DCGM_HEALTH_WATCH_MEM | DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_THERMAL
        | DCGM_HEALTH_WATCH_POWER | DCGM_HEALTH_WATCH_NVLINK | DCGM_HEALTH_WATCH_INFOROM);
    response->version = dcgmHealthResponse_version;

    auto now = Now();

    memset(groupInfo.get(), 0, sizeof(*groupInfo));
    groupInfo->version = dcgmGroupInfo_version;
    result             = dcgmGroupGetInfo(m_dcgmHandle, m_gpuGroup, groupInfo.get());
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupGetInfo failed with %d\n", (int)result);
        return result;
    }

    if (groupInfo->count < 1)
    {
        printf("Skipping TestXidSeverityLevels due to no GPUs being present");
        return DCGM_ST_OK;
    }

    result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmHealthSet failed with %d\n", (int)result);
        return result;
    }

    // Test WARN level XID
    result = TestSingleXid(groupInfo->entityList[0].entityId,
                           31,
                           "MMU Error (Memory subsystem)",
                           DCGM_HEALTH_RESULT_WARN,
                           DCGM_FR_XID_ERROR,
                           now,
                           response,
                           DCGM_HEALTH_WATCH_MEM);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Test FAIL level XID
    result = TestSingleXid(groupInfo->entityList[0].entityId,
                           48,
                           "Double Bit ECC Error (Memory subsystem)",
                           DCGM_HEALTH_RESULT_FAIL,
                           DCGM_FR_XID_ERROR,
                           now + 1s,
                           response,
                           DCGM_HEALTH_WATCH_ALL);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Test that FAIL-level XIDs override WARN-level XIDs for the rest of the test.
    result = TestSingleXid(groupInfo->entityList[0].entityId,
                           31,
                           "MMU Error (Memory subsystem)",
                           DCGM_HEALTH_RESULT_FAIL,
                           DCGM_FR_XID_ERROR,
                           now + 2s,
                           response,
                           DCGM_HEALTH_WATCH_MEM);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Then inject a FAIL level XID - should override the WARN status
    result = TestSingleXid(groupInfo->entityList[0].entityId,
                           48,
                           "Double Bit ECC Error (Memory subsystem)",
                           DCGM_HEALTH_RESULT_FAIL,
                           DCGM_FR_XID_ERROR,
                           now + 3s,
                           response,
                           DCGM_HEALTH_WATCH_ALL);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

    return DCGM_ST_OK;
}

namespace
{
dcgmReturn_t TestSingleFabricManagerStatus(dcgmHandle_t handle,
                                           dcgmGpuGrp_t gpuGroup,
                                           unsigned int const gpuId,
                                           int64_t const statusValue,
                                           char const *const statusDesc,
                                           dcgmHealthWatchResults_t const expectedHealth,
                                           unsigned int const expectedIncidentCount,
                                           long long const timestamp)
{
    dcgmInjectFieldValue_t fv = {};
    fv.version                = dcgmInjectFieldValue_version;
    fv.fieldId                = DCGM_FI_DEV_FABRIC_MANAGER_STATUS;
    fv.fieldType              = DCGM_FT_INT64;
    fv.status                 = DCGM_ST_OK;
    fv.value.i64              = statusValue;
    fv.ts                     = timestamp;

    dcgmReturn_t ret = dcgmInjectFieldValue(handle, gpuId, &fv);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmInjectFieldValue returned %d for %s\n", (int)ret, statusDesc);
        return ret;
    }

    auto response     = MakeUniqueZero<dcgmHealthResponse_t>();
    response->version = dcgmHealthResponse_version;

    ret = dcgmHealthCheck(handle, gpuGroup, response.get());
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmHealthCheck returned %d for %s\n", (int)ret, statusDesc);
        return ret;
    }

    if (response->incidentCount != expectedIncidentCount)
    {
        fprintf(stderr,
                "Expected %u incidents for %s, got %u\n",
                expectedIncidentCount,
                statusDesc,
                response->incidentCount);
        ret = DCGM_ST_GENERIC_ERROR;
    }

    if (ret == DCGM_ST_OK && response->overallHealth != expectedHealth)
    {
        fprintf(stderr,
                "Expected health %d for %s, got %d\n",
                (int)expectedHealth,
                statusDesc,
                (int)response->overallHealth);
        ret = DCGM_ST_GENERIC_ERROR;
    }

    if (ret == DCGM_ST_OK && expectedIncidentCount > 0)
    {
        if (response->incidents[0].system != DCGM_HEALTH_WATCH_NVLINK)
        {
            fprintf(stderr,
                    "Expected NVLINK system for %s incident, got %d\n",
                    statusDesc,
                    (int)response->incidents[0].system);
            ret = DCGM_ST_GENERIC_ERROR;
        }

        if (ret == DCGM_ST_OK && response->incidents[0].health != expectedHealth)
        {
            fprintf(stderr,
                    "Expected health %d for %s incident, got %d\n",
                    (int)expectedHealth,
                    statusDesc,
                    (int)response->incidents[0].health);
            ret = DCGM_ST_GENERIC_ERROR;
        }

        if (ret == DCGM_ST_OK && response->incidents[0].error.code != DCGM_FR_FABRIC_PROBE_STATE)
        {
            fprintf(stderr,
                    "Expected error code %d (DCGM_FR_FABRIC_PROBE_STATE) for %s incident, got %d\n",
                    DCGM_FR_FABRIC_PROBE_STATE,
                    statusDesc,
                    response->incidents[0].error.code);
            ret = DCGM_ST_GENERIC_ERROR;
        }
    }

    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "TestSingleFabricManagerStatus failed for %s\n", statusDesc);
        for (auto const &incident :
             std::span(response->incidents,
                       std::min(static_cast<size_t>(response->incidentCount), std::size(response->incidents))))
        {
            fprintf(stderr,
                    "  Incident: system=%d health=%d entityId=%u msg='%s'\n",
                    incident.system,
                    incident.health,
                    incident.entityInfo.entityId,
                    incident.error.msg);
        }
        return ret;
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t ClearNvlinkFields(dcgmHandle_t handle, unsigned int const gpuId, long long const timestamp)
{
    std::array<std::pair<unsigned short, int64_t>, 8> const nvlinkFields = { {
        { DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL, 0 },
        { DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL, 0 },
        { DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL, 0 },
        { DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL, 0 },
        { DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS, 0 },
        { DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER, DCGM_INT64_BLANK },
        { DCGM_FI_DEV_FABRIC_HEALTH_MASK, 0 },
        { DCGM_FI_DEV_FABRIC_MANAGER_STATUS, DCGM_INT64_BLANK },
    } };

    for (auto const &[fieldId, healthyValue] : nvlinkFields)
    {
        dcgmInjectFieldValue_t fv = {};
        fv.version                = dcgmInjectFieldValue_version;
        fv.fieldId                = fieldId;
        fv.fieldType              = DCGM_FT_INT64;
        fv.status                 = DCGM_ST_OK;
        fv.value.i64              = healthyValue;
        fv.ts                     = timestamp;

        dcgmReturn_t ret = dcgmInjectFieldValue(handle, gpuId, &fv);
        if (ret != DCGM_ST_OK)
        {
            fprintf(stderr, "Failed to clear field %u: %d\n", fieldId, (int)ret);
            return ret;
        }
    }

    return DCGM_ST_OK;
}
} // namespace

int TestHealthMonitor::TestHMCheckResidual()
{
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_ALL);

    auto result = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, newSystems);
    if (result != DCGM_ST_OK)
    {
        fmt::print(stderr, "dcgmHealthSet failed: {}\n", errorString(result));
        return result;
    }

    // Allow time for field updates
    result = dcgmUpdateAllFields(m_dcgmHandle, 1);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fmt::print(stderr, "dcgmUpdateAllFields failed: {}\n", errorString(result));
        // Continue - dcgmHealthCheck may still work
    }

    auto response     = MakeUniqueZero<dcgmHealthResponse_t>();
    response->version = dcgmHealthResponse_version;

    result = dcgmHealthCheck(m_dcgmHandle, m_gpuGroup, response.get());
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fmt::print(stderr, "dcgmHealthCheck failed: {}\n", errorString(result));
        dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));
        return result;
    }

    // Filter out XID-related incidents since XID history cannot be cleared through public API
    std::vector<size_t> nonXidIndices;

    auto incidents = std::span<dcgmIncidentInfo_t>(
        response->incidents, std::min(static_cast<size_t>(response->incidentCount), std::size(response->incidents)));
    auto nonXidIncidents = std::ranges::filter_view(incidents, [](auto const &incident) {
        return incident.error.code != DCGM_FR_XID_ERROR && incident.error.code != DCGM_FR_FALLEN_OFF_BUS
               && incident.error.code != DCGM_FR_UNCONTAINED_ERROR;
    });

    if (!nonXidIncidents.empty())
    {
        fmt::print(stderr,
                   "Residual health check FAILED: overallHealth={}, totalIncidents={}, nonXidIncidents={}\n",
                   static_cast<int>(response->overallHealth),
                   response->incidentCount,
                   std::ranges::distance(nonXidIncidents));

        fmt::print(stderr, "Previous tests may have left unhealthy field values in cache\n");

        size_t idx = 0;
        for (auto const &incident : nonXidIncidents)
        {
            fmt::print(stderr,
                       "  Non-XID Incident {}: system={}, health={}, entity={}/{}, error.code={}, msg='{}'\n",
                       idx++,
                       static_cast<int>(incident.system),
                       static_cast<int>(incident.health),
                       static_cast<int>(incident.entityInfo.entityGroupId),
                       incident.entityInfo.entityId,
                       static_cast<int>(incident.error.code),
                       incident.error.msg);
        }

        dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));
        return DCGM_ST_GENERIC_ERROR;
    }

    // If we only have XID incidents, report them but don't fail
    if (response->incidentCount > 0)
    {
        fmt::print("Residual health check: {} XID incidents ignored (cannot be cleared via public API)\n",
                   response->incidentCount);
    }

    // Cleanup: Clear health watches
    dcgmHealthSet(m_dcgmHandle, m_gpuGroup, static_cast<dcgmHealthSystems_t>(0));

    return DCGM_ST_OK;
}

/*************************************************************************/
int TestHealthMonitor::TestHMCheckFabricManagerStatus()
{
    dcgmReturn_t ret = dcgmHealthSet(m_dcgmHandle, m_gpuGroup, DCGM_HEALTH_WATCH_NVLINK);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmHealthSet returned %d\n", (int)ret);
        return -1;
    }

    if (m_gpus.empty())
    {
        fprintf(stderr, "No GPUs available for testing\n");
        return -1;
    }

    unsigned int const gpuId  = m_gpus[0];
    auto const now            = Now();
    long long timestampOffset = 10;

    ret = ClearNvlinkFields(m_dcgmHandle, gpuId, ToLegacyTimestamp(now + std::chrono::seconds(timestampOffset)));
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "ClearNvlinkFields failed: %d\n", (int)ret);
        return -1;
    }
    timestampOffset += 10;

    struct TestCase
    {
        int64_t statusValue;
        char const *statusDesc;
        dcgmHealthWatchResults_t expectedHealth;
        unsigned int expectedIncidentCount;
    };

    TestCase constexpr testCases[] = {
        { 3, "Success", DCGM_HEALTH_RESULT_PASS, 0 },    { 0, "NotSupported", DCGM_HEALTH_RESULT_PASS, 0 },
        { 2, "InProgress", DCGM_HEALTH_RESULT_WARN, 1 }, { 1, "NotStarted", DCGM_HEALTH_RESULT_FAIL, 1 },
        { 4, "Failure", DCGM_HEALTH_RESULT_FAIL, 1 },    { 5, "Unrecognized", DCGM_HEALTH_RESULT_FAIL, 1 },
        { 6, "NvmlTooOld", DCGM_HEALTH_RESULT_FAIL, 1 },
    };

    for (auto const &test : testCases)
    {
        ret = TestSingleFabricManagerStatus(m_dcgmHandle,
                                            m_gpuGroup,
                                            gpuId,
                                            test.statusValue,
                                            test.statusDesc,
                                            test.expectedHealth,
                                            test.expectedIncidentCount,
                                            ToLegacyTimestamp(now + std::chrono::seconds(timestampOffset)));
        if (ret != DCGM_ST_OK)
        {
            return -1;
        }
        timestampOffset += 10;
    }

    ret = ClearNvlinkFields(m_dcgmHandle, gpuId, ToLegacyTimestamp(now + std::chrono::seconds(timestampOffset)));
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Final ClearNvlinkFields failed: %d\n", (int)ret);
        return -1;
    }

    return 0;
}
