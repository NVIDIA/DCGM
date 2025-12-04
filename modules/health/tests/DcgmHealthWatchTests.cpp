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

#include <DcgmHealthResponse.h>
#include <DcgmHealthWatch.h>
#include <DcgmStringHelpers.h>
#include <UniquePtrUtil.h>
#include <catch2/catch_all.hpp>
#include <dcgm_structs.h>
#include <mock/FileHandleMock.h>

// Mock callback function for testing
dcgmReturn_t mockPostCallback(dcgm_module_command_header_t *, void *)
{
    return DCGM_ST_OK;
}

// Test helper class that has friend access to DcgmHealthWatch
class DcgmHealthWatchTestHelper
{
public:
    static void InjectXidForTesting(DcgmHealthWatch &healthWatch, uint64_t xid, dcgm_field_eid_t gpuId)
    {
        DcgmLockGuard dlg(healthWatch.m_mutex);
        healthWatch.m_gpuXidHistory[xid].insert(gpuId);
    }

    static void ClearXidHistoryForTesting(DcgmHealthWatch &healthWatch)
    {
        DcgmLockGuard dlg(healthWatch.m_mutex);
        healthWatch.m_gpuXidHistory.clear();
    }

    static void MonitorDevastatingXids(DcgmHealthWatch &healthWatch,
                                       dcgm_field_entity_group_t entityGroupId,
                                       dcgm_field_eid_t entityId,
                                       DcgmHealthResponse &response)
    {
        healthWatch.MonitorDevastatingXids(entityGroupId, entityId, response);
    }

    static void MonitorSubsystemXids(DcgmHealthWatch &healthWatch,
                                     dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId,
                                     dcgmHealthSystems_t system,
                                     DcgmHealthResponse &response)
    {
        healthWatch.MonitorSubsystemXids(entityGroupId, entityId, system, response);
    }

    static void OnFieldValuesUpdate(DcgmHealthWatch &healthWatch, DcgmFvBuffer *fvBuffer)
    {
        healthWatch.OnFieldValuesUpdate(fvBuffer);
    }
};

// Test fixture class for common setup
class DcgmHealthWatchFixture
{
public:
    dcgmCoreCallbacks_t callbacks;
    DcgmHealthWatch healthWatch;
    DcgmHealthResponse response;
    const dcgm_field_eid_t testGpuId                  = 0;
    const dcgm_field_entity_group_t testEntityGroupId = DCGM_FE_GPU;

    DcgmHealthWatchFixture()
        : callbacks(createCallbacks())
        , healthWatch(callbacks)
    {
        // Clear any previous state and reset response
        DcgmHealthWatchTestHelper::ClearXidHistoryForTesting(healthWatch);
        response = DcgmHealthResponse();
    }

private:
    static dcgmCoreCallbacks_t createCallbacks()
    {
        dcgmCoreCallbacks_t callbacks = {};
        callbacks.version             = dcgmCoreCallbacks_version;
        callbacks.postfunc            = mockPostCallback;
        callbacks.poster              = nullptr;
        callbacks.loggerfunc          = nullptr;
        return callbacks;
    }
};

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorDevastatingXids")
{
    SECTION("No devastating XIDs detected")
    {
        // Don't inject any XIDs
        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify no incidents were added
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("Multiple devastating XIDs detected")
    {
        // Inject multiple devastating XIDs for the test GPU
        DcgmHealthWatchTestHelper::InjectXidForTesting(healthWatch, 48, testGpuId);  // Double Bit ECC Error
        DcgmHealthWatchTestHelper::InjectXidForTesting(healthWatch, 95, testGpuId);  // Uncontained Error
        DcgmHealthWatchTestHelper::InjectXidForTesting(healthWatch, 140, testGpuId); // ECC unrecovered error

        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify incidents were added
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 3);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL);

        // All incidents should be for the same GPU
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            REQUIRE(healthResponse.incidents[i].system == DCGM_HEALTH_WATCH_ALL);
            REQUIRE(healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_FAIL);
            REQUIRE(healthResponse.incidents[i].entityInfo.entityGroupId == testEntityGroupId);
            REQUIRE(healthResponse.incidents[i].entityInfo.entityId == testGpuId);
        }
    }

    SECTION("Non Devastating XIDs - not detected in response")
    {
        // Inject multiple devastating XIDs for the test GPU
        DcgmHealthWatchTestHelper::InjectXidForTesting(healthWatch, 31, testGpuId);

        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify incidents were added
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::Complete XID Flow Test")
{
    SECTION("Complete flow: Devastating XID processing and monitoring")
    {
        // Set up health monitoring for the group
        dcgmHealthSystems_t systems = DCGM_HEALTH_WATCH_ALL;
        dcgmReturn_t ret            = healthWatch.SetWatches(1, systems, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        // Create a field value buffer with XID data
        DcgmFvBuffer fvBuffer;
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERRORS,
                               48,
                               timelib_usecSince1970(),
                               DCGM_ST_OK); // Double Bit ECC Error - devastating XID

        // Process the field value update and monitor
        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify the response
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_ALL);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityGroupId == testEntityGroupId);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityId == testGpuId);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_XID_ERROR);
    }

    SECTION("Complete flow: Subsystem XID processing and monitoring")
    {
        // Set up health monitoring for memory subsystem
        dcgmHealthSystems_t systems = DCGM_HEALTH_WATCH_MEM;
        dcgmReturn_t ret            = healthWatch.SetWatches(1, systems, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        // Create a field value buffer with subsystem XID data
        DcgmFvBuffer fvBuffer;
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERRORS,
                               31,
                               timelib_usecSince1970(),
                               DCGM_ST_OK); // MMU Error - memory subsystem XID

        // Process the field value update and monitor
        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorSubsystemXids(
            healthWatch, testEntityGroupId, testGpuId, DCGM_HEALTH_WATCH_MEM, response);

        // Verify the response
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityGroupId == testEntityGroupId);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityId == testGpuId);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_XID_ERROR);
    }

    SECTION("Complete flow: Multiple XIDs processing")
    {
        // Set up health monitoring for all systems
        dcgmHealthSystems_t systems
            = static_cast<dcgmHealthSystems_t>(DCGM_HEALTH_WATCH_ALL | DCGM_HEALTH_WATCH_MEM | DCGM_HEALTH_WATCH_PCIE);
        dcgmReturn_t ret = healthWatch.SetWatches(1, systems, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        // Create field value buffer with multiple XIDs
        DcgmFvBuffer fvBuffer;

        // Add devastating XID
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERRORS,
                               48,
                               timelib_usecSince1970(),
                               DCGM_ST_OK); // Double Bit ECC Error

        // Add memory subsystem XID
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERRORS,
                               31,
                               timelib_usecSince1970() + 1000,
                               DCGM_ST_OK); // MMU Error

        // Add PCIe subsystem XID
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERRORS,
                               38,
                               timelib_usecSince1970() + 2000,
                               DCGM_ST_OK); // PCIe Bus Error

        // Process all field value updates and monitor
        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Monitor subsystem XIDs
        DcgmHealthWatchTestHelper::MonitorSubsystemXids(
            healthWatch, testEntityGroupId, testGpuId, DCGM_HEALTH_WATCH_MEM, response);
        DcgmHealthWatchTestHelper::MonitorSubsystemXids(
            healthWatch, testEntityGroupId, testGpuId, DCGM_HEALTH_WATCH_PCIE, response);

        // Verify the response
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);

        // Should have 3 incidents: 1 devastating + 2 subsystem
        REQUIRE(healthResponse.incidentCount == 3);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL); // FAIL overrides WARN

        // Verify all incidents are for the same GPU
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            REQUIRE(healthResponse.incidents[i].entityInfo.entityGroupId == testEntityGroupId);
            REQUIRE(healthResponse.incidents[i].entityInfo.entityId == testGpuId);
        }
    }

    SECTION("Complete flow: Unknown XID ignored")
    {
        // Set up health monitoring
        dcgmHealthSystems_t systems = DCGM_HEALTH_WATCH_ALL;
        dcgmReturn_t ret            = healthWatch.SetWatches(1, systems, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        // Create field value buffer with unknown XID
        DcgmFvBuffer fvBuffer;
        fvBuffer.AddInt64Value(
            DCGM_FE_GPU, testGpuId, DCGM_FI_DEV_XID_ERRORS, 999, timelib_usecSince1970(), DCGM_ST_OK); // Unknown XID

        // Process the field value update and monitor
        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify no incidents were added
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }
}