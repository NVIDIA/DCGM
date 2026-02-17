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

#include <DcgmHealthResponse.h>
#include <DcgmHealthWatch.h>
#include <DcgmImexManager.h>
#include <DcgmStringHelpers.h>
#include <UniquePtrUtil.h>
#include <catch2/catch_all.hpp>
#include <dcgm_core_communication.h>
#include <dcgm_core_structs.h>
#include <dcgm_structs.h>
#include <mock/FileHandleMock.h>
#include <unordered_map>

/**
 * Mock cache context for simulating DcgmCacheManager responses
 */
struct MockCacheContext
{
    std::unordered_map<unsigned short, dcgmReturn_t> fieldReturnCodes; //!< Configured return codes per field ID
    std::unordered_map<unsigned short, dcgmcm_sample_t> fieldSamples;  //!< Configured sample data per field ID
    std::string managedString;                                         //!< Managed string to ensure char* validity

    /**
     * Configures a field to return specific data when queried via GetLatestSample.
     *
     * @param fieldId Field identifier to configure
     * @param retCode Return code for the field query (e.g., DCGM_ST_OK, DCGM_ST_NOT_WATCHED)
     * @param sample Sample data to return if retCode is DCGM_ST_OK
     */
    void SetFieldSample(unsigned short fieldId, dcgmReturn_t retCode, dcgmcm_sample_t sample = {})
    {
        fieldReturnCodes[fieldId] = retCode;
        if (retCode == DCGM_ST_OK)
        {
            fieldSamples[fieldId] = sample;
        }
    }

    /**
     * Adds a string to the managed strings vector.
     *
     * @param str String to add
     */
    void AddManagedString(std::string_view str)
    {
        managedString = std::string(str);
    }

    /**
     * Returns the managed string.
     *
     * @return View of the managed string
     */
    std::string_view ManagedString() const
    {
        return managedString;
    }
};

/**
 * Mock postfunc callback - returns configured response
 *
 * @param header Command header containing request details
 * @param poster Pointer to MockCacheContext with configured responses
 * @return DCGM_ST_OK on success, DCGM_ST_BADPARAM if header is null
 */
static dcgmReturn_t MockPostWithCache(dcgm_module_command_header_t *header, void *poster)
{
    if (!header)
    {
        return DCGM_ST_BADPARAM;
    }

    auto *context = static_cast<MockCacheContext *>(poster);

    // Handle GetLatestSample requests
    if (header->subCommand == DcgmCoreReqIdCMGetLatestSample)
    {
        auto *gls              = reinterpret_cast<dcgmCoreGetLatestSample_t *>(header);
        unsigned short fieldId = gls->request.fieldId;

        // Check if we have a configured response for this field
        if (context)
        {
            auto retIt = context->fieldReturnCodes.find(fieldId);
            if (retIt != context->fieldReturnCodes.end())
            {
                gls->response.ret = retIt->second;

                // If returning OK, populate the sample
                if (retIt->second == DCGM_ST_OK && gls->request.populateSamples)
                {
                    auto sampleIt = context->fieldSamples.find(fieldId);
                    if (sampleIt != context->fieldSamples.end())
                    {
                        gls->response.sample = sampleIt->second;
                    }
                }
                return DCGM_ST_OK;
            }
        }

        // Default: return NOT_WATCHED for unconfigured fields
        gls->response.ret = DCGM_ST_NOT_WATCHED;
        return DCGM_ST_OK;
    }

    // Other requests: return OK
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

    static dcgmReturn_t MonitorGlobalHealthChecks(DcgmHealthWatch &healthWatch,
                                                  dcgmHealthSystems_t healthSystemsMask,
                                                  DcgmHealthResponse &response)
    {
        return healthWatch.MonitorGlobalHealthChecks(healthSystemsMask, response);
    }
};

// Test fixture class for common setup
class DcgmHealthWatchFixture
{
public:
    MockCacheContext mockCache;
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
    dcgmCoreCallbacks_t createCallbacks()
    {
        dcgmCoreCallbacks_t cb = {};
        cb.version             = dcgmCoreCallbacks_version;
        cb.postfunc            = MockPostWithCache;
        cb.poster              = &mockCache; // Pass context for mock to use
        cb.loggerfunc          = [](void const *) { /* do nothing */ };
        return cb;
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

/**
 * Tests for MonitorGlobalHealthChecks
 *
 * Verifies that global health checks are correctly dispatched based on the health systems mask.
 * Tests that checks are skipped when the corresponding watch is disabled.
 */
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorGlobalHealthChecks dispatch")
{
    struct TestCase
    {
        char const *name;
        dcgmHealthSystems_t mask;
    };

    auto test = GENERATE(TestCase { "NVLINK watch disabled - skips MonitorImex", DCGM_HEALTH_WATCH_MEM },
                         TestCase { "NVLINK watch enabled, no cache data - returns OK", DCGM_HEALTH_WATCH_NVLINK },
                         TestCase { "No watches enabled - no checks performed", static_cast<dcgmHealthSystems_t>(0) });

    DYNAMIC_SECTION(test.name)
    {
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorGlobalHealthChecks(healthWatch, test.mask, response);

        REQUIRE(ret == DCGM_ST_OK);
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }
}

namespace DcgmNs::ImexTests
{
/**
 * Configures IMEX domain status field.
 *
 * @param ctx MockCacheContext to configure
 * @param status Domain status string (e.g., "UP", "DOWN", "DEGRADED", "NOT_INSTALLED")
 */
void SetImexDomainStatus(MockCacheContext &ctx, char const *status)
{
    ctx.AddManagedString(status);

    dcgmcm_sample_t sample = {};
    sample.timestamp       = timelib_usecSince1970();
    sample.val.str         = const_cast<char *>(ctx.ManagedString().data());

    ctx.SetFieldSample(DCGM_FI_IMEX_DOMAIN_STATUS, DCGM_ST_OK, sample);
}

/**
 * Configures IMEX daemon status field.
 *
 * @param ctx MockCacheContext to configure
 * @param status Daemon status value (DcgmImexDaemonStatus enum or integral type for invalid values)
 */
template <typename T>
    requires std::same_as<T, DcgmImexDaemonStatus> || std::integral<T>
void SetImexDaemonStatus(MockCacheContext &ctx, T status)
{
    dcgmcm_sample_t sample = {};
    sample.timestamp       = timelib_usecSince1970();
    sample.val.i64         = static_cast<int64_t>(status);
    ctx.SetFieldSample(DCGM_FI_IMEX_DAEMON_STATUS, DCGM_ST_OK, sample);
}

/**
 * Helper to run MonitorGlobalHealthChecks and validate common response patterns.
 *
 * @param healthWatch Health watch instance
 * @param response Response object to populate
 * @param expectedRet Expected return code from MonitorGlobalHealthChecks
 * @param expectedIncidents Expected number of incidents (only checked if expectedRet == DCGM_ST_OK)
 * @param expectedHealth Expected overall health result (only checked if expectedRet == DCGM_ST_OK)
 * @param expectedErrorCode Expected error code for first incident (only checked if expectedIncidents > 0)
 */
void ValidateImexResponse(DcgmHealthWatch &healthWatch,
                          DcgmHealthResponse &response,
                          dcgmReturn_t expectedRet,
                          unsigned int expectedIncidents          = 0,
                          dcgmHealthWatchResults_t expectedHealth = DCGM_HEALTH_RESULT_PASS,
                          dcgmError_t expectedErrorCode           = DCGM_FR_OK)
{
    dcgmReturn_t ret
        = DcgmHealthWatchTestHelper::MonitorGlobalHealthChecks(healthWatch, DCGM_HEALTH_WATCH_NVLINK, response);

    REQUIRE(ret == expectedRet);

    if (expectedRet == DCGM_ST_OK)
    {
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == expectedIncidents);
        REQUIRE(healthResponse.overallHealth == expectedHealth);

        if (expectedIncidents > 0 && expectedErrorCode != DCGM_FR_OK)
        {
            REQUIRE(healthResponse.incidents[0].error.code == expectedErrorCode);
        }
    }
}

/**
 * Test case structure for IMEX health checks
 */
struct ImexTest
{
    char const *name;
    char const *domain                    = "UP";
    int64_t daemon                        = static_cast<int64_t>(DcgmImexDaemonStatus::READY);
    std::optional<dcgmReturn_t> domainErr = std::nullopt;
    std::optional<dcgmReturn_t> daemonErr = std::nullopt;
    bool expectFail                       = false;
    dcgmReturn_t expectRet                = DCGM_ST_OK;
};

/**
 * Factory function for PASS test cases.
 */
auto Pass(char const *name, char const *domain, int64_t daemon) -> ImexTest
{
    return { name, domain, daemon };
}

/**
 * Factory function for FAIL test cases (expects DCGM_FR_IMEX_UNHEALTHY).
 */
auto Fail(char const *name, char const *domain, int64_t daemon) -> ImexTest
{
    return { name, domain, daemon, std::nullopt, std::nullopt, true };
}
} //namespace DcgmNs::ImexTests

/**
 * Tests for MonitorImex logic.
 */
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorImex")
{
    using namespace DcgmNs::ImexTests;

    // Run once (before GENERATE)
    SECTION("Domain unhealthy and daemon returns error")
    {
        SetImexDomainStatus(mockCache, "DOWN");
        mockCache.SetFieldSample(DCGM_FI_IMEX_DAEMON_STATUS, DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGlobalHealthChecks(healthWatch, DCGM_HEALTH_WATCH_NVLINK, response);

        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
        // Need to verify incident was recorded despite error return
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
    }

    auto test
        = GENERATE(Pass("Healthy IMEX state", "UP", static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Fail("Domain DOWN", "DOWN", static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Fail("Domain DEGRADED", "DEGRADED", static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Fail("Daemon INITIALIZING", "UP", static_cast<int64_t>(DcgmImexDaemonStatus::INITIALIZING)),
                   Pass("NOT_INSTALLED", "NOT_INSTALLED", static_cast<int64_t>(DcgmImexDaemonStatus::NOT_INSTALLED)),
                   Pass("UNAVAILABLE", "UNAVAILABLE", static_cast<int64_t>(DcgmImexDaemonStatus::UNAVAILABLE)),
                   Pass("NOT_CONFIGURED", "NOT_CONFIGURED", static_cast<int64_t>(DcgmImexDaemonStatus::NOT_CONFIGURED)),
                   Fail("Invalid domain status", "INVALID_GARBAGE", static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Fail("Invalid daemon status", "UP", 999),
                   Pass("Blank domain status", DCGM_STR_BLANK, static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Pass("Blank daemon status", "UP", DCGM_INT64_BLANK),
                   ImexTest { .name      = "Error retrieving domain",
                              .daemon    = static_cast<int64_t>(DcgmImexDaemonStatus::READY),
                              .domainErr = DCGM_ST_GENERIC_ERROR,
                              .expectRet = DCGM_ST_GENERIC_ERROR },
                   ImexTest { .name      = "Error retrieving daemon",
                              .domain    = "UP",
                              .daemonErr = DCGM_ST_GENERIC_ERROR,
                              .expectRet = DCGM_ST_GENERIC_ERROR },
                   ImexTest { .name      = "NO_DATA for domain",
                              .daemon    = static_cast<int64_t>(DcgmImexDaemonStatus::READY),
                              .domainErr = DCGM_ST_NO_DATA },
                   ImexTest { .name = "NO_DATA for daemon", .domain = "UP", .daemonErr = DCGM_ST_NO_DATA });

    DYNAMIC_SECTION(test.name)
    {
        if (test.domainErr)
        {
            mockCache.SetFieldSample(DCGM_FI_IMEX_DOMAIN_STATUS, *test.domainErr);
        }
        else
        {
            SetImexDomainStatus(mockCache, test.domain);
        }

        if (test.daemonErr)
        {
            mockCache.SetFieldSample(DCGM_FI_IMEX_DAEMON_STATUS, *test.daemonErr);
        }
        else
        {
            SetImexDaemonStatus(mockCache, test.daemon);
        }

        if (test.expectRet != DCGM_ST_OK)
        {
            ValidateImexResponse(healthWatch, response, test.expectRet);
        }
        else if (test.expectFail)
        {
            ValidateImexResponse(healthWatch, response, DCGM_ST_OK, 1, DCGM_HEALTH_RESULT_FAIL, DCGM_FR_IMEX_UNHEALTHY);
        }
        else
        {
            ValidateImexResponse(healthWatch, response, DCGM_ST_OK, 0, DCGM_HEALTH_RESULT_PASS);
        }
    }
}
