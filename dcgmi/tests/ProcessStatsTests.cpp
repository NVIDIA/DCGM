/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <MockDcgmOps.h>
#include <ProcessStats.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <catch2/catch_all.hpp>

#include <cstring>
#include <memory>
#include <string>

namespace detail
{
/**
 * Populate one PID info record with deterministic values for CLI rendering tests.
 */
inline void FillPidSingleInfo(dcgmPidSingleInfo_t &info, unsigned int gpuId)
{
    info.gpuId                    = gpuId;
    info.energyConsumed           = 9000;
    info.pcieRxBandwidth          = { 1000000, 3000000, 2000000 };
    info.pcieTxBandwidth          = { 2000000, 4000000, 3000000 };
    info.pcieReplays              = 3;
    info.startTime                = 1000000;
    info.endTime                  = 4000000;
    info.maxGpuMemoryUsed         = 123456;
    info.eccDoubleBit             = 2;
    info.numXidCriticalErrors     = 1;
    info.xidCriticalErrorsTs[0]   = 2000000;
    info.numOtherComputePids      = 1;
    info.otherComputePids[0]      = 111;
    info.numOtherGraphicsPids     = 1;
    info.otherGraphicsPids[0]     = 222;
    info.processUtilization       = { 777, 55.0, 66.0 };
    info.smUtilization            = { 10, 30, 20 };
    info.memoryUtilization        = { 20, 40, 30 };
    info.memoryClock              = { 1000, 1200, 1100 };
    info.smClock                  = { 800, 1000, 900 };
    info.powerViolationTime       = 1000000;
    info.thermalViolationTime     = 500000;
    info.reliabilityViolationTime = 0;
    info.boardLimitViolationTime  = 0;
    info.lowUtilizationTime       = 0;
    info.syncBoostTime            = 0;
    info.overallHealth            = DCGM_HEALTH_RESULT_WARN;
    info.incidentCount            = 1;
    info.systems[0].system        = DCGM_HEALTH_WATCH_PCIE;
    info.systems[0].health        = DCGM_HEALTH_RESULT_FAIL;
}

/**
 * Populate one job usage record with deterministic values for CLI rendering tests.
 */
inline void FillJobUsageInfo(dcgmGpuUsageInfo_t &info, unsigned int gpuId)
{
    info.gpuId                  = gpuId;
    info.energyConsumed         = 12000;
    info.powerUsage             = { 50.0, 100.0, 75.0 };
    info.pcieRxBandwidth        = { 1000000, 3000000, 2000000 };
    info.pcieTxBandwidth        = { 2000000, 4000000, 3000000 };
    info.pcieReplays            = 4;
    info.startTime              = 1000000;
    info.endTime                = 5000000;
    info.maxGpuMemoryUsed       = 654321;
    info.eccSingleBit           = 1;
    info.eccDoubleBit           = 2;
    info.numXidCriticalErrors   = 1;
    info.xidCriticalErrorsTs[0] = 3000000;
    info.numComputePids         = 1;
    info.computePidInfo[0]      = { 333, 44.0, 55.0 };
    info.numGraphicsPids        = 1;
    info.graphicsPidInfo[0]     = { 444, 22.0, 33.0 };
    info.smUtilization          = { 10, 30, 20 };
    info.memoryUtilization      = { 20, 40, 30 };
    info.memoryClock            = { 1000, 1200, 1100 };
    info.smClock                = { 800, 1000, 900 };
    info.powerViolationTime     = 1000000;
    info.thermalViolationTime   = 500000;
    info.overallHealth          = DCGM_HEALTH_RESULT_PASS;
    info.incidentCount          = 1;
    info.systems[0].system      = DCGM_HEALTH_WATCH_POWER;
    info.systems[0].health      = DCGM_HEALTH_RESULT_WARN;
}
} //namespace detail

namespace
{
struct ProcessStatsTestFixture
{
    ProcessStatsTestFixture()
        : dcgmOps(&mockDcgmOps)
        , stats(dcgmOps)
    {
        dcgmPidInfo_t pidInfo {};
        pidInfo.version                         = dcgmPidInfo_version;
        mockDcgmOps.m_processStatsState.pidInfo = pidInfo;

        dcgmJobInfo_t jobInfo {};
        jobInfo.version                         = dcgmJobInfo_version;
        mockDcgmOps.m_processStatsState.jobInfo = jobInfo;
    }

    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps;
    ProcessStats stats;
};
} //namespace

TEST_CASE_METHOD(ProcessStatsTestFixture, "ProcessStats::EnableWatches")
{
    auto handle = static_cast<dcgmHandle_t>(0x31);

    SECTION("Success")
    {
        CoutCapture capture;

        CHECK(stats.EnableWatches(handle, 7, 200, 30) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_processStatsState.watchPidCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.lastHandle == handle);
        CHECK(mockDcgmOps.m_processStatsState.lastGroupId == 7);
        CHECK(mockDcgmOps.m_processStatsState.lastUpdateFreq == 200000);
        CHECK(mockDcgmOps.m_processStatsState.lastMaxKeepAge == 30.0);
        CHECK(mockDcgmOps.m_processStatsState.lastMaxKeepSamples == 0);
        CHECK(capture.str().find("Successfully started process watches.") != std::string::npos);
    }

    SECTION("Not configured")
    {
        CoutCapture capture;
        mockDcgmOps.m_processStatsState.watchPidReturn = DCGM_ST_NOT_CONFIGURED;

        CHECK(stats.EnableWatches(handle, 7, 200, 30) == DCGM_ST_NOT_CONFIGURED);
        CHECK(capture.str().find("Error: Unable to enable process watches for group") != std::string::npos);
    }
}

TEST_CASE_METHOD(ProcessStatsTestFixture, "ProcessStats job lifecycle APIs")
{
    auto handle = static_cast<dcgmHandle_t>(0x32);

    SECTION("Start, stop, remove, and remove-all succeed")
    {
        CoutCapture capture;

        CHECK(stats.StartJob(handle, 9, "jobA") == DCGM_ST_OK);
        CHECK(stats.StopJob(handle, "jobA") == DCGM_ST_OK);
        CHECK(stats.RemoveJob(handle, "jobA") == DCGM_ST_OK);
        CHECK(stats.RemoveAllJobs(handle) == DCGM_ST_OK);

        CHECK(mockDcgmOps.m_processStatsState.startCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.stopCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.removeCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.removeAllCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.lastGroupId == 9);
        auto output = capture.str();
        CHECK(output.find("Successfully started recording stats for jobA.") != std::string::npos);
        CHECK(output.find("Successfully stopped recording stats for jobA.") != std::string::npos);
        CHECK(output.find("Successfully removed job jobA.") != std::string::npos);
        CHECK(output.find("Successfully removed all jobs.") != std::string::npos);
    }

    SECTION("Job ID length accepts safe values")
    {
        CoutCapture capture;
        std::string maxSafeJobId(ProcessStats::c_dcgmJobIdBufferLength - 1, 'a');

        CHECK(stats.StartJob(handle, 9, maxSafeJobId) == DCGM_ST_OK);
        CHECK(stats.StopJob(handle, maxSafeJobId) == DCGM_ST_OK);
        CHECK(stats.RemoveJob(handle, maxSafeJobId) == DCGM_ST_OK);
        CHECK(stats.ViewJobStats(handle, maxSafeJobId, false) == DCGM_ST_OK);

        CHECK(mockDcgmOps.m_processStatsState.startCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.stopCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.removeCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.getJobCallCount == 1);
        CHECK(capture.str().find("Job ID too long") == std::string::npos);
    }

    SECTION("Job ID length is validated before calling DCGM")
    {
        CoutCapture capture;
        std::string bufferSizedJobId(ProcessStats::c_dcgmJobIdBufferLength, 'x');

        CHECK(stats.StartJob(handle, 9, bufferSizedJobId) == DCGM_ST_BADPARAM);
        CHECK(stats.StopJob(handle, bufferSizedJobId) == DCGM_ST_BADPARAM);
        CHECK(stats.RemoveJob(handle, bufferSizedJobId) == DCGM_ST_BADPARAM);
        CHECK(stats.ViewJobStats(handle, bufferSizedJobId, false) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.m_processStatsState.startCallCount == 0);
        CHECK(mockDcgmOps.m_processStatsState.stopCallCount == 0);
        CHECK(mockDcgmOps.m_processStatsState.removeCallCount == 0);
        CHECK(mockDcgmOps.m_processStatsState.getJobCallCount == 0);
        CHECK(capture.str().find("Job ID too long") != std::string::npos);
    }

    SECTION("Errors are reported")
    {
        mockDcgmOps.m_processStatsState.startReturn     = DCGM_ST_DUPLICATE_KEY;
        mockDcgmOps.m_processStatsState.stopReturn      = DCGM_ST_NO_DATA;
        mockDcgmOps.m_processStatsState.removeReturn    = DCGM_ST_NO_DATA;
        mockDcgmOps.m_processStatsState.removeAllReturn = DCGM_ST_BADPARAM;

        {
            CoutCapture capture;

            CHECK(stats.StartJob(handle, 9, "jobA") == DCGM_ST_DUPLICATE_KEY);
            CHECK(capture.str().find("This Job Id is in use") != std::string::npos);
        }
        {
            CoutCapture capture;

            CHECK(stats.StopJob(handle, "jobA") == DCGM_ST_NO_DATA);
            CHECK(capture.str().find("Error: Job jobA was not found.") != std::string::npos);
        }
        {
            CoutCapture capture;

            CHECK(stats.RemoveJob(handle, "jobA") == DCGM_ST_NO_DATA);
            CHECK(capture.str().find("Error: Job jobA was not found.") != std::string::npos);
        }
        {
            CoutCapture capture;

            CHECK(stats.RemoveAllJobs(handle) == DCGM_ST_BADPARAM);
            CHECK(capture.str().find("Error: Unable to remove jobs") != std::string::npos);
        }
    }
}

TEST_CASE_METHOD(ProcessStatsTestFixture, "ProcessStats::ViewProcessStats")
{
    auto handle = static_cast<dcgmHandle_t>(0x33);

    dcgmPidInfo_t pidInfo {};
    pidInfo.version = dcgmPidInfo_version;
    pidInfo.pid     = 1234;
    pidInfo.numGpus = 1;
    detail::FillPidSingleInfo(pidInfo.summary, 0);
    detail::FillPidSingleInfo(pidInfo.gpus[0], 3);
    mockDcgmOps.m_processStatsState.pidInfo = pidInfo;

    SECTION("Summary output")
    {
        CoutCapture capture;

        CHECK(stats.ViewProcessStats(handle, 7, 1234, false) == DCGM_ST_OK);
        auto output = capture.str();
        CHECK(output.find("Successfully retrieved process info for PID: 1234") != std::string::npos);
        CHECK(output.find("Summary") != std::string::npos);
        CHECK(output.find("Execution Stats") != std::string::npos);
        CHECK(mockDcgmOps.m_processStatsState.getPidCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.lastGroupId == 7);
    }

    SECTION("Verbose output")
    {
        CoutCapture capture;

        CHECK(stats.ViewProcessStats(handle, 7, 1234, true) == DCGM_ST_OK);
        auto output = capture.str();
        CHECK(output.find("GPU ID: 3") != std::string::npos);
        CHECK(output.find("Process Utilization") != std::string::npos);
    }

    SECTION("Not watched")
    {
        CoutCapture capture;
        mockDcgmOps.m_processStatsState.getPidReturn = DCGM_ST_NOT_WATCHED;

        CHECK(stats.ViewProcessStats(handle, 7, 1234, false) == DCGM_ST_NOT_WATCHED);
        CHECK(capture.str().find("Watches have not been enabled") != std::string::npos);
    }

    SECTION("Group not configured")
    {
        CoutCapture capture;
        mockDcgmOps.m_processStatsState.getPidReturn = DCGM_ST_NOT_CONFIGURED;

        CHECK(stats.ViewProcessStats(handle, 7, 1234, false) == DCGM_ST_NOT_CONFIGURED);
        CHECK(capture.str().find("The Group is not found") != std::string::npos);
    }
}

TEST_CASE_METHOD(ProcessStatsTestFixture, "ProcessStats::ViewJobStats")
{
    auto handle = static_cast<dcgmHandle_t>(0x34);

    dcgmJobInfo_t jobInfo {};
    jobInfo.version = dcgmJobInfo_version;
    jobInfo.numGpus = 1;
    detail::FillJobUsageInfo(jobInfo.summary, 0);
    detail::FillJobUsageInfo(jobInfo.gpus[0], 4);
    mockDcgmOps.m_processStatsState.jobInfo = jobInfo;

    SECTION("Summary output")
    {
        CoutCapture capture;

        CHECK(stats.ViewJobStats(handle, "jobA", false) == DCGM_ST_OK);
        auto output = capture.str();
        CHECK(output.find("Successfully retrieved statistics for job: jobA") != std::string::npos);
        CHECK(output.find("Summary") != std::string::npos);
        CHECK(output.find("Execution Stats") != std::string::npos);
        CHECK(mockDcgmOps.m_processStatsState.getJobCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.lastJobId == "jobA");
    }

    SECTION("Verbose output")
    {
        CoutCapture capture;

        CHECK(stats.ViewJobStats(handle, "jobA", true) == DCGM_ST_OK);
        auto output = capture.str();
        CHECK(output.find("GPU ID: 4") != std::string::npos);
        CHECK(output.find("Process Utilization") != std::string::npos);
    }

    SECTION("DCGM error")
    {
        CoutCapture capture;
        mockDcgmOps.m_processStatsState.getJobReturn = DCGM_ST_NO_DATA;

        CHECK(stats.ViewJobStats(handle, "jobA", false) == DCGM_ST_NO_DATA);
        CHECK(capture.str().find("Error: Unable to retrieve job statistics") != std::string::npos);
    }
}
