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
/*
 * ProcessStats.cpp
 *
 *  Created on: Oct 16, 2015
 *      Author: chris
 */

#include "ProcessStats.h"
#include "CommandOutputController.h"
#include "DcgmLogging.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgm_test_apis.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <time.h>

/**************************************************************************************/

/* Process Info */
char STATS_HEADER[] = "+------------------------------------------------------------------------------+\n"
                      "| <HEADER_INFO                                                                >|\n"
                      "+====================================+=========================================+\n";

char STATS_DATA[] = "| <DATA_NAME                        >| <DATA_INFO                            > |\n";

char STATS_FOOTER[] = "+------------------------------------+-----------------------------------------+\n";

char STATS_EXECUTION[] = "|-----  Execution Stats  ------------+-----------------------------------------|\n";

char STATS_EVENTS[] = "+-----  Event Stats  ----------------+-----------------------------------------+\n";

char STATS_PERFORMANCE[] = "+-----  Performance Stats  ----------+-----------------------------------------+\n";

char STATS_SLOWDOWN[] = "+-----  Slowdown Stats  -------------+-----------------------------------------+\n";

char STATS_PROCESS_UTIL[] = "+-----  Process Utilization  --------+-----------------------------------------+\n";

char STATS_COMPUTE_PROCESS_UTIL[]
    = "+--  Compute Process Utilization  ---+-----------------------------------------+\n";

char STATS_GRAPHICS_PROCESS_UTIL[]
    = "+--  Graphics Process Utilization  ---+-----------------------------------------+\n";

char OVERALL_HEALTH[] = "+-----  Overall Health  -------------+-----------------------------------------+\n";


#define PID_TAG         "<PROCESS_ID"
#define HEADER_INFO_TAG "<HEADER_INFO"
#define DATA_NAME_TAG   "<DATA_NAME"
#define DATA_INFO_TAG   "<DATA_INFO"

/*****************************************************************************************/

ProcessStats::ProcessStats()
{
    // TODO Auto-generated constructor stub
}

ProcessStats::~ProcessStats()
{
    // TODO Auto-generated destructor stub
}

/***************************************************************************************/
dcgmReturn_t ProcessStats::EnableWatches(dcgmHandle_t mNvcmHandle,
                                         dcgmGpuGrp_t groupId,
                                         int updateIntervalMs,
                                         int maxKeepAgeSec)
{
    dcgmReturn_t result;

    result = dcgmWatchPidFields(mNvcmHandle, groupId, 1000 * (long long)updateIntervalMs, (double)maxKeepAgeSec, 0);

    if (result != DCGM_ST_OK)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to enable process watches for group. Return: " << error << "." << std::endl;
        PRINT_ERROR("%u %d",
                    "Error starting watches for group with ID: %u. Return: %d",
                    (unsigned int)(uintptr_t)groupId,
                    result);
    }
    else
    {
        std::cout << "Successfully started process watches." << std::endl;
    }

    return result;
}

/***************************************************************************************/
dcgmReturn_t ProcessStats::DisableWatches(dcgmHandle_t mNvcmHandle, dcgmGpuGrp_t groupId)
{
    dcgmReturn_t result;
    dcgmGroupInfo_t stNvcmGroupInfo;

    stNvcmGroupInfo.version = dcgmGroupInfo_version;
    result                  = dcgmGroupGetInfo(mNvcmHandle, groupId, &stNvcmGroupInfo);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to get group information. Return: " << error << std::endl;
        PRINT_ERROR(
            "%u,%d", "Error: GroupGetInfo for GroupId: %u. Return: %d", (unsigned int)(uintptr_t)groupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    std::cout << "Process stats watches cannot be disabled at this time." << std::endl;

    return result;
}

/***************************************************************************************/
dcgmReturn_t ProcessStats::ViewProcessStats(dcgmHandle_t mNvcmHandle,
                                            dcgmGpuGrp_t groupId,
                                            unsigned int pid,
                                            bool verbose)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmPidInfo_t pidInfo;

    memset(&pidInfo, 0, sizeof(pidInfo));

    pidInfo.version = dcgmPidInfo_version;
    pidInfo.pid     = pid;

    result = dcgmGetPidInfo(mNvcmHandle, groupId, &pidInfo);

    if (result == DCGM_ST_NOT_WATCHED)
    {
        std::cout << "Error: Watches have not been enabled. Unable to get PID info." << std::endl;
        PRINT_ERROR("%u %d", "Error getting information for process with pid: %u. Return: %d", pid, result);
        return result;
    }

    if (result != DCGM_ST_OK)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to get information for process with PID " << pid << ". Return: " << error << "."
                  << std::endl;
        PRINT_ERROR("%u %d", "Error getting information for process with pid: %u. Return: %d", pid, result);
        return result;
    }

    // Display Process Info
    std::cout << "Successfully retrieved process info for PID: " << pidInfo.pid << ". Process ran on "
              << pidInfo.numGpus << " GPUs." << std::endl;

    if (verbose)
    {
        for (int i = 0; i < pidInfo.numGpus; i++)
        {
            HelperDisplayPidExecutionStats(&pidInfo.gpus[i], true);
            HelperDisplayPidPerformanceStats(&pidInfo.gpus[i], true);
            HelperDisplayPidEventStats(&pidInfo.gpus[i], true);
            HelperDisplayPidViolationStats(&pidInfo.summary, true);
            HelperDisplayPidProcessUtilization(&pidInfo.gpus[i], true);
            HelperDisplayOverAllHealth(&pidInfo.gpus[i], true);
            std::cout << STATS_FOOTER << std::endl;
        }
    }
    else
    {
        HelperDisplayPidExecutionStats(&pidInfo.summary, false);
        HelperDisplayPidPerformanceStats(&pidInfo.summary, false);
        HelperDisplayPidEventStats(&pidInfo.summary, false);
        HelperDisplayPidViolationStats(&pidInfo.summary, false);
        HelperDisplayOverAllHealth(&pidInfo.summary, false);
        std::cout << STATS_FOOTER << std::endl;
    }

    std::cout
        << "(*) Represents a process statistic. Otherwise device statistic during \n    process lifetime listed.\n"
        << std::endl;

    return result;
}

/***************************************************************************************/
dcgmReturn_t ProcessStats::StartJob(dcgmHandle_t mNvcmHandle, dcgmGpuGrp_t groupId, std::string jobId)
{
    dcgmReturn_t result;

    if (jobId.length() > 64)
    {
        std::cout << "Error: Unable to start job for group. Return: Job ID too long." << std::endl;
        return DCGM_ST_BADPARAM;
    }

    result = dcgmJobStartStats(mNvcmHandle, groupId, (char *)jobId.c_str());

    if (result != DCGM_ST_OK)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)  ? "The Group is not found"
                            : (result == DCGM_ST_DUPLICATE_KEY) ? "This Job Id is in use. Please use another job Id"
                                                                : errorString(result);
        std::cout << "Error: Unable to start job for group. Return: " << error << "." << std::endl;
        PRINT_ERROR("%u %d",
                    "Error starting job watches for group with ID: %u. Return: %d",
                    (unsigned int)(uintptr_t)groupId,
                    result);
    }
    else
    {
        std::cout << "Successfully started recording stats for " << jobId << "." << std::endl;
    }

    return result;
}


/***************************************************************************************/
dcgmReturn_t ProcessStats::StopJob(dcgmHandle_t mNvcmHandle, std::string jobId)
{
    dcgmReturn_t result;

    if (jobId.length() > 64)
    {
        std::cout << "Error: Unable to stop job for group. Return: Job ID too long." << std::endl;
        return DCGM_ST_BADPARAM;
    }

    result = dcgmJobStopStats(mNvcmHandle, (char *)jobId.c_str());

    if (result != DCGM_ST_OK)
    {
        if (result == DCGM_ST_NO_DATA)
            std::cout << "Error: Job " << jobId << " was not found." << std::endl;
        else
            std::cout << "Error: Unable to stop job. Return: " << errorString(result) << "." << std::endl;
        PRINT_ERROR("%d", "Error stoping job watches. Return: %d", result);
    }

    else
    {
        std::cout << "Successfully stopped recording stats for " << jobId << "." << std::endl;
    }

    return result;
}

/***************************************************************************************/
dcgmReturn_t ProcessStats::RemoveJob(dcgmHandle_t mNvcmHandle, std::string jobId)
{
    dcgmReturn_t result;

    if (jobId.length() > 64)
    {
        std::cout << "Error: Unable to stop job for group. Return: Job ID too long." << std::endl;
        return DCGM_ST_BADPARAM;
    }

    result = dcgmJobRemove(mNvcmHandle, (char *)jobId.c_str());
    if (result != DCGM_ST_OK)
    {
        if (result == DCGM_ST_NO_DATA)
            std::cout << "Error: Job " << jobId << " was not found." << std::endl;
        else
            std::cout << "Error: Unable to remove job. Return: " << errorString(result) << "." << std::endl;
        PRINT_ERROR("%d", "Error stoping removing job. Return: %d", result);
    }
    else
    {
        std::cout << "Successfully removed job " << jobId << "." << std::endl;
    }

    return result;
}

/***************************************************************************************/
dcgmReturn_t ProcessStats::RemoveAllJobs(dcgmHandle_t mNvcmHandle)
{
    dcgmReturn_t result;

    result = dcgmJobRemoveAll(mNvcmHandle);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to remove jobs. Return: " << errorString(result) << "." << std::endl;
        PRINT_ERROR("%d", "Error removing jobs. Return: %d", result);
    }
    else
    {
        std::cout << "Successfully removed all jobs." << std::endl;
    }

    return result;
}

/***************************************************************************************/
dcgmReturn_t ProcessStats::ViewJobStats(dcgmHandle_t mNvcmHandle, std::string jobId, bool verbose)
{
    dcgmReturn_t result;
    dcgmJobInfo_t jobInfo;
    jobInfo.version = dcgmJobInfo_version;
    if (jobId.length() > 64)
    {
        std::cout << "Error: Unable to stop job for group. Return: Job ID too long." << std::endl;
        return DCGM_ST_BADPARAM;
    }

    result = dcgmJobGetStats(mNvcmHandle, (char *)jobId.c_str(), &jobInfo);

    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to retrieve job statistics. Return: " << errorString(result) << "." << std::endl;
        PRINT_ERROR("%d", "Error getting job stats. Return: %d", result);
        return result;
    }

    // Display Process Info
    std::cout << "Successfully retrieved statistics for job: " << jobId << ". \n";

    if (verbose)
    {
        for (int i = 0; i < jobInfo.numGpus; i++)
        {
            HelperDisplayJobExecutionStats(&jobInfo.gpus[i], true);
            HelperDisplayJobPerformanceStats(&jobInfo.gpus[i], true);
            HelperDisplayJobEventStats(&jobInfo.gpus[i], true);
            HelperDisplayJobViolationStats(&jobInfo.summary, true);
            HelperDisplayJobProcessUtilization(&jobInfo.gpus[i], true);
            HelperDisplayOverAllHealth(&jobInfo.gpus[i], true);
            std::cout << STATS_FOOTER << std::endl;
        }
    }
    else
    {
        HelperDisplayJobExecutionStats(&jobInfo.summary, false);
        HelperDisplayJobPerformanceStats(&jobInfo.summary, false);
        HelperDisplayJobEventStats(&jobInfo.summary, false);
        HelperDisplayJobViolationStats(&jobInfo.summary, false);
        HelperDisplayOverAllHealth(&jobInfo.summary, false);
        std::cout << STATS_FOOTER << std::endl;
    }

    return result;
}


/***************************************************************************************/
void ProcessStats::HelperDisplayJobExecutionStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose)
{
    CommandOutputController cmdView = CommandOutputController();
    std::stringstream ss;

    // Header

    if (verbose)
    {
        ss << "GPU ID: " << jobInfo->gpuId;
    }
    else
    {
        ss << "Summary";
    }

    cmdView.setDisplayStencil(STATS_HEADER);
    cmdView.addDisplayParameter(HEADER_INFO_TAG, ss.str());
    cmdView.display();
    ss.str("");

    // Body

    cmdView.setDisplayStencil(STATS_DATA);

    std::cout << STATS_EXECUTION;

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Start Time                     ");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimestamp(jobInfo->startTime, true));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "End Time                       ");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimestamp(jobInfo->endTime, false));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Total Execution Time (sec)     ");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimeDifference(jobInfo->startTime, jobInfo->endTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "No. of Processes   ");
    cmdView.addDisplayParameter(DATA_INFO_TAG, jobInfo->numComputePids + jobInfo->numGraphicsPids);
    cmdView.display();
}

/***************************************************************************************/
void ProcessStats::HelperDisplayJobPerformanceStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << STATS_PERFORMANCE;

    cmdView.setDisplayStencil(STATS_DATA);

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Energy Consumed (Joules)");

    cmdView.addDisplayParameter(DATA_INFO_TAG,
                                DCGM_INT64_IS_BLANK(jobInfo->energyConsumed)
                                    ? jobInfo->energyConsumed
                                    : (jobInfo->energyConsumed / 1000)); // 1000 mWs = 1 J
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Power Usage (Watts)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStatFp64Summary(jobInfo->powerUsage));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Max GPU Memory Used (bytes)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, jobInfo->maxGpuMemoryUsed);
    cmdView.display();

    if (verbose)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, "SM Clock (MHz)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStat32Summary(jobInfo->smClock));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "Memory Clock (MHz)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStat32Summary(jobInfo->memoryClock));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "SM Utilization (%)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStat32Summary(jobInfo->smUtilization));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "Memory Utilization (%)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStat32Summary(jobInfo->memoryUtilization));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "PCIe Rx Bandwidth (megabytes)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStatPCISummary(jobInfo->pcieRxBandwidth));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "PCIe Tx Bandwidth (megabytes)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStatPCISummary(jobInfo->pcieTxBandwidth));
        cmdView.display();
    }
    else
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, "Clocks and PCIe Performance");
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Available per GPU in verbose mode");
        cmdView.display();
    }
}

/***************************************************************************************/
void ProcessStats::HelperDisplayJobEventStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose)
{
    CommandOutputController cmdView = CommandOutputController();
    double buffer;

    cmdView.setDisplayStencil(STATS_DATA);

    std::cout << STATS_EVENTS;

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Single Bit ECC Errors");
    cmdView.addDisplayParameter(DATA_INFO_TAG, jobInfo->eccSingleBit);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Double Bit ECC Errors");
    cmdView.addDisplayParameter(DATA_INFO_TAG, jobInfo->eccDoubleBit);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "PCIe Replay Warnings");
    cmdView.addDisplayParameter(DATA_INFO_TAG, jobInfo->pcieReplays);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Critical XID Errors");
    cmdView.addDisplayParameter(DATA_INFO_TAG, jobInfo->numXidCriticalErrors);
    cmdView.display();

    if (verbose)
    {
        for (int i = 0; i < jobInfo->numXidCriticalErrors; i++)
        {
            buffer = ((double)(jobInfo->xidCriticalErrorsTs[i] - jobInfo->startTime) / 10000) / (double)100.0;

            cmdView.addDisplayParameter(DATA_NAME_TAG, (!i) ? "XID (Hr:Min:Sec since start)" : "");
            cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTotalTime(buffer));
            cmdView.display();
        }
    }
}

/***************************************************************************************/
void ProcessStats::HelperDisplayJobViolationStats(dcgmGpuUsageInfo_t *jobInfo, bool /*verbose*/)
{
    CommandOutputController cmdView = CommandOutputController();
    long long totalTime             = (jobInfo->endTime - jobInfo->startTime);

    cmdView.setDisplayStencil(STATS_DATA);

    if (totalTime <= 0)
    {
        totalTime = (unsigned long long)~0 >> 1; // if totalTime is zero make largest value to make all output below 0.
    }

    std::cout << STATS_SLOWDOWN;

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Due to - Power (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, jobInfo->powerViolationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Thermal (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, jobInfo->thermalViolationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Reliability (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, jobInfo->reliabilityViolationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Board Limit (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, jobInfo->boardLimitViolationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Low Utilization (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, jobInfo->lowUtilizationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Sync Boost (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, jobInfo->syncBoostTime));
    cmdView.display();
}

/***************************************************************************************/
void ProcessStats::HelperDisplayJobProcessUtilization(dcgmGpuUsageInfo_t *jobInfo, bool /*verbose*/)
{
    CommandOutputController cmdView = CommandOutputController();
    int sampleIter                  = 0;

    if (jobInfo->numComputePids > 0)
    {
        std::cout << STATS_COMPUTE_PROCESS_UTIL;
        cmdView.setDisplayStencil(STATS_DATA);
        for (sampleIter = 0; sampleIter < jobInfo->numComputePids; sampleIter++)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, "PID");
            cmdView.addDisplayParameter(DATA_INFO_TAG, jobInfo->computePidInfo[sampleIter].pid);
            cmdView.display();

            cmdView.addDisplayParameter(DATA_NAME_TAG, "    Avg SM Utilization (%)");
            if (DCGM_INT32_IS_BLANK(jobInfo->computePidInfo[sampleIter].smUtil))
                cmdView.addDisplayParameter(DATA_INFO_TAG, "Not Found");
            else
                cmdView.addDisplayParameter(DATA_INFO_TAG, (unsigned int)jobInfo->computePidInfo[sampleIter].smUtil);
            cmdView.display();

            cmdView.addDisplayParameter(DATA_NAME_TAG, "    Avg Memory Utilization (%)");
            if (DCGM_INT32_IS_BLANK(jobInfo->computePidInfo[sampleIter].memUtil))
                cmdView.addDisplayParameter(DATA_INFO_TAG, "Not Found");
            else
                cmdView.addDisplayParameter(DATA_INFO_TAG, (unsigned int)jobInfo->computePidInfo[sampleIter].memUtil);
            cmdView.display();
        }
    }

    if (jobInfo->numGraphicsPids > 0)
    {
        std::cout << STATS_GRAPHICS_PROCESS_UTIL;
        cmdView.setDisplayStencil(STATS_DATA);
        for (sampleIter = 0; sampleIter < jobInfo->numGraphicsPids; sampleIter++)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, "PID");
            cmdView.addDisplayParameter(DATA_INFO_TAG, jobInfo->graphicsPidInfo[sampleIter].pid);
            cmdView.display();

            cmdView.addDisplayParameter(DATA_NAME_TAG, "    Avg SM Utilization (%)");
            if (DCGM_INT32_IS_BLANK(jobInfo->graphicsPidInfo[sampleIter].smUtil))
                cmdView.addDisplayParameter(DATA_INFO_TAG, "Not Found");
            else
                cmdView.addDisplayParameter(DATA_INFO_TAG, (unsigned int)jobInfo->graphicsPidInfo[sampleIter].smUtil);
            cmdView.display();

            cmdView.addDisplayParameter(DATA_NAME_TAG, "    Avg Memory Utilization (%)");
            if (DCGM_INT32_IS_BLANK(jobInfo->graphicsPidInfo[sampleIter].memUtil))
                cmdView.addDisplayParameter(DATA_INFO_TAG, "Not Found");
            else
                cmdView.addDisplayParameter(DATA_INFO_TAG, (unsigned int)jobInfo->graphicsPidInfo[sampleIter].memUtil);
            cmdView.display();
        }
    }
}

/***************************************************************************************/
void ProcessStats::HelperDisplayPidProcessUtilization(dcgmPidSingleInfo_t *pidInfo, bool /*verbose*/)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << STATS_PROCESS_UTIL;
    cmdView.setDisplayStencil(STATS_DATA);
    cmdView.addDisplayParameter(DATA_NAME_TAG, "PID");
    cmdView.addDisplayParameter(DATA_INFO_TAG, pidInfo->processUtilization.pid);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "    Avg SM Utilization (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, (unsigned int)pidInfo->processUtilization.smUtil);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "    Avg Memory Utilization (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, (unsigned int)pidInfo->processUtilization.memUtil);
    cmdView.display();
}

/***************************************************************************************/
void ProcessStats::HelperDisplayOverAllHealth(dcgmGpuUsageInfo_t *jobInfo, bool verbose)
{
    CommandOutputController cmdView = CommandOutputController();

    cmdView.setDisplayStencil(STATS_DATA);
    std::cout << OVERALL_HEALTH;

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Overall Health");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperHealthResultToString(jobInfo->overallHealth));
    cmdView.display();

    if (verbose)
    {
        for (unsigned int incident = 0; incident < jobInfo->incidentCount; incident++)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG,
                                        "     " + HelperHealthSystemToString(jobInfo->systems[incident].system));
            cmdView.addDisplayParameter(DATA_INFO_TAG, HelperHealthResultToString(jobInfo->systems[incident].health));
            cmdView.display();
        }
    }
}


/***************************************************************************************/
void ProcessStats::HelperDisplayOverAllHealth(dcgmPidSingleInfo_t *pidInfo, bool verbose)
{
    CommandOutputController cmdView = CommandOutputController();

    cmdView.setDisplayStencil(STATS_DATA);
    std::cout << OVERALL_HEALTH;

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Overall Health");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperHealthResultToString(pidInfo->overallHealth));
    cmdView.display();

    if (verbose)
    {
        for (unsigned int incident = 0; incident < pidInfo->incidentCount; incident++)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG,
                                        "     " + HelperHealthSystemToString(pidInfo->systems[incident].system));
            cmdView.addDisplayParameter(DATA_INFO_TAG, HelperHealthResultToString(pidInfo->systems[incident].health));
            cmdView.display();
        }
    }
}


/***************************************************************************************/
void ProcessStats::HelperDisplayPidExecutionStats(dcgmPidSingleInfo_t *pidInfo, bool verbose)
{
    CommandOutputController cmdView = CommandOutputController();
    std::stringstream ss;

    // Header

    if (verbose)
    {
        ss << "GPU ID: " << pidInfo->gpuId;
    }
    else
    {
        ss << "Summary";
    }

    cmdView.setDisplayStencil(STATS_HEADER);
    cmdView.addDisplayParameter(HEADER_INFO_TAG, ss.str());
    cmdView.display();
    ss.str("");

    // Body

    cmdView.setDisplayStencil(STATS_DATA);

    std::cout << STATS_EXECUTION;

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Start Time                     *");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimestamp(pidInfo->startTime, true));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "End Time                       *");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimestamp(pidInfo->endTime, false));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Total Execution Time (sec)     *");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTimeDifference(pidInfo->startTime, pidInfo->endTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "No. of Conflicting Processes   *");
    cmdView.addDisplayParameter(DATA_INFO_TAG, pidInfo->numOtherComputePids + pidInfo->numOtherGraphicsPids);
    cmdView.display();

    if (verbose)
    {
        for (int i = 0; i < pidInfo->numOtherComputePids; i++)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, (!i) ? "Conflicting Compute PID" : "");
            cmdView.addDisplayParameter(DATA_INFO_TAG, pidInfo->otherComputePids[i]);
            cmdView.display();
        }

        for (int i = 0; i < pidInfo->numOtherGraphicsPids; i++)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, (!i) ? "Conflicting Graphics PID" : "");
            cmdView.addDisplayParameter(DATA_INFO_TAG, pidInfo->otherGraphicsPids[i]);
            cmdView.display();
        }
    }
}

/***************************************************************************************/
void ProcessStats::HelperDisplayPidPerformanceStats(dcgmPidSingleInfo_t *pidInfo, bool verbose)
{
    CommandOutputController cmdView = CommandOutputController();

    std::cout << STATS_PERFORMANCE;

    cmdView.setDisplayStencil(STATS_DATA);

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Energy Consumed (Joules)");

    cmdView.addDisplayParameter(DATA_INFO_TAG,
                                DCGM_INT64_IS_BLANK(pidInfo->energyConsumed)
                                    ? pidInfo->energyConsumed
                                    : (pidInfo->energyConsumed / 1000)); // 1000 mWs = 1 J
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Max GPU Memory Used (bytes)    *");
    cmdView.addDisplayParameter(DATA_INFO_TAG, pidInfo->maxGpuMemoryUsed);
    cmdView.display();

    if (verbose)
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, "SM Clock (MHz)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStat32Summary(pidInfo->smClock));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "Memory Clock (MHz)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStat32Summary(pidInfo->memoryClock));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "SM Utilization (%)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStat32Summary(pidInfo->smUtilization));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "Memory Utilization (%)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStat32Summary(pidInfo->memoryUtilization));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "PCIe Rx Bandwidth (megabytes)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStatPCISummary(pidInfo->pcieRxBandwidth));
        cmdView.display();

        cmdView.addDisplayParameter(DATA_NAME_TAG, "PCIe Tx Bandwidth (megabytes)");
        cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatStatPCISummary(pidInfo->pcieTxBandwidth));
        cmdView.display();
    }
    else
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, "Clocks and PCIe Performance");
        cmdView.addDisplayParameter(DATA_INFO_TAG, "Available per GPU in verbose mode");
        cmdView.display();
    }
}

/***************************************************************************************/
void ProcessStats::HelperDisplayPidEventStats(dcgmPidSingleInfo_t *pidInfo, bool verbose)
{
    CommandOutputController cmdView = CommandOutputController();
    double buffer;

    cmdView.setDisplayStencil(STATS_DATA);

    std::cout << STATS_EVENTS;

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Double Bit ECC Errors");
    cmdView.addDisplayParameter(DATA_INFO_TAG, pidInfo->eccDoubleBit);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "PCIe Replay Warnings");
    cmdView.addDisplayParameter(DATA_INFO_TAG, pidInfo->pcieReplays);
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Critical XID Errors");
    cmdView.addDisplayParameter(DATA_INFO_TAG, pidInfo->numXidCriticalErrors);
    cmdView.display();

    if (verbose)
    {
        for (int i = 0; i < pidInfo->numXidCriticalErrors; i++)
        {
            buffer = ((double)(pidInfo->xidCriticalErrorsTs[i] - pidInfo->startTime) / 10000) / (double)100.0;

            cmdView.addDisplayParameter(DATA_NAME_TAG, (!i) ? "XID (Hr:Min:Sec since start)" : "");
            cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatTotalTime(buffer));
            cmdView.display();
        }
    }
}

/***************************************************************************************/
void ProcessStats::HelperDisplayPidViolationStats(dcgmPidSingleInfo_t *pidInfo, bool /*verbose*/)
{
    CommandOutputController cmdView = CommandOutputController();
    long long totalTime             = (pidInfo->endTime - pidInfo->startTime);

    cmdView.setDisplayStencil(STATS_DATA);

    if (totalTime <= 0)
    {
        totalTime = (unsigned long long)~0 >> 1; // if totalTime is zero make largest value to make all output below 0.
    }

    std::cout << STATS_SLOWDOWN;

    cmdView.addDisplayParameter(DATA_NAME_TAG, "Due to - Power (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, pidInfo->powerViolationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Thermal (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, pidInfo->thermalViolationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Reliability (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, pidInfo->reliabilityViolationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Board Limit (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, pidInfo->boardLimitViolationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Low Utilization (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, pidInfo->lowUtilizationTime));
    cmdView.display();

    cmdView.addDisplayParameter(DATA_NAME_TAG, "       - Sync Boost (%)");
    cmdView.addDisplayParameter(DATA_INFO_TAG, HelperFormatPercentTime(totalTime, pidInfo->syncBoostTime));
    cmdView.display();
}

/***************************************************************************************/
std::string ProcessStats::HelperFormatTimeDifference(long long timestamp1, long long timestamp2)
{
    std::stringstream ss;

    if (DCGM_INT64_IS_BLANK(timestamp1))
        timestamp1 = 0;
    if (DCGM_INT64_IS_BLANK(timestamp2))
        timestamp2 = 0;

    if (!timestamp1)
        return "Has not started";
    if (!timestamp2)
        return "Still Running";

    ss << (((timestamp2 - timestamp1) / 10000) / (double)100.0);
    return ss.str();
}

/***************************************************************************************/
std::string ProcessStats::HelperFormatTimestamp(long long timestamp, bool isStartTime)
{
    std::stringstream ss;

    if (DCGM_INT64_IS_BLANK(timestamp))
    {
        switch (timestamp)
        {
            case DCGM_INT64_BLANK:
                return DCGM_STR_BLANK;

            case DCGM_INT64_NOT_FOUND:
                return DCGM_STR_NOT_FOUND;

            case DCGM_INT64_NOT_SUPPORTED:
                return DCGM_STR_NOT_SUPPORTED;

            case DCGM_INT64_NOT_PERMISSIONED:
                return DCGM_STR_NOT_PERMISSIONED;
            default:
                return DCGM_STR_BLANK;
        }
    }

    if (!timestamp)
    {
        if (isStartTime)
            return "Has Not Started";
        else
            return "Still Running";
    }

    long long temp  = timestamp / 1000000;
    std::string str = ctime((long *)&temp);

    // Remove returned next line character
    str = str.substr(0, str.length() - 1);

    ss << str; //<< ":" << std::setw(4) << std::setfill('0') <<timestamp % 1000000;

    return ss.str();
}

/***************************************************************************************/
std::string ProcessStats::HelperFormatTotalTime(double time)
{
    std::stringstream ss;
    ss << (int)time / 3600 << ":" << std::setw(2) << std::setfill('0') << (int)time / 60 << ":" << std::setw(2)
       << std::setfill('0') << (int)time % 60;

    return ss.str();
}

/***************************************************************************************/
double ProcessStats::HelperFormatPercentTime(long long denominator, long long numerator)
{
    if (denominator == 0)
    {
        return DCGM_FP64_BLANK;
    }

    if (DCGM_INT64_IS_BLANK(numerator))
    {
        switch (numerator)
        {
            case DCGM_INT64_BLANK:
                return DCGM_FP64_BLANK;

            case DCGM_INT64_NOT_FOUND:
                return DCGM_FP64_NOT_FOUND;

            case DCGM_INT64_NOT_SUPPORTED:
                return DCGM_FP64_NOT_SUPPORTED;

            case DCGM_INT64_NOT_PERMISSIONED:
                return DCGM_FP64_NOT_PERMISSIONED;
            default:
                return DCGM_FP64_BLANK;
        }
    }

    return ((((double)numerator * 10000) / denominator) / (double)100);
}

/***************************************************************************************/
std::string ProcessStats::HelperFormatStat32Summary(dcgmStatSummaryInt32_t &summary)
{
    std::stringstream ss;
    std::string temp;
    ss << "Avg: " << Helper32IntToString(summary.average);
    ss << ", Max: " << Helper32IntToString(summary.maxValue);
    ss << ", Min: " << Helper32IntToString(summary.minValue);
    return ss.str();
}

/***************************************************************************************/
std::string ProcessStats::HelperFormatStatPCISummary(dcgmStatSummaryInt64_t &summary)
{
    std::stringstream ss;
    std::string temp;
    ss << "Avg: " << HelperPCILongToString(summary.average);
    ss << ", Max: " << HelperPCILongToString(summary.maxValue);
    ss << ", Min: " << HelperPCILongToString(summary.minValue);
    return ss.str();
}

/***************************************************************************************/
std::string ProcessStats::HelperFormatStatFp64Summary(dcgmStatSummaryFp64_t &summary)
{
    std::stringstream ss;
    std::string temp;
    ss << "Avg: " << HelperDoubleToString(summary.average);
    ss << ", Max: " << HelperDoubleToString(summary.maxValue);
    ss << ", Min: " << HelperDoubleToString(summary.minValue);
    return ss.str();
}

/***************************************************************************************/
std::string ProcessStats::Helper32IntToString(int num)
{
    std::stringstream ss;
    if (DCGM_INT32_IS_BLANK(num))
    {
        return "N/A";
    }
    ss << num;
    return ss.str();
}

/***************************************************************************************/
std::string ProcessStats::HelperDoubleToString(double num)
{
    std::stringstream ss;
    if (DCGM_FP64_IS_BLANK(num))
    {
        return "N/A";
    }
    ss << num;
    return ss.str();
}

/***************************************************************************************/
std::string ProcessStats::HelperPCILongToString(long long num)
{
    std::stringstream ss;
    if (DCGM_INT64_IS_BLANK(num))
    {
        return "N/A";
    }
    ss << num / 1000000; // Bytes to MBytes
    return ss.str();
}

std::string ProcessStats::HelperHealthSystemToString(dcgmHealthSystems_enum system)
{
    switch (system)
    {
        case DCGM_HEALTH_WATCH_PCIE:
            return "PCIE Health";
        case DCGM_HEALTH_WATCH_NVLINK:
            return "NVLink Health";
        case DCGM_HEALTH_WATCH_PMU:
            return "PMU Health";
        case DCGM_HEALTH_WATCH_MCU:
            return "MCU Health";
        case DCGM_HEALTH_WATCH_MEM:
            return "Mem Health";
        case DCGM_HEALTH_WATCH_SM:
            return "SM Health";
        case DCGM_HEALTH_WATCH_INFOROM:
            return "InfoRom Health";
        case DCGM_HEALTH_WATCH_THERMAL:
            return "Thermal Health";
        case DCGM_HEALTH_WATCH_POWER:
            return "Power Health";
        case DCGM_HEALTH_WATCH_DRIVER:
            return "Driver Health";
        default:
            return DCGM_STR_BLANK;
    }
}

std::string ProcessStats::HelperHealthResultToString(dcgmHealthWatchResults_t health)
{
    if (health == DCGM_HEALTH_RESULT_PASS)
        return "Healthy";
    else if (health == DCGM_HEALTH_RESULT_WARN)
        return "Warning";
    else if (health == DCGM_HEALTH_RESULT_FAIL)
        return "Failure";
    else
        return "Internal error";
}


/*****************************************************************************
 *****************************************************************************
 * Enable Watches Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
EnableWatches::EnableWatches(std::string hostname, unsigned int groupId, int updateIntervalMs, int maxKeepAgeSec)
{
    m_hostName         = hostname;
    m_groupId          = (dcgmGpuGrp_t)(long long)groupId;
    m_updateIntervalMs = updateIntervalMs;
    m_maxKeepAgeSec    = maxKeepAgeSec;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t EnableWatches::DoExecuteConnected()
{
    return m_processStatsObj.EnableWatches(m_dcgmHandle, m_groupId, m_updateIntervalMs, m_maxKeepAgeSec);
}

/*****************************************************************************
 *****************************************************************************
 * Disable Watches Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
DisableWatches::DisableWatches(std::string hostname, unsigned int groupId)
{
    m_hostName = hostname;
    mGroupId   = (dcgmGpuGrp_t)(long long)groupId;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t DisableWatches::DoExecuteConnected()
{
    return mProcessStatsObj.DisableWatches(m_dcgmHandle, mGroupId);
}

/*****************************************************************************
 *****************************************************************************
 * View Process Stats Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
ViewProcessStats::ViewProcessStats(std::string hostname, unsigned int groupId, unsigned int pid, bool verbose)
{
    m_hostName = hostname;
    mPid       = pid;
    mVerbose   = verbose;
    mGroupId   = (dcgmGpuGrp_t)(long long)groupId;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t ViewProcessStats::DoExecuteConnected()
{
    return mProcessStatsObj.ViewProcessStats(m_dcgmHandle, mGroupId, mPid, mVerbose);
}


/*****************************************************************************
 *****************************************************************************
 * Start job invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
StartJob::StartJob(std::string hostname, unsigned int groupId, std::string jobId)
{
    m_hostName  = hostname;
    mGroupId    = (dcgmGpuGrp_t)(long long)groupId;
    this->jobId = jobId;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t StartJob::DoExecuteConnected()
{
    return mProcessStatsObj.StartJob(m_dcgmHandle, mGroupId, jobId);
}


/*****************************************************************************
 *****************************************************************************
 * Stop Job Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
StopJob::StopJob(std::string hostname, std::string jobId)
{
    m_hostName  = hostname;
    this->jobId = jobId;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t StopJob::DoExecuteConnected()
{
    return mProcessStatsObj.StopJob(m_dcgmHandle, jobId);
}

/*****************************************************************************
 *****************************************************************************
 * Remove Job Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
RemoveJob::RemoveJob(std::string hostname, std::string jobId)
{
    m_hostName  = hostname;
    this->jobId = jobId;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t RemoveJob::DoExecuteConnected()
{
    return mProcessStatsObj.RemoveJob(m_dcgmHandle, jobId);
}

/*****************************************************************************
 *****************************************************************************
 * Remove Job Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
RemoveAllJobs::RemoveAllJobs(std::string hostname)
{
    m_hostName = hostname;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t RemoveAllJobs::DoExecuteConnected()
{
    return mProcessStatsObj.RemoveAllJobs(m_dcgmHandle);
}

/*****************************************************************************
 *****************************************************************************
 * View Process Stats Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
ViewJobStats::ViewJobStats(std::string hostname, std::string jobId, bool verbose)
{
    m_hostName    = hostname;
    this->jobId   = jobId;
    this->verbose = verbose;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t ViewJobStats::DoExecuteConnected()
{
    return mProcessStatsObj.ViewJobStats(m_dcgmHandle, jobId, verbose);
}
