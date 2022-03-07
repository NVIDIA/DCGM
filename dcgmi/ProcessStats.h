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
 * ProcessStats.h
 *
 *  Created on: Oct 16, 2015
 *      Author: chris
 */

#ifndef PROCESSSTATS_H_
#define PROCESSSTATS_H_

#include "Command.h"

class ProcessStats
{
public:
    ProcessStats();
    virtual ~ProcessStats();

    /*****************************************************************************
     * This method is used to enable watches on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t EnableWatches(dcgmHandle_t mNvcmHandle, dcgmGpuGrp_t groupId, int updateIntervalMs, int maxKeepAgeSec);

    /*****************************************************************************
     * This method is used to disable watches on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisableWatches(dcgmHandle_t mNvcmHandle, dcgmGpuGrp_t groupId);
    /*****************************************************************************
     * This method is used to display process stats on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t ViewProcessStats(dcgmHandle_t mNvcmHandle, dcgmGpuGrp_t groupId, unsigned int pid, bool verbose);

    /*****************************************************************************
     * This method is used to start a job on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t StartJob(dcgmHandle_t mNvcmHandle, dcgmGpuGrp_t groupId, std::string jobId);

    /*****************************************************************************
     * This method is used to stop a job on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t StopJob(dcgmHandle_t mNvcmHandle, std::string jobId);

    /*****************************************************************************
     * This method is used to display job stats on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t ViewJobStats(dcgmHandle_t mNvcmHandle, std::string jobId, bool verbose);

    /*****************************************************************************
     * This method is used to remove job stats for a job on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RemoveJob(dcgmHandle_t mNvcmHandle, std::string jobId);

    /*****************************************************************************
     * This method is used to remove all job stats on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RemoveAllJobs(dcgmHandle_t mNvcmHandle);

private:
    /*****************************************************************************
     * Function to display all execution stats from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidExecutionStats(dcgmPidSingleInfo_t *pidInfo, bool verbose);
    /*****************************************************************************
     *   Function to display all event stats from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidEventStats(dcgmPidSingleInfo_t *pidInfo, bool verbose);
    /*****************************************************************************
    Function to display all performance stats from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidPerformanceStats(dcgmPidSingleInfo_t *pidInfo, bool verbose);
    /*****************************************************************************
     * Function to display all violation stats from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidViolationStats(dcgmPidSingleInfo_t *pidInfo, bool verbose);

    /*****************************************************************************
     * Function to display process utilization from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidProcessUtilization(dcgmPidSingleInfo_t *pidInfo, bool verbose);

    /*****************************************************************************
     * Function to display over all Health of the system while the process is running
     *****************************************************************************/
    void HelperDisplayOverAllHealth(dcgmPidSingleInfo_t *pidInfo, bool verbose);


    /*****************************************************************************
     * Function to display all execution stats from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobExecutionStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose);
    /*****************************************************************************
     *   Function to display all event stats from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobEventStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose);
    /*****************************************************************************
        Function to display all performance stats from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobPerformanceStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose);
    /*****************************************************************************
     * Function to display all violation stats from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobViolationStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose);

    /*****************************************************************************
     * Function to display process utilization from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobProcessUtilization(dcgmGpuUsageInfo_t *jobInfo, bool verbose);


    /*****************************************************************************
     * Function to display over all Health of the system while the job was running
     *****************************************************************************/
    void HelperDisplayOverAllHealth(dcgmGpuUsageInfo_t *jobInfo, bool verbose);

    /*****************************************************************************
     *  Converts double count of seconds to hr:min:sec format
     *****************************************************************************/
    std::string HelperFormatTotalTime(double time);
    /*****************************************************************************
     * Converts long long timeseries into a human readable date and time
     *****************************************************************************/
    std::string HelperFormatTimestamp(long long ts, bool isStartTime);

    /*****************************************************************************
     * Converts a time difference into a string. If one or both are blank, prints
     * the appropriate error text.
     *****************************************************************************/
    std::string HelperFormatTimeDifference(long long timestamp1, long long timestamp2);

    /*****************************************************************************
     * Converts long long times into percentage, checking for DCGM_INT64_BLANK prior
     *****************************************************************************/
    double HelperFormatPercentTime(long long denominator, long long numerator);

    /*****************************************************************************
     * Converts a 32 bit interger stat summary into avg: min: max: string
     *****************************************************************************/
    std::string HelperFormatStat32Summary(dcgmStatSummaryInt32_t &summary);

    /*****************************************************************************
     * Converts a double stat summary into avg: min: max: string
     *****************************************************************************/
    std::string HelperFormatStatFp64Summary(dcgmStatSummaryFp64_t &summary);

    /*****************************************************************************
     *Converts a 64 bit interger stat sumamry into avg: min: max: string
     *****************************************************************************/
    std::string HelperFormatStatPCISummary(dcgmStatSummaryInt64_t &summary);

    /*****************************************************************************
     * Checks if integer is blank and returns N/A if so. Number as string otherwise
     *****************************************************************************/
    std::string Helper32IntToString(int num);

    /*****************************************************************************
     * Checks if double is blank and returns N/A if so. Number as string otherwise
     *****************************************************************************/
    std::string HelperDoubleToString(double num);

    /*****************************************************************************
     * Checks if long long is blank and returns N/A if so. Number as string otherwise
     *****************************************************************************/
    std::string HelperPCILongToString(long long num);

    std::string HelperHealthSystemToString(dcgmHealthSystems_enum system);

    std::string HelperHealthResultToString(dcgmHealthWatchResults_t health);
};


/**
 * Enable Watches Invoker
 */
class EnableWatches : public Command
{
public:
    EnableWatches(std::string hostname, unsigned int groupId, int updateIntervalMs, int maxKeepAgeSec);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    ProcessStats m_processStatsObj;
    dcgmGpuGrp_t m_groupId;
    int m_updateIntervalMs;
    int m_maxKeepAgeSec;
};

/**
 * Enable Watches Invoker
 */
class DisableWatches : public Command
{
public:
    DisableWatches(std::string hostname, unsigned int groupId);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    ProcessStats mProcessStatsObj;
    dcgmGpuGrp_t mGroupId;
};

/**
 * Get Process Information Watches Invoker
 */
class ViewProcessStats : public Command
{
public:
    ViewProcessStats(std::string hostname, unsigned int mGroupId, unsigned int pid, bool verbose);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    ProcessStats mProcessStatsObj;
    unsigned int mPid;
    bool mVerbose;
    dcgmGpuGrp_t mGroupId;
};

/**
 * Start Job Invoker
 */
class StartJob : public Command
{
public:
    StartJob(std::string hostname, unsigned int groupId, std::string jobId);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    ProcessStats mProcessStatsObj;
    std::string jobId;
    dcgmGpuGrp_t mGroupId;
};

/**
 * Stop Job Invoker
 */
class StopJob : public Command
{
public:
    StopJob(std::string hostname, std::string jobId);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    ProcessStats mProcessStatsObj;
    std::string jobId;
};

/**
 * Remove Job Invoker
 */
class RemoveJob : public Command
{
public:
    RemoveJob(std::string hostname, std::string jobId);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    ProcessStats mProcessStatsObj;
    std::string jobId;
};

/**
 * Remove All Jobs Invoker
 */
class RemoveAllJobs : public Command
{
public:
    RemoveAllJobs(std::string hostname);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    ProcessStats mProcessStatsObj;
};


/**
 * Get Process Information Watches Invoker
 */
class ViewJobStats : public Command
{
public:
    ViewJobStats(std::string hostname, std::string jobId, bool verbose);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    ProcessStats mProcessStatsObj;
    std::string jobId;
    bool verbose;
};


#endif /* PROCESSSTATS_H_ */
