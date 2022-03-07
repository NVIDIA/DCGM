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
#ifndef _NVVS_NVVS_Plugin_H_
#define _NVVS_NVVS_Plugin_H_

#include <iostream>
#include <map>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CustomStatHolder.h"
#include "DcgmError.h"
#include "DcgmMutex.h"
#include "DcgmRecorder.h"
#include "Gpu.h"
#include "NvvsCommon.h"
#include "NvvsStructs.h"
#include "Output.h"
#include "TestParameters.h"
#include <NvvsStructs.h>

extern const double DUMMY_TEMPERATURE_VALUE;

// This is a base class for all test plugins
// Once the EUD and Healthmon are converted to a plugin,
// this will likely go away and functionality moved to the Test
// class

typedef struct
{
    std::string shortDescription;
    std::string testGroups;
    dcgmPerGpuTestIndices_t testIndex; /* from dcgmPerGpuTestIndices_enum */
    void *customEntry;
    bool selfParallel;
    TestParameters *defaultTestParameters;
    std::string logFileTag; /* Name to add onto log files to have different
                               tests generate different log files */
} infoStruct_t;

// observedMetrics: map the metric name to a map of GPU ID -> value
typedef std::map<std::string, std::map<unsigned int, double>> observedMetrics_t;

class Plugin
{
    /***************************PUBLIC***********************************/
public:
    Plugin();
    virtual ~Plugin();

    /* Interface methods for running the plugin */
    virtual void Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters) = 0;

    /* Getters and Setters */
    /*************************************************************************/
    infoStruct_t GetInfoStruct() const
    {
        return m_infoStruct;
    }

    /*************************************************************************/
    void SetOutputObject(Output *obj)
    {
        progressOut = obj;
    }

    /* Plugin results */
    /*************************************************************************/
    /*
     * Gets overall result from the specified results.
     *
     * Returns:
     *      - NVVS_RESULT_PASS if all GPUs had result NVVS_RESULT_PASS
     *      - NVVS_RESULT_FAIL if *any* GPU had result NVVS_RESULT_FAIL
     *      - NVVS_RESULT_WARN if *any* GPU had result NVVS_RESULT_WARN
     *      - NVVS_RESULT_SKIP if all GPUs had result NVVS_RESULT_SKIP
     */
    static nvvsPluginResult_t GetOverallResult(const nvvsPluginGpuResults_t &results);

    /*************************************************************************/
    /*
     * Gets overall result for this test.
     *
     * Returns:
     *      - NVVS_RESULT_PASS if all GPUs had result NVVS_RESULT_PASS
     *      - NVVS_RESULT_FAIL if *any* GPU had result NVVS_RESULT_FAIL
     *      - NVVS_RESULT_WARN if *any* GPU had result NVVS_RESULT_WARN
     *      - NVVS_RESULT_SKIP if all GPUs had result NVVS_RESULT_SKIP
     */
    nvvsPluginResult_t GetResult();

    /*************************************************************************/
    /*
     * Get results for all GPUs.
     *
     * Returns:
     *      - nvvsPluginGpuResults_t (map from gpu id to results)
     */
    const nvvsPluginGpuResults_t &GetGpuResults() const
    {
        return m_results;
    }

    /*************************************************************************/
    /*
     * Sets the result for all GPUs to the result given by res.
     *
     */
    void SetResult(nvvsPluginResult_t res);

    /*************************************************************************/
    /*
     * Sets the result for the GPU given by gpuId to the result given by res.
     *
     */
    void SetResultForGpu(unsigned int gpuId, nvvsPluginResult_t res);

    /*************************************************************************/
    const std::vector<std::string> &GetWarnings() const
    {
        return m_warnings;
    }

    /*************************************************************************/
    const std::vector<DcgmError> &GetErrors() const
    {
        return m_errors;
    }

    /*************************************************************************/
    const nvvsPluginGpuErrors_t &GetGpuErrors() const
    {
        return m_errorsPerGPU;
    }

    /*************************************************************************/
    const nvvsPluginGpuMessages_t &GetGpuWarnings() const
    {
        return m_warningsPerGPU;
    }

    /*************************************************************************/
    const std::vector<std::string> &GetVerboseInfo() const
    {
        return m_verboseInfo;
    }

    /*************************************************************************/
    const nvvsPluginGpuMessages_t &GetGpuVerboseInfo() const
    {
        return m_verboseInfoPerGPU;
    }

    /*************************************************************************/
    inline void RecordObservedMetric(unsigned int gpuId, const std::string &valueName, double value)
    {
        m_values[valueName][gpuId] = value;
    }

    /*************************************************************************/
    observedMetrics_t GetObservedMetrics() const
    {
        return m_values;
    }

    bool UsingFakeGpus() const
    {
        return m_fakeGpus;
    }

    /* Methods */
    /*************************************************************************/
    /*
     * Initializes internal result and message structures for use with the gpus given by gpuList.
     * This method **MUST** be called before the plugin logs any messages or sets a result.
     *
     * This method clears any existing warnings, info messages, and results as a side effect.
     * Sets m_gpuList to a copy of the given gpuInfo.
     *
     */
    void InitializeForGpuList(const dcgmDiagPluginGpuList_t &gpuInfo);

    /*************************************************************************/
    /*
     * Adds an error for this plugin
     *
     * Thread-safe.
     */
    void AddError(const DcgmError &error);

    /*************************************************************************/
    /*
     * Adds an error for the GPU specified by gpuId
     *
     * Thread-safe.
     */
    void AddErrorForGpu(unsigned int gpuId, const DcgmError &error);

    /*************************************************************************/
    /*
     * Logs an info message.
     *
     * Thread-safe.
     */
    void AddInfo(const std::string &info);

    /*************************************************************************/
    /*
     * Adds a non-GPU specific verbose message.
     *
     * Thread-safe.
     */
    void AddInfoVerbose(const std::string &info);

    /*************************************************************************/
    /*
     * Adds a verbose message for the GPU given by gpuId.
     *
     * Thread-safe.
     */
    void AddInfoVerboseForGpu(unsigned int gpuId, const std::string &info);

    /*************************************************************************/
    /*
     * Fills in the results struct for this plugin object
     *
     * @param results - the struct in which results are stored
     */
    dcgmReturn_t GetResults(dcgmDiagResults_t *results);

    /*
     * Adds a custom timeseries statistic for GPU gpuId with the specified name and value
     */
    void SetGpuStat(unsigned int gpuId, const std::string &name, double value);

    /*
     * Adds a custom timeseries statistic for GPU gpuId with the specified name and value
     */
    void SetGpuStat(unsigned int gpuId, const std::string &name, long long value);

    /*
     * Adds a custom field fo GPU gpuId
     */
    void SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value);

    /*
     * Adds a custom statistic for group groupName with the specified name and value
     */
    void SetGroupedStat(const std::string &groupName, const std::string &name, double value);

    /*
     * Adds a custom statistic for group groupName with the specified name and value
     */
    void SetGroupedStat(const std::string &groupName, const std::string &name, long long value);

    /*
     * Get the data associated with the custom gpu stat recorded
     */
    std::vector<dcgmTimeseriesInfo_t> GetCustomGpuStat(unsigned int gpuId, const std::string &name);

    /*
     * Populate the struct with statistics from where we've left off in iteration. It could be from the beginning.
     */
    void PopulateCustomStats(dcgmDiagCustomStats_t &customStats);

    /* Variables */
    Output *progressOut; // Output object passed in from the test framework for progress updates

    /***************************PRIVATE**********************************/
private:
    /* Methods */
    /*************************************************************************/
    /*
     * Clears all warnings, info messages, and results.
     *
     */
    void ResetResultsAndMessages();

    nvvsPluginGpuResults_t m_results;            /* Per GPU results: Pass | Fail | Skip | Warn */
    std::vector<std::string> m_warnings;         /* List of general warnings from the plugin */
    std::vector<DcgmError> m_errors;             /* List of errors from the plugin */
    nvvsPluginGpuErrors_t m_errorsPerGPU;        // Per GPU list of errors from the plugin
    nvvsPluginGpuMessages_t m_warningsPerGPU;    /* Per GPU list of warnings from the plugin */
    std::vector<std::string> m_verboseInfo;      /* List of general verbose output from the plugin */
    nvvsPluginGpuMessages_t m_verboseInfoPerGPU; /* Per GPU list of verbose output from the plugin */

    /* Variables */
    observedMetrics_t m_values; /* Record the values found for pass/fail criteria */
    bool m_fakeGpus;            /* Whether or not this plugin is using fake gpus */

    /* Mutexes */
    DcgmMutex m_dataMutex; /* Mutex for plugin data */

    /* Returns the external display name for the specified test type */
    std::string GetDisplayName();

    /***************************PROTECTED********************************/
protected:
    /* Variables */
    infoStruct_t m_infoStruct;
    std::string m_logFile;
    DcgmMutex m_mutex;                   /* mutex for locking the plugin (for use by subclasses). */
    std::vector<unsigned int> m_gpuList; /* list of GPU ids for this plugin */
    CustomStatHolder m_customStatHolder; /* hold stats that aren't DCGM fields */

    long long DetermineMaxTemp(unsigned int gpuId,
                               double parameterValue,
                               DcgmRecorder &dr,
                               dcgmDeviceThermals_t &thermals);
};

// typedef for easier referencing for the factory
typedef Plugin *maker_t();
extern "C" {
extern std::map<std::string, maker_t *, std::less<std::string>> factory;
}
#endif //_NVVS_NVVS_Plugin_H_
