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
#include "PluginTest.h"
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
    std::string testCategories;
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
    virtual void Go(std::string const &testName,
                    dcgmDiagPluginEntityList_v1 const *entityInfo,
                    unsigned int numParameters,
                    const dcgmDiagPluginTestParameter_t *testParameters)
        = 0;

    /* Getters and Setters */
    /*************************************************************************/
    infoStruct_t GetInfoStruct() const
    {
        return m_infoStruct;
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
    nvvsPluginResult_t GetResult(std::string const &testName) const;

    /*************************************************************************/
    /*
     * Get results for all GPUs.
     *
     * Returns:
     *      - nvvsPluginGpuResults_t (map from gpu id to results)
     */
    const nvvsPluginGpuResults_t &GetGpuResults(std::string const &testName) const
    {
        return m_tests.at(testName).GetGpuResults();
    }

    /*************************************************************************/
    /*
     * Get results for all entites.
     *
     * Returns:
     *      - nvvsPluginEntityResults_t (map from entity to results)
     */
    const nvvsPluginEntityResults_t &GetEntityResults(std::string const &testName) const
    {
        return m_tests.at(testName).GetEntityResults();
    }

    /*************************************************************************/
    /*
     * Sets the result for all entities to the result given by res.
     *
     */
    void SetResult(std::string const &testName, nvvsPluginResult_t res);

    /*************************************************************************/
    /*
     * Sets the result for the GPU given by gpuId to the result given by res.
     *
     */
    void SetResultForGpu(std::string const &testName, unsigned int gpuId, nvvsPluginResult_t res);

    /*************************************************************************/
    /*
     * Sets the result for a specific entity to the result given by res.
     *
     */
    void SetResultForEntity(std::string const &testName, dcgmGroupEntityPair_t const &entity, nvvsPluginResult_t res);

    void SetNonGpuResult(std::string const &testName, nvvsPluginResult_t res);

    /*************************************************************************/
    std::vector<std::string> const &GetWarnings(std::string const &testName) const
    {
        return m_tests.at(testName).GetWarnings();
    }

    /*************************************************************************/
    std::vector<DcgmError> const &GetErrors(std::string const &testName) const
    {
        return m_tests.at(testName).GetErrors();
    }

    /*************************************************************************/
    nvvsPluginEntityErrors_t const &GetEntityErrors(std::string const &testName) const
    {
        return m_tests.at(testName).GetEntityErrors();
    }

    /*************************************************************************/
    nvvsPluginGpuErrors_t const &GetGpuErrors(std::string const &testName) const
    {
        return m_tests.at(testName).GetGpuErrors();
    }

    /*************************************************************************/
    nvvsPluginGpuMessages_t const &GetGpuWarnings(std::string const &testName) const
    {
        return m_tests.at(testName).GetGpuWarnings();
    }

    /*************************************************************************/
    /* Deprecated: Use GetEntityVerboseInfo() instead. */
    std::vector<std::string> const &GetVerboseInfo(std::string const &testName) const
    {
        return m_tests.at(testName).GetVerboseInfo();
    }

    /*************************************************************************/
    nvvsPluginEntityMsgs_t const &GetEntityVerboseInfo(std::string const &testName) const
    {
        return m_tests.at(testName).GetEntityVerboseInfo();
    }

    /*************************************************************************/
    /* Deprecated: Use GetEnityVerboseInfo() instead. */
    nvvsPluginGpuMessages_t const &GetGpuVerboseInfo(std::string const &testName) const
    {
        return m_tests.at(testName).GetGpuVerboseInfo();
    }

    /*************************************************************************/
    inline void RecordObservedMetric(std::string const &testName,
                                     unsigned int gpuId,
                                     const std::string &valueName,
                                     double value)
    {
        m_tests.at(testName).RecordObservedMetric(gpuId, valueName, value);
    }

    /*************************************************************************/
    observedMetrics_t GetObservedMetrics(std::string const &testName) const
    {
        return m_tests.at(testName).GetObservedMetrics();
    }

    bool UsingFakeGpus(std::string const &testName) const
    {
        return m_tests.at(testName).UsingFakeGpus();
    }

    /*
     * Initializes internal structures for use with the entityInfo.
     * This method **MUST** be called before the plugin logs any messages or sets a result.
     *
     * This method clears any existing warnings, info messages, and results as a side effect.
     *
     */
    void InitializeForEntityList(std::string const &testName, dcgmDiagPluginEntityList_v1 const &entityInfo);

    /*************************************************************************/
    /*
     * Deprecated: Use AddInfo() or AddError() instead.
     * Adds a warning for this plugin
     *
     * Thread-safe.
     */
    [[deprecated("Use AddInfo() or AddError() instead")]] void AddWarning(std::string const &testName,
                                                                          std::string const &error);

    /*************************************************************************/
    /*
     * Adds an error for this plugin associated with the entity specified by entityPair.
     *
     * Thread-safe when lock holds m_dataMutex.
     */
    DcgmLockGuard AddError(DcgmLockGuard &&lock, std::string const &testName, dcgmDiagError_v1 const &error);

    /*************************************************************************/
    /*
     * Adds an error for this plugin associated with the entity specified by entityPair.
     *
     * Thread-safe.
     */

    void AddError(std::string const &testName, dcgmDiagError_v1 const &error);

    /*************************************************************************/
    /*
     * Deprecated: Use AddError(diagError) instead.
     * Adds an error for this plugin
     *
     * Thread-safe.
     */
    void AddError(std::string const &testName, DcgmError const &error);

    /*************************************************************************/
    /*
     * Adds an error that should be stored, but shouldn't be reported if other errors were found.
     *
     * These errors tend to be very generic; generally, they are accompanied by a more specific
     * error and can be ignored. However, if no other error is found they should be reported.
     */
    void AddOptionalError(std::string const &testName, DcgmError const &error);

    /*************************************************************************/
    /*
     * Logs an info message.
     *
     * Thread-safe.
     */
    void AddInfo(std::string const &testName, std::string const &info);

    /*************************************************************************/
    /*
     * Deprecated: Use AddInfoVerboseForEntity() instead.
     * Adds a non-GPU specific verbose message.
     *
     * Thread-safe.
     */
    void AddInfoVerbose(std::string const &testName, std::string const &info);

    /*************************************************************************/
    /*
     * Adds a verbose message for the GPU given by gpuId.
     *
     * Thread-safe when lock holds m_dataMutex.
     */
    DcgmLockGuard AddInfoVerboseForEntity(DcgmLockGuard &&lock,
                                          std::string const &testName,
                                          dcgmGroupEntityPair_t entity,
                                          std::string const &info);

    /*************************************************************************/
    /*
     * Adds a verbose message for the GPU given by gpuId.
     *
     * Thread-safe.
     */
    void AddInfoVerboseForEntity(std::string const &testName, dcgmGroupEntityPair_t entity, std::string const &info);


    /*************************************************************************/
    /*
     * Deprecated: Use AddInfoVerboseForEntity() instead.
     * Adds a verbose message for the GPU given by gpuId.
     *
     * Thread-safe.
     */
    void AddInfoVerboseForGpu(std::string const &testName, unsigned int gpuId, std::string const &info);

    /**************************************************************************
     * Get results for all entities.
     *
     * @param[in] testName Name of the test for which results are being retrieved
     * @param[out] entityResults Pointer to a structure where the results will be stored
     *
     * @returns DCGM_ST_OK on success, appropriate error code otherwise
     */
    virtual dcgmReturn_t GetResults(std::string const &testName, dcgmDiagEntityResults_v2 *entityResults);

    /**************************************************************************
     * Get results for all entities.
     *
     * @deprecated Use GetResults(dcgmDiagEntityResults_v2 *) instead.
     * @param[in] testName Name of the test for which results are being retrieved
     * @param[out] entityResults Pointer to a structure where the results will be stored
     *
     * @returns DCGM_ST_OK on success, appropriate error code otherwise
     */
    virtual dcgmReturn_t GetResults(std::string const &testName, dcgmDiagEntityResults_v1 *entityResults);

    /*
     * Adds a custom timeseries statistic for GPU gpuId with the specified name and value
     */
    void SetGpuStat(std::string const &testName, unsigned int gpuId, std::string const &name, double value);

    /*
     * Adds a custom timeseries statistic for GPU gpuId with the specified name and value
     */
    void SetGpuStat(std::string const &testName, unsigned int gpuId, std::string const &name, long long value);

    /*
     * Adds a custom field fo GPU gpuId
     */
    void SetSingleGroupStat(std::string const &testName,
                            std::string const &gpuId,
                            std::string const &name,
                            std::string const &value);

    /*
     * Adds a custom statistic for group groupName with the specified name and value
     */
    void SetGroupedStat(std::string const &testName,
                        std::string const &groupName,
                        std::string const &name,
                        double value);

    /*
     * Adds a custom statistic for group groupName with the specified name and value
     */
    void SetGroupedStat(std::string const &testName,
                        std::string const &groupName,
                        std::string const &name,
                        long long value);

    /*
     * Get the data associated with the custom gpu stat recorded
     */
    std::vector<dcgmTimeseriesInfo_t> GetCustomGpuStat(std::string const &testName,
                                                       unsigned int gpuId,
                                                       std::string const &name);

    /*
     * Populate the struct with statistics from where we've left off in iteration. It could be from the beginning.
     */
    void PopulateCustomStats(std::string const &testName, dcgmDiagCustomStats_t &customStats);

    /*
     * Initialize logging from within this plugin with the given severity and logging callback.
     * The main NVVS process will do the actual logging via the provided callback.
     */
    void InitializeLogging(DcgmLoggingSeverity_t loggingSeverity, hostEngineAppenderCallbackFp_t loggingCallback);

    /* Returns the external display name for the specified test type */
    std::string GetDisplayName();

    void SetPluginAttr(dcgmDiagPluginAttr_v1 const *pluginAttr);

    int GetPluginId() const;

    void ParseIgnoreErrorCodesParam(std::string const &testName, std::string const &param);

    gpuIgnoreErrorCodeMap_t const &GetIgnoreErrorCodes(std::string const &testName) const;

    bool ShouldIgnoreError(std::string const &testName,
                           dcgmGroupEntityPair_t const &entity,
                           unsigned int errorCode) const;

    /***************************PRIVATE**********************************/
#ifndef DCGM_PLUGIN_TEST
private:
#endif
    /* Methods */

    /*************************************************************************/
    /*
     * Fills in the results struct for this plugin object
     * Assumes caller has verified entityResults is valid.
     *
     * @tparam EntityResultsType Must be either dcgmDiagEntityResults_v1 or dcgmDiagEntityResults_v2
     * @param[in] testName Name of the test
     * @param[out] entityResults Struct in which results are stored
     *
     * @return DCGM_ST_OK if successful, or error code if unsuccessful
     *         See PluginTest::GetResults() for specific error codes
     */
    template <typename EntityResultsType>
        requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
                 || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
    dcgmReturn_t GetResultsImpl(std::string const &testName, EntityResultsType *entityResults);

    dcgmDiagPluginAttr_v1 m_pluginAttr; /* The plugin attributes assigned from nvvs */

    /***************************PROTECTED********************************/
protected:
    /* Variables */
    infoStruct_t m_infoStruct;
    std::unordered_map<std::string, PluginTest> m_tests; /* Map test name to its PluginTest object */

    /* Mutexes */
    mutable DcgmMutex m_dataMutex; /* Mutex for plugin data */
    mutable DcgmMutex m_mutex;     /* mutex for locking the plugin (for use by subclasses). */

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
