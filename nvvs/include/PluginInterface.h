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
#pragma once

#include "NvvsCommon.h"

#include <dcgm_api_export.h>
#include <dcgm_structs.h>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

#define FAIL_EARLY          "fail_early"
#define FAIL_CHECK_INTERVAL "fail_check_interval"

#define DCGM_MAX_PLUGIN_FIELD_IDS 32

typedef struct
{
    unsigned int gpuId;
    DcgmEntityStatus_t status;
    dcgmDeviceAttributes_t attributes;
} dcgmDiagPluginGpuInfo_t;

typedef struct
{
    unsigned int numGpus;
    dcgmDiagPluginGpuInfo_t gpus[DCGM_MAX_NUM_DEVICES];
} dcgmDiagPluginGpuList_t;

#define DCGM_MAX_PLUGIN_DESC_LEN       128
#define DCGM_MAX_PLUGIN_NAME_LEN       20
#define DCGM_MAX_PARAMETERS_PER_PLUGIN 32
#define DCGM_MAX_PARAMETER_NAME_LEN    50
#define DCGM_DIAG_MAX_VALUE_LEN        50

typedef enum
{
    DcgmPluginParamNone   = 0,
    DcgmPluginParamInt    = 1,
    DcgmPluginParamFloat  = 2,
    DcgmPluginParamString = 3,
    DcgmPluginParamBool   = 4,

    DcgmPluginParamEnd = 5
} dcgmPluginValue_t;

typedef struct
{
    char parameterName[DCGM_MAX_PARAMETER_NAME_LEN]; //!< the name of the parameter
    dcgmPluginValue_t parameterType;                 //!< the type of the parameter
} dcgmDiagPluginParameterInfo_t;

typedef struct
{
    char pluginName[DCGM_MAX_PLUGIN_NAME_LEN];                                     //!< the plugin name
    unsigned int numValidParameters;                                               //!< the number of valid parameters
    dcgmDiagPluginParameterInfo_t validParameters[DCGM_MAX_PARAMETERS_PER_PLUGIN]; //!< an array of valid parameters
    char testGroup[DCGM_MAX_PLUGIN_NAME_LEN];                                      //!< the name of the test group
    char description[DCGM_MAX_PLUGIN_DESC_LEN]; //!< A short description of the plugin
} dcgmDiagPluginInfo_t;

typedef struct
{
    unsigned int numFieldIds;
    unsigned short fieldIds[DCGM_MAX_PLUGIN_FIELD_IDS];
} dcgmDiagPluginStatFieldIds_t;

typedef struct
{
    char parameterName[DCGM_MAX_PARAMETER_NAME_LEN];
    char parameterValue[DCGM_DIAG_MAX_VALUE_LEN];
    dcgmPluginValue_t type;
} dcgmDiagPluginTestParameter_t;

typedef struct
{
    dcgmPluginValue_t type; //!< The type of the stat
    long long timestamp;    //!< The timestamp
    union
    {
        int i;
        double dbl;
        char str[DCGM_DIAG_MAX_VALUE_LEN];
    } value; //!< The value for the stat
} dcgmDiagValue_t;

#define DCGM_DIAG_MAX_VALUES          128
#define DCGM_CUSTOM_STAT_TYPE_GPU     0
#define DCGM_CUSTOM_STAT_TYPE_GROUPED 1
#define DCGM_CUSTOM_STAT_TYPE_SINGLE  2

typedef struct
{
    char statName[DCGM_DIAG_MAX_VALUE_LEN];       //!< the name of the stat
    char category[DCGM_DIAG_MAX_VALUE_LEN];       //!< the category for the stat, if any
    unsigned short type;                          //!< the type of stat (one of DCGM_CUSTOM_STAT_TYPE_*
    unsigned int gpuId;                           //!< The GPU id if relevant
    unsigned int numValues;                       //!< The number of values populated
    dcgmDiagValue_t values[DCGM_DIAG_MAX_VALUES]; //!< The timestamp and value
} dcgmDiagCustomStat_t;

// Use a large size to avoid having too many vector entries on large GPU systems
#define DCGM_DIAG_MAX_CUSTOM_STATS 2048

typedef struct
{
    unsigned int moreStats;                                 /* !< Set to 1 if the diag should ask again for more,
                                                                  0 otherwise */
    unsigned int numStats;                                  // !< The number of stats populated
    dcgmDiagCustomStat_t stats[DCGM_DIAG_MAX_CUSTOM_STATS]; // !< the stats
} dcgmDiagCustomStats_t;

#define DCGM_EVENT_MSG_LEN   1024
#define DCGM_DIAG_MAX_ERRORS 32
#define DCGM_DIAG_MAX_INFO   32
#define DCGM_DIAG_MAX_SKIP   32

#define DCGM_DIAG_ALL_GPUS -1

typedef struct
{
    int errorCode;                /* !< Error code. < 0 is a DCGM error code, 0 is no error, and > 0 is some other error
                                        Note that the error code will be ignored if this is in the info field */
    int gpuId;                    //!< -1 is all GPUs, >= 0 is a GPU id
    char msg[DCGM_EVENT_MSG_LEN]; //!< Text explaining the error
} dcgmDiagEvent_t;

typedef struct
{
    int gpuId;                 //!< The GPU id for this result
    nvvsPluginResult_t result; //!< The result (PASS, SKIP, FAIL)
} dcgmDiagSimpleResult_t;

typedef struct
{
    unsigned int numResults;
    dcgmDiagSimpleResult_t perGpuResults[DCGM_MAX_NUM_DEVICES];
    unsigned int numErrors;
    dcgmDiagEvent_t errors[DCGM_DIAG_MAX_ERRORS];
    unsigned int numInfo;
    dcgmDiagEvent_t info[DCGM_DIAG_MAX_INFO];
} dcgmDiagResults_t;

/**
 * Make sure this plugin is compatible with our version of the diagnostic and get parameter information.
 *
 * @param pluginInterfaceVersion[in] - the plugin interface version
 * @param info[out]                  - information describing the plugin to the diagnostic
 *
 * @return 0                         - if this plugin can be run
 *         <0                        - if there is a DCGM_ST_* error that has made this plugin unable to run
 *         >0                        - if there is some other reason the plugin cannot run.
 */
DCGM_PUBLIC_API dcgmReturn_t GetPluginInfo(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info);

typedef dcgmReturn_t (*dcgmDiagGetPluginInfo_f)(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info);

/**
 * Function called from the diagnostic to initialize the plugin. This will be called once for each plugin
 * The plugin should perform all setup necessary for it to be ready to execute.
 * This function will have a user controllable timeout which defaults to 10 seconds in order to complete successfully.
 *
 * @param handle[in]                 - the DCGM handle that the plugin should use
 * @param gpuInfo[in]                - information about each GPU the plugin should use for its test
 * @param statFieldIds[out]          - information to specify additional fields ids to watch and append to the stats
 * @param userData[out]              - data the plugin would like passed back to the RunTest(), RetrieveCustomStats(),
 *                                     and RetrieveResults() functions. It can be ignored if the plugin wishes.
 *file
 *
 * @return DCGM_ST_OK                - if the plugin has been set up sufficiently to run.
 *         DCGM_ST_*                   if an error condition matching a DCGM error code has caused the plugin to not be
 *                                     runnable. a positive value if for some other reason the plugin should not be run.
 *                                     (error details can be provided to the diagnostic through RetrieveResults.)
 **/
DCGM_PUBLIC_API dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                                              dcgmDiagPluginGpuList_t *gpuInfo,
                                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                                              void **userData);

typedef dcgmReturn_t (*dcgmDiagInitializePlugin_f)(dcgmHandle_t handle,
                                                   dcgmDiagPluginGpuList_t *gpuInfo,
                                                   dcgmDiagPluginStatFieldIds_t *statFieldIds,
                                                   void **userData);

/*
 * Function called to run the test.
 *
 * @param timeout[in]         - the maximum time allowed for running this test.
 * @param numParameters[in]   - the number of parameters populated in testParameters
 * @param testParameters[in]  - an array of parameters to control different functions for the
 * @param userData[in]        - the user data set in InitializePlugin()
 */
DCGM_PUBLIC_API void RunTest(unsigned int timeout,
                             unsigned int numParameters,
                             const dcgmDiagPluginTestParameter_t *testParameters,
                             void *userData);
typedef void (*dcgmDiagRunTest_f)(unsigned int timeout,
                                  unsigned int numParameters,
                                  const dcgmDiagPluginTestParameter_t *testParameters,
                                  void *userData);

/**
 * Pass custom stats (not covered by field ids) to the DCGM diagnostic
 *
 * @param customStats[out]  - the plugin should write any custom stats to be added to the stats file here
 * @param userData[in]      - the user data set in InitializePlugin()
 */
DCGM_PUBLIC_API void RetrieveCustomStats(dcgmDiagCustomStats_t *customStats, void *userData);
typedef void (*dcgmDiagRetrieveCustomStats_f)(dcgmDiagCustomStats_t *customStats, void *userData);

/**
 * Pass results from the plugin to the diagnostic.
 * Also, perform any shutdown and cleanup required by the plugin
 *
 * @param results[out] - detailed results for the plugin
 * @param userData[in] - the user data set in InitializePlugin()
 */
DCGM_PUBLIC_API void RetrieveResults(dcgmDiagResults_t *results, void *userData);
typedef void (*dcgmDiagRetrieveResults_f)(dcgmDiagResults_t *results, void *userData);

#ifdef __cplusplus
} // END extern "C"
#endif
