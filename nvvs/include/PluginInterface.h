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

#define DCGM_MAX_PLUGIN_FIELD_IDS 96

#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION_1 1
#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION_2 2 /* 2.4.0 -> 3.1.7 */
#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION_3 3 /* 3.1.8 -> 3.2.3 */
#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION_4 4 /* 3.2.5 -> 3.3.6 */
#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION_5 5 /* 3.3.7 -> 3.3.8 */
#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION_6 6 /* Current version - 4.0 */
#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION   DCGM_DIAG_PLUGIN_INTERFACE_VERSION_6

/* IMPORTANT:
 *
 * If you change any of the following struct or callback definitions, you need to increment
 * DCGM_DIAG_PLUGIN_INTERFACE_VERSION
 */
typedef struct
{
    DcgmEntityStatus_t status         = {};
    dcgmDeviceAttributes_t attributes = {};
} dcgmDiagPluginEntityGpuAux_v1;

union dcgmDiagPluginEntityAux_v1
{
    dcgmDiagPluginEntityGpuAux_v1 gpu;
};

typedef struct
{
    dcgmGroupEntityPair_t entity;
    dcgmDiagPluginEntityAux_v1 auxField;
} dcgmDiagPluginEntityInfo_v1;

typedef struct
{
    unsigned int numEntities                                         = 0;
    dcgmDiagPluginEntityInfo_v1 entities[DCGM_GROUP_MAX_ENTITIES_V2] = {};
} dcgmDiagPluginEntityList_v1;

#define DCGM_MAX_PLUGIN_DESC_LEN       128
#define DCGM_MAX_PLUGIN_NAME_LEN       20
#define DCGM_MAX_PARAMETERS_PER_PLUGIN 64
#define DCGM_MAX_PARAMETER_NAME_LEN    50
#define DCGM_DIAG_MAX_VALUE_LEN        50
static_assert(DCGM_MAX_PLUGIN_NAME_LEN <= std::min({ sizeof(dcgmDiagTestRun_v1().name),
                                                     sizeof(dcgmDiagTestRun_v1().pluginName),
                                                     sizeof(dcgmDiagResponse_v11().categories[0]) }));
static_assert(DCGM_MAX_PLUGIN_NAME_LEN <= std::min({ sizeof(dcgmDiagTestRun_v2().name),
                                                     sizeof(dcgmDiagTestRun_v2().pluginName),
                                                     sizeof(dcgmDiagResponse_v12().categories[0]) }));

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
    char testName[DCGM_MAX_PLUGIN_NAME_LEN];                                       //!< Test name
    char description[DCGM_MAX_PLUGIN_DESC_LEN];                                    //!< A short description of the test
    unsigned int numValidParameters;                                               //!< the number of valid parameters
    dcgmDiagPluginParameterInfo_t validParameters[DCGM_MAX_PARAMETERS_PER_PLUGIN]; //!< an array of valid parameters
    char testCategory[DCGM_MAX_PLUGIN_NAME_LEN];                                   //!< the name of the test category
    dcgm_field_entity_group_t targetEntityGroup;                                   //!< Target audience for this test
} dcgmDiagPluginTest_t;

#define DCGM_MAX_PLUGIN_TEST_NUM 20
static_assert(DCGM_MAX_PLUGIN_TEST_NUM <= std::size(dcgmDiagResponse_v11().tests));
static_assert(DCGM_MAX_PLUGIN_TEST_NUM <= std::size(dcgmDiagResponse_v12().tests));

typedef struct
{
    char pluginName[DCGM_MAX_PLUGIN_NAME_LEN];            //!< the plugin name
    char description[DCGM_MAX_PLUGIN_DESC_LEN];           //!< A short description of the plugin
    dcgmDiagPluginTest_t tests[DCGM_MAX_PLUGIN_TEST_NUM]; //!< Tests supported by this plugin
    unsigned int numTests;                                //!< The number of valid tests
} dcgmDiagPluginInfo_t;

typedef struct
{
    unsigned int numFieldIds;
    unsigned short fieldIds[DCGM_MAX_PLUGIN_FIELD_IDS];
} dcgmDiagPluginStatFieldIds_t;

typedef struct
{
    char parameterName[DCGM_MAX_PARAMETER_NAME_LEN];
    char parameterValue[DCGM_MAX_TEST_PARMS_LEN_V2];
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

#define DCGM_EVENT_MSG_LEN 1024
/* Pcie test can generate at least (NUM_GPUs * 6) entries */
/* NOTE: dcgmi condenses discreet entries into per-gpu output */
#define DCGM_DIAG_MAX_ERRORS 128
// DCGM_DIAG_MAX_INFO was removed. Use DCGM_DIAG_RESPONSE_INFO_MAX_V2 instead.

#define DCGM_DIAG_ALL_GPUS -1

typedef struct
{
    int gpuId;                 //!< The GPU id for this result
    nvvsPluginResult_t result; //!< The result (PASS, SKIP, FAIL)
} dcgmDiagSimpleResult_t;

enum dcgmDiagAuxDataType
{
    UNINITIALIZED_AUX_DATA_TYPE = 0, //!< The data type is not initialized (AUX data does not exist). Ver: 1
    JSON_VALUE_AUX_DATA_TYPE,        //!< The data is a string that can be parsed as JSON. Ver: 1
};

/**
 * @brief Auxiliary data for a diagnostic result
 * This is used to pass back arbitrary data from a diagnostic plugin to the caller
 * It's up to the caller to know upfront what the data is and how to interpret it.
 */
typedef struct dcgmDiagAuxData_tag
{
    unsigned int version;     //!< Version of this structure. Set to dcgmDiagAuxData_version. Ver: 1
    dcgmDiagAuxDataType type; //!< Type of data in this structure. Ver: 1
    size_t size;              //!< Size of the buffer pointed to by data. Ver: 1
    void *data;               //!< Pointer to the data. Ver: 1
} dcgmDiagAuxData_t;

#define dcgmDiagAuxData_version1 MAKE_DCGM_VERSION(dcgmDiagAuxData_t, 1)
#define dcgmDiagAuxData_version  = dcgmDiagAuxData_version1

/**
 * @brief Plugin attributes which provided by NVVS. Plugins can use attributes to align its view with NVVS.
 */
typedef struct
{
    int pluginId; //!< Plugin index. Plugin uses this value to correctly return its results.
} dcgmDiagPluginAttr_v1;

/**
 * Test results.
 * @deprecated Use dcgmDiagEntityResults_v2 instead.
 * As of DCGM 4.0.
 */
typedef struct
{
    unsigned char numErrors;                                         //!< Number of errors included in this response.
    unsigned char numInfo;                                           //!< Number of info msgs included in this response.
    unsigned short numResults;                                       //!< Number of results included in this response.
    dcgmDiagError_v1 errors[DCGM_DIAG_RESPONSE_ERRORS_MAX];          //!< Per-entity errors.
    dcgmDiagInfo_v1 info[DCGM_DIAG_RESPONSE_INFO_MAX];               //!< Per-entity info messages.
    dcgmDiagEntityResult_v1 results[DCGM_DIAG_TEST_RUN_RESULTS_MAX]; //!< Per-entity result for this test.
    dcgmDiagAuxData_t auxData;                                       //!< Auxiliary data for this test.
} dcgmDiagEntityResults_v1;

/**
 * @brief Test results.
 * As of DCGM 4.2.
 */
typedef struct
{
    unsigned char numErrors;                                         //!< Number of errors included in this response.
    unsigned char numInfo;                                           //!< Number of info msgs included in this response.
    unsigned short numResults;                                       //!< Number of results included in this response.
    dcgmDiagError_v1 errors[DCGM_DIAG_RESPONSE_ERRORS_MAX];          //!< Per-entity errors.
    dcgmDiagInfo_v1 info[DCGM_DIAG_RESPONSE_INFO_MAX_V2];            //!< Per-entity info messages.
    dcgmDiagEntityResult_v1 results[DCGM_DIAG_TEST_RUN_RESULTS_MAX]; //!< Per-entity result for this test.
    dcgmDiagAuxData_t auxData;                                       //!< Auxiliary data for this test.
} dcgmDiagEntityResults_v2;

/**
 * Get the version of this plugin.
 *
 * @return - the DCGM_DIAG_PLUGIN_INTERFACE_VERSION the plugin was compiled against. This should be checked against
 *           our DCGM_DIAG_PLUGIN_INTERFACE_VERSION to make sure they are the same. Otherwise this plugin cannot
 *           be loaded.
 */
DCGM_PUBLIC_API unsigned int GetPluginInterfaceVersion(void);

typedef unsigned int (*dcgmDiagGetPluginInterfaceVersion_f)(void);

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
 * @param statFieldIds[out]          - information to specify additional fields ids to watch and append to the stats
 * @param userData[out]              - data the plugin would like passed back to the RunTest(), RetrieveCustomStats(),
 *                                     and RetrieveResults() functions. It can be ignored if the plugin wishes.
 * @param loggingSeverity[in]        - Severity at which this plugin should log. See DcgmLoggingSeverity_t
 * @param loggingCallback[in]        - Callback to use to log. The nvvs process will log on each plugin's behalf
 *file
 * @param pluginAttr[in]             - Plugin attributes which provided by NVVS.
 *
 * @return DCGM_ST_OK                - if the plugin has been set up sufficiently to run.
 *         DCGM_ST_*                   if an error condition matching a DCGM error code has caused the plugin to not be
 *                                     runnable. a positive value if for some other reason the plugin should not be run.
 *                                     (error details can be provided to the diagnostic through RetrieveResults.)
 **/
DCGM_PUBLIC_API dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                                              void **userData,
                                              DcgmLoggingSeverity_t loggingSeverity,
                                              hostEngineAppenderCallbackFp_t loggingCallback,
                                              dcgmDiagPluginAttr_v1 const *pluginAttr);

typedef dcgmReturn_t (*dcgmDiagInitializePlugin_f)(dcgmHandle_t handle,
                                                   dcgmDiagPluginStatFieldIds_t *statFieldIds,
                                                   void **userData,
                                                   DcgmLoggingSeverity_t loggingSeverity,
                                                   hostEngineAppenderCallbackFp_t loggingCallback,
                                                   dcgmDiagPluginAttr_v1 const *pluginAttr);

/**
 * @brief Shuts down the plugin.
 *
 * This function would be called when the plugin class is destructing.
 * It is responsible for releasing any resources that the plugin has allocated,
 * and ensuring that the plugin is properly cleaned up before exit.
 *
 * @param userData[in]        - the user data set in InitializePlugin()
 *
 * @return DCGM_ST_OK         - if successed.
 *         DCGM_ST_*          - if an error occured.
 **/
DCGM_PUBLIC_API dcgmReturn_t ShutdownPlugin(void *userData);

typedef dcgmReturn_t (*dcgmDiagShutdownPlugin_f)(void *userData);

/*
 * Function called to run the test.
 *
 * @param testName[in]        - target test
 * @param timeout[in]         - the maximum time allowed for running this test.
 * @param numParameters[in]   - the number of parameters populated in testParameters
 * @param testParameters[in]  - an array of parameters to control different functions for the
 * @param entityInfo[in]      - information about each entity which should be used for this test
 * @param userData[in]        - the user data set in InitializePlugin()
 */
DCGM_PUBLIC_API void RunTest(char const *testName,
                             unsigned int timeout,
                             unsigned int numParameters,
                             const dcgmDiagPluginTestParameter_t *testParameters,
                             dcgmDiagPluginEntityList_v1 const *entityInfo,
                             void *userData);
typedef void (*dcgmDiagRunTest_f)(char const *testName,
                                  unsigned int timeout,
                                  unsigned int numParameters,
                                  const dcgmDiagPluginTestParameter_t *testParameters,
                                  dcgmDiagPluginEntityList_v1 const *entityInfo,
                                  void *userData);

/**
 * Pass custom stats (not covered by field ids) to the DCGM diagnostic
 *
 * @param testName[in]      - target test
 * @param customStats[out]  - the plugin should write any custom stats to be added to the stats file here
 * @param userData[in]      - the user data set in InitializePlugin()
 */
DCGM_PUBLIC_API void RetrieveCustomStats(char const *testName, dcgmDiagCustomStats_t *customStats, void *userData);
typedef void (*dcgmDiagRetrieveCustomStats_f)(char const *testName, dcgmDiagCustomStats_t *customStats, void *userData);

/**
 * Pass results from the plugin to the diagnostic.
 * Also, perform any shutdown and cleanup required by the plugin
 *
 * @param testName[in] - target test
 * @param entityResults[out] - detailed results of entity for the test
 * @param userData[in] - the user data set in InitializePlugin()
 */
DCGM_PUBLIC_API void RetrieveResults(char const *testName, dcgmDiagEntityResults_v2 *entityResults, void *userData);
typedef void (*dcgmDiagRetrieveResults_f)(char const *testName,
                                          dcgmDiagEntityResults_v2 *entityResults,
                                          void *userData);

#ifdef __cplusplus
} // END extern "C"
#endif
