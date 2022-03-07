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

#include "DcgmError.h"
#include "DcgmLogging.h"
#include "Gpu.h"
#include <map>
#include <set>
#include <stdint.h>
#include <string>
#include <sysexits.h>
#include <vector>

#define DEPRECATION_WARNING "NVVS has been deprecated. Please use dcgmi diag to invoke these tests."

#define NVVS_ENV_LOG_PREFIX "__NVVS_DBG"

/* Main return codes */
#define MAIN_RET_OK 0
#define MAIN_RET_ERROR                                                                            \
    1 /* Return a single code for now. In the future, we could use the standard sysexits.h codes, \
                             but those are pretty ambiguous as well */

/* Has the user requested a stop? 1=yes. 0=no. Defined in main.cpp */
extern int main_should_stop;

enum suiteNames_enum
{
    NVVS_SUITE_QUICK,
    NVVS_SUITE_MEDIUM,
    NVVS_SUITE_LONG,
    NVVS_SUITE_CUSTOM,
};


/* Logfile output types */
enum logFileType_enum
{
    NVVS_LOGFILE_TYPE_JSON,  /* JSON data without line breaks */
    NVVS_LOGFILE_TYPE_TEXT,  /* Indented plain text */
    NVVS_LOGFILE_TYPE_BINARY /* Binary log format */
    /* Note if you add values here, you must change the ranges where this
     * is used in tp->AddDouble(). Currently NvidiaValidationSuite.cpp */
};

/* Plugin test result states */
typedef enum nvvsPluginResult_enum
{
    NVVS_RESULT_PASS,
    NVVS_RESULT_WARN,
    NVVS_RESULT_FAIL,
    NVVS_RESULT_SKIP
} nvvsPluginResult_t;

// nvvsPluginGpuResults: map GPU IDs to the NVVS Plugin result for the GPU (i.e. Pass | Fail | Warn | Skip)
typedef std::map<unsigned int, nvvsPluginResult_t> nvvsPluginGpuResults_t;

// nvvsPluginGpuMessages: map GPU IDs to vector of (string) messages for that GPU
typedef std::map<unsigned int, std::vector<std::string>> nvvsPluginGpuMessages_t;
typedef std::map<unsigned int, std::vector<DcgmError>> nvvsPluginGpuErrors_t;

/* Internal function return codes */
typedef enum nvvsReturn_enum
{
    NVVS_ST_SUCCESS       = 0,
    NVVS_ST_BADPARAM      = -1, // A bad parameter was passed to a function
    NVVS_ST_GENERIC_ERROR = -2, // A generic, unspecified error
    NVVS_ST_REQUIRES_ROOT = -3, // This function or one of its children requires root to run
} nvvsReturn_t;

class NvvsCommon
{
public:
    NvvsCommon();
    NvvsCommon(const NvvsCommon &other);
    NvvsCommon &operator=(const NvvsCommon &other);
    void Init();
    void SetStatsPath(const std::string &statsPath);

    // structure for global variables
    std::string logFile;          /* file prefix for statistics */
    std::string m_statsPath;      /* Path where statistics files should be saved */
    logFileType_enum logFileType; /* format for statistics to log */
    bool parse;                   /* output parseable format */
    bool quietMode;               /* quiet mode prints no output to the console, all results
                                   * are available via logs or return code only */
    bool serialize;               /* serialize tests that would normally be parallel */
    bool overrideMinMax;          /* override parm min/max values */
    bool verbose;                 /* enable verbose metric reporting */
    bool requirePersistenceMode;  /* require persistence mode to be enabled (default true) */
    bool configless;              /* enable configless operation */
    bool statsOnlyOnFail;         /* enable output of statistics files only on an failure */
    unsigned long long errorMask; /* error mask for inforom */
    int mainReturnCode;     /* MAIN_RET_? #define of the error to return or MAIN_RET_OK if no errors have occured */
    std::string pluginPath; /* Path given in command line for plugins */
    std::set<std::string> desiredTest; /* Specific test(s) asked for on the command line */
    std::string indexString;           /* A potentially comma-separated list of GPUs to run NVVS on */
    std::string fakegpusString;        /* run diagnostics on fake gpus only */
    std::string parmsString;           /* unparsed subtest parameters */
    std::map<std::string, std::map<std::string, std::string>> parms; /* test parameters to set from the command line */
    bool jsonOutput;                                                 /* Produce json output as documented below */
    bool fromDcgm;                   // Note that this run was initiated by DCGM to avoid the deprecation warning
    std::string dcgmHostname;        /* Host name where DCGM is running */
    bool training;                   // Run NVVS in training mode to generate golden values for this configuration
    bool forceTraining;              // Generate golden values despite warnings.
    uint64_t throttleIgnoreMask;     // Mask of throttling reasons to ignore.
    unsigned int trainingIterations; // Number of iterations of each test to perform when training.
    double trainingVariancePcnt;     // Variance allowed as a percentage of the mean in training mode.
    double trainingTolerancePcnt;    // Percentage of tolerance towards meeting the golden value in training mode.
    std::string goldenValuesFile;    // Filename where golden values should be saved
    bool failEarly;             // enable failure checks throughout test rather than at the end so we stop test sooner
    uint64_t failCheckInterval; /* how often failure checks should occur when running tests (in seconds). Only
                                       applies if failEarly is enabled. */
    Gpu *m_gpus[DCGM_MAX_NUM_DEVICES]; // Pointers to the gpu objects that are active for this run
};

extern NvvsCommon nvvsCommon;

/* When jsonOutput is set to true, NVVS writes output in the format:
 * {
 *   "DCGM GPU Diagnostic" : {
 *     "test_categories" : [
 *       {
 *         "category" : "<header>",    # One of Deployment|Hardware|Integration|Performance|Custom
 *         "tests" : [
 *           {
 *             "name" : <name>,
 *             "results" : [
 *               {
 *                 "gpu_ids" : <gpu_ids>, # GPU ID - field name is left as "gpu_ids" to maintain backwards compatibility
 *                 "status : "<status>",  # One of PASS|FAIL|WARN|SKIPPED
 *                 "warnings" : [         # Optional, depends on test output and result
 *                   "<warning_text>", ...
 *                 ],
 *                 "info" : [             # Optional, depends on test output and result
 *                    "<info_text>", ...
 *                 ]
 *               }, ...
 *             ]
 *           }, ...
 *         ]
 *       }, ...
 *     ],
 *     "version" : "<version_str>" # 1.7
 *   }
 * }
 */


std::string GetTestDisplayName(dcgmPerGpuTestIndices_t testIndex);
dcgmPerGpuTestIndices_t GetTestIndex(const std::string testName);