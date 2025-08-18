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

#include "DcgmError.h"
#include "DcgmLogging.h"
#include "Gpu.h"
#include <atomic>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <sysexits.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEPRECATION_WARNING "NVVS has been deprecated. Please use dcgmi diag to invoke these tests."

#define NVVS_ENV_LOG_PREFIX "__NVVS_DBG"

/* Has the user requested a stop? 1=yes. 0=no. Defined in main.cpp */
extern std::atomic_int32_t main_should_stop;

/* Each suite adds additional tests to the previous level with XLONG being the superset */
enum suiteNames_enum
{
    NVVS_SUITE_CUSTOM,
    NVVS_SUITE_QUICK,
    NVVS_SUITE_MEDIUM,
    NVVS_SUITE_LONG,
    NVVS_SUITE_PRODUCTION_TESTING,
    NVVS_SUITE_XLONG,
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
using nvvsPluginEntityMsgs_t    = std::map<dcgmGroupEntityPair_t, std::vector<std::string>>;
using nvvsPluginEntityErrors_t  = std::map<dcgmGroupEntityPair_t, std::vector<dcgmDiagError_v1>>;
using nvvsPluginEntityResults_t = std::map<dcgmGroupEntityPair_t, nvvsPluginResult_t>;

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
    std::string dcgmHostname;                                        /* Host name where DCGM is running */
    uint64_t clocksEventIgnoreMask;                                  // Mask of clocks event reasons to ignore.
    bool failEarly;             // enable failure checks throughout test rather than at the end so we stop test sooner
    uint64_t failCheckInterval; /* how often failure checks should occur when running tests (in seconds). Only
                                       applies if failEarly is enabled. */
    unsigned int currentIteration;      /* the current iteration of the diagnostic being executed.
                                           See dcgmi/CommandLineParser.h */
    unsigned int totalIterations;       /* the total number of iterations of the diagnostic that will run. */
    std::string entityIds;              // Comma-separated list of entity ids.
    int channelFd;                      // A file description used to send back response to caller.
    unsigned int diagResponseVersion;   // The version of diag response to be returned via channel-fd.
    bool rerunAsRoot;                   // Flag indicating if this round is the second attemping or not.
    unsigned int watchFrequency;        // The watch frequency for fields being watched
    std::string ignoreErrorCodesString; // String of error codes to be ignored on different entities
    std::map<dcgmGroupEntityPair_t, std::unordered_set<unsigned int>>
        parsedIgnoreErrorCodes; // String of error codes to be ignored on different entities

private:
    static unsigned int constexpr DEFAULT_WATCH_FREQUENCY_IN_MICROSECONDS { 5000000 };
};

extern NvvsCommon nvvsCommon;

std::string GetTestDisplayName(dcgmPerGpuTestIndices_t testIndex);
dcgmPerGpuTestIndices_t GetTestIndex(const std::string &testName);
dcgmDiagResult_t NvvsPluginResultToDiagResult(nvvsPluginResult_enum nvvsResult);
nvvsPluginResult_t DcgmResultToNvvsResult(dcgmDiagResult_t const result);
