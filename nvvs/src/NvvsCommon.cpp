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
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

#include "DcgmLogging.h"
#include "NvvsCommon.h"
#include "PluginStrings.h"
#include "dcgm_structs.h"
#include <DcgmStringHelpers.h>
#include <NvvsExitCode.h>

NvvsCommon::NvvsCommon()
    : logFile()
    , m_statsPath("./")
    , logFileType(NVVS_LOGFILE_TYPE_JSON)
    , parse(false)
    , quietMode(false)
    , serialize(false)
    , overrideMinMax(false)
    , verbose(false)
    , requirePersistenceMode(true)
    , configless(false)
    , statsOnlyOnFail(false)
    , errorMask(0)
    , mainReturnCode(NVVS_ST_SUCCESS)
    , pluginPath()
    , desiredTest()
    , indexString()
    , fakegpusString()
    , parmsString()
    , parms()
    , dcgmHostname()
    , clocksEventIgnoreMask(DCGM_INT64_BLANK)
    , failEarly(false)
    , failCheckInterval(5)
    , currentIteration(0)
    , totalIterations(1)
    , channelFd(-1)
    , diagResponseVersion(dcgmDiagResponse_version12)
    , rerunAsRoot(false)
    , watchFrequency(DEFAULT_WATCH_FREQUENCY_IN_MICROSECONDS)
{}

NvvsCommon::NvvsCommon(const NvvsCommon &other)
    : logFile(other.logFile)
    , m_statsPath(other.m_statsPath)
    , logFileType(other.logFileType)
    , parse(other.parse)
    , quietMode(other.quietMode)
    , serialize(other.serialize)
    , overrideMinMax(other.overrideMinMax)
    , verbose(other.verbose)
    , requirePersistenceMode(other.requirePersistenceMode)
    , configless(other.configless)
    , statsOnlyOnFail(other.statsOnlyOnFail)
    , errorMask(other.errorMask)
    , mainReturnCode(other.mainReturnCode)
    , pluginPath(other.pluginPath)
    , desiredTest(other.desiredTest)
    , indexString(other.indexString)
    , fakegpusString(other.fakegpusString)
    , parmsString(other.parmsString)
    , parms(other.parms)
    , dcgmHostname(other.dcgmHostname)
    , clocksEventIgnoreMask(other.clocksEventIgnoreMask)
    , failEarly(other.failEarly)
    , failCheckInterval(other.failCheckInterval)
    , currentIteration(other.currentIteration)
    , totalIterations(other.totalIterations)
    , channelFd(other.channelFd)
    , diagResponseVersion(other.diagResponseVersion)
    , rerunAsRoot(other.rerunAsRoot)
    , watchFrequency(other.watchFrequency)
    , ignoreErrorCodesString(other.ignoreErrorCodesString)
    , parsedIgnoreErrorCodes(other.parsedIgnoreErrorCodes)
{}

NvvsCommon &NvvsCommon::operator=(const NvvsCommon &other)
{
    logFile                = other.logFile;
    m_statsPath            = other.m_statsPath;
    logFileType            = other.logFileType;
    parse                  = other.parse;
    quietMode              = other.quietMode;
    serialize              = other.serialize;
    overrideMinMax         = other.overrideMinMax;
    verbose                = other.verbose;
    requirePersistenceMode = other.requirePersistenceMode;
    configless             = other.configless;
    statsOnlyOnFail        = other.statsOnlyOnFail;
    errorMask              = other.errorMask;
    mainReturnCode         = other.mainReturnCode;
    pluginPath             = other.pluginPath;
    desiredTest            = other.desiredTest;
    indexString            = other.indexString;
    fakegpusString         = other.fakegpusString;
    parmsString            = other.parmsString;
    parms                  = other.parms;
    dcgmHostname           = other.dcgmHostname;
    clocksEventIgnoreMask  = other.clocksEventIgnoreMask;
    failEarly              = other.failEarly;
    failCheckInterval      = other.failCheckInterval;
    currentIteration       = other.currentIteration;
    totalIterations        = other.totalIterations;
    channelFd              = other.channelFd;
    diagResponseVersion    = other.diagResponseVersion;
    rerunAsRoot            = other.rerunAsRoot;
    watchFrequency         = other.watchFrequency;
    ignoreErrorCodesString = other.ignoreErrorCodesString;
    parsedIgnoreErrorCodes = other.parsedIgnoreErrorCodes;

    return *this;
}

void NvvsCommon::Init()
{
    logFile                = "";
    m_statsPath            = "./";
    logFileType            = NVVS_LOGFILE_TYPE_JSON;
    parse                  = false;
    quietMode              = false;
    serialize              = false;
    overrideMinMax         = false;
    verbose                = false;
    requirePersistenceMode = true;
    configless             = false;
    statsOnlyOnFail        = false;
    errorMask              = 0;
    mainReturnCode         = NVVS_ST_SUCCESS;
    pluginPath             = "";
    desiredTest.clear();
    indexString    = "";
    fakegpusString = "";
    parmsString    = "";
    parms.clear();
    dcgmHostname           = "";
    clocksEventIgnoreMask  = DCGM_INT64_BLANK;
    failEarly              = false;
    failCheckInterval      = 5;
    currentIteration       = 0;
    totalIterations        = 1;
    channelFd              = -1;
    watchFrequency         = DEFAULT_WATCH_FREQUENCY_IN_MICROSECONDS;
    ignoreErrorCodesString = "";
    parsedIgnoreErrorCodes.clear();
}

void NvvsCommon::SetStatsPath(const std::string &statsPath)
{
    std::stringstream buf;

    if (statsPath.empty())
    {
        return;
    }

    if (access(statsPath.c_str(), 0) == 0)
    {
        struct stat status;
        int st = stat(statsPath.c_str(), &status);

        if (st != 0 || !(status.st_mode & S_IFDIR)) // not a dir
        {
            buf << "Error: statspath '" << statsPath << "' is not a directory.";
            log_error(buf.str());
            throw std::runtime_error(buf.str());
        }
    }
    else
    {
        buf << "Error: cannot access statspath '" << statsPath << "': " << strerror(errno);
        log_error(buf.str());
        throw std::runtime_error(buf.str());
    }

    m_statsPath = statsPath;
}

std::string GetTestDisplayName(dcgmPerGpuTestIndices_t testIndex)
{
    switch (testIndex)
    {
        case DCGM_MEMORY_INDEX:
            return std::string(MEMORY_PLUGIN_NAME);
        case DCGM_DIAGNOSTIC_INDEX:
            return std::string(DIAGNOSTIC_PLUGIN_NAME);
        case DCGM_PCI_INDEX:
            return std::string(PCIE_PLUGIN_NAME);
        case DCGM_SM_STRESS_INDEX:
            return std::string(SMSTRESS_PLUGIN_NAME);
        case DCGM_TARGETED_STRESS_INDEX:
            return std::string(TS_PLUGIN_NAME);
        case DCGM_TARGETED_POWER_INDEX:
            return std::string(TP_PLUGIN_NAME);
        case DCGM_MEMORY_BANDWIDTH_INDEX:
            return std::string(MEMBW_PLUGIN_NAME);
        case DCGM_SOFTWARE_INDEX:
            return std::string(SW_PLUGIN_NAME);
        case DCGM_CONTEXT_CREATE_INDEX:
            return std::string(CTXCREATE_PLUGIN_NAME);
        case DCGM_MEMTEST_INDEX:
            return std::string(MEMTEST_PLUGIN_NAME);
        case DCGM_PULSE_TEST_INDEX:
            return std::string(PULSE_TEST_PLUGIN_NAME);
        case DCGM_EUD_TEST_INDEX:
            return std::string(EUD_PLUGIN_NAME);
        case DCGM_NVBANDWIDTH_INDEX:
            return std::string(NVBANDWIDTH_PLUGIN_NAME);
        default:
            return std::string("Unknown");
    }
}

dcgmPerGpuTestIndices_t GetTestIndex(const std::string &name)
{
    std::string testName = name;
    std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);

    if (testName == MEMBW_PLUGIN_NAME)
    {
        return DCGM_MEMORY_BANDWIDTH_INDEX;
    }
    else if (testName == TP_PLUGIN_NAME)
    {
        return DCGM_TARGETED_POWER_INDEX;
    }
    else if (testName == TS_PLUGIN_NAME)
    {
        return DCGM_TARGETED_STRESS_INDEX;
    }
    else if (testName == SMSTRESS_PLUGIN_NAME)
    {
        return DCGM_SM_STRESS_INDEX;
    }
    else if (testName == DIAGNOSTIC_PLUGIN_NAME)
    {
        return DCGM_DIAGNOSTIC_INDEX;
    }
    else if (testName == PCIE_PLUGIN_NAME)
    {
        return DCGM_PCI_INDEX;
    }
    else if (testName == SW_PLUGIN_NAME)
    {
        return DCGM_SOFTWARE_INDEX;
    }
    else if (testName == MEMORY_PLUGIN_NAME)
    {
        return DCGM_MEMORY_INDEX;
    }
    else if (testName == MEMTEST_PLUGIN_NAME)
    {
        return DCGM_MEMTEST_INDEX;
    }
    else if (testName == PULSE_TEST_PLUGIN_NAME)
    {
        return DCGM_PULSE_TEST_INDEX;
    }
    else if (testName == CTXCREATE_PLUGIN_NAME)
    {
        return DCGM_CONTEXT_CREATE_INDEX;
    }
    else if (testName == EUD_PLUGIN_NAME)
    {
        return DCGM_EUD_TEST_INDEX;
    }
    else if (testName == NVBANDWIDTH_PLUGIN_NAME)
    {
        return DCGM_NVBANDWIDTH_INDEX;
    }

    return DCGM_UNKNOWN_INDEX;
}

dcgmDiagResult_t NvvsPluginResultToDiagResult(nvvsPluginResult_enum nvvsResult)
{
    switch (nvvsResult)
    {
        case NVVS_RESULT_PASS:
            return DCGM_DIAG_RESULT_PASS;
        case NVVS_RESULT_WARN:
            return DCGM_DIAG_RESULT_WARN;
        case NVVS_RESULT_FAIL:
            return DCGM_DIAG_RESULT_FAIL;
        case NVVS_RESULT_SKIP:
            return DCGM_DIAG_RESULT_SKIP;
        default:
        {
            log_error("Unknown NVVS result {}", static_cast<int>(nvvsResult));
            return DCGM_DIAG_RESULT_FAIL;
        }
    }
}

nvvsPluginResult_t DcgmResultToNvvsResult(dcgmDiagResult_t const result)
{
    switch (result)
    {
        case DCGM_DIAG_RESULT_PASS:
            return NVVS_RESULT_PASS;
        case DCGM_DIAG_RESULT_WARN:
            return NVVS_RESULT_WARN;
        case DCGM_DIAG_RESULT_FAIL:
            return NVVS_RESULT_FAIL;
        case DCGM_DIAG_RESULT_SKIP:
        case DCGM_DIAG_RESULT_NOT_RUN:
        default:
            return NVVS_RESULT_SKIP;
    }
}
