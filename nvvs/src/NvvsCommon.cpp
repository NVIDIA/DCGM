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
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

#include "DcgmLogging.h"
#include "NvvsCommon.h"
#include "PluginStrings.h"
#include "dcgm_structs.h"

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
    , mainReturnCode(MAIN_RET_OK)
    , pluginPath()
    , desiredTest()
    , indexString()
    , fakegpusString()
    , parmsString()
    , parms()
    , jsonOutput(false)
    , fromDcgm(false)
    , dcgmHostname()
    , training(false)
    , forceTraining(false)
    , throttleIgnoreMask(DCGM_INT64_BLANK)
    , trainingIterations(0)
    , trainingVariancePcnt(.0)
    , trainingTolerancePcnt(.0)
    , failEarly(false)
    , failCheckInterval(5)

{
    memset(m_gpus, 0, sizeof(m_gpus));
}

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
    , jsonOutput(other.jsonOutput)
    , fromDcgm(other.fromDcgm)
    , dcgmHostname(other.dcgmHostname)
    , training(other.training)
    , forceTraining(other.forceTraining)
    , throttleIgnoreMask(other.throttleIgnoreMask)
    , trainingIterations(other.trainingIterations)
    , trainingVariancePcnt(other.trainingVariancePcnt)
    , trainingTolerancePcnt(other.trainingTolerancePcnt)
    , failEarly(other.failEarly)
    , failCheckInterval(other.failCheckInterval)
{
    memset(m_gpus, 0, sizeof(m_gpus));
}

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
    jsonOutput             = other.jsonOutput;
    fromDcgm               = other.fromDcgm;
    dcgmHostname           = other.dcgmHostname;
    training               = other.training;
    forceTraining          = other.forceTraining;
    throttleIgnoreMask     = other.throttleIgnoreMask;
    failEarly              = other.failEarly;
    failCheckInterval      = other.failCheckInterval;

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
    mainReturnCode         = MAIN_RET_OK;
    pluginPath             = "";
    desiredTest.clear();
    indexString    = "";
    fakegpusString = "";
    parmsString    = "";
    parms.clear();
    jsonOutput         = false;
    fromDcgm           = false;
    dcgmHostname       = "";
    training           = false;
    forceTraining      = false;
    throttleIgnoreMask = DCGM_INT64_BLANK;
    failEarly          = false;
    failCheckInterval  = 5;
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
            PRINT_ERROR("%s", "%s", buf.str().c_str());
            throw std::runtime_error(buf.str());
        }
    }
    else
    {
        buf << "Error: cannot access statspath '" << statsPath << "': " << strerror(errno);
        PRINT_ERROR("%s", "%s", buf.str().c_str());
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
        default:
            return std::string("Unknown");
    }
}

dcgmPerGpuTestIndices_t GetTestIndex(const std::string name)
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
    else if (testName == CTXCREATE_PLUGIN_NAME)
    {
        return DCGM_CONTEXT_CREATE_INDEX;
    }

    return DCGM_UNKNOWN_INDEX;
}
