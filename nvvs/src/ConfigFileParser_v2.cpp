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
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// for dirname and readlink
#include <libgen.h>
#include <unistd.h>

#include "ConfigFileParser_v2.h"
#include "FallbackDiagConfig.h"
#include "ParsingUtility.h"

using namespace DcgmNs::Nvvs;

const static char c_configFileName[] = "diag-skus.yaml";

#define SET_FWCFG(X, Y)                                                         \
    while (1)                                                                   \
    {                                                                           \
        if (!m_fwcfg.SetFrameworkConfigValue(X, Y))                             \
        {                                                                       \
            DCGM_LOG_ERROR << "Unable to set value " << X << " in FWCFG";       \
            throw std::runtime_error("Unable to set value in FrameworkConfig"); \
        }                                                                       \
        break;                                                                  \
    }

/*****************************************************************************/
FrameworkConfig::FrameworkConfig()
    : m_config()
{
    // set the default config
    m_config.dataFile           = "stats";
    m_config.dataFileType       = NVVS_LOGFILE_TYPE_JSON;
    m_config.overrideSerial     = false;
    m_config.scriptable         = false;
    m_config.requirePersistence = true;

    m_config.index.clear();
    m_config.brand = "";
    m_config.name  = "";
    m_config.busid = "";
    m_config.uuid  = "";

    m_config.testname = "Long";
}

FrameworkConfig::FrameworkConfig(const FrameworkConfig &other)
    : m_config(other.m_config)
{}

FrameworkConfig &FrameworkConfig::operator=(const FrameworkConfig &other)
{
    m_config = other.m_config;
    return *this;
}

/*****************************************************************************/
FrameworkConfig::~FrameworkConfig()
{}

/*****************************************************************************/
template <class T>
bool FrameworkConfig::SetFrameworkConfigValue(nvvs_fwcfg_enum field, const T &value)
{
    return false;
}

/*****************************************************************************/
template <>
bool FrameworkConfig::SetFrameworkConfigValue(nvvs_fwcfg_enum field, const bool &value)
{
    // there is no checking here for the type of T on purpose
    switch (field)
    {
        case NVVS_FWCFG_GLOBAL_OVERRIDESERIAL:
            m_config.overrideSerial = value;
            break;
        case NVVS_FWCFG_GLOBAL_SCRIPTABLE:
            m_config.scriptable = value;
            break;
        case NVVS_FWCFG_GLOBAL_PERSISTENCE:
            m_config.requirePersistence = value;
            break;
        default:
            DCGM_LOG_WARNING << "Unhandled field: " << field;
            return false;
    }

    DCGM_LOG_DEBUG << "Set field " << field << " to " << value;
    return true;
}

/*****************************************************************************/
template <>
bool FrameworkConfig::SetFrameworkConfigValue(nvvs_fwcfg_enum field, const logFileType_enum &value)
{
    switch (field)
    {
        case NVVS_FWCFG_GLOBAL_DATAFILETYPE:
            m_config.dataFileType = value;
            break;
        default:
            DCGM_LOG_ERROR << "Unexpected field " << field;
            return false;
    }

    DCGM_LOG_DEBUG << "Set field " << field << " to " << value;
    return true;
}

/*****************************************************************************/
template <>
bool FrameworkConfig::SetFrameworkConfigValue(nvvs_fwcfg_enum field, const std::vector<unsigned int> &value)
{
    switch (field)
    {
        case NVVS_FWCFG_GPU_INDEX:
            m_config.index = value;
            break;
        default:
            DCGM_LOG_ERROR << "Unexpected field " << field;
            return false;
    }

    DCGM_LOG_DEBUG << "Set field " << field << " to " << value.size() << " values.";
    return true;
}

/*****************************************************************************/
template <>
bool FrameworkConfig::SetFrameworkConfigValue(nvvs_fwcfg_enum field, const std::string &value)
{
    // there is no checking here for the type of T on purpose
    switch (field)
    {
        case NVVS_FWCFG_GLOBAL_DATAFILE:
            m_config.dataFile = value;
            break;
        case NVVS_FWCFG_GPU_BRAND:
            m_config.brand = value;
            break;
        case NVVS_FWCFG_GPU_NAME:
            m_config.name = value;
            break;
        case NVVS_FWCFG_GPU_BUSID:
            m_config.busid = value;
            break;
        case NVVS_FWCFG_GPU_UUID:
            m_config.uuid = value;
            break;
        case NVVS_FWCFG_TEST_NAME:
            m_config.testname = value;
            break;
        case NVVS_FWCFG_GPUSET_NAME:
            m_config.gpuSetIdentifier = value;
            break;
        default:
            DCGM_LOG_ERROR << "Unexpected field " << field;
            return false;
    }

    DCGM_LOG_DEBUG << "Set field " << field << " to " << value;
    return true;
}

/*****************************************************************************/
/* ctor saves off the input parameters to local copies/references and opens
 * the config file
 */
ConfigFileParser_v2::ConfigFileParser_v2(const std::string &configFile, const FrameworkConfig &fwcfg)
{
    DCGM_LOG_DEBUG << "ConfigFileParser_v2 ctor with configFile" << configFile;

    // save the pertinent info and object pointer
    m_configFile = configFile; // initial configuration file
    m_fwcfg      = fwcfg;      // initial frameworkconfig object

    // Read config from dcgm-config package

    try
    {
        char execLocation[DCGM_PATH_LEN];
        ssize_t ret = readlink("/proc/self/exe", execLocation, sizeof(execLocation));
        // Ensure it's null-terminated
        execLocation[DCGM_PATH_LEN - 1] = '\0';

        if (ret < 0)
        {
            throw std::runtime_error("Could not find nvvs executable's directory");
        }
        const auto packageYamlLocation = std::filesystem::path(dirname(execLocation)) / c_configFileName;

        DCGM_LOG_DEBUG << "Loading package YAML from " << packageYamlLocation.string();
        m_fallbackYaml = YAML::LoadFile(packageYamlLocation);
        DCGM_LOG_DEBUG << "Loaded package YAML";
    }
    catch (const std::exception &e)
    {
        DCGM_LOG_ERROR
            << "Could not read package diag config. Please ensure the datacanter-gpu-manager-config package is installed";
        DCGM_LOG_ERROR << "Exception: " << e.what();
        m_fallbackYaml = YAML::Load(c_fallbackBakedDiagYaml);
    }

    // To aid debugging, the spec and version must be present in the fallback YAML
    try
    {
        auto spec    = m_fallbackYaml["spec"].as<std::string>();
        auto version = m_fallbackYaml["version"].as<std::string>();
        DCGM_LOG_INFO << "Loaded packaged YAML with version: " << version << ", spec: " << spec;
    }
    catch (const YAML::Exception &e)
    {
        // Rethrow with a useful message
        throw std::runtime_error("Packaged YAML did not contain version and/or spec");
    }
}

/*****************************************************************************/
/* Close the stream if needed and initialize an fstream to the config file
 * setting YAML at the top level document
 */
bool ConfigFileParser_v2::Init()
{
    if (!m_configFile.empty())
    {
        m_userYaml = YAML::LoadFile(m_configFile);
    }

    ParseYaml();
    return true;
}

/*****************************************************************************/
static void parseSubTests(YAML::Node &dstTest, const YAML::Node srcTest)
{
    for (auto srcIt = srcTest.begin(); srcIt != srcTest.end(); ++srcIt)
    {
        std::string key = srcIt->first.Scalar();
        auto srcChild   = srcIt->second;
        auto dstChild   = dstTest[key];
        // TODO Check we have a valid key and valid value type
        if (!srcChild.IsScalar())
        {
            throw std::runtime_error("Subtest child must be a scalar. Key: " + key);
        }
        dstChild = srcChild;
    }
}

/*****************************************************************************/
static void parseTests(YAML::Node &dstSku, const YAML::Node srcSku)
{
    for (auto srcTestIt = srcSku.begin(); srcTestIt != srcSku.end(); ++srcTestIt)
    {
        std::string testKey = srcTestIt->first.Scalar();
        auto srcTest        = srcTestIt->second;
        auto dstTest        = dstSku[testKey];

        DCGM_LOG_VERBOSE << "Checking key: " << testKey;
        // "name" and "id" are allowed to be scalars
        if ((testKey == "name" || testKey == "id"))
        {
            if (!srcTest.IsScalar())
            {
                throw std::runtime_error("Key must be scalar: " + testKey);
            }
        }
        // TODO Check that we have a valid test name here
        else if (!srcTest.IsMap())
        {
            throw std::runtime_error("A SKU's child that is not in {name, id} must be a map. Key: " + testKey);
        }

        // Valid types. Descend into level 2 and start copying to dst
        for (auto paramIt = srcTest.begin(); paramIt != srcTest.end(); paramIt++)
        {
            std::string paramKey   = paramIt->first.Scalar();
            auto srcSubtestOrParam = paramIt->second;
            auto dstSubtestOrParam = dstTest[paramKey];
            if (srcSubtestOrParam.IsMap())
            {
                parseSubTests(dstSubtestOrParam, srcSubtestOrParam);
            }
            else if (srcSubtestOrParam.IsScalar())
            {
                // TODO check that we have a valid param
                dstSubtestOrParam = srcSubtestOrParam;
            }
            else
            {
                throw std::runtime_error("A test's child node must be a param (scalar) or a subtest(map). Key"
                                         + paramKey);
            }
        }
    }
}

/*****************************************************************************/
void ConfigFileParser_v2::ParseYaml()
{
    for (auto srcConfig : { m_fallbackYaml, m_userYaml })
    {
        DCGM_LOG_DEBUG << "Parsing SKUs";

        auto srcSkus = srcConfig["skus"];
        if (!srcSkus.IsSequence())
        {
            auto mark = srcSkus.Mark();
            DCGM_LOG_ERROR << "skus is not a sequence; ignoring. Position: " << mark.line << "," << mark.column;
            continue;
        }

        DCGM_LOG_DEBUG << "Going through SKUs";
        for (auto srcSku : srcSkus)
        {
            std::string id;
            DCGM_LOG_VERBOSE << "Checking SKU is a map";
            if (!srcSku.IsMap())
            {
                auto mark = srcSku.Mark();
                // TODO Convert this to an exception
                DCGM_LOG_ERROR << "sku is not a map; ignoring. Position: " << mark.line << "," << mark.column;
                continue;
            }

            try
            {
                id = srcSku["id"].as<std::string>();
                DCGM_LOG_VERBOSE << "Found SKU with ID " << id;
            }
            catch (const YAML::Exception &e)
            {
                DCGM_LOG_ERROR << "SKU ID could not be read. Ignoring";
                continue;
            }

            YAML::Node &dstSku = GetOrAddSku(id);

            DCGM_LOG_VERBOSE << "Descending to SKU's children";
            parseTests(dstSku, srcSku);
        }
    }
}

/*****************************************************************************/
const std::unordered_map<std::string, YAML::Node> &ConfigFileParser_v2::GetSkus() const
{
    return m_skus;
}

/*****************************************************************************/
/* Miscellaneous Initialization Code
 */
void ConfigFileParser_v2::legacyGlobalStructHelper()
{
    /* gpu stuff */
    auto gpuSet               = std::make_unique<GpuSet>();
    NvvsFrameworkConfig fwcfg = m_fwcfg.GetFWCFG();

    gpuSet->name = fwcfg.gpuSetIdentifier;
    if (!fwcfg.brand.empty() || !fwcfg.name.empty() || !fwcfg.busid.empty() || !fwcfg.uuid.empty()
        || !fwcfg.index.empty())
    {
        gpuSet->properties.present = true;
    }

    gpuSet->properties.brand = fwcfg.brand;
    gpuSet->properties.name  = fwcfg.name;
    gpuSet->properties.busid = fwcfg.busid;
    gpuSet->properties.uuid  = fwcfg.uuid;
    if (!nvvsCommon.fakegpusString.empty())
    {
        std::vector<unsigned int> indexVector;
        // potentially a csv
        std::stringstream ss(nvvsCommon.fakegpusString);
        int i;

        while (ss >> i)
        {
            indexVector.push_back(i);
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
        gpuSet->properties.index   = indexVector;
        gpuSet->properties.present = true; // so that things are parsed further down
    }
    else if (!nvvsCommon.indexString.empty())
    {
        std::vector<unsigned int> indexVector;
        // potentially a csv
        std::stringstream ss(nvvsCommon.indexString);
        int i;

        while (ss >> i)
        {
            indexVector.push_back(i);
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
        gpuSet->properties.index   = indexVector;
        gpuSet->properties.present = true; // so that things are parsed further down
    }
    else
    {
        gpuSet->properties.index = fwcfg.index;
    }

    // Ensure that GPU ID vector does not contain duplicates
    if (gpuSet->properties.index.size() > 1)
    {
        std::unordered_set<unsigned int> ids;
        ids.reserve(gpuSet->properties.index.size());
        for (auto const &idx : gpuSet->properties.index)
        {
            auto isValueInserted = ids.insert(idx);
            if (!isValueInserted.second)
            {
                throw std::runtime_error("The given GPU ID list contains duplicate IDs. "
                                         "Please remove duplicate entries and verify that the list is correct.");
            }
        }
    }

    if (!nvvsCommon.desiredTest.empty())
    {
        for (auto const &testName : nvvsCommon.desiredTest)
        {
            std::map<std::string, std::string> tempMap;
            tempMap["name"] = testName;
            gpuSet->testsRequested.push_back(tempMap);
        }
    }
    else
    {
        std::map<std::string, std::string> tempMap;
        tempMap["name"] = fwcfg.testname;
        gpuSet->testsRequested.push_back(tempMap);
    }

    gpuSets.push_back(std::move(gpuSet));

    /* globals */
    nvvsCommon.logFile     = fwcfg.dataFile;
    nvvsCommon.logFileType = fwcfg.dataFileType;
    nvvsCommon.serialize   = fwcfg.overrideSerial;
    if (nvvsCommon.parse == false) // if it was turned on in the command line, don't overwrite it
    {
        nvvsCommon.parse = fwcfg.scriptable;
    }
    nvvsCommon.requirePersistenceMode = fwcfg.requirePersistence;
}

/*****************************************************************************/
/* Returns reference to m_skus[id]. Adds the SKU if it does not exist */
YAML::Node &ConfigFileParser_v2::GetOrAddSku(const std::string &id)
{
    if (m_skus.find(id) == m_skus.end())
    {
        m_skus[id] = YAML::Node();
    }
    return m_skus[id];
}
