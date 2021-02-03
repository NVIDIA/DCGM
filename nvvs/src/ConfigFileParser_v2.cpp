/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "ConfigFileParser_v2.h"
#include "ParsingUtility.h"


#define SET_FWCFG(X, Y)                                                         \
    while (1)                                                                   \
    {                                                                           \
        if (!m_fwcfg.SetFrameworkConfigValue(X, Y))                             \
        {                                                                       \
            PRINT_DEBUG("%d", "Unable to set value %d in FWCFG", X);            \
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
    m_config.overrideMinMax     = false;
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
        case NVVS_FWCFG_GLOBAL_OVERRIDEMINMAX:
            m_config.overrideMinMax = value;
            break;
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
            return false;
    }
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
            return false;
    }
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
            return false;
    }
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
            return false;
    }
    return true;
    ;
}

/*****************************************************************************/
/* ctor saves off the input parameters to local copies/references and opens
 * the config file
 */
ConfigFileParser_v2::ConfigFileParser_v2(const std::string &configFile, const FrameworkConfig &fwcfg)
{
    PRINT_DEBUG("%s", "ConfigFileParser_v2 ctor with configFile %s", configFile.c_str());

    // save the pertinent info and object pointer
    m_configFile = configFile; // initial configuration file
    m_fwcfg      = fwcfg;      // initial frameworkconfig object
}

/*****************************************************************************/
ConfigFileParser_v2::~ConfigFileParser_v2()
{
    if (m_inputstream.is_open())
        m_inputstream.close();
}

/*****************************************************************************/
/* Close the stream if needed and initialize an fstream to the config file
 * setting YAML at the top level document
 */
bool ConfigFileParser_v2::Init()
{
    if (m_inputstream.is_open())
        m_inputstream.close();

    m_inputstream.open(m_configFile.c_str());
    if (!m_inputstream.good())
    {
        return false;
    }
    m_yamltoplevelnode = YAML::Load(m_inputstream);


    return true;
}


/*****************************************************************************/
/* Look for the gpuset, properties, and tests tags and ship those nodes to
 * the appropriate handler functions
 */
void ConfigFileParser_v2::handleGpuSetBlock(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering handleGpuSetBlock");

    {
        auto const gpuSetNode = node["gpuset"];
        if (gpuSetNode.IsDefined())
        {
            if (gpuSetNode.Type() == YAML::NodeType::Scalar)
            {
                std::string const &name = gpuSetNode.Scalar();
                SET_FWCFG(NVVS_FWCFG_GPUSET_NAME, name);
            }
            else
            {
                throw std::runtime_error("gpuset tag in config file is not a single value");
            }
        }
    }

    {
        auto const propsNode = node["properties"];
        if (propsNode.IsDefined())
        {
            handleGpuSetParameters(propsNode);
        }
    }

    {
        auto const testsNode = node["tests"];
        if (testsNode.IsDefined())
        {
            handleGpuSetTests(testsNode);
        }
    }

    PRINT_DEBUG("", "Leaving handleGpuSetBlock");
}

/*****************************************************************************/
/* look for a name tag.  Only support one name tag for now
 */
void ConfigFileParser_v2::handleGpuSetTests(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering handleGpuSetTests");

    std::string tempVal;

    if (node.Type() == YAML::NodeType::Sequence)
    {
        if (node.size() > 1)
        {
            throw std::runtime_error("Only one test name is supported in the gpu stanza at this time");
        }
        else
        {
            handleGpuSetTests(node[0]);
        }
    }
    else if (node.Type() == YAML::NodeType::Map)
    {
        auto const pName = node["name"];
        if (pName.IsDefined())
        {
            tempVal = pName.Scalar();
            SET_FWCFG(NVVS_FWCFG_TEST_NAME, tempVal);
        }
    }
    else
    {
        throw std::runtime_error("Parsing error in tests section of config file.");
    }

    PRINT_DEBUG("", "Leaving handleGpuSetTests");
}

/*****************************************************************************/
/* look for the name, brand, busid, uuid, and index tags
 */
void ConfigFileParser_v2::handleGpuSetParameters(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering handleGpuSetParameters");

    std::string tempVal;

    if (node.Type() != YAML::NodeType::Map)
    {
        throw std::runtime_error("There is an error in the gpus section of the config file.");
    }

    {
        auto const pName = node["name"];
        if (pName.IsDefined())
        {
            tempVal = pName.Scalar();
            SET_FWCFG(NVVS_FWCFG_GPU_NAME, tempVal);
        }
    }

    {
        auto const pName = node["brand"];
        if (pName.IsDefined())
        {
            tempVal = pName.Scalar();
            SET_FWCFG(NVVS_FWCFG_GPU_BRAND, tempVal);
        }
    }

    {
        auto const pName = node["busid"];
        if (pName.IsDefined())
        {
            tempVal = pName.Scalar();
            SET_FWCFG(NVVS_FWCFG_GPU_BUSID, tempVal);
        }
    }

    {
        auto const pName = node["uuid"];
        if (pName.IsDefined())
        {
            tempVal = pName.Scalar();
            SET_FWCFG(NVVS_FWCFG_GPU_UUID, tempVal);
        }
    }

    {
        auto const pName = node["index"];
        if (pName.IsDefined())
        {
            std::vector<unsigned int> indexVector;
            // potentially a csv
            std::string const &tempString = pName.Scalar();
            std::stringstream ss(tempString);
            int i;

            while (ss >> i)
            {
                indexVector.push_back(i);
                if (ss.peek() == ',')
                {
                    ss.ignore();
                }
            }
            SET_FWCFG(NVVS_FWCFG_GPU_INDEX, indexVector);
        }
    }

    PRINT_DEBUG("", "Leaving handleGpuSetParameters");
}

/*****************************************************************************/
/* Go through the gpus stanza and find the first map
 */
void ConfigFileParser_v2::CheckTokens_gpus(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering CheckTokens_gpu");

    /* Dig down until we find a map.
     * This map should be the only one and contain the optional tags: gpuset, properties, and tests
     */

    if (node.Type() == YAML::NodeType::Sequence)
    {
        if (node.size() > 1)
        {
            throw std::runtime_error("NVVS does not currently support more than one gpuset.");
        }
        CheckTokens_gpus(node[0]);
    }
    else if (node.Type() == YAML::NodeType::Map)
    {
        handleGpuSetBlock(node);
    }
    else
    {
        throw std::runtime_error("Could not parse the gpus stanza of the config file.");
    }

    PRINT_DEBUG("", "Leaving CheckTokens_gpu");
}

/*****************************************************************************/
/* go through the "globals" stanza looking for specific keywords and save them
 * to m_fwcfg
 */
void ConfigFileParser_v2::CheckTokens_globals(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering CheckTokens_global");

    if (node.Type() == YAML::NodeType::Map)
    {
        for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            std::string key;
            std::string value;
            std::string lowerValue;
            key   = it->first.Scalar();
            value = it->second.Scalar();

            PRINT_DEBUG("%s %s", "CheckTokens_global key %s, value %s", key.c_str(), value.c_str());

            /* Get a lowercase version of value for case-insensitive operations */
            lowerValue = value;
            std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(), ::tolower);

            if (key == "logfile")
            {
                SET_FWCFG(NVVS_FWCFG_GLOBAL_DATAFILE, value);
            }
            if (key == "logfile_type")
            {
                if (lowerValue == "json")
                {
                    SET_FWCFG(NVVS_FWCFG_GLOBAL_DATAFILETYPE, NVVS_LOGFILE_TYPE_JSON);
                }
                else if (lowerValue == "text")
                {
                    SET_FWCFG(NVVS_FWCFG_GLOBAL_DATAFILETYPE, NVVS_LOGFILE_TYPE_TEXT);
                }
                else if (lowerValue == "binary")
                {
                    SET_FWCFG(NVVS_FWCFG_GLOBAL_DATAFILETYPE, NVVS_LOGFILE_TYPE_BINARY);
                }
                else
                {
                    std::stringstream ss;
                    ss << "Unknown logfile_type \"" << value << "\". Allowed: json, text, or binary";
                    throw std::runtime_error(ss.str());
                }
            }
            if (key == "overrideMinMax")
            {
                // default is false
                if (lowerValue == "yes" || lowerValue == "true")
                {
                    SET_FWCFG(NVVS_FWCFG_GLOBAL_OVERRIDEMINMAX, true);
                }
            }
            if (key == "scriptable")
            {
                // default is false
                if (lowerValue == "yes" || lowerValue == "true")
                {
                    SET_FWCFG(NVVS_FWCFG_GLOBAL_SCRIPTABLE, true);
                }
            }
            if (key == "serial_override")
            {
                // default is false
                if (lowerValue == "yes" || lowerValue == "true")
                {
                    SET_FWCFG(NVVS_FWCFG_GLOBAL_OVERRIDESERIAL, true);
                }
            }
            if (key == "require_persistence_mode")
            {
                // default is true
                if (lowerValue == "no" || lowerValue == "false")
                {
                    SET_FWCFG(NVVS_FWCFG_GLOBAL_PERSISTENCE, false);
                }
            }
            if (key == "throttle-mask" && !lowerValue.empty())
            {
                /* Note: The mask is directly set in nvvsCommon for convenience as otherwise we need to add a field
                to NvvsFrameworkConfig, and update the legacyGlobalStructHelper to copy from NvvsFramworkConfig to
                nvvsCommon */
                nvvsCommon.throttleIgnoreMask = GetThrottleIgnoreReasonMaskFromString(lowerValue);
            }
        }
    }
    else
    {
        throw std::runtime_error("Unable to parse the globals section of the config file.");
    }

    PRINT_DEBUG("", "Leaving CheckTokens_global");
}

/*****************************************************************************/
void ConfigFileParser_v2::ParseGlobalsAndGpu()
{
    // because we now have a sense of what is "default" then neither of
    // these not being found is an error

    if (YAML::Node pName = m_yamltoplevelnode["globals"])
    {
        CheckTokens_globals(pName);
    }
    if (YAML::Node pName = m_yamltoplevelnode["gpus"])
    {
        CheckTokens_gpus(pName);
    }

    /* TO BE DELETED */
    legacyGlobalStructHelper();
}

/*****************************************************************************/
/* We are looking for the test name (which can be an individual test, suite, or class
 * and then looking for the "custom" tag.  Wherever that node is, if it exists, drill
 * down from there.
 */
void ConfigFileParser_v2::ParseTestOverrides(std::string testName, TestParameters &tp)
{
    // first look for the test name
    auto const pName = m_yamltoplevelnode[testName];
    if (pName.IsDefined()) // found something at the top level, leave it to the helper to dig down
    {
        handleTestDefaults(pName, tp, false);
    }

    /* getting here can mean one of several things
     * 1) the test name and the override section label differ in case
     * 2) it is a legacy config with a "custom" or suite tag
     */

    std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);

    for (YAML::const_iterator it = m_yamltoplevelnode.begin(); it != m_yamltoplevelnode.end(); it++)
    {
        std::string key;
        std::string value;
        key = it->first.Scalar();
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        if (key == testName || key == "custom" || key == "long" || key == "medium"
            || key == "quick") // start the drill down if we find a known legacy tag
        {
            CheckTokens_testDefaults(it->second, testName, tp);
        }
    }
}

/*****************************************************************************/
/* Initial search function for the test name... for newer config files this
 * will fall straight to handleTestDefaults as the node will be a single
 * entry map with the key == testName.  For legacy configs, we have to
 * drill past "custom" a bit.
 */
void ConfigFileParser_v2::CheckTokens_testDefaults(const YAML::Node &node,
                                                   std::string const &testName,
                                                   TestParameters &tp)
{
    // no care for anything but maps and maps of maps, ignore everything else
    if (node.Type() == YAML::NodeType::Sequence)
    {
        for (auto const &n : node)
        {
            CheckTokens_testDefaults(n, testName, tp);
        }
    }
    else if (node.Type() == YAML::NodeType::Map)
    {
        for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            std::string key;
            key = it->first.Scalar();
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            if (key == testName)
            {
                try
                {
                    handleTestDefaults(it->second, tp, false);
                }
                catch (std::exception const &e)
                {
                    std::stringstream ss;
                    ss << "Test " << key << " parsing failed with: \n\t" << e.what();
                    PRINT_ERROR("%s", "%s", ss.str().c_str());
                    throw std::runtime_error(ss.str());
                }
            }
            if (it->second.Type() == YAML::NodeType::Map)
            {
                CheckTokens_testDefaults(it->second, testName, tp);
            }
        }
    }
    else
    {
        std::stringstream ss;
        ss << "There is an error in the \"" << testName << "\" section of the config file.";
        throw std::runtime_error(ss.str());
    }
}

/*****************************************************************************/
/* handle actually putting the specified parameters in to the TestParms obj
 */
void ConfigFileParser_v2::handleTestDefaults(const YAML::Node &node, TestParameters &tp, bool subTest)
{
    PRINT_DEBUG("%d", "Entering handleTestDefaults subTest=%d", (int)subTest);

    unsigned int result;
    static std::string subTestName;

    if (node.Type() == YAML::NodeType::Map)
    {
        for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            std::string key;
            std::string value;
            key = it->first.Scalar();

            if (it->second.Type() == YAML::NodeType::Map)
            {
                if (key == "subtests")
                {
                    handleTestDefaults(it->second, tp, true);
                }
                else
                {
                    if (subTest)
                    {
                        subTestName = key;
                    }
                    handleTestDefaults(it->second, tp, subTest);
                }
            }
            else if (it->second.Type() == YAML::NodeType::Scalar)
            {
                value = it->second.Scalar();

                if (subTest)
                {
                    result = tp.SetSubTestString(subTestName, key, value);
                }
                else
                {
                    result = tp.SetString(key, value);
                }

                if (result != 0)
                {
                    std::stringstream ss;
                    switch (result)
                    {
                        case TP_ST_BADPARAM:
                            ss << "The parameter given for \"" << key << "\" caused an internal error .";
                            break;
                        case TP_ST_NOTFOUND:
                            ss << "The key \"" << key << "\" was not found.";
                            break;
                        case TP_ST_ALREADYEXISTS:
                            // should never happen since we are using set not add
                            ss << "The key \"" << key << "\" was added but already exists.";
                            break;
                        case TP_ST_CANTCOERCE:
                            ss << "The parameter given for \"" << key << "\" cannot be coerced to the type needed.";
                            break;
                        case TP_ST_OUTOFRANGE:
                            ss << "The parameter given for \"" << key
                               << "\" is out of the reasonable range for that key.";
                            break;
                        default:
                            ss << "Received an unknown value from the test parameter system.";
                            break;
                    }
                    throw std::runtime_error(ss.str());
                }
            }
            else
            {
                /* We would be here for a Sequence or Null (whatever Null means) */
                std::stringstream ss;
                ss << "Error in parameters section for  " << key;
                throw CFPv2Exception(node.Mark(), ss.str());
            }
        }
    }
    else
    {
        /* We would be here for a Sequence or Null (whatever Null means) */
        std::stringstream ss;
        ss << "error in \"key: value\" pairs";
        throw CFPv2Exception(node.Mark(), ss.str());
    }
}

/*****************************************************************************/
/* THE BELOW FUNCTIONS ARE ONLY FOR COMPATITBILITY UNTIL HIGHER LAYERS
 * ARE REWRITTEN!
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
    nvvsCommon.logFile        = fwcfg.dataFile;
    nvvsCommon.logFileType    = fwcfg.dataFileType;
    nvvsCommon.overrideMinMax = fwcfg.overrideMinMax;
    nvvsCommon.serialize      = fwcfg.overrideSerial;
    if (nvvsCommon.parse == false) // if it was turned on in the command line, don't overwrite it
    {
        nvvsCommon.parse = fwcfg.scriptable;
    }
    nvvsCommon.requirePersistenceMode = fwcfg.requirePersistence;
}
