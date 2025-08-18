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
#include "CpuHelpers.h"
#include "CpuSet.h"
#include "FallbackDiagConfig.h"
#include "NvvsCommon.h"
#include "ParsingUtility.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <EntityListHelpers.h>

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
        const std::filesystem::path execLocation = [] {
            thread_local char execLocation[PATH_MAX] { 0 };
            ssize_t ret = readlink("/proc/self/exe", execLocation, sizeof(execLocation) - 1);

            if (ret < 0)
            {
                throw std::runtime_error("Could not find nvvs executable's directory");
            }
            return std::filesystem::path { execLocation };
        }();

        const std::filesystem::path packageYamlLocation = [&execLocation] {
            const auto libexecSubdirectory = execLocation.parent_path();
            const auto packageName         = libexecSubdirectory.filename();
            return libexecSubdirectory.parent_path().parent_path() / "share" / packageName / c_configFileName;
        }();

        DCGM_LOG_DEBUG << "Loading package YAML from " << packageYamlLocation.string();
        m_fallbackYaml = YAML::LoadFile(packageYamlLocation);
        DCGM_LOG_DEBUG << "Loaded package YAML";
    }
    catch (const std::exception &e)
    {
        DCGM_LOG_ERROR
            << "Could not read package diag config. Please ensure the datacenter-gpu-manager-config package is installed";
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
        std::string key      = srcIt->first.Scalar();
        auto const &srcChild = srcIt->second;
        auto dstChild        = dstTest[key];
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
        for (auto const &kv : srcTest)
        {
            std::string const &paramKey   = kv.first.Scalar();
            auto const &srcSubtestOrParam = kv.second;
            auto dstSubtestOrParam        = dstTest[paramKey];
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

template <typename ContainerType>
static bool ContainDuplicate(ContainerType const &container)
{
    if (container.empty())
    {
        return false;
    }
    std::unordered_set<typename ContainerType::value_type> hashSet;
    hashSet.reserve(container.size());

    for (auto const &idx : container)
    {
        auto isValueInserted = hashSet.insert(idx);
        if (!isValueInserted.second)
        {
            return true;
        }
    }
    return false;
}

std::unique_ptr<EntitySet> ConfigFileParser_v2::PrepareGpuSet(std::vector<dcgmGroupEntityPair_t> const &entityGroups)
{
    /* gpu stuff */
    auto gpuSet               = std::make_unique<GpuSet>();
    NvvsFrameworkConfig fwcfg = m_fwcfg.GetFWCFG();

    gpuSet->SetName(fwcfg.gpuSetIdentifier);
    if (!fwcfg.brand.empty() || !fwcfg.name.empty() || !fwcfg.busid.empty() || !fwcfg.uuid.empty()
        || !fwcfg.index.empty())
    {
        gpuSet->GetProperties().present = true;
    }

    std::vector<unsigned> gpuIndexesFromEntityGroups;
    for (auto const &entityPair : entityGroups)
    {
        if (entityPair.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        gpuIndexesFromEntityGroups.push_back(entityPair.entityId);
    }

    gpuSet->GetProperties().brand = fwcfg.brand;
    gpuSet->GetProperties().name  = fwcfg.name;
    gpuSet->GetProperties().busid = fwcfg.busid;
    gpuSet->GetProperties().uuid  = fwcfg.uuid;
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
        gpuSet->GetProperties().index   = indexVector;
        gpuSet->GetProperties().present = true; // so that things are parsed further down
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
        gpuSet->GetProperties().index   = indexVector;
        gpuSet->GetProperties().present = true; // so that things are parsed further down
    }
    else if (!gpuIndexesFromEntityGroups.empty())
    {
        gpuSet->GetProperties().index   = gpuIndexesFromEntityGroups;
        gpuSet->GetProperties().present = true; // so that things are parsed further down
    }
    else
    {
        gpuSet->GetProperties().index = fwcfg.index;
    }

    if (ContainDuplicate(gpuSet->GetProperties().index))
    {
        throw std::runtime_error("The given GPU ID list contains duplicate IDs. "
                                 "Please remove duplicate entries and verify that the list is correct.");
    }

    if (!gpuSet->GetProperties().present)
    {
        return nullptr;
    }

    if (nvvsCommon.desiredTest.empty())
    {
        nvvsCommon.desiredTest.insert(fwcfg.testname);
    }

    for (auto const &idx : gpuSet->GetProperties().index)
    {
        gpuSet->AddEntityId(idx);
    }
    gpuSet->SetEntityGroup(DCGM_FE_GPU);
    return gpuSet;
}

std::unique_ptr<EntitySet> ConfigFileParser_v2::PrepareCpuSet(std::vector<dcgmGroupEntityPair_t> const &entityGroups)
{
    std::vector<dcgm_field_eid_t> cpuEntityIds;

    cpuEntityIds.reserve(entityGroups.size());
    for (auto const entity : entityGroups)
    {
        if (entity.entityGroupId != DCGM_FE_CPU)
        {
            continue;
        }

        cpuEntityIds.push_back(entity.entityId);
    }

    if (cpuEntityIds.empty())
    {
        return nullptr;
    }

    if (ContainDuplicate(cpuEntityIds))
    {
        throw std::runtime_error("Error: The given CPU ID list contains duplicate IDs. "
                                 "Please remove duplicate entries and verify that the list is correct.");
    }

    CpuHelpers cpuHelpers;

    if (cpuHelpers.GetVendor() != CpuHelpers::GetNvidiaVendorName() && !CpuHelpers::SupportNonNvidiaCpu())
    {
        throw std::runtime_error("Error: Only support Nvidia CPUs.");
    }
    auto systemCpuIds = cpuHelpers.GetCpuIds();

    sort(systemCpuIds.begin(), systemCpuIds.end());
    sort(cpuEntityIds.begin(), cpuEntityIds.end());

    if (systemCpuIds != cpuEntityIds)
    {
        throw std::runtime_error(fmt::format("Error: The given CPU ID list [{}] does not align with system [{}]. "
                                             "Please provide all presented CPU.",
                                             fmt::to_string(fmt::join(cpuEntityIds, ",")),
                                             fmt::to_string(fmt::join(systemCpuIds, ","))));
    }

    auto cpuSet = std::make_unique<CpuSet>();
    for (auto const cpuEntityId : cpuEntityIds)
    {
        cpuSet->AddEntityId(cpuEntityId);
    }
    return cpuSet;
}

void ConfigFileParser_v2::PrepareEntitySets(dcgmHandle_t dcgmHandle)
{
    std::vector<dcgmGroupEntityPair_t> entityGroups;
    auto err = DcgmNs::EntityListWithMigAndUuidParser(dcgmHandle, nvvsCommon.entityIds, entityGroups);
    if (!err.empty())
    {
        std::string errMsg = fmt::format("failed to parse entity ids: {} with err: {}", nvvsCommon.entityIds, err);
        log_error(errMsg);
        throw std::runtime_error(errMsg);
    }

    for (auto const &creator : { &ConfigFileParser_v2::PrepareGpuSet, &ConfigFileParser_v2::PrepareCpuSet })
    {
        auto set = std::invoke(creator, this, entityGroups);
        if (set)
        {
            m_entitySets.emplace_back(std::move(set));
        }
    }
}

/*****************************************************************************/
/* Miscellaneous Initialization Code
 */
void ConfigFileParser_v2::legacyGlobalStructHelper()
{
    NvvsFrameworkConfig fwcfg = m_fwcfg.GetFWCFG();
    nvvsCommon.logFile        = fwcfg.dataFile;
    nvvsCommon.logFileType    = fwcfg.dataFileType;
    nvvsCommon.serialize      = fwcfg.overrideSerial;
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
