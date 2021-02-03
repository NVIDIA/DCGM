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
#ifndef _NVVS_NVVS_ConfigFileParser2_H_
#define _NVVS_NVVS_ConfigFileParser2_H_

#include "GpuSet.h"
#include "TestParameters.h"
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

enum nvvs_fwcfg_enum
{
    NVVS_FWCFG_GLOBAL_DATAFILE = 0,
    NVVS_FWCFG_GLOBAL_DATAFILETYPE,
    NVVS_FWCFG_GLOBAL_OVERRIDEMINMAX,
    NVVS_FWCFG_GLOBAL_OVERRIDESERIAL,
    NVVS_FWCFG_GLOBAL_SCRIPTABLE,
    NVVS_FWCFG_GLOBAL_PERSISTENCE,

    NVVS_FWCFG_GPUSET_NAME,

    NVVS_FWCFG_GPU_INDEX,
    NVVS_FWCFG_GPU_BRAND,
    NVVS_FWCFG_GPU_NAME,
    NVVS_FWCFG_GPU_BUSID,
    NVVS_FWCFG_GPU_UUID,

    NVVS_FWCFG_TEST_NAME,
};

class NvvsFrameworkConfig
{
public:
    /* GLOBALS */
    std::string dataFile;          /* name of the file to output data */
    logFileType_enum dataFileType; /* type of data output */
    bool overrideMinMax;           /* allow override of the min and max whitelist values */
    bool overrideSerial;           /* force serialization of naturally parallel plugins */
    bool scriptable;               /* give a concise colon-separated output for easy parsing */
    bool requirePersistence;       /* require that persistence mode be on */

    /* GPUSET */
    std::string gpuSetIdentifier; /* name to identify the gpuset for human readability purposes */

    /* GPUS */
    std::vector<unsigned int> index; /* comma separated list of gpu indexes */
    std::string brand;               /* brand of the GPU */
    std::string name;                /* name of the GPU */
    std::string busid;               /* busID of the GPU */
    std::string uuid;                /* UUID of the GPU */

    /* TESTNAME */
    std::string testname; /* Name of the test/suite/class that should be executed */

    NvvsFrameworkConfig()
        : dataFileType(NVVS_LOGFILE_TYPE_JSON)
        , overrideMinMax(false)
        , overrideSerial(false)
        , scriptable(false)
        , requirePersistence(true)
    {}

    NvvsFrameworkConfig(const NvvsFrameworkConfig &) = default;
    NvvsFrameworkConfig(NvvsFrameworkConfig &&)      = default;
    NvvsFrameworkConfig &operator=(NvvsFrameworkConfig const &) = default;
    NvvsFrameworkConfig &operator=(NvvsFrameworkConfig &&) = default;
};

/* Class that represents a run, nominally set via the configuration file */

class FrameworkConfig
{
public:
    /***************************************************************/
    /* ctor is responsible for filling in the default values for the config */
    FrameworkConfig();
    ~FrameworkConfig();
    FrameworkConfig(const FrameworkConfig &other);
    FrameworkConfig &operator=(const FrameworkConfig &other);

    /***************************************************************/
    /* setter
     * a return of true indicates that the value was set properly
     */
    template <class T>
    bool SetFrameworkConfigValue(nvvs_fwcfg_enum field, const T &value);

    /* TO BE DELETED */
    NvvsFrameworkConfig GetFWCFG()
    {
        return m_config;
    }

private:
    NvvsFrameworkConfig m_config;
};

/* Class to contain all methods related to parsing the configuration file
 * and represent those values to calling entities.
 */
class ConfigFileParser_v2
{
public:
    /***************************************************************/
    /* ctor/dtor are responsible for entering default values into that object.
     * The default is a long run on all available GPUs using standard whitelist values
     * It is assumed that a higher layer is responsible for the memory
     * management of the FrameworkConfig object
     */
    ConfigFileParser_v2(const std::string &configFile, const FrameworkConfig &fwcfg);
    ~ConfigFileParser_v2();

    /***************************************************************/
    /* Open the stringstream for the config file and initialize
     * YAML to the most upper layer doc
     */
    bool Init();

    /***************************************************************/
    /* Parse the config file for globals and gpu specifications
     * if configFile is empty then return immediate success and assume
     * defaults are fine.
     * This function will throw an exception on error.
     */
    void ParseGlobalsAndGpu();

    /***************************************************************/
    /* Parse the test overrides for a given test and fill in the
     * appropriate fields in the TestParameters object.  If the
     * configFile is empty then return immediate success and assume
     * the defaults already in the TestParameters object are fine
     * This function will throw an exception on error.
     */
    void ParseTestOverrides(std::string testName, TestParameters &tp);

    /***************************************************************/
    /* Allow the config file to be overridden
     * This closes the currently opened stream and resets everything
     */
    void setConfigFile(std::string newConfig)
    {
        m_configFile = newConfig;
        Init();
    }

    FrameworkConfig &_test_getConfig()
    {
        return m_fwcfg;
    }

    std::vector<std::unique_ptr<GpuSet>> &getGpuSetVec()
    {
        return gpuSets;
    }
    void legacyGlobalStructHelper();

private:
    FrameworkConfig m_fwcfg;
    std::string m_configFile;
    std::ifstream m_inputstream;
    YAML::Node m_yamltoplevelnode;

    std::vector<std::unique_ptr<GpuSet>> gpuSets;

    /* private functions to recursively go through the gpu and globals stanzas looking for known tokens */
    void CheckTokens_globals(const YAML::Node &node);
    void CheckTokens_gpus(const YAML::Node &node);
    void CheckTokens_testDefaults(const YAML::Node &node, std::string const &testName, TestParameters &tp);
    void handleGpuSetBlock(const YAML::Node &node);
    void handleGpuSetParameters(const YAML::Node &node);
    void handleGpuSetTests(const YAML::Node &node);
    void handleTestDefaults(const YAML::Node &node, TestParameters &tp, bool subTest);
};

class CFPv2Exception : public std::runtime_error
{
public:
    CFPv2Exception(const YAML::Mark &mark_, const std::string &msg_)
        : std::runtime_error(build_what(mark_, msg_))
        , mark(mark_)
        , msg(msg_)
    {}
    virtual ~CFPv2Exception() throw()
    {}

    YAML::Mark mark;
    std::string msg;

private:
    static const std::string build_what(const YAML::Mark &mark, const std::string &msg)
    {
        std::stringstream output;
        output << "Config file error at line " << mark.line + 1 << ", column " << mark.column + 1 << ": " << msg;
        return output.str();
    }
};

#endif //_NVVS_NVVS_ConfigFileParser2_H_
