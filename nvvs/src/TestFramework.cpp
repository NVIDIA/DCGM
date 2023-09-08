/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "dcgm_structs.h"
#include <DcgmHandle.h>
#include <DcgmRecorder.h>
#include <DcgmSystem.h>
#include <Gpu.h>
#include <JsonOutput.h>
#include <NvvsCommon.h>
#include <PluginLib.h>
#include <PluginStrings.h>
#include <TestFramework.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

extern DcgmSystem dcgmSystem;
extern DcgmHandle dcgmHandle;

/* Global boolean to say whether we should be exiting or not. This is set by the signal handler if we receive a CTRL-C
 * or other terminating signal */
std::atomic_int32_t main_should_stop __attribute__((visibility("default"))) = 0;

// globals
extern "C" {
std::map<std::string, maker_t *, std::less<std::string>> factory;
}

/*****************************************************************************/
TestFramework::TestFramework()
    : m_testList()
    , testGroup()
    , dlList()
    , skipRest(false)
    , m_nvvsBinaryMode(0)
    , m_nvvsOwnerUid(0)
    , m_nvvsOwnerGid(0)
    , m_validGpuId(0)
    , m_skipLibraryList()
{
    m_output = new Output();

    InitSkippedLibraries();
}

/*****************************************************************************/
TestFramework::TestFramework(bool jsonOutput, GpuSet *gpuSet)
    : m_testList()
    , testGroup()
    , dlList()
    , m_output(0)
    , skipRest(false)
    , m_nvvsBinaryMode(0)
    , m_nvvsOwnerUid(0)
    , m_nvvsOwnerGid(0)
    , m_plugins()
    , m_gpuInfo()
{
    std::vector<Gpu *> gpuList = gpuSet->gpuObjs;
    std::vector<unsigned int> gpuIndices;
    for (auto &&gpu : gpuList)
    {
        gpuIndices.push_back(gpu->GetGpuId());
    }

    PopulateGpuInfoForPlugins(gpuList, m_gpuInfo);

    if (jsonOutput)
    {
        m_output = new JsonOutput(gpuIndices);
    }
    else
    {
        m_output = new Output();
    }

    if (!gpuIndices.empty())
    {
        m_validGpuId = gpuIndices[0];
    }
    else
    {
        // This should never happen
        m_validGpuId = 0;
    }

    InitSkippedLibraries();
}

/*****************************************************************************/
void TestFramework::InitSkippedLibraries()
{
    m_skipLibraryList.push_back("libcurand.so");
    m_skipLibraryList.push_back("libcupti.so");
}

/*****************************************************************************/
TestFramework::~TestFramework()
{
    // close the plugins
    for (std::vector<Test *>::iterator itr = m_testList.begin(); itr != m_testList.end(); itr++)
        if (*itr)
            delete (*itr);

    // close the shared libs
    for (std::list<void *>::iterator itr = dlList.begin(); itr != dlList.end(); itr++)
        if (*itr)
            dlclose(*itr);

    delete m_output;
}

std::string TestFramework::GetPluginBaseDir()
{
    char szTmp[32] = { 0 };
    char buf[1024] = { 0 };
    std::string binaryPath;
    std::string pluginsPath;
    std::vector<std::string> searchPaths;
    std::vector<std::string>::iterator pathIt;
    struct stat statBuf = {};

    snprintf(szTmp, sizeof(szTmp), "/proc/%d/exe", getpid());
    const ssize_t nRead = readlink(szTmp, buf, sizeof(buf) - 1);
    if (nRead <= 0)
    {
        std::stringstream ss;
        ss << "Unable to read nvvs binary path from /proc: " << strerror(errno);
        log_error(ss.str());
        throw std::runtime_error(ss.str());
    }
    // nRead expected to be <= (sizeof(buf) - 1) so this should be safe
    buf[nRead] = '\0';

    // out starting point is the binary path... plugins should be in a relative path to this
    binaryPath = buf;
    if (stat(buf, &statBuf))
    {
        std::stringstream errBuf;
        errBuf << "Cannot stat NVVS binary '" << buf << "' : '" << strerror(errno)
               << "', so we cannot securely load the plugins.";
        log_error(errBuf.str());
        throw std::runtime_error(errBuf.str());
    }

    m_nvvsBinaryMode = statBuf.st_mode;
    m_nvvsOwnerUid   = statBuf.st_uid;
    m_nvvsOwnerGid   = statBuf.st_gid;

    if (nvvsCommon.pluginPath.size() == 0)
    {
        binaryPath = binaryPath.substr(0, binaryPath.find_last_of("/"));

        searchPaths.push_back("/plugins");
        searchPaths.push_back("/../share/nvidia-validation-suite/plugins");

        for (pathIt = searchPaths.begin(); pathIt != searchPaths.end(); pathIt++)
        {
            pluginsPath = binaryPath + (*pathIt);
            log_debug("Searching {} for plugins.", pluginsPath.c_str());
            if (access(pluginsPath.c_str(), 0) == 0)
            {
                struct stat status;
                int st = stat(pluginsPath.c_str(), &status);

                if (st != 0 || !(status.st_mode & S_IFDIR)) // not a dir
                    continue;
                else
                    break;
            }
        }
        if (pathIt == searchPaths.end())
            throw std::runtime_error("Plugins directory was not found.  Please check paths or use -p to set it.");
    }
    else
        pluginsPath = nvvsCommon.pluginPath;

    return pluginsPath;
}

std::string TestFramework::GetPluginDirExtension() const
{
    unsigned int cudaMajorVersion = dcgmSystem.GetCudaMajorVersion();
    unsigned int cudaMinorVersion = dcgmSystem.GetCudaMinorVersion();
    if (cudaMajorVersion == 0)
    {
        throw std::runtime_error("Unable to detect Cuda version. Please verify that libcuda.so is on the system.");
    }

    log_debug("The following Cuda version will be used for plugins: {}.{}", cudaMajorVersion, cudaMinorVersion);

    static const std::string CUDA_12_EXTENSION("/cuda12/");
    static const std::string CUDA_11_EXTENSION("/cuda11/");
    static const std::string CUDA_10_EXTENSION("/cuda10/");

    switch (cudaMajorVersion)
    {
        case 10:
            return CUDA_10_EXTENSION;
        case 11:
            return CUDA_11_EXTENSION;
        case 12:
            return CUDA_12_EXTENSION;
        default:
            log_error("Detected unsupported Cuda version: {}.{}", cudaMajorVersion, cudaMinorVersion);
            throw std::runtime_error("Detected unsupported Cuda version");
    }

    // NOT-REACHED
    return "";
}

std::string TestFramework::GetPluginDir()
{
    return GetPluginBaseDir() + GetPluginDirExtension();
}

void TestFramework::LoadLibrary(const char *libraryPath, const char *libraryName)
{
    for (const auto &skipLibrary : m_skipLibraryList)
    {
        if (!strncmp(skipLibrary.c_str(), libraryName, skipLibrary.size()))
        {
            DCGM_LOG_DEBUG << "Skipping library " << libraryName << " because it matches " << skipLibrary
                           << " in the skip list.";
            return;
        }
    }

    if (!strncmp("libpluginCommon.so", libraryName, 18))
    {
        /* libpluginCommon.so is a resource for the plugins, so it won't contain the symbols a pure plugin has.
           Process it separately and load it immediately. */
        void *dlib = dlopen(libraryPath, RTLD_LAZY);
        if (dlib == NULL)
        {
            std::string const dlopen_error = dlerror();
            log_error("Unable to open plugin {} due to: {}", libraryName, dlopen_error);
        }
        else
        {
            dlList.insert(dlList.end(), dlib);
            log_debug("Successfully loaded dlib {}", libraryName);
        }
    }
    else
    {
        auto pl = std::make_unique<PluginLib>();

        dcgmReturn_t ret = pl->LoadPlugin(libraryPath, libraryName);
        if (ret == DCGM_ST_OK)
        {
            ret = pl->InitializePlugin(dcgmHandle.GetHandle(), m_gpuInfo);
            if (ret == DCGM_ST_OK)
            {
                ret = pl->GetPluginInfo();
                if (ret == DCGM_ST_OK)
                {
                    m_plugins.push_back(std::move(pl));
                }
                else
                {
                    DCGM_LOG_ERROR << "Ignoring plugin '" << libraryName
                                   << "' because we can't retrieve information about it: " << errorString(ret);
                }
            }
            else
            {
                DCGM_LOG_ERROR << "Ignoring plugin '" << libraryName
                               << "' which cannot be initialized: " << errorString(ret);
            }
        }
        else
        {
            DCGM_LOG_ERROR << "Ignoring unloadable plugin '" << libraryName << "': " << errorString(ret);
        }
    }
}

/*****************************************************************************/
bool TestFramework::PluginPermissionsMatch(const std::string &pluginDir, const std::string &plugin)
{
    bool match = false;
    struct stat statBuf;

    if (stat(plugin.c_str(), &statBuf))
    {
        log_error("Not loading plugin '{}' in dir '{}' because I cannot stat the file : '{}'",
                  plugin,
                  pluginDir,
                  strerror(errno));
    }
    else if (m_nvvsBinaryMode != statBuf.st_mode)
    {
        log_error("Not loading plugin '{}' in dir '{}' because its permissions '{:o}' do not match "
                  "the diagnostic's : '{:o}'",
                  plugin,
                  pluginDir,
                  statBuf.st_mode,
                  m_nvvsBinaryMode);
    }
    else if (m_nvvsOwnerUid != statBuf.st_uid)
    {
        log_error("Not loading plugin '{}' in dir '{}' because its owner uid '{}' does not match "
                  "the diagnostic's : '{}'",
                  plugin,
                  pluginDir,
                  statBuf.st_uid,
                  m_nvvsOwnerUid);
    }
    else if (m_nvvsOwnerGid != statBuf.st_gid)
    {
        log_error("Not loading plugin '{}' in dir '{}' because its owner gid '{:o}' does not match "
                  "the diagnostic's : '{:o}'",
                  plugin,
                  pluginDir,
                  statBuf.st_gid,
                  m_nvvsOwnerGid);
    }
    else
    {
        match = true;
    }

    return match;
}

/*****************************************************************************/
void TestFramework::loadPlugins()
{
    // plugin discovery
    char oldPath[2048]    = { 0 };
    std::string pluginDir = GetPluginDir();
    struct dirent *dirent = 0;
    DIR *dir;
    std::string dotSo(".so");
    std::map<std::string, maker_t *, std::less<std::string>>::iterator fitr;
    std::stringstream errbuf;

    if (getcwd(oldPath, sizeof(oldPath)) == 0)
    {
        errbuf << "Cannot load plugins: unable to get current dir: '" << strerror(errno) << "'";
        log_error(errbuf.str());
        throw std::runtime_error(errbuf.str());
    }

    if (chdir(pluginDir.c_str()))
    {
        errbuf << "Error: Cannot load plugins. Unable to change to the plugin dir '" << pluginDir << "': '"
               << strerror(errno) << "'";
        log_error(errbuf.str());
        throw std::runtime_error(errbuf.str());
    }

    // Load the pluginCommon library first so that those libraries can find the appropriate symbols
    LoadLibrary("./libpluginCommon.so", "libpluginCommon.so");

    if ((dir = opendir(".")) == 0)
    {
        errbuf << "Cannot load plugins: unable to open the current dir '" << pluginDir << "': '" << strerror(errno)
               << "'";
        log_error(errbuf.str());
        throw std::runtime_error(errbuf.str());
    }

    // Read the entire directory and get the .sos
    while ((dirent = readdir(dir)) != NULL)
    {
        // Skip files that don't end in .so
        char *dot = strrchr(dirent->d_name, '.');
        if (dot == NULL || dotSo != dot)
            continue;

        // Tell dlopen to look in this directory
        char buf[2048];
        snprintf(buf, sizeof(buf), "./%s", dirent->d_name);

        // Do not load any plugins with different permissions / owner than the diagnostic binary
        if (!PluginPermissionsMatch(pluginDir, buf))
            continue;

        LoadLibrary(buf, dirent->d_name);
    }

    closedir(dir);

    chdir(oldPath);

    for (size_t i = 0; i < m_plugins.size(); i++)
    {
        auto temp = new Test(
            GetTestIndex(m_plugins[i]->GetName()), m_plugins[i]->GetDescription(), m_plugins[i]->GetTestGroup());
        m_testList.insert(m_testList.end(), temp);
        insertIntoTestGroup(m_plugins[i]->GetTestGroup(), temp);
    }
}

/*****************************************************************************/
void TestFramework::insertIntoTestGroup(std::string groupName, Test *testObject)
{
    // groupName may be a CSV list
    std::istringstream ss(groupName);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        while (token[0] == ' ')
        {
            token.erase(token.begin());
        }
        testGroup[token].push_back(testObject);
    }
}

/*****************************************************************************/
void TestFramework::go(std::vector<std::unique_ptr<GpuSet>> &gpuSets)
{
    bool pulseTestWillExecute = false;

    // Check if the pulse test will execute
    for (auto &gpuSet : gpuSets)
    {
        std::vector<Test *> testList;
        std::vector<Gpu *> gpuList = gpuSet->gpuObjs;

        testList = gpuSet->m_hardwareTestObjs;
        if (testList.size() > 0)
        {
            pulseTestWillExecute = WillExecutePulseTest(testList);
            if (pulseTestWillExecute)
            {
                break;
            }
        }

        testList = gpuSet->m_customTestObjs;
        if (testList.size() > 0)
        {
            pulseTestWillExecute = WillExecutePulseTest(testList);
            if (pulseTestWillExecute)
            {
                break;
            }
        }
    }

    // iterate through all GPU sets
    for (auto &gpuSet : gpuSets)
    {
        std::vector<Test *> testList;
        std::vector<Gpu *> gpuList = gpuSet->gpuObjs;

        m_output->AddGpusAndDriverVersion(gpuList);

        testList = gpuSet->m_softwareTestObjs;
        if (testList.size() > 0)
        {
            goList(Test::NVVS_CLASS_SOFTWARE, testList, gpuList, pulseTestWillExecute);
        }

        testList = gpuSet->m_hardwareTestObjs;
        if (testList.size() > 0)
        {
            goList(Test::NVVS_CLASS_HARDWARE, testList, gpuList, pulseTestWillExecute);
        }

        testList = gpuSet->m_integrationTestObjs;
        if (testList.size() > 0)
        {
            goList(Test::NVVS_CLASS_INTEGRATION, testList, gpuList, pulseTestWillExecute);
        }

        testList = gpuSet->m_performanceTestObjs;
        if (testList.size() > 0)
        {
            goList(Test::NVVS_CLASS_PERFORMANCE, testList, gpuList, pulseTestWillExecute);
        }

        testList = gpuSet->m_customTestObjs;
        if (testList.size() > 0)
        {
            goList(Test::NVVS_CLASS_CUSTOM, testList, gpuList, pulseTestWillExecute);
        }
    }

    m_output->print();
}

/*****************************************************************************/
void TestFramework::GetAndOutputHeader(Test::testClasses_enum classNum)
{
    // Don't do anything if we're in the legacy parse mode
    if (nvvsCommon.parse)
        return;

    std::string header;
    switch (classNum)
    {
        case Test::NVVS_CLASS_SOFTWARE:
            header = "Deployment";
            break;
        case Test::NVVS_CLASS_HARDWARE:
            header = "Hardware";
            break;
        case Test::NVVS_CLASS_INTEGRATION:
            header = "Integration";
            break;
        case Test::NVVS_CLASS_PERFORMANCE:
            header = "Stress";
            break;
        case Test::NVVS_CLASS_CUSTOM:
        default:
            header = "Custom";
            break;
    }
    m_output->header(header);
}

/*****************************************************************************/
void TestFramework::PopulateGpuInfoForPlugins(std::vector<Gpu *> &gpuList,
                                              std::vector<dcgmDiagPluginGpuInfo_t> &gpuInfo)
{
    for (size_t i = 0; i < gpuList.size(); i++)
    {
        dcgmDiagPluginGpuInfo_t gi = {};
        gi.gpuId                   = gpuList[i]->GetGpuId();
        gi.attributes              = gpuList[i]->GetAttributes();
        gi.status                  = gpuList[i]->GetDeviceEntityStatus();
        gpuInfo.push_back(gi);
    }
}

std::string TestFramework::GetCompareName(Test::testClasses_enum classNum, const std::string &pluginName, bool reverse)
{
    std::string compareName;
    if (classNum == Test::NVVS_CLASS_SOFTWARE)
    {
        compareName = "software";
    }
    else if (!reverse)
    {
        compareName = pluginName;
        std::transform(compareName.begin(), compareName.end(), compareName.begin(), [](unsigned char c) {
            if (c == ' ')
            {
                return '_';
            }
            return (char)std::tolower(c);
        });
    }
    else
    {
        compareName = pluginName;
        std::transform(compareName.begin(), compareName.end(), compareName.begin(), [](unsigned char c) {
            if (c == '_')
            {
                return ' ';
            }
            return (char)std::tolower(c);
        });
    }

    return compareName;
}

int TestFramework::GetPluginIndex(Test::testClasses_enum classNum, const std::string &pluginName)
{
    std::string compareName = GetCompareName(classNum, pluginName);

    for (unsigned int i = 0; i < m_plugins.size(); i++)
    {
        if (m_plugins[i]->GetName() == compareName)
        {
            return i;
        }
    }

    return -1;
}

std::string TestFramework::GetTestDisplayName(dcgmPerGpuTestIndices_t index)
{
    switch (index)
    {
        case DCGM_MEMORY_INDEX:
            return std::string("Memory");
        case DCGM_DIAGNOSTIC_INDEX:
            return std::string("Diagnostic");
        case DCGM_PCI_INDEX:
            return std::string("PCIe");
        case DCGM_SM_STRESS_INDEX:
            return std::string("SM Stress");
        case DCGM_TARGETED_STRESS_INDEX:
            return std::string("Targeted Stress");
        case DCGM_TARGETED_POWER_INDEX:
            return std::string("Targeted Power");
        case DCGM_MEMORY_BANDWIDTH_INDEX:
            return std::string("Memory Bandwidth");
        case DCGM_MEMTEST_INDEX:
            return std::string("Memtest");
        case DCGM_PULSE_TEST_INDEX:
            return std::string("Pulse Test");
        case DCGM_SOFTWARE_INDEX:
            return std::string("Software");
        case DCGM_CONTEXT_CREATE_INDEX:
            return std::string("Context Create");
        default:
            return std::string("Unknown");
    }
}

bool TestFramework::WillExecutePulseTest(std::vector<Test *> &testsList) const
{
    for (auto const &test : testsList)
    {
        if (test->GetTestIndex() == DCGM_PULSE_TEST_INDEX)
        {
            return true;
        }
    }

    return false;
}

/*****************************************************************************/
void TestFramework::goList(Test::testClasses_enum classNum,
                           std::vector<Test *> testsList,
                           std::vector<Gpu *> gpuList,
                           bool checkFileCreation)
{
    GetAndOutputHeader(classNum);

    // iterate through all tests giving them the GPU objects needed
    for (std::vector<Test *>::iterator testItr = testsList.begin(); testItr != testsList.end(); testItr++)
    {
        Test *test = (*testItr); // readability

        unsigned int vecSize = test->getArgVectorSize(classNum);
        for (unsigned int i = 0; i < vecSize; i++)
        {
            TestParameters *tp = test->popArgVectorElement(classNum);
            std::string name   = tp->GetString(PS_PLUGIN_NAME);

            int pluginIndex = GetPluginIndex(classNum, name);

            if (test->GetTestIndex() == DCGM_PULSE_TEST_INDEX)
            {
                // Add the iterations parameters
                tp->AddDouble(PULSE_TEST_STR_CURRENT_ITERATION, nvvsCommon.currentIteration);
                tp->AddDouble(PULSE_TEST_STR_TOTAL_ITERATIONS, nvvsCommon.totalIterations);
            }

            if (pluginIndex == -1)
            {
                // Error! Didn't find the named plugin. Report fake results for it
                DCGM_LOG_ERROR << "Couldn't find the plugin '" << name << "'";
                std::vector<dcgmDiagSimpleResult_t> perGpuResults;
                std::vector<dcgmDiagEvent_t> errors;
                std::vector<dcgmDiagEvent_t> info;
                dcgmDiagEvent_t error = {};
                error.errorCode       = -1;
                error.gpuId           = -1;
                snprintf(error.msg, sizeof(error.msg), "Unable to find plugin '%s'", name.c_str());
                errors.push_back(error);

                m_output->Result(NVVS_RESULT_FAIL, perGpuResults, errors, info);
                continue;
            }

            m_output->prep(name);
            if (!skipRest && !main_should_stop)
            {
                if (classNum == Test::NVVS_CLASS_SOFTWARE)
                {
                    if (!nvvsCommon.requirePersistenceMode)
                        tp->AddString(SW_STR_REQUIRE_PERSISTENCE, "False");
                    if (name == "Denylist")
                        tp->AddString(SW_STR_DO_TEST, "denylist");
                    else if (name == "NVML Library")
                        tp->AddString(SW_STR_DO_TEST, "libraries_nvml");
                    else if (name == "CUDA Main Library")
                        tp->AddString(SW_STR_DO_TEST, "libraries_cuda");
                    else if (name == "CUDA Toolkit Libraries")
                        tp->AddString(SW_STR_DO_TEST, "libraries_cudatk");
                    else if (name == "Permissions and OS-related Blocks")
                    {
                        tp->AddString(SW_STR_DO_TEST, "permissions");
                        if (checkFileCreation)
                        {
                            tp->AddString(SW_STR_CHECK_FILE_CREATION, "True");
                        }
                        else
                        {
                            tp->AddString(SW_STR_CHECK_FILE_CREATION, "False");
                        }
                    }
                    else if (name == "Persistence Mode")
                        tp->AddString(SW_STR_DO_TEST, "persistence_mode");
                    else if (name == "Environmental Variables")
                        tp->AddString(SW_STR_DO_TEST, "env_variables");
                    else if (name == "Page Retirement/Row Remap")
                        tp->AddString(SW_STR_DO_TEST, "page_retirement");
                    else if (name == "Graphics Processes")
                        tp->AddString(SW_STR_DO_TEST, "graphics_processes");
                    else if (name == "Inforom")
                        tp->AddString(SW_STR_DO_TEST, "inforom");
                }

                DcgmRecorder dcgmRecorder(dcgmHandle.GetHandle());

                m_plugins[pluginIndex]->RunTest(600, tp);

                m_output->Result(m_plugins[pluginIndex]->GetResult(),
                                 m_plugins[pluginIndex]->GetResults(),
                                 m_plugins[pluginIndex]->GetErrors(),
                                 m_plugins[pluginIndex]->GetInfo());

                if (classNum == Test::NVVS_CLASS_SOFTWARE)
                {
                    /* reinitialize plugin, reset errors between software runs */
                    m_plugins[pluginIndex]->InitializePlugin(dcgmHandle.GetHandle(), m_gpuInfo);
                }
            }
            else
            {
                /* If the test hasn't been run (test->go() was not called), test->GetResults() returns
                 * empty results, which is treated as the test being skipped.
                 */
                m_output->Result(m_plugins[pluginIndex]->GetResult(),
                                 m_plugins[pluginIndex]->GetResults(),
                                 m_plugins[pluginIndex]->GetErrors(),
                                 m_plugins[pluginIndex]->GetInfo());
            }

            DCGM_LOG_DEBUG << "Test " << name << " had over result " << m_plugins[pluginIndex]->GetResult()
                           << ". Configless is " << nvvsCommon.configless;

            if (m_plugins[pluginIndex]->GetResult() == NVVS_RESULT_FAIL
                && ((!nvvsCommon.configless) || nvvsCommon.failEarly))

            {
                skipRest = true;
            }
        }
    }
}

void TestFramework::addInfoStatement(const std::string &info)
{
    m_output->addInfoStatement(info);
}


std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> TestFramework::GetSubtestParameters()
{
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> parms;

    for (unsigned int i = 0; i < m_plugins.size(); i++)
    {
        std::string Name = GetCompareName(Test::NVVS_CLASS_CUSTOM, m_plugins[i]->GetName(), true);
        parms[Name]      = m_plugins[i]->GetParameterInfo();
    }

    return parms;
}
