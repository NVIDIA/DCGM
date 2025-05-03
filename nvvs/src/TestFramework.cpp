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

#include "DcgmStringHelpers.h"
#include "EntitySet.h"
#include "PluginInterface.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <DcgmHandle.h>
#include <DcgmRecorder.h>
#include <DcgmSystem.h>
#include <FdChannelClient.h>
#include <Gpu.h>
#include <NvvsCommon.h>
#include <PluginLib.h>
#include <PluginStrings.h>
#include <TestFramework.h>

#include <DcgmBuildInfo.hpp>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <list>
#include <memory>
#include <ranges>
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

namespace
{

void FillGpuEntityAuxData(EntitySet *entitySet, std::vector<dcgmDiagPluginEntityInfo_v1> &entityInfos)
{
    assert(entitySet->GetEntityGroup() == DCGM_FE_GPU);
    GpuSet *gpuSet = ToGpuSet(entitySet);
    assert(gpuSet);
    assert(gpuSet->GetGpuObjs().size() == entityInfos.size());
    for (unsigned idx = 0; idx < entityInfos.size(); ++idx)
    {
        entityInfos[idx].auxField.gpu.attributes = gpuSet->GetGpuObjs()[idx]->GetAttributes();
        entityInfos[idx].auxField.gpu.status     = gpuSet->GetGpuObjs()[idx]->GetDeviceEntityStatus();
    }
}

std::unordered_set<unsigned int> GetGpuIds(EntitySet *entitySet)
{
    if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
    {
        return {};
    }

    std::unordered_set<unsigned int> res;
    for (const auto gpuId : entitySet->GetEntityIds())
    {
        res.insert(gpuId);
    }

    return res;
}

} //namespace


/*****************************************************************************/
TestFramework::TestFramework()
    : m_testList()
    , m_testCategories()
    , dlList()
    , skipRest(false)
    , m_nvvsBinaryMode(0)
    , m_nvvsOwnerUid(0)
    , m_nvvsOwnerGid(0)
    , m_validGpuId(0)
    , m_completedTests(0)
    , m_numTestsToRun(0)
    , m_skipLibraryList()
{
    InitSkippedLibraries();
}

/*****************************************************************************/
TestFramework::TestFramework(std::vector<std::unique_ptr<EntitySet>> &entitySets)
    : m_testList()
    , m_testCategories()
    , dlList()
    , skipRest(false)
    , m_nvvsBinaryMode(0)
    , m_nvvsOwnerUid(0)
    , m_nvvsOwnerGid(0)
    , m_plugins()
{
    std::vector<unsigned int> gpuIndices;
    for (auto const &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        GpuSet *gpuSet                    = ToGpuSet(entitySet.get());
        std::vector<Gpu *> const &gpuList = gpuSet->GetGpuObjs();
        for (auto &&gpu : gpuList)
        {
            gpuIndices.push_back(gpu->GetGpuId());
        }
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
        searchPaths.push_back("/../libexec/datacenter-gpu-manager-4/plugins");

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
    {
        pluginsPath = nvvsCommon.pluginPath;
        if (!std::filesystem::exists(pluginsPath) || !std::filesystem::is_directory(pluginsPath))
        {
            throw std::runtime_error("Plugins directory was not found.  Please check paths or use -p to set it.");
        }
    }

    return pluginsPath;
}

std::optional<std::string> TestFramework::GetPluginCudaDirExtension() const
{
    unsigned int cudaMajorVersion = dcgmSystem.GetCudaMajorVersion();
    unsigned int cudaMinorVersion = dcgmSystem.GetCudaMinorVersion();
    if (cudaMajorVersion == 0)
    {
        log_error("Unable to detect Cuda version.");
        return std::nullopt;
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
        case 13:
            // FIXME: Update to CUDA 13 directory once CUDA 13 support is added
            return CUDA_12_EXTENSION;
        default:
            log_error("Detected unsupported Cuda version: {}.{}", cudaMajorVersion, cudaMinorVersion);
            throw std::runtime_error("Detected unsupported Cuda version");
    }

    // NOT-REACHED
    return std::nullopt;
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
            int pluginId = m_plugins.size();
            ret          = pl->InitializePlugin(dcgmHandle.GetHandle(), pluginId);
            if (ret == DCGM_ST_OK)
            {
                ret = pl->GetPluginInfo();
                if (ret == DCGM_ST_OK)
                {
                    m_plugins.push_back(std::move(pl));
                    log_debug("Successfully loaded dlib {}", libraryName);
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

std::optional<std::string> TestFramework::GetPluginUsingDriverDir()
{
    auto baseDir                                      = GetPluginBaseDir();
    std::optional<std::string> pluginCudaDirExtension = GetPluginCudaDirExtension();
    if (!pluginCudaDirExtension)
    {
        return std::nullopt;
    }
    return baseDir + *pluginCudaDirExtension;
}

void TestFramework::LoadPluginWithDir(std::string const &pluginDir)
{
    // plugin discovery
    char oldPath[2048] = { 0 };

    if (getcwd(oldPath, sizeof(oldPath)) == 0)
    {
        std::string errorMessage = std::format("Cannot load plugins: unable to get current dir: '{}'", strerror(errno));
        log_error(errorMessage);
        throw std::runtime_error(std::move(errorMessage));
    }

    if (chdir(pluginDir.c_str()))
    {
        std::string errorMessage = std::format(
            "Error: Cannot load plugins. Unable to change to the plugin dir '{}': '{}'", pluginDir, strerror(errno));

        log_error(errorMessage);
        throw std::runtime_error(std::move(errorMessage));
    }

    const struct ResetWorkingDirectory
    {
        char const *path;
        ~ResetWorkingDirectory()
        {
            // Previous implementation did not account for possible failure. Deferring error handling
            chdir(path);
        }
    } resetWorkingDirectory { oldPath };

    DIR *const dir = opendir(".");
    if (dir == 0)
    {
        std::string errorMessage = std::format(
            "Error: Cannot load plugins: unable to open the current dir '{}': '{}'", pluginDir, strerror(errno));

        log_error(errorMessage);
        throw std::runtime_error(std::move(errorMessage));
    }

    const struct CloseDirectory
    {
        DIR *directory;
        ~CloseDirectory()
        {
            // Previous implementation did not account for possible failure. Deferring error handling
            closedir(directory);
        }
    } closeDirectory { dir };

    // Read the entire directory and get the shared library files (.*\.so\.[0-9]+)
    for (dirent *entry; (entry = readdir(dir)) != NULL;)
    {
        char const *const begin   = entry->d_name;
        char const *const lastDot = strrchr(begin, '.');

        if (lastDot == NULL || lastDot == begin)
        {
            continue;
        }

        {
            char const *it = lastDot + 1;

            /* d_name is garaunteed to be NULL terminated.
             * https://man7.org/linux/man-pages/man3/readdir.3.html
             *
             * No need to compare against a bounding end iterator
             */
            while (std::isdigit(*it))
            {
                ++it;
            }

            if ((*it != '\0') || (it == lastDot + 1))
            {
                continue;
            }
        }

        {
            static constexpr std::string_view reference("os.");
            const std::string_view stem(begin, lastDot);

            const auto mismatch = std::ranges::mismatch(std::views::reverse(stem), reference);

            if (mismatch.in2 != reference.end())
            {
                continue;
            }
        }

        // Tell dlopen to look in this directory
        thread_local char buf[sizeof(entry->d_name) + 2];
        snprintf(buf, sizeof(buf), "./%s", entry->d_name);

        // Do not load any plugins with different permissions / owner than the diagnostic binary
        if (!PluginPermissionsMatch(pluginDir, buf))
        {
            continue;
        }

        LoadLibrary(buf, entry->d_name);
    }
}

std::string TestFramework::GetPluginCudalessDir()
{
    return GetPluginBaseDir() + "/cudaless/";
}

/*****************************************************************************/
void TestFramework::loadPlugins()
{
    std::string cudalessPluginDir                   = GetPluginCudalessDir();
    std::string pluginCommonPath                    = cudalessPluginDir + "/libpluginCommon.so.4";
    std::optional<std::string> usingDriverPluginDir = GetPluginUsingDriverDir();

    // Load the pluginCommon library first so that those libraries can find the appropriate symbols
    LoadLibrary(pluginCommonPath.c_str(), "libpluginCommon.so.4");
    LoadPluginWithDir(cudalessPluginDir);

    if (usingDriverPluginDir)
    {
        LoadPluginWithDir(*usingDriverPluginDir);
    }
    else
    {
        log_error(
            "failed to get path of plugin using driver, it may lack of driver. Skip loading plugins that need driver.");
    }

    for (size_t pluginIdx = 0; pluginIdx < m_plugins.size(); pluginIdx++)
    {
        auto const &supportedTest = m_plugins[pluginIdx]->GetSupportedTests();
        for (auto const &[testName, pluginTest] : supportedTest)
        {
            auto *temp = new Test(pluginIdx,
                                  m_plugins[pluginIdx]->GetName(),
                                  pluginTest.GetTestName(),
                                  pluginTest.GetDescription(),
                                  pluginTest.GetTargetEntityGroup(),
                                  pluginTest.GetTestCategory());
            if (temp == nullptr)
            {
                log_error("failed to allocate structure for holding test: {}", testName);
                continue;
            }
            m_testList.insert(m_testList.end(), temp);
            InsertIntoTestCategory(pluginTest.GetTestCategory(), temp);
        }
    }
}

/*****************************************************************************/
void TestFramework::InsertIntoTestCategory(std::string groupName, Test *testObject)
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
        m_testCategories[token].push_back(testObject);
    }
}

dcgmReturn_t TestFramework::SetDiagResponseVersion(unsigned int version)
{
    return m_diagResponse.SetVersion(version);
}

/*****************************************************************************/
void TestFramework::Go(std::vector<std::unique_ptr<EntitySet>> &entitySets)
{
    bool pulseTestWillExecute = false;
    m_numTestsToRun           = 0;
    m_completedTests          = 0;

    m_diagResponse.PopulateDefault(entitySets);
    // Check if the pulse test will execute
    for (auto &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        GpuSet *gpuSet = ToGpuSet(entitySet.get());
        std::optional<std::vector<Test *>> testList;
        std::vector<Gpu *> gpuList = gpuSet->GetGpuObjs();

        testList = gpuSet->GetTestObjList(HARDWARE_TEST_OBJS);
        if (testList && testList->size() > 0)
        {
            pulseTestWillExecute = WillExecutePulseTest(*testList);
            if (pulseTestWillExecute)
            {
                break;
            }
        }

        testList = gpuSet->GetTestObjList(CUSTOM_TEST_OBJS);
        if (testList && testList->size() > 0)
        {
            pulseTestWillExecute = WillExecutePulseTest(*testList);
            if (pulseTestWillExecute)
            {
                break;
            }
        }
    }
    // Iterate over all the entitySets and get the total number of tests
    // that will be run.
    for (auto const &entitySet : entitySets)
    {
        m_numTestsToRun += entitySet->GetNumTests();
    }
    m_numTestsToRun++; // include the software test in the test count

    // software is a special plugin, which is not included in the m_plugins.
    // we use the size of m_plugins to represent its index.
    int const softwarePluginId = m_plugins.size();
    dcgmDiagPluginAttr_v1 pluginAttr { .pluginId = softwarePluginId };
    // run software External plugin
    runSoftwarePlugin(entitySets, &pluginAttr);

    // iterate through all entity sets
    for (auto &entitySet : entitySets)
    {
        std::optional<std::vector<Test *>> testList;

        testList = entitySet->GetTestObjList(HARDWARE_TEST_OBJS);
        if (testList && testList->size() > 0)
        {
            GoList(Test::NVVS_CLASS_HARDWARE, *testList, entitySet.get(), pulseTestWillExecute);
        }

        testList = entitySet->GetTestObjList(INTEGRATION_TEST_OBJS);
        if (testList && testList->size() > 0)
        {
            GoList(Test::NVVS_CLASS_INTEGRATION, *testList, entitySet.get(), pulseTestWillExecute);
        }

        testList = entitySet->GetTestObjList(PERFORMANCE_TEST_OBJS);
        if (testList && testList->size() > 0)
        {
            GoList(Test::NVVS_CLASS_PERFORMANCE, *testList, entitySet.get(), pulseTestWillExecute);
        }

        testList = entitySet->GetTestObjList(CUSTOM_TEST_OBJS);
        if (testList && testList->size() > 0)
        {
            GoList(Test::NVVS_CLASS_CUSTOM, *testList, entitySet.get(), pulseTestWillExecute);
        }
    }

    if (nvvsCommon.channelFd == -1)
    {
        m_diagResponse.Print();
    }
    else
    {
        if (!FdChannelClient(nvvsCommon.channelFd).Write(m_diagResponse.RawBinaryBlob()))
        {
            log_error("failed to write diag response to caller.");
        }
    }
}

/*****************************************************************************/
std::vector<dcgmDiagPluginEntityInfo_v1> TestFramework::PopulateEntityInfoForPlugins(EntitySet *entitySet)
{
    std::vector<dcgmDiagPluginEntityInfo_v1> entityInfo;
    auto const &entityIds = entitySet->GetEntityIds();

    for (size_t i = 0; i < entityIds.size(); i++)
    {
        dcgmDiagPluginEntityInfo_v1 ei = {};
        ei.entity.entityId             = entityIds[i];
        ei.entity.entityGroupId        = entitySet->GetEntityGroup();
        entityInfo.push_back(ei);
    }

    if (entitySet->GetEntityGroup() == DCGM_FE_GPU)
    {
        FillGpuEntityAuxData(entitySet, entityInfo);
    }

    return entityInfo;
}

std::string TestFramework::GetCompareName(const std::string &pluginName, bool reverse)
{
    std::string compareName;
    if (!reverse)
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
        if (test->GetTestName() == PULSE_TEST_PLUGIN_NAME)
        {
            return true;
        }
    }

    return false;
}

void TestFramework::WriteDiagStatusToChannel(std::string_view testName, unsigned int errorCode) const
{
    dcgmDiagStatus_v1 diagInfo = {};
    diagInfo.version           = dcgmDiagStatus_version1;
    diagInfo.totalTests        = m_numTestsToRun;
    diagInfo.completedTests    = m_completedTests;

    if (m_completedTests > m_numTestsToRun)
    {
        // Something is wrong with this run, do not write to the channel
        log_error("Completed test count {} greater than total test count {}", m_completedTests, m_numTestsToRun);
        return;
    }

    SafeCopyTo(diagInfo.testName, testName.data());
    diagInfo.errorCode = errorCode;
    std::span<char const> infoBinary(reinterpret_cast<char const *>(&diagInfo), sizeof(diagInfo));
    log_debug("Writing diag info for test {}", testName);
    if (!FdChannelClient(nvvsCommon.channelFd).Write(infoBinary))
    {
        log_error("Failed to write diag info to caller.");
    }
}

template <class T>
static unsigned int GetFirstError(std::vector<T> const &errors)
    requires(std::is_same_v<T, dcgmDiagErrorDetail_v2> || std::is_same_v<T, dcgmDiagError_v1>)
{
    unsigned int firstError = DCGM_FR_OK;
    if (errors.size() > 0)
    {
        firstError = errors[0].code;
    }
    return firstError;
}

/*****************************************************************************/
void TestFramework::GoList(Test::testClasses_enum classNum,
                           std::vector<Test *> testsList,
                           EntitySet *entitySet,
                           bool /* checkFileCreation */)
{
    auto const gpuIds = GetGpuIds(entitySet);

    // iterate through all tests giving them the GPU objects needed
    for (std::vector<Test *>::iterator testItr = testsList.begin(); testItr != testsList.end(); testItr++)
    {
        Test *test = (*testItr); // readability

        unsigned int vecSize = test->getArgVectorSize(classNum);
        for (unsigned int i = 0; i < vecSize; i++)
        {
            bool testSkipped       = false;
            TestParameters *tp     = test->popArgVectorElement(classNum);
            std::string pluginName = tp->GetString(PS_PLUGIN_NAME);

            unsigned int pluginIndex    = test->GetPluginIndex();
            std::string const &testName = test->GetTestName();

            if (pluginIndex >= m_plugins.size() || m_diagResponse.TestSlotsFull())
            {
                std::string errMsg = fmt::format("Invalid index {} or too many tests", pluginIndex);
                log_error(errMsg);
                std::vector<dcgmDiagSimpleResult_t> perGpuResults;
                std::vector<dcgmDiagErrorDetail_v2> errors;
                std::vector<dcgmDiagErrorDetail_v2> info;
                dcgmDiagErrorDetail_v2 error = {};
                error.code                   = -1;
                error.gpuId                  = -1;
                SafeCopyTo(error.msg, errMsg.c_str());
                errors.push_back(error);

                m_diagResponse.SetSystemError(errMsg, DCGM_ST_GENERIC_ERROR);
                continue;
            }

            if (testName == std::string(PULSE_TEST_PLUGIN_NAME))
            {
                // Add the iterations parameters
                tp->AddDouble(PULSE_TEST_STR_CURRENT_ITERATION, nvvsCommon.currentIteration);
                tp->AddDouble(PULSE_TEST_STR_TOTAL_ITERATIONS, nvvsCommon.totalIterations);
            }

            if (!skipRest && !main_should_stop)
            {
                dcgmDiagEntityResults_v2 const &entityResults
                    = m_plugins[pluginIndex]->GetEntityResults<dcgmDiagEntityResults_v2>(testName);
                DcgmRecorder dcgmRecorder(dcgmHandle.GetHandle());
                std::vector<dcgmDiagPluginEntityInfo_v1> entityInfos = PopulateEntityInfoForPlugins(entitySet);

                log_debug("Test {} start", testName);

                m_plugins[pluginIndex]->SetTestRunningState(testName, TestRuningState::Running);
                m_plugins[pluginIndex]->RunTest(testName, entityInfos, 600, tp);
                m_plugins[pluginIndex]->SetTestRunningState(testName, TestRuningState::Done);

                if (auto ret = m_diagResponse.SetTestResult(
                        pluginName, testName, entityResults, m_plugins[pluginIndex]->GetAuxData(testName));
                    ret != DCGM_ST_OK)
                {
                    log_error("failed to set test result to test [{}], ret: [{}].", testName, ret);
                }
            }
            else
            {
                if (auto ret = m_diagResponse.SetTestSkipped(pluginName, testName); ret != DCGM_ST_OK)
                {
                    log_error("failed to set skipped result to test [{}], ret: [{}].", testName, ret);
                }
                testSkipped = true;
            }

            m_diagResponse.AddTestCategory(testName, test->GetCategory());

            log_debug("Test {} had result {}. Configless is {}",
                      testName,
                      m_plugins[pluginIndex]->GetResult(testName),
                      nvvsCommon.configless);

            if (m_plugins[pluginIndex]->GetResult(testName) == NVVS_RESULT_FAIL
                && ((!nvvsCommon.configless) || nvvsCommon.failEarly))

            {
                skipRest = true;
            }

            unsigned int firstError = DCGM_FR_TEST_SKIPPED;
            if (!testSkipped && !(m_plugins[pluginIndex]->GetResult(testName) == NVVS_RESULT_SKIP))
            {
                firstError = GetFirstError(m_plugins[pluginIndex]->GetErrors(testName));
            }
            m_completedTests++;
            WriteDiagStatusToChannel(testName, firstError);
        }
    }
}

std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> TestFramework::GetSubtestParameters()
{
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> parms;
    auto addCommonParams = [](std::vector<dcgmDiagPluginParameterInfo_t> &p) {
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_SUITE_LEVEL, DcgmPluginParamInt });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_LOGFILE, DcgmPluginParamString });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_LOGFILE_TYPE, DcgmPluginParamFloat });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_RUN_IF_GOM_ENABLED, DcgmPluginParamBool });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_IGNORE_ERROR_CODES, DcgmPluginParamString });
    };

    for (unsigned int i = 0; i < m_plugins.size(); i++)
    {
        auto const &supportedTests = m_plugins[i]->GetSupportedTests();
        for (auto const &[testName, _] : supportedTests)
        {
            std::string Name = GetCompareName(testName, true);
            auto &p          = parms[Name];
            p                = m_plugins[i]->GetParameterInfo(testName);
            addCommonParams(p);
        }
    }

    auto &swParameterInfo = parms["software"];
    swParameterInfo       = GetSwParameterInfo();
    addCommonParams(swParameterInfo);
    return parms;
}


void TestFramework::runSoftwarePlugin(std::vector<std::unique_ptr<EntitySet>> const &entitySets,
                                      dcgmDiagPluginAttr_v1 const *pluginAttr)
{
    bool gpuSetFound = false;

    for (const auto &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        GpuSet *gpuSet = ToGpuSet(entitySet.get());
        // init SoftwarePlugin
        m_softwarePluginFramework = std::make_unique<SoftwarePluginFramework>(gpuSet->GetGpuObjs());

        // run softwarePlugin
        m_softwarePluginFramework->Run(m_diagResponse, pluginAttr, nvvsCommon.parms);
        gpuSetFound = true;
    }

    unsigned int firstError = DCGM_FR_TEST_SKIPPED;
    if (gpuSetFound)
    {
        m_diagResponse.AddTestCategory(SW_PLUGIN_NAME, SW_PLUGIN_CATEGORY);
        firstError = GetFirstError(m_softwarePluginFramework->GetErrors());
    }
    m_completedTests++;
    WriteDiagStatusToChannel(SW_PLUGIN_NAME, firstError);
}
