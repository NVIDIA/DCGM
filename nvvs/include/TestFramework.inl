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

#include <TestFramework.h>

#include "DcgmStringHelpers.h"
#include "PluginInterface.h"

#include <DcgmBuildInfo.hpp>
#include <DcgmHandle.h>
#include <DcgmSystem.h>
#include <Defer.hpp>
#include <FdChannelClient.h>
#include <PluginStrings.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <format>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

extern DcgmSystem dcgmSystem;
extern DcgmHandle dcgmHandle;
extern std::atomic_int32_t main_should_stop;

namespace impl
{

namespace
{

    void FillGpuEntityAuxData(EntitySet &entitySet, std::vector<dcgmDiagPluginEntityInfo_v1> &entityInfos)
    {
        assert(entitySet.GetEntityGroup() == DCGM_FE_GPU);
        GpuSet *gpuSet = ToGpuSet(&entitySet);
        assert(gpuSet);
        auto const &gpuObjs = gpuSet->GetGpuObjs();
        for (unsigned idx = 0; idx < entityInfos.size(); ++idx)
        {
            for (auto const &gpuObj : gpuObjs)
            {
                if (gpuObj->GetGpuId() == entityInfos[idx].entity.entityId
                    && entityInfos[idx].entity.entityGroupId == DCGM_FE_GPU)
                {
                    entityInfos[idx].auxField.gpu.attributes = gpuObj->GetAttributes();
                    entityInfos[idx].auxField.gpu.status     = gpuObj->GetDeviceEntityStatus();
                    break;
                }
            }
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

    void AppendSkippedEntityInfos(std::unordered_map<dcgm_field_eid_t, std::string> const &skippedEntities,
                                  dcgm_field_entity_group_t entityGroupId,
                                  dcgmDiagEntityResults_v2 &entityResults)
    {
        for (auto const &[entityId, reason] : skippedEntities)
        {
            if (entityResults.numInfo < DCGM_DIAG_RESPONSE_INFO_MAX_V2)
            {
                entityResults.info[entityResults.numInfo].entity.entityId      = entityId;
                entityResults.info[entityResults.numInfo].entity.entityGroupId = entityGroupId;
                SafeCopyTo(entityResults.info[entityResults.numInfo].msg, reason.c_str());
                entityResults.numInfo++;
            }
            else
            {
                log_error("Too many info messages. Skipping entity {} in group {}.", entityId, entityGroupId);
            }

            if (entityResults.numResults < DCGM_DIAG_TEST_RUN_RESULTS_MAX)
            {
                entityResults.results[entityResults.numResults].entity.entityId      = entityId;
                entityResults.results[entityResults.numResults].entity.entityGroupId = entityGroupId;
                entityResults.results[entityResults.numResults].result               = DCGM_DIAG_RESULT_SKIP;
                entityResults.numResults++;
            }
            else
            {
                log_error("Too many results. Skipping entity {} in group {}.", entityId, entityGroupId);
            }
        }
    }

} // namespace

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
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::TestFramework()
    : m_completedTests(0)
    , m_monitor(nullptr)
    , m_numTestsToRun(0)
    , m_testList()
    , m_testCategories()
    , dlList()
    , skipRest(false)
    , m_nvvsOwnerUid(0)
    , m_nvvsOwnerGid(0)
    , m_validGpuId()
    , m_plugins()
    , m_entityInfo()
    , m_skipLibraryList()
{
    InitSkippedLibraries();
}

/*****************************************************************************/
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::TestFramework(
    std::vector<std::unique_ptr<EntitySet>> &entitySets)
    : m_completedTests(0)
    , m_monitor(nullptr)
    , m_numTestsToRun(0)
    , m_testList()
    , m_testCategories()
    , dlList()
    , skipRest(false)
    , m_nvvsOwnerUid(0)
    , m_nvvsOwnerGid(0)
    , m_validGpuId()
    , m_plugins()
    , m_entityInfo()
    , m_skipLibraryList()
{
    for (auto const &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        GpuSet *gpuSet                    = ToGpuSet(entitySet.get());
        std::vector<Gpu *> const &gpuList = gpuSet->GetGpuObjs();
        if (!gpuList.empty())
        {
            m_validGpuId = gpuList[0]->GetGpuId();
        }
    }
    InitSkippedLibraries();
}

/*****************************************************************************/
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    InitSkippedLibraries()
{
    m_skipLibraryList.push_back("libcurand.so");
    m_skipLibraryList.push_back("libcupti.so");
}

/*****************************************************************************/
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::~TestFramework()
{
    // close the plugins
    for (std::vector<Test *>::iterator itr = m_testList.begin(); itr != m_testList.end(); itr++)
        if (*itr)
            delete (*itr);

    // close the shared libs
    for (std::list<void *>::iterator itr = dlList.begin(); itr != dlList.end(); itr++)
    {
        if (*itr)
        {
            m_dynamicLibrary.Close(*itr);
        }
    }
}

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
std::string TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    GetPluginBaseDir()
{
    char szTmp[32] = { 0 };
    char buf[1024] = { 0 };
    std::string binaryPath;
    std::string pluginsPath;
    std::vector<std::string> searchPaths;
    std::vector<std::string>::iterator pathIt;
    struct stat statBuf = {};

    snprintf(szTmp, sizeof(szTmp), "/proc/%d/exe", getpid());
    const ssize_t nRead = m_fileSystem.ReadLink(szTmp, buf, sizeof(buf) - 1);
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
    if (m_fileSystem.Stat(buf, &statBuf))
    {
        std::stringstream errBuf;
        errBuf << "Cannot stat NVVS binary '" << buf << "' : '" << strerror(errno)
               << "', so we cannot securely load the plugins.";
        log_error(errBuf.str());
        throw std::runtime_error(errBuf.str());
    }

    m_nvvsOwnerUid = statBuf.st_uid;
    m_nvvsOwnerGid = statBuf.st_gid;

    if (nvvsCommon.pluginPath.size() == 0)
    {
        binaryPath = binaryPath.substr(0, binaryPath.find_last_of("/"));

        searchPaths.push_back("/plugins");
        searchPaths.push_back("/../libexec/datacenter-gpu-manager-4/plugins");

        for (pathIt = searchPaths.begin(); pathIt != searchPaths.end(); pathIt++)
        {
            pluginsPath = binaryPath + (*pathIt);
            log_debug("Searching {} for plugins.", pluginsPath.c_str());
            if (m_fileSystem.Access(pluginsPath.c_str(), F_OK) == 0)
            {
                struct stat status;
                int st = m_fileSystem.Stat(pluginsPath.c_str(), &status);

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
        if (!m_fileSystem.IsDirectory(pluginsPath))
        {
            throw std::runtime_error("Plugins directory was not found.  Please check paths or use -p to set it.");
        }
    }

    return pluginsPath;
}

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
std::optional<std::string> TestFramework<PluginLibT,
                                         DynamicLibraryLoaderT,
                                         SoftwarePluginFrameworkT,
                                         FileSystemOperatorT>::GetPluginCudaDirExtension() const
{
    unsigned int cudaMajorVersion = dcgmSystem.GetCudaMajorVersion();
    unsigned int cudaMinorVersion = dcgmSystem.GetCudaMinorVersion();
    if (cudaMajorVersion == 0)
    {
        log_error("Unable to detect Cuda version.");
        return std::nullopt;
    }

    static const std::string CUDA_13_EXTENSION("/cuda13/");
    static const std::string CUDA_12_EXTENSION("/cuda12/");
    static const std::string CUDA_11_EXTENSION("/cuda11/");

    if (m_validGpuId.has_value())
    {
        cudaMajorVersion = GetCompatibleCudaMajorVersion(*m_validGpuId, cudaMajorVersion, cudaMinorVersion);
    }

    log_debug("The following Cuda version will be used for plugins: {}.{}", cudaMajorVersion, cudaMinorVersion);

    switch (cudaMajorVersion)
    {
        case 11:
            return CUDA_11_EXTENSION;
        case 12:
            return CUDA_12_EXTENSION;
        case 13:
            return CUDA_13_EXTENSION;
        default:
            if (cudaMajorVersion > MAX_CUDA_MAJOR_VERSION)
            {
                log_debug(
                    "Detected unsupported CUDA version {}; defaulting to {}", cudaMajorVersion, MAX_CUDA_MAJOR_VERSION);
                return CUDA_13_EXTENSION;
            }

            log_error("Detected unsupported Cuda version: {}.{}", cudaMajorVersion, cudaMinorVersion);
            throw std::runtime_error("Detected unsupported Cuda version");
    }

    // NOT-REACHED
    return std::nullopt;
}

/*****************************************************************************/
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
unsigned int TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    GetCompatibleCudaMajorVersion(unsigned int gpuId,
                                  unsigned int cudaDriverMajorVersion,
                                  unsigned int cudaDriverMinorVersion)
{
    if (cudaDriverMajorVersion == 0)
    {
        log_warning("User provided CUDA version is 0. Using default.");
        return cudaDriverMajorVersion;
    }

    auto chipArchitecture = dcgmSystem.GetGpuChipArchitecture(gpuId);
    if (chipArchitecture.is_error())
    {
        log_warning(
            "Unable to get GPU chip architecture for GPU {}, error: {}. Using provided cudaDriverMajorVersion: {}",
            gpuId,
            chipArchitecture.error(),
            cudaDriverMajorVersion);
        return cudaDriverMajorVersion;
    }

    // CUDA 13.0 drops support for everything that is < 7.5 SM Cuda Compatibility. These older SKUs need to run
    // a 12.9-linked application.
    // Note: We only check one GPU because NVVS can only run on homogenous GPUs.
    if ((*chipArchitecture == DCGM_CHIP_ARCH_MAXWELL || *chipArchitecture == DCGM_CHIP_ARCH_PASCAL
         || *chipArchitecture == DCGM_CHIP_ARCH_VOLTA)
        && cudaDriverMajorVersion == 13 && cudaDriverMinorVersion == 0)
    {
        log_info("Using CUDA 12 for GPU {} because it is Maxwell, Pascal, or Volta and CUDA 13.0 is detected", gpuId);
        return 12;
    }

    return cudaDriverMajorVersion;
}

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::LoadLibrary(
    const char *libraryPath,
    const char *libraryName)
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
        void *dlib = m_dynamicLibrary.Open(libraryPath, RTLD_LAZY);
        if (dlib == NULL)
        {
            char const *const err          = m_dynamicLibrary.Error();
            std::string const dlopen_error = err ? err : "";
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
        auto pl = std::make_unique<PluginLibT>();
        if (m_monitor != nullptr)
        {
            pl->SetHangDetectMonitor(m_monitor);
        }

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
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
bool TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    PluginPermissionsMatch(const std::string &pluginDir, const std::string &plugin)
{
    bool loadingPermitted = true;

    if (struct stat statBuf; m_fileSystem.Stat(plugin.c_str(), &statBuf))
    {
        log_warning("Failed to stat '{}' file: '{}'", plugin, strerror(errno));
        loadingPermitted = false;
    }
    else
    {
        if (statBuf.st_uid != m_nvvsOwnerUid)
        {
            log_info("'nvvs' file UID: {:o}", m_nvvsOwnerUid);
            log_info("'{}' file UID: {:o}", plugin, statBuf.st_uid);
            log_warning("'nvvs' executable and '{}' file do not share a UID", plugin);
            loadingPermitted = false;
        }

        if (statBuf.st_gid != m_nvvsOwnerGid)
        {
            log_info("'nvvs' file GID: {:o}", m_nvvsOwnerGid);
            log_info("'{}' file GID: {:o}", plugin, statBuf.st_gid);
            log_warning("'nvvs' executable and '{}' file do not share a GID", plugin);
            loadingPermitted = false;
        }

        if (statBuf.st_mode & S_IWGRP)
        {
            log_warning("'{}' file is group writable", plugin);
            loadingPermitted = false;
        }

        if (statBuf.st_mode & S_IWOTH)
        {
            log_warning("'{}' file is world writable", plugin);
            loadingPermitted = false;
        }
    }

    if (!loadingPermitted)
    {
        log_warning("Refusing to load plugin '{}/{}'", pluginDir, plugin);
    }

    return loadingPermitted;
}

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
std::optional<std::string> TestFramework<PluginLibT,
                                         DynamicLibraryLoaderT,
                                         SoftwarePluginFrameworkT,
                                         FileSystemOperatorT>::GetPluginUsingDriverDir()
{
    auto baseDir                                      = GetPluginBaseDir();
    std::optional<std::string> pluginCudaDirExtension = GetPluginCudaDirExtension();
    if (!pluginCudaDirExtension)
    {
        return std::nullopt;
    }
    return baseDir + *pluginCudaDirExtension;
}

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::LoadPluginWithDir(
    std::string const &pluginDir)
{
    char oldPath[2048] = { 0 };

    if (m_fileSystem.GetCurrentWorkingDirectory(oldPath, sizeof(oldPath)) == nullptr)
    {
        std::string errorMessage = std::format("Cannot load plugins: unable to get current dir: '{}'", strerror(errno));
        log_error(errorMessage);
        throw std::runtime_error(std::move(errorMessage));
    }

    if (m_fileSystem.ChangeDirectory(pluginDir.c_str()))
    {
        std::string errorMessage = std::format(
            "Error: Cannot load plugins. Unable to change to the plugin dir '{}': '{}'", pluginDir, strerror(errno));
        log_error(errorMessage);
        throw std::runtime_error(std::move(errorMessage));
    }

    const struct ResetWorkingDirectory
    {
        char const *path;
        FileSystemOperatorT *fs;
        ~ResetWorkingDirectory()
        {
            // Previous implementation did not account for possible failure. Deferring error handling
            if (fs != nullptr)
            {
                (void)fs->ChangeDirectory(path);
            }
        }
    } resetWorkingDirectory { oldPath, &m_fileSystem };

    std::optional<std::vector<std::string>> const entries = m_fileSystem.ListDirectoryEntries(pluginDir);
    if (!entries.has_value())
    {
        std::string errorMessage = std::format(
            "Error: Cannot load plugins: unable to open the plugin dir '{}': '{}'", pluginDir, strerror(errno));

        log_error(errorMessage);
        throw std::runtime_error(std::move(errorMessage));
    }

    // Read the entire directory and get the shared library files (.*\.so\.[0-9]+)
    for (std::string const &name : *entries)
    {
        char const *const begin   = name.c_str();
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
        thread_local char buf[256 + 2];
        snprintf(buf, sizeof(buf), "./%s", name.c_str());

        // Do not load any plugins with different permissions / owner than the diagnostic binary
        if (!PluginPermissionsMatch(pluginDir, buf))
        {
            continue;
        }

        LoadLibrary(buf, name.c_str());
    }
}

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
std::string TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    GetPluginCudalessDir()
{
    return GetPluginBaseDir() + "/cudaless/";
}

/*****************************************************************************/
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::loadPlugins(
    HangDetectMonitor *monitor)
{
    m_monitor = monitor;

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
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    InsertIntoTestCategory(std::string groupName, Test *testObject)
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

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
dcgmReturn_t TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    SetDiagResponseVersion(unsigned int version)
{
    return m_diagResponse.SetVersion(version);
}

/*****************************************************************************/
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::Go(
    std::vector<std::unique_ptr<EntitySet>> &entitySets)
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
    RunSoftwarePlugin(entitySets, &pluginAttr);

    std::array constexpr testClasses = { std::make_tuple(Test::NVVS_CLASS_HARDWARE, HARDWARE_TEST_OBJS),
                                         std::make_tuple(Test::NVVS_CLASS_INTEGRATION, INTEGRATION_TEST_OBJS),
                                         std::make_tuple(Test::NVVS_CLASS_PERFORMANCE, PERFORMANCE_TEST_OBJS),
                                         std::make_tuple(Test::NVVS_CLASS_CUSTOM, CUSTOM_TEST_OBJS) };

    // iterate through all entity sets
    for (auto &entitySet : entitySets)
    {
        for (auto const &[classNum, testObjList] : testClasses)
        {
            std::optional<std::vector<Test *>> testList = entitySet->GetTestObjList(testObjList);
            if (testList && testList->size() > 0)
            {
                GoList(classNum, *testList, entitySet.get(), pulseTestWillExecute);
            }
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
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
std::vector<dcgmDiagPluginEntityInfo_v1> TestFramework<PluginLibT,
                                                       DynamicLibraryLoaderT,
                                                       SoftwarePluginFrameworkT,
                                                       FileSystemOperatorT>::PopulateEntityInfoForPlugins(EntitySet &
                                                                                                              entitySet)
{
    std::vector<dcgmDiagPluginEntityInfo_v1> entityInfo = entitySet.PopulateEntityInfo();

    if (entitySet.GetEntityGroup() == DCGM_FE_GPU)
    {
        FillGpuEntityAuxData(entitySet, entityInfo);
    }
    return entityInfo;
}

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
std::string TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    GetCompareName(const std::string &pluginName, bool reverse)
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

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
bool TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    WillExecutePulseTest(std::vector<Test *> &testsList) const
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

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    WriteDiagStatusToChannel(std::string_view testName, unsigned int errorCode) const
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

/*****************************************************************************/
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::GoList(
    Test::testClasses_enum classNum,
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

            if (testName == PULSE_TEST_PLUGIN_NAME)
            {
                // Add the iterations parameters
                tp->AddDouble(PULSE_TEST_STR_CURRENT_ITERATION, nvvsCommon.currentIteration);
                tp->AddDouble(PULSE_TEST_STR_TOTAL_ITERATIONS, nvvsCommon.totalIterations);
            }

            std::chrono::milliseconds testDuration { 0 };
            if (!skipRest && !main_should_stop)
            {
                std::unordered_map<dcgm_field_eid_t, std::string> savedSkips;
                if (nvvsCommon.isEudOnly)
                {
                    savedSkips = entitySet->SaveAndClearRowRemapSkips();
                    if (!savedSkips.empty())
                    {
                        log_debug("EUD explicitly requested: bypassing {} row-remapping skip(s)", savedSkips.size());
                    }
                }

                DcgmNs::Defer restoreSkips([entitySet, &savedSkips]() {
                    if (!savedSkips.empty())
                    {
                        entitySet->RestoreSkips(savedSkips);
                    }
                });

                std::vector<dcgmDiagPluginEntityInfo_v1> entityInfos = PopulateEntityInfoForPlugins(*entitySet);

                log_debug("Test {} start", testName);
                auto const testStart = std::chrono::steady_clock::now();

                m_plugins[pluginIndex]->SetTestRunningState(testName, TestRuningState::Running);
                if (!entityInfos.empty())
                {
                    m_plugins[pluginIndex]->RunTest(testName, entityInfos, 600, tp);
                }
                m_plugins[pluginIndex]->SetTestRunningState(testName, TestRuningState::Done);
                testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()
                                                                                     - testStart);

                dcgmDiagEntityResults_v2 entityResults
                    = m_plugins[pluginIndex]->template GetEntityResults<dcgmDiagEntityResults_v2>(testName);
                AppendSkippedEntityInfos(entitySet->GetSkippedEntities(), entitySet->GetEntityGroup(), entityResults);
                if (auto ret = m_diagResponse.SetTestResult(
                        pluginName, testName, entityResults, m_plugins[pluginIndex]->GetAuxData(testName));
                    ret != DCGM_ST_OK)
                {
                    log_error("failed to set test result to test [{}], ret: [{}].", testName, ret);
                }
                entitySet->UpdateSkippedEntities(entityResults);

                if (nvvsCommon.isEudOnly && !savedSkips.empty())
                {
                    restoreSkips.Trigger();
                    log_debug("Restored {} row-remapping skip(s) after EUD", savedSkips.size());
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

            log_debug("{}",
                      FormatTestEndMessage(
                          testName, m_plugins[pluginIndex]->GetResult(testName), testDuration, nvvsCommon.configless));

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

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>>
TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::GetSubtestParameters()
{
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> parms;
    auto addCommonParams = [](std::vector<dcgmDiagPluginParameterInfo_t> &p) {
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_SUITE_LEVEL, DcgmPluginParamInt });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_LOGFILE, DcgmPluginParamString });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_LOGFILE_TYPE, DcgmPluginParamFloat });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_RUN_IF_GOM_ENABLED, DcgmPluginParamBool });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_IGNORE_ERROR_CODES, DcgmPluginParamString });
        p.emplace_back(dcgmDiagPluginParameterInfo_t { PS_USE_GENERIC_MODE, DcgmPluginParamBool });
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

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::
    CreateSoftwarePluginFrameworks(std::vector<std::unique_ptr<EntitySet>> const &entitySets)
{
    m_softwarePluginFrameworks.clear();
    for (const auto &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        GpuSet *gpuSet = ToGpuSet(entitySet.get());
        m_softwarePluginFrameworks.push_back(
            std::make_unique<SoftwarePluginFrameworkT>(gpuSet->GetGpuObjs(), *entitySet));
    }
}

template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
void TestFramework<PluginLibT, DynamicLibraryLoaderT, SoftwarePluginFrameworkT, FileSystemOperatorT>::RunSoftwarePlugin(
    std::vector<std::unique_ptr<EntitySet>> const &entitySets,
    dcgmDiagPluginAttr_v1 const *pluginAttr)
{
    size_t numGpuEntitySets = 0;
    for (auto const &es : entitySets)
    {
        if (es->GetEntityGroup() == DCGM_FE_GPU)
        {
            numGpuEntitySets += 1;
        }
    }

    assert(m_softwarePluginFrameworks.size() == numGpuEntitySets);
    if (m_softwarePluginFrameworks.size() != numGpuEntitySets)
    {
        log_error("Software plugin frameworks not initialized for this run: expected {} GPU frameworks, have {}. ",
                  numGpuEntitySets,
                  m_softwarePluginFrameworks.size());
        m_completedTests++;
        WriteDiagStatusToChannel(SW_PLUGIN_NAME, DCGM_FR_TEST_SKIPPED);
        return;
    }
    bool const gpuSetFound = numGpuEntitySets > 0;
    size_t gpuIdx          = 0;
    for (const auto &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }
        m_softwarePluginFrameworks[gpuIdx++]->Run(m_diagResponse, pluginAttr, nvvsCommon.parms);
    }

    unsigned int firstError = DCGM_FR_TEST_SKIPPED;
    if (gpuSetFound)
    {
        m_diagResponse.AddTestCategory(SW_PLUGIN_NAME, SW_PLUGIN_CATEGORY);
        firstError = GetFirstError(m_softwarePluginFrameworks.back()->GetErrors());
    }
    m_completedTests++;
    WriteDiagStatusToChannel(SW_PLUGIN_NAME, firstError);
}

} // namespace impl
