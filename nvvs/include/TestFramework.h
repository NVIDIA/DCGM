/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
/*
 * A base class responsible for looking for and loading all NVVS plugins.  Though
 * the Plugin objects are instantiated here through the factory interface, they are
 * destroyed as part of the Test destructor.  Test objects and the dynamic library
 * closing is done as part of this destructor.
 */
#ifndef _NVVS_NVVS_TestFramework_H
#define _NVVS_NVVS_TestFramework_H

#include <map>
#include <memory>
#include <optional>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "DcgmRecorder.h"
#include "DynamicLibraryLoader.h"
#include "FileSystemOperator.h"
#include "GoldenValueCalculator.h"
#include "GpuSet.h"
#include "HangDetectMonitor.h"
#include "NvvsStructs.h"
#include "PluginLib.h"
#include "SoftwarePluginFramework.h"
#include "Test.h"

#include <DcgmNvvsResponseWrapper.h>
#include <dcgm_structs.h>

namespace impl
{

/** @brief TestFramework is a base class for running NVVS tests.
 *
 * @tparam PluginLibT The type of the plugin library.
 * @tparam DynamicLibraryLoaderT The type of the dynamic library loader.
 * @tparam SoftwarePluginFrameworkT The type of the software plugin framework.
 * @tparam FileSystemOperatorT The type of the filesystem operator.
 */
template <typename PluginLibT,
          typename DynamicLibraryLoaderT,
          typename SoftwarePluginFrameworkT,
          typename FileSystemOperatorT>
class TestFramework
{
public:
    TestFramework();
    TestFramework(std::vector<std::unique_ptr<EntitySet>> &entitySet);
    ~TestFramework();

    // methods
    void Go(std::vector<std::unique_ptr<EntitySet>> &entitySets);

    void loadPlugins(HangDetectMonitor *monitor);

    // Getters
    std::vector<Test *> getTests()
    {
        return m_testList;
    }

    std::map<std::string, std::vector<Test *>> GetTestCategories()
    {
        return m_testCategories;
    }

    /**
     * @brief Builds one @c SoftwarePluginFrameworkT per GPU entity set.
     *
     * Instances are stored in @c m_softwarePluginFrameworks. Call before @c Go() or @c RunSoftwarePlugin();
     * callers may adjust instances (for example test mocks) after creation.
     *
     * @param[in] entitySets One framework is created for each GPU entity set in this vector.
     */
    void CreateSoftwarePluginFrameworks(std::vector<std::unique_ptr<EntitySet>> const &entitySets);

    /**
     * @brief Runs the software plugin as a separate entity.
     *
     * Requires @c m_softwarePluginFrameworks to match the GPU entity set count from
     * @c CreateSoftwarePluginFrameworks(); otherwise logs an error, records failure, and returns without running.
     *
     * @param[in] entitySets GPU entity sets corresponding to the software plugin frameworks.
     * @param[in] pluginAttr Plugin attributes passed to the software plugin run.
     */
    void RunSoftwarePlugin(std::vector<std::unique_ptr<EntitySet>> const &entitySets,
                           dcgmDiagPluginAttr_v1 const *pluginAttr);

    /********************************************************************/
    /*
     * Returns the name that should be used to compare the local name with the name given from the plugin
     *
     * Transforms ' ' and '_' depending on the value of "reverse"
     */
    std::string GetCompareName(const std::string &pluginName, bool reverse = false);

    /********************************************************************/
    /*
     * Returns a map of test names and their valid parameters
     */
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> GetSubtestParameters();

    dcgmReturn_t SetDiagResponseVersion(unsigned int version);

    /********************************************************************/
    /**
     * Determines the appropriate CUDA major version to use based on GPU architecture compatibility
     *
     * CUDA 13.0 drops support for Maxwell, Pascal, and Volta GPUs (compute capability < 7.5).
     * This static method checks the GPU architecture and returns CUDA 12 for older GPUs when CUDA 13.0 is detected.
     *
     * @param gpuId GPU ID to check architecture for
     * @param cudaDriverMajorVersion Current CUDA driver major version
     * @param cudaDriverMinorVersion Current CUDA driver minor version
     * @return unsigned int The CUDA major version to use (12 for older GPUs with CUDA 13.0, otherwise original version)
     */
    static unsigned int GetCompatibleCudaMajorVersion(unsigned int gpuId,
                                                      unsigned int cudaDriverMajorVersion,
                                                      unsigned int cudaDriverMinorVersion);

protected:
    /** @brief Put methods and members that are not part of the public interface in protected section so that the test
 derived class can access them */

    /** @brief Aggregated diagnostic response and result slots for NVVS. */
    DcgmNvvsResponseWrapper m_diagResponse;

    /** @brief One software plugin framework per GPU entity set (see @c CreateSoftwarePluginFrameworks). */
    std::vector<std::unique_ptr<SoftwarePluginFrameworkT>> m_softwarePluginFrameworks;

    /** @brief Loads and unloads plugin shared objects. */
    DynamicLibraryLoaderT m_dynamicLibrary;

    /** @brief Filesystem operations used to locate plugins and validate paths. */
    FileSystemOperatorT m_fileSystem;

    /**
     * @brief Resolves and returns the absolute plugin base directory.
     *
     * Also records the NVVS binary owner and group for subsequent plugin permission checks.
     *
     * @return Path to the directory containing plugin subdirectories.
     * @throws std::runtime_error If the binary path cannot be read, the binary cannot be stat'd, or no plugin directory
     *         is found.
     */
    std::string GetPluginBaseDir();

    /**
     * @brief Returns the path to CUDA-free plugins (@c cudaless), under the plugin base directory.
     *
     * @return Base plugin path concatenated with @c "/cudaless/".
     */
    std::string GetPluginCudalessDir();

    /**
     * @brief Appends a loaded plugin library to the internal plugin list.
     *
     * Used when constructing the framework with pre-loaded or mock plugins (for example in unit tests).
     *
     * @param[in] plugin Ownership of the plugin library instance; stored in @c m_plugins.
     */
    void PushPlugin(std::unique_ptr<PluginLibT> plugin)
    {
        m_plugins.push_back(std::move(plugin));
    }

private:
    /** @brief Number of tests that have completed execution. */
    unsigned int m_completedTests;

    /** @brief Optional hang-detection monitor set during @c loadPlugins. */
    HangDetectMonitor *m_monitor;

    /** @brief Total number of tests scheduled to run. */
    unsigned int m_numTestsToRun;

    std::vector<Test *> m_testList;
    std::map<std::string, std::vector<Test *>> m_testCategories;
    std::list<void *> dlList;
    bool skipRest;
    uid_t m_nvvsOwnerUid;
    gid_t m_nvvsOwnerGid;
    // This is one of the GPU IDs used for the tests. If a GPU set is present, it is set to the first GPU in the entity
    // set. Since NVVS can only run on homogenous GPUs, this is sufficient for us to determine the CUDA version, GPU
    // Arch, etc.
    std::optional<dcgm_field_eid_t> m_validGpuId;

    // new plugin loading
    std::vector<std::unique_ptr<PluginLibT>> m_plugins;
    std::vector<dcgmDiagPluginEntityInfo_v1> m_entityInfo;
    std::vector<std::string> m_skipLibraryList;

    // methods
    void InsertIntoTestCategory(std::string, Test *);
    void LoadLibrary(const char *libPath, const char *libName);
    void StartStatWatches(DcgmRecorder &dcgmRecorder, int pluginIndex, std::vector<Gpu *> gpuList);
    void EndStatWatches(DcgmRecorder &dcgmRecorder,
                        int pluginIndex,
                        timelib64_t startTime,
                        Test *test,
                        int logFileType);

    void LoadPluginWithDir(std::string const &pluginDir);

    /********************************************************************/
    std::vector<dcgmDiagPluginEntityInfo_v1> PopulateEntityInfoForPlugins(EntitySet &entitySet);

    /********************************************************************/
    /*
     * Returns true if the plugin has the same permissions and owner as the NVVS binary
     */
    bool PluginPermissionsMatch(const std::string &pluginDir, const std::string &plugin);

    /********************************************************************/
    /*
     * Initializes the values that should be in the skipped libraries list
     */
    void InitSkippedLibraries();

    /********************************************************************/
    /*
     * Returns true if we are going to execute the pulse test
     */
    bool WillExecutePulseTest(std::vector<Test *> &testsList) const;

    /********************************************************************/
    /*
     * Writes dcgmDiagStatus_t to the NVVS channel
     */
    void WriteDiagStatusToChannel(std::string_view pluginName, unsigned int errorCode) const;

    /**
     * @brief Returns the CUDA-version subdirectory name for driver-dependent plugins (for example @c "/cuda12/").
     *
     * Uses @c GetCompatibleCudaMajorVersion when a valid GPU id is known. Returns nullopt if the CUDA version cannot
     * be determined or is not mapped to a plugin tree.
     *
     * @return Subdirectory name including leading and trailing slashes, or @c std::nullopt on failure.
     */
    std::optional<std::string> GetPluginCudaDirExtension() const;

    /**
     * @brief Runs a list of tests for one entity set by dequeuing parameters and invoking the loaded plugins.
     *
     * Iterates @p testsList for @p suite, pops @c TestParameters per test, validates plugin indices, and calls into the
     * plugin @c RunTest path as appropriate for the entity set.
     *
     * @param[in] suite Test class / suite identifier used to select the argument vector on each @c Test.
     * @param[in] testsList Tests to execute for this pass.
     * @param[in,out] entitySet GPU entity set (GPUs and skip state) for this run; may be updated (for example skip
     *                state) during execution.
     * @param[in] checkFileCreation Reserved; currently unused in the implementation.
     */
    void GoList(Test::testClasses_enum suite,
                std::vector<Test *> testsList,
                EntitySet *entitySet,
                bool checkFileCreation);

    /**
     * @brief Returns the full path to CUDA-specific, driver-dependent plugins.
     *
     * Combines @c GetPluginBaseDir() with @c GetPluginCudaDirExtension().
     *
     * @return Full plugin directory path, or @c std::nullopt if the CUDA extension cannot be resolved.
     */
    std::optional<std::string> GetPluginUsingDriverDir();
};

} // namespace impl

/*
 * Production type: default plugin backend is PluginLib. The class template lives in impl::TestFramework.
 */
using TestFramework = impl::TestFramework<PluginLib, DynamicLibraryLoader, SoftwarePluginFramework, FileSystemOperator>;

#endif //  _NVVS_NVVS_TestFramework_H
