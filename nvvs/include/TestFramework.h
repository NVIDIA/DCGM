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
/*
 * A base class responsible for looking for and loading all NVVS plugins.  Though
 * the Plugin objects are instantiated here through the factory interface, they are
 * destroyed as part of the Test destructor.  Test objects and the dynamic library
 * closing is done as part of this destructor.
 */
#ifndef _NVVS_NVVS_TestFramework_H
#define _NVVS_NVVS_TestFramework_H

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "DcgmRecorder.h"
#include "GoldenValueCalculator.h"
#include "GpuSet.h"
#include "NvvsStructs.h"
#include "PluginLib.h"
#include "SoftwarePluginFramework.h"
#include "Test.h"

#include <DcgmNvvsResponseWrapper.h>
#include <dcgm_structs.h>

class TestFramework
{
public:
    TestFramework();
    TestFramework(std::vector<std::unique_ptr<EntitySet>> &entitySet);
    ~TestFramework();

    // methods
    void Go(std::vector<std::unique_ptr<EntitySet>> &entitySets);
    void loadPlugins();

    // Getters
    std::vector<Test *> getTests()
    {
        return m_testList;
    }

    std::map<std::string, std::vector<Test *>> GetTestCategories()
    {
        return m_testCategories;
    }

    void LoadPlugins();

    /*
     * Runs the software plugin as a seperate entity
     */
    void runSoftwarePlugin(std::vector<std::unique_ptr<EntitySet>> const &entitySets,
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

protected:
    std::vector<Test *> m_testList;
    std::map<std::string, std::vector<Test *>> m_testCategories;
    std::list<void *> dlList;
    bool skipRest;
    mode_t m_nvvsBinaryMode;
    uid_t m_nvvsOwnerUid;
    gid_t m_nvvsOwnerGid;
    unsigned int m_validGpuId;
    std::unique_ptr<SoftwarePluginFramework> m_softwarePluginFramework;
    DcgmNvvsResponseWrapper m_diagResponse;
    unsigned int m_completedTests;
    unsigned int m_numTestsToRun;

    // new plugin loading
    std::vector<std::unique_ptr<PluginLib>> m_plugins;
    std::vector<dcgmDiagPluginEntityInfo_v1> m_entityInfo;
    std::vector<std::string> m_skipLibraryList;

    // methods
    std::string GetTestDisplayName(dcgmPerGpuTestIndices_t index);
    void InsertIntoTestCategory(std::string, Test *);
    void GoList(Test::testClasses_enum suite,
                std::vector<Test *> testsList,
                EntitySet *entitySet,
                bool checkFileCreation);
    void LoadLibrary(const char *libPath, const char *libName);
    void StartStatWatches(DcgmRecorder &dcgmRecorder, int pluginIndex, std::vector<Gpu *> gpuList);
    void EndStatWatches(DcgmRecorder &dcgmRecorder,
                        int pluginIndex,
                        timelib64_t startTime,
                        Test *test,
                        int logFileType);

    void LoadPluginWithDir(std::string const &pluginDir);

    /********************************************************************/
    std::optional<std::string> GetPluginUsingDriverDir();

    /********************************************************************/
    std::string GetPluginCudalessDir();

    /********************************************************************/
    std::vector<dcgmDiagPluginEntityInfo_v1> PopulateEntityInfoForPlugins(EntitySet *entitySet);

    /********************************************************************/
    /*
     * Determines and returns the base directory for the plugins, and gets the
     * appropriate permissions to check against later
     */
    std::string GetPluginBaseDir();

    /********************************************************************/
    /*
     * Checks the driver version and returns /cuda{version} to load the correct plugins
     * return std::nullopt if failed.
     */
    std::optional<std::string> GetPluginCudaDirExtension() const;

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
};

#endif //  _NVVS_NVVS_TestFramework_H
