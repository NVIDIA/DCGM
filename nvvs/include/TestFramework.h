/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "Output.h"
#include "PluginLib.h"
#include "Test.h"

class TestFramework
{
public:
    TestFramework();
    TestFramework(bool jsonOutput, GpuSet *gpuSet);
    ~TestFramework();

    // methods
    void go(std::vector<std::unique_ptr<GpuSet>> &gpuSet);
    void loadPlugins();
    void addInfoStatement(const std::string &info);

    // Getters
    std::vector<Test *> getTests()
    {
        return m_testList;
    }

    std::map<std::string, std::vector<Test *>> getTestGroups()
    {
        return testGroup;
    }

    void LoadPlugins();

    int GetPluginIndex(Test::testClasses_enum classNum, const std::string &pluginName);

    /********************************************************************/
    /*
     * Returns the name that should be used to compare the local name with the name given from the plugin
     *
     * Transforms ' ' and '_' depending on the value of "reverse"
     */
    std::string GetCompareName(Test::testClasses_enum classNum, const std::string &testName, bool reverse = false);

    /********************************************************************/
    /*
     * Returns a map of test names and their valid parameters
     */
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> GetSubtestParameters();

    std::string GetPluginNameFromTestName(std::string const &testName) const;

protected:
    std::vector<Test *> m_testList;
    std::map<std::string, std::vector<Test *>> testGroup;
    std::list<void *> dlList;
    Output *m_output;
    bool skipRest;
    mode_t m_nvvsBinaryMode;
    uid_t m_nvvsOwnerUid;
    gid_t m_nvvsOwnerGid;
    unsigned int m_validGpuId;

    // new plugin loading
    std::vector<std::unique_ptr<PluginLib>> m_plugins;
    std::vector<dcgmDiagPluginGpuInfo_t> m_gpuInfo;
    std::vector<std::string> m_skipLibraryList;
    // As one plugin can provide more than one test, this map holds the test name to the name of its belonging.
    std::unordered_map<std::string, std::string> m_testNameToPluginName;

    // methods
    std::string GetTestDisplayName(dcgmPerGpuTestIndices_t index);
    void insertIntoTestGroup(std::string, Test *);
    void goList(Test::testClasses_enum suite,
                std::vector<Test *> testsList,
                std::vector<Gpu *> gpuList,
                bool checkFileCreation);
    void LoadLibrary(const char *libPath, const char *libName);
    void GetAndOutputHeader(Test::testClasses_enum classNum);
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
    void PopulateGpuInfoForPlugins(std::vector<Gpu *> &gpuList, std::vector<dcgmDiagPluginGpuInfo_t> &gpuInfo);

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

    /***************************PROTECTED********************************/
protected:
};

#endif //  _NVVS_NVVS_TestFramework_H
