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
#ifndef NVVS_NVVS_SoftwarePluginFramework_H
#define NVVS_NVVS_SoftwarePluginFramework_H

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "Software.h"

#include <DcgmNvvsResponseWrapper.h>

class SoftwarePluginFramework
{
public:
    /*************************************************************************/
    /**
     * default constructor
     */
    SoftwarePluginFramework() = default;

    /*************************************************************************/
    /**
     * constructor - sets the variables to default values, initializes testname Map, populated data about GPU, set
     * output, load the plugin library, set the tests within the plugin
     * @params gpuList - list of gpu objects on which tests will be run
     */
    explicit SoftwarePluginFramework(std::vector<Gpu *> gpuList);

    /*************************************************************************/
    /**
     * constructor - allocates memory for m_entityList
     * @params entityList - unique_ptr with memory allocated via new or make_unique.
     */
    explicit SoftwarePluginFramework(std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList);

    /*************************************************************************/
    /**
     * destructor - deleting the pointers for the test list, closing the plugin
     */
    ~SoftwarePluginFramework();

    /*************************************************************************/
    /**
     * method for running the plugin. Iterates over all gpus, and for each gpu iterate over all the tests withing
     * software plugin, run each test.
     * @params diagResponse - Structure to store the test results
     * @params pluginAttr - Plugin attributes assigned from nvvs used for software plugin
     * @params userParms - User provided parameters.
     */
    void Run(DcgmNvvsResponseWrapper &diagResponse,
             dcgmDiagPluginAttr_v1 const *pluginAttr,
             std::map<std::string, std::map<std::string, std::string>> const &userParms);

    void SetSoftwarePlugin(std::unique_ptr<Software> softwareObj);


    /*************************************************************************/
    /**
     * Get all the errors from the last run. An empty vector is returned if there
     * are no errors.
     * Return: the errors vector
     */
    std::vector<dcgmDiagError_v1> const &GetErrors() const;


private:
    std::map<std::string, std::string> m_testNameMap;
    std::map<std::string, std::unique_ptr<TestParameters>> m_testParamMap;
    std::unique_ptr<Software> m_softwareObj;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> m_entityList;
    std::vector<dcgmDiagError_v1> m_errors;

protected:
    /*************************************************************************/
    /**
     * Method to initialize the test name map, which holds sub test names
     */
    void initTestNameMap();

    /*************************************************************************/
    /**
     * Method to initialize the test name map, which holds sub test names
     */
    void initTestParametersMap();

    /*************************************************************************/
    /**
     * method to set gpu parameters - gpuid, attributes, status
     * @params gpuList - list of gpu objects on which tests will be run
     */
    void populateGpuInfo(const std::vector<Gpu *> &gpuList);

    /*************************************************************************/
    /**
     * helper method to set the result
     * @params testName - Test name is being used to specify which test result to set
     * @params diagResponse - Structure to store the test result
     */
    void SetResult(std::string_view const testName, DcgmNvvsResponseWrapper &diagResponse);

    /*************************************************************************/
    /**
     * returns the testNameMap
     * Return: Map of key string and value string
     */
    std::map<std::string, std::string> const &getTestNameMap() const
    {
        return m_testNameMap;
    }

    /*************************************************************************/
    /**
     * method to get test parameters map
     * Return: Map, key string value unique_ptr
     */
    std::map<std::string, std::unique_ptr<TestParameters>> const &getTestParamMap()
    {
        return m_testParamMap;
    }

    /*************************************************************************/
    /**
     * method to return the entity information struct
     * Return: entity list struct dcgmDiagPluginEntityList_v1
     */
    dcgmDiagPluginEntityList_v1 const &getEntityList() const
    {
        return *(m_entityList);
    }
};

#endif //  NVVS_NVVS_SoftwarePluginFramework_H
