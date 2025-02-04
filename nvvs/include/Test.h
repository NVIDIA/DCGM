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
/* The Test class is a base interface to a plugin two primary goals:
 *    1   Obfuscate the plugin class from the rest of NVVS thus most
 *        calls are simply passed through to the corresponding plugin.
 *        But that allows plugins to be compiled completely separate.
 *    2   Catch exceptions thrown by the plugin and make sure they
 *        do not make it all the way up and kill all of NVVS.
 */
#ifndef _NVVS_NVVS_TEST_H
#define _NVVS_NVVS_TEST_H

#include "Gpu.h"
#include "NvvsCommon.h"
#include "NvvsStructs.h"
#include "Plugin.h"
#include "TestParameters.h"
#include <iostream>
#include <list>
#include <string>
#include <vector>

class Test
{
    /***************************PUBLIC***********************************/
public:
    Test(unsigned pluginIndex,
         std::string const &pluginName,
         std::string const &testName,
         std::string const &description,
         dcgm_field_entity_group_t targetEntityGroup,
         std::string category);
    ~Test();

    enum testClasses_enum
    {
        NVVS_CLASS_SOFTWARE,
        NVVS_CLASS_HARDWARE,
        NVVS_CLASS_INTEGRATION,
        NVVS_CLASS_PERFORMANCE,
        NVVS_CLASS_CUSTOM,
        NVVS_CLASS_LAST
    };

    void go(TestParameters *);
    void go(TestParameters *, dcgmDiagGpuInfo_t &);
    void go(TestParameters *, dcgmDiagGpuList_t &);

    std::string GetTestName() const
    {
        return m_name;
    }

    std::string getTestDesc() const
    {
        return m_description;
    }

    unsigned int getArgVectorSize(testClasses_enum num)
    {
        return (m_argMap[num]).size();
    }

    void pushArgVectorElement(testClasses_enum num, TestParameters *obj)
    {
        m_argMap[num].push_back(obj);
    }

    TestParameters *popArgVectorElement(testClasses_enum num)
    {
        if (m_argMap.find(num) == m_argMap.end())
            return 0;

        TestParameters *tp = m_argMap[num].front();
        if (tp)
            m_argMap[num].erase(m_argMap[num].begin());
        return tp;
    }

    // Get per-test log file tag to distinguish tests' log files from each other
    std::string GetLogFileTag() const
    {
        return m_name;
    }

    unsigned GetPluginIndex() const
    {
        return m_pluginIdx;
    }

    dcgm_field_entity_group_t GetTargetEntityGroup() const
    {
        return m_targetEntityGroup;
    }

    std::string const &GetPluginName() const
    {
        return m_pluginName;
    }

    std::string const &GetCategory() const
    {
        return m_category;
    }

    /***************************PRIVATE**********************************/
private:
    /* Methods */
    void getOut(std::string error);

    /* Variables */
    unsigned m_pluginIdx;
    std::string m_pluginName;
    std::string m_name;
    std::string m_category;
    std::map<testClasses_enum, std::vector<TestParameters *>> m_argMap;
    bool m_skipTest = false;
    std::string m_description;
    dcgm_field_entity_group_t m_targetEntityGroup;
    static const nvvsPluginGpuResults_t m_emptyGpuResults;
    static const nvvsPluginGpuMessages_t m_emptyGpuMessages;
    static const std::vector<std::string> m_emptyMessages;
    static const std::vector<DcgmError> m_emptyErrors;
    static const nvvsPluginGpuErrors_t m_emptyPerGpuErrors;

    /***************************PROTECTED********************************/
protected:
};

#endif //_NVVS_NVVS_TEST_H
