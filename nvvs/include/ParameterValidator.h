/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef _NVVS_NVVS_ParameterValidator_H
#define _NVVS_NVVS_ParameterValidator_H

#include <PluginInterface.h>
#include <map>
#include <set>
#include <string>

typedef struct
{
    std::string testname;
    std::set<std::string> parameters;
} subtestInfo_t;

class TestInfo
{
public:
    void Clear();
    void AddParameter(const std::string &parameter);
    void SetName(const std::string &testname);

    bool HasParameter(const std::string &parameter) const;
    bool HasSubtest(const std::string &subtest) const;
    bool HasSubtestParameter(const std::string &subtest, const std::string &parameter) const;
    subtestInfo_t m_info;
    std::map<std::string, subtestInfo_t> m_subtests;
};

class ParameterValidator
{
public:
    ParameterValidator(const std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> &parms);
    ParameterValidator();

    /*
     * IsValidTestName()
     *
     * Return true if the test name is among the valid choices
     *        false if it isn't
     */
    bool IsValidTestName(const std::string &testname) const;

    /*
     * IsValidParameter()
     *
     * Return true if the parameter is among the valid choices for the specified test
     *        false if it isn't
     */
    bool IsValidParameter(const std::string &testname, const std::string &parameter) const;

    /*
     * IsValidSubtest()
     *
     * Return true if the subtest is among the valid choices for the specified test
     *        false if it isn't
     */
    bool IsValidSubtest(const std::string &testname, const std::string &subtest) const;


    /*
     * IsValidSubtestParameter()
     *
     * Return true if the parameter is among the valid choices for the specified test and subtest
     *        false if it isn't
     */
    bool IsValidSubtestParameter(const std::string &testname,
                                 const std::string &subtest,
                                 const std::string &parameter) const;

    /*
     * Performs the work of initializing the parameter validator from parms
     *
     */
    void Initialize(const std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> &parms);

    /*
     * Makes sure the test name is always lower case and never has spaces
     *
     * Return a lower-case version of testname with all spaces transformed to underscores
     */
    static std::string TransformTestName(const std::string &testname);

private:
    std::map<std::string, TestInfo> m_possiblePlugins;
};

#endif
