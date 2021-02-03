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
#include <algorithm>

#include "ParameterValidator.h"
#include "PluginInterface.h"
#include "PluginStrings.h"

void TestInfo::SetName(const std::string &testname)
{
    m_info.testname = testname;
}

void TestInfo::AddParameter(const std::string &parameter)
{
    m_info.parameters.insert(parameter);
}

void TestInfo::Clear()
{
    m_info.testname.clear();
    m_info.parameters.clear();
    m_subtests.clear();
}

void TestInfo::AddSubtest(const std::string &subtest)
{
    m_subtests[subtest].testname = subtest;
}

void TestInfo::AddSubtestParameter(const std::string &subtest, const std::string &parameter)
{
    m_subtests[subtest].parameters.insert(parameter);
}

bool TestInfo::HasParameter(const std::string &parameter) const
{
    return m_info.parameters.find(parameter) != m_info.parameters.end();
}

bool TestInfo::HasSubtestParameter(const std::string &subtest, const std::string &parameter) const
{
    auto iter = m_subtests.find(subtest);

    return !(iter == m_subtests.end() || iter->second.parameters.find(parameter) == iter->second.parameters.end());
}

bool TestInfo::HasSubtest(const std::string &subtest) const
{
    return m_subtests.find(subtest) != m_subtests.end();
}

bool ParameterValidator::IsValidTestName(const std::string &testname) const
{
    std::string loweredTest(testname);
    std::transform(loweredTest.begin(), loweredTest.end(), loweredTest.begin(), ::tolower);
    return m_possiblePlugins.find(loweredTest) != m_possiblePlugins.end();
}

bool ParameterValidator::IsValidParameter(const std::string &testname, const std::string &parameter) const
{
    std::string loweredTest(testname);
    std::string loweredParam(parameter);
    std::transform(loweredTest.begin(), loweredTest.end(), loweredTest.begin(), ::tolower);
    std::transform(loweredParam.begin(), loweredParam.end(), loweredParam.begin(), ::tolower);
    auto iter = m_possiblePlugins.find(loweredTest);

    return !(iter == m_possiblePlugins.end() || !iter->second.HasParameter(loweredParam));
}

bool ParameterValidator::IsValidSubtest(const std::string &testname, const std::string &subtest) const
{
    std::string loweredTest(testname);
    std::string loweredSub(subtest);
    std::transform(loweredTest.begin(), loweredTest.end(), loweredTest.begin(), ::tolower);
    std::transform(loweredSub.begin(), loweredSub.end(), loweredSub.begin(), ::tolower);

    auto iter = m_possiblePlugins.find(loweredTest);

    return !(iter == m_possiblePlugins.end() || !iter->second.HasSubtest(loweredSub));
}

bool ParameterValidator::IsValidSubtestParameter(const std::string &testname,
                                                 const std::string &subtest,
                                                 const std::string &parameter) const
{
    std::string loweredTest(testname);
    std::string loweredSub(subtest);
    std::string loweredParam(parameter);

    std::transform(loweredTest.begin(), loweredTest.end(), loweredTest.begin(), ::tolower);
    std::transform(loweredSub.begin(), loweredSub.end(), loweredSub.begin(), ::tolower);
    std::transform(loweredParam.begin(), loweredParam.end(), loweredParam.begin(), ::tolower);

    auto iter = m_possiblePlugins.find(loweredTest);

    return !(iter == m_possiblePlugins.end() || !iter->second.HasSubtestParameter(subtest, parameter));
}

ParameterValidator::ParameterValidator(const std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> &parms)
{
    for (auto const &cur : parms)
    {
        TestInfo ti;
        ti.SetName(cur.first);
        for (auto const &iter : cur.second)
        {
            ti.AddParameter(iter.parameterName);
        }

        m_possiblePlugins[cur.first] = ti;
    }
}