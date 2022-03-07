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
#ifndef TESTPARAMETERS_H
#define TESTPARAMETERS_H

#include <PluginInterface.h>
#include <map>
#include <string>

/* Parameter value types */
#define TP_T_STRING 0
#define TP_T_DOUBLE 1

/*****************************************************************************/
#define TP_ST_OK       0
#define TP_ST_BADPARAM 1 /* Bad parameter to function */
#define TP_ST_NOTFOUND                                                       \
    2                         /* The requested TestParameter or sub test was \
                                 not found */
#define TP_ST_ALREADYEXISTS 3 /* Tried to add TestParameter more than once */
#define TP_ST_CANTCOERCE                            \
    4 /* Was unable to coerce a value from one type \
                                 to another (double <=> string) */
#define TP_ST_OUTOFRANGE                             \
    5 /* Tried to set a value outside of the value's \
                                 allowed range */

/*****************************************************************************/
class TestParameterValue
{
public:
    TestParameterValue(std::string defaultValue);
    TestParameterValue(double defaultValue);
    TestParameterValue(const TestParameterValue &copyMe);
    ~TestParameterValue();

    int GetValueType();

    /*************************************************************************/
    /* Setters. Use this from the configuration file reader. These return a
     * TP_ST_? #define on error (!0)
     */
    int Set(std::string value);
    int Set(double value);

    /*************************************************************************/
    /* Getters. Note that the object also supports direct conversion to double
     * and std::string
     **/
    double GetDouble();
    std::string GetString();

    /*************************************************************************/

private:
    int m_valueType; /* TP_T_? #define of the value type */

    /* Actual parameter value */
    std::string m_stringValue;
    double m_doubleValue;
};

/*****************************************************************************/
/* Class for holding all of the parameters that will be passed to a test */
class TestParameters
{
public:
    TestParameters();
    ~TestParameters();
    TestParameters(TestParameters &copyMe);
    TestParameters &operator=(const TestParameters &other);

    /*************************************************************************/
    /* Add a global parameter to the test. Call this from the plugin stub */
    int AddString(std::string key, std::string value);
    int AddDouble(std::string key, double value);

    /* Add a subtest parameter. Call this from the plugin stub  */
    int AddSubTestString(std::string subTest, std::string key, std::string value);
    int AddSubTestDouble(std::string subTest, std::string key, double value);

    /*************************************************************************/
    /* Setters. Call these from the config parser */
    int SetString(std::string key, std::string value, bool silent = false);
    int SetDouble(std::string key, double value);

    /* Add a subtest parameter. Call this from the plugin stub  */
    int SetSubTestString(std::string subTest, std::string key, std::string value, bool create = false);
    int SetSubTestDouble(std::string subTest, std::string key, double value);

    /*************************************************************************/
    /* Getters. Call these from within the plugin */
    std::string GetString(std::string key);
    double GetDouble(std::string key);
    int GetBoolFromString(std::string key);
    std::string GetSubTestString(std::string subTest, std::string key);
    double GetSubTestDouble(std::string subTest, std::string key);
    int GetBoolFromSubTestString(std::string subTest, std::string key);
    bool HasKey(const std::string &key);

    /*************************************************************************/
    /*
     * Override / add the parameters from sourceTp into this TestParameters object
     *
     */
    int OverrideFrom(TestParameters *sourceTp);
    int OverrideFromString(const std::string &name, const std::string &value);

    /*************************************************************************/
    int SetFromStruct(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *tpStruct);

    /*************************************************************************/
    std::vector<dcgmDiagPluginTestParameter_t> GetParametersAsStruct() const;

    /*************************************************************************/

private:
    std::map<std::string, TestParameterValue *> m_globalParameters;
    std::map<std::string, std::map<std::string, TestParameterValue *>> m_subTestParameters;

    /*************************************************************************/
    int SetStructValue(TestParameterValue *tpv, dcgmDiagPluginTestParameter_t &param) const;
};

/*****************************************************************************/
#endif // TESTPARAMETERS_H
