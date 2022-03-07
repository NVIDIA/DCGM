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
#include "TestParameters.h"
#include "NvvsCommon.h"
#include "float.h"
#include <cstdlib>
#include <sstream>

/*****************************************************************************/
TestParameterValue::TestParameterValue(std::string defaultValue)
    : m_valueType(TP_T_STRING)
    , m_stringValue(defaultValue)
    , m_doubleValue(0.0)
{}

/*****************************************************************************/
TestParameterValue::TestParameterValue(double defaultValue)
    : m_valueType(TP_T_DOUBLE)
    , m_stringValue("")
    , m_doubleValue(defaultValue)
{}

/*****************************************************************************/
TestParameterValue::TestParameterValue(const TestParameterValue &copyMe)
    : m_valueType(copyMe.m_valueType)
    , m_stringValue(copyMe.m_stringValue)
    , m_doubleValue(copyMe.m_doubleValue)
{}

/*****************************************************************************/
int TestParameterValue::GetValueType()
{
    return m_valueType;
}

/*****************************************************************************/
TestParameterValue::~TestParameterValue()
{
    m_stringValue = "";
}

/*****************************************************************************/
int TestParameterValue::Set(std::string value)
{
    /* Possibly coerce the value into the other type */
    if (m_valueType == TP_T_STRING)
    {
        m_stringValue = value;
        return 0;
    }

    double beforeDoubleValue = m_doubleValue;
    m_doubleValue            = std::stof(value);

    if (m_doubleValue == 0.0 && value.c_str()[0] != '0')
    {
        m_doubleValue = beforeDoubleValue;
        return TP_ST_CANTCOERCE; /* atof failed. Must be a bad value */
    }

    return 0;
}

/*****************************************************************************/
int TestParameterValue::Set(double value)
{
    /* Possibly coerce the value into the other type */
    if (m_valueType == TP_T_DOUBLE)
    {
        m_doubleValue = value;
        return 0;
    }

    std::stringstream ss;
    ss << value;
    m_stringValue = ss.str();
    return 0;
}

/*****************************************************************************/
double TestParameterValue::GetDouble(void)
{
    /* Possibly coerce the value into the other type */
    if (m_valueType == TP_T_DOUBLE)
    {
        return m_doubleValue;
    }

    return std::stof(m_stringValue);
}

/*****************************************************************************/
std::string TestParameterValue::GetString(void)
{
    /* Possibly coerce the value into the other type */
    if (m_valueType == TP_T_STRING)
    {
        return m_stringValue;
    }

    std::stringstream ss;
    ss << m_doubleValue;
    return ss.str();
}

/*****************************************************************************/
/*****************************************************************************/
TestParameters::TestParameters()
{
    m_globalParameters.clear();
    m_subTestParameters.clear();
}

/*****************************************************************************/
TestParameters::~TestParameters()
{
    std::map<std::string, TestParameterValue *>::iterator it;

    for (it = m_globalParameters.begin(); it != m_globalParameters.end(); it++)
    {
        if (it->second)
            delete (it->second);
    }
    m_globalParameters.clear();

    std::map<std::string, std::map<std::string, TestParameterValue *>>::iterator outerIt;

    for (outerIt = m_subTestParameters.begin(); outerIt != m_subTestParameters.end(); outerIt++)
    {
        for (it = outerIt->second.begin(); it != outerIt->second.end(); it++)
        {
            delete (it->second);
        }
        outerIt->second.clear();
    }
    m_subTestParameters.clear();
}

/*****************************************************************************/
TestParameters::TestParameters(TestParameters &copyMe)
{
    /* do a deep copy of the source object */
    std::map<std::string, TestParameterValue *>::iterator it;

    for (it = copyMe.m_globalParameters.begin(); it != copyMe.m_globalParameters.end(); it++)
    {
        m_globalParameters[std::string(it->first)] = new TestParameterValue(*(it->second));
    }

    std::map<std::string, std::map<std::string, TestParameterValue *>>::iterator outerIt;

    for (outerIt = copyMe.m_subTestParameters.begin(); outerIt != copyMe.m_subTestParameters.end(); outerIt++)
    {
        for (it = outerIt->second.begin(); it != outerIt->second.end(); it++)
        {
            m_subTestParameters[outerIt->first][it->first] = new TestParameterValue(*(it->second));
        }
    }
}

TestParameters &TestParameters::operator=(const TestParameters &other)
{
    if (this == &other)
    {
        return *this;
    }

    for (auto &[name, value] : other.m_globalParameters)
    {
        m_globalParameters[name] = new TestParameterValue(*value);
    }

    for (auto &[subtest, parameters] : other.m_subTestParameters)
    {
        for (auto &[name, value] : parameters)
        {
            m_subTestParameters[subtest][name] = new TestParameterValue(*value);
        }
    }

    return *this;
}

/*****************************************************************************/
int TestParameters::AddString(std::string key, std::string value)
{
    if (m_globalParameters.find(key) != m_globalParameters.end())
    {
        PRINT_WARNING("%s %s", "Tried to add parameter %s => %s, but it already exists", key.c_str(), value.c_str());
        return TP_ST_ALREADYEXISTS;
    }

    m_globalParameters[key] = new TestParameterValue((std::string)value);
    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::AddDouble(std::string key, double value)
{
    if (m_globalParameters.find(key) != m_globalParameters.end())
    {
        DCGM_LOG_WARNING << "Tried to add parameter " << key << " => " << value << ", but it already exists";
        return TP_ST_ALREADYEXISTS;
    }

    m_globalParameters[key] = new TestParameterValue((double)value);
    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::AddSubTestString(std::string subTest, std::string key, std::string value)
{
    std::map<std::string, std::map<std::string, TestParameterValue *>>::iterator outerIt;
    std::map<std::string, TestParameterValue *>::iterator it;

    outerIt = m_subTestParameters.find(subTest);
    if (outerIt != m_subTestParameters.end())
    {
        it = outerIt->second.find(key);
        if (it != outerIt->second.end())
        {
            PRINT_WARNING("%s %s %s",
                          "Tried to add subtest %s parameter %s => %s, "
                          "but it already exists",
                          subTest.c_str(),
                          key.c_str(),
                          value.c_str());
            return TP_ST_ALREADYEXISTS;
        }
    }

    m_subTestParameters[subTest][key] = new TestParameterValue((std::string)value);
    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::AddSubTestDouble(std::string subTest, std::string key, double value)
{
    std::map<std::string, std::map<std::string, TestParameterValue *>>::iterator outerIt;
    std::map<std::string, TestParameterValue *>::iterator it;

    outerIt = m_subTestParameters.find(subTest);
    if (outerIt != m_subTestParameters.end())
    {
        it = outerIt->second.find(key);
        if (it != outerIt->second.end())
        {
            PRINT_WARNING("%s %s %f",
                          "Tried to add subtest %s parameter %s => %f, but it already exists",
                          subTest.c_str(),
                          key.c_str(),
                          value);
            return TP_ST_ALREADYEXISTS;
        }
    }

    m_subTestParameters[subTest][key] = new TestParameterValue((double)value);
    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::SetString(std::string key, std::string value, bool silent)
{
    if (m_globalParameters.find(key) == m_globalParameters.end())
    {
        if (silent == false)
        {
            PRINT_WARNING("%s %s", "Tried to set unknown parameter %s to %s", key.c_str(), value.c_str());
        }
        return TP_ST_NOTFOUND;
    }

    int st = m_globalParameters[key]->Set((std::string)value);
    DCGM_LOG_DEBUG << "Set global parameter " << key << " -> " << value << ". st " << st;
    return st;
}

/*****************************************************************************/
int TestParameters::SetDouble(std::string key, double value)
{
    if (m_globalParameters.find(key) == m_globalParameters.end())
    {
        PRINT_WARNING("%s %f", "Tried to set unknown parameter %s to %f", key.c_str(), value);
        return TP_ST_NOTFOUND;
    }

    int st = m_globalParameters[key]->Set(value);
    DCGM_LOG_DEBUG << "Set global parameter " << key << " -> " << value << ". st " << st;
    return st;
}

/*****************************************************************************/
int TestParameters::SetSubTestString(std::string subTest, std::string key, std::string value, bool create)
{
    std::map<std::string, std::map<std::string, TestParameterValue *>>::iterator outerIt;

    outerIt = m_subTestParameters.find(subTest);
    if (outerIt == m_subTestParameters.end())
    {
        if (create)
        {
            return AddSubTestString(subTest, key, value);
        }
        else
        {
            DCGM_LOG_WARNING << "Tried to set unknown subtest " << subTest << "'s parameter " << key << " to " << value;
            return TP_ST_NOTFOUND;
        }
    }
    else if (outerIt->second.find(key) == outerIt->second.end())
    {
        if (create)
        {
            return AddSubTestString(subTest, key, value);
        }
        else
        {
            DCGM_LOG_WARNING << "Tried to set subtest " << subTest << "'s unknown parameter " << key << " to " << value;
            return TP_ST_NOTFOUND;
        }
    }

    return m_subTestParameters[subTest][key]->Set((std::string)value);
}

/*****************************************************************************/
int TestParameters::SetSubTestDouble(std::string subTest, std::string key, double value)
{
    std::map<std::string, std::map<std::string, TestParameterValue *>>::iterator outerIt;

    outerIt = m_subTestParameters.find(subTest);
    if (outerIt == m_subTestParameters.end())
    {
        PRINT_WARNING(
            "%s %s %f", "Tried to set unknown subtest %s's parameter %s to %f", subTest.c_str(), key.c_str(), value);
        return TP_ST_NOTFOUND;
    }

    m_subTestParameters[subTest][key]->Set((double)value);
    return TP_ST_OK;
}

/*****************************************************************************/
std::string TestParameters::GetString(std::string key)
{
    return m_globalParameters[key]->GetString();
}

/*****************************************************************************/
static int bool_string_to_bool(std::string str)
{
    const char *cStr = str.c_str();
    char firstChar   = *cStr;

    if (str.size() < 1)
        return 0; /* Empty string is false */

    if (firstChar == 't' || firstChar == 'T' || firstChar == '1' || firstChar == 'Y' || firstChar == 'y')
        return 1;
    else
        return 0; /* Everything else is false */
}

/*****************************************************************************/
int TestParameters::GetBoolFromString(std::string key)
{
    std::string str = m_globalParameters[key]->GetString();
    return bool_string_to_bool(str);
}

/*****************************************************************************/
double TestParameters::GetDouble(std::string key)
{
    return m_globalParameters[key]->GetDouble();
}

/*****************************************************************************/
std::string TestParameters::GetSubTestString(std::string subTest, std::string key)
{
    return m_subTestParameters[subTest][key]->GetString();
}

/*****************************************************************************/
bool TestParameters::HasKey(const std::string &key)
{
    return m_globalParameters.find(key) != m_globalParameters.end();
}

/*****************************************************************************/
double TestParameters::GetSubTestDouble(std::string subTest, std::string key)
{
    return m_subTestParameters[subTest][key]->GetDouble();
}

/*****************************************************************************/
int TestParameters::GetBoolFromSubTestString(std::string subTest, std::string key)
{
    std::string str = m_subTestParameters[subTest][key]->GetString();
    return bool_string_to_bool(str);
}

/*****************************************************************************/
int TestParameters::OverrideFrom(TestParameters *sourceTp)
{
    std::map<std::string, TestParameterValue *>::iterator destIt, sourceIt;
    TestParameterValue *deleteTpv = 0;

    if (sourceTp == nullptr)
    {
        DCGM_LOG_ERROR << "Cannot override my test parameters with a null object";
        return TP_ST_BADPARAM;
    }

    /* Global parameters */
    for (auto &[name, value] : sourceTp->m_globalParameters)
    {
        destIt = m_globalParameters.find(name);
        if (destIt == m_globalParameters.end())
        {
            m_globalParameters[name] = new TestParameterValue(*value);
            DCGM_LOG_DEBUG << "Added parameter " << name << "=" << value->GetString();
        }
        else
        {
            deleteTpv                = destIt->second;
            m_globalParameters[name] = new TestParameterValue(*value);
            DCGM_LOG_DEBUG << "Overrode parameter " << name << " with value " << value->GetString() << " (previously "
                           << deleteTpv->GetString() << ")";
            delete deleteTpv;
        }
    }

    /* Subtest parameters */
    std::map<std::string, std::map<std::string, TestParameterValue *>>::iterator outerDestIt, outerSourceIt;

    for (auto &[subtestName, paramMap] : sourceTp->m_subTestParameters)
    {
        for (auto &[name, value] : paramMap)
        {
            outerDestIt = m_subTestParameters.find(subtestName);
            if (outerDestIt != m_subTestParameters.end())
            {
                destIt = outerDestIt->second.find(name);
                if (destIt != outerDestIt->second.end())
                {
                    deleteTpv                              = m_subTestParameters[subtestName][name];
                    m_subTestParameters[subtestName][name] = new TestParameterValue(*value);
                    DCGM_LOG_DEBUG << "Overrode subtest " << subtestName << " parameter " << name << " with value "
                                   << value->GetString() << " (previously " << deleteTpv->GetString() << ")";
                    delete deleteTpv;
                }
                else
                {
                    m_subTestParameters[subtestName][name] = new TestParameterValue(*value);
                    DCGM_LOG_DEBUG << "Added subtest " << subtestName << " parameter " << name << "="
                                   << value->GetString();
                }
            }
            else
            {
                m_subTestParameters[subtestName][name] = new TestParameterValue(*value);
                DCGM_LOG_DEBUG << "Added subtest " << subtestName << " parameter " << name << "=" << value->GetString();
            }
        }
    }

    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::OverrideFromString(const std::string &name, const std::string &value)
{
    int rc = this->SetString(name, value, true);

    if (rc == TP_ST_NOTFOUND)
    {
        // If name has a '.' this could be a subtest parameter
        size_t dot = name.find('.');
        if (dot != std::string::npos)
        {
            std::string subtestName(name.substr(0, dot));
            std::string key(name.substr(dot + 1));

            rc = this->SetSubTestString(subtestName, key, value, true);
        }
        else
        {
            rc = AddString(name, value);
        }
    }

    return rc;
}

/*****************************************************************************/
int TestParameters::SetFromStruct(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *tpStruct)
{
    int rc = TP_ST_OK;
    for (unsigned int i = 0; i < numParameters; i++)
    {
        switch (tpStruct[i].type)
        {
            case DcgmPluginParamInt:
            case DcgmPluginParamFloat:
            case DcgmPluginParamString:
            case DcgmPluginParamBool:
            {
                int tmpRc = this->OverrideFromString(tpStruct[i].parameterName, tpStruct[i].parameterValue);
                if (rc == TP_ST_OK)
                {
                    rc = tmpRc;
                }
                break;
            }
            default:
            {
                rc = TP_ST_BADPARAM;
                DCGM_LOG_ERROR << "Cannot set parameter " << tpStruct[i].parameterName << "="
                               << tpStruct[i].parameterValue << " due to unknown type " << tpStruct[i].type;
            }
        }
    }

    return rc;
}

/*****************************************************************************/
int TestParameters::SetStructValue(TestParameterValue *tpv, dcgmDiagPluginTestParameter_t &param) const
{
    if (tpv == nullptr)
    {
        DCGM_LOG_DEBUG << "Null parameter sent to " << __func__ << ", cannot process it!";
        return TP_ST_BADPARAM;
    }

    int type = tpv->GetValueType();

    switch (type)
    {
        case TP_T_STRING:
        {
            param.type = DcgmPluginParamString;
            snprintf(param.parameterValue, sizeof(param.parameterValue), "%s", tpv->GetString().c_str());
            break;
        }
        case TP_T_DOUBLE:
        {
            param.type = DcgmPluginParamFloat;
            snprintf(param.parameterValue, sizeof(param.parameterValue), "%f", tpv->GetDouble());
            break;
        }

        default:
        {
            // Should never get here
            DCGM_LOG_DEBUG << "Cannot process parameter type '" << type << "' for sending to the plugin.";
            return TP_ST_BADPARAM;
        }
    }

    return TP_ST_OK;
}

/*****************************************************************************/
std::vector<dcgmDiagPluginTestParameter_t> TestParameters::GetParametersAsStruct() const
{
    std::vector<dcgmDiagPluginTestParameter_t> params;

    for (auto &[name, value] : m_globalParameters)
    {
        dcgmDiagPluginTestParameter_t param = {};
        snprintf(param.parameterName, sizeof(param.parameterName), "%s", name.c_str());
        if (SetStructValue(value, param) != TP_ST_OK)
        {
            DCGM_LOG_ERROR << "Unable to send parameter '" << name << "' to the plugin.";
            continue;
        }

        params.push_back(param);
    }

    for (auto &[name, subtestMap] : m_subTestParameters)
    {
        for (auto &[subtestName, value] : subtestMap)
        {
            dcgmDiagPluginTestParameter_t param = {};
            snprintf(param.parameterName, sizeof(param.parameterName), "%s.%s", name.c_str(), subtestName.c_str());

            if (SetStructValue(value, param) != TP_ST_OK)
            {
                DCGM_LOG_ERROR << "Unable to send parameter '" << name << "' to the plugin.";
                continue;
            }

            params.push_back(param);
        }
    }

    return params;
}
