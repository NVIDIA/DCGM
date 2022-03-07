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
#include <fstream>

#include "GoldenValueCalculator.h"
#include "PluginStrings.h"
#include "dcgm_structs.h"
#include "timelib.h"
#include "yaml-cpp/yaml.h"


/*************************************************************************/
void GoldenValueCalculator::RecordGoldenValueInputs(const std::string &testname, const observedMetrics_t &metrics)
{
    for (observedMetrics_t::const_iterator outer = metrics.begin(); outer != metrics.end(); ++outer)
    {
        for (std::map<unsigned int, double>::const_iterator inner = outer->second.begin(); inner != outer->second.end();
             ++inner)
        {
            std::string paramName = GetParameterName(testname, outer->first);
            AddToAllInputs(testname, paramName, inner->second);
            AddToInputsPerGpu(testname, paramName, inner->first, inner->second);
        }
    }
}

/*************************************************************************/
valueWithVariance_t GoldenValueCalculator::CalculateMeanAndVariance(std::vector<double> &data) const
{
    double total = 0.0;
    valueWithVariance_t output;

    for (size_t i = 0; i < data.size(); i++)
    {
        total += data[i];
    }

    double avg   = total / static_cast<double>(data.size());
    output.value = avg;

    if (data.size() > 1)
    {
        double scratchVariance = 0.0;

        for (size_t i = 0; i < data.size(); i++)
        {
            double diff = data[i] - avg;
            scratchVariance += diff * diff;
        }

        output.variance = scratchVariance / static_cast<double>(data.size() - 1);
    }
    else
    {
        output.variance = 0.0;
    }

    return output;
}

/*************************************************************************/
double GoldenValueCalculator::ToleranceAdjustFactor(const std::string &paramName) const
{
    if ((paramName == PCIE_STR_MAX_LATENCY) || (paramName == PCIE_STR_MAX_MEMORY_CLOCK)
        || (paramName == PCIE_STR_MAX_GRAPHICS_CLOCK) || (paramName == TP_STR_TEMPERATURE_MAX)
        || (paramName == TS_STR_TEMPERATURE_MAX) || (paramName == SMSTRESS_STR_TEMPERATURE_MAX)
        || (paramName == DIAGNOSTIC_STR_TEMPERATURE_MAX))
    {
        // Max values are increased to be slightly more permissive
        return nvvsCommon.trainingTolerancePcnt;
    }
    else if (paramName == PCIE_STR_MIN_PCI_GEN)
    {
        return 0;
    }
    else
    {
        /*
         * Minimum values are decreased to be slightly more permissive
         * Parameters that should return -1:
         * PCIE_STR_MIN_BANDWIDTH
         * TS_STR_TARGET_PERF
         * SMSTRESS_STR_TARGET_PERF
         * TP_STR_TARGET_POWER
         * MEMBW_STR_MINIMUM_BANDWIDTH
         */
        return -1 * nvvsCommon.trainingTolerancePcnt;
    }
}

/*************************************************************************/
void GoldenValueCalculator::AdjustGoldenValue(const std::string &testname, const std::string &paramName)
{
    double fraction = m_calculatedGoldenValues[testname][paramName].value * ToleranceAdjustFactor(paramName);
    m_calculatedGoldenValues[testname][paramName].value += fraction;
}

void GoldenValueCalculator::DumpObservedMetricsWithGpuIds(int64_t timestamp) const
{
    std::ofstream fileWithGpuIds;
    std::stringstream dataWithGpuIdsFilename;
    dataWithGpuIdsFilename << "/tmp/dcgmgd_withgpuids" << timestamp << ".txt";

    fileWithGpuIds.open(dataWithGpuIdsFilename.str().c_str(), std::ofstream::out);

    for (std::map<std::string, paramToGpuToValues_t>::const_iterator testIt = m_inputsPerGpu.begin();
         testIt != m_inputsPerGpu.end();
         ++testIt)
    {
        for (paramToGpuToValues_t::const_iterator paramsIt = testIt->second.begin(); paramsIt != testIt->second.end();
             ++paramsIt)
        {
            for (std::map<unsigned int, std::vector<double>>::const_iterator gpusIt = paramsIt->second.begin();
                 gpusIt != paramsIt->second.end();
                 ++gpusIt)
            {
                fileWithGpuIds << testIt->first << "." << paramsIt->first << "." << gpusIt->first;
                for (size_t i = 0; i < gpusIt->second.size(); i++)
                {
                    fileWithGpuIds << "," << gpusIt->second[i];
                }
                fileWithGpuIds << "\n";
            }
        }
    }

    fileWithGpuIds.close();
}

/*************************************************************************/
void GoldenValueCalculator::DumpObservedMetricsAll(int64_t timestamp) const
{
    std::stringstream dataFilename;
    dataFilename << "/tmp/dcgmgd" << timestamp << ".txt";

    std::ofstream file;
    file.open(dataFilename.str().c_str(), std::ofstream::out);

    for (std::map<std::string, paramToValues_t>::const_iterator testIt = m_inputs.begin(); testIt != m_inputs.end();
         ++testIt)
    {
        for (paramToValues_t::const_iterator paramsIt = testIt->second.begin(); paramsIt != testIt->second.end();
             ++paramsIt)
        {
            file << testIt->first << "." << paramsIt->first;
            for (size_t i = 0; i < paramsIt->second.size(); i++)
            {
                file << "," << paramsIt->second[i];
            }
            file << "\n";
        }
    }

    file.close();
}

/*************************************************************************/
void GoldenValueCalculator::DumpObservedMetrics() const
{
    int64_t timestamp = timelib_usecSince1970();
    DumpObservedMetricsAll(timestamp);
    DumpObservedMetricsWithGpuIds(timestamp);
}

/*************************************************************************/
dcgmReturn_t GoldenValueCalculator::IsVarianceAcceptable(const std::string &testname,
                                                         const std::string &paramName,
                                                         const valueWithVariance_t &vwv) const
{
    dcgmReturn_t ret = DCGM_ST_OK;
    // Make acceptable variance a percentage of the value since the variables we're measuring have such
    // different scales
    // The formula is sqrt(variance) > value * tolerance, but I've converted it to avoid using the sqrt
    double tolerance  = nvvsCommon.trainingVariancePcnt;
    double acceptable = tolerance * tolerance * vwv.value * vwv.value;
    if (vwv.variance > acceptable)
    {
        ret = DCGM_ST_DIAG_VARIANCE;
        PRINT_ERROR("%s %s %.02f %.02f",
                    "Golden value %s.%s has variance of %.02f, outside the acceptable "
                    "standard of %.02f",
                    testname.c_str(),
                    paramName.c_str(),
                    vwv.variance,
                    acceptable);
    }

    return ret;
}

/*************************************************************************/
dcgmReturn_t GoldenValueCalculator::CalculateAndWriteGoldenValues(const std::string &filename)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    m_calculatedGoldenValues.clear();
    m_averageGpuValues.clear();

    for (std::map<std::string, paramToValues_t>::iterator testIt = m_inputs.begin(); testIt != m_inputs.end(); ++testIt)
    {
        for (paramToValues_t::iterator paramIt = testIt->second.begin(); paramIt != testIt->second.end(); ++paramIt)
        {
            m_calculatedGoldenValues[testIt->first][paramIt->first] = CalculateMeanAndVariance(paramIt->second);

            ret = IsVarianceAcceptable(
                testIt->first, paramIt->first, m_calculatedGoldenValues[testIt->first][paramIt->first]);

            AdjustGoldenValue(testIt->first, paramIt->first);
        }
    }

    for (std::map<std::string, paramToGpuToValues_t>::iterator testIt = m_inputsPerGpu.begin();
         testIt != m_inputsPerGpu.end();
         ++testIt)
    {
        for (paramToGpuToValues_t::iterator paramsIt = testIt->second.begin(); paramsIt != testIt->second.end();
             ++paramsIt)
        {
            for (std::map<unsigned int, std::vector<double>>::iterator gpusIt = paramsIt->second.begin();
                 gpusIt != paramsIt->second.end();
                 ++gpusIt)
            {
                dcgmGpuToValue_t tmp;
                tmp.gpuId                                          = gpusIt->first;
                tmp.calculatedValue                                = CalculateMeanAndVariance(gpusIt->second);
                m_averageGpuValues[testIt->first][paramsIt->first] = tmp;
            }
        }
    }

    DumpObservedMetrics();

    if (ret == DCGM_ST_OK || nvvsCommon.forceTraining)
    {
        WriteConfigFile(filename);
    }

    return ret;
}

/*************************************************************************/
void GoldenValueCalculator::WriteConfigFile(const std::string &filename)
{
    // Build the YAML
    YAML::Emitter yamlOut;

    yamlOut << YAML::BeginMap << YAML::Key << "custom";
    yamlOut << YAML::Value << YAML::BeginSeq << YAML::BeginMap << YAML::Key << "custom" << YAML::Value;
    yamlOut << YAML::BeginMap; // begin the map of tests

    std::map<std::string, std::map<std::string, double>> subtestParamValue;

    for (std::map<std::string, std::map<std::string, valueWithVariance_t>>::iterator testIt
         = m_calculatedGoldenValues.begin();
         testIt != m_calculatedGoldenValues.end();
         ++testIt)
    {
        std::string testname = testIt->first;

        // Each test name is a key in the testmap, and each value is a map of parameters
        yamlOut << YAML::Key << testname << YAML::Value << YAML::BeginMap;
        bool pcie = testname == PCIE_PLUGIN_NAME;

        for (std::map<std::string, valueWithVariance_t>::iterator paramIt = testIt->second.begin();
             paramIt != testIt->second.end();
             ++paramIt)
        {
            size_t pos;
            if (pcie && (pos = paramIt->first.find('.')) != std::string::npos)
            {
                // Save the subtests names to a separate map because they all have to written at once
                std::string subtest                   = paramIt->first.substr(0, pos);
                std::string paramName                 = paramIt->first.substr(pos + 1);
                subtestParamValue[subtest][paramName] = paramIt->second.value;
            }
            else
            {
                // Each test parameter is a key / value pair
                yamlOut << YAML::Key << paramIt->first << YAML::Value << paramIt->second.value;
            }
        }

        if (pcie && subtestParamValue.size())
        {
            // If a test has subtests, then those subtests are a map
            yamlOut << YAML::Key << "subtests" << YAML::Value << YAML::BeginMap;

            for (std::map<std::string, std::map<std::string, double>>::iterator subtestIt = subtestParamValue.begin();
                 subtestIt != subtestParamValue.end();
                 ++subtestIt)
            {
                // each subtest name is a key and the value is a map of parameters to the subtest
                yamlOut << YAML::Key << subtestIt->first << YAML::Value << YAML::BeginMap;

                for (std::map<std::string, double>::iterator paramIt = subtestIt->second.begin();
                     paramIt != subtestIt->second.end();
                     ++paramIt)
                {
                    // Each subtest parameter is a key / value pair
                    yamlOut << YAML::Key << paramIt->first << YAML::Value << paramIt->second;
                }

                yamlOut << YAML::EndMap;
            }

            yamlOut << YAML::EndMap;
        }

        yamlOut << YAML::EndMap;
    }

    yamlOut << YAML::EndMap; // End the map of tests
    // End the inner custom map, the sequence, and the outer custom map
    yamlOut << YAML::EndMap << YAML::EndSeq << YAML::EndMap;

    std::ofstream file;
    file.open(filename.c_str(), std::ofstream::out);
    file << yamlOut.c_str();
    file.close();
}

/*************************************************************************/
std::string GoldenValueCalculator::GetParameterName(const std::string &testname, const std::string &metricName) const
{
    std::string parm;

    if (testname == PCIE_PLUGIN_NAME)
    {
        size_t pos = metricName.find('-');
        if (pos == std::string::npos)
        {
            parm = metricName;
        }
        else
        {
            parm = metricName.substr(0, pos);
        }
    }
    else
    {
        parm = metricName;
    }

    return parm;
}
