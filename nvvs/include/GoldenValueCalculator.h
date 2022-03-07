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
#ifndef GOLDEN_VALUE_CALCULATOR_H
#define GOLDEN_VALUE_CALCULATOR_H

#include <map>
#include <string>
#include <vector>

#include "Plugin.h"
#include "dcgm_structs.h"

typedef struct
{
    double value;    //!< The golden value (mean value) calculated
    double variance; //!< The variance calculated for this mean in the dataset
} valueWithVariance_t;

typedef struct
{
    unsigned int gpuId;                  //!< The GPU id that these values were calculated for
    valueWithVariance_t calculatedValue; //!< The value and its variance
} dcgmGpuToValue_t;

typedef std::map<std::string, std::map<unsigned int, std::vector<double>>> paramToGpuToValues_t;
typedef std::map<std::string, std::vector<double>> paramToValues_t;

class GoldenValueCalculator
{
public:
    /*************************************************************************/
    GoldenValueCalculator()
        : m_inputs()
        , m_inputsPerGpu()
        , m_averageGpuValues()
        , m_calculatedGoldenValues()
    {}

    /*************************************************************************/
    /*
     * Store the values observed from the test in m_inputs and m_inputsPerGpu
     */
    void RecordGoldenValueInputs(const std::string &testName, const observedMetrics_t &metrics);

    /*************************************************************************/
    /*
     * Calculate and write the golden values and store them in m_calculatedGoldenValues using the inputs
     * recorded to this point
     *
     * Return DCGM_ST_VARIANCE if the values do not converge
     *        DCGM_ST_* on other errors
     *        DCGM_ST_OK on success
     */
    dcgmReturn_t CalculateAndWriteGoldenValues(const std::string &filename);

protected:
    // "testname.parameter" -> array of values
    std::map<std::string, paramToValues_t> m_inputs;
    // "testname.parameter" -> gpuId -> array of values
    std::map<std::string, paramToGpuToValues_t> m_inputsPerGpu;
    // "testname.parameter" -> gpuId -> average value
    std::map<std::string, std::map<std::string, dcgmGpuToValue_t>> m_averageGpuValues;
    // "testname.parameter" -> average value
    std::map<std::string, std::map<std::string, valueWithVariance_t>> m_calculatedGoldenValues;

    // Methods
    /*************************************************************************/
    /*
     * Writes the configure file to the specified filename.
     */
    void WriteConfigFile(const std::string &filename);

    /*************************************************************************/
    /*
     * Returns the correct parameter name from the supplied metric name and test name
     * In many cases, it returns a string equal to metricName
     */
    std::string GetParameterName(const std::string &testname, const std::string &metricName) const;

    /*************************************************************************/
    /*
     * Records the value for the given parameter in our map of inputs
     */
    inline void AddToAllInputs(const std::string &testname, const std::string &paramName, double value)
    {
        m_inputs[testname][paramName].push_back(value);
    }

    /*************************************************************************/
    /*
     * Records the value for the given parameter in our map of inputs which records gpu Id
     */
    inline void AddToInputsPerGpu(const std::string &testname,
                                  const std::string &paramName,
                                  unsigned int gpuId,
                                  double value)
    {
        m_inputsPerGpu[testname][paramName][gpuId].push_back(value);
    }

    /*************************************************************************/
    /*
     * Returns a calculated mean and variance for the data in the vector
     */
    valueWithVariance_t CalculateMeanAndVariance(std::vector<double> &data) const;

    /*************************************************************************/
    /*
     * Determines if this parameter should be adjusted up, down, or not at all
     * Returns a positive value to adjust up, 0 for not at all, and a negative value for down
     */
    double ToleranceAdjustFactor(const std::string &paramName) const;

    /*************************************************************************/
    /*
     * Adjusts the calculated golden value for the testname and parameter to allow some
     * tolerance in the test runs.
     */
    void AdjustGoldenValue(const std::string &testname, const std::string &paramName);

    /*************************************************************************/
    /*
     * Adjusts the calculated golden value for the testname and parameter to allow some
     * tolerance in the test runs.
     */
    dcgmReturn_t IsVarianceAcceptable(const std::string &testname,
                                      const std::string &paramName,
                                      const valueWithVariance_t &vwv) const;

    /*************************************************************************/
    /*
     * Dumps all of the metrics we've recorded into a CSV file for inspection
     */
    void DumpObservedMetrics() const;

    /*************************************************************************/
    /*
     * Dumps all of the metrics we've recorded, but dumps the information per GPU
     */
    void DumpObservedMetricsWithGpuIds(int64_t timestamp) const;

    /*************************************************************************/
    /*
     * Dumps all of the metrics we've recorded with no information about which GPU it came from
     */
    void DumpObservedMetricsAll(int64_t timestamp) const;
};

#endif
