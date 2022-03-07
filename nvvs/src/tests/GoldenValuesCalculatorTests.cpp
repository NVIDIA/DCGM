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
#include <catch2/catch.hpp>

#include <GoldenValueCalculator.h>
#include <PluginStrings.h>

class WrapperGoldenValueCalculator : protected GoldenValueCalculator
{
public:
    std::map<std::string, paramToValues_t> &WrapperGetInputs();
    std::map<std::string, paramToGpuToValues_t> &WrapperGetInputsPerGpu();

    std::map<std::string, std::map<std::string, dcgmGpuToValue_t>> m_averageGpuValues;

    std::map<std::string, std::map<std::string, valueWithVariance_t>> WrapperGetCalculatedGoldenValues();

    void WrapperRecordGoldenValueInputs(const std::string &testname, const observedMetrics_t &metrics);

    void WrapperWriteConfigFile(const std::string &filename);

    std::string WrapperGetParameterName(const std::string &testname, const std::string &metricName) const;

    void WrapperAddToAllInputs(const std::string &testname, const std::string &paramName, double value);

    void WrapperAddToInputsPerGpu(const std::string &testname,
                                  const std::string &paramName,
                                  unsigned int gpuId,
                                  double value);

    valueWithVariance_t WrapperCalculateMeanAndVariance(std::vector<double> &data) const;

    double WrapperToleranceAdjustFactor(const std::string &paramName) const;

    void WrapperAdjustGoldenValue(const std::string &testname, const std::string &paramName);

    dcgmReturn_t WrapperIsVarianceAcceptable(const std::string &testname,
                                             const std::string &paramName,
                                             const valueWithVariance_t &vwv) const;

    void WrapperDumpObservedMetrics() const;

    void WrapperDumpObservedMetricsWithGpuIds(int64_t timestamp) const;

    void WrapperDumpObservedMetricsAll(int64_t timestamp) const;
};

std::map<std::string, paramToValues_t> &WrapperGoldenValueCalculator::WrapperGetInputs()
{
    return m_inputs;
}

std::map<std::string, paramToGpuToValues_t> &WrapperGoldenValueCalculator::WrapperGetInputsPerGpu()
{
    return m_inputsPerGpu;
}

std::map<std::string, std::map<std::string, valueWithVariance_t>> WrapperGoldenValueCalculator::
    WrapperGetCalculatedGoldenValues()
{
    return m_calculatedGoldenValues;
}

void setValues(std::map<unsigned int, double> &metric, double val0, double val1, double val2, double val3)
{
    metric[0] = val0;
    metric[1] = val1;
    metric[2] = val2;
    metric[3] = val3;
}

void WrapperGoldenValueCalculator::WrapperRecordGoldenValueInputs(const std::string &testname,
                                                                  const observedMetrics_t &metrics)
{
    RecordGoldenValueInputs(testname, metrics);
}

std::string WrapperGoldenValueCalculator::WrapperGetParameterName(const std::string &testname,
                                                                  const std::string &metricName) const

{
    return GetParameterName(testname, metricName);
}

void WrapperGoldenValueCalculator::WrapperAddToAllInputs(const std::string &testname,
                                                         const std::string &paramName,
                                                         double value)
{
    AddToAllInputs(testname, paramName, value);
}

void WrapperGoldenValueCalculator::WrapperAddToInputsPerGpu(const std::string &testname,
                                                            const std::string &paramName,
                                                            unsigned int gpuId,
                                                            double value)
{
    AddToInputsPerGpu(testname, paramName, gpuId, value);
}

valueWithVariance_t WrapperGoldenValueCalculator::WrapperCalculateMeanAndVariance(std::vector<double> &data) const
{
    return CalculateMeanAndVariance(data);
}

double WrapperGoldenValueCalculator::WrapperToleranceAdjustFactor(const std::string &paramName) const
{
    return ToleranceAdjustFactor(paramName);
}

void WrapperGoldenValueCalculator::WrapperAdjustGoldenValue(const std::string &testname, const std::string &paramName)
{
    return AdjustGoldenValue(testname, paramName);
}

dcgmReturn_t WrapperGoldenValueCalculator::WrapperIsVarianceAcceptable(const std::string &testname,
                                                                       const std::string &paramName,
                                                                       const valueWithVariance_t &vwv) const
{
    return IsVarianceAcceptable(testname, paramName, vwv);
}

SCENARIO("WrapperGetParameterName returns parameter name")
{
    WrapperGoldenValueCalculator gv;
    CHECK(gv.WrapperGetParameterName("pcie", "param-metric") == "param");
    CHECK(gv.WrapperGetParameterName("pcie", "param") == "param");
    CHECK(gv.WrapperGetParameterName("pcie", "param-") == "param");

    CHECK(gv.WrapperGetParameterName("not-PCIe", "param-metric") == "param-metric");
    CHECK(gv.WrapperGetParameterName("not-PCIe", "param-") == "param-");
}

SCENARIO("GoldenValueCalculator records inputs correctly for GPU and aggregate metrics")
{
    WrapperGoldenValueCalculator gv;
    observedMetrics_t metrics;
    std::map<unsigned int, double> singleMetric;

    setValues(singleMetric, 1, 2, 3, 4);
    metrics["1"] = singleMetric;

    setValues(singleMetric, 2, 4, 6, 8);
    metrics["2"] = singleMetric;

    gv.WrapperRecordGoldenValueInputs("test1", metrics);

    setValues(singleMetric, 3, 4, 5, 6);
    metrics["1"] = singleMetric;

    setValues(singleMetric, 4, 6, 8, 10);
    metrics["2"] = singleMetric;

    gv.WrapperRecordGoldenValueInputs("test2", metrics);

    auto inputs       = gv.WrapperGetInputs();
    auto inputsPerGpu = gv.WrapperGetInputsPerGpu();

    CHECK(inputs["test1"]["1"][0] == 1);
    CHECK(inputs["test1"]["1"][1] == 2);
    CHECK(inputs["test1"]["1"][2] == 3);
    CHECK(inputs["test1"]["1"][3] == 4);

    CHECK(inputs["test1"]["2"][0] == 2);
    CHECK(inputs["test1"]["2"][1] == 4);
    CHECK(inputs["test1"]["2"][2] == 6);
    CHECK(inputs["test1"]["2"][3] == 8);

    CHECK(inputs["test2"]["1"][0] == 3);
    CHECK(inputs["test2"]["1"][1] == 4);
    CHECK(inputs["test2"]["1"][2] == 5);
    CHECK(inputs["test2"]["1"][3] == 6);

    CHECK(inputs["test2"]["2"][0] == 4);
    CHECK(inputs["test2"]["2"][1] == 6);
    CHECK(inputs["test2"]["2"][2] == 8);
    CHECK(inputs["test2"]["2"][3] == 10);

    CHECK(inputsPerGpu["test1"]["1"][0][0] == 1);
    CHECK(inputsPerGpu["test1"]["1"][1][0] == 2);
    CHECK(inputsPerGpu["test1"]["1"][2][0] == 3);
    CHECK(inputsPerGpu["test1"]["1"][3][0] == 4);

    CHECK(inputsPerGpu["test1"]["2"][0][0] == 2);
    CHECK(inputsPerGpu["test1"]["2"][1][0] == 4);
    CHECK(inputsPerGpu["test1"]["2"][2][0] == 6);
    CHECK(inputsPerGpu["test1"]["2"][3][0] == 8);

    CHECK(inputsPerGpu["test2"]["1"][0][0] == 3);
    CHECK(inputsPerGpu["test2"]["1"][1][0] == 4);
    CHECK(inputsPerGpu["test2"]["1"][2][0] == 5);
    CHECK(inputsPerGpu["test2"]["1"][3][0] == 6);

    CHECK(inputsPerGpu["test2"]["2"][0][0] == 4);
    CHECK(inputsPerGpu["test2"]["2"][1][0] == 6);
    CHECK(inputsPerGpu["test2"]["2"][2][0] == 8);
    CHECK(inputsPerGpu["test2"]["2"][3][0] == 10);
}

SCENARIO("CalculateMeanAndVariance calculates the correct mean and variance")
{
    WrapperGoldenValueCalculator gv;

    std::vector<double> data = { 1 };
    valueWithVariance_t res  = gv.WrapperCalculateMeanAndVariance(data);
    CHECK(res.value == 1);
    CHECK(res.variance == 0);

    data = { 0, 2 };
    res  = gv.WrapperCalculateMeanAndVariance(data);
    CHECK(res.value == 1);
    CHECK(res.variance == 2);

    data = { 1, 5 };
    res  = gv.WrapperCalculateMeanAndVariance(data);
    CHECK(res.value == 3);
    CHECK(res.variance == 8);

    data = { 1, 2, 3, 4 };
    res  = gv.WrapperCalculateMeanAndVariance(data);
    CHECK(res.value == 2.5);
    CHECK((res.variance - 1.666) < 0.1);
}

SCENARIO("ToleranceAdjustFactor returns the correct adjustment factor")
{
    WrapperGoldenValueCalculator gv;

    nvvsCommon.trainingTolerancePcnt = 1;
    CHECK(gv.WrapperToleranceAdjustFactor(PCIE_STR_MAX_LATENCY) == 1);
    CHECK(gv.WrapperToleranceAdjustFactor(SMSTRESS_STR_TARGET_PERF) == -1);

    nvvsCommon.trainingTolerancePcnt = 2;
    CHECK(gv.WrapperToleranceAdjustFactor(PCIE_STR_MAX_MEMORY_CLOCK) == 2);
    CHECK(gv.WrapperToleranceAdjustFactor(PCIE_STR_MIN_BANDWIDTH) == -2);

    CHECK(gv.WrapperToleranceAdjustFactor(PCIE_STR_MIN_PCI_GEN) == 0);
}

SCENARIO("IsVarianceAcceptable returns OK if acceptable, DIAG_VARIANCE otherwise")
{
    WrapperGoldenValueCalculator gv;
    nvvsCommon.trainingVariancePcnt = 1;

    valueWithVariance_t acceptable = { 1, 1 };
    CHECK(DCGM_ST_OK == gv.WrapperIsVarianceAcceptable("test", "param", acceptable));

    valueWithVariance_t not_acceptable = { 1, 1.011 };
    CHECK(DCGM_ST_DIAG_VARIANCE == gv.WrapperIsVarianceAcceptable("test", "param", not_acceptable));

    // not_acceptable becomes acceptable if we modify the nvvsCommon parameter
    nvvsCommon.trainingVariancePcnt = 1.01;
    CHECK(DCGM_ST_OK == gv.WrapperIsVarianceAcceptable("test", "param", not_acceptable));
}

SCENARIO("GoldenValueCalculator calculates and adjusts golden values accurately")
{}
