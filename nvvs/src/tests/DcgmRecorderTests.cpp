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
#include "DcgmDiagUnitTestCommon.h"
#include <DcgmRecorder.h>
#include <NvvsCommon.h>
#include <catch2/catch.hpp>
#include <fstream>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

static long long g_timestamp = 0;

long long initializeDataCommon(dcgmTimeseriesInfo_t &data, unsigned short fieldId)
{
    g_timestamp++;
    data.fieldId   = fieldId;
    data.timestamp = g_timestamp;
    return g_timestamp;
}

long long initializeDataDouble(dcgmTimeseriesInfo_t &data, unsigned short fieldId, double val)
{
    data.isInt    = false;
    data.val.fp64 = val;
    return initializeDataCommon(data, fieldId);
}

long long initializeDataI64(dcgmTimeseriesInfo_t &data, unsigned short fieldId, uint64_t val)
{
    data.isInt   = true;
    data.val.i64 = val;
    return initializeDataCommon(data, fieldId);
}

const unsigned int gpuId                       = 0;
const std::string gpuIdStr                     = "0";
const std::string groupName                    = "group";
const std::vector<long long> i64Vals           = { 1, 2 };
const std::vector<double> doubleVals           = { 3 };
const std::vector<std::string> singleGroupGpus = { "gpu1", "gpu2" };
const std::vector<std::string> singleGroupKeys = { "key1", "key2" };
const std::vector<std::string> singleGroupVals = { "val1", "val2" };

void fillGpuStats(DcgmRecorder &dr)
{
    dr.SetGpuStat(gpuId, "i64", i64Vals[0]);
    dr.SetGpuStat(gpuId, "i64", i64Vals[1]);

    dr.SetGpuStat(gpuId, "dbl", doubleVals[0]);
}

void fillGroupStats(DcgmRecorder &dr)
{
    dr.SetGroupedStat(groupName, "i64", i64Vals[0]);
    dr.SetGroupedStat(groupName, "i64", i64Vals[1]);

    dr.SetGroupedStat(groupName, "dbl", doubleVals[0]);
}

void fillSingleGroupStats(DcgmRecorder &dr)
{
    dr.SetSingleGroupStat(singleGroupGpus[0], singleGroupKeys[0], singleGroupVals[0]);
    dr.SetSingleGroupStat(singleGroupGpus[0], singleGroupKeys[1], singleGroupVals[1]);
    dr.SetSingleGroupStat(singleGroupGpus[1], singleGroupKeys[1], singleGroupVals[1]);
}

SCENARIO("void DcgmRecorder::GetTagFromFieldId(unsigned short fieldId, std::string &tag)")
{
    DcgmRecorder dr;
    std::string str;
    dr.GetTagFromFieldId(0, str);
    CHECK(str == "0");
}

// List of private/protected functions tested as part of the block below:
// void DcgmRecorder::InsertCustomData(unsigned int gpuId, const std::string &name, dcgmTimeseriesInfo_t &data)
// std::vector<dcgmTimeseriesInfo_t> DcgmRecorder::GetCustomGpuStat(unsigned int gpuId, const std::string &name)
// void DcgmRecorder::SetGpuStat(unsigned int gpuId, const std::string &name, double value)
// void DcgmRecorder::SetGpuStat(unsigned int gpuId, const std::string &name, long long value)
SCENARIO("void DcgmRecorder::ClearCustomData()")
{
    DcgmRecorder dr;
    fillGpuStats(dr);
    auto i64Res = dr.GetCustomGpuStat(gpuId, "i64");
    auto dblRes = dr.GetCustomGpuStat(gpuId, "dbl");

    CHECK(i64Res.size() == 2);
    CHECK(i64Res[0].isInt);
    CHECK(i64Res[0].val.i64 == i64Vals[0]);
    CHECK(i64Res[1].isInt);
    CHECK(i64Res[1].val.i64 == i64Vals[1]);

    CHECK(dblRes.size() == 1);
    CHECK(dblRes[0].val.fp64 == doubleVals[0]);
    CHECK(!dblRes[0].isInt);

    dr.ClearCustomData();

    i64Res = dr.GetCustomGpuStat(gpuId, "i64");
    dblRes = dr.GetCustomGpuStat(gpuId, "dbl");

    CHECK(i64Res.size() == 0);
    CHECK(dblRes.size() == 0);
}

// List of private/protected functions tested as part of the block below:
// void DcgmRecorder::SetGroupedStat(const std::string &groupName, const std::string &name, double value)
// void DcgmRecorder::SetGroupedStat(const std::string &groupName, const std::string &name, long long value)
SCENARIO("void DcgmRecorder::InsertGroupedData(const std::string &groupName, const std::string &name, "
         "dcgmTimeseriesInfo_t &data)")
{
    DcgmRecorder dr;

    fillGroupStats(dr);

    auto i64Res = dr.GetGroupedStat(groupName, "i64");
    auto dblRes = dr.GetGroupedStat(groupName, "dbl");

    CHECK(i64Res.size() == 2);
    CHECK(i64Res[0].isInt);
    CHECK(i64Res[0].val.i64 == i64Vals[0]);
    CHECK(i64Res[1].isInt);
    CHECK(i64Res[1].val.i64 == i64Vals[1]);

    CHECK(dblRes.size() == 1);
    CHECK(dblRes[0].val.fp64 == doubleVals[0]);
    CHECK(!dblRes[0].isInt);
}

// List of private/protected functions tested as part of the block below:
// void DcgmRecorder::AddGpuDataToJson(Json::Value &jv)
// void DcgmRecorder::AddCustomData(Json::Value &jv)
// void DcgmRecorder::AddCustomTimeseriesVector(Json::Value &jv, std::vector<dcgmTimeseriesInfo_t> &vec)
// void DcgmRecorder::AddNonTimeseriesDataToJson(Json::Value &jv)
// std::string DcgmRecorder::GetWatchedFieldsAsJson(Json::Value &jv, long long ts)
// std::string DcgmRecorder::GetWatchedFieldsAsString(std::string &output, long long ts)
// void DcgmRecorder::AddGroupedDataToJson(Json::Value &jv)
// void DcgmRecorder::SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value)
SCENARIO("int DcgmRecorder::WriteToFile(const std::string &filename, int logFileType, long long testStart)")
{
    DcgmRecorder dr;
    Json::Value json;

    std::string jsonFileName = createTmpFile("json");
    std::string txtFileName  = createTmpFile("txt");

    fillGpuStats(dr);
    fillGroupStats(dr);
    fillSingleGroupStats(dr);

    dr.WriteToFile(txtFileName, NVVS_LOGFILE_TYPE_TEXT, 0);
    dr.WriteToFile(jsonFileName, NVVS_LOGFILE_TYPE_JSON, 0);

    std::ifstream jsonStream(jsonFileName);
    jsonStream >> json;

    CHECK(json[GPUS][0]["i64"][0]["value"].asInt() == i64Vals[0]);
    CHECK(json[GPUS][0]["i64"][1]["value"].asInt() == i64Vals[1]);
    CHECK(json[GPUS][0]["dbl"][0]["value"].asDouble() == doubleVals[0]);

    CHECK(json[groupName]["i64"][0]["value"].asInt() == i64Vals[0]);
    CHECK(json[groupName]["i64"][1]["value"].asInt() == i64Vals[1]);
    CHECK(json[groupName]["dbl"][0]["value"].asDouble() == doubleVals[0]);

    CHECK(json[singleGroupKeys[0]][singleGroupGpus[0]].asString() == singleGroupVals[0]);
    CHECK(json[singleGroupKeys[1]][singleGroupGpus[0]].asString() == singleGroupVals[1]);
    CHECK(json[singleGroupKeys[1]][singleGroupGpus[1]].asString() == singleGroupVals[1]);
}

SCENARIO("int DcgmRecorder::GetValueIndex(unsigned short fieldId)")
{
    DcgmRecorder dr;
    CHECK(dr.GetValueIndex(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL) == 2);
    CHECK(dr.GetValueIndex(256) == 0);
    CHECK(dr.GetValueIndex(DCGM_FI_DEV_THERMAL_VIOLATION) == 1);
}
