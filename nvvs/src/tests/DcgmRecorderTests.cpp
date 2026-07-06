/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <Defer.hpp>
#include <NvvsCommon.h>
#include <catch2/catch_all.hpp>
#include <fstream>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

static long long g_timestamp                                = 0;
static dcgmReturn_t g_watchFieldsRet                        = DCGM_ST_OK;
static dcgmReturn_t g_getValuesSinceRet                     = DCGM_ST_OK;
static dcgmFieldGrp_t g_fieldGroupId                        = 0;
static dcgmGpuGrp_t g_groupId                               = 0;
static unsigned int g_injectedGpuId                         = 0;
static dcgmFieldValue_v1 g_injectedFieldVal                 = {};
static bool g_hasInjectedValue                              = false;
static dcgmReturn_t g_getFieldSummaryRet                    = DCGM_ST_OK;
static dcgmSummaryResponse_t g_injectedFieldSummaryResponse = {};

class WrapperDcgmRecorder : protected DcgmRecorder
{
public:
    WrapperDcgmRecorder(dcgmHandle_t handle)
        : DcgmRecorder(handle)
    {}
    void WrapperFormatFieldViolationError(DcgmError &d,
                                          unsigned short fieldId,
                                          unsigned int gpuId,
                                          timelib64_t start,
                                          int64_t intValue,
                                          double dblValue,
                                          const std::string &fieldName);
    using DcgmRecorder::AddWatches;
};

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
    CHECK(i64Res[0].val.i64 == static_cast<uint64_t>(i64Vals[0]));
    CHECK(i64Res[1].isInt);
    CHECK(i64Res[1].val.i64 == static_cast<uint64_t>(i64Vals[1]));

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
    CHECK(i64Res[0].val.i64 == static_cast<uint64_t>(i64Vals[0]));
    CHECK(i64Res[1].isInt);
    CHECK(i64Res[1].val.i64 == static_cast<uint64_t>(i64Vals[1]));

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

dcgmReturn_t dcgmGroupCreate(dcgmHandle_t /* handle */,
                             dcgmGroupType_t /* type */,
                             const char * /* groupName */,
                             dcgmGpuGrp_t *groupId)
{
    if (groupId != nullptr)
    {
        *groupId = g_groupId;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t /* handle */,
                                  int /* numFieldIds */,
                                  const unsigned short * /* fieldIdArray */,
                                  const char * /* name */,
                                  dcgmFieldGrp_t *m_fieldGroupId)
{
    if (m_fieldGroupId != nullptr)
    {
        *m_fieldGroupId = g_fieldGroupId;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t dcgmGroupAddDevice(dcgmHandle_t /* pDcgmHandle */, dcgmGpuGrp_t /* groupId */, unsigned int /* gpuId */)
{
    return DCGM_ST_OK;
}

dcgmReturn_t dcgmWatchFields(dcgmHandle_t /* pDcgmHandle */,
                             dcgmGpuGrp_t /* groupId */,
                             dcgmFieldGrp_t /* fieldGroupId */,
                             long long /* updateFreq */,
                             double /* maxKeepAge */,
                             int /* maxKeepSamples */)
{
    return g_watchFieldsRet;
}

dcgmReturn_t dcgmGetValuesSince(dcgmHandle_t /* pDcgmHandle */,
                                dcgmGpuGrp_t /* groupId */,
                                dcgmFieldGrp_t /* fieldGroupId */,
                                long long /* sinceTimestamp */,
                                long long *nextSinceTimestamp,
                                dcgmFieldValueEnumeration_f /* enumCB */,
                                void * /* userData */)
{
    if (nextSinceTimestamp != nullptr)
    {
        *nextSinceTimestamp = g_timestamp + 1;
    }
    return g_getValuesSinceRet;
}

dcgmReturn_t dcgmGetValuesSince_v2(dcgmHandle_t /* pDcgmHandle */,
                                   dcgmGpuGrp_t /* groupId */,
                                   dcgmFieldGrp_t /* fieldGroupId */,
                                   long long /* sinceTimestamp */,
                                   long long * /* nextSinceTimestamp */,
                                   dcgmFieldValueEntityEnumeration_f enumCB,
                                   void *userData)
{
    if (g_getValuesSinceRet == DCGM_ST_OK && enumCB != nullptr && g_hasInjectedValue)
    {
        enumCB(DCGM_FE_GPU, g_injectedGpuId, &g_injectedFieldVal, 1, userData);
    }

    return g_getValuesSinceRet;
}

// Needed because DcgmRecorder::Shutdown destroys the group when AddWatches has initialized m_dcgmGroup
dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t /* pDcgmHandle */, dcgmGpuGrp_t /* groupId */)
{
    return DCGM_ST_OK;
}

// Needed because DcgmRecorder::Shutdown destroys the field group when AddWatches has set m_fieldGroupId
dcgmReturn_t dcgmFieldGroupDestroy(dcgmHandle_t /* pDcgmHandle */, dcgmFieldGrp_t /* fieldGroupId */)
{
    return DCGM_ST_OK;
}

// Needed because DcgmGroup::FieldGroupDestroy calls dcgmUnwatchFields during cleanup
dcgmReturn_t dcgmUnwatchFields(dcgmHandle_t /* pDcgmHandle */,
                               dcgmGpuGrp_t /* groupId */,
                               dcgmFieldGrp_t /* fieldGroupId */)
{
    return DCGM_ST_OK;
}

// Needed because DcgmRecorder::GetFieldSummary calls dcgmGetFieldSummary
dcgmReturn_t dcgmGetFieldSummary(dcgmHandle_t /* handle */, dcgmFieldSummaryRequest_t *request)
{
    if (request != nullptr)
    {
        request->response = g_injectedFieldSummaryResponse;
    }
    return g_getFieldSummaryRet;
}

SCENARIO("AddWatches")
{
    DcgmRecorder dr((dcgmHandle_t)1);
    std::vector<unsigned short> fieldIds;
    std::vector<unsigned int> gpuIds;

    // Fail with empty field ids
    CHECK(dr.AddWatches(fieldIds, gpuIds, false, "field_group1", "group1", 300.0) == DCGM_ST_BADPARAM);
    fieldIds.push_back(DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    // Fail with empty gpu IDs
    CHECK(dr.AddWatches(fieldIds, gpuIds, false, "field_group1", "group1", 300.0) == DCGM_ST_BADPARAM);
    gpuIds.push_back(0);

    g_groupId        = 1;
    g_fieldGroupId   = 1;
    g_watchFieldsRet = DCGM_ST_GPU_IS_LOST;
    CHECK(dr.AddWatches(fieldIds, gpuIds, false, "field_group1", "group1", 300.0) == DCGM_ST_GPU_IS_LOST);
}

SCENARIO("DcgmRecorder::WriteToFile with field watch errors")
{
    dcgmHandle_t handle = (dcgmHandle_t)1;
    DcgmRecorder dr(handle);
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS };
    std::vector<unsigned int> gpuIds     = { 0 };

    std::string jsonFileName = createTmpFile("json");
    DcgmNs::Defer cleanup([&jsonFileName]() { CHECK(unlink(jsonFileName.c_str()) == 0); });

    g_groupId        = 1;
    g_fieldGroupId   = 1;
    g_watchFieldsRet = DCGM_ST_OK;

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);

    SECTION("fields not watched")
    {
        g_getValuesSinceRet = DCGM_ST_NOT_WATCHED;
    }

    SECTION("no data available")
    {
        g_getValuesSinceRet = DCGM_ST_NO_DATA;
    }

    SECTION("GPU lost")
    {
        g_getValuesSinceRet = DCGM_ST_GPU_IS_LOST;
    }

    dr.WriteToFile(jsonFileName, NVVS_LOGFILE_TYPE_JSON, 0);

    std::ifstream fileStream(jsonFileName);
    std::string content((std::istreambuf_iterator<char>(fileStream)), std::istreambuf_iterator<char>());

    CHECK_FALSE(content.empty());

    Json::Value json;
    Json::CharReaderBuilder builder;
    std::istringstream stream(content);
    std::string errs;
    Json::parseFromStream(builder, stream, &json, &errs);
    CHECK(json.isNull());
}

void WrapperDcgmRecorder::WrapperFormatFieldViolationError(DcgmError &d,
                                                           unsigned short fieldId,
                                                           unsigned int gpuId,
                                                           timelib64_t startTime,
                                                           int64_t intValue,
                                                           double dblValue,
                                                           const std::string &fieldName)
{
    FormatFieldViolationError(d, fieldId, gpuId, startTime, intValue, dblValue, fieldName);
}

SCENARIO("FormatFieldViolationError")
{
    WrapperDcgmRecorder dr((dcgmHandle_t)1);
    unsigned int gpuId = 2;
    std::string fieldName("bob");

    for (unsigned int fieldId = DCGM_FR_UNKNOWN; fieldId < DCGM_FR_ERROR_SENTINEL; fieldId++)
    {
        DcgmError d { gpuId };
        dr.WrapperFormatFieldViolationError(d, fieldId, gpuId, 14, 6, DCGM_FP64_BLANK, fieldName);
        switch (fieldId)
        {
            case DCGM_FI_DEV_THERMAL_VIOLATION:
            {
                CHECK(d.GetCode() == DCGM_FR_THERMAL_VIOLATIONS);
                break;
            }
            case DCGM_FI_DEV_XID_ERROR:
            {
                CHECK(d.GetCode() == DCGM_FR_XID_ERROR);
                break;
            }
            default:
            {
                CHECK(d.GetCode() == DCGM_FR_FIELD_VIOLATION);
                // FormatFieldViolation trusts the caller on whether or not the field is a double. When the int
                // value is blank, it formats a double.
                dr.WrapperFormatFieldViolationError(d, fieldId, gpuId, 14, DCGM_INT64_BLANK, 10.0, fieldName);
                CHECK(d.GetCode() == DCGM_FR_FIELD_VIOLATION_DBL);
                break;
            }
        }
    }
}

SCENARIO("DcgmRecorder: negative test for DCGM_FR_CLOCKS_EVENT_VIOLATION")
{
    DcgmRecorder dr((dcgmHandle_t)42);
    unsigned int targetGpuId = 1;
    timelib64_t startTime    = 1000000;
    std::vector<DcgmError> fatalErrors;
    std::vector<DcgmError> ignoredErrors;

    // Set up field watches so CheckForClocksEvent can query field values
    g_groupId                            = 1;
    g_fieldGroupId                       = 1;
    g_watchFieldsRet                     = DCGM_ST_OK;
    g_getValuesSinceRet                  = DCGM_ST_OK;
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_CLOCKS_EVENT_REASONS };
    std::vector<unsigned int> gpuIds     = { targetGpuId };

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);

    // Inject a HW slowdown clocks event for the target GPU via the dcgmGetValuesSince_v2 stub
    memset(&g_injectedFieldVal, 0, sizeof(g_injectedFieldVal));
    g_injectedFieldVal.version   = dcgmFieldValue_version1;
    g_injectedFieldVal.fieldId   = DCGM_FI_DEV_CLOCKS_EVENT_REASONS;
    g_injectedFieldVal.fieldType = DCGM_FT_INT64;
    g_injectedFieldVal.status    = DCGM_ST_OK;
    g_injectedFieldVal.ts        = startTime + 5000000;
    g_injectedFieldVal.value.i64 = DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN;
    g_injectedGpuId              = targetGpuId;
    g_hasInjectedValue           = true;

    DcgmNs::Defer cleanup([] {
        g_fieldGroupId      = 0;
        g_groupId           = 0;
        g_watchFieldsRet    = DCGM_ST_OK;
        g_getValuesSinceRet = DCGM_ST_OK;
        g_hasInjectedValue  = false;
        g_injectedGpuId     = 0;
        memset(&g_injectedFieldVal, 0, sizeof(g_injectedFieldVal));
    });

    // CheckForClocksEvent should detect the injected event and return DR_VIOLATION
    int ret = dr.CheckForClocksEvent(targetGpuId, startTime, fatalErrors, ignoredErrors);
    CHECK(ret == DR_VIOLATION);
    REQUIRE(fatalErrors.size() == 1);

    // The error must carry the correct failure reason and be attributed to the affected GPU
    CHECK(fatalErrors[0].GetCode() == DCGM_FR_CLOCKS_EVENT_VIOLATION);
    CHECK(fatalErrors[0].GetEntity().entityId == targetGpuId);
    CHECK(fatalErrors[0].GetSeverity() == DCGM_ERROR_MONITOR);
    CHECK(fatalErrors[0].GetMessage().find("Clocks event for GPU") != std::string::npos);
}

SCENARIO("DcgmRecorder: negative test for DCGM_FR_THERMAL_VIOLATIONS_TS")
{
    WrapperDcgmRecorder dr((dcgmHandle_t)42);
    constexpr unsigned int targetGpuId = 1;
    constexpr timelib64_t startTime    = 1000000;

    // Set up field watches so FormatFieldViolationError can query field values
    g_groupId                            = 1;
    g_fieldGroupId                       = 1;
    g_watchFieldsRet                     = DCGM_ST_OK;
    g_getValuesSinceRet                  = DCGM_ST_OK;
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_CLOCKS_EVENT_REASONS };
    std::vector<unsigned int> gpuIds     = { targetGpuId };

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);

    // Inject a HW slowdown clocks event for the target GPU via the dcgmGetValuesSince_v2 stub
    memset(&g_injectedFieldVal, 0, sizeof(g_injectedFieldVal));
    g_injectedFieldVal.version   = dcgmFieldValue_version1;
    g_injectedFieldVal.fieldId   = DCGM_FI_DEV_CLOCKS_EVENT_REASONS;
    g_injectedFieldVal.fieldType = DCGM_FT_INT64;
    g_injectedFieldVal.status    = DCGM_ST_OK;
    g_injectedFieldVal.ts        = startTime + 5000000;
    g_injectedFieldVal.value.i64 = DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN;
    g_injectedGpuId              = targetGpuId;
    g_hasInjectedValue           = true;

    DcgmNs::Defer cleanup([] {
        g_fieldGroupId      = 0;
        g_groupId           = 0;
        g_watchFieldsRet    = DCGM_ST_OK;
        g_getValuesSinceRet = DCGM_ST_OK;
        g_hasInjectedValue  = false;
        g_injectedGpuId     = 0;
        memset(&g_injectedFieldVal, 0, sizeof(g_injectedFieldVal));
    });

    // FormatFieldViolationError should pick up the injected reason and emit the _TS variant
    DcgmError d { targetGpuId };
    dr.WrapperFormatFieldViolationError(
        d, DCGM_FI_DEV_THERMAL_VIOLATION, targetGpuId, startTime, 5000000000LL, 0.0, "");

    // The error must carry the correct failure reason and be attributed to the affected GPU
    CHECK(d.GetCode() == DCGM_FR_THERMAL_VIOLATIONS_TS);
    CHECK(d.GetEntity().entityId == targetGpuId);
    CHECK(d.GetSeverity() == DCGM_ERROR_MONITOR);
    CHECK(d.GetMessage().find("Thermal violations totaling") != std::string::npos);
}

SCENARIO("DcgmRecorder: negative test for DCGM_FR_UNSUPPORTED_FIELD_TYPE")
{
    DcgmRecorder dr((dcgmHandle_t)42);
    constexpr unsigned int targetGpuId = 1;
    constexpr timelib64_t startTime    = 1000000;
    std::vector<DcgmError> fatalErrors;
    std::vector<DcgmError> ignoredErrors;

    // Set up field watches so CheckErrorFields can query field values
    g_groupId            = 1;
    g_fieldGroupId       = 1;
    g_watchFieldsRet     = DCGM_ST_OK;
    g_getValuesSinceRet  = DCGM_ST_OK;
    g_getFieldSummaryRet = DCGM_ST_OK;

    // DCGM_FI_DEV_GPU_UUID has fieldType DCGM_FT_STRING, which forces the unsupported-type branch
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_UUID };
    std::vector<unsigned int> gpuIds     = { targetGpuId };

    REQUIRE(DcgmFieldsInit() == 0);
    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);

    DcgmNs::Defer cleanup([] {
        g_fieldGroupId       = 0;
        g_groupId            = 0;
        g_watchFieldsRet     = DCGM_ST_OK;
        g_getValuesSinceRet  = DCGM_ST_OK;
        g_getFieldSummaryRet = DCGM_ST_OK;
        memset(&g_injectedFieldSummaryResponse, 0, sizeof(g_injectedFieldSummaryResponse));
        DcgmFieldsTerm();
    });

    // maxTemp is set above the zeroed response highTemp=0 so the bundled CheckGpuTemperature stays silent
    int ret = dr.CheckErrorFields(fieldIds, nullptr, targetGpuId, 1000, fatalErrors, ignoredErrors, startTime);
    CHECK(ret == DR_VIOLATION);
    REQUIRE(fatalErrors.size() == 1);

    // The error must carry the correct failure reason and be attributed to the affected GPU
    CHECK(fatalErrors[0].GetCode() == DCGM_FR_UNSUPPORTED_FIELD_TYPE);
    CHECK(fatalErrors[0].GetEntity().entityId == targetGpuId);
    CHECK(fatalErrors[0].GetSeverity() == DCGM_ERROR_TRIAGE);
    CHECK(fatalErrors[0].GetMessage().find("not supported") != std::string::npos);
    CHECK(ignoredErrors.empty());
}
