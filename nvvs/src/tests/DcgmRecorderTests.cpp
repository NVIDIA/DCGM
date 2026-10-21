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
#include <DcgmRecorder.h>
#include <dcgm_test_apis.h>

#include <Defer.hpp>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginStrings.h>
#include <ScopedDcgmFields.hpp>
#include <TempDir.hpp>
#include <TempFile.hpp>
#include <algorithm>
#include <catch2/catch_all.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <yaml-cpp/yaml.h>


namespace
{
struct DcgmRecorderTestFixture
{
    DcgmRecorderTestFixture()
    {
        mockDcgmOps.m_fieldGroupState.nextFieldGroupId = 1;
    }

    DcgmNs::MockDcgmOps mockDcgmOps;
    NvvsCommon nvvsCommon;
};

void InjectGpuInt64Field(DcgmNs::MockDcgmOps &mockOps,
                         unsigned int gpuId,
                         unsigned short fieldId,
                         long long value,
                         dcgmReturn_t status = DCGM_ST_OK)
{
    dcgmGroupEntityPair_t entity { DCGM_FE_GPU, gpuId };
    DcgmNs::MockDcgmEntity gpu(entity);
    dcgmFieldValue_v2 fv {};
    fv.version       = dcgmFieldValue_version2;
    fv.fieldId       = fieldId;
    fv.fieldType     = DCGM_FT_INT64;
    fv.status        = status;
    fv.entityGroupId = DCGM_FE_GPU;
    fv.entityId      = gpuId;
    fv.value.i64     = value;
    gpu.InjectFieldValue(fieldId, fv);
    mockOps.AddMockedEntity(gpu);
}

void InjectGpuInt64Fields(DcgmNs::MockDcgmOps &mockOps,
                          unsigned int gpuId,
                          std::initializer_list<std::pair<unsigned short, long long>> fields)
{
    dcgmGroupEntityPair_t entity { DCGM_FE_GPU, gpuId };
    DcgmNs::MockDcgmEntity gpu(entity);
    for (auto const &[fieldId, value] : fields)
    {
        dcgmFieldValue_v2 fv {};
        fv.version       = dcgmFieldValue_version2;
        fv.fieldId       = fieldId;
        fv.fieldType     = DCGM_FT_INT64;
        fv.status        = DCGM_ST_OK;
        fv.entityGroupId = DCGM_FE_GPU;
        fv.entityId      = gpuId;
        fv.value.i64     = value;
        gpu.InjectFieldValue(fieldId, fv);
    }
    mockOps.AddMockedEntity(gpu);
}
} // namespace

static long long g_timestamp = 0;

class WrapperDcgmRecorder : protected DcgmRecorder
{
public:
    WrapperDcgmRecorder(dcgmHandle_t handle, DcgmNs::DcgmOpsAnyPtr dcgmOps, NvvsCommon *nvvsCommon)
        : DcgmRecorder(handle, dcgmOps, nvvsCommon)
    {}
    void WrapperFormatFieldViolationError(DcgmError &d,
                                          unsigned short fieldId,
                                          unsigned int gpuId,
                                          timelib64_t start,
                                          int64_t intValue,
                                          double dblValue,
                                          const std::string &fieldName);
    using DcgmRecorder::AddWatches;
    using DcgmRecorder::Shutdown;
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

SCENARIO_METHOD(DcgmRecorderTestFixture,
                "void DcgmRecorder::GetTagFromFieldId(unsigned short fieldId, std::string &tag)")
{
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    std::string str;
    dr.GetTagFromFieldId(0, str);
    CHECK(str == "0");
}

// List of private/protected functions tested as part of the block below:
// void DcgmRecorder::InsertCustomData(unsigned int gpuId, const std::string &name, dcgmTimeseriesInfo_t &data)
// std::vector<dcgmTimeseriesInfo_t> DcgmRecorder::GetCustomGpuStat(unsigned int gpuId, const std::string &name)
// void DcgmRecorder::SetGpuStat(unsigned int gpuId, const std::string &name, double value)
// void DcgmRecorder::SetGpuStat(unsigned int gpuId, const std::string &name, long long value)
SCENARIO_METHOD(DcgmRecorderTestFixture, "void DcgmRecorder::ClearCustomData()")
{
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
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
SCENARIO_METHOD(DcgmRecorderTestFixture,
                "void DcgmRecorder::InsertGroupedData(const std::string &groupName, const std::string &name, "
                "dcgmTimeseriesInfo_t &data)")
{
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);

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
SCENARIO_METHOD(DcgmRecorderTestFixture,
                "int DcgmRecorder::WriteToFile(const std::string &filename, int logFileType, long long testStart)")
{
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    Json::Value json;

    auto jsonFile = TempFile::Random("", "dcgm-recorder-json-");
    auto txtFile  = TempFile::Random("", "dcgm-recorder-txt-");

    auto const jsonFileName = jsonFile.Path().string();
    auto const txtFileName  = txtFile.Path().string();

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

SCENARIO_METHOD(DcgmRecorderTestFixture, "int DcgmRecorder::GetValueIndex(unsigned short fieldId)")
{
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    CHECK(dr.GetValueIndex(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL) == 2);
    CHECK(dr.GetValueIndex(256) == 0);
    CHECK(dr.GetValueIndex(DCGM_FI_DEV_THERMAL_VIOLATION) == 1);
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorderBase returns unsupported defaults")
{
    DcgmRecorderBase recorder;
    dcgmFieldValue_v2 value {};
    dcgmFieldSummaryRequest_t request {};

    CHECK(recorder.GetCurrentFieldValue(0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, value, 0) == DCGM_ST_NOT_SUPPORTED);
    CHECK(recorder.GetFieldSummary(request) == DCGM_ST_NOT_SUPPORTED);
    CHECK(recorder.GetGpuUtilizationNote(0, 0).empty());
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder ignore-error handling separates fatal and ignored errors")
{
    DcgmRecorder recorder(static_cast<dcgmHandle_t>(0), &mockDcgmOps, &nvvsCommon);
    dcgmGroupEntityPair_t entity { DCGM_FE_GPU, 3 };
    DcgmError ignored { entity.entityId };
    ignored.SetCode(DCGM_FR_INTERNAL);
    DcgmError fatal { entity.entityId };
    fatal.SetCode(DCGM_FR_UNKNOWN);
    std::vector<DcgmError> fatalErrors;
    std::vector<DcgmError> ignoredErrors;
    int retCode = DR_VIOLATION;

    recorder.SetIgnoreErrorCodes({ { entity, { DCGM_FR_INTERNAL } } });
    recorder.AddOrClearError(ignored, fatalErrors, ignoredErrors, retCode);
    CHECK(retCode == DR_SUCCESS);
    CHECK(fatalErrors.empty());
    REQUIRE(ignoredErrors.size() == 1);
    CHECK(ignoredErrors[0].GetCode() == DCGM_FR_INTERNAL);

    retCode = DR_VIOLATION;
    recorder.AddOrClearError(fatal, fatalErrors, ignoredErrors, retCode);
    CHECK(retCode == DR_VIOLATION);
    REQUIRE(fatalErrors.size() == 1);
    CHECK(fatalErrors[0].GetCode() == DCGM_FR_UNKNOWN);
    CHECK(ignoredErrors.size() == 1);
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "AddWatches")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    std::vector<unsigned short> fieldIds;
    std::vector<unsigned int> gpuIds;

    // Fail with empty field ids
    CHECK(dr.AddWatches(fieldIds, gpuIds, false, "field_group1", "group1", 300.0) == DCGM_ST_BADPARAM);
    fieldIds.push_back(DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    // Fail with empty gpu IDs
    CHECK(dr.AddWatches(fieldIds, gpuIds, false, "field_group1", "group1", 300.0) == DCGM_ST_BADPARAM);
    gpuIds.push_back(0);

    mockDcgmOps.m_fieldGroupState.watchReturn = DCGM_ST_GPU_IS_LOST;
    CHECK(dr.AddWatches(fieldIds, gpuIds, false, "field_group1", "group1", 300.0) == DCGM_ST_GPU_IS_LOST);
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder::WriteToFile with field watch errors")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS };
    std::vector<unsigned int> gpuIds     = { 0 };

    auto jsonFile           = TempFile::Random("", "dcgm-recorder-errors-json-");
    auto const jsonFileName = jsonFile.Path().string();

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);

    SECTION("fields not watched")
    {
        mockDcgmOps.m_valuesSinceState.returnCode = DCGM_ST_NOT_WATCHED;
    }

    SECTION("no data available")
    {
        mockDcgmOps.m_valuesSinceState.returnCode = DCGM_ST_NO_DATA;
    }

    SECTION("GPU lost")
    {
        mockDcgmOps.m_valuesSinceState.returnCode = DCGM_ST_GPU_IS_LOST;
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

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder::WriteToFile omits empty JSON stats")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
    std::vector<unsigned int> gpuIds     = { 0 };
    auto jsonDir                         = TempDir::Random("dcgm-recorder-empty-json-");
    auto const jsonPath                  = jsonDir.Path() / "stats.json";
    auto const jsonFileName              = jsonPath.string();

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "empty_json_fields", "empty_json_group", 300.0) == DCGM_ST_OK);

    int const ret = dr.WriteToFile(jsonFileName, NVVS_LOGFILE_TYPE_JSON, 0);

    CHECK(ret == 0);
    CHECK_FALSE(std::filesystem::exists(jsonPath));
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder::WriteToFile omits skeleton-only JSON stats")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
    std::vector<unsigned int> gpuIds     = { 0 };
    auto jsonDir                         = TempDir::Random("dcgm-recorder-skeleton-json-");
    auto const jsonPath                  = jsonDir.Path() / "stats.json";
    auto const jsonFileName              = jsonPath.string();

    dcgmFieldValue_v1 fieldValue {};
    fieldValue.version                           = dcgmFieldValue_version1;
    fieldValue.fieldId                           = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
    fieldValue.fieldType                         = DCGM_FT_STRING;
    fieldValue.status                            = DCGM_ST_OK;
    fieldValue.ts                                = 1234567;
    mockDcgmOps.m_valuesSinceState.entityGroupId = DCGM_FE_GPU;
    mockDcgmOps.m_valuesSinceState.entityId      = 0;
    mockDcgmOps.m_valuesSinceState.values        = std::vector<dcgmFieldValue_v1> { fieldValue };

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "skeleton_json_fields", "skeleton_json_group", 300.0) == DCGM_ST_OK);

    int const ret = dr.WriteToFile(jsonFileName, NVVS_LOGFILE_TYPE_JSON, 0);

    CHECK(ret == 0);
    CHECK_FALSE(std::filesystem::exists(jsonPath));
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder::WriteToFile preserves watched JSON stats")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
    std::vector<unsigned int> gpuIds     = { 0 };
    auto jsonFile                        = TempFile::Random("", "dcgm-recorder-watched-json-");
    auto const jsonFileName              = jsonFile.Path().string();

    dcgmFieldValue_v1 fieldValue {};
    fieldValue.version                           = dcgmFieldValue_version1;
    fieldValue.fieldId                           = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
    fieldValue.fieldType                         = DCGM_FT_INT64;
    fieldValue.status                            = DCGM_ST_OK;
    fieldValue.ts                                = 1234567;
    fieldValue.value.i64                         = 42;
    mockDcgmOps.m_valuesSinceState.entityGroupId = DCGM_FE_GPU;
    mockDcgmOps.m_valuesSinceState.entityId      = 0;
    mockDcgmOps.m_valuesSinceState.values        = std::vector<dcgmFieldValue_v1> { fieldValue };

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "watched_json_fields", "watched_json_group", 300.0) == DCGM_ST_OK);

    int const ret = dr.WriteToFile(jsonFileName, NVVS_LOGFILE_TYPE_JSON, 0);

    REQUIRE(ret == 0);
    Json::Value json;
    std::ifstream jsonStream(jsonFileName);
    jsonStream >> json;

    INFO(json.toStyledString());
    std::string const expectedFieldTag = std::to_string(static_cast<unsigned int>(fieldValue.fieldId));

    REQUIRE(json[GPUS].isArray());
    REQUIRE(json[GPUS].size() == 1);
    CHECK(json[GPUS][0]["gpuId"].asUInt() == 0);
    REQUIRE(json[GPUS][0][expectedFieldTag].isArray());
    REQUIRE(json[GPUS][0][expectedFieldTag].size() == 1);
    CHECK(json[GPUS][0][expectedFieldTag][0]["timestamp"].asInt64() == fieldValue.ts);
    CHECK(json[GPUS][0][expectedFieldTag][0]["value"].asInt64() == fieldValue.value.i64);
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

SCENARIO_METHOD(DcgmRecorderTestFixture, "FormatFieldViolationError")
{
    WrapperDcgmRecorder dr((dcgmHandle_t)1, &mockDcgmOps, &nvvsCommon);
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


SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder field violation formatting covers field-specific diagnostics")
{
    GIVEN("a recorder wrapper and a target GPU")
    {
        WrapperDcgmRecorder recorder(static_cast<dcgmHandle_t>(1), &mockDcgmOps, &nvvsCommon);
        unsigned int const gpuId = 2;
        std::string const fieldName("field-name");

        auto format = [&](unsigned short fieldId, int64_t intValue, double doubleValue = DCGM_FP64_BLANK) {
            DcgmError error { gpuId };
            recorder.WrapperFormatFieldViolationError(error, fieldId, gpuId, 14, intValue, doubleValue, fieldName);
            return error;
        };

        WHEN("ECC and PCIe replay counters violate integer thresholds")
        {
            auto sbe  = format(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, 5);
            auto dbe  = format(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, 6);
            auto pcie = format(DCGM_FI_DEV_PCIE_REPLAY_TOTAL, 7);

            THEN("each field maps to its dedicated DCGM error code")
            {
                CHECK(sbe.GetCode() == DCGM_FR_SBE_VIOLATION);
                CHECK(dbe.GetCode() == DCGM_FR_DBE_VIOLATION);
                CHECK(pcie.GetCode() == DCGM_FR_PCIE_REPLAY_VIOLATION);
            }
        }

        WHEN("XID values include contained and uncontained reset signals")
        {
            auto uncontained = format(DCGM_FI_DEV_XID_ERROR, 95);
            auto contained   = format(DCGM_FI_DEV_XID_ERROR, 94);
            auto generic     = format(DCGM_FI_DEV_XID_ERROR, 31);

            THEN("the specific reset classifications are preserved")
            {
                CHECK(uncontained.GetCode() == DCGM_FR_UNCONTAINED_ERROR);
                CHECK(contained.GetCode() == DCGM_FR_CONTAINED_ERROR);
                CHECK(generic.GetCode() == DCGM_FR_XID_ERROR);
            }
        }

        WHEN("row-remap and switch fatal errors are reported")
        {
            auto rowFailed  = format(DCGM_FI_DEV_ROW_REMAP_FAILED, 1);
            auto rowPending = format(DCGM_FI_DEV_ROW_REMAP_PENDING, 1);
            auto sxid       = format(DCGM_FI_DEV_SXID_FATAL_ERROR, 42);

            THEN("the failure reason maps to the corresponding diagnostic code")
            {
                CHECK(rowFailed.GetCode() == DCGM_FR_ROW_REMAP_FAILURE);
                CHECK(rowPending.GetCode() == DCGM_FR_PENDING_ROW_REMAP);
                CHECK(sxid.GetCode() == DCGM_FR_SXID_ERROR);
            }
        }
    }
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder move construction preserves basic recorder behavior")
{
    GIVEN("an empty recorder")
    {
        DcgmRecorder source(static_cast<dcgmHandle_t>(0), &mockDcgmOps, &nvvsCommon);

        WHEN("the recorder is move-constructed")
        {
            DcgmRecorder moved(std::move(source));

            THEN("the moved recorder remains usable for public helper queries")
            {
                CHECK(moved.GetValueIndex(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL) == 2);
                CHECK(moved.GetValueIndex(DCGM_FI_DEV_THERMAL_VIOLATION) == 1);
                CHECK(moved.GetCustomGpuStat(gpuId, "missing").empty());
                CHECK(moved.GetGroupedStat(groupName, "missing").empty());
            }
        }
    }
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder: negative test for DCGM_FR_CLOCKS_EVENT_VIOLATION")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    unsigned int targetGpuId = 1;
    timelib64_t startTime    = 1000000;
    std::vector<DcgmError> fatalErrors;
    std::vector<DcgmError> ignoredErrors;

    // Set up field watches so CheckForClocksEvent can query field values
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_CLOCKS_EVENT_REASONS };
    std::vector<unsigned int> gpuIds     = { targetGpuId };

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);

    // Inject a HW slowdown clocks event for the target GPU via the mock values-since state.
    dcgmFieldValue_v1 injectedFieldVal {};
    injectedFieldVal.version                = dcgmFieldValue_version1;
    injectedFieldVal.fieldId                = DCGM_FI_DEV_CLOCKS_EVENT_REASONS;
    injectedFieldVal.fieldType              = DCGM_FT_INT64;
    injectedFieldVal.status                 = DCGM_ST_OK;
    injectedFieldVal.ts                     = startTime + 5000000;
    injectedFieldVal.value.i64              = DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN;
    mockDcgmOps.m_valuesSinceState.entityId = targetGpuId;
    mockDcgmOps.m_valuesSinceState.values   = std::vector<dcgmFieldValue_v1> { injectedFieldVal };


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

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder: negative test for DCGM_FR_THERMAL_VIOLATIONS_TS")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    WrapperDcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    constexpr unsigned int targetGpuId = 1;
    constexpr timelib64_t startTime    = 1000000;

    // Set up field watches so FormatFieldViolationError can query field values
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_CLOCKS_EVENT_REASONS };
    std::vector<unsigned int> gpuIds     = { targetGpuId };

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);

    // Inject a HW slowdown clocks event for the target GPU via the mock values-since state.
    dcgmFieldValue_v1 injectedFieldVal {};
    injectedFieldVal.version                = dcgmFieldValue_version1;
    injectedFieldVal.fieldId                = DCGM_FI_DEV_CLOCKS_EVENT_REASONS;
    injectedFieldVal.fieldType              = DCGM_FT_INT64;
    injectedFieldVal.status                 = DCGM_ST_OK;
    injectedFieldVal.ts                     = startTime + 5000000;
    injectedFieldVal.value.i64              = DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN;
    mockDcgmOps.m_valuesSinceState.entityId = targetGpuId;
    mockDcgmOps.m_valuesSinceState.values   = std::vector<dcgmFieldValue_v1> { injectedFieldVal };


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

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder: negative test for DCGM_FR_UNSUPPORTED_FIELD_TYPE")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    constexpr unsigned int targetGpuId = 1;
    constexpr timelib64_t startTime    = 1000000;
    std::vector<DcgmError> fatalErrors;
    std::vector<DcgmError> ignoredErrors;

    // DCGM_FI_DEV_GPU_UUID has fieldType DCGM_FT_STRING, which forces the unsupported-type branch
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_UUID };
    std::vector<unsigned int> gpuIds     = { targetGpuId };

    ScopedDcgmFields dcgmFields;
    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);


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

TEST_CASE_METHOD(DcgmRecorderTestFixture, "DcgmRecorder: negative test for DCGM_FR_CANNOT_GET_FIELD_TAG")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    constexpr unsigned int targetGpuId      = 1;
    constexpr unsigned short invalidFieldId = 65535;
    constexpr timelib64_t startTime         = 1000000;
    constexpr long long maxTemp             = 1000;

    std::vector<unsigned short> fieldIds = { invalidFieldId };
    std::vector<DcgmError> fatalErrors;
    std::vector<DcgmError> ignoredErrors;

    ScopedDcgmFields dcgmFields;

    int ret = dr.CheckErrorFields(fieldIds, nullptr, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

    CHECK(ret == DR_COMM_ERROR);
    REQUIRE(fatalErrors.size() == 1);
    CHECK(fatalErrors[0].GetCode() == DCGM_FR_CANNOT_GET_FIELD_TAG);
    CHECK(fatalErrors[0].GetEntity().entityGroupId == DCGM_FE_GPU);
    CHECK(fatalErrors[0].GetEntity().entityId == targetGpuId);
    CHECK(fatalErrors[0].GetSeverity() == DCGM_ERROR_MONITOR);
    CHECK(fatalErrors[0].GetMessage().find("Unable to get field information") != std::string::npos);
    CHECK(ignoredErrors.empty());
}


SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder: negative test for DCGM_FR_FIELD_THRESHOLD")
{
    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    constexpr unsigned int targetGpuId = 1;
    constexpr timelib64_t startTime    = 1000000;
    constexpr int64_t thresholdValue   = 50;
    // Set maxTemp high enough that CheckGpuTemperature won't fire on any injected test value
    constexpr int64_t maxTemp = 1000000000;
    std::vector<DcgmError> fatalErrors;
    std::vector<DcgmError> ignoredErrors;

    ScopedDcgmFields dcgmFields;

    // DCGM_FI_DEV_GPU_TEMP_CELSIUS is INT64 and not special-cased, so over-threshold routes to DCGM_FR_FIELD_THRESHOLD
    std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
    std::vector<unsigned int> gpuIds     = { targetGpuId };

    REQUIRE(dr.AddWatches(fieldIds, gpuIds, false, "test_field_group", "test_group", 300.0) == DCGM_ST_OK);

    std::vector<dcgmTimeseriesInfo_t> thresholds(fieldIds.size());
    thresholds[0].isInt   = true;
    thresholds[0].val.i64 = thresholdValue;

    // DCGM_SUMMARY_MAX is the default index (0) for fields that do not appear in GetValueIndex's switch.
    auto &injectedValue = mockDcgmOps.m_fieldSummaryState.response.values[0].i64;

    SECTION("value strictly above threshold emits DCGM_FR_FIELD_THRESHOLD attributed to the affected GPU only")
    {
        injectedValue = thresholdValue + 1;

        int ret
            = dr.CheckErrorFields(fieldIds, &thresholds, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

        CHECK(ret == DR_VIOLATION);
        REQUIRE(fatalErrors.size() == 1);
        CHECK(fatalErrors[0].GetCode() == DCGM_FR_FIELD_THRESHOLD);
        CHECK(fatalErrors[0].GetEntity().entityGroupId == DCGM_FE_GPU);
        CHECK(fatalErrors[0].GetEntity().entityId == targetGpuId);
        CHECK(fatalErrors[0].GetSeverity() == DCGM_ERROR_MONITOR);
        CHECK(fatalErrors[0].GetMessage().find("above the threshold") != std::string::npos);
        CHECK(ignoredErrors.empty());
    }

    SECTION("value far above threshold still emits a single DCGM_FR_FIELD_THRESHOLD for the affected GPU")
    {
        injectedValue = thresholdValue * 1000;

        int ret
            = dr.CheckErrorFields(fieldIds, &thresholds, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

        CHECK(ret == DR_VIOLATION);
        REQUIRE(fatalErrors.size() == 1);
        CHECK(fatalErrors[0].GetCode() == DCGM_FR_FIELD_THRESHOLD);
        CHECK(fatalErrors[0].GetEntity().entityId == targetGpuId);
    }

    SECTION("value equal to threshold produces no DCGM_FR_FIELD_THRESHOLD error (boundary is exclusive)")
    {
        injectedValue = thresholdValue;

        int ret
            = dr.CheckErrorFields(fieldIds, &thresholds, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

        CHECK(ret == DR_SUCCESS);
        CHECK(fatalErrors.empty());
        CHECK(ignoredErrors.empty());
    }

    SECTION("value below threshold produces no DCGM_FR_FIELD_THRESHOLD error")
    {
        injectedValue = thresholdValue - 1;

        int ret
            = dr.CheckErrorFields(fieldIds, &thresholds, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

        CHECK(ret == DR_SUCCESS);
        CHECK(fatalErrors.empty());
        CHECK(ignoredErrors.empty());
    }

    SECTION("blank INT64 sentinel never triggers DCGM_FR_FIELD_THRESHOLD")
    {
        injectedValue = DCGM_INT64_BLANK;

        int ret
            = dr.CheckErrorFields(fieldIds, &thresholds, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

        CHECK(ret == DR_SUCCESS);
        CHECK(fatalErrors.empty());
        CHECK(ignoredErrors.empty());
    }

    SECTION("null failureThresholds disables the threshold path entirely")
    {
        injectedValue = thresholdValue + 1;

        int ret = dr.CheckErrorFields(fieldIds, nullptr, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

        CHECK(ret == DR_VIOLATION);
        REQUIRE_FALSE(fatalErrors.empty());
        for (auto const &err : fatalErrors)
        {
            CHECK(err.GetCode() != DCGM_FR_FIELD_THRESHOLD);
        }
    }

    SECTION(
        "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL over threshold routes to DCGM_FR_DBE_THRESHOLD_VIOLATION, not DCGM_FR_FIELD_THRESHOLD")
    {
        DcgmRecorder drSpecial(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        std::vector<unsigned short> specialIds = { DCGM_FI_DEV_ECC_DBE_VOL_TOTAL };
        std::vector<dcgmTimeseriesInfo_t> specialThresholds(1);
        specialThresholds[0].isInt   = true;
        specialThresholds[0].val.i64 = thresholdValue;

        // GetValueIndex maps ECC_DBE_VOL_TOTAL to slot 2
        mockDcgmOps.m_fieldSummaryState.response.values[2].i64 = thresholdValue + 1;

        REQUIRE(drSpecial.AddWatches(specialIds, gpuIds, false, "special_field_group", "special_group", 300.0)
                == DCGM_ST_OK);

        int ret = drSpecial.CheckErrorFields(
            specialIds, &specialThresholds, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

        CHECK(ret == DR_VIOLATION);
        REQUIRE(fatalErrors.size() == 1);
        CHECK(fatalErrors[0].GetCode() == DCGM_FR_DBE_THRESHOLD_VIOLATION);
        CHECK(fatalErrors[0].GetCode() != DCGM_FR_FIELD_THRESHOLD);
        CHECK(fatalErrors[0].GetEntity().entityId == targetGpuId);
    }
}


TEST_CASE_METHOD(DcgmRecorderTestFixture, "DcgmRecorder: negative test for DCGM_FR_FIELD_QUERY")
{
    constexpr unsigned int targetGpuId = 1;
    constexpr timelib64_t startTime    = 1000000;
    constexpr long long maxTemp        = 1000;

    mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
    DcgmRecorder dr(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
    std::vector<DcgmError> fatalErrors;
    std::vector<DcgmError> ignoredErrors;

    ScopedDcgmFields dcgmFields;

    auto requireFieldQueryError
        = [&](dcgm_field_entity_group_t entityGroup, dcgm_field_eid_t entityId, std::string_view messageFragment) {
              REQUIRE(fatalErrors.size() == 1);
              CHECK(fatalErrors[0].GetCode() == DCGM_FR_FIELD_QUERY);
              CHECK(fatalErrors[0].GetEntity().entityGroupId == entityGroup);
              CHECK(fatalErrors[0].GetEntity().entityId == entityId);
              CHECK(fatalErrors[0].GetSeverity() == DCGM_ERROR_MONITOR);
              CHECK(fatalErrors[0].GetMessage().find(messageFragment) != std::string::npos);
              CHECK(ignoredErrors.empty());
          };

    SECTION("CheckErrorFields reports field summary query failure")
    {
        std::vector<unsigned short> fieldIds                                         = { DCGM_FI_DEV_POWER_VIOLATION };
        mockDcgmOps.m_fieldSummaryState.returnByFieldId[DCGM_FI_DEV_POWER_VIOLATION] = DCGM_ST_GENERIC_ERROR;

        int ret = dr.CheckErrorFields(fieldIds, nullptr, targetGpuId, maxTemp, fatalErrors, ignoredErrors, startTime);

        CHECK(ret == DR_COMM_ERROR);
        requireFieldQueryError(DCGM_FE_GPU, targetGpuId, "power_violation");
    }

    SECTION("CheckPerSecondErrorConditions reports values-since query failure")
    {
        std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_POWER_VIOLATION };
        std::vector<dcgmFieldValue_v1> failureThreshold(fieldIds.size());
        failureThreshold[0].fieldId   = DCGM_FI_DEV_POWER_VIOLATION;
        failureThreshold[0].fieldType = DCGM_FT_INT64;
        failureThreshold[0].status    = DCGM_ST_OK;
        failureThreshold[0].value.i64 = 1;

        mockDcgmOps.m_valuesSinceState.returnCode = DCGM_ST_GENERIC_ERROR;

        dcgmReturn_t ret
            = dr.CheckPerSecondErrorConditions(fieldIds, failureThreshold, targetGpuId, fatalErrors, startTime);

        CHECK(ret == DR_COMM_ERROR);
        requireFieldQueryError(DCGM_FE_GPU, targetGpuId, "power_violation");
    }

    SECTION("CheckGpuTemperature reports field summary query failure")
    {
        std::string infoMsg;
        long long highTemp                         = 123;
        mockDcgmOps.m_fieldSummaryState.returnCode = DCGM_ST_GENERIC_ERROR;

        int ret = dr.CheckGpuTemperature(targetGpuId, fatalErrors, maxTemp, infoMsg, startTime, highTemp);

        CHECK(ret == DR_COMM_ERROR);
        CHECK(highTemp == 0);
        requireFieldQueryError(DCGM_FE_GPU, targetGpuId, "gpu temperature");
    }

    SECTION("CheckForClocksEvent reports values-since query failure")
    {
        mockDcgmOps.m_valuesSinceState.returnCode = DCGM_ST_GENERIC_ERROR;

        int ret = dr.CheckForClocksEvent(targetGpuId, startTime, fatalErrors, ignoredErrors);

        CHECK(ret == DR_COMM_ERROR);
        requireFieldQueryError(DCGM_FE_GPU, targetGpuId, "clocks event");
    }

    SECTION("GetLatestValuesForWatchedFields reports all-watched-fields query failure")
    {
        // Use a fresh recorder because AddWatches mutates recorder watch state for this section only
        DcgmNs::MockDcgmOps watchedDcgmLibMock;
        watchedDcgmLibMock.m_groupManagementState.addReturn = DCGM_ST_OK;
        DcgmRecorder watchedDr(watchedDcgmLibMock.m_handleId, &watchedDcgmLibMock, &nvvsCommon);
        std::vector<unsigned short> fieldIds = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        std::vector<unsigned int> gpuIds     = { targetGpuId };

        REQUIRE(watchedDr.AddWatches(fieldIds, gpuIds, false, "field_query_fields", "field_query_group", 300.0)
                == DCGM_ST_OK);

        watchedDcgmLibMock.m_entitiesGetLatestValuesState.returnCode = DCGM_ST_GENERIC_ERROR;

        int ret = watchedDr.GetLatestValuesForWatchedFields(0, fatalErrors);

        CHECK(ret == DR_COMM_ERROR);
        DcgmError unknownGpu { DcgmError::GpuIdTag::Unknown };
        requireFieldQueryError(
            unknownGpu.GetEntity().entityGroupId, unknownGpu.GetEntity().entityId, "all watched fields");
    }

    SECTION("VerifyHBMTemperature reports HBM temperature query failure")
    {
        mockDcgmOps.m_fieldSummaryState.returnCode = DCGM_ST_GENERIC_ERROR;

        int ret = dr.VerifyHBMTemperature(targetGpuId, fatalErrors, maxTemp, startTime);

        CHECK(ret == DR_COMM_ERROR);
        requireFieldQueryError(DCGM_FE_GPU, targetGpuId, "HBM temperature");
    }
}


SCENARIO_METHOD(DcgmRecorderTestFixture,
                "DcgmRecorder preserves custom stat separation and reports file output failures")
{
    GIVEN("custom GPU and group statistics across multiple keys")
    {
        DcgmRecorder dr(static_cast<dcgmHandle_t>(0), &mockDcgmOps, &nvvsCommon);
        dr.SetGpuStat(0, "temperature", 70LL);
        dr.SetGpuStat(1, "temperature", 80LL);
        dr.SetGpuStat(1, "power", 250.5);
        dr.SetGroupedStat("aggregate", "count", 2LL);
        dr.SetGroupedStat("aggregate", "power", 500.0);

        THEN("lookups remain scoped by GPU, group, and stat name")
        {
            auto gpu0Temp  = dr.GetCustomGpuStat(0, "temperature");
            auto gpu1Temp  = dr.GetCustomGpuStat(1, "temperature");
            auto gpu1Power = dr.GetCustomGpuStat(1, "power");
            auto missing   = dr.GetCustomGpuStat(0, "missing");

            REQUIRE(gpu0Temp.size() == 1);
            CHECK(gpu0Temp[0].isInt);
            CHECK(gpu0Temp[0].val.i64 == 70ULL);
            REQUIRE(gpu1Temp.size() == 1);
            CHECK(gpu1Temp[0].val.i64 == 80ULL);
            REQUIRE(gpu1Power.size() == 1);
            CHECK_FALSE(gpu1Power[0].isInt);
            CHECK(gpu1Power[0].val.fp64 == 250.5);
            CHECK(missing.empty());

            auto groupedCount = dr.GetGroupedStat("aggregate", "count");
            auto groupedPower = dr.GetGroupedStat("aggregate", "power");
            REQUIRE(groupedCount.size() == 1);
            CHECK(groupedCount[0].val.i64 == 2ULL);
            REQUIRE(groupedPower.size() == 1);
            CHECK(groupedPower[0].val.fp64 == 500.0);
        }
    }

    GIVEN("a recorder asked to write to an invalid file target")
    {
        DcgmRecorder dr(static_cast<dcgmHandle_t>(0), &mockDcgmOps, &nvvsCommon);
        dr.SetGpuStat(0, "samples", 11LL);

        THEN("the write fails with the documented error code")
        {
            CHECK(dr.WriteToFile(".", NVVS_LOGFILE_TYPE_JSON, 0) == -1);
        }
    }

    GIVEN("a recorder with custom data and an unknown log file type")
    {
        DcgmRecorder dr(static_cast<dcgmHandle_t>(0), &mockDcgmOps, &nvvsCommon);
        dr.SetGpuStat(0, "samples", 11LL);
        auto jsonFile           = TempFile::Random("", "recorder-default-json-");
        auto const jsonFileName = jsonFile.Path().string();

        WHEN("the file is written")
        {
            int const ret = dr.WriteToFile(jsonFileName, 9999, 0);

            THEN("the default JSON path is used")
            {
                REQUIRE(ret == 0);
                Json::Value json;
                std::ifstream jsonStream(jsonFileName);
                jsonStream >> json;
                CHECK(json[GPUS][0]["samples"][0]["value"].asInt() == 11);
            }
        }
    }
}


SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder reports summary-derived temperatures and utilization notes")
{
    GIVEN("a recorder with an injected field-summary response")
    {
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        std::vector<DcgmError> errors;
        std::string infoMsg;
        long long highTemp = 0;

        WHEN("the reported GPU temperature exceeds the configured maximum")
        {
            mockDcgmOps.m_fieldSummaryState.response.values[0].i64 = 92;
            mockDcgmOps.m_fieldSummaryState.response.values[1].i64 = 77;
            auto const result = recorder.CheckGpuTemperature(3, errors, 85, infoMsg, 1000, highTemp);

            THEN("the violation is reported with the observed maximum and average text")
            {
                CHECK(result == DR_VIOLATION);
                CHECK(highTemp == 92);
                REQUIRE(errors.size() == 1);
                CHECK(errors[0].GetCode() == DCGM_FR_TEMP_VIOLATION);
                CHECK(errors[0].GetEntity().entityId == 3);
                CHECK_THAT(infoMsg, Catch::Matchers::ContainsSubstring("temperature average"));
                CHECK_THAT(infoMsg, Catch::Matchers::ContainsSubstring("77 C"));
            }
        }

        WHEN("the summary query itself fails")
        {
            mockDcgmOps.m_fieldSummaryState.returnCode = DCGM_ST_GENERIC_ERROR;
            auto const result = recorder.CheckGpuTemperature(4, errors, 85, infoMsg, 1000, highTemp);

            THEN("a communication error is returned and high temperature is cleared")
            {
                CHECK(result == DR_COMM_ERROR);
                CHECK(highTemp == 0);
                REQUIRE(errors.size() == 1);
                CHECK(errors[0].GetCode() == DCGM_FR_FIELD_QUERY);
            }
        }

        WHEN("GPU utilization is below the diagnostic threshold")
        {
            mockDcgmOps.m_fieldSummaryState.response.values[0].i64 = 30;
            auto const note                                        = recorder.GetGpuUtilizationNote(5, 1000);

            THEN("the note asks the user to investigate competing GPU work")
            {
                CHECK_THAT(note, Catch::Matchers::ContainsSubstring("GPU usage was only 30"));
                CHECK_THAT(note, Catch::Matchers::ContainsSubstring("GPU 5"));
            }
        }

        WHEN("GPU utilization meets the diagnostic threshold")
        {
            mockDcgmOps.m_fieldSummaryState.response.values[0].i64 = 95;
            auto const note                                        = recorder.GetGpuUtilizationNote(6, 1000);

            THEN("no utilization note is returned")
            {
                CHECK(note.empty());
            }
        }
    }
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder per-second error checks validate threshold inputs")
{
    GIVEN("one watched field and no matching threshold")
    {
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        std::vector<unsigned short> fieldIds { DCGM_FI_DEV_PCIE_REPLAY_TOTAL };
        std::vector<dcgmFieldValue_v1> thresholds;
        std::vector<DcgmError> errors;

        WHEN("per-second conditions are checked")
        {
            dcgmReturn_t const result = recorder.CheckPerSecondErrorConditions(fieldIds, thresholds, 0, errors, 1000);

            THEN("the public API rejects the mismatched inputs without adding field errors")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(errors.empty());
            }
        }
    }
}

SCENARIO_METHOD(DcgmRecorderTestFixture,
                "DcgmRecorder watch frequency and error strings are observable through public APIs")
{
    GIVEN("a recorder with successful group and field creation")
    {
        mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        std::vector<unsigned int> gpuIds { 2 };

        WHEN("a custom watch frequency is configured before adding watches")
        {
            recorder.SetWatchFrequency(1234567);
            auto const result = recorder.AddWatches(fieldIds, gpuIds, false, "fields", "gpus", 11.5);

            THEN("the DCGM watch shim receives the custom frequency and duration padding")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_fieldGroupState.lastUpdateFreq == 1234567);
                CHECK_THAT(mockDcgmOps.m_fieldGroupState.lastMaxKeepAge, Catch::Matchers::WithinAbs(41.5, 0.000001));
            }
        }

        WHEN("known and unknown DCGM status codes are formatted")
        {
            THEN("known values use DCGM messages and unknown values include the numeric code")
            {
                CHECK_THAT(recorder.ErrorAsString(DCGM_ST_BADPARAM),
                           Catch::Matchers::ContainsSubstring("Bad parameter"));
                CHECK_THAT(recorder.ErrorAsString(static_cast<dcgmReturn_t>(-9999)),
                           Catch::Matchers::ContainsSubstring("Unknown error code -9999"));
            }
        }
    }
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder summary checks report threshold and HBM temperature failures")
{
    DcgmNs::Defer cleanupFields([] { DcgmFieldsTerm(); });
    REQUIRE(DcgmFieldsInit() == 0);

    GIVEN("field summaries above configured integer and double thresholds")
    {
        mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        constexpr unsigned int targetGpuId = 2;
        constexpr timelib64_t startTime    = 1000000;
        std::vector<DcgmError> fatalErrors;
        std::vector<DcgmError> ignoredErrors;

        WHEN("a replay counter exceeds its integer threshold")
        {
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_PCIE_REPLAY_TOTAL };
            std::vector<dcgmTimeseriesInfo_t> thresholds(1);
            thresholds[0].isInt                                    = true;
            thresholds[0].val.i64                                  = 7;
            mockDcgmOps.m_fieldSummaryState.response.values[2].i64 = 9;

            int const ret = recorder.CheckErrorFields(
                fieldIds, &thresholds, targetGpuId, 1000, fatalErrors, ignoredErrors, startTime);

            THEN("the recorder emits the field-specific threshold error")
            {
                CHECK(ret == DR_VIOLATION);
                REQUIRE(fatalErrors.size() == 1);
                CHECK(fatalErrors[0].GetCode() == DCGM_FR_PCIE_REPLAY_THRESHOLD_VIOLATION);
                CHECK(fatalErrors[0].GetEntity().entityId == targetGpuId);
                CHECK(ignoredErrors.empty());
            }
        }

        WHEN("a double-valued field exceeds its threshold")
        {
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_BOARD_POWER_WATTS };
            std::vector<dcgmTimeseriesInfo_t> thresholds(1);
            thresholds[0].isInt                                     = false;
            thresholds[0].val.fp64                                  = 250.0;
            mockDcgmOps.m_fieldSummaryState.response.values[0].fp64 = 275.5;

            int const ret = recorder.CheckErrorFields(
                fieldIds, &thresholds, targetGpuId, 1000, fatalErrors, ignoredErrors, startTime);

            THEN("the recorder emits a double threshold violation")
            {
                CHECK(ret == DR_VIOLATION);
                CHECK(std::any_of(fatalErrors.begin(), fatalErrors.end(), [](DcgmError const &error) {
                    return error.GetCode() == DCGM_FR_FIELD_THRESHOLD_DBL;
                }));
                CHECK(std::all_of(fatalErrors.begin(), fatalErrors.end(), [](DcgmError const &error) {
                    return error.GetEntity().entityId == targetGpuId;
                }));
                CHECK(ignoredErrors.empty());
            }
        }
    }

    GIVEN("HBM temperature summaries")
    {
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        constexpr unsigned int targetGpuId = 3;
        std::vector<DcgmError> errors;

        WHEN("the maximum HBM temperature exceeds the allowed value")
        {
            mockDcgmOps.m_fieldSummaryState.response.values[0].i64 = 101;
            int const ret = recorder.VerifyHBMTemperature(targetGpuId, errors, 95, 1000);

            THEN("a temperature violation is reported")
            {
                CHECK(ret == DR_VIOLATION);
                REQUIRE(errors.size() == 1);
                CHECK(errors[0].GetCode() == DCGM_FR_TEMP_VIOLATION);
                CHECK(errors[0].GetEntity().entityId == targetGpuId);
            }
        }

        WHEN("the summary query fails")
        {
            mockDcgmOps.m_fieldSummaryState.returnCode = DCGM_ST_GENERIC_ERROR;
            int const ret                              = recorder.VerifyHBMTemperature(targetGpuId, errors, 95, 1000);

            THEN("the error is converted to a communication failure")
            {
                CHECK(ret == DR_COMM_ERROR);
                REQUIRE(errors.size() == 1);
                CHECK(errors[0].GetCode() == DCGM_FR_FIELD_QUERY);
            }
        }
    }
}


SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder HBM error checks use public field queries")
{
    GIVEN("a recorder and GPU entity metadata")
    {
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        dcgmDiagPluginEntityInfo_v1 entityInfo {};
        entityInfo.entity.entityGroupId = DCGM_FE_GPU;
        entityInfo.entity.entityId      = 7;
        std::vector<DcgmError> errors;

        WHEN("current HBM temperature exceeds the live max temperature")
        {
            InjectGpuInt64Field(mockDcgmOps, 7, DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS, 95);
            mockDcgmOps.m_fieldSummaryState.response.values[0].i64 = 102;

            THEN("CheckHBMErrorFields reports a temperature violation")
            {
                CHECK(recorder.CheckHBMErrorFields(entityInfo, errors, 1234) == DR_VIOLATION);
                REQUIRE(errors.size() == 1);
                CHECK(errors[0].GetCode() == DCGM_FR_TEMP_VIOLATION);
                CHECK(errors[0].GetEntity().entityId == 7);
            }
        }

        WHEN("the max HBM temperature is unavailable for a real GPU")
        {
            mockDcgmOps.m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS]
                = DCGM_ST_NO_DATA;
            entityInfo.auxField.gpu.status = DcgmEntityStatusOk;

            THEN("the HBM check is skipped without manufacturing a violation")
            {
                CHECK(recorder.CheckHBMErrorFields(entityInfo, errors, 1234) == DR_SUCCESS);
                CHECK(errors.empty());
            }
        }

        WHEN("the max HBM temperature is unavailable for a fake GPU")
        {
            mockDcgmOps.m_entitiesGetLatestValuesState.returnByFieldId[DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS]
                = DCGM_ST_NO_DATA;
            entityInfo.auxField.gpu.status                         = DcgmEntityStatusFake;
            mockDcgmOps.m_fieldSummaryState.response.values[0].i64 = 101;

            THEN("the fake-GPU default threshold is used")
            {
                CHECK(recorder.CheckHBMErrorFields(entityInfo, errors, 1234) == DR_VIOLATION);
                REQUIRE(errors.size() == 1);
                CHECK(errors[0].GetCode() == DCGM_FR_TEMP_VIOLATION);
            }
        }

        WHEN("the entity is not a GPU")
        {
            entityInfo.entity.entityGroupId = DCGM_FE_CPU;

            THEN("the HBM check is a no-op")
            {
                CHECK(recorder.CheckHBMErrorFields(entityInfo, errors, 1234) == DR_SUCCESS);
                CHECK(errors.empty());
            }
        }
    }
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder common-error checks use update and XID collaborators")
{
    DcgmNs::Defer cleanupFields([] { DcgmFieldsTerm(); });
    REQUIRE(DcgmFieldsInit() == 0);

    GIVEN("a GPU entity and clean summary/XID responses")
    {
        mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
        InjectGpuInt64Fields(
            mockDcgmOps,
            2,
            { { DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS, 95 }, { DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS, 95 } });
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        TestParameters tp;
        tp.AddDouble(PCIE_STR_MAX_PCIE_REPLAYS, 1000.0);
        nvvsPluginResult_t result = NVVS_RESULT_PASS;
        dcgmDiagPluginEntityInfo_v1 entityInfo {};
        entityInfo.entity.entityGroupId = DCGM_FE_GPU;
        entityInfo.entity.entityId      = 2;
        entityInfo.auxField.gpu.status  = DcgmEntityStatusFake;
        std::vector<dcgmDiagPluginEntityInfo_v1> entityInfos { entityInfo };
        std::vector<DcgmError> fatalErrors;
        std::vector<DcgmError> ignoredErrors;

        WHEN("common errors are checked")
        {
            recorder.CheckCommonErrors(tp, 1234, result, entityInfos, fatalErrors, ignoredErrors);

            THEN("field updates are requested and no errors are manufactured")
            {
                CHECK(result == NVVS_RESULT_PASS);
                CHECK(fatalErrors.empty());
                CHECK(ignoredErrors.empty());
                CHECK(mockDcgmOps.m_updateAllFieldsState.callCount == 1);
                CHECK(mockDcgmOps.m_updateAllFieldsState.lastWaitForUpdate == 1);
            }
        }

        WHEN("a replay counter exceeds the configured threshold")
        {
            mockDcgmOps.m_fieldSummaryState.response.values[2].i64 = 1001;
            recorder.CheckCommonErrors(tp, 1234, result, entityInfos, fatalErrors, ignoredErrors);

            THEN("the plugin result is failed and a field error is recorded")
            {
                CHECK(result == NVVS_RESULT_FAIL);
                CHECK_FALSE(fatalErrors.empty());
                CHECK(std::any_of(fatalErrors.begin(), fatalErrors.end(), [](DcgmError const &error) {
                    return error.GetCode() == DCGM_FR_PCIE_REPLAY_THRESHOLD_VIOLATION;
                }));
            }
        }
    }
}


SCENARIO_METHOD(DcgmRecorderTestFixture,
                "DcgmRecorder field-summary public helpers normalize no-data and surface errors")
{
    GIVEN("a field summary request")
    {
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        dcgmFieldSummaryRequest_t request {};
        request.fieldId       = DCGM_FI_DEV_GPU_UTIL;
        request.entityGroupId = DCGM_FE_GPU;
        request.entityId      = 3;

        WHEN("DCGM reports no summary data")
        {
            mockDcgmOps.m_fieldSummaryState.returnCode = DCGM_ST_NO_DATA;
            dcgmReturn_t const ret                     = recorder.GetFieldSummary(request);

            THEN("the recorder treats lack of samples as a successful empty summary")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(request.version == dcgmFieldSummaryRequest_version1);
            }
        }

        WHEN("the utilization summary query fails")
        {
            mockDcgmOps.m_fieldSummaryState.returnCode = DCGM_ST_GENERIC_ERROR;
            std::string const note                     = recorder.GetGpuUtilizationNote(3, 1000);

            THEN("the public note contains the formatted DCGM error")
            {
                CHECK_FALSE(note.empty());
                CHECK_THAT(note, Catch::Matchers::ContainsSubstring("Generic"));
            }
        }
    }
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder latest-value collection reports success and DCGM query failures")
{
    GIVEN("a recorder with one watched field")
    {
        mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
        InjectGpuInt64Field(mockDcgmOps, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 72);
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        std::vector<unsigned int> gpuIds { 0 };
        REQUIRE(recorder.AddWatches(fieldIds, gpuIds, false, "latest_fields", "latest_gpus", 5.0) == DCGM_ST_OK);

        WHEN("latest values are available")
        {
            std::vector<DcgmError> errors;
            mockDcgmOps.m_entitiesGetLatestValuesState.returnCode = DCGM_ST_OK;

            int const result = recorder.GetLatestValuesForWatchedFields(0, errors);

            THEN("the collection succeeds without recording fatal errors")
            {
                CHECK(result == DR_SUCCESS);
                CHECK(errors.empty());
            }
        }

        WHEN("the DCGM latest-value query fails")
        {
            std::vector<DcgmError> errors;
            mockDcgmOps.m_entitiesGetLatestValuesState.returnCode = DCGM_ST_GENERIC_ERROR;

            int const result = recorder.GetLatestValuesForWatchedFields(DCGM_FV_FLAG_LIVE_DATA, errors);

            THEN("the query failure is converted to a recorder communication error")
            {
                CHECK(result == DR_COMM_ERROR);
                REQUIRE(errors.size() == 1);
                CHECK(errors[0].GetCode() == DCGM_FR_FIELD_QUERY);
            }
        }
    }
}


SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder watch lifecycle preserves public error contracts")
{
    GIVEN("a recorder without a connected DCGM handle")
    {
        WrapperDcgmRecorder recorder(static_cast<dcgmHandle_t>(0), &mockDcgmOps, &nvvsCommon);
        std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        std::vector<unsigned int> gpuIds { 0, 1 };

        WHEN("watches are requested")
        {
            dcgmReturn_t const result = recorder.AddWatches(fieldIds, gpuIds, false, "fields", "disconnected", 1.0);

            THEN("the recorder reports the invalid connection")
            {
                CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
            }
        }
    }

    GIVEN("a recorder with successful group and field shims")
    {
        mockDcgmOps.m_groupManagementState.addReturn = DCGM_ST_OK;
        WrapperDcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        std::vector<unsigned int> gpuIds { 3, 4 };

        WHEN("watches are added and then shut down")
        {
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS };
            dcgmReturn_t const watchResult
                = recorder.AddWatches(fieldIds, gpuIds, false, "recorder_fields", "recorder_group", 2.0);
            dcgmReturn_t const shutdownResult = recorder.Shutdown();

            THEN("the recorder cleans up the watch lifecycle")
            {
                CHECK(watchResult == DCGM_ST_OK);
                CHECK(shutdownResult == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_fieldGroupState.lastUpdateFreq == 5000000);
                CHECK_THAT(mockDcgmOps.m_fieldGroupState.lastMaxKeepAge, Catch::Matchers::WithinAbs(32.0, 0.000001));
            }
        }
    }
}

SCENARIO_METHOD(DcgmRecorderTestFixture, "DcgmRecorder current-field wrappers surface DCGM query failures")
{
    GIVEN("a recorder and caller-owned output buffers")
    {
        DcgmRecorder recorder(mockDcgmOps.m_handleId, &mockDcgmOps, &nvvsCommon);
        dcgmFieldValue_v2 currentValue {};
        WHEN("current field retrieval fails through the DCGM entity-value shim")
        {
            mockDcgmOps.m_entitiesGetLatestValuesState.returnCode = DCGM_ST_GENERIC_ERROR;
            dcgmReturn_t const result
                = recorder.GetCurrentFieldValue(9, DCGM_FI_DEV_GPU_TEMP_CELSIUS, currentValue, DCGM_FV_FLAG_LIVE_DATA);

            THEN("the DCGM error is returned through the public wrapper")
            {
                CHECK(result == DCGM_ST_GENERIC_ERROR);
                CHECK(currentValue.status == 0);
            }
        }
    }
}
