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
#include <DcgmValuesSinceHolder.h>
#include <catch2/catch.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

static const char *c_GPUS  = "GPUS";
static const char *c_value = "value";

SCENARIO("void DcgmEntityTimeSeries::AddValue(unsigned short fieldId, dcgmFieldValue_v1 &val)")
{
    // also bool DcgmEntityTimeSeries::IsFieldStored(unsigned short fieldId)
    DcgmEntityTimeSeries timeSeries;
    dcgmFieldValue_v1 fv = {};

    CHECK(!timeSeries.IsFieldStored(1));
    timeSeries.AddValue(1, fv);
    CHECK(timeSeries.IsFieldStored(1));
    CHECK(!timeSeries.IsFieldStored(2));
}

SCENARIO("void DcgmEntityTimeSeries::GetFirstNonZero(unsigned short fieldId, dcgmFieldValue_v1 &dfv, uint64_t mask)")
{
    DcgmEntityTimeSeries timeSeries;
    dcgmFieldValue_v1 fv          = {};
    const dcgmFieldValue_v1 zeros = {};

    // INT64

    fv.fieldId   = 1;
    fv.fieldType = DCGM_FT_INT64;

    // First we inject a zero
    fv.value.i64 = 0;
    timeSeries.AddValue(1, fv);

    // Then a non-zero
    fv.value.i64 = 33;
    timeSeries.AddValue(1, fv);

    // Now we test retrieving the value
    memset(&fv, 0, sizeof(fv));
    timeSeries.GetFirstNonZero(1, fv, 1);
    CHECK(33 == fv.value.i64);

    // Test again with mask = 0
    memset(&fv, 0, sizeof(fv));
    timeSeries.GetFirstNonZero(1, fv, 0);
    CHECK(33 == fv.value.i64);

    // Test again with mask = 2. This should not fetch a value
    memset(&fv, 0, sizeof(fv));
    timeSeries.GetFirstNonZero(1, fv, 2);
    CHECK(0 == memcmp(&fv, &zeros, sizeof(fv)));

    // DOUBLE

    fv.fieldId   = 2;
    fv.fieldType = DCGM_FT_DOUBLE;

    // First we inject a zero
    fv.value.dbl = 0.0;
    timeSeries.AddValue(2, fv);

    // Then a non-zero
    fv.value.dbl = 33.0;
    timeSeries.AddValue(2, fv);

    memset(&fv, 0, sizeof(fv));

    // Pass 2 mask to ensure the double logic ignores the mask
    timeSeries.GetFirstNonZero(2, fv, 2);
    CHECK(33.0 == fv.value.dbl);

    // INVALID FIELD (UNINITIALIZED)

    memset(&fv, 0, sizeof(fv));
    timeSeries.GetFirstNonZero(3, fv, 0);
    CHECK(0 == memcmp(&fv, &zeros, sizeof(fv)));
}

SCENARIO("void DcgmEntityTimeSeries::AddToJson(Json::Value &jv, unsigned int jsonIndex)")
{
    const unsigned int jsonIndex = 0;
    DcgmEntityTimeSeries timeSeries;
    dcgmFieldValue_v1 fv = {};
    Json::Value jv;

    // INT64

    fv.fieldId   = 1;
    fv.fieldType = DCGM_FT_INT64;

    fv.value.i64 = 0;
    timeSeries.AddValue(1, fv);

    fv.value.i64 = 1;
    timeSeries.AddValue(1, fv);

    // DOUBLE
    fv.fieldId   = 2;
    fv.fieldType = DCGM_FT_DOUBLE;

    fv.value.dbl = 0.0;
    timeSeries.AddValue(2, fv);

    fv.value.dbl = 1.0;
    timeSeries.AddValue(2, fv);

    timeSeries.AddToJson(jv, jsonIndex);
    CHECK(jv[c_GPUS][jsonIndex]["gpuId"].asUInt() == 0);

    Json::Value &field1 = jv[c_GPUS][jsonIndex]["1"];
    Json::Value &field2 = jv[c_GPUS][jsonIndex]["2"];
    CHECK(field1[0][c_value].asInt() == 0);
    CHECK(field1[1][c_value].asInt() == 1);
    CHECK(field2[0][c_value].asDouble() == 0.0);
    CHECK(field2[1][c_value].asDouble() == 1.0);
}

SCENARIO("void DcgmValuesSinceHolder::AddValue(dcgm_field_entity_group_t entityGroupId, "
         "dcgm_field_eid_t entityId, unsigned short fieldId, dcgmFieldValue_v1 &val)")
{
    // Also bool DcgmValuesSinceHolder::IsStored(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
    //                                           unsigned short fieldId)
    DcgmValuesSinceHolder dvsh;
    dcgmFieldValue_v1 fv         = {};
    const unsigned short fieldId = 1;

    fv.fieldId   = fieldId;
    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = 1;

    CHECK(!dvsh.IsStored(DCGM_FE_GPU, 0, fieldId));

    dvsh.AddValue(DCGM_FE_GPU, 0, fieldId, fv);

    CHECK(dvsh.IsStored(DCGM_FE_GPU, 0, fieldId));
    CHECK(!dvsh.IsStored(DCGM_FE_SWITCH, 0, fieldId));
    CHECK(!dvsh.IsStored(DCGM_FE_GPU, 0, fieldId + 1));
}

SCENARIO("void DcgmValuesSinceHolder::GetFirstNonZero(dcgm_field_entity_group_t entityGroupId, "
         "dcgm_field_eid_t entityId, unsigned short fieldId, dcgmFieldValue_v1 &dfv, uint64_t mask)")
{
    // Keeping this simple as it mostly reuses the logic in DcgmEntityTimeSeries::AddValue for the most part
    DcgmValuesSinceHolder dvsh;
    dcgmFieldValue_v1 fv          = {};
    const dcgmFieldValue_v1 zeros = {};
    const unsigned short fieldId  = 1;

    // We expect this not to fetch anything
    dvsh.GetFirstNonZero(DCGM_FE_GPU, 0, fieldId, fv, 0);
    CHECK(0 == memcmp(&fv, &zeros, sizeof(fv)));

    fv.fieldId   = fieldId;
    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = 0;
    dvsh.AddValue(DCGM_FE_GPU, 0, fieldId, fv);

    fv.value.i64 = 1;
    dvsh.AddValue(DCGM_FE_GPU, 0, fieldId, fv);

    memset(&fv, 0, sizeof(fv));

    // We expect this to fetch a result
    dvsh.GetFirstNonZero(DCGM_FE_GPU, 0, fieldId, fv, 0);
    CHECK(fv.value.i64 == 1);
}

SCENARIO("void DcgmValuesSinceHolder::ClearEntries(dcgm_field_entity_group_t entityGroupId,"
         "dcgm_field_eid_t entityId, unsigned short fieldId)")
{
    // Also void DcgmValuesSinceHolder::ClearCache()
    DcgmValuesSinceHolder dvsh;
    dcgmFieldValue_v1 fv   = {};
    unsigned short fieldId = 1;

    fv.fieldId   = fieldId;
    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = 0;
    dvsh.AddValue(DCGM_FE_GPU, 0, fieldId, fv);
    CHECK(dvsh.IsStored(DCGM_FE_GPU, 0, fieldId));

    fieldId    = 2;
    fv.fieldId = fieldId;
    dvsh.AddValue(DCGM_FE_GPU, 0, fieldId, fv);
    CHECK(dvsh.IsStored(DCGM_FE_GPU, 0, fieldId));

    fieldId    = 3;
    fv.fieldId = fieldId;
    dvsh.AddValue(DCGM_FE_GPU, 0, fieldId, fv);
    CHECK(dvsh.IsStored(DCGM_FE_GPU, 0, fieldId));

    // Clear first field, not second or third
    dvsh.ClearEntries(DCGM_FE_GPU, 0, 1);
    // Clear uninitialized field--checking the code does not throw
    dvsh.ClearEntries(DCGM_FE_GPU, 0, 20);

    CHECK(!dvsh.IsStored(DCGM_FE_GPU, 0, 1));
    CHECK(dvsh.IsStored(DCGM_FE_GPU, 0, 2));
    CHECK(dvsh.IsStored(DCGM_FE_GPU, 0, 3));

    // Clear all fields
    dvsh.ClearCache();

    CHECK(!dvsh.IsStored(DCGM_FE_GPU, 0, 1));
    CHECK(!dvsh.IsStored(DCGM_FE_GPU, 0, 2));
    CHECK(!dvsh.IsStored(DCGM_FE_GPU, 0, 3));
}

SCENARIO("void DcgmValuesSinceHolder::AddToJson(Json::Value &jv)")
{
    Json::Value jv;
    DcgmValuesSinceHolder dvsh;
    dcgmFieldValue_v1 fv   = {};
    unsigned short fieldId = 1;
    unsigned int entityId  = 0;

    fv.fieldId   = fieldId;
    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = 0;
    dvsh.AddValue(DCGM_FE_GPU, entityId, fieldId, fv);

    // Change field id
    fieldId    = 2;
    fv.fieldId = fieldId;
    dvsh.AddValue(DCGM_FE_GPU, entityId, fieldId, fv);

    dvsh.AddToJson(jv);

    Json::Value &field1 = jv[c_GPUS][0]["1"];
    Json::Value &field2 = jv[c_GPUS][1]["2"];

    CHECK(field1[0][c_value].asInt() == 0);
    CHECK(field2[0][c_value].asInt() == 0);
}

SCENARIO("bool DcgmValuesSinceHolder::DoesValuePassPerSecondThreshold(unsigned short fieldId, "
         "const dcgmFieldValue_v1 &dfv, unsigned int gpuId, const char *fieldName, "
         "std::vector<DcgmError> &errorList, timelib64_t startTime)")
{
    std::vector<DcgmError> errorList;
    DcgmValuesSinceHolder dvsh;
    dcgmFieldValue_v1 fv        = {};
    dcgmFieldValue_v1 threshold = {};
    unsigned short fieldId      = 1;
    unsigned int entityId       = 0;

    // INT64

    threshold.fieldId   = fieldId;
    threshold.fieldType = DCGM_FT_INT64;
    threshold.value.i64 = 2;

    fv.fieldId   = fieldId;
    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = 0;
    dvsh.AddValue(DCGM_FE_GPU, entityId, fieldId, fv);

    fv.value.i64 = 1;
    dvsh.AddValue(DCGM_FE_GPU, entityId, fieldId, fv);

    CHECK(!dvsh.DoesValuePassPerSecondThreshold(fieldId, threshold, entityId, "field name", errorList, 0));

    fv.value.i64 = 3;
    dvsh.AddValue(DCGM_FE_GPU, entityId, fieldId, fv);
    CHECK(dvsh.DoesValuePassPerSecondThreshold(fieldId, threshold, entityId, "field name", errorList, 0));

    // DOUBLE

    fieldId = 2;

    threshold.fieldId   = fieldId;
    threshold.fieldType = DCGM_FT_DOUBLE;
    threshold.value.dbl = 2.0;

    fv.fieldId   = fieldId;
    fv.fieldType = DCGM_FT_DOUBLE;
    fv.value.dbl = 0.0;
    dvsh.AddValue(DCGM_FE_GPU, entityId, fieldId, fv);

    fv.value.dbl = 1.0;
    dvsh.AddValue(DCGM_FE_GPU, entityId, fieldId, fv);

    CHECK(!dvsh.DoesValuePassPerSecondThreshold(fieldId, threshold, entityId, "field name", errorList, 0));

    fv.value.dbl = 3.0;
    dvsh.AddValue(DCGM_FE_GPU, entityId, fieldId, fv);
    CHECK(dvsh.DoesValuePassPerSecondThreshold(fieldId, threshold, entityId, "field name", errorList, 0));
}
