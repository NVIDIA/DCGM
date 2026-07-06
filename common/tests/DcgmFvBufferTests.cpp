/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmFvBuffer.h>
#include <Defer.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cstring>
#include <string>

namespace
{
size_t EntryCount(DcgmFvBuffer &buffer)
{
    size_t count {};
    REQUIRE(buffer.GetSize(nullptr, &count) == DCGM_ST_OK);
    return count;
}
} //namespace

TEST_CASE("DcgmFvBuffer handles size and input validation")
{
    GIVEN("an empty buffer")
    {
        DcgmFvBuffer buffer(0);

        SECTION("GetSize requires at least one output pointer")
        {
            CHECK(buffer.GetSize(nullptr, nullptr) == DCGM_ST_BADPARAM);
        }

        SECTION("SetFromBuffer rejects empty inputs")
        {
            CHECK(buffer.SetFromBuffer(nullptr, 8) == DCGM_ST_BADPARAM);
            CHECK(buffer.SetFromBuffer("payload", 0) == DCGM_ST_BADPARAM);
        }

        SECTION("GetNextFv returns null while empty")
        {
            dcgmBufferedFvCursor_t cursor {};
            CHECK(buffer.GetNextFv(&cursor) == nullptr);
        }

        SECTION("Clear resets entries but keeps the object reusable")
        {
            REQUIRE(buffer.AddInt64Value(DCGM_FE_GPU, 3, DCGM_FI_DEV_BOARD_POWER_WATTS, 123, 456, DCGM_ST_OK)
                    != nullptr);
            REQUIRE(EntryCount(buffer) == 1);

            buffer.Clear();

            CHECK(EntryCount(buffer) == 0);
            CHECK(buffer.AddDoubleValue(DCGM_FE_GPU, 3, DCGM_FI_DEV_GPU_UTIL_RATIO, 7.5, 789, DCGM_ST_OK) != nullptr);
            CHECK(EntryCount(buffer) == 1);
        }
    }
}

TEST_CASE("DcgmFvBuffer stores and converts field values")
{
    GIVEN("a buffer with several field types")
    {
        DcgmFvBuffer buffer(0, true);

        auto *intFv = buffer.AddInt64Value(DCGM_FE_GPU, 7, DCGM_FI_DEV_BOARD_POWER_WATTS, 42000, 1000, DCGM_ST_OK);
        REQUIRE(intFv != nullptr);

        auto *doubleFv = buffer.AddDoubleValue(DCGM_FE_GPU, 7, DCGM_FI_DEV_GPU_UTIL_RATIO, 87.5, 1001, DCGM_ST_OK);
        REQUIRE(doubleFv != nullptr);

        auto *stringFv = buffer.AddStringValue(DCGM_FE_GPU, 7, DCGM_FI_DEV_GPU_NAME, "unit-test-gpu", 1002, DCGM_ST_OK);
        REQUIRE(stringFv != nullptr);

        std::array<char, 4> blob { 'd', 'c', 'g', 'm' };
        auto *blobFv = buffer.AddBlobValue(
            DCGM_FE_GPU, 7, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, blob.data(), blob.size(), 1003, DCGM_ST_OK);
        REQUIRE(blobFv != nullptr);

        SECTION("entries can be walked with a cursor")
        {
            dcgmBufferedFvCursor_t cursor {};

            auto *first = buffer.GetNextFv(&cursor);
            REQUIRE(first != nullptr);
            CHECK(first->fieldType == DCGM_FT_INT64);
            CHECK(first->value.i64 == 42000);

            auto *second = buffer.GetNextFv(&cursor);
            REQUIRE(second != nullptr);
            CHECK(second->fieldType == DCGM_FT_DOUBLE);
            CHECK(second->value.dbl == 87.5);

            auto *third = buffer.GetNextFv(&cursor);
            REQUIRE(third != nullptr);
            CHECK(third->fieldType == DCGM_FT_STRING);
            CHECK(std::string(third->value.str) == "unit-test-gpu");

            auto *fourth = buffer.GetNextFv(&cursor);
            REQUIRE(fourth != nullptr);
            CHECK(fourth->fieldType == DCGM_FT_BINARY);
            CHECK(std::memcmp(fourth->value.blob, blob.data(), blob.size()) == 0);

            CHECK(buffer.GetNextFv(&cursor) == nullptr);
        }

        SECTION("buffered values convert to FV1 and FV2")
        {
            dcgmFieldValue_v1 fv1 {};
            DcgmFvBuffer::ConvertBufferedFvToFv1(doubleFv, &fv1);
            CHECK(fv1.version == dcgmFieldValue_version1);
            CHECK(fv1.fieldId == DCGM_FI_DEV_GPU_UTIL_RATIO);
            CHECK(fv1.value.dbl == 87.5);

            dcgmFieldValue_v2 fv2 {};
            DcgmFvBuffer::ConvertBufferedFvToFv2(intFv, &fv2);
            CHECK(fv2.version == dcgmFieldValue_version2);
            CHECK(fv2.entityGroupId == DCGM_FE_GPU);
            CHECK(fv2.entityId == 7);
            CHECK(fv2.value.i64 == 42000);
        }

        SECTION("GetAllAsFv1 copies available entries up to caller capacity")
        {
            std::array<dcgmFieldValue_v1, 3> values {};
            size_t stored {};

            REQUIRE(buffer.GetAllAsFv1(values.data(), values.size(), &stored) == DCGM_ST_OK);

            CHECK(stored == values.size());
            CHECK(values[0].value.i64 == 42000);
            CHECK(values[1].value.dbl == 87.5);
            CHECK(std::string(values[2].value.str) == "unit-test-gpu");
        }

        SECTION("SetFromBuffer copies and recounts serialized entries")
        {
            size_t bytes {};
            size_t elements {};
            REQUIRE(buffer.GetSize(&bytes, &elements) == DCGM_ST_OK);
            REQUIRE(elements == 4);

            DcgmFvBuffer copy(0);
            REQUIRE(copy.SetFromBuffer(buffer.GetBuffer(), bytes) == DCGM_ST_OK);

            CHECK(EntryCount(copy) == elements);
        }
    }
}

TEST_CASE("DcgmFvBuffer rejects malformed values")
{
    GIVEN("a mutable copy of one valid entry")
    {
        DcgmFvBuffer source(0);
        REQUIRE(source.AddInt64Value(DCGM_FE_GPU, 1, DCGM_FI_DEV_BOARD_POWER_WATTS, 1, 2, DCGM_ST_OK) != nullptr);

        size_t bytes {};
        REQUIRE(source.GetSize(&bytes, nullptr) == DCGM_ST_OK);
        std::string raw(source.GetBuffer(), bytes);

        SECTION("SetFromBuffer rejects bad versions")
        {
            auto *fv    = reinterpret_cast<dcgmBufferedFv_t *>(raw.data());
            fv->version = dcgmBufferedFv_version + 1;

            DcgmFvBuffer dest(0);
            CHECK(dest.SetFromBuffer(raw.data(), raw.size()) == DCGM_ST_GENERIC_ERROR);
        }

        SECTION("SetFromBuffer rejects entries that run past the buffer")
        {
            auto *fv   = reinterpret_cast<dcgmBufferedFv_t *>(raw.data());
            fv->length = static_cast<unsigned short>(raw.size() + 1);

            DcgmFvBuffer dest(0);
            CHECK(dest.SetFromBuffer(raw.data(), raw.size()) == DCGM_ST_GENERIC_ERROR);
        }

        SECTION("GetNextFv rejects corrupted internal entries")
        {
            DcgmFvBuffer dest(0);
            REQUIRE(dest.SetFromBuffer(source.GetBuffer(), bytes) == DCGM_ST_OK);

            auto *fv    = reinterpret_cast<dcgmBufferedFv_t *>(const_cast<char *>(dest.GetBuffer()));
            fv->version = dcgmBufferedFv_version + 1;

            dcgmBufferedFvCursor_t cursor {};
            CHECK(dest.GetNextFv(&cursor) == nullptr);
        }
    }

    GIVEN("direct add helpers")
    {
        DcgmFvBuffer buffer(0);

        SECTION("string values require nonempty data and bounded length")
        {
            CHECK(buffer.AddStringValue(DCGM_FE_GPU, 1, DCGM_FI_DEV_GPU_NAME, nullptr, 0, DCGM_ST_OK) == nullptr);
            CHECK(buffer.AddStringValue(DCGM_FE_GPU, 1, DCGM_FI_DEV_GPU_NAME, "", 0, DCGM_ST_OK) == nullptr);

            std::string tooLarge(DCGM_MAX_STR_LENGTH, 'x');
            CHECK(buffer.AddStringValue(DCGM_FE_GPU, 1, DCGM_FI_DEV_GPU_NAME, tooLarge.c_str(), 0, DCGM_ST_OK)
                  == nullptr);
        }

        SECTION("blob values require data and bounded length")
        {
            char oneByte           = 1;
            auto addAccountingBlob = [&buffer](void *value, size_t valueSize) {
                return buffer.AddBlobValue(
                    DCGM_FE_GPU, 1, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, value, valueSize, 0, DCGM_ST_OK);
            };

            CHECK(addAccountingBlob(nullptr, 1) == nullptr);
            CHECK(addAccountingBlob(&oneByte, 0) == nullptr);

            std::array<char, DCGM_MAX_BLOB_LENGTH + 1> tooLarge {};
            CHECK(addAccountingBlob(tooLarge.data(), tooLarge.size()) == nullptr);
        }

        SECTION("blank values use field metadata")
        {
            REQUIRE(DcgmFieldsInit() == 0);
            DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

            CHECK(buffer.AddBlankValue(DCGM_FE_GPU, 1, DCGM_FI_DEV_BOARD_POWER_WATTS, DCGM_ST_OK) != nullptr);
            CHECK(buffer.AddBlankValue(DCGM_FE_GPU, 1, DCGM_FI_DEV_GPU_UTIL_RATIO, DCGM_ST_OK) != nullptr);
            CHECK(buffer.AddBlankValue(DCGM_FE_GPU, 1, DCGM_FI_DEV_GPU_NAME, DCGM_ST_OK) != nullptr);
            CHECK(buffer.AddBlankValue(DCGM_FE_GPU, 1, DCGM_FI_SYSTEM_FIELD_UNKNOWN, DCGM_ST_OK) == nullptr);
        }
    }
}
