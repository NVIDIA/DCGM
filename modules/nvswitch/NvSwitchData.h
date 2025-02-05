/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cstring>
#include <sstream>
#include <string>
#include <timelib.h>

#include <nscq.h>

#include <DcgmFvBuffer.h>
#include <dcgm_structs.h>

/**
 * Here we find data storage wrapper classes for NSCQ-provided
 * telemetry information from which we extract field data.
 *
 * Each class is responsible for having a BufferAdd function
 * to add data to a DcgmFvBuffer.
 */

namespace DcgmNs::NvSwitch::Data
{

/**
 * This holds int64_t data.
 */
struct Int64Data
{
    int64_t value;

    Int64Data(void);
    Int64Data(int64_t value);
    ~Int64Data() = default;

    void BufferAdd(dcgm_field_entity_group_t entityGroupId,
                   dcgm_field_eid_t entityId,
                   unsigned short fieldId,
                   timelib64_t now,
                   DcgmFvBuffer &buf) const;

    std::string Str(void) const;
};

/**
 * This holds uint64_t data.
 */
struct Uint64Data
{
    int64_t value; /* careful! We are storing uint64_t */

    Uint64Data(void);
    Uint64Data(uint64_t value);
    ~Uint64Data() = default;

    void BufferAdd(dcgm_field_entity_group_t entityGroupId,
                   dcgm_field_eid_t entityId,
                   unsigned short fieldId,
                   timelib64_t now,
                   DcgmFvBuffer &buf) const;

    std::string Str(void) const;
};

/**
 * This holds Switch error data. We get passed passed a vector of nscq_error_t
 * from NSCQ and each element has a value and time element. The time element
 * tags the error value in the buffer of error values ultimately produced.
 */
struct ErrorData
{
    int64_t value; /* it comes in as a uint_32 */
    timelib64_t time;

    ErrorData(void);
    ErrorData(const nscq_error_t &in);
    ~ErrorData() = default;

    void BufferAdd(dcgm_field_entity_group_t entityGroupId,
                   dcgm_field_eid_t entityId,
                   unsigned short fieldId,
                   timelib64_t now,
                   DcgmFvBuffer &buf) const;

    std::string Str(void) const;
};

/**
 * This holds uuid data.
 */
struct UuidData
{
    char value[sizeof(nscq_uuid_t().bytes) + 1];

    UuidData(void)
    {
        std::memset(value, 0, sizeof(value));
    }

    UuidData(nscq_uuid_t _value)
    {
        std::memcpy(value, _value.bytes, sizeof(value) - 1);
    }

    ~UuidData() = default;

    void BufferAdd(dcgm_field_entity_group_t entityGroupId,
                   dcgm_field_eid_t entityId,
                   unsigned short fieldId,
                   timelib64_t now,
                   DcgmFvBuffer &buf) const
    {
        buf.AddStringValue(entityGroupId, entityId, fieldId, value, now, DCGM_ST_OK);
    }

    std::string Str(void) const
    {
        std::ostringstream os;

        os << value;

        return os.str();
    }
};

} // namespace DcgmNs::NvSwitch::Data