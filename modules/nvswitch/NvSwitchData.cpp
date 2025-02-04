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
#include "NvSwitchData.h"

#include <nscq.h>
#include <sstream>
#include <string>
#include <timelib.h>

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
 * Int64Data implementation.
 */
Int64Data::Int64Data(void)
    : value(DCGM_INT64_BLANK)
{}

Int64Data::Int64Data(int64_t value)
    : value(value)
{}

void Int64Data::BufferAdd(dcgm_field_entity_group_t entityGroupId,
                          dcgm_field_eid_t entityId,
                          unsigned short fieldId,
                          timelib64_t now,
                          DcgmFvBuffer &buf) const
{
    buf.AddInt64Value(entityGroupId, entityId, fieldId, value, now, DCGM_ST_OK);
}

std::string Int64Data::Str(void) const
{
    std::ostringstream os;

    os << value;

    return os.str();
}

/**
 * Uint64Data implementation.
 */
Uint64Data::Uint64Data(void)
    : value(DCGM_INT64_BLANK)
{}

Uint64Data::Uint64Data(uint64_t value)
    : value(value & 0x7fffffffffffffff)
{}

void Uint64Data::BufferAdd(dcgm_field_entity_group_t entityGroupId,
                           dcgm_field_eid_t entityId,
                           unsigned short fieldId,
                           timelib64_t now,
                           DcgmFvBuffer &buf) const
{
    buf.AddInt64Value(entityGroupId, entityId, fieldId, value, now, DCGM_ST_OK);
}

std::string Uint64Data::Str(void) const
{
    std::ostringstream os;

    os << value;

    return os.str();
}

/**
 * ErrorData implementation.
 */

ErrorData::ErrorData(void)
    : value(DCGM_INT64_BLANK)
    , time(0)
{}

ErrorData::ErrorData(const nscq_error_t &in)
{
    value = in.error_value;
    time  = (int64_t)in.time;
}

void ErrorData::BufferAdd(dcgm_field_entity_group_t entityGroupId,
                          dcgm_field_eid_t entityId,
                          unsigned short fieldId,
                          timelib64_t /* now */,
                          DcgmFvBuffer &buf) const
{
    buf.AddInt64Value(entityGroupId, entityId, fieldId, value, time, DCGM_ST_OK);
}

std::string ErrorData::Str(void) const
{
    std::ostringstream os;

    os << value << '@' << time;

    return os.str();
}

} // namespace DcgmNs::NvSwitch::Data
