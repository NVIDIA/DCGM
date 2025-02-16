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

#include "FieldIds.h"

#include <dcgm_fields.h>

#include "DcgmNscqManager.h"
#include "NvSwitchData.h"
#include "UpdateFunctions.h"

/**
 * Field Id Control definitions are in this file.
 */

namespace DcgmNs
{

/**
 * This macro simply allows us to return a reference to a static singleton
 * FieldIdControlType<fieldId> upcast to FieldIdControlType<DCGM_FI_UNKNOWN>.
 * It allows for mappings of fieldId to the "magic"
 * FieldIdControlType<fieldId> that can provide the NSCQ path and Update
 * function to use. It is intended to be included in every Field ID
 * specialised FieldIdControlType definition.
 */

#define SELF_REF                                                 \
    static const FieldIdControlType<DCGM_FI_UNKNOWN> &Self(void) \
    {                                                            \
        static FieldIdControlType<fieldId> self;                 \
        return self;                                             \
    }


/**
 * Here, we actually start defining the templated FieldIdInteralType classes
 * that identify the nscq data types, dcgm data types, and a static pointer
 * to the appriate update function.
 *
 * Where required we also incluide a FieldIdStorageType speclialization when
 * returned NSCQ data items are complex structures that require fieldId-specific
 * member extraction, or extraction of both a data item and a timestamp.
 *
 * Also, where required we defined dcgmFieldType classes when a new internal type
 * needs to be introduced.
 */
template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = int32_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_temperature_current;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = int32_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_temperature_limit_slowdown;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = int32_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_temperature_limit_shutdown;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = const nscq_link_throughput_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_throughput_counters;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_link_throughput_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_throughput_counters;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

/**
 * Here we define a storage type to extract the tx field of returned NSCQ
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.tx)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract the rx field of returned NSCQ
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.rx)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS;

public:
    using dcgmFieldType = NvSwitch::Data::ErrorData;
    using nscqFieldType = nscq_error_t; /* NSCQ give is a vector of these. */

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_error_fatal;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS;

public:
    using dcgmFieldType = NvSwitch::Data::ErrorData;
    using nscqFieldType = nscq_error_t; /* NSCQ passes us a vector of these. */

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_error_nonfatal;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

/**
 * Here we define a storage type to convert fatal error returned NSCQ
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to convert non-fatal error returned NSCQ
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_error_replay_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_error_recovery_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_error_flit_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS;

public:
    using nscqFieldType = uint64_t;
    using dcgmFieldType = NvSwitch::Data::Uint64Data;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_error_lane_crc_err_count_aggregate;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_error_lane_ecc_err_count_aggregate;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_link_throughput_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_port_throughput_counters;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_link_throughput_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_port_throughput_counters;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

/**
 * Here we define a storage type to extract the tx field from returned NSCQ
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.tx)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract the rx field from returned NSCQ
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.rx)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS;

public:
    using dcgmFieldType = NvSwitch::Data::ErrorData;
    using nscqFieldType = nscq_error_t; /* NSCQ passes us a vector of these. */

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_error_fatal;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkVectorFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS;

public:
    using dcgmFieldType = NvSwitch::Data::ErrorData;
    using nscqFieldType = nscq_error_t; /* NSCQ passes us a vector of these. */

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_error_nonfatal;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkVectorFieldType<fieldId>::updateFunc;
    }
};

/**
 * Here we define a storage type to convert returned NSCQ fatal error callback
 * data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS>::dcgmFieldType
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to convert returned NSCQ non-fatal error
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS>::dcgmFieldType
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_lane_crc_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_lane_crc_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_lane_crc_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_lane_crc_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_lane_ecc_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_lane_ecc_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_lane_ecc_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_lane_ecc_err_count;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_nvlink_voltage_info_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_voltage_info;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_nvlink_current_info_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_current_info;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_nvlink_current_info_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_current_info;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_VDD> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_POWER_VDD;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_nvswitch_power_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_power;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_DVDD> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_POWER_DVDD;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_nvswitch_power_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_power;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_HVDD> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_POWER_HVDD;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_nvswitch_power_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_power;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_nvlink_current_info_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_current_info;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3;

public:
    using dcgmFieldType = NvSwitch::Data::Uint64Data;
    using nscqFieldType = nscq_vc_latency_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_vc_latency;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLaneScalarFieldType<fieldId>::updateFunc;
    }
};

/**
 * Here we define the three storage type to extract returned NSCQ voltage callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.voltage_mvolt)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define 3 storage types to extract returned NSCQ current callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.iddq)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.iddq_rev)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.iddq_dvdd)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define 3 storage types to extract returned NSCQ power callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_VDD>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_VDD>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_POWER_VDD;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.vdd_w)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_DVDD>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_DVDD>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_POWER_DVDD;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.dvdd_w)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_HVDD>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_HVDD>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_POWER_HVDD;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.hvdd_w)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ low latency callback
 * data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.low)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ medium latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.medium)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ high latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.high)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ panic latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.panic)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ latency count
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.count)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ low latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.low)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ medium latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.medium)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ high latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.high)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ panic latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.panic)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ latency count
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.count)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ low latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.low)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ medium latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.medium)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ high latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.high)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ panic latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.panic)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ latency count
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.count)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ low latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.low)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ medium latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.medium)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ high latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.high)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ panic latency
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.panic)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a storage type to extract returned NSCQ latency count
 * callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.count)
    {}

    ~FieldIdStorageType() = default;
};

template <>
class FieldIdControlType<DCGM_FI_DEV_UUID> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_UUID;

public:
    using dcgmFieldType = NvSwitch::Data::UuidData;
    using nscqFieldType = nscq_uuid_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_uuid;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PHYS_ID> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PHYS_ID;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_phys_id;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_reset_required;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_ID> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_ID;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_nvlink_id;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_STATUS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_STATUS;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_link_status;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_TYPE> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_TYPE;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_remote_device_type;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};


template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = nscq_pcie_location_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_remote_device_pcie_location;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = nscq_pcie_location_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_remote_device_pcie_location;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = nscq_pcie_location_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_remote_device_pcie_location;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = nscq_pcie_location_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_remote_device_pcie_location;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_ID> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_ID;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_remote_device_link;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_SID> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_SID;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = uint64_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_remote_device_nvlink;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_DEVICE_UUID> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_DEVICE_UUID;

public:
    using dcgmFieldType = NvSwitch::Data::UuidData;
    using nscqFieldType = nscq_uuid_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_port_remote_device_uuid;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateLinkScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_BUS> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PCIE_BUS;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = nscq_pcie_location_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_pcie_location;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = nscq_pcie_location_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_pcie_location;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = nscq_pcie_location_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_pcie_location;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

template <>
class FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION> : public FieldIdControlType<DCGM_FI_UNKNOWN>
{
    static constexpr unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION;

public:
    using dcgmFieldType = NvSwitch::Data::Int64Data;
    using nscqFieldType = nscq_pcie_location_t;

    SELF_REF

    const char *NscqPath(void) const override
    {
        return nscq_nvswitch_pcie_location;
    }

    UpdateFuncType UpdateFunc(void) const override
    {
        return UpdateNvSwitchScalarFieldType<fieldId>::updateFunc;
    }
};

/**
 * Storage type to extract returned NSCQ pcie device callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.device)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Storage type to extract returned NSCQ pcie bus callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_PCIE_BUS>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_BUS>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PCIE_BUS;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.bus)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Storage type to extract returned NSCQ pcie domain callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.domain)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Storage type to extract returned NSCQ pcie function callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.function)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Storage type to extract returned NSCQ pcie bus callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.bus)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Storage type to extract returned NSCQ pcie device callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.device)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Storage type to extract returned NSCQ pcie domain callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.domain)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Storage type to extract returned NSCQ pcie function callback data.
 */
template <>
class FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION>
    : public FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION>::dcgmFieldType
{
    constexpr static unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION;

public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &in)
        : FieldIdControlType<fieldId>::dcgmFieldType(in.function)
    {}

    ~FieldIdStorageType() = default;
};
} // namespace DcgmNs