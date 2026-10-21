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
#include "DcgmNvLinkCachedWatch.h"

#include "DcgmCMUtils.h"

#include <DcgmUtilities.h>

#include <dcgm_fields.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <tuple>

namespace DcgmNs::Cache
{
namespace
{

    struct NvLinkFieldRange
    {
        unsigned int firstFieldId; //!< First field ID in the contiguous range.
        unsigned int lastFieldId;  //!< Last field ID in the contiguous range.
        unsigned int firstLinkId;  //!< Link ID represented by `firstFieldId`.
    };

    template <std::size_t N>
    bool TryGetLinkIdFromRanges(unsigned int fieldId,
                                std::array<NvLinkFieldRange, N> const &ranges,
                                unsigned int &linkId)
    {
        for (auto const &range : ranges)
        {
            if (fieldId >= range.firstFieldId && fieldId <= range.lastFieldId)
            {
                linkId = range.firstLinkId + fieldId - range.firstFieldId;
                return true;
            }
        }

        return false;
    }

} // namespace


bool CanApplyNvLinkCachedWatchResult(NvLinkCachedWatchSnapshot currentWatch, uint64_t expectedGeneration)
{
    return currentWatch.isWatched && currentWatch.generation == expectedGeneration;
}

bool IsExpectedNvLinkCachedWatchFieldValueError(nvmlReturn_t nvmlReturn)
{
    return nvmlReturn == NVML_ERROR_NOT_SUPPORTED || nvmlReturn == NVML_ERROR_FUNCTION_NOT_FOUND;
}

std::vector<NvmlFieldValueChunk> BuildNvmlFieldValueChunks(std::size_t fieldValueCount)
{
    std::vector<NvmlFieldValueChunk> chunks;
    for (std::size_t offset = 0; offset < fieldValueCount;)
    {
        std::size_t const chunkSize = std::min(fieldValueCount - offset, static_cast<std::size_t>(NVML_FI_MAX));
        chunks.push_back({ .offset = offset, .size = chunkSize });
        offset += chunkSize;
    }
    return chunks;
}

std::vector<nvmlFieldValue_t> MakePaddedNvmlFieldValueBuffer(std::span<nvmlFieldValue_t const> requestedFieldValues)
{
    assert(requestedFieldValues.size() <= NVML_FI_MAX);
    std::vector<nvmlFieldValue_t> fieldValues(requestedFieldValues.size() + 1);
    assert(fieldValues.size() == requestedFieldValues.size() + 1);
    std::copy(requestedFieldValues.begin(), requestedFieldValues.end(), fieldValues.begin());
    return fieldValues;
}

bool HasUnexpectedNvmlFieldValueWrite(std::span<nvmlFieldValue_t const> fieldValues,
                                      std::size_t requestedFieldValueCount)
{
    if (requestedFieldValueCount > fieldValues.size())
    {
        return true;
    }

    for (auto const &fieldValue : fieldValues.subspan(requestedFieldValueCount))
    {
        auto const *bytes = reinterpret_cast<unsigned char const *>(&fieldValue);
        if (std::any_of(bytes, bytes + sizeof(fieldValue), [](unsigned char byte) { return byte != 0; }))
        {
            return true;
        }
    }

    return false;
}

uint8_t NvLinkBandwidthDirectionMask(DcgmcmDirection_t direction)
{
    switch (direction)
    {
        case DcgmcmDirectionAll:
            return DcgmNvLinkBandwidthDirectionRx | DcgmNvLinkBandwidthDirectionTx;
        case DcgmcmDirectionRx:
            return DcgmNvLinkBandwidthDirectionRx;
        case DcgmcmDirectionTx:
            return DcgmNvLinkBandwidthDirectionTx;
        default:
            return 0;
    }
}

uint8_t NvLinkBerFieldMask(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO:
            return DcgmNvLinkBerFieldSymbol;
        case DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO:
            return DcgmNvLinkBerFieldEffective;
        default:
            return 0;
    }
}

std::optional<unsigned short> GetNvLinkBerRawFieldId(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO:
            return NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER;
        case DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO:
            return NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER;
        default:
            return std::nullopt;
    }
}

std::optional<DcgmcmDirection_t> GetNvLinkPerLinkBandwidthDirection(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_THROUGHPUT_PER_LINK:
            return DcgmcmDirectionAll;
        case DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_PER_LINK:
            return DcgmcmDirectionTx;
        case DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_PER_LINK:
            return DcgmcmDirectionRx;
        default:
            return std::nullopt;
    }
}

std::optional<DcgmcmDirection_t> GetNvLinkAggregateBandwidthDirection(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL:
        case DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND:
            return DcgmcmDirectionAll;
        case DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_TOTAL:
            return DcgmcmDirectionTx;
        case DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL:
            return DcgmcmDirectionRx;
        default:
            return std::nullopt;
    }
}

bool IsNvLinkCombinedThroughputBytesPerSecondField(unsigned short fieldId) noexcept
{
    return DCGM_FI_DEV_NVLINK_THROUGHPUT_L0_BYTES_PER_SECOND <= fieldId
           && fieldId <= DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND;
}

double ScaleNvLinkCombinedThroughputKiBPerSec(double kibPerSec, unsigned short fieldId) noexcept
{
    if (IsNvLinkCombinedThroughputBytesPerSecondField(fieldId))
    {
        return kibPerSec * 1024.0; /* KiB/sec -> bytes/sec */
    }

    return kibPerSec / 1000.0; /* Convert to MB/sec (legacy published unit) */
}

bool TryGetNvLinkBandwidthFieldInfo(unsigned int fieldId, unsigned int &linkId, DcgmcmDirection_t &direction)
{
    static constexpr std::array<NvLinkFieldRange, 5> allRanges = { {
        { DCGM_FI_DEV_NVLINK_THROUGHPUT_L0, DCGM_FI_DEV_NVLINK_THROUGHPUT_L5, 0 },
        { DCGM_FI_DEV_NVLINK_THROUGHPUT_L6, DCGM_FI_DEV_NVLINK_THROUGHPUT_L11, 6 },
        { DCGM_FI_DEV_NVLINK_THROUGHPUT_L12, DCGM_FI_DEV_NVLINK_THROUGHPUT_L14, 12 },
        { DCGM_FI_DEV_NVLINK_THROUGHPUT_L15, DCGM_FI_DEV_NVLINK_THROUGHPUT_L17, 15 },
        { DCGM_FI_DEV_NVLINK_THROUGHPUT_L0_BYTES_PER_SECOND, DCGM_FI_DEV_NVLINK_THROUGHPUT_L17_BYTES_PER_SECOND, 0 },
    } };
    static constexpr std::array<NvLinkFieldRange, 1> txRanges  = { {
        { DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L0, DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L17, 0 },
    } };
    static constexpr std::array<NvLinkFieldRange, 1> rxRanges  = { {
        { DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L0, DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L17, 0 },
    } };

    if (TryGetLinkIdFromRanges(fieldId, allRanges, linkId))
    {
        direction = DcgmcmDirectionAll;
        return true;
    }

    if (TryGetLinkIdFromRanges(fieldId, txRanges, linkId))
    {
        direction = DcgmcmDirectionTx;
        return true;
    }

    if (TryGetLinkIdFromRanges(fieldId, rxRanges, linkId))
    {
        direction = DcgmcmDirectionRx;
        return true;
    }

    return false;
}

bool IsNvLinkPerLinkBandwidthField(dcgm_field_entity_group_t entityGroupId, unsigned short fieldId)
{
    return entityGroupId == DCGM_FE_LINK && GetNvLinkPerLinkBandwidthDirection(fieldId).has_value();
}

std::optional<NvLinkBerReadRequest> TryGetNvLinkCachedWatchBerRead(dcgmcm_watch_info_t const &watchInfo,
                                                                   unsigned short fieldId,
                                                                   bool isNvLink5OrNewer)
{
    if (watchInfo.practicalEntityGroupId != DCGM_FE_GPU)
    {
        return std::nullopt;
    }

    if (NvLinkBerFieldMask(fieldId) == 0)
    {
        return std::nullopt;
    }

    return NvLinkBerReadRequest { .key              = { .gpuId = watchInfo.practicalEntityId },
                                  .entityGroupId    = watchInfo.practicalEntityGroupId,
                                  .entityId         = watchInfo.practicalEntityId,
                                  .fieldId          = fieldId,
                                  .isNvLink5OrNewer = isNvLink5OrNewer,
                                  .watchGeneration  = watchInfo.watchGeneration,
                                  .maxAgeUsec       = watchInfo.maxAgeUsec };
}

std::optional<unsigned int> GetNvLinkErrorCounterFieldValueId(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_PER_LINK_TOTAL:
            return NVML_FI_DEV_NVLINK_ERROR_DL_REPLAY;
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_PER_LINK_TOTAL:
            return NVML_FI_DEV_NVLINK_ERROR_DL_RECOVERY;
        default:
            return std::nullopt;
    }
}

bool IsHopper(dcgmChipArchitecture_t arch)
{
    return arch == DCGM_CHIP_ARCH_HOPPER;
}

std::optional<NvLinkErrorCounterReadRequest> TryGetNvLinkCachedWatchErrorCounterRead(
    dcgmcm_watch_info_t const &watchInfo,
    unsigned short fieldId,
    dcgmChipArchitecture_t arch)
{
    if (watchInfo.practicalEntityGroupId != DCGM_FE_LINK || !IsHopper(arch))
    {
        return std::nullopt;
    }

    auto const nvmlFieldId = GetNvLinkErrorCounterFieldValueId(fieldId);
    if (!nvmlFieldId)
    {
        return std::nullopt;
    }

    dcgm_link_t const link { .raw = watchInfo.watchKey.entityId };
    if (link.parsed.type != DCGM_FE_GPU)
    {
        return std::nullopt;
    }

    return NvLinkErrorCounterReadRequest {
        .key             = { .gpuId = link.parsed.gpuId, .scopeId = link.parsed.index, .nvmlFieldId = *nvmlFieldId },
        .entityGroupId   = watchInfo.practicalEntityGroupId,
        .entityId        = watchInfo.practicalEntityId,
        .fieldId         = fieldId,
        .watchGeneration = watchInfo.watchGeneration,
        .maxAgeUsec      = watchInfo.maxAgeUsec
    };
}

std::optional<NvLinkStateReadRequest> TryGetNvLinkCachedWatchStateRead(dcgmcm_watch_info_t const &watchInfo,
                                                                       unsigned short fieldId)
{
    if (watchInfo.practicalEntityGroupId != DCGM_FE_LINK || fieldId != DCGM_FI_DEV_NVLINK_LINK_STATUS)
    {
        return std::nullopt;
    }

    dcgm_link_t const link { .raw = watchInfo.watchKey.entityId };
    if (link.parsed.type != DCGM_FE_GPU)
    {
        return std::nullopt;
    }

    return NvLinkStateReadRequest { .key             = { .gpuId = link.parsed.gpuId, .scopeId = link.parsed.index },
                                    .entityGroupId   = watchInfo.practicalEntityGroupId,
                                    .entityId        = watchInfo.practicalEntityId,
                                    .fieldId         = fieldId,
                                    .watchGeneration = watchInfo.watchGeneration };
}

std::optional<NvLinkBandwidthReadRequest> TryGetNvLinkCachedWatchBandwidthRead(dcgmcm_watch_info_t const &watchInfo,
                                                                               unsigned short fieldId,
                                                                               timelib64_t now)
{
    if (watchInfo.practicalEntityGroupId == DCGM_FE_LINK)
    {
        auto direction = GetNvLinkPerLinkBandwidthDirection(fieldId);
        if (!direction)
        {
            return std::nullopt;
        }

        dcgm_link_t const link { .raw = watchInfo.watchKey.entityId };
        if (link.parsed.type != DCGM_FE_GPU)
        {
            return std::nullopt;
        }

        return NvLinkBandwidthReadRequest { .key = { .gpuId = link.parsed.gpuId, .scopeId = link.parsed.index },
                                            .entityGroupId         = watchInfo.practicalEntityGroupId,
                                            .entityId              = watchInfo.practicalEntityId,
                                            .fieldId               = fieldId,
                                            .direction             = *direction,
                                            .watchGeneration       = watchInfo.watchGeneration,
                                            .maxAgeUsec            = watchInfo.maxAgeUsec,
                                            .lastQueriedUsec       = 0,
                                            .updateLastQueriedUsec = true };
    }

    if (watchInfo.practicalEntityGroupId == DCGM_FE_GPU)
    {
        auto direction              = GetNvLinkAggregateBandwidthDirection(fieldId);
        unsigned int scopeId        = std::numeric_limits<unsigned>::max();
        bool const isAggregateField = direction.has_value();
        if (!direction)
        {
            DcgmcmDirection_t legacyDirection = DcgmcmDirectionAll;
            if (!TryGetNvLinkBandwidthFieldInfo(fieldId, scopeId, legacyDirection))
            {
                return std::nullopt;
            }
            direction = legacyDirection;
        }

        return NvLinkBandwidthReadRequest { .key = { .gpuId = watchInfo.practicalEntityId, .scopeId = scopeId },
                                            .entityGroupId         = watchInfo.practicalEntityGroupId,
                                            .entityId              = watchInfo.practicalEntityId,
                                            .fieldId               = fieldId,
                                            .direction             = *direction,
                                            .watchGeneration       = watchInfo.watchGeneration,
                                            .maxAgeUsec            = watchInfo.maxAgeUsec,
                                            .lastQueriedUsec       = isAggregateField ? 0 : now,
                                            .updateLastQueriedUsec = true };
    }

    return std::nullopt;
}


NvLinkBandwidthSampleIntent TranslateNvLinkBandwidthResult(DcgmNvLinkBandwidthRawResult const &result,
                                                           DcgmcmDirection_t direction,
                                                           std::optional<NvLinkBandwidthPreviousSample> previous,
                                                           timelib64_t readEnd,
                                                           unsigned short fieldId)
{
    if (!result.hasDeviceHandle)
    {
        return { .shouldAppend        = true,
                 .value               = DCGM_INT64_BLANK,
                 .rawValue            = 0,
                 .timestampUsec       = readEnd,
                 .shouldSetLastStatus = false,
                 .lastStatus          = NVML_SUCCESS,
                 .logReason           = NvLinkBandwidthLogReason::None,
                 .logNvmlReturn       = NVML_SUCCESS,
                 .logNvmlFieldId      = 0 };
    }

    if (result.nvmlReturn != NVML_SUCCESS)
    {
        return { .shouldAppend        = false,
                 .value               = 0,
                 .rawValue            = 0,
                 .timestampUsec       = readEnd,
                 .shouldSetLastStatus = true,
                 .lastStatus          = result.nvmlReturn,
                 .logReason           = NvLinkBandwidthLogReason::TopLevelError,
                 .logNvmlReturn       = result.nvmlReturn,
                 .logNvmlFieldId      = 0 };
    }

    auto const directionMask = NvLinkBandwidthDirectionMask(direction);
    if ((directionMask & DcgmNvLinkBandwidthDirectionRx) && result.rx.nvmlReturn != NVML_SUCCESS)
    {
        return { .shouldAppend        = false,
                 .value               = 0,
                 .rawValue            = 0,
                 .timestampUsec       = readEnd,
                 .shouldSetLastStatus = true,
                 .lastStatus          = result.rx.nvmlReturn,
                 .logReason           = NvLinkBandwidthLogReason::FieldError,
                 .logNvmlReturn       = result.rx.nvmlReturn,
                 .logNvmlFieldId      = result.rx.fieldId };
    }
    if ((directionMask & DcgmNvLinkBandwidthDirectionTx) && result.tx.nvmlReturn != NVML_SUCCESS)
    {
        return { .shouldAppend        = false,
                 .value               = 0,
                 .rawValue            = 0,
                 .timestampUsec       = readEnd,
                 .shouldSetLastStatus = true,
                 .lastStatus          = result.tx.nvmlReturn,
                 .logReason           = NvLinkBandwidthLogReason::FieldError,
                 .logNvmlReturn       = result.tx.nvmlReturn,
                 .logNvmlFieldId      = result.tx.fieldId };
    }

    long long currentSum       = 0;
    timelib64_t fieldTimestamp = 0;
    if (directionMask & DcgmNvLinkBandwidthDirectionRx)
    {
        currentSum += static_cast<long long>(result.rx.value.ullVal);
        fieldTimestamp = std::max(fieldTimestamp, static_cast<timelib64_t>(result.rx.timestamp));
    }
    if (directionMask & DcgmNvLinkBandwidthDirectionTx)
    {
        currentSum += static_cast<long long>(result.tx.value.ullVal);
        fieldTimestamp = std::max(fieldTimestamp, static_cast<timelib64_t>(result.tx.timestamp));
    }
    if (fieldTimestamp == 0)
    {
        fieldTimestamp = readEnd;
    }

    if (currentSum == 0 || !previous.has_value())
    {
        return { .shouldAppend        = true,
                 .value               = 0,
                 .rawValue            = currentSum,
                 .timestampUsec       = fieldTimestamp,
                 .shouldSetLastStatus = true,
                 .lastStatus          = NVML_SUCCESS,
                 .logReason           = NvLinkBandwidthLogReason::None,
                 .logNvmlReturn       = NVML_SUCCESS,
                 .logNvmlFieldId      = 0 };
    }

    double const timeDiffSec = (fieldTimestamp - previous->timestampUsec) / 1000000.0;
    if (timeDiffSec == 0.0)
    {
        return { .shouldAppend        = true,
                 .value               = 0,
                 .rawValue            = currentSum,
                 .timestampUsec       = fieldTimestamp,
                 .shouldSetLastStatus = true,
                 .lastStatus          = NVML_SUCCESS,
                 .logReason           = NvLinkBandwidthLogReason::ZeroTimeDiff,
                 .logNvmlReturn       = NVML_SUCCESS,
                 .logNvmlFieldId      = 0 };
    }

    double valueDbl = static_cast<double>(currentSum - previous->rawValue);
    valueDbl /= timeDiffSec; /* Convert to KiB/sec */
    valueDbl = ScaleNvLinkCombinedThroughputKiBPerSec(valueDbl, fieldId);

    return { .shouldAppend        = true,
             .value               = static_cast<long long>(valueDbl),
             .rawValue            = currentSum,
             .timestampUsec       = fieldTimestamp,
             .shouldSetLastStatus = true,
             .lastStatus          = NVML_SUCCESS,
             .logReason           = NvLinkBandwidthLogReason::None,
             .logNvmlReturn       = NVML_SUCCESS,
             .logNvmlFieldId      = 0 };
}

NvLinkBerSampleIntent TranslateNvLinkBerResult(DcgmNvLinkBerRawResult const &result,
                                               unsigned short fieldId,
                                               timelib64_t readEnd)
{
    if (!result.hasDeviceHandle)
    {
        return {
            .shouldAppend        = true,
            .value               = DCGM_FP64_BLANK,
            .rawValue            = 0,
            .timestampUsec       = readEnd,
            .shouldSetLastStatus = false,
            .lastStatus          = NVML_SUCCESS,
            .logReason           = NvLinkBerLogReason::None,
            .logNvmlReturn       = NVML_SUCCESS,
            .logFieldReturn      = NVML_SUCCESS,
            .logNvmlFieldId      = 0,
        };
    }

    auto const rawNvmlFieldId = GetNvLinkBerRawFieldId(fieldId);
    if (!rawNvmlFieldId)
    {
        return {
            .shouldAppend        = false,
            .value               = 0,
            .rawValue            = 0,
            .timestampUsec       = readEnd,
            .shouldSetLastStatus = false,
            .lastStatus          = NVML_SUCCESS,
            .logReason           = NvLinkBerLogReason::None,
            .logNvmlReturn       = NVML_SUCCESS,
            .logFieldReturn      = NVML_SUCCESS,
            .logNvmlFieldId      = 0,
        };
    }

    if (result.nvmlReturn != NVML_SUCCESS)
    {
        return {
            .shouldAppend        = true,
            .value               = NvmlErrorToDoubleValue(result.nvmlReturn),
            .rawValue            = 0,
            .timestampUsec       = readEnd,
            .shouldSetLastStatus = true,
            .lastStatus          = result.nvmlReturn,
            .logReason           = NvLinkBerLogReason::TopLevelError,
            .logNvmlReturn       = result.nvmlReturn,
            .logFieldReturn      = NVML_SUCCESS,
            .logNvmlFieldId      = *rawNvmlFieldId,
        };
    }

    nvmlFieldValue_t const *fieldValue = nullptr;
    if (fieldId == DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO)
    {
        fieldValue = &result.symbol;
    }
    else if (fieldId == DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO)
    {
        fieldValue = &result.effective;
    }

    if (fieldValue == nullptr)
    {
        return {
            .shouldAppend        = false,
            .value               = 0,
            .rawValue            = 0,
            .timestampUsec       = readEnd,
            .shouldSetLastStatus = true,
            .lastStatus          = NVML_SUCCESS,
            .logReason           = NvLinkBerLogReason::None,
            .logNvmlReturn       = NVML_SUCCESS,
            .logFieldReturn      = NVML_SUCCESS,
            .logNvmlFieldId      = *rawNvmlFieldId,
        };
    }

    if (fieldValue->nvmlReturn != NVML_SUCCESS)
    {
        return {
            .shouldAppend        = true,
            .value               = NvmlErrorToDoubleValue(fieldValue->nvmlReturn),
            .rawValue            = 0,
            .timestampUsec       = readEnd,
            .shouldSetLastStatus = true,
            .lastStatus          = fieldValue->nvmlReturn,
            .logReason           = NvLinkBerLogReason::FieldError,
            .logNvmlReturn       = result.nvmlReturn,
            .logFieldReturn      = fieldValue->nvmlReturn,
            .logNvmlFieldId      = *rawNvmlFieldId,
        };
    }

    auto const [mantissa, exponent, ber] = DcgmNs::Utils::NvmlBerParser(fieldValue->value.ullVal);
    (void)mantissa;
    (void)exponent;

    return {
        .shouldAppend        = true,
        .value               = ber,
        .rawValue            = 0,
        .timestampUsec       = readEnd,
        .shouldSetLastStatus = true,
        .lastStatus          = NVML_SUCCESS,
        .logReason           = NvLinkBerLogReason::None,
        .logNvmlReturn       = NVML_SUCCESS,
        .logFieldReturn      = NVML_SUCCESS,
        .logNvmlFieldId      = *rawNvmlFieldId,
    };
}

NvLinkInt64SampleIntent TranslateNvLinkStateResult(DcgmNvLinkStateRawResult const &result, timelib64_t readStart)
{
    if (!result.hasDeviceHandle)
    {
        return { .value                  = DCGM_INT64_BLANK,
                 .rawValue               = 0,
                 .timestampUsec          = readStart,
                 .shouldSetLastStatus    = false,
                 .lastStatus             = NVML_SUCCESS,
                 .sawTopLevelUnsupported = false };
    }
    if (result.nvmlReturn != NVML_SUCCESS)
    {
        return { .value                  = NvmlErrorToInt64Value(result.nvmlReturn),
                 .rawValue               = 0,
                 .timestampUsec          = readStart,
                 .shouldSetLastStatus    = false,
                 .lastStatus             = NVML_SUCCESS,
                 .sawTopLevelUnsupported = false };
    }
    if (result.state.nvmlReturn != NVML_SUCCESS)
    {
        return { .value                  = NvmlErrorToInt64Value(result.state.nvmlReturn),
                 .rawValue               = 0,
                 .timestampUsec          = readStart,
                 .shouldSetLastStatus    = false,
                 .lastStatus             = NVML_SUCCESS,
                 .sawTopLevelUnsupported = false };
    }
    return { .value                  = static_cast<long long>(result.state.value.uiVal),
             .rawValue               = 0,
             .timestampUsec          = readStart,
             .shouldSetLastStatus    = false,
             .lastStatus             = NVML_SUCCESS,
             .sawTopLevelUnsupported = false };
}

NvLinkInt64SampleIntent TranslateNvLinkErrorCounterResult(DcgmNvLinkErrorCounterRawResult const &result,
                                                          timelib64_t readEnd)
{
    if (!result.hasDeviceHandle)
    {
        return { .value                  = DCGM_INT64_BLANK,
                 .rawValue               = 0,
                 .timestampUsec          = readEnd,
                 .shouldSetLastStatus    = false,
                 .lastStatus             = NVML_SUCCESS,
                 .sawTopLevelUnsupported = false };
    }
    if (result.nvmlReturn != NVML_SUCCESS)
    {
        return { .value                  = NvmlErrorToInt64Value(result.nvmlReturn),
                 .rawValue               = 0,
                 .timestampUsec          = readEnd,
                 .shouldSetLastStatus    = true,
                 .lastStatus             = result.nvmlReturn,
                 .sawTopLevelUnsupported = result.nvmlReturn == NVML_ERROR_NOT_SUPPORTED };
    }
    if (result.value.nvmlReturn != NVML_SUCCESS)
    {
        return { .value                  = NvmlErrorToInt64Value(result.value.nvmlReturn),
                 .rawValue               = 0,
                 .timestampUsec          = readEnd,
                 .shouldSetLastStatus    = true,
                 .lastStatus             = result.value.nvmlReturn,
                 .sawTopLevelUnsupported = false };
    }
    return { .value                  = static_cast<long long>(result.value.value.ullVal),
             .rawValue               = 0,
             .timestampUsec          = readEnd,
             .shouldSetLastStatus    = true,
             .lastStatus             = NVML_SUCCESS,
             .sawTopLevelUnsupported = false };
}

std::pair<std::map<unsigned int, std::vector<DcgmNvLinkBandwidthReadSpec>>,
          std::map<DcgmNvLinkBandwidthReadKey, std::vector<NvLinkBandwidthReadRequest>>>
GroupNvLinkBandwidthReads(std::span<NvLinkBandwidthReadRequest const> requests)
{
    std::map<DcgmNvLinkBandwidthReadKey, uint8_t> directionMasks;
    std::map<DcgmNvLinkBandwidthReadKey, std::vector<NvLinkBandwidthReadRequest>> requestsByKey;
    for (auto const &request : requests)
    {
        directionMasks[request.key] |= NvLinkBandwidthDirectionMask(request.direction);
        requestsByKey[request.key].push_back(request);
    }

    std::map<unsigned int, std::vector<DcgmNvLinkBandwidthReadSpec>> readSpecsByGpu;
    for (auto const &[key, directionMask] : directionMasks)
    {
        readSpecsByGpu[key.gpuId].push_back({ .key = key, .directionMask = directionMask });
    }

    return { readSpecsByGpu, requestsByKey };
}

std::pair<std::map<unsigned int, std::vector<DcgmNvLinkStateReadSpec>>,
          std::map<DcgmNvLinkStateReadKey, std::vector<NvLinkStateReadRequest>>>
GroupNvLinkStateReads(std::span<NvLinkStateReadRequest const> requests)
{
    std::map<DcgmNvLinkStateReadKey, std::vector<NvLinkStateReadRequest>> requestsByKey;
    for (auto const &request : requests)
    {
        requestsByKey[request.key].push_back(request);
    }

    std::map<unsigned int, std::vector<DcgmNvLinkStateReadSpec>> readSpecsByGpu;
    for (auto const &[key, keyRequests] : requestsByKey)
    {
        (void)keyRequests;
        readSpecsByGpu[key.gpuId].push_back({ .key = key });
    }

    return { readSpecsByGpu, requestsByKey };
}

std::pair<std::map<unsigned int, std::vector<DcgmNvLinkErrorCounterReadSpec>>,
          std::map<DcgmNvLinkErrorCounterReadKey, std::vector<NvLinkErrorCounterReadRequest>>>
GroupNvLinkErrorCounterReads(std::span<NvLinkErrorCounterReadRequest const> requests)
{
    std::map<DcgmNvLinkErrorCounterReadKey, std::vector<NvLinkErrorCounterReadRequest>> requestsByKey;
    for (auto const &request : requests)
    {
        requestsByKey[request.key].push_back(request);
    }

    std::map<unsigned int, std::vector<DcgmNvLinkErrorCounterReadSpec>> readSpecsByGpu;
    for (auto const &[key, keyRequests] : requestsByKey)
    {
        (void)keyRequests;
        readSpecsByGpu[key.gpuId].push_back({ .key = key });
    }

    return { readSpecsByGpu, requestsByKey };
}

std::tuple<std::map<DcgmNvLinkBerReadKey, uint8_t>,
           std::map<DcgmNvLinkBerReadKey, bool>,
           std::map<DcgmNvLinkBerReadKey, std::vector<NvLinkBerReadRequest>>>
GroupNvLinkBerReads(std::span<NvLinkBerReadRequest const> requests)
{
    std::map<DcgmNvLinkBerReadKey, uint8_t> fieldMasks;
    std::map<DcgmNvLinkBerReadKey, bool> isNvLink5OrNewer;
    std::map<DcgmNvLinkBerReadKey, std::vector<NvLinkBerReadRequest>> requestsByKey;
    for (auto const &request : requests)
    {
        fieldMasks[request.key] |= NvLinkBerFieldMask(request.fieldId);
        isNvLink5OrNewer[request.key] = request.isNvLink5OrNewer;
        requestsByKey[request.key].push_back(request);
    }

    return { fieldMasks, isNvLink5OrNewer, requestsByKey };
}

} // namespace DcgmNs::Cache
