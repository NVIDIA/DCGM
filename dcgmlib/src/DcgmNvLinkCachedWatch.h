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
#pragma once

#include "DcgmCacheManager.h"

#include <DcgmLogging.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace DcgmNs::Cache
{

/**
 * Cached-watch request for one public NVLink bandwidth field.
 */

/**
 * Watch state captured when deciding whether a cached-watch NVLink result may still be applied.
 */
struct NvLinkCachedWatchSnapshot
{
    bool isWatched;      //!< Whether the field is still watched.
    uint64_t generation; //!< Current watch generation.
};

/**
 * Contiguous NVML field-values request range.
 */
struct NvmlFieldValueChunk
{
    std::size_t offset; //!< First field-value index in the chunk.
    std::size_t size;   //!< Number of field-values in the chunk.
};

struct NvLinkBandwidthReadRequest
{
    DcgmNvLinkBandwidthReadKey key;          //!< GPU and raw NVML scope to query.
    dcgm_field_entity_group_t entityGroupId; //!< Public entity group to update after the read.
    dcgm_field_eid_t entityId;               //!< Public entity ID to update after the read.
    unsigned short fieldId;                  //!< Public DCGM field ID to update after the read.
    DcgmcmDirection_t direction;             //!< Public bandwidth direction requested by `fieldId`.
    uint64_t watchGeneration;                //!< Watch generation captured before the unlocked backend call.
    timelib64_t maxAgeUsec;                  //!< Maximum cache age to apply to the public sample.
    timelib64_t lastQueriedUsec;             //!< Queried timestamp to preserve for legacy GPU fields.
    bool updateLastQueriedUsec;              //!< Whether cached-watch apply should update the watch query timestamp.
};

/**
 * Cached-watch request for one public NVLink BER ratio field.
 */
struct NvLinkBerReadRequest
{
    DcgmNvLinkBerReadKey key;                //!< GPU whose raw BER fields should be queried.
    dcgm_field_entity_group_t entityGroupId; //!< Public entity group to update after the read.
    dcgm_field_eid_t entityId;               //!< Public entity ID to update after the read.
    unsigned short fieldId;                  //!< Public DCGM BER ratio field ID to update after the read.
    bool isNvLink5OrNewer;                   //!< Whether aggregate NVLink5+ raw scope should be used.
    uint64_t watchGeneration;                //!< Watch generation captured before the unlocked backend call.
    timelib64_t maxAgeUsec;                  //!< Maximum cache age to apply to the public sample.
};

/**
 * Cached-watch request for one public NVLink link-state field.
 */
struct NvLinkStateReadRequest
{
    DcgmNvLinkStateReadKey key;              //!< GPU and raw NVML scope to query.
    dcgm_field_entity_group_t entityGroupId; //!< Public entity group to update after the read.
    dcgm_field_eid_t entityId;               //!< Public entity ID to update after the read.
    unsigned short fieldId;                  //!< Public DCGM link-state field ID to update after the read.
    uint64_t watchGeneration;                //!< Watch generation captured before the unlocked backend call.
};

/**
 * Cached-watch request for one public NVLink error-counter field.
 */
struct NvLinkErrorCounterReadRequest
{
    DcgmNvLinkErrorCounterReadKey key;       //!< GPU, raw NVML scope, and raw NVML field to query.
    dcgm_field_entity_group_t entityGroupId; //!< Public entity group to update after the read.
    dcgm_field_eid_t entityId;               //!< Public DCGM entity ID to update after the read.
    unsigned short fieldId;                  //!< Public DCGM error-counter field ID to update after the read.
    uint64_t watchGeneration;                //!< Watch generation captured before the unlocked backend call.
    timelib64_t maxAgeUsec;                  //!< Maximum cache age to apply to the public sample.
};


/**
 * Return whether a cached-watch NVLink result may still be applied to a watched field.
 *
 * @param[in] currentWatch Current watch state observed after the unlocked backend call.
 * @param[in] expectedGeneration Watch generation captured before the unlocked backend call.
 * @return True only when the field is still watched and the generation has not changed.
 */
[[nodiscard]] bool CanApplyNvLinkCachedWatchResult(NvLinkCachedWatchSnapshot currentWatch, uint64_t expectedGeneration);

/**
 * Return whether an NVML field-values error is an expected unsupported-path result.
 *
 * @param[in] nvmlReturn NVML status to classify.
 * @return True for unsupported or missing NVML field-values support, false otherwise.
 */
[[nodiscard]] bool IsExpectedNvLinkCachedWatchFieldValueError(nvmlReturn_t nvmlReturn);

/**
 * Split an NVML field-values request into chunks that do not exceed `NVML_FI_MAX`.
 *
 * @param[in] fieldValueCount Number of requested field-values.
 * @return Contiguous chunks covering each requested field-value exactly once.
 */
[[nodiscard]] std::vector<NvmlFieldValueChunk> BuildNvmlFieldValueChunks(std::size_t fieldValueCount);

/**
 * Create temporary NVML output storage with one zeroed guard record.
 *
 * @param[in] requestedFieldValues Requested field-value records to copy into temporary storage.
 * @return Requested records followed by a zeroed guard record.
 */
[[nodiscard]] std::vector<nvmlFieldValue_t> MakePaddedNvmlFieldValueBuffer(
    std::span<nvmlFieldValue_t const> requestedFieldValues);

/**
 * Return whether output storage beyond the requested field-value count changed.
 *
 * @param[in] fieldValues Temporary field-value storage containing requested records and guard storage.
 * @param[in] requestedFieldValueCount Number of requested field-value records.
 * @return True when guard storage changed or the requested count exceeds the storage size.
 */
[[nodiscard]] bool HasUnexpectedNvmlFieldValueWrite(std::span<nvmlFieldValue_t const> fieldValues,
                                                    std::size_t requestedFieldValueCount);

/**
 * Invoke an NVML field-values reader without allowing a one-record overrun to modify caller storage.
 *
 * @param[in,out] fieldValues Requested field-values and destination for accepted output.
 * @param[in] getFieldValues Callback that invokes the NVML reader.
 * @return Success after copying accepted output. An observed overflow logs at debug level and returns
 * `NVML_ERROR_UNKNOWN`, including when the callback also reports an error. A callback error without an observed
 * overflow leaves `fieldValues` unchanged while returning its respective NVML error. Empty or over-`NVML_FI_MAX`
 * input returns `NVML_ERROR_INVALID_ARGUMENT` and does not invoke `getFieldValues`.
 */
template <typename GetFieldValues>
[[nodiscard]] std::expected<void, nvmlReturn_t> ReadNvmlFieldValues(std::span<nvmlFieldValue_t> fieldValues,
                                                                    GetFieldValues &&getFieldValues)
{
    if (fieldValues.empty() || fieldValues.size() > NVML_FI_MAX)
    {
        return std::unexpected(NVML_ERROR_INVALID_ARGUMENT);
    }

    auto paddedFieldValues        = MakePaddedNvmlFieldValueBuffer(fieldValues);
    nvmlReturn_t const nvmlReturn = std::forward<GetFieldValues>(getFieldValues)(
        static_cast<unsigned int>(fieldValues.size()), paddedFieldValues.data());
    if (HasUnexpectedNvmlFieldValueWrite(paddedFieldValues, fieldValues.size()))
    {
        log_debug("nvmlDeviceGetFieldValues requested {} field-values; detected output beyond the requested count.",
                  fieldValues.size());
        return std::unexpected(NVML_ERROR_UNKNOWN);
    }

    if (nvmlReturn != NVML_SUCCESS)
    {
        return std::unexpected(nvmlReturn);
    }

    std::copy_n(paddedFieldValues.begin(), fieldValues.size(), fieldValues.begin());
    return {};
}

/**
 * Convert a public bandwidth direction to the raw RX/TX field mask.
 *
 * @param[in] direction Public DCGM bandwidth direction.
 * @return Bitmask of raw bandwidth directions to request, or 0 for unsupported input.
 */
[[nodiscard]] uint8_t NvLinkBandwidthDirectionMask(DcgmcmDirection_t direction);

/**
 * Convert a public BER ratio field ID to the raw BER field mask.
 *
 * @param[in] fieldId Public DCGM BER ratio field ID.
 * @return Bitmask of raw BER fields to request, or 0 when `fieldId` is not a supported BER ratio field.
 */
[[nodiscard]] uint8_t NvLinkBerFieldMask(unsigned short fieldId);

/**
 * Map a public BER ratio field ID to the raw NVML BER field ID.
 *
 * @param[in] fieldId Public DCGM BER ratio field ID.
 * @return Raw NVML field ID, or `std::nullopt` when `fieldId` is not a supported BER ratio field.
 */
[[nodiscard]] std::optional<unsigned short> GetNvLinkBerRawFieldId(unsigned short fieldId);

/**
 * Map a FE_LINK throughput field ID to its public bandwidth direction.
 *
 * @param[in] fieldId Public DCGM FE_LINK throughput field ID.
 * @return Public bandwidth direction, or `std::nullopt` when `fieldId` is not a supported FE_LINK throughput field.
 */
[[nodiscard]] std::optional<DcgmcmDirection_t> GetNvLinkPerLinkBandwidthDirection(unsigned short fieldId);

/**
 * Map a GPU aggregate throughput field ID to its public bandwidth direction.
 *
 * @param[in] fieldId Public DCGM aggregate throughput field ID.
 * @return Public bandwidth direction, or `std::nullopt` when `fieldId` is not a supported aggregate throughput field.
 */
[[nodiscard]] std::optional<DcgmcmDirection_t> GetNvLinkAggregateBandwidthDirection(unsigned short fieldId);

/**
 * Translate a legacy GPU NVLink throughput field to link index and direction.
 *
 * @param[in] fieldId DCGM legacy GPU NVLink throughput field ID.
 * @param[out] linkId Link index represented by `fieldId`.
 * @param[out] direction Public bandwidth direction represented by `fieldId`.
 * @return True when `fieldId` is a supported legacy GPU NVLink throughput field.
 */
[[nodiscard]] bool TryGetNvLinkBandwidthFieldInfo(unsigned int fieldId,
                                                  unsigned int &linkId,
                                                  DcgmcmDirection_t &direction);

/**
 * Return whether an entity/field pair is a cached-watch FE_LINK throughput candidate.
 *
 * @param[in] entityGroupId Public DCGM entity group.
 * @param[in] fieldId Public DCGM field ID.
 * @return True when the pair is a supported FE_LINK throughput field.
 */
[[nodiscard]] bool IsNvLinkPerLinkBandwidthField(dcgm_field_entity_group_t entityGroupId, unsigned short fieldId);

/**
 * Build a cached-watch BER request for a watched field when the field is supported.
 *
 * @param[in] watchInfo Current watch information captured while the watch table is locked.
 * @param[in] fieldId Public DCGM field ID.
 * @param[in] isNvLink5OrNewer Whether aggregate NVLink5+ raw scope should be used.
 * @return Cached-watch BER request, or `std::nullopt` when the watch should use the existing path.
 */
[[nodiscard]] std::optional<NvLinkBerReadRequest> TryGetNvLinkCachedWatchBerRead(dcgmcm_watch_info_t const &watchInfo,
                                                                                 unsigned short fieldId,
                                                                                 bool isNvLink5OrNewer);

/**
 * Map a supported public FE_LINK error-counter field to a raw NVML field-value ID.
 *
 * @param[in] fieldId Public DCGM FE_LINK error-counter field ID.
 * @return Raw NVML field-value ID, or `std::nullopt` when `fieldId` is not supported by the cached-watch path.
 */
[[nodiscard]] std::optional<unsigned int> GetNvLinkErrorCounterFieldValueId(unsigned short fieldId);

/**
 * Return whether an architecture is exactly Hopper.
 *
 * @param[in] arch DCGM chip architecture value.
 * @return True only for `DCGM_CHIP_ARCH_HOPPER`.
 */
[[nodiscard]] bool IsHopper(dcgmChipArchitecture_t arch);

/**
 * Build a cached-watch Hopper error-counter request for a watched field when supported.
 *
 * @param[in] watchInfo Current watch information captured while the watch table is locked.
 * @param[in] fieldId Public DCGM field ID.
 * @param[in] arch GPU architecture for the watched entity.
 * @return Cached-watch error-counter request, or `std::nullopt` when the watch should use the existing path.
 */
[[nodiscard]] std::optional<NvLinkErrorCounterReadRequest> TryGetNvLinkCachedWatchErrorCounterRead(
    dcgmcm_watch_info_t const &watchInfo,
    unsigned short fieldId,
    dcgmChipArchitecture_t arch);

/**
 * Build a cached-watch link-state request for a watched field when supported.
 *
 * @param[in] watchInfo Current watch information captured while the watch table is locked.
 * @param[in] fieldId Public DCGM field ID.
 * @return Cached-watch state request, or `std::nullopt` when the watch should use the existing path.
 */
[[nodiscard]] std::optional<NvLinkStateReadRequest> TryGetNvLinkCachedWatchStateRead(
    dcgmcm_watch_info_t const &watchInfo,
    unsigned short fieldId);

/**
 * Build a cached-watch bandwidth request for a watched field when supported.
 *
 * @param[in] watchInfo Current watch information captured while the watch table is locked.
 * @param[in] fieldId Public DCGM field ID.
 * @param[in] now Current timestamp used to preserve legacy GPU-field query timing.
 * @return Cached-watch bandwidth request, or `std::nullopt` when the watch should use the existing path.
 */
[[nodiscard]] std::optional<NvLinkBandwidthReadRequest> TryGetNvLinkCachedWatchBandwidthRead(
    dcgmcm_watch_info_t const &watchInfo,
    unsigned short fieldId,
    timelib64_t now);


/**
 * Public INT64 append decision for a cached-watch NVLink read result.
 */
struct NvLinkInt64SampleIntent
{
    long long value;             //!< Public sample value to append.
    long long rawValue;          //!< Secondary raw value to append.
    timelib64_t timestampUsec;   //!< Public sample timestamp.
    bool shouldSetLastStatus;    //!< Whether `lastStatus` should update watch state.
    nvmlReturn_t lastStatus;     //!< Watch status when `shouldSetLastStatus` is true.
    bool sawTopLevelUnsupported; //!< Whether the top-level backend status was `NVML_ERROR_NOT_SUPPORTED`.
};

/**
 * Previous raw bandwidth counter sample used to calculate the public rate.
 */
struct NvLinkBandwidthPreviousSample
{
    long long rawValue;        //!< Previous raw RX/TX counter sum.
    timelib64_t timestampUsec; //!< Previous raw counter timestamp.
};

/**
 * Reason a cached-watch bandwidth result should be logged by the caller.
 */
enum class NvLinkBandwidthLogReason
{
    None,          //!< No bandwidth-specific log is needed.
    TopLevelError, //!< The backend call failed before per-field values were valid.
    FieldError,    //!< A requested raw bandwidth field reported a per-field error.
    ZeroTimeDiff,  //!< The rate calculation could not use a zero time delta.
};

/**
 * Public append decision for a cached-watch NVLink bandwidth result.
 */
struct NvLinkBandwidthSampleIntent
{
    bool shouldAppend;                  //!< Whether a public sample should be appended.
    long long value;                    //!< Public bandwidth rate value to append.
    long long rawValue;                 //!< Raw RX/TX counter sum to append.
    timelib64_t timestampUsec;          //!< Public sample timestamp.
    bool shouldSetLastStatus;           //!< Whether `lastStatus` should update watch state.
    nvmlReturn_t lastStatus;            //!< Watch status when `shouldSetLastStatus` is true.
    NvLinkBandwidthLogReason logReason; //!< Optional caller log reason.
    nvmlReturn_t logNvmlReturn;         //!< NVML status associated with `logReason`.
    unsigned int logNvmlFieldId;        //!< Raw NVML field ID associated with `logReason`.
};

/**
 * Reason a cached-watch BER result should be logged by the caller.
 */
enum class NvLinkBerLogReason
{
    None,          //!< No BER-specific log is needed.
    TopLevelError, //!< The backend call failed before per-field values were valid.
    FieldError,    //!< A requested raw BER field reported a per-field error.
};

/**
 * Public append decision for a cached-watch NVLink BER result.
 */
struct NvLinkBerSampleIntent
{
    bool shouldAppend;            //!< Whether a public sample should be appended.
    double value;                 //!< Public BER value to append.
    double rawValue;              //!< Secondary raw value to append.
    timelib64_t timestampUsec;    //!< Public sample timestamp.
    bool shouldSetLastStatus;     //!< Whether `lastStatus` should update watch state.
    nvmlReturn_t lastStatus;      //!< Watch status when `shouldSetLastStatus` is true.
    NvLinkBerLogReason logReason; //!< Optional caller log reason.
    nvmlReturn_t logNvmlReturn;   //!< Top-level NVML status associated with `logReason`.
    nvmlReturn_t logFieldReturn;  //!< Per-field NVML status associated with `logReason`.
    unsigned int logNvmlFieldId;  //!< Raw NVML field ID associated with `logReason`.
};

/**
 * Translate a cached-watch NVLink bandwidth read result to a public sample or status-only update.
 *
 * @param[in] result Raw backend result for one GPU/scope key.
 * @param[in] direction Public bandwidth direction requested by the public field.
 * @param[in] previous Previous raw counter sample, or `std::nullopt` when no prior sample exists.
 * @param[in] readEnd Timestamp captured after the backend read.
 * @param[in] fieldId Public DCGM field ID that requested this sample.
 * @return Public bandwidth append/status intent.
 */
[[nodiscard]] NvLinkBandwidthSampleIntent TranslateNvLinkBandwidthResult(
    DcgmNvLinkBandwidthRawResult const &result,
    DcgmcmDirection_t direction,
    std::optional<NvLinkBandwidthPreviousSample> previous,
    timelib64_t readEnd,
    unsigned short fieldId);

/**
 * True when `fieldId` is a combined NVLink throughput field published in bytes per second.
 */
[[nodiscard]] bool IsNvLinkCombinedThroughputBytesPerSecondField(unsigned short fieldId) noexcept;

/**
 * Convert a combined NVLink throughput rate from KiB/s to the published unit for `fieldId`.
 *
 * Bytes-per-second fields use ×1024. Legacy MB/s fields use ÷1000.
 *
 * @param[in] kibPerSec Rate in kibibytes per second.
 * @param[in] fieldId Public combined-throughput field ID selecting the published unit.
 * @return Rate in the published unit for `fieldId`.
 */
[[nodiscard]] double ScaleNvLinkCombinedThroughputKiBPerSec(double kibPerSec, unsigned short fieldId) noexcept;

/**
 * Translate a cached-watch NVLink BER read result to a public double sample or status-only update.
 *
 * @param[in] result Raw backend result for one GPU key.
 * @param[in] fieldId Public DCGM BER ratio field ID requested by the public field.
 * @param[in] readEnd Timestamp captured after the backend read.
 * @return Public BER append/status intent.
 */
[[nodiscard]] NvLinkBerSampleIntent TranslateNvLinkBerResult(DcgmNvLinkBerRawResult const &result,
                                                             unsigned short fieldId,
                                                             timelib64_t readEnd);

/**
 * Translate a cached-watch NVLink state read result to a public INT64 sample.
 *
 * @param[in] result Raw backend result for one GPU/scope key.
 * @param[in] readStart Timestamp captured before the backend read.
 * @return Public sample intent.
 */
[[nodiscard]] NvLinkInt64SampleIntent TranslateNvLinkStateResult(DcgmNvLinkStateRawResult const &result,
                                                                 timelib64_t readStart);

/**
 * Translate a cached-watch Hopper NVLink error-counter read result to a public INT64 sample.
 *
 * @param[in] result Raw backend result for one GPU/scope/field key.
 * @param[in] readEnd Timestamp captured after the backend read.
 * @return Public sample intent.
 */
[[nodiscard]] NvLinkInt64SampleIntent TranslateNvLinkErrorCounterResult(DcgmNvLinkErrorCounterRawResult const &result,
                                                                        timelib64_t readEnd);

/**
 * Group bandwidth requests by raw GPU/scope key.
 *
 * @param[in] requests Cached-watch public bandwidth requests.
 * @return Raw read specs by GPU plus public fan-out metadata by raw key.
 */
[[nodiscard]] std::pair<std::map<unsigned int, std::vector<DcgmNvLinkBandwidthReadSpec>>,
                        std::map<DcgmNvLinkBandwidthReadKey, std::vector<NvLinkBandwidthReadRequest>>>
GroupNvLinkBandwidthReads(std::span<NvLinkBandwidthReadRequest const> requests);

/**
 * Group link-state requests by raw GPU/scope key.
 *
 * @param[in] requests Cached-watch public link-state requests.
 * @return Raw read specs by GPU plus public fan-out metadata by raw key.
 */
[[nodiscard]] std::pair<std::map<unsigned int, std::vector<DcgmNvLinkStateReadSpec>>,
                        std::map<DcgmNvLinkStateReadKey, std::vector<NvLinkStateReadRequest>>>
GroupNvLinkStateReads(std::span<NvLinkStateReadRequest const> requests);

/**
 * Group Hopper error-counter requests by raw GPU/scope/field key.
 *
 * @param[in] requests Cached-watch public error-counter requests.
 * @return Raw read specs by GPU plus public fan-out metadata by raw key.
 */
[[nodiscard]] std::pair<std::map<unsigned int, std::vector<DcgmNvLinkErrorCounterReadSpec>>,
                        std::map<DcgmNvLinkErrorCounterReadKey, std::vector<NvLinkErrorCounterReadRequest>>>
GroupNvLinkErrorCounterReads(std::span<NvLinkErrorCounterReadRequest const> requests);

/**
 * Group BER requests by raw GPU key.
 *
 * @param[in] requests Cached-watch public BER requests.
 * @return Raw field masks, NVLink version flags, and public fan-out metadata by raw key.
 */
[[nodiscard]] std::tuple<std::map<DcgmNvLinkBerReadKey, uint8_t>,
                         std::map<DcgmNvLinkBerReadKey, bool>,
                         std::map<DcgmNvLinkBerReadKey, std::vector<NvLinkBerReadRequest>>>
GroupNvLinkBerReads(std::span<NvLinkBerReadRequest const> requests);

} // namespace DcgmNs::Cache
