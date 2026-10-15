/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <dcgm_fields.h>

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

namespace DcgmNs::Cache
{
/** State used to decide whether a deferred PRM-counter result remains current. */
struct DeferredPrmCounterWatchState
{
    bool isWatched;      //!< Whether the watch remains active.
    uint64_t generation; //!< Current generation of the watch.
};

/**
 * Determines whether a deferred PRM-counter result may update a watch.
 *
 * @param[in] currentState Current state of the watch under the cache-manager lock.
 * @param[in] snapshotGeneration Generation captured before the deferred read.
 * @return true when the watch remains active at the captured generation.
 */
[[nodiscard]] constexpr bool CanApplyDeferredPrmCounterResult(DeferredPrmCounterWatchState currentState,
                                                              uint64_t snapshotGeneration) noexcept
{
    return currentState.isWatched && currentState.generation == snapshotGeneration;
}

/**
 * DCGM field pairs that temporarily dual-publish one NVML PRM counter (same identity, different units).
 * When both are queued, the lower field ID is requested from NVML and the result is applied to both.
 */
constexpr std::array kPrmCounterDualPublishFieldPairs {
    std::pair { DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO,
                DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS },
};

/**
 * Return the dual-publish sibling field ID for a PRM counter field.
 *
 * @param[in] fieldId DCGM field ID to classify.
 * @return The paired field ID, or std::nullopt when fieldId is not in a dual-publish pair.
 */
[[nodiscard]] constexpr std::optional<unsigned short> TryGetDualPublishSiblingFieldId(unsigned short fieldId) noexcept
{
    for (auto const &[firstFieldId, secondFieldId] : kPrmCounterDualPublishFieldPairs)
    {
        if (fieldId == firstFieldId)
        {
            return secondFieldId;
        }
        if (fieldId == secondFieldId)
        {
            return firstFieldId;
        }
    }
    return std::nullopt;
}

/**
 * Return whether a dual-publish sibling should be omitted from an NVML PRM counter batch.
 *
 * The lower field ID is the NVML representative when both keys are present in queuedKeys.
 *
 * @param[in] queuedKeys Keys that will actually be requested or applied (must provide contains()).
 * @param[in] key Candidate batch key. Must have a copyable fieldId member.
 * @return true when key is the higher dual-publish sibling and the lower sibling is in queuedKeys.
 */
template <typename KeyContainer, typename Key>
[[nodiscard]] bool ShouldOmitDualPublishSiblingFromBatch(KeyContainer const &queuedKeys, Key const &key)
{
    auto const siblingFieldId = TryGetDualPublishSiblingFieldId(key.fieldId);
    if (!siblingFieldId || key.fieldId < *siblingFieldId)
    {
        return false;
    }

    Key siblingKey     = key;
    siblingKey.fieldId = *siblingFieldId;
    return queuedKeys.contains(siblingKey);
}
} // namespace DcgmNs::Cache
