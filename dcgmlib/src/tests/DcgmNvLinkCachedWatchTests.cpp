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

#include <DcgmCMUtils.h>
#include <DcgmNvLinkCachedWatch.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <limits>
#include <vector>

namespace
{

class DcgmNvLinkCachedWatchTestFixture
{
protected:
    static dcgm_field_eid_t LinkEntityId(unsigned int gpuId, unsigned int linkId)
    {
        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = gpuId;
        link.parsed.index = linkId;
        return link.raw;
    }

    static dcgmcm_watch_info_t GpuWatch(unsigned int gpuId)
    {
        dcgmcm_watch_info_t watchInfo {};
        watchInfo.watchKey.entityGroupId = DCGM_FE_GPU;
        watchInfo.watchKey.entityId      = gpuId;
        watchInfo.practicalEntityGroupId = DCGM_FE_GPU;
        watchInfo.practicalEntityId      = gpuId;
        watchInfo.watchGeneration        = 17;
        watchInfo.maxAgeUsec             = 900;
        return watchInfo;
    }

    static dcgmcm_watch_info_t LinkWatch(unsigned int gpuId, unsigned int linkId)
    {
        dcgmcm_watch_info_t watchInfo {};
        watchInfo.watchKey.entityGroupId = DCGM_FE_LINK;
        watchInfo.watchKey.entityId      = LinkEntityId(gpuId, linkId);
        watchInfo.practicalEntityGroupId = DCGM_FE_LINK;
        watchInfo.practicalEntityId      = watchInfo.watchKey.entityId;
        watchInfo.watchGeneration        = 23;
        watchInfo.maxAgeUsec             = 1200;
        return watchInfo;
    }

    static nvmlFieldValue_t BandwidthField(unsigned int fieldId,
                                           nvmlReturn_t nvmlReturn,
                                           unsigned long long value,
                                           unsigned long long timestamp)
    {
        nvmlFieldValue_t fieldValue {};
        fieldValue.fieldId      = fieldId;
        fieldValue.nvmlReturn   = nvmlReturn;
        fieldValue.value.ullVal = value;
        fieldValue.timestamp    = timestamp;
        return fieldValue;
    }

    static nvmlFieldValue_t BerField(unsigned int fieldId, nvmlReturn_t nvmlReturn, unsigned long long value)
    {
        nvmlFieldValue_t fieldValue {};
        fieldValue.fieldId      = fieldId;
        fieldValue.nvmlReturn   = nvmlReturn;
        fieldValue.valueType    = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
        fieldValue.value.ullVal = value;
        return fieldValue;
    }
};

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: classifies NVLink bandwidth watches")
{
    using namespace DcgmNs::Cache;

    GIVEN("a FE_LINK throughput watch")
    {
        auto watchInfo = LinkWatch(2, 7);

        WHEN("the public field is per-link TX throughput")
        {
            auto request
                = TryGetNvLinkCachedWatchBandwidthRead(watchInfo, DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_PER_LINK, 12345);

            THEN("the request identifies the raw GPU, link, and direction")
            {
                REQUIRE(request.has_value());
                CHECK(request->key.gpuId == 2);
                CHECK(request->key.scopeId == 7);
                CHECK(request->entityGroupId == DCGM_FE_LINK);
                CHECK(request->entityId == watchInfo.practicalEntityId);
                CHECK(request->fieldId == DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_PER_LINK);
                CHECK(request->direction == DcgmcmDirectionTx);
                CHECK(request->watchGeneration == 23);
                CHECK(request->maxAgeUsec == 1200);
                CHECK(request->lastQueriedUsec == 0);
                CHECK(request->updateLastQueriedUsec);
            }
        }
    }

    GIVEN("a GPU aggregate throughput watch")
    {
        auto watchInfo = GpuWatch(3);

        WHEN("the public field is aggregate RX throughput")
        {
            auto request
                = TryGetNvLinkCachedWatchBandwidthRead(watchInfo, DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL, 67890);

            THEN("the request uses the aggregate NVML scope")
            {
                REQUIRE(request.has_value());
                CHECK(request->key.gpuId == 3);
                CHECK(request->key.scopeId == std::numeric_limits<unsigned>::max());
                CHECK(request->direction == DcgmcmDirectionRx);
                CHECK(request->lastQueriedUsec == 0);
            }
        }
    }

    GIVEN("a legacy GPU throughput watch")
    {
        auto watchInfo = GpuWatch(4);

        WHEN("the public field is a legacy L7 TX field")
        {
            auto request = TryGetNvLinkCachedWatchBandwidthRead(watchInfo, DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L7, 111);

            THEN("the request preserves the legacy link index and query timestamp")
            {
                REQUIRE(request.has_value());
                CHECK(request->key.gpuId == 4);
                CHECK(request->key.scopeId == 7);
                CHECK(request->direction == DcgmcmDirectionTx);
                CHECK(request->lastQueriedUsec == 111);
            }
        }

        WHEN("the public field is combined lane L17 in bytes per second")
        {
            auto request = TryGetNvLinkCachedWatchBandwidthRead(
                watchInfo, DCGM_FI_DEV_NVLINK_THROUGHPUT_L17_BYTES_PER_SECOND, 112);

            THEN("the request maps link 17 with combined RX+TX direction")
            {
                REQUIRE(request.has_value());
                CHECK(request->key.gpuId == 4);
                CHECK(request->key.scopeId == 17);
                CHECK(request->direction == DcgmcmDirectionAll);
                CHECK(request->fieldId == DCGM_FI_DEV_NVLINK_THROUGHPUT_L17_BYTES_PER_SECOND);
                CHECK(request->lastQueriedUsec == 112);
            }
        }

        WHEN("the public field is combined all-lanes throughput in bytes per second")
        {
            auto request
                = TryGetNvLinkCachedWatchBandwidthRead(watchInfo, DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND, 113);

            THEN("the request uses the aggregate NVML scope with combined RX+TX direction")
            {
                REQUIRE(request.has_value());
                CHECK(request->key.gpuId == 4);
                CHECK(request->key.scopeId == std::numeric_limits<unsigned>::max());
                CHECK(request->direction == DcgmcmDirectionAll);
                CHECK(request->fieldId == DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND);
                CHECK(request->lastQueriedUsec == 0);
            }
        }
    }

    GIVEN("an unsupported field")
    {
        auto watchInfo = GpuWatch(5);

        WHEN("classification is attempted")
        {
            auto request = TryGetNvLinkCachedWatchBandwidthRead(watchInfo, DCGM_FI_DEV_GPU_NAME, 222);

            THEN("no cached-watch request is produced")
            {
                CHECK_FALSE(request.has_value());
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: classifies NVLink BER watches")
{
    using namespace DcgmNs::Cache;

    GIVEN("a GPU BER watch")
    {
        auto watchInfo = GpuWatch(1);

        WHEN("the public field is symbol BER")
        {
            auto request = TryGetNvLinkCachedWatchBerRead(watchInfo, DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO, true);

            THEN("the request identifies the GPU and raw BER field family")
            {
                REQUIRE(request.has_value());
                CHECK(request->key.gpuId == 1);
                CHECK(request->fieldId == DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO);
                CHECK(request->isNvLink5OrNewer);
                CHECK(NvLinkBerFieldMask(request->fieldId) == DcgmNvLinkBerFieldSymbol);
                REQUIRE(GetNvLinkBerRawFieldId(request->fieldId).has_value());
                CHECK(*GetNvLinkBerRawFieldId(request->fieldId) == NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER);
            }
        }
    }

    GIVEN("a FE_LINK watch")
    {
        auto watchInfo = LinkWatch(1, 2);

        WHEN("BER classification is attempted")
        {
            auto request = TryGetNvLinkCachedWatchBerRead(watchInfo, DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO, false);

            THEN("no cached-watch BER request is produced")
            {
                CHECK_FALSE(request.has_value());
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: classifies NVLink state watches")
{
    using namespace DcgmNs::Cache;

    GIVEN("a FE_LINK link-status watch")
    {
        auto watchInfo = LinkWatch(6, 4);

        WHEN("classification is attempted")
        {
            auto request = TryGetNvLinkCachedWatchStateRead(watchInfo, DCGM_FI_DEV_NVLINK_LINK_STATUS);

            THEN("the request identifies the raw GPU and link scope")
            {
                REQUIRE(request.has_value());
                CHECK(request->key.gpuId == 6);
                CHECK(request->key.scopeId == 4);
                CHECK(request->fieldId == DCGM_FI_DEV_NVLINK_LINK_STATUS);
                CHECK(request->watchGeneration == 23);
            }
        }
    }

    GIVEN("a GPU watch")
    {
        auto watchInfo = GpuWatch(6);

        WHEN("link-status classification is attempted")
        {
            auto request = TryGetNvLinkCachedWatchStateRead(watchInfo, DCGM_FI_DEV_NVLINK_LINK_STATUS);

            THEN("no cached-watch state request is produced")
            {
                CHECK_FALSE(request.has_value());
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: classifies Hopper error-counter watches")
{
    using namespace DcgmNs::Cache;

    GIVEN("a Hopper FE_LINK replay-error watch")
    {
        auto watchInfo = LinkWatch(8, 3);

        WHEN("classification is attempted")
        {
            auto request = TryGetNvLinkCachedWatchErrorCounterRead(
                watchInfo, DCGM_FI_DEV_NVLINK_REPLAY_ERROR_PER_LINK_TOTAL, DCGM_CHIP_ARCH_HOPPER);

            THEN("the request identifies the raw field-values counter")
            {
                REQUIRE(request.has_value());
                CHECK(request->key.gpuId == 8);
                CHECK(request->key.scopeId == 3);
                CHECK(request->key.nvmlFieldId == NVML_FI_DEV_NVLINK_ERROR_DL_REPLAY);
                CHECK(request->fieldId == DCGM_FI_DEV_NVLINK_REPLAY_ERROR_PER_LINK_TOTAL);
            }
        }
    }

    GIVEN("a non-Hopper FE_LINK replay-error watch")
    {
        auto watchInfo = LinkWatch(8, 3);

        WHEN("classification is attempted")
        {
            auto request = TryGetNvLinkCachedWatchErrorCounterRead(
                watchInfo, DCGM_FI_DEV_NVLINK_REPLAY_ERROR_PER_LINK_TOTAL, DCGM_CHIP_ARCH_BLACKWELL);

            THEN("no cached-watch field-values request is produced")
            {
                CHECK_FALSE(request.has_value());
            }
        }
    }

    GIVEN("a Hopper FE_LINK CRC split-field watch")
    {
        auto watchInfo = LinkWatch(8, 3);

        WHEN("classification is attempted")
        {
            auto request = TryGetNvLinkCachedWatchErrorCounterRead(
                watchInfo, DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_PER_LINK_TOTAL, DCGM_CHIP_ARCH_HOPPER);

            THEN("no cached-watch field-values request is produced")
            {
                CHECK_FALSE(request.has_value());
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: groups NVLink bandwidth requests")
{
    using namespace DcgmNs::Cache;

    GIVEN("watched RX and TX public fields for one raw link plus another link")
    {
        auto rxWatch    = LinkWatch(0, 1);
        auto txWatch    = LinkWatch(0, 1);
        auto otherWatch = LinkWatch(0, 2);
        auto rxRequest  = TryGetNvLinkCachedWatchBandwidthRead(rxWatch, DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_PER_LINK, 10);
        auto txRequest  = TryGetNvLinkCachedWatchBandwidthRead(txWatch, DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_PER_LINK, 10);
        auto otherRequest
            = TryGetNvLinkCachedWatchBandwidthRead(otherWatch, DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_PER_LINK, 10);
        REQUIRE(rxRequest.has_value());
        REQUIRE(txRequest.has_value());
        REQUIRE(otherRequest.has_value());
        std::vector<NvLinkBandwidthReadRequest> requests { *rxRequest, *txRequest, *otherRequest };

        WHEN("the requests are grouped")
        {
            auto [readSpecsByGpu, requestsByKey] = GroupNvLinkBandwidthReads(requests);

            THEN("siblings coalesce by raw key and distinct links remain separate")
            {
                REQUIRE(readSpecsByGpu.contains(0));
                REQUIRE(readSpecsByGpu.at(0).size() == 2);
                CHECK(readSpecsByGpu.at(0)[0].key.scopeId == 1);
                CHECK(readSpecsByGpu.at(0)[0].directionMask
                      == (DcgmNvLinkBandwidthDirectionRx | DcgmNvLinkBandwidthDirectionTx));
                CHECK(readSpecsByGpu.at(0)[1].key.scopeId == 2);
                CHECK(readSpecsByGpu.at(0)[1].directionMask == DcgmNvLinkBandwidthDirectionRx);

                DcgmNvLinkBandwidthReadKey const coalescedKey { .gpuId = 0, .scopeId = 1 };
                REQUIRE(requestsByKey.contains(coalescedKey));
                CHECK(requestsByKey.at(coalescedKey).size() == 2);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: groups BER requests")
{
    using namespace DcgmNs::Cache;

    GIVEN("watched symbol and effective BER fields for one GPU")
    {
        auto watchInfo        = GpuWatch(2);
        auto symbolRequest    = TryGetNvLinkCachedWatchBerRead(watchInfo, DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO, true);
        auto effectiveRequest = TryGetNvLinkCachedWatchBerRead(watchInfo, DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO, true);
        REQUIRE(symbolRequest.has_value());
        REQUIRE(effectiveRequest.has_value());
        std::vector<NvLinkBerReadRequest> requests { *symbolRequest, *effectiveRequest };

        WHEN("the requests are grouped")
        {
            auto [fieldMasks, isNvLink5OrNewer, requestsByKey] = GroupNvLinkBerReads(requests);

            THEN("one raw BER read carries both requested raw fields")
            {
                DcgmNvLinkBerReadKey const key { .gpuId = 2 };
                REQUIRE(fieldMasks.contains(key));
                CHECK(fieldMasks.at(key) == (DcgmNvLinkBerFieldSymbol | DcgmNvLinkBerFieldEffective));
                CHECK(isNvLink5OrNewer.at(key));
                REQUIRE(requestsByKey.contains(key));
                CHECK(requestsByKey.at(key).size() == 2);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: groups state and error-counter requests")
{
    using namespace DcgmNs::Cache;

    GIVEN("two state watches on different GPUs")
    {
        auto firstRequest  = TryGetNvLinkCachedWatchStateRead(LinkWatch(0, 1), DCGM_FI_DEV_NVLINK_LINK_STATUS);
        auto secondRequest = TryGetNvLinkCachedWatchStateRead(LinkWatch(1, 1), DCGM_FI_DEV_NVLINK_LINK_STATUS);
        REQUIRE(firstRequest.has_value());
        REQUIRE(secondRequest.has_value());
        std::vector<NvLinkStateReadRequest> requests { *firstRequest, *secondRequest };

        WHEN("the requests are grouped")
        {
            auto [readSpecsByGpu, requestsByKey] = GroupNvLinkStateReads(requests);

            THEN("raw reads are partitioned by GPU")
            {
                CHECK(readSpecsByGpu.size() == 2);
                CHECK(readSpecsByGpu.at(0).size() == 1);
                CHECK(readSpecsByGpu.at(1).size() == 1);
                CHECK(requestsByKey.size() == 2);
            }
        }
    }

    GIVEN("two Hopper error-counter public fields on one link")
    {
        auto watchInfo     = LinkWatch(3, 5);
        auto replayRequest = TryGetNvLinkCachedWatchErrorCounterRead(
            watchInfo, DCGM_FI_DEV_NVLINK_REPLAY_ERROR_PER_LINK_TOTAL, DCGM_CHIP_ARCH_HOPPER);
        auto recoveryRequest = TryGetNvLinkCachedWatchErrorCounterRead(
            watchInfo, DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_PER_LINK_TOTAL, DCGM_CHIP_ARCH_HOPPER);
        REQUIRE(replayRequest.has_value());
        REQUIRE(recoveryRequest.has_value());
        std::vector<NvLinkErrorCounterReadRequest> requests { *replayRequest, *recoveryRequest };

        WHEN("the requests are grouped")
        {
            auto [readSpecsByGpu, requestsByKey] = GroupNvLinkErrorCounterReads(requests);

            THEN("each raw NVML field remains a separate read spec")
            {
                REQUIRE(readSpecsByGpu.contains(3));
                REQUIRE(readSpecsByGpu.at(3).size() == 2);
                CHECK(readSpecsByGpu.at(3)[0].key.nvmlFieldId == NVML_FI_DEV_NVLINK_ERROR_DL_REPLAY);
                CHECK(readSpecsByGpu.at(3)[1].key.nvmlFieldId == NVML_FI_DEV_NVLINK_ERROR_DL_RECOVERY);
                CHECK(requestsByKey.size() == 2);
            }
        }
    }
}


SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: classifies expected NVML errors")
{
    using namespace DcgmNs::Cache;

    GIVEN("NVML reports an unsupported field-values path")
    {
        WHEN("the status is classified")
        {
            THEN("the error is expected for cached-watch fallback decisions")
            {
                CHECK(IsExpectedNvLinkCachedWatchFieldValueError(NVML_ERROR_NOT_SUPPORTED));
                CHECK(IsExpectedNvLinkCachedWatchFieldValueError(NVML_ERROR_FUNCTION_NOT_FOUND));
            }
        }
    }

    GIVEN("NVML reports another error")
    {
        WHEN("the status is classified")
        {
            THEN("the error remains unexpected")
            {
                CHECK_FALSE(IsExpectedNvLinkCachedWatchFieldValueError(NVML_ERROR_UNKNOWN));
                CHECK_FALSE(IsExpectedNvLinkCachedWatchFieldValueError(NVML_SUCCESS));
            }
        }
    }
}


SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: translates bandwidth read results")
{
    using namespace DcgmNs::Cache;

    GIVEN("no usable device handle")
    {
        DcgmNvLinkBandwidthRawResult result { .hasDeviceHandle = false };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionAll, std::nullopt, 900, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("a blank value is appended without changing watch status")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == DCGM_INT64_BLANK);
                CHECK(intent.rawValue == 0);
                CHECK(intent.timestampUsec == 900);
                CHECK_FALSE(intent.shouldSetLastStatus);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::None);
            }
        }
    }

    GIVEN("a top-level NVML error")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_ERROR_UNKNOWN };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionAll, std::nullopt, 901, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("no sample is appended and the watch status records the error")
            {
                CHECK_FALSE(intent.shouldAppend);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_ERROR_UNKNOWN);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::TopLevelError);
                CHECK(intent.logNvmlReturn == NVML_ERROR_UNKNOWN);
            }
        }
    }

    GIVEN("a requested RX field error")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.rx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, NVML_ERROR_NOT_SUPPORTED, 0, 0);
        result.tx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, NVML_SUCCESS, 7, 1000);

        WHEN("the result is translated for all bandwidth")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionAll, std::nullopt, 902, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("the RX field error is returned without appending a sample")
            {
                CHECK_FALSE(intent.shouldAppend);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_ERROR_NOT_SUPPORTED);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::FieldError);
                CHECK(intent.logNvmlFieldId == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX);
            }
        }
    }

    GIVEN("a requested TX field error")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.rx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, NVML_SUCCESS, 3, 1000);
        result.tx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, NVML_ERROR_UNKNOWN, 0, 0);

        WHEN("the result is translated for all bandwidth")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionAll, std::nullopt, 903, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("the TX field error is returned after RX succeeds")
            {
                CHECK_FALSE(intent.shouldAppend);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_ERROR_UNKNOWN);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::FieldError);
                CHECK(intent.logNvmlFieldId == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX);
            }
        }
    }

    GIVEN("a successful first sample")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.rx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, NVML_SUCCESS, 10, 1000);
        result.tx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, NVML_SUCCESS, 15, 1500);

        WHEN("the result is translated without a previous sample")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionAll, std::nullopt, 904, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("zero rate and the raw counter sum are appended")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == 0);
                CHECK(intent.rawValue == 25);
                CHECK(intent.timestampUsec == 1500);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_SUCCESS);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::None);
            }
        }
    }

    GIVEN("a successful zero counter sample")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.rx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, NVML_SUCCESS, 0, 2000);
        result.tx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, NVML_SUCCESS, 0, 3000);
        NvLinkBandwidthPreviousSample previous { .rawValue = 100, .timestampUsec = 1000 };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionAll, previous, 905, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("zero rate and the zero raw counter are appended")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == 0);
                CHECK(intent.rawValue == 0);
                CHECK(intent.timestampUsec == 3000);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::None);
            }
        }
    }

    GIVEN("a successful sample with the same timestamp as the previous sample")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.rx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, NVML_SUCCESS, 500, 4000);
        NvLinkBandwidthPreviousSample previous { .rawValue = 100, .timestampUsec = 4000 };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionRx, previous, 906, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("zero rate is appended and the zero-time-diff condition is reported")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == 0);
                CHECK(intent.rawValue == 500);
                CHECK(intent.timestampUsec == 4000);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_SUCCESS);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::ZeroTimeDiff);
            }
        }
    }

    GIVEN("a successful RX-only sample with a previous sample")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.rx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, NVML_SUCCESS, 3100, 3000000);
        result.tx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, NVML_ERROR_UNKNOWN, 999, 3000000);
        NvLinkBandwidthPreviousSample previous { .rawValue = 100, .timestampUsec = 1000000 };

        WHEN("the result is translated for RX")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionRx, previous, 907, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("only RX participates in the rate calculation")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == 1);
                CHECK(intent.rawValue == 3100);
                CHECK(intent.timestampUsec == 3000000);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::None);
            }
        }
    }

    GIVEN("a successful TX-only sample without raw timestamps")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.rx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, NVML_ERROR_UNKNOWN, 999, 0);
        result.tx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, NVML_SUCCESS, 4200, 0);
        NvLinkBandwidthPreviousSample previous { .rawValue = 200, .timestampUsec = 1000000 };

        WHEN("the result is translated for TX")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionTx, previous, 3000000, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL);

            THEN("only TX is used and readEnd becomes the timestamp")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == 2);
                CHECK(intent.rawValue == 4200);
                CHECK(intent.timestampUsec == 3000000);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::None);
            }
        }
    }

    GIVEN("a successful combined sample for a bytes-per-second field")
    {
        DcgmNvLinkBandwidthRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.rx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, NVML_SUCCESS, 1100, 3000000);
        result.tx = BandwidthField(NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, NVML_SUCCESS, 1100, 3000000);
        NvLinkBandwidthPreviousSample previous { .rawValue = 200, .timestampUsec = 1000000 };

        WHEN("the result is translated for combined all-lanes bytes per second")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionAll, previous, 908, DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND);

            THEN("KiB/s is converted directly to bytes per second")
            {
                // delta = 2000 KiB over 2 s -> 1000 KiB/s -> 1000 * 1024 bytes/s
                CHECK(intent.shouldAppend);
                CHECK(intent.value == 1000LL * 1024LL);
                CHECK(intent.rawValue == 2200);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::None);
            }
        }

        WHEN("the result is translated for combined lane L0 bytes per second")
        {
            auto intent = TranslateNvLinkBandwidthResult(
                result, DcgmcmDirectionAll, previous, 908, DCGM_FI_DEV_NVLINK_THROUGHPUT_L0_BYTES_PER_SECOND);

            THEN("KiB/s is converted directly to bytes per second")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == 1000LL * 1024LL);
                CHECK(intent.rawValue == 2200);
                CHECK(intent.logReason == NvLinkBandwidthLogReason::None);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture,
                "DcgmNvLinkCachedWatch: classifies combined throughput bytes-per-second fields")
{
    using namespace DcgmNs::Cache;

    GIVEN("field IDs in and around the combined throughput bytes-per-second range")
    {
        THEN("lane and all-lanes IDs in 1630-1648 are classified as bytes-per-second")
        {
            CHECK(IsNvLinkCombinedThroughputBytesPerSecondField(DCGM_FI_DEV_NVLINK_THROUGHPUT_L0_BYTES_PER_SECOND));
            CHECK(IsNvLinkCombinedThroughputBytesPerSecondField(DCGM_FI_DEV_NVLINK_THROUGHPUT_L17_BYTES_PER_SECOND));
            CHECK(IsNvLinkCombinedThroughputBytesPerSecondField(DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND));
        }

        THEN("legacy combined MB/s and unrelated fields are not classified as bytes-per-second")
        {
            CHECK_FALSE(IsNvLinkCombinedThroughputBytesPerSecondField(DCGM_FI_DEV_NVLINK_THROUGHPUT_L0));
            CHECK_FALSE(IsNvLinkCombinedThroughputBytesPerSecondField(DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL));
            CHECK_FALSE(IsNvLinkCombinedThroughputBytesPerSecondField(DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L0));
            CHECK_FALSE(IsNvLinkCombinedThroughputBytesPerSecondField(DCGM_FI_DEV_C2C_MAX_BANDWIDTH_BYTES_PER_SECOND));
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture,
                "DcgmNvLinkCachedWatch: scales combined throughput from KiB/s to published units")
{
    using namespace DcgmNs::Cache;

    STATIC_REQUIRE(noexcept(ScaleNvLinkCombinedThroughputKiBPerSec(0.0, 0)));

    GIVEN("KiB/s rates for bytes-per-second combined fields")
    {
        THEN("inclusive range bounds and an interior ID multiply by 1024")
        {
            CHECK(ScaleNvLinkCombinedThroughputKiBPerSec(0.0, DCGM_FI_DEV_NVLINK_THROUGHPUT_L0_BYTES_PER_SECOND)
                  == 0.0);
            CHECK(ScaleNvLinkCombinedThroughputKiBPerSec(1.0, DCGM_FI_DEV_NVLINK_THROUGHPUT_L0_BYTES_PER_SECOND)
                  == 1024.0);
            CHECK(ScaleNvLinkCombinedThroughputKiBPerSec(1000.0, DCGM_FI_DEV_NVLINK_THROUGHPUT_L8_BYTES_PER_SECOND)
                  == 1000.0 * 1024.0);
            CHECK(ScaleNvLinkCombinedThroughputKiBPerSec(1000.0, DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND)
                  == 1000.0 * 1024.0);
        }
    }

    GIVEN("KiB/s rates for legacy MB/s combined fields")
    {
        THEN("IDs outside the bytes-per-second range divide by 1000")
        {
            CHECK(ScaleNvLinkCombinedThroughputKiBPerSec(0.0, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL) == 0.0);
            CHECK(ScaleNvLinkCombinedThroughputKiBPerSec(1000.0, DCGM_FI_DEV_NVLINK_THROUGHPUT_L0) == 1.0);
            CHECK(ScaleNvLinkCombinedThroughputKiBPerSec(1000.0, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL) == 1.0);
            CHECK(ScaleNvLinkCombinedThroughputKiBPerSec(2500.0, DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL) == 2.5);
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: translates BER read results")
{
    using namespace DcgmNs::Cache;

    GIVEN("no usable device handle")
    {
        DcgmNvLinkBerRawResult result { .hasDeviceHandle = false };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBerResult(result, DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO, 1000);

            THEN("a blank value is appended without changing watch status")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == DCGM_FP64_BLANK);
                CHECK(intent.rawValue == 0);
                CHECK(intent.timestampUsec == 1000);
                CHECK_FALSE(intent.shouldSetLastStatus);
                CHECK(intent.logReason == NvLinkBerLogReason::None);
            }
        }
    }

    GIVEN("an unsupported public field")
    {
        DcgmNvLinkBerRawResult result { .nvmlReturn = NVML_SUCCESS };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBerResult(result, DCGM_FI_DEV_GPU_NAME, 1001);

            THEN("nothing is appended or logged")
            {
                CHECK_FALSE(intent.shouldAppend);
                CHECK_FALSE(intent.shouldSetLastStatus);
                CHECK(intent.logReason == NvLinkBerLogReason::None);
            }
        }
    }

    GIVEN("a top-level NVML error")
    {
        DcgmNvLinkBerRawResult result { .nvmlReturn = NVML_ERROR_NOT_SUPPORTED };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBerResult(result, DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO, 1002);

            THEN("the public NVML error sentinel and watch status are returned")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == NvmlErrorToDoubleValue(NVML_ERROR_NOT_SUPPORTED));
                CHECK(intent.timestampUsec == 1002);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_ERROR_NOT_SUPPORTED);
                CHECK(intent.logReason == NvLinkBerLogReason::TopLevelError);
                CHECK(intent.logNvmlFieldId == NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER);
            }
        }
    }

    GIVEN("a per-field NVML error")
    {
        DcgmNvLinkBerRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.effective = BerField(NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER, NVML_ERROR_UNKNOWN, 0);

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBerResult(result, DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO, 1003);

            THEN("the per-field sentinel and watch status are returned")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == NvmlErrorToDoubleValue(NVML_ERROR_UNKNOWN));
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_ERROR_UNKNOWN);
                CHECK(intent.logReason == NvLinkBerLogReason::FieldError);
                CHECK(intent.logFieldReturn == NVML_ERROR_UNKNOWN);
                CHECK(intent.logNvmlFieldId == NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER);
            }
        }
    }

    GIVEN("a successful symbol BER field-value")
    {
        DcgmNvLinkBerRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.symbol = BerField(NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER, NVML_SUCCESS, 0x0200000000000003ULL);
        auto const [mantissa, exponent, expectedBer] = DcgmNs::Utils::NvmlBerParser(result.symbol.value.ullVal);
        (void)mantissa;
        (void)exponent;

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkBerResult(result, DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO, 1004);

            THEN("the parsed BER value is appended")
            {
                CHECK(intent.shouldAppend);
                CHECK(intent.value == expectedBer);
                CHECK(intent.rawValue == 0);
                CHECK(intent.timestampUsec == 1004);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_SUCCESS);
                CHECK(intent.logReason == NvLinkBerLogReason::None);
                CHECK(intent.logNvmlFieldId == NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: translates state read results")
{
    using namespace DcgmNs::Cache;

    GIVEN("a successful state field-value")
    {
        DcgmNvLinkStateRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.state.nvmlReturn  = NVML_SUCCESS;
        result.state.value.uiVal = NVML_FEATURE_ENABLED;

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkStateResult(result, 700);

            THEN("the public state value is appended")
            {
                CHECK(intent.value == NVML_FEATURE_ENABLED);
                CHECK(intent.rawValue == 0);
                CHECK(intent.timestampUsec == 700);
                CHECK_FALSE(intent.shouldSetLastStatus);
            }
        }
    }

    GIVEN("a top-level NVML error")
    {
        DcgmNvLinkStateRawResult result { .nvmlReturn = NVML_ERROR_NOT_SUPPORTED };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkStateResult(result, 701);

            THEN("the public NVML error sentinel is appended")
            {
                CHECK(intent.value == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(intent.timestampUsec == 701);
            }
        }
    }

    GIVEN("a per-field NVML error")
    {
        DcgmNvLinkStateRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.state.nvmlReturn = NVML_ERROR_UNKNOWN;

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkStateResult(result, 702);

            THEN("the per-field NVML error sentinel is appended")
            {
                CHECK(intent.value == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));
                CHECK(intent.timestampUsec == 702);
            }
        }
    }

    GIVEN("no usable device handle")
    {
        DcgmNvLinkStateRawResult result { .hasDeviceHandle = false };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkStateResult(result, 703);

            THEN("a blank value is appended")
            {
                CHECK(intent.value == DCGM_INT64_BLANK);
                CHECK(intent.timestampUsec == 703);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: translates error-counter read results")
{
    using namespace DcgmNs::Cache;

    GIVEN("a successful error-counter field-value")
    {
        DcgmNvLinkErrorCounterRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.value.nvmlReturn   = NVML_SUCCESS;
        result.value.value.ullVal = 99;

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkErrorCounterResult(result, 800);

            THEN("the public counter value and success status are returned")
            {
                CHECK(intent.value == 99);
                CHECK(intent.timestampUsec == 800);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_SUCCESS);
                CHECK_FALSE(intent.sawTopLevelUnsupported);
            }
        }
    }

    GIVEN("a top-level unsupported error")
    {
        DcgmNvLinkErrorCounterRawResult result { .nvmlReturn = NVML_ERROR_NOT_SUPPORTED };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkErrorCounterResult(result, 801);

            THEN("the unsupported sentinel and watch status are returned")
            {
                CHECK(intent.value == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(intent.timestampUsec == 801);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_ERROR_NOT_SUPPORTED);
                CHECK(intent.sawTopLevelUnsupported);
            }
        }
    }

    GIVEN("a per-field NVML error")
    {
        DcgmNvLinkErrorCounterRawResult result { .nvmlReturn = NVML_SUCCESS };
        result.value.nvmlReturn = NVML_ERROR_UNKNOWN;

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkErrorCounterResult(result, 802);

            THEN("the per-field sentinel and watch status are returned")
            {
                CHECK(intent.value == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));
                CHECK(intent.timestampUsec == 802);
                CHECK(intent.shouldSetLastStatus);
                CHECK(intent.lastStatus == NVML_ERROR_UNKNOWN);
                CHECK_FALSE(intent.sawTopLevelUnsupported);
            }
        }
    }

    GIVEN("no usable device handle")
    {
        DcgmNvLinkErrorCounterRawResult result { .hasDeviceHandle = false };

        WHEN("the result is translated")
        {
            auto intent = TranslateNvLinkErrorCounterResult(result, 803);

            THEN("a blank value is appended without changing watch status")
            {
                CHECK(intent.value == DCGM_INT64_BLANK);
                CHECK(intent.timestampUsec == 803);
                CHECK_FALSE(intent.shouldSetLastStatus);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: validates cached-watch apply snapshots")
{
    using namespace DcgmNs::Cache;

    GIVEN("a current watched field with the expected generation")
    {
        NvLinkCachedWatchSnapshot const currentWatch { .isWatched = true, .generation = 41 };

        WHEN("the cached-watch result is checked")
        {
            bool const canApply = CanApplyNvLinkCachedWatchResult(currentWatch, 41);

            THEN("the result may be applied")
            {
                CHECK(canApply);
            }
        }
    }

    GIVEN("a field whose watch was removed")
    {
        NvLinkCachedWatchSnapshot const currentWatch { .isWatched = false, .generation = 41 };

        WHEN("the cached-watch result is checked")
        {
            bool const canApply = CanApplyNvLinkCachedWatchResult(currentWatch, 41);

            THEN("the result is rejected")
            {
                CHECK_FALSE(canApply);
            }
        }
    }

    GIVEN("a field whose watch generation changed")
    {
        NvLinkCachedWatchSnapshot const currentWatch { .isWatched = true, .generation = 42 };

        WHEN("the cached-watch result is checked")
        {
            bool const canApply = CanApplyNvLinkCachedWatchResult(currentWatch, 41);

            THEN("the stale result is rejected")
            {
                CHECK_FALSE(canApply);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: chunks NVML field-values requests")
{
    using namespace DcgmNs::Cache;

    GIVEN("no requested field-values")
    {
        WHEN("chunks are built")
        {
            auto chunks = BuildNvmlFieldValueChunks(0);

            THEN("no chunks are returned")
            {
                CHECK(chunks.empty());
            }
        }
    }

    GIVEN("exactly NVML_FI_MAX requested field-values")
    {
        WHEN("chunks are built")
        {
            auto chunks = BuildNvmlFieldValueChunks(NVML_FI_MAX);

            THEN("one full chunk is returned")
            {
                REQUIRE(chunks.size() == 1);
                CHECK(chunks[0].offset == 0);
                CHECK(chunks[0].size == NVML_FI_MAX);
            }
        }
    }

    GIVEN("one more than NVML_FI_MAX requested field-values")
    {
        WHEN("chunks are built")
        {
            auto chunks = BuildNvmlFieldValueChunks(NVML_FI_MAX + 1);

            THEN("the overflow starts a second chunk")
            {
                REQUIRE(chunks.size() == 2);
                CHECK(chunks[0].offset == 0);
                CHECK(chunks[0].size == NVML_FI_MAX);
                CHECK(chunks[1].offset == NVML_FI_MAX);
                CHECK(chunks[1].size == 1);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: returns requested NVML field-value output")
{
    using namespace DcgmNs::Cache;

    GIVEN("two requested field-values")
    {
        std::array<nvmlFieldValue_t, 2> fieldValues {};
        fieldValues[0].fieldId = 1;
        fieldValues[1].fieldId = 2;

        WHEN("NVML returns values for the requested fields")
        {
            auto result = ReadNvmlFieldValues(fieldValues, [](unsigned int count, nvmlFieldValue_t *values) {
                REQUIRE(count == 2);
                CHECK(values[0].fieldId == 1);
                CHECK(values[1].fieldId == 2);
                values[0].value.ullVal = 17;
                values[1].value.ullVal = 23;
                return NVML_SUCCESS;
            });

            THEN("the output is returned")
            {
                CHECK(result.has_value());
                CHECK(fieldValues[0].fieldId == 1);
                CHECK(fieldValues[1].fieldId == 2);
                CHECK(fieldValues[0].value.ullVal == 17);
                CHECK(fieldValues[1].value.ullVal == 23);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture,
                "DcgmNvLinkCachedWatch: detects NVML output beyond the requested field count")
{
    using namespace DcgmNs::Cache;

    GIVEN("two requested field-values")
    {
        std::array<nvmlFieldValue_t, 2> requestedFields {};
        requestedFields[0].fieldId = 1;
        requestedFields[1].fieldId = 2;

        WHEN("NVML writes an additional field-value")
        {
            auto fieldValues = requestedFields;
            auto result      = ReadNvmlFieldValues(fieldValues, [&](unsigned int count, nvmlFieldValue_t *values) {
                REQUIRE(count == requestedFields.size());
                CHECK(values[0].fieldId == requestedFields[0].fieldId);
                CHECK(values[1].fieldId == requestedFields[1].fieldId);
                values[0].value.ullVal  = 17;
                values[1].value.ullVal  = 23;
                values[count].timestamp = 1786393219000128;
                return NVML_SUCCESS;
            });

            THEN("the result is rejected without changing requested values")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK(result.error() == NVML_ERROR_UNKNOWN);
                CHECK(fieldValues[0].fieldId == requestedFields[0].fieldId);
                CHECK(fieldValues[1].fieldId == requestedFields[1].fieldId);
                CHECK(fieldValues[0].value.ullVal == 0);
                CHECK(fieldValues[1].value.ullVal == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture,
                "DcgmNvLinkCachedWatch: preserves requested storage when NVML returns an error")
{
    using namespace DcgmNs::Cache;

    GIVEN("two requested field-values")
    {
        std::array<nvmlFieldValue_t, 2> requestedFields {};
        requestedFields[0].fieldId = 1;
        requestedFields[1].fieldId = 2;

        WHEN("NVML returns an error after modifying temporary output")
        {
            auto fieldValues = requestedFields;
            auto result      = ReadNvmlFieldValues(fieldValues, [](unsigned int, nvmlFieldValue_t *values) {
                values[0].value.ullVal = 17;
                values[1].value.ullVal = 23;
                return NVML_ERROR_NOT_SUPPORTED;
            });

            THEN("the error is returned without changing requested values")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK(result.error() == NVML_ERROR_NOT_SUPPORTED);
                CHECK(fieldValues[0].fieldId == requestedFields[0].fieldId);
                CHECK(fieldValues[1].fieldId == requestedFields[1].fieldId);
                CHECK(fieldValues[0].value.ullVal == 0);
                CHECK(fieldValues[1].value.ullVal == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture,
                "DcgmNvLinkCachedWatch: prioritizes unexpected output over an NVML callback error")
{
    using namespace DcgmNs::Cache;

    GIVEN("two requested field-values")
    {
        std::array<nvmlFieldValue_t, 2> requestedFields {};
        requestedFields[0].fieldId = 1;
        requestedFields[1].fieldId = 2;

        WHEN("NVML returns an error after writing the guard record")
        {
            auto fieldValues = requestedFields;
            auto result      = ReadNvmlFieldValues(fieldValues, [&](unsigned int count, nvmlFieldValue_t *values) {
                values[0].value.ullVal  = 17;
                values[1].value.ullVal  = 23;
                values[count].timestamp = 1786393219000128;
                return NVML_ERROR_NOT_SUPPORTED;
            });

            THEN("the unexpected output is reported without changing requested values")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK(result.error() == NVML_ERROR_UNKNOWN);
                CHECK(fieldValues[0].fieldId == requestedFields[0].fieldId);
                CHECK(fieldValues[1].fieldId == requestedFields[1].fieldId);
                CHECK(fieldValues[0].value.ullVal == 0);
                CHECK(fieldValues[1].value.ullVal == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmNvLinkCachedWatchTestFixture, "DcgmNvLinkCachedWatch: rejects invalid field-value counts")
{
    using namespace DcgmNs::Cache;

    GIVEN("no requested field-values")
    {
        std::array<nvmlFieldValue_t, 0> fieldValues;
        bool wasCalled = false;

        WHEN("the reader is invoked")
        {
            auto result = ReadNvmlFieldValues(fieldValues, [&](unsigned int, nvmlFieldValue_t *) {
                wasCalled = true;
                return NVML_SUCCESS;
            });

            THEN("the request is rejected without calling NVML")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK(result.error() == NVML_ERROR_INVALID_ARGUMENT);
                CHECK_FALSE(wasCalled);
            }
        }
    }

    GIVEN("more than NVML_FI_MAX requested field-values")
    {
        std::vector<nvmlFieldValue_t> fieldValues(NVML_FI_MAX + 1);
        bool wasCalled = false;

        WHEN("the reader is invoked")
        {
            auto result = ReadNvmlFieldValues(fieldValues, [&](unsigned int, nvmlFieldValue_t *) {
                wasCalled = true;
                return NVML_SUCCESS;
            });

            THEN("the request is rejected without calling NVML")
            {
                REQUIRE_FALSE(result.has_value());
                CHECK(result.error() == NVML_ERROR_INVALID_ARGUMENT);
                CHECK_FALSE(wasCalled);
            }
        }
    }
}

} // namespace
