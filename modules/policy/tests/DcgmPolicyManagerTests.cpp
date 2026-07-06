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
#include <DcgmPolicyManager.h>
#include <catch2/catch_all.hpp>
#include <dcgm_core_communication.h>
#include <dcgm_fields.h>
#include <dcgm_policy_structs.h>
#include <dcgm_structs.h>
#include <map>
#include <vector>

/**
 * Mock context for all core proxy calls made by DcgmPolicyManager.
 */
struct MockPolicyContext
{
    // GPU inventory returned by GetGpuCount / GetGpuIds
    unsigned int m_numGpus             = 1;
    std::vector<unsigned int> m_gpuIds = { 0 };

    // Configurable GetSamples response per fieldId.
    // The proxy sets gs->response.samples = caller's stack array before calling
    // postfunc, so the mock writes directly through that pointer.
    struct SamplesData
    {
        dcgmReturn_t m_ret = DCGM_ST_NO_DATA;
        dcgmcm_sample_t m_curr {}; // samples[0]: newest (DESCENDING order)
        dcgmcm_sample_t m_prev {}; // samples[1]: older
        int m_numSamples = 0;
    };
    std::map<unsigned short, SamplesData> m_samplesMap;

    // Configurable GetLatestSample response per fieldId
    struct LatestData
    {
        dcgmReturn_t m_ret = DCGM_ST_NO_DATA;
        dcgmcm_sample_t m_sample {};
    };
    std::map<unsigned short, LatestData> m_latestMap;

    // Violation notification tracking
    int m_notifyCount                     = 0;
    dcgmPolicyCondition_t m_lastCondition = static_cast<dcgmPolicyCondition_t>(0);
    unsigned int m_lastNotifyGpuId        = UINT_MAX;
};

static dcgmReturn_t MockPolicyPostfunc(dcgm_module_command_header_t *header, void *poster)
{
    if (!header)
        return DCGM_ST_BADPARAM;

    auto *ctx = static_cast<MockPolicyContext *>(poster);

    switch (static_cast<dcgmCoreReqCmd_t>(header->subCommand))
    {
        case DcgmCoreReqIdCMGetGpuCount:
        {
            auto *ggc                = reinterpret_cast<dcgmCoreGetGpuCount_t *>(header);
            ggc->response.ret        = DCGM_ST_OK;
            ggc->response.uintAnswer = ctx ? ctx->m_numGpus : 1u;
            return DCGM_ST_OK;
        }
        case DcgmCoreReqIdCMGetGpuIds:
        {
            auto *cgg         = reinterpret_cast<dcgmCoreGetGpuList_t *>(header);
            cgg->response.ret = DCGM_ST_OK;
            if (ctx)
            {
                cgg->response.gpuCount
                    = static_cast<unsigned int>(std::min(ctx->m_gpuIds.size(), std::size(cgg->response.gpuIds)));

                for (size_t i = 0; i < cgg->response.gpuCount; ++i)
                    cgg->response.gpuIds[i] = ctx->m_gpuIds[i];
            }
            else
            {
                cgg->response.gpuCount  = 1;
                cgg->response.gpuIds[0] = 0;
            }
            return DCGM_ST_OK;
        }
        case DcgmCoreReqIdGMVerifyAndUpdateGroupId:
        {
            auto *bq                = reinterpret_cast<dcgmCoreBasicQuery_t *>(header);
            bq->response.ret        = DCGM_ST_OK;
            bq->response.uintAnswer = bq->request.entityId; // return groupId unchanged
            return DCGM_ST_OK;
        }
        case DcgmCoreReqIdGMGetGroupEntities:
        {
            auto *gge                      = reinterpret_cast<dcgmCoreGetGroupEntities_t *>(header);
            gge->response.ret              = DCGM_ST_OK;
            gge->response.entityPairsCount = 1;
            gge->response.entityPairs[0]   = { DCGM_FE_GPU, 0 };
            return DCGM_ST_OK;
        }
        case DcgmCoreReqIdCMAddFieldWatch:
        case DcgmCoreReqIdCMUpdateAllFields:
        case DcgmCoreReqIdNotifyRequestOfCompletion:
            return DCGM_ST_OK;
        case DcgmCoreReqIdCMGetSamples:
        {
            auto *gs               = reinterpret_cast<dcgmCoreGetSamples_t *>(header);
            unsigned short fieldId = gs->request.fieldId;

            gs->response.numSamples = 0;
            if (!ctx || ctx->m_samplesMap.find(fieldId) == ctx->m_samplesMap.end())
            {
                gs->response.ret = DCGM_ST_NO_DATA;
                return DCGM_ST_OK;
            }

            auto &data       = ctx->m_samplesMap.at(fieldId);
            gs->response.ret = data.m_ret;
            if (data.m_ret == DCGM_ST_OK && gs->response.samples != nullptr && data.m_numSamples >= 2)
            {
                gs->response.samples[0] = data.m_curr;
                gs->response.samples[1] = data.m_prev;
                gs->response.numSamples = 2;
            }
            return DCGM_ST_OK;
        }
        case DcgmCoreReqIdCMGetLatestSample:
        {
            auto *gls              = reinterpret_cast<dcgmCoreGetLatestSample_t *>(header);
            unsigned short fieldId = gls->request.fieldId;

            if (!ctx || ctx->m_latestMap.find(fieldId) == ctx->m_latestMap.end())
            {
                gls->response.ret = DCGM_ST_NO_DATA;
                return DCGM_ST_OK;
            }

            auto &data           = ctx->m_latestMap.at(fieldId);
            gls->response.ret    = data.m_ret;
            gls->response.sample = data.m_sample;
            return DCGM_ST_OK;
        }
        case DcgmCoreReqIdSendRawMessage:
        {
            auto *msg     = reinterpret_cast<dcgmCoreSendRawMessage_t *>(header);
            msg->response = DCGM_ST_OK;
            if (ctx)
            {
                ctx->m_notifyCount++;
                if (msg->request.msgData != nullptr && msg->request.msgSize == sizeof(dcgm_msg_policy_notify_t))
                {
                    auto *notify           = static_cast<dcgm_msg_policy_notify_t *>(msg->request.msgData);
                    ctx->m_lastCondition   = notify->response.condition;
                    ctx->m_lastNotifyGpuId = notify->response.gpuId;
                }
            }
            return DCGM_ST_OK;
        }
        default:
            return DCGM_ST_OK;
    }
}

// ---------------------------------------------------------------------------
// Fixture: one GPU (id=0), watcher registered for all conditions, thresholds set
// ---------------------------------------------------------------------------
static constexpr dcgm_connection_id_t kConnId  = 1;
static constexpr dcgm_request_id_t kReqId      = 1;
static constexpr unsigned int kGpuId           = 0;
static constexpr dcgmGpuGrp_t kGroupId         = static_cast<dcgmGpuGrp_t>(1);
static constexpr unsigned int kMaxThermal      = 80;
static constexpr unsigned int kMaxPower        = 200;
static constexpr unsigned int kMaxRetiredPages = 10;

static constexpr dcgmPolicyCondition_t kAllConditions = static_cast<dcgmPolicyCondition_t>(
    DCGM_POLICY_COND_DBE | DCGM_POLICY_COND_PCI | DCGM_POLICY_COND_MAX_PAGES_RETIRED | DCGM_POLICY_COND_THERMAL
    | DCGM_POLICY_COND_POWER | DCGM_POLICY_COND_NVLINK | DCGM_POLICY_COND_XID);

static MockPolicyContext::SamplesData MakeCounterSamples(int64_t current, int64_t previous)
{
    MockPolicyContext::SamplesData data;
    data.m_ret          = DCGM_ST_OK;
    data.m_curr.val.i64 = current;
    data.m_prev.val.i64 = previous;
    data.m_numSamples   = 2;
    return data;
}

static MockPolicyContext::LatestData MakeLatestSample(int64_t value, timelib64_t timestamp = 6000000)
{
    MockPolicyContext::LatestData data;
    data.m_ret              = DCGM_ST_OK;
    data.m_sample.val.i64   = value;
    data.m_sample.timestamp = timestamp;
    return data;
}

static void AddGpuInt64Value(DcgmFvBuffer &fvBuffer,
                             unsigned short fieldId,
                             int64_t value,
                             timelib64_t timestamp = 6000000,
                             dcgmReturn_t status   = DCGM_ST_OK)
{
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, fieldId, value, timestamp, status);
}

class DcgmPolicyManagerFixture
{
public:
    MockPolicyContext ctx;
    dcgmCoreCallbacks_t ccb;
    DcgmPolicyManager mgr;

    DcgmPolicyManagerFixture()
        : ccb(makeCallbacks())
        , mgr(ccb)
    {
        dcgm_policy_msg_register_t regMsg = {};
        regMsg.header.connectionId        = kConnId;
        regMsg.header.requestId           = kReqId;
        regMsg.groupId                    = kGroupId;
        regMsg.condition                  = kAllConditions;
        mgr.RegisterForPolicy(&regMsg);

        dcgm_policy_msg_set_policy_t setMsg                                   = {};
        setMsg.header.connectionId                                            = kConnId;
        setMsg.groupId                                                        = kGroupId;
        setMsg.policy.version                                                 = dcgmPolicy_version;
        setMsg.policy.condition                                               = kAllConditions;
        setMsg.policy.parms[DCGM_POLICY_COND_IDX_THERMAL].tag                 = dcgmPolicyConditionParams_st::LLONG;
        setMsg.policy.parms[DCGM_POLICY_COND_IDX_THERMAL].val.llval           = kMaxThermal;
        setMsg.policy.parms[DCGM_POLICY_COND_IDX_POWER].tag                   = dcgmPolicyConditionParams_st::LLONG;
        setMsg.policy.parms[DCGM_POLICY_COND_IDX_POWER].val.llval             = kMaxPower;
        setMsg.policy.parms[DCGM_POLICY_COND_IDX_MAX_PAGES_RETIRED].tag       = dcgmPolicyConditionParams_st::LLONG;
        setMsg.policy.parms[DCGM_POLICY_COND_IDX_MAX_PAGES_RETIRED].val.llval = kMaxRetiredPages;
        mgr.ProcessSetPolicy(&setMsg);
    }

private:
    dcgmCoreCallbacks_t makeCallbacks()
    {
        dcgmCoreCallbacks_t cb = {};
        cb.version             = dcgmCoreCallbacks_version;
        cb.postfunc            = MockPolicyPostfunc;
        cb.poster              = &ctx;
        cb.loggerfunc          = [](void const *) {
        };
        return cb;
    }
};

// ===========================================================================
// Tests: fields silently ignored when GPU has no policy / not a GPU entity
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::OnFieldValuesUpdate ignores non-GPU entity group")
{
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_CPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 999, 1000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

TEST_CASE("DcgmPolicyManager::OnFieldValuesUpdate ignores GPU without policy set")
{
    MockPolicyContext ctx;
    dcgmCoreCallbacks_t ccb = {};
    ccb.version             = dcgmCoreCallbacks_version;
    ccb.postfunc            = MockPolicyPostfunc;
    ccb.poster              = &ctx;
    ccb.loggerfunc          = [](void const *) {
    };

    DcgmPolicyManager mgr(ccb); // no RegisterForPolicy or ProcessSetPolicy

    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 999, 1000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

// ===========================================================================
// Tests: CheckThermalValues
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckThermalValues triggers violation above threshold")
{
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal + 1, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 1);
    REQUIRE(ctx.m_lastCondition == DCGM_POLICY_COND_THERMAL);
    REQUIRE(ctx.m_lastNotifyGpuId == kGpuId);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckThermalValues no violation at or below threshold")
{
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckThermalValues skips fv with error status")
{
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(
        DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal + 10, 6000000, DCGM_ST_NOT_WATCHED);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

// ===========================================================================
// Tests: CheckPowerValues
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckPowerValues triggers violation above threshold")
{
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddDoubleValue(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, kMaxPower + 1.0, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 1);
    REQUIRE(ctx.m_lastCondition == DCGM_POLICY_COND_POWER);
    REQUIRE(ctx.m_lastNotifyGpuId == kGpuId);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckPowerValues no violation at or below threshold")
{
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddDoubleValue(
        DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, static_cast<double>(kMaxPower), 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

// ===========================================================================
// Tests: CheckXIDErrors
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckXIDErrors triggers violation for any XID value")
{
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_XID_ERROR, 79, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 1);
    REQUIRE(ctx.m_lastCondition == DCGM_POLICY_COND_XID);
    REQUIRE(ctx.m_lastNotifyGpuId == kGpuId);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckXIDErrors skips fv with error status")
{
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_XID_ERROR, 79, 6000000, DCGM_ST_NOT_WATCHED);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::OnFieldValuesUpdate ignores unknown fields")
{
    GIVEN("a GPU field value with a field id that policy does not handle")
    {
        DcgmFvBuffer fvBuffer;
        AddGpuInt64Value(fvBuffer, DCGM_FI_SYSTEM_GPU_QUANTITY, 999);

        WHEN("field values are processed")
        {
            mgr.OnFieldValuesUpdate(&fvBuffer);

            CHECK(ctx.m_notifyCount == 0);
        }
    }
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::OnFieldValuesUpdate ignores detached GPUs")
{
    GIVEN("a policy manager with detached GPUs")
    {
        REQUIRE(mgr.DetachGpus() == DCGM_ST_OK);

        DcgmFvBuffer fvBuffer;
        AddGpuInt64Value(fvBuffer, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal + 1);

        WHEN("a violating temperature sample arrives")
        {
            mgr.OnFieldValuesUpdate(&fvBuffer);

            CHECK(ctx.m_notifyCount == 0);
        }
    }
}

// ===========================================================================
// Tests: CheckEccErrors (uses GetCounterDelta → GetSamples mock)
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::CheckEccErrors triggers violation when counter increases")
{
    MockPolicyContext::SamplesData data;
    data.m_ret                                    = DCGM_ST_OK;
    data.m_curr.val.i64                           = 5;
    data.m_prev.val.i64                           = 3;
    data.m_numSamples                             = 2;
    ctx.m_samplesMap[DCGM_FI_DEV_ECC_DBE_VOL_DEV] = data;

    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_ECC_DBE_VOL_DEV, 5, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 1);
    REQUIRE(ctx.m_lastCondition == DCGM_POLICY_COND_DBE);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckEccErrors no violation when counter unchanged")
{
    MockPolicyContext::SamplesData data;
    data.m_ret                                    = DCGM_ST_OK;
    data.m_curr.val.i64                           = 5;
    data.m_prev.val.i64                           = 5;
    data.m_numSamples                             = 2;
    ctx.m_samplesMap[DCGM_FI_DEV_ECC_DBE_VOL_DEV] = data;

    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_ECC_DBE_VOL_DEV, 5, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckEccErrors no violation on counter decrease (reset)")
{
    MockPolicyContext::SamplesData data;
    data.m_ret                                    = DCGM_ST_OK;
    data.m_curr.val.i64                           = 2; // went down — counter was reset
    data.m_prev.val.i64                           = 5;
    data.m_numSamples                             = 2;
    ctx.m_samplesMap[DCGM_FI_DEV_ECC_DBE_VOL_DEV] = data;

    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_ECC_DBE_VOL_DEV, 2, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::CheckEccErrors no violation when GetSamples returns insufficient data")
{
    // samplesMap not configured → GetSamples returns NO_DATA → GetCounterDelta returns false
    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_ECC_DBE_VOL_DEV, 5, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckEccErrors no violation for blank sample values")
{
    GIVEN("ECC counter samples where the latest value is blank")
    {
        ctx.m_samplesMap[DCGM_FI_DEV_ECC_DBE_VOL_DEV] = MakeCounterSamples(DCGM_INT64_BLANK, 3);

        DcgmFvBuffer fvBuffer;
        AddGpuInt64Value(fvBuffer, DCGM_FI_DEV_ECC_DBE_VOL_DEV, 5);

        WHEN("the ECC field value is processed")
        {
            mgr.OnFieldValuesUpdate(&fvBuffer);

            CHECK(ctx.m_notifyCount == 0);
        }
    }
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckEccErrors no violation for out of range samples")
{
    GIVEN("ECC counter samples where the latest value is outside the supported unsigned range")
    {
        ctx.m_samplesMap[DCGM_FI_DEV_ECC_DBE_VOL_DEV] = MakeCounterSamples(static_cast<int64_t>(UINT_MAX) + 1, 3);

        DcgmFvBuffer fvBuffer;
        AddGpuInt64Value(fvBuffer, DCGM_FI_DEV_ECC_DBE_VOL_DEV, 5);

        WHEN("the ECC field value is processed")
        {
            mgr.OnFieldValuesUpdate(&fvBuffer);

            CHECK(ctx.m_notifyCount == 0);
        }
    }
}

// ===========================================================================
// Tests: CheckPcieErrors (same counter-delta pattern)
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::CheckPcieErrors triggers violation when counter increases")
{
    MockPolicyContext::SamplesData data;
    data.m_ret                                      = DCGM_ST_OK;
    data.m_curr.val.i64                             = 10;
    data.m_prev.val.i64                             = 7;
    data.m_numSamples                               = 2;
    ctx.m_samplesMap[DCGM_FI_DEV_PCIE_REPLAY_TOTAL] = data;

    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_PCIE_REPLAY_TOTAL, 10, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 1);
    REQUIRE(ctx.m_lastCondition == DCGM_POLICY_COND_PCI);
}

// ===========================================================================
// Tests: CheckNVLinkErrors
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::CheckNVLinkErrors triggers violation when counter increases")
{
    GIVEN("one of the NVLink error counter fields increased")
    {
        auto const fieldId = GENERATE(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL);
        CAPTURE(fieldId);

        ctx.m_samplesMap[fieldId] = MakeCounterSamples(10, 7);

        DcgmFvBuffer fvBuffer;
        AddGpuInt64Value(fvBuffer, fieldId, 10);

        WHEN("the NVLink field value is processed")
        {
            mgr.OnFieldValuesUpdate(&fvBuffer);

            CHECK(ctx.m_notifyCount == 1);
            CHECK(ctx.m_lastCondition == DCGM_POLICY_COND_NVLINK);
        }
    }
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckNVLinkErrors no violation when counter is reset")
{
    GIVEN("an NVLink counter that moved backward after reset")
    {
        ctx.m_samplesMap[DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL] = MakeCounterSamples(2, 7);

        DcgmFvBuffer fvBuffer;
        AddGpuInt64Value(fvBuffer, DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, 2);

        WHEN("the NVLink field value is processed")
        {
            mgr.OnFieldValuesUpdate(&fvBuffer);

            CHECK(ctx.m_notifyCount == 0);
        }
    }
}

// ===========================================================================
// Tests: CheckRetiredPages
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::CheckRetiredPages triggers violation when total exceeds max")
{
    // fv carries DBE count=5; mock returns SBE count=8 via GetLatestSample
    // total 13 > kMaxRetiredPages (10)
    MockPolicyContext::LatestData sbeData;
    sbeData.m_ret                                       = DCGM_ST_OK;
    sbeData.m_sample.val.i64                            = 8;
    ctx.m_latestMap[DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL] = sbeData;

    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, 5, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 1);
    REQUIRE(ctx.m_lastCondition == DCGM_POLICY_COND_MAX_PAGES_RETIRED);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture, "DcgmPolicyManager::CheckRetiredPages no violation when total within limit")
{
    // total 2 + 3 = 5 <= kMaxRetiredPages (10)
    MockPolicyContext::LatestData sbeData;
    sbeData.m_ret                                       = DCGM_ST_OK;
    sbeData.m_sample.val.i64                            = 3;
    ctx.m_latestMap[DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL] = sbeData;

    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, 2, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer);

    REQUIRE(ctx.m_notifyCount == 0);
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::CheckRetiredPages triggers violation when SBE field plus DBE latest exceed max")
{
    GIVEN("an SBE retired-page update with DBE latest sample above the combined threshold")
    {
        ctx.m_latestMap[DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL] = MakeLatestSample(7, 5000000);

        DcgmFvBuffer fvBuffer;
        AddGpuInt64Value(fvBuffer, DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, 6);

        WHEN("the retired-page field value is processed")
        {
            mgr.OnFieldValuesUpdate(&fvBuffer);

            CHECK(ctx.m_notifyCount == 1);
            CHECK(ctx.m_lastCondition == DCGM_POLICY_COND_MAX_PAGES_RETIRED);
        }
    }
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::CheckRetiredPages ignores not-supported companion sample")
{
    GIVEN("a DBE retired-page update with an unsupported companion SBE sample")
    {
        ctx.m_latestMap[DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL] = MakeLatestSample(DCGM_INT64_NOT_SUPPORTED);

        DcgmFvBuffer fvBuffer;
        AddGpuInt64Value(fvBuffer, DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, 20);

        WHEN("the retired-page field value is processed")
        {
            mgr.OnFieldValuesUpdate(&fvBuffer);

            CHECK(ctx.m_notifyCount == 0);
        }
    }
}

// ===========================================================================
// Tests: SetViolation rate limiting (5-second cooldown per alert type per watcher)
// ===========================================================================

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::SetViolation suppresses repeated violations within 5 second window")
{
    DcgmFvBuffer fvBuffer1;
    fvBuffer1.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal + 5, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer1);
    REQUIRE(ctx.m_notifyCount == 1);

    // 1 second later — below the 5-second minimum signal interval
    DcgmFvBuffer fvBuffer2;
    fvBuffer2.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal + 5, 7000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer2);
    REQUIRE(ctx.m_notifyCount == 1); // suppressed
}

TEST_CASE_METHOD(DcgmPolicyManagerFixture,
                 "DcgmPolicyManager::SetViolation fires again after 5 second cooldown expires")
{
    DcgmFvBuffer fvBuffer1;
    fvBuffer1.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal + 5, 6000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer1);
    REQUIRE(ctx.m_notifyCount == 1);

    // 6 seconds later — above the 5-second minimum
    DcgmFvBuffer fvBuffer2;
    fvBuffer2.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal + 5, 12000000, DCGM_ST_OK);
    mgr.OnFieldValuesUpdate(&fvBuffer2);
    REQUIRE(ctx.m_notifyCount == 2);
}
