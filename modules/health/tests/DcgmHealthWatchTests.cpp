/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmGPUHardwareLimits.h>
#include <DcgmHealthResponse.h>
#include <DcgmHealthWatch.h>
#include <DcgmImexManager.h>
#include <DcgmStringHelpers.h>
#include <MockFileHandle.h>
#include <UniquePtrUtil.h>
#include <array>
#include <catch2/catch_all.hpp>
#include <dcgm_core_communication.h>
#include <dcgm_core_structs.h>
#include <dcgm_structs.h>
#include <map>
#include <optional>
#include <unordered_map>

const char *EntityToString(dcgm_field_entity_group_t entityGroupId);

/**
 * Mock cache context for simulating DcgmCacheManager responses
 */
struct MockGetLatestSampleContext
{
    std::unordered_map<unsigned short, dcgmReturn_t> fieldReturnCodes; //!< Configured return codes per field ID
    std::unordered_map<unsigned short, dcgmcm_sample_t> fieldSamples;  //!< Configured sample data per field ID
    std::string managedString;                                         //!< Managed string to ensure char* validity
    std::vector<unsigned int> gpuIds;                                  //!< GPU IDs to return from GetGpuIds
    bool hasNvLinks      = false;                                      //!< Whether GPUs report having NVLinks
    unsigned int groupId = 1;                                          //!< Group ID for GetGroupEntities mock

    /**
     * Configures a field to return specific data when queried via GetLatestSample.
     *
     * @param fieldId Field identifier to configure
     * @param retCode Return code for the field query (e.g., DCGM_ST_OK, DCGM_ST_NOT_WATCHED)
     * @param sample Sample data to return if retCode is DCGM_ST_OK
     */
    void SetFieldSample(unsigned short fieldId, dcgmReturn_t retCode, dcgmcm_sample_t sample = {})
    {
        fieldReturnCodes[fieldId] = retCode;
        if (retCode == DCGM_ST_OK)
        {
            fieldSamples[fieldId] = sample;
        }
    }

    /**
     * Adds a string to the managed strings vector.
     *
     * @param str String to add
     */
    void AddManagedString(std::string_view str)
    {
        managedString = std::string(str);
    }

    /**
     * Returns the managed string.
     *
     * @return View of the managed string
     */
    std::string_view ManagedString() const
    {
        return managedString;
    }

    /**
     * Configures the mock to report GPUs with or without NVLinks.
     *
     * @param numGpus Number of GPUs to simulate
     * @param withNvLinks If true, GPU 0 link 0 reports Up; otherwise all NotSupported
     */
    void SetupGpuNvLinks(unsigned int numGpus, bool withNvLinks)
    {
        gpuIds.clear();
        for (unsigned int i = 0; i < numGpus; i++)
        {
            gpuIds.push_back(i);
        }
        hasNvLinks = withNvLinks;
    }

    /**
     * Builds an entity list from the configured GPU IDs (for passing to MonitorGlobalHealthChecks).
     */
    std::vector<dcgmGroupEntityPair_t> GetEntities() const
    {
        std::vector<dcgmGroupEntityPair_t> entities;
        for (auto id : gpuIds)
        {
            entities.push_back({ DCGM_FE_GPU, id });
        }
        return entities;
    }
};

/**
 * Mock postfunc callback - returns configured response
 *
 * @param header Command header containing request details
 * @param poster Pointer to MockGetLatestSampleContext with configured responses
 * @return DCGM_ST_OK on success, DCGM_ST_BADPARAM if header is null
 */
static dcgmReturn_t MockCoreCallbackGetLatestSample(dcgm_module_command_header_t *header, void *poster)
{
    if (!header)
    {
        return DCGM_ST_BADPARAM;
    }

    auto *context = static_cast<MockGetLatestSampleContext *>(poster);

    // Handle GetLatestSample requests
    if (header->subCommand == DcgmCoreReqIdCMGetLatestSample)
    {
        auto *gls              = reinterpret_cast<dcgmCoreGetLatestSample_t *>(header);
        unsigned short fieldId = gls->request.fieldId;

        // Check if we have a configured response for this field
        if (context)
        {
            auto retIt = context->fieldReturnCodes.find(fieldId);
            if (retIt != context->fieldReturnCodes.end())
            {
                gls->response.ret = retIt->second;

                // If returning OK, populate the sample
                if (retIt->second == DCGM_ST_OK && gls->request.populateSamples)
                {
                    auto sampleIt = context->fieldSamples.find(fieldId);
                    if (sampleIt != context->fieldSamples.end())
                    {
                        gls->response.sample = sampleIt->second;
                    }
                }
                return DCGM_ST_OK;
            }
        }

        if (fieldId == DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY)
        {
            gls->response.ret = DCGM_ST_OK;
            if (gls->request.populateSamples)
            {
                gls->response.sample.val.i64 = (9 << 16) | 0;
            }
            return DCGM_ST_OK;
        }

        // Default: return NOT_WATCHED for unconfigured fields
        gls->response.ret = DCGM_ST_NOT_WATCHED;
        return DCGM_ST_OK;
    }

    // Handle GetGpuIds requests
    if (header->subCommand == DcgmCoreReqIdCMGetGpuIds)
    {
        auto *cgg = reinterpret_cast<dcgmCoreGetGpuList_t *>(header);
        if (context)
        {
            cgg->response.ret      = DCGM_ST_OK;
            cgg->response.gpuCount = context->gpuIds.size();
            for (size_t i = 0; i < context->gpuIds.size() && i < DCGM_MAX_NUM_DEVICES; i++)
            {
                cgg->response.gpuIds[i] = context->gpuIds[i];
            }
        }
        else
        {
            cgg->response.ret      = DCGM_ST_OK;
            cgg->response.gpuCount = 0;
        }
        return DCGM_ST_OK;
    }

    // Handle GetEntityNvLinkLinkStatus requests
    if (header->subCommand == DcgmCoreReqIdCMGetEntityNvLinkLinkStatus)
    {
        auto *nls         = reinterpret_cast<dcgmCoreGetEntityNvLinkLinkStatus_t *>(header);
        nls->response.ret = DCGM_ST_OK;
        for (auto &ls : nls->response.linkStates)
        {
            ls = DcgmNvLinkLinkStateNotSupported;
        }
        if (context && context->hasNvLinks && nls->request.entityId == 0)
        {
            nls->response.linkStates[0] = DcgmNvLinkLinkStateUp;
        }
        return DCGM_ST_OK;
    }

    // Handle GetAllGpuInfo requests. Default to the explicit "unknown" architecture
    // sentinel so tests that configure compute capability can exercise that fallback.
    if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
    {
        auto *qgi               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
        qgi->response.ret       = DCGM_ST_OK;
        qgi->response.infoCount = 0;

        if (context)
        {
            for (auto const gpuId : context->gpuIds)
            {
                if (qgi->response.infoCount >= DCGM_MAX_NUM_DEVICES)
                {
                    break;
                }

                qgi->response.info[qgi->response.infoCount]       = {};
                qgi->response.info[qgi->response.infoCount].gpuId = gpuId;
                qgi->response.info[qgi->response.infoCount].arch  = DCGM_CHIP_ARCH_UNKNOWN;
                qgi->response.infoCount++;
            }
        }

        if (qgi->response.infoCount == 0)
        {
            qgi->response.infoCount     = 1;
            qgi->response.info[0]       = {};
            qgi->response.info[0].gpuId = 0;
            qgi->response.info[0].arch  = DCGM_CHIP_ARCH_UNKNOWN;
        }

        return DCGM_ST_OK;
    }

    // Handle GetGroupEntities requests
    if (header->subCommand == DcgmCoreReqIdGMGetGroupEntities)
    {
        auto *gge         = reinterpret_cast<dcgmCoreGetGroupEntities_t *>(header);
        gge->response.ret = DCGM_ST_OK;
        if (context)
        {
            unsigned int count = 0;
            for (size_t i = 0; i < context->gpuIds.size() && i < DCGM_GROUP_MAX_ENTITIES_V2; i++)
            {
                gge->response.entityPairs[i].entityGroupId = DCGM_FE_GPU;
                gge->response.entityPairs[i].entityId      = context->gpuIds[i];
                count++;
            }
            gge->response.entityPairsCount = count;
        }
        else
        {
            gge->response.entityPairsCount = 0;
        }
        return DCGM_ST_OK;
    }

    // Handle GetSamples requests — return NO_DATA for unconfigured fields
    if (header->subCommand == DcgmCoreReqIdCMGetSamples)
    {
        auto *gs                = reinterpret_cast<dcgmCoreGetSamples_t *>(header);
        gs->response.ret        = DCGM_ST_NO_DATA;
        gs->response.numSamples = 0;
        return DCGM_ST_OK;
    }

    // Other requests: return OK
    return DCGM_ST_OK;
}

/**
 * Mock context for GetSamples responses.
 */
struct MockGetSamplesContext
{
    struct Key
    {
        unsigned short fieldId;
        dcgmOrder_t order;

        bool operator<(Key const &other) const
        {
            if (fieldId != other.fieldId)
            {
                return fieldId < other.fieldId;
            }
            return order < other.order;
        }
    };

    std::map<Key, dcgmReturn_t> returnCodes;     //!< Configured return codes per field+order
    std::map<Key, dcgmcm_sample_t> sampleValues; //!< Configured sample data per field+order

    /**
     * Configures one ordered GetSamples response.
     */
    void SetGetSamplesResult(unsigned short fieldId,
                             dcgmOrder_t order,
                             dcgmReturn_t retCode,
                             dcgmcm_sample_t sample = {})
    {
        Key key { fieldId, order };
        returnCodes[key] = retCode;
        if (retCode == DCGM_ST_OK)
        {
            sampleValues[key] = sample;
        }
    }

    /**
     * Convenience helper to configure ascending and descending GetSamples responses.
     */
    void SetGetSamplesResults(unsigned short fieldId,
                              dcgmReturn_t retCode,
                              dcgmcm_sample_t ascending  = {},
                              dcgmcm_sample_t descending = {})
    {
        SetGetSamplesResult(fieldId, DCGM_ORDER_ASCENDING, retCode, ascending);
        SetGetSamplesResult(fieldId, DCGM_ORDER_DESCENDING, retCode, descending);
    }
};

/**
 * Queue-based mock context for GetSamples. Enqueued responses are consumed in FIFO order per
 * (fieldId, order) key, allowing the same field+order to return different values on successive calls.
 */
struct MockGetSamplesQueueContext
{
    struct Key
    {
        unsigned short fieldId;
        dcgmOrder_t order;

        bool operator<(Key const &other) const
        {
            if (fieldId != other.fieldId)
            {
                return fieldId < other.fieldId;
            }
            return order < other.order;
        }
    };

    struct Response
    {
        dcgmReturn_t retCode;
        dcgmcm_sample_t sample;
    };

    std::map<Key, std::queue<Response>> queues;

    /**
     * Enqueues a GetSamples response for the given field and order.
     * Responses are returned in the order they were enqueued.
     */
    void Enqueue(unsigned short fieldId, dcgmOrder_t order, dcgmReturn_t retCode, dcgmcm_sample_t sample = {})
    {
        queues[Key { fieldId, order }].push(Response { retCode, sample });
    }
};

/**
 * Mock postfunc callback for queue-based GetSamples responses.
 */
static dcgmReturn_t MockCoreCallbackGetSamplesQueue(dcgm_module_command_header_t *header, void *poster)
{
    if (!header)
    {
        return DCGM_ST_BADPARAM;
    }

    if (header->subCommand != DcgmCoreReqIdCMGetSamples)
    {
        return DCGM_ST_OK;
    }

    auto *context = static_cast<MockGetSamplesQueueContext *>(poster);
    auto *gs      = reinterpret_cast<dcgmCoreGetSamples_t *>(header);

    if (!context)
    {
        gs->response.ret        = DCGM_ST_NOT_WATCHED;
        gs->response.numSamples = 0;
        return DCGM_ST_OK;
    }

    MockGetSamplesQueueContext::Key key { gs->request.fieldId, gs->request.order };
    auto queueIt = context->queues.find(key);
    if (queueIt == context->queues.end() || queueIt->second.empty())
    {
        gs->response.ret        = DCGM_ST_NOT_WATCHED;
        gs->response.numSamples = 0;
        return DCGM_ST_OK;
    }

    auto &resp       = queueIt->second.front();
    gs->response.ret = resp.retCode;
    if (resp.retCode == DCGM_ST_OK && gs->response.samples != nullptr && gs->request.maxSamples > 0)
    {
        gs->response.samples[0] = resp.sample;
        gs->response.numSamples = 1;
    }
    else
    {
        gs->response.numSamples = 0;
    }
    queueIt->second.pop();

    return DCGM_ST_OK;
}

/**
 * Mock postfunc callback for GetSamples.
 */
static dcgmReturn_t MockCoreCallbackGetSamples(dcgm_module_command_header_t *header, void *poster)
{
    if (!header)
    {
        return DCGM_ST_BADPARAM;
    }

    if (header->subCommand != DcgmCoreReqIdCMGetSamples)
    {
        return DCGM_ST_OK;
    }

    auto *context = static_cast<MockGetSamplesContext *>(poster);
    auto *gs      = reinterpret_cast<dcgmCoreGetSamples_t *>(header);

    if (!context)
    {
        gs->response.ret        = DCGM_ST_NOT_WATCHED;
        gs->response.numSamples = 0;
        return DCGM_ST_OK;
    }

    MockGetSamplesContext::Key key { gs->request.fieldId, gs->request.order };
    auto retIt = context->returnCodes.find(key);
    if (retIt == context->returnCodes.end())
    {
        gs->response.ret        = DCGM_ST_NOT_WATCHED;
        gs->response.numSamples = 0;
        return DCGM_ST_OK;
    }

    gs->response.ret = retIt->second;
    if (retIt->second != DCGM_ST_OK)
    {
        gs->response.numSamples = 0;
        return DCGM_ST_OK;
    }

    auto sampleIt = context->sampleValues.find(key);
    if (sampleIt != context->sampleValues.end() && gs->response.samples != nullptr && gs->request.maxSamples > 0)
    {
        gs->response.samples[0] = sampleIt->second;
        gs->response.numSamples = 1;
    }
    else
    {
        gs->response.numSamples = 0;
    }

    return DCGM_ST_OK;
}

struct DcgmHealthWatchBerCorePoster
{
    MockGetSamplesContext *mockGetSamples {};
    MockGetLatestSampleContext *mockLatestSample {};
    bool getAllGpuInfoEnabled = false;
    dcgmcm_gpu_info_cached_t singleGpuInfo {};
};

static dcgmReturn_t MockCoreCallbackSamplesLatestGpuInfo(dcgm_module_command_header_t *header, void *poster)
{
    if (!header)
    {
        return DCGM_ST_BADPARAM;
    }

    auto *ctx = static_cast<DcgmHealthWatchBerCorePoster *>(poster);
    if (!ctx)
    {
        return DCGM_ST_BADPARAM;
    }

    switch (header->subCommand)
    {
        case DcgmCoreReqIdCMGetSamples:
            return MockCoreCallbackGetSamples(header, ctx->mockGetSamples);
        case DcgmCoreReqIdCMGetLatestSample:
            return MockCoreCallbackGetLatestSample(header, ctx->mockLatestSample);
        case DcgmCoreReqIdCMGetAllGpuInfo:
        {
            auto *qgi = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
            if (!ctx->getAllGpuInfoEnabled)
            {
                qgi->response.ret           = DCGM_ST_OK;
                qgi->response.infoCount     = 1;
                qgi->response.info[0]       = {};
                qgi->response.info[0].gpuId = 0;
                qgi->response.info[0].arch  = DCGM_CHIP_ARCH_UNKNOWN;
                return DCGM_ST_OK;
            }
            qgi->response.ret       = DCGM_ST_OK;
            qgi->response.infoCount = 1;
            qgi->response.info[0]   = ctx->singleGpuInfo;
            return DCGM_ST_OK;
        }
        default:
            return DCGM_ST_OK;
    }
}

// Test helper class that has friend access to DcgmHealthWatch
class DcgmHealthWatchTestHelper
{
public:
    static void InjectXidForTesting(DcgmHealthWatch &healthWatch, uint64_t xid, dcgm_field_eid_t gpuId)
    {
        DcgmLockGuard dlg(healthWatch.m_mutex);
        healthWatch.m_gpuXidHistory[xid].insert(gpuId);
    }

    static void ClearXidHistoryForTesting(DcgmHealthWatch &healthWatch)
    {
        DcgmLockGuard dlg(healthWatch.m_mutex);
        healthWatch.m_gpuXidHistory.clear();
    }

    static void MonitorDevastatingXids(DcgmHealthWatch &healthWatch,
                                       dcgm_field_entity_group_t entityGroupId,
                                       dcgm_field_eid_t entityId,
                                       DcgmHealthResponse &response)
    {
        healthWatch.MonitorDevastatingXids(entityGroupId, entityId, response);
    }

    static void MonitorSubsystemXids(DcgmHealthWatch &healthWatch,
                                     dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId,
                                     dcgmHealthSystems_t system,
                                     DcgmHealthResponse &response)
    {
        healthWatch.MonitorSubsystemXids(entityGroupId, entityId, system, response);
    }

    static void OnFieldValuesUpdate(DcgmHealthWatch &healthWatch, DcgmFvBuffer *fvBuffer)
    {
        healthWatch.OnFieldValuesUpdate(fvBuffer);
    }

    static dcgmReturn_t MonitorGlobalHealthChecks(DcgmHealthWatch &healthWatch,
                                                  dcgmHealthSystems_t healthSystemsMask,
                                                  DcgmHealthResponse &response)
    {
        return healthWatch.MonitorGlobalHealthChecks(healthSystemsMask, response);
    }

    static bool SystemHasNvLinks(DcgmHealthWatch &healthWatch, std::vector<dcgmGroupEntityPair_t> const &entities)
    {
        return healthWatch.SystemHasNvLinks(entities);
    }

    static void SetGroupWatchState(DcgmHealthWatch &healthWatch, unsigned int groupId, dcgmHealthSystems_t systems)
    {
        DcgmLockGuard dlg(healthWatch.m_mutex);
        healthWatch.mGroupWatchState[groupId] = HealthWatchState { systems, 0, 30000000, 600.0 };
    }

    static dcgmReturn_t MonitorWatches(DcgmHealthWatch &healthWatch, unsigned int groupId, DcgmHealthResponse &response)
    {
        return healthWatch.MonitorWatches(groupId, 0, 0, response);
    }

    static dcgmReturn_t MonitorConnectX(DcgmHealthWatch &healthWatch,
                                        dcgm_field_entity_group_t entityGroupId,
                                        dcgm_field_eid_t entityId,
                                        DcgmHealthResponse &response)
    {
        return healthWatch.MonitorConnectX(entityGroupId, entityId, response);
    }

    static dcgmReturn_t MonitorNVLinkErrorFields(DcgmHealthWatch &healthWatch,
                                                 dcgm_field_entity_group_t entityGroupId,
                                                 dcgm_field_eid_t entityId,
                                                 DcgmHealthResponse &response)
    {
        return healthWatch.MonitorNVLinkErrorFields(entityGroupId, entityId, response);
    }

    static dcgmReturn_t MonitorFabricFields(DcgmHealthWatch &healthWatch,
                                            dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId,
                                            DcgmHealthResponse &response)
    {
        return healthWatch.MonitorFabricFields(entityGroupId, entityId, 0, 0, response);
    }

    static dcgmReturn_t MonitorNVLink4Fields(DcgmHealthWatch &healthWatch,
                                             dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId,
                                             long long startTime,
                                             long long endTime,
                                             DcgmHealthResponse &response)
    {
        return healthWatch.MonitorNVLink4Fields(entityGroupId, entityId, startTime, endTime, response);
    }

    static dcgmReturn_t MonitorNVLink5Fields(DcgmHealthWatch &healthWatch,
                                             dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId,
                                             long long startTime,
                                             long long endTime,
                                             DcgmHealthResponse &response)
    {
        return healthWatch.MonitorNVLink5Fields(entityGroupId, entityId, startTime, endTime, response);
    }

    static dcgmReturn_t MonitorMemVolatileDbes(DcgmHealthWatch &healthWatch,
                                               dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               long long startTime,
                                               long long endTime,
                                               DcgmHealthResponse &response)
    {
        return healthWatch.MonitorMemVolatileDbes(entityGroupId, entityId, startTime, endTime, response);
    }

    static dcgmReturn_t MonitorMemRetiredPending(DcgmHealthWatch &healthWatch,
                                                 dcgm_field_entity_group_t entityGroupId,
                                                 dcgm_field_eid_t entityId,
                                                 long long startTime,
                                                 long long endTime,
                                                 DcgmHealthResponse &response)
    {
        return healthWatch.MonitorMemRetiredPending(entityGroupId, entityId, startTime, endTime, response);
    }

    static dcgmReturn_t MonitorMemRowRemapFailures(DcgmHealthWatch &healthWatch,
                                                   dcgm_field_entity_group_t entityGroupId,
                                                   dcgm_field_eid_t entityId,
                                                   long long startTime,
                                                   long long endTime,
                                                   DcgmHealthResponse &response)
    {
        return healthWatch.MonitorMemRowRemapFailures(entityGroupId, entityId, startTime, endTime, response);
    }

    static dcgmReturn_t MonitorMemRowRemapUncorrectable(DcgmHealthWatch &healthWatch,
                                                        dcgm_field_entity_group_t entityGroupId,
                                                        dcgm_field_eid_t entityId,
                                                        DcgmHealthResponse &response)
    {
        return healthWatch.MonitorMemRowRemapUncorrectable(entityGroupId, entityId, response);
    }

    static int GetComputeCapabilityAmpere()
    {
        return DcgmHealthWatch::computeCapabilityAmpere;
    }

    static dcgmReturn_t MonitorMemUnrepairableFlag(DcgmHealthWatch &healthWatch,
                                                   dcgm_field_entity_group_t entityGroupId,
                                                   dcgm_field_eid_t entityId,
                                                   DcgmHealthResponse &response)
    {
        return healthWatch.MonitorMemUnrepairableFlag(entityGroupId, entityId, response);
    }

    static dcgmReturn_t MonitorMem(DcgmHealthWatch &healthWatch,
                                   dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId,
                                   long long startTime,
                                   long long endTime,
                                   DcgmHealthResponse &response)
    {
        return healthWatch.MonitorMem(entityGroupId, entityId, startTime, endTime, response);
    }

    static dcgmReturn_t MonitorGpuRecoveryAction(DcgmHealthWatch &healthWatch,
                                                 dcgm_field_entity_group_t entityGroupId,
                                                 dcgm_field_eid_t entityId,
                                                 DcgmHealthResponse &response)
    {
        return healthWatch.MonitorGpuRecoveryAction(entityGroupId, entityId, response);
    }

    static dcgmReturn_t MonitorMemSbeDbeRetiredPages(DcgmHealthWatch &healthWatch,
                                                     dcgm_field_entity_group_t entityGroupId,
                                                     dcgm_field_eid_t entityId,
                                                     long long startTime,
                                                     long long endTime,
                                                     DcgmHealthResponse &response)
    {
        return healthWatch.MonitorMemSbeDbeRetiredPages(entityGroupId, entityId, startTime, endTime, response);
    }

    static dcgmReturn_t MonitorNvSwitchErrorCounts(DcgmHealthWatch &healthWatch,
                                                   bool fatal,
                                                   dcgm_field_entity_group_t entityGroupId,
                                                   dcgm_field_eid_t entityId,
                                                   long long startTime,
                                                   long long endTime,
                                                   DcgmHealthResponse &response)
    {
        return healthWatch.MonitorNvSwitchErrorCounts(fatal, entityGroupId, entityId, startTime, endTime, response);
    }

    static dcgmReturn_t GetExpectedPcieReplayRate(DcgmHealthWatch &hw,
                                                  dcgm_field_entity_group_t eg,
                                                  dcgm_field_eid_t eid,
                                                  int &rate)
    {
        return hw.GetExpectedPcieReplayRate(eg, eid, rate);
    }

    static dcgmReturn_t MonitorPcie(DcgmHealthWatch &hw,
                                    dcgm_field_entity_group_t eg,
                                    dcgm_field_eid_t eid,
                                    long long s,
                                    long long e,
                                    DcgmHealthResponse &r)
    {
        return hw.MonitorPcie(eg, eid, s, e, r);
    }

    static dcgmReturn_t MonitorThermal(DcgmHealthWatch &hw,
                                       dcgm_field_entity_group_t eg,
                                       dcgm_field_eid_t eid,
                                       long long s,
                                       long long e,
                                       DcgmHealthResponse &r)
    {
        return hw.MonitorThermal(eg, eid, s, e, r);
    }

    static dcgmReturn_t MonitorPower(DcgmHealthWatch &hw,
                                     dcgm_field_entity_group_t eg,
                                     dcgm_field_eid_t eid,
                                     long long s,
                                     long long e,
                                     DcgmHealthResponse &r)
    {
        return hw.MonitorPower(eg, eid, s, e, r);
    }

    static dcgmReturn_t MonitorCpuThermal(DcgmHealthWatch &hw,
                                          dcgm_field_entity_group_t eg,
                                          dcgm_field_eid_t eid,
                                          long long s,
                                          long long e,
                                          DcgmHealthResponse &r)
    {
        return hw.MonitorCpuThermal(eg, eid, s, e, r);
    }

    static dcgmReturn_t MonitorCpuPower(DcgmHealthWatch &hw,
                                        dcgm_field_entity_group_t eg,
                                        dcgm_field_eid_t eid,
                                        DcgmHealthResponse &r)
    {
        return hw.MonitorCpuPower(eg, eid, 0, 0, r);
    }

    static dcgmReturn_t MonitorInforom(DcgmHealthWatch &hw,
                                       dcgm_field_entity_group_t eg,
                                       dcgm_field_eid_t eid,
                                       DcgmHealthResponse &r)
    {
        return hw.MonitorInforom(eg, eid, 0, 0, r);
    }

    static bool IsGpuMigEnabled(DcgmHealthWatch &hw, dcgm_field_eid_t gpuId)
    {
        return hw.IsGpuMigEnabled(gpuId);
    }

    static dcgmReturn_t MonitorNVLinkStatus(DcgmHealthWatch &hw,
                                            dcgm_field_entity_group_t eg,
                                            dcgm_field_eid_t eid,
                                            DcgmHealthResponse &r)
    {
        return hw.MonitorNVLinkStatus(eg, eid, r);
    }

    static dcgmReturn_t MonitorNVLink(DcgmHealthWatch &hw,
                                      dcgm_field_entity_group_t eg,
                                      dcgm_field_eid_t eid,
                                      long long s,
                                      long long e,
                                      DcgmHealthResponse &r)
    {
        return hw.MonitorNVLink(eg, eid, s, e, r);
    }
};

// Test fixture class for common setup
class DcgmHealthWatchFixture
{
public:
    MockGetLatestSampleContext mockLatestSample;
    dcgmCoreCallbacks_t callbacks;
    DcgmHealthWatch healthWatch;
    DcgmHealthResponse response;
    dcgmHealthResponse_v5 healthResponse              = {};
    const dcgm_field_eid_t testGpuId                  = 0;
    const dcgm_field_entity_group_t testEntityGroupId = DCGM_FE_GPU;

    DcgmHealthWatchFixture()
        : callbacks(createCallbacks())
        , healthWatch(callbacks)
    {
        // Clear any previous state and reset response
        DcgmHealthWatchTestHelper::ClearXidHistoryForTesting(healthWatch);
        response = DcgmHealthResponse();
    }

private:
    dcgmCoreCallbacks_t createCallbacks()
    {
        dcgmCoreCallbacks_t cb = {};
        cb.version             = dcgmCoreCallbacks_version;
        cb.postfunc            = MockCoreCallbackGetLatestSample;
        cb.poster              = &mockLatestSample; // Pass context for mock to use
        cb.loggerfunc          = [](void const *) { /* do nothing */ };
        return cb;
    }
};

class DcgmHealthWatchGetSamplesFixture
{
public:
    MockGetSamplesContext mockGetSamples;
    MockGetLatestSampleContext mockLatestSample;
    DcgmHealthWatchBerCorePoster corePoster;
    dcgmCoreCallbacks_t callbacks;
    DcgmHealthWatch healthWatch;
    DcgmHealthResponse response;
    dcgmHealthResponse_v5 healthResponse              = {};
    const dcgm_field_eid_t testGpuId                  = 0;
    const dcgm_field_entity_group_t testEntityGroupId = DCGM_FE_GPU;

    DcgmHealthWatchGetSamplesFixture()
        : callbacks(createCallbacks())
        , healthWatch(callbacks)
    {
        DcgmHealthWatchTestHelper::ClearXidHistoryForTesting(healthWatch);
        response = DcgmHealthResponse();
    }

private:
    dcgmCoreCallbacks_t createCallbacks()
    {
        corePoster.mockGetSamples       = &mockGetSamples;
        corePoster.mockLatestSample     = &mockLatestSample;
        corePoster.getAllGpuInfoEnabled = false;
        dcgmCoreCallbacks_t cb          = {};
        cb.version                      = dcgmCoreCallbacks_version;
        cb.postfunc                     = MockCoreCallbackSamplesLatestGpuInfo;
        cb.poster                       = &corePoster;
        cb.loggerfunc                   = [](void const *) { /* do nothing */ };
        return cb;
    }
};

/**
 * Combined mock context for tests that need both GetSamples and GetEntityNvLinkLinkStatus responses.
 */
struct MockNvSwitchContext
{
    MockGetSamplesContext getSamples;
    dcgmNvLinkLinkState_t linkStates[DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH] = {};
    dcgmReturn_t linkStatusRet                                           = DCGM_ST_OK;
};

/**
 * Mock postfunc callback that handles both GetSamples and GetEntityNvLinkLinkStatus.
 */
static dcgmReturn_t MockCoreCallbackNvSwitch(dcgm_module_command_header_t *header, void *poster)
{
    if (!header)
    {
        return DCGM_ST_BADPARAM;
    }

    auto *context = static_cast<MockNvSwitchContext *>(poster);

    if (header->subCommand == DcgmCoreReqIdCMGetSamples)
    {
        return MockCoreCallbackGetSamples(header, context ? &context->getSamples : nullptr);
    }

    if (header->subCommand == DcgmCoreReqIdCMGetEntityNvLinkLinkStatus)
    {
        auto *nls         = reinterpret_cast<dcgmCoreGetEntityNvLinkLinkStatus_t *>(header);
        nls->response.ret = context ? context->linkStatusRet : DCGM_ST_OK;
        if (context && context->linkStatusRet == DCGM_ST_OK)
        {
            memcpy(nls->response.linkStates, context->linkStates, sizeof(nls->response.linkStates));
        }
        return DCGM_ST_OK;
    }

    return DCGM_ST_OK;
}

class DcgmHealthWatchNvSwitchFixture
{
public:
    MockNvSwitchContext mockNvSwitch;
    dcgmCoreCallbacks_t callbacks;
    DcgmHealthWatch healthWatch;
    DcgmHealthResponse response;
    dcgmHealthResponse_v5 healthResponse              = {};
    const dcgm_field_eid_t testSwitchId               = 0;
    const dcgm_field_entity_group_t testEntityGroupId = DCGM_FE_SWITCH;

    DcgmHealthWatchNvSwitchFixture()
        : callbacks(createCallbacks())
        , healthWatch(callbacks)
    {
        DcgmHealthWatchTestHelper::ClearXidHistoryForTesting(healthWatch);
        response = DcgmHealthResponse();
    }

private:
    dcgmCoreCallbacks_t createCallbacks()
    {
        dcgmCoreCallbacks_t cb = {};
        cb.version             = dcgmCoreCallbacks_version;
        cb.postfunc            = MockCoreCallbackNvSwitch;
        cb.poster              = &mockNvSwitch;
        cb.loggerfunc          = [](void const *) { /* do nothing */ };
        return cb;
    }
};

class DcgmHealthWatchMemRetiredPagesFixture
{
public:
    MockGetSamplesQueueContext mockGetSamples;
    dcgmCoreCallbacks_t callbacks;
    DcgmHealthWatch healthWatch;
    DcgmHealthResponse response;
    dcgmHealthResponse_v5 healthResponse              = {};
    const dcgm_field_eid_t testGpuId                  = 0;
    const dcgm_field_entity_group_t testEntityGroupId = DCGM_FE_GPU;

    DcgmHealthWatchMemRetiredPagesFixture()
        : callbacks(createCallbacks())
        , healthWatch(callbacks)
    {
        DcgmHealthWatchTestHelper::ClearXidHistoryForTesting(healthWatch);
        response = DcgmHealthResponse();
    }

    /**
     * Enqueues DBE (descending) then SBE (descending) responses with the given page counts.
     * A blank value (DCGM_INT64_BLANK) can be passed to simulate a blank sample.
     */
    void SetDbeAndSbe(int64_t dbePages, int64_t sbePages)
    {
        dcgmcm_sample_t dbeSample = {};
        dbeSample.val.i64         = dbePages;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeSample);

        dcgmcm_sample_t sbeSample = {};
        sbeSample.val.i64         = sbePages;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sbeSample);
    }

private:
    dcgmCoreCallbacks_t createCallbacks()
    {
        dcgmCoreCallbacks_t cb = {};
        cb.version             = dcgmCoreCallbacks_version;
        cb.postfunc            = MockCoreCallbackGetSamplesQueue;
        cb.poster              = &mockGetSamples;
        cb.loggerfunc          = [](void const *) { /* do nothing */ };
        return cb;
    }
};

/**
 * Combined mock context for GPU NVLink status tests. Handles both
 * GetEntityNvLinkLinkStatus (for MonitorNVLinkStatus) and GetLatestSample
 * (for IsGpuMigEnabled).
 */
struct MockGpuNvLinkContext
{
    MockGetLatestSampleContext latestSample;
    dcgmNvLinkLinkState_t linkStates[DCGM_NVLINK_MAX_LINKS_PER_GPU] = {};
    dcgmReturn_t linkStatusRet                                      = DCGM_ST_OK;
};

static dcgmReturn_t MockCoreCallbackGpuNvLink(dcgm_module_command_header_t *header, void *poster)
{
    if (!header)
        return DCGM_ST_BADPARAM;

    auto *ctx = static_cast<MockGpuNvLinkContext *>(poster);

    if (header->subCommand == DcgmCoreReqIdCMGetEntityNvLinkLinkStatus)
    {
        auto *nls         = reinterpret_cast<dcgmCoreGetEntityNvLinkLinkStatus_t *>(header);
        nls->response.ret = ctx ? ctx->linkStatusRet : DCGM_ST_OK;
        if (ctx && ctx->linkStatusRet == DCGM_ST_OK)
        {
            for (int i = 0; i < DCGM_NVLINK_MAX_LINKS_PER_GPU; ++i)
                nls->response.linkStates[i] = ctx->linkStates[i];
        }
        return DCGM_ST_OK;
    }

    if (header->subCommand == DcgmCoreReqIdCMGetLatestSample)
        return MockCoreCallbackGetLatestSample(header, ctx ? &ctx->latestSample : nullptr);

    return DCGM_ST_OK;
}

class DcgmHealthWatchGpuNvLinkFixture
{
public:
    MockGpuNvLinkContext mockCtx;
    dcgmCoreCallbacks_t callbacks;
    DcgmHealthWatch healthWatch;
    DcgmHealthResponse response;
    dcgmHealthResponse_v5 healthResponse              = {};
    const dcgm_field_eid_t testGpuId                  = 0;
    const dcgm_field_entity_group_t testEntityGroupId = DCGM_FE_GPU;

    DcgmHealthWatchGpuNvLinkFixture()
        : callbacks(createCallbacks())
        , healthWatch(callbacks)
    {
        DcgmHealthWatchTestHelper::ClearXidHistoryForTesting(healthWatch);
        response = DcgmHealthResponse();
    }

private:
    dcgmCoreCallbacks_t createCallbacks()
    {
        dcgmCoreCallbacks_t cb = {};
        cb.version             = dcgmCoreCallbacks_version;
        cb.postfunc            = MockCoreCallbackGpuNvLink;
        cb.poster              = &mockCtx;
        cb.loggerfunc          = [](void const *) { /* do nothing */ };
        return cb;
    }
};

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorDevastatingXids")
{
    SECTION("No devastating XIDs detected")
    {
        // Don't inject any XIDs
        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify no incidents were added
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("Multiple devastating XIDs detected")
    {
        // Inject multiple devastating XIDs for the test GPU
        DcgmHealthWatchTestHelper::InjectXidForTesting(healthWatch, 48, testGpuId);  // Double Bit ECC Error
        DcgmHealthWatchTestHelper::InjectXidForTesting(healthWatch, 95, testGpuId);  // Uncontained Error
        DcgmHealthWatchTestHelper::InjectXidForTesting(healthWatch, 140, testGpuId); // ECC unrecovered error

        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify incidents were added
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 3);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL);

        // All incidents should be for the same GPU
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            REQUIRE(healthResponse.incidents[i].system == DCGM_HEALTH_WATCH_ALL);
            REQUIRE(healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_FAIL);
            REQUIRE(healthResponse.incidents[i].entityInfo.entityGroupId == testEntityGroupId);
            REQUIRE(healthResponse.incidents[i].entityInfo.entityId == testGpuId);
        }
    }

    SECTION("Non Devastating XIDs - not detected in response")
    {
        // Inject multiple devastating XIDs for the test GPU
        DcgmHealthWatchTestHelper::InjectXidForTesting(healthWatch, 31, testGpuId);

        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify incidents were added
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::Complete XID Flow Test")
{
    SECTION("Complete flow: Devastating XID processing and monitoring")
    {
        // Set up health monitoring for the group
        dcgmHealthSystems_t systems = DCGM_HEALTH_WATCH_ALL;
        dcgmReturn_t ret            = healthWatch.SetWatches(1, systems, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        // Create a field value buffer with XID data
        DcgmFvBuffer fvBuffer;
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERROR,
                               48,
                               timelib_usecSince1970(),
                               DCGM_ST_OK); // Double Bit ECC Error - devastating XID

        // Process the field value update and monitor
        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify the response
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_ALL);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityGroupId == testEntityGroupId);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityId == testGpuId);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_XID_ERROR);
    }

    SECTION("Complete flow: Subsystem XID processing and monitoring")
    {
        struct Row
        {
            char const *name;
            dcgmHealthSystems_t system;
            int64_t xid;
            char const *msgSubstring; //!< Must appear somewhere in the incident message
        };

        auto r = GENERATE(Row { "MEM XID 31", DCGM_HEALTH_WATCH_MEM, 31, "31" },
                          Row { "PCIe XID 42", DCGM_HEALTH_WATCH_PCIE, 42, "PCIe Replay Error" },
                          Row { "Thermal XID 62", DCGM_HEALTH_WATCH_THERMAL, 62, "Thermal Violation" },
                          Row { "NVLink XID 67", DCGM_HEALTH_WATCH_NVLINK, 67, "NVLink Error" });

        CAPTURE(r.name);

        DcgmHealthWatchTestHelper::ClearXidHistoryForTesting(healthWatch);
        response = DcgmHealthResponse();

        dcgmReturn_t ret = healthWatch.SetWatches(1, r.system, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        DcgmFvBuffer fvBuffer;
        fvBuffer.AddInt64Value(
            DCGM_FE_GPU, testGpuId, DCGM_FI_DEV_XID_ERROR, r.xid, timelib_usecSince1970(), DCGM_ST_OK);

        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorSubsystemXids(healthWatch, testEntityGroupId, testGpuId, r.system, response);

        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == r.system);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityGroupId == testEntityGroupId);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityId == testGpuId);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_XID_ERROR);

        std::string_view msg = healthResponse.incidents[0].error.msg;
        REQUIRE(msg.find(r.msgSubstring) != std::string_view::npos);
        REQUIRE(msg.find("triggered by XID") != std::string_view::npos);
    }

    SECTION("Complete flow: Multiple XIDs processing")
    {
        // Set up health monitoring for all systems
        dcgmHealthSystems_t systems
            = static_cast<dcgmHealthSystems_t>(DCGM_HEALTH_WATCH_ALL | DCGM_HEALTH_WATCH_MEM | DCGM_HEALTH_WATCH_PCIE);
        dcgmReturn_t ret = healthWatch.SetWatches(1, systems, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        // Create field value buffer with multiple XIDs
        DcgmFvBuffer fvBuffer;

        // Add devastating XID
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERROR,
                               48,
                               timelib_usecSince1970(),
                               DCGM_ST_OK); // Double Bit ECC Error

        // Add memory subsystem XID
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERROR,
                               31,
                               timelib_usecSince1970() + 1000,
                               DCGM_ST_OK); // MMU Error

        // Add PCIe subsystem XID
        fvBuffer.AddInt64Value(DCGM_FE_GPU,
                               testGpuId,
                               DCGM_FI_DEV_XID_ERROR,
                               38,
                               timelib_usecSince1970() + 2000,
                               DCGM_ST_OK); // PCIe Bus Error

        // Process all field value updates and monitor
        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Monitor subsystem XIDs
        DcgmHealthWatchTestHelper::MonitorSubsystemXids(
            healthWatch, testEntityGroupId, testGpuId, DCGM_HEALTH_WATCH_MEM, response);
        DcgmHealthWatchTestHelper::MonitorSubsystemXids(
            healthWatch, testEntityGroupId, testGpuId, DCGM_HEALTH_WATCH_PCIE, response);

        // Verify the response
        response.PopulateHealthResponse(healthResponse);

        // Should have 3 incidents: 1 devastating + 2 subsystem
        REQUIRE(healthResponse.incidentCount == 3);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL); // FAIL overrides WARN

        // Verify all incidents are for the same GPU
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            REQUIRE(healthResponse.incidents[i].entityInfo.entityGroupId == testEntityGroupId);
            REQUIRE(healthResponse.incidents[i].entityInfo.entityId == testGpuId);
        }
    }

    SECTION("Complete flow: Unknown XID ignored")
    {
        // Set up health monitoring
        dcgmHealthSystems_t systems = DCGM_HEALTH_WATCH_ALL;
        dcgmReturn_t ret            = healthWatch.SetWatches(1, systems, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        // Create field value buffer with unknown XID
        DcgmFvBuffer fvBuffer;
        fvBuffer.AddInt64Value(
            DCGM_FE_GPU, testGpuId, DCGM_FI_DEV_XID_ERROR, 999, timelib_usecSince1970(), DCGM_ST_OK); // Unknown XID

        // Process the field value update and monitor
        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorDevastatingXids(healthWatch, testEntityGroupId, testGpuId, response);

        // Verify no incidents were added
        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }
}

/**
 * XID 64 (row remapping failure) must produce DCGM_HEALTH_RESULT_FAIL, not WARN.
 * NVIDIA's XID catalog and GPU Debug Guidelines both mandate immediate GPU reset or host reboot
 * for this event, so both the XID path and the field (DCGM_FI_DEV_ROW_REMAP_FAILED) path must agree.
 */
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::XID 64 Row Remap Failure is FAIL not WARN")
{
    SECTION("XID 64 via subsystem XID path produces FAIL")
    {
        dcgmReturn_t ret = healthWatch.SetWatches(1, DCGM_HEALTH_WATCH_MEM, 0, 1000000, 3600.0);
        REQUIRE(ret == DCGM_ST_OK);

        DcgmFvBuffer fvBuffer;
        fvBuffer.AddInt64Value(DCGM_FE_GPU, testGpuId, DCGM_FI_DEV_XID_ERROR, 64, timelib_usecSince1970(), DCGM_ST_OK);

        DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);
        DcgmHealthWatchTestHelper::MonitorSubsystemXids(
            healthWatch, testEntityGroupId, testGpuId, DCGM_HEALTH_WATCH_MEM, response);

        response.PopulateHealthResponse(healthResponse);

        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_ROW_REMAP_FAILURE);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityGroupId == testEntityGroupId);
        REQUIRE(healthResponse.incidents[0].entityInfo.entityId == testGpuId);
    }
}

/**
 * Tests that XID 93 (corrupt InfoROM) is only raised on Volta GPUs.
 */
TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::XID 93 InfoROM Volta-only filter")
{
    struct TestCase
    {
        char const *name;
        dcgmChipArchitecture_t arch;
        bool shouldReport;
    };

    constexpr std::array testCases {
        TestCase { "Volta: XID 93 must be reported", DCGM_CHIP_ARCH_VOLTA, true },
        TestCase { "Turing: XID 93 must be ignored", DCGM_CHIP_ARCH_TURING, false },
        TestCase { "Ampere: XID 93 must be ignored", DCGM_CHIP_ARCH_AMPERE, false },
        TestCase { "Hopper: XID 93 must be ignored", DCGM_CHIP_ARCH_HOPPER, false },
        TestCase { "Blackwell: XID 93 must be ignored", DCGM_CHIP_ARCH_BLACKWELL, false },
    };

    auto const &tc = GENERATE_REF(from_range(testCases));
    CAPTURE(tc.name);

    corePoster.getAllGpuInfoEnabled = true;
    corePoster.singleGpuInfo        = {};
    corePoster.singleGpuInfo.gpuId  = testGpuId;
    corePoster.singleGpuInfo.arch   = tc.arch;

    dcgmReturn_t ret = healthWatch.SetWatches(1, DCGM_HEALTH_WATCH_INFOROM, 0, 1000000, 3600.0);
    REQUIRE(ret == DCGM_ST_OK);

    DcgmHealthWatchTestHelper::ClearXidHistoryForTesting(healthWatch);
    response = DcgmHealthResponse();

    DcgmFvBuffer fvBuffer;
    fvBuffer.AddInt64Value(DCGM_FE_GPU, testGpuId, DCGM_FI_DEV_XID_ERROR, 93, timelib_usecSince1970(), DCGM_ST_OK);
    DcgmHealthWatchTestHelper::OnFieldValuesUpdate(healthWatch, &fvBuffer);

    DcgmHealthWatchTestHelper::MonitorSubsystemXids(
        healthWatch, testEntityGroupId, testGpuId, DCGM_HEALTH_WATCH_INFOROM, response);

    response.PopulateHealthResponse(healthResponse);

    if (tc.shouldReport)
    {
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_INFOROM);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_CORRUPT_INFOROM);
    }
    else
    {
        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }
}

/**
 * Tests for MonitorGlobalHealthChecks
 *
 * Verifies that global health checks are correctly dispatched based on the health systems mask.
 * Since MonitorGlobalHealthChecks trusts the mask it receives (filtering is done by the caller),
 * these tests verify mask-based dispatch only.
 */
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorGlobalHealthChecks dispatch")
{
    struct TestCase
    {
        char const *name;
        dcgmHealthSystems_t mask;
    };

    auto test = GENERATE(TestCase { "NVLINK watch disabled - skips MonitorImex", DCGM_HEALTH_WATCH_MEM },
                         TestCase { "NVLINK watch enabled, no cache data - returns OK", DCGM_HEALTH_WATCH_NVLINK },
                         TestCase { "No watches enabled - no checks performed", static_cast<dcgmHealthSystems_t>(0) });

    DYNAMIC_SECTION(test.name)
    {
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorGlobalHealthChecks(healthWatch, test.mask, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }
}

/**
 * Tests for SystemHasNvLinks
 *
 * Verifies that SystemHasNvLinks checks only the GPUs in the provided entity list.
 */
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::SystemHasNvLinks scoped to entities")
{
    struct TestCase
    {
        char const *name;
        unsigned int numGpus;
        bool withNvLinks;
        std::vector<dcgmGroupEntityPair_t> entityOverride;
        bool useOverride;
        bool expected;
    };

    auto test = GENERATE(TestCase { "Returns true when group GPU has NVLinks", 1, true, {}, false, true },
                         TestCase { "Returns false when group GPUs have no NVLinks", 4, false, {}, false, false },
                         TestCase { "Returns false for empty entity list", 0, false, {}, true, false },
                         TestCase { "Ignores non-GPU entities", 1, true, { { DCGM_FE_SWITCH, 0 } }, true, false });

    DYNAMIC_SECTION(test.name)
    {
        if (test.numGpus > 0)
        {
            mockLatestSample.SetupGpuNvLinks(test.numGpus, test.withNvLinks);
        }

        auto entities = test.useOverride ? test.entityOverride : mockLatestSample.GetEntities();
        REQUIRE(DcgmHealthWatchTestHelper::SystemHasNvLinks(healthWatch, entities) == test.expected);
    }
}

namespace DcgmNs::ImexTests
{
/**
 * Configures IMEX domain status field.
 *
 * @param ctx MockGetLatestSampleContext to configure
 * @param status Domain status string (e.g., "UP", "DOWN", "DEGRADED", "NOT_INSTALLED")
 */
void SetImexDomainStatus(MockGetLatestSampleContext &ctx, char const *status)
{
    ctx.AddManagedString(status);

    dcgmcm_sample_t sample = {};
    sample.timestamp       = timelib_usecSince1970();
    sample.val.str         = const_cast<char *>(ctx.ManagedString().data());

    ctx.SetFieldSample(DCGM_FI_IMEX_DOMAIN_STATUS, DCGM_ST_OK, sample);
}

/**
 * Configures IMEX daemon status field.
 *
 * @param ctx MockGetLatestSampleContext to configure
 * @param status Daemon status value (DcgmImexDaemonStatus enum or integral type for invalid values)
 */
template <typename T>
    requires std::same_as<T, DcgmImexDaemonStatus> || std::integral<T>
void SetImexDaemonStatus(MockGetLatestSampleContext &ctx, T status)
{
    dcgmcm_sample_t sample = {};
    sample.timestamp       = timelib_usecSince1970();
    sample.val.i64         = static_cast<int64_t>(status);
    ctx.SetFieldSample(DCGM_FI_IMEX_DAEMON_STATUS, DCGM_ST_OK, sample);
}

/**
 * Helper to run MonitorGlobalHealthChecks and validate common response patterns.
 *
 * @param healthWatch Health watch instance
 * @param response Response object to populate
 * @param expectedRet Expected return code from MonitorGlobalHealthChecks
 * @param expectedIncidents Expected number of incidents (only checked if expectedRet == DCGM_ST_OK)
 * @param expectedHealth Expected overall health result (only checked if expectedRet == DCGM_ST_OK)
 * @param expectedErrorCode Expected error code for first incident (only checked if expectedIncidents > 0)
 */
void ValidateImexResponse(DcgmHealthWatch &healthWatch,
                          DcgmHealthResponse &response,
                          dcgmReturn_t expectedRet,
                          unsigned int expectedIncidents          = 0,
                          dcgmHealthWatchResults_t expectedHealth = DCGM_HEALTH_RESULT_PASS,
                          dcgmError_t expectedErrorCode           = DCGM_FR_OK)
{
    dcgmReturn_t ret
        = DcgmHealthWatchTestHelper::MonitorGlobalHealthChecks(healthWatch, DCGM_HEALTH_WATCH_NVLINK, response);

    REQUIRE(ret == expectedRet);

    if (expectedRet == DCGM_ST_OK)
    {
        dcgmHealthResponse_v5 healthResponse = {};
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == expectedIncidents);
        REQUIRE(healthResponse.overallHealth == expectedHealth);

        if (expectedIncidents > 0 && expectedErrorCode != DCGM_FR_OK)
        {
            REQUIRE(healthResponse.incidents[0].error.code == expectedErrorCode);
        }
    }
}

/**
 * Test case structure for IMEX health checks
 */
struct ImexTest
{
    char const *name;
    char const *domain                    = "UP";
    int64_t daemon                        = static_cast<int64_t>(DcgmImexDaemonStatus::READY);
    std::optional<dcgmReturn_t> domainErr = std::nullopt;
    std::optional<dcgmReturn_t> daemonErr = std::nullopt;
    bool expectFail                       = false;
    dcgmReturn_t expectRet                = DCGM_ST_OK;
};

/**
 * Factory function for PASS test cases.
 */
auto Pass(char const *name, char const *domain, int64_t daemon) -> ImexTest
{
    return { name, domain, daemon };
}

/**
 * Factory function for FAIL test cases (expects DCGM_FR_IMEX_UNHEALTHY).
 */
auto Fail(char const *name, char const *domain, int64_t daemon) -> ImexTest
{
    return { name, domain, daemon, std::nullopt, std::nullopt, true };
}
} //namespace DcgmNs::ImexTests

/**
 * Tests for MonitorImex logic.
 */
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorImex")
{
    using namespace DcgmNs::ImexTests;

    // All IMEX tests require NVLinks to be present so MonitorImex is not skipped
    mockLatestSample.SetupGpuNvLinks(1, true);

    SECTION("Domain unhealthy and daemon returns error")
    {
        SetImexDomainStatus(mockLatestSample, "DOWN");
        mockLatestSample.SetFieldSample(DCGM_FI_IMEX_DAEMON_STATUS, DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGlobalHealthChecks(healthWatch, DCGM_HEALTH_WATCH_NVLINK, response);

        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
        // Need to verify incident was recorded despite error return
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
    }

    auto test
        = GENERATE(Pass("Healthy IMEX state", "UP", static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Fail("Domain DOWN", "DOWN", static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Fail("Domain DEGRADED", "DEGRADED", static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Fail("Daemon INITIALIZING", "UP", static_cast<int64_t>(DcgmImexDaemonStatus::INITIALIZING)),
                   Pass("NOT_INSTALLED", "NOT_INSTALLED", static_cast<int64_t>(DcgmImexDaemonStatus::NOT_INSTALLED)),
                   Pass("UNAVAILABLE", "UNAVAILABLE", static_cast<int64_t>(DcgmImexDaemonStatus::UNAVAILABLE)),
                   Pass("NOT_CONFIGURED", "NOT_CONFIGURED", static_cast<int64_t>(DcgmImexDaemonStatus::NOT_CONFIGURED)),
                   Fail("Invalid domain status", "INVALID_GARBAGE", static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Fail("Invalid daemon status", "UP", 999),
                   Pass("Blank domain status", DCGM_STR_BLANK, static_cast<int64_t>(DcgmImexDaemonStatus::READY)),
                   Pass("Blank daemon status", "UP", DCGM_INT64_BLANK),
                   ImexTest { .name      = "Error retrieving domain",
                              .daemon    = static_cast<int64_t>(DcgmImexDaemonStatus::READY),
                              .domainErr = DCGM_ST_GENERIC_ERROR,
                              .expectRet = DCGM_ST_GENERIC_ERROR },
                   ImexTest { .name      = "Error retrieving daemon",
                              .domain    = "UP",
                              .daemonErr = DCGM_ST_GENERIC_ERROR,
                              .expectRet = DCGM_ST_GENERIC_ERROR },
                   ImexTest { .name      = "NO_DATA for domain",
                              .daemon    = static_cast<int64_t>(DcgmImexDaemonStatus::READY),
                              .domainErr = DCGM_ST_NO_DATA },
                   ImexTest { .name = "NO_DATA for daemon", .domain = "UP", .daemonErr = DCGM_ST_NO_DATA });

    DYNAMIC_SECTION(test.name)
    {
        if (test.domainErr)
        {
            mockLatestSample.SetFieldSample(DCGM_FI_IMEX_DOMAIN_STATUS, *test.domainErr);
        }
        else
        {
            SetImexDomainStatus(mockLatestSample, test.domain);
        }

        if (test.daemonErr)
        {
            mockLatestSample.SetFieldSample(DCGM_FI_IMEX_DAEMON_STATUS, *test.daemonErr);
        }
        else
        {
            SetImexDaemonStatus(mockLatestSample, test.daemon);
        }

        if (test.expectRet != DCGM_ST_OK)
        {
            ValidateImexResponse(healthWatch, response, test.expectRet);
        }
        else if (test.expectFail)
        {
            ValidateImexResponse(healthWatch, response, DCGM_ST_OK, 1, DCGM_HEALTH_RESULT_FAIL, DCGM_FR_IMEX_UNHEALTHY);
        }
        else
        {
            ValidateImexResponse(healthWatch, response, DCGM_ST_OK, 0, DCGM_HEALTH_RESULT_PASS);
        }
    }
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorImex skipped without NVLinks")
{
    // Simulate a system with GPUs but no NVLinks (like RTX PRO 6000 with PHB topology)
    mockLatestSample.SetupGpuNvLinks(4, false);

    // Configure IMEX to report COMMAND_ERROR (the customer's actual scenario)
    DcgmNs::ImexTests::SetImexDomainStatus(mockLatestSample, "UNAVAILABLE");
    DcgmNs::ImexTests::SetImexDaemonStatus(mockLatestSample, DcgmImexDaemonStatus::COMMAND_ERROR);

    // Register NVLINK watch for this group and call MonitorWatches directly.
    // MonitorWatches should detect no NVLinks and skip the IMEX check.
    unsigned int groupId = mockLatestSample.groupId;
    DcgmHealthWatchTestHelper::SetGroupWatchState(healthWatch, groupId, DCGM_HEALTH_WATCH_NVLINK);

    dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorWatches(healthWatch, groupId, response);

    REQUIRE(ret == DCGM_ST_OK);
    dcgmHealthResponse_v5 healthResponse = {};
    response.PopulateHealthResponse(healthResponse);
    REQUIRE(healthResponse.incidentCount == 0);
    REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorConnectX")
{
    const dcgm_field_eid_t testConnectXId               = 0;
    const dcgm_field_entity_group_t testConnectXGroupId = DCGM_FE_CONNECTX;

    SECTION("Healthy ConnectX - all checks pass")
    {
        dcgmcm_sample_t sample = {};

        sample.val.i64 = DcgmEntityStatusOk;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_HEALTH, DCGM_ST_OK, sample);

        sample.val.i64 = 0;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS, DCGM_ST_OK, sample);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_MASK, DCGM_ST_OK, sample);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_SEVERITY, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorConnectX(healthWatch, testConnectXGroupId, testConnectXId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Health status not healthy (lost) - warning")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = DcgmEntityStatusLost;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_HEALTH, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorConnectX(healthWatch, testConnectXGroupId, testConnectXId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_CONNECTX);
    }

    SECTION("Uncorrectable fatal errors - failure")
    {
        dcgmcm_sample_t sample = {};

        sample.val.i64 = 0x1000;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS, DCGM_ST_OK, sample);

        sample.val.i64 = 0;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = 0x1000;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_SEVERITY, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorConnectX(healthWatch, testConnectXGroupId, testConnectXId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_CONNECTX);
    }

    SECTION("Uncorrectable non-fatal errors - warning")
    {
        dcgmcm_sample_t sample = {};

        sample.val.i64 = 0x1000;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS, DCGM_ST_OK, sample);

        sample.val.i64 = 0;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = 0;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_SEVERITY, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorConnectX(healthWatch, testConnectXGroupId, testConnectXId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_CONNECTX);
    }

    SECTION("Uncorrectable errors masked - no incidents")
    {
        dcgmcm_sample_t sample = {};

        sample.val.i64 = 0x100000;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS, DCGM_ST_OK, sample);

        sample.val.i64 = 0x180000;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = 0x463010;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_SEVERITY, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorConnectX(healthWatch, testConnectXGroupId, testConnectXId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Multiple unmasked errors - fatal and non-fatal")
    {
        dcgmcm_sample_t sample = {};

        sample.val.i64 = 0x19998;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS, DCGM_ST_OK, sample);

        sample.val.i64 = 0;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = 0x1010;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_SEVERITY, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorConnectX(healthWatch, testConnectXGroupId, testConnectXId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 2);

        bool foundFatal = false;
        bool foundWarn  = false;
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            REQUIRE(healthResponse.incidents[i].system == DCGM_HEALTH_WATCH_CONNECTX);
            if (healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_FAIL)
                foundFatal = true;
            if (healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_WARN)
                foundWarn = true;
        }
        REQUIRE(foundFatal);
        REQUIRE(foundWarn);
    }

    SECTION("Fields not watched - no incidents")
    {
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_HEALTH, DCGM_ST_NOT_WATCHED);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS, DCGM_ST_NOT_WATCHED);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorConnectX(healthWatch, testConnectXGroupId, testConnectXId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorNVLinkErrorFields")
{
    const dcgm_field_eid_t testGpuId                  = 0;
    const dcgm_field_entity_group_t testEntityGroupId = DCGM_FE_GPU;

    SECTION("No NVLink errors - healthy (old fields, pre-Blackwell)")
    {
        dcgmcm_sample_t sample = {};

        // Set compute capability to SM 9.0 (Hopper, pre-Blackwell)
        sample.val.i64 = (9 << 16) | 0; // Major 9, Minor 0
        mockLatestSample.SetFieldSample(DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, DCGM_ST_OK, sample);

        // All old error counts are zero
        sample.val.i64 = 0;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_CRC_ERROR_TOTAL, DCGM_ST_OK, sample);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_TOTAL, DCGM_ST_OK, sample);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_REPLAY_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("No NVLink errors - healthy (new fields, Blackwell+)")
    {
        dcgmcm_sample_t sample = {};

        // Set compute capability to SM 10.0 (Blackwell)
        sample.val.i64 = (10 << 16) | 0;
        mockLatestSample.SetFieldSample(DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, DCGM_ST_OK, sample);

        // All new recovery event counts are zero
        sample.val.i64 = 0;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL, DCGM_ST_OK, sample);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL, DCGM_ST_OK, sample);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_EVENT_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("NVLink DL CRC errors detected - failure")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 5; // CRC errors
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_CRC_ERROR_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
    }

    SECTION("NVLink DL Recovery errors detected - failure")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 3; // Recovery errors
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
    }

    SECTION("NVLink DL Replay errors detected - failure")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 7; // Replay errors
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_REPLAY_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
    }

    SECTION("Multiple NVLink error types - multiple failures")
    {
        dcgmcm_sample_t sample = {};

        // CRC errors
        sample.val.i64 = 2;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_CRC_ERROR_TOTAL, DCGM_ST_OK, sample);

        // Recovery errors
        sample.val.i64 = 4;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_TOTAL, DCGM_ST_OK, sample);

        // Replay errors
        sample.val.i64 = 6;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_REPLAY_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 3);
    }

    SECTION("Link recovery successful events - warning")
    {
        dcgmcm_sample_t sample = {};

        // Inject Blackwell+ compute capability (CC 10.0)
        sample.val.i64 = (10 << 16) | 0;
        mockLatestSample.SetFieldSample(DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, DCGM_ST_OK, sample);

        sample.val.i64 = 5; // Some successful recovery events
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
    }

    SECTION("Link recovery failed events - failure")
    {
        dcgmcm_sample_t sample = {};

        // Inject Blackwell+ compute capability (CC 10.0)
        sample.val.i64 = (10 << 16) | 0;
        mockLatestSample.SetFieldSample(DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, DCGM_ST_OK, sample);

        sample.val.i64 = 2; // Failed recovery events indicate serious issues
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
    }

    SECTION("Multiple new recovery fields - multiple incidents")
    {
        dcgmcm_sample_t sample = {};

        // Inject Blackwell+ compute capability (CC 10.0)
        sample.val.i64 = (10 << 16) | 0;
        mockLatestSample.SetFieldSample(DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, DCGM_ST_OK, sample);

        // Successful recovery events
        sample.val.i64 = 3;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL, DCGM_ST_OK, sample);

        // Failed recovery events
        sample.val.i64 = 1;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 2); // 1 warning + 1 failure
    }

    SECTION("Pre-Blackwell GPU only checks old fields")
    {
        dcgmcm_sample_t sample = {};

        // Inject pre-Blackwell compute capability (CC 9.0)
        sample.val.i64 = (9 << 16) | 0; // Major 9, Minor 0
        mockLatestSample.SetFieldSample(DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, DCGM_ST_OK, sample);

        // Set both old and new fields with errors
        sample.val.i64 = 1;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_CRC_ERROR_TOTAL, DCGM_ST_OK, sample);

        sample.val.i64 = 2;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        // Only old field (DL_CRC) should be checked, new field ignored
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
    }

    SECTION("Fields not watched - no incidents")
    {
        // All fields return NOT_WATCHED (default behavior)
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_NVLINK_CRC_ERROR_TOTAL, DCGM_ST_NOT_WATCHED);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkErrorFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }
}

/**
 * Tests for MonitorFabricFields.
 *
 * Fabric health mask bit layout (NVML_GPU_FABRIC_HEALTH_MASK_* in dcgm_nvml.h):
 *   bits[1:0]  = DEGRADED_BW             (NOT_SUPPORTED=0, TRUE=1, FALSE=2)
 *   bits[3:2]  = ROUTE_RECOVERY          (NOT_SUPPORTED=0, TRUE=1, FALSE=2)
 *   bits[5:4]  = ROUTE_UNHEALTHY         (NOT_SUPPORTED=0, TRUE=1, FALSE=2)
 *   bits[7:6]  = ACCESS_TIMEOUT_RECOVERY (NOT_SUPPORTED=0, TRUE=1, FALSE=2)
 *   bits[11:8] = INCORRECT_CONFIGURATION (NOT_SUPPORTED=0, NONE=1, errors=2-7)
 *
 * Healthy mask (all non-error): 0x1AA
 *   DEGRADED_BW=FALSE(2)<<0=2, ROUTE_RECOVERY=FALSE(2)<<2=8,
 *   ROUTE_UNHEALTHY=FALSE(2)<<4=32, ACCESS_TIMEOUT_RECOVERY=FALSE(2)<<6=128,
 *   INCORRECT_CONFIGURATION=NONE(1)<<8=256  →  total=426=0x1AA
 */
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorFabricFields")
{
    SECTION("No data available - no incidents")
    {
        // Default mock returns NOT_WATCHED for all fields → CollectLatestInt64Sample returns nullopt
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("All fabric fields healthy - no incidents")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x1AA; // all non-error bits
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = static_cast<int64_t>(DcgmFMStatusSuccess);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_MANAGER_STATUS, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("Route unhealthy - failure")
    {
        // Replace ROUTE_UNHEALTHY=FALSE(2)<<4=32 with TRUE(1)<<4=16 → 0x1AA - 32 + 16 = 0x19A
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x19A;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FIELD_VIOLATION);
        std::string_view msg = healthResponse.incidents[0].error.msg;
        REQUIRE(msg.find("Fabric Health: Route Unhealthy") != std::string_view::npos);
    }

    SECTION("Degraded bandwidth - failure")
    {
        // Replace DEGRADED_BW=FALSE(2)<<0=2 with TRUE(1)<<0=1 → 0x1AA - 2 + 1 = 0x1A9
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x1A9;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FIELD_VIOLATION);
        std::string_view msg = healthResponse.incidents[0].error.msg;
        REQUIRE(msg.find("Fabric Health: Bandwidth Degraded") != std::string_view::npos);
    }

    SECTION("Route recovery in progress - failure")
    {
        // Replace ROUTE_RECOVERY=FALSE(2)<<2=8 with TRUE(1)<<2=4 → 0x1AA - 8 + 4 = 0x1A6
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x1A6;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FIELD_VIOLATION);
        std::string_view msg = healthResponse.incidents[0].error.msg;
        REQUIRE(msg.find("Fabric Health: Route Recovery") != std::string_view::npos);
    }

    SECTION("Access timeout recovery in progress - failure")
    {
        // Replace ACCESS_TIMEOUT_RECOVERY=FALSE(2)<<6=128 with TRUE(1)<<6=64 → 0x1AA - 128 + 64 = 0x16A
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x16A;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FIELD_VIOLATION);
        std::string_view msg = healthResponse.incidents[0].error.msg;
        REQUIRE(msg.find("Fabric Health: Access Timeout Recovery") != std::string_view::npos);
    }

    SECTION("Incorrect configuration set - failure")
    {
        // Replace INCORRECT_CONFIGURATION=NONE(1)<<8=256 with INCORRECT_SYSGUID(2)<<8=512 → 0x1AA - 256 + 512 = 0x2AA
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x2AA;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FIELD_VIOLATION);
        std::string_view msg = healthResponse.incidents[0].error.msg;
        REQUIRE(msg.find("Fabric Health: Incorrect Configuration") != std::string_view::npos);
    }

    SECTION("Incorrect configuration = NOT_SUPPORTED - no incident")
    {
        // INCORRECT_CONFIGURATION=NOT_SUPPORTED(0)<<8=0; 0x1AA - 256 + 0 = 0x0AA
        // Condition: !TEST(_NOT_SUPPORTED) && !TEST(_NONE) → TEST(_NOT_SUPPORTED) is true → whole expr false → no
        // incident
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x0AA;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("All five fabric health faults active - five failures")
    {
        // All fault bits: DEGRADED_BW_TRUE(1) | ROUTE_RECOVERY_TRUE(4) | ROUTE_UNHEALTHY_TRUE(16)
        //               | ACCESS_TIMEOUT_RECOVERY_TRUE(64) | INCORRECT_CONFIGURATION_SYSGUID(512) = 0x255
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x255;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = static_cast<int64_t>(DcgmFMStatusSuccess);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_MANAGER_STATUS, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 5);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL);

        std::string allMsgs;
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            REQUIRE(healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_FAIL);
            REQUIRE(healthResponse.incidents[i].system == DCGM_HEALTH_WATCH_NVLINK);
            REQUIRE(healthResponse.incidents[i].error.code == DCGM_FR_FIELD_VIOLATION);
            allMsgs += healthResponse.incidents[i].error.msg;
            allMsgs += '\n';
        }
        REQUIRE(allMsgs.find("Fabric Health: Bandwidth Degraded") != std::string::npos);
        REQUIRE(allMsgs.find("Fabric Health: Route Recovery") != std::string::npos);
        REQUIRE(allMsgs.find("Fabric Health: Route Unhealthy") != std::string::npos);
        REQUIRE(allMsgs.find("Fabric Health: Access Timeout Recovery") != std::string::npos);
        REQUIRE(allMsgs.find("Fabric Health: Incorrect Configuration") != std::string::npos);
    }

    SECTION("Fabric health mask fetch error - propagates error without checking FM status")
    {
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("FM Status: NotSupported - no incident")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x1AA;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = static_cast<int64_t>(DcgmFMStatusNotSupported);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_MANAGER_STATUS, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    struct FmStatusCase
    {
        char const *name;
        dcgmFabricManagerStatus_t status;
        dcgmHealthWatchResults_t expectedHealth;
    };

    auto fmTest = GENERATE(FmStatusCase { "InProgress", DcgmFMStatusInProgress, DCGM_HEALTH_RESULT_WARN },
                           FmStatusCase { "NotStarted", DcgmFMStatusNotStarted, DCGM_HEALTH_RESULT_FAIL },
                           FmStatusCase { "Failure", DcgmFMStatusFailure, DCGM_HEALTH_RESULT_FAIL },
                           FmStatusCase { "Unrecognized", DcgmFMStatusUnrecognized, DCGM_HEALTH_RESULT_FAIL },
                           FmStatusCase { "NvmlTooOld", DcgmFMStatusNvmlTooOld, DCGM_HEALTH_RESULT_FAIL });

    DYNAMIC_SECTION("FM Status: " << fmTest.name << " - incident reported")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x1AA;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = static_cast<int64_t>(fmTest.status);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_MANAGER_STATUS, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == fmTest.expectedHealth);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FABRIC_PROBE_STATE);
    }

    SECTION("FM Status: unknown value - default case reports failure")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x1AA;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        sample.val.i64 = static_cast<int64_t>(DcgmFMStatusCount) + 1; // Beyond last known status
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_MANAGER_STATUS, DCGM_ST_OK, sample);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FABRIC_PROBE_STATE);
    }

    SECTION("FM Status fetch error - propagates error")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0x1AA;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_OK, sample);

        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_MANAGER_STATUS, DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorFabricFields(healthWatch, testEntityGroupId, testGpuId, response);

        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }
}

/**
 * Tests for MonitorNVLink4Fields.
 *
 * The function monitors four NVLink error counters over a time window [startTime, endTime]:
 *   - DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL  (CRC — warn/fail by rate)
 *   - DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL   (CRC — warn/fail by rate)
 *   - DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL     (always FAIL if >= 1)
 *   - DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL   (always FAIL if >= 1)
 *
 * Error is only counted when delta (|start - end|) >= DCGM_LIMIT_MAX_NVLINK_ERROR (1).
 * CRC fields additionally use per-second rate vs DCGM_LIMIT_MAX_NVLINK_CRC_ERROR (100/s):
 *   - rate >= 100/s  → FAIL  (DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD)
 *   - delta >= 1 but rate < 100/s → WARN (DCGM_FR_NVLINK_ERROR_THRESHOLD)
 * Replay / Recovery fields: delta >= 1 → FAIL (DCGM_FR_NVLINK_ERROR_CRITICAL).
 *
 * Uses DcgmHealthWatchGetSamplesFixture (GetSamples mock).
 * startTime / endTime are passed explicitly to pin the time window and avoid
 * relying on timelib_usecSince1970() in the production code.
 */
TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorNVLink4Fields")
{
    // Fixed 60-second window: endTime - startTime = 60 000 000 µs
    long long const startTime = 1000000000LL;
    long long const endTime   = startTime + 60000000LL;

    std::array constexpr nvlink4Fields = { DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL };

    // Marks all four fields as having no data so the production loop skips them.
    // Call this before setField() to avoid NOT_WATCHED falling through the descending check.
    auto setNoDataForAllFields = [&]() {
        for (auto const fieldId : nvlink4Fields)
        {
            mockGetSamples.SetGetSamplesResult(fieldId, DCGM_ORDER_ASCENDING, DCGM_ST_NO_DATA);
            mockGetSamples.SetGetSamplesResult(fieldId, DCGM_ORDER_DESCENDING, DCGM_ST_NO_DATA);
        }
    };

    auto setField = [&](unsigned short fieldId, int64_t startVal, int64_t endVal) {
        dcgmcm_sample_t s = {};
        s.val.i64         = startVal;
        mockGetSamples.SetGetSamplesResult(fieldId, DCGM_ORDER_ASCENDING, DCGM_ST_OK, s);
        s.val.i64 = endVal;
        mockGetSamples.SetGetSamplesResult(fieldId, DCGM_ORDER_DESCENDING, DCGM_ST_OK, s);
    };

    SECTION("No data for any field - no incidents")
    {
        setNoDataForAllFields();

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("All fields zero - no incidents")
    {
        setNoDataForAllFields();
        setField(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL, 0, 0);
        setField(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL, 0, 0);
        setField(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, 0, 0);
        setField(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL, 0, 0);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Replay errors >= 1 - failure")
    {
        setNoDataForAllFields();
        // delta = DCGM_LIMIT_MAX_NVLINK_ERROR → FAIL
        setField(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, 0, DCGM_LIMIT_MAX_NVLINK_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_ERROR_CRITICAL);
        std::string_view msg = healthResponse.incidents[0].error.msg;
        REQUIRE(msg.find("NvLink errors") != std::string_view::npos);
    }

    SECTION("Recovery errors >= 1 - failure")
    {
        setNoDataForAllFields();
        // delta = DCGM_LIMIT_MAX_NVLINK_ERROR → FAIL
        setField(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL, 0, DCGM_LIMIT_MAX_NVLINK_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_ERROR_CRITICAL);
    }

    SECTION("CRC flit errors below rate threshold - warning")
    {
        setNoDataForAllFields();
        // delta = DCGM_LIMIT_MAX_NVLINK_ERROR, rate = delta/60s << DCGM_LIMIT_MAX_NVLINK_CRC_ERROR → WARN
        setField(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL, 0, DCGM_LIMIT_MAX_NVLINK_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_ERROR_THRESHOLD);
    }

    SECTION("CRC flit errors above rate threshold - failure")
    {
        setNoDataForAllFields();
        // delta / windowSec > DCGM_LIMIT_MAX_NVLINK_CRC_ERROR → FAIL
        double const windowSec       = (endTime - startTime) / 1000000.0;
        int64_t const aboveRateDelta = static_cast<int64_t>(DCGM_LIMIT_MAX_NVLINK_CRC_ERROR * windowSec) + 1;
        setField(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL, 0, aboveRateDelta);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD);
    }

    SECTION("CRC data errors below rate threshold - warning")
    {
        setNoDataForAllFields();
        // delta = DCGM_LIMIT_MAX_NVLINK_ERROR, rate << DCGM_LIMIT_MAX_NVLINK_CRC_ERROR → WARN
        setField(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL, 0, DCGM_LIMIT_MAX_NVLINK_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_ERROR_THRESHOLD);
    }

    SECTION("CRC data errors above rate threshold - failure")
    {
        setNoDataForAllFields();
        // delta / windowSec > DCGM_LIMIT_MAX_NVLINK_CRC_ERROR → FAIL
        double const windowSec       = (endTime - startTime) / 1000000.0;
        int64_t const aboveRateDelta = static_cast<int64_t>(DCGM_LIMIT_MAX_NVLINK_CRC_ERROR * windowSec) + 1;
        setField(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL, 0, aboveRateDelta);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD);
    }

    SECTION("Delta computed correctly when start > end (counter wrap)")
    {
        setNoDataForAllFields();
        // start > end → delta = start - end = DCGM_LIMIT_MAX_NVLINK_ERROR → replay FAIL
        setField(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_LIMIT_MAX_NVLINK_ERROR, 0);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_ERROR_CRITICAL);
    }

    SECTION("Ascending GetSamples returns fatal error - propagated immediately")
    {
        setNoDataForAllFields();
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL, DCGM_ORDER_ASCENDING, DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Descending GetSamples returns fatal error - propagated immediately")
    {
        setNoDataForAllFields();
        dcgmcm_sample_t s = {};
        s.val.i64         = 1;
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL, DCGM_ORDER_ASCENDING, DCGM_ST_OK, s);
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Ascending returns NOT_WATCHED - field skipped, no incident")
    {
        setNoDataForAllFields();
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_ORDER_ASCENDING, DCGM_ST_NOT_WATCHED);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Ascending returns NO_DATA - field skipped, no incident")
    {
        setNoDataForAllFields();
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_ORDER_ASCENDING, DCGM_ST_NO_DATA);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Descending returns NO_DATA - field skipped, no incident")
    {
        setNoDataForAllFields();
        dcgmcm_sample_t s = {};
        s.val.i64         = 5;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_ORDER_ASCENDING, DCGM_ST_OK, s);
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_NO_DATA);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Ascending value is blank - field skipped, no incident")
    {
        setNoDataForAllFields();
        dcgmcm_sample_t s = {};
        s.val.i64         = DCGM_INT64_BLANK;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_ORDER_ASCENDING, DCGM_ST_OK, s);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Ascending value is NOT_SUPPORTED - field skipped, no incident")
    {
        setNoDataForAllFields();
        dcgmcm_sample_t s = {};
        s.val.i64         = DCGM_INT64_NOT_SUPPORTED;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_ORDER_ASCENDING, DCGM_ST_OK, s);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Descending value is blank - field skipped, no incident")
    {
        setNoDataForAllFields();
        dcgmcm_sample_t s = {};
        s.val.i64         = DCGM_LIMIT_MAX_NVLINK_ERROR;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_ORDER_ASCENDING, DCGM_ST_OK, s);
        s.val.i64 = DCGM_INT64_BLANK;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, s);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Blank startTime (0) - replaced with now - 60s, errors still detected")
    {
        setNoDataForAllFields();
        setField(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, 0, DCGM_LIMIT_MAX_NVLINK_ERROR);

        // startTime=0 triggers: startTime = now - oneMinuteInUsec inside the function
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, 0, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_ERROR_CRITICAL);
    }

    SECTION("Blank endTime (0) for CRC field - uses now as end time for rate calculation")
    {
        setNoDataForAllFields();
        // Anchor startTime 60s before now. The function will compute timeDiffInSec = (now - startTime) / 1e6.
        // Use 2x the threshold delta to tolerate timing variance up to one additional minute.
        long long const blankEndStartTime = static_cast<long long>(timelib_usecSince1970()) - 60000000LL;
        double const approxWindowSec      = 60.0;
        int64_t const aboveRateDelta = static_cast<int64_t>(DCGM_LIMIT_MAX_NVLINK_CRC_ERROR * approxWindowSec * 2) + 1;
        setField(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL, 0, aboveRateDelta);

        // endTime=0 triggers: timeDiffInSec = (now - startTime) / 1e6
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, blankEndStartTime, 0, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD);
    }

    SECTION("Multiple fields with errors - one incident per field")
    {
        setNoDataForAllFields();
        // CRC flit: delta = DCGM_LIMIT_MAX_NVLINK_ERROR, 60s window → below rate threshold → WARN
        setField(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL, 0, DCGM_LIMIT_MAX_NVLINK_ERROR);
        // Replay: delta = DCGM_LIMIT_MAX_NVLINK_ERROR → FAIL
        setField(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL, 0, DCGM_LIMIT_MAX_NVLINK_ERROR);
        // Recovery: delta = DCGM_LIMIT_MAX_NVLINK_ERROR → FAIL
        setField(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL, 0, DCGM_LIMIT_MAX_NVLINK_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink4Fields(
            healthWatch, testEntityGroupId, testGpuId, startTime, endTime, response);

        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 3);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_FAIL);

        bool foundWarn = false, foundFail = false;
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            REQUIRE(healthResponse.incidents[i].system == DCGM_HEALTH_WATCH_NVLINK);
            if (healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_WARN)
                foundWarn = true;
            if (healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_FAIL)
                foundFail = true;
        }
        REQUIRE(foundWarn);
        REQUIRE(foundFail);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorNVLink5Fields BER thresholds")
{
    std::array constexpr nvlink5Fields = { DCGM_FI_DEV_NVLINK_RX_SYMBOL_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RAW,
                                           DCGM_FI_DEV_NVLINK_RX_PACKET_MALFORMED_TOTAL,
                                           DCGM_FI_DEV_NVLINK_RX_PACKET_DROPPED_TOTAL,
                                           DCGM_FI_DEV_NVLINK_RX_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_RX_REMOTE_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_RX_GENERAL_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_INTEGRITY_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_RECOVERY_EVENT_TOTAL,
                                           DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL,
                                           DCGM_FI_DEV_NVLINK_SYMBOL_BER_RAW,
                                           DCGM_FI_DEV_NVLINK_ECC_ERROR_TOTAL };

    struct BerInput
    {
        uint64_t mantissa;
        uint64_t exponent;
    };

    struct TestCase
    {
        BerInput effectiveBer;
        BerInput symbolBer;
        dcgmHealthWatchResults_t expectedHealth;
        unsigned int expectedErrorCode;
        std::string_view name;
    };

    auto encodeBer = [](uint64_t mantissa, uint64_t exponent) {
        return static_cast<int64_t>((mantissa << 8) | exponent);
    };
    auto setBer = [&](unsigned short fieldId, BerInput input) {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = encodeBer(input.mantissa, input.exponent);
        mockGetSamples.SetGetSamplesResults(fieldId, DCGM_ST_OK, sample, sample);
    };
    auto setBerRaw = [&](unsigned short fieldId, int64_t rawValue) {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = rawValue;
        mockGetSamples.SetGetSamplesResults(fieldId, DCGM_ST_OK, sample, sample);
    };
    auto setDefaultNoDataForOtherFields = [&]() {
        for (auto const fieldId : nvlink5Fields)
        {
            if (fieldId == DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RAW || fieldId == DCGM_FI_DEV_NVLINK_SYMBOL_BER_RAW)
            {
                continue;
            }
            mockGetSamples.SetGetSamplesResult(fieldId, DCGM_ORDER_ASCENDING, DCGM_ST_NO_DATA);
            mockGetSamples.SetGetSamplesResult(fieldId, DCGM_ORDER_DESCENDING, DCGM_ST_NO_DATA);
        }
    };
    auto verifyResult = [&](dcgmHealthWatchResults_t expectedHealth, unsigned int expectedErrorCode) {
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNVLink5Fields(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        auto healthResponseUptr = std::make_unique<dcgmHealthResponse_v5>();
        auto &healthResponse    = *healthResponseUptr;
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.overallHealth == expectedHealth);

        if (expectedHealth == DCGM_HEALTH_RESULT_PASS)
        {
            REQUIRE(healthResponse.incidentCount == 0);
        }
        else
        {
            REQUIRE(healthResponse.incidentCount == 1);
            REQUIRE(healthResponse.incidents[0].health == expectedHealth);
            REQUIRE(healthResponse.incidents[0].entityInfo.entityId == testGpuId);
            REQUIRE(healthResponse.incidents[0].entityInfo.entityGroupId == testEntityGroupId);
            REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
            REQUIRE(healthResponse.incidents[0].error.code == expectedErrorCode);
        }
    };

    std::array const testCases = {
        TestCase { .effectiveBer      = { 9, 13 },
                   .symbolBer         = { 15, 255 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_PASS,
                   .expectedErrorCode = DCGM_ST_OK,
                   .name              = "effective BER below 1e-12 passes" },
        TestCase { .effectiveBer      = { 1, 12 },
                   .symbolBer         = { 15, 255 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_PASS,
                   .expectedErrorCode = DCGM_ST_OK,
                   .name              = "effective BER at 1e-12 passes" },
        TestCase { .effectiveBer      = { 2, 12 },
                   .symbolBer         = { 15, 255 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_FAIL,
                   .expectedErrorCode = DCGM_FR_NVLINK_EFFECTIVE_BER_THRESHOLD,
                   .name              = "effective BER above 1e-12 fails" },
        TestCase { .effectiveBer      = { 10, 13 },
                   .symbolBer         = { 15, 255 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_PASS,
                   .expectedErrorCode = DCGM_ST_OK,
                   .name              = "effective BER at 10e-13 passes" },
        TestCase { .effectiveBer      = { 11, 13 },
                   .symbolBer         = { 15, 255 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_FAIL,
                   .expectedErrorCode = DCGM_FR_NVLINK_EFFECTIVE_BER_THRESHOLD,
                   .name              = "effective BER at 11e-13 fails" },
        TestCase { .effectiveBer      = { 15, 255 },
                   .symbolBer         = { 1, 13 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_FAIL,
                   .expectedErrorCode = DCGM_FR_NVLINK_SYMBOL_BER_THRESHOLD,
                   .name              = "non-zero symbol BER fails" },
        TestCase { .effectiveBer      = { 15, 255 },
                   .symbolBer         = { 15, 255 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_PASS,
                   .expectedErrorCode = DCGM_ST_OK,
                   .name              = "symbol BER at 1.5e-254 passes" },
        TestCase { .effectiveBer      = { 15, 255 },
                   .symbolBer         = { 0, 0 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_PASS,
                   .expectedErrorCode = DCGM_ST_OK,
                   .name              = "raw zero symbol BER passes" },
        TestCase { .effectiveBer      = { 15, 255 },
                   .symbolBer         = { 0, 5 },
                   .expectedHealth    = DCGM_HEALTH_RESULT_PASS,
                   .expectedErrorCode = DCGM_ST_OK,
                   .name              = "zero mantissa nonzero exponent symbol BER passes" },
    };

    for (auto const &testCase : testCases)
    {
        SECTION(std::string(testCase.name))
        {
            setDefaultNoDataForOtherFields();
            setBer(DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RAW, testCase.effectiveBer);
            setBer(DCGM_FI_DEV_NVLINK_SYMBOL_BER_RAW, testCase.symbolBer);
            verifyResult(testCase.expectedHealth, testCase.expectedErrorCode);
        }
    }

    struct ArchTestCase
    {
        std::string_view name;
        dcgmReturn_t computeCapabilityRet;
        int64_t computeCapability;
        bool getAllGpuInfoEnabled;
        dcgmChipArchitecture_t fallbackArch;
        int64_t symbolBerRaw;
        dcgmHealthWatchResults_t expectedHealth;
        unsigned int expectedErrorCode;
    };

    std::array const archTestCases = {
        ArchTestCase { .name                 = "Blackwell via compute capability aggregate symbol BER raw 73710 passes",
                       .computeCapabilityRet = DCGM_ST_OK,
                       .computeCapability    = (10 << 16) | 0,
                       .getAllGpuInfoEnabled = false,
                       .fallbackArch         = DCGM_CHIP_ARCH_UNKNOWN,
                       .symbolBerRaw         = 73710,
                       .expectedHealth       = DCGM_HEALTH_RESULT_PASS,
                       .expectedErrorCode    = DCGM_ST_OK },
        ArchTestCase { .name = "Blackwell via compute capability symbol BER worse than floor exponent fails",
                       .computeCapabilityRet = DCGM_ST_OK,
                       .computeCapability    = (10 << 16) | 0,
                       .getAllGpuInfoEnabled = false,
                       .fallbackArch         = DCGM_CHIP_ARCH_UNKNOWN,
                       .symbolBerRaw         = encodeBer(15, 237),
                       .expectedHealth       = DCGM_HEALTH_RESULT_FAIL,
                       .expectedErrorCode    = DCGM_FR_NVLINK_SYMBOL_BER_THRESHOLD },
        ArchTestCase { .name                 = "Pre-Blackwell via compute capability symbol BER floor raw 4095 passes",
                       .computeCapabilityRet = DCGM_ST_OK,
                       .computeCapability    = (9 << 16) | 0,
                       .getAllGpuInfoEnabled = false,
                       .fallbackArch         = DCGM_CHIP_ARCH_UNKNOWN,
                       .symbolBerRaw         = encodeBer(15, 255),
                       .expectedHealth       = DCGM_HEALTH_RESULT_PASS,
                       .expectedErrorCode    = DCGM_ST_OK },
        ArchTestCase {
            .name = "Blackwell via GetAllGpuInfo fallback when compute capability unwatched aggregate 73710 passes",
            .computeCapabilityRet = DCGM_ST_NOT_WATCHED,
            .computeCapability    = 0,
            .getAllGpuInfoEnabled = true,
            .fallbackArch         = DCGM_CHIP_ARCH_BLACKWELL,
            .symbolBerRaw         = 73710,
            .expectedHealth       = DCGM_HEALTH_RESULT_PASS,
            .expectedErrorCode    = DCGM_ST_OK },
        ArchTestCase {
            .name = "Blackwell via GetAllGpuInfo when compute capability reports pre-Blackwell aggregate 73710 passes",
            .computeCapabilityRet = DCGM_ST_OK,
            .computeCapability    = (9 << 16) | 0,
            .getAllGpuInfoEnabled = true,
            .fallbackArch         = DCGM_CHIP_ARCH_BLACKWELL,
            .symbolBerRaw         = 73710,
            .expectedHealth       = DCGM_HEALTH_RESULT_PASS,
            .expectedErrorCode    = DCGM_ST_OK },
        ArchTestCase { .name                 = "Pre-Blackwell zero mantissa nonzero exponent symbol BER passes",
                       .computeCapabilityRet = DCGM_ST_OK,
                       .computeCapability    = (9 << 16) | 0,
                       .getAllGpuInfoEnabled = false,
                       .fallbackArch         = DCGM_CHIP_ARCH_UNKNOWN,
                       .symbolBerRaw         = encodeBer(0, 5),
                       .expectedHealth       = DCGM_HEALTH_RESULT_PASS,
                       .expectedErrorCode    = DCGM_ST_OK },
        ArchTestCase { .name                 = "Blackwell zero mantissa nonzero exponent symbol BER passes",
                       .computeCapabilityRet = DCGM_ST_OK,
                       .computeCapability    = (10 << 16) | 0,
                       .getAllGpuInfoEnabled = false,
                       .fallbackArch         = DCGM_CHIP_ARCH_UNKNOWN,
                       .symbolBerRaw         = encodeBer(0, 5),
                       .expectedHealth       = DCGM_HEALTH_RESULT_PASS,
                       .expectedErrorCode    = DCGM_ST_OK },
    };

    for (auto const &testCase : archTestCases)
    {
        SECTION(std::string(testCase.name))
        {
            dcgmcm_sample_t ccSample = {};
            ccSample.val.i64         = testCase.computeCapability;
            mockLatestSample.SetFieldSample(
                DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, testCase.computeCapabilityRet, ccSample);

            corePoster.getAllGpuInfoEnabled = testCase.getAllGpuInfoEnabled;
            corePoster.singleGpuInfo        = {};
            corePoster.singleGpuInfo.gpuId  = testGpuId;
            corePoster.singleGpuInfo.arch   = testCase.fallbackArch;

            setDefaultNoDataForOtherFields();
            setBer(DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RAW, BerInput { 15, 255 });
            setBerRaw(DCGM_FI_DEV_NVLINK_SYMBOL_BER_RAW, testCase.symbolBerRaw);
            verifyResult(testCase.expectedHealth, testCase.expectedErrorCode);
        }
    }
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch: negative test for DCGM_FR_FAULTY_MEMORY")
{
    // Set unrepairable memory flag to non-zero, triggering the faulty memory error path
    dcgmcm_sample_t sample = {};
    sample.val.i64         = 1;
    mockLatestSample.SetFieldSample(DCGM_FI_DEV_MEMORY_UNREPAIRABLE, DCGM_ST_OK, sample);

    dcgmReturn_t ret
        = DcgmHealthWatchTestHelper::MonitorMemUnrepairableFlag(healthWatch, testEntityGroupId, testGpuId, response);
    REQUIRE(ret == DCGM_ST_OK);

    auto pHealthResponse = MakeUniqueZero<dcgmHealthResponse_v5>();
    response.PopulateHealthResponse(*pHealthResponse);
    REQUIRE(pHealthResponse->incidentCount == 1);
    REQUIRE(pHealthResponse->overallHealth == DCGM_HEALTH_RESULT_FAIL);
    REQUIRE(pHealthResponse->incidents[0].system == DCGM_HEALTH_WATCH_MEM);
    REQUIRE(pHealthResponse->incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
    REQUIRE(pHealthResponse->incidents[0].entityInfo.entityGroupId == testEntityGroupId);
    REQUIRE(pHealthResponse->incidents[0].entityInfo.entityId == testGpuId);
    REQUIRE(pHealthResponse->incidents[0].error.code == DCGM_FR_FAULTY_MEMORY);
}

TEST_CASE_METHOD(DcgmHealthWatchNvSwitchFixture, "DcgmHealthWatch::MonitorNvSwitchErrorCounts")
{
    long long const startTime = 0LL;
    long long const endTime   = 0LL;

    SECTION("Non-fatal: no errors returns PASS")
    {
        // Field returns OK but sample value is 0 → no incident
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_NON_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, false, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("Non-fatal: sample > 0 reports WARN incident")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 5;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_NON_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, false, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVSWITCH_NONFATAL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVSWITCH_NON_FATAL_ERROR);
    }

    SECTION("Fatal: no errors and all links up returns PASS")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);
        // All link states default to DcgmNvLinkLinkStateNotSupported (0), not Down

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, true, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("Fatal: sample > 0 reports FAIL incident")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 3;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, true, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVSWITCH_FATAL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVSWITCH_FATAL_ERROR);
    }

    SECTION("GetSamples failure is skipped via continue")
    {
        // GetSamples returns error → loop continues, no incident added
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_NO_DATA);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, true, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Blank sample value is skipped")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = DCGM_INT64_BLANK;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, true, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Fatal: GetEntityNvLinkLinkStatus failure returns error")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);
        mockNvSwitch.linkStatusRet = DCGM_ST_GENERIC_ERROR;

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, true, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Fatal: one link down reports NVLINK_DOWN FAIL incident")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);
        mockNvSwitch.linkStates[7] = DcgmNvLinkLinkStateDown;

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, true, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVSWITCH_FATAL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_DOWN);
    }

    SECTION("Fatal: multiple links down reports multiple NVLINK_DOWN incidents")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);
        mockNvSwitch.linkStates[0]  = DcgmNvLinkLinkStateDown;
        mockNvSwitch.linkStates[10] = DcgmNvLinkLinkStateDown;
        mockNvSwitch.linkStates[20] = DcgmNvLinkLinkStateDown;

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, true, testEntityGroupId, testSwitchId, startTime, endTime, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 3);
        for (unsigned int i = 0; i < 3; i++)
        {
            REQUIRE(healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_FAIL);
            REQUIRE(healthResponse.incidents[i].system == DCGM_HEALTH_WATCH_NVSWITCH_FATAL);
            REQUIRE(healthResponse.incidents[i].error.code == DCGM_FR_NVLINK_DOWN);
        }
    }

    SECTION("Non-fatal does not call GetEntityNvLinkLinkStatus")
    {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 0;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_NON_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);
        // Set link status to fail — if called, function would return error
        mockNvSwitch.linkStatusRet = DCGM_ST_GENERIC_ERROR;
        // Set a link down — if called, function would add an incident
        mockNvSwitch.linkStates[0] = DcgmNvLinkLinkStateDown;

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, false, testEntityGroupId, testSwitchId, startTime, endTime, response);
        // Non-fatal path skips GetEntityNvLinkLinkStatus entirely
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Blank startTime is filled in from current time")
    {
        // startTime=0 is blank; function computes now - 60s. Provide a sample with OK count.
        dcgmcm_sample_t sample = {};
        sample.val.i64         = 2;
        mockNvSwitch.getSamples.SetGetSamplesResult(
            DCGM_FI_DEV_SXID_FATAL_ERROR, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sample);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorNvSwitchErrorCounts(
            healthWatch, true, testEntityGroupId, testSwitchId, 0LL, 0LL, response);
        REQUIRE(ret == DCGM_ST_OK);

        // Incident should be reported — the blank startTime fill-in did not prevent the call
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount >= 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVSWITCH_FATAL);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchMemRetiredPagesFixture, "DcgmHealthWatch::MonitorMemSbeDbeRetiredPages")
{
    SECTION("Zero pages: PASS")
    {
        SetDbeAndSbe(0, 0);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("SBE + DBE combined below limit: PASS")
    {
        // 30 + 30 = 60 < DCGM_LIMIT_MAX_RETIRED_PAGES (63)
        SetDbeAndSbe(30, 30);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("SBE + DBE combined equals limit: FAIL with DCGM_FR_RETIRED_PAGES_LIMIT")
    {
        // 30 + 33 = 63 == DCGM_LIMIT_MAX_RETIRED_PAGES
        SetDbeAndSbe(33, 30);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_RETIRED_PAGES_LIMIT);
    }

    SECTION("SBE + DBE combined exceeds limit: FAIL with DCGM_FR_RETIRED_PAGES_LIMIT")
    {
        // 40 + 30 = 70 > DCGM_LIMIT_MAX_RETIRED_PAGES (63)
        SetDbeAndSbe(40, 30);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_RETIRED_PAGES_LIMIT);
    }

    SECTION("Blank DBE sample treated as zero in total")
    {
        // Blank DBE: only SBE contributes, total = 5 < 63
        dcgmcm_sample_t dbeSample = {};
        dbeSample.val.i64         = DCGM_INT64_BLANK;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeSample);
        dcgmcm_sample_t sbeSample = {};
        sbeSample.val.i64         = 5;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sbeSample);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Blank SBE sample treated as zero in total")
    {
        // Blank SBE: only DBE contributes, total = 5 < 63
        dcgmcm_sample_t dbeSample = {};
        dbeSample.val.i64         = 5;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeSample);
        dcgmcm_sample_t sbeSample = {};
        sbeSample.val.i64         = DCGM_INT64_BLANK;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sbeSample);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("DBE GetSamples fatal error returns error")
    {
        // Hard error on DBE query → function returns the error immediately
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("SBE GetSamples fatal error returns error")
    {
        // DBE query succeeds, SBE query fails with hard error
        dcgmcm_sample_t dbeSample = {};
        dbeSample.val.i64         = 0;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeSample);
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("DBE above soft limit, rate > 1 per week: FAIL with DCGM_FR_RETIRED_PAGES_DBE_LIMIT")
    {
        // DBE = 20 > DCGM_LIMIT_MAX_RETIRED_PAGES_SOFT_LIMIT (15), total = 20 < 63
        // Week-ago DBE = 5, delta = 20 - 5 = 15 > 1 → FAIL
        dcgmcm_sample_t dbeCurrent = {};
        dbeCurrent.val.i64         = 20;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeCurrent);

        dcgmcm_sample_t sbeSample = {};
        sbeSample.val.i64         = 0;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sbeSample);

        dcgmcm_sample_t dbeWeekAgo = {};
        dbeWeekAgo.val.i64         = 5;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeWeekAgo);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_RETIRED_PAGES_DBE_LIMIT);
    }

    SECTION("DBE above soft limit, rate exactly 1 per week: PASS")
    {
        // delta = 20 - 19 = 1, not > 1, so no incident
        dcgmcm_sample_t dbeCurrent = {};
        dbeCurrent.val.i64         = 20;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeCurrent);

        dcgmcm_sample_t sbeSample = {};
        sbeSample.val.i64         = 0;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sbeSample);

        dcgmcm_sample_t dbeWeekAgo = {};
        dbeWeekAgo.val.i64         = 19;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeWeekAgo);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("DBE above soft limit, week-ago sample is blank: PASS (no incident)")
    {
        // Blank week-ago → function returns early with no incident
        dcgmcm_sample_t dbeCurrent = {};
        dbeCurrent.val.i64         = 20;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeCurrent);

        dcgmcm_sample_t sbeSample = {};
        sbeSample.val.i64         = 0;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sbeSample);

        dcgmcm_sample_t dbeWeekAgo = {};
        dbeWeekAgo.val.i64         = DCGM_INT64_BLANK;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, dbeWeekAgo);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("DBE at or below soft limit: week-ago query is not made")
    {
        // DBE = 15 == DCGM_LIMIT_MAX_RETIRED_PAGES_SOFT_LIMIT, not > 15, so no week-ago check
        SetDbeAndSbe(15, 0);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("NO_DATA on DBE query: treated as zero, continues")
    {
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_NO_DATA);
        dcgmcm_sample_t sbeSample = {};
        sbeSample.val.i64         = 0;
        mockGetSamples.Enqueue(DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, sbeSample);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemSbeDbeRetiredPages(
            healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorGpuRecoveryAction")
{
    auto setAction = [&](dcgmReturn_t retCode, std::optional<int64_t> actionValue = std::nullopt) {
        dcgmcm_sample_t sample = {};
        if (actionValue)
        {
            sample.val.i64 = *actionValue;
        }
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_GPU_RECOVERY_ACTION, retCode, sample);
    };

    SECTION("NO_DATA returns OK with no incident")
    {
        setAction(DCGM_ST_NO_DATA);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("NOT_WATCHED returns OK with no incident")
    {
        setAction(DCGM_ST_NOT_WATCHED);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Hard error from GetLatestSample is propagated")
    {
        setAction(DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Blank sample returns OK with no incident")
    {
        setAction(DCGM_ST_OK, DCGM_INT64_BLANK);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Out-of-range action value returns OK with no incident")
    {
        setAction(DCGM_ST_OK, 99);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("NONE: no incident")
    {
        setAction(DCGM_ST_OK, NVML_GPU_RECOVERY_ACTION_NONE);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("GPU_RESET: FAIL incident on DRIVER system")
    {
        setAction(DCGM_ST_OK, NVML_GPU_RECOVERY_ACTION_GPU_RESET);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_DRIVER);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_GPU_RECOVERY_RESET);
    }

    SECTION("NODE_REBOOT: FAIL incident on DRIVER system")
    {
        setAction(DCGM_ST_OK, NVML_GPU_RECOVERY_ACTION_NODE_REBOOT);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_DRIVER);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_GPU_RECOVERY_REBOOT);
    }

    SECTION("DRAIN_P2P: WARN incident on DRIVER system")
    {
        setAction(DCGM_ST_OK, NVML_GPU_RECOVERY_ACTION_DRAIN_P2P);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_DRIVER);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_GPU_RECOVERY_DRAIN_P2P);
    }

    SECTION("DRAIN_AND_RESET: WARN incident on DRIVER system")
    {
        setAction(DCGM_ST_OK, NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorGpuRecoveryAction(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_DRIVER);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_GPU_RECOVERY_DRAIN_RESET);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorMem GetSamples sub-functions")
{
    using InvokeFn = dcgmReturn_t (*)(
        DcgmHealthWatch &, dcgm_field_entity_group_t, dcgm_field_eid_t, long long, long long, DcgmHealthResponse &);

    struct Params
    {
        const char *name;
        unsigned short fieldId;
        InvokeFn invoke;
        dcgmError_t errorCode;
        dcgmHealthWatchResults_t expectedHealth;
    };

    auto params = GENERATE(Params { "MonitorMemVolatileDbes",
                                    DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                    DcgmHealthWatchTestHelper::MonitorMemVolatileDbes,
                                    DCGM_FR_VOLATILE_DBE_DETECTED,
                                    DCGM_HEALTH_RESULT_FAIL },
                           Params { "MonitorMemRetiredPending",
                                    DCGM_FI_DEV_PAGE_RETIRED_PENDING,
                                    DcgmHealthWatchTestHelper::MonitorMemRetiredPending,
                                    DCGM_FR_PENDING_PAGE_RETIREMENTS,
                                    DCGM_HEALTH_RESULT_WARN },
                           Params { "MonitorMemRowRemapFailures",
                                    DCGM_FI_DEV_ROW_REMAP_FAILED,
                                    DcgmHealthWatchTestHelper::MonitorMemRowRemapFailures,
                                    DCGM_FR_ROW_REMAP_FAILURE,
                                    DCGM_HEALTH_RESULT_FAIL });

    auto setField = [&](dcgmReturn_t retCode, int64_t value = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = value;
        mockGetSamples.SetGetSamplesResult(params.fieldId, DCGM_ORDER_DESCENDING, retCode, s);
    };

    DYNAMIC_SECTION(params.name << ": NOT_WATCHED returns OK with no incident")
    {
        setField(DCGM_ST_NOT_WATCHED);

        dcgmReturn_t ret = params.invoke(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    DYNAMIC_SECTION(params.name << ": hard error is propagated")
    {
        setField(DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret = params.invoke(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    DYNAMIC_SECTION(params.name << ": blank sample returns OK with no incident")
    {
        setField(DCGM_ST_OK, DCGM_INT64_BLANK);

        dcgmReturn_t ret = params.invoke(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    DYNAMIC_SECTION(params.name << ": zero value returns OK with no incident")
    {
        setField(DCGM_ST_OK, 0);

        dcgmReturn_t ret = params.invoke(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    DYNAMIC_SECTION(params.name << ": non-zero value reports incident")
    {
        setField(DCGM_ST_OK, 1);

        dcgmReturn_t ret = params.invoke(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == params.expectedHealth);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].error.code == params.errorCode);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorMemRowRemapUncorrectable")
{
    auto setComputeCapability = [&](int majorVersion) {
        dcgmcm_sample_t sample = {};
        sample.val.i64         = (majorVersion << 16) | 0;
        mockLatestSample.SetFieldSample(DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, DCGM_ST_OK, sample);
    };

    auto setField = [&](dcgmReturn_t retCode, std::optional<int64_t> value = std::nullopt) {
        dcgmcm_sample_t sample = {};
        if (value)
        {
            sample.val.i64 = *value;
        }
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL, retCode, sample);
    };

    // Set mock architecture to Ampere by default so the function runs
    setComputeCapability(DcgmHealthWatchTestHelper::GetComputeCapabilityAmpere());

    SECTION("NO_DATA returns OK with no incident")
    {
        setField(DCGM_ST_NO_DATA);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemRowRemapUncorrectable(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("NOT_WATCHED returns OK with no incident")
    {
        setField(DCGM_ST_NOT_WATCHED);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemRowRemapUncorrectable(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Hard error is propagated")
    {
        setField(DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemRowRemapUncorrectable(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Blank sample returns OK with no incident")
    {
        setField(DCGM_ST_OK, DCGM_INT64_BLANK);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemRowRemapUncorrectable(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Pre-Ampere architecture returns OK with no incident")
    {
        setComputeCapability(DcgmHealthWatchTestHelper::GetComputeCapabilityAmpere() - 1);
        setField(DCGM_ST_OK, DCGM_LIMIT_MAX_ROW_REMAP_UNCORRECTABLE + 1);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemRowRemapUncorrectable(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Under threshold returns OK with no incident")
    {
        setField(DCGM_ST_OK, DCGM_LIMIT_MAX_ROW_REMAP_UNCORRECTABLE - 1);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemRowRemapUncorrectable(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("At or above threshold reports WARN")
    {
        setField(DCGM_ST_OK, DCGM_LIMIT_MAX_ROW_REMAP_UNCORRECTABLE);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemRowRemapUncorrectable(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_UNCORRECTABLE_ROW_REMAP_LIMIT);
    }
}

TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorMemUnrepairableFlag")
{
    auto setField = [&](dcgmReturn_t retCode, std::optional<int64_t> value = std::nullopt) {
        dcgmcm_sample_t s = {};
        if (value)
        {
            s.val.i64 = *value;
        }
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_MEMORY_UNREPAIRABLE, retCode, s);
    };

    SECTION("NO_DATA returns OK with no incident")
    {
        setField(DCGM_ST_NO_DATA);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemUnrepairableFlag(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("NOT_WATCHED returns OK with no incident")
    {
        setField(DCGM_ST_NOT_WATCHED);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemUnrepairableFlag(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Hard error is propagated")
    {
        setField(DCGM_ST_GENERIC_ERROR);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemUnrepairableFlag(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Blank sample returns OK with no incident")
    {
        setField(DCGM_ST_OK, DCGM_INT64_BLANK);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemUnrepairableFlag(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Zero flag returns OK with no incident")
    {
        setField(DCGM_ST_OK, 0);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemUnrepairableFlag(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Non-zero flag reports FAIL")
    {
        setField(DCGM_ST_OK, 1);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorMemUnrepairableFlag(
            healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FAULTY_MEMORY);
    }
}

// ---------------------------------------------------------------------------
// MonitorMem (orchestrator)
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorMem")
{
    // With no mock data configured, all GetSamples calls return NOT_WATCHED (sample=0),
    // and GetLatestSample for UnrepairableFlag returns OK with sample=0 — no incidents.

    SECTION("All sub-functions clean returns OK with no incidents")
    {
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorMem(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
        REQUIRE(healthResponse.overallHealth == DCGM_HEALTH_RESULT_PASS);
    }

    SECTION("Volatile DBE error triggers FAIL incident")
    {
        dcgmcm_sample_t s = {};
        s.val.i64         = 2;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_OK, s);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorMem(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_OK);

        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_VOLATILE_DBE_DETECTED);
    }

    SECTION("Sub-function hard error is propagated while other sub-functions still run")
    {
        // DBE volatile returns hard error — MonitorMem accumulates it but continues
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, DCGM_ORDER_DESCENDING, DCGM_ST_GENERIC_ERROR);
        // Retired pending also has an incident — proves sub-functions after the error still run
        dcgmcm_sample_t s = {};
        s.val.i64         = 1;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_PAGE_RETIRED_PENDING, DCGM_ORDER_DESCENDING, DCGM_ST_OK, s);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorMem(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);

        response.PopulateHealthResponse(healthResponse);
        // The pending retirement incident from the later sub-function was still recorded
        REQUIRE(healthResponse.incidentCount >= 1);
        bool foundPending = false;
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            if (healthResponse.incidents[i].error.code == DCGM_FR_PENDING_PAGE_RETIREMENTS)
            {
                foundPending = true;
            }
        }
        REQUIRE(foundPending);
    }
}

// ---------------------------------------------------------------------------
// GetExpectedPcieReplayRate
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::GetExpectedPcieReplayRate")
{
    auto setGen = [&](dcgmReturn_t ret, int64_t val = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = val;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_PCIE_LINK_GEN, ret, s);
    };
    auto setWidth = [&](dcgmReturn_t ret, int64_t val = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = val;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_PCIE_LINK_WIDTH, ret, s);
    };

    SECTION("LINK_GEN returns NO_DATA")
    {
        setGen(DCGM_ST_NO_DATA);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_NO_DATA);
    }

    SECTION("LINK_GEN returns NOT_WATCHED")
    {
        setGen(DCGM_ST_NOT_WATCHED);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_NOT_WATCHED);
    }

    SECTION("LINK_GEN value is blank")
    {
        setGen(DCGM_ST_OK, DCGM_INT64_BLANK);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_NO_DATA);
    }

    SECTION("LINK_WIDTH returns NO_DATA")
    {
        setGen(DCGM_ST_OK, 4);
        setWidth(DCGM_ST_NO_DATA);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_NO_DATA);
    }

    SECTION("LINK_WIDTH value is blank")
    {
        setGen(DCGM_ST_OK, 4);
        setWidth(DCGM_ST_OK, DCGM_INT64_BLANK);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_NO_DATA);
    }

    SECTION("Gen1 4 lanes — rate == 1")
    {
        setGen(DCGM_ST_OK, 1);
        setWidth(DCGM_ST_OK, 4);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_OK);
        REQUIRE(rate == 1); // ceil(0.15 * 4) = 1
    }

    SECTION("Gen4 16 lanes — rate == 16")
    {
        setGen(DCGM_ST_OK, 4);
        setWidth(DCGM_ST_OK, 16);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_OK);
        REQUIRE(rate == 16); // ceil(0.96 * 16) = 16
    }

    SECTION("Gen5 16 lanes — rate == 31")
    {
        setGen(DCGM_ST_OK, 5);
        setWidth(DCGM_ST_OK, 16);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_OK);
        REQUIRE(rate == 31); // ceil(1.92 * 16) = 31
    }

    SECTION("Gen exceeds max — capped to gen6 rate")
    {
        setGen(DCGM_ST_OK, 99);
        setWidth(DCGM_ST_OK, 16);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_OK);
        REQUIRE(rate == 62); // ceil(3.84 * 16) = 62
    }

    SECTION("Hard error from GetLatestSample is propagated")
    {
        setGen(DCGM_ST_GENERIC_ERROR);
        int rate = 0;
        REQUIRE(DcgmHealthWatchTestHelper::GetExpectedPcieReplayRate(healthWatch, testEntityGroupId, testGpuId, rate)
                == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// MonitorPcie
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorPcie")
{
    constexpr long long kStart = 1000000LL;
    constexpr long long kEnd   = 2000000LL;

    auto setAsc = [&](dcgmReturn_t ret, int64_t val = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = val;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_PCIE_REPLAY_TOTAL, DCGM_ORDER_ASCENDING, ret, s);
    };
    auto setDesc = [&](dcgmReturn_t ret, int64_t val = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = val;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_PCIE_REPLAY_TOTAL, DCGM_ORDER_DESCENDING, ret, s);
    };
    auto setPcieGenWidth = [&](int64_t gen, int64_t width) {
        dcgmcm_sample_t g = {}, w = {};
        g.val.i64 = gen;
        w.val.i64 = width;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_PCIE_LINK_GEN, DCGM_ST_OK, g);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_PCIE_LINK_WIDTH, DCGM_ST_OK, w);
    };

    SECTION("Ascending GetSamples returns NO_DATA — OK, no incidents")
    {
        setAsc(DCGM_ST_NO_DATA);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Ascending GetSamples returns NOT_WATCHED — OK, no incidents")
    {
        setAsc(DCGM_ST_NOT_WATCHED);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Ascending value is blank — OK, no incidents")
    {
        setAsc(DCGM_ST_OK, DCGM_INT64_BLANK);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Descending GetSamples returns NO_DATA after valid ascending — OK, no incidents")
    {
        setAsc(DCGM_ST_OK, 100);
        setDesc(DCGM_ST_NO_DATA);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Descending value is blank — OK, no incidents")
    {
        setAsc(DCGM_ST_OK, 100);
        setDesc(DCGM_ST_OK, DCGM_INT64_BLANK);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("GetExpectedPcieReplayRate returns NO_DATA — OK, no incidents")
    {
        setAsc(DCGM_ST_OK, 0);
        setDesc(DCGM_ST_OK, 100);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_PCIE_LINK_GEN, DCGM_ST_NO_DATA);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Replay rate within expected limit — no incidents")
    {
        setAsc(DCGM_ST_OK, 0);
        setDesc(DCGM_ST_OK, 1); // delta = 1
        setPcieGenWidth(4, 16); // expected = ceil(0.96 * 16) = 16; 1 <= 16
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Replay rate exceeds expected — WARN with DCGM_FR_PCI_REPLAY_RATE")
    {
        setAsc(DCGM_ST_OK, 0);
        setDesc(DCGM_ST_OK, 20); // delta = 20
        setPcieGenWidth(4, 16);  // expected = 16; 20 > 16 → WARN
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_PCIE);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_PCI_REPLAY_RATE);
    }

    SECTION("Counter wrap (start > end) — delta still triggers WARN")
    {
        setAsc(DCGM_ST_OK, 20);
        setDesc(DCGM_ST_OK, 0); // delta = |0 - 20| = 20
        setPcieGenWidth(4, 16); // expected = 16; 20 > 16 → WARN
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_PCI_REPLAY_RATE);
    }

    SECTION("Ascending GetSamples returns hard error — propagated")
    {
        setAsc(DCGM_ST_GENERIC_ERROR);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPcie(healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// MonitorThermal
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorThermal")
{
    constexpr long long kStart = 1000000LL;
    constexpr long long kEnd   = 2000000LL;

    auto setAsc = [&](dcgmReturn_t ret, int64_t val = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = val;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_THERMAL_VIOLATION, DCGM_ORDER_ASCENDING, ret, s);
    };
    auto setLatest = [&](dcgmReturn_t ret, int64_t val = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = val;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_THERMAL_VIOLATION, ret, s);
    };

    SECTION("GetSamples returns NO_DATA — OK, no incidents")
    {
        setAsc(DCGM_ST_NO_DATA);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("GetSamples returns NOT_WATCHED — OK, no incidents")
    {
        setAsc(DCGM_ST_NOT_WATCHED);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Ascending value is blank — OK, no incidents")
    {
        setAsc(DCGM_ST_OK, DCGM_INT64_BLANK);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("GetLatestSample returns NO_DATA — OK, no incidents")
    {
        setAsc(DCGM_ST_OK, 100);
        setLatest(DCGM_ST_NO_DATA);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("GetLatestSample returns NOT_WATCHED — OK, no incidents")
    {
        setAsc(DCGM_ST_OK, 100);
        setLatest(DCGM_ST_NOT_WATCHED);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("GetLatestSample value is blank — OK, no incidents")
    {
        setAsc(DCGM_ST_OK, 100);
        setLatest(DCGM_ST_OK, DCGM_INT64_BLANK);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Zero violation time (start == end) — no incidents")
    {
        setAsc(DCGM_ST_OK, 500);
        setLatest(DCGM_ST_OK, 500);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Positive violation time (end > start) — WARN with DCGM_FR_CLOCKS_EVENT_THERMAL")
    {
        setAsc(DCGM_ST_OK, 100);
        setLatest(DCGM_ST_OK, 200);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_THERMAL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_CLOCKS_EVENT_THERMAL);
    }

    SECTION("Counter wrap (start > end) — WARN")
    {
        setAsc(DCGM_ST_OK, 200);
        setLatest(DCGM_ST_OK, 100);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_CLOCKS_EVENT_THERMAL);
    }

    SECTION("GetSamples hard error is propagated")
    {
        setAsc(DCGM_ST_GENERIC_ERROR);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// MonitorPower
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorPower")
{
    constexpr long long kStart = 1000000LL;
    constexpr long long kEnd   = 2000000LL;

    auto setAsc = [&](dcgmReturn_t ret, int64_t val = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = val;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_POWER_VIOLATION, DCGM_ORDER_ASCENDING, ret, s);
    };
    auto setDesc = [&](dcgmReturn_t ret, int64_t val = 0) {
        dcgmcm_sample_t s = {};
        s.val.i64         = val;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_POWER_VIOLATION, DCGM_ORDER_DESCENDING, ret, s);
    };

    SECTION("GPU entity with blank board power — WARN with DCGM_FR_POWER_UNREADABLE")
    {
        dcgmcm_sample_t board = {};
        board.val.d           = DCGM_FP64_BLANK;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_BOARD_POWER_WATTS, DCGM_ST_OK, board);
        setAsc(DCGM_ST_NO_DATA);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_POWER_UNREADABLE);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_POWER);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
    }

    SECTION("GPU entity with NOT_SUPPORTED board power — no POWER_UNREADABLE warning")
    {
        dcgmcm_sample_t board = {};
        board.val.d           = DCGM_FP64_NOT_SUPPORTED;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_BOARD_POWER_WATTS, DCGM_ST_OK, board);
        setAsc(DCGM_ST_NO_DATA);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Non-GPU entity skips board power check — no POWER_UNREADABLE warning")
    {
        dcgmcm_sample_t board = {};
        board.val.d           = DCGM_FP64_BLANK;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_BOARD_POWER_WATTS, DCGM_ST_OK, board);
        setAsc(DCGM_ST_NO_DATA);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorPower(healthWatch, DCGM_FE_SWITCH, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("POWER_VIOLATION ascending NO_DATA — OK, no incidents")
    {
        setAsc(DCGM_ST_NO_DATA);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("POWER_VIOLATION ascending NOT_WATCHED — OK, no incidents")
    {
        setAsc(DCGM_ST_NOT_WATCHED);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Ascending value is blank — no incidents")
    {
        setAsc(DCGM_ST_OK, DCGM_INT64_BLANK);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Descending NO_DATA after valid ascending — no incidents")
    {
        setAsc(DCGM_ST_OK, 100);
        setDesc(DCGM_ST_NO_DATA);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Zero violation time (start == end) — no incidents")
    {
        setAsc(DCGM_ST_OK, 500);
        setDesc(DCGM_ST_OK, 500);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Positive violation time — WARN with DCGM_FR_CLOCKS_EVENT_POWER")
    {
        setAsc(DCGM_ST_OK, 100);
        setDesc(DCGM_ST_OK, 300);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_POWER);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_CLOCKS_EVENT_POWER);
    }

    SECTION("Both board power unreadable and violation — two incidents")
    {
        dcgmcm_sample_t board = {};
        board.val.d           = DCGM_FP64_BLANK;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_BOARD_POWER_WATTS, DCGM_ST_OK, board);
        setAsc(DCGM_ST_OK, 100);
        setDesc(DCGM_ST_OK, 300);

        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 2);
    }

    SECTION("Ascending GetSamples hard error is propagated")
    {
        setAsc(DCGM_ST_GENERIC_ERROR);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorPower(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// MonitorCpuThermal
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorCpuThermal")
{
    constexpr long long kStart = 1000000LL;
    constexpr long long kEnd   = 2000000LL;

    auto makeSample = [](double d) {
        dcgmcm_sample_t s = {};
        s.val.d           = d;
        return s;
    };

    // Configure ascending GetSamples for all three CPU temp fields.
    auto setAllGetSamples = [&](double startTemp, double startWarn, double startCrit) {
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_CPU_TEMP_CELSIUS, DCGM_ORDER_ASCENDING, DCGM_ST_OK, makeSample(startTemp));
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_CPU_TEMP_WARNING_CELSIUS, DCGM_ORDER_ASCENDING, DCGM_ST_OK, makeSample(startWarn));
        mockGetSamples.SetGetSamplesResult(
            DCGM_FI_DEV_CPU_TEMP_CRITICAL_CELSIUS, DCGM_ORDER_ASCENDING, DCGM_ST_OK, makeSample(startCrit));
    };

    // Configure GetLatestSample for all three CPU temp fields.
    auto setAllLatestSamples = [&](double endTemp, double warnThresh, double critThresh) {
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CPU_TEMP_CELSIUS, DCGM_ST_OK, makeSample(endTemp));
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CPU_TEMP_WARNING_CELSIUS, DCGM_ST_OK, makeSample(warnThresh));
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CPU_TEMP_CRITICAL_CELSIUS, DCGM_ST_OK, makeSample(critThresh));
    };

    SECTION("First GetSamples field returns NO_DATA — OK, no incidents")
    {
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_CPU_TEMP_CELSIUS, DCGM_ORDER_ASCENDING, DCGM_ST_NO_DATA);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorCpuThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("First ascending field value is blank — OK, no incidents")
    {
        dcgmcm_sample_t blankSample = {};
        blankSample.val.i64         = DCGM_INT64_BLANK;
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_CPU_TEMP_CELSIUS, DCGM_ORDER_ASCENDING, DCGM_ST_OK, blankSample);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorCpuThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("First GetLatestSample field returns NO_DATA — OK, no incidents")
    {
        setAllGetSamples(70.0, 85.0, 95.0);
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CPU_TEMP_CELSIUS, DCGM_ST_NO_DATA);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorCpuThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Normal temperatures — no incidents")
    {
        setAllGetSamples(70.0, 85.0, 95.0);
        setAllLatestSamples(75.0, 85.0, 95.0);
        // avg=(70+75)/2=72.5 < 85 → no WARN; end=75 < 95 → no FAIL
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorCpuThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Average temp >= warning threshold — WARN with DCGM_FR_FIELD_THRESHOLD_DBL")
    {
        setAllGetSamples(85.0, 85.0, 95.0);
        setAllLatestSamples(87.0, 85.0, 95.0);
        // avg=(85+87)/2=86 >= 85 → WARN; end=87 < 95 → no FAIL
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorCpuThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_THERMAL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FIELD_THRESHOLD_DBL);
    }

    SECTION("End temp >= critical threshold — WARN and FAIL incidents")
    {
        setAllGetSamples(90.0, 85.0, 95.0);
        setAllLatestSamples(95.0, 85.0, 95.0);
        // avg=(90+95)/2=92.5 >= 85 → WARN; end=95 >= 95 → FAIL
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorCpuThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 2);
        bool foundWarn = false, foundFail = false;
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            if (healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_WARN)
                foundWarn = true;
            if (healthResponse.incidents[i].health == DCGM_HEALTH_RESULT_FAIL)
                foundFail = true;
        }
        REQUIRE(foundWarn);
        REQUIRE(foundFail);
    }

    SECTION("End temp >= critical but average below warning — only FAIL incident")
    {
        setAllGetSamples(50.0, 85.0, 95.0);
        setAllLatestSamples(95.0, 85.0, 95.0);
        // avg=(50+95)/2=72.5 < 85 → no WARN; end=95 >= 95 → FAIL
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorCpuThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FIELD_THRESHOLD_DBL);
    }

    SECTION("GetSamples hard error is propagated")
    {
        mockGetSamples.SetGetSamplesResult(DCGM_FI_DEV_CPU_TEMP_CELSIUS, DCGM_ORDER_ASCENDING, DCGM_ST_GENERIC_ERROR);
        dcgmReturn_t ret = DcgmHealthWatchTestHelper::MonitorCpuThermal(
            healthWatch, testEntityGroupId, testGpuId, kStart, kEnd, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// MonitorInforom
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorInforom")
{
    auto setField = [&](dcgmReturn_t ret, std::optional<int64_t> val = std::nullopt) {
        dcgmcm_sample_t s = {};
        if (val.has_value())
            s.val.i64 = *val;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_INFOROM_VALID, ret, s);
    };

    SECTION("Field not watched — OK, no incidents")
    {
        // Default mock returns NOT_WATCHED; CollectLatestInt64Sample maps to nullopt.
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorInforom(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Field returns NO_DATA — OK, no incidents")
    {
        setField(DCGM_ST_NO_DATA);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorInforom(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Field returns blank value — OK, no incidents")
    {
        setField(DCGM_ST_OK, DCGM_INT64_BLANK);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorInforom(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Field returns NOT_SUPPORTED — OK, no incidents")
    {
        setField(DCGM_ST_OK, DCGM_INT64_NOT_SUPPORTED);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorInforom(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Valid inforom (value == 1) — no incidents")
    {
        setField(DCGM_ST_OK, 1);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorInforom(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Corrupt inforom (value == 0) — WARN with DCGM_FR_CORRUPT_INFOROM")
    {
        setField(DCGM_ST_OK, 0);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorInforom(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_INFOROM);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_CORRUPT_INFOROM);
    }

    SECTION("GetLatestSample hard error is propagated")
    {
        setField(DCGM_ST_GENERIC_ERROR);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorInforom(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// MonitorCpuPower
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchFixture, "DcgmHealthWatch::MonitorCpuPower")
{
    auto setPower = [&](dcgmReturn_t ret, double val = 0.0) {
        dcgmcm_sample_t s = {};
        s.val.d           = val;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CPU_POWER_WATTS, ret, s);
    };
    auto setLimit = [&](dcgmReturn_t ret, double val = 0.0) {
        dcgmcm_sample_t s = {};
        s.val.d           = val;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CPU_POWER_LIMIT_WATTS, ret, s);
    };

    SECTION("CPU_POWER_UTIL_CURRENT returns NO_DATA — OK, no incidents")
    {
        setPower(DCGM_ST_NO_DATA);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorCpuPower(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("CPU_POWER_UTIL_CURRENT value is blank — OK, no incidents")
    {
        dcgmcm_sample_t s = {};
        s.val.i64         = DCGM_INT64_BLANK;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CPU_POWER_WATTS, DCGM_ST_OK, s);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorCpuPower(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("CPU_POWER_LIMIT returns NO_DATA — OK, no incidents")
    {
        setPower(DCGM_ST_OK, 100.0);
        setLimit(DCGM_ST_NO_DATA);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorCpuPower(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("CPU_POWER_LIMIT value is blank — OK, no incidents")
    {
        setPower(DCGM_ST_OK, 100.0);
        dcgmcm_sample_t s = {};
        s.val.i64         = DCGM_INT64_BLANK;
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_CPU_POWER_LIMIT_WATTS, DCGM_ST_OK, s);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorCpuPower(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Power below limit — no incidents")
    {
        setPower(DCGM_ST_OK, 100.0);
        setLimit(DCGM_ST_OK, 200.0);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorCpuPower(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Power exactly at limit — FAIL with DCGM_FR_FIELD_THRESHOLD_DBL")
    {
        setPower(DCGM_ST_OK, 200.0);
        setLimit(DCGM_ST_OK, 200.0);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorCpuPower(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_POWER);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_FIELD_THRESHOLD_DBL);
    }

    SECTION("Power exceeds limit — FAIL")
    {
        setPower(DCGM_ST_OK, 250.0);
        setLimit(DCGM_ST_OK, 200.0);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorCpuPower(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
    }

    SECTION("GetLatestSample hard error is propagated")
    {
        setPower(DCGM_ST_GENERIC_ERROR);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorCpuPower(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// MonitorNVLinkStatus
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchGpuNvLinkFixture, "DcgmHealthWatch::MonitorNVLinkStatus")
{
    SECTION("Non-GPU entity group — returns OK, no incidents")
    {
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkStatus(healthWatch, DCGM_FE_SWITCH, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("GetEntityNvLinkLinkStatus returns error — propagated")
    {
        mockCtx.linkStatusRet = DCGM_ST_GENERIC_ERROR;
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkStatus(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("All links NotSupported (default) — no incidents")
    {
        // linkStates is zero-initialized to DcgmNvLinkLinkStateNotSupported
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkStatus(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("All links Disabled — no incidents")
    {
        for (int i = 0; i < DCGM_NVLINK_MAX_LINKS_PER_GPU; ++i)
            mockCtx.linkStates[i] = DcgmNvLinkLinkStateDisabled;
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkStatus(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("One link Down, MIG disabled — FAIL with DCGM_FR_NVLINK_DOWN")
    {
        mockCtx.linkStates[3] = DcgmNvLinkLinkStateDown;
        dcgmcm_sample_t s     = {};
        s.val.i64             = 0; // MIG disabled
        mockCtx.latestSample.SetFieldSample(DCGM_FI_DEV_MIG_MODE, DCGM_ST_OK, s);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkStatus(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 1);
        REQUIRE(healthResponse.incidents[0].health == DCGM_HEALTH_RESULT_FAIL);
        REQUIRE(healthResponse.incidents[0].system == DCGM_HEALTH_WATCH_NVLINK);
        REQUIRE(healthResponse.incidents[0].error.code == DCGM_FR_NVLINK_DOWN);
    }

    SECTION("One link Down, MIG enabled — treated as disabled, no incidents")
    {
        mockCtx.linkStates[3] = DcgmNvLinkLinkStateDown;
        dcgmcm_sample_t s     = {};
        s.val.i64             = 1; // MIG enabled
        mockCtx.latestSample.SetFieldSample(DCGM_FI_DEV_MIG_MODE, DCGM_ST_OK, s);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkStatus(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Multiple links Down, MIG disabled — one FAIL per down link")
    {
        mockCtx.linkStates[0] = DcgmNvLinkLinkStateDown;
        mockCtx.linkStates[5] = DcgmNvLinkLinkStateDown;
        mockCtx.linkStates[9] = DcgmNvLinkLinkStateDown;
        dcgmcm_sample_t s     = {};
        s.val.i64             = 0; // MIG disabled
        mockCtx.latestSample.SetFieldSample(DCGM_FI_DEV_MIG_MODE, DCGM_ST_OK, s);

        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkStatus(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 3);
        for (unsigned int i = 0; i < healthResponse.incidentCount; i++)
        {
            REQUIRE(healthResponse.incidents[i].error.code == DCGM_FR_NVLINK_DOWN);
        }
    }

    SECTION("Link Up — no incidents")
    {
        mockCtx.linkStates[0] = DcgmNvLinkLinkStateUp;
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLinkStatus(healthWatch, testEntityGroupId, testGpuId, response);
        REQUIRE(ret == DCGM_ST_OK);
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }
}

// ---------------------------------------------------------------------------
// MonitorNVLink (orchestrator)
// ---------------------------------------------------------------------------
TEST_CASE_METHOD(DcgmHealthWatchGetSamplesFixture, "DcgmHealthWatch::MonitorNVLink")
{
    // MockCoreCallbackSamplesLatestGpuInfo returns DCGM_ST_OK for unknown subcommands
    // (including GetEntityNvLinkLinkStatus), so all link states are zero = NotSupported.
    // All field data unconfigured → NOT_WATCHED/NO_DATA → no incidents.

    SECTION("All sub-methods clean — no incidents")
    {
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLink(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        // MonitorNVLink5Fields returns NOT_WATCHED when its descending GetSamples sees an
        // unwatched field whose ascending sample is non-blank (zero-init value). That bubbles
        // up through MonitorNVLink's last-error-wins accumulation. Accept both OK and
        // NOT_WATCHED as "clean" — the important property is zero incidents.
        REQUIRE((ret == DCGM_ST_OK || ret == DCGM_ST_NOT_WATCHED));
        response.PopulateHealthResponse(healthResponse);
        REQUIRE(healthResponse.incidentCount == 0);
    }

    SECTION("Sub-method error is propagated — MonitorFabricFields returns error last")
    {
        // Trigger an error in MonitorFabricFields (4th of 5 sub-methods). MonitorNVLinkErrorFields
        // (5th) returns OK for unconfigured fields, so GENERIC_ERROR is the final return value.
        mockLatestSample.SetFieldSample(DCGM_FI_DEV_FABRIC_HEALTH_MASK, DCGM_ST_GENERIC_ERROR);
        dcgmReturn_t ret
            = DcgmHealthWatchTestHelper::MonitorNVLink(healthWatch, testEntityGroupId, testGpuId, 0, 0, response);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
    }
}

TEST_CASE("DcgmHealthWatch string helpers")
{
    SECTION("GIVEN health systems WHEN names are requested THEN known values are mapped")
    {
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_PCIE) == "PCIe");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_NVLINK) == "NVLink");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_PMU) == "PMU");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_MCU) == "MCU");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_MEM) == "Memory");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_SM) == "SM");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_INFOROM) == "Inforom");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_THERMAL) == "Thermal");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_POWER) == "Power");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_DRIVER) == "Driver");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_NVSWITCH_NONFATAL)
              == "NVSwitch non-fatal errors");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_NVSWITCH_FATAL) == "NVSwitch fatal errors");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_CONNECTX) == "ConnectX");
        CHECK(DcgmHealthWatch::GetHealthSystemAsString(DCGM_HEALTH_WATCH_ALL) == "Unknown");
    }

    SECTION("GIVEN health results WHEN names are requested THEN known values are mapped")
    {
        CHECK(DcgmHealthWatch::GetHealthResultAsString(DCGM_HEALTH_RESULT_PASS) == "PASS");
        CHECK(DcgmHealthWatch::GetHealthResultAsString(DCGM_HEALTH_RESULT_WARN) == "WARNING");
        CHECK(DcgmHealthWatch::GetHealthResultAsString(DCGM_HEALTH_RESULT_FAIL) == "FAILURE");
        CHECK(DcgmHealthWatch::GetHealthResultAsString(static_cast<dcgmHealthWatchResults_t>(0xFFFFFFFF)) == "UNKNOWN");
    }

    SECTION("GIVEN entity groups WHEN names are requested THEN known values are mapped")
    {
        CHECK(std::string(EntityToString(DCGM_FE_GPU)) == "GPU");
        CHECK(std::string(EntityToString(DCGM_FE_VGPU)) == "VGPU");
        CHECK(std::string(EntityToString(DCGM_FE_SWITCH)) == "NvSwitch");
        CHECK(std::string(EntityToString(DCGM_FE_GPU_I)) == "GPU Instance");
        CHECK(std::string(EntityToString(DCGM_FE_GPU_CI)) == "Compute Instance");
        CHECK(std::string(EntityToString(DCGM_FE_LINK)) == "Link");
        CHECK(std::string(EntityToString(DCGM_FE_NONE)) == "Unknown");
    }
}
