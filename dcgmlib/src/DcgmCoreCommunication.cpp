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
#include <chrono>
#include <fmt/format.h>
#include <mutex>
#include <thread>

#include <DcgmStringHelpers.h>
#include <dcgm_module_structs.h>

#include "DcgmCoreCommunication.h"
#include "DcgmHostEngineHandler.h"
#include "DcgmModule.h"
#include "dcgm_core_communication.h"
#include "dcgm_nvswitch_structs.h"
#include "dcgm_structs.h"

dcgmReturn_t PostRequestToCore(dcgm_module_command_header_t *header, void *poster)
{
    if (poster == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    auto communicator = reinterpret_cast<DcgmCoreCommunication *>(poster);
    return communicator->ProcessRequestInCore(header);
}

void DcgmCoreCommunication::Init(DcgmCacheManager *cm, DcgmGroupManager *gm)
{
    m_cacheManagerPtr = cm;
    m_groupManagerPtr = gm;
}

bool DcgmCoreCommunication::IsInitialized() const
{
    return m_cacheManagerPtr != nullptr && m_groupManagerPtr != nullptr;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetGpuIds(dcgm_module_command_header_t *header)
{
    dcgmCoreGetGpuList_t cgg;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetGpuList_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&cgg, header, sizeof(cgg));

    std::vector<unsigned int> gpuIds;
    cgg.response.ret = m_cacheManagerPtr->GetGpuIds(cgg.request.flag, gpuIds);

    cgg.response.gpuCount = gpuIds.size();
    for (size_t i = 0; i < gpuIds.size(); i++)
    {
        cgg.response.gpuIds[i] = gpuIds[i];
    }

    memcpy(header, &cgg, sizeof(cgg));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessAreAllGpuIdsSameSku(dcgm_module_command_header_t *header)
{
    dcgmCoreQueryGpuList_t qg;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreQueryGpuList_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&qg, header, sizeof(qg));

    std::vector<unsigned int> gpuIds;
    for (unsigned int i = 0; i < qg.request.gpuCount; i++)
    {
        gpuIds.push_back(qg.request.gpuIds[i]);
    }

    qg.response.uintAnswer = m_cacheManagerPtr->AreAllGpuIdsSameSku(gpuIds);

    memcpy(header, &qg, sizeof(qg));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetDriverVersion(dcgm_module_command_header_t *header)
{
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreReqIdGetDriverVersion_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    // Cast header to the actual struct type and write directly
    auto *gdv = reinterpret_cast<dcgmCoreReqIdGetDriverVersion_t *>(header);

    auto driverVersion = m_cacheManagerPtr->GetDriverVersion();
    SafeCopyTo(gdv->driverVersion, driverVersion.c_str());
    log_verbose("Writing driver version {} from CacheManager", driverVersion);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetGpuCount(dcgm_module_command_header_t *header)
{
    dcgmCoreGetGpuCount_t ggc;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetGpuCount_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&ggc, header, sizeof(ggc));

    ggc.response.uintAnswer = m_cacheManagerPtr->GetGpuCount(ggc.request.flag);

    memcpy(header, &ggc, sizeof(ggc));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessAddFieldWatch(dcgm_module_command_header_t *header)
{
    dcgmCoreAddFieldWatch_v2 afw;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreAddFieldWatch_version2);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&afw, header, sizeof(afw));

    DcgmWatcher watcher(afw.request.watcherType, afw.request.connectionId);
    afw.ret = m_cacheManagerPtr->AddFieldWatch(afw.request.entityGroupId,
                                               afw.request.entityId,
                                               afw.request.fieldId,
                                               afw.request.monitorFreqUsec,
                                               afw.request.maxSampleAge,
                                               afw.request.maxKeepSamples,
                                               watcher,
                                               afw.request.subscribeForUpdates,
                                               afw.request.updateOnFirstWatch,
                                               afw.response.wereFirstWatcher);

    memcpy(header, &afw, sizeof(afw));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetInt64SummaryData(dcgm_module_command_header_t *header)
{
    dcgmCoreGetInt64SummaryData_t gisd;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetInt64SummaryData_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&gisd, header, sizeof(gisd));

    gisd.response.ret = m_cacheManagerPtr->GetInt64SummaryData(gisd.request.entityGroupId,
                                                               gisd.request.entityId,
                                                               gisd.request.fieldId,
                                                               gisd.request.summaryCount,
                                                               gisd.request.summaries,
                                                               gisd.response.summaryValues,
                                                               gisd.request.startTime,
                                                               gisd.request.endTime,
                                                               gisd.request.useEntryCB,
                                                               gisd.request.userData);

    memcpy(header, &gisd, sizeof(gisd));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetLatestSample(dcgm_module_command_header_t *header)
{
    dcgmCoreGetLatestSample_t gls;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetLatestSample_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&gls, header, sizeof(gls));

    dcgmcm_sample_p samplePtr = nullptr;
    dcgmcm_sample_t sample {};
    DcgmFvBuffer fvbuf(0);

    if (gls.request.populateSamples)
    {
        samplePtr = &sample;
    }

    gls.response.ret = m_cacheManagerPtr->GetLatestSample(
        gls.request.entityGroupId, gls.request.entityId, gls.request.fieldId, samplePtr, &fvbuf);
    gls.response.sample = sample;

    if (gls.request.populateFvBuffer)
    {
        size_t elementCount;
        fvbuf.GetSize(&gls.response.bufferSize, &elementCount);
        if (gls.response.bufferSize > sizeof(gls.response.buffer))
        {
            gls.response.bufferSize = sizeof(gls.response.buffer);
        }

        memcpy(&gls.response.buffer, fvbuf.GetBuffer(), gls.response.bufferSize);
    }

    memcpy(header, &gls, sizeof(gls));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetEntityNvLinkLinkStatus(dcgm_module_command_header_t *header)
{
    dcgmCoreGetEntityNvLinkLinkStatus_t nls;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetEntityNvLinkLinkStatus_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&nls, header, sizeof(nls));

    /* Dispatch NvSwitch requests to the NvSwitch module */
    if (nls.request.entityGroupId == DCGM_FE_SWITCH)
    {
        dcgm_nvswitch_msg_get_link_states_t nvsMsg {};

        nvsMsg.header.length       = sizeof(nvsMsg);
        nvsMsg.header.moduleId     = DcgmModuleIdNvSwitch;
        nvsMsg.header.subCommand   = DCGM_NVSWITCH_SR_GET_LINK_STATES;
        nvsMsg.header.connectionId = header->connectionId;
        nvsMsg.header.requestId    = header->requestId;
        nvsMsg.header.version      = dcgm_nvswitch_msg_get_link_states_version;
        nvsMsg.entityId            = nls.request.entityId;

        nls.response.ret = DcgmHostEngineHandler::Instance()->ProcessModuleCommand(&nvsMsg.header);

        static_assert(sizeof(nls.response.linkStates) == sizeof(nvsMsg.linkStates), "size mismatch");
        memcpy(nls.response.linkStates, nvsMsg.linkStates, sizeof(nls.response.linkStates));
    }
    else
    {
        /* It's not a NvSwitch. Dispatch to the cache manager */
        nls.response.ret = m_cacheManagerPtr->GetEntityNvLinkLinkStatus(
            nls.request.entityGroupId, nls.request.entityId, nls.response.linkStates);
    }

    memcpy(header, &nls, sizeof(nls));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetSamples(dcgm_module_command_header_t *header)
{
    dcgmCoreGetSamples_t gs;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetSamples_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&gs, header, sizeof(gs));

    gs.response.numSamples = gs.request.maxSamples;
    gs.response.ret        = m_cacheManagerPtr->GetSamples(gs.request.entityGroupId,
                                                    gs.request.entityId,
                                                    gs.request.fieldId,
                                                    gs.response.samples,
                                                    &gs.response.numSamples,
                                                    gs.request.startTime,
                                                    gs.request.endTime,
                                                    gs.request.order,
                                                    nullptr);

    memcpy(header, &gs, sizeof(gs));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGpuIdToNvmlIndex(dcgm_module_command_header_t *header)
{
    dcgmCoreBasicQuery_t br;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreBasicQuery_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&br, header, sizeof(br));

    br.response.uintAnswer = m_cacheManagerPtr->GpuIdToNvmlIndex(br.request.entityId);

    memcpy(header, &br, sizeof(br));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetMultipleLatestLiveSamples(dcgm_module_command_header_t *header)
{
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetMultipleLatestLiveSamples_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    std::unique_ptr<dcgmCoreGetMultipleLatestLiveSamples_t> gml
        = std::make_unique<dcgmCoreGetMultipleLatestLiveSamples_t>();

    memcpy(gml.get(), header, sizeof(*gml));

    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;

    // If the buffer isn't large enough to copy the entire buffer cache into, then we copy by pieces and the
    // caller has to request the follow-on pieces. The caller does so by setting the position of the buffer
    // to copy from. If the position is 0, then that means this is a new request and the cached buffer should
    // not be used even if there's data in it.
    if (gml->request.bufferPosition == 0)
    {
        delete[] m_fvBufferCache;
        m_fvBufferCache = nullptr;

        for (size_t i = 0; i < gml->request.numFieldIds; i++)
        {
            fieldIds.push_back(gml->request.fieldIds[i]);
        }

        for (unsigned int i = 0; i < gml->request.entityPairCount; i++)
        {
            entities.push_back(gml->request.entityPairs[i]);
        }

        DcgmFvBuffer fv;

        gml->response.ret  = m_cacheManagerPtr->GetMultipleLatestLiveSamples(entities, fieldIds, &fv);
        const char *buffer = fv.GetBuffer();
        size_t elementCount;
        fv.GetSize(&gml->response.bufferSize, &elementCount);

        if (gml->response.bufferSize > sizeof(gml->response.buffer))
        {
            // We need to cache this response and copy it piecemeal
            m_fvBufferCache = new char[gml->response.bufferSize];
            memcpy(m_fvBufferCache, buffer, gml->response.bufferSize);
            m_fvbufferCacheSize      = gml->response.bufferSize;
            gml->response.bufferSize = sizeof(gml->response.buffer);
            memcpy(&gml->response.buffer, buffer, sizeof(gml->response.buffer));
            gml->response.dataDidNotFit = 1;
        }
        else
        {
            // The entire response fits, so copy the whole thing into the buffer
            memcpy(&gml->response.buffer, buffer, gml->response.bufferSize);
            gml->response.dataDidNotFit = 0;
        }
    }
    else
    {
        // This is a follow-on request, so do not query for new data, copy from the cached data

        if (m_fvBufferCache == nullptr)
        {
            DCGM_LOG_ERROR << "Requested to continue copying latest values, but no values are buffered!";
            return DCGM_ST_BADPARAM;
        }

        size_t remainingToCopy = m_fvbufferCacheSize - gml->request.bufferPosition;
        if (m_fvbufferCacheSize < gml->request.bufferPosition)
        {
            // Somehow they're requesting data beyond our cached buffer size, send back an empty buffer
            gml->response.dataDidNotFit = 0;
            gml->response.bufferSize    = 0;
        }
        else if (remainingToCopy > sizeof(gml->response.buffer))
        {
            // The remaining data doesn't fit either, copy a piece of it.
            memcpy(&gml->response.buffer, m_fvBufferCache + gml->request.bufferPosition, sizeof(gml->response.buffer));
            gml->response.dataDidNotFit = 1;
            gml->response.bufferSize    = sizeof(gml->response.buffer);
        }
        else
        {
            // The remaining data fits in the buffer, send it all.
            memcpy(&gml->response.buffer, m_fvBufferCache + gml->request.bufferPosition, remainingToCopy);
            gml->response.dataDidNotFit = 0;
            gml->response.bufferSize    = remainingToCopy;
        }
    }

    memcpy(header, gml.get(), sizeof(*gml));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessRemoveFieldWatch(dcgm_module_command_header_t *header)
{
    dcgmCoreRemoveFieldWatch_t rfw;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreRemoveFieldWatch_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&rfw, header, sizeof(rfw));

    DcgmWatcher watcher(rfw.request.watcherType, rfw.request.connectionId);
    rfw.ret = m_cacheManagerPtr->RemoveFieldWatch(
        rfw.request.entityGroupId, rfw.request.entityId, rfw.request.fieldId, rfw.request.clearCache, watcher);

    memcpy(header, &rfw, sizeof(rfw));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetAllGpuInfo(dcgm_module_command_header_t *header)
{
    std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;
    dcgmCoreQueryGpuInfo_t ginfo;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreQueryGpuInfo_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&ginfo, header, sizeof(ginfo));

    ginfo.response.ret = m_cacheManagerPtr->GetAllGpuInfo(gpuInfo);

    for (unsigned int i = 0; i < gpuInfo.size(); i++)
    {
        ginfo.response.info[i] = gpuInfo[i];
    }

    ginfo.response.infoCount = gpuInfo.size();

    memcpy(header, &ginfo, sizeof(ginfo));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessAppendSamples(dcgm_module_command_header_t *header)
{
    dcgmCoreAppendSamples_t as;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreAppendSamples_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&as, header, sizeof(as));
    DcgmFvBuffer fvbuf;
    fvbuf.SetFromBuffer(as.request.buffer, as.request.bufferSize);

    as.ret = m_cacheManagerPtr->AppendSamples(&fvbuf);
    memcpy(header, &as, sizeof(as));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessSetValue(dcgm_module_command_header_t *header)
{
    dcgmCoreSetValue_t sv;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreSetValue_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&sv, header, sizeof(sv));

    sv.ret = m_cacheManagerPtr->SetValue(sv.request.gpuId, sv.request.fieldId, sv.request.value);
    memcpy(header, &sv, sizeof(sv));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessNvmlIndexToGpuId(dcgm_module_command_header_t *header)
{
    dcgmCoreBasicQuery_t bq;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreBasicQuery_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&bq, header, sizeof(bq));

    bq.response.uintAnswer = m_cacheManagerPtr->NvmlIndexToGpuId(bq.request.entityId);
    memcpy(header, &bq, sizeof(bq));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessUpdateAllFields(dcgm_module_command_header_t *header)
{
    dcgmCoreGetGpuCount_t qg;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetGpuCount_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&qg, header, sizeof(qg));

    qg.response.ret = m_cacheManagerPtr->UpdateAllFields(qg.request.flag);
    memcpy(header, &qg, sizeof(qg));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessVerifyAndUpdateGroupId(dcgm_module_command_header_t *header)
{
    dcgmCoreBasicQuery_t bq;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreBasicQuery_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&bq, header, sizeof(bq));
    bq.response.uintAnswer = bq.request.entityId;

    bq.response.ret = m_groupManagerPtr->verifyAndUpdateGroupId(&bq.response.uintAnswer);
    memcpy(header, &bq, sizeof(bq));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetGroupEntities(dcgm_module_command_header_t *header)
{
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetGroupEntities_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    std::unique_ptr<dcgmCoreGetGroupEntities_t> gge = std::make_unique<dcgmCoreGetGroupEntities_t>();

    memcpy(gge.get(), header, sizeof(*gge));

    std::vector<dcgmGroupEntityPair_t> entities;
    gge->response.ret = m_groupManagerPtr->GetGroupEntities(gge->request.groupId, entities);

    if (entities.size() > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        DCGM_LOG_ERROR << fmt::format("Too many entities in the group {}. Provided {}, supported up to {}",
                                      gge->request.groupId,
                                      entities.size(),
                                      DCGM_GROUP_MAX_ENTITIES_V2);
        return DCGM_ST_MAX_LIMIT;
    }

    for (size_t i = 0; i < entities.size(); i++)
    {
        gge->response.entityPairs[i] = entities[i];
    }

    gge->response.entityPairsCount = entities.size();

    memcpy(header, gge.get(), sizeof(*gge));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessAreAllTheSameSku(dcgm_module_command_header_t *header)
{
    dcgmCoreBasicGroup_t bg;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreBasicGroup_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&bg, header, sizeof(bg));
    int allSameSku = 0;

    bg.response.ret = m_groupManagerPtr->AreAllTheSameSku(bg.request.connectionId, bg.request.groupId, &allSameSku);
    bg.response.uintAnswer = allSameSku;

    memcpy(header, &bg, sizeof(bg));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetGroupGpuIds(dcgm_module_command_header_t *header)
{
    dcgmCoreGetGroupGpuIds_t ggg;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetGroupGpuIds_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&ggg, header, sizeof(ggg));

    std::vector<unsigned int> gpuIds;
    ggg.response.ret      = m_groupManagerPtr->GetGroupGpuIds(ggg.request.connectionId, ggg.request.groupId, gpuIds);
    ggg.response.gpuCount = gpuIds.size();
    for (size_t i = 0; i < gpuIds.size(); i++)
    {
        ggg.response.gpuIds[i] = gpuIds[i];
    }

    memcpy(header, &ggg, sizeof(ggg));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessLoggingGetSeverity(dcgm_module_command_header_t *header)
{
    dcgmCoreGetSeverity_t getSeverity;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetSeverity_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&getSeverity, header, sizeof(getSeverity));

    bool success = false;

    switch (getSeverity.request)
    {
        case BASE_LOGGER:
            getSeverity.response = GetLoggerSeverity(BASE_LOGGER);
            success              = true;
            break;
        case SYSLOG_LOGGER:
            getSeverity.response = GetLoggerSeverity(SYSLOG_LOGGER);
            success              = true;
            break;
        case CONSOLE_LOGGER:
            getSeverity.response = GetLoggerSeverity(CONSOLE_LOGGER);
            success              = true;
            break;
        case FILE_LOGGER:
            getSeverity.response = GetLoggerSeverity(FILE_LOGGER);
            success              = true;
            break;
            // Do not add a default case so that the compiler catches missing loggers
    }

    if (!success)
    {
        getSeverity.response = DcgmLoggingSeverityUnspecified;
    }

    memcpy(header, &getSeverity, sizeof(getSeverity));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessSendModuleCommand(dcgm_module_command_header_t *header)
{
    dcgmCoreSendModuleCommand_t smc;
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreSendModuleCommand_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&smc, header, sizeof(smc));

    smc.response = DcgmHostEngineHandler::Instance()->ProcessModuleCommand((dcgm_module_command_header_t *)smc.command);

    memcpy(header, &smc, sizeof(smc));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessSendRawMessageToClient(dcgm_module_command_header_t *header)
{
    dcgmCoreSendRawMessage_t msg;
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreSendRawMessage_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&msg, header, sizeof(msg));

    msg.response = DcgmHostEngineHandler::Instance()->SendRawMessageToClient(msg.request.connectionId,
                                                                             msg.request.msgType,
                                                                             msg.request.requestId,
                                                                             msg.request.msgData,
                                                                             msg.request.msgSize,
                                                                             msg.request.status);

    memcpy(header, &msg, sizeof(msg));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessNotifyRequestOfCompletion(dcgm_module_command_header_t *header)
{
    dcgmCoreNotifyRequestOfCompletion_t msg;
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreNotifyRequestOfCompletion_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&msg, header, sizeof(msg));

    DcgmHostEngineHandler::Instance()->NotifyRequestOfCompletion(msg.request.connectionId, msg.request.requestId);

    msg.response = DCGM_ST_OK;

    memcpy(header, &msg, sizeof(msg));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessPopulateFieldGroupGetAll(dcgm_module_command_header_t *header)
{
    dcgmCorePopulateFieldGroups_t req;

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCorePopulateFieldGroups_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&req, header, sizeof(req));

    DcgmFieldGroupManager *fieldGroupManager = DcgmHostEngineHandler::Instance()->GetFieldGroupManager();

    req.response.status = fieldGroupManager->PopulateFieldGroupGetAll(&req.response.groups);

    memcpy(header, &req, sizeof(req));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetFieldGroupGetFields(dcgm_module_command_header_t *header)
{
    dcgmCoreGetFieldGroupFields_t req = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreGetFieldGroupFields_version);

    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&req, header, sizeof(req));

    std::vector<unsigned short> fields;
    DcgmFieldGroupManager *fieldGroupManager = DcgmHostEngineHandler::Instance()->GetFieldGroupManager();

    req.response.ret = fieldGroupManager->GetFieldGroupFields(req.fieldGrp, fields);

    for (size_t i = 0; i < fields.size(); i++)
    {
        req.response.fieldIds[i] = fields[i];
    }

    req.response.numFieldIds = fields.size();

    memcpy(header, &req, sizeof(req));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetMigUtilization(dcgm_module_command_header_t *header)
{
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmCoreGetMigUtilization_t query {};
    auto ret = DcgmModule::CheckVersion(header, dcgmCoreGetMigUtilization_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));
    switch (query.request.entityGroupId)
    {
        case DCGM_FE_GPU:
        {
            query.response.ret = m_cacheManagerPtr->GetMigGpuPopulation(
                query.request.gpuId, &query.response.capacity, &query.response.usage);
            break;
        }
        case DCGM_FE_GPU_I:
            query.response.ret = m_cacheManagerPtr->GetMigInstancePopulation(
                query.request.gpuId,
                DcgmNs::Mig::Nvml::GpuInstanceId { query.request.instanceId },
                &query.response.capacity,
                &query.response.usage);
            break;
        case DCGM_FE_GPU_CI:
            query.response.ret = m_cacheManagerPtr->GetMigComputeInstancePopulation(
                query.request.gpuId,
                DcgmNs::Mig::Nvml::GpuInstanceId { query.request.instanceId },
                DcgmNs::Mig::Nvml::ComputeInstanceId { query.request.computeInstanceId },
                &query.response.capacity,
                &query.response.usage);
            break;
        default:
            DCGM_LOG_ERROR << "[Mig] Unable to provide MIG utilization for entity group "
                           << query.request.entityGroupId;
            query.response.ret = DCGM_ST_NO_DATA;
            break;
    }

    memcpy(header, &query, sizeof(query));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetMigInstanceEntityId(dcgm_module_command_header_t *header)
{
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    dcgmCoreGetComputeInstanceEntityId_t query {};
    auto ret = DcgmModule::CheckVersion(header, dcgmCoreGetComputeInstanceEntityId_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    using DcgmNs::Mig::Nvml::ComputeInstanceId;
    using DcgmNs::Mig::Nvml::GpuInstanceId;
    auto entityId
        = query.request.entityGroupId == DCGM_FE_GPU_CI
              ? m_cacheManagerPtr->GetComputeInstanceEntityId(query.request.gpuId,
                                                              ComputeInstanceId { query.request.computeInstanceId },
                                                              GpuInstanceId { query.request.instanceId })
              : m_cacheManagerPtr->GetInstanceEntityId(query.request.gpuId, GpuInstanceId { query.request.instanceId });


    query.response.uintAnswer = entityId;
    query.response.ret        = entityId == DCGM_BLANK_ENTITY_ID ? DCGM_ST_NO_DATA : DCGM_ST_OK;

    memcpy(header, &query, sizeof(query));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetMigIndicesForEntity(dcgm_module_command_header_t *header)
{
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    dcgmCoreGetMigIndicesForEntity_t query {};
    auto ret = DcgmModule::CheckVersion(header, dcgmCoreGetMigIndicesForEntity_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }
    memcpy(&query, header, sizeof(query));

    auto const entityGroupId = query.request.entityGroupId;
    auto const entityId      = query.request.entityId;

    if (entityGroupId != DCGM_FE_GPU_I && entityGroupId != DCGM_FE_GPU_CI)
    {
        query.response.ret = DCGM_ST_BADPARAM;
    }
    else
    {
        unsigned int gpuId = {};
        DcgmNs::Mig::GpuInstanceId instanceId {};
        DcgmNs::Mig::ComputeInstanceId computeInstanceId {};
        auto result = m_cacheManagerPtr->GetMigIndicesForEntity(
            { entityGroupId, entityId }, &gpuId, &instanceId, &computeInstanceId);
        query.response.ret = result;
        if (result == DCGM_ST_OK)
        {
            query.response.gpuId      = gpuId;
            query.response.instanceId = instanceId.id;
        }
    }

    memcpy(header, &query, sizeof(query));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetServiceAccount(dcgm_module_command_header_t *header)
{
    dcgmCoreGetServiceAccount_v1 query = {};
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (auto const ret = DcgmModule::CheckVersion(header, dcgmCoreGetServiceAccount_version1); ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    auto const &serviceAccount = DcgmHostEngineHandler::Instance()->GetServiceAccount();

    std::strncpy(query.response.serviceAccount, serviceAccount.c_str(), sizeof(query.response.serviceAccount));
    query.response.serviceAccount[sizeof(query.response.serviceAccount) - 1] = '\0';

    memcpy(header, &query, sizeof(query));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessGetGpuInstanceHierarchy(dcgm_module_command_header_t *header)
{
    if (header == nullptr || header->length != sizeof(dcgmCoreGetGpuInstanceHierarchy_t))
    {
        return DCGM_ST_BADPARAM;
    }

    if (auto const ret = DcgmModule::CheckVersion(header, dcgmCoreGetGpuInstanceHierarchy_version1); ret != DCGM_ST_OK)
    {
        return ret;
    }

    auto *query = reinterpret_cast<dcgmCoreGetGpuInstanceHierarchy_t *>(header);
    auto ret    = m_cacheManagerPtr->PopulateMigHierarchy(query->response);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessResourceReserve(dcgm_module_command_header_t *header)
{
    dcgmCoreResourceReserve_v1 query = {};
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (auto const ret = DcgmModule::CheckVersion(header, dcgmCoreResourceReserve_version1); ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    auto token = DcgmHostEngineHandler::Instance()->ReserveResources();

    if (token)
    {
        query.response.ret   = DCGM_ST_OK;
        query.response.token = *token;
    }
    else
    {
        query.response.ret = DCGM_ST_IN_USE;
    }

    memcpy(header, &query, sizeof(query));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessResourceFree(dcgm_module_command_header_t *header)
{
    dcgmCoreResourceFree_v1 query = {};
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (auto const ret = DcgmModule::CheckVersion(header, dcgmCoreResourceFree_version1); ret != DCGM_ST_OK)
    {
        return ret;
    }

    if (header->length < sizeof(query))
    {
        DCGM_LOG_ERROR << "Header length " << header->length << " is smaller than expected size " << sizeof(query);
        return DCGM_ST_BADPARAM;
    }
    memcpy(&query, header, sizeof(query));

    query.ret
        = DcgmHostEngineHandler::Instance()->FreeResources(query.request.token) ? DCGM_ST_OK : DCGM_ST_GENERIC_ERROR;

    memcpy(header, &query, sizeof(query));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessSpawn(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessSpawn_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessSpawn_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessSpawn(query.request, query.handle, query.pid);

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessStop(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessStop_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessStop_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessStop(query.handle, query.force);

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessGetStatus(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessGetStatus_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessGetStatus_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessGetStatus(query.handle, query.status);

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessWait(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessWait_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessWait_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessWait(query.handle, query.timeoutSec);

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessDestroy(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessDestroy_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessDestroy_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessDestroy(query.handle, query.sigTermTimeoutSec);

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessGetStdErrHandle(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessGetStdErrHandle_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessGetStdErrHandle_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessGetStdErrHandle(query.handle, query.fd);

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessGetStdOutHandle(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessGetStdOutHandle_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessGetStdOutHandle_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessGetStdOutHandle(query.handle, query.fd);

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessGetDataChannelHandle(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessGetDataChannelHandle_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessGetDataChannelHandle_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessGetDataChannelHandle(query.handle, query.fd);

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessChildProcessManagerReset(dcgm_module_command_header_t *header)
{
    dcgmCoreChildProcessManagerReset_t query = {};

    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DcgmModule::CheckVersion(header, dcgmCoreChildProcessManagerReset_version1);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    memcpy(&query, header, sizeof(query));

    query.ret = DcgmHostEngineHandler::Instance()->ChildProcessManagerReset();

    memcpy(header, &query, sizeof(query));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreCommunication::ProcessRequestInCore(dcgm_module_command_header_t *header)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    if (header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (IsInitialized() == false)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    // NOTE: None of the headers passed in are owned by us, and we do not delete them.
    switch (header->subCommand)
    {
        case DcgmCoreReqIdCMGetGpuIds:
        {
            ret = ProcessGetGpuIds(header);
            break;
        }

        case DcgmCoreReqIdCMAreAllGpuIdsSameSku:
        {
            ret = ProcessAreAllGpuIdsSameSku(header);
            break;
        }

        case DcgmCoreReqIdCMGetGpuCount:
        {
            ret = ProcessGetGpuCount(header);
            break;
        }

        case DcgmCoreReqIdCMAddFieldWatch:
        {
            ret = ProcessAddFieldWatch(header);
            break;
        }

        case DcgmCoreReqIdCMGetInt64SummaryData:
        {
            ret = ProcessGetInt64SummaryData(header);
            break;
        }

        case DcgmCoreReqIdCMGetLatestSample:
        {
            ret = ProcessGetLatestSample(header);
            break;
        }

        case DcgmCoreReqIdCMGetEntityNvLinkLinkStatus:
        {
            ret = ProcessGetEntityNvLinkLinkStatus(header);
            break;
        }

        case DcgmCoreReqIdCMGetSamples:
        {
            ret = ProcessGetSamples(header);
            break;
        }

        case DcgmCoreReqIdCMGpuIdToNvmlIndex:
        {
            ret = ProcessGpuIdToNvmlIndex(header);
            break;
        }

        case DcgmCoreReqIdCMGetMultipleLatestLiveSamples:
        {
            ret = ProcessGetMultipleLatestLiveSamples(header);
            break;
        }

        case DcgmCoreReqIdCMRemoveFieldWatch:
        {
            ret = ProcessRemoveFieldWatch(header);
            break;
        }

        case DcgmCoreReqIdCMGetAllGpuInfo:
        {
            ret = ProcessGetAllGpuInfo(header);
            break;
        }

        case DcgmCoreReqIdCMAppendSamples:
        {
            ret = ProcessAppendSamples(header);
            break;
        }

        case DcgmCoreReqIdCMSetValue:
        {
            ret = ProcessSetValue(header);
            break;
        }

        case DcgmCoreReqIdCMNvmlIndexToGpuId:
        {
            ret = ProcessNvmlIndexToGpuId(header);
            break;
        }

        case DcgmCoreReqIdCMUpdateAllFields:
        {
            ret = ProcessUpdateAllFields(header);
            break;
        }

        case DcgmCoreReqIdGMVerifyAndUpdateGroupId:
        {
            ret = ProcessVerifyAndUpdateGroupId(header);
            break;
        }

        case DcgmCoreReqIdGMGetGroupEntities:
        {
            ret = ProcessGetGroupEntities(header);
            break;
        }

        case DcgmCoreReqIdGMAreAllTheSameSku:
        {
            ret = ProcessAreAllTheSameSku(header);
            break;
        }

        case DcgmCoreReqIdGMGetGroupGpuIds:
        {
            ret = ProcessGetGroupGpuIds(header);
            break;
        }

        case DcgmCoreReqIdLoggingGetSeverity:
        {
            ret = ProcessLoggingGetSeverity(header);
            break;
        }

        case DcgmCoreReqIdSendModuleCommand:
        {
            ret = ProcessSendModuleCommand(header);
            break;
        }

        case DcgmCoreReqIdSendRawMessage:
        {
            ret = ProcessSendRawMessageToClient(header);
            break;
        }

        case DcgmCoreReqIdFGMPopulateFieldGroups:
        {
            ret = ProcessPopulateFieldGroupGetAll(header);
            break;
        }

        case DcgmCoreReqIdNotifyRequestOfCompletion:
        {
            ret = ProcessNotifyRequestOfCompletion(header);
            break;
        }

        case DcgmCoreReqIdFGMGetFieldGroupFields:
        {
            ret = ProcessGetFieldGroupGetFields(header);
            break;
        }

        case DcgmCoreReqIdGetMigInstanceEntityId:
        {
            ret = ProcessGetMigInstanceEntityId(header);
            break;
        }

        case DcgmCoreReqIdGetMigUtilization:
        {
            ret = ProcessGetMigUtilization(header);
            break;
        }

        case DcgmCoreReqMigIndicesForEntity:
        {
            ret = ProcessGetMigIndicesForEntity(header);
            break;
        }

        case DcgmCoreReqGetServiceAccount:
        {
            ret = ProcessGetServiceAccount(header);
            break;
        }

        case DcgmCoreReqPopulateMigHierarchy:
        {
            ret = ProcessGetGpuInstanceHierarchy(header);
            break;
        }

        case DcgmCoreReqIdResourceReserve:
        {
            ret = ProcessResourceReserve(header);
            break;
        }

        case DcgmCoreReqIdResourceFree:
        {
            ret = ProcessResourceFree(header);
            break;
        }

        case DcgmCoreReqIdChildProcessSpawn:
        {
            ret = ProcessChildProcessSpawn(header);
            break;
        }

        case DcgmCoreReqIdChildProcessStop:
        {
            ret = ProcessChildProcessStop(header);
            break;
        }

        case DcgmCoreReqIdChildProcessGetStatus:
        {
            ret = ProcessChildProcessGetStatus(header);
            break;
        }

        case DcgmCoreReqIdChildProcessWait:
        {
            ret = ProcessChildProcessWait(header);
            break;
        }

        case DcgmCoreReqIdChildProcessDestroy:
        {
            ret = ProcessChildProcessDestroy(header);
            break;
        }

        case DcgmCoreReqIdChildProcessGetStdErrHandle:
        {
            ret = ProcessChildProcessGetStdErrHandle(header);
            break;
        }

        case DcgmCoreReqIdChildProcessGetStdOutHandle:
        {
            ret = ProcessChildProcessGetStdOutHandle(header);
            break;
        }

        case DcgmCoreReqIdChildProcessGetDataChannelHandle:
        {
            ret = ProcessChildProcessGetDataChannelHandle(header);
            break;
        }

        case DcgmCoreReqIdGetDriverVersion:
        {
            ret = ProcessGetDriverVersion(header);
            break;
        }

        case DcgmCoreReqIdChildProcessManagerReset:
        {
            ret = ProcessChildProcessManagerReset(header);
            break;
        }

        default:
            DCGM_LOG_DEBUG << "Unhandled sub command " << header->subCommand << " received and ignored.";
            break;
    }

    return ret;
}
