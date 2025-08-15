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
#include <cstring>
#include <fmt/format.h>

#include "DcgmCoreProxy.h"
#include "dcgm_core_communication.h"
#include <DcgmStringHelpers.h>


DcgmCoreProxy::DcgmCoreProxy(const dcgmCoreCallbacks_t coreCallbacks)
    : m_coreCallbacks(coreCallbacks)
{}

void initializeCoreHeader(dcgm_module_command_header_t &header,
                          dcgmCoreReqCmd_t cmd,
                          unsigned int version,
                          size_t reqSize)
{
    header.version    = version;
    header.moduleId   = DcgmModuleIdCore;
    header.subCommand = cmd;
    header.length     = reqSize;
}

dcgmReturn_t DcgmCoreProxy::GetGpuIds(int activeOnly, std::vector<unsigned int> &gpuIds)
{
    dcgmCoreGetGpuList_t cgg = {};
    cgg.request.flag         = activeOnly;

    initializeCoreHeader(cgg.header, DcgmCoreReqIdCMGetGpuIds, dcgmCoreGetGpuList_version, sizeof(cgg));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&cgg.header, m_coreCallbacks.poster);

    gpuIds.clear();
    if (ret == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < cgg.response.gpuCount; i++)
        {
            gpuIds.push_back(cgg.response.gpuIds[i]);
        }

        ret = cgg.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get all GPU ids.";
    }

    return ret;
}

bool DcgmCoreProxy::AreAllGpuIdsSameSku(std::vector<unsigned int> &gpuIds) const
{
    dcgmCoreQueryGpuList_t qg = {};

    size_t end = gpuIds.size();
    if (end > std::extent_v<decltype(qg.request.gpuIds)>)
    {
        end = std::extent_v<decltype(qg.request.gpuIds)>;
        log_error("Too many GPUs in the list. Only the first {} will be used.", end);
    }
    for (size_t i = 0; i < end; i++)
    {
        qg.request.gpuIds[i] = gpuIds[i];
    }
    qg.request.gpuCount = gpuIds.size();

    initializeCoreHeader(qg.header, DcgmCoreReqIdCMAreAllGpuIdsSameSku, dcgmCoreQueryGpuList_version, sizeof(qg));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qg.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        bool sameSku = (qg.response.uintAnswer != 0);
        return sameSku;
    }

    log_error("Error '{}' while determining if all GPUs in our list are the same SKU", errorString(ret));
    return false;
}

dcgmReturn_t DcgmCoreProxy::GetDriverVersion(std::string &driverVersion) const
{
    dcgmCoreReqIdGetDriverVersion_t gdv = {};
    initializeCoreHeader(gdv.header, DcgmCoreReqIdGetDriverVersion, dcgmCoreReqIdGetDriverVersion_version, sizeof(gdv));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&gdv.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        driverVersion = gdv.driverVersion;
        log_verbose("Got driver version {} from core", driverVersion);
    }
    else
    {
        log_error("Error '{}' while getting driver version", errorString(ret));
    }
    return ret;
}

unsigned int DcgmCoreProxy::GetGpuCount(int activeOnly)
{
    dcgmCoreGetGpuCount_t ggc = {};
    ggc.request.flag          = activeOnly;

    initializeCoreHeader(ggc.header, DcgmCoreReqIdCMGetGpuCount, dcgmCoreGetGpuCount_version, sizeof(ggc));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&ggc.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        unsigned int count = ggc.response.uintAnswer;
        return count;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get a GPU count.";
        return 0;
    }
}

dcgmReturn_t DcgmCoreProxy::GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> &gpuInfo)
{
    dcgmCoreQueryGpuInfo_t qgi = {};
    initializeCoreHeader(qgi.header, DcgmCoreReqIdCMGetAllGpuInfo, dcgmCoreQueryGpuInfo_version, sizeof(qgi));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qgi.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < qgi.response.infoCount; i++)
        {
            gpuInfo.push_back(qgi.response.info[i]);
        }

        ret = qgi.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get all GPU information";
    }

    return ret;
}

unsigned int DcgmCoreProxy::NvmlIndexToGpuId(int nvmlIndex)
{
    dcgmCoreBasicQuery_t bq = {};
    bq.request.entityId     = nvmlIndex;

    initializeCoreHeader(bq.header, DcgmCoreReqIdCMNvmlIndexToGpuId, dcgmCoreBasicQuery_version, sizeof(bq));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&bq.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        unsigned int index = bq.response.uintAnswer;
        return index;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << " while attempting to get a GPU id for NVML index "
                       << nvmlIndex << ".";
        return DCGM_MAX_NUM_DEVICES;
    }
}

dcgmReturn_t DcgmCoreProxy::UpdateAllFields(int waitForUpdate)
{
    dcgmCoreGetGpuCount_t qg = {};
    qg.request.flag          = waitForUpdate;

    initializeCoreHeader(qg.header, DcgmCoreReqIdCMUpdateAllFields, dcgmCoreGetGpuCount_version, sizeof(qg));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qg.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret = qg.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << " while attempting to trigger an update for all fields.";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetEntityNvLinkLinkStatus(dcgm_field_entity_group_t entityGroupId,
                                                      dcgm_field_eid_t entityId,
                                                      dcgmNvLinkLinkState_t *linkStates)
{
    dcgmCoreGetEntityNvLinkLinkStatus_t nls = {};
    nls.request.entityGroupId               = entityGroupId;
    nls.request.entityId                    = entityId;

    initializeCoreHeader(
        nls.header, DcgmCoreReqIdCMGetEntityNvLinkLinkStatus, dcgmCoreGetEntityNvLinkLinkStatus_version, sizeof(nls));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&nls.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret = nls.response.ret;

        unsigned int count;
        if (entityGroupId == DCGM_FE_GPU)
        {
            count = DCGM_NVLINK_MAX_LINKS_PER_GPU;
        }
        else
        {
            count = DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH;
        }

        memcpy(linkStates, nls.response.linkStates, count * sizeof(dcgmNvLinkLinkState_t));
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret)
                       << "' while attempting to get NvLinkLinkStatus for entity group " << entityGroupId << " id "
                       << entityId;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::AddFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                          dcgm_field_eid_t entityId,
                                          unsigned short fieldId,
                                          timelib64_t monitorFreqUsec,
                                          double maxSampleAge,
                                          int maxKeepSamples,
                                          DcgmWatcher watcher,
                                          bool subscribeForUpdates,
                                          bool updateOnFirstWatch,
                                          bool &wereFirstWatcher)
{
    dcgmCoreAddFieldWatch_v2 afw    = {};
    afw.request.entityGroupId       = entityGroupId;
    afw.request.entityId            = entityId;
    afw.request.fieldId             = fieldId;
    afw.request.monitorFreqUsec     = monitorFreqUsec;
    afw.request.maxSampleAge        = maxSampleAge;
    afw.request.maxKeepSamples      = maxKeepSamples;
    afw.request.watcherType         = watcher.watcherType;
    afw.request.connectionId        = watcher.connectionId;
    afw.request.subscribeForUpdates = subscribeForUpdates;
    afw.request.updateOnFirstWatch  = updateOnFirstWatch;

    initializeCoreHeader(afw.header, DcgmCoreReqIdCMAddFieldWatch, dcgmCoreAddFieldWatch_version2, sizeof(afw));

    wereFirstWatcher = afw.request.updateOnFirstWatch;

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&afw.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret = afw.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to add field watch: entity group "
                       << entityGroupId << ", entity " << entityId << ", field " << fieldId << ".";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetInt64SummaryData(dcgm_field_entity_group_t entityGroupId,
                                                dcgm_field_eid_t entityId,
                                                unsigned short fieldId,
                                                int numSummaryTypes,
                                                DcgmcmSummaryType_t *summaryTypes,
                                                long long *summaryValues,
                                                timelib64_t startTime,
                                                timelib64_t endTime,
                                                pfUseEntryForSummary pfUseEntryCB,
                                                void *userData)
{
    dcgmCoreGetInt64SummaryData_t gisd = {};
    gisd.request.entityGroupId         = entityGroupId;
    gisd.request.entityId              = entityId;
    gisd.request.fieldId               = fieldId;

    gisd.request.summaryCount = numSummaryTypes;
    for (int i = 0; i < numSummaryTypes; i++)
    {
        gisd.request.summaries[i] = summaryTypes[i];
    }

    gisd.request.startTime  = startTime;
    gisd.request.endTime    = endTime;
    gisd.request.useEntryCB = pfUseEntryCB;
    gisd.request.userData   = userData;

    initializeCoreHeader(
        gisd.header, DcgmCoreReqIdCMGetInt64SummaryData, dcgmCoreGetInt64SummaryData_version, sizeof(gisd));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&gisd.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret = gisd.response.ret;
        memcpy(summaryValues, gisd.response.summaryValues, numSummaryTypes * sizeof(long long));
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get summary data for entity group"
                       << entityGroupId << ", entity " << entityId << ", field " << fieldId;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetLatestSample(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId,
                                            unsigned short fieldId,
                                            dcgmcm_sample_p sample,
                                            DcgmFvBuffer *fvBuffer)
{
    dcgmCoreGetLatestSample_t gls = {};
    gls.request.entityGroupId     = entityGroupId;
    gls.request.entityId          = entityId;
    gls.request.fieldId           = fieldId;
    gls.request.populateSamples   = sample != nullptr;
    gls.request.populateFvBuffer  = fvBuffer != nullptr;

    initializeCoreHeader(gls.header, DcgmCoreReqIdCMGetLatestSample, dcgmCoreGetLatestSample_version, sizeof(gls));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&gls.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret = gls.response.ret;
        if (gls.request.populateSamples)
        {
            memcpy(sample, &gls.response.sample, sizeof(*sample));
        }

        if (gls.request.populateFvBuffer)
        {
            fvBuffer->SetFromBuffer(gls.response.buffer, gls.response.bufferSize);
        }
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get latest sample for entity group "
                       << entityGroupId << ", entity " << entityId << ", field " << fieldId;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetSamples(dcgm_field_entity_group_t entityGroupId,
                                       dcgm_field_eid_t entityId,
                                       unsigned short fieldId,
                                       dcgmcm_sample_p samples,
                                       int *Msamples,
                                       timelib64_t startTime,
                                       timelib64_t endTime,
                                       dcgmOrder_t order)
{
    dcgmCoreGetSamples_t gs = {};
    initializeCoreHeader(gs.header, DcgmCoreReqIdCMGetSamples, dcgmCoreGetSamples_version, sizeof(gs));
    gs.request.entityGroupId = entityGroupId;
    gs.request.entityId      = entityId;
    gs.request.fieldId       = fieldId;
    gs.request.startTime     = startTime;
    gs.request.endTime       = endTime;
    gs.request.maxSamples    = *Msamples;
    gs.request.order         = order;
    gs.response.samples      = samples;

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&gs.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret       = gs.response.ret;
        *Msamples = gs.response.numSamples;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get samples for entity group "
                       << entityGroupId << ", entity " << entityId << ", field " << fieldId;
    }

    return ret;
}

class Blob
{
public:
    Blob()
        : m_size(8192)
        , m_used(0)
        , m_blob(std::make_unique<char[]>(m_size))
    {}

    ~Blob() = default;

    void AppendToBlob(const char *toAppend, size_t appendSize)
    {
        if (m_size - m_used < appendSize)
        {
            GrowBlob(appendSize);
        }

        memcpy(m_blob.get() + m_used, toAppend, appendSize);
        m_used += appendSize;
    }

    void GrowBlob(size_t toAppend)
    {
        size_t bigSize = m_size * 4;

        // Multiply the existing size by 4 until we have enough room to append new data to our blob
        while (bigSize - m_used < toAppend)
        {
            bigSize *= 4;
        }

        char *biggerBlob = new char[bigSize];
        memcpy(biggerBlob, m_blob.get(), m_used);
        m_blob.reset(biggerBlob);
        m_size = bigSize;
    }

    const char *GetBlob() const
    {
        return m_blob.get();
    }

    size_t GetBlobSize() const
    {
        return m_size;
    }

    size_t GetUsed() const
    {
        return m_used;
    }

private:
    size_t m_size;
    size_t m_used;
    std::unique_ptr<char[]> m_blob;
};

dcgmReturn_t DcgmCoreProxy::GetMultipleLatestLiveSamples(std::vector<dcgmGroupEntityPair_t> const &entities,
                                                         std::vector<unsigned short> const &fieldIds,
                                                         DcgmFvBuffer *fvBuffer) const
{
    if (entities.size() > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        DCGM_LOG_ERROR << fmt::format("Too many entities in the group. Provided {}, supported up to {}",
                                      entities.size(),
                                      DCGM_GROUP_MAX_ENTITIES_V2);
        return DCGM_ST_MAX_LIMIT;
    }

    std::unique_ptr<dcgmCoreGetMultipleLatestLiveSamples_t> gml
        = std::make_unique<dcgmCoreGetMultipleLatestLiveSamples_t>();
    memset(gml.get(), 0, sizeof(*gml));

    gml->request.entityPairCount = entities.size();
    for (size_t i = 0; i < entities.size(); i++)
    {
        gml->request.entityPairs[i] = entities[i];
    }

    gml->request.fieldIds    = fieldIds.data();
    gml->request.numFieldIds = fieldIds.size();

    initializeCoreHeader(gml->header,
                         DcgmCoreReqIdCMGetMultipleLatestLiveSamples,
                         dcgmCoreGetMultipleLatestLiveSamples_version,
                         sizeof(*gml));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&gml->header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        Blob blob;
        blob.AppendToBlob(gml->response.buffer, gml->response.bufferSize);

        // If the data didn't fit in one call, repeatedly call the API and copy off each portion until we're done.
        while (gml->response.dataDidNotFit)
        {
            gml->request.bufferPosition = blob.GetUsed();
            // coverity[overrun-buffer-val]
            ret = m_coreCallbacks.postfunc(&gml->header, m_coreCallbacks.poster);

            if (ret != DCGM_ST_OK)
            {
                break;
            }
            else
            {
                blob.AppendToBlob(gml->response.buffer, gml->response.bufferSize);
            }
        }

        if (ret == DCGM_ST_OK)
        {
            fvBuffer->SetFromBuffer(blob.GetBlob(), blob.GetUsed());
        }
    }

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get latest samples for fields ";
        for (size_t i = 0; i < fieldIds.size(); i++)
        {
            if (i != 0)
            {
                DCGM_LOG_ERROR << "," << fieldIds[i];
            }
            else
            {
                DCGM_LOG_ERROR << fieldIds[i];
            }
        }
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::RemoveFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId,
                                             unsigned short fieldId,
                                             int clearCache,
                                             DcgmWatcher watcher) const
{
    dcgmCoreRemoveFieldWatch_t rfw = {};
    rfw.request.entityGroupId      = entityGroupId;
    rfw.request.entityId           = entityId;
    rfw.request.fieldId            = fieldId;
    rfw.request.clearCache         = clearCache;
    rfw.request.watcherType        = watcher.watcherType;
    rfw.request.connectionId       = watcher.connectionId;

    initializeCoreHeader(rfw.header, DcgmCoreReqIdCMRemoveFieldWatch, dcgmCoreRemoveFieldWatch_version, sizeof(rfw));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&rfw.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret = rfw.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret)
                       << "' while while attempting to remove field watch: entity group " << entityGroupId << " entity "
                       << entityId << " field " << fieldId << ".";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::AppendSamples(DcgmFvBuffer *fvBuffer) const
{
    dcgmCoreAppendSamples_t as = {};
    as.request.buffer          = fvBuffer->GetBuffer();
    size_t elementCount        = 0;
    fvBuffer->GetSize(&as.request.bufferSize, &elementCount);

    if (as.request.bufferSize < 1)
    {
        DCGM_LOG_ERROR << "AppendSamples got an empty fvBuffer";
        return DCGM_ST_BADPARAM;
    }

    initializeCoreHeader(as.header, DcgmCoreReqIdCMAppendSamples, dcgmCoreAppendSamples_version, sizeof(as));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&as.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret = as.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to append samples to the cache.";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::SetValue(int gpuId, unsigned short fieldId, dcgmcm_sample_p value) const
{
    dcgmCoreSetValue_t sv = {};
    sv.request.gpuId      = gpuId;
    sv.request.fieldId    = fieldId;
    sv.request.value      = value;

    initializeCoreHeader(sv.header, DcgmCoreReqIdCMSetValue, dcgmCoreSetValue_version, sizeof(sv));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&sv.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret = sv.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to set a value for GPU " << gpuId
                       << " and field " << fieldId << ".";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGroupEntities(unsigned int groupId, std::vector<dcgmGroupEntityPair_t> &entities) const
{
    std::unique_ptr<dcgmCoreGetGroupEntities_t> gge = std::make_unique<dcgmCoreGetGroupEntities_t>();
    memset(gge.get(), 0, sizeof(*gge));
    gge->request.connectionId = 0;
    gge->request.groupId      = groupId;

    initializeCoreHeader(gge->header, DcgmCoreReqIdGMGetGroupEntities, dcgmCoreGetGroupEntities_version, sizeof(*gge));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&gge->header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < gge->response.entityPairsCount; i++)
        {
            entities.push_back(gge->response.entityPairs[i]);
        }

        ret = gge->response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get entities for group " << groupId;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::AreAllTheSameSku(dcgm_connection_id_t connectionId,
                                             unsigned int groupId,
                                             bool *areAllSameSku) const
{
    if (areAllSameSku == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmCoreBasicGroup_t bg = {};
    bg.request.connectionId = connectionId;
    bg.request.groupId      = groupId;

    initializeCoreHeader(bg.header, DcgmCoreReqIdGMAreAllTheSameSku, dcgmCoreBasicGroup_version, sizeof(bg));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&bg.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *areAllSameSku = bg.response.uintAnswer != 0;
        ret            = bg.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString << "' while asking if all GPUs in group " << groupId
                       << " are the same SKU.";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGroupGpuIds(dcgm_connection_id_t connectionId,
                                           unsigned int groupId,
                                           std::vector<unsigned int> &gpuIds) const
{
    dcgmCoreGetGroupGpuIds_t ggg = {};
    ggg.request.connectionId     = connectionId;
    ggg.request.groupId          = groupId;

    initializeCoreHeader(ggg.header, DcgmCoreReqIdGMGetGroupGpuIds, dcgmCoreGetGroupGpuIds_version, sizeof(ggg));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&ggg.header, m_coreCallbacks.poster);

    gpuIds.clear();
    if (ret == DCGM_ST_OK)
    {
        gpuIds.reserve(ggg.response.gpuCount);
        for (unsigned int i = 0; i < ggg.response.gpuCount; i++)
        {
            gpuIds.push_back(ggg.response.gpuIds[i]);
        }

        ret = ggg.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get GPU ids from group " << groupId;
    }

    return ret;
}

int DcgmCoreProxy::GpuIdToNvmlIndex(unsigned int gpuId) const
{
    dcgmCoreBasicQuery_t bq = {};
    bq.request.entityId     = gpuId;

    initializeCoreHeader(bq.header, DcgmCoreReqIdCMGpuIdToNvmlIndex, dcgmCoreBasicQuery_version, sizeof(bq));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&bq.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        int nvmlIndex = bq.response.uintAnswer;
        return nvmlIndex;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to translate GPU " << gpuId
                       << " id to its NVML id.";
        return -1;
    }
}

dcgmReturn_t DcgmCoreProxy::VerifyAndUpdateGroupId(unsigned int *groupId) const
{
    dcgmCoreBasicQuery_t bq = {};
    bq.request.entityId     = *groupId;

    initializeCoreHeader(bq.header, DcgmCoreReqIdGMVerifyAndUpdateGroupId, dcgmCoreBasicQuery_version, sizeof(bq));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&bq.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        ret      = bq.response.ret;
        *groupId = bq.response.uintAnswer;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to verify and update the group id "
                       << *groupId;
    }

    return ret;
}

DcgmLoggingSeverity_t DcgmCoreProxy::GetLoggerSeverity(loggerCategory_t logger) const
{
    dcgmCoreGetSeverity_t getSeverity = {};

    getSeverity.request = logger;

    initializeCoreHeader(
        getSeverity.header, DcgmCoreReqIdLoggingGetSeverity, dcgmCoreGetSeverity_version, sizeof(getSeverity));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&getSeverity.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        return getSeverity.response;
    }

    DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get logger severity";

    return DcgmLoggingSeverityUnspecified;
}

dcgmReturn_t DcgmCoreProxy::SendModuleCommand(void *msg) const
{
    dcgmCoreSendModuleCommand_t smc = {};

    initializeCoreHeader(smc.header, DcgmCoreReqIdSendModuleCommand, dcgmCoreSendModuleCommand_version, sizeof(smc));
    smc.command = msg;

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&smc.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        return smc.response;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::SendRawMessageToClient(dcgm_connection_id_t connectionId,
                                                   unsigned int msgType,
                                                   dcgm_request_id_t requestId,
                                                   void *msgData,
                                                   unsigned int msgSize,
                                                   dcgmReturn_t status) const
{
    dcgmCoreSendRawMessage_t msg = {};

    initializeCoreHeader(msg.header, DcgmCoreReqIdSendRawMessage, dcgmCoreSendRawMessage_version, sizeof(msg));
    msg.request.connectionId = connectionId;
    msg.request.msgType      = msgType;
    msg.request.requestId    = requestId;
    msg.request.msgData      = msgData;
    msg.request.msgSize      = msgSize;
    msg.request.status       = status;

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&msg.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        return msg.response;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::NotifyRequestOfCompletion(dcgm_connection_id_t connectionId, dcgm_request_id_t requestId)
{
    dcgmCoreNotifyRequestOfCompletion_t req = {};

    initializeCoreHeader(
        req.header, DcgmCoreReqIdNotifyRequestOfCompletion, dcgmCoreNotifyRequestOfCompletion_version, sizeof(req));
    req.request.connectionId = connectionId;
    req.request.requestId    = requestId;

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&req.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        return req.response;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::PopulateFieldGroupGetAll(dcgmAllFieldGroup_t *allGroupInfo)
{
    dcgmCorePopulateFieldGroups_t req = {};

    initializeCoreHeader(
        req.header, DcgmCoreReqIdFGMPopulateFieldGroups, dcgmCorePopulateFieldGroups_version, sizeof(req));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&req.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        memcpy(allGroupInfo, &req.response.groups, sizeof(dcgmAllFieldGroup_t));

        return req.response.status;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetFieldGroupFields(dcgmFieldGrp_t fieldGrp, std::vector<unsigned short> &fieldIds)
{
    dcgmCoreGetFieldGroupFields_t fgf = {};

    fgf.fieldGrp = fieldGrp;

    fieldIds.clear();
    initializeCoreHeader(
        fgf.header, DcgmCoreReqIdFGMGetFieldGroupFields, dcgmCoreGetFieldGroupFields_version, sizeof(fgf));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&fgf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < fgf.response.numFieldIds; i++)
        {
            fieldIds.push_back(fgf.response.fieldIds[i]);
        }

        ret = fgf.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get fields for group " << fieldGrp;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetMigInstanceEntityIdHelper(unsigned int gpuId,
                                                         DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                                         DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
                                                         dcgm_field_entity_group_t desiredLevel,
                                                         dcgm_field_eid_t *entityId) const
{
    if (entityId == nullptr || (desiredLevel != DCGM_FE_GPU_I && desiredLevel != DCGM_FE_GPU_CI))
    {
        return DCGM_ST_BADPARAM;
    }
    dcgmCoreGetComputeInstanceEntityId_t query {};
    query.request.gpuId             = gpuId;
    query.request.instanceId        = instanceId.id;
    query.request.computeInstanceId = computeInstanceId.id;
    query.request.entityGroupId     = desiredLevel;

    initializeCoreHeader(
        query.header, DcgmCoreReqIdGetMigInstanceEntityId, dcgmCoreGetComputeInstanceEntityId_version1, sizeof(query));


    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        *entityId = query.response.uintAnswer;
        return query.response.ret;
    }

    DCGM_LOG_ERROR << "[CoreProxy] Got error: " << errorString(ret) << " while getting the "
                   << (desiredLevel == DCGM_FE_GPU_I ? "gpu" : "compute") << " instance entity id. GpuId: " << gpuId
                   << ", InstanceId: " << instanceId.id << ", ComputeInstanceId: " << computeInstanceId.id;
    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGpuInstanceEntityId(unsigned int gpuId,
                                                   DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                                   dcgm_field_eid_t *entityId) const
{
    using namespace DcgmNs::Mig;
    return GetMigInstanceEntityIdHelper(gpuId, instanceId, Nvml::ComputeInstanceId {}, DCGM_FE_GPU_I, entityId);
}

dcgmReturn_t DcgmCoreProxy::GetComputeInstanceEntityId(unsigned int gpuId,
                                                       DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                                       DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
                                                       dcgm_field_eid_t *entityId) const
{
    return GetMigInstanceEntityIdHelper(gpuId, instanceId, computeInstanceId, DCGM_FE_GPU_CI, entityId);
}

dcgmReturn_t DcgmCoreProxy::GetMigInstancePopulation(unsigned int gpuId,
                                                     DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                                     size_t *capacityGpcs,
                                                     size_t *usedGpcs) const
{
    return GetMigUtilizationHelper(gpuId, instanceId, {}, DCGM_FE_GPU_I, capacityGpcs, usedGpcs);
}
dcgmReturn_t DcgmCoreProxy::GetMigGpuPopulation(unsigned int gpuId, size_t *capacityGpcs, size_t *usedGpcs) const
{
    return GetMigUtilizationHelper(gpuId, {}, {}, DCGM_FE_GPU, capacityGpcs, usedGpcs);
}

dcgmReturn_t DcgmCoreProxy::GetMigComputeInstancePopulation(
    unsigned int gpuId,
    DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
    DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
    size_t *capacityGpcs,
    size_t *usedGpcs)
{
    return GetMigUtilizationHelper(gpuId, instanceId, computeInstanceId, DCGM_FE_GPU_CI, capacityGpcs, usedGpcs);
}

dcgmReturn_t DcgmCoreProxy::GetGpuInstanceHierarchy(dcgmMigHierarchy_v2 &migHierarchy)
{
    dcgmCoreGetGpuInstanceHierarchy_t query {};

    initializeCoreHeader(
        query.header, DcgmCoreReqPopulateMigHierarchy, dcgmCoreGetGpuInstanceHierarchy_version1, sizeof(query));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret != DCGM_ST_OK)
    {
        log_error("[CoreProxy] Got error: {}, while getting the GPU instance hierachy.", errorString(ret));
        return ret;
    }

    std::memcpy(&migHierarchy, &query.response, sizeof(migHierarchy));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreProxy::GetMigUtilizationHelper(
    unsigned int gpuId,
    DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
    DcgmNs::Mig::Nvml::ComputeInstanceId const & /* computeInstanceId */,
    dcgm_field_entity_group_t const entityGroupId,
    size_t *capacityGpcs,
    size_t *usedGpcs) const
{
    dcgmCoreGetMigUtilization_t query {};
    query.request.gpuId         = gpuId;
    query.request.instanceId    = instanceId.id;
    query.request.entityGroupId = entityGroupId;

    initializeCoreHeader(
        query.header, DcgmCoreReqIdGetMigUtilization, dcgmCoreGetMigUtilization_version1, sizeof(query));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        *capacityGpcs = query.response.capacity;
        *usedGpcs     = query.response.usage;
        return query.response.ret;
    }

    DCGM_LOG_ERROR << "[CoreProxy] Got error: " << errorString(ret)
                   << " while getting the profile ID for gpuId: " << gpuId << ", InstanceId: " << instanceId.id;

    return ret;
}
dcgmReturn_t DcgmCoreProxy::GetMigIndicesForEntity(dcgmGroupEntityPair_t const &entity,
                                                   unsigned int *gpuId,
                                                   dcgm_field_eid_t *instanceId) const
{
    dcgmCoreGetMigIndicesForEntity_t query {};
    query.request.entityGroupId = entity.entityGroupId;
    query.request.entityId      = entity.entityId;

    initializeCoreHeader(
        query.header, DcgmCoreReqMigIndicesForEntity, dcgmCoreGetMigIndicesForEntity_version1, sizeof(query));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "[CoreProxy] Got error: " << errorString(ret) << " while getting MIG instance IDs";

        return ret;
    }

    if (gpuId != nullptr)
    {
        if (query.response.gpuId != std::numeric_limits<decltype(query.response.gpuId)>::max())
        {
            *gpuId = query.response.gpuId;
        }
        else
        {
            DCGM_LOG_ERROR << "Requested entityId was not a MIG instance";
            return DCGM_ST_BADPARAM;
        }
    }

    if (instanceId != nullptr)
    {
        if (query.response.instanceId != std::numeric_limits<decltype(query.response.instanceId)>::max())
        {
            *instanceId = query.response.instanceId;
        }
        else
        {
            DCGM_LOG_ERROR << "Requested entityId was not a MIG instance";
            return DCGM_ST_BADPARAM;
        }
    }

    return query.response.ret;
}

dcgmReturn_t DcgmCoreProxy::GetServiceAccount(std::string &serviceAccount) const
{
    dcgmCoreGetServiceAccount_t query {};
    initializeCoreHeader(query.header, DcgmCoreReqGetServiceAccount, dcgmCoreGetServiceAccount_version1, sizeof(query));
    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "[CoreProxy] Got error: " << errorString(ret) << " while getting service account name";
        return ret;
    }

    auto const nameLen = strnlen(query.response.serviceAccount, sizeof(query.response.serviceAccount));
    serviceAccount     = std::string(query.response.serviceAccount, nameLen);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCoreProxy::ReserveResources(unsigned int &token)
{
    dcgmCoreResourceReserve_t query = {};

    // Initialize the header
    initializeCoreHeader(query.header, DcgmCoreReqIdResourceReserve, dcgmCoreResourceReserve_version, sizeof(query));

    // Send the request to the core
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "[CoreProxy] Got error: " << errorString(ret) << " while attempting to reserve resources.";
        return ret;
    }

    // Check the response
    if (query.response.ret == DCGM_ST_OK)
    {
        token = query.response.token;
    }

    return query.response.ret;
}

dcgmReturn_t DcgmCoreProxy::FreeResources(unsigned int token)
{
    dcgmCoreResourceFree_t query = {};

    // Set token
    query.request.token = token;

    // Initialize the header
    initializeCoreHeader(query.header, DcgmCoreReqIdResourceFree, dcgmCoreResourceFree_version, sizeof(query));

    // Send the request to the core
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "[CoreProxy] Got error: " << errorString(ret) << " while attempting to free resources.";
        return ret;
    }

    return query.ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessSpawn(dcgmChildProcessParams_t const &params,
                                              ChildProcessHandle_t &handle,
                                              int &pid)
{
    dcgmCoreChildProcessSpawn_v1 query = {};
    query.request                      = params;

    initializeCoreHeader(
        query.header, DcgmCoreReqIdChildProcessSpawn, dcgmCoreChildProcessSpawn_version1, sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        handle = query.handle;
        pid    = query.pid;
        ret    = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to spawn child process", errorString(ret));
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessStop(ChildProcessHandle_t handle, bool force)
{
    dcgmCoreChildProcessStop_v1 query = {};
    query.handle                      = handle;
    query.force                       = force;

    initializeCoreHeader(query.header, DcgmCoreReqIdChildProcessStop, dcgmCoreChildProcessStop_version1, sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        ret = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to stop child process", errorString(ret));
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessGetStatus(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status)
{
    dcgmCoreChildProcessGetStatus_v1 query = {};
    query.handle                           = handle;
    query.status                           = status;

    initializeCoreHeader(
        query.header, DcgmCoreReqIdChildProcessGetStatus, dcgmCoreChildProcessGetStatus_version1, sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        status = query.status;
        ret    = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to get child process status", errorString(ret));
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessWait(ChildProcessHandle_t handle, int timeoutSec)
{
    dcgmCoreChildProcessWait_v1 query = {};
    query.handle                      = handle;
    query.timeoutSec                  = timeoutSec;

    initializeCoreHeader(query.header, DcgmCoreReqIdChildProcessWait, dcgmCoreChildProcessWait_version1, sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        ret = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to wait for child process", errorString(ret));
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessDestroy(ChildProcessHandle_t handle, int sigTermTimeoutSec)
{
    dcgmCoreChildProcessDestroy_v1 query = {};
    query.handle                         = handle;
    query.sigTermTimeoutSec              = sigTermTimeoutSec;

    initializeCoreHeader(
        query.header, DcgmCoreReqIdChildProcessDestroy, dcgmCoreChildProcessDestroy_version1, sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        ret = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to destroy child process", errorString(ret));
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessGetStdErrHandle(ChildProcessHandle_t handle, int &fd)
{
    dcgmCoreChildProcessGetStdErrHandle_v1 query = {};
    query.handle                                 = handle;
    query.fd                                     = fd;

    initializeCoreHeader(query.header,
                         DcgmCoreReqIdChildProcessGetStdErrHandle,
                         dcgmCoreChildProcessGetStdErrHandle_version1,
                         sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        fd  = query.fd;
        ret = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to get child process stderr handle", errorString(ret));
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessGetStdOutHandle(ChildProcessHandle_t handle, int &fd)
{
    dcgmCoreChildProcessGetStdOutHandle_v1 query = {};
    query.handle                                 = handle;
    query.fd                                     = fd;

    initializeCoreHeader(query.header,
                         DcgmCoreReqIdChildProcessGetStdOutHandle,
                         dcgmCoreChildProcessGetStdOutHandle_version1,
                         sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        fd  = query.fd;
        ret = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to get child process stdout handle", errorString(ret));
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessGetDataChannelHandle(ChildProcessHandle_t handle, int &fd)
{
    dcgmCoreChildProcessGetDataChannelHandle_v1 query = {};
    query.handle                                      = handle;
    query.fd                                          = fd;

    initializeCoreHeader(query.header,
                         DcgmCoreReqIdChildProcessGetDataChannelHandle,
                         dcgmCoreChildProcessGetDataChannelHandle_version1,
                         sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        fd  = query.fd;
        ret = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to get child process data channel handle", errorString(ret));
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::ChildProcessManagerReset()
{
    dcgmCoreChildProcessManagerReset_v1 query = {};

    initializeCoreHeader(
        query.header, DcgmCoreReqIdChildProcessManagerReset, dcgmCoreChildProcessManagerReset_version1, sizeof(query));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&query.header, m_coreCallbacks.poster);
    if (ret == DCGM_ST_OK)
    {
        ret = query.ret;
    }
    else
    {
        log_error("Error '{}' while attempting to reset child process manager", errorString(ret));
    }

    return ret;
}
