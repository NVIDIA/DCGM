/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

dcgmReturn_t DcgmCoreProxy::IsGlobalFieldWatched(unsigned short dcgmFieldId, bool *isWatched)
{
    dcgmCoreQueryField_t qf = {};

    if (nullptr == isWatched)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = dcgmFieldId;

    initializeCoreHeader(qf.header, DcgmCoreReqIdCMIsGlobalFieldWatched, dcgmCoreQueryField_version, sizeof(qf));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *isWatched = (qf.response.uintAnswer != 0);
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret)
                       << "' while determining if all GPUs in our list are the same SKU";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::IsGpuFieldWatchedOnAnyGpu(unsigned short dcgmFieldId, bool *isWatched)
{
    dcgmCoreQueryField_t qf = {};

    if (nullptr == isWatched)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = dcgmFieldId;

    initializeCoreHeader(qf.header, DcgmCoreReqIdCMIsGlobalFieldWatched, dcgmCoreQueryField_version, sizeof(qf));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *isWatched = (qf.response.uintAnswer != 0);
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while determining if field is watched on any GPU";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGpuFieldBytesUsed(unsigned int gpuId, unsigned short fieldId, long long *value)
{
    dcgmCoreGetField_t qf = {};

    if (nullptr == value)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = fieldId;
    qf.request.gpuId   = gpuId;

    initializeCoreHeader(qf.header, DcgmCoreReqIDCMGetGpuFieldBytesUsed, dcgmCoreGetField_version, sizeof(qf));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *value = (qf.response.longAnswer != 0);
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while retrieving GPU field bytes used";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGlobalFieldBytesUsed(unsigned short fieldId, long long *value)
{
    dcgmCoreGetField_t qf = {};

    if (nullptr == value)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = fieldId;

    initializeCoreHeader(qf.header, DcgmCoreReqIDCMGetGlobalFieldBytesUsed, dcgmCoreGetField_version, sizeof(qf));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *value = (qf.response.longAnswer != 0);
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while retrieving global field bytes used";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGpuFieldExecTimeUsec(unsigned int gpuId, unsigned short fieldId, long long *value)
{
    dcgmCoreGetField_t qf = {};

    if (nullptr == value)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = fieldId;
    qf.request.gpuId   = gpuId;

    initializeCoreHeader(qf.header, DcgmCoreReqIDCMGetGpuFieldExecTimeUsec, dcgmCoreGetField_version, sizeof(qf));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *value = (qf.response.longAnswer != 0);
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while retrieving field exec time used";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGlobalFieldExecTimeUsec(unsigned short fieldId, long long *value)
{
    dcgmCoreGetField_t qf = {};

    if (nullptr == value)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = fieldId;

    initializeCoreHeader(qf.header, DcgmCoreReqIDCMGetGlobalFieldExecTimeUsec, dcgmCoreGetField_version, sizeof(qf));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *value = (qf.response.longAnswer != 0);
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while retrieving global field exec time used";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGpuFieldFetchCount(unsigned int gpuId, unsigned short fieldId, long long *value)
{
    dcgmCoreGetField_t qf = {};

    if (nullptr == value)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = fieldId;
    qf.request.gpuId   = gpuId;

    initializeCoreHeader(qf.header, DcgmCoreReqIDCMGetGpuFieldFetchCount, dcgmCoreGetField_version, sizeof(qf));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *value = (qf.response.longAnswer != 0);
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while retrieving GPU field fetch count";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetGlobalFieldFetchCount(unsigned short fieldId, long long *value)
{
    dcgmCoreGetField_t qf = {};

    if (nullptr == value)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = fieldId;

    initializeCoreHeader(qf.header, DcgmCoreReqIDCMGetGlobalFieldFetchCount, dcgmCoreGetField_version, sizeof(qf));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *value = (qf.response.longAnswer != 0);
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while retrieving global field fetch count";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::GetFieldWatchFreq(unsigned int gpuId, unsigned short fieldId, timelib64_t *value)
{
    dcgmCoreGetField_t qf = {};

    if (nullptr == value)
    {
        return DCGM_ST_BADPARAM;
    }

    qf.request.fieldId = fieldId;
    qf.request.gpuId   = gpuId;

    initializeCoreHeader(qf.header, DcgmCoreReqIDCMGetFieldWatchFreq, dcgmCoreGetField_version, sizeof(qf));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        *value = qf.response.longAnswer;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while retrieving field watch freq";
    }

    return ret;
}

bool DcgmCoreProxy::AreAllGpuIdsSameSku(std::vector<unsigned int> &gpuIds)
{
    dcgmCoreQueryGpuList_t qg = {};

    for (size_t i = 0; i < gpuIds.size(); i++)
    {
        qg.request.gpuIds[i] = gpuIds[i];
    }
    qg.request.gpuCount = gpuIds.size();

    initializeCoreHeader(qg.header, DcgmCoreReqIdCMAreAllGpuIdsSameSku, dcgmCoreQueryGpuList_version, sizeof(qg));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qg.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        bool sameSku = (qg.response.uintAnswer != 0);
        return sameSku;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret)
                       << "' while determining if all GPUs in our list are the same SKU";
        return false;
    }
}

bool DcgmCoreProxy::AnyGlobalFieldsWatched(std::vector<unsigned short> *fieldIds)
{
    dcgmCoreQueryFieldList_t qf = {};

    if (nullptr != fieldIds)
    {
        qf.request.fieldIds    = fieldIds->data();
        qf.request.numFieldIds = fieldIds->size();
    }

    initializeCoreHeader(qf.header, DcgmCoreReqIdCMAnyGlobalFieldsWatched, dcgmCoreQueryFieldList_version, sizeof(qf));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        bool anyWatched = (qf.response.uintAnswer != 0);
        return anyWatched;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while determining if any fields watched";
        return false;
    }
}

bool DcgmCoreProxy::AnyFieldsWatched(std::vector<unsigned short> *fieldIds)
{
    dcgmCoreQueryFieldList_t qf = {};

    if (nullptr != fieldIds)
    {
        qf.request.fieldIds    = fieldIds->data();
        qf.request.numFieldIds = fieldIds->size();
    }

    initializeCoreHeader(qf.header, DcgmCoreReqIdCMAnyFieldsWatched, dcgmCoreQueryFieldList_version, sizeof(qf));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        bool anyWatched = (qf.response.uintAnswer != 0);
        return anyWatched;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while determining if any fields watched";
        return false;
    }
}

bool DcgmCoreProxy::AnyGpuFieldsWatched(unsigned int gpuId, std::vector<unsigned short> *fieldIds)
{
    dcgmCoreQueryFieldList_t qf = {};

    if (nullptr != fieldIds)
    {
        qf.request.fieldIds    = fieldIds->data();
        qf.request.numFieldIds = fieldIds->size();
    }

    qf.request.gpuIds[0] = gpuId;
    qf.request.gpuCount  = 1;

    initializeCoreHeader(qf.header, DcgmCoreReqIdCMAnyGpuFieldsWatched, dcgmCoreQueryFieldList_version, sizeof(qf));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        bool anyWatched = (qf.response.uintAnswer != 0);
        return anyWatched;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while determining if any fields watched";
        return false;
    }
}

bool DcgmCoreProxy::AnyGpuFieldsWatchedAnywhere(std::vector<unsigned short> *fieldIds)
{
    dcgmCoreQueryFieldList_t qf = {};

    if (nullptr != fieldIds)
    {
        qf.request.fieldIds    = fieldIds->data();
        qf.request.numFieldIds = fieldIds->size();
    }

    initializeCoreHeader(
        qf.header, DcgmCoreReqIdCMAnyGpuFieldsWatchedAnywhere, dcgmCoreQueryFieldList_version, sizeof(qf));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&qf.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        bool anyWatched = (qf.response.uintAnswer != 0);
        return anyWatched;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while determining if any fields watched";
        return false;
    }
}

int DcgmCoreProxy::GetGpuCount(int activeOnly)
{
    dcgmCoreGetGpuCount_t ggc = {};
    ggc.request.flag          = activeOnly;

    initializeCoreHeader(ggc.header, DcgmCoreReqIdCMGetGpuCount, dcgmCoreGetGpuCount_version, sizeof(ggc));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&ggc.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        int count = ggc.response.uintAnswer;
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
                                          bool subscribeForUpdates)
{
    dcgmCoreAddFieldWatch_t afw     = {};
    afw.request.entityGroupId       = entityGroupId;
    afw.request.entityId            = entityId;
    afw.request.fieldId             = fieldId;
    afw.request.monitorFreqUsec     = monitorFreqUsec;
    afw.request.maxSampleAge        = maxSampleAge;
    afw.request.maxKeepSamples      = maxKeepSamples;
    afw.request.watcherType         = watcher.watcherType;
    afw.request.connectionId        = watcher.connectionId;
    afw.request.subscribeForUpdates = subscribeForUpdates;

    initializeCoreHeader(afw.header, DcgmCoreReqIdCMAddFieldWatch, dcgmCoreAddFieldWatch_version, sizeof(afw));

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
                                                         DcgmFvBuffer *fvBuffer)
{
    dcgmCoreGetMultipleLatestLiveSamples_t gml = {};

    if (entities.size() > DCGM_GROUP_MAX_ENTITIES)
    {
        DCGM_LOG_ERROR << fmt::format("Too many entities in the group. Provided {}, supported up to {}",
                                      entities.size(),
                                      DCGM_GROUP_MAX_ENTITIES);
        return DCGM_ST_MAX_LIMIT;
    }

    gml.request.entityPairCount = entities.size();
    for (size_t i = 0; i < entities.size(); i++)
    {
        gml.request.entityPairs[i] = entities[i];
    }

    gml.request.fieldIds    = fieldIds.data();
    gml.request.numFieldIds = fieldIds.size();

    initializeCoreHeader(gml.header,
                         DcgmCoreReqIdCMGetMultipleLatestLiveSamples,
                         dcgmCoreGetMultipleLatestLiveSamples_version,
                         sizeof(gml));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&gml.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        Blob blob;
        blob.AppendToBlob(gml.response.buffer, gml.response.bufferSize);

        // If the data didn't fit in one call, repeatedly call the API and copy off each portion until we're done.
        while (gml.response.dataDidNotFit)
        {
            gml.request.bufferPosition = blob.GetUsed();
            ret                        = m_coreCallbacks.postfunc(&gml.header, m_coreCallbacks.poster);

            if (ret != DCGM_ST_OK)
            {
                break;
            }
            else
            {
                blob.AppendToBlob(gml.response.buffer, gml.response.bufferSize);
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
                                             DcgmWatcher watcher)
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

dcgmReturn_t DcgmCoreProxy::AppendSamples(DcgmFvBuffer *fvBuffer)
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

dcgmReturn_t DcgmCoreProxy::SetValue(int gpuId, unsigned short fieldId, dcgmcm_sample_p value)
{
    dcgmCoreSetValue_t sv = {};
    sv.request.gpuId      = gpuId;
    sv.request.fieldId    = fieldId;
    sv.request.value      = value;

    initializeCoreHeader(sv.header, DcgmCoreReqIdCMSetValue, dcgmCoreSetValue_version, sizeof(sv));

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
    dcgmCoreGetGroupEntities_t gge = {};
    gge.request.connectionId       = 0;
    gge.request.groupId            = groupId;

    initializeCoreHeader(gge.header, DcgmCoreReqIdGMGetGroupEntities, dcgmCoreGetGroupEntities_version, sizeof(gge));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&gge.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < gge.response.entityPairsCount; i++)
        {
            entities.push_back(gge.response.entityPairs[i]);
        }

        ret = gge.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get entities for group " << groupId;
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::AreAllTheSameSku(dcgm_connection_id_t connectionId,
                                             unsigned int groupId,
                                             int *areAllSameSku) const
{
    if (areAllSameSku == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmCoreBasicGroup_t bg = {};
    bg.request.connectionId = connectionId;
    bg.request.groupId      = groupId;

    initializeCoreHeader(bg.header, DcgmCoreReqIdGMAreAllTheSameSku, dcgmCoreBasicGroup_version, sizeof(bg));

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
                                           std::vector<unsigned int> &gpuIds)
{
    dcgmCoreGetGroupGpuIds_t ggg = {};
    ggg.request.connectionId     = connectionId;
    ggg.request.groupId          = groupId;

    initializeCoreHeader(ggg.header, DcgmCoreReqIdGMGetGroupGpuIds, dcgmCoreGetGroupGpuIds_version, sizeof(ggg));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&ggg.header, m_coreCallbacks.poster);

    gpuIds.clear();
    if (ret == DCGM_ST_OK)
    {
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

int DcgmCoreProxy::GpuIdToNvmlIndex(unsigned int gpuId)
{
    dcgmCoreBasicQuery_t bq = {};
    bq.request.entityId     = gpuId;

    initializeCoreHeader(bq.header, DcgmCoreReqIdCMGpuIdToNvmlIndex, dcgmCoreBasicQuery_version, sizeof(bq));

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

DcgmLoggingSeverity_t DcgmCoreProxy::GetLoggerSeverity(dcgm_connection_id_t connectionId, loggerCategory_t logger)
{
    dcgmCoreGetSeverity_t getSeverity = {};

    getSeverity.request = logger;

    initializeCoreHeader(
        getSeverity.header, DcgmCoreReqIdLoggingGetSeverity, dcgmCoreGetSeverity_version, sizeof(getSeverity));

    dcgmReturn_t ret = m_coreCallbacks.postfunc(&getSeverity.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        return getSeverity.response;
    }

    DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to get logger severity";

    return DcgmLoggingSeverityUnspecified;
}

dcgmReturn_t DcgmCoreProxy::SendModuleCommand(void *msg)
{
    dcgmCoreSendModuleCommand_t smc = {};

    initializeCoreHeader(smc.header, DcgmCoreReqIdSendModuleCommand, dcgmCoreSendModuleCommand_version, sizeof(smc));
    smc.command = msg;

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
                                                   dcgmReturn_t status)
{
    dcgmCoreSendRawMessage_t msg = {};

    initializeCoreHeader(msg.header, DcgmCoreReqIdSendRawMessage, dcgmCoreSendRawMessage_version, sizeof(msg));
    msg.request.connectionId = connectionId;
    msg.request.msgType      = msgType;
    msg.request.requestId    = requestId;
    msg.request.msgData      = msgData;
    msg.request.msgSize      = msgSize;
    msg.request.status       = status;

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

dcgmReturn_t DcgmCoreProxy::PopulateWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                              std::vector<unsigned short> *fieldIds)
{
    dcgmCorePopulateGlobalWatchInfo_t pwi = {};

    if (nullptr != fieldIds)
    {
        pwi.request.fieldIds    = fieldIds->data();
        pwi.request.numFieldIds = fieldIds->size();
    }

    watchInfo.clear();
    initializeCoreHeader(pwi.header, DcgmCoreReqIdCMPopulateWatchInfo, dcgmCorePopulateWatchInfo_version, sizeof(pwi));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&pwi.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < pwi.response.numFields; i++)
        {
            watchInfo.push_back(pwi.response.fields[i]);
        }

        ret = pwi.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to populate global watch info";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::PopulateGlobalWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                                    std::vector<unsigned short> *fieldIds)
{
    dcgmCorePopulateGlobalWatchInfo_t pwi = {};

    if (nullptr != fieldIds)
    {
        pwi.request.fieldIds    = fieldIds->data();
        pwi.request.numFieldIds = fieldIds->size();
    }

    watchInfo.clear();
    initializeCoreHeader(
        pwi.header, DcgmCoreReqIdCMPopulateGlobalWatchInfo, dcgmCorePopulateGlobalWatchInfo_version, sizeof(pwi));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&pwi.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < pwi.response.numFields; i++)
        {
            watchInfo.push_back(pwi.response.fields[i]);
        }

        ret = pwi.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to populate global watch info";
    }

    return ret;
}

dcgmReturn_t DcgmCoreProxy::PopulateGpuWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                                 unsigned int gpuId,
                                                 std::vector<unsigned short> *fieldIds)
{
    dcgmCorePopulateGpuWatchInfo_t pwi = {};

    pwi.request.gpuIds[0] = gpuId;
    pwi.request.gpuCount  = 1;

    if (nullptr != fieldIds)
    {
        pwi.request.fieldIds    = fieldIds->data();
        pwi.request.numFieldIds = fieldIds->size();
    }

    watchInfo.clear();

    initializeCoreHeader(
        pwi.header, DcgmCoreReqIdCMPopulateGpuWatchInfo, dcgmCorePopulateGpuWatchInfo_version, sizeof(pwi));

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = m_coreCallbacks.postfunc(&pwi.header, m_coreCallbacks.poster);

    if (ret == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < pwi.response.numFields; i++)
        {
            watchInfo.push_back(pwi.response.fields[i]);
        }

        ret = pwi.response.ret;
    }
    else
    {
        DCGM_LOG_ERROR << "Error '" << errorString(ret) << "' while attempting to populate global watch info";
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

dcgmReturn_t DcgmCoreProxy::GetMigUtilizationHelper(unsigned int gpuId,
                                                    DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                                    DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
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
