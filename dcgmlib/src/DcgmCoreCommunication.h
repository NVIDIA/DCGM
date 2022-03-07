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
#ifndef DCGM_CORE_COMMUNICATION_CLASS_H
#define DCGM_CORE_COMMUNICATION_CLASS_H

#include <mutex>
#include <string>

#include <dcgm_core_communication.h>
#include <dcgm_module_structs.h>

#include "DcgmCacheManager.h"
#include "DcgmGroupManager.h"
#include "dcgm_structs.h"


// These are the callbacks which are passed to the modules
dcgmReturn_t PostRequestToCore(dcgm_module_command_header_t *req, void *poster);

class DcgmCoreCommunication
{
public:
    DcgmCoreCommunication()
        : m_cacheManagerPtr(nullptr)
        , m_groupManagerPtr(nullptr)
        , m_fvBufferCache(nullptr)
        , m_fvbufferCacheSize(0)
    {}

    ~DcgmCoreCommunication()
    {
        delete m_fvBufferCache;
        m_fvBufferCache = nullptr;
    }

    void Init(DcgmCacheManager *cm, DcgmGroupManager *gm);

    /**
     * @return true if this has been initialized, false otherwise
     */
    bool IsInitialized() const;

    /**
     * Process the core module request.
     *
     * @return DCGM_ST_OK if we were able to process the request.
     *         DCGM_ST_*  if we were not able to process the request.
     * @note if the request was processed and had an error, that is returned in the struct itself
     */
    dcgmReturn_t ProcessRequestInCore(dcgm_module_command_header_t *header);

private:
    DcgmCacheManager *m_cacheManagerPtr; // Owned elsewhere, not freed. Used to process API requests
    DcgmGroupManager *m_groupManagerPtr; // Owned elsewhere, not freed. Used to process API requests

    // These two parameters are used to hold a copy of the values retrieved with GetMultipleLatestLiveValues()
    // if the size of the buffer is larger than the buffer in the response struct. These are cleaned out the
    // next time ProcessGetMultipleLiveLatestSamples() is called with the flag signifying it is a fresh call.
    char *m_fvBufferCache;
    size_t m_fvbufferCacheSize;

    /**
     * Methods for handling each core module API call
     */
    dcgmReturn_t ProcessGetGpuIds(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessAreAllGpuIdsSameSku(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetGpuCount(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessAddFieldWatch(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetInt64SummaryData(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetLatestSample(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetEntityNvLinkLinkStatus(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetSamples(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGpuIdToNvmlIndex(dcgm_module_command_header_t *header);

    /*
     * If the response buffer isn't large enough to copy all of the samples into, then we copy by pieces and set
     * a flag in the response indicating there is more to request. The caller has to request the follow-on
     * pieces. The caller does so by setting the position of the buffer to copy from. If the position is 0,
     * then that means this is a new request and the cached buffer should not be used even if there's data in it.
     */
    dcgmReturn_t ProcessGetMultipleLatestLiveSamples(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessRemoveFieldWatch(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetAllGpuInfo(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessAppendSamples(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessSetValue(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessNvmlIndexToGpuId(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessUpdateAllFields(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessVerifyAndUpdateGroupId(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetGroupEntities(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessAreAllTheSameSku(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetGroupGpuIds(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessAnyGlobalFieldsWatched(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessAnyFieldsWatched(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessAnyGpuFieldsWatched(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessAnyGpuFieldsWatchedAnywhere(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessIsGlobalFieldWatched(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessIsGpuFieldWatchedOnAnyGpu(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetFieldBytesUsed(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetGlobalFieldBytesUsed(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetGpuFieldExecTimeUsec(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetGlobalFieldExecTimeUsec(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetGpuFieldFetchCount(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetGlobalFieldFetchCount(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetFieldWatchFreq(dcgm_module_command_header_t *header);
#if 0
    dcgmReturn_t ProcessForEachWatchedGlobalField(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessForEachWatchedGpuField(dcgm_module_command_header_t *header);
#endif
    dcgmReturn_t ProcessLoggingGetSeverity(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessSendModuleCommand(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessSendRawMessageToClient(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessNotifyRequestOfCompletion(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessPopulateFieldGroupGetAll(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetFieldGroupGetFields(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessPopulateGlobalWatchInfo(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessPopulateGpuWatchInfo(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessPopulateWatchInfo(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetMigInstanceEntityId(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetMigUtilization(dcgm_module_command_header_t *header);
    dcgmReturn_t ProcessGetMigIndicesForEntity(dcgm_module_command_header_t *header);
};

#if 0
void DcgmCoreCommunication::ProcessForEachWatchedGlobalField(dcgm_module_command_header_t *header)
{
    if (header == nullptr)
    {
        return;
    }
}

void DcgmCoreCommunication::ProcessForEachWatchedGpuField(dcgm_module_command_header_t *header)
{
    if (header == nullptr)
    {
        return;
    }
}
#endif

#endif
