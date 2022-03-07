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
#pragma once

#include "DcgmProtocol.h"

#include <DcgmMigTypes.hpp>
#include <DcgmWatchTable.h>
#include <dcgm_core_communication.h>
#include <dcgm_module_structs.h>

#include <vector>


class DcgmCoreProxy
{
public:
    /**
     * Constructor
     */
    explicit DcgmCoreProxy(dcgmCoreCallbacks_t coreCallbacks);

    /**
     *
     * @param[in]  activeOnly - if true, only get active GPU ids, otherwise get all visible GPU ids
     * @param[out] gpuIds - the GPU ids retrieved
     */
    dcgmReturn_t GetGpuIds(int activeOnly, std::vector<unsigned int> &gpuIds);

    /**
     * @param[in]  dcgmFieldId - field to check
     * @param[out] isWatched - boolean whether the field is watched
     */
    dcgmReturn_t IsGlobalFieldWatched(unsigned short dcgmFieldId, bool *isWatched);

    /**
     * @param[in]  dcgmFieldId - field to check
     * @param[out] isWatched - boolean whether the field is watched
     */
    dcgmReturn_t IsGpuFieldWatchedOnAnyGpu(unsigned short dcgmFieldId, bool *isWatched);

    /**
     * @param[in] gpuId - gpu to query
     * @param[in] fieldId - field to query
     * @param[out] bytesUsed - bytes used by gpu
     */
    dcgmReturn_t GetGpuFieldBytesUsed(unsigned int gpuId, unsigned short fieldId, long long *bytesUsed);

    /**
     * @param[in] fieldId - field to query
     * @param[out] bytesUsed - bytes used globally
     */
    dcgmReturn_t GetGlobalFieldBytesUsed(unsigned short fieldId, long long *bytesUsed);

    /**
     * @param[in] gpuId - gpu to query
     * @param[in] fieldId - field to query
     * @param[out] bytesUsed - total usec by gpu
     */
    dcgmReturn_t GetGpuFieldExecTimeUsec(unsigned int gpuId, unsigned short fieldId, long long *totalUsec);

    /**
     * @param[in] fieldId - field to query
     * @param[out] bytesUsed - total usec used globally
     */
    dcgmReturn_t GetGlobalFieldExecTimeUsec(unsigned short fieldId, long long *totalUsec);

    /**
     * @param[in] gpuId - gpu to query
     * @param[in] fieldId - field to query
     * @param[out] fetchCount - gpu fetch count
     */
    dcgmReturn_t GetGpuFieldFetchCount(unsigned int gpuId, unsigned short fieldId, long long *fetchCount);

    /**
     * @param[in] fieldId - field to query
     * @param[out] fetchCount - total fetch count
     */
    dcgmReturn_t GetGlobalFieldFetchCount(unsigned short fieldId, long long *fetchCount);

    /**
     * @param[in] fieldId - field to query
     * @param[out] freqUsec - frequency
     */
    dcgmReturn_t GetFieldWatchFreq(unsigned int gpuId, unsigned short fieldId, timelib64_t *freqUsec);


    /**
     * @param[in] gpuIds - the list of GPU ids to check
     */
    bool AreAllGpuIdsSameSku(std::vector<unsigned int> &gpuIds);

    /**
     * @param[in] fieldIds - the list of field ids to check
     */
    bool AnyGlobalFieldsWatched(std::vector<unsigned short> *fieldIds);

    /**
     * @param[in] fieldIds - the list of field ids to check
     */
    bool AnyFieldsWatched(std::vector<unsigned short> *fieldIds);

    /**
     * @param[in] GpuId - Gpu to query
     * @param[in] FieldIds - the list of field ids to check
     */
    bool AnyGpuFieldsWatched(unsigned int gpuId, std::vector<unsigned short> *fieldIds);

    /**
     * @param[in] GpuId - Gpu to query
     * @param[in] FieldIds - the list of field ids to check
     */
    bool AnyGpuFieldsWatchedAnywhere(std::vector<unsigned short> *fieldIds);

    /**
     * @param[in] activeOnly - if true, get a count of only the active GPUs, otherwise count all visible GPUs
     */
    int GetGpuCount(int activeOnly);

    /**
     * @param[out] gpuInfo - a list of all the GPU information
     */
    dcgmReturn_t GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> &gpuInfo);

    /**
     * @param[in] nvmlIndex - the NVML index to translate to a DCGM GPU id
     */
    unsigned int NvmlIndexToGpuId(int nvmlIndex);

    /**
     * @param[in] waitForUpdate - if true, don't return until an update has been processed
     */
    dcgmReturn_t UpdateAllFields(int waitForUpdate);

    /**
     * @param[in] msgheader - header containing module and command info
     */
    dcgmReturn_t SendModuleCommand(void *header);

    /**
     * @param[in] connectionId - recipient of message
     * @param[in] msgType - type of message to be sent
     * @param[in] requestId - requestId that corresponds to message
     * @param[in] msgData - raw message data to send
     * @param[in] msgSize - size of message data
     * @param[out] status - status of message
     */
    dcgmReturn_t SendRawMessageToClient(dcgm_connection_id_t connectionId,
                                        unsigned int msgType,
                                        dcgm_request_id_t requestId,
                                        void *msgData,
                                        unsigned int msgSize,
                                        dcgmReturn_t status);

    /**
     * @param[in] connectionId - connection to notify
     * @param[in] requestId - request that has completed
     */
    dcgmReturn_t NotifyRequestOfCompletion(dcgm_connection_id_t connectionId, dcgm_request_id_t requestId);

    /**
     * @param[out] allGroupInfo - populated on success
     */
    dcgmReturn_t PopulateFieldGroupGetAll(dcgmAllFieldGroup_t *allGroupInfo);

    /**
     * @param[in] fieldGrp - field group to query
     * @param[out] fieldIds - group fields associated with fieldGrp
     */
    dcgmReturn_t GetFieldGroupFields(dcgmFieldGrp_t fieldGrp, std::vector<unsigned short> &fieldIds);

    /**
     *
     * @param[in]  entityGroupId - identifies which group of entities this belongs to
     * @param[in]  entityId - identifies which entity in that group this is
     * @param[out] linkStates - the status of that NvLink
     */
    dcgmReturn_t GetEntityNvLinkLinkStatus(dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId,
                                           dcgmNvLinkLinkState_t *linkStates);

    /**
     * @param[in] entityGroupId - which entity group we are dealing with
     * @param[in] entityId - the id of that entity
     * @param[in] fieldId - the id of the field we want to watch
     * @param[in] monitorFreqUsec - the frequency of capturing a sample for this field in microseconds.
     * @param[in] maxSampleAge - the maximum amount of time a sample is cached before deletion
     * @param[in] maxKeepSamples - the maximum number of samples to keep in the cache at a time
     * @param[in] watcher - the object tracking who is watching this field
     * @param[in] subscribeForUpdates - boolean for whether or not the watcher is subscribing for updates.
     */
    dcgmReturn_t AddFieldWatch(dcgm_field_entity_group_t entityGroupId,
                               dcgm_field_eid_t entityId,
                               unsigned short fieldId,
                               timelib64_t monitorFreqUsec,
                               double maxSampleAge,
                               int maxKeepSamples,
                               DcgmWatcher watcher,
                               bool subscribeForUpdates);

    /**
     * @param[in]  entityGroupId - which entity group we are dealing with
     * @param[in]  entityId - the id of that entity
     * @param[in]  fieldId - the id of the field we want summarized
     * @param[in]  numSummaryTypes - the number of summary types we are requesting
     * @param[in]  summaryTypes - the summary types we are requesting
     * @param[out] summaryValues - populated with the values for each kind of summary
     * @param[in]  startTime - the earliest timestamp for a value to be included in the summary
     * @param[in]  endTime - the latest timestamp for a value to be included in the summary
     * @param[in]  pfUseEntryCB - callback function called once with each timestamped value
     * @param[in]  userData - caller-supplied data which is passed to the callback along with each entry
     */
    dcgmReturn_t GetInt64SummaryData(dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId,
                                     unsigned short fieldId,
                                     int numSummaryTypes,
                                     DcgmcmSummaryType_t *summaryTypes,
                                     long long *summaryValues,
                                     timelib64_t startTime,
                                     timelib64_t endTime,
                                     pfUseEntryForSummary pfUseEntryCB,
                                     void *userData);

    /**
     * @param[in]  entityGroupId - which entity group we are dealing with
     * @param[in]  entityId - the id of that entity
     * @param[in]  fieldId - the id of the field we want summarized
     * @param[out] sample - populated with the latest sample
     * @param[out] fvBuffer - the field value buffer populated with the latest sample
     */
    dcgmReturn_t GetLatestSample(dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId,
                                 unsigned short fieldId,
                                 dcgmcm_sample_p sample,
                                 DcgmFvBuffer *fvBuffer);

    /**
     * @param[in]  entityGroupId - which entity group we are dealing with
     * @param[in]  entityId - the id of that entity
     * @param[in]  fieldId - the id of the field we want summarized
     * @param[out] samples - populated with the cached samples
     * @param[in,out] Msamples - the max space for samples going in and the number of samples in the copied buffer
     *                           going out
     * @param[in]     startTime - the earliest timestamp to be included
     * @param[in]     endTime - the latest timestamp to be included
     * @param[in]     order - the order in which the samples should be inserted into the buffer
     */
    dcgmReturn_t GetSamples(dcgm_field_entity_group_t entityGroupId,
                            dcgm_field_eid_t entityId,
                            unsigned short fieldId,
                            dcgmcm_sample_p samples,
                            int *Msamples,
                            timelib64_t startTime,
                            timelib64_t endTime,
                            dcgmOrder_t order);

    /**
     * @param[in]  entities - a list of each entity id and group id
     * @param[in]  fieldIds - a list of the field ids to be retrieved
     * @param[out] fvBuffer - the buffer to be populated with with the samples
     */
    dcgmReturn_t GetMultipleLatestLiveSamples(std::vector<dcgmGroupEntityPair_t> const &entities,
                                              std::vector<unsigned short> const &fieldIds,
                                              DcgmFvBuffer *fvBuffer);

    /**
     * @param[in] entityGroupId - which entity group we are dealing with
     * @param[in] entityId - the id of that entity
     * @param[in] fieldId - the id of the field we want summarized
     * @param[in] clearCache - if true, the values for this field are cleared from the cache if there are
     *                         no other watchers.
     * @param[in] watcher - the object describing who was tracking this field
     */
    dcgmReturn_t RemoveFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                  dcgm_field_eid_t entityId,
                                  unsigned short fieldId,
                                  int clearCache,
                                  DcgmWatcher watcher);

    /**
     * @param[in] fvBuffer - buffer of field values to add to the cache
     */
    dcgmReturn_t AppendSamples(DcgmFvBuffer *fvBuffer);

    /**
     * @param[in] gpuId - the id of the GPU whose calue is being set
     * @param[in] fieldId - the field id that is getting populated
     * @param[in] value - the value being set
     */
    dcgmReturn_t SetValue(int gpuId, unsigned short fieldId, dcgmcm_sample_p value);

    /**
     * @param[in]  groupId - the id of the group whose entities are being retrieved
     * @param[out] entities - the list of entities with information about entity group and id
     */
    dcgmReturn_t GetGroupEntities(unsigned int groupId, std::vector<dcgmGroupEntityPair_t> &entities) const;

    /**
     * @param[in]  connectionId - the id of the connection
     * @param[in]  groupId - the id of the group we're checking
     * @param[out] areAllSameSku - set to true if all GPUs in this group and the same SKU, false otherwise.
     */
    dcgmReturn_t AreAllTheSameSku(dcgm_connection_id_t connectionId, unsigned int groupId, int *areAllSameSku) const;

    /**
     * @param[in]  connectionId - the id of the connection
     * @param[in]  groupId - the id of the group we're checking
     * @param[out]  gpuIds - populated with the complete list of GPU ids for this group
     */
    dcgmReturn_t GetGroupGpuIds(dcgm_connection_id_t connectionId,
                                unsigned int groupId,
                                std::vector<unsigned int> &gpuIds);


    /**
     * @param[in] gpuId - the DCGM GPU ID we'd like to convert to an NVML GPU index
     * @return the NVML index of this GPU ID
     */
    int GpuIdToNvmlIndex(unsigned int gpuId);

    /**
     * @param[in,out] groupId - the group ID to verify and update
     */
    dcgmReturn_t VerifyAndUpdateGroupId(unsigned int *groupId) const;

    DcgmLoggingSeverity_t GetLoggerSeverity(dcgm_connection_id_t connectionId, loggerCategory_t logger = BASE_LOGGER);

    /**
     * @param[out] watchInfo - array of relevant global watch information from cache manager
     * @param[in] fields - optional array of fields to query
     */
    dcgmReturn_t PopulateGlobalWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                         std::vector<unsigned short> *fieldIds = nullptr);

    /**
     * @param[out] watchInfo - array of relevant gpu watch information from cache manager
     * @param[in] gpuId - gpu to query
     * @param[in] fields - optionl array of fields to query
     */
    dcgmReturn_t PopulateGpuWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                      unsigned int gpuId,
                                      std::vector<unsigned short> *fieldIds = nullptr);

    /**
     * @param[out] watchInfo - array of relevant watch information from cache manager
     * @param[in] fields - optional array of fields to query
     */
    dcgmReturn_t PopulateWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                   std::vector<unsigned short> *fieldIds = nullptr);

    /**
     * Returns DCGM entity id for the compute instance with given NVML indices.
     * @param gpuId[in]                 GPU ID
     * @param instanceId[in]            NVML index for GPU Instance
     * @param computeInstanceId[in]     NVML index for Compute instance
     * @param entityId[out]             Result EntityId
     * @return
     *      \ref DCGM_ST_OK         EntityId was found
     *      \ref DCGM_ST_NO_DATA    No entity id was found or the NVML indices are invalid
     *      \ref DCGM_ST_BADPARAM   Wrong parameters were passed to the function
     *      \ref DCGM_ST_*          Other generic errors
     */
    dcgmReturn_t GetComputeInstanceEntityId(unsigned int gpuId,
                                            DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                            DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
                                            dcgm_field_eid_t *entityId) const;

    /**
     * Returns DCGM entity id for the GPU instance with given NVML index.
     * @param gpuId[in]                 GPU ID
     * @param instanceId[in]            NVML index for GPU Instance
     * @param entityId[out]             Result EntityId
     * @return
     *      \ref DCGM_ST_OK         EntityId was found
     *      \ref DCGM_ST_NO_DATA    No entity id was found or the NVML indices are invalid
     *      \ref DCGM_ST_BADPARAM   Wrong parameters were passed to the function
     *      \ref DCGM_ST_*          Other generic errors
     */
    dcgmReturn_t GetGpuInstanceEntityId(unsigned int gpuId,
                                        DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                        dcgm_field_eid_t *entityId) const;

    /**
     * @brief For the given entity pair returns MIG ids if the requested entity is either GPU_I or GPU_CI
     * @param[in]   entity      Entity pair for the desired MIG instance.
     *                          The entityGroupId should be either \c DCGM_FE_GPU_I or \c DCGM_FE_GPU_CI
     * @param[out]  gpuId       GPU ID of the requested MIG instance
     * @param[out]  instanceId  GPU Instance ID if the requested entityGroupId was \c DCGM_FE_GPU_CI
     * @return
     *          \c\b DCGM_ST_OK       Success<br>
     *          \c\b DCGM_ST_BADPARAM In case entity is not GPU_I or GPU_CI<br>
     */
    dcgmReturn_t GetMigIndicesForEntity(dcgmGroupEntityPair_t const &entity,
                                        unsigned int *gpuId,
                                        dcgm_field_eid_t *instanceId = nullptr) const;

    /**
     * Returns information about how many GPCs are allocated and occupied for the GPU instance.
     * @param gpuId[in]             GPU ID
     * @param instanceId[in]        NVML index for GPU Instance
     * @param capacityGpcs[out]     Number of available GPCs
     * @param usedGpcs[out]         Number of occupied GPCs
     * @return
     *      \ref DCGM_ST_OK         Values were set successfully
     *      \ref DCGM_ST_NO_DATA    No GPU Instance or no values for it were found
     *      \ref DCGM_ST_BADPARAM   Wrong parameters were passed to the function
     *      \ref DCGM_ST_*          Other generic errors
     */
    dcgmReturn_t GetMigInstancePopulation(unsigned int gpuId,
                                          const DcgmNs::Mig::Nvml::GpuInstanceId &instanceId,
                                          size_t *capacityGpcs,
                                          size_t *usedGpcs) const;

    /**
     * Returns information about how many GPCs are available and occupied on the GPU.
     * @param gpuId[in]             GPU ID
     * @param capacityGpcs[out]     Number of available GPCs
     * @param usedGpcs[out]         Number of occupied GPCs
     * @return
     *      \ref DCGM_ST_OK         Values were set successfully
     *      \ref DCGM_ST_NO_DATA    No GPU or values for it were found
     *      \ref DCGM_ST_BADPARAM   Wrong parameters were passed to the function
     *      \ref DCGM_ST_*          Other generic errors
     */
    dcgmReturn_t GetMigGpuPopulation(unsigned int gpuId, size_t *capacityGpcs, size_t *usedGpcs) const;

    /**
     * Returns information about how many GPCs occupied by the Compute Instance.
     * @param gpuId[in]             GPU ID
     * @param instanceId[in]        NVML index for GPU instance
     * @param computeInstanceId[in] NVML index for Compute instance
     * @param capacityGpcs[out]     Number of available GPCs
     * @param usedGpcs[out]         Number of occupied GPCs
     * @return
     *      \ref DCGM_ST_OK         Values were set successfully<br>
     *      \ref DCGM_ST_NO_DATA    No Compute instance or values for it were found<br>
     *      \ref DCGM_ST_BADPARAM   Wrong parameters were passed to the function<br>
     *      \ref DCGM_ST_*          Other generic errors<br>
     *
     * @note    capacityGpcs and usedGpcs are the same for Ampere GPUs
     *
     */
    dcgmReturn_t GetMigComputeInstancePopulation(unsigned int gpuId,
                                                 DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                                 DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
                                                 size_t *capacityGpcs,
                                                 size_t *usedGpcs);

private:
    dcgmCoreCallbacks_t m_coreCallbacks;

    dcgmReturn_t GetMigInstanceEntityIdHelper(unsigned int gpuId,
                                              DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                              DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
                                              dcgm_field_entity_group_t desiredLevel,
                                              dcgm_field_eid_t *entityId) const;

    dcgmReturn_t GetMigUtilizationHelper(unsigned int gpuId,
                                         DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                         DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
                                         dcgm_field_entity_group_t entityGroupId,
                                         size_t *capacityGpcs,
                                         size_t *usedGpcs) const;
};

void InitializeCoreHeader(dcgm_module_command_header_t &header,
                          dcgmCoreReqCmd_t cmd,
                          unsigned int version,
                          size_t reqSize);
