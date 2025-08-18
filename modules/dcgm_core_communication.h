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
#pragma once

#include <DcgmCacheManager.h>
#include <DcgmLogging.h>
#include <timelib.h>
#include <timeseries.h>

#include "DcgmProtocol.h"
#include "dcgm_module_structs.h"

typedef unsigned int dcgmCoreReqId_t; // A unique identifier for the request we sent to libdcgm

// A list of each possible kind of request to libdcgm.so
typedef enum
{
    DcgmCoreReqIdCMGetGpuIds                      = 0,  // DcgmCacheManager::GetGpuIds()
    DcgmCoreReqIdCMAreAllGpuIdsSameSku            = 1,  // DcgmCacheManager::AreAllGpuIdsSameSku()
    DcgmCoreReqIdCMGetGpuCount                    = 2,  // DcgmCacheManager::GetGpuCount()
    DcgmCoreReqIdCMAddFieldWatch                  = 3,  // DcgmCacheManager::AddFieldWatch
    DcgmCoreReqIdCMGetInt64SummaryData            = 4,  // DcgmCacheManager::GetInt64SummaryData()
    DcgmCoreReqIdCMGetLatestSample                = 5,  // DcgmCacheManager::GetLatestSample()
    DcgmCoreReqIdCMGetEntityNvLinkLinkStatus      = 6,  // DcgmCacheManager::GetEntityNvLinkLinkStatus()
    DcgmCoreReqIdCMGetSamples                     = 7,  // DcgmCacheManager::GetSamples()
    DcgmCoreReqIdCMGpuIdToNvmlIndex               = 8,  // DcgmCacheManager::GpuIdToNvmlIndex()
    DcgmCoreReqIdCMGetMultipleLatestLiveSamples   = 9,  // DcgmCacheManager::GetMultipleLatestLiveSamples()
    DcgmCoreReqIdCMRemoveFieldWatch               = 10, // DcgmCacheManager::RemoveFieldWatch()
    DcgmCoreReqIdCMGetAllGpuInfo                  = 11, // DcgmCacheManager::GetAllGpuInfo()
    DcgmCoreReqIdCMAppendSamples                  = 12, // DcgmCacheManager::AppendSamples()
    DcgmCoreReqIdCMSetValue                       = 13, // DcgmCacheManager::SetValue()
    DcgmCoreReqIdCMNvmlIndexToGpuId               = 14, // DcgmCacheManager::NvmlIndexToGpuId()
    DcgmCoreReqIdCMUpdateAllFields                = 15, // DcgmCacheManager::UpdateAllFields()
    DcgmCoreReqIdGMVerifyAndUpdateGroupId         = 16, // DcgmGroupManager::VerifyAndUpdateGroupId()
    DcgmCoreReqIdGMGetGroupEntities               = 17, // DcgmGroupManager::GetGroupEntities()
    DcgmCoreReqIdGMAreAllTheSameSku               = 18, // DcgmGroupManager::AreAllTheSameSku()
    DcgmCoreReqIdGMGetGroupGpuIds                 = 19, // DcgmGroupManager::GetGroupGpuIds()
    DcgmCoreReqIdLoggingGetSeverity               = 36, // LoggingGetSeverity()
    DcgmCoreReqIdSendModuleCommand                = 37, // DcgmHostEngineHandler::ProcessModuleCommand()
    DcgmCoreReqIdSendRawMessage                   = 38, // DcgmHostEngineHandler::SendRawMessageToClient()
    DcgmCoreReqIdNotifyRequestOfCompletion        = 39, // DcgmHostEngineHandler::NotifyRequestOfCompletion()
    DcgmCoreReqIdFGMPopulateFieldGroups           = 40, // FieldGroupManager::PopulateFieldGroups()
    DcgmCoreReqIdFGMGetFieldGroupFields           = 41, // FieldGroupManager::GetFieldGroupFields()
    DcgmCoreReqIdGetMigInstanceEntityId           = 45, // DcgmCacheManager::GetComputeInstanceEntityId()
    DcgmCoreReqIdGetMigUtilization                = 46, // DcgmCacheManager::GetMigUtilization()
    DcgmCoreReqMigIndicesForEntity                = 47, // DcgmCacheManager::GetMigIndicesForEntity()
    DcgmCoreReqGetServiceAccount                  = 48, // DcgmHostEngineHandler::GetServiceAccount()
    DcgmCoreReqPopulateMigHierarchy               = 49, // DcgmCacheManager::PopulateMigHierarchy()
    DcgmCoreReqIdResourceReserve                  = 50, // DcgmHostEngineHandler::ReserveResources()
    DcgmCoreReqIdResourceFree                     = 51, // DcgmHostEngineHandler::FreeResources()
    DcgmCoreReqIdChildProcessSpawn                = 52, // ChildProcessManager::Spawn()
    DcgmCoreReqIdChildProcessStop                 = 53, // ChildProcessManager::Stop()
    DcgmCoreReqIdChildProcessGetStatus            = 54, // ChildProcessManager::GetStatus()
    DcgmCoreReqIdChildProcessWait                 = 55, // ChildProcessManager::Wait()
    DcgmCoreReqIdChildProcessDestroy              = 56, // ChildProcessManager::Destroy()
    DcgmCoreReqIdChildProcessGetStdErrHandle      = 57, // ChildProcessManager::GetStdErrHandle()
    DcgmCoreReqIdChildProcessGetStdOutHandle      = 58, // ChildProcessManager::GetStdOutHandle()
    DcgmCoreReqIdChildProcessGetDataChannelHandle = 59, // ChildProcessManager::GetDataChannelHandle()
    DcgmCoreReqIdGetDriverVersion                 = 60, // DcgmCacheManager::GetDriverVersion()
    DcgmCoreReqIdChildProcessManagerReset         = 61, // ChildProcessManager::Reset()
    DcgmCoreReqIdCount                                  // Always keep this one last
} dcgmCoreReqCmd_t;

/* Callback functions for allocating and freeing DcgmModules. These are found
   in the modules' shared library with dlsym */
typedef dcgmReturn_t (*dcgmCoreReqPost_f)(dcgm_module_command_header_t *req, void *);
typedef dcgmReturn_t (*dcgmCoreGetResponse_f)(dcgmCoreReqId_t reqId, unsigned int timeout);
/**
 * Contains the callbacks that should be used for communicating between the modules and libdcgm
 */
typedef struct
{
    unsigned int version;            // !< the version of the callback structure
    dcgmCoreReqPost_f postfunc;      // !< function pointer to post a request to the core library
    void *poster;                    // !< pointer to the object that will forward the request to the core modules
    dcgmLoggerCallback_f loggerfunc; // !< function pointer to send logging messages to the hostengine
} dcgmCoreCallbacks_v1;

#define dcgmCoreCallbacks_version1 MAKE_DCGM_VERSION(dcgmCoreCallbacks_v1, 1)

#define dcgmCoreCallbacks_version dcgmCoreCallbacks_version1

typedef dcgmCoreCallbacks_v1 dcgmCoreCallbacks_t;

/**
 * Basic information covering simple requests that just specify an ID and maybe an entity group as well
 */
typedef struct
{
    dcgm_field_entity_group_t entityGroupId; // !< entity group id
    dcgm_field_eid_t entityId;               // !< the entity id
} dcgmCoreBasicReqParams_t;

/**
 * Used for sending a module command from one module to another through the CoreProxy
 */
typedef struct
{
    void *msg;
} dcgmCoreModuleCommand_t;

typedef struct
{
    dcgm_connection_id_t connectionId;
    unsigned int msgType;
    dcgm_request_id_t requestId;
    void *msgData;
    unsigned int msgSize;
    dcgmReturn_t status;
} dcgmCoreRawMessage_t;

typedef struct
{
    dcgm_connection_id_t connectionId;
    dcgm_request_id_t requestId;

} dcgmCoreNotifyRequestMessage_t;

typedef struct
{
    dcgmAllFieldGroup_t groups;
    dcgmReturn_t status;
} dcgmCoreFieldGroups_t;

typedef struct
{
    dcgmReturn_t ret;                            // !< The status of the function call
    unsigned short fieldIds[DCGM_FI_MAX_FIELDS]; // !< List of the field ids we want samples for
    size_t numFieldIds;                          // !< The number of field ids specified
} dcgmCoreGetFieldGroupFieldsResponse_t;

/*
 * Basic parameters for group manager requests
 */
typedef struct
{
    dcgm_connection_id_t connectionId; // !< the connection id this request is associated with
    unsigned int groupId;              // !< the group id for this request
} dcgmCoreBasicGroupParams_t;

/*
 * Struct covering the parameters for simple requests about all the GPUs
 */
typedef struct
{
    int flag; // !< a true or false flag specifying a single parameter for basic GPU queries to libdcgm
} dcgmCoreGpuQueryParams_t;

/*
 * Struct covering the parameter for querying specified field
 */
typedef struct
{
    unsigned int gpuId;     // !< optional gpu id for query
    unsigned short fieldId; // !< a true or false flag specifying a single parameter for basic GPU queries to libdcgm
} dcgmCoreQueryFieldParams_t;

/*
 * Struct for queries that are just specifying a list of GPUs to ask about
 */
typedef struct
{
    unsigned int gpuCount;
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES];
} dcgmCoreGpuListQueryParams_t;

/*
 * Struct for queries that are just specifying a list of fieldIds to ask about
 */
typedef struct
{
    unsigned int gpuCount;
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES];

    unsigned short *fieldIds; // !< List of the field ids we want samples for
    size_t numFieldIds;       // !< The number of field ids specified
} dcgmCoreFieldListQueryParams_t;

/**
 * This struct holds all of the request parameters necessary to call AddFieldWatch()
 */
typedef struct
{
    dcgm_field_entity_group_t entityGroupId; // !< entity group id
    dcgm_field_eid_t entityId;               // !< the entity id
    unsigned short fieldId;                  // !< DCGM field id
    timelib64_t monitorFreqUsec;             // !< refresh frequency in microseconds
    double maxSampleAge;                     // !< maximum age of sample to be kept
    int maxKeepSamples;                      // !< maximum number of samples to be kept
    DcgmWatcherType_t watcherType;           // !< identifier for who is watching
    dcgm_connection_id_t connectionId;       // !< connection id of the client watching
    bool subscribeForUpdates;                // !< flag for whether or not updates are subribed to
    bool updateOnFirstWatch;                 // !< Whether we should do an UpdateAllFields(true) if we were
                                             // !< the first watcher or not. Pass true if you want to guarantee
                                             // !< there is a value in the cache after this call. Pass false if you
                                             // !< don't care or plan to batch together a bunch of watches before
                                             // !<  an UpdateAllFields() at the end
} dcgmCoreAddFieldWatchReqParams_v1;

typedef struct
{
    bool wereFirstWatcher; // !< Whether we were the first watcher (true) or not (false). If so,
                           // !< you will need to call UpdateAllFields(true) for a value to be
                           // !< present in the cache.
} dcgmCoreAddFieldWatchRespParams_v1;

/**
 * This struct holds all of the parameters necessary to call GetInt64SummaryData()
 */
typedef struct
{
    dcgm_field_entity_group_t entityGroupId;              // !< entity group id
    dcgm_field_eid_t entityId;                            // !< the entity id
    unsigned short fieldId;                               // !< DCGM field id
    unsigned int summaryCount;                            // !< The number of summaries requested
    DcgmcmSummaryType_t summaries[DcgmcmSummaryTypeSize]; // !< the summaries we are requesting
    timelib64_t startTime;                                // !< the earliest timestamp a sample may have
    timelib64_t endTime;                                  // !< the latest timestamp a sample may have
    pfUseEntryForSummary useEntryCB;                      // !< callback function
    void *userData;                                       // !< input to the callback function
} dcgmCoreGetInt64SummaryDataParams_t;

/**
 * This struct holds all of the parameters necessary to call GetLatestSample()
 */
typedef struct
{
    dcgm_field_entity_group_t entityGroupId; // !< entity group id
    dcgm_field_eid_t entityId;               // !< the entity id
    unsigned short fieldId;                  // !< DCGM field id
    bool populateSamples;                    // !< flag for populating samples
    bool populateFvBuffer;                   // !< flag for populating a DcgmFvBuffer
} dcgmCoreGetLatestSampleParams_t;

/**
 * This struct holds all of the parameters necessary to call GetSamples()
 */
typedef struct
{
    dcgm_field_entity_group_t entityGroupId; // !< entity group id
    dcgm_field_eid_t entityId;               // !< the entity id
    unsigned short fieldId;                  // !< DCGM field id
    timelib64_t startTime;                   // !< the earliest timestamp a sample may have
    timelib64_t endTime;                     // !< the latest timestamp a sample may have
    int maxSamples;                          // !< the maximum number of samples that can be copied in
    dcgmOrder_t order;                       // !< flag specifying the ordering of samples
} dcgmCoreGetSamplesParams_t;


typedef struct
{
    unsigned int entityPairCount;                                  // !< The number of entities requested
    dcgmGroupEntityPair_t entityPairs[DCGM_GROUP_MAX_ENTITIES_V2]; // !< Entity group and id info for each entity
    unsigned short const *fieldIds;                                // !< List of the field ids we want samples for
    size_t numFieldIds;                                            // !< The number of field ids specified
    size_t bufferPosition;                                         // !< The start position for buffer copying
} dcgmCoreGetMultipleLatestLiveSamplesParams_v1;

#define dcgmCoreGetMultipleLatestLiveSamplesParams_version1 \
    MAKE_DCGM_VERSION(dcgmCoreGetMultipleLatestLiveSamplesParams_v1, 1)
#define dcgmCoreGetMultipleLatestLiveSamplesParams_version dcgmCoreGetMultipleLatestLiveSamplesParams_version1
typedef dcgmCoreGetMultipleLatestLiveSamplesParams_v1 dcgmCoreGetMultipleLatestLiveSamplesParams_t;

/**
 * This struct holds all of the parameters necessary to call RemoveFieldWatch()
 */
typedef struct
{
    dcgm_field_entity_group_t entityGroupId; // !< entity group id
    dcgm_field_eid_t entityId;               // !< the entity id
    unsigned short fieldId;                  // !< DCGM field id
    bool clearCache;                         // !< flag for whether or not to clear the cache
    DcgmWatcherType_t watcherType;           // !< identifier for who is watching
    dcgm_connection_id_t connectionId;       // !< connection id of the client watching
} dcgmCoreRemoveFieldWatchParams_t;

/**
 * This struct holds all of the parameters necessary to call AppendSamples()
 */
typedef struct
{
    const char *buffer; // !< Pointer to a raw DcgmFvBuffer. Owned by the object.
    size_t bufferSize;  // !< Size of the buffer
} dcgmCoreAppendSamplesParams_t;

/**
 * This struct holds all of the parameters necessary to call SetValue()
 */
typedef struct
{
    dcgmReturn_t ret;       // !< dcgmReturn_t from libdcgm
    int gpuId;              // !< the GPU id for the value
    unsigned short fieldId; // !< the field id for the value
    dcgmcm_sample_p value;  // !< the value we are setting
} dcgmCoreSetValueParams_t;

/**
 * Struct representing the most basic replies from libdcgm
 */
typedef struct
{
    dcgmReturn_t ret;        // !< dcgmReturn_t from libdcgm, if any
    unsigned int uintAnswer; // !< unsigned int response, if any
} dcgmCoreBasicResponse_t;

typedef struct
{
    dcgmReturn_t ret;     // !< dcgmReturn_t from libdcgm, if any
    long long longAnswer; // !< long response, if any
} dcgmCoreLongResponse_t;

/**
 *
 */
typedef struct
{
    dcgmReturn_t ret;
    unsigned int gpuCount;
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES];
} dcgmCoreGpuIdsResponse_t;

/**
 *
 */
typedef struct
{
    dcgmReturn_t ret;
    unsigned int infoCount;
    dcgmcm_gpu_info_cached_t info[DCGM_MAX_NUM_DEVICES];
} dcgmCoreGpuInfoResponse_t;

typedef struct
{
    dcgmReturn_t ret;                                              // !< The status of the function call
    dcgmGroupEntityPair_t entityPairs[DCGM_GROUP_MAX_ENTITIES_V2]; // !< Entity group and id info for each entity
    unsigned int entityPairsCount;                                 // !< Number of entity pairs in the list
} dcgmCoreEntityInfoResponse_v1;

#define dcgmCoreEntityInfoResponse_version1 MAKE_DCGM_VERSION(dcgmCoreEntityInfoResponse_v1, 1)
#define dcgmCoreEntityInfoResponse_version  dcgmCoreEntityInfoResponse_version1
typedef dcgmCoreEntityInfoResponse_v1 dcgmCoreEntityInfoResponse_t;

typedef struct
{
    dcgmReturn_t ret;                                                     // !< The status of the function call
    dcgmNvLinkLinkState_t linkStates[DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH]; // !< The status of each NvLink
} dcgmCoreNvLinkStatusResponse_t;

typedef struct
{
    dcgmReturn_t ret;                           // !< The status of the function call
    unsigned int numSummaryTypes;               // !< The number of summary types stored in this response
    long long summaryValues[DCGM_SUMMARY_SIZE]; // !< Array for storing each summary value
} dcgmCoreGetSummaryResponse_t;

typedef struct
{
    dcgmReturn_t ret;       // !< The status of the function call
    dcgmcm_sample_t sample; // !< The sample data
    char buffer[sizeof(dcgmBufferedFv_t)];
    size_t bufferSize;
} dcgmCoreGetLatestSampleResponse_t;

typedef struct
{
    dcgmReturn_t ret;        // !< The status of the function call
    dcgmcm_sample_p samples; // !< The memory address of the samples
    int numSamples;          // !< The number of samples stored at that address.
} dcgmCoreGetSamplesResponse_t;

#define SAMPLES_BUFFER_SIZE_V1 16384

typedef struct
{
    dcgmReturn_t ret;                    // !< The status of the function call
    char buffer[SAMPLES_BUFFER_SIZE_V1]; // !< The buffer we're copying the raw data into
    size_t bufferSize;                   // !< The length of data written into buffer
    unsigned char dataDidNotFit;         // !< 0 if the data all fit, 1 otherwise
} dcgmCoreGetMultipleLatestSamplesResponse_v1;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreBasicReqParams_t request;    // Parameters to pass to the method that will get called
    dcgmCoreBasicResponse_t response;    //
} dcgmCoreBasicQuery_v1;

#define dcgmCoreBasicQuery_version1 MAKE_DCGM_VERSION(dcgmCoreBasicQuery_v1, 1)
#define dcgmCoreBasicQuery_version  dcgmCoreBasicQuery_version1
typedef dcgmCoreBasicQuery_v1 dcgmCoreBasicQuery_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    void *command;
    dcgmReturn_t response;
} dcgmCoreSendModuleCommand_v1;

#define dcgmCoreSendModuleCommand_version1 MAKE_DCGM_VERSION(dcgmCoreSendModuleCommand_v1, 1)
#define dcgmCoreSendModuleCommand_version  dcgmCoreSendModuleCommand_version1
typedef dcgmCoreSendModuleCommand_v1 dcgmCoreSendModuleCommand_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreRawMessage_t request;
    dcgmReturn_t response;
} dcgmCoreSendRawMessage_v1;

#define dcgmCoreSendRawMessage_version1 MAKE_DCGM_VERSION(dcgmCoreSendRawMessage_v1, 1)
#define dcgmCoreSendRawMessage_version  dcgmCoreSendRawMessage_version1
typedef dcgmCoreSendRawMessage_v1 dcgmCoreSendRawMessage_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreNotifyRequestMessage_t request;
    dcgmReturn_t response;
} dcgmCoreNotifyRequestOfCompletion_v1;

#define dcgmCoreNotifyRequestOfCompletion_version1 MAKE_DCGM_VERSION(dcgmCoreNotifyRequestOfCompletion_v1, 1)
#define dcgmCoreNotifyRequestOfCompletion_version  dcgmCoreNotifyRequestOfCompletion_version1
typedef dcgmCoreNotifyRequestOfCompletion_v1 dcgmCoreNotifyRequestOfCompletion_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmFieldGrp_t fieldGrp;
    dcgmCoreFieldGroups_t response;
} dcgmCorePopulateFieldGroups_v1;

#define dcgmCorePopulateFieldGroups_version1 MAKE_DCGM_VERSION(dcgmCorePopulateFieldGroups_v1, 1)
#define dcgmCorePopulateFieldGroups_version  dcgmCorePopulateFieldGroups_version1
typedef dcgmCorePopulateFieldGroups_v1 dcgmCorePopulateFieldGroups_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmFieldGrp_t fieldGrp;
    dcgmCoreGetFieldGroupFieldsResponse_t response;
} dcgmCoreGetFieldGroupFields_v1;

#define dcgmCoreGetFieldGroupFields_version1 MAKE_DCGM_VERSION(dcgmCoreGetFieldGroupFields_v1, 1)
#define dcgmCoreGetFieldGroupFields_version  dcgmCoreGetFieldGroupFields_version1
typedef dcgmCoreGetFieldGroupFields_v1 dcgmCoreGetFieldGroupFields_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreBasicGroupParams_t request;
    dcgmCoreBasicResponse_t response;
} dcgmCoreBasicGroup_v1;

#define dcgmCoreBasicGroup_version1 MAKE_DCGM_VERSION(dcgmCoreBasicGroup_v1, 1)
#define dcgmCoreBasicGroup_version  dcgmCoreBasicGroup_version1
typedef dcgmCoreBasicGroup_v1 dcgmCoreBasicGroup_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreGpuQueryParams_t request;
    dcgmCoreBasicResponse_t response;
} dcgmCoreGetGpuCount_v1;

#define dcgmCoreGetGpuCount_version1 MAKE_DCGM_VERSION(dcgmCoreGetGpuCount_v1, 1)
#define dcgmCoreGetGpuCount_version  dcgmCoreGetGpuCount_version1
typedef dcgmCoreGetGpuCount_v1 dcgmCoreGetGpuCount_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreGpuQueryParams_t request;
    dcgmCoreGpuIdsResponse_t response;
} dcgmCoreGetGpuList_v1;

#define dcgmCoreGetGpuList_version1 MAKE_DCGM_VERSION(dcgmCoreGetGpuList_v1, 1)
#define dcgmCoreGetGpuList_version  dcgmCoreGetGpuList_version1
typedef dcgmCoreGetGpuList_v1 dcgmCoreGetGpuList_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreQueryFieldParams_t request;
    dcgmCoreBasicResponse_t response;
} dcgmCoreQueryField_v1;

#define dcgmCoreQueryField_version1 MAKE_DCGM_VERSION(dcgmCoreGetGpuList_v1, 1)
#define dcgmCoreQueryField_version  dcgmCoreGetGpuList_version1
typedef dcgmCoreQueryField_v1 dcgmCoreQueryField_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreGpuListQueryParams_t request;
    dcgmCoreBasicResponse_t response;
} dcgmCoreQueryGpuList_v1;

#define dcgmCoreQueryGpuList_version1 MAKE_DCGM_VERSION(dcgmCoreQueryGpuList_v1, 1)
#define dcgmCoreQueryGpuList_version  dcgmCoreQueryGpuList_version1
typedef dcgmCoreQueryGpuList_v1 dcgmCoreQueryGpuList_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreFieldListQueryParams_t request;
    dcgmCoreBasicResponse_t response;
} dcgmCoreQueryFieldList_v1;

#define dcgmCoreQueryFieldList_version1 MAKE_DCGM_VERSION(dcgmCoreQueryFieldList_v1, 1)
#define dcgmCoreQueryFieldList_version  dcgmCoreQueryFieldList_version1
typedef dcgmCoreQueryFieldList_v1 dcgmCoreQueryFieldList_t;


typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreQueryFieldParams_t request;
    dcgmCoreLongResponse_t response;
} dcgmCoreGetField_v1;

#define dcgmCoreGetField_version1 MAKE_DCGM_VERSION(dcgmCoreGetField_v1, 1)
#define dcgmCoreGetField_version  dcgmCoreGetField_version1
typedef dcgmCoreGetField_v1 dcgmCoreGetField_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreBasicReqParams_t request;
    dcgmCoreGpuInfoResponse_t response;
} dcgmCoreQueryGpuInfo_v1;

#define dcgmCoreQueryGpuInfo_version1 MAKE_DCGM_VERSION(dcgmCoreQueryGpuInfo_v1, 1)
#define dcgmCoreQueryGpuInfo_version  dcgmCoreQueryGpuInfo_version1
typedef dcgmCoreQueryGpuInfo_v1 dcgmCoreQueryGpuInfo_t;

typedef struct
{
    dcgm_module_command_header_t header;     // Command header
    dcgmCoreBasicReqParams_t request;        // Entity id and entity group id for the request
    dcgmCoreNvLinkStatusResponse_t response; // The NVLink status
} dcgmCoreGetEntityNvLinkLinkStatus_v1;

#define dcgmCoreGetEntityNvLinkLinkStatus_version1 MAKE_DCGM_VERSION(dcgmCoreGetEntityNvLinkLinkStatus_v1, 1)
#define dcgmCoreGetEntityNvLinkLinkStatus_version  dcgmCoreGetEntityNvLinkLinkStatus_version1
typedef dcgmCoreGetEntityNvLinkLinkStatus_v1 dcgmCoreGetEntityNvLinkLinkStatus_t;

typedef struct
{
    dcgm_module_command_header_t header;         // Command header
    dcgmCoreAddFieldWatchReqParams_v1 request;   // parameters for setting the field watch
    dcgmCoreAddFieldWatchRespParams_v1 response; // Output parameters from setting the field watch
    dcgmReturn_t ret;                            // Success or DCGM_ST_* error code
} dcgmCoreAddFieldWatch_v2;

#define dcgmCoreAddFieldWatch_version2 MAKE_DCGM_VERSION(dcgmCoreAddFieldWatch_v2, 2)

typedef struct
{
    dcgm_module_command_header_t header;         // Command header
    dcgmCoreGetInt64SummaryDataParams_t request; // IN:  parameters for getting the summary data
    dcgmCoreGetSummaryResponse_t response;       // OUT: The summary data
} dcgmCoreGetInt64SummaryData_v1;

#define dcgmCoreGetInt64SummaryData_version1 MAKE_DCGM_VERSION(dcgmCoreGetInt64SummaryData_v1, 1)
#define dcgmCoreGetInt64SummaryData_version  dcgmCoreGetInt64SummaryData_version1
typedef dcgmCoreGetInt64SummaryData_v1 dcgmCoreGetInt64SummaryData_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreGetLatestSampleParams_t request;
    dcgmCoreGetLatestSampleResponse_t response;
} dcgmCoreGetLatestSample_v1;

#define dcgmCoreGetLatestSample_version1 MAKE_DCGM_VERSION(dcgmCoreGetLatestSample_v1, 1)
#define dcgmCoreGetLatestSample_version  dcgmCoreGetLatestSample_version1
typedef dcgmCoreGetLatestSample_v1 dcgmCoreGetLatestSample_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreGetSamplesParams_t request;
    dcgmCoreGetSamplesResponse_t response;
} dcgmCoreGetSamples_v1;

#define dcgmCoreGetSamples_version1 MAKE_DCGM_VERSION(dcgmCoreGetSamples_v1, 1)
#define dcgmCoreGetSamples_version  dcgmCoreGetSamples_version1
typedef dcgmCoreGetSamples_v1 dcgmCoreGetSamples_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreGetMultipleLatestLiveSamplesParams_v1 request;
    dcgmCoreGetMultipleLatestSamplesResponse_v1 response;
} dcgmCoreGetMultipleLatestLiveSamples_v2;

#define dcgmCoreGetMultipleLatestLiveSamples_version2 MAKE_DCGM_VERSION(dcgmCoreGetMultipleLatestLiveSamples_v2, 2)
#define dcgmCoreGetMultipleLatestLiveSamples_version  dcgmCoreGetMultipleLatestLiveSamples_version2
typedef dcgmCoreGetMultipleLatestLiveSamples_v2 dcgmCoreGetMultipleLatestLiveSamples_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreRemoveFieldWatchParams_t request;
    dcgmReturn_t ret;
} dcgmCoreRemoveFieldWatch_v1;

#define dcgmCoreRemoveFieldWatch_version1 MAKE_DCGM_VERSION(dcgmCoreRemoveFieldWatch_v1, 1)
#define dcgmCoreRemoveFieldWatch_version  dcgmCoreRemoveFieldWatch_version1
typedef dcgmCoreRemoveFieldWatch_v1 dcgmCoreRemoveFieldWatch_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreAppendSamplesParams_t request;
    dcgmReturn_t ret;
} dcgmCoreAppendSamples_v1;

#define dcgmCoreAppendSamples_version1 MAKE_DCGM_VERSION(dcgmCoreAppendSamples_v1, 1)
#define dcgmCoreAppendSamples_version  dcgmCoreAppendSamples_version1
typedef dcgmCoreAppendSamples_v1 dcgmCoreAppendSamples_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreSetValueParams_t request;
    dcgmReturn_t ret;
} dcgmCoreSetValue_v1;

#define dcgmCoreSetValue_version1 MAKE_DCGM_VERSION(dcgmCoreSetValue_v1, 1)
#define dcgmCoreSetValue_version  dcgmCoreSetValue_version1
typedef dcgmCoreSetValue_v1 dcgmCoreSetValue_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreBasicGroupParams_t request;
    dcgmCoreEntityInfoResponse_t response;
} dcgmCoreGetGroupEntities_v2;

#define dcgmCoreGetGroupEntities_version2 MAKE_DCGM_VERSION(dcgmCoreGetGroupEntities_v2, 2)
#define dcgmCoreGetGroupEntities_version  dcgmCoreGetGroupEntities_version2
typedef dcgmCoreGetGroupEntities_v2 dcgmCoreGetGroupEntities_t;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    dcgmCoreBasicGroupParams_t request;
    dcgmCoreGpuIdsResponse_t response;
} dcgmCoreGetGroupGpuIds_v1;

#define dcgmCoreGetGroupGpuIds_version1 MAKE_DCGM_VERSION(dcgmCoreGetGroupGpuIds_v1, 1)
#define dcgmCoreGetGroupGpuIds_version  dcgmCoreGetGroupGpuIds_version1
typedef dcgmCoreGetGroupGpuIds_v1 dcgmCoreGetGroupGpuIds_t;

using DcgmLoggingRecord = plog::Record;

typedef struct
{
    dcgm_module_command_header_t header; // Command header
    loggerCategory_t request;            // which logger
    DcgmLoggingSeverity_t response;      // severity
} dcgmCoreGetSeverity_v1;

#define dcgmCoreGetSeverity_version1 MAKE_DCGM_VERSION(dcgmCoreGetSeverity_v1, 1)
#define dcgmCoreGetSeverity_version  dcgmCoreGetSeverity_version1
typedef dcgmCoreGetSeverity_v1 dcgmCoreGetSeverity_t;

typedef struct
{
    unsigned int gpuId;
    unsigned int instanceId;
    unsigned int computeInstanceId;
    dcgm_field_entity_group_t entityGroupId;
} dcgmCoreGetComputeInstanceEntityIdParams_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmCoreGetComputeInstanceEntityIdParams_t request;
    dcgmCoreBasicResponse_t response;
} dcgmCoreGetComputeInstanceEntityId_v1;

#define dcgmCoreGetComputeInstanceEntityId_version1 MAKE_DCGM_VERSION(dcgmCoreGetComputeInstanceEntityId_v1, 1)
#define dcgmCoreGetComputeInstanceEntityId_version  dcgmCoreGetComputeInstanceEntityId_version1
typedef dcgmCoreGetComputeInstanceEntityId_v1 dcgmCoreGetComputeInstanceEntityId_t;

typedef struct
{
    unsigned int gpuId;
    unsigned int instanceId;
    unsigned int computeInstanceId;
    dcgm_field_entity_group_t entityGroupId;
} dcgmCoreGetMigUtilizationParams_t;

typedef struct
{
    dcgmReturn_t ret;
    size_t capacity;
    size_t usage;
} dcgmCoreGetMigUtilizationResponse_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmCoreGetMigUtilizationParams_t request;
    dcgmCoreGetMigUtilizationResponse_t response;
} dcgmCoreGetMigUtilization_v1;

#define dcgmCoreGetMigUtilization_version1 MAKE_DCGM_VERSION(dcgmCoreGetMigUtilization_v1, 1)
#define dcgmCoreGetMigUtilization_version  dcgmCoreGetMigUtilization_version1
typedef dcgmCoreGetMigUtilization_v1 dcgmCoreGetMigUtilization_t;

typedef struct
{
    dcgm_field_entity_group_t entityGroupId;
    dcgm_field_eid_t entityId;
} dcgmCoreGetMigIndicesForEntityParams_t;

typedef struct
{
    dcgmReturn_t ret;
    unsigned int gpuId;
    dcgm_field_eid_t instanceId;
} dcgmCoreGetMigIndicesForEntityResponse_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmCoreGetMigIndicesForEntityParams_t request;
    dcgmCoreGetMigIndicesForEntityResponse_t response;
} dcgmCoreGetMigIndicesForEntity_v1;

#define dcgmCoreGetMigIndicesForEntity_version1 MAKE_DCGM_VERSION(dcgmCoreGetMigIndicesForEntity_v1, 1)
#define dcgmCoreGetMigIndicesForEntity_version  dcgmCoreGetMigIndicesForEntity_version1
typedef dcgmCoreGetMigIndicesForEntity_v1 dcgmCoreGetMigIndicesForEntity_t;

typedef struct
{
    char serviceAccount[256];
} dcgmCoreGetServiceAccountResponse_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmCoreGetServiceAccountResponse_t response;
} dcgmCoreGetServiceAccount_v1;

#define dcgmCoreGetServiceAccount_version1 MAKE_DCGM_VERSION(dcgmCoreGetServiceAccount_v1, 1)
#define dcgmCoreGetServiceAccount_version  dcgmCoreGetServiceAccount_version1
typedef dcgmCoreGetServiceAccount_v1 dcgmCoreGetServiceAccount_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmMigHierarchy_v2 response;
} dcgmCoreGetGpuInstanceHierarchy_v1;

#define dcgmCoreGetGpuInstanceHierarchy_version1 MAKE_DCGM_VERSION(dcgmCoreGetGpuInstanceHierarchy_v1, 1)
#define dcgmCoreGetGpuInstanceHierarchy_version  dcgmCoreGetGpuInstanceHierarchy_version1
typedef dcgmCoreGetGpuInstanceHierarchy_v1 dcgmCoreGetGpuInstanceHierarchy_t;

/**
 * Response for resource reservation request
 */
typedef struct
{
    dcgmReturn_t ret;   // !< The status of the function call
    unsigned int token; // !< Token for the reservation, used to free resources
} dcgmCoreResourceReserveRespParams_t;

/**
 * Parameters for resource freeing request
 */
typedef struct
{
    unsigned int token; // !< Token received during resource reservation
} dcgmCoreResourceFreeReqParams_t;

/**
 * Complete message structure for resource reservation
 */
typedef struct
{
    dcgm_module_command_header_t header;          // Command header
    dcgmCoreResourceReserveRespParams_t response; // Response parameters
} dcgmCoreResourceReserve_v1;

/**
 * Complete message structure for resource freeing
 */
typedef struct
{
    dcgm_module_command_header_t header;     // Command header
    dcgmCoreResourceFreeReqParams_t request; // Request parameters
    dcgmReturn_t ret;                        // Return status
} dcgmCoreResourceFree_v1;

#define dcgmCoreResourceReserve_version1 MAKE_DCGM_VERSION(dcgmCoreResourceReserve_v1, 1)
#define dcgmCoreResourceReserve_version  dcgmCoreResourceReserve_version1
typedef dcgmCoreResourceReserve_v1 dcgmCoreResourceReserve_t;

#define dcgmCoreResourceFree_version1 MAKE_DCGM_VERSION(dcgmCoreResourceFree_v1, 1)
#define dcgmCoreResourceFree_version  dcgmCoreResourceFree_version1
typedef dcgmCoreResourceFree_v1 dcgmCoreResourceFree_t;

/**
 * ChildProcess message structures
 */
typedef struct
{
    dcgm_module_command_header_t header;
    dcgmChildProcessParams_t request;
    dcgmReturn_t ret;
    ChildProcessHandle_t handle;
    int pid;
} dcgmCoreChildProcessSpawn_v1;

#define dcgmCoreChildProcessSpawn_version1 MAKE_DCGM_VERSION(dcgmCoreChildProcessSpawn_v1, 1)
#define dcgmCoreChildProcessSpawn_version  dcgmCoreChildProcessSpawn_version1
typedef dcgmCoreChildProcessSpawn_v1 dcgmCoreChildProcessSpawn_t;

typedef struct
{
    dcgm_module_command_header_t header;
    ChildProcessHandle_t handle;
    bool force;
    dcgmReturn_t ret;
} dcgmCoreChildProcessStop_v1;

#define dcgmCoreChildProcessStop_version1 MAKE_DCGM_VERSION(dcgmCoreChildProcessStop_v1, 1)
#define dcgmCoreChildProcessStop_version  dcgmCoreChildProcessStop_version1
typedef dcgmCoreChildProcessStop_v1 dcgmCoreChildProcessStop_t;

typedef struct
{
    dcgm_module_command_header_t header;
    ChildProcessHandle_t handle;
    dcgmChildProcessStatus_t status;
    dcgmReturn_t ret;
} dcgmCoreChildProcessGetStatus_v1;

#define dcgmCoreChildProcessGetStatus_version1 MAKE_DCGM_VERSION(dcgmCoreChildProcessGetStatus_v1, 1)
#define dcgmCoreChildProcessGetStatus_version  dcgmCoreChildProcessGetStatus_version1
typedef dcgmCoreChildProcessGetStatus_v1 dcgmCoreChildProcessGetStatus_t;

typedef struct
{
    dcgm_module_command_header_t header;
    ChildProcessHandle_t handle;
    int timeoutSec;
    dcgmReturn_t ret;
} dcgmCoreChildProcessWait_v1;

#define dcgmCoreChildProcessWait_version1 MAKE_DCGM_VERSION(dcgmCoreChildProcessWait_v1, 1)
#define dcgmCoreChildProcessWait_version  dcgmCoreChildProcessWait_version1
typedef dcgmCoreChildProcessWait_v1 dcgmCoreChildProcessWait_t;

typedef struct
{
    dcgm_module_command_header_t header;
    ChildProcessHandle_t handle;
    int sigTermTimeoutSec;
    dcgmReturn_t ret;
} dcgmCoreChildProcessDestroy_v1;

#define dcgmCoreChildProcessDestroy_version1 MAKE_DCGM_VERSION(dcgmCoreChildProcessDestroy_v1, 1)
#define dcgmCoreChildProcessDestroy_version  dcgmCoreChildProcessDestroy_version1
typedef dcgmCoreChildProcessDestroy_v1 dcgmCoreChildProcessDestroy_t;

typedef struct
{
    dcgm_module_command_header_t header;
    ChildProcessHandle_t handle;
    int fd;
    dcgmReturn_t ret;
} dcgmCoreChildProcessGetStdErrHandle_v1;

#define dcgmCoreChildProcessGetStdErrHandle_version1 MAKE_DCGM_VERSION(dcgmCoreChildProcessGetStdErrHandle_v1, 1)
#define dcgmCoreChildProcessGetStdErrHandle_version  dcgmCoreChildProcessGetStdErrHandle_version1
typedef dcgmCoreChildProcessGetStdErrHandle_v1 dcgmCoreChildProcessGetStdErrHandle_t;

typedef struct
{
    dcgm_module_command_header_t header;
    ChildProcessHandle_t handle;
    int fd;
    dcgmReturn_t ret;
} dcgmCoreChildProcessGetStdOutHandle_v1;

#define dcgmCoreChildProcessGetStdOutHandle_version1 MAKE_DCGM_VERSION(dcgmCoreChildProcessGetStdOutHandle_v1, 1)
#define dcgmCoreChildProcessGetStdOutHandle_version  dcgmCoreChildProcessGetStdOutHandle_version1
typedef dcgmCoreChildProcessGetStdOutHandle_v1 dcgmCoreChildProcessGetStdOutHandle_t;

typedef struct
{
    dcgm_module_command_header_t header;
    ChildProcessHandle_t handle;
    int fd;
    dcgmReturn_t ret;
} dcgmCoreChildProcessGetDataChannelHandle_v1;

#define dcgmCoreChildProcessGetDataChannelHandle_version1 \
    MAKE_DCGM_VERSION(dcgmCoreChildProcessGetDataChannelHandle_v1, 1)
#define dcgmCoreChildProcessGetDataChannelHandle_version dcgmCoreChildProcessGetDataChannelHandle_version1
typedef dcgmCoreChildProcessGetDataChannelHandle_v1 dcgmCoreChildProcessGetDataChannelHandle_t;

typedef struct
{
    dcgm_module_command_header_t header;
    char driverVersion[DCGM_MAX_STR_LENGTH];
    dcgmCoreBasicResponse_t response;
} dcgmCoreReqIdGetDriverVersion_v1;

#define dcgmCoreReqIdGetDriverVersion_version1 MAKE_DCGM_VERSION(dcgmCoreReqIdGetDriverVersion_v1, 1)
#define dcgmCoreReqIdGetDriverVersion_version  dcgmCoreReqIdGetDriverVersion_version1
typedef dcgmCoreReqIdGetDriverVersion_v1 dcgmCoreReqIdGetDriverVersion_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmReturn_t ret;
} dcgmCoreChildProcessManagerReset_v1;

#define dcgmCoreChildProcessManagerReset_version1 MAKE_DCGM_VERSION(dcgmCoreChildProcessManagerReset_v1, 1)
#define dcgmCoreChildProcessManagerReset_version  dcgmCoreChildProcessManagerReset_version1
typedef dcgmCoreChildProcessManagerReset_v1 dcgmCoreChildProcessManagerReset_t;