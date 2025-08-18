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
#include "dcgm_multinode_internal.h"
#include "dcgm_structs.h"
#include "dcgm_test_apis.h"
#include "dcgm_util.h"
#include "nvcmvalue.h"

#include "DcgmBuildInfo.hpp"
#include "DcgmFvBuffer.h"
#include "DcgmLogging.h"
#include "DcgmModuleApi.h"
#include "DcgmPolicyRequest.h"
#include "DcgmSettings.h"
#include "DcgmStatus.h"
#include "DcgmVersion.hpp"
#include "dcgm_config_structs.h"
#include "dcgm_diag_structs.h"
#include "dcgm_health_structs.h"
#include "dcgm_introspect_structs.h"
#include "dcgm_mndiag_structs.hpp"
#include "dcgm_nvswitch_structs.h"
#include "dcgm_policy_structs.h"
#include "dcgm_profiling_structs.h"
#include "dcgm_sysmon_structs.h"
#include "nvml_injection.h"
#include <DcgmStringHelpers.h>

#include <fmt/core.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <type_traits>
#include <unistd.h>

#ifdef INJECTION_LIBRARY_AVAILABLE
void nvmlClearLibraryHandleIfNeeded(void);
#endif

// Wrap each dcgmFunction with apiEnter and apiExit
#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)         \
    static dcgmReturn_t tsapiFuncname argtypes;                                   \
    dcgmReturn_t DCGM_PUBLIC_API dcgmFuncname argtypes                            \
    {                                                                             \
        dcgmReturn_t result;                                                      \
        log_debug("Entering {}{} " fmt, #dcgmFuncname, #argtypes, ##__VA_ARGS__); \
        result = apiEnter();                                                      \
        if (result != DCGM_ST_OK)                                                 \
        {                                                                         \
            return result;                                                        \
        }                                                                         \
        try                                                                       \
        {                                                                         \
            result = tsapiFuncname(__VA_ARGS__);                                  \
        }                                                                         \
        catch (const std::exception &e)                                           \
        {                                                                         \
            DCGM_LOG_ERROR << "Caught exception " << e.what();                    \
            result = DCGM_ST_GENERIC_ERROR;                                       \
        }                                                                         \
        catch (...)                                                               \
        {                                                                         \
            DCGM_LOG_ERROR << "Unknown exception ";                               \
            result = DCGM_ST_GENERIC_ERROR;                                       \
        }                                                                         \
        apiExit();                                                                \
        log_debug("Returning {}", result);                                        \
        return result;                                                            \
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
extern "C" {
#include "entry_point.h"
}
#pragma GCC diagnostic pop

#include "DcgmClientHandler.h"
#include "DcgmHostEngineHandler.h"
#include <mutex>

/* Define these outside of C linkage since they use C++ features */

/**
 * Structure for representing the global variables of DCGM within a process
 *
 * Variables of this structure are controlled by dcgmGlobalsMutex. Call dcgmGlobalsLock() and dcgmGlobalsUnlock()
 * to access these variables consistently
 */
typedef struct dcgm_globals_t
{
    int isInitialized;                /* Has DcgmInit() been successfully called? dcgmShutdown() sets this back to 0 */
    int fieldsAreInitialized;         /* Has DcgmFieldsInit() been successfully called? */
    int embeddedEngineStarted;        /* Have we started an embedded host engine? */
    int clientHandlerRefCount;        /* How many threads are currently using the client handler? This should be
                                  0 unless threads are in requests */
    DcgmClientHandler *clientHandler; /* Pointer to our client handler. This cannot be freed unless
                                         clientHandlerRefCount reaches 0. Eventually, this should be replaced
                                         with a shared_ptr */
} dcgm_globals_t, *dcgm_globals_p;

// Globals
static std::mutex g_dcgmGlobalsMutex;     /* Lock to control access to g_dcgmGlobals. This
                                         is declared outside of g_dcgmGlobals so g_dcgmGlobals
                                         can be memset to 0 in dcgmShutdown() */
static dcgm_globals_t g_dcgmGlobals = {}; /* Declared static so we don't export it */
static std::mutex g_dcgmPolicyCbMutex;    /* Lock to prevent users from unregistering policy callbacks while we
                                             are in the process of handling them. */

/*****************************************************************************
 * Functions used for locking/unlocking the globals of DCGM within a process
 ******************************************************************************/

static void dcgmGlobalsLock(void)
{
    g_dcgmGlobalsMutex.lock();
}

static void dcgmGlobalsUnlock(void)
{
    g_dcgmGlobalsMutex.unlock();
}

/*****************************************************************************
 *****************************************************************************/
/*****************************************************************************
 * Helper methods to unify code for remote cases and ISV agent case
 *****************************************************************************/
int helperUpdateErrorCodes(dcgmStatus_t statusHandle, int numStatuses, dcgm_config_status_t *statuses)
{
    DcgmStatus *pStatusObj;
    int i;

    if (!statusHandle || numStatuses < 1 || !statuses)
        return -1;

    pStatusObj = (DcgmStatus *)statusHandle;

    for (i = 0; i < numStatuses; i++)
    {
        pStatusObj->Enqueue(statuses[i].gpuId, statuses[i].fieldId, statuses[i].errorCode);
    }

    return 0;
}

/*****************************************************************************/
/* Get a pointer to the client handler. If this returns non-NULL, you need to call
   dcgmapiReleaseClientHandler() to decrease the reference count to it */
static DcgmClientHandler *dcgmapiAcquireClientHandler(bool shouldAllocate)
{
    DcgmClientHandler *retVal = NULL;

    dcgmGlobalsLock();

    if (g_dcgmGlobals.clientHandler)
    {
        g_dcgmGlobals.clientHandlerRefCount++;
        retVal = g_dcgmGlobals.clientHandler;
        log_debug("Incremented the client handler to {}", g_dcgmGlobals.clientHandlerRefCount);
    }
    else if (shouldAllocate)
    {
        try
        {
            g_dcgmGlobals.clientHandler = new DcgmClientHandler();
        }
        catch (const std::exception &e)
        {
            DCGM_LOG_ERROR << "Got system error exception: " << e.what();
            dcgmGlobalsUnlock();
            return nullptr;
        }

        DCGM_LOG_INFO << "Allocated the client handler";
        retVal                              = g_dcgmGlobals.clientHandler;
        g_dcgmGlobals.clientHandlerRefCount = 1;
    }
    /* Else: retVal is left as NULL. We want this in case another thread gets in and
             changes g_dcgmGlobals.clientHandler between the unlock and return */

    dcgmGlobalsUnlock();

    return retVal;
}

/*****************************************************************************/
/* Release a client handler that was acquired with dcgmapiAcquireClientHandler */
static void dcgmapiReleaseClientHandler()
{
    dcgmGlobalsLock();

    if (g_dcgmGlobals.clientHandler)
    {
        if (g_dcgmGlobals.clientHandlerRefCount < 1)
        {
            log_error("Client handler ref count underflowed. Tried to decrement from {}",
                      g_dcgmGlobals.clientHandlerRefCount);
        }
        else
        {
            g_dcgmGlobals.clientHandlerRefCount--;
            log_debug("Decremented the client handler to {}", g_dcgmGlobals.clientHandlerRefCount);
        }
    }

    dcgmGlobalsUnlock();
}

/*****************************************************************************/
/* free the client handler that was allocated with dcgmapiAcquireClientHandler */
static void dcgmapiFreeClientHandler()
{
    while (1)
    {
        /* We must not have the globals lock here or nobody else will be able to decrement the ref count */
        while (g_dcgmGlobals.clientHandlerRefCount > 0)
        {
            log_info("Waiting to destroy the client handler. Current refCount: {}",
                     g_dcgmGlobals.clientHandlerRefCount);
            sleep(1);
        }

        dcgmGlobalsLock();

        /* Now that we have the lock, we have to re-check state */

        if (!g_dcgmGlobals.clientHandler)
        {
            /* Another thread did our work for us. Unlock and get out */
            log_info("Another thread freed the client handler for us.");
            dcgmGlobalsUnlock();
            return;
        }
        if (g_dcgmGlobals.clientHandlerRefCount > 0)
        {
            /* Someone else got the lock and incremented the ref count while we were sleeping. Start over */
            dcgmGlobalsUnlock();
            log_info("Another thread acquired the client handler while we were sleeping.");
            continue;
        }

        delete g_dcgmGlobals.clientHandler;
        g_dcgmGlobals.clientHandler = NULL;
        dcgmGlobalsUnlock();
        log_info("Freed the client handler");
        break;
    }
}

/*****************************************************************************/
dcgmReturn_t processModuleCommandAtRemoteHostEngine(dcgmHandle_t pDcgmHandle,
                                                    dcgm_module_command_header_t *moduleCommand,
                                                    std::unique_ptr<DcgmRequest> request = nullptr,
                                                    unsigned int timeout                 = 60000,
                                                    size_t maxResponseSize               = 0)
{
    dcgmReturn_t ret;

    /* Check for Host Engine handle. This check is only needed in standalone
       case */
    if ((dcgmHandle_t) nullptr == pDcgmHandle)
    {
        DCGM_LOG_ERROR << "Invalid DCGM handle passed to processAtRemoteHostEngine. Handle = nullptr";
        return DCGM_ST_BADPARAM;
    }

    DcgmClientHandler *clientHandler = dcgmapiAcquireClientHandler(true);
    if (!clientHandler)
    {
        DCGM_LOG_ERROR << "Unable to acqire the client handler";
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the connectionId when not embedded */
    moduleCommand->connectionId = pDcgmHandle;

    /* Invoke method on the client side */
    ret = clientHandler->ExchangeModuleCommandAsync(
        pDcgmHandle, moduleCommand, std::move(request), maxResponseSize, timeout);

    dcgmapiReleaseClientHandler();
    return ret;
}

/*****************************************************************************/
dcgmReturn_t processModuleCommandAtEmbeddedHostEngine(dcgm_module_command_header_t *moduleCommand,
                                                      std::unique_ptr<DcgmRequest> request = nullptr)
{
    DcgmHostEngineHandler *pHEHandlerInstance = nullptr;
    dcgmReturn_t ret;
    dcgm_request_id_t requestId = DCGM_REQUEST_ID_NONE;

    /* Get Instance to Host Engine Handler */
    pHEHandlerInstance = DcgmHostEngineHandler::Instance();
    if (NULL == pHEHandlerInstance)
    {
        DCGM_LOG_ERROR << "DcgmHostEngineHandler::Instance() returned NULL";
        return DCGM_ST_UNINITIALIZED;
    }

    if (request != nullptr)
    {
        /* Subscribe to be updated when this request updates. This will also assign request->requestId */
        ret = pHEHandlerInstance->AddRequestWatcher(std::move(request), requestId);
        if (ret != DCGM_ST_OK)
        {
            log_error("AddRequestWatcher returned {}", ret);
            return ret;
        }
    }

    moduleCommand->requestId = requestId;

    /* Invoke Request handler method on the host engine */
    ret = (dcgmReturn_t)pHEHandlerInstance->ProcessModuleCommand(moduleCommand);
    return ret;
}


/*****************************************************************************/
dcgmReturn_t processModuleCommandAtHostEngine(dcgmHandle_t pDcgmHandle,
                                              dcgm_module_command_header_t *moduleCommand,
                                              size_t maxResponseSize               = 0,
                                              std::unique_ptr<DcgmRequest> request = nullptr,
                                              unsigned int timeout                 = 60000)
{
    if (pDcgmHandle != (dcgmHandle_t)DCGM_EMBEDDED_HANDLE) /* Remote DCGM */
    {
        // coverity[overrun]
        return processModuleCommandAtRemoteHostEngine(
            pDcgmHandle, moduleCommand, std::move(request), timeout, maxResponseSize);
    }
    else /* Implies Embedded HE mode. ISV Case */
    {
        // coverity[overrun]
        return processModuleCommandAtEmbeddedHostEngine(moduleCommand, std::move(request));
    }
}

/*****************************************************************************/
static dcgmReturn_t tsapiEngineGroupCreate(dcgmHandle_t pDcgmHandle,
                                           dcgmGroupType_t type,
                                           const char *groupName,
                                           dcgmGpuGrp_t *pDcgmGrpId)
{
    dcgmReturn_t ret;

    if ((groupName == NULL) || (pDcgmGrpId == NULL))
    {
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_create_group_v1 msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_CREATE_GROUP;
    msg.header.version    = dcgm_core_msg_create_group_version;
    msg.cg.groupType      = type;
    SafeCopyTo(msg.cg.groupName, groupName);

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    if (DCGM_ST_OK != msg.cg.cmdRet)
    {
        return msg.cg.cmdRet;
    }

    *pDcgmGrpId = msg.cg.newGroupId;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t tsapiEngineGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t grpId)
{
    dcgmReturn_t ret;

    dcgm_core_msg_group_destroy_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GROUP_DESTROY;
    msg.header.version    = dcgm_core_msg_group_destroy_version;

    msg.gd.groupId = grpId;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (DCGM_ST_OK != msg.gd.cmdRet)
    {
        return (dcgmReturn_t)msg.gd.cmdRet;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId)
{
    dcgm_core_msg_add_remove_entity_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GROUP_ADD_ENTITY;
    msg.header.version    = dcgm_core_msg_add_remove_entity_version;

    msg.re.groupId       = groupId;
    msg.re.entityGroupId = entityGroupId;
    msg.re.entityId      = entityId;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.re.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t helperGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t grpId, unsigned int gpuId)
{
    return cmHelperGroupAddEntity(pDcgmHandle, grpId, DCGM_FE_GPU, gpuId);
}

/*****************************************************************************/
dcgmReturn_t tsapiGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId)
{
    return cmHelperGroupAddEntity(pDcgmHandle, groupId, entityGroupId, entityId);
}

/*****************************************************************************/
dcgmReturn_t cmHelperGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                       dcgmGpuGrp_t groupId,
                                       dcgm_field_entity_group_t entityGroupId,
                                       dcgm_field_eid_t entityId)
{
    dcgmReturn_t ret;

    dcgm_core_msg_add_remove_entity_v1 msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_REMOVE_ENTITY;
    msg.header.version    = dcgm_core_msg_add_remove_entity_version;

    /* Set group ID to be removed from the hostengine */
    msg.re.groupId       = groupId;
    msg.re.entityGroupId = entityGroupId;
    msg.re.entityId      = entityId;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.re.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t helperGroupRemoveDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t grpId, unsigned int gpuId)
{
    return cmHelperGroupRemoveEntity(pDcgmHandle, grpId, DCGM_FE_GPU, gpuId);
}

/*****************************************************************************/
dcgmReturn_t tsapiGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId)
{
    return cmHelperGroupRemoveEntity(pDcgmHandle, groupId, entityGroupId, entityId);
}

/*****************************************************************************/
dcgmReturn_t tsapiGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t *pGroupIdList, unsigned int *pCount)
{
    if (!pGroupIdList || !pCount)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret;

    dcgm_core_msg_group_get_all_ids_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GROUP_GET_ALL_IDS;
    msg.header.version    = dcgm_core_msg_group_get_all_ids_version;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (msg.groups.cmdRet != DCGM_ST_OK)
    {
        return (dcgmReturn_t)msg.groups.cmdRet;
    }

    *pCount = msg.groups.numGroups;

    for (unsigned int i = 0; i < msg.groups.numGroups; ++i)
    {
        pGroupIdList[i] = msg.groups.groupIds[i];
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperGroupGetInfo(dcgmHandle_t pDcgmHandle,
                                dcgmGpuGrp_t groupId,
                                dcgmGroupInfo_t *pDcgmGroupInfo,
                                long long *hostEngineTimestamp)
{
    if (!pDcgmGroupInfo)
    {
        return DCGM_ST_BADPARAM;
    }

    /* Check for version */
    if ((pDcgmGroupInfo->version < dcgmGroupInfo_version3) || (pDcgmGroupInfo->version > dcgmGroupInfo_version))
    {
        DCGM_LOG_ERROR << "Struct version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    dcgmReturn_t ret;

    std::unique_ptr<dcgm_core_msg_group_get_info_t> msg = std::make_unique<dcgm_core_msg_group_get_info_t>();
    memset(msg.get(), 0, sizeof(*msg));
    msg->header.length     = sizeof(*msg);
    msg->header.moduleId   = DcgmModuleIdCore;
    msg->header.subCommand = DCGM_CORE_SR_GROUP_GET_INFO;
    msg->header.version    = dcgm_core_msg_group_get_info_version;

    msg->gi.groupId = groupId;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg->header, sizeof(*msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (msg->gi.cmdRet != DCGM_ST_OK)
    {
        return (dcgmReturn_t)msg->gi.cmdRet;
    }

    SafeCopyTo(pDcgmGroupInfo->groupName, msg->gi.groupInfo.groupName);

    if (hostEngineTimestamp)
    {
        *hostEngineTimestamp = msg->gi.timestamp;
    }

    pDcgmGroupInfo->count = msg->gi.groupInfo.count;

    if (pDcgmGroupInfo->count > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        DCGM_LOG_ERROR << "Invalid number of GPU Ids returned from the hostengine";
        return DCGM_ST_GENERIC_ERROR;
    }

    for (unsigned int index = 0; index < msg->gi.groupInfo.count; ++index)
    {
        pDcgmGroupInfo->entityList[index].entityGroupId = msg->gi.groupInfo.entityList[index].entityGroupId;
        pDcgmGroupInfo->entityList[index].entityId      = msg->gi.groupInfo.entityList[index].entityId;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 * This method is a common helper to get value for multiple fields
 *
 * dcgmHandle       IN: Handle to the host engine
 * groupId          IN: Optional groupId that will be resolved by the host engine.
 *                      This is ignored if entityList is provided.
 * entityList       IN: List of entities to retrieve values for. This value takes
 *                      precedence over groupId
 * entityListCount  IN: How many entries are contained in entityList[]
 * fieldGroupId     IN: Optional fieldGroupId that will be resolved by the host engine.
 *                      This is ignored if fieldIdList[] is provided
 * fieldIdList      IN: List of field IDs to retrieve values for. This value takes
 *                      precedence over fieldGroupId
 * fieldIdListCount IN: How many entries are contained in fieldIdList
 * fvBuffer        OUT: Field value buffer to save values into
 * flags            IN: Mask of DCGM_GMLV_FLAG_? flags that modify this request
 *
 *
 * @return DCGM_ST_OK on success
 *         Other DCGM_ST_? status code on error
 *
 *****************************************************************************/
dcgmReturn_t helperGetLatestValuesForFields(dcgmHandle_t dcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmGroupEntityPair_t *entityList,
                                            unsigned int entityListCount,
                                            dcgmFieldGrp_t fieldGroupId,
                                            unsigned short fieldIdList[],
                                            unsigned int fieldIdListCount,
                                            DcgmFvBuffer *fvBuffer,
                                            unsigned int flags)
{
    dcgmReturn_t ret;

    // Don't put a 4 MB object on the stack
    std::unique_ptr<dcgm_core_msg_entities_get_latest_values_v3> msg
        = std::make_unique<dcgm_core_msg_entities_get_latest_values_v3>();

    msg->header.length
        = sizeof(*msg) - SAMPLES_BUFFER_SIZE_V2; /* avoid transferring the large buffer when making request */
    msg->header.moduleId   = DcgmModuleIdCore;
    msg->header.subCommand = DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3;
    msg->header.version    = dcgm_core_msg_entities_get_latest_values_version3;

    if ((entityList && !entityListCount) || (fieldIdList && !fieldIdListCount) || !fvBuffer
        || entityListCount > DCGM_GROUP_MAX_ENTITIES_V2 || fieldIdListCount > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
    {
        DCGM_LOG_ERROR << "Bad parameter";
        return DCGM_ST_BADPARAM;
    }

    msg->ev.flags = flags;

    if (entityList)
    {
        memmove(&msg->ev.entities[0], entityList, entityListCount * sizeof(entityList[0]));
        msg->ev.entitiesCount = entityListCount;
    }
    else
    {
        msg->ev.groupId = groupId;
    }

    if (fieldIdList)
    {
        memmove(&msg->ev.fieldIdList[0], fieldIdList, fieldIdListCount * sizeof(fieldIdList[0]));
        msg->ev.fieldIdCount = fieldIdListCount;
    }
    else
    {
        msg->ev.fieldGroupId = fieldGroupId;
    }

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg->header, sizeof(*msg));
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    /* Did the request return a global request error (vs a field value status)? */
    if (DCGM_ST_OK != msg->ev.cmdRet)
    {
        DCGM_LOG_ERROR << "Got message status " << msg->ev.cmdRet;
        return (dcgmReturn_t)msg->ev.cmdRet;
    }

    /* Make a FV buffer from our protobuf string */
    fvBuffer->SetFromBuffer(msg->ev.buffer, msg->ev.bufferSize);
    return DCGM_ST_OK;
}

/****************************************************************************/
dcgmReturn_t tsapiEntitiesGetLatestValues(dcgmHandle_t dcgmHandle,
                                          dcgmGroupEntityPair_t entities[],
                                          unsigned int entityCount,
                                          unsigned short fields[],
                                          unsigned int fieldCount,
                                          unsigned int flags,
                                          dcgmFieldValue_v2 values[])
{
    dcgmReturn_t dcgmReturn;

    if (!entities || entityCount < 1 || !fields || fieldCount < 1 || !values)
    {
        DCGM_LOG_ERROR << "Bad parameter";
        return DCGM_ST_BADPARAM;
    }

    DcgmFvBuffer fvBuffer(0);

    dcgmReturn
        = helperGetLatestValuesForFields(dcgmHandle, 0, entities, entityCount, 0, fields, fieldCount, &fvBuffer, flags);
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    size_t bufferSize = 0, elementCount = 0;
    dcgmReturn = fvBuffer.GetSize(&bufferSize, &elementCount);
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Check that we got as many fields back as we requested */
    if (elementCount != fieldCount * entityCount)
    {
        DCGM_LOG_ERROR << "Returned FV mismatch. Requested " << entityCount * fieldCount << " != returned "
                       << elementCount;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Convert the buffered FVs to our output array */
    dcgmBufferedFvCursor_t cursor = 0;
    unsigned int valuesIndex      = 0;
    for (dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        fvBuffer.ConvertBufferedFvToFv2(fv, &values[valuesIndex]);
        valuesIndex++;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 * Common helper method for standalone and embedded case to fetch DCGM GPU Ids from
 * the system
 * @param mode          IN  :   Should be one of DCGM_MODE_?
 * @param pDcgmHandle   IN  :   HE Handle for Standalone case. NULL for Embedded case
 * @param pGpuIdList    OUT :   List of DCGM GPU Ids
 * @param pCount        OUT :   Number of GPUs in the list
 * @param onlySupported IN  :   Whether or not to only return devices that are supported
 *                              by DCGM. 1=only return DCGM-supported devices.
 *                                       0=return all devices in the system
 * @return
 *****************************************************************************/
dcgmReturn_t cmHelperGetAllDevices(dcgmHandle_t pDcgmHandle, unsigned int *pGpuIdList, int *pCount, int onlySupported)
{
    if ((NULL == pGpuIdList) || (NULL == pCount))
    {
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_get_all_devices_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_ALL_DEVICES;
    msg.header.version    = dcgm_core_msg_get_all_devices_version;

    msg.dev.supported = onlySupported;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    *pCount = msg.dev.count;

    for (int index = 0; index < msg.dev.count; index++)
    {
        pGpuIdList[index] = msg.dev.devices[index];
    }

    return (dcgmReturn_t)msg.dev.cmdRet;
}

/**
 * Common helper to get device attributes
 * @param mode
 * @param pDcgmHandle
 * @param gpuId
 * @param pDcgmDeviceAttr
 * @return
 */
dcgmReturn_t helperDeviceGetAttributes(dcgmHandle_t pDcgmHandle, int gpuId, dcgmDeviceAttributes_t *pDcgmDeviceAttr)
{
    unsigned short fieldIds[34];
    dcgmBufferedFv_t *fv;
    unsigned int count = 0;
    dcgmReturn_t ret;

    if (NULL == pDcgmDeviceAttr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (pDcgmDeviceAttr->version != dcgmDeviceAttributes_version)
    {
        return DCGM_ST_VER_MISMATCH;
    }


    fieldIds[count++] = DCGM_FI_DEV_SLOWDOWN_TEMP;
    fieldIds[count++] = DCGM_FI_DEV_SHUTDOWN_TEMP;
    fieldIds[count++] = DCGM_FI_DEV_ENFORCED_POWER_LIMIT;
    fieldIds[count++] = DCGM_FI_DEV_POWER_MGMT_LIMIT;
    fieldIds[count++] = DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF;
    fieldIds[count++] = DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX;
    fieldIds[count++] = DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN;
    fieldIds[count++] = DCGM_FI_DEV_SUPPORTED_CLOCKS;
    fieldIds[count++] = DCGM_FI_DEV_UUID;
    fieldIds[count++] = DCGM_FI_DEV_VBIOS_VERSION;
    fieldIds[count++] = DCGM_FI_DEV_INFOROM_IMAGE_VER;
    fieldIds[count++] = DCGM_FI_DEV_BRAND;
    fieldIds[count++] = DCGM_FI_DEV_NAME;
    fieldIds[count++] = DCGM_FI_DEV_SERIAL;
    fieldIds[count++] = DCGM_FI_DEV_PCI_BUSID;
    fieldIds[count++] = DCGM_FI_DEV_PCI_COMBINED_ID;
    fieldIds[count++] = DCGM_FI_DEV_PCI_SUBSYS_ID;
    fieldIds[count++] = DCGM_FI_DEV_BAR1_TOTAL;
    fieldIds[count++] = DCGM_FI_DEV_FB_TOTAL;
    fieldIds[count++] = DCGM_FI_DEV_FB_USED;
    fieldIds[count++] = DCGM_FI_DEV_FB_FREE;
    fieldIds[count++] = DCGM_FI_DRIVER_VERSION;
    fieldIds[count++] = DCGM_FI_DEV_VIRTUAL_MODE;
    fieldIds[count++] = DCGM_FI_DEV_PERSISTENCE_MODE;
    fieldIds[count++] = DCGM_FI_DEV_MIG_MODE;
    fieldIds[count++] = DCGM_FI_DEV_CC_MODE;

    if (count >= sizeof(fieldIds) / sizeof(fieldIds[0]))
    {
        log_error("Update DeviceGetAttributes to accommodate more fields");
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_GPU;
    entityPair.entityId      = gpuId;
    DcgmFvBuffer fvBuffer(0);
    ret = helperGetLatestValuesForFields(
        pDcgmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, DCGM_FV_FLAG_LIVE_DATA);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    size_t bufferSize = 0, elementCount = 0;
    ret = fvBuffer.GetSize(&bufferSize, &elementCount);
    if (elementCount != count)
    {
        log_error("Unexpected elementCount {} != count {} or ret {}", (int)elementCount, count, (int)ret);
        /* Keep going. We will only process what we have */
    }

    dcgmBufferedFvCursor_t cursor = 0;
    for (fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        switch (fv->fieldId)
        {
            case DCGM_FI_DEV_SLOWDOWN_TEMP:
                pDcgmDeviceAttr->thermalSettings.slowdownTemp = (unsigned int)nvcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_SHUTDOWN_TEMP:
                pDcgmDeviceAttr->thermalSettings.shutdownTemp = (unsigned int)nvcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT:
                pDcgmDeviceAttr->powerLimits.curPowerLimit = (unsigned int)nvcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_ENFORCED_POWER_LIMIT:
                pDcgmDeviceAttr->powerLimits.enforcedPowerLimit
                    = (unsigned int)nvcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF:
                pDcgmDeviceAttr->powerLimits.defaultPowerLimit = (unsigned int)nvcmvalue_double_to_int32(fv->value.dbl);

                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX:
                pDcgmDeviceAttr->powerLimits.maxPowerLimit = (unsigned int)nvcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN:
                pDcgmDeviceAttr->powerLimits.minPowerLimit = (unsigned int)nvcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_UUID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.uuid))
                {
                    log_error("String overflow error for the requested UUID field");
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.uuid, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.uuid, fv->value.str);
                }

                break;
            }

            case DCGM_FI_DEV_VBIOS_VERSION:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.vbios))
                {
                    log_error("String overflow error for the requested VBIOS field");
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.vbios, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.vbios, fv->value.str);
                }

                break;
            }

            case DCGM_FI_DEV_INFOROM_IMAGE_VER:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.inforomImageVersion))
                {
                    log_error("String overflow error for the requested Inforom field");
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.inforomImageVersion, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.inforomImageVersion, fv->value.str);
                }

                break;
            }

            case DCGM_FI_DEV_BRAND:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.brandName))
                {
                    log_error("String overflow error for the requested brand name field");
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.brandName, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.brandName, fv->value.str);
                }

                break;
            }

            case DCGM_FI_DEV_NAME:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.deviceName))
                {
                    log_error("String overflow error for the requested device name field");
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.deviceName, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.deviceName, fv->value.str);
                }

                break;
            }

            case DCGM_FI_DEV_SERIAL:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.serial))
                {
                    log_error("String overflow error for the requested serial field");
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.serial, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.serial, fv->value.str);
                }

                break;
            }

            case DCGM_FI_DEV_PCI_BUSID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.pciBusId))
                {
                    log_error("String overflow error for the requested serial field");
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.pciBusId, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.pciBusId, fv->value.str);
                }

                break;
            }

            case DCGM_FI_DEV_SUPPORTED_CLOCKS:
            {
                dcgmDeviceSupportedClockSets_t *supClocks = (dcgmDeviceSupportedClockSets_t *)fv->value.blob;

                if (!supClocks)
                {
                    memset(&pDcgmDeviceAttr->clockSets, 0, sizeof(pDcgmDeviceAttr->clockSets));
                    log_error("Null field value for DCGM_FI_DEV_SUPPORTED_CLOCKS");
                }
                else if (supClocks->version != dcgmDeviceSupportedClockSets_version)
                {
                    memset(&pDcgmDeviceAttr->clockSets, 0, sizeof(pDcgmDeviceAttr->clockSets));
                    log_error("Expected dcgmDeviceSupportedClockSets_version {}. Got {}",
                              (int)dcgmDeviceSupportedClockSets_version,
                              (int)supClocks->version);
                }
                else
                {
                    int payloadSize = (sizeof(*supClocks) - sizeof(supClocks->clockSet))
                                      + (supClocks->count * sizeof(supClocks->clockSet[0]));
                    if (payloadSize > (int)(fv->length - (sizeof(*fv) - sizeof(fv->value))))
                    {
                        log_error("DCGM_FI_DEV_SUPPORTED_CLOCKS calculated size {} > possible size {}",
                                  payloadSize,
                                  (int)(fv->length - (sizeof(*fv) - sizeof(fv->value))));
                        memset(&pDcgmDeviceAttr->clockSets, 0, sizeof(pDcgmDeviceAttr->clockSets));
                    }
                    else
                    {
                        /* Success */
                        memcpy(&pDcgmDeviceAttr->clockSets, supClocks, payloadSize);
                    }
                }
                break;
            }

            case DCGM_FI_DEV_PCI_COMBINED_ID:
                pDcgmDeviceAttr->identifiers.pciDeviceId = fv->value.i64;
                break;

            case DCGM_FI_DEV_PCI_SUBSYS_ID:
                pDcgmDeviceAttr->identifiers.pciSubSystemId = fv->value.i64;
                break;

            case DCGM_FI_DEV_BAR1_TOTAL:
                pDcgmDeviceAttr->memoryUsage.bar1Total = fv->value.i64;
                break;

            case DCGM_FI_DEV_FB_TOTAL:
                pDcgmDeviceAttr->memoryUsage.fbTotal = fv->value.i64;
                break;

            case DCGM_FI_DEV_FB_USED:
                pDcgmDeviceAttr->memoryUsage.fbUsed = fv->value.i64;
                break;

            case DCGM_FI_DEV_FB_FREE:
                pDcgmDeviceAttr->memoryUsage.fbFree = fv->value.i64;
                break;

            case DCGM_FI_DRIVER_VERSION:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.driverVersion))
                {
                    log_error("String overflow error for the requested driver version field");
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.driverVersion, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmDeviceAttr->identifiers.driverVersion, fv->value.str);
                }

                break;
            }

            case DCGM_FI_DEV_VIRTUAL_MODE:
                pDcgmDeviceAttr->identifiers.virtualizationMode = (unsigned int)nvcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_PERSISTENCE_MODE:
                pDcgmDeviceAttr->settings.persistenceModeEnabled = fv->value.i64;
                break;

            case DCGM_FI_DEV_MIG_MODE:
                pDcgmDeviceAttr->settings.migModeEnabled = fv->value.i64;
                break;

            case DCGM_FI_DEV_CC_MODE:
                pDcgmDeviceAttr->settings.confidentialComputeMode = fv->value.i64;
                break;

            default:
                /* This should never happen */
                return DCGM_ST_GENERIC_ERROR;
                break;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperWatchFieldValue(dcgmHandle_t pDcgmHandle,
                                   int gpuId,
                                   unsigned short fieldId,
                                   long long updateFreq,
                                   double maxKeepAge,
                                   int maxKeepSamples)
{
    if (!fieldId || updateFreq <= 0 || (maxKeepSamples <= 0 && maxKeepAge <= 0.0))
        return DCGM_ST_BADPARAM;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById((unsigned short)fieldId);
    if (NULL == fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        DCGM_LOG_ERROR << "field ID " << fieldId << " is not a valid field ID";
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_watch_field_value_v2 msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_WATCH_FIELD_VALUE_V2;
    msg.header.version    = dcgm_core_msg_watch_field_value_version2;

    msg.fv.fieldId        = fieldId;
    msg.fv.entityId       = gpuId;
    msg.fv.updateFreq     = updateFreq;
    msg.fv.maxKeepAge     = maxKeepAge;
    msg.fv.maxKeepSamples = maxKeepSamples;
    msg.fv.entityGroupId  = DCGM_FE_GPU;
    if (fieldMeta->scope == DCGM_FS_GLOBAL)
        msg.fv.entityGroupId = DCGM_FE_NONE;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Return code " << ret;
        return ret;
    }

    return (dcgmReturn_t)msg.fv.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t helperUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate)
{
    dcgm_core_msg_update_all_fields_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_UPDATE_ALL_FIELDS;
    msg.header.version    = dcgm_core_msg_update_all_fields_version;

    msg.uf.waitForUpdate = waitForUpdate;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Return code " << ret;
        return ret;
    }

    return (dcgmReturn_t)msg.uf.cmdRet;
}

/*****************************************************************************/
static dcgmReturn_t tsapiEngineUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate)
{
    return helperUpdateAllFields(pDcgmHandle, waitForUpdate);
}

/*****************************************************************************/
/**
 * Common helper to get vGPU device attributes
 * @param mode
 * @param pDcgmHandle
 * @param gpuId
 * @param pDcgmVgpuDeviceAttr
 * @return
 */
dcgmReturn_t helperVgpuDeviceGetAttributes(dcgmHandle_t pDcgmHandle,
                                           int gpuId,
                                           dcgmVgpuDeviceAttributes_t *pDcgmVgpuDeviceAttr)
{
    dcgmReturn_t ret;
    long long updateFreq = 30000000;
    double maxKeepAge    = 14400.0;
    int maxKeepSamples   = 480;

    if (NULL == pDcgmVgpuDeviceAttr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (pDcgmVgpuDeviceAttr->version != dcgmVgpuDeviceAttributes_version7)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    unsigned short fieldIds[] = { DCGM_FI_DEV_SUPPORTED_TYPE_INFO,
                                  DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS,
                                  DCGM_FI_DEV_VGPU_INSTANCE_IDS,
                                  DCGM_FI_DEV_VGPU_UTILIZATIONS,
                                  DCGM_FI_DEV_GPU_UTIL,
                                  DCGM_FI_DEV_MEM_COPY_UTIL,
                                  DCGM_FI_DEV_ENC_UTIL,
                                  DCGM_FI_DEV_DEC_UTIL };
    const unsigned int count  = sizeof(fieldIds) / sizeof(fieldIds[0]);
    DCGM_CASSERT(count <= 32, 0);

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_GPU;
    entityPair.entityId      = gpuId;

    DcgmFvBuffer fvBuffer(0);

    ret = helperGetLatestValuesForFields(pDcgmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, 0);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    dcgmBufferedFv_t *fv;
    dcgmBufferedFvCursor_t cursor = 0;

    int anyWatched = 0; /* Did we add any watches? */
    for (fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        if (fv->status == DCGM_ST_NOT_WATCHED)
        {
            ret = helperWatchFieldValue(pDcgmHandle, gpuId, fv->fieldId, updateFreq, maxKeepAge, maxKeepSamples);
            if (DCGM_ST_OK != ret)
            {
                return ret;
            }
            anyWatched = 1;
        }
    }

    if (anyWatched)
    {
        /* Make sure the new watches have updated */
        helperUpdateAllFields(pDcgmHandle, 1);

        /* Get all of the field values again now that everything has been watched */
        entityPair.entityGroupId = DCGM_FE_GPU;
        entityPair.entityId      = gpuId;
        ret = helperGetLatestValuesForFields(pDcgmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, 0);
        if (DCGM_ST_OK != ret)
        {
            return ret;
        }
    }

    size_t bufferSize = 0, elementCount = 0;
    ret = fvBuffer.GetSize(&bufferSize, &elementCount);
    if (elementCount != count)
    {
        log_error("Unexpected elementCount {} != count {} or ret {}", (int)elementCount, count, (int)ret);
        /* Keep going. We will only process what we have */
    }

    cursor = 0;
    for (fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        switch (fv->fieldId)
        {
            case DCGM_FI_DEV_SUPPORTED_TYPE_INFO:
            {
                dcgmDeviceVgpuTypeInfo_t *vgpuTypeInfo = (dcgmDeviceVgpuTypeInfo_t *)fv->value.blob;

                if (!vgpuTypeInfo)
                {
                    memset(&pDcgmVgpuDeviceAttr->supportedVgpuTypeInfo,
                           0,
                           sizeof(pDcgmVgpuDeviceAttr->supportedVgpuTypeInfo));
                    log_error("Null field value for DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS");
                    pDcgmVgpuDeviceAttr->supportedVgpuTypeCount = 0;
                    break;
                }
                else
                {
                    pDcgmVgpuDeviceAttr->supportedVgpuTypeCount = vgpuTypeInfo[0].vgpuTypeInfo.supportedVgpuTypeCount;
                }

                if (sizeof(pDcgmVgpuDeviceAttr->supportedVgpuTypeInfo)
                    < sizeof(*vgpuTypeInfo) * (pDcgmVgpuDeviceAttr->supportedVgpuTypeCount))
                {
                    memset(&pDcgmVgpuDeviceAttr->supportedVgpuTypeInfo,
                           0,
                           sizeof(pDcgmVgpuDeviceAttr->supportedVgpuTypeInfo));
                    log_error("vGPU Type ID static info array size {} too small for {} vGPU static info",
                              (int)sizeof(pDcgmVgpuDeviceAttr->supportedVgpuTypeInfo),
                              (int)sizeof(*vgpuTypeInfo) * (pDcgmVgpuDeviceAttr->supportedVgpuTypeCount));
                }
                else
                {
                    // coverity[overrun-buffer-arg] vgpuTypeInfo is an array of size supportedVgpuTypeCount
                    memcpy(&pDcgmVgpuDeviceAttr->supportedVgpuTypeInfo,
                           vgpuTypeInfo + 1,
                           sizeof(*vgpuTypeInfo) * (pDcgmVgpuDeviceAttr->supportedVgpuTypeCount));
                }
                break;
            }

            case DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS:
            {
                unsigned int *temp = (unsigned int *)fv->value.blob;

                if (!temp)
                {
                    memset(&pDcgmVgpuDeviceAttr->creatableVgpuTypeIds,
                           0,
                           sizeof(pDcgmVgpuDeviceAttr->creatableVgpuTypeIds));
                    log_error("Null field value for DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS");
                    pDcgmVgpuDeviceAttr->creatableVgpuTypeCount = 0;
                    break;
                }
                else
                {
                    pDcgmVgpuDeviceAttr->creatableVgpuTypeCount = temp[0];
                }

                if (sizeof(pDcgmVgpuDeviceAttr->creatableVgpuTypeIds)
                    < sizeof(*temp) * (pDcgmVgpuDeviceAttr->creatableVgpuTypeCount))
                {
                    memset(&pDcgmVgpuDeviceAttr->creatableVgpuTypeIds,
                           0,
                           sizeof(pDcgmVgpuDeviceAttr->creatableVgpuTypeIds));
                    log_error("Creatable vGPU Type IDs array size {} too small for {} Id value",
                              (int)sizeof(pDcgmVgpuDeviceAttr->creatableVgpuTypeIds),
                              (int)sizeof(*temp) * (pDcgmVgpuDeviceAttr->creatableVgpuTypeCount));
                }
                else
                {
                    memcpy(&pDcgmVgpuDeviceAttr->creatableVgpuTypeIds,
                           temp + 1,
                           sizeof(*temp) * (pDcgmVgpuDeviceAttr->creatableVgpuTypeCount));
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_INSTANCE_IDS:
            {
                unsigned int *temp = (unsigned int *)fv->value.blob;

                if (!temp)
                {
                    memset(&pDcgmVgpuDeviceAttr->activeVgpuInstanceIds,
                           0,
                           sizeof(pDcgmVgpuDeviceAttr->activeVgpuInstanceIds));
                    log_error("Null field value for DCGM_FI_DEV_VGPU_INSTANCE_IDS");
                    pDcgmVgpuDeviceAttr->activeVgpuInstanceCount = 0;
                    break;
                }
                else
                {
                    pDcgmVgpuDeviceAttr->activeVgpuInstanceCount = temp[0];
                }

                if (sizeof(pDcgmVgpuDeviceAttr->activeVgpuInstanceIds)
                    < sizeof(*temp) * (pDcgmVgpuDeviceAttr->activeVgpuInstanceCount))
                {
                    memset(&pDcgmVgpuDeviceAttr->activeVgpuInstanceIds,
                           0,
                           sizeof(pDcgmVgpuDeviceAttr->activeVgpuInstanceIds));
                    log_error("Active vGPU Instance IDs array size {} too small for {} Id value",
                              (int)sizeof(pDcgmVgpuDeviceAttr->activeVgpuInstanceIds),
                              (int)sizeof(*temp) * (pDcgmVgpuDeviceAttr->activeVgpuInstanceCount));
                }
                else
                {
                    memcpy(&pDcgmVgpuDeviceAttr->activeVgpuInstanceIds,
                           temp + 1,
                           sizeof(*temp) * (pDcgmVgpuDeviceAttr->activeVgpuInstanceCount));
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_UTILIZATIONS:
            {
                dcgmDeviceVgpuUtilInfo_t *vgpuUtilInfo = (dcgmDeviceVgpuUtilInfo_t *)fv->value.blob;

                if (!vgpuUtilInfo)
                {
                    memset(&pDcgmVgpuDeviceAttr->vgpuUtilInfo, 0, sizeof(pDcgmVgpuDeviceAttr->vgpuUtilInfo));
                    log_error("Null field value for DCGM_FI_DEV_VGPU_UTILIZATIONS");
                    break;
                }

                if (sizeof(pDcgmVgpuDeviceAttr->vgpuUtilInfo)
                    < sizeof(*vgpuUtilInfo) * (pDcgmVgpuDeviceAttr->activeVgpuInstanceCount))
                {
                    memset(&pDcgmVgpuDeviceAttr->vgpuUtilInfo, 0, sizeof(pDcgmVgpuDeviceAttr->vgpuUtilInfo));
                    log_error("Active vGPU Instance IDs utilizations array size {} too small for {} Id value",
                              (int)sizeof(pDcgmVgpuDeviceAttr->vgpuUtilInfo),
                              (int)sizeof(*vgpuUtilInfo) * (pDcgmVgpuDeviceAttr->activeVgpuInstanceCount));
                }
                else
                {
                    memcpy(&pDcgmVgpuDeviceAttr->vgpuUtilInfo,
                           vgpuUtilInfo,
                           sizeof(*vgpuUtilInfo) * (pDcgmVgpuDeviceAttr->activeVgpuInstanceCount));
                }
                break;
            }

            case DCGM_FI_DEV_GPU_UTIL:
            {
                pDcgmVgpuDeviceAttr->gpuUtil = fv->value.i64;
                break;
            }

            case DCGM_FI_DEV_MEM_COPY_UTIL:
            {
                pDcgmVgpuDeviceAttr->memCopyUtil = fv->value.i64;
                break;
            }

            case DCGM_FI_DEV_ENC_UTIL:
            {
                pDcgmVgpuDeviceAttr->encUtil = fv->value.i64;
                break;
            }

            case DCGM_FI_DEV_DEC_UTIL:
            {
                pDcgmVgpuDeviceAttr->decUtil = fv->value.i64;
                break;
            }

            default:
                /* This should never happen */
                return DCGM_ST_GENERIC_ERROR;
                break;
        }
    }

    return DCGM_ST_OK;
}

/**
 * Common helper to get attributes specific to vGPU instance
 * @param mode
 * @param pDcgmHandle
 * @param vgpuId
 * @param pDcgmVgpuInstanceAttr
 * @return
 */
dcgmReturn_t helperVgpuInstanceGetAttributes(dcgmHandle_t pDcgmHandle,
                                             int vgpuId,
                                             dcgmVgpuInstanceAttributes_t *pDcgmVgpuInstanceAttr)
{
    dcgmReturn_t ret;

    if (NULL == pDcgmVgpuInstanceAttr)
    {
        return DCGM_ST_BADPARAM;
    }

    if ((pDcgmVgpuInstanceAttr->version < dcgmVgpuInstanceAttributes_version1)
        || (pDcgmVgpuInstanceAttr->version > dcgmVgpuInstanceAttributes_version))
    {
        return DCGM_ST_VER_MISMATCH;
    }

    unsigned short fieldIds[] = { DCGM_FI_DEV_VGPU_VM_ID,
                                  DCGM_FI_DEV_VGPU_VM_NAME,
                                  DCGM_FI_DEV_VGPU_TYPE,
                                  DCGM_FI_DEV_VGPU_UUID,
                                  DCGM_FI_DEV_VGPU_DRIVER_VERSION,
                                  DCGM_FI_DEV_VGPU_MEMORY_USAGE,
                                  DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATE,
                                  DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT };
    const unsigned int count  = sizeof(fieldIds) / sizeof(fieldIds[0]);
    DCGM_CASSERT(count <= 32, 0);

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_VGPU;
    entityPair.entityId      = vgpuId;

    DcgmFvBuffer fvBuffer(0);

    ret = helperGetLatestValuesForFields(pDcgmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, 0);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    dcgmBufferedFv_t *fv;
    dcgmBufferedFvCursor_t cursor = 0;

    for (fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        switch (fv->fieldId)
        {
            case DCGM_FI_DEV_VGPU_VM_ID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmVgpuInstanceAttr->vmId))
                {
                    log_error("String overflow error for the requested vGPU instance VM ID field");
                    SafeCopyTo(pDcgmVgpuInstanceAttr->vmId, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmVgpuInstanceAttr->vmId, fv->value.str);
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_VM_NAME:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmVgpuInstanceAttr->vmName))
                {
                    log_error("String overflow error for the requested vGPU instance VM name field");
                    SafeCopyTo(pDcgmVgpuInstanceAttr->vmName, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmVgpuInstanceAttr->vmName, fv->value.str);
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_TYPE:
                pDcgmVgpuInstanceAttr->vgpuTypeId = fv->value.i64;
                break;

            case DCGM_FI_DEV_VGPU_UUID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmVgpuInstanceAttr->vgpuUuid))
                {
                    log_error("String overflow error for the requested vGPU instance UUID field");
                    SafeCopyTo(pDcgmVgpuInstanceAttr->vgpuUuid, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmVgpuInstanceAttr->vgpuUuid, fv->value.str);
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_DRIVER_VERSION:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmVgpuInstanceAttr->vgpuDriverVersion))
                {
                    log_error("String overflow error for the requested vGPU instance driver version field");
                    SafeCopyTo(pDcgmVgpuInstanceAttr->vgpuDriverVersion, DCGM_STR_BLANK);
                }
                else
                {
                    SafeCopyTo(pDcgmVgpuInstanceAttr->vgpuDriverVersion, fv->value.str);
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_MEMORY_USAGE:
                pDcgmVgpuInstanceAttr->fbUsage = fv->value.i64;
                break;

            case DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATE:
                pDcgmVgpuInstanceAttr->licenseStatus = fv->value.i64;
                break;

            case DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT:
                pDcgmVgpuInstanceAttr->frameRateLimit = fv->value.i64;
                break;

            default:
                /* This should never happen */
                return DCGM_ST_GENERIC_ERROR;
                break;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 * Helper method to set the configuration for a group
 *****************************************************************************/
dcgmReturn_t helperConfigSet(dcgmHandle_t dcgmHandle,
                             dcgmGpuGrp_t groupId,
                             dcgmConfig_t *pDeviceConfig,
                             dcgmStatus_t pDcgmStatusList)
{
    dcgm_config_msg_set_v1 msg;
    dcgmReturn_t dcgmReturn;

    if (!pDeviceConfig)
    {
        log_error("Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    if (pDeviceConfig->version != dcgmConfig_version)
    {
        log_error("Version mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdConfig;
    msg.header.subCommand = DCGM_CONFIG_SR_SET;
    msg.header.version    = dcgm_config_msg_set_version;
    msg.groupId           = groupId;
    memcpy(&msg.config, pDeviceConfig, sizeof(msg.config));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Update error codes from the error list if there are any */
    if (msg.numStatuses && pDcgmStatusList)
    {
        helperUpdateErrorCodes(pDcgmStatusList, msg.numStatuses, msg.statuses);
    }

    return dcgmReturn;
}

/*****************************************************************************
 * Helper method to set the vGPU configuration for a group
 *****************************************************************************/
dcgmReturn_t helperVgpuConfigSet(dcgmHandle_t /* pDcgmHandle */,
                                 dcgmGpuGrp_t /* groupId */,
                                 dcgmVgpuConfig_t *pDeviceConfig,
                                 dcgmStatus_t /* pDcgmStatusList */)
{
    if (!pDeviceConfig)
    {
        DCGM_LOG_ERROR << "bad pDeviceConfig " << (void *)pDeviceConfig;
        return DCGM_ST_BADPARAM;
    }

    if ((pDeviceConfig->version < dcgmVgpuConfig_version1) || (pDeviceConfig->version > dcgmVgpuConfig_version))
    {
        log_error("VgpuConfigSet version {:x} mismatches current version {:x}",
                  pDeviceConfig->version,
                  dcgmVgpuConfig_version);
        return DCGM_ST_VER_MISMATCH;
    }

    /* This code never worked, this API is private, and the tests in test_vgpu.py are disabled.
       Returning NOT_SUPPORTED for now */
    return DCGM_ST_NOT_SUPPORTED;
}

/*****************************************************************************/
dcgmReturn_t helperConfigGet(dcgmHandle_t dcgmHandle,
                             dcgmGpuGrp_t groupId,
                             dcgmConfigType_t reqType,
                             int count,
                             dcgmConfig_t *pDeviceConfigList,
                             dcgmStatus_t pDcgmStatusList)
{
    dcgm_config_msg_get_v1 msg;
    dcgmReturn_t dcgmReturn;
    int i;
    unsigned int versionAtBaseIndex;

    if ((!pDeviceConfigList) || (count <= 0)
        || ((reqType != DCGM_CONFIG_TARGET_STATE) && (reqType != DCGM_CONFIG_CURRENT_STATE)))
    {
        log_error("Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    /* Get version at the index 0 */
    versionAtBaseIndex = pDeviceConfigList[0].version;

    /* Verify requested version in the list of output parameters */
    for (i = 0; i < count; ++i)
    {
        if (pDeviceConfigList[i].version != versionAtBaseIndex)
        {
            log_error("Version mismatch");
            return DCGM_ST_VER_MISMATCH;
        }

        if (pDeviceConfigList[i].version != dcgmConfig_version)
        {
            log_error("Version mismatch");
            return DCGM_ST_VER_MISMATCH;
        }
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdConfig;
    msg.header.subCommand = DCGM_CONFIG_SR_GET;
    msg.header.version    = dcgm_config_msg_get_version;
    msg.groupId           = groupId;
    msg.reqType           = reqType;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Update error codes from the error list if there are any */
    if (msg.numStatuses && pDcgmStatusList)
    {
        helperUpdateErrorCodes(pDcgmStatusList, msg.numStatuses, msg.statuses);
    }

    /* Copy the configs back to the caller's array */
    if (msg.numConfigs > 0)
    {
        unsigned int numToCopy = std::min(msg.numConfigs, (unsigned int)count);
        memcpy(pDeviceConfigList, msg.configs, numToCopy * sizeof(msg.configs[0]));
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperVgpuConfigGet(dcgmHandle_t /* pDcgmHandle */,
                                 dcgmGpuGrp_t /* groupId */,
                                 dcgmConfigType_t /* reqType */,
                                 int count,
                                 dcgmVgpuConfig_t *pDeviceConfigList,
                                 dcgmStatus_t /* pDcgmStatusList */)
{
    if (!pDeviceConfigList || count < 1)
        return DCGM_ST_BADPARAM;

    /* Get version at the index 0 */
    unsigned int versionAtBaseIndex = pDeviceConfigList[0].version;
    int index;

    /* Verify requested version in the list of output parameters */
    for (index = 0; index < count; ++index)
    {
        if (pDeviceConfigList[index].version != versionAtBaseIndex)
        {
            return DCGM_ST_VER_MISMATCH;
        }

        if ((pDeviceConfigList[index].version < dcgmVgpuConfig_version1)
            || (pDeviceConfigList[index].version > dcgmVgpuConfig_version))
        {
            return DCGM_ST_VER_MISMATCH;
        }
    }

    /* This code never worked, this API is private, and the tests in test_vgpu.py are disabled.
       Returning NOT_SUPPORTED for now */
    return DCGM_ST_NOT_SUPPORTED;
}

/*****************************************************************************/
dcgmReturn_t helperConfigEnforce(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t pDcgmStatusList)
{
    dcgm_config_msg_enforce_group_v1 msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdConfig;
    msg.header.subCommand = DCGM_CONFIG_SR_ENFORCE_GROUP;
    msg.header.version    = dcgm_config_msg_enforce_group_version;
    msg.groupId           = groupId;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Update error codes from the error list if there are any */
    if (msg.numStatuses && pDcgmStatusList)
    {
        helperUpdateErrorCodes(pDcgmStatusList, msg.numStatuses, msg.statuses);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperVgpuConfigEnforce(dcgmHandle_t /* pDcgmHandle */,
                                     dcgmGpuGrp_t /* groupId */,
                                     dcgmStatus_t /* pDcgmStatusList */)
{
    /* This code never worked, this API is private, and the tests in test_vgpu.py are disabled.
       Returning NOT_SUPPORTED for now */
    return DCGM_ST_NOT_SUPPORTED;
}

/*****************************************************************************/
static dcgmReturn_t tsapiInjectEntityFieldValue(dcgmHandle_t pDcgmHandle,
                                                dcgm_field_entity_group_t entityGroupId,
                                                dcgm_field_eid_t entityId,
                                                dcgmInjectFieldValue_t *pDcgmInjectFieldValue)
{
    if (NULL == pDcgmInjectFieldValue)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_inject_field_value_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_INJECT_FIELD_VALUE;
    msg.header.version    = dcgm_core_msg_inject_field_value_version;

    msg.iv.entityGroupId = entityGroupId;
    msg.iv.entityId      = entityId;
    memcpy(&msg.iv.fieldValue, pDcgmInjectFieldValue, sizeof(msg.iv.fieldValue));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        log_debug("dcgmModuleSendBlockingFixedRequest returned {}", (int)ret);
        return ret;
    }

    return (dcgmReturn_t)msg.iv.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t helperInjectFieldValue(dcgmHandle_t pDcgmHandle,
                                    unsigned int gpuId,
                                    dcgmInjectFieldValue_t *pDcgmInjectFieldValue)
{
    return tsapiInjectEntityFieldValue(pDcgmHandle, DCGM_FE_GPU, gpuId, pDcgmInjectFieldValue);
}

/*****************************************************************************/
dcgmReturn_t tsapiEngineGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle, dcgmCacheManagerFieldInfo_v4_t *fieldInfo)
{
    if (!fieldInfo)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_get_cache_manager_field_info_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_CACHE_MANAGER_FIELD_INFO;
    msg.header.version    = dcgm_core_msg_get_cache_manager_field_info_version2;

    memcpy(&msg.fi.fieldInfo, fieldInfo, sizeof(msg.fi.fieldInfo));
    msg.fi.fieldInfo.version = dcgmCacheManagerFieldInfo_version4;
    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        log_debug("dcgmModuleSendBlockingFixedRequest returned {}", (int)ret);
        return ret;
    }

    if (DCGM_ST_OK != msg.fi.cmdRet)
    {
        return (dcgmReturn_t)msg.fi.cmdRet;
    }

    memcpy(fieldInfo, &msg.fi.fieldInfo, sizeof(dcgmCacheManagerFieldInfo_v4_t));

    return (dcgmReturn_t)msg.fi.cmdRet;
}

/*****************************************************************************/
static dcgmReturn_t helperGetMultipleValuesForFieldFvBuffer(dcgmHandle_t pDcgmHandle,
                                                            dcgm_field_entity_group_t entityGroup,
                                                            dcgm_field_eid_t entityId,
                                                            unsigned int fieldId,
                                                            int *count,
                                                            long long startTs,
                                                            long long endTs,
                                                            dcgmOrder_t order,
                                                            DcgmFvBuffer *fvBuffer)
{
    dcgmReturn_t ret;
    int maxCount;
    dcgm_field_meta_p fieldMeta;

    if (!count || (*count) < 1 || !fieldId || fvBuffer == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    maxCount = *count;
    *count   = 0;

    log_debug("helperGetMultipleValuesForField eg {} eid {}, "
              "fieldId {}, maxCount {}, startTs {} endTs {}, order {}",
              entityGroup,
              entityId,
              (int)fieldId,
              maxCount,
              startTs,
              endTs,
              (int)order);

    /* Validate the fieldId */
    fieldMeta = DcgmFieldGetById(fieldId);
    if (!fieldMeta)
    {
        log_error("Invalid fieldId {}", fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    auto msg = std::make_unique<dcgm_core_msg_get_multiple_values_for_field_v2>();

    msg->header.length
        = sizeof(*msg) - sizeof(msg->fv.buffer); /* avoid transferring the large buffer when making request */
    msg->header.moduleId   = DcgmModuleIdCore;
    msg->header.subCommand = DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2;
    msg->header.version    = dcgm_core_msg_get_multiple_values_for_field_version2;

    msg->fv.entityId = entityId;
    msg->fv.fieldId  = fieldId;
    msg->fv.startTs  = startTs;
    msg->fv.endTs    = endTs;
    msg->fv.count    = maxCount;
    msg->fv.order    = order;

    if (fieldMeta->scope == DCGM_FS_GLOBAL)
        msg->fv.entityGroupId = DCGM_FE_NONE;
    else
        msg->fv.entityGroupId = entityGroup;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg->header, sizeof(*msg));

    if (DCGM_ST_OK != ret)
    {
        log_debug("dcgmModuleSendBlockingFixedRequest returned {}", (int)ret);
        return ret;
    }

    /* Check the status of the DCGM command */
    ret = (dcgmReturn_t)msg->fv.cmdRet;
    if (ret == DCGM_ST_NO_DATA || ret == DCGM_ST_NOT_SUPPORTED)
    {
        DCGM_LOG_WARNING << "Handling ret " << ret << " for eg " << entityGroup << " eid " << entityId
                         << " by returning a single fv with that error code.";
        /* Handle these returns by setting the field value status rather than failing the API */
        *count = 1;
        fvBuffer->AddBlankValue(entityGroup, entityId, fieldId, ret);
        return DCGM_ST_OK;
    }
    else if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_WARNING << "vecCmdsRef[0]->status() " << ret;
        return ret;
    }

    *count = msg->fv.count;
    fvBuffer->SetFromBuffer(msg->fv.buffer, msg->fv.bufferSize);
    return DCGM_ST_OK;
}

/*****************************************************************************/
static dcgmReturn_t helperGetMultipleValuesForFieldFV1s(dcgmHandle_t pDcgmHandle,
                                                        dcgm_field_entity_group_t entityGroup,
                                                        dcgm_field_eid_t entityId,
                                                        unsigned int fieldId,
                                                        int *count,
                                                        long long startTs,
                                                        long long endTs,
                                                        dcgmOrder_t order,
                                                        dcgmFieldValue_v1 values[])
{
    int i;

    memset(values, 0, sizeof(values[0]) * (*count));

    DcgmFvBuffer fvBuffer(0);

    dcgmReturn_t dcgmReturn = helperGetMultipleValuesForFieldFvBuffer(
        pDcgmHandle, entityGroup, entityId, fieldId, count, startTs, endTs, order, &fvBuffer);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got " << dcgmReturn << " from helperGetMultipleValuesForFieldFvBuffer()";
        return dcgmReturn;
    }

    /* Convert the buffered FVs to our output array */
    dcgmBufferedFvCursor_t cursor = 0;
    i                             = 0;
    for (dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor); fv && i < (*count); fv = fvBuffer.GetNextFv(&cursor))
    {
        fvBuffer.ConvertBufferedFvToFv1(fv, &values[i]);
        i++;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t tsapiEngineUnwatchFieldValue(dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldId, int clearCache)
{
    dcgmReturn_t ret;

    if (!fieldId)
        return DCGM_ST_BADPARAM;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById((unsigned short)fieldId);
    if (NULL == fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        log_error("field ID {} is not a valid field ID", fieldId);
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_unwatch_field_value_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_UNWATCH_FIELD_VALUE;
    msg.header.version    = dcgm_core_msg_unwatch_field_value_version;

    msg.uf.fieldId       = fieldId;
    msg.uf.clearCache    = clearCache;
    msg.uf.entityGroupId = DCGM_FE_GPU;
    msg.uf.gpuId         = gpuId;
    if (fieldMeta->scope == DCGM_FS_GLOBAL)
        msg.uf.entityGroupId = DCGM_FE_NONE;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.uf.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t helperPolicyGet(dcgmHandle_t dcgmHandle,
                             dcgmGpuGrp_t groupId,
                             int count,
                             dcgmPolicy_t dcgmPolicy[],
                             dcgmStatus_t /* dcgmStatusList */)
{
    dcgm_policy_msg_get_policies_t msg;
    dcgmReturn_t dcgmReturn;
    int i;

    if ((NULL == dcgmPolicy) || (count <= 0))
    {
        log_error("Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    /* Note: dcgmStatusList has always been ignored by this request. Continuing this tradition on
             as I refactor this for modularity */

    /* Verify requested version in the list of output parameters */
    for (i = 0; i < count; i++)
    {
        if (dcgmPolicy[i].version != dcgmPolicy_version)
        {
            log_error("Version mismatch at index {}", i);
            return DCGM_ST_VER_MISMATCH;
        }
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdPolicy;
    msg.header.subCommand = DCGM_POLICY_SR_GET_POLICIES;
    msg.header.version    = dcgm_policy_msg_get_policies_version;
    msg.groupId           = groupId;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (dcgmReturn == DCGM_ST_OK)
    {
        if (msg.numPolicies > count)
            dcgmReturn = DCGM_ST_INSUFFICIENT_SIZE; /* Tell the user we only copied "count" entries */

        msg.numPolicies = std::min(count, msg.numPolicies);

        memcpy(dcgmPolicy, msg.policies, msg.numPolicies * sizeof(msg.policies[0]));
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperPolicySet(dcgmHandle_t dcgmHandle,
                             dcgmGpuGrp_t groupId,
                             dcgmPolicy_t *dcgmPolicy,
                             dcgmStatus_t /* dcgmStatusList */)
{
    dcgm_policy_msg_set_policy_t msg;
    dcgmReturn_t dcgmReturn;

    if (NULL == dcgmPolicy)
        return DCGM_ST_BADPARAM;

    /* Note: dcgmStatusList has always been ignored by this request. Continuing this tradition on
             as I refactor this for modularity */

    if (dcgmPolicy->version != dcgmPolicy_version)
    {
        log_error("Version mismatch.");
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdPolicy;
    msg.header.subCommand = DCGM_POLICY_SR_SET_POLICY;
    msg.header.version    = dcgm_policy_msg_set_policy_version;
    msg.groupId           = groupId;
    memcpy(&msg.policy, dcgmPolicy, sizeof(msg.policy));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperPolicyRegister(dcgmHandle_t dcgmHandle,
                                  dcgmGpuGrp_t groupId,
                                  dcgmPolicyCondition_t condition,
                                  fpRecvUpdates callback,
                                  uint64_t userData)
{
    dcgmReturn_t dcgmReturn;
    dcgm_policy_msg_register_t msg;

    /* Make an ansync object. We're going to pass ownership off, so we won't have to free it */
    std::unique_ptr<DcgmPolicyRequest> policyRequest
        = std::make_unique<DcgmPolicyRequest>(callback, userData, g_dcgmPolicyCbMutex);

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdPolicy;
    msg.header.subCommand = DCGM_POLICY_SR_REGISTER;
    msg.header.version    = dcgm_policy_msg_register_version;
    msg.groupId           = groupId;
    msg.condition         = condition;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg), std::move(policyRequest));
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t helperPolicyUnregister(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition)
{
    dcgmReturn_t dcgmReturn;
    dcgm_policy_msg_unregister_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdPolicy;
    msg.header.subCommand = DCGM_POLICY_SR_UNREGISTER;
    msg.header.version    = dcgm_policy_msg_unregister_version;
    msg.groupId           = groupId;
    msg.condition         = condition;

    /* Prevent users from unregistering callbacks while we are in the process of handling them. */
    if (g_dcgmPolicyCbMutex.try_lock())
    {
        // coverity[overrun-buffer-arg]
        dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
        g_dcgmPolicyCbMutex.unlock();
    }
    else
    {
        dcgmReturn = DCGM_ST_IN_USE;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiSetEntityNvLinkLinkState(dcgmHandle_t dcgmHandle, dcgmSetNvLinkLinkState_v1 *linkState)
{
    if (!linkState)
        return DCGM_ST_BADPARAM;
    if (linkState->version != dcgmSetNvLinkLinkState_version1)
        return DCGM_ST_VER_MISMATCH;

    dcgmReturn_t dcgmReturn;
    dcgm_core_msg_set_entity_nvlink_state_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_SET_ENTITY_LINK_STATE;
    msg.header.version    = dcgm_core_msg_set_entity_nvlink_state_version;

    memcpy(&msg.state, linkState, sizeof(msg.state));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperGetFieldValuesSince(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              long long sinceTimestamp,
                                              unsigned short *fieldIds,
                                              int numFieldIds,
                                              long long *nextSinceTimestamp,
                                              dcgmFieldValueEnumeration_f enumCB,
                                              void *userData)
{
    dcgmReturn_t dcgmSt, retDcgmSt;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo = std::make_unique<dcgmGroupInfo_t>();
    memset(groupInfo.get(), 0, sizeof(*groupInfo));
    unsigned int i;
    int j;
    unsigned int gpuId, fieldId;
    int valuesAtATime
        = SAMPLES_BUFFER_SIZE_V2 / DCGM_BUFFERED_FV1_MIN_ENTRY_SIZE; /* How many values should we fetch at a time */
    int retNumFieldValues;
    int callbackSt              = 0;
    long long endQueryTimestamp = 0;
    DcgmFvBuffer fvBuffer(0);

    retDcgmSt = DCGM_ST_OK;

    if (!fieldIds || !enumCB || !nextSinceTimestamp || numFieldIds < 1)
    {
        log_error("Bad param to helperGetFieldValuesSince");
        return DCGM_ST_BADPARAM;
    }

    log_debug("helperGetFieldValuesSince groupId {}, sinceTs {}, numFieldIds {}, userData {}",
              (void *)groupId,
              sinceTimestamp,
              numFieldIds,
              (void *)userData);

    *nextSinceTimestamp = sinceTimestamp;

    groupInfo->version = dcgmGroupInfo_version;

    /* Convert groupId to list of GPUs. Note that this is an extra round trip to the server
     * in the remote case, but it keeps the code much simpler */
    dcgmSt = helperGroupGetInfo(pDcgmHandle, groupId, groupInfo.get(), &endQueryTimestamp);
    if (dcgmSt != DCGM_ST_OK)
    {
        log_error("helperGroupGetInfo groupId {} returned {}", (void *)groupId, (int)dcgmSt);
        return dcgmSt;
    }

    log_debug("Got group {} with {} entities", groupInfo->groupName, groupInfo->count);

    /* Pre-check the group for non-GPU/non-global entities */
    for (i = 0; i < groupInfo->count; i++)
    {
        if (groupInfo->entityList[i].entityGroupId != DCGM_FE_GPU
            && groupInfo->entityList[i].entityGroupId != DCGM_FE_NONE)
        {
            log_error("helperGetFieldValuesSince called on groupId {} with non-GPU eg {}, eid {}.",
                      (void *)groupId,
                      groupInfo->entityList[i].entityGroupId,
                      groupInfo->entityList[i].entityId);
            return DCGM_ST_NOT_SUPPORTED;
        }
    }

    /* Fetch valuesAtATime values for each GPU for each field since sinceTimestamp.
     * Make valuesAtATime large enough to offset the fact that this is a round trip
     * to the server for each combo of gpuId, fieldId, and valuesAtATime values
     */

    dcgmFieldValue_v1 fv1;

    for (i = 0; i < groupInfo->count; i++)
    {
        gpuId = groupInfo->entityList[i].entityId;

        for (j = 0; j < numFieldIds; j++)
        {
            fieldId = fieldIds[j];

            fvBuffer.Clear();

            retNumFieldValues = valuesAtATime;
            dcgmSt            = helperGetMultipleValuesForFieldFvBuffer(pDcgmHandle,
                                                             DCGM_FE_GPU,
                                                             gpuId,
                                                             fieldId,
                                                             &retNumFieldValues,
                                                             sinceTimestamp,
                                                             endQueryTimestamp,
                                                             DCGM_ORDER_ASCENDING,
                                                             &fvBuffer);
            if (dcgmSt == DCGM_ST_NO_DATA)
            {
                log_debug("DCGM_ST_NO_DATA for gpuId {}, fieldId {}, sinceTs {}", gpuId, fieldId, sinceTimestamp);
                continue;
            }
            else if (dcgmSt != DCGM_ST_OK)
            {
                log_error(
                    "Got st {} from helperGetMultipleValuesForField gpuId {}, fieldId {}", (int)dcgmSt, gpuId, fieldId);
                return dcgmSt;
            }

            log_debug("Got {} values for gpuId {}, fieldId {}", retNumFieldValues, gpuId, fieldId);

            dcgmBufferedFvCursor_t cursor = 0;

            /* Loop over each returned value and call our callback for it */
            for (dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
            {
                fvBuffer.ConvertBufferedFvToFv1(fv, &fv1);
                callbackSt = enumCB(gpuId, &fv1, 1, userData);
                if (callbackSt != 0)
                {
                    log_debug("User requested callback exit");
                    /* Leaving status as OK. User requested the exit */
                    return DCGM_ST_OK;
                }
            }
        }
    }

    /* Success. We can advance the caller's next query timestamp */
    *nextSinceTimestamp = endQueryTimestamp + 1;
    return retDcgmSt;
}

/*****************************************************************************/
static dcgmReturn_t helperGetValuesSince(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgmFieldGrp_t fieldGroupId,
                                         long long sinceTimestamp,
                                         long long *nextSinceTimestamp,
                                         dcgmFieldValueEnumeration_f enumCB,
                                         dcgmFieldValueEntityEnumeration_f enumCBv2,
                                         void *userData)
{
    dcgmReturn_t dcgmSt, retDcgmSt;
    std::unique_ptr<dcgmGroupInfo_t> groupInfo = std::make_unique<dcgmGroupInfo_t>();
    memset(groupInfo.get(), 0, sizeof(*groupInfo));
    dcgmFieldGroupInfo_t fieldGroupInfo = {};
    unsigned int i;
    int j;
    unsigned int fieldId;
    int valuesAtATime
        = SAMPLES_BUFFER_SIZE_V2 / DCGM_BUFFERED_FV1_MIN_ENTRY_SIZE; /* How many values should we fetch at a time */
    int retNumFieldValues;
    int callbackSt              = 0;
    long long endQueryTimestamp = 0;
    DcgmFvBuffer fvBuffer(0);

    retDcgmSt = DCGM_ST_OK;

    if ((!enumCB && !enumCBv2) || !nextSinceTimestamp)
    {
        log_error("Bad param to helperGetValuesSince");
        return DCGM_ST_BADPARAM;
    }

    *nextSinceTimestamp = sinceTimestamp;

    fieldGroupInfo.version      = dcgmFieldGroupInfo_version;
    fieldGroupInfo.fieldGroupId = fieldGroupId;
    dcgmSt                      = dcgmFieldGroupGetInfo(pDcgmHandle, &fieldGroupInfo);
    if (dcgmSt != DCGM_ST_OK)
    {
        log_error(
            "Got dcgmSt {} from dcgmFieldGroupGetInfo() fieldGroupId {}", (dcgmReturn_t)dcgmSt, (void *)fieldGroupId);
        return dcgmSt;
    }

    log_debug("fieldGroup {}, name {}, numFieldIds {}",
              (void *)fieldGroupId,
              fieldGroupInfo.fieldGroupName,
              fieldGroupInfo.numFieldIds);

    /* Convert groupId to list of GPUs. Note that this is an extra round trip to the server
     * in the remote case, but it keeps the code much simpler */
    groupInfo->version = dcgmGroupInfo_version;
    dcgmSt             = helperGroupGetInfo(pDcgmHandle, groupId, groupInfo.get(), &endQueryTimestamp);
    if (dcgmSt != DCGM_ST_OK)
    {
        log_error("helperGroupGetInfo groupId {} returned {}", (void *)groupId, (int)dcgmSt);
        return dcgmSt;
    }

    log_debug(
        "Got group {} with {} GPUs, endQueryTimestamp {}", groupInfo->groupName, groupInfo->count, endQueryTimestamp);

    /* Pre-check the group for non-GPU/non-global entities */
    if (!enumCBv2)
    {
        for (i = 0; i < groupInfo->count; i++)
        {
            if (groupInfo->entityList[i].entityGroupId != DCGM_FE_GPU
                && groupInfo->entityList[i].entityGroupId != DCGM_FE_NONE)
            {
                log_error("helperGetValuesSince called on groupId {} with non-GPU eg {}, eid {}.",
                          (void *)groupId,
                          groupInfo->entityList[i].entityGroupId,
                          groupInfo->entityList[i].entityId);
                return DCGM_ST_NOT_SUPPORTED;
            }
        }
    }

    /* Fetch valuesAtATime values for each GPU for each field since sinceTimestamp.
     * Make valuesAtATime large enough to offset the fact that this is a round trip
     * to the server for each combo of gpuId, fieldId, and valuesAtATime values
     */

    dcgmFieldValue_v1 fv1;

    for (i = 0; i < groupInfo->count; i++)
    {
        for (j = 0; j < (int)fieldGroupInfo.numFieldIds; j++)
        {
            fieldId = fieldGroupInfo.fieldIds[j];

            fvBuffer.Clear();

            /* Using endQueryTimestamp as endTime here so we don't get values that update after the
               nextSinceTimestamp we're returning to the client */
            retNumFieldValues = valuesAtATime;
            dcgmSt            = helperGetMultipleValuesForFieldFvBuffer(pDcgmHandle,
                                                             groupInfo->entityList[i].entityGroupId,
                                                             groupInfo->entityList[i].entityId,
                                                             fieldId,
                                                             &retNumFieldValues,
                                                             sinceTimestamp,
                                                             endQueryTimestamp,
                                                             DCGM_ORDER_ASCENDING,
                                                             &fvBuffer);
            if (dcgmSt == DCGM_ST_NO_DATA)
            {
                log_debug("DCGM_ST_NO_DATA for eg {}, eid {}, fieldId {}, sinceTs {}",
                          groupInfo->entityList[i].entityGroupId,
                          groupInfo->entityList[i].entityId,
                          fieldId,
                          sinceTimestamp);
                continue;
            }
            else if (dcgmSt != DCGM_ST_OK)
            {
                log_error("Got st {} from helperGetMultipleValuesForField eg {}, eid {}, fieldId {}",
                          (int)dcgmSt,
                          groupInfo->entityList[i].entityGroupId,
                          groupInfo->entityList[i].entityId,
                          fieldId);
                return dcgmSt;
            }

            log_debug("Got {} values for eg {}, eid {}, fieldId {}",
                      retNumFieldValues,
                      groupInfo->entityList[i].entityGroupId,
                      groupInfo->entityList[i].entityId,
                      fieldId);

            dcgmBufferedFvCursor_t cursor = 0;

            /* Loop over each returned value and call our callback for it */
            for (dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
            {
                fvBuffer.ConvertBufferedFvToFv1(fv, &fv1);
                if (enumCB)
                {
                    callbackSt = enumCB(groupInfo->entityList[i].entityId, &fv1, 1, userData);
                    if (callbackSt != 0)
                    {
                        log_debug("User requested callback exit");
                        /* Leaving status as OK. User requested the exit */
                        return DCGM_ST_OK;
                    }
                }
                if (enumCBv2)
                {
                    callbackSt = enumCBv2(
                        groupInfo->entityList[i].entityGroupId, groupInfo->entityList[i].entityId, &fv1, 1, userData);
                    if (callbackSt != 0)
                    {
                        log_debug("User requested callback exit");
                        /* Leaving status as OK. User requested the exit */
                        return DCGM_ST_OK;
                    }
                }
            }
        }
    }

    /* Success. We can advance the caller's next query timestamp */
    *nextSinceTimestamp = endQueryTimestamp + 1;
    log_debug("nextSinceTimestamp advanced to {}", *nextSinceTimestamp);

    return retDcgmSt;
}

/*****************************************************************************/
static dcgmReturn_t helperGetLatestValues(dcgmHandle_t pDcgmHandle,
                                          dcgmGpuGrp_t groupId,
                                          dcgmFieldGrp_t fieldGroupId,
                                          dcgmFieldValueEnumeration_f enumCB,
                                          dcgmFieldValueEntityEnumeration_f enumCBv2,
                                          void *userData)
{
    dcgmReturn_t dcgmSt, retDcgmSt;
    int callbackSt = 0;
    retDcgmSt      = DCGM_ST_OK;

    /* At least one enumCB must be provided */
    if (!enumCB && !enumCBv2)
    {
        DCGM_LOG_ERROR << "Bad param to helperLatestValues";
        return DCGM_ST_BADPARAM;
    }

    DcgmFvBuffer fvBuffer(0);

    dcgmSt = helperGetLatestValuesForFields(pDcgmHandle, groupId, 0, 0, fieldGroupId, 0, 0, &fvBuffer, 0);
    if (dcgmSt != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Return code " << dcgmSt;
        return dcgmSt;
    }

    dcgmBufferedFv_t *fv;
    dcgmFieldValue_v1 fieldValueV1; /* Converted from fv */
    dcgmBufferedFvCursor_t cursor = 0;

    /* Loop over each returned value and call our callback for it */
    for (fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        DCGM_LOG_DEBUG << "Got value for eg " << fv->entityGroupId << ", eid " << fv->entityId << ", fieldId "
                       << fv->fieldId;

        /* Get a v1 version to pass to our callbacks */
        fvBuffer.ConvertBufferedFvToFv1(fv, &fieldValueV1);

        if (enumCB)
        {
            if (fv->entityGroupId != DCGM_FE_GPU && fv->entityGroupId != DCGM_FE_NONE)
            {
                DCGM_LOG_DEBUG << "helperGetLatestValues called on groupId " << (void *)groupId << " with non-GPU eg "
                               << fv->entityGroupId << ", eid " << fv->entityId;
                continue;
            }
            callbackSt = enumCB(fv->entityId, &fieldValueV1, 1, userData);
            if (callbackSt != 0)
            {
                DCGM_LOG_DEBUG << "User requested callback exit";
                /* Leaving status as OK. User requested the exit */
                break;
            }
        }
        if (enumCBv2)
        {
            callbackSt
                = enumCBv2((dcgm_field_entity_group_t)fv->entityGroupId, fv->entityId, &fieldValueV1, 1, userData);
            if (callbackSt != 0)
            {
                DCGM_LOG_DEBUG << "User requested callback exit";
                /* Leaving status as OK. User requested the exit */
                break;
            }
        }
    }

    return retDcgmSt;
}

dcgmReturn_t tsapiWatchFields(dcgmHandle_t pDcgmHandle,
                              dcgmGpuGrp_t groupId,
                              dcgmFieldGrp_t fieldGroupId,
                              long long updateFreq,
                              double maxKeepAge,
                              int maxKeepSamples)
{
    if (!groupId)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_watch_fields_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_WATCH_FIELDS;
    msg.header.version    = dcgm_core_msg_watch_fields_version;

    msg.watchInfo.groupId        = groupId;
    msg.watchInfo.fieldGroupId   = fieldGroupId;
    msg.watchInfo.updateFreq     = updateFreq;
    msg.watchInfo.maxKeepAge     = maxKeepAge;
    msg.watchInfo.maxKeepSamples = maxKeepSamples;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.watchInfo.cmdRet;
}

dcgmReturn_t tsapiUnwatchFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId)
{
    if (!groupId)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_watch_fields_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_UNWATCH_FIELDS;
    msg.header.version    = dcgm_core_msg_watch_fields_version;

    msg.watchInfo.groupId      = groupId;
    msg.watchInfo.fieldGroupId = fieldGroupId;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.watchInfo.cmdRet;
}

dcgmReturn_t tsapiFieldGroupCreate(dcgmHandle_t pDcgmHandle,
                                   int numFieldIds,
                                   const unsigned short *fieldIds,
                                   const char *fieldGroupName,
                                   dcgmFieldGrp_t *dcgmFieldGroupId)
{
    if (numFieldIds < 1 || numFieldIds >= DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP || !fieldGroupName
        || strlen(fieldGroupName) >= DCGM_MAX_STR_LENGTH || !fieldGroupName[0] || !fieldIds || !dcgmFieldGroupId)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_fieldgroup_op_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_FIELDGROUP_CREATE;
    msg.header.version    = dcgm_core_msg_fieldgroup_op_version;

    msg.info.fg.version = dcgmFieldGroupInfo_version;
    SafeCopyTo(msg.info.fg.fieldGroupName, fieldGroupName);
    msg.info.fg.numFieldIds = numFieldIds;
    memcpy(msg.info.fg.fieldIds, fieldIds, sizeof(fieldIds[0]) * numFieldIds);

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Return code " << ret;
        return ret;
    }

    *dcgmFieldGroupId = msg.info.fg.fieldGroupId;
    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiFieldGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId)
{
    DCGM_LOG_DEBUG << "dcgmFieldGroupDestroy fieldGroupId " << (void *)dcgmFieldGroupId;

    dcgm_core_msg_fieldgroup_op_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_FIELDGROUP_DESTROY;
    msg.header.version    = dcgm_core_msg_fieldgroup_op_version;

    msg.info.fg.version      = dcgmFieldGroupInfo_version;
    msg.info.fg.fieldGroupId = dcgmFieldGroupId;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    DCGM_LOG_DEBUG << "tsapiFieldGroupDestroy ret " << ret << ", fieldGroupId " << (void *)msg.info.fg.fieldGroupId;

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiFieldGroupGetInfo(dcgmHandle_t pDcgmHandle, dcgmFieldGroupInfo_t *fieldGroupInfo)
{
    if (!fieldGroupInfo)
        return DCGM_ST_BADPARAM;

    /* Valid version can't be 0 */
    if (!fieldGroupInfo->version)
        return DCGM_ST_VER_MISMATCH;

    dcgm_core_msg_fieldgroup_op_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_FIELDGROUP_GET_INFO;
    msg.header.version    = dcgm_core_msg_fieldgroup_op_version;

    memcpy(&msg.info.fg, fieldGroupInfo, sizeof(msg.info.fg));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    DCGM_LOG_DEBUG << "tsapiFieldGroupGetInfo got ret " << ret;

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    memcpy(fieldGroupInfo, &msg.info.fg, sizeof(msg.info.fg));

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiFieldGroupGetAll(dcgmHandle_t pDcgmHandle, dcgmAllFieldGroup_t *allGroupInfo)
{
    if (!allGroupInfo)
        return DCGM_ST_BADPARAM;

    /* Valid version can't be 0 or just any random number  */
    if (allGroupInfo->version != dcgmAllFieldGroup_version)
    {
        DCGM_LOG_ERROR << "Version Mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_fieldgroup_get_all_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_FIELDGROUP_GET_ALL;
    msg.header.version    = dcgm_core_msg_fieldgroup_get_all_version;

    memcpy(&msg.info.fg, allGroupInfo, sizeof(msg.info.fg));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    DCGM_LOG_DEBUG << "tsapiFieldGroupGetAll got ret " << ret;

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    memcpy(allGroupInfo, &msg.info.fg, sizeof(msg.info.fg));

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t helperHealthSet(dcgmHandle_t dcgmHandle, dcgmHealthSetParams_v2 *healthSetParams)
{
    dcgm_health_msg_set_systems_t msg;
    dcgmReturn_t dcgmReturn;

    if (healthSetParams == nullptr)
    {
        DCGM_LOG_ERROR << "Null healthSetParams";
        return DCGM_ST_BADPARAM;
    }
    if (healthSetParams->version != dcgmHealthSetParams_version2)
    {
        DCGM_LOG_ERROR << "Version mismatch " << std::hex << healthSetParams->version
                       << " != " << dcgmHealthSetParams_version2;
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_SET_SYSTEMS_V2;
    msg.header.version    = dcgm_health_msg_set_systems_version;
    msg.healthSet         = *healthSetParams;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    return dcgmReturn;
}

dcgmReturn_t helperHealthGet(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems)
{
    dcgm_health_msg_get_systems_t msg;
    dcgmReturn_t dcgmReturn;

    if (!systems)
    {
        DCGM_LOG_ERROR << "bad systems " << (void *)systems;
        return DCGM_ST_BADPARAM;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_GET_SYSTEMS;
    msg.header.version    = dcgm_health_msg_get_systems_version;
    msg.groupId           = groupId;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    *systems = msg.systems;
    return dcgmReturn;
}

template <typename MsgType>
    requires std::is_same_v<MsgType, dcgm_diag_msg_run_v12> || std::is_same_v<MsgType, dcgm_diag_msg_run_v11>
             || std::is_same_v<MsgType, dcgm_diag_msg_run_v10> || std::is_same_v<MsgType, dcgm_diag_msg_run_v9>
             || std::is_same_v<MsgType, dcgm_diag_msg_run_v8> || std::is_same_v<MsgType, dcgm_diag_msg_run_v7>
             || std::is_same_v<MsgType, dcgm_diag_msg_run_v6> || std::is_same_v<MsgType, dcgm_diag_msg_run_v5>
static inline dcgm_module_command_header_t *helperInitDiagMsgRun(MsgType *msg,
                                                                 unsigned int const msgVersion,
                                                                 dcgmPolicyAction_t action)
{
    memset(msg, 0, sizeof(*msg));
    msg->header.length  = sizeof(*msg);
    msg->header.version = msgVersion;
    msg->action         = action;

    switch (msg->header.version)
    {
        case dcgm_diag_msg_run_version12:
            msg->diagResponse.version = dcgmDiagResponse_version12;
            break;
        case dcgm_diag_msg_run_version11:
        case dcgm_diag_msg_run_version10:
            msg->diagResponse.version = dcgmDiagResponse_version11;
            break;
        case dcgm_diag_msg_run_version9:
        case dcgm_diag_msg_run_version8:
            msg->diagResponse.version = dcgmDiagResponse_version10;
            break;
        case dcgm_diag_msg_run_version7:
            msg->diagResponse.version = dcgmDiagResponse_version9;
            break;
        case dcgm_diag_msg_run_version6:
            msg->diagResponse.version = dcgmDiagResponse_version8;
            break;
        case dcgm_diag_msg_run_version5:
            msg->diagResponse.version = dcgmDiagResponse_version7;
            break;
        default:
            log_error("Unexpected run diag version: [{:x}], treating as [{:x}].",
                      msg->header.version,
                      dcgm_diag_msg_run_version12);
            msg->diagResponse.version = dcgmDiagResponse_version12;
            break;
    }

    return &(msg->header);
}

dcgmReturn_t helperActionManager(dcgmHandle_t dcgmHandle,
                                 dcgmRunDiag_v10 *drd,
                                 dcgmPolicyAction_t action,
                                 dcgmDiagResponse_v12 *response)
{
    if (!drd || !response)
    {
        log_error("drd {} or response {} was NULL.", (void *)drd, (void *)response);
        return DCGM_ST_BADPARAM;
    }

    auto msg12 = std::make_unique<dcgm_diag_msg_run_v12>();
    auto msg11 = std::make_unique<dcgm_diag_msg_run_v11>();
    auto msg10 = std::make_unique<dcgm_diag_msg_run_v10>();
    auto msg9  = std::make_unique<dcgm_diag_msg_run_v9>();
    auto msg8  = std::make_unique<dcgm_diag_msg_run_v8>();
    auto msg7  = std::make_unique<dcgm_diag_msg_run_v7>();
    auto msg6  = std::make_unique<dcgm_diag_msg_run_v6>();
    auto msg5  = std::make_unique<dcgm_diag_msg_run_v5>();

    switch (drd->version)
    {
        case dcgmRunDiag_version10:
        case dcgmRunDiag_version9:
        case dcgmRunDiag_version8:
        case dcgmRunDiag_version7:
            break;
        default:
            // unknown drd version
            log_error("dcgmRunDiag version mismatch {:X} != {:X} and isn't in accepted list",
                      drd->version,
                      dcgmRunDiag_version10);
            return DCGM_ST_VER_MISMATCH;
    }

    dcgm_module_command_header_t *header;

    switch (response->version)
    {
        case dcgmDiagResponse_version12:
            if (drd->version == dcgmRunDiag_version10)
            {
                header = helperInitDiagMsgRun(msg12.get(), dcgm_diag_msg_run_version12, action);
                memcpy(&msg12->runDiag, drd, sizeof(dcgmRunDiag_v10));
            }
            else if (drd->version == dcgmRunDiag_version9)
            {
                header = helperInitDiagMsgRun(msg10.get(), dcgm_diag_msg_run_version10, action);
                memcpy(&msg10->runDiag, drd, sizeof(dcgmRunDiag_v9));
            }
            else
            {
                log_error("Unexpected run diag version: [{:x}], expected versions: [{:x}, {:x}].",
                          drd->version,
                          dcgmRunDiag_version10,
                          dcgmRunDiag_version9);
                return DCGM_ST_VER_MISMATCH;
            }
            break;
        case dcgmDiagResponse_version11:
            if (drd->version == dcgmRunDiag_version10)
            {
                header = helperInitDiagMsgRun(msg11.get(), dcgm_diag_msg_run_version11, action);
                memcpy(&msg11->runDiag, drd, sizeof(dcgmRunDiag_v10));
            }
            else if (drd->version == dcgmRunDiag_version9)
            {
                header = helperInitDiagMsgRun(msg10.get(), dcgm_diag_msg_run_version10, action);
                memcpy(&msg10->runDiag, drd, sizeof(dcgmRunDiag_v9));
            }
            else
            {
                log_error("Unexpected run diag version: [{:x}], expected versions: [{:x}, {:x}].",
                          drd->version,
                          dcgmRunDiag_version10,
                          dcgmRunDiag_version9);
                return DCGM_ST_VER_MISMATCH;
            }
            break;

        case dcgmDiagResponse_version10:
            if (drd->version == dcgmRunDiag_version8)
            {
                header = helperInitDiagMsgRun(msg9.get(), dcgm_diag_msg_run_version9, action);
                memcpy(&msg9->runDiag, drd, sizeof(dcgmRunDiag_v8));
            }
            else if (drd->version == dcgmRunDiag_version7)
            {
                header = helperInitDiagMsgRun(msg8.get(), dcgm_diag_msg_run_version8, action);
                memcpy(&msg8->runDiag, drd, sizeof(dcgmRunDiag_v7));
            }
            else
            {
                log_error("Unexpected run diag version: [{:x}], expected versions: [{:x}, {:x}].",
                          drd->version,
                          dcgmRunDiag_version8,
                          dcgmRunDiag_version7);
                return DCGM_ST_VER_MISMATCH;
            }
            break;
        case dcgmDiagResponse_version9:
            header = helperInitDiagMsgRun(msg7.get(), dcgm_diag_msg_run_version7, action);
            if (drd->version != dcgmRunDiag_version7)
            {
                log_error("Unexpected run diag version: [{:x}], expected version: [{:x}].",
                          drd->version,
                          dcgmRunDiag_version7);
                return DCGM_ST_VER_MISMATCH;
            }
            memcpy(&msg7->runDiag, drd, sizeof(dcgmRunDiag_v7));
            break;

        case dcgmDiagResponse_version8:
            header = helperInitDiagMsgRun(msg6.get(), dcgm_diag_msg_run_version6, action);
            if (drd->version != dcgmRunDiag_version7)
            {
                log_error("Unexpected run diag version: [{:x}], expected version: [{:x}].",
                          drd->version,
                          dcgmRunDiag_version7);
                return DCGM_ST_VER_MISMATCH;
            }
            memcpy(&msg6->runDiag, drd, sizeof(dcgmRunDiag_v7));
            break;

        case dcgmDiagResponse_version7:
            header = helperInitDiagMsgRun(msg5.get(), dcgm_diag_msg_run_version5, action);
            if (drd->version != dcgmRunDiag_version7)
            {
                log_error("Unexpected run diag version: [{:x}], expected version: [{:x}].",
                          drd->version,
                          dcgmRunDiag_version7);
                return DCGM_ST_VER_MISMATCH;
            }
            memcpy(&msg5->runDiag, drd, sizeof(dcgmRunDiag_v7));
            break;

        default:
            log_error("response->version 0x{:x} doesn't match a valid version", response->version);
            return DCGM_ST_VER_MISMATCH;
    }

    header->moduleId   = DcgmModuleIdDiag;
    header->subCommand = DCGM_DIAG_SR_RUN;

    // Compatibility requirement
    static_assert(sizeof(msg11->diagResponse) <= sizeof(msg12->diagResponse));
    static_assert(sizeof(msg10->diagResponse) <= sizeof(msg11->diagResponse));
    static_assert(sizeof(msg9->diagResponse) <= sizeof(msg10->diagResponse));
    static_assert(sizeof(msg8->diagResponse) <= sizeof(msg9->diagResponse));
    static_assert(sizeof(msg7->diagResponse) <= sizeof(msg8->diagResponse));
    static_assert(sizeof(msg6->diagResponse) <= sizeof(msg7->diagResponse));
    static_assert(sizeof(msg5->diagResponse) <= sizeof(msg6->diagResponse));

    unsigned int timeoutSeconds              = 0;
    constexpr unsigned int secToMsMultiplier = 1000;

    if (drd->version == dcgmRunDiag_version10)
    {
        timeoutSeconds = drd->timeoutSeconds;
    }
    else if (drd->version == dcgmRunDiag_version9)
    {
        timeoutSeconds = (reinterpret_cast<dcgmRunDiag_v9 *>(drd))->timeoutSeconds;
    }
    else if (drd->version == dcgmRunDiag_version8)
    {
        timeoutSeconds = (reinterpret_cast<dcgmRunDiag_v8 *>(drd))->timeoutSeconds;
    }
    else if (drd->version == dcgmRunDiag_version7)
    {
        timeoutSeconds = (reinterpret_cast<dcgmRunDiag_v7 *>(drd))->timeoutSeconds;
    }

    // coverity[overrun-buffer-arg]
    dcgmReturn_t dcgmReturn = dcgmModuleSendBlockingFixedRequest(
        dcgmHandle, header, sizeof(*msg12), nullptr, timeoutSeconds * secToMsMultiplier);

    switch (response->version)
    {
        case dcgmDiagResponse_version12:
            if (drd->version == dcgmRunDiag_version10)
            {
                memcpy(response, &msg12->diagResponse, sizeof(msg12->diagResponse));
            }
            else if (drd->version == dcgmRunDiag_version9)
            {
                memcpy(response, &msg10->diagResponse, sizeof(msg10->diagResponse));
            }
            break;
        case dcgmDiagResponse_version11:
            if (drd->version == dcgmRunDiag_version10)
            {
                memcpy(response, &msg11->diagResponse, sizeof(msg11->diagResponse));
            }
            else if (drd->version == dcgmRunDiag_version9)
            {
                memcpy(response, &msg10->diagResponse, sizeof(msg10->diagResponse));
            }
            break;

        case dcgmDiagResponse_version10:
            if (drd->version == dcgmRunDiag_version8)
            {
                memcpy(response, &msg9->diagResponse, sizeof(msg9->diagResponse));
            }
            else if (drd->version == dcgmRunDiag_version7)
            {
                memcpy(response, &msg8->diagResponse, sizeof(msg8->diagResponse));
            }
            break;

        case dcgmDiagResponse_version9:
            memcpy(response, &msg7->diagResponse, sizeof(msg7->diagResponse));
            break;

        case dcgmDiagResponse_version8:
            memcpy(response, &msg6->diagResponse, sizeof(msg6->diagResponse));
            break;

        case dcgmDiagResponse_version7:
            memcpy(response, &msg5->diagResponse, sizeof(msg5->diagResponse));
            break;

        default:
            DCGM_LOG_ERROR << "Unexpected response->version 0x" << std::hex << response->version;
            return DCGM_ST_GENERIC_ERROR;
    }

    return dcgmReturn;
}

dcgmReturn_t helperStopDiag(dcgmHandle_t dcgmHandle)
{
    dcgm_diag_msg_stop_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdDiag;
    msg.header.subCommand = DCGM_DIAG_SR_STOP;
    msg.header.version    = dcgm_diag_msg_stop_version;

    static const int SIXTY_MINUTES_IN_MS = 3600000;
    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg), nullptr, SIXTY_MINUTES_IN_MS);

    return dcgmReturn;
}

uid_t helperGetEffectiveUid()
{
    uid_t eUid = geteuid();
    return eUid;
}

dcgmReturn_t helperRunMnDiag(dcgmHandle_t dcgmHandle, dcgmRunMnDiag_v1 const *drmnd, dcgmMnDiagResponse_v1 *response)
{
    std::unique_ptr<dcgm_mndiag_msg_run_v1> msg = std::make_unique<dcgm_mndiag_msg_run_v1>();
    dcgmReturn_t dcgmReturn { DCGM_ST_OK };

    memset(msg.get(), 0, sizeof(*msg));
    msg->header.length     = sizeof(*msg);
    msg->header.version    = dcgm_mndiag_msg_run_version1;
    msg->header.moduleId   = DcgmModuleIdMnDiag;
    msg->header.subCommand = DCGM_MNDIAG_SR_RUN;

    switch (response->version)
    {
        case dcgmMnDiagResponse_version1:
            memcpy(&msg->params, drmnd, sizeof(dcgmRunMnDiag_v1));
            memcpy(&msg->response, response, sizeof(dcgmMnDiagResponse_v1));
            break;
        default:
            log_error("Unexpected response->version 0x{:X}", response->version);
            return DCGM_ST_GENERIC_ERROR;
    }

    // Populate the effective uid of the caller
    msg->effectiveUid = helperGetEffectiveUid();

    constexpr unsigned int TWENTY_FOUR_HOURS_IN_MS = 86400000;
    // coverity[overrun-buffer-arg]
    dcgmReturn
        = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg->header, sizeof(*msg), nullptr, TWENTY_FOUR_HOURS_IN_MS);

    switch (response->version)
    {
        case dcgmMnDiagResponse_version1:
            memcpy(response, &msg->response, sizeof(dcgmMnDiagResponse_v1));
            break;
        default:
            log_error("Unexpected response->version 0x{:X}", response->version);
            return DCGM_ST_GENERIC_ERROR;
    }

    return dcgmReturn;
}

dcgmReturn_t helperStopMnDiag(dcgmHandle_t dcgmHandle)
{
    dcgm_mndiag_msg_stop_t msg {};
    dcgmReturn_t dcgmReturn { DCGM_ST_OK };

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdMnDiag;
    msg.header.subCommand = DCGM_MNDIAG_SR_STOP;
    msg.header.version    = dcgm_mndiag_msg_stop_version;

    static const int SIXTY_MINUTES_IN_MS = 3600000;
    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg), nullptr, SIXTY_MINUTES_IN_MS);

    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperHealthCheckV5(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_v5 *response)
{
    std::unique_ptr<dcgm_health_msg_check_v5> msg = std::make_unique<dcgm_health_msg_check_v5>();
    dcgmReturn_t dcgmReturn;

    memset(msg.get(), 0, sizeof(*msg));
    msg->header.length     = sizeof(*msg);
    msg->header.moduleId   = DcgmModuleIdHealth;
    msg->header.subCommand = DCGM_HEALTH_SR_CHECK_V5;
    msg->header.version    = dcgm_health_msg_check_version5;

    msg->groupId   = groupId;
    msg->startTime = 0;
    msg->endTime   = 0;

    memcpy(&msg->response, response, sizeof(msg->response));
    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg->header, sizeof(*msg));
    memcpy(response, &msg->response, sizeof(msg->response));
    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperGetPidInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo)
{
    if (!pidInfo)
        return DCGM_ST_BADPARAM;

    /* Valid version can't be 0 or just any random number  */
    if (pidInfo->version != dcgmPidInfo_version)
    {
        DCGM_LOG_ERROR << "Version Mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    if (!pidInfo->pid)
    {
        DCGM_LOG_ERROR << "Bad parameter";
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_pid_get_info_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_PID_GET_INFO;
    msg.header.version    = dcgm_core_msg_pid_get_info_version;

    msg.info.groupId = groupId;

    memcpy(&msg.info.pidInfo, pidInfo, sizeof(msg.info.pidInfo));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    memcpy(pidInfo, &msg.info.pidInfo, sizeof(msg.info.pidInfo));

    return (dcgmReturn_t)msg.info.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t helperGetTopologyAffinity(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmAffinity_t *groupAffinity)
{
    dcgmReturn_t ret;

    if (NULL == groupAffinity)
        return DCGM_ST_BADPARAM;

    dcgm_core_msg_get_topology_affinity_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_TOPOLOGY_AFFINITY;
    msg.header.version    = dcgm_core_msg_get_topology_affinity_version;

    msg.affinity.groupId = groupId;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    memcpy(groupAffinity, &msg.affinity.affinity, sizeof(msg.affinity.affinity));

    return (dcgmReturn_t)msg.affinity.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t helperSelectGpusByTopology(dcgmHandle_t pDcgmHandle,
                                        uint64_t inputGpuIds,
                                        uint32_t numGpus,
                                        uint64_t *outputGpuIds,
                                        uint64_t hintFlags)
{
    dcgmReturn_t ret;

    if ((dcgmHandle_t) nullptr == pDcgmHandle || !outputGpuIds)
    {
        DCGM_LOG_ERROR << "bad outputGpuIds " << (void *)outputGpuIds << " or pDcgmHandle " << (void *)pDcgmHandle;
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_select_topology_gpus_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_SELECT_TOPOLOGY_GPUS;
    msg.header.version    = dcgm_core_msg_select_topology_gpus_version;

    msg.sgt.inputGpus  = inputGpuIds;
    msg.sgt.numGpus    = numGpus;
    msg.sgt.flags      = hintFlags;
    msg.sgt.outputGpus = 0;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    *outputGpuIds = msg.sgt.outputGpus;

    return (dcgmReturn_t)msg.sgt.cmdRet;
}

static dcgmReturn_t helperGetFieldSummary(dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request)
{
    if (!request)
        return DCGM_ST_BADPARAM;

    if (request->version != dcgmFieldSummaryRequest_version1)
        return DCGM_ST_VER_MISMATCH;

    dcgm_core_msg_get_field_summary_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_FIELD_SUMMARY;
    msg.header.version    = dcgm_core_msg_get_field_summary_version;

    memcpy(&msg.info.fsr, request, sizeof(msg.info.fsr));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    DCGM_LOG_DEBUG << "helperGetFieldSummary retrieved " << request->response.summaryCount << " summary types. Return "
                   << ret;

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    memcpy(request, &msg.info.fsr, sizeof(msg.info.fsr));

    return (dcgmReturn_t)msg.info.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t helperGetTopologyPci(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmTopology_t *groupTopology)
{
    dcgmReturn_t ret;

    if (NULL == groupTopology)
        return DCGM_ST_BADPARAM;

    dcgm_core_msg_get_topology_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_TOPOLOGY;
    msg.header.version    = dcgm_core_msg_get_topology_version;

    msg.topo.groupId = groupId;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    DCGM_LOG_DEBUG << "helperGetTopologyPci returned " << ret;

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    memcpy(groupTopology, &msg.topo.topology, sizeof(msg.topo.topology));

    return (dcgmReturn_t)msg.topo.cmdRet;
}

/*****************************************************************************
 The entry points for DCGM Host Engine APIs
 *****************************************************************************/


static dcgmReturn_t tsapiEngineRun(unsigned short portNumber, char const *socketPath, unsigned int isConnectionTCP)
{
    if (NULL == DcgmHostEngineHandler::Instance())
    {
        return DCGM_ST_UNINITIALIZED;
    }

    return (dcgmReturn_t)DcgmHostEngineHandler::Instance()->RunServer(portNumber, socketPath, isConnectionTCP);
}

static dcgmReturn_t tsapiEngineGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId)
{
    return cmHelperGroupAddEntity(pDcgmHandle, groupId, DCGM_FE_GPU, gpuId);
}

static dcgmReturn_t tsapiEngineGroupRemoveDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId)
{
    return cmHelperGroupRemoveEntity(pDcgmHandle, groupId, DCGM_FE_GPU, gpuId);
}

static dcgmReturn_t tsapiEngineGroupGetInfo(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmGroupInfo_t *pDcgmGroupInfo)
{
    return helperGroupGetInfo(pDcgmHandle, groupId, pDcgmGroupInfo, nullptr);
}

static dcgmReturn_t tsapiStatusCreate(dcgmStatus_t *pDcgmStatusList)
{
    if ((dcgmStatus_t) nullptr == pDcgmStatusList)
    {
        return DCGM_ST_BADPARAM;
    }

    *pDcgmStatusList = (dcgmStatus_t) new DcgmStatus;
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiStatusDestroy(dcgmStatus_t pDcgmStatusList)
{
    if ((dcgmStatus_t) nullptr == pDcgmStatusList)
    {
        return DCGM_ST_BADPARAM;
    }

    delete (DcgmStatus *)pDcgmStatusList;
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiStatusGetCount(dcgmStatus_t pDcgmStatusList, unsigned int *count)
{
    if (((dcgmStatus_t) nullptr == pDcgmStatusList) || (nullptr == count))
    {
        return DCGM_ST_BADPARAM;
    }

    *count = ((DcgmStatus *)pDcgmStatusList)->GetNumErrors();
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiStatusPopError(dcgmStatus_t pDcgmStatusList, dcgmErrorInfo_t *pDcgmErrorInfo)
{
    if (((dcgmStatus_t) nullptr == pDcgmStatusList) || (nullptr == pDcgmErrorInfo))
    {
        return DCGM_ST_BADPARAM;
    }

    if (((DcgmStatus *)pDcgmStatusList)->IsEmpty())
    {
        return DCGM_ST_NO_DATA;
    }

    (void)((DcgmStatus *)pDcgmStatusList)->Dequeue(pDcgmErrorInfo);

    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiStatusClear(dcgmStatus_t pDcgmStatusList)
{
    if ((dcgmStatus_t) nullptr == pDcgmStatusList)
    {
        return DCGM_ST_BADPARAM;
    }

    (void)((DcgmStatus *)pDcgmStatusList)->RemoveAll();

    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiEngineGetAllDevices(dcgmHandle_t pDcgmHandle,
                                             unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                             int *count)
{
    return cmHelperGetAllDevices(pDcgmHandle, gpuIdList, count, 0);
}

static dcgmReturn_t tsapiEngineGetAllSupportedDevices(dcgmHandle_t pDcgmHandle,
                                                      unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                                      int *count)
{
    return cmHelperGetAllDevices(pDcgmHandle, gpuIdList, count, 1);
}

dcgmReturn_t tsapiGetEntityGroupEntities(dcgmHandle_t dcgmHandle,
                                         dcgm_field_entity_group_t entityGroup,
                                         dcgm_field_eid_t *entities,
                                         int *numEntities,
                                         unsigned int flags)
{
    if (!entities || !numEntities)
    {
        return DCGM_ST_BADPARAM;
    }

    int entitiesCapacity = *numEntities;

    dcgmReturn_t ret;

    std::unique_ptr<dcgm_core_msg_get_entity_group_entities_t> msg
        = std::make_unique<dcgm_core_msg_get_entity_group_entities_t>();
    memset(msg.get(), 0, sizeof(*msg));

    msg->header.length     = sizeof(*msg);
    msg->header.moduleId   = DcgmModuleIdCore;
    msg->header.subCommand = DCGM_CORE_SR_GET_ENTITY_GROUP_ENTITIES;
    msg->header.version    = dcgm_core_msg_get_entity_group_entities_version;

    msg->entities.flags       = flags;
    msg->entities.entityGroup = entityGroup;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg->header, sizeof(*msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (msg->entities.cmdRet != DCGM_ST_OK)
    {
        return (dcgmReturn_t)msg->entities.cmdRet;
    }

    *numEntities = msg->entities.numEntities;

    if (msg->entities.numEntities > static_cast<unsigned int>(entitiesCapacity))
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    for (unsigned int i = 0; i < msg->entities.numEntities; ++i)
    {
        entities[i] = msg->entities.entities[i];
    }

    return DCGM_ST_OK;
}

dcgmReturn_t tsapiGetGpuChipArchitecture(dcgmHandle_t dcgmHandle,
                                         unsigned int gpuId,
                                         dcgmChipArchitecture_t *chipArchitecture)
{
    if (chipArchitecture == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_get_gpu_chip_architecture_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_GPU_CHIP_ARCHITECTURE;
    msg.header.version    = dcgm_core_msg_get_gpu_chip_architecture_version;

    msg.info.gpuId = gpuId;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    *chipArchitecture = msg.info.data;

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiGetGpuInstanceHierarchy(dcgmHandle_t dcgmHandle, dcgmMigHierarchy_v2 *hierarchy)
{
    dcgmReturn_t ret = DCGM_ST_NOT_SUPPORTED;

    if (hierarchy == nullptr)
    {
        DCGM_LOG_ERROR << "Invalid call to dcgmGetGpuInstanceHierarchy. "
                          "The hierarchy parameter must be specified.";
        return DCGM_ST_BADPARAM;
    }

    if (hierarchy->version != dcgmMigHierarchy_version2)
    {
        DCGM_LOG_ERROR << "The dcgmEntityHierarchy was called with an invalid hierarchy argument version."
                       << "Expected dcgmMigHierarchy_version2 (" << dcgmMigHierarchy_version2 << ") "
                       << "Given argument version is " << hierarchy->version;
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_get_gpu_instance_hierarchy_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY;
    msg.header.version    = dcgm_core_msg_get_gpu_instance_hierarchy_version;

    memcpy(&msg.info.data, hierarchy, sizeof(msg.info.data));

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    DCGM_LOG_DEBUG << "Got total GPUs/GPU Instances/GPU Compute Instances back: " << hierarchy->count
                   << ". dcgmReturn: " << ret;

    memcpy(hierarchy, &msg.info.data, sizeof(msg.info.data));

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiCreateMigEntity(dcgmHandle_t dcgmHandle, dcgmCreateMigEntity_t *cme)
{
    if (cme == nullptr)
    {
        DCGM_LOG_ERROR << "Invalid pointer to the struct specifying how to create the MIG entity.";
        return DCGM_ST_BADPARAM;
    }

    if (cme->version != dcgmCreateMigEntity_version)
    {
        DCGM_LOG_ERROR << "Got bad version " << cme->version << ".";
        return DCGM_ST_VER_MISMATCH;
    }

    if (cme->createOption == DcgmMigCreateComputeInstance
        && (cme->profile < DcgmMigProfileComputeInstanceSlice1
            || cme->profile > DcgmMigProfileComputeInstanceSlice1Rev1))
    {
        DCGM_LOG_ERROR << "Invalid profile " << cme->profile << " for creating a compute instance";
        return DCGM_ST_BADPARAM;
    }
    else if (cme->createOption == DcgmMigCreateGpuInstance
             && (cme->profile < DcgmMigProfileGpuInstanceSlice1 || cme->profile > DcgmMigProfileGpuInstanceSlice1Rev2))
    {
        DCGM_LOG_ERROR << "Invalid profile " << cme->profile << " for creating a GPU instance";
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_create_mig_entity_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_MIG_ENTITY_CREATE;
    msg.header.version    = dcgm_core_msg_create_mig_entity_version;
    memcpy(&(msg.cme), cme, sizeof(msg.cme));

    // coverity[overrun-buffer-arg]
    return dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
}

dcgmReturn_t tsapiDeleteMigEntity(dcgmHandle_t dcgmHandle, dcgmDeleteMigEntity_t *dme)
{
    if (dme == nullptr)
    {
        DCGM_LOG_ERROR << "Invalid pointer to the struct specifying which MIG entity should be deleted.";
        return DCGM_ST_BADPARAM;
    }

    if (dme->version != dcgmDeleteMigEntity_version)
    {
        DCGM_LOG_ERROR << "Got bad version " << dme->version << ".";
        return DCGM_ST_VER_MISMATCH;
    }

    switch (dme->entityGroupId)
    {
        case DCGM_FE_GPU_I:

            if (dme->entityId >= DCGM_MAX_INSTANCES)
            {
                DCGM_LOG_ERROR << "Entity id " << dme->entityId << " is above the maximum limit " << DCGM_MAX_INSTANCES;
                return DCGM_ST_BADPARAM;
            }
            break;

        case DCGM_FE_GPU_CI:
            if (dme->entityId >= DCGM_MAX_COMPUTE_INSTANCES)
            {
                DCGM_LOG_ERROR << "Entity id " << dme->entityId << " is above the maximum limit "
                               << DCGM_MAX_COMPUTE_INSTANCES;
                return DCGM_ST_BADPARAM;
            }
            break;

        default:
            DCGM_LOG_ERROR << "Invalid group specified for dcgmDeleteMigEntity: " << dme->entityGroupId;
            return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_delete_mig_entity_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_MIG_ENTITY_DELETE;
    msg.header.version    = dcgm_core_msg_delete_mig_entity_version;
    memcpy(&(msg.dme), dme, sizeof(msg.dme));

    // coverity[overrun-buffer-arg]
    return dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
}

dcgmReturn_t tsapiGetNvLinkLinkStatus(dcgmHandle_t dcgmHandle, dcgmNvLinkStatus_v4 *linkStatus)
{
    if (!linkStatus)
        return DCGM_ST_BADPARAM;

    if (linkStatus->version != dcgmNvLinkStatus_version4)
        return DCGM_ST_VER_MISMATCH;

    dcgm_core_msg_get_nvlink_status_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_NVLINK_STATUS;
    msg.header.version    = dcgm_core_msg_get_nvlink_status_version;
    memcpy(&msg.info.ls, linkStatus, sizeof(msg.info.ls));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Return code " << ret;
        return ret;
    }

    /* Check the status of the DCGM command */
    if (msg.info.cmdRet != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Return code " << ret;
        return (dcgmReturn_t)msg.info.cmdRet;
    }

    memcpy(linkStatus, &msg.info.ls, sizeof(msg.info.ls));

    DCGM_LOG_DEBUG << "Got " << linkStatus->numGpus << " GPUs and " << linkStatus->numNvSwitches
                   << " back. Return: " << msg.info.cmdRet;

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiGetNvLinkP2PStatus(dcgmHandle_t dcgmHandle, dcgmNvLinkP2PStatus_v1 *linkStatus)
{
    if (!linkStatus)
    {
        DCGM_LOG_ERROR << "Invalid pointer to the struct specifying which NvLink P2P statuses should be returned.";

        return DCGM_ST_BADPARAM;
    }

    if (linkStatus->version != dcgmNvLinkP2PStatus_version1)
    {
        DCGM_LOG_ERROR << "Got bad dcgmNvLinkP2PStatus version " << linkStatus->version << " instead of "
                       << dcgmNvLinkP2PStatus_version1 << ".";

        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_get_nvlink_p2p_status_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_NVLINK_P2P_STATUS;
    msg.header.version    = dcgm_core_msg_get_nvlink_p2p_status_version;
    memcpy(&msg.info.ls, linkStatus, sizeof(msg.info.ls));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Return code " << ret;
        return ret;
    }

    /* Check the status of the DCGM command */
    if (msg.info.cmdRet != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Return code " << ret;
        return (dcgmReturn_t)msg.info.cmdRet;
    }

    memcpy(linkStatus, &msg.info.ls, sizeof(msg.info.ls));

    DCGM_LOG_DEBUG << "Got " << linkStatus->numGpus << " GPUs back. Return: " << msg.info.cmdRet;

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t helperGetCpuHierarchySysmonMsg(dcgmHandle_t dcgmHandle, dcgm_sysmon_msg_get_cpus_t &sysmonMsg)
{
    sysmonMsg.header.length     = sizeof(sysmonMsg);
    sysmonMsg.header.version    = dcgm_sysmon_msg_get_cpus_version;
    sysmonMsg.header.moduleId   = DcgmModuleIdSysmon;
    sysmonMsg.header.subCommand = DCGM_SYSMON_SR_GET_CPUS;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &sysmonMsg.header, sizeof(sysmonMsg));

    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Received {}", errorString(dcgmReturn));
        return dcgmReturn;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t tsapiGetCpuHierarchy(dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v1 *cpuHierarchy)
{
    if (!cpuHierarchy)
        return DCGM_ST_BADPARAM;

    if (cpuHierarchy->version != dcgmCpuHierarchy_version1)
        return DCGM_ST_VER_MISMATCH;

    dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
    dcgmReturn_t dcgmReturn = helperGetCpuHierarchySysmonMsg(dcgmHandle, sysmonMsg);
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    for (unsigned int node = 0; node < sysmonMsg.cpuCount; node++)
    {
        const auto &nodeObject = sysmonMsg.cpus[node];

        cpuHierarchy->cpus[node].cpuId      = nodeObject.cpuId;
        cpuHierarchy->cpus[node].ownedCores = nodeObject.ownedCores;
        cpuHierarchy->numCpus++;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t tsapiGetCpuHierarchy_v2(dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v2 *cpuHierarchy)
{
    if (!cpuHierarchy)
    {
        return DCGM_ST_BADPARAM;
    }

    if (cpuHierarchy->version != dcgmCpuHierarchy_version2)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
    dcgmReturn_t dcgmReturn = helperGetCpuHierarchySysmonMsg(dcgmHandle, sysmonMsg);
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    for (unsigned int node = 0; node < sysmonMsg.cpuCount; node++)
    {
        const auto &nodeObject = sysmonMsg.cpus[node];

        cpuHierarchy->cpus[node].cpuId      = nodeObject.cpuId;
        cpuHierarchy->cpus[node].ownedCores = nodeObject.ownedCores;
        cpuHierarchy->numCpus++;
        SafeCopyTo(cpuHierarchy->cpus[node].serial, (char *)nodeObject.serial);
    }

    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiEngineGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                   unsigned int gpuId,
                                                   dcgmDeviceAttributes_t *pDcgmDeviceAttr)
{
    if (pDcgmDeviceAttr == NULL)
    {
        return DCGM_ST_BADPARAM;
    }

    if (pDcgmDeviceAttr->version != dcgmDeviceAttributes_version3)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    return helperDeviceGetAttributes(pDcgmHandle, gpuId, pDcgmDeviceAttr);
}

static dcgmReturn_t tsapiEngineGetVgpuDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                       unsigned int gpuId,
                                                       dcgmVgpuDeviceAttributes_t *pDcgmVgpuDeviceAttr)
{
    return helperVgpuDeviceGetAttributes(pDcgmHandle, gpuId, pDcgmVgpuDeviceAttr);
}

static dcgmReturn_t tsapiEngineGetVgpuInstanceAttributes(dcgmHandle_t pDcgmHandle,
                                                         unsigned int vgpuId,
                                                         dcgmVgpuInstanceAttributes_t *pDcgmVgpuInstanceAttr)
{
    return helperVgpuInstanceGetAttributes(pDcgmHandle, vgpuId, pDcgmVgpuInstanceAttr);
}

static dcgmReturn_t tsapiEngineConfigSet(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgmConfig_t *pDeviceConfig,
                                         dcgmStatus_t statusHandle)
{
    return helperConfigSet(pDcgmHandle, groupId, pDeviceConfig, statusHandle);
}

static dcgmReturn_t tsapiEngineVgpuConfigSet(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmVgpuConfig_t *pDeviceConfig,
                                             dcgmStatus_t statusHandle)
{
    return helperVgpuConfigSet(pDcgmHandle, groupId, pDeviceConfig, statusHandle);
}

static dcgmReturn_t tsapiEngineConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle)
{
    return helperConfigEnforce(pDcgmHandle, groupId, statusHandle);
}

static dcgmReturn_t tsapiEngineVgpuConfigEnforce(dcgmHandle_t pDcgmHandle,
                                                 dcgmGpuGrp_t groupId,
                                                 dcgmStatus_t statusHandle)
{
    return helperVgpuConfigEnforce(pDcgmHandle, groupId, statusHandle);
}

static dcgmReturn_t tsapiEngineConfigGet(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgmConfigType_t type,
                                         int count,
                                         dcgmConfig_t deviceConfigList[],
                                         dcgmStatus_t statusHandle)
{
    return helperConfigGet(pDcgmHandle, groupId, type, count, deviceConfigList, statusHandle);
}

static dcgmReturn_t tsapiEngineVgpuConfigGet(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmConfigType_t type,
                                             int count,
                                             dcgmVgpuConfig_t deviceConfigList[],
                                             dcgmStatus_t statusHandle)
{
    return helperVgpuConfigGet(pDcgmHandle, groupId, type, count, deviceConfigList, statusHandle);
}

static dcgmReturn_t tsapiEngineInjectFieldValue(dcgmHandle_t pDcgmHandle,
                                                unsigned int gpuId,
                                                dcgmInjectFieldValue_t *pDcgmInjectFieldValue)
{
    return helperInjectFieldValue(pDcgmHandle, gpuId, pDcgmInjectFieldValue);
}

static dcgmReturn_t tsapiGetGpuStatus(dcgmHandle_t pDcgmHandle, unsigned int gpuId, DcgmEntityStatus_t *status)
{
    if (status == nullptr)
    {
        DCGM_LOG_ERROR << "Invalid status pointer when getting gpu status.";
        return DCGM_ST_BADPARAM;
    }

    *status = DcgmEntityStatusUnknown;

    dcgm_core_msg_get_gpu_status_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_GPU_STATUS;
    msg.header.version    = dcgm_core_msg_get_gpu_status_version;
    msg.gpuId             = gpuId;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK == ret)
    {
        *status = msg.status;
    }

    return ret;
}

static dcgmReturn_t tsapiCreateFakeEntities(dcgmHandle_t pDcgmHandle, dcgmCreateFakeEntities_v2 *createFakeEntities)
{
    if (!createFakeEntities)
        return DCGM_ST_BADPARAM;

    if (createFakeEntities->version != dcgmCreateFakeEntities_version)
        return DCGM_ST_VER_MISMATCH;

    dcgm_core_msg_create_fake_entities_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_CREATE_FAKE_ENTITIES;
    msg.header.version    = dcgm_core_msg_create_fake_entities_version;

    memcpy(&msg.info.fe, createFakeEntities, sizeof(msg.info.fe));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    DCGM_LOG_DEBUG << "Created " << createFakeEntities->numToCreate << " Return: " << ret;

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    memcpy(createFakeEntities, &msg.info.fe, sizeof(msg.info.fe));

    return (dcgmReturn_t)msg.info.cmdRet;
}

static dcgmReturn_t tsapiEngineGetLatestValuesForFields(dcgmHandle_t pDcgmHandle,
                                                        int gpuId,
                                                        unsigned short fieldIds[],
                                                        unsigned int count,
                                                        dcgmFieldValue_v1 values[])
{
    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_GPU;
    entityPair.entityId      = gpuId;
    DcgmFvBuffer fvBuffer(0);
    dcgmReturn_t dcgmReturn
        = helperGetLatestValuesForFields(pDcgmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, 0);
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = fvBuffer.GetAllAsFv1(values, count, 0);
    return dcgmReturn;
}

static dcgmReturn_t tsapiEngineEntityGetLatestValues(dcgmHandle_t pDcgmHandle,
                                                     dcgm_field_entity_group_t entityGroup,
                                                     int entityId,
                                                     unsigned short fieldIds[],
                                                     unsigned int count,
                                                     dcgmFieldValue_v1 values[])
{
    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = entityGroup;
    entityPair.entityId      = entityId;
    DcgmFvBuffer fvBuffer(0);
    dcgmReturn_t dcgmReturn
        = helperGetLatestValuesForFields(pDcgmHandle, 0, &entityPair, 1, 0, fieldIds, count, &fvBuffer, 0);

    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = fvBuffer.GetAllAsFv1(values, count, 0);
    return dcgmReturn;
}

static dcgmReturn_t tsapiEngineGetMultipleValuesForField(dcgmHandle_t pDcgmHandle,
                                                         int gpuId,
                                                         unsigned short fieldId,
                                                         int *count,
                                                         long long startTs,
                                                         long long endTs,
                                                         dcgmOrder_t order,
                                                         dcgmFieldValue_v1 values[])
{
    return helperGetMultipleValuesForFieldFV1s(
        pDcgmHandle, DCGM_FE_GPU, gpuId, fieldId, count, startTs, endTs, order, values);
}

static dcgmReturn_t tsapiEngineWatchFieldValue(dcgmHandle_t pDcgmHandle,
                                               int gpuId,
                                               unsigned short fieldId,
                                               long long updateFreq,
                                               double maxKeepAge,
                                               int maxKeepSamples)
{
    return helperWatchFieldValue(pDcgmHandle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepSamples);
}

static dcgmReturn_t tsapiEnginePolicyGet(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         int count,
                                         dcgmPolicy_t policy[],
                                         dcgmStatus_t statusHandle)
{
    return helperPolicyGet(pDcgmHandle, groupId, count, policy, statusHandle);
}

static dcgmReturn_t tsapiEnginePolicySet(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgmPolicy_t *policy,
                                         dcgmStatus_t statusHandle)
{
    return helperPolicySet(pDcgmHandle, groupId, policy, statusHandle);
}

static dcgmReturn_t tsapiEnginePolicyTrigger(dcgmHandle_t /* pDcgmHandle */)
{
    /* Policy management is now edge-triggered, so this function has no reason
       to exist anymore. Also, it only ever worked in the embedded case.
       Just returning OK to not break old clients. */
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiEnginePolicyRegister(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgmPolicyCondition_t condition,
                                              fpRecvUpdates callback,
                                              uint64_t userData)
{
    return helperPolicyRegister(pDcgmHandle, groupId, condition, callback, userData);
}

static dcgmReturn_t tsapiEnginePolicyUnregister(dcgmHandle_t pDcgmHandle,
                                                dcgmGpuGrp_t groupId,
                                                dcgmPolicyCondition_t condition)
{
    return helperPolicyUnregister(pDcgmHandle, groupId, condition);
}

static dcgmReturn_t tsapiEngineGetFieldValuesSince(dcgmHandle_t pDcgmHandle,
                                                   dcgmGpuGrp_t groupId,
                                                   long long sinceTimestamp,
                                                   unsigned short *fieldIds,
                                                   int numFieldIds,
                                                   long long *nextSinceTimestamp,
                                                   dcgmFieldValueEnumeration_f enumCB,
                                                   void *userData)
{
    return helperGetFieldValuesSince(
        pDcgmHandle, groupId, sinceTimestamp, fieldIds, numFieldIds, nextSinceTimestamp, enumCB, userData);
}

static dcgmReturn_t tsapiEngineGetValuesSince(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgmFieldGrp_t fieldGroupId,
                                              long long sinceTimestamp,
                                              long long *nextSinceTimestamp,
                                              dcgmFieldValueEnumeration_f enumCB,
                                              void *userData)
{
    return helperGetValuesSince(
        pDcgmHandle, groupId, fieldGroupId, sinceTimestamp, nextSinceTimestamp, enumCB, 0, userData);
}

static dcgmReturn_t tsapiEngineGetValuesSince_v2(dcgmHandle_t pDcgmHandle,
                                                 dcgmGpuGrp_t groupId,
                                                 dcgmFieldGrp_t fieldGroupId,
                                                 long long sinceTimestamp,
                                                 long long *nextSinceTimestamp,
                                                 dcgmFieldValueEntityEnumeration_f enumCB,
                                                 void *userData)
{
    return helperGetValuesSince(
        pDcgmHandle, groupId, fieldGroupId, sinceTimestamp, nextSinceTimestamp, 0, enumCB, userData);
}

static dcgmReturn_t tsapiEngineGetLatestValues(dcgmHandle_t pDcgmHandle,
                                               dcgmGpuGrp_t groupId,
                                               dcgmFieldGrp_t fieldGroupId,
                                               dcgmFieldValueEnumeration_f enumCB,
                                               void *userData)
{
    return helperGetLatestValues(pDcgmHandle, groupId, fieldGroupId, enumCB, 0, userData);
}

static dcgmReturn_t tsapiEngineGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                                  dcgmGpuGrp_t groupId,
                                                  dcgmFieldGrp_t fieldGroupId,
                                                  dcgmFieldValueEntityEnumeration_f enumCB,
                                                  void *userData)
{
    return helperGetLatestValues(pDcgmHandle, groupId, fieldGroupId, 0, enumCB, userData);
}

static dcgmReturn_t tsapiEngineHealthSet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t systems)
{
    dcgmHealthSetParams_v2 params {};

    params.version = dcgmHealthSetParams_version2;
    params.groupId = groupId;
    params.systems = systems;
    /* Default legacy watches to 30 seconds. Keep the data for 10 minutes */
    params.updateInterval = 30000000;
    params.maxKeepAge     = 600.0;

    return helperHealthSet(pDcgmHandle, &params);
}

static dcgmReturn_t tsapiEngineHealthSet_v2(dcgmHandle_t pDcgmHandle, dcgmHealthSetParams_v2 *params)
{
    return helperHealthSet(pDcgmHandle, params);
}

static dcgmReturn_t tsapiEngineHealthGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems)
{
    return helperHealthGet(pDcgmHandle, groupId, systems);
}

static dcgmReturn_t tsapiEngineHealthCheck(dcgmHandle_t pDcgmHandle,
                                           dcgmGpuGrp_t groupId,
                                           dcgmHealthResponse_t *response)
{
    if (!response)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    if (response->version != dcgmHealthResponse_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    return helperHealthCheckV5(pDcgmHandle, groupId, reinterpret_cast<dcgmHealthResponse_v5 *>(response));
}

static dcgmReturn_t tsapiEngineActionValidate_v2(dcgmHandle_t pDcgmHandle,
                                                 dcgmRunDiag_v10 *drd,
                                                 dcgmDiagResponse_v12 *response)
{
    return helperActionManager(pDcgmHandle, drd, DCGM_POLICY_ACTION_NONE, response);
}

static dcgmReturn_t tsapiEngineActionValidate(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgmPolicyValidation_t validate,
                                              dcgmDiagResponse_v12 *response)
{
    dcgmRunDiag_v10 drd10 = {};

    drd10.version  = dcgmRunDiag_version10;
    drd10.validate = validate;
    drd10.groupId  = groupId;

    return helperActionManager(pDcgmHandle, &drd10, DCGM_POLICY_ACTION_NONE, response);
}

static dcgmReturn_t tsapiEngineRunDiagnostic(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmDiagnosticLevel_t diagLevel,
                                             dcgmDiagResponse_v12 *diagResponse)
{
    dcgmPolicyValidation_t validation = DCGM_POLICY_VALID_NONE;

    if (!diagResponse)
        return DCGM_ST_BADPARAM;

    if (!diagResponse->version)
    {
        log_debug("Version missing");
        return DCGM_ST_VER_MISMATCH;
    }

    /* diagLevel -> validation */
    switch (diagLevel)
    {
        case DCGM_DIAG_LVL_SHORT:
            validation = DCGM_POLICY_VALID_SV_SHORT;
            break;

        case DCGM_DIAG_LVL_MED:
            validation = DCGM_POLICY_VALID_SV_MED;
            break;

        case DCGM_DIAG_LVL_LONG:
            validation = DCGM_POLICY_VALID_SV_LONG;
            break;

        case DCGM_DIAG_LVL_XLONG:
            validation = DCGM_POLICY_VALID_SV_XLONG;
            break;

        case DCGM_DIAG_LVL_INVALID:
        default:
            log_error("Invalid diagLevel {}", (int)diagLevel);
            return DCGM_ST_BADPARAM;
    }

    if (diagResponse->version == dcgmDiagResponse_version12 || diagResponse->version == dcgmDiagResponse_version11)
    {
        dcgmRunDiag_v10 drd = {};
        drd.version         = dcgmRunDiag_version10;
        drd.groupId         = groupId;
        drd.validate        = validation;

        return helperActionManager(pDcgmHandle, &drd, DCGM_POLICY_ACTION_NONE, diagResponse);
    }
    else
    {
        dcgmRunDiag_v7 drd = {};
        drd.version        = dcgmRunDiag_version7;
        drd.groupId        = groupId;
        drd.validate       = validation;

        return helperActionManager(
            pDcgmHandle, reinterpret_cast<dcgmRunDiag_v10 *>(&drd), DCGM_POLICY_ACTION_NONE, diagResponse);
    }
}

static dcgmReturn_t tsapiEngineStopDiagnostic(dcgmHandle_t pDcgmHandle)
{
    return helperStopDiag(pDcgmHandle);
}

static dcgmReturn_t tsapiEngineRunMnDiagnostic(dcgmHandle_t pDcgmHandle,
                                               dcgmRunMnDiag_v1 const *drmnd,
                                               dcgmMnDiagResponse_v1 *response)
{
    return helperRunMnDiag(pDcgmHandle, drmnd, response);
}

static dcgmReturn_t helperMultinodeRequest(dcgmHandle_t pDcgmHandle, dcgmMultinodeRequest_t *pRequest)
{
    if (!pRequest)
    {
        return DCGM_ST_BADPARAM;
    }

    if (pRequest->version != dcgmMultinodeRequest_version1)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    // At this time, only mnubergemm is supported
    if (pRequest->testType != mnubergemm)
    {
        log_error("Invalid test type {}", (int)pRequest->testType);
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t dcgmReturn { DCGM_ST_OK };
    std::unique_ptr<dcgm_mndiag_msg_resource_v1> resourceMsg;
    std::unique_ptr<dcgm_mndiag_msg_authorization_v1> authorizationMsg;
    std::unique_ptr<dcgm_mndiag_msg_run_params_v1> runParamsMsg;
    std::unique_ptr<dcgm_mndiag_msg_node_info_v1> nodeInfoMsg;

    auto initResourceMsg = [&resourceMsg, &pRequest](unsigned int subCommand) {
        resourceMsg = std::make_unique<dcgm_mndiag_msg_resource_v1>();
        memset(resourceMsg.get(), 0, sizeof(*resourceMsg));
        resourceMsg->header.length     = sizeof(*resourceMsg);
        resourceMsg->header.version    = dcgm_mndiag_msg_resource_version1;
        resourceMsg->header.moduleId   = DcgmModuleIdMnDiag;
        resourceMsg->header.subCommand = subCommand;
        memcpy(&resourceMsg->resource, &pRequest->requestData.resource, sizeof(pRequest->requestData.resource));
    };

    auto initAuthorizationMsg = [&authorizationMsg, &pRequest](unsigned int subCommand) {
        authorizationMsg = std::make_unique<dcgm_mndiag_msg_authorization_v1>();
        memset(authorizationMsg.get(), 0, sizeof(*authorizationMsg));
        authorizationMsg->header.length     = sizeof(*authorizationMsg);
        authorizationMsg->header.version    = dcgm_mndiag_msg_authorization_version1;
        authorizationMsg->header.moduleId   = DcgmModuleIdMnDiag;
        authorizationMsg->header.subCommand = subCommand;
        memcpy(&authorizationMsg->authorization,
               &pRequest->requestData.authorization,
               sizeof(pRequest->requestData.authorization));
    };

    auto initRunParamsMsg = [&runParamsMsg, &pRequest](unsigned int subCommand) {
        runParamsMsg = std::make_unique<dcgm_mndiag_msg_run_params_v1>();
        memset(runParamsMsg.get(), 0, sizeof(*runParamsMsg));
        runParamsMsg->header.length     = sizeof(*runParamsMsg);
        runParamsMsg->header.version    = dcgm_mndiag_msg_run_params_version1;
        runParamsMsg->header.moduleId   = DcgmModuleIdMnDiag;
        runParamsMsg->header.subCommand = subCommand;
        memcpy(&runParamsMsg->runParams, &pRequest->requestData.runParams, sizeof(pRequest->requestData.runParams));
    };

    auto initNodeInfoMsg = [&nodeInfoMsg, &pRequest](unsigned int subCommand) {
        nodeInfoMsg = std::make_unique<dcgm_mndiag_msg_node_info_v1>();
        memset(nodeInfoMsg.get(), 0, sizeof(*nodeInfoMsg));
        nodeInfoMsg->header.length     = sizeof(*nodeInfoMsg);
        nodeInfoMsg->header.version    = dcgm_mndiag_msg_node_info_version1;
        nodeInfoMsg->header.moduleId   = DcgmModuleIdMnDiag;
        nodeInfoMsg->header.subCommand = subCommand;
    };

    constexpr unsigned int SIXTY_MINUTES_IN_MS = 3600000;

    switch (pRequest->requestType)
    {
        case ReserveResources:
            initResourceMsg(DCGM_MNDIAG_SR_RESERVE_RESOURCES);
            // coverity[overrun-buffer-arg]
            dcgmReturn = dcgmModuleSendBlockingFixedRequest(
                pDcgmHandle, &resourceMsg->header, sizeof(*resourceMsg), nullptr, SIXTY_MINUTES_IN_MS);
            if (dcgmReturn == DCGM_ST_OK)
            {
                memcpy(&pRequest->requestData.resource, &resourceMsg->resource, sizeof(pRequest->requestData.resource));
            }
            break;

        case ReleaseResources:
            initResourceMsg(DCGM_MNDIAG_SR_RELEASE_RESOURCES);
            // coverity[overrun-buffer-arg]
            dcgmReturn = dcgmModuleSendBlockingFixedRequest(
                pDcgmHandle, &resourceMsg->header, sizeof(*resourceMsg), nullptr, SIXTY_MINUTES_IN_MS);
            if (dcgmReturn == DCGM_ST_OK)
            {
                memcpy(&pRequest->requestData.resource, &resourceMsg->resource, sizeof(pRequest->requestData.resource));
            }
            break;

        case DetectProcess:
            initResourceMsg(DCGM_MNDIAG_SR_DETECT_PROCESS);
            // coverity[overrun-buffer-arg]
            dcgmReturn = dcgmModuleSendBlockingFixedRequest(
                pDcgmHandle, &resourceMsg->header, sizeof(*resourceMsg), nullptr, SIXTY_MINUTES_IN_MS);
            if (dcgmReturn == DCGM_ST_OK)
            {
                memcpy(&pRequest->requestData.resource, &resourceMsg->resource, sizeof(pRequest->requestData.resource));
            }
            break;

        case AuthorizeConnection:
            initAuthorizationMsg(DCGM_MNDIAG_SR_AUTHORIZE_CONNECTION);
            // coverity[overrun-buffer-arg]
            dcgmReturn = dcgmModuleSendBlockingFixedRequest(
                pDcgmHandle, &authorizationMsg->header, sizeof(*authorizationMsg), nullptr, SIXTY_MINUTES_IN_MS);
            break;

        case RevokeAuthorization:
            initAuthorizationMsg(DCGM_MNDIAG_SR_REVOKE_AUTHORIZATION);
            // coverity[overrun-buffer-arg]
            dcgmReturn = dcgmModuleSendBlockingFixedRequest(
                pDcgmHandle, &authorizationMsg->header, sizeof(*authorizationMsg), nullptr, SIXTY_MINUTES_IN_MS);
            break;

        case BroadcastRunParameters:
            initRunParamsMsg(DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS);
            // coverity[overrun-buffer-arg]
            dcgmReturn = dcgmModuleSendBlockingFixedRequest(
                pDcgmHandle, &runParamsMsg->header, sizeof(*runParamsMsg), nullptr, SIXTY_MINUTES_IN_MS);
            break;

        case GetNodeInfo:
            initNodeInfoMsg(DCGM_MNDIAG_SR_GET_NODE_INFO);
            // coverity[overrun-buffer-arg]
            dcgmReturn = dcgmModuleSendBlockingFixedRequest(
                pDcgmHandle, &nodeInfoMsg->header, sizeof(*nodeInfoMsg), nullptr, SIXTY_MINUTES_IN_MS);
            memcpy(&pRequest->requestData.nodeInfo, &nodeInfoMsg->nodeInfo, sizeof(pRequest->requestData.nodeInfo));
            break;

        default:
            log_error("Invalid request type {}", (int)pRequest->requestType);
            return DCGM_ST_BADPARAM;
    }

    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Failed to send message of dcgmMultinodeRequestType_t type {}. Return: ({}): {}",
                  std::to_underlying(pRequest->requestType),
                  std::to_underlying(dcgmReturn),
                  errorString(dcgmReturn));
        return dcgmReturn;
    }
    return dcgmReturn;
}

static dcgmReturn_t tsapiEngineMultinodeRequest(dcgmHandle_t pDcgmHandle, dcgmMultinodeRequest_t *pRequest)
{
    return helperMultinodeRequest(pDcgmHandle, pRequest);
}

static dcgmReturn_t tsapiEngineStopMnDiagnostic(dcgmHandle_t pDcgmHandle)
{
    return helperStopMnDiag(pDcgmHandle);
}

static dcgmReturn_t tsapiEngineGetPidInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo)
{
    return helperGetPidInfo(pDcgmHandle, groupId, pidInfo);
}

static dcgmReturn_t tsapiEngineWatchPidFields(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              long long updateFreq,
                                              double maxKeepAge,
                                              int maxKeepSamples)
{
    dcgm_core_msg_watch_predefined_fields_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_WATCH_PREDEFINED_FIELDS;
    msg.header.version    = dcgm_core_msg_watch_predefined_fields_version;

    msg.info.wpf.version         = dcgmWatchPredefined_version;
    msg.info.wpf.watchPredefType = DCGM_WATCH_PREDEF_PID;
    msg.info.wpf.groupId         = groupId;
    msg.info.wpf.updateFreq      = updateFreq;
    msg.info.wpf.maxKeepAge      = maxKeepAge;
    msg.info.wpf.maxKeepSamples  = maxKeepSamples;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

static dcgmReturn_t tsapiEngineWatchJobFields(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              long long updateFreq,
                                              double maxKeepAge,
                                              int maxKeepSamples)
{
    dcgm_core_msg_watch_predefined_fields_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_WATCH_PREDEFINED_FIELDS;
    msg.header.version    = dcgm_core_msg_watch_predefined_fields_version;

    msg.info.wpf.version         = dcgmWatchPredefined_version;
    msg.info.wpf.watchPredefType = DCGM_WATCH_PREDEF_JOB;
    msg.info.wpf.groupId         = groupId;
    msg.info.wpf.updateFreq      = updateFreq;
    msg.info.wpf.maxKeepAge      = maxKeepAge;
    msg.info.wpf.maxKeepSamples  = maxKeepSamples;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t helperJobStatCmd(dcgmHandle_t pDcgmHandle, unsigned int groupId, const char JobId[64], unsigned int jobCmd)
{
    dcgm_core_msg_job_cmd_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = jobCmd;
    msg.header.version    = dcgm_core_msg_job_cmd_version;

    switch (jobCmd)
    {
        case DCGM_CORE_SR_JOB_START_STATS:

            msg.jc.groupId = groupId;
            SafeCopyTo(msg.jc.jobId, JobId);
            break;

        case DCGM_CORE_SR_JOB_STOP_STATS:
        case DCGM_CORE_SR_JOB_REMOVE:

            SafeCopyTo(msg.jc.jobId, JobId);
            break;

        case DCGM_CORE_SR_JOB_REMOVE_ALL:
            /* nothing to specify here */
            break;

        default:

            return DCGM_ST_GENERIC_ERROR;
    }

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.jc.cmdRet;
}

dcgmReturn_t tsapiEngineJobStartStats(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64])
{
    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    return helperJobStatCmd(pDcgmHandle, groupId, jobId, DCGM_CORE_SR_JOB_START_STATS);
}

dcgmReturn_t tsapiEngineJobStopStats(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    return helperJobStatCmd(pDcgmHandle, 0, jobId, DCGM_CORE_SR_JOB_STOP_STATS);
}

dcgmReturn_t tsapiEngineJobGetStats(dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo)
{
    if ((NULL == jobId) || (NULL == pJobInfo))
        return DCGM_ST_BADPARAM;

    /* Valid version can't be 0 or just any random number  */
    if (pJobInfo->version != dcgmJobInfo_version)
    {
        DCGM_LOG_DEBUG << "Version Mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    if ((0 == jobId[0]))
    {
        DCGM_LOG_DEBUG << "Job ID was NULL";
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_job_get_stats_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_JOB_GET_STATS;
    msg.header.version    = dcgm_core_msg_job_get_stats_version;

    SafeCopyTo(msg.jc.jobId, jobId);
    msg.jc.jobStats.version = dcgmJobInfo_version;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    if (DCGM_ST_OK != msg.jc.cmdRet)
    {
        return (dcgmReturn_t)msg.jc.cmdRet;
    }

    memcpy(pJobInfo, &msg.jc.jobStats, sizeof(dcgmJobInfo_t));

    return DCGM_ST_OK;
}

dcgmReturn_t tsapiEngineJobRemove(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    return helperJobStatCmd(pDcgmHandle, 0, jobId, DCGM_CORE_SR_JOB_REMOVE);
}

dcgmReturn_t tsapiEngineJobRemoveAll(dcgmHandle_t pDcgmHandle)
{
    return helperJobStatCmd(pDcgmHandle, 0, "", DCGM_CORE_SR_JOB_REMOVE_ALL);
}

static dcgmReturn_t tsapiEngineGetWorkloadPowerProfileInfo(dcgmHandle_t pDcgmHandle,
                                                           unsigned int gpuId,
                                                           dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo,
                                                           dcgmDeviceWorkloadPowerProfilesStatus_v1 *profilesStatus)
{
    dcgmReturn_t dcgmReturn;

    if (profilesInfo == nullptr || profilesStatus == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (profilesInfo->version != dcgmWorkloadPowerProfileProfilesInfo_version
        || profilesStatus->version != dcgmDeviceWorkloadPowerProfilesStatus_version)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_get_workload_power_profiles_status_v1 msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_WORKLOAD_POWER_PROFILES_STATUS;
    msg.header.version    = dcgm_core_msg_get_workload_power_profiles_status_version;

    msg.pp.gpuId = gpuId;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK == dcgmReturn)
    {
        /* Copy the response back over the request */
        memcpy(profilesInfo, &msg.pp.profilesInfo, sizeof(dcgmWorkloadPowerProfileProfilesInfo_v1));
        memcpy(profilesStatus, &msg.pp.profilesStatus, sizeof(dcgmDeviceWorkloadPowerProfilesStatus_v1));
    }

    return (dcgmReturn_t)msg.cmdRet;
}

static dcgmReturn_t tsapiEngineGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                                 unsigned int gpuId,
                                                 dcgmDeviceTopology_t *deviceTopology)
{
    dcgmTopology_t groupTopology;
    dcgmAffinity_t groupAffinity;
    dcgmReturn_t ret = DCGM_ST_OK;

    if (!deviceTopology)
    {
        DCGM_LOG_ERROR << "bad deviceTopology " << (void *)deviceTopology;
        return DCGM_ST_BADPARAM;
    }

    unsigned int numGpusInTopology = 0;

    memset(&groupTopology, 0, sizeof(groupTopology));
    memset(&groupAffinity, 0, sizeof(groupAffinity));
    deviceTopology->version = dcgmDeviceTopology_version;

    ret = helperGetTopologyPci(pDcgmHandle,
                               (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS,
                               &groupTopology); // retrieve the topology for the entire system
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "helperGetTopologyPci returned " << ret;
        return ret;
    }

    // numElements from topology is going to be zero here if DCGM_ST_NO_DATA is returned

    // go through the entire topology looking for a match of gpuId in their gpuA or gpuB of the paths structs
    for (unsigned int index = 0; index < groupTopology.numElements; index++)
    {
        unsigned int gpuA = groupTopology.element[index].dcgmGpuA;
        unsigned int gpuB = groupTopology.element[index].dcgmGpuB;

        if (gpuA == gpuId || gpuB == gpuId)
        {
            deviceTopology->gpuPaths[numGpusInTopology].gpuId = (gpuA == gpuId) ? gpuB : gpuA;
            deviceTopology->gpuPaths[numGpusInTopology].path  = groupTopology.element[index].path;
            // the GPU topo info is store always lowGpuId connected to highGpuId
            // i.e. 0->1, 1->2, 1->4 ... never 3->1.
            // thus if gpuId == gpuA then we need to use the AtoBNvLinkIds entry as GPU A will always be a lower number
            // if gpuId == gpuB then use BtoANvLinkIds.
            if (gpuA == gpuId)
                deviceTopology->gpuPaths[numGpusInTopology].localNvLinkIds = groupTopology.element[index].AtoBNvLinkIds;
            else
                deviceTopology->gpuPaths[numGpusInTopology].localNvLinkIds = groupTopology.element[index].BtoANvLinkIds;
            numGpusInTopology++;
        }
    }
    deviceTopology->numGpus = numGpusInTopology;

    // it is ok at this point to have numGpusInTopology == 0 because there may only be one GPU on the system.

    ret = helperGetTopologyAffinity(pDcgmHandle, (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS, &groupAffinity);
    if (DCGM_ST_OK != ret)
        return ret;

    bool found = false;
    for (unsigned int index = 0; index < groupAffinity.numGpus; index++)
    {
        if (groupAffinity.affinityMasks[index].dcgmGpuId == gpuId)
        {
            found = true;
            memcpy(deviceTopology->cpuAffinityMask,
                   groupAffinity.affinityMasks[index].bitmask,
                   sizeof(unsigned long) * DCGM_AFFINITY_BITMASK_ARRAY_SIZE);
            break;
        }
    }

    if (!found) // the gpuId was illegal as ALL GPUs should have some affinity
    {
        DCGM_LOG_WARNING << "No GPU topology found for gpuId " << gpuId << " leaving blank.";
    }

    return ret;
}

/*
 * Compare two topologies. Returns -1 if a is better than b. 0 if same. 1 if b is better than a.
 * This is meant to be used in a qsort() callback, resulting in the elements being sorted in descending order of P2P
 * speed
 */
static int dcgmGpuTopologyLevelCmpCB(dcgmGpuTopologyLevel_t a, dcgmGpuTopologyLevel_t b)
{
    // This code has to be complicated because a lower PCI value is better
    // but a higher NvLink value is better. All NvLinks are better than all PCI
    unsigned int nvLinkPathA = DCGM_TOPOLOGY_PATH_NVLINK(a);
    unsigned int pciPathA    = DCGM_TOPOLOGY_PATH_PCI(a);

    unsigned int nvLinkPathB = DCGM_TOPOLOGY_PATH_NVLINK(b);
    unsigned int pciPathB    = DCGM_TOPOLOGY_PATH_PCI(b);

    /* If both have NvLinks, compare those. More is better */
    if (nvLinkPathA && nvLinkPathB)
    {
        if (nvLinkPathA > nvLinkPathB)
            return -1;
        else if (nvLinkPathA < nvLinkPathB)
            return 1;
        else
            return 0; /* Ignore PCI topology if we have NvLink */
    }

    /* If one or the other has NvLink, that one is faster */
    if (nvLinkPathA && !nvLinkPathB)
        return -1;
    if (nvLinkPathB && !nvLinkPathA)
        return 1;

    /* Neither has NvLink. Compare the PCI topologies. Less is better */
    if (pciPathA < pciPathB)
        return -1;
    else if (pciPathA > pciPathB)
        return 1;

    return 0;
}

dcgmGpuTopologyLevel_t GetSlowestPath(dcgmTopology_t &topology)
{
    dcgmGpuTopologyLevel_t slowestPath = DCGM_TOPOLOGY_UNINITIALIZED;

    // go through the entire topology looking for the slowest path
    // numElements from topology is going to be zero here if DCGM_ST_NO_DATA is returned
    for (unsigned int index = 0; index < topology.numElements; index++)
    {
        /* If slowest path hasn't been set yet or slowest path is better than what we're comparing to */
        if (slowestPath == DCGM_TOPOLOGY_UNINITIALIZED
            || 0 > dcgmGpuTopologyLevelCmpCB(slowestPath, topology.element[index].path))
            slowestPath = topology.element[index].path;
    }

    return slowestPath;
}

static dcgmReturn_t tsapiEngineGroupTopology(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmGroupTopology_t *groupTopology)
{
    dcgmTopology_t topology;
    dcgmAffinity_t affinity;
    dcgmReturn_t ret = DCGM_ST_OK;

    if (!groupTopology)
    {
        DCGM_LOG_ERROR << "bad groupTopology " << (void *)groupTopology;
        return DCGM_ST_BADPARAM;
    }

    groupTopology->version = dcgmGroupTopology_version;

    ret = helperGetTopologyPci(pDcgmHandle, groupId, &topology); // retrieve the topology for this group
    if (DCGM_ST_OK != ret && DCGM_ST_NO_DATA != ret)
        return ret;

    groupTopology->slowestPath = GetSlowestPath(topology);

    ret = helperGetTopologyAffinity(pDcgmHandle, groupId, &affinity);
    if (DCGM_ST_OK != ret)
        return ret;

    bool foundDifference = false;

    // iterate through each element of the bitmask OR'ing them together and locating if there was a difference
    for (unsigned int i = 0; i < DCGM_AFFINITY_BITMASK_ARRAY_SIZE; i++)
    {
        unsigned long overallMask = 0;
        for (unsigned int index = 0; index < affinity.numGpus; index++)
        {
            overallMask |= affinity.affinityMasks[index].bitmask[i];
            if (overallMask != affinity.affinityMasks[index].bitmask[i])
                foundDifference = true;
        }
        groupTopology->groupCpuAffinityMask[i] = overallMask;
    }

    groupTopology->numaOptimalFlag = (foundDifference) ? 0 : 1;
    return ret;
}

static dcgmReturn_t tsapiIntrospectGetHostengineMemoryUsage(dcgmHandle_t dcgmHandle,
                                                            dcgmIntrospectMemory_t *memoryInfo,
                                                            int waitIfNoData)
{
    dcgm_introspect_msg_he_mem_usage_v1 msg;
    dcgmReturn_t dcgmReturn;

    if (!memoryInfo)
        return DCGM_ST_BADPARAM;
    if (memoryInfo->version != dcgmIntrospectMemory_version1)
    {
        log_error("Version mismatch x{:X} != x{:X}", memoryInfo->version, dcgmIntrospectMemory_version1);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE;
    msg.header.version    = dcgm_introspect_msg_he_mem_usage_version1;
    msg.waitIfNoData      = waitIfNoData;
    memcpy(&msg.memoryInfo, memoryInfo, sizeof(*memoryInfo));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Copy the response back over the request */
    memcpy(memoryInfo, &msg.memoryInfo, sizeof(*memoryInfo));
    return dcgmReturn;
}

static dcgmReturn_t tsapiIntrospectGetHostengineCpuUtilization(dcgmHandle_t dcgmHandle,
                                                               dcgmIntrospectCpuUtil_t *cpuUtil,
                                                               int waitIfNoData)
{
    dcgm_introspect_msg_he_cpu_util_v1 msg;
    dcgmReturn_t dcgmReturn;

    if (!cpuUtil)
        return DCGM_ST_BADPARAM;
    if (cpuUtil->version != dcgmIntrospectCpuUtil_version1)
    {
        log_error("Version mismatch x{:X} != x{:X}", cpuUtil->version, dcgmIntrospectCpuUtil_version1);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL;
    msg.header.version    = dcgm_introspect_msg_he_cpu_util_version1;

    msg.waitIfNoData = waitIfNoData;

    memcpy(&msg.cpuUtil, cpuUtil, sizeof(*cpuUtil));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Copy the response back over the request */
    memcpy(cpuUtil, &msg.cpuUtil, sizeof(*cpuUtil));
    return dcgmReturn;
}

static dcgmReturn_t tsapiSelectGpusByTopology(dcgmHandle_t pDcgmHandle,
                                              uint64_t inputGpuIds,
                                              uint32_t numGpus,
                                              uint64_t *outputGpuIds,
                                              uint64_t hintFlags)
{
    return helperSelectGpusByTopology(pDcgmHandle, inputGpuIds, numGpus, outputGpuIds, hintFlags);
}

static dcgmReturn_t tsapiGetFieldSummary(dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request)
{
    return helperGetFieldSummary(pDcgmHandle, request);
}

/*****************************************************************************/
static dcgmReturn_t tsapiModuleDenylist(dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId)
{
    if (moduleId <= DcgmModuleIdCore || moduleId >= DcgmModuleIdCount)
    {
        DCGM_LOG_ERROR << "Bad module ID " << moduleId;
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_module_denylist_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_MODULE_DENYLIST;
    msg.header.version    = dcgm_core_msg_module_denylist_version;

    msg.bl.moduleId = moduleId;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.bl.cmdRet;
}

/*****************************************************************************/
static dcgmReturn_t tsapiModuleGetStatuses(dcgmHandle_t pDcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses)
{
    if (!moduleStatuses)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    if (moduleStatuses->version != dcgmModuleGetStatuses_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_module_status_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_MODULE_STATUS;
    msg.header.version    = dcgm_core_msg_module_status_version;

    memcpy(&msg.info.st, moduleStatuses, sizeof(msg.info.st));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    if (msg.info.cmdRet == DCGM_ST_OK)
    {
        memcpy(moduleStatuses, &msg.info.st, sizeof(msg.info.st));
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}


/*****************************************************************************/
dcgmReturn_t tsapiProfGetSupportedMetricGroups(dcgmHandle_t dcgmHandle, dcgmProfGetMetricGroups_t *metricGroups)
{
    dcgmReturn_t dcgmReturn;

    if (!metricGroups)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    if (metricGroups->version != dcgmProfGetMetricGroups_version3)
    {
        DCGM_LOG_ERROR << "Version mismatch " << std::hex << metricGroups->version
                       << " != " << dcgmProfGetMetricGroups_version3;
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_get_metric_groups_t msg;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_PROF_GET_METRIC_GROUPS;
    msg.header.version    = dcgm_core_msg_get_metric_groups_version;

    memcpy(&msg.metricGroups, metricGroups, sizeof(*metricGroups));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Copy the response back over the request */
    memcpy(metricGroups, &msg.metricGroups, sizeof(*metricGroups));

    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperProfPauseResume(dcgmHandle_t dcgmHandle, bool pause)
{
    dcgm_core_msg_pause_resume_v1 msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_PAUSE_RESUME;
    msg.header.version    = dcgm_core_msg_pause_resume_version1;
    msg.pause             = pause;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
    {
        /* Note that for GPM-enabled platforms, the profiling module may not be loadable.
           return a more appropriate "Not supported" status */
        DCGM_LOG_WARNING << "Converting prof pause/resume return DCGM_ST_MODULE_NOT_LOADED -> DCGM_ST_NOT_SUPPORTED";
        return DCGM_ST_NOT_SUPPORTED;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiProfPause(dcgmHandle_t dcgmHandle)
{
    return helperProfPauseResume(dcgmHandle, true);
}

/*****************************************************************************/
dcgmReturn_t tsapiProfResume(dcgmHandle_t dcgmHandle)
{
    return helperProfPauseResume(dcgmHandle, false);
}

/*****************************************************************************/
dcgmReturn_t tsapiVersionInfo(dcgmVersionInfo_t *pVersionInfo)
{
    return GetVersionInfo(pVersionInfo);
}

dcgmReturn_t tsapiHostengineVersionInfo(dcgmHandle_t dcgmHandle, dcgmVersionInfo_t *pVersionInfo)
{
    if (pVersionInfo == nullptr)
    {
        DCGM_LOG_ERROR << "Invalid pointer when getting hostengine version.";
        return DCGM_ST_BADPARAM;
    }

    /* Still sending this here since we need coverage for the embedded case. DcgmClientHandler only
       handles remote hostengines */

    dcgm_core_msg_hostengine_version_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_HOSTENGINE_VERSION;
    msg.header.version    = dcgm_core_msg_hostengine_version_version;

    memcpy(&msg.version, pVersionInfo, sizeof(msg.version));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK == ret)
    {
        memcpy(pVersionInfo, &msg.version, sizeof(dcgmVersionInfo_t));
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t tsapiHostengineSetLoggingSeverity(dcgmHandle_t dcgmHandle, dcgmSettingsSetLoggingSeverity_t *logging)
{
    if (logging->version != dcgmSettingsSetLoggingSeverity_version2)
    {
        DCGM_LOG_ERROR << "dcgmHostengineSetLoggingSeverity version mismatch " << logging->version
                       << " != " << dcgmSettingsSetLoggingSeverity_version2;
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_set_severity_t msg = {};
    // coverity[overrun-buffer-arg]
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_SET_LOGGING_SEVERITY;
    msg.header.version    = dcgm_core_msg_set_severity_version;
    memcpy(&msg.logging, logging, sizeof(msg.logging));

    // coverity[overrun-buffer-arg]
    return dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
}

dcgmReturn_t tsapiHostengineIsHealthy(dcgmHandle_t dcgmHandle, dcgmHostengineHealth_t *heHealth)
{
    if (heHealth == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (heHealth->version != dcgmHostengineHealth_version)
    {
        DCGM_LOG_ERROR << "dcgmHostengineHealth version mismatch " << heHealth->version
                       << " != " << dcgmHostengineHealth_version;
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_hostengine_health_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_HOSTENGINE_HEALTH;
    msg.header.version    = dcgm_core_msg_hostengine_health_version;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK == ret)
    {
        heHealth->overallHealth = msg.info.overallHealth;

        return (dcgmReturn_t)msg.info.cmdRet;
    }

    return ret;
}

dcgmReturn_t tsapiInjectEntityFieldValueToNvml(dcgmHandle_t dcgmHandle,
                                               dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               dcgmInjectFieldValue_t *value)
{
    if (value == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_nvml_inject_field_value_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_NVML_INJECT_FIELD_VALUE;
    msg.header.version    = dcgm_core_msg_inject_field_value_version;

    msg.iv.entityGroupId = entityGroupId;
    msg.iv.entityId      = entityId;
    memcpy(&msg.iv.fieldValue, value, sizeof(msg.iv.fieldValue));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    return (dcgmReturn_t)msg.iv.cmdRet;
}

dcgmReturn_t tsapiCreateNvmlInjectionGpu(dcgmHandle_t dcgmHandle, unsigned int index)
{
    dcgm_core_msg_nvml_create_injection_gpu_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_NVML_CREATE_FAKE_ENTITY;
    msg.header.version    = dcgm_core_msg_nvml_create_injection_gpu_version;

    msg.info.index = index;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

#ifdef INJECTION_LIBRARY_AVAILABLE
dcgmReturn_t tsapiInjectNvmlDevice(dcgmHandle_t dcgmHandle,
                                   unsigned int gpuId,
                                   const char *key,
                                   const injectNvmlVal_t *extraKeys,
                                   unsigned int extraKeyCount,
                                   const injectNvmlRet_t *injectNvmlRet)
{
    if (key == nullptr)
    {
        DCGM_LOG_ERROR << "Key is required to process injecting an NVML device.";
        return DCGM_ST_BADPARAM;
    }
    if (injectNvmlRet == nullptr)
    {
        DCGM_LOG_ERROR << "Injected NVML return is required to process injecting an NVML device.";
        return DCGM_ST_BADPARAM;
    }
    if (injectNvmlRet->nvmlRet == NVML_SUCCESS && injectNvmlRet->valueCount == 0)
    {
        DCGM_LOG_ERROR << "Value is required to process injecting an NVML device when nvmlRet is NVML_SUCCESS.";
        return DCGM_ST_BADPARAM;
    }

    if (extraKeyCount > NVML_INJECTION_MAX_EXTRA_KEYS)
    {
        DCGM_LOG_ERROR << "Cannot process " << extraKeyCount << " extra keys. The maximum is "
                       << NVML_INJECTION_MAX_EXTRA_KEYS;
        return DCGM_ST_BADPARAM;
    }
    else if (extraKeys != nullptr && extraKeyCount == 0)
    {
        DCGM_LOG_ERROR << "Extra keys cannot be processed specifying how many there are. (0 specified.)";
        return DCGM_ST_BADPARAM;
    }
    else if (injectNvmlRet->valueCount > NVML_INJECTION_MAX_VALUES)
    {
        DCGM_LOG_ERROR << "Cannot process " << injectNvmlRet->valueCount << " values. The maximum is "
                       << NVML_INJECTION_MAX_VALUES;
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_nvml_inject_device_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_NVML_INJECT_DEVICE;
    msg.header.version    = dcgm_core_msg_nvml_inject_device_version;

    snprintf(msg.info.key, sizeof(msg.info.key), "%s", key);
    if (extraKeys != nullptr && extraKeyCount > 0)
    {
        memcpy(&msg.info.extraKeys, extraKeys, sizeof(injectNvmlVal_t) * extraKeyCount);
    }
    msg.info.extraKeyCount = extraKeyCount;
    msg.info.gpuId         = gpuId;
    memcpy(&msg.info.injectNvmlRet, injectNvmlRet, sizeof(*injectNvmlRet));

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiInjectNvmlDeviceForFollowingCalls(dcgmHandle_t dcgmHandle,
                                                    unsigned int gpuId,
                                                    const char *key,
                                                    const injectNvmlVal_t *extraKeys,
                                                    unsigned int extraKeyCount,
                                                    const injectNvmlRet_t *injectNvmlRets,
                                                    unsigned int retCount)
{
    if (key == nullptr)
    {
        DCGM_LOG_ERROR << "Key is required to process injecting an NVML device.";
        return DCGM_ST_BADPARAM;
    }

    if (extraKeyCount > NVML_INJECTION_MAX_EXTRA_KEYS)
    {
        DCGM_LOG_ERROR << "Cannot process " << extraKeyCount << " extra keys. The maximum is "
                       << NVML_INJECTION_MAX_EXTRA_KEYS;
        return DCGM_ST_BADPARAM;
    }
    else if (extraKeys != nullptr && extraKeyCount == 0)
    {
        DCGM_LOG_ERROR << "Extra keys cannot be processed specifying how many there are. (0 specified.)";
        return DCGM_ST_BADPARAM;
    }
    else if (retCount > NVML_INJECTION_MAX_RETURNS)
    {
        DCGM_LOG_ERROR << "Cannot process " << retCount << " returns. The maximum is " << NVML_INJECTION_MAX_RETURNS;
        return DCGM_ST_BADPARAM;
    }
    else if (retCount > 0 && injectNvmlRets == nullptr)
    {
        DCGM_LOG_ERROR << "injectNvmlRets is expected";
        return DCGM_ST_BADPARAM;
    }

    for (unsigned i = 0; i < retCount; ++i)
    {
        if (injectNvmlRets[i].valueCount > NVML_INJECTION_MAX_VALUES)
        {
            DCGM_LOG_ERROR << "The value count in injectNvmlRets[" << i << "] is too large. The maximum is "
                           << NVML_INJECTION_MAX_VALUES;
            return DCGM_ST_BADPARAM;
        }
        if (injectNvmlRets[i].nvmlRet == NVML_SUCCESS && injectNvmlRets[i].valueCount == 0)
        {
            DCGM_LOG_ERROR << "Value is required to process injecting an NVML device when nvmlRet is NVML_SUCCESS.";
            return DCGM_ST_BADPARAM;
        }
    }

    dcgm_core_msg_nvml_inject_device_for_following_calls_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_NVML_INJECT_DEVICE_FOR_FOLLOWING_CALLS;
    msg.header.version    = dcgm_core_msg_nvml_inject_device_for_following_calls_version;

    snprintf(msg.info.key, sizeof(msg.info.key), "%s", key);
    if (extraKeys != nullptr && extraKeyCount > 0)
    {
        memcpy(&msg.info.extraKeys, extraKeys, sizeof(injectNvmlVal_t) * extraKeyCount);
    }
    msg.info.extraKeyCount = extraKeyCount;
    msg.info.gpuId         = gpuId;
    msg.info.retCount      = retCount;
    memcpy(msg.info.injectNvmlRets, injectNvmlRets, sizeof(injectNvmlRet_t) * retCount);

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiGetNvmlInjectFuncCallCount(dcgmHandle_t dcgmHandle, injectNvmlFuncCallCounts_t *nvmlFuncCallCounts)
{
    if (nvmlFuncCallCounts == nullptr)
    {
        DCGM_LOG_ERROR << "nvmlFuncCallCounts nullptr; bad param";
        return DCGM_ST_BADPARAM;
    }
    dcgm_core_msg_get_nvml_inject_func_call_count_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_NVML_INJECT_FUNC_CALL_COUNT;
    msg.header.version    = dcgm_core_msg_get_nvml_inject_func_call_count_version;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }
    *nvmlFuncCallCounts = msg.info.funcCallCounts;

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiResetNvmlInjectFuncCallCount(dcgmHandle_t dcgmHandle)
{
    dcgm_core_msg_reset_nvml_inject_func_call_count_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_RESET_NVML_FUNC_CALL_COUNT;
    msg.header.version    = dcgm_core_msg_reset_nvml_inject_func_call_count_version;

    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t tsapiInjectedNvmlDeviceReset(dcgmHandle_t dcgmHandle, unsigned int gpuId)
{
    dcgm_core_msg_nvml_inject_device_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_NVML_INJECTED_DEVICE_RESET;
    msg.header.version    = dcgm_core_msg_nvml_injected_device_reset_version;
    msg.info.gpuId        = gpuId;
    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }
    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiRemoveNvmlInjectedGpu(dcgmHandle_t dcgmHandle, char const *uuid)
{
    dcgm_core_msg_remove_restore_nvml_injected_gpu_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_REMOVE_NVML_INJECTED_GPU;
    msg.header.version    = dcgm_core_msg_remove_restore_nvml_injected_gpu_version;
    SafeCopyTo(msg.info.uuid, uuid);

    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

dcgmReturn_t tsapiRestoreNvmlInjectedGpu(dcgmHandle_t dcgmHandle, char const *uuid)
{
    dcgm_core_msg_remove_restore_nvml_injected_gpu_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_RESTORE_NVML_INJECTED_GPU;
    msg.header.version    = dcgm_core_msg_remove_restore_nvml_injected_gpu_version;
    SafeCopyTo(msg.info.uuid, uuid);

    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_DEBUG << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

#endif

/*****************************************************************************/
namespace
{
dcgmReturn_t StartEmbeddedV2(dcgmStartEmbeddedV2Params_v1 &params)
{
    /*
     * Do not use DCGM_LOG* macros until the DcgmLogger is actually initialized
     */
    if ((params.opMode != DCGM_OPERATION_MODE_AUTO) && (params.opMode != DCGM_OPERATION_MODE_MANUAL))
    {
        return DCGM_ST_BADPARAM;
    }

    if (!g_dcgmGlobals.isInitialized)
    {
        return DCGM_ST_UNINITIALIZED;
    }

#ifdef INJECTION_LIBRARY_AVAILABLE
    // Force close the handle so injection NVML works with embedded tests
    nvmlClearLibraryHandleIfNeeded();
#endif

    // Change to home dir
    std::string homeDirMsg;
    bool chdirFail      = false;
    const char *homeDir = getenv(DCGM_HOME_DIR_VAR_NAME);
    if (homeDir == nullptr)
    {
        homeDirMsg = fmt::format("Not changing to a home directory - '{}' is not defined in the environment.",
                                 DCGM_HOME_DIR_VAR_NAME);
    }
    else
    {
        if (chdir(homeDir))
        {
            char errbuf[1024];
            strerror_r(errno, errbuf, sizeof(errbuf));

            std::string cwd(std::filesystem::current_path());
            homeDirMsg = fmt::format("Couldn't change to directory '{}' from '{}': {}.", homeDir, cwd, errbuf);
            chdirFail  = true;
        }
    }

    dcgmGlobalsLock();

    /* Check again after lock */
    if (!g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        return DCGM_ST_UNINITIALIZED;
    }

    /* Initialize the logger */
    std::string paramsLogFile;
    if (params.logFile != nullptr)
    {
        paramsLogFile = params.logFile;
    }

    const std::string logFile
        = GetLogFilenameFromArgAndEnv(paramsLogFile, DCGM_LOGGING_DEFAULT_HOSTENGINE_FILE, DCGM_ENV_LOG_PREFIX);

    // If logging severity is unspecified, pass empty string as the arg to the
    // helper. Empty arg => no arg => look at env
    const std::string loggingSeverityArg
        = params.severity == DcgmLoggingSeverityUnspecified
              ? ""
              : LoggingSeverityToString(params.severity, DCGM_LOGGING_SEVERITY_STRING_ERROR);

    const std::string logSeverity = GetLogSeverityFromArgAndEnv(
        loggingSeverityArg, DCGM_LOGGING_DEFAULT_HOSTENGINE_SEVERITY, DCGM_ENV_LOG_PREFIX);

    DcgmLoggingSeverity_t loggingSeverity = LoggingSeverityFromString(logSeverity.c_str(), DcgmLoggingSeverityError);
    DcgmLoggingInit(logFile.c_str(), loggingSeverity, DcgmLoggingSeverityNone);
    /* Set severity explicitly in case logging is already initialized and ignoring logFile + loggingSeverity */
    SetLoggerSeverity(BASE_LOGGER, loggingSeverity);
    RouteLogToBaseLogger(SYSLOG_LOGGER);
    log_debug("Initialized base logger");
    DCGM_LOG_SYSLOG_DEBUG << "Initialized syslog logger";

    if (homeDirMsg.empty() == false)
    {
        if (chdirFail)
        {
            DCGM_LOG_ERROR << homeDirMsg << " Exiting.";
            dcgmGlobalsUnlock();
            return DCGM_ST_INIT_ERROR;
        }
        else
        {
            DCGM_LOG_DEBUG << homeDirMsg;
        }
    }

    DCGM_LOG_INFO << DcgmNs::DcgmBuildInfo().GetBuildInfoStr();
    /* See if the host engine is running already */
    void *pHostEngineInstance = DcgmHostEngineHandler::Instance();
    if (pHostEngineInstance != nullptr)
    {
        g_dcgmGlobals.embeddedEngineStarted = 1; /* No harm in making sure this is true */
        dcgmGlobalsUnlock();
        DCGM_LOG_DEBUG << "dcgmStartEmbedded(): host engine was already running";
        return DCGM_ST_OK;
    }

    pHostEngineInstance = DcgmHostEngineHandler::Init(params);
    if (pHostEngineInstance == nullptr)
    {
        dcgmGlobalsUnlock();
        DCGM_LOG_ERROR << "DcgmHostEngineHandler::Init failed";
        return DCGM_ST_INIT_ERROR;
    }

    g_dcgmGlobals.embeddedEngineStarted = 1;

    dcgmGlobalsUnlock();

    params.dcgmHandle = (dcgmHandle_t)DCGM_EMBEDDED_HANDLE;
    DCGM_LOG_DEBUG << "dcgmStartEmbedded(): Embedded host engine started";

    return DCGM_ST_OK;
}

dcgmReturn_t StartEmbeddedV2(dcgmStartEmbeddedV2Params_v2 &params)
{
    dcgmStartEmbeddedV2Params_v1 proxyParams { .version       = dcgmStartEmbeddedV2Params_version1,
                                               .opMode        = params.opMode,
                                               .dcgmHandle    = params.dcgmHandle,
                                               .logFile       = params.logFile,
                                               .severity      = params.severity,
                                               .denyListCount = params.denyListCount,
                                               .denyList      = {} };

    memcpy(proxyParams.denyList, params.denyList, sizeof(proxyParams.denyList));

    if (auto dcgmResult = StartEmbeddedV2(proxyParams); dcgmResult != DCGM_ST_OK)
    {
        return dcgmResult;
    }

    params.dcgmHandle = proxyParams.dcgmHandle;

    auto *instance = DcgmHostEngineHandler::Instance();
    instance->SetServiceAccount(params.serviceAccount);

    return DCGM_ST_OK;
}

dcgmReturn_t StartEmbeddedV2(dcgmStartEmbeddedV2Params_v3 &params)
{
    dcgmStartEmbeddedV2Params_v2 proxyParams { .version    = dcgmStartEmbeddedV2Params_version2,
                                               .opMode     = params.opMode,
                                               .dcgmHandle = params.dcgmHandle,
                                               .logFile    = params.logFile,
                                               .severity   = params.severity,
                                               .denyListCount
                                               = std::min(params.denyListCount, (unsigned int)DCGM_MODULE_ID_COUNT_V1),
                                               .serviceAccount = params.serviceAccount,
                                               .denyList       = {} };

    // Copy as much of the denyList as will fit in v2
    memcpy(proxyParams.denyList, params.denyList, sizeof(proxyParams.denyList));

    if (auto dcgmResult = StartEmbeddedV2(proxyParams); dcgmResult != DCGM_ST_OK)
    {
        return dcgmResult;
    }

    params.dcgmHandle = proxyParams.dcgmHandle;

    // If there are more denyList entries than can fit in v2, handle them separately
    if (params.denyListCount > DCGM_MODULE_ID_COUNT_V1)
    {
        auto *instance = DcgmHostEngineHandler::Instance();

        dcgmReturn_t result = instance->ApplyModuleDenylist(&params.denyList[DCGM_MODULE_ID_COUNT_V1],
                                                            params.denyListCount - DCGM_MODULE_ID_COUNT_V1);

        if (result != DCGM_ST_OK)
        {
            return result;
        }
    }

    return DCGM_ST_OK;
}

/**
 * @brief Safely casts a constructor argument to the \c To type and back
 *
 * This class is meant to be used as a argument placeholder during a function call.
 * This is an adopted version of std::bit_cast that is more relaxed and unsafe - it does not check that the size of the
 * source and target types match.
 * This class copies memory from the memory the constructors \a inputArgument points to into an internal storage of the
 * type \c To.
 * When the object is destroyed the memory of the internal storage is copied back to the original location the \a
 * inputArguments points to.
 * Object of this class can be implicitly casted to the reference of the To type (aka To&), so any changes made to the
 * To& storage will be copied back to the original location \a inputArgument points to.
 * @example
 * @code{.cpp}
 *      struct ParamV1 { int version; ... };
 *      struct ParamV2 { int version; char name[255]; ... };
 *      void APIFunc(ParamV1 *ptr) {
 *          if (ptr->version == 1) Func(*ptr);
 *          else if (ptr->version == 2) Func(SafeArgumentCast<ParamV2>{ptr});
 *      }
 *      void Func(ParamV1& v1) {...}
 *      void Func(ParamV2& v2) {... memcpy(v2.name, "Hello", 6); }
 *
 *      void main() {
 *          ParamV1 v1 {.version = 1 };
 *          ParamV2 v2 {.version = 2; .name = {}};
 *          APIFunc(&v1);
 *          APIFunc((ParamV1*)&v2);
 *          assert("Hello" == v2.name);
 *      }
 * @endcode
 *
 * @tparam To The target type of the versioned argument. The \c To type is required to be trivially copyable and
 *            trivially constructible (aka trivial type). Additionally, the \c To should have unique object
 *            representation in memory - i.e. if \c To is a structure, it should not be padded. Also, it will not work
 *            if a structure has float or double fields.
 */
template <class To>
    requires std::is_trivial_v<To> && std::has_unique_object_representations_v<To>
class SafeArgumentCast
{
public:
    SafeArgumentCast(SafeArgumentCast const &)            = delete;
    SafeArgumentCast(SafeArgumentCast &&)                 = delete;
    SafeArgumentCast &operator=(SafeArgumentCast const &) = delete;
    SafeArgumentCast &operator=(SafeArgumentCast &&)      = delete;

    template <class From>
    explicit SafeArgumentCast(From *inputArgument)
        : m_target(inputArgument)
    {
        assert(inputArgument != nullptr);
        memcpy(&m_storage, m_target, sizeof(To));
    }

    ~SafeArgumentCast()
    {
        memcpy(m_target, &m_storage, sizeof(To));
    }

    operator To &() // NOLINT(google-explicit-constructor)
    {
        return m_storage;
    }

private:
    void *m_target;
    To m_storage {};
};

} // namespace

dcgmReturn_t DCGM_PUBLIC_API dcgmStartEmbedded_v2(dcgmStartEmbeddedV2Params_v1 *params)
{
    /*
     * Do not log anything with DCGM_LOG_* before the actual StartEmbeddedV2 function are called.
     * Those functions initialize the DCGM logger for the first time.
     */
    if (params == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    switch (params->version)
    {
        case dcgmStartEmbeddedV2Params_version1:
        {
            return StartEmbeddedV2(SafeArgumentCast<dcgmStartEmbeddedV2Params_v1> { params });
        }
        case dcgmStartEmbeddedV2Params_version2:
        {
            return StartEmbeddedV2(SafeArgumentCast<dcgmStartEmbeddedV2Params_v2> { params });
        }
        case dcgmStartEmbeddedV2Params_version3:
        {
            return StartEmbeddedV2(SafeArgumentCast<dcgmStartEmbeddedV2Params_v3> { params });
        }
        default:
            return DCGM_ST_VER_MISMATCH;
    }
}

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle)
{
    if (pDcgmHandle == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmStartEmbeddedV2Params_v1 params {};
    params.version  = dcgmStartEmbeddedV2Params_version1;
    params.opMode   = opMode;
    params.logFile  = nullptr;
    params.severity = DcgmLoggingSeverityUnspecified;

    dcgmReturn_t dcgmReturn = dcgmStartEmbedded_v2(&params);
    *pDcgmHandle            = params.dcgmHandle;
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmStopEmbedded(dcgmHandle_t pDcgmHandle)
{
    if (!g_dcgmGlobals.isInitialized)
    {
        log_error("dcgmStopEmbedded before dcgmInit()");
        return DCGM_ST_UNINITIALIZED;
    }
    if (pDcgmHandle != (dcgmHandle_t)DCGM_EMBEDDED_HANDLE)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmGlobalsLock();

    /* Check again after lock */
    if (!g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        log_error("dcgmStopEmbedded before dcgmInit() after lock");
        return DCGM_ST_UNINITIALIZED;
    }

    if (g_dcgmGlobals.embeddedEngineStarted)
    {
        DcgmHostEngineHandler *heHandler = DcgmHostEngineHandler::Instance();

        if (!heHandler)
        {
            log_error("embeddedEngineStarted was set but heHandler is NULL");
        }
        else
        {
            // Invoke the cleanup method
            (void)DcgmHostEngineHandler::Instance()->Cleanup();
            log_debug("embedded host engine cleaned up");
        }
        g_dcgmGlobals.embeddedEngineStarted = 0;
    }

    dcgmGlobalsUnlock();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmConnect(const char *ipAddress, dcgmHandle_t *pDcgmHandle)
{
    dcgmConnectV2Params_t connectParams;

    /* Set up default parameters for dcgmConnect_v2 */
    memset(&connectParams, 0, sizeof(connectParams));
    connectParams.version                = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = 0;

    return dcgmConnect_v2(ipAddress, &connectParams, pDcgmHandle);
}

/*****************************************************************************/
dcgmReturn_t cmSendClientLogin(dcgmHandle_t dcgmHandle, dcgmConnectV2Params_t *connectParams)
{
    if (!connectParams)
        return DCGM_ST_BADPARAM;

    dcgm_core_msg_client_login_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_CLIENT_LOGIN;
    msg.header.version    = dcgm_core_msg_client_login_version;

    msg.info.persistAfterDisconnect = connectParams->persistAfterDisconnect;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    return (dcgmReturn_t)msg.info.cmdRet;
}

/*****************************************************************************/
dcgmReturn_t sendClientLogin(dcgmHandle_t dcgmHandle, dcgmConnectV2Params_t *connectParams)
{
    dcgmReturn_t ret;

    /* First, try the module command version. If that fails, send the legacy protobuf version */
    ret = cmSendClientLogin(dcgmHandle, connectParams);
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmConnect_v2(const char *ipAddress,
                                            dcgmConnectV2Params_t *connectParams,
                                            dcgmHandle_t *pDcgmHandle)
{
    dcgmReturn_t dcgmReturn;
    dcgmConnectV2Params_v2 paramsCopy;

    if (!ipAddress || !ipAddress[0] || !pDcgmHandle || !connectParams)
        return DCGM_ST_BADPARAM;
    if (!g_dcgmGlobals.isInitialized)
    {
        log_error("dcgmConnect before dcgmInit()");
        return DCGM_ST_UNINITIALIZED;
    }

    /* Handle the old version by copying its parameters to the new version and changing the
       pointer to our local struct */
    if (connectParams->version == dcgmConnectV2Params_version1)
    {
        memset(&paramsCopy, 0, sizeof(paramsCopy));
        paramsCopy.version                = dcgmConnectV2Params_version;
        paramsCopy.persistAfterDisconnect = connectParams->persistAfterDisconnect;
        /* Other fields default to 0 from the memset above */
        connectParams = &paramsCopy;
    }
    else if (connectParams->version != dcgmConnectV2Params_version)
    {
        log_error("dcgmConnect_v2 Version mismatch {:X} != {:X}", connectParams->version, dcgmConnectV2Params_version);
        return DCGM_ST_VER_MISMATCH;
    }

    DcgmClientHandler *clientHandler = dcgmapiAcquireClientHandler(true);
    if (!clientHandler)
    {
        log_error("Unable to allocate client handler");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Add connection to the client handler */
    dcgmReturn_t status = clientHandler->GetConnHandleForHostEngine(
        ipAddress, pDcgmHandle, connectParams->timeoutMs, connectParams->addressIsUnixSocket ? true : false);
    dcgmapiReleaseClientHandler();
    if (DCGM_ST_OK != status)
    {
        log_error("GetConnHandleForHostEngine ip {} returned {}", ipAddress, (int)status);
        return status;
    }

    log_debug("Connected to ip {} as dcgmHandle {}", ipAddress, (void *)*pDcgmHandle);

    /* Send our connection options to the host engine */
    dcgmReturn = sendClientLogin(*pDcgmHandle, connectParams);
    if (dcgmReturn != DCGM_ST_OK)
    {
        /* Abandon the connection if we can't login */
        log_error("Got error {} from sendClientLogin on connection {}. Abandoning connection.",
                  (int)dcgmReturn,
                  (void *)*pDcgmHandle);
        return dcgmDisconnect(*pDcgmHandle);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmDisconnect(dcgmHandle_t pDcgmHandle)
{
    if (!g_dcgmGlobals.isInitialized)
    {
        log_warning("dcgmDisconnect before dcgmInit()");
        /* Returning OK here to prevent errors from being logged from the
           python framework when DcgmHandle objects are garbage collected after
           dcgmShutdown has already been called. */
        return DCGM_ST_OK;
    }

    DcgmClientHandler *clientHandler = dcgmapiAcquireClientHandler(false);
    if (!clientHandler)
    {
        log_warning("dcgmDisconnect called while client handler was not allocated.");
        /* Returning OK here to prevent errors from being logged from the
           python framework when DcgmHandle objects are garbage collected after
           dcgmShutdown has already been called. */
        return DCGM_ST_OK;
    }

    /* Actually close the connection */
    clientHandler->CloseConnForHostEngine(pDcgmHandle);

    dcgmapiReleaseClientHandler();

    log_debug("dcgmDisconnect closed connection with handle {}", (void *)pDcgmHandle);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmInit(void)
{
    if (g_dcgmGlobals.isInitialized)
    {
        log_debug("dcgmInit was already initialized");
        return DCGM_ST_OK;
    }

    dcgmGlobalsLock();

    /* Check again now that we have the lock */
    if (g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        log_debug("dcgmInit was already initialized after lock");
        return DCGM_ST_OK;
    }

    /* globals are uninitialized. Zero the structure */
    memset(&g_dcgmGlobals, 0, sizeof(g_dcgmGlobals));

    int ret = DcgmFieldsInit();
    if (ret != DCGM_ST_OK)
    {
        /* Undo any initialization done above */
        dcgmGlobalsUnlock();
        log_error("DcgmFieldsInit failed");
        return DCGM_ST_INIT_ERROR;
    }
    g_dcgmGlobals.fieldsAreInitialized = 1;

    /* Fully-initialized. Mark structure as such */
    g_dcgmGlobals.isInitialized = 1;

    dcgmGlobalsUnlock();

    log_debug("dcgmInit was successful");
    return DCGM_ST_OK;
}

/*****************************************************************************/

dcgmReturn_t DCGM_PUBLIC_API dcgmShutdown()
{
    if (!g_dcgmGlobals.isInitialized)
    {
        log_debug("dcgmShutdown called when DCGM was uninitialized.");
        return DCGM_ST_OK;
    }

    /* Clean up remote connections - must NOT have dcgmGlobalsLock() here or we will
       deadlock */
    log_debug("Before dcgmapiFreeClientHandler");
    dcgmapiFreeClientHandler();
    log_debug("After dcgmapiFreeClientHandler");

    dcgmGlobalsLock();

    if (!g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        log_debug("dcgmShutdown called when DCGM was uninitialized - after lock.");
        return DCGM_ST_UNINITIALIZED;
    }

    if (g_dcgmGlobals.embeddedEngineStarted)
    {
        DcgmHostEngineHandler *heHandler = DcgmHostEngineHandler::Instance();

        if (!heHandler)
        {
            log_error("embeddedEngineStarted was set but heHandler is NULL");
        }
        else
        {
            // Invoke the cleanup method
            (void)DcgmHostEngineHandler::Instance()->Cleanup();
            log_debug("host engine cleaned up");
        }
        g_dcgmGlobals.embeddedEngineStarted = 0;
    }

    DcgmFieldsTerm();
    g_dcgmGlobals.fieldsAreInitialized = 0;

    g_dcgmGlobals.isInitialized = 0;

    dcgmGlobalsUnlock();

    log_debug("dcgmShutdown completed successfully");

    return DCGM_ST_OK;
}

#define MODULE_CORE_NAME       "Core"
#define MODULE_NVSWITCH_NAME   "NvSwitch"
#define MODULE_VGPU_NAME       "VGPU"
#define MODULE_INTROSPECT_NAME "Introspection"
#define MODULE_HEALTH_NAME     "Health"
#define MODULE_POLICY_NAME     "Policy"
#define MODULE_CONFIG_NAME     "Config"
#define MODULE_DIAG_NAME       "Diag"
#define MODULE_PROFILING_NAME  "Profiling"
#define MODULE_SYSMON_NAME     "SysMon"
#define MODULE_MNDIAG_NAME     "MnDiag"

dcgmReturn_t tsapiDcgmModuleIdToName(dcgmModuleId_t id, char const **name)
{
    if (name == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    static const std::unordered_map<dcgmModuleId_t, char const *> moduleNames = {
        { DcgmModuleIdCore, MODULE_CORE_NAME },           { DcgmModuleIdNvSwitch, MODULE_NVSWITCH_NAME },
        { DcgmModuleIdVGPU, MODULE_VGPU_NAME },           { DcgmModuleIdIntrospect, MODULE_INTROSPECT_NAME },
        { DcgmModuleIdHealth, MODULE_HEALTH_NAME },       { DcgmModuleIdPolicy, MODULE_POLICY_NAME },
        { DcgmModuleIdConfig, MODULE_CONFIG_NAME },       { DcgmModuleIdDiag, MODULE_DIAG_NAME },
        { DcgmModuleIdProfiling, MODULE_PROFILING_NAME }, { DcgmModuleIdSysmon, MODULE_SYSMON_NAME },
        { DcgmModuleIdMnDiag, MODULE_MNDIAG_NAME },
    };

    assert(moduleNames.size() == DcgmModuleIdCount);

    auto const it = moduleNames.find(id);
    if (it == moduleNames.end())
    {
        return DCGM_ST_BADPARAM;
    }

    *name = it->second;
    return DCGM_ST_OK;
}

/*****************************************************************************/
static dcgmReturn_t HelperDcgmPauseResume(dcgmHandle_t pDcgmHandle, bool pause)
{
    dcgm_core_msg_pause_resume_v1 msg;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_PAUSE_RESUME;
    msg.header.version    = dcgm_core_msg_pause_resume_version1;
    msg.pause             = pause;

    // coverity[overrun-buffer-arg]
    return dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));
}

dcgmReturn_t DCGM_PUBLIC_API tsapiPause(dcgmHandle_t pDcgmHandle)
{
    return HelperDcgmPauseResume(pDcgmHandle, true);
}

dcgmReturn_t DCGM_PUBLIC_API tsapiResume(dcgmHandle_t pDcgmHandle)
{
    return HelperDcgmPauseResume(pDcgmHandle, false);
}

dcgmReturn_t DCGM_PUBLIC_API tsapiNvswitchGetBackend(dcgmHandle_t pDcgmHandle,
                                                     bool *active,
                                                     char *backendName,
                                                     unsigned int backendNameLength)
{
    dcgm_core_msg_nvswitch_get_backend_t msg {};

    if (sizeof(msg.backendName) > backendNameLength)
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    backendNameLength = std::min(backendNameLength, (unsigned int)sizeof(msg.backendName));

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_NVSWITCH_GET_BACKEND;
    msg.header.version    = dcgm_core_msg_nvswitch_get_backend_version;

    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    *active = msg.active;
    snprintf(backendName, backendNameLength, "%s", msg.backendName);

    return ret;
}

/*****************************************************************************/
