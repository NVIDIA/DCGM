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
#include "dcgm_test_apis.h"
#include "dcgm_util.h"
#include "nvcmvalue.h"
#include <cstdint>
#include <cstdio>

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
#include "dcgm_nvswitch_structs.h"
#include "dcgm_policy_structs.h"
#include "dcgm_profiling_structs.h"
#include <DcgmStringHelpers.h>

// Wrap each dcgmFunction with apiEnter and apiExit
#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)                                 \
    static dcgmReturn_t tsapiFuncname argtypes;                                                           \
    dcgmReturn_t DCGM_PUBLIC_API dcgmFuncname argtypes                                                    \
    {                                                                                                     \
        dcgmReturn_t result;                                                                              \
        PRINT_DEBUG("Entering %s%s " fmt, "Entering %s%s " fmt, #dcgmFuncname, #argtypes, ##__VA_ARGS__); \
        result = apiEnter();                                                                              \
        if (result != DCGM_ST_OK)                                                                         \
        {                                                                                                 \
            return result;                                                                                \
        }                                                                                                 \
        try                                                                                               \
        {                                                                                                 \
            result = tsapiFuncname(__VA_ARGS__);                                                          \
        }                                                                                                 \
        catch (const std::exception &e)                                                                   \
        {                                                                                                 \
            DCGM_LOG_ERROR << "Caught exception " << e.what();                                            \
            result = DCGM_ST_GENERIC_ERROR;                                                               \
        }                                                                                                 \
        catch (...)                                                                                       \
        {                                                                                                 \
            DCGM_LOG_ERROR << "Unknown exception ";                                                       \
            result = DCGM_ST_GENERIC_ERROR;                                                               \
        }                                                                                                 \
        apiExit();                                                                                        \
        PRINT_DEBUG("Returning %d", "Returning %d", result);                                              \
        return result;                                                                                    \
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
int helperUpdateErrorCodes(dcgmStatus_t statusHandle, dcgm::Command *pGroupCmd)
{
    DcgmStatus *pStatusObj;
    int index;

    if (((dcgmStatus_t) nullptr == statusHandle) || (nullptr == pGroupCmd))
    {
        DCGM_LOG_ERROR << "Got null statusHandle or pGroupCmd";
        return -1;
    }

    pStatusObj = (DcgmStatus *)statusHandle;


    for (index = 0; index < pGroupCmd->errlist_size(); index++)
    {
        pStatusObj->Enqueue(pGroupCmd->errlist(index).gpuid(),
                            pGroupCmd->errlist(index).fieldid(),
                            pGroupCmd->errlist(index).errorcode());
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
        PRINT_DEBUG("%d", "Incremented the client handler to %d", g_dcgmGlobals.clientHandlerRefCount);
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
            PRINT_ERROR("%d",
                        "Client handler ref count underflowed. Tried to decrement from %d",
                        g_dcgmGlobals.clientHandlerRefCount);
        }
        else
        {
            g_dcgmGlobals.clientHandlerRefCount--;
            PRINT_DEBUG("%d", "Decremented the client handler to %d", g_dcgmGlobals.clientHandlerRefCount);
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
            PRINT_INFO("%d",
                       "Waiting to destroy the client handler. Current refCount: %d",
                       g_dcgmGlobals.clientHandlerRefCount);
            sleep(1);
        }

        dcgmGlobalsLock();

        /* Now that we have the lock, we have to re-check state */

        if (!g_dcgmGlobals.clientHandler)
        {
            /* Another thread did our work for us. Unlock and get out */
            PRINT_INFO("", "Another thread freed the client handler for us.");
            dcgmGlobalsUnlock();
            return;
        }
        if (g_dcgmGlobals.clientHandlerRefCount > 0)
        {
            /* Someone else got the lock and incremented the ref count while we were sleeping. Start over */
            dcgmGlobalsUnlock();
            PRINT_INFO("", "Another thread acquired the client handler while we were sleeping.");
            continue;
        }

        delete g_dcgmGlobals.clientHandler;
        g_dcgmGlobals.clientHandler = NULL;
        dcgmGlobalsUnlock();
        PRINT_INFO("", "Freed the client handler");
        break;
    }
}

/*****************************************************************************/
dcgmReturn_t processProtobufAtRemoteHostEngine(dcgmHandle_t pDcgmHandle,
                                               DcgmProtobuf *pEncodePrb,
                                               DcgmProtobuf *pDecodePrb,
                                               std::vector<dcgm::Command *> *pVecCmds,
                                               std::unique_ptr<DcgmRequest> request = nullptr,
                                               unsigned int timeout                 = 60000)
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

    /* Invoke method on the client side */
    ret = clientHandler->ExchangeMsgAsync(pDcgmHandle, pEncodePrb, pDecodePrb, pVecCmds, std::move(request), timeout);

    dcgmapiReleaseClientHandler();

    return ret;
}

/*****************************************************************************/
dcgmReturn_t processProtobufAtEmbeddedHostEngine(DcgmProtobuf *pEncodePrb,
                                                 std::vector<dcgm::Command *> *pVecCmds,
                                                 std::unique_ptr<DcgmRequest> request = nullptr)
{
    DcgmHostEngineHandler *pHEHandlerInstance = nullptr;
    dcgmReturn_t ret;
    dcgm_request_id_t requestId = 0;

    /* Get Instance to Host Engine Handler */
    pHEHandlerInstance = DcgmHostEngineHandler::Instance();
    if (nullptr == pHEHandlerInstance)
    {
        DCGM_LOG_ERROR << "DcgmHostEngineHandler::Instance() returned nullptr";
        return DCGM_ST_UNINITIALIZED;
    }

    /* Get Vector of commands from the protobuf messages */
    if (0 != pEncodePrb->GetAllCommands(pVecCmds))
    {
        /* This should never happen */
        DCGM_LOG_ERROR << "GetAllCommands failed";
        return DCGM_ST_GENERIC_ERROR;
    }

    if (request != nullptr)
    {
        /* Subscribe to be updated when this request updates. This will also assign request->requestId */
        ret = pHEHandlerInstance->AddRequestWatcher(std::move(request), requestId);
        if (ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "AddRequestWatcher returned %d", ret);
            return ret;
        }
    }

    /* Invoke Request handler method on the host engine */
    ret = (dcgmReturn_t)pHEHandlerInstance->HandleCommands(pVecCmds, DCGM_CONNECTION_ID_NONE, requestId);
    return ret;
}

/*****************************************************************************/
dcgmReturn_t processAtHostEngine(dcgmHandle_t pDcgmHandle,
                                 DcgmProtobuf *encodePrb,
                                 DcgmProtobuf *decodePrb,
                                 std::vector<dcgm::Command *> *vecCmdsRef,
                                 std::unique_ptr<DcgmRequest> request = nullptr,
                                 unsigned int timeout                 = 60000)
{
    if (pDcgmHandle != (dcgmHandle_t)DCGM_EMBEDDED_HANDLE) /* Remote DCGM */
    {
        return processProtobufAtRemoteHostEngine(
            pDcgmHandle, encodePrb, decodePrb, vecCmdsRef, std::move(request), timeout);
    }
    else /* Implies Embedded HE mode. ISV Case */
    {
        return processProtobufAtEmbeddedHostEngine(encodePrb, vecCmdsRef, std::move(request));
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
            PRINT_ERROR("%d", "AddRequestWatcher returned %d", ret);
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
dcgmReturn_t helperGroupCreate(dcgmHandle_t pDcgmHandle,
                               dcgmGroupType_t type,
                               const char *groupName,
                               dcgmGpuGrp_t *pDcgmGrpId)
{
    dcgm::GroupInfo *pGroupInfo;             /* Protobuf equivalent structure of the output parameter. */
    dcgm::GroupInfo *pGroupInfoOut;          /* Protobuf equivalent structure of the output parameter. */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if ((groupName == NULL) || (pDcgmGrpId == NULL))
    {
        return DCGM_ST_BADPARAM;
    }
    unsigned int length = strlen(groupName);

    /* Update the desired group type and group name*/
    pGroupInfo = new dcgm::GroupInfo;
    pGroupInfo->set_grouptype(type);
    pGroupInfo->set_groupname(groupName, length);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GROUP_CREATE, dcgm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp)
    {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_grpinfo()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Update the Protobuf reference with the results */
    pGroupInfoOut = vecCmdsRef[0]->mutable_arg(0)->mutable_grpinfo();

    if (pGroupInfoOut->has_groupid())
    {
        *pDcgmGrpId = (dcgmGpuGrp_t)(long long)pGroupInfoOut->groupid();
    }
    else
    {
        PRINT_ERROR("", "Failed to create group");
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperGroupCreate(dcgmHandle_t pDcgmHandle,
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
dcgmReturn_t helperGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t grpId)
{
    dcgm::GroupInfo *pGroupInfo;             /* Protobuf equivalent structure of the output parameter. */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    /* Set group ID to be removed from the hostengine */
    pGroupInfo = new dcgm::GroupInfo;
    pGroupInfo->set_groupid((intptr_t)grpId);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GROUP_DESTROY, dcgm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp)
    {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                  dcgmGpuGrp_t groupId,
                                  dcgm_field_entity_group_t entityGroupId,
                                  dcgm_field_eid_t entityId)
{
    dcgm::GroupInfo *pGroupInfo;             /* Protobuf equivalent structure of the output parameter. */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;


    pGroupInfo = new dcgm::GroupInfo;
    pGroupInfo->set_groupid((intptr_t)groupId);

    dcgm::EntityIdPair *eidPair = pGroupInfo->add_entity();
    eidPair->set_entitygroupid((unsigned int)entityGroupId);
    eidPair->set_entityid((unsigned int)entityId);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GROUP_ADD_DEVICE, dcgm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp)
    {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t grpId)
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
    return helperGroupAddEntity(pDcgmHandle, grpId, DCGM_FE_GPU, gpuId);
}

/*****************************************************************************/
dcgmReturn_t tsapiGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId)
{
    return helperGroupAddEntity(pDcgmHandle, groupId, entityGroupId, entityId);
}

/*****************************************************************************/
dcgmReturn_t helperGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                     dcgmGpuGrp_t groupId,
                                     dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId)
{
    dcgm::GroupInfo *pGroupInfo;             /* Protobuf equivalent structure of the output parameter. */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    /* Set group ID to be removed from the hostengine */
    pGroupInfo = new dcgm::GroupInfo;
    pGroupInfo->set_groupid((intptr_t)groupId);

    dcgm::EntityIdPair *eidPair = pGroupInfo->add_entity();
    eidPair->set_entitygroupid((unsigned int)entityGroupId);
    eidPair->set_entityid((unsigned int)entityId);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GROUP_REMOVE_DEVICE, dcgm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp)
    {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
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
    return helperGroupRemoveEntity(pDcgmHandle, grpId, DCGM_FE_GPU, gpuId);
}

/*****************************************************************************/
dcgmReturn_t tsapiGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId)
{
    return helperGroupRemoveEntity(pDcgmHandle, groupId, entityGroupId, entityId);
}

/*****************************************************************************/
dcgmReturn_t helperGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t *pGroupIdList, unsigned int *pCount)
{
    dcgm::FieldMultiValues *pListGrpIdsOutput;
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if ((NULL == pGroupIdList) || (NULL == pCount))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GROUP_GETALL_IDS, dcgm::OPERATION_SYSTEM, -1, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_fieldmultivalues()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Update the Protobuf reference with the results */
    pListGrpIdsOutput = vecCmdsRef[0]->mutable_arg(0)->mutable_fieldmultivalues();

    *pCount = pListGrpIdsOutput->vals_size();
    for (int index = 0; index < pListGrpIdsOutput->vals_size(); index++)
    {
        pGroupIdList[index] = (dcgmGpuGrp_t)pListGrpIdsOutput->mutable_vals(index)->i64();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t *pGroupIdList, unsigned int *pCount)
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

    for (int i = 0; i < msg.groups.numGroups; i++)
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
    dcgm::GroupInfo *pGroupInfo;             /* Protobuf equivalent structure of the output parameter. */
    dcgm::GroupInfo *pGroupInfoOut;          /* Protobuf equivalent structure of the output parameter. */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    /* Input parameter validation */
    if (NULL == pDcgmGroupInfo)
    {
        PRINT_ERROR("", "NULL pDcgmGroupInfo");
        return DCGM_ST_BADPARAM;
    }

    /* Check for version */
    if ((pDcgmGroupInfo->version < dcgmGroupInfo_version2) || (pDcgmGroupInfo->version > dcgmGroupInfo_version))
    {
        PRINT_ERROR("%X", "helperGroupGetInfo version mismatch on x%X", pDcgmGroupInfo->version);
        return DCGM_ST_VER_MISMATCH;
    }

    pGroupInfo = new dcgm::GroupInfo;
    pGroupInfo->set_groupid((intptr_t)groupId);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GROUP_INFO, dcgm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp)
    {
        delete pGroupInfo;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_grpinfo(pGroupInfo);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }


    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_grpinfo()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Update the Protobuf reference with the results */
    pGroupInfoOut = vecCmdsRef[0]->mutable_arg(0)->mutable_grpinfo();


    if (pGroupInfoOut->has_groupname())
    {
        size_t length;
        length = strlen(pGroupInfoOut->groupname().c_str());
        if (length + 1 > DCGM_MAX_STR_LENGTH)
        {
            PRINT_ERROR("", "String overflow error for the requested field");
            return DCGM_ST_MEMORY;
        }

        dcgmStrncpy(pDcgmGroupInfo->groupName, pGroupInfoOut->groupname().c_str(), sizeof(pDcgmGroupInfo->groupName));
    }
    else
    {
        PRINT_ERROR("", "Can't find group name in the returned info from the hostengine");
        return DCGM_ST_GENERIC_ERROR;
    }

    if (pGroupInfoOut->entity_size() > DCGM_GROUP_MAX_ENTITIES)
    {
        PRINT_ERROR("", "Invalid number of GPU Ids returned from the hostengine");
        return DCGM_ST_GENERIC_ERROR;
    }

    if (hostEngineTimestamp)
    {
        if (vecCmdsRef[0]->has_timestamp())
        {
            *hostEngineTimestamp = vecCmdsRef[0]->timestamp();
        }
        else
        {
            PRINT_ERROR("", "No timestamp in command. Caller requires one.");
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    pDcgmGroupInfo->count = pGroupInfoOut->entity_size();

    for (int index = 0; index < pGroupInfoOut->entity_size(); index++)
    {
        const dcgm::EntityIdPair eidPair = pGroupInfoOut->entity(index);

        pDcgmGroupInfo->entityList[index].entityGroupId = (dcgm_field_entity_group_t)eidPair.entitygroupid();
        pDcgmGroupInfo->entityList[index].entityId      = (dcgm_field_eid_t)eidPair.entityid();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperGroupGetInfo(dcgmHandle_t pDcgmHandle,
                                  dcgmGpuGrp_t groupId,
                                  dcgmGroupInfo_t *pDcgmGroupInfo,
                                  long long *hostEngineTimestamp)
{
    if (!pDcgmGroupInfo)
    {
        return DCGM_ST_BADPARAM;
    }

    /* Check for version */
    if ((pDcgmGroupInfo->version < dcgmGroupInfo_version2) || (pDcgmGroupInfo->version > dcgmGroupInfo_version))
    {
        DCGM_LOG_ERROR << "Struct version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    dcgmReturn_t ret;

    dcgm_core_msg_group_get_info_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GROUP_GET_INFO;
    msg.header.version    = dcgm_core_msg_group_get_info_version;

    msg.gi.groupId = groupId;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (msg.gi.cmdRet != DCGM_ST_OK)
    {
        return (dcgmReturn_t)msg.gi.cmdRet;
    }

    dcgmStrncpy(pDcgmGroupInfo->groupName, msg.gi.groupInfo.groupName, sizeof(pDcgmGroupInfo->groupName));

    if (hostEngineTimestamp)
    {
        *hostEngineTimestamp = msg.gi.timestamp;
    }

    pDcgmGroupInfo->count = msg.gi.groupInfo.count;

    if (pDcgmGroupInfo->count > DCGM_GROUP_MAX_ENTITIES)
    {
        DCGM_LOG_ERROR << "Invalid number of GPU Ids returned from the hostengine";
        return DCGM_ST_GENERIC_ERROR;
    }

    for (int index = 0; index < msg.gi.groupInfo.count; index++)
    {
        pDcgmGroupInfo->entityList[index].entityGroupId = msg.gi.groupInfo.entityList[index].entityGroupId;
        pDcgmGroupInfo->entityList[index].entityId      = msg.gi.groupInfo.entityList[index].entityId;
    }

    return DCGM_ST_OK;
}

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
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    dcgmGetMultipleLatestValues_t msg;

    if ((entityList && !entityListCount) || (fieldIdList && !fieldIdListCount) || !fvBuffer
        || entityListCount > DCGM_GROUP_MAX_ENTITIES || fieldIdListCount > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to be sent over the network */
    pCmdTemp = encodePrb.AddCommand(dcgm::GET_MULTIPLE_LATEST_VALUES, dcgm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp)
    {
        PRINT_ERROR("", "encodePrb.AddCommand failed.");
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgm::CmdArg *pCmdArg = pCmdTemp->add_arg();

    memset(&msg, 0, sizeof(msg));
    msg.version = dcgmGetMultipleLatestValues_version;
    msg.flags   = flags;

    if (entityList)
    {
        memmove(&msg.entities[0], entityList, entityListCount * sizeof(entityList[0]));
        msg.entitiesCount = entityListCount;
    }
    else
    {
        msg.groupId = groupId;
    }

    if (fieldIdList)
    {
        memmove(&msg.fieldIds[0], fieldIdList, fieldIdListCount * sizeof(fieldIdList[0]));
        msg.fieldIdCount = fieldIdListCount;
    }
    else
    {
        msg.fieldGroupId = fieldGroupId;
    }

    pCmdArg->set_blob(&msg, sizeof(msg));

    ret = processAtHostEngine(dcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        PRINT_ERROR("%d", "processAtHostEngine returned %d", (int)ret);
        return ret;
    }

    if (vecCmdsRef.size() < 1 || vecCmdsRef[0]->arg_size() < 1)
    {
        PRINT_ERROR("", "Malformed GET_MULTIPLE_LATEST_VALUES response 1.");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Did the request return a global request error (vs a field value status)? */
    if (vecCmdsRef[0]->has_status() && vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got message status %d", (int)vecCmdsRef[0]->status());
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    if (!vecCmdsRef[0]->arg(0).has_blob())
    {
        PRINT_ERROR("", "Malformed GET_MULTIPLE_LATEST_VALUES missing blob.");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Make a FV buffer from our protobuf string */
    fvBuffer->SetFromBuffer(vecCmdsRef[0]->arg(0).blob().c_str(), vecCmdsRef[0]->arg(0).blob().size());
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
dcgmReturn_t cmHelperGetLatestValuesForFields(dcgmHandle_t dcgmHandle,
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

    dcgm_core_msg_entities_get_latest_values_t msg = {};

    msg.header.length = sizeof(msg) - SAMPLES_BUFFER_SIZE; /* avoid transferring the large buffer when making request */
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES;
    msg.header.version    = dcgm_core_msg_entities_get_latest_values_version;

    if ((entityList && !entityListCount) || (fieldIdList && !fieldIdListCount) || !fvBuffer
        || entityListCount > DCGM_GROUP_MAX_ENTITIES || fieldIdListCount > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
    {
        DCGM_LOG_ERROR << "Bad parameter";
        return DCGM_ST_BADPARAM;
    }

    msg.ev.flags = flags;

    if (entityList)
    {
        memmove(&msg.ev.entities[0], entityList, entityListCount * sizeof(entityList[0]));
        msg.ev.entitiesCount = entityListCount;
    }
    else
    {
        msg.ev.groupId = groupId;
    }

    if (fieldIdList)
    {
        memmove(&msg.ev.fieldIdList[0], fieldIdList, fieldIdListCount * sizeof(fieldIdList[0]));
        msg.ev.fieldIdCount = fieldIdListCount;
    }
    else
    {
        msg.ev.fieldGroupId = fieldGroupId;
    }

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "dcgmModuleSendBlockingFixedRequest returned " << ret;
        return ret;
    }

    /* Did the request return a global request error (vs a field value status)? */
    if (DCGM_ST_OK != msg.ev.cmdRet)
    {
        DCGM_LOG_ERROR << "Got message status " << msg.ev.cmdRet;
        return (dcgmReturn_t)msg.ev.cmdRet;
    }

    /* Make a FV buffer from our protobuf string */
    fvBuffer->SetFromBuffer(msg.ev.buffer, msg.ev.bufferSize);
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

dcgmReturn_t helperGetAllDevices(dcgmHandle_t pDcgmHandle, unsigned int *pGpuIdList, int *pCount, int onlySupported)
{
    dcgm::FieldMultiValues *pListGpuIdsOutput;
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    dcgm::CmdArg *cmdArg = 0;

    if ((NULL == pGpuIdList) || (NULL == pCount))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::DISCOVER_DEVICES, dcgm::OPERATION_SYSTEM, -1, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }


    cmdArg = pCmdTemp->add_arg();
    /* Use the int32 parameter to pass "onlySupported" */
    cmdArg->set_i32(onlySupported);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_fieldmultivalues()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Update the Protobuf reference with the results */
    pListGpuIdsOutput = vecCmdsRef[0]->mutable_arg(0)->mutable_fieldmultivalues();

    *pCount = pListGpuIdsOutput->vals_size();
    for (int index = 0; index < pListGpuIdsOutput->vals_size(); index++)
    {
        pGpuIdList[index] = pListGpuIdsOutput->mutable_vals(index)->i64();
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

    if (count >= sizeof(fieldIds) / sizeof(fieldIds[0]))
    {
        PRINT_ERROR("", "Update DeviceGetAttributes to accommodate more fields\n");
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
        PRINT_ERROR("%d %d %d", "Unexpected elementCount %d != count %d or ret %d", (int)elementCount, count, (int)ret);
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
                    PRINT_ERROR("", "String overflow error for the requested UUID field");
                    dcgmStrncpy(
                        pDcgmDeviceAttr->identifiers.uuid, DCGM_STR_BLANK, sizeof(pDcgmDeviceAttr->identifiers.uuid));
                }
                else
                {
                    dcgmStrncpy(
                        pDcgmDeviceAttr->identifiers.uuid, fv->value.str, sizeof(pDcgmDeviceAttr->identifiers.uuid));
                }

                break;
            }

            case DCGM_FI_DEV_VBIOS_VERSION:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.vbios))
                {
                    PRINT_ERROR("", "String overflow error for the requested VBIOS field");
                    dcgmStrncpy(
                        pDcgmDeviceAttr->identifiers.vbios, DCGM_STR_BLANK, sizeof(pDcgmDeviceAttr->identifiers.vbios));
                }
                else
                {
                    dcgmStrncpy(
                        pDcgmDeviceAttr->identifiers.vbios, fv->value.str, sizeof(pDcgmDeviceAttr->identifiers.vbios));
                }

                break;
            }

            case DCGM_FI_DEV_INFOROM_IMAGE_VER:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.inforomImageVersion))
                {
                    PRINT_ERROR("", "String overflow error for the requested Inforom field");
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.inforomImageVersion,
                                DCGM_STR_BLANK,
                                sizeof(pDcgmDeviceAttr->identifiers.inforomImageVersion));
                }
                else
                {
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.inforomImageVersion,
                                fv->value.str,
                                sizeof(pDcgmDeviceAttr->identifiers.inforomImageVersion));
                }

                break;
            }

            case DCGM_FI_DEV_BRAND:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.brandName))
                {
                    PRINT_ERROR("", "String overflow error for the requested brand name field");
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.brandName,
                                DCGM_STR_BLANK,
                                sizeof(pDcgmDeviceAttr->identifiers.brandName));
                }
                else
                {
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.brandName,
                                fv->value.str,
                                sizeof(pDcgmDeviceAttr->identifiers.brandName));
                }

                break;
            }

            case DCGM_FI_DEV_NAME:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.deviceName))
                {
                    PRINT_ERROR("", "String overflow error for the requested device name field");
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.deviceName,
                                DCGM_STR_BLANK,
                                sizeof(pDcgmDeviceAttr->identifiers.deviceName));
                }
                else
                {
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.deviceName,
                                fv->value.str,
                                sizeof(pDcgmDeviceAttr->identifiers.deviceName));
                }

                break;
            }

            case DCGM_FI_DEV_SERIAL:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.serial))
                {
                    PRINT_ERROR("", "String overflow error for the requested serial field");
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.serial,
                                DCGM_STR_BLANK,
                                sizeof(pDcgmDeviceAttr->identifiers.serial));
                }
                else
                {
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.serial,
                                fv->value.str,
                                sizeof(pDcgmDeviceAttr->identifiers.serial));
                }

                break;
            }

            case DCGM_FI_DEV_PCI_BUSID:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmDeviceAttr->identifiers.pciBusId))
                {
                    PRINT_ERROR("", "String overflow error for the requested serial field");
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.pciBusId,
                                DCGM_STR_BLANK,
                                sizeof(pDcgmDeviceAttr->identifiers.pciBusId));
                }
                else
                {
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.pciBusId,
                                fv->value.str,
                                sizeof(pDcgmDeviceAttr->identifiers.pciBusId));
                }

                break;
            }

            case DCGM_FI_DEV_SUPPORTED_CLOCKS:
            {
                dcgmDeviceSupportedClockSets_t *supClocks = (dcgmDeviceSupportedClockSets_t *)fv->value.blob;

                if (!supClocks)
                {
                    memset(&pDcgmDeviceAttr->clockSets, 0, sizeof(pDcgmDeviceAttr->clockSets));
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_SUPPORTED_CLOCKS");
                }
                else if (supClocks->version != dcgmDeviceSupportedClockSets_version)
                {
                    memset(&pDcgmDeviceAttr->clockSets, 0, sizeof(pDcgmDeviceAttr->clockSets));
                    PRINT_ERROR("%d %d",
                                "Expected dcgmDeviceSupportedClockSets_version %d. Got %d",
                                (int)dcgmDeviceSupportedClockSets_version,
                                (int)supClocks->version);
                }
                else
                {
                    int payloadSize = (sizeof(*supClocks) - sizeof(supClocks->clockSet))
                                      + (supClocks->count * sizeof(supClocks->clockSet[0]));
                    if (payloadSize > (int)(fv->length - (sizeof(*fv) - sizeof(fv->value))))
                    {
                        PRINT_ERROR("%d %d",
                                    "DCGM_FI_DEV_SUPPORTED_CLOCKS calculated size %d > possible size %d",
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
                    PRINT_ERROR("", "String overflow error for the requested driver version field");
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.driverVersion,
                                DCGM_STR_BLANK,
                                sizeof(pDcgmDeviceAttr->identifiers.driverVersion));
                }
                else
                {
                    dcgmStrncpy(pDcgmDeviceAttr->identifiers.driverVersion,
                                fv->value.str,
                                sizeof(pDcgmDeviceAttr->identifiers.driverVersion));
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
    dcgm::WatchFieldValue *pProtoWatchFieldValue; /* Protobuf Arg */
    DcgmProtobuf encodePrb;                       /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                       /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                      /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef;      /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (!fieldId || updateFreq <= 0 || (maxKeepSamples <= 0 && maxKeepAge <= 0.0))
        return DCGM_ST_BADPARAM;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById((unsigned short)fieldId);
    if (NULL == fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "field ID %u is not a valid field ID", fieldId);
        return DCGM_ST_BADPARAM;
    }

    pProtoWatchFieldValue = new dcgm::WatchFieldValue;
    pProtoWatchFieldValue->set_version(dcgmWatchFieldValue_version);
    pProtoWatchFieldValue->set_fieldid(fieldId);
    pProtoWatchFieldValue->set_updatefreq(updateFreq);
    pProtoWatchFieldValue->set_maxkeepage(maxKeepAge);
    pProtoWatchFieldValue->set_maxkeepsamples(maxKeepSamples);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::WATCH_FIELD_VALUE, dcgm::OPERATION_SINGLE_ENTITY, gpuId, 0);
    if (NULL == pCmdTemp)
    {
        delete pProtoWatchFieldValue;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the entityGroupId */
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;
    if (fieldMeta->scope == DCGM_FS_GLOBAL)
        entityGroupId = DCGM_FE_NONE;
    pCmdTemp->set_entitygroupid((int)entityGroupId);


    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_watchfieldvalue(pProtoWatchFieldValue);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperWatchFieldValue(dcgmHandle_t pDcgmHandle,
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

    dcgm_core_msg_watch_field_value_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_WATCH_FIELD_VALUE;
    msg.header.version    = dcgm_core_msg_watch_field_value_version;

    msg.fv.fieldId        = fieldId;
    msg.fv.gpuId          = gpuId;
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
    dcgm::UpdateAllFields *pProtoUpdateAllFields; /* Protobuf Arg */
    DcgmProtobuf encodePrb;                       /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                       /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                      /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef;      /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    pProtoUpdateAllFields = new dcgm::UpdateAllFields;
    pProtoUpdateAllFields->set_version(dcgmUpdateAllFields_version);
    pProtoUpdateAllFields->set_waitforupdate(waitForUpdate);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::UPDATE_ALL_FIELDS, dcgm::OPERATION_SINGLE_ENTITY, -1, 0);
    if (NULL == pCmdTemp)
    {
        delete pProtoUpdateAllFields;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_updateallfields(pProtoUpdateAllFields);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate)
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
    unsigned short fieldIds[32];
    unsigned int count = 0;
    dcgmReturn_t ret;
    long long updateFreq = 30000000;
    double maxKeepAge    = 14400.0;
    int maxKeepSamples   = 480;

    if (NULL == pDcgmVgpuDeviceAttr)
    {
        return DCGM_ST_BADPARAM;
    }

    if ((pDcgmVgpuDeviceAttr->version < dcgmVgpuDeviceAttributes_version6)
        || (pDcgmVgpuDeviceAttr->version > dcgmVgpuDeviceAttributes_version))
    {
        return DCGM_ST_VER_MISMATCH;
    }

    fieldIds[count++] = DCGM_FI_DEV_SUPPORTED_TYPE_INFO;
    fieldIds[count++] = DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_INSTANCE_IDS;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_UTILIZATIONS;
    fieldIds[count++] = DCGM_FI_DEV_GPU_UTIL;
    fieldIds[count++] = DCGM_FI_DEV_MEM_COPY_UTIL;
    fieldIds[count++] = DCGM_FI_DEV_ENC_UTIL;
    fieldIds[count++] = DCGM_FI_DEV_DEC_UTIL;

    if (count >= 32)
    {
        PRINT_ERROR("", "Update DeviceGetAttributes to accommodate more fields\n");
        return DCGM_ST_GENERIC_ERROR;
    }

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
        PRINT_ERROR("%d %d %d", "Unexpected elementCount %d != count %d or ret %d", (int)elementCount, count, (int)ret);
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
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS");
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
                    PRINT_ERROR("%d %d",
                                "vGPU Type ID static info array size %d too small for %d vGPU static info",
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
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS");
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
                    PRINT_ERROR("%d %d",
                                "Creatable vGPU Type IDs array size %d too small for %d Id value",
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
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_VGPU_INSTANCE_IDS");
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
                    PRINT_ERROR("%d %d",
                                "Active vGPU Instance IDs array size %d too small for %d Id value",
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
                    PRINT_ERROR("", "Null field value for DCGM_FI_DEV_VGPU_UTILIZATIONS");
                    break;
                }

                if (sizeof(pDcgmVgpuDeviceAttr->vgpuUtilInfo)
                    < sizeof(*vgpuUtilInfo) * (pDcgmVgpuDeviceAttr->activeVgpuInstanceCount))
                {
                    memset(&pDcgmVgpuDeviceAttr->vgpuUtilInfo, 0, sizeof(pDcgmVgpuDeviceAttr->vgpuUtilInfo));
                    PRINT_ERROR("%d %d",
                                "Active vGPU Instance IDs utilizations array size %d too small for %d Id value",
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
    unsigned short fieldIds[32];
    unsigned int count = 0;
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

    fieldIds[count++] = DCGM_FI_DEV_VGPU_VM_ID;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_VM_NAME;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_TYPE;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_UUID;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_DRIVER_VERSION;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_MEMORY_USAGE;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_LICENSE_INSTANCE_STATUS;
    fieldIds[count++] = DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT;

    if (count >= 32)
    {
        PRINT_ERROR("", "Update DeviceGetAttributes to accommodate more fields\n");
        return DCGM_ST_GENERIC_ERROR;
    }

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
                    PRINT_ERROR("", "String overflow error for the requested vGPU instance VM ID field");
                    dcgmStrncpy(pDcgmVgpuInstanceAttr->vmId, DCGM_STR_BLANK, sizeof(pDcgmVgpuInstanceAttr->vmId));
                }
                else
                {
                    dcgmStrncpy(pDcgmVgpuInstanceAttr->vmId, fv->value.str, sizeof(pDcgmVgpuInstanceAttr->vmId));
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_VM_NAME:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmVgpuInstanceAttr->vmName))
                {
                    PRINT_ERROR("", "String overflow error for the requested vGPU instance VM name field");
                    dcgmStrncpy(pDcgmVgpuInstanceAttr->vmName, DCGM_STR_BLANK, sizeof(pDcgmVgpuInstanceAttr->vmName));
                }
                else
                {
                    dcgmStrncpy(pDcgmVgpuInstanceAttr->vmName, fv->value.str, sizeof(pDcgmVgpuInstanceAttr->vmName));
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
                    PRINT_ERROR("", "String overflow error for the requested vGPU instance UUID field");
                    dcgmStrncpy(
                        pDcgmVgpuInstanceAttr->vgpuUuid, DCGM_STR_BLANK, sizeof(pDcgmVgpuInstanceAttr->vgpuUuid));
                }
                else
                {
                    dcgmStrncpy(
                        pDcgmVgpuInstanceAttr->vgpuUuid, fv->value.str, sizeof(pDcgmVgpuInstanceAttr->vgpuUuid));
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_DRIVER_VERSION:
            {
                size_t length;
                length = strlen(fv->value.str);
                if (length + 1 > sizeof(pDcgmVgpuInstanceAttr->vgpuDriverVersion))
                {
                    PRINT_ERROR("", "String overflow error for the requested vGPU instance driver version field");
                    dcgmStrncpy(pDcgmVgpuInstanceAttr->vgpuDriverVersion,
                                DCGM_STR_BLANK,
                                sizeof(pDcgmVgpuInstanceAttr->vgpuDriverVersion));
                }
                else
                {
                    dcgmStrncpy(pDcgmVgpuInstanceAttr->vgpuDriverVersion,
                                fv->value.str,
                                sizeof(pDcgmVgpuInstanceAttr->vgpuDriverVersion));
                }
                break;
            }

            case DCGM_FI_DEV_VGPU_MEMORY_USAGE:
                pDcgmVgpuInstanceAttr->fbUsage = fv->value.i64;
                break;

            case DCGM_FI_DEV_VGPU_LICENSE_INSTANCE_STATUS:
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
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    if (pDeviceConfig->version != dcgmConfig_version)
    {
        PRINT_ERROR("", "Version mismatch");
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
dcgmReturn_t helperVgpuConfigSet(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgmVgpuConfig_t *pDeviceConfig,
                                 dcgmStatus_t pDcgmStatusList)
{
    if (!pDeviceConfig)
    {
        DCGM_LOG_ERROR << "bad pDeviceConfig " << (void *)pDeviceConfig;
        return DCGM_ST_BADPARAM;
    }

    if ((pDeviceConfig->version < dcgmVgpuConfig_version1) || (pDeviceConfig->version > dcgmVgpuConfig_version))
    {
        PRINT_ERROR("%x %x",
                    "VgpuConfigSet version %x mismatches current version %x",
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
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    /* Get version at the index 0 */
    versionAtBaseIndex = pDeviceConfigList[0].version;

    /* Verify requested version in the list of output parameters */
    for (i = 0; i < count; ++i)
    {
        if (pDeviceConfigList[i].version != versionAtBaseIndex)
        {
            PRINT_ERROR("", "Version mismatch");
            return DCGM_ST_VER_MISMATCH;
        }

        if (pDeviceConfigList[i].version != dcgmConfig_version)
        {
            PRINT_ERROR("", "Version mismatch");
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
dcgmReturn_t helperVgpuConfigGet(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgmConfigType_t reqType,
                                 int count,
                                 dcgmVgpuConfig_t *pDeviceConfigList,
                                 dcgmStatus_t pDcgmStatusList)
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
dcgmReturn_t helperVgpuConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t pDcgmStatusList)
{
    /* This code never worked, this API is private, and the tests in test_vgpu.py are disabled.
       Returning NOT_SUPPORTED for now */
    return DCGM_ST_NOT_SUPPORTED;
}

/*****************************************************************************/
dcgmReturn_t tsapiInjectEntityFieldValue(dcgmHandle_t pDcgmHandle,
                                         dcgm_field_entity_group_t entityGroupId,
                                         dcgm_field_eid_t entityId,
                                         dcgmInjectFieldValue_t *pDcgmInjectFieldValue)
{
    std::unique_ptr<dcgm::InjectFieldValue>
        pProtoInjectFieldValue;              /* Protobuf equivalent structure of the output parameter. */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (NULL == pDcgmInjectFieldValue)
    {
        return DCGM_ST_BADPARAM;
    }

    /* Allocate and set version in the protobuf struct */
    pProtoInjectFieldValue = std::make_unique<dcgm::InjectFieldValue>();
    pProtoInjectFieldValue->set_entitygroupid(entityGroupId);
    pProtoInjectFieldValue->set_entityid(entityId);
    pProtoInjectFieldValue->set_version(dcgmInjectFieldValue_version);
    pProtoInjectFieldValue->mutable_fieldvalue()->set_fieldid(pDcgmInjectFieldValue->fieldId);
    pProtoInjectFieldValue->mutable_fieldvalue()->set_status(DCGM_ST_OK);
    pProtoInjectFieldValue->mutable_fieldvalue()->set_ts(pDcgmInjectFieldValue->ts);
    pProtoInjectFieldValue->mutable_fieldvalue()->set_version(dcgmFieldValue_version2);


    switch (pDcgmInjectFieldValue->fieldType)
    {
        case DCGM_FT_DOUBLE:
            pProtoInjectFieldValue->mutable_fieldvalue()->set_fieldtype(dcgm::DBL);
            pProtoInjectFieldValue->mutable_fieldvalue()->mutable_val()->set_dbl(pDcgmInjectFieldValue->value.dbl);
            break;
        case DCGM_FT_INT64:
            pProtoInjectFieldValue->mutable_fieldvalue()->set_fieldtype(dcgm::INT64);
            pProtoInjectFieldValue->mutable_fieldvalue()->mutable_val()->set_i64(pDcgmInjectFieldValue->value.i64);
            break;
        case DCGM_FT_STRING:
            pProtoInjectFieldValue->mutable_fieldvalue()->set_fieldtype(dcgm::STR);
            pProtoInjectFieldValue->mutable_fieldvalue()->mutable_val()->set_str(pDcgmInjectFieldValue->value.str);
            break;
        default:
            return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::INJECT_FIELD_VALUE, dcgm::OPERATION_SINGLE_ENTITY, entityId, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_injectfieldvalue(pProtoInjectFieldValue.release());

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmTsapiInjectEntityFieldValue(dcgmHandle_t pDcgmHandle,
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
        PRINT_DEBUG("%d", "dcgmModuleSendBlockingFixedRequest returned %d", (int)ret);
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
dcgmReturn_t helperGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle, dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    // dcgm::InjectFieldValue *pProtoInjectFieldValue; /* Protobuf equivalent structure of the output parameter. */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    std::string retFieldInfoStr;

    if (!fieldInfo)
        return DCGM_ST_BADPARAM;

    fieldInfo->version = dcgmCacheManagerFieldInfo_version;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::CACHE_MANAGER_FIELD_INFO, dcgm::OPERATION_SINGLE_ENTITY, fieldInfo->gpuId, 0);
    if (!pCmdTemp)
    {
        PRINT_ERROR("", "encodePrb.AddCommand returned NULL");
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_cachemanagerfieldinfo(fieldInfo, sizeof(*fieldInfo));

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    retFieldInfoStr = *(vecCmdsRef[0]->mutable_arg(0)->mutable_cachemanagerfieldinfo());
    if (retFieldInfoStr.size() != sizeof(*fieldInfo))
    {
        PRINT_ERROR("%d %d",
                    "Got CACHE_MANAGER_FIELD_INFO of %d bytes. Expected %d bytes",
                    (int)retFieldInfoStr.size(),
                    (int)sizeof(*fieldInfo));
        return DCGM_ST_VER_MISMATCH;
    }

    memcpy(fieldInfo, (dcgmCacheManagerFieldInfo_t *)retFieldInfoStr.c_str(), sizeof(*fieldInfo));

    if (fieldInfo->version != dcgmCacheManagerFieldInfo_version)
        return DCGM_ST_VER_MISMATCH; /* Same size struct with different version */

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle, dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    if (!fieldInfo)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_get_cache_manager_field_info_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_CACHE_MANAGER_FIELD_INFO;
    msg.header.version    = dcgm_core_msg_get_cache_manager_field_info_version;

    memcpy(&msg.fi.fieldInfo, fieldInfo, sizeof(msg.fi.fieldInfo));
    msg.fi.fieldInfo.version = dcgmCacheManagerFieldInfo_version;
    // coverity[overrun-buffer-arg]
    dcgmReturn_t ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        PRINT_DEBUG("%d", "dcgmModuleSendBlockingFixedRequest returned %d", (int)ret);
        return ret;
    }

    if (DCGM_ST_OK != msg.fi.cmdRet)
    {
        return (dcgmReturn_t)msg.fi.cmdRet;
    }

    memcpy(fieldInfo, &msg.fi.fieldInfo, sizeof(dcgmCacheManagerFieldInfo_t));

    return (dcgmReturn_t)msg.fi.cmdRet;
}

/*****************************************************************************/
static void setFvValueAsBlank(dcgmFieldValue_v1 &fv, unsigned short fieldType)
{
    switch (fieldType)
    {
        case DCGM_FT_INT64:
        case DCGM_FT_TIMESTAMP:
            fv.value.i64 = DCGM_INT64_BLANK;
            break;

        case DCGM_FT_DOUBLE:
            fv.value.dbl = DCGM_FP64_BLANK;
            break;

        case DCGM_FT_STRING:
            dcgmStrncpy(fv.value.str, DCGM_STR_BLANK, sizeof(fv.value.str));
            break;

        case DCGM_FT_BINARY:
            fv.value.blob[0] = 0;
            break;

        default:
            DCGM_LOG_ERROR << "fieldType " << fieldType << " is unhandled.";
            break;
    }
}

/*****************************************************************************/
dcgmReturn_t helperGetMultipleValuesForField(dcgmHandle_t pDcgmHandle,
                                             dcgm_field_entity_group_t entityGroup,
                                             dcgm_field_eid_t entityId,
                                             unsigned int fieldId,
                                             int *count,
                                             long long startTs,
                                             long long endTs,
                                             dcgmOrder_t order,
                                             dcgmFieldValue_v1 values[])
{
    dcgm::FieldMultiValues *pProtoGetMultiValuesForField = 0;
    dcgm::FieldMultiValues *pResponse                    = 0;
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    int maxCount, i;
    int fieldType;
    dcgm_field_meta_p fieldMeta;

    if (!count || (*count) < 1 || !fieldId || !values)
        return DCGM_ST_BADPARAM;

    maxCount = *count;
    *count   = 0;

    PRINT_DEBUG("%u %u %d %d %lld %lld %d",
                "helperGetMultipleValuesForField eg %u eid %u, "
                "fieldId %d, maxCount %d, startTs %lld endTs %lld, order %d",
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
        PRINT_ERROR("%u", "Invalid fieldId %u", fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    memset(values, 0, sizeof(values[0]) * maxCount);

    pProtoGetMultiValuesForField = new dcgm::FieldMultiValues;
    pProtoGetMultiValuesForField->set_version(dcgmGetMultipleValuesForField_version);
    pProtoGetMultiValuesForField->set_fieldid(fieldId);
    // fieldType not required for request
    pProtoGetMultiValuesForField->set_startts(startTs);
    pProtoGetMultiValuesForField->set_endts(endTs);
    pProtoGetMultiValuesForField->set_maxcount(maxCount);
    pProtoGetMultiValuesForField->set_orderflag((dcgm::MultiValuesOrder)order);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GET_FIELD_MULTIPLE_VALUES, dcgm::OPERATION_SINGLE_ENTITY, entityId, 0);
    if (NULL == pCmdTemp)
    {
        PRINT_ERROR("", "encodePrb.AddCommand failed");
        delete pProtoGetMultiValuesForField;
        return DCGM_ST_GENERIC_ERROR;
    }

    if (fieldMeta->scope == DCGM_FS_GLOBAL)
        pCmdTemp->set_entitygroupid(DCGM_FE_NONE);
    else
        pCmdTemp->set_entitygroupid(entityGroup);

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_fieldmultivalues(pProtoGetMultiValuesForField);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        PRINT_DEBUG("%d", "ProcessAtEmbeddedHostEngine returned %d", (int)ret);
        return ret;
    }

    /* Check the status of the DCGM command */
    ret = (dcgmReturn_t)vecCmdsRef[0]->status();
    if (ret == DCGM_ST_NO_DATA || ret == DCGM_ST_NOT_SUPPORTED)
    {
        DCGM_LOG_WARNING << "Handling ret " << ret << " for eg " << entityGroup << " eid " << entityId
                         << " by returning a single fv with that error code.";
        /* Handle these returns by setting the field value status rather than failing the API */
        *count = 1;
        memset(&values[0], 0, sizeof(values[0]));
        values[0].version   = dcgmFieldValue_version1;
        values[0].fieldId   = fieldId;
        values[0].fieldType = fieldMeta->fieldType;
        setFvValueAsBlank(values[0], fieldMeta->fieldType);
        values[0].status = ret;
        return DCGM_ST_OK;
    }
    else if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_WARNING << "vecCmdsRef[0]->status() " << ret;
        return ret;
    }

    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_fieldmultivalues()))
    {
        PRINT_ERROR("", "arg or fieldmultivalue missing");
        return DCGM_ST_GENERIC_ERROR;
    }

    pResponse = vecCmdsRef[0]->mutable_arg(0)->mutable_fieldmultivalues();

    if (pResponse->version() != dcgmGetMultipleValuesForField_version1)
        return DCGM_ST_VER_MISMATCH;
    if (!pResponse->has_fieldtype())
    {
        PRINT_ERROR("", "Field type missing");
        return DCGM_ST_GENERIC_ERROR;
    }

    *count = pResponse->vals_size();

    fieldType = pResponse->fieldtype();

    for (i = 0; i < (*count); i++)
    {
        dcgm::Value *responseValue = pResponse->mutable_vals(i);

        if (responseValue->has_timestamp())
            values[i].ts = responseValue->timestamp();
        else
            PRINT_WARNING("%d %d", "timestamp missing at index %d/%d", i, (*count));


        values[i].version   = dcgmFieldValue_version1;
        values[i].fieldId   = fieldId;
        values[i].fieldType = fieldType;
        values[i].status    = 0;

        switch (values[i].fieldType)
        {
            case DCGM_FT_DOUBLE:
                values[i].value.dbl = responseValue->dbl();
                break;

            case DCGM_FT_INT64:
            case DCGM_FT_TIMESTAMP:
                values[i].value.i64 = responseValue->i64();
                break;

            case DCGM_FT_STRING:
                size_t length;
                length = strlen(responseValue->str().c_str());
                if (length + 1 > DCGM_MAX_STR_LENGTH)
                {
                    PRINT_ERROR("", "String overflow error for the requested field");
                    return DCGM_ST_GENERIC_ERROR;
                }

                dcgmStrncpy(values[i].value.str, responseValue->str().c_str(), sizeof(values[i].value.str) - 1);
                break;

            case DCGM_FT_BINARY:
            {
                if (responseValue->blob().size() > sizeof(values[i].value.blob))
                {
                    PRINT_ERROR("%d %d %d",
                                "Buffer values[index].value.blob size %d too small for %d. fieldType %d",
                                (int)sizeof(values[i].value.blob),
                                (int)responseValue->blob().size(),
                                (int)values[i].fieldType);
                    return DCGM_ST_MEMORY;
                }

                memcpy(values[i].value.blob, (void *)responseValue->blob().c_str(), responseValue->blob().size());
                break;
            }

            default:
                PRINT_ERROR("%c", "Uknown type: %c", (char)fieldType);
                return DCGM_ST_GENERIC_ERROR;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperGetMultipleValuesForField(dcgmHandle_t pDcgmHandle,
                                               dcgm_field_entity_group_t entityGroup,
                                               dcgm_field_eid_t entityId,
                                               unsigned int fieldId,
                                               int *count,
                                               long long startTs,
                                               long long endTs,
                                               dcgmOrder_t order,
                                               dcgmFieldValue_v1 values[])
{
    dcgmReturn_t ret;
    int maxCount, i;
    dcgm_field_meta_p fieldMeta;

    if (!count || (*count) < 1 || !fieldId || !values)
        return DCGM_ST_BADPARAM;

    maxCount = *count;
    *count   = 0;

    PRINT_DEBUG("%u %u %d %d %lld %lld %d",
                "helperGetMultipleValuesForField eg %u eid %u, "
                "fieldId %d, maxCount %d, startTs %lld endTs %lld, order %d",
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
        PRINT_ERROR("%u", "Invalid fieldId %u", fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    memset(values, 0, sizeof(values[0]) * maxCount);

    dcgm_core_msg_get_multiple_values_for_field_t msg = {};

    msg.header.length = sizeof(msg) - SAMPLES_BUFFER_SIZE; /* avoid transferring the large buffer when making request */
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD;
    msg.header.version    = dcgm_core_msg_get_multiple_values_for_field_version;

    msg.fv.entityId = entityId;
    msg.fv.fieldId  = fieldId;
    msg.fv.startTs  = startTs;
    msg.fv.endTs    = endTs;
    msg.fv.count    = maxCount;
    msg.fv.order    = order;

    if (fieldMeta->scope == DCGM_FS_GLOBAL)
        msg.fv.entityGroupId = DCGM_FE_NONE;
    else
        msg.fv.entityGroupId = entityGroup;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        PRINT_DEBUG("%d", "dcgmModuleSendBlockingFixedRequest returned %d", (int)ret);
        return ret;
    }

    /* Check the status of the DCGM command */
    ret = (dcgmReturn_t)msg.fv.cmdRet;
    if (ret == DCGM_ST_NO_DATA || ret == DCGM_ST_NOT_SUPPORTED)
    {
        DCGM_LOG_WARNING << "Handling ret " << ret << " for eg " << entityGroup << " eid " << entityId
                         << " by returning a single fv with that error code.";
        /* Handle these returns by setting the field value status rather than failing the API */
        *count = 1;
        memset(&values[0], 0, sizeof(values[0]));
        values[0].version   = dcgmFieldValue_version1;
        values[0].fieldId   = fieldId;
        values[0].fieldType = fieldMeta->fieldType;
        setFvValueAsBlank(values[0], fieldMeta->fieldType);
        values[0].status = ret;
        return DCGM_ST_OK;
    }
    else if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_WARNING << "vecCmdsRef[0]->status() " << ret;
        return ret;
    }

    *count = msg.fv.count;

    DcgmFvBuffer fvBuffer(0);
    fvBuffer.SetFromBuffer(msg.fv.buffer, msg.fv.bufferSize);
    /* Convert the buffered FVs to our output array */
    dcgmBufferedFvCursor_t cursor = 0;
    i                             = 0;
    for (dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor); fv && i < maxCount && i < msg.fv.count;
         fv                   = fvBuffer.GetNextFv(&cursor))
    {
        fvBuffer.ConvertBufferedFvToFv1(fv, &values[i]);
        i++;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t helperSendStructRequest(dcgmHandle_t pDcgmHandle,
                                     unsigned int cmdType,
                                     int gpuId,
                                     int groupId,
                                     void *structData,
                                     int structSize)
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    unsigned int opMode;
    int opModeId;

    if (gpuId >= 0 && groupId >= 0)
    {
        PRINT_WARNING("%d %d", "Invalid combo of gpuId %d and groupId %d", gpuId, groupId);
        return DCGM_ST_BADPARAM;
    }

    if (!structData || structSize < 1)
        return DCGM_ST_BADPARAM;

    /* We already validated mutual exclusivity above. Now prepare to pass gpuId/groupId */
    if (gpuId >= 0)
    {
        opMode   = dcgm::OPERATION_SINGLE_ENTITY;
        opModeId = gpuId;
    }
    else if (groupId >= 0) /* groupId */
    {
        opMode   = dcgm::OPERATION_GROUP_ENTITIES;
        opModeId = groupId;
    }
    else
    {
        opMode   = dcgm::OPERATION_SYSTEM;
        opModeId = -1;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(cmdType, opMode, opModeId, 0);
    if (NULL == pCmdTemp)
    {
        PRINT_ERROR("", "Error from AddCommand");
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_blob(structData, structSize);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    if (!vecCmdsRef[0]->arg_size())
    {
        PRINT_ERROR("", "Arg size of 0 unexpected");
        return DCGM_ST_GENERIC_ERROR;
    }

    if (!vecCmdsRef[0]->arg(0).has_blob())
    {
        PRINT_ERROR("", "Response missing blob");
        return DCGM_ST_GENERIC_ERROR;
    }

    if ((int)vecCmdsRef[0]->arg(0).blob().size() > structSize)
    {
        PRINT_ERROR(
            "%d %d", "Returned blob size %d > structSize %d", (int)vecCmdsRef[0]->arg(0).blob().size(), structSize);
        return DCGM_ST_GENERIC_ERROR;
    }

    memcpy(structData, (void *)vecCmdsRef[0]->arg(0).blob().c_str(), vecCmdsRef[0]->arg(0).blob().size());

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t helperUnwatchFieldValue(dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldId, int clearCache)
{
    dcgm::UnwatchFieldValue *pProtoUnwatchFieldValue; /* Protobuf Arg */
    DcgmProtobuf encodePrb;                           /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                           /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                          /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef;          /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (!fieldId)
        return DCGM_ST_BADPARAM;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById((unsigned short)fieldId);
    if (NULL == fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "field ID %u is not a valid field ID", fieldId);
        return DCGM_ST_BADPARAM;
    }

    pProtoUnwatchFieldValue = new dcgm::UnwatchFieldValue;
    pProtoUnwatchFieldValue->set_version(dcgmUnwatchFieldValue_version);
    pProtoUnwatchFieldValue->set_fieldid(fieldId);
    pProtoUnwatchFieldValue->set_clearcache(clearCache);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::UNWATCH_FIELD_VALUE, dcgm::OPERATION_SINGLE_ENTITY, gpuId, 0);
    if (NULL == pCmdTemp)
    {
        delete pProtoUnwatchFieldValue;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the entityGroupId */
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;
    if (fieldMeta->scope == DCGM_FS_GLOBAL)
        entityGroupId = DCGM_FE_NONE;
    pCmdTemp->set_entitygroupid((int)entityGroupId);

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_unwatchfieldvalue(pProtoUnwatchFieldValue);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t cmHelperUnwatchFieldValue(dcgmHandle_t pDcgmHandle, int gpuId, unsigned short fieldId, int clearCache)
{
    dcgmReturn_t ret;

    if (!fieldId)
        return DCGM_ST_BADPARAM;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById((unsigned short)fieldId);
    if (NULL == fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "field ID %u is not a valid field ID", fieldId);
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
                             dcgmStatus_t dcgmStatusList)
{
    dcgm_policy_msg_get_policies_t msg;
    dcgmReturn_t dcgmReturn;
    int i;

    if ((NULL == dcgmPolicy) || (count <= 0))
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    /* Note: dcgmStatusList has always been ignored by this request. Continuing this tradition on
             as I refactor this for modularity */

    /* Verify requested version in the list of output parameters */
    for (i = 0; i < count; i++)
    {
        if (dcgmPolicy[i].version != dcgmPolicy_version)
        {
            PRINT_ERROR("%d", "Version mismatch at index %d", i);
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
                             dcgmStatus_t dcgmStatusList)
{
    dcgm_policy_msg_set_policy_t msg;
    dcgmReturn_t dcgmReturn;

    if (NULL == dcgmPolicy)
        return DCGM_ST_BADPARAM;

    /* Note: dcgmStatusList has always been ignored by this request. Continuing this tradition on
             as I refactor this for modularity */

    if (dcgmPolicy->version != dcgmPolicy_version)
    {
        PRINT_ERROR("", "Version mismatch.");
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
                                  fpRecvUpdates beginCallback,
                                  fpRecvUpdates finishCallback)
{
    dcgmReturn_t dcgmReturn;
    dcgm_policy_msg_register_t msg;

    /* Make an ansync object. We're going to pass ownership off, so we won't have to free it */
    std::unique_ptr<DcgmPolicyRequest> policyRequest
        = std::make_unique<DcgmPolicyRequest>(beginCallback, finishCallback);

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

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
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
    dcgmGroupInfo_t groupInfo = {};
    unsigned int i;
    int j;
    unsigned int gpuId, fieldId;
    int valuesAtATime = 100; /* How many values should we fetch at a time */
    int retNumFieldValues;
    dcgmFieldValue_v1 *fieldValues = 0;
    int callbackSt                 = 0;
    long long endQueryTimestamp    = 0;

    retDcgmSt = DCGM_ST_OK;

    if (!fieldIds || !enumCB || !nextSinceTimestamp || numFieldIds < 1)
    {
        PRINT_ERROR("", "Bad param to helperGetFieldValuesSince");
        return DCGM_ST_BADPARAM;
    }

    PRINT_DEBUG("%p %lld %d %p",
                "helperGetFieldValuesSince groupId %p, sinceTs %lld, numFieldIds %d, userData %p",
                (void *)groupId,
                sinceTimestamp,
                numFieldIds,
                (void *)userData);

    *nextSinceTimestamp = sinceTimestamp;

    groupInfo.version = dcgmGroupInfo_version;

    /* Convert groupId to list of GPUs. Note that this is an extra round trip to the server
     * in the remote case, but it keeps the code much simpler */
    dcgmSt = helperGroupGetInfo(pDcgmHandle, groupId, &groupInfo, &endQueryTimestamp);
    if (dcgmSt != DCGM_ST_OK)
    {
        PRINT_ERROR("%p %d", "helperGroupGetInfo groupId %p returned %d", (void *)groupId, (int)dcgmSt);
        return dcgmSt;
    }

    PRINT_DEBUG("%s %d", "Got group %s with %d entities", groupInfo.groupName, groupInfo.count);

    /* Pre-check the group for non-GPU/non-global entities */
    for (i = 0; i < groupInfo.count; i++)
    {
        if (groupInfo.entityList[i].entityGroupId != DCGM_FE_GPU
            && groupInfo.entityList[i].entityGroupId != DCGM_FE_NONE)
        {
            PRINT_ERROR("%p %u %u",
                        "helperGetFieldValuesSince called on groupId %p with non-GPU eg %u, eid %u.",
                        (void *)groupId,
                        groupInfo.entityList[i].entityGroupId,
                        groupInfo.entityList[i].entityId);
            return DCGM_ST_NOT_SUPPORTED;
        }
    }

    fieldValues = (dcgmFieldValue_v1 *)malloc(sizeof(*fieldValues) * valuesAtATime);
    if (!fieldValues)
    {
        PRINT_ERROR("%d", "Unable to alloc %d bytes", (int)(sizeof(*fieldValues) * valuesAtATime));
        return DCGM_ST_MEMORY;
    }
    memset(fieldValues, 0, sizeof(*fieldValues) * valuesAtATime);

    /* Fetch valuesAtATime values for each GPU for each field since sinceTimestamp.
     * Make valuesAtATime large enough to offset the fact that this is a round trip
     * to the server for each combo of gpuId, fieldId, and valuesAtATime values
     */

    for (i = 0; i < groupInfo.count; i++)
    {
        gpuId = groupInfo.entityList[i].entityId;

        for (j = 0; j < numFieldIds; j++)
        {
            fieldId = fieldIds[j];

            retNumFieldValues = valuesAtATime;
            dcgmSt            = helperGetMultipleValuesForField(pDcgmHandle,
                                                     DCGM_FE_GPU,
                                                     gpuId,
                                                     fieldId,
                                                     &retNumFieldValues,
                                                     sinceTimestamp,
                                                     endQueryTimestamp,
                                                     DCGM_ORDER_ASCENDING,
                                                     fieldValues);
            if (dcgmSt == DCGM_ST_NO_DATA)
            {
                PRINT_DEBUG("%u, %u %lld",
                            "DCGM_ST_NO_DATA for gpuId %u, fieldId %u, sinceTs %lld",
                            gpuId,
                            fieldId,
                            sinceTimestamp);
                continue;
            }
            else if (dcgmSt != DCGM_ST_OK)
            {
                PRINT_ERROR("%d %u %u",
                            "Got st %d from helperGetMultipleValuesForField gpuId %u, fieldId %u",
                            (int)dcgmSt,
                            gpuId,
                            fieldId);
                retDcgmSt = dcgmSt;
                goto CLEANUP;
            }

            PRINT_DEBUG("%d %u %u", "Got %d values for gpuId %u, fieldId %u", retNumFieldValues, gpuId, fieldId);

            callbackSt = enumCB(gpuId, fieldValues, retNumFieldValues, userData);
            if (callbackSt != 0)
            {
                PRINT_DEBUG("", "User requested callback exit");
                /* Leaving status as OK. User requested the exit */
                goto CLEANUP;
            }
        }
    }

    /* Success. We can advance the caller's next query timestamp */
    *nextSinceTimestamp = endQueryTimestamp + 1;


CLEANUP:
    if (fieldValues)
    {
        free(fieldValues);
        fieldValues = 0;
    }


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
    dcgmGroupInfo_t groupInfo           = {};
    dcgmFieldGroupInfo_t fieldGroupInfo = {};
    unsigned int i;
    int j;
    unsigned int fieldId;
    int valuesAtATime = 100; /* How many values should we fetch at a time */
    int retNumFieldValues;
    dcgmFieldValue_v1 *fieldValues = 0;
    int callbackSt                 = 0;
    long long endQueryTimestamp    = 0;

    retDcgmSt = DCGM_ST_OK;

    if ((!enumCB && !enumCBv2) || !nextSinceTimestamp)
    {
        PRINT_ERROR("", "Bad param to helperGetValuesSince");
        return DCGM_ST_BADPARAM;
    }

    *nextSinceTimestamp = sinceTimestamp;

    fieldGroupInfo.version      = dcgmFieldGroupInfo_version;
    fieldGroupInfo.fieldGroupId = fieldGroupId;
    dcgmSt                      = dcgmFieldGroupGetInfo(pDcgmHandle, &fieldGroupInfo);
    if (dcgmSt != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %p",
                    "Got dcgmSt %d from dcgmFieldGroupGetInfo() fieldGroupId %p",
                    (dcgmReturn_t)dcgmSt,
                    (void *)fieldGroupId);
        return dcgmSt;
    }

    PRINT_DEBUG("%p %s %u",
                "fieldGroup %p, name %s, numFieldIds %u",
                (void *)fieldGroupId,
                fieldGroupInfo.fieldGroupName,
                fieldGroupInfo.numFieldIds);

    /* Convert groupId to list of GPUs. Note that this is an extra round trip to the server
     * in the remote case, but it keeps the code much simpler */
    groupInfo.version = dcgmGroupInfo_version;
    dcgmSt            = helperGroupGetInfo(pDcgmHandle, groupId, &groupInfo, &endQueryTimestamp);
    if (dcgmSt != DCGM_ST_OK)
    {
        PRINT_ERROR("%p %d", "helperGroupGetInfo groupId %p returned %d", (void *)groupId, (int)dcgmSt);
        return dcgmSt;
    }

    PRINT_DEBUG("%s %d %lld",
                "Got group %s with %d GPUs, endQueryTimestamp %lld",
                groupInfo.groupName,
                groupInfo.count,
                endQueryTimestamp);

    /* Pre-check the group for non-GPU/non-global entities */
    if (!enumCBv2)
    {
        for (i = 0; i < groupInfo.count; i++)
        {
            if (groupInfo.entityList[i].entityGroupId != DCGM_FE_GPU
                && groupInfo.entityList[i].entityGroupId != DCGM_FE_NONE)
            {
                PRINT_ERROR("%p %u %u",
                            "helperGetValuesSince called on groupId %p with non-GPU eg %u, eid %u.",
                            (void *)groupId,
                            groupInfo.entityList[i].entityGroupId,
                            groupInfo.entityList[i].entityId);
                return DCGM_ST_NOT_SUPPORTED;
            }
        }
    }

    fieldValues = (dcgmFieldValue_v1 *)malloc(sizeof(*fieldValues) * valuesAtATime);
    if (!fieldValues)
    {
        PRINT_ERROR("%d", "Unable to alloc %d bytes", (int)(sizeof(*fieldValues) * valuesAtATime));
        return DCGM_ST_MEMORY;
    }
    memset(fieldValues, 0, sizeof(*fieldValues) * valuesAtATime);

    /* Fetch valuesAtATime values for each GPU for each field since sinceTimestamp.
     * Make valuesAtATime large enough to offset the fact that this is a round trip
     * to the server for each combo of gpuId, fieldId, and valuesAtATime values
     */

    for (i = 0; i < groupInfo.count; i++)
    {
        for (j = 0; j < (int)fieldGroupInfo.numFieldIds; j++)
        {
            fieldId = fieldGroupInfo.fieldIds[j];

            /* Using endQueryTimestamp as endTime here so we don't get values that update after the
               nextSinceTimestamp we're returning to the client */
            retNumFieldValues = valuesAtATime;
            dcgmSt            = helperGetMultipleValuesForField(pDcgmHandle,
                                                     groupInfo.entityList[i].entityGroupId,
                                                     groupInfo.entityList[i].entityId,
                                                     fieldId,
                                                     &retNumFieldValues,
                                                     sinceTimestamp,
                                                     endQueryTimestamp,
                                                     DCGM_ORDER_ASCENDING,
                                                     fieldValues);
            if (dcgmSt == DCGM_ST_NO_DATA)
            {
                PRINT_DEBUG("%u %u, %u %lld",
                            "DCGM_ST_NO_DATA for eg %u, eid %u, fieldId %u, sinceTs %lld",
                            groupInfo.entityList[i].entityGroupId,
                            groupInfo.entityList[i].entityId,
                            fieldId,
                            sinceTimestamp);
                continue;
            }
            else if (dcgmSt != DCGM_ST_OK)
            {
                PRINT_ERROR("%d %u %u %u",
                            "Got st %d from helperGetMultipleValuesForField eg %u, eid %u, fieldId %u",
                            (int)dcgmSt,
                            groupInfo.entityList[i].entityGroupId,
                            groupInfo.entityList[i].entityId,
                            fieldId);
                retDcgmSt = dcgmSt;
                goto CLEANUP;
            }

            PRINT_DEBUG("%d %u %u %u",
                        "Got %d values for eg %u, eid %u, fieldId %u",
                        retNumFieldValues,
                        groupInfo.entityList[i].entityGroupId,
                        groupInfo.entityList[i].entityId,
                        fieldId);

            if (enumCB)
            {
                callbackSt = enumCB(groupInfo.entityList[i].entityId, fieldValues, retNumFieldValues, userData);
                if (callbackSt != 0)
                {
                    PRINT_DEBUG("", "User requested callback exit");
                    /* Leaving status as OK. User requested the exit */
                    goto CLEANUP;
                }
            }
            if (enumCBv2)
            {
                callbackSt = enumCBv2(groupInfo.entityList[i].entityGroupId,
                                      groupInfo.entityList[i].entityId,
                                      fieldValues,
                                      retNumFieldValues,
                                      userData);
                if (callbackSt != 0)
                {
                    PRINT_DEBUG("", "User requested callback exit");
                    /* Leaving status as OK. User requested the exit */
                    goto CLEANUP;
                }
            }
        }
    }

    /* Success. We can advance the caller's next query timestamp */
    *nextSinceTimestamp = endQueryTimestamp + 1;
    PRINT_DEBUG("%lld", "nextSinceTimestamp advanced to %lld", *nextSinceTimestamp);

CLEANUP:
    if (fieldValues)
    {
        free(fieldValues);
        fieldValues = 0;
    }


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
    dcgm::WatchFields *pWatchFields;         /* Request message */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (!groupId)
    {
        PRINT_ERROR("", "Bad param");
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::WATCH_FIELDS, dcgm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp)
    {
        PRINT_ERROR("", "encodePrb.AddCommand failed");
        return DCGM_ST_GENERIC_ERROR;
    }

    pWatchFields = pCmdTemp->add_arg()->mutable_watchfields();
    pWatchFields->set_version(dcgmWatchFields_version);
    pWatchFields->set_fieldgroupid((uintptr_t)fieldGroupId);
    pWatchFields->set_updatefreq(updateFreq);
    pWatchFields->set_maxkeepage(maxKeepAge);
    pWatchFields->set_maxkeepsamples(maxKeepSamples);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

dcgmReturn_t cmTsapiWatchFields(dcgmHandle_t pDcgmHandle,
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
    dcgm::UnwatchFields *pUnwatchFields;     /* Request message */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (!groupId)
    {
        PRINT_ERROR("", "Bad param");
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::UNWATCH_FIELDS, dcgm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp)
    {
        PRINT_ERROR("", "encodePrb.AddCommand failed");
        return DCGM_ST_GENERIC_ERROR;
    }

    pUnwatchFields = pCmdTemp->add_arg()->mutable_unwatchfields();
    pUnwatchFields->set_fieldgroupid((uintptr_t)fieldGroupId);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "processAtHostEngine returned %d", ret);
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        PRINT_DEBUG("%d", "vecCmdsRef[0]->status() returned %d", (int)vecCmdsRef[0]->status());
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    return DCGM_ST_OK;
}

dcgmReturn_t cmTsapiUnwatchFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId)
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
                                   unsigned short *fieldIds,
                                   char *fieldGroupName,
                                   dcgmFieldGrp_t *dcgmFieldGroupId)
{
    dcgmReturn_t dcgmReturn;
    dcgmFieldGroupInfo_t fieldGroupInfo;

    if (numFieldIds < 1 || numFieldIds >= DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP || !fieldGroupName
        || strlen(fieldGroupName) >= DCGM_MAX_STR_LENGTH || !fieldGroupName[0] || !fieldIds || !dcgmFieldGroupId)
    {
        return DCGM_ST_BADPARAM;
    }

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version = dcgmFieldGroupInfo_version;
    dcgmStrncpy(fieldGroupInfo.fieldGroupName, fieldGroupName, sizeof(fieldGroupInfo.fieldGroupName) - 1);
    fieldGroupInfo.numFieldIds = numFieldIds;
    memcpy(fieldGroupInfo.fieldIds, fieldIds, sizeof(fieldIds[0]) * numFieldIds);

    dcgmReturn = helperSendStructRequest(
        pDcgmHandle, dcgm::FIELD_GROUP_CREATE, -1, -1, &fieldGroupInfo, sizeof(fieldGroupInfo));

    PRINT_DEBUG("%d", "tsapiFieldGroupCreate dcgmSt %d", (int)dcgmReturn);

    *dcgmFieldGroupId = fieldGroupInfo.fieldGroupId;
    return dcgmReturn;
}

dcgmReturn_t cmTsapiFieldGroupCreate(dcgmHandle_t pDcgmHandle,
                                     int numFieldIds,
                                     unsigned short *fieldIds,
                                     char *fieldGroupName,
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
    dcgmStrncpy(msg.info.fg.fieldGroupName, fieldGroupName, sizeof(msg.info.fg.fieldGroupName) - 1);
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

dcgmReturn_t helperActionManager(dcgmHandle_t dcgmHandle,
                                 dcgmRunDiag_t *drd,
                                 dcgmPolicyAction_t action,
                                 dcgmDiagResponse_t *response)
{
    dcgm_diag_msg_run_v4 msg4;
    dcgmReturn_t dcgmReturn;

    if (!drd || !response)
    {
        PRINT_ERROR("%p %p", "drd %p or response %p was NULL.", (void *)drd, (void *)response);
        return DCGM_ST_BADPARAM;
    }

    switch (drd->version)
    {
        case dcgmRunDiag_version7:
            break;
        default:
            // unknown drd version
            PRINT_ERROR("%X %X",
                        "dcgmRunDiag version mismatch %X != %X and isn't in accepted list",
                        drd->version,
                        dcgmRunDiag_version);
            return DCGM_ST_VER_MISMATCH;
    }

    dcgm_module_command_header_t *header;
    dcgmRunDiag_t *runDiag;

    switch (response->version)
    {
        case dcgmDiagResponse_version6:
            memset(&msg4, 0, sizeof(msg4));
            msg4.header.length        = sizeof(msg4);
            msg4.header.version       = dcgm_diag_msg_run_version4;
            msg4.diagResponse.version = dcgmDiagResponse_version6;
            msg4.action               = action;
            runDiag                   = &(msg4.runDiag);
            header                    = &(msg4.header);
            break;

        default:
            DCGM_LOG_ERROR << "response->version 0x" << std::hex << response->version
                           << " doesn't match a valid version";
            return DCGM_ST_VER_MISMATCH;
    }

    header->moduleId   = DcgmModuleIdDiag;
    header->subCommand = DCGM_DIAG_SR_RUN;

    switch (drd->version)
    {
        case dcgmRunDiag_version7:
            memcpy(runDiag, drd, sizeof(dcgmRunDiag_v7));
            break;
        default:
            // unknown dcgmRunDiag version
            PRINT_ERROR("%X %X",
                        "dcgmRunDiag_version mismatch %X != %X and isn't in accepted list",
                        drd->version,
                        dcgmRunDiag_version);
            return DCGM_ST_VER_MISMATCH;
    }

    static const int SIXTY_MINUTES_IN_MS = 3600000;
    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, header, sizeof(msg4), nullptr, SIXTY_MINUTES_IN_MS);

    switch (response->version)
    {
        case dcgmDiagResponse_version6:
            memcpy(response, &msg4.diagResponse, sizeof(msg4.diagResponse));
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

/*****************************************************************************/
static dcgmReturn_t helperHealthCheckV4(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_v4 *response)
{
    dcgm_health_msg_check_v4 msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_CHECK_V4;
    msg.header.version    = dcgm_health_msg_check_version4;

    msg.groupId   = groupId;
    msg.startTime = 0;
    msg.endTime   = 0;

    memcpy(&msg.response, response, sizeof(msg.response));
    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    memcpy(response, &msg.response, sizeof(msg.response));
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
static dcgmReturn_t helperGetTopologyAffinity(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgmAffinity_t *groupAffinity)
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgm::Command *pGroupCmd;                /* Temp reference to the command */
    dcgmReturn_t ret;

    if (NULL == groupAffinity)
        return DCGM_ST_BADPARAM;

    /* Add Command to the protobuf encoder object */
    pCmdTemp
        = encodePrb.AddCommand(dcgm::GET_TOPOLOGY_INFO_AFFINITY, dcgm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    // check to see if topology information was even applicable
    if (pGroupCmd->status() == DCGM_ST_NO_DATA)
        return (dcgmReturn_t)pGroupCmd->status();

    if (!(pGroupCmd->arg_size()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }

    if (pGroupCmd->mutable_arg(0)->has_blob())
    {
        memcpy(groupAffinity, (void *)pGroupCmd->mutable_arg(0)->blob().c_str(), sizeof(dcgmAffinity_t));
    }

    return (dcgmReturn_t)pGroupCmd->status();
}

/*****************************************************************************/
dcgmReturn_t cmHelperGetTopologyAffinity(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmAffinity_t *groupAffinity)
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
static dcgmReturn_t helperSelectGpusByTopology(dcgmHandle_t pDcgmHandle,
                                               uint64_t inputGpuIds,
                                               uint32_t numGpus,
                                               uint64_t *outputGpuIds,
                                               uint64_t hintFlags)
{
    dcgm::Command *pCmdTemp;
    dcgmReturn_t ret;
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgm::Command *pGroupCmd;                /* Temp reference to the command */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */

    dcgm::SchedulerHintRequest *shr;

    if ((dcgmHandle_t) nullptr == pDcgmHandle || !outputGpuIds)
    {
        DCGM_LOG_ERROR << "bad outputGpuIds " << (void *)outputGpuIds << " or pDcgmHandle " << (void *)pDcgmHandle;
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::SELECT_GPUS_BY_TOPOLOGY, dcgm::OPERATION_GROUP_ENTITIES, 0, 0);
    if (pCmdTemp == NULL)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    shr = new dcgm::SchedulerHintRequest;
    shr->set_version(dcgmTopoSchedHint_version1);
    shr->set_inputgpuids(inputGpuIds);
    shr->set_numgpus(numGpus);
    shr->set_hintflags(hintFlags);

    pCmdTemp->add_arg()->set_allocated_schedulerhintrequest(shr);

    // This should be fast. We'll do a 30 second timeout
    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    if (!(pGroupCmd->arg_size()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        PRINT_DEBUG("", "Return argument is missing");
        return DCGM_ST_GENERIC_ERROR;
    }

    if (pGroupCmd->mutable_arg(0)->has_i64())
    {
        *outputGpuIds = pGroupCmd->mutable_arg(0)->i64();
    }

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

/*****************************************************************************/
dcgmReturn_t cmHelperSelectGpusByTopology(dcgmHandle_t pDcgmHandle,
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
static dcgmReturn_t helperGetTopologyPci(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmTopology_t *groupTopology)
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgm::Command *pGroupCmd;                /* Temp reference to the command */
    dcgmReturn_t ret;

    if (NULL == groupTopology)
        return DCGM_ST_BADPARAM;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GET_TOPOLOGY_INFO_IO, dcgm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    // no data is okay, topology struct returned will just numElements == 0
    if (DCGM_ST_OK != ret && DCGM_ST_NO_DATA != ret)
    {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    if (!(pGroupCmd->arg_size()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        return DCGM_ST_GENERIC_ERROR;
    }

    if (pGroupCmd->mutable_arg(0)->has_blob())
    {
        memcpy(groupTopology, (void *)pGroupCmd->mutable_arg(0)->blob().c_str(), sizeof(dcgmTopology_t));
    }

    return (dcgmReturn_t)pGroupCmd->status();
}

/*****************************************************************************/
dcgmReturn_t cmHelperGetTopologyPci(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmTopology_t *groupTopology)
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

static dcgmReturn_t tsapiGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupIdList[], unsigned int *count)
{
    return helperGroupGetAllIds(pDcgmHandle, groupIdList, count);
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

static dcgmReturn_t tsapiEngineGroupCreate(dcgmHandle_t pDcgmHandle,
                                           dcgmGroupType_t type,
                                           char *groupName,
                                           dcgmGpuGrp_t *pDcgmGrpId)
{
    return helperGroupCreate(pDcgmHandle, type, groupName, pDcgmGrpId);
}

static dcgmReturn_t tsapiEngineGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId)
{
    return helperGroupDestroy(pDcgmHandle, groupId);
}

static dcgmReturn_t tsapiEngineGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId)
{
    return helperGroupAddDevice(pDcgmHandle, groupId, gpuId);
}

static dcgmReturn_t tsapiEngineGroupRemoveDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId)
{
    return helperGroupRemoveDevice(pDcgmHandle, groupId, gpuId);
}

static dcgmReturn_t tsapiEngineGroupGetInfo(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmGroupInfo_t *pDcgmGroupInfo)
{
    return helperGroupGetInfo(pDcgmHandle, groupId, pDcgmGroupInfo, 0);
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
    return helperGetAllDevices(pDcgmHandle, gpuIdList, count, 0);
}

static dcgmReturn_t tsapiEngineGetAllSupportedDevices(dcgmHandle_t pDcgmHandle,
                                                      unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                                      int *count)
{
    return helperGetAllDevices(pDcgmHandle, gpuIdList, count, 1);
}

static dcgmReturn_t tsapiGetEntityGroupEntities(dcgmHandle_t dcgmHandle,
                                                dcgm_field_entity_group_t entityGroup,
                                                dcgm_field_eid_t *entities,
                                                int *numEntities,
                                                unsigned int flags)
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    dcgm::CmdArg *cmdArg          = 0;
    dcgm::EntityList *pEntityList = NULL;

    if (!entities || !numEntities)
    {
        return DCGM_ST_BADPARAM;
    }
    int entitiesCapacity = *numEntities;

    int onlySupported = (flags & DCGM_GEGE_FLAG_ONLY_SUPPORTED) ? 1 : 0;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::GET_ENTITY_LIST, dcgm::OPERATION_SYSTEM, -1, 0);
    if (!pCmdTemp)
    {
        PRINT_ERROR("", "AddCommand failed");
        return DCGM_ST_GENERIC_ERROR;
    }

    cmdArg = pCmdTemp->add_arg();

    pEntityList = new dcgm::EntityList();
    pEntityList->set_entitygroupid(entityGroup);
    pEntityList->set_onlysupported(onlySupported);
    cmdArg->set_allocated_entitylist(pEntityList);

    ret = processAtHostEngine(dcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        PRINT_DEBUG("%d", "vecCmdsRef[0]->status() %d", vecCmdsRef[0]->status());
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    /* Make sure that the returned message has the required structures */
    if (!(vecCmdsRef[0]->arg_size() && vecCmdsRef[0]->arg(0).has_entitylist()))
    {
        /* This should never happen unless there is a bug in message packing at
           the host engine side */
        PRINT_ERROR("", "Returned message was malformed");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Update the Protobuf reference with the results */
    pEntityList = vecCmdsRef[0]->mutable_arg(0)->mutable_entitylist();

    *numEntities = pEntityList->entity_size();

    if (pEntityList->entity_size() > entitiesCapacity)
    {
        PRINT_DEBUG("%d %d", "Insufficient capacity: %d > %d", pEntityList->entity_size(), entitiesCapacity);
        *numEntities = pEntityList->entity_size();
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    for (int index = 0; index < pEntityList->entity_size(); index++)
    {
        entities[index] = pEntityList->mutable_entity(index)->entityid();
    }

    return DCGM_ST_OK;
}

dcgmReturn_t cmTsapiGetEntityGroupEntities(dcgmHandle_t dcgmHandle,
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

    dcgm_core_msg_get_entity_group_entities_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_ENTITY_GROUP_ENTITIES;
    msg.header.version    = dcgm_core_msg_get_entity_group_entities_version;

    msg.entities.flags       = flags;
    msg.entities.entityGroup = entityGroup;

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (msg.entities.cmdRet != DCGM_ST_OK)
    {
        return (dcgmReturn_t)msg.entities.cmdRet;
    }

    *numEntities = msg.entities.numEntities;

    if (msg.entities.numEntities > entitiesCapacity)
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    for (int i = 0; i < msg.entities.numEntities; i++)
    {
        entities[i] = msg.entities.entities[i];
    }

    return DCGM_ST_OK;
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

    if (hierarchy->version != dcgmMigHierarchy_version1 && hierarchy->version != dcgmMigHierarchy_version2)
    {
        DCGM_LOG_ERROR << "The dcgmEntityHierarchy was called with an invalid hierarchy argument version."
                       << " Expected versions are either "
                       << "dcgmMigHierarchy_version2 (" << dcgmMigHierarchy_version2 << ") or "
                       << "dcgmMigHierarchy_version1 (" << dcgmMigHierarchy_version1 << ")."
                       << " Given argument version is " << hierarchy->version;
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_core_msg_get_gpu_instance_hierarchy_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY;
    msg.header.version    = dcgm_core_msg_get_gpu_instance_hierarchy_version;

    if (hierarchy->version == dcgmMigHierarchy_version1)
    {
        memcpy(&msg.info.mh.v1, hierarchy, sizeof(msg.info.mh.v1));
        msg.info.v2 = 0;
    }
    else
    {
        memcpy(&msg.info.mh.v2, hierarchy, sizeof(msg.info.mh.v2));
        msg.info.v2 = 1;
    }

    // coverity[overrun-buffer-arg]
    ret = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    DCGM_LOG_DEBUG << "Got total GPUs/GPU Instances/GPU Compute Instances back: " << hierarchy->count
                   << ". dcgmReturn: " << ret;

    if (hierarchy->version == dcgmMigHierarchy_version1)
    {
        memcpy(hierarchy, &msg.info.mh.v1, sizeof(msg.info.mh.v1));
    }
    else
    {
        memcpy(hierarchy, &msg.info.mh.v2, sizeof(msg.info.mh.v2));
    }

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
        && (cme->profile < DcgmMigProfileComputeInstanceSlice1 || cme->profile > DcgmMigProfileComputeInstanceSlice7))
    {
        DCGM_LOG_ERROR << "Invalid profile " << cme->profile << " for creating a compute instance";
        return DCGM_ST_BADPARAM;
    }
    else if (cme->createOption == DcgmMigCreateGpuInstance
             && (cme->profile < DcgmMigProfileGpuInstanceSlice1 || cme->profile > DcgmMigProfileGpuInstanceSlice7))
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

dcgmReturn_t tsapiGetNvLinkLinkStatus(dcgmHandle_t dcgmHandle, dcgmNvLinkStatus_v2 *linkStatus)
{
    if (!linkStatus)
        return DCGM_ST_BADPARAM;

    if (linkStatus->version != dcgmNvLinkStatus_version2)
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

static dcgmReturn_t tsapiEngineGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                   unsigned int gpuId,
                                                   dcgmDeviceAttributes_t *pDcgmDeviceAttr)
{
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

static dcgmReturn_t tsapiEngineGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle,
                                                        dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    return helperGetCacheManagerFieldInfo(pDcgmHandle, fieldInfo);
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
    return helperGetMultipleValuesForField(
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

static dcgmReturn_t tsapiEngineUnwatchFieldValue(dcgmHandle_t pDcgmHandle,
                                                 int gpuId,
                                                 unsigned short fieldId,
                                                 int clearCache)
{
    return helperUnwatchFieldValue(pDcgmHandle, gpuId, fieldId, clearCache);
}

static dcgmReturn_t tsapiEngineUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate)
{
    return helperUpdateAllFields(pDcgmHandle, waitForUpdate);
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

static dcgmReturn_t tsapiEnginePolicyTrigger(dcgmHandle_t pDcgmHandle)
{
    /* Policy management is now edge-triggered, so this function has no reason
       to exist anymore. Also, it only ever worked in the embedded case.
       Just returning OK to not break old clients. */
    return DCGM_ST_OK;
}

static dcgmReturn_t tsapiEnginePolicyRegister(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgmPolicyCondition_t condition,
                                              fpRecvUpdates beginCallback,
                                              fpRecvUpdates finishCallback)
{
    return helperPolicyRegister(pDcgmHandle, groupId, condition, beginCallback, finishCallback);
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

    return helperHealthCheckV4(pDcgmHandle, groupId, reinterpret_cast<dcgmHealthResponse_v4 *>(response));
}

static dcgmReturn_t tsapiEngineActionValidate_v2(dcgmHandle_t pDcgmHandle,
                                                 dcgmRunDiag_t *drd,
                                                 dcgmDiagResponse_t *response)
{
    return helperActionManager(pDcgmHandle, drd, DCGM_POLICY_ACTION_NONE, response);
}

static dcgmReturn_t tsapiEngineActionValidate(dcgmHandle_t pDcgmHandle,
                                              dcgmGpuGrp_t groupId,
                                              dcgmPolicyValidation_t validate,
                                              dcgmDiagResponse_t *response)
{
    dcgmRunDiag_t drd = {};
    drd.version       = dcgmRunDiag_version7;
    drd.validate      = validate;
    drd.groupId       = groupId;
    return helperActionManager(pDcgmHandle, &drd, DCGM_POLICY_ACTION_NONE, response);
}

static dcgmReturn_t tsapiEngineRunDiagnostic(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmDiagnosticLevel_t diagLevel,
                                             dcgmDiagResponse_t *diagResponse)
{
    dcgmPolicyValidation_t validation = DCGM_POLICY_VALID_NONE;
    dcgmRunDiag_t drd                 = {};

    if (!diagResponse)
        return DCGM_ST_BADPARAM;

    if (!diagResponse->version)
    {
        PRINT_DEBUG("", "Version missing");
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

        case DCGM_DIAG_LVL_INVALID:
        default:
            PRINT_ERROR("%d", "Invalid diagLevel %d", (int)diagLevel);
            return DCGM_ST_BADPARAM;
    }

    drd.version  = dcgmRunDiag_version;
    drd.groupId  = groupId;
    drd.validate = validation;

    return helperActionManager(pDcgmHandle, &drd, DCGM_POLICY_ACTION_NONE, diagResponse);
}

static dcgmReturn_t tsapiEngineStopDiagnostic(dcgmHandle_t pDcgmHandle)
{
    return helperStopDiag(pDcgmHandle);
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

static dcgmReturn_t tsapiEngineJobStartStats(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64])
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgm::Command *pGroupCmd;                /* Temp reference to the command */
    dcgmReturn_t ret;

    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::JOB_START_STATS, dcgm::OPERATION_GROUP_ENTITIES, (intptr_t)groupId, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_str(jobId);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

dcgmReturn_t cmTsapiEngineJobStartStats(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64])
{
    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    return helperJobStatCmd(pDcgmHandle, groupId, jobId, DCGM_CORE_SR_JOB_START_STATS);
}

static dcgmReturn_t tsapiEngineJobStopStats(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgm::Command *pGroupCmd;                /* Temp reference to the command */
    dcgmReturn_t ret;

    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::JOB_STOP_STATS, dcgm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_str(jobId);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

dcgmReturn_t cmTsapiEngineJobStopStats(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    return helperJobStatCmd(pDcgmHandle, 0, jobId, DCGM_CORE_SR_JOB_STOP_STATS);
}

static dcgmReturn_t tsapiEngineJobGetStats(dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo)
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if ((NULL == jobId) || (NULL == pJobInfo))
        return DCGM_ST_BADPARAM;

    /* Valid version can't be 0 or just any random number  */
    if (pJobInfo->version != dcgmJobInfo_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    if ((0 == jobId[0]))
    {
        PRINT_DEBUG("", "Job ID was NULL");
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::JOB_GET_INFO, dcgm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Add the args to the command to be sent over the network */
    pCmdTemp->add_arg()->set_str(jobId);
    pCmdTemp->add_arg()->set_blob(pJobInfo, sizeof(*pJobInfo));

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Check the status of the DCGM command */
    if (vecCmdsRef[0]->status() != DCGM_ST_OK)
    {
        return (dcgmReturn_t)vecCmdsRef[0]->status();
    }

    if (!vecCmdsRef[0]->arg_size())
    {
        PRINT_ERROR("", "Arg size of 0 unexpected");
        return DCGM_ST_GENERIC_ERROR;
    }

    if (!vecCmdsRef[0]->arg(0).has_str())
    {
        PRINT_ERROR("", "Response missing job id");
        return DCGM_ST_GENERIC_ERROR;
    }

    if (!vecCmdsRef[0]->arg(1).has_blob())
    {
        PRINT_ERROR("", "Response missing blob");
        return DCGM_ST_GENERIC_ERROR;
    }

    if (vecCmdsRef[0]->arg(1).blob().size() > sizeof(*pJobInfo))
    {
        PRINT_ERROR("%d %d",
                    "Returned blob size %d > structSize %d",
                    (int)vecCmdsRef[0]->arg(0).blob().size(),
                    (int)sizeof(*pJobInfo));
        return DCGM_ST_GENERIC_ERROR;
    }

    memcpy(pJobInfo, (void *)vecCmdsRef[0]->arg(1).blob().c_str(), vecCmdsRef[0]->arg(1).blob().size());

    return DCGM_ST_OK;
}

dcgmReturn_t cmTsapiEngineJobGetStats(dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo)
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

static dcgmReturn_t tsapiEngineJobRemove(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgm::Command *pGroupCmd;                /* Temp reference to the command */
    dcgmReturn_t ret;

    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::JOB_REMOVE, dcgm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    pCmdTemp->add_arg()->set_str(jobId);

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

dcgmReturn_t cmTsapiEngineJobRemove(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    if ((NULL == jobId) || (0 == jobId[0]))
    {
        return DCGM_ST_BADPARAM;
    }

    return helperJobStatCmd(pDcgmHandle, 0, jobId, DCGM_CORE_SR_JOB_REMOVE);
}

static dcgmReturn_t tsapiEngineJobRemoveAll(dcgmHandle_t pDcgmHandle)
{
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgm::Command *pGroupCmd;                /* Temp reference to the command */
    dcgmReturn_t ret;

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::JOB_REMOVE_ALL, dcgm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = processAtHostEngine(pDcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Get reference to returned command in a local reference */
    pGroupCmd = vecCmdsRef[0];

    /* Return the global status returned from the operation at the hostengine */
    return (dcgmReturn_t)pGroupCmd->status();
}

dcgmReturn_t cmTsapiEngineJobRemoveAll(dcgmHandle_t pDcgmHandle)
{
    return helperJobStatCmd(pDcgmHandle, 0, "", DCGM_CORE_SR_JOB_REMOVE_ALL);
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
        return DCGM_ST_BADPARAM;

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

static dcgmReturn_t tsapiMetadataToggleState(dcgmHandle_t dcgmHandle, dcgmIntrospectState_t enabledStatus)
{
    dcgm_introspect_msg_toggle_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_STATE_TOGGLE;
    msg.header.version    = dcgm_introspect_msg_toggle_version;
    msg.enabledStatus     = enabledStatus;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    return dcgmReturn;
}

static dcgmReturn_t tsapiMetadataStateSetRunInterval(dcgmHandle_t dcgmHandle, unsigned int runIntervalMs)
{
    dcgm_introspect_msg_set_interval_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_STATE_SET_RUN_INTERVAL;
    msg.header.version    = dcgm_introspect_msg_set_interval_version;

    msg.runIntervalMs = runIntervalMs;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    return dcgmReturn;
}

static dcgmReturn_t tsapiMetadataUpdateAll(dcgmHandle_t dcgmHandle, int waitForUpdate)
{
    dcgm_introspect_msg_update_all_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_UPDATE_ALL;
    msg.header.version    = dcgm_introspect_msg_update_all_version;

    msg.waitForUpdate = waitForUpdate;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
    return dcgmReturn;
}

static dcgmReturn_t tsapiIntrospectGetFieldsMemoryUsage(dcgmHandle_t dcgmHandle,
                                                        dcgmIntrospectContext_t *context,
                                                        dcgmIntrospectFullMemory_t *memoryInfo,
                                                        int waitIfNoData)
{
    dcgm_introspect_msg_fields_mem_usage_t msg;
    dcgmReturn_t dcgmReturn;

    if ((!context) || (!memoryInfo))
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    /* Valid version can't be 0 or just any random number  */
    if (context->version != dcgmIntrospectContext_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    /* Valid version can't be 0 or just any random number  */
    if (memoryInfo->version != dcgmIntrospectFullMemory_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    if (static_cast<std::underlying_type_t<dcgmIntrospectLevel_t>>(context->introspectLvl)
            <= static_cast<std::underlying_type_t<dcgmIntrospectLevel_t>>(DCGM_INTROSPECT_LVL_INVALID)
        || static_cast<std::underlying_type_t<dcgmIntrospectLevel_t>>(context->introspectLvl)
               > static_cast<std::underlying_type_t<dcgmIntrospectLevel_t>>(DCGM_INTROSPECT_LVL_ALL_FIELDS))
    {
        PRINT_ERROR("", "Bad introspection level");
        return DCGM_ST_BADPARAM;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_FIELDS_MEM_USAGE;
    msg.header.version    = dcgm_introspect_msg_fields_mem_usage_version;

    memcpy(&msg.context, context, sizeof(msg.context));
    memcpy(&msg.memoryInfo, memoryInfo, sizeof(msg.memoryInfo));
    msg.waitIfNoData = waitIfNoData;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    memcpy(memoryInfo, &msg.memoryInfo, sizeof(msg.memoryInfo));
    return dcgmReturn;
}

static dcgmReturn_t tsapiIntrospectGetFieldMemoryUsage(dcgmHandle_t dcgmHandle,
                                                       unsigned short fieldId,
                                                       dcgmIntrospectFullMemory_t *memoryInfo,
                                                       int waitIfNoData)
{
    dcgmIntrospectContext_t context;
    context.version       = dcgmIntrospectContext_version;
    context.introspectLvl = DCGM_INTROSPECT_LVL_FIELD;
    context.fieldId       = fieldId;


    return tsapiIntrospectGetFieldsMemoryUsage(dcgmHandle, &context, memoryInfo, waitIfNoData);
}

static dcgmReturn_t tsapiIntrospectGetFieldsExecTime(dcgmHandle_t dcgmHandle,
                                                     dcgmIntrospectContext_t *context,
                                                     dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                     int waitIfNoData)
{
    dcgm_introspect_msg_fields_exec_time_t msg;
    dcgmReturn_t dcgmReturn;

    if ((!context) || (!execTime))
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    /* Valid version can't be 0 or just any random number  */
    if (context->version != dcgmIntrospectContext_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    /* Valid version can't be 0 or just any random number  */
    if (execTime->version != dcgmIntrospectFullFieldsExecTime_version)
    {
        PRINT_DEBUG("", "Version Mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    if (static_cast<std::underlying_type_t<dcgmIntrospectLevel_t>>(context->introspectLvl)
            <= static_cast<std::underlying_type_t<dcgmIntrospectLevel_t>>(DCGM_INTROSPECT_LVL_INVALID)
        || static_cast<std::underlying_type_t<dcgmIntrospectLevel_t>>(context->introspectLvl)
               > static_cast<std::underlying_type_t<dcgmIntrospectLevel_t>>(DCGM_INTROSPECT_LVL_ALL_FIELDS))
    {
        PRINT_ERROR("", "Bad introspection level");
        return DCGM_ST_BADPARAM;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_FIELDS_EXEC_TIME;
    msg.header.version    = dcgm_introspect_msg_fields_exec_time_version;

    memcpy(&msg.context, context, sizeof(msg.context));
    memcpy(&msg.execTime, execTime, sizeof(msg.execTime));
    msg.waitIfNoData = waitIfNoData;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    memcpy(execTime, &msg.execTime, sizeof(msg.execTime));
    return dcgmReturn;
}

static dcgmReturn_t tsapiIntrospectGetFieldExecTime(dcgmHandle_t dcgmHandle,
                                                    unsigned short fieldId,
                                                    dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                    int waitIfNoData)
{
    dcgmIntrospectContext_t context;
    context.version       = dcgmIntrospectContext_version;
    context.introspectLvl = DCGM_INTROSPECT_LVL_FIELD;
    context.fieldId       = fieldId;

    return tsapiIntrospectGetFieldsExecTime(dcgmHandle, &context, execTime, waitIfNoData);
}

static dcgmReturn_t tsapiIntrospectGetHostengineMemoryUsage(dcgmHandle_t dcgmHandle,
                                                            dcgmIntrospectMemory_t *memoryInfo,
                                                            int waitIfNoData)
{
    dcgm_introspect_msg_he_mem_usage_t msg;
    dcgmReturn_t dcgmReturn;

    if (!memoryInfo)
        return DCGM_ST_BADPARAM;
    if (memoryInfo->version != dcgmIntrospectMemory_version)
    {
        PRINT_ERROR("%X %X", "Version mismatch x%X != x%X", memoryInfo->version, dcgmIntrospectMemory_version);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE;
    msg.header.version    = dcgm_introspect_msg_he_mem_usage_version;
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
    dcgm_introspect_msg_he_cpu_util_t msg;
    dcgmReturn_t dcgmReturn;

    if (!cpuUtil)
        return DCGM_ST_BADPARAM;
    if (cpuUtil->version != dcgmIntrospectCpuUtil_version)
    {
        PRINT_ERROR("%X %X", "Version mismatch x%X != x%X", cpuUtil->version, dcgmIntrospectCpuUtil_version);
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdIntrospect;
    msg.header.subCommand = DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL;
    msg.header.version    = dcgm_introspect_msg_he_cpu_util_version;

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
static dcgmReturn_t tsapiModuleBlacklist(dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId)
{
    if (moduleId <= DcgmModuleIdCore || moduleId >= DcgmModuleIdCount)
    {
        DCGM_LOG_ERROR << "Bad module ID " << moduleId;
        return DCGM_ST_BADPARAM;
    }

    dcgm_core_msg_module_blacklist_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_MODULE_BLACKLIST;
    msg.header.version    = dcgm_core_msg_module_blacklist_version;

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
    dcgm_profiling_msg_get_mgs_t msg;
    dcgmReturn_t dcgmReturn;

    if (!metricGroups)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    if (metricGroups->version != dcgmProfGetMetricGroups_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_GET_MGS;
    msg.header.version    = dcgm_profiling_msg_get_mgs_version;

    memcpy(&msg.metricGroups, metricGroups, sizeof(*metricGroups));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Copy the response back over the request */
    memcpy(metricGroups, &msg.metricGroups, sizeof(*metricGroups));
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiProfWatchFields(dcgmHandle_t dcgmHandle, dcgmProfWatchFields_t *watchFields)
{
    dcgm_profiling_msg_watch_fields_t msg;
    dcgmReturn_t dcgmReturn;

    if (!watchFields)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    if (watchFields->version != dcgmProfWatchFields_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_WATCH_FIELDS;
    msg.header.version    = dcgm_profiling_msg_watch_fields_version;

    memcpy(&msg.watchFields, watchFields, sizeof(*watchFields));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Copy the response back over the request */
    memcpy(watchFields, &msg.watchFields, sizeof(*watchFields));
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiProfUnwatchFields(dcgmHandle_t dcgmHandle, dcgmProfUnwatchFields_t *unwatchFields)
{
    dcgm_profiling_msg_unwatch_fields_t msg;
    dcgmReturn_t dcgmReturn;

    if (!unwatchFields)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    if (unwatchFields->version != dcgmProfUnwatchFields_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_UNWATCH_FIELDS;
    msg.header.version    = dcgm_profiling_msg_unwatch_fields_version;

    memcpy(&msg.unwatchFields, unwatchFields, sizeof(*unwatchFields));

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));

    /* Copy the response back over the request */
    memcpy(unwatchFields, &msg.unwatchFields, sizeof(*unwatchFields));
    return dcgmReturn;
}

/*****************************************************************************/
static dcgmReturn_t helperProfPauseResume(dcgmHandle_t dcgmHandle, bool pause)
{
    dcgm_profiling_msg_pause_resume_t msg;
    dcgmReturn_t dcgmReturn;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_PAUSE_RESUME;
    msg.header.version    = dcgm_profiling_msg_pause_resume_version;
    msg.pause             = pause;

    // coverity[overrun-buffer-arg]
    dcgmReturn = dcgmModuleSendBlockingFixedRequest(dcgmHandle, &msg.header, sizeof(msg));
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

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmStartEmbedded_v2(dcgmStartEmbeddedV2Params_v1 *params)
{
    if (params == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (params->version != dcgmStartEmbeddedV2Params_version1)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    if ((params->opMode != DCGM_OPERATION_MODE_AUTO) && (params->opMode != DCGM_OPERATION_MODE_MANUAL))
    {
        return DCGM_ST_BADPARAM;
    }

    if (!g_dcgmGlobals.isInitialized)
    {
        PRINT_ERROR("", "dcgmStartEmbedded before dcgmInit()");
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmGlobalsLock();

    /* Check again after lock */
    if (!g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        PRINT_ERROR("", "dcgmStartEmbedded before dcgmInit() after lock");
        return DCGM_ST_UNINITIALIZED;
    }

    /* Initialize the logger */
    std::string paramsLogFile;
    if (params->logFile != nullptr)
        paramsLogFile = params->logFile;

    const std::string logFile = DcgmLogging::getLogFilenameFromArgAndEnv(
        paramsLogFile, DCGM_LOGGING_DEFAULT_HOSTENGINE_FILE, DCGM_ENV_LOG_PREFIX);

    // If logging severity is unspecified, pass empty string as the arg to the
    // helper. Empty arg => no arg => look at env
    const std::string loggingSeverityArg
        = params->severity == DcgmLoggingSeverityUnspecified
              ? ""
              : DcgmLogging::severityToString(params->severity, DCGM_LOGGING_SEVERITY_STRING_ERROR);

    const std::string logSeverity = DcgmLogging::getLogSeverityFromArgAndEnv(
        loggingSeverityArg, DCGM_LOGGING_DEFAULT_HOSTENGINE_SEVERITY, DCGM_ENV_LOG_PREFIX);

    DcgmLogging::init(logFile.c_str(), DcgmLogging::severityFromString(logSeverity.c_str(), DcgmLoggingSeverityError));
    DcgmLogging &logging = DcgmLogging::getInstance();
    logging.routeLogToBaseLogger<SYSLOG_LOGGER>();
    DCGM_LOG_DEBUG << "Initialized base logger";
    DCGM_LOG_SYSLOG_DEBUG << "Initialized syslog logger";

    DCGM_LOG_INFO << DcgmNs::DcgmBuildInfo().GetBuildInfoStr();
    /* See if the host engine is running already */
    void *pHostEngineInstance = DcgmHostEngineHandler::Instance();
    if (pHostEngineInstance)
    {
        g_dcgmGlobals.embeddedEngineStarted = 1; /* No harm in making sure this is true */
        dcgmGlobalsUnlock();
        PRINT_DEBUG("", "dcgmStartEmbedded(): host engine was already running");
        return DCGM_ST_OK;
    }

    pHostEngineInstance = DcgmHostEngineHandler::Init(*params);
    if (pHostEngineInstance == nullptr)
    {
        dcgmGlobalsUnlock();
        PRINT_ERROR("", "DcgmHostEngineHandler::Init failed");
        return DCGM_ST_INIT_ERROR;
    }

    g_dcgmGlobals.embeddedEngineStarted = 1;

    dcgmGlobalsUnlock();

    params->dcgmHandle = (dcgmHandle_t)DCGM_EMBEDDED_HANDLE;
    PRINT_DEBUG("", "dcgmStartEmbedded(): Embedded host engine started");

    return DCGM_ST_OK;
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
        PRINT_ERROR("", "dcgmStopEmbedded before dcgmInit()");
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
        PRINT_ERROR("", "dcgmStopEmbedded before dcgmInit() after lock");
        return DCGM_ST_UNINITIALIZED;
    }

    if (g_dcgmGlobals.embeddedEngineStarted)
    {
        DcgmHostEngineHandler *heHandler = DcgmHostEngineHandler::Instance();

        if (!heHandler)
        {
            PRINT_ERROR("", "embeddedEngineStarted was set but heHandler is NULL");
        }
        else
        {
            // Invoke the cleanup method
            (void)DcgmHostEngineHandler::Instance()->Cleanup();
            PRINT_DEBUG("", "embedded host engine cleaned up");
        }
        g_dcgmGlobals.embeddedEngineStarted = 0;
    }

    dcgmGlobalsUnlock();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmConnect(char *ipAddress, dcgmHandle_t *pDcgmHandle)
{
    dcgmConnectV2Params_t connectParams;

    /* Set up default parameters for dcgmConnect_v2 */
    memset(&connectParams, 0, sizeof(connectParams));
    connectParams.version                = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = 0;

    return dcgmConnect_v2(ipAddress, &connectParams, pDcgmHandle);
}

/*****************************************************************************/
static dcgmReturn_t sendClientLogin(dcgmHandle_t dcgmHandle, dcgmConnectV2Params_t *connectParams)
{
    dcgm::ClientLogin *pClientLogin;         /* Protobuf Arg */
    DcgmProtobuf encodePrb;                  /* Protobuf message for encoding */
    DcgmProtobuf decodePrb;                  /* Protobuf message for decoding */
    dcgm::Command *pCmdTemp;                 /* Pointer to proto command for intermediate usage */
    std::vector<dcgm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;

    if (!connectParams)
        return DCGM_ST_BADPARAM;

    pClientLogin = new dcgm::ClientLogin;
    pClientLogin->set_persistafterdisconnect(connectParams->persistAfterDisconnect);

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(dcgm::CLIENT_LOGIN, dcgm::OPERATION_SYSTEM, 0, 0);
    if (NULL == pCmdTemp)
    {
        delete pClientLogin;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set the arg to passed as a proto message. After this point the arg is
     * managed by protobuf library, so don't worry about freeing it after this point */
    pCmdTemp->add_arg()->set_allocated_clientlogin(pClientLogin);

    ret = processAtHostEngine(dcgmHandle, &encodePrb, &decodePrb, &vecCmdsRef);
    if (DCGM_ST_OK != ret)
    {
        PRINT_ERROR("%d", "Got st %d from processAtHostEngine", (int)ret);
        return ret;
    }

    /* Check the status of the DCGM command */
    ret = (dcgmReturn_t)vecCmdsRef[0]->status();
    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got st %d from vecCmdsRef[0]->status()", (int)vecCmdsRef[0]->status());
    }

    return ret;
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
        PRINT_ERROR("", "dcgmConnect before dcgmInit()");
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
        PRINT_ERROR(
            "%X %X", "dcgmConnect_v2 Version mismatch %X != %X", connectParams->version, dcgmConnectV2Params_version);
        return DCGM_ST_VER_MISMATCH;
    }

    DcgmClientHandler *clientHandler = dcgmapiAcquireClientHandler(true);
    if (!clientHandler)
    {
        PRINT_ERROR("", "Unable to allocate client handler");
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Add connection to the client handler */
    dcgmReturn_t status = clientHandler->GetConnHandleForHostEngine(
        ipAddress, pDcgmHandle, connectParams->timeoutMs, connectParams->addressIsUnixSocket ? true : false);
    dcgmapiReleaseClientHandler();
    if (DCGM_ST_OK != status)
    {
        PRINT_ERROR("%s %d", "GetConnHandleForHostEngine ip %s returned %d", ipAddress, (int)status);
        return status;
    }

    PRINT_DEBUG("%s %p", "Connected to ip %s as dcgmHandle %p", ipAddress, (void *)*pDcgmHandle);

    /* Send our connection options to the host engine */
    dcgmReturn = sendClientLogin(*pDcgmHandle, connectParams);
    if (dcgmReturn != DCGM_ST_OK)
    {
        /* Abandon the connection if we can't login */
        PRINT_ERROR("%d %p",
                    "Got error %d from sendClientLogin on connection %p. Abandoning connection.",
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
        PRINT_WARNING("", "dcgmDisconnect before dcgmInit()");
        /* Returning OK here to prevent errors from being logged from the
           python framework when DcgmHandle objects are garbage collected after
           dcgmShutdown has already been called. */
        return DCGM_ST_OK;
    }

    DcgmClientHandler *clientHandler = dcgmapiAcquireClientHandler(false);
    if (!clientHandler)
    {
        PRINT_WARNING("", "dcgmDisconnect called while client handler was not allocated.");
        /* Returning OK here to prevent errors from being logged from the
           python framework when DcgmHandle objects are garbage collected after
           dcgmShutdown has already been called. */
        return DCGM_ST_OK;
    }

    /* Actually close the connection */
    clientHandler->CloseConnForHostEngine(pDcgmHandle);

    dcgmapiReleaseClientHandler();

    PRINT_DEBUG("%p", "dcgmDisconnect closed connection with handle %p", (void *)pDcgmHandle);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DCGM_PUBLIC_API dcgmInit(void)
{
    if (g_dcgmGlobals.isInitialized)
    {
        PRINT_DEBUG("", "dcgmInit was already initialized");
        return DCGM_ST_OK;
    }

    dcgmGlobalsLock();

    /* Check again now that we have the lock */
    if (g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        PRINT_DEBUG("", "dcgmInit was already initialized after lock");
        return DCGM_ST_OK;
    }

    /* globals are uninitialized. Zero the structure */
    memset(&g_dcgmGlobals, 0, sizeof(g_dcgmGlobals));

    int ret = DcgmFieldsInit();
    if (ret != DCGM_ST_OK)
    {
        /* Undo any initialization done above */
        dcgmGlobalsUnlock();
        PRINT_ERROR("", "DcgmFieldsInit failed");
        return DCGM_ST_INIT_ERROR;
    }
    g_dcgmGlobals.fieldsAreInitialized = 1;

    /* Fully-initialized. Mark structure as such */
    g_dcgmGlobals.isInitialized = 1;

    dcgmGlobalsUnlock();

    PRINT_DEBUG("", "dcgmInit was successful");
    return DCGM_ST_OK;
}

/*****************************************************************************/

dcgmReturn_t DCGM_PUBLIC_API dcgmShutdown()
{
    if (!g_dcgmGlobals.isInitialized)
    {
        PRINT_DEBUG("", "dcgmShutdown called when DCGM was uninitialized.");
        return DCGM_ST_OK;
    }

    /* Clean up remote connections - must NOT have dcgmGlobalsLock() here or we will
       deadlock */
    PRINT_DEBUG("", "Before dcgmapiFreeClientHandler");
    dcgmapiFreeClientHandler();
    PRINT_DEBUG("", "After dcgmapiFreeClientHandler");

    dcgmGlobalsLock();

    if (!g_dcgmGlobals.isInitialized)
    {
        dcgmGlobalsUnlock();
        PRINT_DEBUG("", "dcgmShutdown called when DCGM was uninitialized - after lock.");
        return DCGM_ST_UNINITIALIZED;
    }

    if (g_dcgmGlobals.embeddedEngineStarted)
    {
        DcgmHostEngineHandler *heHandler = DcgmHostEngineHandler::Instance();

        if (!heHandler)
        {
            PRINT_ERROR("", "embeddedEngineStarted was set but heHandler is NULL");
        }
        else
        {
            // Invoke the cleanup method
            (void)DcgmHostEngineHandler::Instance()->Cleanup();
            PRINT_DEBUG("", "host engine cleaned up");
        }
        g_dcgmGlobals.embeddedEngineStarted = 0;
    }

    DcgmFieldsTerm();
    g_dcgmGlobals.fieldsAreInitialized = 0;

    g_dcgmGlobals.isInitialized = 0;

    dcgmGlobalsUnlock();

    PRINT_DEBUG("", "dcgmShutdown completed successfully");

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
        { DcgmModuleIdProfiling, MODULE_PROFILING_NAME },
    };

    auto it = moduleNames.find(id);
    if (it == moduleNames.end())
    {
        return DCGM_ST_BADPARAM;
    }

    *name = it->second;
    return DCGM_ST_OK;
}

/*****************************************************************************/
