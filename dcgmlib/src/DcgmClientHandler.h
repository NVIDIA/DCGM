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

#include "DcgmIpc.h"
#include "DcgmMutex.h"
#include "DcgmRequest.h"
#include "dcgm_module_structs.h"
#include "dcgm_structs.h"
#include <DcgmBuildInfo.hpp>
#include <iostream>

typedef struct
{
    dcgmReturn_t dcgmReturn;               /* Any error returned from the request like a disconnect */
    std::unique_ptr<DcgmMessage> response; /* The reply message we waited for */
} DcgmClientBlockingResponse_t;

/* Attributes of each client connection of the DcgmClientHandler */
class DCHConnectionAttributes
{
public:
    DCHConnectionAttributes(const char *buildInfoString)
        : m_buildInfo(DcgmNs::DcgmBuildInfo(buildInfoString))
    {}

    DcgmNs::DcgmBuildInfo m_buildInfo; /* Build info retrieved from the server */
};

class DcgmClientHandler
{
public:
    /*****************************************************************************
     * Constructor/destructor
     *****************************************************************************/
    DcgmClientHandler();
    virtual ~DcgmClientHandler();

    /*****************************************************************************
     * This method is used to get Connection to the host engine corresponding to
     * the IP address
     *****************************************************************************/
    dcgmReturn_t GetConnHandleForHostEngine(const char *identifier,
                                            dcgmHandle_t *pDcgmHandle,
                                            unsigned int timeoutMs,
                                            bool addressIsUnixSocket);

    /*****************************************************************************
     * This method is used to close connection with the Host Engine
     *****************************************************************************/
    void CloseConnForHostEngine(dcgmHandle_t pConnHandle);

    /*****************************************************************************
     * Module command version of ExchangeMsgAsync
     *
     *****************************************************************************/
    dcgmReturn_t ExchangeModuleCommandAsync(dcgmHandle_t dcgmHandle,
                                            dcgm_module_command_header_t *moduleCommand,
                                            std::unique_ptr<DcgmRequest> request,
                                            size_t maxResponseSize,
                                            unsigned int timeoutMs = 60000);

#ifndef DCGMLIB_TESTS
private:
#endif
    DcgmIpc m_dcgmIpc;
    std::atomic<dcgm_request_id_t> m_requestId = DCGM_REQUEST_ID_NONE;
    DcgmMutex m_mutex; /* Lock to use for threaded data structures */

    /* Requests that block and are no longer tracked once they complete.
       Protected by m_mutex */
    std::unordered_map<dcgm_request_id_t, std::promise<DcgmClientBlockingResponse_t>> m_blockingReqs;

    /* Requests that block on initial response and then receive notifications (like policy monitoring)
       Protected by m_mutex */
    std::unordered_map<dcgm_request_id_t, std::unique_ptr<DcgmRequest>> m_persistentReqs;

    /* A map of connectionId-> set of requestIds. This is here so we can delete outstanding
        m_blockingReqs and m_persistentReqs entries when a client unexpectedly disconnects.
        Protected by m_mutex */
    std::unordered_map<dcgm_connection_id_t, std::unordered_set<dcgm_request_id_t>> m_connectionRequests;

    /* A map of connectionId -> attributes for each connection */
    std::unordered_map<dcgm_connection_id_t, DCHConnectionAttributes> m_connectionAttributes;

    dcgmReturn_t splitIdentifierAndPort(std::string_view identifier,
                                        bool const addressIsUnixSocket,
                                        std::vector<char> &result,
                                        unsigned int &portNumber);

    dcgmReturn_t TryConnectingToHostEngine(char const identifier[],
                                           unsigned int portNumber,
                                           dcgmHandle_t *pDcgmHandle,
                                           bool addressIsUnixSocket,
                                           int connectionTimeoutMs);

    /*************************************************************************/
    /* Callback functions for DcgmIpc to call */
    static void ProcessMessageStatic(dcgm_connection_id_t connectionId,
                                     std::unique_ptr<DcgmMessage> dcgmMessage,
                                     void *userData);

    void ProcessMessage(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> dcgmMessage);

    static void ProcessDisconnectStatic(dcgm_connection_id_t connectionId, void *userData);

    void ProcessDisconnect(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /* Populate connection attributes for a newly established connection.
     *
     * Returns DCGM_ST_OK if OK
     *         Other nonzero DCGM_ST_? on error.
     */
    dcgmReturn_t PopulateConnectionAttributes(dcgmHandle_t dcgmHandle);

    /*************************************************************************/

    /* Get the next request ID to use for a client request */
    dcgm_request_id_t GetNextRequestId(void);

    /* Helpers for manipulating m_blockingReqs */
    std::future<DcgmClientBlockingResponse_t> AddBlockingRequest(dcgm_connection_id_t connectionId,
                                                                 dcgm_request_id_t requestId);
    void RemoveBlockingRequest(dcgm_connection_id_t connectionId,
                               dcgm_request_id_t requestId,
                               std::optional<dcgmReturn_t> errorToSetForFuture);

    /* Helpers for manipulating m_persistentReqs */
    void AddPersistentRequest(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmRequest> request);
    void RemovePersistentRequest(dcgm_connection_id_t connectionId, dcgm_request_id_t requestId);
};
