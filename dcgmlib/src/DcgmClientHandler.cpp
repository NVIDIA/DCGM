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

#include "DcgmClientHandler.h"
#include "DcgmLogging.h"
#include "DcgmMutex.h"
#include "DcgmProtocol.h"
#include "DcgmRequest.h"
#include "DcgmSettings.h"
#include "dcgm_util.h"
#include "timelib.h"
#include <DcgmIpc.h>
#include <dcgm_core_structs.h>
#include <iostream>
#include <netinet/in.h>
#include <stdexcept>
#include <stdlib.h>

/*****************************************************************************
 * Constructor
 *****************************************************************************/
DcgmClientHandler::DcgmClientHandler()
    : m_dcgmIpc(1)
    , m_mutex(0)
{
    dcgmReturn_t dcgmReturn = m_dcgmIpc.Init(std::nullopt,
                                             std::nullopt,
                                             DcgmClientHandler::ProcessMessageStatic,
                                             this,
                                             DcgmClientHandler::ProcessDisconnectStatic,
                                             this);
    if (dcgmReturn != DCGM_ST_OK)
    {
        std::stringstream ss;
        ss << "m_dcgmIpc.Init raised error " << errorString(dcgmReturn);
        const std::string errorStr = ss.str();
        DCGM_LOG_ERROR << errorStr;
        throw std::runtime_error(errorStr);
    }
}

/*****************************************************************************
 * Destructor
 *****************************************************************************/
DcgmClientHandler::~DcgmClientHandler()
{
    /* Explicitly tell DcgmIpc to shut down so we don't get async callbacks after
       this destructor */
    try
    {
        if (m_dcgmIpc.StopAndWait(60000))
        {
            DCGM_LOG_ERROR << "m_dcgmIpc.StopAndWait returned that it was still running.";
        }
    }
    catch (std::exception &ex)
    {
        DCGM_LOG_ERROR << "m_dcgmIpc.StopAndWait raised exception " << ex.what();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "m_dcgmIpc.StopAndWait raised unknown exception";
    }

    /* Clear all of the structures that are protected by locks */
    {
        DcgmLockGuard dlg(&m_mutex);
        m_blockingReqs.clear();
        m_persistentReqs.clear();
        m_connectionRequests.clear();
        m_connectionAttributes.clear();
    }
    // The lock guard above will obey RAII. m_mutex will be destructed with this instance.
    // coverity[missing_unlock: FALSE]
}

/*****************************************************************************/
void DcgmClientHandler::ProcessDisconnect(dcgm_connection_id_t connectionId)
{
    DCGM_LOG_VERBOSE << "ProcessDisconnect for connectionId " << connectionId;

    DcgmLockGuard dlg(&m_mutex);

    auto it = m_connectionRequests.find(connectionId);
    if (it == m_connectionRequests.end())
    {
        DCGM_LOG_DEBUG << "No requests were outstanding for connectionId " << connectionId;
        return;
    }

    int numFound = 0;

    /* Each iteration invalidates requestIt. So just get the first element
       until there are no more elements */
    for (auto requestIt = it->second.begin(); requestIt != it->second.end(); requestIt = it->second.begin())
    {
        /* For now, naively try to remove each request from both tables */
        dcgm_request_id_t requestId = *requestIt;
        RemoveBlockingRequest(connectionId, requestId, DCGM_ST_CONNECTION_NOT_VALID);
        RemovePersistentRequest(connectionId, requestId);
        DCGM_LOG_DEBUG << "Erased requestId " << requestId << " for connectionId " << connectionId;
        numFound++;
    }

    m_connectionRequests.erase(connectionId);

    m_connectionAttributes.erase(connectionId);

    DCGM_LOG_DEBUG << "Erased " << numFound << " requests for connectionId " << connectionId;
}

/*****************************************************************************/
void DcgmClientHandler::ProcessDisconnectStatic(dcgm_connection_id_t connectionId, void *userData)
{
    DcgmClientHandler *ch = (DcgmClientHandler *)userData;
    ch->ProcessDisconnect(connectionId);
}


/*****************************************************************************/
void DcgmClientHandler::ProcessMessage(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> dcgmMessage)
{
    dcgm_message_header_t *msgHdr = dcgmMessage->GetMessageHdr();

    if (msgHdr->requestId == DCGM_REQUEST_ID_NONE)
    {
        DCGM_LOG_ERROR << "Unexpected requestId of DCGM_REQUEST_ID_NONE for connectionId " << connectionId;
        return;
    }

    DcgmLockGuard dlg { &m_mutex };

    /* Only mark requests as complete if this is a response to a request (not an async notification) */
    if (dcgmMessage->IsAsyncNotification())
    {
        DCGM_LOG_DEBUG << "Not marking blocking request as done for async notification msgType " << msgHdr->msgType
                       << " requestId " << msgHdr->requestId;
    }
    else
    {
        auto itB = m_blockingReqs.find(msgHdr->requestId);
        if (itB != m_blockingReqs.end())
        {
            DCGM_LOG_DEBUG << "Found blocking request for requestId " << msgHdr->requestId;
            DcgmClientBlockingResponse_t response;
            response.dcgmReturn = DCGM_ST_OK;
            response.response   = std::move(dcgmMessage);
            itB->second.set_value(std::move(response));
            return;
        }
    }

    auto itP = m_persistentReqs.find(msgHdr->requestId);
    if (itP == m_persistentReqs.end())
    {
        DCGM_LOG_ERROR << "requestId " << msgHdr->requestId << " not found for connectionId " << connectionId;
        return;
    }

    DCGM_LOG_DEBUG << "Processed persistent requestId " << msgHdr->requestId;
    itP->second->ProcessMessage(std::move(dcgmMessage));
}

/*****************************************************************************/
void DcgmClientHandler::ProcessMessageStatic(dcgm_connection_id_t connectionId,
                                             std::unique_ptr<DcgmMessage> dcgmMessage,
                                             void *userData)
{
    DcgmClientHandler *clientHandler = (DcgmClientHandler *)userData;

    clientHandler->ProcessMessage(connectionId, std::move(dcgmMessage));
}

/*****************************************************************************/
/*
 * helper function for trying to connect to the hostengine.  Returns 0 on success in which
 * case pDcgmHandle will be set as well.
 */
dcgmReturn_t DcgmClientHandler::TryConnectingToHostEngine(char const identifier[],
                                                          unsigned int portNumber,
                                                          dcgmHandle_t *pDcgmHandle,
                                                          bool addressIsUnixSocket,
                                                          int connectionTimeoutMs)
{
    dcgm_connection_id_t connectionId { DCGM_CONNECTION_ID_NONE };
    dcgmReturn_t dcgmReturn;

    if (addressIsUnixSocket)
    {
        dcgmReturn = m_dcgmIpc.ConnectDomain(identifier, connectionId, connectionTimeoutMs);
    }
    else
    {
        dcgmReturn = m_dcgmIpc.ConnectTcp(identifier, portNumber, connectionId, connectionTimeoutMs);
    }

    *pDcgmHandle = (dcgmHandle_t)connectionId;
    return dcgmReturn;
}

/* For ip and port pairs, return the identifier and the port number.
   For addressIsUnixSocket, this is a no-op. */
dcgmReturn_t DcgmClientHandler::splitIdentifierAndPort(std::string_view identifier,
                                                       bool const addressIsUnixSocket,
                                                       std::vector<char> &result,
                                                       unsigned int &portNumber)
{
    if (identifier.empty())
    {
        log_warning("Rejecting empty identifier");
        return DCGM_ST_NO_DATA;
    }

    if (addressIsUnixSocket)
    {
        constexpr size_t SUN_PATH_MAX = 108;
        if (SUN_PATH_MAX < identifier.length())
        {
            log_warning("Rejecting AF_UNIX identifier with excessive length {}", identifier.length());
            return DCGM_ST_BADPARAM;
        }
    }
    // Accomodate '[' ... ']' ':' port
    else if ((2 * INET6_ADDRSTRLEN) < identifier.length())
    {
        log_warning("Rejecting AF_INET identifier with excessive length {}", identifier.length());
        return DCGM_ST_BADPARAM;
    }

    // create local copy of identifier
    std::vector<char> identifierTemp(identifier.length() + 1);
    std::copy(identifier.begin(), identifier.end(), identifierTemp.begin());
    identifierTemp.back() = '\0';

    /* Attempt to find :port */
    size_t pos = identifier.rfind(':');

    /* If the identifier is a unix socket, a bracketed address w/no port, or no port, use the default port */
    if (addressIsUnixSocket || identifier.back() == ']' || pos == identifier.npos)
    {
        portNumber = DCGM_HE_PORT_NUMBER;
        result     = std::move(identifierTemp);
        return DCGM_ST_OK;
    }

    /* If empty before/after the ':', address must not be valid */
    if (pos == 0 || identifier.size() <= (pos + 1))
    {
        log_debug("Rejecting partial identifier '{}'", identifier);
        return DCGM_ST_BADPARAM;
    }

    std::string_view portStr = identifier.substr(pos + 1, identifier.size() - pos - 1);
    unsigned int foundPort   = atoi(portStr.data());

    // Check if in valid range 1-65535
    if ((foundPort <= 0) || (foundPort > 65535))
    {
        log_debug("Rejecting invalid port {}'", portStr);
        return DCGM_ST_BADPARAM;
    }

    portNumber = foundPort;
    identifierTemp.resize(pos + 1);
    identifierTemp.back() = '\0';
    result                = std::move(identifierTemp);
    return DCGM_ST_OK;
}

/*****************************************************************************
 * Get Connection to the host engine corresponding to the IP address or FQDN
 *****************************************************************************/
dcgmReturn_t DcgmClientHandler::GetConnHandleForHostEngine(const char *identifier,
                                                           dcgmHandle_t *pDcgmHandle,
                                                           unsigned int timeoutMs,
                                                           bool addressIsUnixSocket)
{
    if (!timeoutMs)
    {
        timeoutMs = 5000; /* 5-second default timeout */
    }

    std::vector<char> identifierTemp {};
    unsigned int portNumber = 0;
    dcgmReturn_t st         = splitIdentifierAndPort(identifier, addressIsUnixSocket, identifierTemp, portNumber);

    if (st != DCGM_ST_OK)
        return st;

    unsigned int attempt       = 0;
    bool connected             = false;
    const unsigned int WAIT_MS = 50;

    timelib64_t start = timelib_usecSince1970();
    timelib64_t now   = start;

    for (;;)
    {
        attempt++;
        if (DCGM_ST_OK
            == TryConnectingToHostEngine(
                identifierTemp.data(), portNumber, pDcgmHandle, addressIsUnixSocket, timeoutMs))
        {
            connected = true;
            break;
        }

        now = timelib_usecSince1970();
        if ((now - start) + WAIT_MS * 1000 > timeoutMs * 1000)
        {
            break;
        }

        log_debug("failed connecting to hostengine, still going to try for {} more ms", timeoutMs - (start - now));
        usleep(WAIT_MS * 1000);
    }

    log_debug("finished {} connection attempts to hostengine in about {} ms", attempt, (now - start) / 1000);

    if (connected)
    {
        DCGM_LOG_DEBUG << "successfully connected to hostengine. Getting connection attributes.";
        PopulateConnectionAttributes((dcgm_connection_id_t)*pDcgmHandle);
        return DCGM_ST_OK;
    }
    else
    {
        log_error("failed to connect to hostengine");
        return DCGM_ST_CONNECTION_NOT_VALID;
    }
}

/*****************************************************************************
 * Closes connection to the host engine
 *****************************************************************************/
void DcgmClientHandler::CloseConnForHostEngine(dcgmHandle_t connHandle)
{
    dcgm_connection_id_t connectionId = (dcgm_connection_id_t)connHandle;

    if (connectionId == DCGM_CONNECTION_ID_NONE)
    {
        DCGM_LOG_ERROR << "null connectionId";
        return;
    }

    m_dcgmIpc.CloseConnection(connectionId);
}

/*****************************************************************************/
dcgm_request_id_t DcgmClientHandler::GetNextRequestId()
{
    dcgm_request_id_t newId = m_requestId++;

    /* Don't allocate a connection as id DCGM_REQUEST_ID_NONE. In practice,
       this will only happen after 2^32 connections */
    if (newId == DCGM_REQUEST_ID_NONE)
    {
        newId = m_requestId++;
    }

    return newId;
}

/*****************************************************************************/
std::future<DcgmClientBlockingResponse_t> DcgmClientHandler::AddBlockingRequest(dcgm_connection_id_t connectionId,
                                                                                dcgm_request_id_t requestId)
{
    DcgmLockGuard dlg(&m_mutex);

    std::promise<DcgmClientBlockingResponse_t> dcbr;
    m_connectionRequests[connectionId].insert(requestId);
    m_blockingReqs.insert({ requestId, std::move(dcbr) });
    return m_blockingReqs[requestId].get_future();
}

/*****************************************************************************/
void DcgmClientHandler::RemoveBlockingRequest(dcgm_connection_id_t connectionId,
                                              dcgm_request_id_t requestId,
                                              std::optional<dcgmReturn_t> errorToSetForFuture)
{
    DcgmLockGuard dlg(&m_mutex);
    auto it = m_blockingReqs.find(requestId);
    if (it != m_blockingReqs.end())
    {
        if (errorToSetForFuture.has_value())
        {
            /* Set the future to satisfied with an error value */
            DcgmClientBlockingResponse_t dcbr {};
            dcbr.dcgmReturn = *errorToSetForFuture; /* Changed this from .value() since coverity thinks we didn't check
                                                       .has_value above */
            it->second.set_value(std::move(dcbr));
        }
        m_blockingReqs.erase(it);
        DCGM_LOG_VERBOSE << "blocking requestId " << requestId << " was removed.";
    }
    else
    {
        DCGM_LOG_VERBOSE << "blocking requestId " << requestId << " was not found.";
    }

    m_connectionRequests[connectionId].erase(requestId);
}

/*****************************************************************************/
void DcgmClientHandler::AddPersistentRequest(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmRequest> request)
{
    DcgmLockGuard dlg(&m_mutex);
    dcgm_request_id_t requestId = request->GetRequestId();
    m_persistentReqs.insert({ requestId, std::move(request) });
    m_connectionRequests[connectionId].insert(requestId);
}

/*****************************************************************************/
void DcgmClientHandler::RemovePersistentRequest(dcgm_connection_id_t connectionId, dcgm_request_id_t requestId)
{
    DcgmLockGuard dlg(&m_mutex);
    auto it = m_persistentReqs.find(requestId);
    if (it != m_persistentReqs.end())
    {
        m_persistentReqs.erase(it);
        DCGM_LOG_VERBOSE << "persistent requestId " << requestId << " was removed.";
    }
    else
    {
        DCGM_LOG_VERBOSE << "persistent requestId " << requestId << " was not found.";
    }

    m_connectionRequests[connectionId].erase(requestId);
}

/*****************************************************************************/
dcgmReturn_t DcgmClientHandler::ExchangeModuleCommandAsync(dcgmHandle_t dcgmHandle,
                                                           dcgm_module_command_header_t *moduleCommand,
                                                           std::unique_ptr<DcgmRequest> request,
                                                           size_t maxResponseSize,
                                                           unsigned int timeoutMs)
{
    std::unique_ptr<DcgmMessage> dcgmSendMsg = std::make_unique<DcgmMessage>();
    dcgm_request_id_t requestId;
    dcgm_connection_id_t connectionId = (dcgm_connection_id_t)dcgmHandle;

    // Get Next Request ID
    requestId       = GetNextRequestId();
    auto requestFut = AddBlockingRequest(connectionId, requestId);

    /* Update Encoded Message with a header to be sent over socket */
    dcgmSendMsg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, requestId, DCGM_ST_OK, moduleCommand->length);
    auto msgData = dcgmSendMsg->GetMsgBytesPtr();
    msgData->resize(moduleCommand->length);
    memcpy(msgData->data(), moduleCommand, moduleCommand->length);

    if (request != nullptr)
    {
        request->SetRequestId(requestId);
        AddPersistentRequest(connectionId, std::move(request));
    }

    // Send the Message
    dcgmReturn_t retSt = m_dcgmIpc.SendMessage(connectionId, std::move(dcgmSendMsg), true);
    if (retSt != DCGM_ST_OK)
    {
        RemoveBlockingRequest(connectionId, requestId, retSt);
        RemovePersistentRequest(connectionId, requestId);
        return retSt;
    }

    if (timeoutMs == 0)
    {
        requestFut.wait();
    }
    else
    {
        auto futStatus = requestFut.wait_for(std::chrono::milliseconds(timeoutMs));
        if (futStatus != std::future_status::ready)
        {
            DCGM_LOG_ERROR << "connectionId " << connectionId << " requestId " << requestId << " timed out after "
                           << timeoutMs << " ms.";
            RemoveBlockingRequest(connectionId, requestId, std::nullopt);
            RemovePersistentRequest(connectionId, requestId);
            return DCGM_ST_TIMEOUT;
        }
    }

    auto response = requestFut.get();
    RemoveBlockingRequest(connectionId, requestId, std::nullopt);

    if (response.dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "connectionId " << connectionId << " requestId " << requestId << " returned "
                       << errorString(response.dcgmReturn);
        return response.dcgmReturn;
    }

    DCGM_LOG_DEBUG << "Request Wait completed for connectionId " << connectionId << " request ID: " << requestId;

    dcgm_message_header_t *recvHeader = response.response->GetMessageHdr();

    if (recvHeader->msgType != DCGM_MSG_MODULE_COMMAND)
    {
        DCGM_LOG_ERROR << "Unexpected response type " << std::hex << recvHeader->msgType << " to module command.";
        return DCGM_ST_GENERIC_ERROR;
    }

    if (static_cast<size_t>(recvHeader->length) > maxResponseSize)
    {
        DCGM_LOG_ERROR << "Module command response size " << recvHeader->length << " was bigger than max allowed of "
                       << maxResponseSize;
        return DCGM_ST_GENERIC_ERROR;
    }

    auto msgBytes = response.response->GetMsgBytesPtr();
    memcpy(moduleCommand, msgBytes->data(), recvHeader->length);

    DCGM_LOG_DEBUG << "Got module command response of length " << recvHeader->length;

    /* If the request was persistent, it still exists in m_persistentReqs */
    return (dcgmReturn_t)recvHeader->status;
}

/*****************************************************************************/
dcgmReturn_t DcgmClientHandler::PopulateConnectionAttributes(dcgmHandle_t dcgmHandle)
{
    dcgm_core_msg_hostengine_version_t msg = {};

    dcgm_connection_id_t connectionId = (dcgm_connection_id_t)dcgmHandle;

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_HOSTENGINE_VERSION;
    msg.header.version    = dcgm_core_msg_hostengine_version_version;
    msg.version.version   = dcgmVersionInfo_version;

    // coverity[overrun-buffer-arg]
    dcgmReturn_t dcgmReturn = ExchangeModuleCommandAsync(connectionId, &msg.header, nullptr, sizeof(msg));
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "ConnectionId " << connectionId << " Got error " << dcgmReturn
                       << " from ExchangeModuleCommandAsync";
        return dcgmReturn;
    }

    /* Don't lock until here. If we had the lock going into ExchangeModuleCommandAsync(), we could
       never receive our message from the libevent processing thread */
    DcgmLockGuard dlg(&m_mutex);

    /* Returns pair of {iterator, wasInsertedBool} */
    auto [attributesIt, wasInserted]
        = m_connectionAttributes.emplace(connectionId, DCHConnectionAttributes(msg.version.rawBuildInfoString));

    DCGM_LOG_DEBUG << "connectionId " << connectionId << " returned version "
                   << attributesIt->second.m_buildInfo.GetVersion();

    return DCGM_ST_OK;
}

/*****************************************************************************/
