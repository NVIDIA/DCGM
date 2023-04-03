/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmRequest.h"
#include "DcgmLogging.h"
#include "dcgm_structs.h"
#include "timelib.h"

/*****************************************************************************/
DcgmRequest::DcgmRequest(dcgm_request_id_t requestId)
    : m_status(DCGM_ST_PENDING)
    , m_requestId(requestId)
    , m_messages()
    , m_mutex()
    , m_condition()
{
    log_debug("DcgmRequest {}, requestId {} created", (void *)this, m_requestId);
}

/*****************************************************************************/
dcgm_request_id_t DcgmRequest::GetRequestId(void)
{
    return m_requestId;
}

/*****************************************************************************/
void DcgmRequest::SetRequestId(dcgm_request_id_t requestId)
{
    m_requestId = requestId;
}

/*****************************************************************************/
DcgmRequest::~DcgmRequest()
{
    Lock();

    m_messages.clear();

    m_status = DCGM_ST_UNINITIALIZED;
    Unlock();

    log_debug("DcgmRequest {} destructed", (void *)this);
}

/*****************************************************************************/
void DcgmRequest::Lock()
{
    m_mutex.lock();
}

/*****************************************************************************/
void DcgmRequest::Unlock()
{
    m_mutex.unlock();
}

/*****************************************************************************/
int DcgmRequest::Wait(int timeoutMs)
{
    int retSt = DCGM_ST_OK;
    std::chrono::milliseconds timeout(timeoutMs); /* 1 second timeout */

    if (m_status != DCGM_ST_PENDING)
    {
        log_debug("DcgmRequest {} already in state {}", (void *)this, m_status);
        return DCGM_ST_OK; /* Already out of pending state */
    }

    /* Technically, we could sleep n x timeoutMs times here if we get n spurious
     * wake-ups. We can deal with this corner case later
     */

    std::unique_lock<std::mutex> unique_lock(m_mutex);
    while (m_status == DCGM_ST_PENDING)
    {
        bool stat = m_condition.wait_for(
            unique_lock, timeout, [this] { return m_status != DCGM_ST_PENDING || !m_messages.empty(); });
        if (!stat)
        {
            retSt = DCGM_ST_TIMEOUT;
            break;
        }

        /* Were we signalled with another status like DCGM_ST_CONNECTION_NOT_VALID? */
        if (m_status != DCGM_ST_PENDING)
        {
            retSt = m_status;
            break;
        }
    }

    log_debug("DcgmRequest {} wait complete. m_status {}, retSt {}", (void *)this, m_status, retSt);

    return retSt;
}

/*****************************************************************************/
int DcgmRequest::MessageCount()
{
    int count;

    /* Don't take any chances */
    Lock();
    count = (int)m_messages.size();
    Unlock();
    return count;
}

/*****************************************************************************/
int DcgmRequest::ProcessMessage(std::unique_ptr<DcgmMessage> msg)
{
    if (!msg)
        return DCGM_ST_BADPARAM;

    log_debug("DcgmRequest::ProcessMessage msg {} DcgmRequest {}", (void *)msg.get(), (void *)this);

    Lock();
    m_status = DCGM_ST_OK;
    m_messages.push_back(std::move(msg));
    Unlock();
    m_condition.notify_all();

    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmRequest::SetStatus(int status)
{
    log_debug("DcgmRequest::SetStatus DcgmRequest {}, status {}", (void *)this, status);

    Lock();
    m_status = status;
    Unlock();
    m_condition.notify_all();
    return DCGM_ST_OK;
}

/*****************************************************************************/
std::unique_ptr<DcgmMessage> DcgmRequest::GetNextMessage()
{
    std::unique_ptr<DcgmMessage> retVal;

    Lock();

    if (m_messages.empty())
    {
        Unlock();
        DCGM_LOG_DEBUG << "found no messages" << (void *)this;
        return nullptr;
    }

    /* Return the first message */
    retVal = std::move(m_messages[0]);
    m_messages.erase(m_messages.begin());
    Unlock();

    return retVal;
}

/*****************************************************************************/
