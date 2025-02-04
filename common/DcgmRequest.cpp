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
{
    log_debug("DcgmRequest {}, requestId {} created", fmt::ptr(this), m_requestId.load());
}

/*****************************************************************************/
dcgm_request_id_t DcgmRequest::GetRequestId(void)
{
    return m_requestId.load();
}

/*****************************************************************************/
void DcgmRequest::SetRequestId(dcgm_request_id_t requestId)
{
    Lock();
    m_requestId = requestId;
    Unlock();
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
