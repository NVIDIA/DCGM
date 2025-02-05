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
#ifndef DCGMREQUEST_H
#define DCGMREQUEST_H

#include "DcgmProtocol.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

/*****************************************************************************/
class DcgmRequest
{
public:
    DcgmRequest(dcgm_request_id_t requestId);
    virtual ~DcgmRequest();

    /*************************************************************************/
    /* Accessors for m_requestId */
    void SetRequestId(dcgm_request_id_t requestId);
    dcgm_request_id_t GetRequestId();

    /*************************************************************************/
    /* Pure virtual for processing an incoming message that any child class
     * needs to implement to processs incoming messages for this request
     *
     * Returns DCGM_ST_SUCCESS on success.
     *         Other DCGM_ST_? status code on error
     */
    virtual int ProcessMessage(std::unique_ptr<DcgmMessage> msg) = 0;

    /*************************************************************************/
protected:
    int m_status;                               /* Status of this request. A DCGM_ST_? constant */
    std::atomic<dcgm_request_id_t> m_requestId; /* Request identifier of this request. Should be unique
                                                   across all requests of this connection */

    std::vector<std::unique_ptr<DcgmMessage>> m_messages; /* Vector of responses in the order they
                                                             were received we will expect one for
                                                             now, but there could be more in the
                                                             future */

    std::mutex m_mutex; /* We need a lock around this because we may be
                           updating this out of band after the initial
                           ack of our data has completed */

    /*************************************************************************/
    /* Used for protecting the internal state of this class */
    void Lock();
    void Unlock();

    /*************************************************************************/
};

/*****************************************************************************/

#endif // DCGMREQUEST_H
