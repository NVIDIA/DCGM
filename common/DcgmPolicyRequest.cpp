/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
/*
 * File:   DcgmPolicyRequest.cpp
 */

#include "DcgmPolicyRequest.h"
#include "DcgmProtocol.h"
#include "DcgmSettings.h"
#include "dcgm_structs.h"

/*****************************************************************************/
DcgmPolicyRequest::DcgmPolicyRequest(fpRecvUpdates callback, uint64_t userData)
    : DcgmRequest(0)
{
    mIsAckRecvd = false;
    mCallback   = callback;
    mUserData   = userData;
}

/*****************************************************************************/
DcgmPolicyRequest::~DcgmPolicyRequest()
{}

/*****************************************************************************/
int DcgmPolicyRequest::ProcessMessage(std::unique_ptr<DcgmMessage> msg)
{
    if (!msg)
        return DCGM_ST_BADPARAM;

    std::unique_lock lock(m_mutex);
    /* The first response is the policy manager confirming that we've watched
       it. Further responses will be policy violations */
    dcgm_message_header_t *header = msg->GetMessageHdr();
    switch (header->msgType)
    {
        case DCGM_MSG_PROTO_REQUEST:
        case DCGM_MSG_PROTO_RESPONSE:
        case DCGM_MSG_MODULE_COMMAND:
            /* Request/response messages complete the initial request */
            if (!mIsAckRecvd)
            {
                m_status    = DCGM_ST_OK;
                mIsAckRecvd = true;
                m_messages.push_back(std::move(msg));
            }
            else
            {
                log_error("Ignoring unexpected duplicate ACK");
            }
            return DCGM_ST_OK;

        case DCGM_MSG_POLICY_NOTIFY:
            break; /* Code handled below */

        default:
            lock.unlock();
            log_error("Unexpected msgType {} received.", header->msgType);
            return DCGM_ST_OK; /* Returning an error here doesn't affect anything we want to affect */
    }

    /* We should only be here if we got a policy notification */
    auto msgBytes = msg->GetMsgBytesPtr();

    /* Buffer needs to be at least large enough to hold the policy notify struct */
    if (msgBytes->size() < sizeof(dcgm_msg_policy_notify_t))
    {
        lock.unlock();
        log_error("Message buffer too small: {} < {}", msgBytes->size(), sizeof(dcgm_msg_policy_notify_t));
        return DCGM_ST_BADPARAM;
    }

    dcgm_msg_policy_notify_t *policy = (dcgm_msg_policy_notify_t *)msgBytes->data();

    /* Make local copies of the callback so we can safely unlock. I don't want this code to deadlock if someone
       removes this object from one of the callbacks */
    fpRecvUpdates callback = mCallback;

    lock.unlock();

    /* Call the callback if it is present */
    if (callback)
    {
        try
        {
            callback(&policy->response, mUserData);
        }
        catch (std::exception const &e)
        {
            log_error("Callback thrown exception: {}.", e.what());
        }
        catch (...)
        {
            log_error("Callback thrown unknown exception.");
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
DcgmPolicyViolationRequest::DcgmPolicyViolationRequest(fpRecvPolicyViolation callback, uint64_t userData)
    : DcgmRequest(0)
{
    mIsAckRecvd = false;
    mCallback   = callback;
    mUserData   = userData;
}

/*****************************************************************************/
DcgmPolicyViolationRequest::~DcgmPolicyViolationRequest()
{}

/*****************************************************************************/
int DcgmPolicyViolationRequest::ProcessMessage(std::unique_ptr<DcgmMessage> msg)
{
    if (!msg)
    {
        return DCGM_ST_BADPARAM;
    }

    std::unique_lock lock(m_mutex);
    dcgm_message_header_t *header = msg->GetMessageHdr();
    switch (header->msgType)
    {
        case DCGM_MSG_PROTO_REQUEST:
        case DCGM_MSG_PROTO_RESPONSE:
        case DCGM_MSG_MODULE_COMMAND:
            if (!mIsAckRecvd)
            {
                m_status    = DCGM_ST_OK;
                mIsAckRecvd = true;
                m_messages.push_back(std::move(msg));
            }
            else
            {
                lock.unlock();
                log_error("Ignoring unexpected duplicate field policy callback ACK");
            }
            return DCGM_ST_OK;

        case DCGM_MSG_POLICY_VIOLATION:
            break;

        default:
            auto const msgType = header->msgType;
            lock.unlock();
            log_error("Unexpected msgType {} received for field policy callback.", msgType);
            return DCGM_ST_OK;
    }

    auto msgBytes = msg->GetMsgBytesPtr();
    if (msgBytes->size() < sizeof(dcgm_msg_policy_violation_t))
    {
        lock.unlock();
        log_error("Message buffer too small: {} < {}", msgBytes->size(), sizeof(dcgm_msg_policy_violation_t));
        return DCGM_ST_BADPARAM;
    }

    auto *policyViolation          = reinterpret_cast<dcgm_msg_policy_violation_t *>(msgBytes->data());
    fpRecvPolicyViolation callback = mCallback;

    lock.unlock();

    if (callback)
    {
        try
        {
            callback(&policyViolation->violation, mUserData);
        }
        catch (std::exception const &e)
        {
            log_error("Field policy callback thrown exception: {}.", e.what());
        }
        catch (...)
        {
            log_error("Field policy callback thrown unknown exception.");
        }
    }

    return DCGM_ST_OK;
}
