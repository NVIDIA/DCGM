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
/*
 * File:   DcgmPolicyRequest.cpp
 */

#include "DcgmPolicyRequest.h"
#include "DcgmProtocol.h"
#include "DcgmSettings.h"
#include "dcgm_structs.h"

/*****************************************************************************/
DcgmPolicyRequest::DcgmPolicyRequest(fpRecvUpdates beginCB, fpRecvUpdates finishCB)
    : DcgmRequest(0)
{
    mIsAckRecvd = false;
    mBeginCB    = beginCB;
    mFinishCB   = finishCB;
}

/*****************************************************************************/
DcgmPolicyRequest::~DcgmPolicyRequest()
{}

/*****************************************************************************/
int DcgmPolicyRequest::ProcessMessage(std::unique_ptr<DcgmMessage> msg)
{
    if (!msg)
        return DCGM_ST_BADPARAM;

    Lock();
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
                m_condition.notify_all(); /* The waiting thread will wake up and read the messages */
            }
            else
            {
                PRINT_ERROR("", "Ignoring unexpected duplicate ACK");
            }
            Unlock();
            return DCGM_ST_OK;

        case DCGM_MSG_POLICY_NOTIFY:
            /* This #if is here because we can either ignore updates before the initial request
               is ACKed or we can process them right away. The default behavior is to process
               callbacks right away, meaning that if there is a policy violation, we are likely
               to call our callbacks from within dcgmPolicyRegister because it sets up the watches,
               calls UpdateAllFields(), and gets instant notifications of FV updates. Leaving the
               alternative here in case customers complain too much :) */
#if 1
            break; /* Code handled below */
#else              /* Notify only after ACK */
            if (mIsAckRecvd)
                break; /* Code handled below */
            else
            {
                PRINT_DEBUG("%p", "Request %p ignoring notification before the initial request was ACKed", this);
            }
            Unlock();
#endif
            return DCGM_ST_OK;

        default:
            PRINT_ERROR("%u", "Unexpected msgType %u received.", header->msgType);
            Unlock();
            return DCGM_ST_OK; /* Returning an error here doesn't affect anything we want to affect */
    }

    /* We should only be here if we got a policy notification */
    auto msgBytes                    = msg->GetMsgBytesPtr();
    dcgm_msg_policy_notify_t *policy = (dcgm_msg_policy_notify_t *)msgBytes->data();

    /* Make local copies of the callbacks so we can safely unlock. I don't want this code to deadlock if someone
       removes this object from one of the callbacks */
    fpRecvUpdates beginCB  = mBeginCB;
    fpRecvUpdates finishCB = mFinishCB;

    Unlock();

    /* Call the callbacks if they're present */
    if (policy->begin && beginCB)
        beginCB(&policy->response);
    if (!policy->begin && finishCB)
        finishCB(&policy->response);

    return DCGM_ST_OK;
}
