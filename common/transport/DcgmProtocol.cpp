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
#include "DcgmProtocol.h"
#include <arpa/inet.h>
#include <memory.h>

DcgmMessage::DcgmMessage()
    : m_messageHdr()
{}

DcgmMessage::DcgmMessage(dcgm_message_header_t *header)
    : m_messageHdr()
{
    memcpy(&m_messageHdr, header, sizeof(m_messageHdr));
}

DcgmMessage::~DcgmMessage() = default;

void DcgmMessage::UpdateMsgHdr(int msgType, dcgm_request_id_t requestId, int status, int length)
{
    m_messageHdr.msgId     = DCGM_PROTO_MAGIC;
    m_messageHdr.requestId = requestId;
    m_messageHdr.msgType   = msgType;
    m_messageHdr.status    = status;
    m_messageHdr.length    = length;
}

dcgm_message_header_t *DcgmMessage::GetMessageHdr()
{
    return &m_messageHdr;
}

std::vector<char> *DcgmMessage::GetMsgBytesPtr()
{
    return &m_msgBytes;
}

size_t DcgmMessage::GetLength()
{
    return m_msgBytes.size();
}

dcgm_request_id_t DcgmMessage::GetRequestId()
{
    return m_messageHdr.requestId;
}

void DcgmMessage::SetRequestId(dcgm_request_id_t requestId)
{
    m_messageHdr.requestId = requestId;
}
