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
 * File:   DcgmProtocol.h
 */

#ifndef DCGM_PROTOCOL_H
#define DCGM_PROTOCOL_H

#include "dcgm_structs.h"
#include <vector>

/* Align to byte boundaries */
#pragma pack(1)

#define DCGM_PROTO_MAGIC \
    0xadbcbcad /* Used to fill up the msgID for dcgm message header. Formerly 0xabbcbcab on DCGM 1.7 and older */

#define DCGM_PROTO_MAX_MESSAGE_SIZE (4 * 1024 * 1024) /* Maximum size of a single DCGM socket message (4 MB) */

/**
 * ID of a request from the client to the host engine. This is unique for a given connection
 * but not unique across connections
 */
typedef unsigned int dcgm_request_id_t;

/* Special constant for an unset request ID */
#define DCGM_REQUEST_ID_NONE ((dcgm_request_id_t)0)

/**
 * DCGM Message Header. Note that this is in host byte order (little endian)
 */
typedef struct
{
    int msgId;                   /* Identifier to represent DCGM protocol (DCGM_PROTO_MAGIC) */
    dcgm_request_id_t requestId; /* Represent Message by a request ID */
    int length;                  /* Length of message body (not including this header). The full size of this
                                    message on the wire is sizeof(*header) + header->length */
    int msgType;                 /* Type of message. One of DCGM_MSG_? */
    int status;                  /* Status. One of DCGM_ST_? #defines from dcgm_structs.h */
} dcgm_message_header_t;

#pragma pack() /* Undo the 1-byte alignment */

/**
 * The following defines are used to recognize type of DCGM messages
 */
#define DCGM_MSG_PROTO_REQUEST  0x0100 /* A Google protobuf-based request */
#define DCGM_MSG_PROTO_RESPONSE 0x0200 /* A Google protobuf-based response to a request */
#define DCGM_MSG_MODULE_COMMAND 0x0300 /* A module command message */
#define DCGM_MSG_POLICY_NOTIFY  0x0400 /* Async notification of a policy violation */
#define DCGM_MSG_REQUEST_NOTIFY 0x0500 /* Notify an async request that it will receive no further updates */

/* DCGM_MSG_POLICY_NOTIFY - Signal a client that a policy has been violated */
typedef struct
{
    int begin;                             /* Whether this is the first response (1) or the second (0).
                                     This will determine if beginCB or finishCB is called. */
    dcgmPolicyCallbackResponse_t response; /* Policy response to pass to client callbacks */
} dcgm_msg_policy_notify_t;

/* DCGM_MSG_REQUEST_NOTIFY - Notify an async request that it will receive
 *                           no further updates
 **/
typedef struct
{
    unsigned int requestId; /* This is redundant with the header requestId, but we need
                               message contents */
} dcgm_msg_request_notify_t;

class DcgmMessage
{
public:
    DcgmMessage();
    DcgmMessage(dcgm_message_header_t *header); /* Set header from a network packet */
    ~DcgmMessage();

    /**
     * Copy constructors. Deleted for now to force moving
     */
    DcgmMessage(const DcgmMessage &other) = delete;
    DcgmMessage &operator=(DcgmMessage other) = delete;

    /**
     * Move Constructors
     */
    DcgmMessage(DcgmMessage &&other) = default;
    DcgmMessage &operator=(DcgmMessage &&other) = default;

    /**
     * This method updates the message header to be sent over socket. mMessageHdr will be
     * the passed in parameters after this call.
     */
    void UpdateMsgHdr(int msgType, dcgm_request_id_t requestId, int status, int length);

    /**
     * This method returns reference to Dcgm Message Header
     */
    dcgm_message_header_t *GetMessageHdr();

    /**
     * This method returns the underlying vector that represents this message's content
     * Note that this is the data past the m_messageHdr on the wire.
     */
    std::vector<char> *GetMsgBytesPtr();

    /**
     * This message is used to get the length of the message
     */
    size_t GetLength();

    /**
     * This message is used to get the msgType
     */
    unsigned int GetMsgType()
    {
        return m_messageHdr.msgType;
    };

    /**
     * This message is used to set request id corresponding to the message
     * @return
     */
    void SetRequestId(dcgm_request_id_t requestId);

    /**
     * This message is used to get request id corresponding to the message
     */
    unsigned int GetRequestId();

private:
    dcgm_message_header_t m_messageHdr {}; /*!< Sender populates the message to be sent */
    std::vector<char> m_msgBytes;          /*!< The bytes of the message that come after m_messageHdr on the
                                               socket stream. The .size() member of this is the size of
                                               the message. This should match m_messageHdr.length */
};

#endif /* DCGM_PROTOCOL_H */
