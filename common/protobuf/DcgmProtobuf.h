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
 * File:   DcgmProtobuf.h
 */

#ifndef DCGMPROTOBUF_H
#define DCGMPROTOBUF_H

#include "DcgmProtocol.h"
#include "dcgm.pb.h"
#include <iostream>
#include <vector>

class DcgmProtobuf
{
public:
    DcgmProtobuf();
    virtual ~DcgmProtobuf();

    /*****************************************************************************
     * Add command to the protobuf message to be sent over the network
     * Returns !NULL on Success
     *          NULL on Error
     *****************************************************************************/
    dcgm::Command *AddCommand(unsigned int cmdType, unsigned int opMode, int id, int status);

    /*****************************************************************************
     This method returns the encoded message to be sent over socket*
     *****************************************************************************/
    dcgmReturn_t GetEncodedMessage(std::vector<char> &encodedMessage);

    /*****************************************************************************
     * Parse the received protobuf message. This method gets reference to all the
     * commands in the message.
     *****************************************************************************/
    dcgmReturn_t ParseRecvdMessage(char *buf, int length, std::vector<dcgm::Command *> *pCommands);

    /*****************************************************************************
     * This method is used to get reference to all the commands in the protobuf message
     *****************************************************************************/
    int GetAllCommands(std::vector<dcgm::Command *> *pCommands);

protected:
    dcgm::Msg *mpProtoMsg;  /* Google protobuf format message */
    char *mpEncodedMessage; /* Encoded message is stored in this buffer */
    int mMsgType;           /* Represents one of Request, Response or Notify */
};

#endif /* DCGMPROTOBUF_H */
