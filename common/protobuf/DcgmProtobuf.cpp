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
 * File:   DcgmProtobuf.cpp
 */

#include "DcgmProtobuf.h"
#include "DcgmLogging.h"
#include "DcgmSettings.h"

/*****************************************************************************
 * Constructor for DCGM Protobuf
 *****************************************************************************/
DcgmProtobuf::DcgmProtobuf()
    : mMsgType(0)
{
    mpProtoMsg       = new dcgm::Msg;
    mpEncodedMessage = NULL;
}

/*****************************************************************************
 * Destructor for DCGM Protobuf
 *****************************************************************************/
DcgmProtobuf::~DcgmProtobuf()
{
    delete mpProtoMsg;
    mpProtoMsg = NULL;
    if (mpEncodedMessage)
    {
        delete[] mpEncodedMessage;
        mpEncodedMessage = NULL;
    }
}

/*****************************************************************************
 This method returns the encoded message to be sent over socket*
 *****************************************************************************/
dcgmReturn_t DcgmProtobuf::GetEncodedMessage(std::vector<char> &encodedMessage)
{
    encodedMessage.resize(mpProtoMsg->ByteSize());
    mpProtoMsg->SerializeToArray(encodedMessage.data(), encodedMessage.size());
    return DCGM_ST_OK;
}

/*****************************************************************************
 * Parse the received protobuf message.
 *****************************************************************************/
dcgmReturn_t DcgmProtobuf::ParseRecvdMessage(char *buf, int bufLength, std::vector<dcgm::Command *> *pCommands)
{
    unsigned int numCmds, j;


    if ((NULL == buf) || (bufLength <= 0))
    {
        return DCGM_ST_BADPARAM;
    }

    if (true != mpProtoMsg->ParseFromArray(buf, bufLength))
    {
        PRINT_ERROR("", "Failed to parse protobuf message");
        return DCGM_ST_BADPARAM;
    }

    numCmds = mpProtoMsg->cmd_size();
    if (numCmds <= 0)
    {
        PRINT_ERROR("", "Invalid number of commands in the protobuf message");
        return DCGM_ST_BADPARAM;
    }

    for (j = 0; j < numCmds; j++)
    {
        const dcgm::Command &cmdMsg = mpProtoMsg->cmd(j);
        pCommands->push_back((dcgm::Command *)&cmdMsg); /* Store reference to the command in protobuf message */
    }

    return DCGM_ST_OK;
}

int DcgmProtobuf::GetAllCommands(std::vector<dcgm::Command *> *pCommands)
{
    unsigned int numCmds, j;

    numCmds = mpProtoMsg->cmd_size();
    if (numCmds <= 0)
    {
        PRINT_ERROR("", "Invalid number of commands in the protobuf message");
        return -1;
    }

    for (j = 0; j < numCmds; j++)
    {
        const dcgm::Command &cmdMsg = mpProtoMsg->cmd(j);
        pCommands->push_back((dcgm::Command *)&cmdMsg); /* Store reference to the command in protobuf message */
    }

    return 0;
}

/*****************************************************************************
 * Add command to the protobuf message to be sent over the network
 *****************************************************************************/
dcgm::Command *DcgmProtobuf::AddCommand(unsigned int cmdType, unsigned int opMode, int id, int status)
{
    dcgm::Command *pCmd;

    pCmd = mpProtoMsg->add_cmd();
    if (NULL == pCmd)
    {
        return NULL;
    }

    pCmd->set_cmdtype((dcgm::CmdType)cmdType);
    pCmd->set_opmode((dcgm::CmdOperationMode)opMode);
    pCmd->set_id(id);
    pCmd->set_status(status);
    return pCmd;
}
