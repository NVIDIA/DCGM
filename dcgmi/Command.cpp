/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * File:   Command.cpp
 */

#include "Command.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgmi_common.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>

/*****************************************************************************/
Command::Command()
    : mHostName()
    , mNvcmHandle(0)
    , mJson(false)
    , mSilent(false)
    , mTimeout(0)
    , mPersistAfterDisconnect(0)
{}

/*****************************************************************************/
Command::~Command()
{
    dcgmReturn_t result;

    if (mNvcmHandle)
    {
        // Disconnect
        result = dcgmDisconnect(mNvcmHandle);
        if (DCGM_ST_OK != result)
        {
            std::cout << "Error: unable to close connection to specified host: " << mHostName << std::endl;
        }
        mNvcmHandle = 0;
    }
}

/*****************************************************************************/
dcgmReturn_t Command::Connect(void)
{
    dcgmReturn_t result;
    dcgmConnectV2Params_t connectParams;
    const char *hostNameStr  = mHostName.c_str();
    bool isUnixSocketAddress = false;

    /* For now, do a global init of DCGM on the start of a command. We can change this later to
     * only connect to the remote host engine from within the command object
     */

    result = dcgmInit();
    if (DCGM_ST_OK != result)
    {
        if (mSilent == false)
            std::cout << "Error: unable to initialize DCGM" << std::endl;
        return result;
    }

    hostNameStr = dcgmi_parse_hostname_string(hostNameStr, &isUnixSocketAddress, !mSilent);
    if (!hostNameStr)
        return DCGM_ST_BADPARAM; /* Don't need to print here. The function above already did */

    memset(&connectParams, 0, sizeof(connectParams));
    connectParams.version                = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = mPersistAfterDisconnect;
    connectParams.addressIsUnixSocket    = isUnixSocketAddress ? 1 : 0;
    connectParams.timeoutMs              = mTimeout;

    result = dcgmConnect_v2((char *)hostNameStr, &connectParams, &mNvcmHandle);
    if (DCGM_ST_OK != result)
    {
        if (mSilent == false)
            std::cout << "Error: unable to establish a connection to the specified host: " << mHostName << std::endl;
        return result;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t Command::Execute()
{
    dcgmReturn_t result;

    result = Connect();
    if (DCGM_ST_OK != result)
    {
        return DoExecuteConnectionFailure(DCGM_ST_CONNECTION_NOT_VALID);
    }

    return DoExecuteConnected();
}

/*****************************************************************************/
void Command::SetPersistAfterDisconnect(unsigned int persistAfterDisconnect)
{
    mPersistAfterDisconnect = persistAfterDisconnect;
}
dcgmReturn_t Command::DoExecuteConnectionFailure(dcgmReturn_t connectionStatus)
{
    if (!mSilent)
    {
        std::cout << "Error: Unable to connect to host engine. " << errorString(connectionStatus) << "." << std::endl;
    }
    return connectionStatus;
}

/*****************************************************************************/
