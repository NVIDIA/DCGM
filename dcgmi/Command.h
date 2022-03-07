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
 * File:   Command.h
 */

#ifndef COMMAND_H
#define COMMAND_H

#include "dcgm_structs.h"

#include <string>


class Command
{
public:
    Command();

    virtual ~Command();

    /*****************************************************************************
     * Connect to the host name
     *****************************************************************************/
    dcgmReturn_t Connect();

    /*****************************************************************************
     * persistAfterDisconnect: Should the host engine persist the watches created
     *                         by this connection after the connection goes away?
     *                         1=yes. 0=no (default).
     *****************************************************************************/
    void SetPersistAfterDisconnect(unsigned int persistAfterDisconnect);

    /*****************************************************************************
     * Execute command on the Host Engine
     *****************************************************************************/
    dcgmReturn_t Execute();

protected:
    /**
     * Virtual function and should be implemented by the derived class.
     * The Execute() calls this after a successful connection
     * @return DCGM_ST_* status of the underlying implementation
     */
    virtual dcgmReturn_t DoExecuteConnected() = 0;

    virtual dcgmReturn_t DoExecuteConnectionFailure(dcgmReturn_t connectionStatus);

    std::string m_hostName;
    dcgmHandle_t m_dcgmHandle {};
    unsigned int m_timeout {};
    unsigned int m_persistAfterDisconnect {}; /*!< Should the host engine persist the watches created
                                                   by this connection after the connection goes away?
                                                   1=yes. 0=no (default). */
    bool m_json {};
    bool m_silent {};
};


#endif /* COMMAND_H */
