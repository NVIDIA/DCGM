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
#include <cstdlib>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>

#include "DcgmHandle.h"
#include "NvvsCommon.h" //logging functions
#include "dcgm_test_apis.h"

DcgmHandle::DcgmHandle()
    : m_lastReturn(DCGM_ST_OK)
    , m_handle(0)
    , m_ownHandle(true)
{}

DcgmHandle::DcgmHandle(DcgmHandle &&other) noexcept
    : m_lastReturn(other.m_lastReturn)
    , m_handle(other.m_handle)
    , m_ownHandle(other.m_ownHandle)
{
    if (m_ownHandle)
    {
        other.m_handle    = 0;
        other.m_ownHandle = false;
    }
}

DcgmHandle &DcgmHandle::operator=(dcgmHandle_t handle)
{
    Cleanup();
    m_handle    = handle;
    m_ownHandle = false;
    return *this;
}

DcgmHandle::~DcgmHandle()
{
    Cleanup();
}

void DcgmHandle::Cleanup()
{
    // If someone else owns the handle, let them manage it
    if (m_ownHandle && m_handle)
    {
        if (m_handle == (dcgmHandle_t)DCGM_EMBEDDED_HANDLE)
            dcgmStopEmbedded(m_handle);
        else
            dcgmDisconnect(m_handle);

        /* Don't call dcgmShutdown here as it will nuke all other connections */
    }
}

std::string DcgmHandle::RetToString(dcgmReturn_t ret)
{
    std::stringstream err;

    if (ret != DCGM_ST_OK)
    {
        const char *tmp = errorString(ret);

        if (tmp == NULL)
            err << "Unknown error from DCGM: " << ret;
        else
            err << tmp;
    }

    return err.str();
}

std::string DcgmHandle::GetLastError()
{
    return RetToString(m_lastReturn);
}

dcgmHandle_t DcgmHandle::GetHandle()
{
    return m_handle;
}

dcgmReturn_t DcgmHandle::ConnectToDcgm(const std::string &dcgmHostname)
{
    char hostname[128];
    dcgmConnectV2Params_t params;
    memset(&params, 0, sizeof(params));
    params.version   = dcgmConnectV2Params_version2;
    params.timeoutMs = 1000; // 1 second should be plenty of time to find the host engine

    if (m_ownHandle == false)
    {
        DCGM_LOG_DEBUG << "Warning: starting new connection despite being initialized with a pre-existing connection.";
    }

    if (dcgmHostname.size() == 0)
        sprintf(hostname, "127.0.0.1");
    else
    {
        if (!strncmp(dcgmHostname.c_str(), "unix://", 7))
        {
            params.addressIsUnixSocket = 1;
            snprintf(hostname, sizeof(hostname), "%s", dcgmHostname.c_str() + 7);
        }
        else
            snprintf(hostname, sizeof(hostname), "%s", dcgmHostname.c_str());
    }

    m_lastReturn = dcgmInit();
    if (m_lastReturn == DCGM_ST_OK)
        m_lastReturn = dcgmConnect_v2(hostname, &params, &m_handle);

    /* If we got connected or another error, return */
    if (m_lastReturn != DCGM_ST_CONNECTION_NOT_VALID)
    {
        return m_lastReturn;
    }

    /* Try to see if an embedded host engine is already running. We have to do this since
       there could be multiple instances of this class, and we don't want them all to start-stop
       the embedded host engine. Only the first instance of this class should do that */
    int count                                    = 0;
    unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES] = { 0 };
    m_lastReturn = dcgmGetAllDevices((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, gpuIdList, &count);
    /* If the API succeeded (already running) or we got another error, return */
    if (m_lastReturn != DCGM_ST_UNINITIALIZED)
    {
        m_handle = (dcgmHandle_t)DCGM_EMBEDDED_HANDLE;
        PRINT_DEBUG("%d", "Skipping starting of embedded host engine due to lastReturn %d.", m_lastReturn);
        return m_lastReturn;
    }

    PRINT_DEBUG("", "Starting embedded host engine.");
    m_lastReturn = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &m_handle);
    return m_lastReturn;
}
