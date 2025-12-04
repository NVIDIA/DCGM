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

#include "DcgmDiagCallers.h"

#include <DcgmLogging.h>

void DcgmDiagCallers::SetConnectionId(dcgm_connection_id_t connectionId)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto [_, inserted] = m_connectionInfo.try_emplace(connectionId, ConnectionInfo {});
        if (inserted)
        {
            return;
        }
    }
    log_debug("Connection ID {} already set, ignoring.", connectionId);
}

void DcgmDiagCallers::ResetConnectionId(dcgm_connection_id_t connectionId)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_connectionInfo.erase(connectionId);
}

void DcgmDiagCallers::SetAlreadyStopped(dcgm_connection_id_t connectionId)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_connectionInfo.find(connectionId);
        if (it != m_connectionInfo.end())
        {
            it->second.alreadyStopped = true;
            return;
        }
    }
    log_debug("Connection ID {} not found, ignoring.", connectionId);
}

bool DcgmDiagCallers::IsAlreadyStopped(dcgm_connection_id_t connectionId) const
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_connectionInfo.find(connectionId);
        if (it != m_connectionInfo.end())
        {
            return it->second.alreadyStopped;
        }
    }
    log_debug("Connection ID {} not found, assuming not stopped.", connectionId);
    return false;
}

bool DcgmDiagCallers::Exists(dcgm_connection_id_t connectionId) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_connectionInfo.contains(connectionId);
}

void DcgmDiagCallers::SetHeartbeatEnabled(dcgm_connection_id_t connectionId, bool enabled)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_connectionInfo.find(connectionId);
        if (it != m_connectionInfo.end())
        {
            it->second.heartbeatEnabled  = enabled;
            it->second.lastHeartbeatTime = std::chrono::steady_clock::now();
            return;
        }
    }
    log_debug("Connection ID {} not found, ignoring.", connectionId);
}

bool DcgmDiagCallers::IsHeartbeatEnabled(dcgm_connection_id_t connectionId) const
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_connectionInfo.find(connectionId);
        if (it != m_connectionInfo.end())
        {
            return it->second.heartbeatEnabled;
        }
    }
    log_debug("Connection ID {} not found, assuming not enabled.", connectionId);
    return false;
}

std::chrono::steady_clock::time_point DcgmDiagCallers::GetLastHeartbeatTime(dcgm_connection_id_t connectionId) const
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_connectionInfo.find(connectionId);
        if (it != m_connectionInfo.end())
        {
            return it->second.lastHeartbeatTime;
        }
    }
    log_debug("Connection ID {} not found, assuming now.", connectionId);
    return std::chrono::steady_clock::now();
}

void DcgmDiagCallers::ReceiveHeartbeat(dcgm_connection_id_t connectionId)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_connectionInfo.find(connectionId);
        if (it != m_connectionInfo.end())
        {
            it->second.lastHeartbeatTime = std::chrono::steady_clock::now();
            return;
        }
    }
    log_debug("Connection ID {} not found, ignoring.", connectionId);
}
