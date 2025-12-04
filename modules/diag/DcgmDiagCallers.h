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
#pragma once

#include <dcgm_structs_internal.h>
#include <mutex>
#include <unordered_map>

class DcgmDiagCallers
{
public:
    struct ConnectionInfo
    {
        // The caller has died
        bool alreadyStopped = false;
        // The heartbeat is enabled
        bool heartbeatEnabled = false;
        // The last heartbeat time
        std::chrono::steady_clock::time_point lastHeartbeatTime = std::chrono::steady_clock::now();
    };

    DcgmDiagCallers()                                   = default;
    DcgmDiagCallers(DcgmDiagCallers const &)            = delete;
    DcgmDiagCallers(DcgmDiagCallers &&)                 = delete;
    DcgmDiagCallers &operator=(DcgmDiagCallers const &) = delete;
    DcgmDiagCallers &operator=(DcgmDiagCallers &&)      = delete;

    /*
     * Set the connection id of the caller
     * @param connectionId
     */
    void SetConnectionId(dcgm_connection_id_t connectionId);

    /**
     * Reset the connection id of the caller
     * @param connectionId
     */
    void ResetConnectionId(dcgm_connection_id_t connectionId);

    /**
     * Set the already stopped flag
     */
    void SetAlreadyStopped(dcgm_connection_id_t connectionId);

    /**
     * Check if the caller has died
     * @return true if the caller has died
     */
    bool IsAlreadyStopped(dcgm_connection_id_t connectionId) const;

    /**
     * Check if the caller exists
     * @param connectionId
     * @return true if the caller exists
     */
    bool Exists(dcgm_connection_id_t connectionId) const;

    /**
     * Set the heartbeat enabled flag
     * @param connectionId
     * @param enabled
     */
    void SetHeartbeatEnabled(dcgm_connection_id_t connectionId, bool enabled);

    /**
     * Get the heartbeat enabled flag
     * @param connectionId
     * @return true if the heartbeat is enabled
     */
    bool IsHeartbeatEnabled(dcgm_connection_id_t connectionId) const;

    /**
     * Get the last heartbeat time
     * @param connectionId
     * @return the last heartbeat time
     */
    std::chrono::steady_clock::time_point GetLastHeartbeatTime(dcgm_connection_id_t connectionId) const;

    /**
     * Set the last heartbeat time
     * @param connectionId
     */
    void ReceiveHeartbeat(dcgm_connection_id_t connectionId);

private:
    // mutex to protect this class
    mutable std::mutex m_mutex;
    // connection of the callers
    std::unordered_map<dcgm_connection_id_t, ConnectionInfo> m_connectionInfo;
};
