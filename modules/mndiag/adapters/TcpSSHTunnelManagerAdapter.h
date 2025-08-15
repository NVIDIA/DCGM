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

#include "TcpSSHTunnelManagerBase.h"
#include <SSHTunnelManager.hpp>

/**
 * @brief Adapter for TcpSSHTunnelManager to use in production code
 */
class TcpSSHTunnelManagerAdapter : public TcpSSHTunnelManagerBase
{
public:
    TcpSSHTunnelManagerAdapter()
        : m_tunnelManager(DcgmNs::Common::RemoteConn::TcpSSHTunnelManager::GetInstance())
    {}
    ~TcpSSHTunnelManagerAdapter() override = default;

    /**
     * @brief Starts an SSH tunnel session using the real TcpSSHTunnelManager
     */
    DcgmNs::Common::RemoteConn::detail::TunnelState StartSession(std::string_view remoteHostname,
                                                                 uint16_t const &remoteAddress,
                                                                 uint16_t &localAddress,
                                                                 std::optional<uid_t> uid) override
    {
        return m_tunnelManager.StartSession(remoteHostname, remoteAddress, localAddress, uid);
    }

    /**
     * @brief Ends an SSH tunnel session using the real TcpSSHTunnelManager
     */
    void EndSession(std::string_view remoteHostname,
                    uint16_t const &remoteAddress,
                    std::optional<uid_t> uid,
                    std::optional<bool> forceEnd) override
    {
        m_tunnelManager.EndSession(remoteHostname, remoteAddress, uid, forceEnd);
    }

    bool SetChildProcessFuncs(DcgmNs::Common::RemoteConn::detail::ChildProcessFuncs const *childProcessFuncs) override
    {
        return m_tunnelManager.SetChildProcessFuncs(childProcessFuncs);
    }

private:
    // Reference to the singleton instance
    DcgmNs::Common::RemoteConn::TcpSSHTunnelManager &m_tunnelManager;
};
