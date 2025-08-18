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

#include <SSHTunnelManager.hpp>
#include <string_view>

/**
 * @brief Base interface for TcpSSHTunnelManager to allow for mocking in tests
 */
class TcpSSHTunnelManagerBase
{
public:
    virtual ~TcpSSHTunnelManagerBase() = default;

    /**
     * @brief Starts an SSH tunnel session
     *
     * @param remoteHostname Remote hostname
     * @param remoteAddress Remote address to forward
     * @param localAddress Local address assigned for the tunnel
     * @return TunnelState Status of the tunnel
     */
    virtual DcgmNs::Common::RemoteConn::detail::TunnelState StartSession(std::string_view remoteHostname,
                                                                         uint16_t const &remoteAddress,
                                                                         uint16_t &localAddress,
                                                                         std::optional<uid_t> uid = std::nullopt)
        = 0;

    /**
     * @brief Ends an SSH tunnel session
     *
     * @param remoteHostname Remote hostname
     * @param remoteAddress Remote address that was forwarded
     */
    virtual void EndSession(std::string_view remoteHostname,
                            uint16_t const &remoteAddress,
                            std::optional<uid_t> uid     = std::nullopt,
                            std::optional<bool> forceEnd = std::nullopt)
        = 0;

    virtual bool SetChildProcessFuncs(DcgmNs::Common::RemoteConn::detail::ChildProcessFuncs const *childProcessFuncs)
        = 0;
};
