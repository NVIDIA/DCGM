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

#include <TcpSSHTunnelManagerBase.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Mock implementation of TcpSSHTunnelManagerBase for testing
 */
class MockTcpSSHTunnelManager : public TcpSSHTunnelManagerBase
{
public:
    using StartSessionCallback
        = std::function<DcgmNs::Common::RemoteConn::detail::
                            TunnelState(std::string_view, uint16_t const &, uint16_t &, std::optional<uid_t>)>;
    using EndSessionCallback
        = std::function<void(std::string_view, uint16_t const &, std::optional<uid_t>, std::optional<bool>)>;

    MockTcpSSHTunnelManager()                                           = default;
    ~MockTcpSSHTunnelManager() override                                 = default;
    MockTcpSSHTunnelManager(const MockTcpSSHTunnelManager &)            = delete;
    MockTcpSSHTunnelManager &operator=(const MockTcpSSHTunnelManager &) = delete;
    MockTcpSSHTunnelManager(MockTcpSSHTunnelManager &&)                 = delete;
    MockTcpSSHTunnelManager &operator=(MockTcpSSHTunnelManager &&)      = delete;

    /**
     * @brief StartSession mock - returns the configured result or uses callback
     */
    DcgmNs::Common::RemoteConn::detail::TunnelState StartSession(std::string_view remoteHostname,
                                                                 uint16_t const &remotePort,
                                                                 uint16_t &localPort,
                                                                 std::optional<uid_t> uid = std::nullopt) override
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Record the call
            m_startSessionCalls.emplace_back(remoteHostname, remotePort, uid);
        }
        m_startSessionCallCount++;

        // Use callback if provided
        if (m_startSessionCallback)
        {
            return m_startSessionCallback(remoteHostname, remotePort, localPort, uid);
        }

        // Otherwise use default behavior
        if (m_startSessionResult == DcgmNs::Common::RemoteConn::detail::TunnelState::Active)
        {
            // Generate a fake local port
            localPort = m_defaultRemotePort;
        }

        return m_startSessionResult;
    }

    /**
     * @brief EndSession mock - calls the configured callback if provided
     */
    void EndSession(std::string_view remoteHostname,
                    uint16_t const &remotePort,
                    std::optional<uid_t> uid     = std::nullopt,
                    std::optional<bool> forceEnd = std::nullopt) override
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Record the call
            m_endSessionCalls.emplace_back(remoteHostname, remotePort, uid, forceEnd);
        }
        m_endSessionCallCount++;

        // Use callback if provided
        if (m_endSessionCallback)
        {
            m_endSessionCallback(remoteHostname, remotePort, uid, forceEnd);
        }
    }

    bool SetChildProcessFuncs(DcgmNs::Common::RemoteConn::detail::ChildProcessFuncs const *) override
    {
        // No-op
        return true;
    }

    // Configuration methods

    /**
     * @brief Set the default local port to return
     */
    void SetDefaultRemotePort(uint16_t port)
    {
        m_defaultRemotePort = port;
    }

    /**
     * @brief Set the result to return for StartSession calls
     */
    void SetStartSessionResult(DcgmNs::Common::RemoteConn::detail::TunnelState result)
    {
        m_startSessionResult = result;
    }

    /**
     * @brief Set a callback for StartSession calls
     */
    void SetStartSessionCallback(StartSessionCallback callback)
    {
        m_startSessionCallback = std::move(callback);
    }

    /**
     * @brief Set a callback for EndSession calls
     */
    void SetEndSessionCallback(EndSessionCallback callback)
    {
        m_endSessionCallback = std::move(callback);
    }

    // Inspection methods

    /**
     * @brief Get the recorded StartSession calls
     */
    const std::vector<std::tuple<std::string, uint16_t, std::optional<uid_t>>> &GetStartSessionCalls() const
    {
        return m_startSessionCalls;
    }

    /**
     * @brief Get the recorded EndSession calls
     */
    const std::vector<std::tuple<std::string, uint16_t, std::optional<uid_t>, std::optional<bool>>> &
    GetEndSessionCalls() const
    {
        return m_endSessionCalls;
    }

    /**
     * @brief Reset tracking state
     */
    void Reset()
    {
        m_startSessionCallCount = 0;
        m_endSessionCallCount   = 0;

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_startSessionCalls.clear();
            m_endSessionCalls.clear();
        }

        // Reset configuration back to defaults
        m_startSessionResult   = DcgmNs::Common::RemoteConn::detail::TunnelState::Active;
        m_defaultRemotePort    = 8888;
        m_startSessionCallback = nullptr;
        m_endSessionCallback   = nullptr;
    }

private:
    // Mutex to protect vector modifications
    mutable std::mutex m_mutex;

    // Tracking for calls - using atomic for thread safety
    std::atomic<int> m_startSessionCallCount { 0 };
    std::atomic<int> m_endSessionCallCount { 0 };

    // Records of calls made
    std::vector<std::tuple<std::string, uint16_t, std::optional<uid_t>>> m_startSessionCalls;
    std::vector<std::tuple<std::string, uint16_t, std::optional<uid_t>, std::optional<bool>>> m_endSessionCalls;

    // Default return values
    DcgmNs::Common::RemoteConn::detail::TunnelState m_startSessionResult
        = DcgmNs::Common::RemoteConn::detail::TunnelState::Active;
    uint16_t m_defaultRemotePort = 8888;

    // Callbacks
    StartSessionCallback m_startSessionCallback = nullptr;
    EndSessionCallback m_endSessionCallback     = nullptr;
};