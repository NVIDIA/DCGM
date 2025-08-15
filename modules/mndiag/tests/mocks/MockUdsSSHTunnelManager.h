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

#include <UdsSSHTunnelManagerBase.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Mock implementation of UdsSSHTunnelManagerBase for testing
 */
class MockUdsSSHTunnelManager : public UdsSSHTunnelManagerBase
{
public:
    using StartSessionCallback
        = std::function<DcgmNs::Common::RemoteConn::detail::
                            TunnelState(std::string_view, std::string const &, std::string &, std::optional<uid_t>)>;
    using EndSessionCallback
        = std::function<void(std::string_view, std::string const &, std::optional<uid_t>, std::optional<bool>)>;

    MockUdsSSHTunnelManager()                                           = default;
    ~MockUdsSSHTunnelManager() override                                 = default;
    MockUdsSSHTunnelManager(const MockUdsSSHTunnelManager &)            = delete;
    MockUdsSSHTunnelManager &operator=(const MockUdsSSHTunnelManager &) = delete;
    MockUdsSSHTunnelManager(MockUdsSSHTunnelManager &&)                 = delete;
    MockUdsSSHTunnelManager &operator=(MockUdsSSHTunnelManager &&)      = delete;

    /**
     * @brief StartSession mock - returns the configured result or uses callback
     */
    DcgmNs::Common::RemoteConn::detail::TunnelState StartSession(std::string_view remoteHostname,
                                                                 std::string const &remoteAddress,
                                                                 std::string &localAddress,
                                                                 std::optional<uid_t> uid = std::nullopt) override
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Record the call
            m_startSessionCalls.emplace_back(remoteHostname, remoteAddress, uid);
        }
        m_startSessionCallCount++;

        // Use callback if provided
        if (m_startSessionCallback)
        {
            return m_startSessionCallback(remoteHostname, remoteAddress, localAddress, uid);
        }

        // Otherwise use default behavior
        if (m_startSessionResult == DcgmNs::Common::RemoteConn::detail::TunnelState::Active)
        {
            // Generate a fake local socket path
            localAddress = m_defaultRemoteSocketPath;
        }

        return m_startSessionResult;
    }

    /**
     * @brief EndSession mock - calls the configured callback if provided
     */
    void EndSession(std::string_view remoteHostname,
                    std::string const &remoteAddress,
                    std::optional<uid_t> uid     = std::nullopt,
                    std::optional<bool> forceEnd = std::nullopt) override
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Record the call
            m_endSessionCalls.emplace_back(remoteHostname, remoteAddress, uid, forceEnd);
        }
        m_endSessionCallCount++;

        // Use callback if provided
        if (m_endSessionCallback)
        {
            m_endSessionCallback(remoteHostname, remoteAddress, uid, forceEnd);
        }
    }

    bool SetChildProcessFuncs(DcgmNs::Common::RemoteConn::detail::ChildProcessFuncs const *) override
    {
        // No-op
        return true;
    }
    // Configuration methods

    /**
     * @brief Set the default local socket path to return
     */
    void SetDefaultRemoteSocketPath(std::string path)
    {
        m_defaultRemoteSocketPath = std::move(path);
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
    const std::vector<std::tuple<std::string, std::string, std::optional<uid_t>>> &GetStartSessionCalls() const
    {
        return m_startSessionCalls;
    }

    /**
     * @brief Get the recorded EndSession calls
     */
    const std::vector<std::tuple<std::string, std::string, std::optional<uid_t>, std::optional<bool>>> &
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
        m_startSessionResult      = DcgmNs::Common::RemoteConn::detail::TunnelState::Active;
        m_defaultRemoteSocketPath = "/tmp/dcgm_socket_mock";
        m_startSessionCallback    = nullptr;
        m_endSessionCallback      = nullptr;
    }

private:
    // Mutex to protect vector modifications
    mutable std::mutex m_mutex;

    // Tracking for calls - using atomic for thread safety
    std::atomic<int> m_startSessionCallCount { 0 };
    std::atomic<int> m_endSessionCallCount { 0 };

    // Records of calls made
    std::vector<std::tuple<std::string, std::string, std::optional<uid_t>>> m_startSessionCalls;
    std::vector<std::tuple<std::string, std::string, std::optional<uid_t>, std::optional<bool>>> m_endSessionCalls;

    // Default return values
    DcgmNs::Common::RemoteConn::detail::TunnelState m_startSessionResult
        = DcgmNs::Common::RemoteConn::detail::TunnelState::Active;
    std::string m_defaultRemoteSocketPath = "/tmp/dcgm_socket_mock";

    // Callbacks
    StartSessionCallback m_startSessionCallback = nullptr;
    EndSessionCallback m_endSessionCallback     = nullptr;
};