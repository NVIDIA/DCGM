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

#include <MnDiagStateMachineBase.h>

/**
 * @brief Mock implementation of MnDiagStateMachineBase for testing
 */
class MockMnDiagStateMachine : public MnDiagStateMachineBase
{
public:
    /**
     * @brief Configuration for mock responses
     */
    struct Config
    {
        bool shouldReserveReturn = true;
        std::vector<std::pair<pid_t, std::string>> detectedProcessInfo;
        bool notifyProcessDetectedReturn    = true;
        bool notifyDiagnosticFinishedReturn = true;
    };

    /**
     * @brief Statistics about method calls
     */
    struct Stats
    {
        bool startCalled                          = false;
        bool stopCalled                           = false;
        bool notifyToReserveCalled                = false;
        bool tryGetDetectedMpiPidCalled           = false;
        bool notifyProcessDetectedCalled          = false;
        bool notifyDiagnosticFinishedCalled       = false;
        bool setProcessExecutionTimeoutCalled     = false;
        std::chrono::seconds lastTimeoutInSeconds = std::chrono::seconds(0);
        bool setMnubergemmPathCalled              = false;
        std::string lastMnubergemmPath;
    };

    /**
     * @brief Constructor with status callback
     *
     * @param setStatusCallback Callback to use for status updates
     */
    explicit MockMnDiagStateMachine(std::function<void(MnDiagStatus)> setStatusCallback)
        : m_statusCallback(std::move(setStatusCallback))
    {}

    bool Start() override
    {
        m_stats.startCalled = true;
        return true;
    }

    void Stop() override
    {
        m_stats.stopCalled = true;
    }

    bool NotifyToReserve() override
    {
        m_stats.notifyToReserveCalled = true;

        if (m_config.shouldReserveReturn)
        {
            m_statusCallback(MnDiagStatus::RESERVED);
        }

        return m_config.shouldReserveReturn;
    }

    dcgmReturn_t TryGetDetectedMpiPid() override
    {
        m_stats.tryGetDetectedMpiPidCalled = true;

        if (m_config.detectedProcessInfo.empty())
        {
            return DCGM_ST_CHILD_SPAWN_FAILED;
        }

        m_statusCallback(MnDiagStatus::RUNNING);

        return DCGM_ST_OK;
    }

    bool NotifyProcessDetected() override
    {
        m_stats.notifyProcessDetectedCalled = true;

        if (m_config.notifyProcessDetectedReturn)
        {
            m_statusCallback(MnDiagStatus::RUNNING);
        }

        return m_config.notifyProcessDetectedReturn;
    }

    bool NotifyDiagnosticFinished() override
    {
        m_stats.notifyDiagnosticFinishedCalled = true;

        if (m_config.notifyDiagnosticFinishedReturn)
        {
            m_statusCallback(MnDiagStatus::READY);
        }

        return m_config.notifyDiagnosticFinishedReturn;
    }

    void SetProcessExecutionTimeout(std::chrono::seconds timeoutInSeconds) override
    {
        m_stats.setProcessExecutionTimeoutCalled = true;
        m_stats.lastTimeoutInSeconds             = timeoutInSeconds;

        if (m_processExecutionTimeoutCallback)
        {
            m_processExecutionTimeoutCallback(timeoutInSeconds);
        }
    }

    void SetMnubergemmPath(std::string const &path) override
    {
        m_stats.setMnubergemmPathCalled = true;
        m_stats.lastMnubergemmPath      = path;

        if (m_mnubergemmPathCallback)
        {
            m_mnubergemmPathCallback(path);
        }
    }

    // Get configuration and statistics
    Config const &GetConfig() const
    {
        return m_config;
    }

    Stats const &GetStats() const
    {
        return m_stats;
    }

    // Set entire configuration at once
    void SetConfig(Config const &config)
    {
        m_config = config;
    }

    // Set callback for SetProcessExecutionTimeout
    void SetProcessExecutionTimeoutCallback(std::function<void(std::chrono::seconds)> callback)
    {
        m_processExecutionTimeoutCallback = std::move(callback);
    }

    // Add setter for the callback:
    void SetMnubergemmPathCallback(std::function<void(std::string const &)> callback)
    {
        m_mnubergemmPathCallback = std::move(callback);
    }

private:
    std::function<void(MnDiagStatus)> m_statusCallback;
    std::function<void(std::chrono::seconds)> m_processExecutionTimeoutCallback;
    std::function<void(std::string const &)> m_mnubergemmPathCallback;

    // Configuration and statistics
    Config m_config;
    Stats m_stats;
};