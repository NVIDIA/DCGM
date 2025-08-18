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

#include "DcgmApiBase.h"
#include <atomic>
#include <dcgm_mndiag_structs.hpp>
#include <dcgm_module_structs.h>
#include <dcgm_structs.h>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

/**
 * @brief Mock implementation of DcgmApiBase for testing
 *
 * This mock allows tests to define custom behavior for the MultinodeRequest method,
 * which wraps dcgmMultinodeRequest in production code.
 */
class MockDcgmApi : public DcgmApiBase
{
public:
    using SendRequestCallback = std::function<dcgmReturn_t(dcgmHandle_t, dcgmMultinodeRequest_t *)>;

    MockDcgmApi()           = default;
    ~MockDcgmApi() override = default;

    dcgmReturn_t MultinodeRequest(dcgmHandle_t handle, dcgmMultinodeRequest_t *request) override
    {
        // Variables to store outside the lock
        SendRequestCallback callback = nullptr;
        auto const key               = std::make_pair(handle, request->requestType);

        bool hasCallback               = false;
        dcgmReturn_t sendRequestResult = DCGM_ST_OK;
        MnDiagStatus reserveResponse   = MnDiagStatus::READY;
        MnDiagStatus releaseResponse   = MnDiagStatus::READY;
        MnDiagStatus detectResponse    = MnDiagStatus::READY;

        // First acquire lock only for reading shared state
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            // Save the last request for inspection
            m_lastHandle = handle;
            m_sendRequestCallCount++;
            m_requestTypeCounts[request->requestType]++;
            // Check if there's a callback
            auto it = m_handleCommandCallbacks.find(key);
            if (it != m_handleCommandCallbacks.end())
            {
                callback    = it->second;
                hasCallback = true;
            }
            else
            {
                // Store these values for use outside the lock
                sendRequestResult = m_sendRequestResult;
                reserveResponse   = m_reserveResourcesResponse;
                releaseResponse   = m_releaseResourcesResponse;
                detectResponse    = m_detectProcessResponse;
            }
        }

        // If we have a callback, use it
        if (hasCallback)
        {
            return callback(handle, request);
        }

        // No callback, check if we need to modify the response
        switch (request->requestType)
        {
            case ReserveResources:
            {
                request->requestData.resource.response = reserveResponse;
                break;
            }

            case ReleaseResources:
            {
                request->requestData.resource.response = releaseResponse;
                break;
            }

            case DetectProcess:
            {
                request->requestData.resource.response = detectResponse;
                break;
            }

            default:; // Do nothing
        }

        // Return the default result (handles both MnDiag and non-MnDiag cases)
        return sendRequestResult;
    }

    /**
     * @brief Connect_v2 mock - returns the configured result
     */
    dcgmReturn_t Connect_v2(const char *ipAddress,
                            dcgmConnectV2Params_t *connectParams,
                            dcgmHandle_t *pDcgmHandle) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Save the last connection parameters for inspection
        m_lastIpAddress = ipAddress ? ipAddress : "";
        if (connectParams)
        {
            m_lastConnectParams = *connectParams;
        }
        else
        {
            m_lastConnectParams = {};
        }
        m_connectCallCount++;

        // By default, use our handle counter
        dcgmHandle_t handle = ++m_nextHandle;
        if (pDcgmHandle)
        {
            *pDcgmHandle = handle;
        }

        // Store connection info for this handle
        ConnectionInfo info;
        info.ipAddress                = ipAddress ? ipAddress : "";
        info.params                   = connectParams ? *connectParams : dcgmConnectV2Params_t {};
        m_connectionsByHandle[handle] = info;

        return m_connectResult;
    }

    /**
     * @brief Disconnect mock - returns the configured result
     */
    dcgmReturn_t Disconnect(dcgmHandle_t pDcgmHandle) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_lastDisconnectedHandle = pDcgmHandle;
        m_disconnectCallCount++;

        return m_disconnectResult;
    }

    // Configuration methods

    /**
     * @brief Set the result to return for SendRequest calls
     */
    void SetSendRequestResult(dcgmReturn_t result)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_sendRequestResult = result;
    }

    /**
     * @brief Set a specific callback for a handle+command combination
     */
    void SetHandleCommandCallback(dcgmHandle_t handle,
                                  dcgmMultinodeRequestType_t requestType,
                                  SendRequestCallback callback)
    {
        // Create key outside of lock
        auto const key = std::make_pair(handle, requestType);
        // Create a local copy of the callback
        SendRequestCallback callbackCopy = std::move(callback);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_handleCommandCallbacks.insert_or_assign(key, std::move(callbackCopy));
    }

    /**
     * @brief Set the response value for reserve resources messages
     */
    void SetReserveResourcesResponse(MnDiagStatus response)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_reserveResourcesResponse = response;
    }

    /**
     * @brief Set the response value for release resources messages
     */
    void SetReleaseResourcesResponse(MnDiagStatus response)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_releaseResourcesResponse = response;
    }

    /**
     * @brief Set the response value for detect process messages
     */
    void SetDetectProcessResponse(MnDiagStatus response)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_detectProcessResponse = response;
    }

    /**
     * @brief Set the result to return for Connect_v2 calls
     */
    void SetConnectResult(dcgmReturn_t result)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connectResult = result;
    }

    /**
     * @brief Set the result to return for Disconnect calls
     */
    void SetDisconnectResult(dcgmReturn_t result)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_disconnectResult = result;
    }

    // Inspection methods

    /**
     * @brief Get the last handle used for SendRequest
     */
    dcgmHandle_t GetLastHandle() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_lastHandle;
    }

    /**
     * @brief Get the number of times SendRequest was called
     */
    int GetSendRequestCallCount() const
    {
        return m_sendRequestCallCount.load();
    }

    /**
     * @brief Get the number of times a specific request type was called
     */
    unsigned int GetRequestTypeCount(dcgmMultinodeRequestType_t requestType) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_requestTypeCounts.find(requestType);
        if (it != m_requestTypeCounts.end())
        {
            return it->second;
        }
        return 0;
    }

    /**
     * @brief Get the last IP address used for Connect_v2
     */
    std::string GetLastIpAddress() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_lastIpAddress;
    }

    /**
     * @brief Get the last connect parameters used for Connect_v2
     */
    dcgmConnectV2Params_t GetLastConnectParams() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_lastConnectParams;
    }

    /**
     * @brief Get the number of times Connect_v2 was called
     */
    int GetConnectCallCount() const
    {
        return m_connectCallCount.load();
    }

    /**
     * @brief Get the last handle disconnected
     */
    dcgmHandle_t GetLastDisconnectedHandle() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_lastDisconnectedHandle;
    }

    /**
     * @brief Get the number of times Disconnect was called
     */
    int GetDisconnectCallCount() const
    {
        return m_disconnectCallCount.load();
    }

    /**
     * @brief Reset tracking state
     */
    void Reset()
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_lastHandle           = 0;
        m_sendRequestCallCount = 0;
        m_sendRequestResult    = DCGM_ST_OK;

        m_lastIpAddress     = "";
        m_lastConnectParams = {};
        m_connectCallCount  = 0;
        m_connectResult     = DCGM_ST_OK;
        m_nextHandle        = 0;
        m_connectionsByHandle.clear();

        m_lastDisconnectedHandle = 0;
        m_disconnectCallCount    = 0;
        m_disconnectResult       = DCGM_ST_OK;

        m_handleCommandCallbacks.clear();
        m_requestTypeCounts.clear();
        // Reset response values
        m_reserveResourcesResponse = MnDiagStatus::RESERVED;
        m_releaseResourcesResponse = MnDiagStatus::READY;
        m_detectProcessResponse    = MnDiagStatus::READY;
    }

    // Track connections by handle
    struct ConnectionInfo
    {
        std::string ipAddress;
        dcgmConnectV2Params_t params;
    };

    /**
     * @brief Get the connection info for a specific handle
     *
     * @param handle The handle to lookup
     * @return std::optional<ConnectionInfo> Connection info if found, or empty optional if not found
     */
    std::optional<ConnectionInfo> GetConnectionInfo(dcgmHandle_t handle) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        auto it = m_connectionsByHandle.find(handle);
        if (it != m_connectionsByHandle.end())
        {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * @brief Check if a connection exists for a specific handle
     *
     * @param handle The handle to lookup
     * @return bool True if connection exists
     */
    bool HasConnection(dcgmHandle_t handle) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_connectionsByHandle.find(handle) != m_connectionsByHandle.end();
    }

private:
    // Thread synchronization
    mutable std::mutex m_mutex;

    // Default behaviors
    dcgmReturn_t m_sendRequestResult = DCGM_ST_OK;
    dcgmReturn_t m_connectResult     = DCGM_ST_OK;
    dcgmReturn_t m_disconnectResult  = DCGM_ST_OK;

    // Response values for different request types
    MnDiagStatus m_reserveResourcesResponse = MnDiagStatus::RESERVED;
    MnDiagStatus m_releaseResourcesResponse = MnDiagStatus::READY;
    MnDiagStatus m_detectProcessResponse    = MnDiagStatus::READY;

    // Tracking for last call
    dcgmHandle_t m_lastHandle = 0;
    std::atomic<int> m_sendRequestCallCount { 0 };

    // For Connect call tracking
    std::string m_lastIpAddress;
    dcgmConnectV2Params_t m_lastConnectParams = {};
    std::atomic<int> m_connectCallCount { 0 };
    std::atomic<dcgmHandle_t> m_nextHandle { 0 };

    // For Disconnect call tracking
    dcgmHandle_t m_lastDisconnectedHandle = 0;
    std::atomic<int> m_disconnectCallCount { 0 };

    // Storage for callbacks keyed by handle+command
    std::map<std::pair<dcgmHandle_t, uint16_t>, SendRequestCallback> m_handleCommandCallbacks;

    // Storage for connection info
    std::map<dcgmHandle_t, ConnectionInfo> m_connectionsByHandle;
    std::unordered_map<dcgmMultinodeRequestType_t, unsigned int> m_requestTypeCounts;
};