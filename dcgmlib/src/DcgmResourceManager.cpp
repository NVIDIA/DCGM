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

#include "DcgmResourceManager.h"
#include <DcgmLogging.h>
#include <random>

DcgmResourceManager::DcgmResourceManager()  = default;
DcgmResourceManager::~DcgmResourceManager() = default;

// Public
std::optional<unsigned int> DcgmResourceManager::ReserveResources()
{
    bool canReserve = false;
    {
        DcgmLockGuard guard(&m_mutex);
        if (!m_isReserved)
        {
            m_isReserved = true;
            canReserve   = true;
        }
    }

    if (!canReserve)
    {
        DCGM_LOG_ERROR << "Resources not available";
        return std::nullopt;
    }

    unsigned int token = GenerateToken();
    ReservationInfo info;
    info.reservationTime = std::chrono::system_clock::now();

    {
        DcgmLockGuard guard(&m_mutex);
        m_currentToken       = token;
        m_currentReservation = info;
    }

    return token;
}

bool DcgmResourceManager::FreeResources(unsigned int const &token)
{
    bool success = false;
    {
        DcgmLockGuard guard(&m_mutex);

        if (token == m_currentToken && m_isReserved)
        {
            m_currentToken = 0;
            m_isReserved   = false;
            success        = true;
        }
    }

    if (!success)
    {
        DCGM_LOG_WARNING << "Attempt to free resources with invalid token";
        return false;
    }

    DCGM_LOG_INFO << "Freed resources";
    return true;
}

std::optional<ReservationInfo> DcgmResourceManager::GetReservationInfo(unsigned int const &token) const
{
    DcgmLockGuard guard(&m_mutex);

    if (token != m_currentToken || !m_isReserved)
    {
        DCGM_LOG_DEBUG << "No reservation found for token " << token;
        return std::nullopt;
    }

    return m_currentReservation;
}


// Private
unsigned int DcgmResourceManager::GenerateToken()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<unsigned int> dis(1, UINT_MAX);

    unsigned int token;
    do
    {
        token = dis(gen);
    } while (token == m_currentToken);

    return token;
}
