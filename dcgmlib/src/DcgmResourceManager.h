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

#include <DcgmMutex.h>
#include <chrono>
#include <cstdint>
#include <dcgm_structs.h>
#include <functional>
#include <optional>
#include <string>

// ---------------------------------
// Reservation information
struct ReservationInfo
{
    std::chrono::system_clock::time_point reservationTime;
};

class DcgmResourceManager
{
public:
    DcgmResourceManager();
    ~DcgmResourceManager();

    // Reserve resources, returns a token if successful, nullopt if not
    std::optional<unsigned int> ReserveResources();

    // Free resources using the token
    bool FreeResources(unsigned int const &token);

    // Get information about a reservation
    std::optional<ReservationInfo> GetReservationInfo(unsigned int const &token) const;

private:
    // Private methods
    unsigned int GenerateToken();

    // Private variables
    bool m_isReserved           = false;
    unsigned int m_currentToken = 0;      // Current active token
    ReservationInfo m_currentReservation; // Current reservation info
    mutable DcgmMutex m_mutex = DcgmMutex(0);
};
