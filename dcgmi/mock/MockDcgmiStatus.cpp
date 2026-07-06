/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "MockDcgmiStatus.hpp"

MockDcgmiStatusData g_mockDcgmiStatusData;

void ResetMockDcgmiStatus()
{
    g_mockDcgmiStatusData                     = {};
    g_mockDcgmiStatusData.statusCreateReturn  = DCGM_ST_OK;
    g_mockDcgmiStatusData.statusDestroyReturn = DCGM_ST_OK;
    g_mockDcgmiStatusData.statusHandle        = 0xfeed;
}

extern "C" dcgmReturn_t dcgmStatusCreate(dcgmStatus_t *statusHandle)
{
    g_mockDcgmiStatusData.statusCreateCallCount++;

    if (statusHandle == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (g_mockDcgmiStatusData.statusCreateReturn != DCGM_ST_OK)
    {
        return g_mockDcgmiStatusData.statusCreateReturn;
    }

    *statusHandle = g_mockDcgmiStatusData.statusHandle;
    return DCGM_ST_OK;
}

extern "C" dcgmReturn_t dcgmStatusDestroy(dcgmStatus_t statusHandle)
{
    g_mockDcgmiStatusData.statusDestroyCallCount++;
    g_mockDcgmiStatusData.lastDestroyedStatus = statusHandle;
    return g_mockDcgmiStatusData.statusDestroyReturn;
}
