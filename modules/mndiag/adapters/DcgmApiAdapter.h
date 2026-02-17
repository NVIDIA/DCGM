/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmModuleApi.h>
#include <dcgm_agent.h>

/**
 * @brief Adapter for DCGM API functions to use in production code
 */
class DcgmApiAdapter : public DcgmApiBase
{
public:
    DcgmApiAdapter()           = default;
    ~DcgmApiAdapter() override = default;

    /**
     * @brief Send a multi-node request to a host engine using the real dcgmMultinodeRequest
     */
    dcgmReturn_t MultinodeRequest(dcgmHandle_t handle, dcgmMultinodeRequest_t *request) override
    {
        return dcgmMultinodeRequest(handle, request);
    }

    dcgmReturn_t Connect_v3(const char *connectionString,
                            dcgmConnectV3Params_t *connectParams,
                            dcgmHandle_t *pDcgmHandle) override
    {
        return dcgmConnect_v3(connectionString, connectParams, pDcgmHandle);
    }

    dcgmReturn_t Disconnect(dcgmHandle_t pDcgmHandle) override
    {
        return dcgmDisconnect(pDcgmHandle);
    }
};
