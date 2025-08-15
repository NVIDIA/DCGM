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

#include <dcgm_module_structs.h>
#include <dcgm_structs.h>

/**
 * @brief Base interface for DCGM API functions to allow for mocking in tests
 */
class DcgmApiBase
{
public:
    virtual ~DcgmApiBase() = default;

    /**
     * @brief Send a multi-node request to a host engine
     *
     * @param handle Connection handle
     * @param request Multinode request
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    virtual dcgmReturn_t MultinodeRequest(dcgmHandle_t handle, dcgmMultinodeRequest_t *request) = 0;

    /**
     * @brief Connect to a DCGM instance
     *
     * @param ipAddress IP address of the DCGM instance
     * @param connectParams Connection parameters
     * @param pDcgmHandle DCGM handle to connect to
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    virtual dcgmReturn_t Connect_v2(const char *ipAddress,
                                    dcgmConnectV2Params_t *connectParams,
                                    dcgmHandle_t *pDcgmHandle)
        = 0;

    /**
     * @brief Disconnect from a DCGM instance
     *
     * @param pDcgmHandle DCGM handle to disconnect from
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    virtual dcgmReturn_t Disconnect(dcgmHandle_t pDcgmHandle) = 0;
};
