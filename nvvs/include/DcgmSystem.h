/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef DCGM_SYSTEM_H
#define DCGM_SYSTEM_H

#include <vector>

#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include "dcgm_test_apis.h"

class DcgmSystem
{
public:
    DcgmSystem();
    ~DcgmSystem();

    /*
     * Saves a copy of the handle this system object should use
     */
    void Init(dcgmHandle_t handle);

    /*
     * Populates deviceAttr with the attributes retrieved from DCGM
     *
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetDeviceAttributes(unsigned int gpuId, dcgmDeviceAttributes_t &deviceAttr);

    /*
     * Populates gpuStatus with the status retrieved from DCGM
     *
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized or gpuStatus is null
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetGpuStatus(unsigned int gpuId, DcgmEntityStatus_t *gpuStatus);

    /*
     * Populates gpuIdList with the supported GPUs present on this host.
     *
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetAllSupportedDevices(std::vector<unsigned int> &gpuIdList);

    /*
     * Populates gpuIdList with the GPUs present on this host.
     *
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetAllDevices(std::vector<unsigned int> &gpuIdList);

    /*
     * Retrieves the latest value for the specified field and populates dcgmFieldValue_v2 accordingly
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetGpuLatestValue(unsigned int gpuId,
                                   unsigned short fieldId,
                                   unsigned int flags,
                                   dcgmFieldValue_v2 &value);

    /*
     * Retrieves the latest field values for the specified gpus and calls the given
     * dcgmFieldValueEntityEnumeration_f (checker) with the retrieved data and userData.
     * @return:
     *
     * DCGM_ST_OK       : success
     * DCGM_ST_BADPARAM : if the handle hasn't been initialized
     * DCGM_ST_*        : if returned from calls to DCGM
     */
    dcgmReturn_t GetLatestValuesForGpus(const std::vector<unsigned int> &gpuIds,
                                        std::vector<unsigned short> &fieldIds,
                                        unsigned int flags,
                                        dcgmFieldValueEntityEnumeration_f checker,
                                        void *userData);

    /*
     * @return:
     *
     * true             : this system object is initialized
     * false            : this system object isn't initialized
     */
    bool IsInitialized() const;

private:
    dcgmHandle_t m_handle; // We use this handle but we do not own it.
};

#endif