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

#include "NvmlTaskRunner.hpp"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <map>
#include <memory>
#include <vector>


typedef struct
{
    unsigned int gpuId;
    SafeNvmlHandle nvmlDevice;
} dcgm_power_profile_helper_t;

/**
 * Interface for reading workload power profile data from an NVML-facing collaborator.
 *
 * Implementations read NVML workload power profile metadata and current status for
 * a safe NVML device handle. Production callers use the NvmlTaskRunner-backed
 * implementation; unit tests may provide fakes that return deterministic NVML
 * status codes and profile structures.
 */
class DcgmWorkloadPowerProfileReader
{
public:
    /**
     * Destroy the workload power profile reader.
     */
    virtual ~DcgmWorkloadPowerProfileReader() noexcept = default;

    /**
     * Read supported workload power profile metadata for a device.
     *
     * @param[in] device Safe NVML device handle to query.
     * @param[out] profilesInfo NVML profile metadata populated by the implementation.
     *
     * @return NVML_SUCCESS on success; otherwise an NVML error code describing the failure.
     */
    virtual nvmlReturn_t GetWorkloadPowerProfileProfilesInfo(
        SafeNvmlHandle device,
        nvmlWorkloadPowerProfileProfilesInfo_t *profilesInfo) noexcept = 0;

    /**
     * Read current workload power profile masks for a device.
     *
     * @param[in] device Safe NVML device handle to query.
     * @param[out] currentProfiles NVML current profile masks populated by the implementation.
     *
     * @return NVML_SUCCESS on success; otherwise an NVML error code describing the failure.
     */
    virtual nvmlReturn_t GetWorkloadPowerProfileCurrentProfiles(
        SafeNvmlHandle device,
        nvmlWorkloadPowerProfileCurrentProfiles_t *currentProfiles) noexcept = 0;
};

/*************************************************************************/
/*
 * Get information about the profiles supported on this device
 *
 * nvmlDriver    IN: A started NVML task runner
 * gpuInfo       IN: gpu to query
 * profilesInfo OUT: information about supported profiles
 *
 * Returns 0 on success
 *        <0 on error. See DCGM_ST_? #defines
 */
dcgmReturn_t DcgmGetWorkloadPowerProfilesInfo(NvmlTaskRunner &nvmlDriver,
                                              dcgm_power_profile_helper_t const *gpuInfo,
                                              dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo);

/**
 * Get information about workload power profiles supported on this device.
 *
 * @param[in] profileReader Reader used to query NVML-facing workload power profile metadata.
 * @param[in] gpuInfo GPU to query.
 * @param[out] profilesInfo Information about supported profiles.
 *
 * @return DCGM_ST_OK on success; DCGM_ST_BADPARAM for null input or output pointers;
 *         DCGM_ST_NOT_SUPPORTED when NVML reports profile support is unavailable;
 *         DCGM_ST_NVML_ERROR for other NVML failures.
 */
dcgmReturn_t DcgmGetWorkloadPowerProfilesInfo(DcgmWorkloadPowerProfileReader &profileReader,
                                              dcgm_power_profile_helper_t const *gpuInfo,
                                              dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo);

/*************************************************************************/
/*
 * Get status information about the profiles supported on this device
 *
 * nvmlDriver    IN: A started NVML task runner
 * gpuInfo       IN: gpu to query
 * profileStatus OUT: status information about supported profiles
 *
 * Returns 0 on success
 *        <0 on error. See DCGM_ST_? #defines
 */
dcgmReturn_t DcgmGetWorkloadPowerProfilesStatus(NvmlTaskRunner &nvmlDriver,
                                                dcgm_power_profile_helper_t const *gpuInfo,
                                                dcgmDeviceWorkloadPowerProfilesStatus_v1 *profilesStatus);

/**
 * Get current workload power profile status for this device.
 *
 * @param[in] profileReader Reader used to query NVML-facing workload power profile status.
 * @param[in] gpuInfo GPU to query.
 * @param[out] profilesStatus Current, requested, and enforced profile masks.
 *
 * @return DCGM_ST_OK on success; DCGM_ST_BADPARAM for null input or output pointers;
 *         DCGM_ST_NOT_SUPPORTED when NVML reports profile support is unavailable;
 *         DCGM_ST_NVML_ERROR for other NVML failures.
 */
dcgmReturn_t DcgmGetWorkloadPowerProfilesStatus(DcgmWorkloadPowerProfileReader &profileReader,
                                                dcgm_power_profile_helper_t const *gpuInfo,
                                                dcgmDeviceWorkloadPowerProfilesStatus_v1 *profilesStatus);
