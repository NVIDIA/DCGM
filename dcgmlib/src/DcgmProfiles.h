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

#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <map>
#include <memory>
#include <vector>


typedef struct
{
    unsigned int gpuId;
    nvmlDevice_t nvmlDevice;
} dcgm_power_profile_helper_t;

/*************************************************************************/
/*
 * Get information about the profiles supported on this device
 *
 * gpuInfo       IN: gpu to query
 * profilesInfo OUT: information about supported profiles
 *
 * Returns 0 on success
 *        <0 on error. See DCGM_ST_? #defines
 */
dcgmReturn_t DcgmGetWorkloadPowerProfilesInfo(dcgm_power_profile_helper_t const *gpuInfo,
                                              dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo);

/*************************************************************************/
/*
 * Get status information about the profiles supported on this device
 *
 * gpuInfo       IN: gpu to query
 * profileStatus OUT: status information about supported profiles
 *
 * Returns 0 on success
 *        <0 on error. See DCGM_ST_? #defines
 */
dcgmReturn_t DcgmGetWorkloadPowerProfilesStatus(dcgm_power_profile_helper_t const *gpuInfo,
                                                dcgmDeviceWorkloadPowerProfilesStatus_v1 *profilesStatus);