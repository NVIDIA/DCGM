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

#include "DcgmProfiles.h"

#include "DcgmHostEngineHandler.h"
#include "DcgmUtilities.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <bitset>
#include <condition_variable>
#include <dcgm_nvml.h>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

dcgmReturn_t DcgmGetWorkloadPowerProfilesInfo(dcgm_power_profile_helper_t const *gpuInfo,
                                              dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo)
{
    if ((gpuInfo == nullptr) || (profilesInfo == nullptr))
    {
        return DCGM_ST_BADPARAM;
    }

    nvmlWorkloadPowerProfileProfilesInfo_t nvmlProfilesInfo = {};

    nvmlProfilesInfo.version = nvmlWorkloadPowerProfileProfilesInfo_v1;

    nvmlReturn_t nvmlSt = nvmlDeviceWorkloadPowerProfileGetProfilesInfo(gpuInfo->nvmlDevice, &nvmlProfilesInfo);

    if (nvmlSt == NVML_ERROR_NOT_SUPPORTED)
    {
        log_debug("gpuId %d workload power profiles not supported.", gpuInfo->gpuId);
        return DCGM_ST_NOT_SUPPORTED;
    }
    else if (nvmlSt != NVML_SUCCESS)
    {
        log_debug("Could not query workload power profiles info for gpuId %d - rc %d.", gpuInfo->gpuId, nvmlSt);
        return DCGM_ST_NVML_ERROR;
    }

    unsigned int perfIndex = 0;
    for (int i = 0; i < NVML_WORKLOAD_POWER_MAX_PROFILES; i++)
    {
        if (!(NVML_255_MASK_BIT_GET(i, nvmlProfilesInfo.perfProfilesMask)))
        {
            continue;
        }

        profilesInfo->workloadPowerProfile[perfIndex].profileId
            = (dcgmPowerProfileType_t)nvmlProfilesInfo.perfProfile[i].profileId;
        profilesInfo->workloadPowerProfile[perfIndex].priority = nvmlProfilesInfo.perfProfile[i].priority;

        memcpy(profilesInfo->workloadPowerProfile[perfIndex].conflictingMask,
               nvmlProfilesInfo.perfProfile[i].conflictingMask.mask,
               sizeof(profilesInfo->workloadPowerProfile[perfIndex].conflictingMask));

        perfIndex++;
    }

    profilesInfo->profileCount = perfIndex;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmGetWorkloadPowerProfilesStatus(dcgm_power_profile_helper_t const *gpuInfo,
                                                dcgmDeviceWorkloadPowerProfilesStatus_v1 *profilesStatus)
{
    if ((gpuInfo == nullptr) || (profilesStatus == nullptr))
    {
        return DCGM_ST_BADPARAM;
    }

    nvmlWorkloadPowerProfileCurrentProfiles_t profiles = {};

    profiles.version = nvmlWorkloadPowerProfileCurrentProfiles_v1;

    nvmlReturn_t nvmlSt = nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(gpuInfo->nvmlDevice, &profiles);

    if (nvmlSt == NVML_ERROR_NOT_SUPPORTED)
    {
        log_debug("gpuId %d workload power profiles not supported.", gpuInfo->gpuId);
        return DCGM_ST_NOT_SUPPORTED;
    }
    else if (nvmlSt != NVML_SUCCESS)
    {
        log_debug("Could not query workload power profiles status for gpuId %d - rc %d.", gpuInfo->gpuId, nvmlSt);
        return DCGM_ST_NVML_ERROR;
    }

    memcpy(profilesStatus->profileMask, profiles.perfProfilesMask.mask, sizeof(profilesStatus->profileMask));
    memcpy(profilesStatus->requestedProfileMask,
           profiles.requestedProfilesMask.mask,
           sizeof(profilesStatus->requestedProfileMask));
    memcpy(profilesStatus->enforcedProfileMask,
           profiles.enforcedProfilesMask.mask,
           sizeof(profilesStatus->enforcedProfileMask));

    return DCGM_ST_OK;
}
