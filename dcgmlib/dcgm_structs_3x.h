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
#ifndef DCGM_STRUCTS_3X__H
#define DCGM_STRUCTS_3X__H

#include "dcgm_structs.h"

// DCGM_CASSERT
#include "dcgm_structs_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Status of all of the NvLinks in a given system
 */
typedef struct
{
    unsigned int version; //!< Version of this request. Should be dcgmNvLinkStatus_version1
    unsigned int numGpus; //!< Number of entries in gpus[] that are populated
    dcgmNvLinkGpuLinkStatus_v3 gpus[DCGM_MAX_NUM_DEVICES]; //!< Per-GPU NvLink link statuses
    unsigned int numNvSwitches;                            //!< Number of entries in nvSwitches[] that are populated
    dcgmNvLinkNvSwitchLinkStatus_t nvSwitches[DCGM_MAX_NUM_SWITCHES]; //!< Per-NvSwitch link statuses
} dcgmNvLinkStatus_v3;

/**
 * Version 3 of dcgmNvLinkStatus
 */
#define dcgmNvLinkStatus_version3 MAKE_DCGM_VERSION(dcgmNvLinkStatus_v3, 3)

DCGM_CASSERT(dcgmNvLinkStatus_version3 == (long)0x30015bc, 3);

/**
 * Version 1 of dcgmSettingsSetLoggingSeverity_t
 */
typedef struct
{
    int targetLogger;
    DcgmLoggingSeverity_t targetSeverity;
} dcgmSettingsSetLoggingSeverity_v1;


#define dcgmSettingsSetLoggingSeverity_version1 MAKE_DCGM_VERSION(dcgmSettingsSetLoggingSeverity_v1, 1)

DCGM_CASSERT(dcgmSettingsSetLoggingSeverity_version1 == (long)0x1000008, 1);

#ifdef __cplusplus
}
#endif

#endif // DCGM_STRUCTS_3X__H
