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
#pragma once

#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>

/*****************************************************************************/
typedef struct dcgm_nvswitch_info_t

{
    unsigned int physicalId;                                                   /* Physical hardware ID of the NvSwitch
                                  (expected values are 0x10 (10000)  to 0x15 (10101)
                                  for board1 NVSwitches and 0x18(11000) to 0x1D (11101)
                                  for board2 NVSwitches) */
    DcgmEntityStatus_t status;                                                 /* Status of this NvSwitch */
    dcgmNvLinkLinkState_t nvLinkLinkState[DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH]; /* NvLink link state
                                                                                  for each link */
} dcgm_nvswitch_info_t, *dcgm_nvswitch_info_p;
