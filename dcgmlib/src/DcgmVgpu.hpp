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

#include "DcgmCacheManager.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <map>
#include <memory>
#include <vector>


/*************************************************************************/
/*
 * Helpers to fetch the information of active FBC sessions on the given device/vGPU instance
 *
 */
dcgmReturn_t GetDeviceFBCSessionsInfo(DcgmCacheManager &cm,
                                      nvmlDevice_t nvmlDevice,
                                      dcgmcm_update_thread_t &threadCtx,
                                      dcgmcm_watch_info_p watchInfo,
                                      timelib64_t now,
                                      timelib64_t expireTime);

dcgmReturn_t GetVgpuInstanceFBCSessionsInfo(DcgmCacheManager *cm,
                                            nvmlVgpuInstance_t vgpuId,
                                            dcgmcm_update_thread_t &threadCtx,
                                            dcgmcm_watch_info_p watchInfo,
                                            timelib64_t now,
                                            timelib64_t expireTime);

/*************************************************************************/
/*
 * Cache or buffer the latest value for a watched vGPU field
 *
 * Returns 0 if OK
 *        <0 DCGM_ST_? on module error
 */
dcgmReturn_t BufferOrCacheLatestVgpuValue(DcgmCacheManager &cm,
                                          dcgmcm_update_thread_t &threadCtx,
                                          nvmlVgpuInstance_t vgpuId,
                                          dcgm_field_meta_p fieldMeta);
