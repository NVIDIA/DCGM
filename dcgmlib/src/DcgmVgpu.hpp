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

#include "DcgmCacheManager.h"
#include "FbcSessionOpsAnyPtr.hpp"
#include "NvmlTaskRunner.hpp"
#include "VgpuFieldValueOpsAnyPtr.hpp"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

namespace DcgmVgpuDetail
{

/**
 * Type-erased operation bundle for Frame Buffer Capture (FBC) session helpers.
 *
 * Production callers use the public overloads below. Tests construct this wrapper from a pointer to an
 * object with matching member functions and pass it to the DcgmVgpuDetail FBC session overloads.
 */
using FbcSessionOps = FbcSessionOpsAnyPtr;

/**
 * Type-erased operation bundle for vGPU field value buffering.
 *
 * Production callers use the public overloads below. Tests construct this wrapper from a pointer to an
 * object with matching member functions and pass it to the DcgmVgpuDetail field-buffering overload.
 */
using VgpuFieldValueOps = VgpuFieldValueOpsAnyPtr;


/**
 * Fetch and append FBC session information for a GPU through injected dependencies.
 *
 * @param[in,out] cm: Cache manager used by the dependency wrapper.
 * @param[in,out] nvmlDriver: NVML task runner used by the dependency wrapper.
 * @param[in] safeNvmlDevice: Safe NVML device handle for the target GPU.
 * @param[in,out] threadCtx: Cache update context identifying the target entity and field.
 * @param[in,out] watchInfo: Optional watch state updated with NVML status.
 * @param[in] now: Timestamp to use for the appended value.
 * @param[in] expireTime: Oldest timestamp to keep for the appended value.
 * @param[in,out] deps: Dependency wrapper used for allocation, NVML, and cache appends.
 * @returns: DCGM_ST_OK on success, or a DCGM error code.
 */
dcgmReturn_t GetDeviceFBCSessionsInfo(DcgmCacheManager &cm,
                                      NvmlTaskRunner &nvmlDriver,
                                      SafeNvmlHandle safeNvmlDevice,
                                      dcgmcm_update_thread_t &threadCtx,
                                      dcgmcm_watch_info_p watchInfo,
                                      timelib64_t now,
                                      timelib64_t expireTime,
                                      FbcSessionOps &ops);

/**
 * Fetch and append FBC session information for a vGPU through injected dependencies.
 *
 * @param[in,out] cm: Cache manager used by the dependency wrapper.
 * @param[in,out] nvmlDriver: NVML task runner used by the dependency wrapper.
 * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
 * @param[in,out] threadCtx: Cache update context identifying the target entity and field.
 * @param[in,out] watchInfo: Optional watch state updated with NVML status.
 * @param[in] now: Timestamp to use for the appended value.
 * @param[in] expireTime: Oldest timestamp to keep for the appended value.
 * @param[in,out] deps: Dependency wrapper used for allocation, NVML, and cache appends.
 * @returns: DCGM_ST_OK on success, or a DCGM error code.
 */
dcgmReturn_t GetVgpuInstanceFBCSessionsInfo(DcgmCacheManager &cm,
                                            NvmlTaskRunner &nvmlDriver,
                                            SafeVgpuInstance vgpuId,
                                            dcgmcm_update_thread_t &threadCtx,
                                            dcgmcm_watch_info_p watchInfo,
                                            timelib64_t now,
                                            timelib64_t expireTime,
                                            FbcSessionOps &ops);

/**
 * Buffer or cache a vGPU field value through injected dependencies.
 *
 * @param[in,out] cm: Cache manager used by the dependency wrapper.
 * @param[in,out] threadCtx: Cache update context identifying the target entity and field.
 * @param[in,out] nvmlDriver: NVML task runner used by the dependency wrapper.
 * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
 * @param[in] fieldMeta: Metadata for the field being queried.
 * @param[in,out] ops: Operation bundle used for NVML, time, cache appends, and watch updates.
 * @param[in,out] fbcOps: Operation bundle used when the requested field delegates to FBC session collection.
 * @returns: DCGM_ST_OK on success, or a DCGM error code.
 */
dcgmReturn_t BufferOrCacheLatestVgpuValue(DcgmCacheManager &cm,
                                          dcgmcm_update_thread_t &threadCtx,
                                          NvmlTaskRunner &nvmlDriver,
                                          SafeVgpuInstance vgpuId,
                                          dcgm_field_meta_p fieldMeta,
                                          VgpuFieldValueOps &ops,
                                          FbcSessionOps &fbcOps);

} // namespace DcgmVgpuDetail

/*************************************************************************/
/*
 * Helpers to fetch the information of active FBC sessions on the given device/vGPU instance
 *
 */
dcgmReturn_t GetDeviceFBCSessionsInfo(DcgmCacheManager &cm,
                                      NvmlTaskRunner &nvmlDriver,
                                      SafeNvmlHandle safeNvmlDevice,
                                      dcgmcm_update_thread_t &threadCtx,
                                      dcgmcm_watch_info_p watchInfo,
                                      timelib64_t now,
                                      timelib64_t expireTime);

dcgmReturn_t GetVgpuInstanceFBCSessionsInfo(DcgmCacheManager &cm,
                                            NvmlTaskRunner &nvmlDriver,
                                            SafeVgpuInstance vgpuId,
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
                                          NvmlTaskRunner &nvmlDriver,
                                          SafeVgpuInstance vgpuId,
                                          dcgm_field_meta_p fieldMeta);
