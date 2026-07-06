/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "NvmlTaskRunner.hpp"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <cstddef>

/**
 * Operations shared by FBC session collection and vGPU field value buffering.
 */
class DcgmVgpuOpsCommon
{
public:
    /**
     * Allocate memory for vGPU helper payloads.
     *
     * @param[in] size: Number of bytes to allocate.
     * @returns: Pointer to allocated memory, or nullptr if allocation fails.
     *
     * Not thread-safe. Concurrent use of the returned pointer requires external synchronization.
     */
    void *Malloc(std::size_t size);

    /**
     * Release memory allocated by Malloc.
     *
     * @param[in] ptr: Pointer previously returned by Malloc, or nullptr.
     *
     * Not thread-safe. The pointer must not be used after this call.
     */
    void Free(void *ptr);

    /**
     * Append a binary field value for the entity identified by threadCtx.
     *
     * @param[in,out] cm: Cache manager that owns the field buffer.
     * @param[in,out] threadCtx: Update context with entity key and destination buffer.
     * @param[in] value: Binary payload to append.
     * @param[in] valueSize: Size of value in bytes.
     * @param[in] timestamp: Sample timestamp in microseconds since 1970.
     * @param[in] oldestKeepTimestamp: Oldest sample timestamp to retain for this field.
     * @returns: DCGM_ST_OK on success, or a DCGM error code from the cache manager.
     *
     * Not thread-safe. Caller must serialize access to cm and threadCtx.
     */
    dcgmReturn_t AppendEntityBlob(DcgmCacheManager &cm,
                                  dcgmcm_update_thread_t &threadCtx,
                                  void *value,
                                  int valueSize,
                                  timelib64_t timestamp,
                                  timelib64_t oldestKeepTimestamp);

    /**
     * Map an NVML return code to the corresponding DCGM return code.
     *
     * @param[in] nvmlReturn: NVML status from a vGPU or device query.
     * @returns: DCGM_ST_OK when nvmlReturn is NVML_SUCCESS; otherwise the mapped DCGM error code.
     *
     * Thread-safe. Does not access shared mutable state.
     */
    dcgmReturn_t NvmlReturnToDcgmReturn(nvmlReturn_t nvmlReturn);
};

/**
 * Dependency surface for Frame Buffer Capture (FBC) session collection.
 *
 * Production code and tests provide concrete types with this public API. The type-erased wrapper is
 * generated from this class by libclang-code-generators.
 */
class FbcSessionOps : public virtual DcgmVgpuOpsCommon
{
public:
    /**
     * Query active FBC sessions for a GPU.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] safeNvmlDevice: Safe NVML device handle for the target GPU.
     * @param[out] sessionCount: Number of active sessions on return.
     * @param[in,out] sessionInfo: Optional array of length *sessionCount to receive session details.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlDeviceGetFBCSessions(NvmlTaskRunner &nvmlDriver,
                                          SafeNvmlHandle safeNvmlDevice,
                                          unsigned int *sessionCount,
                                          nvmlFBCSessionInfo_t *sessionInfo);

    /**
     * Query active FBC sessions for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] sessionCount: Number of active sessions on return.
     * @param[in,out] sessionInfo: Optional array of length *sessionCount to receive session details.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetFBCSessions(NvmlTaskRunner &nvmlDriver,
                                                SafeVgpuInstance vgpuId,
                                                unsigned int *sessionCount,
                                                nvmlFBCSessionInfo_t *sessionInfo);
};

/**
 * Dependency surface for vGPU field value buffering.
 *
 * Production code and tests provide concrete types with this public API. The type-erased wrapper is
 * generated from this class by libclang-code-generators.
 */
class VgpuFieldValueOps : public virtual DcgmVgpuOpsCommon
{
public:
    /**
     * Append a string field value for the entity identified by threadCtx.
     *
     * @param[in,out] cm: Cache manager that owns the field buffer.
     * @param[in,out] threadCtx: Update context with entity key and destination buffer.
     * @param[in] value: NUL-terminated string to append.
     * @param[in] timestamp: Sample timestamp in microseconds since 1970.
     * @param[in] oldestKeepTimestamp: Oldest sample timestamp to retain for this field.
     * @returns: DCGM_ST_OK on success, or a DCGM error code from the cache manager.
     *
     * Not thread-safe. Caller must serialize access to cm and threadCtx.
     */
    dcgmReturn_t AppendEntityString(DcgmCacheManager &cm,
                                    dcgmcm_update_thread_t &threadCtx,
                                    char const *value,
                                    timelib64_t timestamp,
                                    timelib64_t oldestKeepTimestamp);

    /**
     * Append a 64-bit integer field value for the entity identified by threadCtx.
     *
     * @param[in,out] cm: Cache manager that owns the field buffer.
     * @param[in,out] threadCtx: Update context with entity key and destination buffer.
     * @param[in] value1: Primary integer sample value.
     * @param[in] value2: Secondary integer sample value.
     * @param[in] timestamp: Sample timestamp in microseconds since 1970.
     * @param[in] oldestKeepTimestamp: Oldest sample timestamp to retain for this field.
     * @returns: DCGM_ST_OK on success, or a DCGM error code from the cache manager.
     *
     * Not thread-safe. Caller must serialize access to cm and threadCtx.
     */
    dcgmReturn_t AppendEntityInt64(DcgmCacheManager &cm,
                                   dcgmcm_update_thread_t &threadCtx,
                                   long long value1,
                                   long long value2,
                                   timelib64_t timestamp,
                                   timelib64_t oldestKeepTimestamp);

    /**
     * Update watch timing for a monitored field.
     *
     * @param[in,out] cm: Cache manager that owns the watch table.
     * @param[in,out] watchInfo: Watch entry to update.
     * @param[in] monitorIntervalUsec: Polling interval in microseconds.
     * @param[in] maxAgeUsec: Maximum sample age to retain.
     * @param[in] maxKeepSamples: Maximum number of samples to keep.
     * @param[in] watcher: Identity of the watcher requesting the update.
     * @returns: DCGM_ST_OK on success, or a DCGM error code from the cache manager.
     *
     * Not thread-safe. Caller must serialize access to cm and watchInfo.
     */
    dcgmReturn_t UpdateFieldWatch(DcgmCacheManager &cm,
                                  dcgmcm_watch_info_p watchInfo,
                                  timelib64_t monitorIntervalUsec,
                                  double maxAgeUsec,
                                  int maxKeepSamples,
                                  DcgmWatcher watcher);

    /**
     * Return the current time in microseconds since 1970.
     *
     * @returns: Current timestamp from the production time source.
     *
     * Thread-safe when the underlying time source is thread-safe.
     */
    timelib64_t Now();

    /**
     * Return the DCGM string representation of an NVML error for buffered string fields.
     *
     * @param[in] nvmlReturn: NVML status to translate.
     * @returns: Pointer to a NUL-terminated error string valid for the lifetime of the call.
     *
     * Thread-safe. Does not access shared mutable state.
     */
    char const *NvmlErrorToStringValue(nvmlReturn_t nvmlReturn);

    /**
     * Return the DCGM integer representation of an NVML error for buffered integer fields.
     *
     * @param[in] nvmlReturn: NVML status to translate.
     * @returns: Encoded integer value representing the NVML error.
     *
     * Thread-safe. Does not access shared mutable state.
     */
    long long NvmlErrorToInt64Value(nvmlReturn_t nvmlReturn);

    /**
     * Return a human-readable NVML error string for logging.
     *
     * @param[in,out] nvmlDriver: NVML task runner that supplies error strings.
     * @param[in] nvmlReturn: NVML status to describe.
     * @returns: Pointer to a NUL-terminated description valid for the lifetime of the call.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver.
     */
    char const *NvmlErrorString(NvmlTaskRunner &nvmlDriver, nvmlReturn_t nvmlReturn);

    /**
     * Query the VM identifier associated with a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] buffer: Buffer that receives the VM identifier on success.
     * @param[in] bufferSize: Size of buffer in bytes.
     * @param[out] vmIdType: VM identifier type reported by NVML.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetVmID(NvmlTaskRunner &nvmlDriver,
                                         SafeVgpuInstance vgpuId,
                                         char *buffer,
                                         unsigned int bufferSize,
                                         nvmlVgpuVmIdType_t *vmIdType);

    /**
     * Query the vGPU type identifier for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] vgpuTypeId: vGPU type identifier on success.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetType(NvmlTaskRunner &nvmlDriver, SafeVgpuInstance vgpuId, unsigned int *vgpuTypeId);

    /**
     * Query the UUID for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] buffer: Buffer that receives the UUID on success.
     * @param[in] bufferSize: Size of buffer in bytes.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetUUID(NvmlTaskRunner &nvmlDriver,
                                         SafeVgpuInstance vgpuId,
                                         char *buffer,
                                         unsigned int bufferSize);

    /**
     * Query the VM driver version running on a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] buffer: Buffer that receives the version string on success.
     * @param[in] bufferSize: Size of buffer in bytes.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetVmDriverVersion(NvmlTaskRunner &nvmlDriver,
                                                    SafeVgpuInstance vgpuId,
                                                    char *buffer,
                                                    unsigned int bufferSize);

    /**
     * Query framebuffer usage for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] fbUsage: Framebuffer usage in bytes on success.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetFbUsage(NvmlTaskRunner &nvmlDriver,
                                            SafeVgpuInstance vgpuId,
                                            unsigned long long *fbUsage);

    /**
     * Query vGPU license information for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] licenseInfo: License state and expiry details on success.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetLicenseInfo_v2(NvmlTaskRunner &nvmlDriver,
                                                   SafeVgpuInstance vgpuId,
                                                   nvmlVgpuLicenseInfo_t *licenseInfo);

    /**
     * Query the frame-rate limit for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] frameRateLimit: Frame-rate limit in frames per second on success.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetFrameRateLimit(NvmlTaskRunner &nvmlDriver,
                                                   SafeVgpuInstance vgpuId,
                                                   unsigned int *frameRateLimit);

    /**
     * Query the PCI bus identifier for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] vgpuPciId: Buffer that receives the PCI identifier on success.
     * @param[in,out] length: Input buffer capacity; output length of the identifier on success.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetGpuPciId(NvmlTaskRunner &nvmlDriver,
                                             SafeVgpuInstance vgpuId,
                                             char *vgpuPciId,
                                             unsigned int *length);

    /**
     * Query encoder statistics for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] sessionCount: Number of active encoder sessions on success.
     * @param[out] averageFps: Average frames per second on success.
     * @param[out] averageLatency: Average encoder latency on success.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetEncoderStats(NvmlTaskRunner &nvmlDriver,
                                                 SafeVgpuInstance vgpuId,
                                                 unsigned int *sessionCount,
                                                 unsigned int *averageFps,
                                                 unsigned int *averageLatency);

    /**
     * Query active encoder sessions for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] sessionCount: Number of active sessions on return.
     * @param[in,out] sessionInfo: Optional array of length *sessionCount to receive session details.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetEncoderSessions(NvmlTaskRunner &nvmlDriver,
                                                    SafeVgpuInstance vgpuId,
                                                    unsigned int *sessionCount,
                                                    nvmlEncoderSessionInfo_t *sessionInfo);

    /**
     * Query FBC statistics for a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] fbcStats: FBC session statistics on success.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetFBCStats(NvmlTaskRunner &nvmlDriver,
                                             SafeVgpuInstance vgpuId,
                                             nvmlFBCStats_t *fbcStats);

    /**
     * Query the GPU instance identifier backing a vGPU instance.
     *
     * @param[in,out] nvmlDriver: NVML task runner used for the query.
     * @param[in] vgpuId: Safe vGPU instance handle for the target vGPU.
     * @param[out] gpuInstanceId: GPU instance identifier on success.
     * @returns: NVML_SUCCESS on success, or an NVML error code from the driver.
     *
     * Not thread-safe. Caller must serialize access to nvmlDriver and output parameters.
     */
    nvmlReturn_t NvmlVgpuInstanceGetGpuInstanceId(NvmlTaskRunner &nvmlDriver,
                                                  SafeVgpuInstance vgpuId,
                                                  unsigned int *gpuInstanceId);
};

/**
 * Default implementations of all vGPU dependency operations used in dcgmlib.
 */
struct DcgmVgpuOps
    : FbcSessionOps
    , VgpuFieldValueOps
{};
