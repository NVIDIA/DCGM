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

#include "DcgmVgpuOps.hpp"

#include <DcgmCMUtils.h>
#include <DcgmUtilities.h>
#include <cstdlib>

void *DcgmVgpuOpsCommon::Malloc(std::size_t size)
{
    return std::malloc(size);
}

void DcgmVgpuOpsCommon::Free(void *ptr)
{
    std::free(ptr);
}

dcgmReturn_t DcgmVgpuOpsCommon::AppendEntityBlob(DcgmCacheManager &cm,
                                                 dcgmcm_update_thread_t &threadCtx,
                                                 void *value,
                                                 int valueSize,
                                                 timelib64_t timestamp,
                                                 timelib64_t oldestKeepTimestamp)
{
    return cm.AppendEntityBlob(threadCtx, value, valueSize, timestamp, oldestKeepTimestamp);
}

dcgmReturn_t DcgmVgpuOpsCommon::NvmlReturnToDcgmReturn(nvmlReturn_t nvmlReturn)
{
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
}

nvmlReturn_t FbcSessionOps::NvmlDeviceGetFBCSessions(NvmlTaskRunner &nvmlDriver,
                                                     SafeNvmlHandle safeNvmlDevice,
                                                     unsigned int *sessionCount,
                                                     nvmlFBCSessionInfo_t *sessionInfo)
{
    return nvmlDriver.NvmlDeviceGetFBCSessions(safeNvmlDevice, sessionCount, sessionInfo);
}

nvmlReturn_t FbcSessionOps::NvmlVgpuInstanceGetFBCSessions(NvmlTaskRunner &nvmlDriver,
                                                           SafeVgpuInstance vgpuId,
                                                           unsigned int *sessionCount,
                                                           nvmlFBCSessionInfo_t *sessionInfo)
{
    return nvmlDriver.NvmlVgpuInstanceGetFBCSessions(vgpuId, sessionCount, sessionInfo);
}

dcgmReturn_t VgpuFieldValueOps::AppendEntityString(DcgmCacheManager &cm,
                                                   dcgmcm_update_thread_t &threadCtx,
                                                   char const *value,
                                                   timelib64_t timestamp,
                                                   timelib64_t oldestKeepTimestamp)
{
    return cm.AppendEntityString(threadCtx, value, timestamp, oldestKeepTimestamp);
}

dcgmReturn_t VgpuFieldValueOps::AppendEntityInt64(DcgmCacheManager &cm,
                                                  dcgmcm_update_thread_t &threadCtx,
                                                  long long value1,
                                                  long long value2,
                                                  timelib64_t timestamp,
                                                  timelib64_t oldestKeepTimestamp)
{
    return cm.AppendEntityInt64(threadCtx, value1, value2, timestamp, oldestKeepTimestamp);
}

dcgmReturn_t VgpuFieldValueOps::UpdateFieldWatch(DcgmCacheManager &cm,
                                                 dcgmcm_watch_info_p watchInfo,
                                                 timelib64_t monitorIntervalUsec,
                                                 double maxAgeUsec,
                                                 int maxKeepSamples,
                                                 DcgmWatcher watcher)
{
    return cm.UpdateFieldWatch(watchInfo, monitorIntervalUsec, maxAgeUsec, maxKeepSamples, watcher);
}

timelib64_t VgpuFieldValueOps::Now()
{
    return timelib_usecSince1970();
}

char const *VgpuFieldValueOps::NvmlErrorToStringValue(nvmlReturn_t nvmlReturn)
{
    return ::NvmlErrorToStringValue(nvmlReturn);
}

long long VgpuFieldValueOps::NvmlErrorToInt64Value(nvmlReturn_t nvmlReturn)
{
    return ::NvmlErrorToInt64Value(nvmlReturn);
}

char const *VgpuFieldValueOps::NvmlErrorString(NvmlTaskRunner &nvmlDriver, nvmlReturn_t nvmlReturn)
{
    return nvmlDriver.NvmlErrorString(nvmlReturn);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetVmID(NvmlTaskRunner &nvmlDriver,
                                                        SafeVgpuInstance vgpuId,
                                                        char *buffer,
                                                        unsigned int bufferSize,
                                                        nvmlVgpuVmIdType_t *vmIdType)
{
    return nvmlDriver.NvmlVgpuInstanceGetVmID(vgpuId, buffer, bufferSize, vmIdType);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetType(NvmlTaskRunner &nvmlDriver,
                                                        SafeVgpuInstance vgpuId,
                                                        unsigned int *vgpuTypeId)
{
    return nvmlDriver.NvmlVgpuInstanceGetType(vgpuId, vgpuTypeId);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetUUID(NvmlTaskRunner &nvmlDriver,
                                                        SafeVgpuInstance vgpuId,
                                                        char *buffer,
                                                        unsigned int bufferSize)
{
    return nvmlDriver.NvmlVgpuInstanceGetUUID(vgpuId, buffer, bufferSize);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetVmDriverVersion(NvmlTaskRunner &nvmlDriver,
                                                                   SafeVgpuInstance vgpuId,
                                                                   char *buffer,
                                                                   unsigned int bufferSize)
{
    return nvmlDriver.NvmlVgpuInstanceGetVmDriverVersion(vgpuId, buffer, bufferSize);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetFbUsage(NvmlTaskRunner &nvmlDriver,
                                                           SafeVgpuInstance vgpuId,
                                                           unsigned long long *fbUsage)
{
    return nvmlDriver.NvmlVgpuInstanceGetFbUsage(vgpuId, fbUsage);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetLicenseInfo_v2(NvmlTaskRunner &nvmlDriver,
                                                                  SafeVgpuInstance vgpuId,
                                                                  nvmlVgpuLicenseInfo_t *licenseInfo)
{
    return nvmlDriver.NvmlVgpuInstanceGetLicenseInfo_v2(vgpuId, licenseInfo);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetFrameRateLimit(NvmlTaskRunner &nvmlDriver,
                                                                  SafeVgpuInstance vgpuId,
                                                                  unsigned int *frameRateLimit)
{
    return nvmlDriver.NvmlVgpuInstanceGetFrameRateLimit(vgpuId, frameRateLimit);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetGpuPciId(NvmlTaskRunner &nvmlDriver,
                                                            SafeVgpuInstance vgpuId,
                                                            char *vgpuPciId,
                                                            unsigned int *length)
{
    return nvmlDriver.NvmlVgpuInstanceGetGpuPciId(vgpuId, vgpuPciId, length);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetEncoderStats(NvmlTaskRunner &nvmlDriver,
                                                                SafeVgpuInstance vgpuId,
                                                                unsigned int *sessionCount,
                                                                unsigned int *averageFps,
                                                                unsigned int *averageLatency)
{
    return nvmlDriver.NvmlVgpuInstanceGetEncoderStats(vgpuId, sessionCount, averageFps, averageLatency);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetEncoderSessions(NvmlTaskRunner &nvmlDriver,
                                                                   SafeVgpuInstance vgpuId,
                                                                   unsigned int *sessionCount,
                                                                   nvmlEncoderSessionInfo_t *sessionInfo)
{
    return nvmlDriver.NvmlVgpuInstanceGetEncoderSessions(vgpuId, sessionCount, sessionInfo);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetFBCStats(NvmlTaskRunner &nvmlDriver,
                                                            SafeVgpuInstance vgpuId,
                                                            nvmlFBCStats_t *fbcStats)
{
    return nvmlDriver.NvmlVgpuInstanceGetFBCStats(vgpuId, fbcStats);
}

nvmlReturn_t VgpuFieldValueOps::NvmlVgpuInstanceGetGpuInstanceId(NvmlTaskRunner &nvmlDriver,
                                                                 SafeVgpuInstance vgpuId,
                                                                 unsigned int *gpuInstanceId)
{
    return nvmlDriver.NvmlVgpuInstanceGetGpuInstanceId(vgpuId, gpuInstanceId);
}
