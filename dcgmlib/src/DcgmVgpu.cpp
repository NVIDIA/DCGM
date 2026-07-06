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

#include "DcgmVgpu.hpp"
#include "DcgmVgpuOps.hpp"
#include "NvmlTaskRunner.hpp"

#include <DcgmCMUtils.h>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string_view>

namespace DcgmVgpuDetail
{
namespace detail
{

    inline constexpr std::string_view ConvertNvmlGridLicenseStateToString(unsigned int licenseState)
    {
        switch (licenseState)
        {
            case NVML_GRID_LICENSE_STATE_UNLICENSED_UNRESTRICTED:
                return "Unlicensed (Unrestricted)";
            case NVML_GRID_LICENSE_STATE_UNLICENSED_RESTRICTED:
                return "Unlicensed (Restricted)";
            case NVML_GRID_LICENSE_STATE_UNINITIALIZED:
            case NVML_GRID_LICENSE_STATE_UNLICENSED:
                return "Unlicensed";
            case NVML_GRID_LICENSE_STATE_LICENSED:
                return "Licensed";
            default:
                return "Unknown";
        }
    }

    inline constexpr std::string_view ConvertNvmlLicenseExpiryStatusToString(unsigned int licenseExpiry)
    {
        switch (licenseExpiry)
        {
            case NVML_GRID_LICENSE_EXPIRY_INVALID:
                return "ERR!";
            case NVML_GRID_LICENSE_EXPIRY_NOT_APPLICABLE:
                return "N/A";
            case NVML_GRID_LICENSE_EXPIRY_PERMANENT:
                return "Permanent";
            case NVML_GRID_LICENSE_EXPIRY_VALID:
                return "";
            default:
                return "Not Available";
        }
    }

} // namespace detail

dcgmReturn_t GetDeviceFBCSessionsInfo(DcgmCacheManager &cm,
                                      NvmlTaskRunner &nvmlDriver,
                                      SafeNvmlHandle safeNvmlDevice,
                                      dcgmcm_update_thread_t &threadCtx,
                                      dcgmcm_watch_info_p watchInfo,
                                      timelib64_t now,
                                      timelib64_t expireTime,
                                      FbcSessionOps &deps)
{
    auto *devFbcSessions = static_cast<dcgmDeviceFbcSessions_t *>(deps.Malloc(sizeof(dcgmDeviceFbcSessions_t)));
    if (devFbcSessions == nullptr)
    {
        log_error("malloc of {} bytes failed", static_cast<int>(sizeof(dcgmDeviceFbcSessions_t)));
        return DCGM_ST_MEMORY;
    }

    unsigned int sessionCount = 0;
    nvmlReturn_t nvmlReturn   = deps.NvmlDeviceGetFBCSessions(nvmlDriver, safeNvmlDevice, &sessionCount, nullptr);
    if (watchInfo != nullptr)
    {
        watchInfo->lastStatus = nvmlReturn;
    }

    if (nvmlReturn != NVML_SUCCESS || sessionCount == 0)
    {
        devFbcSessions->version      = dcgmDeviceFbcSessions_version;
        devFbcSessions->sessionCount = 0;
        int payloadSize              = sizeof(*devFbcSessions) - sizeof(devFbcSessions->sessionInfo);
        deps.AppendEntityBlob(cm, threadCtx, devFbcSessions, payloadSize, now, expireTime);
        deps.Free(devFbcSessions);
        return deps.NvmlReturnToDcgmReturn(nvmlReturn);
    }

    auto *sessionInfo = static_cast<nvmlFBCSessionInfo_t *>(deps.Malloc(sizeof(nvmlFBCSessionInfo_t) * sessionCount));
    if (sessionInfo == nullptr)
    {
        log_error("malloc of {} bytes failed", static_cast<int>(sizeof(nvmlFBCSessionInfo_t) * sessionCount));
        deps.Free(devFbcSessions);
        return DCGM_ST_MEMORY;
    }

    nvmlReturn = deps.NvmlDeviceGetFBCSessions(nvmlDriver, safeNvmlDevice, &sessionCount, sessionInfo);
    if (watchInfo != nullptr)
    {
        watchInfo->lastStatus = nvmlReturn;
    }
    if (nvmlReturn != NVML_SUCCESS)
    {
        log_error("nvmlDeviceGetFBCSessions failed with status {}", static_cast<int>(nvmlReturn));
        deps.Free(sessionInfo);
        deps.Free(devFbcSessions);
        return deps.NvmlReturnToDcgmReturn(nvmlReturn);
    }

    devFbcSessions->version      = dcgmDeviceFbcSessions_version;
    devFbcSessions->sessionCount = sessionCount;

    for (unsigned int i = 0; i < sessionCount; i++)
    {
        if (devFbcSessions->sessionCount >= DCGM_MAX_FBC_SESSIONS)
            break;

        devFbcSessions->sessionInfo[i].version        = dcgmDeviceFbcSessionInfo_version;
        devFbcSessions->sessionInfo[i].vgpuId         = sessionInfo[i].vgpuInstance;
        devFbcSessions->sessionInfo[i].sessionId      = sessionInfo[i].sessionId;
        devFbcSessions->sessionInfo[i].pid            = sessionInfo[i].pid;
        devFbcSessions->sessionInfo[i].displayOrdinal = sessionInfo[i].displayOrdinal;
        devFbcSessions->sessionInfo[i].sessionType    = static_cast<dcgmFBCSessionType_t>(sessionInfo[i].sessionType);
        devFbcSessions->sessionInfo[i].sessionFlags   = sessionInfo[i].sessionFlags;
        devFbcSessions->sessionInfo[i].hMaxResolution = sessionInfo[i].hMaxResolution;
        devFbcSessions->sessionInfo[i].vMaxResolution = sessionInfo[i].vMaxResolution;
        devFbcSessions->sessionInfo[i].hResolution    = sessionInfo[i].hResolution;
        devFbcSessions->sessionInfo[i].vResolution    = sessionInfo[i].vResolution;
        devFbcSessions->sessionInfo[i].averageFps     = sessionInfo[i].averageFPS;
        devFbcSessions->sessionInfo[i].averageLatency = sessionInfo[i].averageLatency;
    }

    int payloadSize = (sizeof(*devFbcSessions) - sizeof(devFbcSessions->sessionInfo))
                      + (devFbcSessions->sessionCount * sizeof(devFbcSessions->sessionInfo[0]));

    deps.AppendEntityBlob(cm, threadCtx, devFbcSessions, payloadSize, now, expireTime);
    deps.Free(sessionInfo);
    deps.Free(devFbcSessions);
    return DCGM_ST_OK;
}

dcgmReturn_t GetVgpuInstanceFBCSessionsInfo(DcgmCacheManager &cm,
                                            NvmlTaskRunner &nvmlDriver,
                                            SafeVgpuInstance vgpuId,
                                            dcgmcm_update_thread_t &threadCtx,
                                            dcgmcm_watch_info_p watchInfo,
                                            timelib64_t now,
                                            timelib64_t expireTime,
                                            FbcSessionOps &deps)
{
    auto *vgpuFbcSessions = static_cast<dcgmDeviceFbcSessions_t *>(deps.Malloc(sizeof(dcgmDeviceFbcSessions_t)));
    if (vgpuFbcSessions == nullptr)
    {
        log_error("malloc of {} bytes failed", static_cast<int>(sizeof(dcgmDeviceFbcSessions_t)));
        return DCGM_ST_MEMORY;
    }

    unsigned int sessionCount = 0;
    nvmlReturn_t nvmlReturn   = deps.NvmlVgpuInstanceGetFBCSessions(nvmlDriver, vgpuId, &sessionCount, nullptr);
    if (watchInfo != nullptr)
    {
        watchInfo->lastStatus = nvmlReturn;
    }

    if (nvmlReturn != NVML_SUCCESS || sessionCount == 0)
    {
        vgpuFbcSessions->version      = dcgmDeviceFbcSessions_version;
        vgpuFbcSessions->sessionCount = 0;
        int payloadSize               = sizeof(*vgpuFbcSessions) - sizeof(vgpuFbcSessions->sessionInfo);
        deps.AppendEntityBlob(cm, threadCtx, vgpuFbcSessions, payloadSize, now, expireTime);
        deps.Free(vgpuFbcSessions);
        return deps.NvmlReturnToDcgmReturn(nvmlReturn);
    }

    auto *sessionInfo = static_cast<nvmlFBCSessionInfo_t *>(deps.Malloc(sizeof(nvmlFBCSessionInfo_t) * sessionCount));
    if (sessionInfo == nullptr)
    {
        log_error("malloc of {} bytes failed", static_cast<int>(sizeof(nvmlFBCSessionInfo_t) * sessionCount));
        deps.Free(vgpuFbcSessions);
        return DCGM_ST_MEMORY;
    }

    nvmlReturn = deps.NvmlVgpuInstanceGetFBCSessions(nvmlDriver, vgpuId, &sessionCount, sessionInfo);
    if (watchInfo != nullptr)
    {
        watchInfo->lastStatus = nvmlReturn;
    }
    if (nvmlReturn != NVML_SUCCESS)
    {
        log_error("nvmlVgpuInstanceGetFBCSessions failed with status {} for vgpuId {}",
                  static_cast<int>(nvmlReturn),
                  vgpuId.vgpuInstance);
        deps.Free(sessionInfo);
        deps.Free(vgpuFbcSessions);
        return deps.NvmlReturnToDcgmReturn(nvmlReturn);
    }

    vgpuFbcSessions->version      = dcgmDeviceFbcSessions_version;
    vgpuFbcSessions->sessionCount = sessionCount;

    for (unsigned int i = 0; i < sessionCount; i++)
    {
        if (vgpuFbcSessions->sessionCount >= DCGM_MAX_FBC_SESSIONS)
            break;

        vgpuFbcSessions->sessionInfo[i].version        = dcgmDeviceFbcSessionInfo_version;
        vgpuFbcSessions->sessionInfo[i].vgpuId         = sessionInfo[i].vgpuInstance;
        vgpuFbcSessions->sessionInfo[i].sessionId      = sessionInfo[i].sessionId;
        vgpuFbcSessions->sessionInfo[i].pid            = sessionInfo[i].pid;
        vgpuFbcSessions->sessionInfo[i].displayOrdinal = sessionInfo[i].displayOrdinal;
        vgpuFbcSessions->sessionInfo[i].sessionType    = static_cast<dcgmFBCSessionType_t>(sessionInfo[i].sessionType);
        vgpuFbcSessions->sessionInfo[i].sessionFlags   = sessionInfo[i].sessionFlags;
        vgpuFbcSessions->sessionInfo[i].hMaxResolution = sessionInfo[i].hMaxResolution;
        vgpuFbcSessions->sessionInfo[i].vMaxResolution = sessionInfo[i].vMaxResolution;
        vgpuFbcSessions->sessionInfo[i].hResolution    = sessionInfo[i].hResolution;
        vgpuFbcSessions->sessionInfo[i].vResolution    = sessionInfo[i].vResolution;
        vgpuFbcSessions->sessionInfo[i].averageFps     = sessionInfo[i].averageFPS;
        vgpuFbcSessions->sessionInfo[i].averageLatency = sessionInfo[i].averageLatency;
    }

    int payloadSize = (sizeof(*vgpuFbcSessions) - sizeof(vgpuFbcSessions->sessionInfo))
                      + (vgpuFbcSessions->sessionCount * sizeof(vgpuFbcSessions->sessionInfo[0]));

    deps.AppendEntityBlob(cm, threadCtx, vgpuFbcSessions, payloadSize, now, expireTime);
    deps.Free(sessionInfo);
    deps.Free(vgpuFbcSessions);
    return DCGM_ST_OK;
}

dcgmReturn_t BufferOrCacheLatestVgpuValue(DcgmCacheManager &cm,
                                          dcgmcm_update_thread_t &threadCtx,
                                          NvmlTaskRunner &nvmlDriver,
                                          SafeVgpuInstance vgpuId,
                                          dcgm_field_meta_p fieldMeta,
                                          VgpuFieldValueOps &deps,
                                          FbcSessionOps &fbcOps)
{
    timelib64_t now, expireTime;
    nvmlReturn_t nvmlReturn;

    if (!fieldMeta || fieldMeta->scope != DCGM_FS_DEVICE)
        return DCGM_ST_BADPARAM;

    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    now = deps.Now();

    /* Expiration is either measured in absolute time or 0 */
    expireTime = 0;
    if (watchInfo && watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Set without lock before we possibly return on error so we don't get in a hot
     * polling loop on something that is unsupported or errors. Not getting the lock
     * ahead of time because we don't want to hold the lock across a driver call that
     * could be long */
    if (watchInfo)
        watchInfo->lastQueriedUsec = now;

    switch (fieldMeta->fieldId)
    {
        case DCGM_FI_DEV_VGPU_VM_ID:
        case DCGM_FI_DEV_VGPU_VM_NAME:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;
            nvmlVgpuVmIdType_t vmIdType;

            nvmlReturn = deps.NvmlVgpuInstanceGetVmID(nvmlDriver, vgpuId, buffer, bufferSize, &vmIdType);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                deps.AppendEntityString(cm, threadCtx, deps.NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (fieldMeta->fieldId == DCGM_FI_DEV_VGPU_VM_ID)
            {
                deps.AppendEntityString(cm, threadCtx, buffer, now, expireTime);
            }
            else
            {
#if defined(NV_VMWARE)
                /* Command executed is specific to VMware */
                FILE *fp;
                char cmd[156], tmp_name[DCGM_DEVICE_UUID_BUFFER_SIZE];
                snprintf(cmd,
                         sizeof(cmd),
                         "localcli vm process list | grep \"World ID: %s\" -B 1 | head -1 | cut -f1 -d ':'",
                         buffer);

                if (strlen(cmd) == 0)
                    return DCGM_ST_NO_DATA;

                if (NULL == (fp = popen(cmd, "r")))
                {
                    nvmlReturn = NVML_ERROR_NOT_FOUND;
                    deps.AppendEntityString(cm, threadCtx, deps.NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                    return deps.NvmlReturnToDcgmReturn(nvmlReturn);
                }
                if (fgets(tmp_name, sizeof(tmp_name), fp))
                {
                    char *eol = strchr(tmp_name, '\n');
                    if (eol)
                        *eol = 0;
                    deps.AppendEntityString(cm, threadCtx, tmp_name, now, expireTime);
                }
                else
                {
                    nvmlReturn = NVML_ERROR_NOT_FOUND;
                    deps.AppendEntityString(cm, threadCtx, deps.NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                }
                pclose(fp);
#else
                /* Soon to be implemented for other environments. Appending error string for now. */
                nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
                deps.AppendEntityString(cm, threadCtx, deps.NvmlErrorToStringValue(nvmlReturn), now, expireTime);
#endif
            }
            break;
        }

        case DCGM_FI_DEV_VGPU_TYPE:
        {
            unsigned int vgpuTypeId = 0;

            nvmlReturn = deps.NvmlVgpuInstanceGetType(nvmlDriver, vgpuId, &vgpuTypeId);
            if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("nvmlVgpuInstanceGetType failed with status {} for vgpuId {}",
                          (int)nvmlReturn,
                          vgpuId.vgpuInstance);
                deps.AppendEntityInt64(cm, threadCtx, deps.NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }
            deps.AppendEntityInt64(cm, threadCtx, vgpuTypeId, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_UUID:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;

            nvmlReturn = deps.NvmlVgpuInstanceGetUUID(nvmlDriver, vgpuId, buffer, bufferSize);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("nvmlVgpuInstanceGetUUID failed with status {} for vgpuId {}",
                          (int)nvmlReturn,
                          vgpuId.vgpuInstance);
                deps.AppendEntityString(cm, threadCtx, deps.NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }
            deps.AppendEntityString(cm, threadCtx, buffer, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_DRIVER_VERSION:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;

            nvmlReturn = deps.NvmlVgpuInstanceGetVmDriverVersion(nvmlDriver, vgpuId, buffer, bufferSize);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("nvmlVgpuInstanceGetVmDriverVersion failed with status {} for vgpuId {}",
                          (int)nvmlReturn,
                          vgpuId.vgpuInstance);
                deps.AppendEntityString(cm, threadCtx, deps.NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (strcmp("Not Available", buffer))
            {
                /* Updating the cache frequency to once every 15 minutes after a known driver version is fetched. */
                if (watchInfo && watchInfo->monitorIntervalUsec != 900000000)
                {
                    dcgmReturn_t status = deps.UpdateFieldWatch(
                        cm, watchInfo, 900000000, 900.0, 1, DcgmWatcher(DcgmWatcherTypeCacheManager));
                    if (DCGM_ST_OK != status)
                    {
                        DCGM_LOG_ERROR << "UpdateFieldWatch failed for vgpuId " << vgpuId.vgpuInstance
                                       << " and fieldId " << fieldMeta->fieldId;
                        return status;
                    }
                }
            }

            deps.AppendEntityString(cm, threadCtx, buffer, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_MEMORY_USAGE:
        {
            unsigned long long fbUsage;

            nvmlReturn = deps.NvmlVgpuInstanceGetFbUsage(nvmlDriver, vgpuId, &fbUsage);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                log_error("nvmlVgpuInstanceGetFbUsage failed with status {} for vgpuId {}",
                          (int)nvmlReturn,
                          vgpuId.vgpuInstance);
                deps.AppendEntityInt64(cm, threadCtx, deps.NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }
            fbUsage = fbUsage / (1024 * 1024);
            deps.AppendEntityInt64(cm, threadCtx, fbUsage, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATUS:
        {
            nvmlVgpuLicenseInfo_t licenseInfo;
            char licenseState[NVML_GRID_LICENSE_BUFFER_SIZE]  = { 0 };
            char licenseExpiry[NVML_GRID_LICENSE_BUFFER_SIZE] = { 0 };

            nvmlReturn = deps.NvmlVgpuInstanceGetLicenseInfo_v2(nvmlDriver, vgpuId, &licenseInfo);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }
            if (NVML_SUCCESS != nvmlReturn)
            {
                if (nvmlReturn != NVML_ERROR_DRIVER_NOT_LOADED)
                {
                    log_error("nvmlVgpuInstanceGetLicenseInfo_v2 for vgpuId {} failed with error: ({}) {}",
                              vgpuId.vgpuInstance,
                              nvmlReturn,
                              deps.NvmlErrorString(nvmlDriver, nvmlReturn));
                }
                SafeCopyTo(licenseState, (char const *)"Not Available");
                deps.AppendEntityString(cm, threadCtx, licenseState, now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (licenseInfo.currentState == NVML_GRID_LICENSE_STATE_LICENSED)
            {
                if (licenseInfo.licenseExpiry.status == NVML_GRID_LICENSE_EXPIRY_VALID)
                {
                    auto const &expInfo = licenseInfo.licenseExpiry;
                    using namespace fmt::literals;
                    SafeCopyTo(licenseExpiry,
                               fmt::format(" (Expiry: {year}-{month}-{day} {hour}:{min}:{sec} GMT)",
                                           "year"_a  = expInfo.year,
                                           "month"_a = expInfo.month,
                                           "day"_a   = expInfo.day,
                                           "hour"_a  = expInfo.hour,
                                           "min"_a   = expInfo.min,
                                           "sec"_a   = expInfo.sec)
                                   .c_str());
                }
                else
                {
                    SafeCopyTo(
                        licenseExpiry,
                        fmt::format(" (Expiry: {})",
                                    detail::ConvertNvmlLicenseExpiryStatusToString(licenseInfo.licenseExpiry.status))
                            .c_str());
                }

                SafeCopyTo(licenseState,
                           fmt::format("{}{}",
                                       detail::ConvertNvmlGridLicenseStateToString(licenseInfo.currentState),
                                       licenseExpiry)
                               .c_str());

                /* Updating the cache frequency to once every 20 seconds when VM is licensed. */
                if ((watchInfo != nullptr) && watchInfo->monitorIntervalUsec != 20000000)
                {
                    dcgmReturn_t status = deps.UpdateFieldWatch(
                        cm, watchInfo, 20000000, 20.0, 1, DcgmWatcher(DcgmWatcherTypeCacheManager));
                    if (DCGM_ST_OK != status)
                    {
                        log_error("UpdateFieldWatch failed for vgpuId {} and fieldId {}. Error: ({}){}",
                                  vgpuId.vgpuInstance,
                                  fieldMeta->fieldId,
                                  status,
                                  errorString(status));
                        return status;
                    }
                }
            }
            else
            {
                SafeCopyTo(licenseState, detail::ConvertNvmlGridLicenseStateToString(licenseInfo.currentState).data());

                if ((watchInfo != nullptr) && watchInfo->monitorIntervalUsec != 1000000)
                {
                    /* Updating the cache frequency to once every 1 sec, when VM is unlicensed and current caching
                     * frequency is 20 sec. */
                    dcgmReturn_t status = deps.UpdateFieldWatch(
                        cm, watchInfo, 1000000, 600.0, 600, DcgmWatcher(DcgmWatcherTypeCacheManager));
                    if (DCGM_ST_OK != status)
                    {
                        log_error("UpdateFieldWatch failed for vgpuId {} and fieldId {}. Error: ({}){}",
                                  vgpuId.vgpuInstance,
                                  fieldMeta->fieldId,
                                  status,
                                  errorString(status));
                        return status;
                    }
                }
            }

            deps.AppendEntityString(cm, threadCtx, licenseState, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT:
        {
            unsigned int frameRateLimit;

            nvmlReturn = deps.NvmlVgpuInstanceGetFrameRateLimit(nvmlDriver, vgpuId, &frameRateLimit);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }
            if (NVML_SUCCESS != nvmlReturn)
            {
                // Not Supported returned, if vGPU scheduler is enabled. Don't log.
                if (nvmlReturn != NVML_ERROR_NOT_SUPPORTED)
                {
                    log_error("nvmlVgpuInstanceGetFrameRateLimit for vgpuId {} failed with error: ({}) {}",
                              vgpuId.vgpuInstance,
                              nvmlReturn,
                              deps.NvmlErrorString(nvmlDriver, nvmlReturn));
                }
                deps.AppendEntityInt64(cm, threadCtx, deps.NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }
            deps.AppendEntityInt64(cm, threadCtx, frameRateLimit, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_PCI_ID:
        {
            char vgpuPciId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
            unsigned int length = NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE;

            nvmlReturn = deps.NvmlVgpuInstanceGetGpuPciId(nvmlDriver, vgpuId, vgpuPciId, &length);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }
            if (NVML_SUCCESS != nvmlReturn && NVML_ERROR_DRIVER_NOT_LOADED != nvmlReturn)
            {
                DCGM_LOG_ERROR << fmt::format("nvmlVgpuInstanceGetGpuPciId for vgpuId {} failed with error: ({}) {}",
                                              vgpuId.vgpuInstance,
                                              nvmlReturn,
                                              deps.NvmlErrorString(nvmlDriver, nvmlReturn));
                deps.AppendEntityString(cm, threadCtx, deps.NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Updating the cache frequency to once every 60 minutes after a valid vGPU PCI Id is fetched. */
            if ((strcmp("00000000:00:00.0", vgpuPciId) != 0)
                && (watchInfo && watchInfo->monitorIntervalUsec != 3600000000))
            {
                dcgmReturn_t status = deps.UpdateFieldWatch(
                    cm, watchInfo, 3600000000, 3600.0, 1, DcgmWatcher(DcgmWatcherTypeCacheManager));
                if (DCGM_ST_OK != status)
                {
                    DCGM_LOG_ERROR << fmt::format("UpdateFieldWatch failed for vgpuId {} and fieldId {}. Error: ({}){}",
                                                  vgpuId.vgpuInstance,
                                                  fieldMeta->fieldId,
                                                  status,
                                                  errorString(status));
                    return status;
                }
            }

            deps.AppendEntityString(cm, threadCtx, vgpuPciId, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_ENC_STATS:
        {
            dcgmDeviceEncStats_t vgpuEncStats;
            vgpuEncStats.version = dcgmDeviceEncStats_version;

            nvmlReturn = deps.NvmlVgpuInstanceGetEncoderStats(
                nvmlDriver, vgpuId, &vgpuEncStats.sessionCount, &vgpuEncStats.averageFps, &vgpuEncStats.averageLatency);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                memset(&vgpuEncStats, 0, sizeof(vgpuEncStats));
                deps.AppendEntityBlob(cm, threadCtx, &vgpuEncStats, (int)(sizeof(vgpuEncStats)), now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }
            deps.AppendEntityBlob(cm, threadCtx, &vgpuEncStats, (int)(sizeof(vgpuEncStats)), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO:
        {
            std::unique_ptr<dcgmDeviceVgpuEncSessions_t[]> vgpuEncSessionsInfo;
            std::unique_ptr<nvmlEncoderSessionInfo_t[]> sessionInfo;
            unsigned int sessionCount = 0;

            nvmlReturn = deps.NvmlVgpuInstanceGetEncoderSessions(nvmlDriver, vgpuId, &sessionCount, nullptr);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            sessionInfo         = std::make_unique<nvmlEncoderSessionInfo_t[]>(sessionCount);
            vgpuEncSessionsInfo = std::make_unique<dcgmDeviceVgpuEncSessions_t[]>(sessionCount + 1);

            /* Initialize the first session object since the code below only updates index 1 and beyond */
            memset(vgpuEncSessionsInfo.get(), 0, sizeof(nvmlEncoderSessionInfo_t));
            vgpuEncSessionsInfo[0].version = dcgmDeviceVgpuEncSessions_version;

            if (nvmlReturn != NVML_SUCCESS)
            {
                vgpuEncSessionsInfo[0].encoderSessionInfo.sessionCount = 0;
                deps.AppendEntityBlob(cm,
                                      threadCtx,
                                      &vgpuEncSessionsInfo[0],
                                      (int)(sizeof(dcgmDeviceVgpuEncSessions_t)),
                                      now,
                                      expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (sessionCount != 0)
            {
                nvmlReturn
                    = deps.NvmlVgpuInstanceGetEncoderSessions(nvmlDriver, vgpuId, &sessionCount, sessionInfo.get());
                if (watchInfo != nullptr)
                {
                    watchInfo->lastStatus = nvmlReturn;
                }
                if (nvmlReturn != NVML_SUCCESS)
                {
                    DCGM_LOG_ERROR << fmt::format(
                        "nvmlVgpuInstanceGetEncoderSessions failed for vgpuId {} with status ({}){}",
                        vgpuId.vgpuInstance,
                        nvmlReturn,
                        deps.NvmlErrorString(nvmlDriver, nvmlReturn));
                    return deps.NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            /* First element of the array holds the count */
            vgpuEncSessionsInfo[0].encoderSessionInfo.sessionCount = sessionCount;

            for (unsigned int i = 0; i < sessionCount; i++)
            {
                auto &encInfo    = vgpuEncSessionsInfo[i + 1];
                auto const &info = sessionInfo[i];

                encInfo.version                   = dcgmDeviceVgpuEncSessions_version;
                encInfo.encoderSessionInfo.vgpuId = info.vgpuInstance;
                encInfo.sessionId                 = info.sessionId;
                encInfo.pid                       = info.pid;
                encInfo.codecType                 = (dcgmEncoderType_t)info.codecType;
                encInfo.hResolution               = info.hResolution;
                encInfo.vResolution               = info.vResolution;
                encInfo.averageFps                = info.averageFps;
                encInfo.averageLatency            = info.averageLatency;
            }
            deps.AppendEntityBlob(cm,
                                  threadCtx,
                                  vgpuEncSessionsInfo.get(),
                                  (int)(sizeof(dcgmDeviceVgpuEncSessions_t) * (sessionCount + 1)),
                                  now,
                                  expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_FBC_STATS:
        {
            dcgmDeviceFbcStats_t vgpuFbcStats;
            nvmlFBCStats_t fbcStats;

            nvmlReturn = deps.NvmlVgpuInstanceGetFBCStats(nvmlDriver, vgpuId, &fbcStats);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                log_error("nvmlVgpuInstanceGetFBCStats failed with status {} for vgpuId {}",
                          (int)nvmlReturn,
                          vgpuId.vgpuInstance);
                memset(&vgpuFbcStats, 0, sizeof(vgpuFbcStats));
                deps.AppendEntityBlob(cm, threadCtx, &vgpuFbcStats, (int)(sizeof(vgpuFbcStats)), now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }

            vgpuFbcStats.version        = dcgmDeviceFbcStats_version;
            vgpuFbcStats.sessionCount   = fbcStats.sessionsCount;
            vgpuFbcStats.averageFps     = fbcStats.averageFPS;
            vgpuFbcStats.averageLatency = fbcStats.averageLatency;

            deps.AppendEntityBlob(cm, threadCtx, &vgpuFbcStats, (int)(sizeof(vgpuFbcStats)), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO:
        {
            dcgmReturn_t status = DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
                cm, nvmlDriver, vgpuId, threadCtx, watchInfo, now, expireTime, fbcOps);
            if (DCGM_ST_OK != status)
                return status;
            break;
        }

        case DCGM_FI_DEV_VGPU_GPU_INSTANCE_ID:
        {
            unsigned int gpuInstanceId = INVALID_GPU_INSTANCE_ID;

            nvmlReturn = deps.NvmlVgpuInstanceGetGpuInstanceId(nvmlDriver, vgpuId, &gpuInstanceId);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }
            if (NVML_SUCCESS != nvmlReturn)
            {
                DCGM_LOG_ERROR << fmt::format(
                    "nvmlVgpuInstanceGetGpuInstanceId for vgpuId {} failed with error: ({}) {}",
                    vgpuId.vgpuInstance,
                    nvmlReturn,
                    deps.NvmlErrorString(nvmlDriver, nvmlReturn));
                deps.AppendEntityInt64(cm, threadCtx, deps.NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return deps.NvmlReturnToDcgmReturn(nvmlReturn);
            }
            deps.AppendEntityInt64(cm, threadCtx, gpuInstanceId, 0, now, expireTime);
            break;
        }

        default:
            log_warning("Unimplemented fieldId: {}", (int)fieldMeta->fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}


} // namespace DcgmVgpuDetail

dcgmReturn_t GetDeviceFBCSessionsInfo(DcgmCacheManager &cm,
                                      NvmlTaskRunner &nvmlDriver,
                                      SafeNvmlHandle safeNvmlDevice,
                                      dcgmcm_update_thread_t &threadCtx,
                                      dcgmcm_watch_info_p watchInfo,
                                      timelib64_t now,
                                      timelib64_t expireTime)
{
    DcgmVgpuOps vgpuOps;
    DcgmVgpuDetail::FbcSessionOps deps(&vgpuOps);
    return DcgmVgpuDetail::GetDeviceFBCSessionsInfo(
        cm, nvmlDriver, safeNvmlDevice, threadCtx, watchInfo, now, expireTime, deps);
}

dcgmReturn_t GetVgpuInstanceFBCSessionsInfo(DcgmCacheManager &cm,
                                            NvmlTaskRunner &nvmlDriver,
                                            SafeVgpuInstance vgpuId,
                                            dcgmcm_update_thread_t &threadCtx,
                                            dcgmcm_watch_info_p watchInfo,
                                            timelib64_t now,
                                            timelib64_t expireTime)
{
    DcgmVgpuOps vgpuOps;
    DcgmVgpuDetail::FbcSessionOps deps(&vgpuOps);
    return DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
        cm, nvmlDriver, vgpuId, threadCtx, watchInfo, now, expireTime, deps);
}

/*****************************************************************************/
dcgmReturn_t BufferOrCacheLatestVgpuValue(DcgmCacheManager &cm,
                                          dcgmcm_update_thread_t &threadCtx,
                                          NvmlTaskRunner &nvmlDriver,
                                          SafeVgpuInstance vgpuId,
                                          dcgm_field_meta_p fieldMeta)
{
    DcgmVgpuOps vgpuOps;
    DcgmVgpuDetail::VgpuFieldValueOps ops(&vgpuOps);
    DcgmVgpuDetail::FbcSessionOps fbcOps(&vgpuOps);
    return DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(cm, threadCtx, nvmlDriver, vgpuId, fieldMeta, ops, fbcOps);
}
