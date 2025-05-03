/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include "nvml.h"
#include "nvml_deprecated.h"

NVML_ENTRY_POINT(nvmlDeviceGetClockInfo, tsapiDeviceGetClockInfo,
        (nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock),
        "(%p, %d, %p)",
        device, type, clock)
NVML_ENTRY_POINT(nvmlDeviceGetMaxClockInfo, tsapiDeviceGetMaxClockInfo,
        (nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock),
        "(%p, %d, %p)",
        device, type, clock)
NVML_ENTRY_POINT(nvmlDeviceGetComputeMode, tsapiDeviceGetComputeMode,
        (nvmlDevice_t device, nvmlComputeMode_t *mode),
        "(%p, %p)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceSetComputeMode, tsapiDeviceSetComputeMode,
        (nvmlDevice_t device, nvmlComputeMode_t mode),
        "(%p, %d)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceGetCudaComputeCapability, tsapiDeviceGetCudaComputeCapability,
        (nvmlDevice_t device, int *major, int *minor),
        "(%p, %p, %p)",
        device, major, minor)
NVML_ENTRY_POINT(nvmlDeviceSetDriverModel, tsapiDeviceSetDriverModel,
        (nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags),
        "(%p, %d, 0x%x)",
        device, driverModel, flags)
NVML_ENTRY_POINT(nvmlDeviceGetDriverModel, tsapiDeviceGetDriverModel,
        (nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending),
        "(%p, %p, %p)",
        device, current, pending)
NVML_ENTRY_POINT(nvmlDeviceGetCount, tsapiDeviceGetCount,
        (unsigned int *deviceCount),
        "(%p)",
        deviceCount)
NVML_ENTRY_POINT(nvmlDeviceGetHandleByIndex, tsapiDeviceGetHandleByIndex,
        (unsigned int index, nvmlDevice_t *device),
        "(%d, %p)",
        index, device)
NVML_ENTRY_POINT(nvmlDeviceGetHandleBySerial, tsapiDeviceGetHandleBySerial,
        (const char *serial, nvmlDevice_t *device),
        "(%p, %p)",
        serial, device)
NVML_ENTRY_POINT(nvmlDeviceGetHandleByUUID, tsapiDeviceGetHandleByUUID,
        (const char *uuid, nvmlDevice_t *device),
        "(%p, %p)",
        uuid, device)
NVML_ENTRY_POINT(nvmlDeviceGetHandleByPciBusId, tsapiDeviceGetHandleByPciBusId,
        (const char *pciBusId, nvmlDevice_t *device),
        "(%p, %p)",
        pciBusId, device)
NVML_ENTRY_POINT(nvmlDeviceGetHandleByPciBusId_v2, tsapiDeviceGetHandleByPciBusId_v2,
        (const char *pciBusId, nvmlDevice_t *device),
        "(%p, %p)",
        pciBusId, device)
NVML_ENTRY_POINT(nvmlDeviceGetInforomVersion, tsapiDeviceGetInforomVersion,
        (nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length),
        "(%p, %d, %p, %d)",
        device, object, version, length)
NVML_ENTRY_POINT(nvmlDeviceGetInforomImageVersion, tsapiDeviceGetInforomImageVersion,
        (nvmlDevice_t device, char *version, unsigned int length),
        "(%p, %p, %d)",
        device, version, length)
NVML_ENTRY_POINT(nvmlDeviceGetDisplayMode, tsapiDeviceGetDisplayMode,
        (nvmlDevice_t device, nvmlEnableState_t *mode),
        "(%p, %p)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceGetEccMode, tsapiDeviceGetEccMode,
        (nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending),
        "(%p, %p, %p)",
        device, current, pending)
NVML_ENTRY_POINT(nvmlDeviceGetDefaultEccMode, tsapiDeviceGetDefaultEccMode,
        (nvmlDevice_t device, nvmlEnableState_t *defaultMode),
        "(%p, %p)",
        device, defaultMode)
NVML_ENTRY_POINT(nvmlDeviceGetBoardId, tsapiDeviceGetBoardId,
        (nvmlDevice_t device, unsigned int *boardId),
        "(%p, %p)",
        device, boardId)
NVML_ENTRY_POINT(nvmlDeviceGetMultiGpuBoard, tsapiDeviceGetMultiGpuBoard,
        (nvmlDevice_t device, unsigned int *multiGpuBool),
        "(%p, %p)",
        device, multiGpuBool)
NVML_ENTRY_POINT(nvmlDeviceGetDetailedEccErrors, tsapiDeviceGetDetailedEccErrors,
        (nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType,
         nvmlEccErrorCounts_t *eccCounts),
        "(%p, %d, %d, %p)",
        device, errorType, counterType, eccCounts)
NVML_ENTRY_POINT(nvmlDeviceGetTotalEccErrors, tsapiDeviceGetTotalEccErrors,
        (nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType,
         unsigned long long *eccCounts),
        "(%p, %d, %d, %p)",
        device, errorType, counterType, eccCounts)
NVML_ENTRY_POINT(nvmlDeviceSetEccMode, tsapiDeviceSetEccMode,
        (nvmlDevice_t device, nvmlEnableState_t ecc),
        "(%p, %d)",
        device, ecc)
NVML_ENTRY_POINT(nvmlDeviceClearEccErrorCounts, tsapiDeviceClearEccErrorCounts,
        (nvmlDevice_t device, nvmlEccCounterType_t counterType),
        "(%p, %d)",
        device, counterType)
NVML_ENTRY_POINT(nvmlDeviceGetName, tsapiDeviceGetName,
        (nvmlDevice_t device, char* name, unsigned int length),
        "(%p, %p, %d)",
        device, name, length)
NVML_ENTRY_POINT(nvmlDeviceGetBrand, tsapiDeviceGetBrand,
        (nvmlDevice_t device, nvmlBrandType_t *type),
        "(%p, %p)",
        device, type)
NVML_ENTRY_POINT(nvmlDeviceGetSerial, tsapiDeviceGetSerial,
        (nvmlDevice_t device, char* serial, unsigned int length),
        "(%p, %p, %d)",
        device, serial, length)
NVML_ENTRY_POINT(nvmlDeviceGetBoardPartNumber, tsapiDeviceGetBoardPartNumber,
        (nvmlDevice_t device, char * partNumber, unsigned int length),
        "(%p %p %d)",
        device, partNumber, length)
NVML_ENTRY_POINT(nvmlDeviceGetMemoryAffinity, tsapiDeviceGetMemoryAffinity,
        (nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope),
        "(%p, %d, %p, %d)",
        device, nodeSetSize, nodeSet, scope)
NVML_ENTRY_POINT(nvmlDeviceGetCpuAffinityWithinScope, tsapiDeviceGetCpuAffinityWithinScope,
        (nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope),
        "(%p, %d, %p, %d)",
        device, cpuSetSize, cpuSet, scope)
NVML_ENTRY_POINT(nvmlDeviceGetCpuAffinity, tsapiDeviceGetCpuAffinity,
        (nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet),
        "(%p, %d, %p)",
        device, cpuSetSize, cpuSet)
NVML_ENTRY_POINT(nvmlDeviceSetCpuAffinity, tsapiDeviceSetCpuAffinity,
        (nvmlDevice_t device),
        "(%p)",
        device)
NVML_ENTRY_POINT(nvmlDeviceClearCpuAffinity, tsapiDeviceClearCpuAffinity,
        (nvmlDevice_t device),
        "(%p)",
        device)
NVML_ENTRY_POINT(nvmlDeviceGetUUID, tsapiDeviceGetUUID,
        (nvmlDevice_t device, char *uuid, unsigned int length),
        "(%p, %p, %d)",
        device, uuid, length)
NVML_ENTRY_POINT(nvmlDeviceGetMemoryInfo, tsapiDeviceGetMemoryInfo,
        (nvmlDevice_t device, nvmlMemory_t *memory),
        "(%p, %p)",
        device, memory)
NVML_ENTRY_POINT(nvmlDeviceGetMemoryInfo_v2, tsapiDeviceGetMemoryInfo_v2,
        (nvmlDevice_t device, nvmlMemory_v2_t *memory),
        "(%p, %p)",
        device, memory)
NVML_ENTRY_POINT(nvmlDeviceGetPciInfo, tsapiDeviceGetPciInfo,
        (nvmlDevice_t device, nvmlPciInfo_t *pci),
        "(%p, %p)",
        device, pci)
NVML_ENTRY_POINT(nvmlDeviceGetPciInfo_v2, tsapiDeviceGetPciInfo_v2,
        (nvmlDevice_t device, nvmlPciInfo_t *pci),
        "(%p, %p)",
        device, pci)
NVML_ENTRY_POINT(nvmlDeviceGetPciInfo_v3, tsapiDeviceGetPciInfo_v3,
        (nvmlDevice_t device, nvmlPciInfo_t *pci),
        "(%p, %p)",
        device, pci)
NVML_ENTRY_POINT(nvmlDeviceGetPersistenceMode, tsapiDeviceGetPersistenceMode,
        (nvmlDevice_t device, nvmlEnableState_t *mode),
        "(%p, %p)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceSetPersistenceMode, tsapiDeviceSetPersistenceMode,
        (nvmlDevice_t device, nvmlEnableState_t mode),
        "(%p, %d)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceGetBAR1MemoryInfo, tsapiDeviceGetBAR1MemoryInfo,
        (nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory),
        "(%p %p)",
        device, bar1Memory)
NVML_ENTRY_POINT(nvmlDeviceGetViolationStatus, tsapiDeviceGetViolationStatus,
        (nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime),
        "(%p %d %p)",
        device, perfPolicyType, violTime)

// Internally these are the same function
NVML_ENTRY_POINT(nvmlDeviceGetPowerState, tsapiDeviceGetPerformanceState,
        (nvmlDevice_t device, nvmlPstates_t *pState),
        "(%p, %p)",
        device, pState)
NVML_ENTRY_POINT(nvmlDeviceGetPerformanceState, tsapiDeviceGetPerformanceState,
        (nvmlDevice_t device, nvmlPstates_t *pState),
        "(%p, %p)",
        device, pState)
        
NVML_ENTRY_POINT(nvmlDeviceGetPowerUsage, tsapiDeviceGetPowerUsage,
        (nvmlDevice_t device, unsigned int *power),
        "(%p, %p)",
        device, power)

NVML_ENTRY_POINT(nvmlDeviceGetPowerMode, tsapiDeviceGetPowerMode,
        (nvmlDevice_t device, unsigned int *powerModeId),
        "(%p, %p)",
        device, powerModeId)

NVML_ENTRY_POINT(nvmlDeviceGetSupportedPowerModes, tsapiDeviceGetSupportedPowerModes,
                 (nvmlDevice_t device, unsigned int *supportedPowerModes),
                 "(%p, %p)",
                 device, supportedPowerModes)

NVML_ENTRY_POINT(nvmlDeviceSetPowerMode, tsapiDeviceSetPowerMode,
        (nvmlDevice_t device, unsigned int powerModeId),
        "(%p, %u)",
        device, powerModeId)

NVML_ENTRY_POINT(nvmlDeviceGetTotalEnergyConsumption, tsapiDeviceGetTotalEnergyConsumption,
        (nvmlDevice_t device, unsigned long long *energy),
        "(%p, %p)",
        device, energy)

NVML_ENTRY_POINT(nvmlDeviceGetPowerManagementMode, tsapiDeviceGetPowerManagementMode,
        (nvmlDevice_t device, nvmlEnableState_t *mode),
        "(%p, %p)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceGetPowerManagementLimit, tsapiDeviceGetPowerManagementLimit,
        (nvmlDevice_t device, unsigned int *limit),
        "(%p, %p)",
        device, limit)
NVML_ENTRY_POINT(nvmlDeviceGetTemperature, tsapiDeviceGetTemperature,
        (nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp),
        "(%p, %d, %p)",
        device, sensorType, temp)
NVML_ENTRY_POINT(nvmlDeviceGetTemperatureThreshold, tsapiDeviceGetTemperatureThreshold,
        (nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp),
        "(%p, %d, %p)",
        device, thresholdType, temp)
NVML_ENTRY_POINT(nvmlDeviceSetTemperatureThreshold, tsapiDeviceSetTemperatureThreshold,
        (nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp),
        "(%p, %d, %p)",
        device, thresholdType, temp)
NVML_ENTRY_POINT(nvmlDeviceGetMarginTemperature, tsapiDeviceGetMarginTemperature,
        (nvmlDevice_t device, nvmlMarginTemperature_t *marginTempInfo),
        "(%p, %p)",
        device, marginTempInfo)
NVML_ENTRY_POINT(nvmlDeviceGetFanSpeed, tsapiDeviceGetFanSpeed,
        (nvmlDevice_t device, unsigned int *speed),
        "(%p, %p)",
        device, speed)
NVML_ENTRY_POINT(nvmlDeviceGetFanSpeed_v2, tsapiDeviceGetFanSpeed_v2,
        (nvmlDevice_t device, unsigned int fan, unsigned int * speed),
        "(%p, %u, %p)",
         device, fan, speed)
NVML_ENTRY_POINT(nvmlDeviceGetTargetFanSpeed, tsapiDeviceGetTargetFanSpeed,
        (nvmlDevice_t device, unsigned int fan, unsigned int * targetSpeed),
        "(%p, %u, %p)",
         device, fan, targetSpeed)
NVML_ENTRY_POINT(nvmlDeviceGetNumFans, tsapiDeviceGetNumFans,
        (nvmlDevice_t device, unsigned int *numFans),
        "(%p, %p)",
        device, numFans)
NVML_ENTRY_POINT(nvmlDeviceGetUtilizationRates, tsapiDeviceGetUtilizationRates,
        (nvmlDevice_t device, nvmlUtilization_t *utilization),
        "(%p, %p)",
        device, utilization)
NVML_ENTRY_POINT(nvmlDeviceGetEncoderUtilization, tsapiDeviceGetEncoderUtilization,
        (nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs),
        "(%p, %p, %p)",
        device, utilization, samplingPeriodUs)
NVML_ENTRY_POINT(nvmlDeviceGetDecoderUtilization, tsapiDeviceGetDecoderUtilization,
        (nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs),
        "(%p, %p, %p)",
        device, utilization, samplingPeriodUs)
NVML_ENTRY_POINT(nvmlDeviceGetMaxPcieLinkGeneration, tsapiDeviceGetMaxPcieLinkGeneration,
        (nvmlDevice_t device, unsigned int *maxLinkGen),
        "(%p, %p)",
        device, maxLinkGen)
NVML_ENTRY_POINT(nvmlDeviceGetMaxPcieLinkWidth, tsapiDeviceGetMaxPcieLinkWidth,
        (nvmlDevice_t device, unsigned int *maxLinkWidth),
        "(%p, %p)",
        device, maxLinkWidth)
NVML_ENTRY_POINT(nvmlDeviceGetCurrPcieLinkGeneration, tsapiDeviceGetCurrPcieLinkGeneration,
        (nvmlDevice_t device, unsigned int *currLinkGen),
        "(%p, %p)",
        device, currLinkGen)
NVML_ENTRY_POINT(nvmlDeviceGetCurrPcieLinkWidth, tsapiDeviceGetCurrPcieLinkWidth,
        (nvmlDevice_t device, unsigned int *currLinkWidth),
        "(%p, %p)",
        device, currLinkWidth)
NVML_ENTRY_POINT(nvmlSystemGetDriverVersion, tsapiSystemGetDriverVersion,
        (char* version, unsigned int length),
        "(%p, %d)",
        version, length)
NVML_ENTRY_POINT(nvmlSystemGetNVMLVersion, tsapiSystemGetNVMLVersion,
        (char* version, unsigned int length),
        "(%p, %d)",
        version, length)
NVML_ENTRY_POINT(nvmlSystemGetCudaDriverVersion, tsapiSystemGetCudaDriverVersion,
        (int* cudaDriverVersion),
        "(%p)",
        cudaDriverVersion)
NVML_ENTRY_POINT(nvmlSystemGetCudaDriverVersion_v2, tsapiSystemGetCudaDriverVersion_v2,
        (int* cudaDriverVersion),
        "(%p)",
        cudaDriverVersion)
NVML_ENTRY_POINT(nvmlUnitGetCount, tsapiUnitGetCount,
        (unsigned int *unitCount),
        "(%p)",
        unitCount)
NVML_ENTRY_POINT(nvmlUnitGetFanSpeedInfo, tsapiUnitGetFanSpeedInfo,
        (nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds),
        "(%p, %p)",
        unit, fanSpeeds)
NVML_ENTRY_POINT(nvmlUnitGetHandleByIndex, tsapiUnitGetHandleByIndex,
        (unsigned int index, nvmlUnit_t *unit),
        "(%d, %p)",
        index, unit)
NVML_ENTRY_POINT(nvmlUnitGetLedState, tsapiUnitGetLedState,
        (nvmlUnit_t unit, nvmlLedState_t *state),
        "(%p, %p)",
        unit, state)
NVML_ENTRY_POINT(nvmlUnitSetLedState, tsapiUnitSetLedState,
        (nvmlUnit_t unit, nvmlLedColor_t color),
        "(%p, %d)",
        unit, color)
NVML_ENTRY_POINT(nvmlUnitGetPsuInfo, tsapiUnitGetPsuInfo,
        (nvmlUnit_t unit, nvmlPSUInfo_t *psu),
        "(%p, %p)",
        unit, psu)
NVML_ENTRY_POINT(nvmlUnitGetTemperature, tsapiUnitGetTemperature,
        (nvmlUnit_t unit, unsigned int type, unsigned int *temp),
        "(%p, %d, %p)",
        unit, type, temp)
NVML_ENTRY_POINT(nvmlUnitGetUnitInfo, tsapiUnitGetUnitInfo,
        (nvmlUnit_t unit, nvmlUnitInfo_t *info),
        "(%p, %p)",
        unit, info)
NVML_ENTRY_POINT(nvmlUnitGetDevices, tsapiUnitGetDevices,
        (nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices),
        "(%p, %p, %p)",
        unit, deviceCount, devices)
NVML_ENTRY_POINT(nvmlDeviceGetVbiosVersion, tsapiDeviceGetVbiosVersion,
        (nvmlDevice_t device, char * version, unsigned int length),
        "(%p, %p, %d)",
        device, version, length)
NVML_ENTRY_POINT(nvmlDeviceGetBridgeChipInfo, tsapiDeviceGetBridgeChipInfo,
        (nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy),
        "(%p, %p)",
        device, bridgeHierarchy)
NVML_ENTRY_POINT(nvmlSystemGetHicVersion, tsapiSystemGetHicVersion,
        (unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries),
        "(%p, %p)",
        hwbcCount, hwbcEntries)
NVML_ENTRY_POINT(nvmlEventSetCreate, tsapiEventSetCreate,
        (nvmlEventSet_t *set),
        "(%p)",
        set)
NVML_ENTRY_POINT(nvmlDeviceRegisterEvents, tsapiDeviceRegisterEvents,
        (nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set),
        "(%p, %llu, %p)",
        device, eventTypes, set)
NVML_ENTRY_POINT(nvmlDeviceGetSupportedEventTypes, tsapiDeviceGetSupportedEventTypes,
        (nvmlDevice_t device, unsigned long long *eventTypes),
        "(%p, %p)",
        device, eventTypes)
NVML_ENTRY_POINT(nvmlEventSetWait, tsapiEventSetWait,
        (nvmlEventSet_t set, nvmlEventData_t * data, unsigned int timeoutms),
        "(%p, %p, %u)",
        set, data, timeoutms)
NVML_ENTRY_POINT(nvmlEventSetWait_v2, tsapiEventSetWait_v2,
        (nvmlEventSet_t set, nvmlEventData_t * data, unsigned int timeoutms),
        "(%p, %p, %u)",
        set, data, timeoutms)
NVML_ENTRY_POINT(nvmlEventSetFree, tsapiEventSetFree,
        (nvmlEventSet_t set),
        "(%p)",
        set)
NVML_ENTRY_POINT(nvmlDeviceGetComputeRunningProcesses, tsapiDeviceGetComputeRunningProcesses,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlDeviceGetComputeRunningProcesses_v2, tsapiDeviceGetComputeRunningProcesses_v2,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlDeviceGetComputeRunningProcesses_v3, tsapiDeviceGetComputeRunningProcesses_v3,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlDeviceGetGraphicsRunningProcesses, tsapiDeviceGetGraphicsRunningProcesses,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlDeviceGetGraphicsRunningProcesses_v2, tsapiDeviceGetGraphicsRunningProcesses_v2,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlDeviceGetGraphicsRunningProcesses_v3, tsapiDeviceGetGraphicsRunningProcesses_v3,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlDeviceGetMPSComputeRunningProcesses, tsapiDeviceGetMPSComputeRunningProcesses,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlDeviceGetMPSComputeRunningProcesses_v2, tsapiDeviceGetMPSComputeRunningProcesses_v2,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlDeviceGetMPSComputeRunningProcesses_v3, tsapiDeviceGetMPSComputeRunningProcesses_v3,
        (nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos),
        "(%p, %p, %p)",
        device, infoCount, infos)
NVML_ENTRY_POINT(nvmlSystemGetProcessName, tsapiSystemGetProcessName,
        (unsigned int pid, char *name, unsigned int length),
        "(%u, %p, %u)",
        pid, name, length)
NVML_ENTRY_POINT(nvmlDeviceOnSameBoard, tsapiDeviceOnSameBoard,
        (nvmlDevice_t dev1, nvmlDevice_t dev2, int *onSameBoard),
        "(%p, %p, %p)",
        dev1, dev2, onSameBoard)
NVML_ENTRY_POINT(nvmlDeviceGetInforomConfigurationChecksum, tsapiDeviceGetInforomConfigurationChecksum,
        (nvmlDevice_t device, unsigned int *checksum),
        "(%p, %p)",
        device, checksum)
NVML_ENTRY_POINT(nvmlDeviceValidateInforom, tsapiDeviceValidateInforom,
        (nvmlDevice_t device),
        "(%p)",
        device)
NVML_ENTRY_POINT(nvmlDeviceGetGpuOperationMode, tsapiDeviceGetGpuOperationMode,
        (nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending),
        "(%p, %p, %p)",
        device, current, pending)
NVML_ENTRY_POINT(nvmlDeviceSetGpuOperationMode, tsapiDeviceSetGpuOperationMode,
        (nvmlDevice_t device, nvmlGpuOperationMode_t mode),
        "(%p, %d)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceGetDisplayActive, tsapiDeviceGetDisplayActive,
        (nvmlDevice_t device, nvmlEnableState_t *isActive),
        "(%p, %p)",
        device, isActive)
NVML_ENTRY_POINT(nvmlDeviceGetMemoryErrorCounter, tsapiDeviceGetMemoryErrorCounter,
        (nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType,
         nvmlMemoryLocation_t locationType, unsigned long long *count),
        "(%p, %d, %d, %d, %p)",
        device, errorType, counterType, locationType, count)
NVML_ENTRY_POINT(nvmlDeviceGetSramEccErrorStatus, tsapiDeviceGetSramEccErrorStatus,
        (nvmlDevice_t device, nvmlEccSramErrorStatus_t *status),
        "(%p, %p)",
        device, status)
NVML_ENTRY_POINT(nvmlDeviceSetGpuLockedClocks, tsapiDeviceSetGpuLockedClocks,
        (nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz),
        "(%p, %u, %u)",
        device, minGpuClockMHz, maxGpuClockMHz)
NVML_ENTRY_POINT(nvmlDeviceResetGpuLockedClocks, tsapiDeviceResetGpuLockedClocks,
        (nvmlDevice_t device),
        "(%p)",
        device)
NVML_ENTRY_POINT(nvmlDeviceSetMemoryLockedClocks, tsapiDeviceSetMemoryLockedClocks,
        (nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz),
        "(%p, %u, %u)",
        device, minMemClockMHz, maxMemClockMHz)
NVML_ENTRY_POINT(nvmlDeviceResetMemoryLockedClocks, tsapiDeviceResetMemoryLockedClocks,
        (nvmlDevice_t device),
        "(%p)",
        device)
NVML_ENTRY_POINT(nvmlDeviceSetApplicationsClocks, tsapiDeviceSetApplicationsClocks,
        (nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz),
        "(%p, %u, %u)",
        device, memClockMHz, graphicsClockMHz)
NVML_ENTRY_POINT(nvmlDeviceGetApplicationsClock, tsapiDeviceGetApplicationsClock,
        (nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz),
        "(%p, %d, %p)",
        device, clockType, clockMHz)
NVML_ENTRY_POINT(nvmlDeviceGetMaxCustomerBoostClock, tsapiDeviceGetMaxCustomerBoostClock,
        (nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz),
        "(%p, %d, %p)",
        device, clockType, clockMHz)
NVML_ENTRY_POINT(nvmlDeviceGetClock, tsapiDeviceGetClock,
        (nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz),
        "(%p, %d, %d, %p)",
        device, clockType, clockId, clockMHz)
NVML_ENTRY_POINT(nvmlDeviceGetDefaultApplicationsClock, tsapiDeviceGetDefaultApplicationsClock,
        (nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz),
        "(%p, %d, %p)",
        device, clockType, clockMHz)
NVML_ENTRY_POINT(nvmlDeviceResetApplicationsClocks, tsapiDeviceResetApplicationsClocks,
        (nvmlDevice_t device),
        "(%p)",
        device)
NVML_ENTRY_POINT(nvmlDeviceGetSupportedMemoryClocks, tsapiDeviceGetSupportedMemoryClocks,
        (nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz),
        "(%p, %p, %p)",
        device, count, clocksMHz)
NVML_ENTRY_POINT(nvmlDeviceGetSupportedGraphicsClocks, tsapiDeviceGetSupportedGraphicsClocks,
        (nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz),
        "(%p, %u, %p, %p)",
        device, memoryClockMHz, count, clocksMHz)
NVML_ENTRY_POINT(nvmlDeviceGetAutoBoostedClocksEnabled, tsapiDeviceGetAutoBoostedClocksEnabled,
        (nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled),
        "(%p, %p, %p)",
        device, isEnabled, defaultIsEnabled)
NVML_ENTRY_POINT(nvmlDeviceSetAutoBoostedClocksEnabled, tsapiDeviceSetAutoBoostedClocksEnabled,
        (nvmlDevice_t device, nvmlEnableState_t enabled),
        "(%p, %d)",
        device, enabled)
NVML_ENTRY_POINT(nvmlDeviceSetDefaultAutoBoostedClocksEnabled, tsapiDeviceSetDefaultAutoBoostedClocksEnabled,
        (nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags),
        "(%p, %d, 0x%x)",
        device, enabled, flags)
NVML_ENTRY_POINT(nvmlDeviceGetPowerManagementLimitConstraints, tsapiDeviceGetPowerManagementLimitConstraints,
        (nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit),
        "(%p, %p, %p)",
        device, minLimit, maxLimit)
NVML_ENTRY_POINT(nvmlDeviceGetPowerManagementDefaultLimit, tsapiDeviceGetPowerManagementDefaultLimit,
        (nvmlDevice_t device, unsigned int *defaultLimit),
        "(%p, %p)",
        device, defaultLimit)
NVML_ENTRY_POINT(nvmlDeviceSetPowerManagementLimit, tsapiDeviceSetPowerManagementLimit,
        (nvmlDevice_t device, unsigned int limit),
        "(%p, %u)",
        device, limit)
NVML_ENTRY_POINT(nvmlDeviceGetCurrentClocksEventReasons, tsapiDeviceGetCurrentClocksEventReasons,
        (nvmlDevice_t device, unsigned long long *clocksEventReasons),
        "(%p, %p)",
        device, clocksEventReasons)
NVML_ENTRY_POINT(nvmlDeviceGetCurrentClocksThrottleReasons, tsapiDeviceGetCurrentClocksThrottleReasons,
        (nvmlDevice_t device, unsigned long long *clocksThrottleReasons),
        "(%p, %p)",
        device, clocksThrottleReasons)
NVML_ENTRY_POINT(nvmlDeviceGetSupportedClocksEventReasons, tsapiDeviceGetSupportedClocksEventReasons,
        (nvmlDevice_t device, unsigned long long *supportedClocksEventReasons),
        "(%p, %p)",
        device, supportedClocksEventReasons);
NVML_ENTRY_POINT(nvmlDeviceGetSupportedClocksThrottleReasons, tsapiDeviceGetSupportedClocksThrottleReasons,
        (nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons),
        "(%p, %p)",
        device, supportedClocksThrottleReasons);
NVML_ENTRY_POINT(nvmlDeviceGetIndex, tsapiDeviceGetIndex,
        (nvmlDevice_t device, unsigned int *index),
        "(%p, %p)",
        device, index)
NVML_ENTRY_POINT(nvmlDeviceGetAccountingMode, tsapiDeviceGetAccountingMode, 
        (nvmlDevice_t device, nvmlEnableState_t * mode),
        "(%p, %p)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceSetAccountingMode, tsapiDeviceSetAccountingMode,
        (nvmlDevice_t device, nvmlEnableState_t mode),
        "(%p, %d)",
        device, mode)
NVML_ENTRY_POINT(nvmlDeviceClearAccountingPids, tsapiDeviceClearAccountingPids,
        (nvmlDevice_t device),
        "(%p)",
        device)
NVML_ENTRY_POINT(nvmlDeviceGetAccountingStats, tsapiDeviceGetAccountingStats,
        (nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t * stats),
        "(%p, %d, %p)",
        device, pid, stats)
NVML_ENTRY_POINT(nvmlDeviceGetAccountingPids, tsapiDeviceGetAccountingPids,
        (nvmlDevice_t device, unsigned int *count, unsigned int *pids),
        "(%p, %p, %p)",
        device, count, pids)
NVML_ENTRY_POINT(nvmlDeviceGetAccountingBufferSize, tsapiDeviceGetAccountingBufferSize,
        (nvmlDevice_t device, unsigned int *bufferSize),
        "(%p, %p)",
        device, bufferSize)
NVML_ENTRY_POINT(nvmlDeviceGetRetiredPages, tsapiDeviceGetRetiredPages_v1,
    (nvmlDevice_t device, nvmlPageRetirementCause_t sourceFilter, unsigned int *count, unsigned long long *addresses),
    "(%p, %u, %p, %p)",
    device, sourceFilter, count, addresses)
NVML_ENTRY_POINT(nvmlDeviceGetRetiredPages_v2, tsapiDeviceGetRetiredPages_v2,
    (nvmlDevice_t device, nvmlPageRetirementCause_t sourceFilter, unsigned int *count, unsigned long long *addresses, unsigned long long *timestamps),
    "(%p, %u, %p, %p, %p)",
    device, sourceFilter, count, addresses, timestamps)
NVML_ENTRY_POINT(nvmlDeviceGetRetiredPagesPendingStatus, tsapiDeviceGetRetiredPagesPendingStatus,
    (nvmlDevice_t device, nvmlEnableState_t *isPending),
    "(%p, %p)",
    device, isPending)
NVML_ENTRY_POINT(nvmlDeviceSetAPIRestriction, tsapiDeviceSetAPIRestriction,
        (nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted),
        "(%p, %d, %d)",
        device, apiType, isRestricted)
NVML_ENTRY_POINT(nvmlDeviceGetAPIRestriction, tsapiDeviceGetAPIRestriction,
        (nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted),
        "(%p, %d, %p)",
        device, apiType, isRestricted)
NVML_ENTRY_POINT(nvmlDeviceGetMinorNumber, tsapiDeviceGetMinorNumber,
        (nvmlDevice_t device, unsigned int *minorNumber),
        "(%p, %p)",
        device, minorNumber)

NVML_ENTRY_POINT(nvmlDeviceGetEnforcedPowerLimit, tsapiDeviceGetEnforcedPowerLimit,
        (nvmlDevice_t device, unsigned int *limit),
        "(%p, %p)",
        device, limit)

NVML_ENTRY_POINT(nvmlDeviceGetSamples, tsapiDeviceGetSamples,
        (nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType,
        unsigned int *sampleCount, nvmlSample_t *samples),
        "(%p, %u, %llu, %p, %p, %p)",
        device, type, lastSeenTimeStamp, sampleValType, sampleCount, samples)

NVML_ENTRY_POINT(nvmlDeviceGetPcieThroughput, tsapiDeviceGetPcieThroughput,
        (nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value),
        "(%p, %d, %p)",
        device, counter, value)

NVML_ENTRY_POINT(nvmlDeviceGetPcieReplayCounter, tsapiDeviceGetPcieReplayCounter,
        (nvmlDevice_t device, unsigned int *value),
        "(%p, %p)",
        device, value)

NVML_ENTRY_POINT(nvmlDeviceGetTopologyCommonAncestor, tsapiDeviceGetTopologyCommonAncestor,
        (nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo),
        "(%p, %p, %p)",
        device1, device2, pathInfo)

NVML_ENTRY_POINT(nvmlDeviceGetTopologyNearestGpus, tsapiDeviceGetTopologyNearestGpus,
        (nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray),
        "(%p, %d, %p, %p)",
        device, level, count, deviceArray)

NVML_ENTRY_POINT(nvmlSystemGetTopologyGpuSet, tsapiSystemGetTopologyGpuSet, 
        (unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray),
        "(%d, %p, %p)",
        cpuNumber, count, deviceArray)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkState, tsapiDeviceGetNvLinkState,
        (nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive),
        "(%p, %d, %p)",
        device, link, isActive)

NVML_ENTRY_POINT(nvmlDeviceGetP2PStatus, tsapiDeviceGetP2PStatus, 
        (nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus),
        "(%p, %p, %d %p)",
        device1, device2,p2pIndex,  p2pStatus)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkVersion, tsapiDeviceGetNvLinkVersion,
        (nvmlDevice_t device, unsigned int link, unsigned int *version),
        "(%p, %d, %p)", 
        device, link, version)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkRemotePciInfo, tsapiDeviceGetNvLinkRemotePciInfo,
        (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci),
        "(%p, %d, %p)", 
        device, link, pci)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkRemotePciInfo_v2, tsapiDeviceGetNvLinkRemotePciInfo_v2,
        (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci),
        "(%p, %d, %p)",
        device, link, pci)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkRemoteDeviceType, tsapiDeviceGetNvLinkRemoteDeviceType,
        (nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType),
        "(%p, %d, %p)", 
        device, link, pNvLinkDeviceType)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkCapability, tsapiDeviceGetNvLinkCapability,
        (nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult),
        "(%p, %d, %d, %p)",
        device, link, capability, capResult)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkErrorCounter, tsapiDeviceGetNvLinkErrorCounter,
        (nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue),
        "(%p, %d, %d, %p)",
        device, link, counter, counterValue)

NVML_ENTRY_POINT(nvmlDeviceResetNvLinkErrorCounters, tsapiDeviceResetNvLinkErrorCounters,
        (nvmlDevice_t device, unsigned int link),
        "(%p, %d)",
        device, link)

NVML_ENTRY_POINT(nvmlDeviceSetNvLinkUtilizationControl, tsapiDeviceSetNvLinkUtilizationControl,
        (nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control, unsigned int reset),
        "(%p, %d, %d, %p, %d)",
        device, link, counter, control, reset)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkUtilizationControl, tsapiDeviceGetNvLinkUtilizationControl,
        (nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control),
        "(%p, %d, %d, %p)",
        device, link, counter, control)

NVML_ENTRY_POINT(nvmlDeviceGetNvLinkUtilizationCounter, tsapiDeviceGetNvLinkUtilizationCounter,
        (nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter),
        "(%p, %d, %d, %p, %p)",
        device, link, counter, rxcounter, txcounter)

NVML_ENTRY_POINT(nvmlDeviceFreezeNvLinkUtilizationCounter, tsapiDeviceFreezeNvLinkUtilizationCounter,
        (nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze),
        "(%p, %d, %d, %d)",
        device, link, counter, freeze)

NVML_ENTRY_POINT(nvmlDeviceResetNvLinkUtilizationCounter, tsapiDeviceResetNvLinkUtilizationCounter,
        (nvmlDevice_t device, unsigned int link, unsigned int counter),
        "(%p, %d, %d)",
        device, link, counter)

NVML_ENTRY_POINT(nvmlDeviceGetVirtualizationMode, tsapiDeviceGetVirtualizationMode,
        (nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode),
        "(%p %p)",
        device, pVirtualMode)

NVML_ENTRY_POINT(nvmlDeviceSetVirtualizationMode, tsapiDeviceSetVirtualizationMode,
        (nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode),
        "(%p %d)",
        device, virtualMode)

NVML_ENTRY_POINT(nvmlDeviceGetSupportedVgpus, tsapiDeviceGetSupportedVgpus,
        (nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds),
        "(%p %p %p)",
        device, vgpuCount, vgpuTypeIds)

NVML_ENTRY_POINT(nvmlDeviceGetCreatableVgpus, tsapiDeviceGetCreatableVgpus,
        (nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds),
        "(%p %p %p)",
        device, vgpuCount, vgpuTypeIds)

NVML_ENTRY_POINT(nvmlVgpuTypeGetClass, tsapiVgpuTypeGetClass,
        (nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size),
        "(%d %p %p)",
        vgpuTypeId, vgpuTypeClass, size)

NVML_ENTRY_POINT(nvmlVgpuTypeGetName, tsapiVgpuTypeGetName,
        (nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size),
        "(%d %p %p)",
        vgpuTypeId, vgpuTypeName, size)

NVML_ENTRY_POINT(nvmlVgpuTypeGetGpuInstanceProfileId, tsapiVgpuTypeGetGpuInstanceProfileId,
        (nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId),
        "(%d %p)",
        vgpuTypeId, gpuInstanceProfileId)

NVML_ENTRY_POINT(nvmlVgpuTypeGetDeviceID, tsapiVgpuTypeGetDeviceID,
        (nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID),
        "(%d %p %p)",
        vgpuTypeId, deviceID, subsystemID)

NVML_ENTRY_POINT(nvmlVgpuTypeGetFramebufferSize, tsapiVgpuTypeGetFramebufferSize,
        (nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize),
        "(%d %p)",
        vgpuTypeId, fbSize)

NVML_ENTRY_POINT(nvmlVgpuTypeGetNumDisplayHeads, tsapiVgpuTypeGetNumDisplayHeads,
        (nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads),
        "(%d %p)",
        vgpuTypeId, numDisplayHeads)

NVML_ENTRY_POINT(nvmlVgpuTypeGetResolution, tsapiVgpuTypeGetResolution,
        (nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim),
        "(%d %d %p %p)",
        vgpuTypeId, displayIndex, xdim, ydim)

NVML_ENTRY_POINT(nvmlVgpuTypeGetLicense, tsapiVgpuTypeGetLicense,
        (nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size),
        "(%d %p %d)",
        vgpuTypeId, vgpuTypeLicenseString, size)

NVML_ENTRY_POINT(nvmlVgpuTypeGetFrameRateLimit, tsapiVgpuTypeGetFrameRateLimit,
        (nvmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit),
        "(%d %p)",
        vgpuTypeId, frameRateLimit)

NVML_ENTRY_POINT(nvmlVgpuTypeGetMaxInstances, tsapiVgpuTypeGetMaxInstances,
        (nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount),
        "(%p %d %p)",
        device, vgpuTypeId, vgpuInstanceCount)

NVML_ENTRY_POINT(nvmlVgpuTypeGetMaxInstancesPerVm, tsapiVgpuTypeGetMaxInstancesPerVm,
        (nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm),
        "(%d %p)",
        vgpuTypeId, vgpuInstanceCountPerVm)

NVML_ENTRY_POINT(nvmlDeviceGetActiveVgpus, tsapiDeviceGetActiveVgpus,
        (nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances),
        "(%p %p %p)",
        device, vgpuCount, vgpuInstances)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetVmID, tsapiVgpuInstanceGetVmID,
        (nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType),
        "(%d %p %d %p)",
        vgpuInstance, vmId, size, vmIdType)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetUUID, tsapiVgpuInstanceGetUUID,
        (nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size),
        "(%d %p %d)",
        vgpuInstance, uuid, size)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetMdevUUID, tsapiVgpuInstanceGetMdevUUID,
        (nvmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size),
        "(%d %p %d)",
        vgpuInstance, mdevUuid, size)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetVmDriverVersion, tsapiVgpuInstanceGetVmDriverVersion,
        (nvmlVgpuInstance_t vgpuInstance, char *version, unsigned int length),
        "(%d %p %d)",
        vgpuInstance, version, length)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetFbUsage, tsapiVgpuInstanceGetFbUsage,
        (nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage),
        "(%d %p)",
        vgpuInstance, fbUsage)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetLicenseStatus, tsapiVgpuInstanceGetLicenseStatus,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed),
        "(%d %p)",
        vgpuInstance, licensed)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetLicenseInfo, tsapiVgpuInstanceGetLicenseInfo,
        (nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo),
        "(%d %p)",
        vgpuInstance, licenseInfo)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetLicenseInfo_v2, tsapiVgpuInstanceGetLicenseInfo_v2,
        (nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo),
        "(%d %p)",
        vgpuInstance, licenseInfo)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetType, tsapiVgpuInstanceGetType,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *vgpuTypeId),
        "(%d %p)",
        vgpuInstance, vgpuTypeId)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetFrameRateLimit, tsapiVgpuInstanceGetFrameRateLimit,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit),
        "(%d %p)",
        vgpuInstance, frameRateLimit)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetEccMode, tsapiVgpuInstanceGetEccMode,
        (nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode),
        "(%d %p)",
        vgpuInstance, eccMode)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetEncoderCapacity, tsapiVgpuInstanceGetEncoderCapacity,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity),
        "(%d %p)",
        vgpuInstance, encoderCapacity)

NVML_ENTRY_POINT(nvmlVgpuInstanceSetEncoderCapacity, tsapiVgpuInstanceSetEncoderCapacity,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity),
        "(%d %d)",
        vgpuInstance, encoderCapacity)

NVML_ENTRY_POINT(nvmlDeviceGetVgpuUtilization, tsapiDeviceGetVgpuUtilization,
        (nvmlDevice_t device, unsigned long long lastSeenTimeStamp,
         nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount,
         nvmlVgpuInstanceUtilizationSample_t *utilizationSamples),
        "(%p %llu %p %p %p)",
        device, lastSeenTimeStamp, sampleValType, vgpuInstanceSamplesCount, utilizationSamples)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetMetadata, tsapiVgpuInstanceGetMetadata,
        (nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize),
        "(%d %p %p)",
        vgpuInstance, vgpuMetadata, bufferSize)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetGpuPciId, tsapiVgpuInstanceGetGpuPciId,
        (nvmlVgpuInstance_t vgpuInstance, char *vgpuPciId, unsigned int *length),
        "(%d, %p, %p)",
        vgpuInstance, vgpuPciId, length)

NVML_ENTRY_POINT(nvmlVgpuTypeGetCapabilities, tsapiVgpuTypeGetCapabilities,
        (nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int *capResult),
        "(%d %d %p)",
        vgpuTypeId, capability, capResult)

NVML_ENTRY_POINT(nvmlDeviceGetGspFirmwareVersion, tsapiDeviceGetGspFirmwareVersion,
        (nvmlDevice_t device, char *version),
        "(%p, %p)",
        device, version)

NVML_ENTRY_POINT(nvmlDeviceGetGspFirmwareMode, tsapiDeviceGetGspFirmwareMode,
        (nvmlDevice_t device, unsigned int *isEnabled, unsigned int *defaultMode),
        "(%p, %p, %p)",
        device, isEnabled, defaultMode)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetGpuInstanceId, tsapiVgpuInstanceGetGpuInstanceId,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId),
        "(%d %p)",
        vgpuInstance, gpuInstanceId)

NVML_ENTRY_POINT(nvmlDeviceGetVgpuMetadata, tsapiDeviceGetVgpuMetadata,
        (nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize),
        "(%p %p %p)",
        device, pgpuMetadata, bufferSize)

NVML_ENTRY_POINT(nvmlGetVgpuCompatibility, tsapiGetVgpuCompatibility,
        (nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo),
        "(%p %p %p)",
        vgpuMetadata, pgpuMetadata, compatibilityInfo)

NVML_ENTRY_POINT(nvmlDeviceGetPgpuMetadataString, tsapiDeviceGetPgpuMetadataString,
        (nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize),
        "(%p %p %p)",
        device, pgpuMetadata, bufferSize)

NVML_ENTRY_POINT(nvmlDeviceGetGridLicensableFeatures, tsapiDeviceGetGridLicensableFeatures,
        (nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures),
        "(%p %p)",
        device, pGridLicensableFeatures)

NVML_ENTRY_POINT(nvmlDeviceGetGridLicensableFeatures_v2, tsapiDeviceGetGridLicensableFeatures_v2,
        (nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures),
        "(%p %p)",
        device, pGridLicensableFeatures)

NVML_ENTRY_POINT(nvmlDeviceGetGridLicensableFeatures_v3, tsapiDeviceGetGridLicensableFeatures_v3,
        (nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures),
        "(%p %p)",
        device, pGridLicensableFeatures)

NVML_ENTRY_POINT(nvmlDeviceGetGridLicensableFeatures_v4, tsapiDeviceGetGridLicensableFeatures_v4,
        (nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures),
        "(%p %p)",
        device, pGridLicensableFeatures)

NVML_ENTRY_POINT(nvmlDeviceGetEncoderCapacity, tsapiDeviceGetEncoderCapacity,
        (nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *pEncoderCapacity),
        "(%p %d %p)",
        device, encoderQueryType, pEncoderCapacity)

NVML_ENTRY_POINT(nvmlDeviceGetEncoderStats, tsapiDeviceGetEncoderStats,
        (nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps,
         unsigned int *averageLatency),
        "(%p %p %p %p)",
        device, sessionCount, averageFps, averageLatency)

NVML_ENTRY_POINT(nvmlDeviceGetEncoderSessions, tsapiDeviceGetEncoderSessions,
        (nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos),
        "(%p %p %p)",
        device, sessionCount, sessionInfos)

NVML_ENTRY_POINT(nvmlDeviceGetFBCStats, tsapiDeviceGetFBCStats,
        (nvmlDevice_t device, nvmlFBCStats_t *fbcStats),
        "(%p, %p)",
        device, fbcStats)

NVML_ENTRY_POINT(nvmlDeviceGetFBCSessions, tsapiDeviceGetFBCSessions,
        (nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo),
        "(%p %p %p)",
        device, sessionCount, sessionInfo)

NVML_ENTRY_POINT(nvmlDeviceModifyDrainState, tsapiDeviceModifyDrainState,
        (nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState),
        "(%p, %d)",
        pciInfo, newState)

NVML_ENTRY_POINT(nvmlDeviceQueryDrainState, tsapiDeviceQueryDrainState,
        (nvmlPciInfo_t *pciInfo, nvmlEnableState_t *newState),
        "(%p, %p)",
        pciInfo, newState)

NVML_ENTRY_POINT(nvmlDeviceRemoveGpu, tsapiDeviceRemoveGpu,
        (nvmlPciInfo_t *pciInfo),
        "(%p)",
        pciInfo)

NVML_ENTRY_POINT(nvmlDeviceRemoveGpu_v2, tsapiDeviceRemoveGpu_v2,
        (nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState),
        "(%p, %d, %d)",
        pciInfo, gpuState, linkState)

NVML_ENTRY_POINT(nvmlDeviceDiscoverGpus, tsapiDeviceDiscoverGpus,
        (nvmlPciInfo_t *pciInfo),
        "(%p)",
        pciInfo)

NVML_ENTRY_POINT(nvmlDeviceGetFieldValues, tsapiDeviceGetFieldValues,
        (nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values),
        "(%p, %d, %p)",
        device, valuesCount, values)

NVML_ENTRY_POINT(nvmlDeviceGetVgpuProcessUtilization, tsapiDeviceGetVgpuProcessUtilization,
        (nvmlDevice_t device, unsigned long long lastSeenTimeStamp,
         unsigned int *vgpuProcessSamplesCount,
         nvmlVgpuProcessUtilizationSample_t *utilizationSamples),
        "(%p %llu %p %p)",
        device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetEncoderStats, tsapiVgpuInstanceGetEncoderStats,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount,
        unsigned int *averageFps, unsigned int *averageLatency),
        "(%d %p %p %p)",
        vgpuInstance, sessionCount, averageFps, averageLatency)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetEncoderSessions, tsapiVgpuInstanceGetEncoderSessions,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo),
        "(%d %p %p)",
        vgpuInstance, sessionCount, sessionInfo)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetFBCStats, tsapiVgpuInstanceGetFBCStats,
        (nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats),
        "(%d %p)",
        vgpuInstance, fbcStats)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetFBCSessions, tsapiVgpuInstanceGetFBCSessions,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo),
        "(%d %p %p)",
        vgpuInstance, sessionCount, sessionInfo)

NVML_ENTRY_POINT(nvmlDeviceGetProcessUtilization, tsapiDeviceGetProcessUtilization,
        (nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount,
        unsigned long long lastSeenTimeStamp),
        "(%p, %p, %p, %llu)",
        device, utilization, processSamplesCount, lastSeenTimeStamp)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetAccountingMode, tsapiVgpuInstanceGetAccountingMode,
        (nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode),
        "(%d, %p)",
        vgpuInstance, mode)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetAccountingPids, tsapiVgpuInstanceGetAccountingPids,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids),
        "(%d, %p, %p)",
        vgpuInstance, count, pids)

NVML_ENTRY_POINT(nvmlVgpuInstanceGetAccountingStats, tsapiVgpuInstanceGetAccountingStats,
        (nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats),
        "(%d, %d, %p)",
        vgpuInstance, pid, stats)

NVML_ENTRY_POINT(nvmlVgpuInstanceClearAccountingPids, tsapiVgpuInstanceClearAccountingPids,
        (nvmlVgpuInstance_t vgpuInstance),
        "(%d)",
        vgpuInstance)

NVML_ENTRY_POINT(nvmlGetBlacklistDeviceCount, tsapiGetBlacklistDeviceCount,
        (unsigned int *deviceCount),
        "(%p)",
        deviceCount)

NVML_ENTRY_POINT(nvmlGetBlacklistDeviceInfoByIndex, tsapiGetBlacklistDeviceInfoByIndex,
        (unsigned int index, nvmlBlacklistDeviceInfo_t *info),
        "(%d, %p)",
        index, info)

NVML_ENTRY_POINT(nvmlGetExcludedDeviceCount, tsapiGetExcludedDeviceCount,
        (unsigned int *deviceCount),
        "(%p)",
        deviceCount)

NVML_ENTRY_POINT(nvmlGetExcludedDeviceInfoByIndex, tsapiGetExcludedDeviceInfoByIndex,
        (unsigned int index, nvmlExcludedDeviceInfo_t *info),
        "(%d, %p)",
        index, info)

NVML_ENTRY_POINT(nvmlGetVgpuVersion, tsapiGetVgpuVersion,
        (nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t * current),
        "(%p, %p)",
        supported, current)

NVML_ENTRY_POINT(nvmlSetVgpuVersion, tsapiSetVgpuVersion,
        (nvmlVgpuVersion_t *vgpuVersion),
        "(%p)",
        vgpuVersion)

NVML_ENTRY_POINT(nvmlDeviceGetHostVgpuMode, tsapiDeviceGetHostVgpuMode,
        (nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode),
        "(%p, %p)",
        device, pHostVgpuMode)

NVML_ENTRY_POINT(nvmlDeviceSetMigMode, tsapiDeviceSetMigMode,
        (nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus),
        "(%p, %d, %p)",
        device, mode, activationStatus)

NVML_ENTRY_POINT(nvmlDeviceGetMigMode, tsapiDeviceGetMigMode,
        (nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode),
        "(%p, %p, %p)",
        device, currentMode, pendingMode)

NVML_ENTRY_POINT(nvmlDeviceGetGpuInstanceProfileInfo, tsapiDeviceGetGpuInstanceProfileInfo,
        (nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info),
        "(%p, %u, %p)",
        device, profile, info)

NVML_ENTRY_POINT(nvmlDeviceGetGpuInstanceProfileInfoV, tsapiDeviceGetGpuInstanceProfileInfoV,
        (nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t *info),
        "(%p, %u, %p)",
        device, profile, info)

NVML_ENTRY_POINT(nvmlDeviceGetGpuInstanceRemainingCapacity, tsapiDeviceGetGpuInstanceRemainingCapacity,
        (nvmlDevice_t device, unsigned int profileId, unsigned int *count),
        "(%p, %u, %p)",
        device, profileId, count)

NVML_ENTRY_POINT(nvmlDeviceGetGpuInstancePossiblePlacements, tsapiDeviceGetGpuInstancePossiblePlacements,
        (nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count),
        "(%p, %u, %p, %p)",
        device, profileId, placements, count)

NVML_ENTRY_POINT(nvmlDeviceGetGpuInstancePossiblePlacements_v2, tsapiDeviceGetGpuInstancePossiblePlacements_v2,
        (nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count),
        "(%p, %u, %p, %p)",
        device, profileId, placements, count)

NVML_ENTRY_POINT(nvmlDeviceCreateGpuInstance, tsapiDeviceCreateGpuInstance,
        (nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance),
        "(%p, %d, %p)",
        device, profileId, gpuInstance)

NVML_ENTRY_POINT(nvmlDeviceCreateGpuInstanceWithPlacement, tsapiDeviceCreateGpuInstanceWithPlacement,
        (nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement, nvmlGpuInstance_t *gpuInstance),
        "(%p, %d, %p, %p)",
        device, profileId, placement, gpuInstance)

NVML_ENTRY_POINT(nvmlGpuInstanceDestroy, tsapiGpuInstanceDestroy,
        (nvmlGpuInstance_t gpuInstance),
        "(%p)",
        gpuInstance)

NVML_ENTRY_POINT(nvmlDeviceGetGpuInstances, tsapiDeviceGetGpuInstances,
        (nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *instances, unsigned int *count),
        "(%p, %u, %p, %p)",
        device, profileId, instances, count)

NVML_ENTRY_POINT(nvmlGpuInstanceGetInfo, tsapiGpuInstanceGetInfo,
        (nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info),
        "(%p, %p)",
        gpuInstance, info)

NVML_ENTRY_POINT(nvmlDeviceGetGpuInstanceById, tsapiDeviceGetGpuInstanceById,
        (nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance),
        "(%p, %u, %p)",
        device, id, gpuInstance)

NVML_ENTRY_POINT(nvmlGpuInstanceGetComputeInstanceProfileInfo, tsapiGpuInstanceGetComputeInstanceProfileInfo,
        (nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info),
        "(%p, %u, %u, %p)",
        gpuInstance, profile, engProfile, info)

NVML_ENTRY_POINT(nvmlGpuInstanceGetComputeInstanceProfileInfoV, tsapiGpuInstanceGetComputeInstanceProfileInfoV,
        (nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t *info),
        "(%p, %u, %u, %p)",
        gpuInstance, profile, engProfile, info)

NVML_ENTRY_POINT(nvmlGpuInstanceGetComputeInstanceRemainingCapacity, tsapiGpuInstanceGetComputeInstanceRemainingCapacity,
        (nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count),
        "(%p, %u, %p)",
        gpuInstance, profileId, count)

NVML_ENTRY_POINT(nvmlGpuInstanceCreateComputeInstance, tsapiGpuInstanceCreateComputeInstance,
        (nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance),
        "(%p, %u, %p)",
        gpuInstance, profileId, computeInstance)

NVML_ENTRY_POINT(nvmlComputeInstanceDestroy, tsapiComputeInstanceDestroy,
        (nvmlComputeInstance_t computeInstance),
        "(%p)",
        computeInstance)

NVML_ENTRY_POINT(nvmlGpuInstanceGetComputeInstances, tsapiGpuInstanceGetComputeInstances,
        (nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstances, unsigned int *count),
        "(%p, %u, %p, %p)",
        gpuInstance, profileId, computeInstances, count)

NVML_ENTRY_POINT(nvmlGpuInstanceGetComputeInstanceById, tsapiGpuInstanceGetComputeInstanceById,
        (nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t *computeInstance),
        "(%p, %u, %p)",
        gpuInstance, id, computeInstance)

NVML_ENTRY_POINT(nvmlComputeInstanceGetInfo, tsapiComputeInstanceGetInfo,
        (nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info),
        "(%p, %p)",
        computeInstance, info)

NVML_ENTRY_POINT(nvmlComputeInstanceGetInfo_v2, tsapiComputeInstanceGetInfo_v2,
        (nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info),
        "(%p, %p)",
        computeInstance, info)

NVML_ENTRY_POINT(nvmlDeviceIsMigDeviceHandle, tsapiDeviceIsMigDeviceHandle,
        (nvmlDevice_t device, unsigned int *isMigDevice),
        "(%p, %p)",
        device, isMigDevice)

NVML_ENTRY_POINT(nvmlDeviceGetGpuInstanceId, tsapiDeviceGetGpuInstanceId,
        (nvmlDevice_t device, unsigned int *id),
        "(%p, %p)",
        device, id)

NVML_ENTRY_POINT(nvmlDeviceGetComputeInstanceId, tsapiDeviceGetComputeInstanceId,
        (nvmlDevice_t device, unsigned int *id),
        "(%p, %p)",
        device, id)

NVML_ENTRY_POINT(nvmlDeviceGetMaxMigDeviceCount, tsapiDeviceGetMaxMigDeviceCount,
        (nvmlDevice_t device, unsigned int *migDeviceCount),
        "(%p, %p)",
        device, migDeviceCount)

NVML_ENTRY_POINT(nvmlDeviceGetMigDeviceHandleByIndex, tsapiDeviceGetMigDeviceHandleByIndex,
        (nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice),
        "(%p, %u, %p)",
        device, index, migDevice)

NVML_ENTRY_POINT(nvmlSystemGetConfComputeState, tsapiSystemGetConfComputeState,
        (nvmlConfComputeSystemState_t *ccMode),
        "(%p)",
        ccMode)

NVML_ENTRY_POINT(nvmlDeviceGetDeviceHandleFromMigDeviceHandle, tsapiDeviceGetDeviceHandleFromMigDeviceHandle,
        (nvmlDevice_t migDevice, nvmlDevice_t *device),
        "(%p, %p)",
        migDevice, device)

NVML_ENTRY_POINT(nvmlDeviceGetAttributes, tsapiDeviceGetAttributes,
        (nvmlDevice_t device, nvmlDeviceAttributes_t *attributes),
        "(%p, %p)",
        device, attributes)

NVML_ENTRY_POINT(nvmlDeviceGetAttributes_v2, tsapiDeviceGetAttributes_v2,
        (nvmlDevice_t device, nvmlDeviceAttributes_t *attributes),
        "(%p, %p)",
        device, attributes)

NVML_ENTRY_POINT(nvmlDeviceGetRemappedRows, tsapiDeviceGetRemappedRows,
         (nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows,
          unsigned int *isPending, unsigned int *failureOccurred),
         "(%p, %p, %p, %p, %p)",
         device, corrRows, uncRows, isPending, failureOccurred)

NVML_ENTRY_POINT(nvmlDeviceGetRowRemapperHistogram, tsapiDeviceGetRowRemapperHistogram,
        (nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values),
        "(%p, %p)",
        device, values)

NVML_ENTRY_POINT(nvmlDeviceGetBusType, tsapiDeviceGetBusType,
        (nvmlDevice_t device, nvmlBusType_t *type),
        "(%p, %p)",
        device, type)

NVML_ENTRY_POINT(nvmlDeviceGetIrqNum, tsapiDeviceGetIrqNum,
        (nvmlDevice_t device, unsigned int *irqNum),
        "(%p, %p)",
        device, irqNum)

NVML_ENTRY_POINT(nvmlDeviceGetNumGpuCores, tsapiDeviceGetNumGpuCores,
        (nvmlDevice_t device, unsigned int *numCores),
        "(%p, %p)",
        device, numCores)

NVML_ENTRY_POINT(nvmlDeviceGetPowerSource, tsapiDeviceGetPowerSource,
        (nvmlDevice_t device, nvmlPowerSource_t *powerSource),
        "(%p, %p)",
        device, powerSource)

NVML_ENTRY_POINT(nvmlDeviceGetMemoryBusWidth, tsapiDeviceGetMemoryBusWidth,
        (nvmlDevice_t device, unsigned int *busWidth),
        "(%p, %p)",
        device, busWidth)

NVML_ENTRY_POINT(nvmlDeviceGetPcieLinkMaxSpeed, tsapiDeviceGetPcieLinkMaxSpeed,
        (nvmlDevice_t device, unsigned int *maxSpeed),
        "(%p, %p)",
        device, maxSpeed)

NVML_ENTRY_POINT(nvmlDeviceGetAdaptiveClockInfoStatus, tsapiDeviceGetAdaptiveClockInfoStatus,
        (nvmlDevice_t device, unsigned int *adaptiveClockStatus),
        "(%p, %p)",
        device, adaptiveClockStatus)

NVML_ENTRY_POINT(nvmlDeviceGetPcieSpeed, tsapiDeviceGetPcieSpeed,
        (nvmlDevice_t device, unsigned int *pcieSpeed),
        "(%p, %p)",
        device, pcieSpeed)
NVML_ENTRY_POINT(nvmlDeviceGetDynamicPstatesInfo, tsapiDeviceGetDynamicPstatesInfo,
        (nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t *pDynamicPstatesInfo),
        "(%p %p)",
        device, pDynamicPstatesInfo)
NVML_ENTRY_POINT(nvmlDeviceSetFanSpeed_v2, tsapiDeviceSetFanSpeed_v2,
        (nvmlDevice_t device, unsigned int fan, unsigned int speed),
        "(%p, %u, %u)",
        device, fan, speed)

NVML_ENTRY_POINT(nvmlDeviceSetDefaultFanSpeed_v2, tsapiDeviceSetDefaultFanSpeed_v2,
        (nvmlDevice_t device, unsigned int fan),
        "(%p, %u)",
        device, fan)
NVML_ENTRY_POINT(nvmlDeviceGetThermalSettings, tsapiDeviceGetThermalSettings,
        (nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t *pThermalSettings),
        "(%p %u %p)",
        device, sensorIndex, pThermalSettings)
NVML_ENTRY_POINT(nvmlDeviceGetMinMaxClockOfPState, tsapiDeviceGetMinMaxClockOfPState,
        (nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int * minClockMHz, unsigned int * maxClockMHz),
        "(%p, %u, %u, %p, %p)",
        device, type, pstate, minClockMHz, maxClockMHz)

NVML_ENTRY_POINT(nvmlDeviceGetSupportedPerformanceStates, tsapiDeviceGetSupportedPerformanceStates,
        (nvmlDevice_t device, nvmlPstates_t *pstates, unsigned int size),
        "(%p, %p, %u)",
        device, pstates, size)

NVML_ENTRY_POINT(nvmlDeviceGetGpcClkVfOffset, tsapiDeviceGetGpcClkVfOffset,
        (nvmlDevice_t device, int *offset),
        "(%p, %p)",
        device, offset)

NVML_ENTRY_POINT(nvmlDeviceSetGpcClkVfOffset, tsapiDeviceSetGpcClkVfOffset,
        (nvmlDevice_t device, int offset),
        "(%p, %d)",
        device, offset)

NVML_ENTRY_POINT(nvmlDeviceGetMemClkVfOffset, tsapiDeviceGetMemClkVfOffset,
        (nvmlDevice_t device, int *offset),
        "(%p, %p)",
        device, offset)

NVML_ENTRY_POINT(nvmlDeviceSetMemClkVfOffset, tsapiDeviceSetMemClkVfOffset,
        (nvmlDevice_t device, int offset),
        "(%p, %d)",
        device, offset)

NVML_ENTRY_POINT(nvmlDeviceGetMinMaxFanSpeed, tsapiDeviceGetMinMaxFanSpeed,
        (nvmlDevice_t device, unsigned int *min, unsigned int * max),
        "(%p, %p, %p)",
        device, min, max)

NVML_ENTRY_POINT(nvmlDeviceGetGpcClkMinMaxVfOffset, tsapiDeviceGetGpcClkMinMaxVfOffset,
        (nvmlDevice_t device, int *minOffset, int *maxOffset),
        "(%p, %p, %p)",
        device, minOffset, maxOffset)

NVML_ENTRY_POINT(nvmlDeviceGetMemClkMinMaxVfOffset, tsapiDeviceGetMemClkMinMaxVfOffset,
        (nvmlDevice_t device, int *minOffset, int *maxOffset),
        "(%p, %p, %p)",
        device, minOffset, maxOffset)

NVML_ENTRY_POINT(nvmlGpmMetricsGet, tsapiGpmMetricsGet,
        (nvmlGpmMetricsGet_t *metricsGet),
        "(%p)",
        metricsGet)

NVML_ENTRY_POINT(nvmlGpmSampleAlloc, tsapiGpmSampleAlloc,
        (nvmlGpmSample_t *gpmSample),
        "(%p)",
        gpmSample)

NVML_ENTRY_POINT(nvmlGpmSampleFree, tsapiGpmSampleFree,
        (nvmlGpmSample_t gpmSample),
        "(%p)",
        gpmSample)

NVML_ENTRY_POINT(nvmlGpmSampleGet, tsapiGpmSampleGet,
        (nvmlDevice_t device, nvmlGpmSample_t gpmSample),
        "(%p, %p)",
        device, gpmSample)

NVML_ENTRY_POINT(nvmlGpmMigSampleGet, tsapiGpmMigSampleGet,
        (nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample),
        "(%p, %d, %p)",
        device, gpuInstanceId, gpmSample)

NVML_ENTRY_POINT(nvmlGpmQueryDeviceSupport, tsapiGpmQueryDeviceSupport,
        (nvmlDevice_t device, nvmlGpmSupport_t *gpmSupport),
        "(%p, %p)",
        device, gpmSupport)

NVML_ENTRY_POINT(nvmlDeviceGetGpuFabricInfo,
                 tsapiDeviceGetGpuFabricInfo,
                 (nvmlDevice_t device, nvmlGpuFabricInfo_t *gpuFabricInfo),
                 "(%p, %p)",
                 device, gpuFabricInfo);

NVML_ENTRY_POINT(nvmlDeviceGetGpuFabricInfoV, tsapiDeviceGetGpuFabricInfoV,
                 (nvmlDevice_t device, nvmlGpuFabricInfoV_t *gpuFabricInfo),
                 "(%p, %p)",
                 device, gpuFabricInfo);

NVML_ENTRY_POINT(nvmlDeviceGetCount_v2, tsapiDeviceGetCount_v2,
        (unsigned int *deviceCount),
        "(%p)",
        deviceCount)
NVML_ENTRY_POINT(nvmlDeviceGetHandleByIndex_v2, tsapiDeviceGetHandleByIndex_v2,
        (unsigned int index, nvmlDevice_t *device),
        "(%d, %p)",
        index, device)
NVML_ENTRY_POINT(nvmlDeviceGetArchitecture, tsapiDeviceGetArchitecture,
        (nvmlDevice_t device, nvmlDeviceArchitecture_t *arch),
        "(%p, %p)",
        device, arch)

NVML_ENTRY_POINT(nvmlDeviceWorkloadPowerProfileGetProfilesInfo,
                 tsapiDeviceWorkloadPowerProfileGetProfilesInfo,
                 (nvmlDevice_t device, nvmlWorkloadPowerProfileProfilesInfo_t *profilesInfo),
                 "(%p, %p)",
                 device, profilesInfo);
NVML_ENTRY_POINT(nvmlDeviceWorkloadPowerProfileGetCurrentProfiles,
                 tsapiDeviceWorkloadPowerProfileGetCurrentProfiles,
                 (nvmlDevice_t device, nvmlWorkloadPowerProfileCurrentProfiles_t *currentProfiles),
                 "(%p, %p)",
                 device, currentProfiles);
NVML_ENTRY_POINT(nvmlDeviceWorkloadPowerProfileSetRequestedProfiles,
                 tsapiDeviceWorkloadPowerProfileSetRequestedProfiles,
                 (nvmlDevice_t device, nvmlWorkloadPowerProfileRequestedProfiles_t *requestedProfiles),
                 "(%p, %p)",
                 device, requestedProfiles);
NVML_ENTRY_POINT(nvmlDeviceWorkloadPowerProfileClearRequestedProfiles,
                 tsapiDeviceWorkloadPowerProfileClearRequestedProfiles,
                 (nvmlDevice_t device, nvmlWorkloadPowerProfileRequestedProfiles_t *requestedProfiles),
                 "(%p, %p)",
                 device, requestedProfiles);

NVML_ENTRY_POINT(nvmlDeviceGetPlatformInfo,
                 tsapiDeviceGetPlatformInfo,
                 (nvmlDevice_t device, nvmlPlatformInfo_t *platformInfo),
                 "(%p, %p)",
                 device, platformInfo);
