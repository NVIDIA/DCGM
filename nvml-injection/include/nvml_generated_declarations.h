/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

/*
 * NOTE: This code is auto-generated by generate_nvml_stubs.py
 * DO NOT EDIT MANUALLY
 */


#pragma once

#include <nvml.h>

#define MAX_NVML_ARGS 20
typedef struct
{
    const char *funcname;
    unsigned int argCount;
    injectionArgType_t argTypes[MAX_NVML_ARGS];
} functionInfo_t;

// clang-format off
typedef nvmlReturn_t (*nvmlDeviceGetClockInfo_f)(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock) ;
typedef nvmlReturn_t (*nvmlDeviceGetComputeMode_f)(nvmlDevice_t device, nvmlComputeMode_t *mode) ;
typedef nvmlReturn_t (*nvmlDeviceSetComputeMode_f)(nvmlDevice_t device, nvmlComputeMode_t mode) ;
typedef nvmlReturn_t (*nvmlDeviceGetCudaComputeCapability_f)(nvmlDevice_t device, int *major, int *minor) ;
typedef nvmlReturn_t (*nvmlDeviceSetDriverModel_f)(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) ;
typedef nvmlReturn_t (*nvmlDeviceGetDriverModel_f)(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending) ;
typedef nvmlReturn_t (*nvmlDeviceGetCount_f)(unsigned int *deviceCount) ;
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_f)(unsigned int index, nvmlDevice_t *device) ;
typedef nvmlReturn_t (*nvmlDeviceGetHandleBySerial_f)(const char *serial, nvmlDevice_t *device) ;
typedef nvmlReturn_t (*nvmlDeviceGetInforomVersion_f)(nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length) ;
typedef nvmlReturn_t (*nvmlDeviceGetInforomImageVersion_f)(nvmlDevice_t device, char *version, unsigned int length) ;
typedef nvmlReturn_t (*nvmlDeviceGetDisplayMode_f)(nvmlDevice_t device, nvmlEnableState_t *mode) ;
typedef nvmlReturn_t (*nvmlDeviceGetEccMode_f)(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending) ;
typedef nvmlReturn_t (*nvmlDeviceGetBoardId_f)(nvmlDevice_t device, unsigned int *boardId) ;
typedef nvmlReturn_t (*nvmlDeviceGetDetailedEccErrors_f)(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts) ;
typedef nvmlReturn_t (*nvmlDeviceGetTotalEccErrors_f)(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts) ;
typedef nvmlReturn_t (*nvmlDeviceSetEccMode_f)(nvmlDevice_t device, nvmlEnableState_t ecc) ;
typedef nvmlReturn_t (*nvmlDeviceClearEccErrorCounts_f)(nvmlDevice_t device, nvmlEccCounterType_t counterType) ;
typedef nvmlReturn_t (*nvmlDeviceGetBrand_f)(nvmlDevice_t device, nvmlBrandType_t *type) ;
typedef nvmlReturn_t (*nvmlDeviceGetMemoryAffinity_f)(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope) ;
typedef nvmlReturn_t (*nvmlDeviceGetCpuAffinity_f)(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet) ;
typedef nvmlReturn_t (*nvmlDeviceSetCpuAffinity_f)(nvmlDevice_t device) ;
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_f)(nvmlDevice_t device, nvmlMemory_t *memory) ;
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_v2_f)(nvmlDevice_t device, nvmlMemory_v2_t *memory) ;
typedef nvmlReturn_t (*nvmlDeviceGetPciInfo_f)(nvmlDevice_t device, nvmlPciInfo_t *pci) ;
typedef nvmlReturn_t (*nvmlDeviceGetBAR1MemoryInfo_f)(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory) ;
typedef nvmlReturn_t (*nvmlDeviceGetViolationStatus_f)(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime) ;
typedef nvmlReturn_t (*nvmlDeviceGetPowerState_f)(nvmlDevice_t device, nvmlPstates_t *pState) ;
typedef nvmlReturn_t (*nvmlDeviceSetPowerMode_f)(nvmlDevice_t device, unsigned int powerModeId) ;
typedef nvmlReturn_t (*nvmlDeviceGetTotalEnergyConsumption_f)(nvmlDevice_t device, unsigned long long *energy) ;
typedef nvmlReturn_t (*nvmlDeviceGetTemperature_f)(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp) ;
typedef nvmlReturn_t (*nvmlDeviceGetTemperatureThreshold_f)(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp) ;
typedef nvmlReturn_t (*nvmlDeviceSetTemperatureThreshold_f)(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp) ;
typedef nvmlReturn_t (*nvmlDeviceGetFanSpeed_v2_f)(nvmlDevice_t device, unsigned int fan, unsigned int *speed) ;
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_f)(nvmlDevice_t device, nvmlUtilization_t *utilization) ;
typedef nvmlReturn_t (*nvmlDeviceGetEncoderUtilization_f)(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs) ;
typedef nvmlReturn_t (*nvmlSystemGetDriverVersion_f)(char *version, unsigned int length) ;
typedef nvmlReturn_t (*nvmlSystemGetCudaDriverVersion_f)(int *cudaDriverVersion) ;
typedef nvmlReturn_t (*nvmlUnitGetFanSpeedInfo_f)(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds) ;
typedef nvmlReturn_t (*nvmlUnitGetHandleByIndex_f)(unsigned int index, nvmlUnit_t *unit) ;
typedef nvmlReturn_t (*nvmlUnitGetLedState_f)(nvmlUnit_t unit, nvmlLedState_t *state) ;
typedef nvmlReturn_t (*nvmlUnitSetLedState_f)(nvmlUnit_t unit, nvmlLedColor_t color) ;
typedef nvmlReturn_t (*nvmlUnitGetPsuInfo_f)(nvmlUnit_t unit, nvmlPSUInfo_t *psu) ;
typedef nvmlReturn_t (*nvmlUnitGetTemperature_f)(nvmlUnit_t unit, unsigned int type, unsigned int *temp) ;
typedef nvmlReturn_t (*nvmlUnitGetUnitInfo_f)(nvmlUnit_t unit, nvmlUnitInfo_t *info) ;
typedef nvmlReturn_t (*nvmlUnitGetDevices_f)(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices) ;
typedef nvmlReturn_t (*nvmlDeviceGetBridgeChipInfo_f)(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy) ;
typedef nvmlReturn_t (*nvmlSystemGetHicVersion_f)(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries) ;
typedef nvmlReturn_t (*nvmlEventSetCreate_f)(nvmlEventSet_t *set) ;
typedef nvmlReturn_t (*nvmlDeviceRegisterEvents_f)(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) ;
typedef nvmlReturn_t (*nvmlEventSetWait_f)(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms) ;
typedef nvmlReturn_t (*nvmlEventSetFree_f)(nvmlEventSet_t set) ;
typedef nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses_f)(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v1_t *infos) ;
typedef nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses_v2_f)(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_v2_t *infos) ;
typedef nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses_v3_f)(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) ;
typedef nvmlReturn_t (*nvmlSystemGetProcessName_f)(unsigned int pid, char *name, unsigned int length) ;
typedef nvmlReturn_t (*nvmlDeviceOnSameBoard_f)(nvmlDevice_t dev1, nvmlDevice_t dev2, int *onSameBoard) ;
typedef nvmlReturn_t (*nvmlDeviceGetGpuOperationMode_f)(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending) ;
typedef nvmlReturn_t (*nvmlDeviceSetGpuOperationMode_f)(nvmlDevice_t device, nvmlGpuOperationMode_t mode) ;
typedef nvmlReturn_t (*nvmlDeviceGetMemoryErrorCounter_f)(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long *count) ;
typedef nvmlReturn_t (*nvmlDeviceSetGpuLockedClocks_f)(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) ;
typedef nvmlReturn_t (*nvmlDeviceGetClock_f)(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz) ;
typedef nvmlReturn_t (*nvmlDeviceGetSupportedGraphicsClocks_f)(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz) ;
typedef nvmlReturn_t (*nvmlDeviceSetDefaultAutoBoostedClocksEnabled_f)(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) ;
typedef nvmlReturn_t (*nvmlDeviceGetAccountingStats_f)(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats) ;
typedef nvmlReturn_t (*nvmlDeviceGetRetiredPages_f)(nvmlDevice_t device, nvmlPageRetirementCause_t sourceFilter, unsigned int *count, unsigned long long *addresses) ;
typedef nvmlReturn_t (*nvmlDeviceGetRetiredPages_v2_f)(nvmlDevice_t device, nvmlPageRetirementCause_t sourceFilter, unsigned int *count, unsigned long long *addresses, unsigned long long *timestamps) ;
typedef nvmlReturn_t (*nvmlDeviceSetAPIRestriction_f)(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) ;
typedef nvmlReturn_t (*nvmlDeviceGetAPIRestriction_f)(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted) ;
typedef nvmlReturn_t (*nvmlDeviceGetSamples_f)(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples) ;
typedef nvmlReturn_t (*nvmlDeviceGetPcieThroughput_f)(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value) ;
typedef nvmlReturn_t (*nvmlDeviceGetTopologyCommonAncestor_f)(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo) ;
typedef nvmlReturn_t (*nvmlDeviceGetTopologyNearestGpus_f)(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray) ;
typedef nvmlReturn_t (*nvmlSystemGetTopologyGpuSet_f)(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray) ;
typedef nvmlReturn_t (*nvmlDeviceGetNvLinkState_f)(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) ;
typedef nvmlReturn_t (*nvmlDeviceGetP2PStatus_f)(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus) ;
typedef nvmlReturn_t (*nvmlDeviceGetNvLinkRemotePciInfo_f)(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) ;
typedef nvmlReturn_t (*nvmlDeviceGetNvLinkRemoteDeviceType_f)(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType) ;
typedef nvmlReturn_t (*nvmlDeviceGetNvLinkCapability_f)(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult) ;
typedef nvmlReturn_t (*nvmlDeviceGetNvLinkErrorCounter_f)(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue) ;
typedef nvmlReturn_t (*nvmlDeviceSetNvLinkUtilizationControl_f)(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control, unsigned int reset) ;
typedef nvmlReturn_t (*nvmlDeviceGetNvLinkUtilizationControl_f)(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control) ;
typedef nvmlReturn_t (*nvmlDeviceGetNvLinkUtilizationCounter_f)(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter) ;
typedef nvmlReturn_t (*nvmlDeviceFreezeNvLinkUtilizationCounter_f)(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze) ;
typedef nvmlReturn_t (*nvmlDeviceGetVirtualizationMode_f)(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode) ;
typedef nvmlReturn_t (*nvmlDeviceSetVirtualizationMode_f)(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) ;
typedef nvmlReturn_t (*nvmlDeviceGetSupportedVgpus_f)(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds) ;
typedef nvmlReturn_t (*nvmlVgpuTypeGetClass_f)(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size) ;
typedef nvmlReturn_t (*nvmlVgpuTypeGetGpuInstanceProfileId_f)(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId) ;
typedef nvmlReturn_t (*nvmlVgpuTypeGetDeviceID_f)(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID) ;
typedef nvmlReturn_t (*nvmlVgpuTypeGetFramebufferSize_f)(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize) ;
typedef nvmlReturn_t (*nvmlVgpuTypeGetResolution_f)(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim) ;
typedef nvmlReturn_t (*nvmlVgpuTypeGetLicense_f)(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size) ;
typedef nvmlReturn_t (*nvmlVgpuTypeGetMaxInstances_f)(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount) ;
typedef nvmlReturn_t (*nvmlDeviceGetActiveVgpus_f)(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetVmID_f)(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetUUID_f)(nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetFbUsage_f)(nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetLicenseStatus_f)(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetLicenseInfo_f)(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetEccMode_f)(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceSetEncoderCapacity_f)(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) ;
typedef nvmlReturn_t (*nvmlDeviceGetVgpuUtilization_f)(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetMetadata_f)(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetGpuPciId_f)(nvmlVgpuInstance_t vgpuInstance, char *vgpuPciId, unsigned int *length) ;
typedef nvmlReturn_t (*nvmlVgpuTypeGetCapabilities_f)(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int *capResult) ;
typedef nvmlReturn_t (*nvmlDeviceGetGspFirmwareVersion_f)(nvmlDevice_t device, char *version) ;
typedef nvmlReturn_t (*nvmlDeviceGetVgpuMetadata_f)(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize) ;
typedef nvmlReturn_t (*nvmlGetVgpuCompatibility_f)(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo) ;
typedef nvmlReturn_t (*nvmlDeviceGetPgpuMetadataString_f)(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize) ;
typedef nvmlReturn_t (*nvmlDeviceGetGridLicensableFeatures_f)(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) ;
typedef nvmlReturn_t (*nvmlDeviceGetEncoderCapacity_f)(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *pEncoderCapacity) ;
typedef nvmlReturn_t (*nvmlDeviceGetEncoderStats_f)(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) ;
typedef nvmlReturn_t (*nvmlDeviceGetEncoderSessions_f)(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos) ;
typedef nvmlReturn_t (*nvmlDeviceGetFBCStats_f)(nvmlDevice_t device, nvmlFBCStats_t *fbcStats) ;
typedef nvmlReturn_t (*nvmlDeviceGetFBCSessions_f)(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) ;
typedef nvmlReturn_t (*nvmlDeviceModifyDrainState_f)(nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState) ;
typedef nvmlReturn_t (*nvmlDeviceQueryDrainState_f)(nvmlPciInfo_t *pciInfo, nvmlEnableState_t *newState) ;
typedef nvmlReturn_t (*nvmlDeviceRemoveGpu_f)(nvmlPciInfo_t *pciInfo) ;
typedef nvmlReturn_t (*nvmlDeviceRemoveGpu_v2_f)(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) ;
typedef nvmlReturn_t (*nvmlDeviceGetFieldValues_f)(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values) ;
typedef nvmlReturn_t (*nvmlDeviceGetVgpuProcessUtilization_f)(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t *utilizationSamples) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetEncoderStats_f)(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetEncoderSessions_f)(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetFBCStats_f)(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetFBCSessions_f)(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo) ;
typedef nvmlReturn_t (*nvmlDeviceGetProcessUtilization_f)(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetAccountingPids_f)(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceGetAccountingStats_f)(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats) ;
typedef nvmlReturn_t (*nvmlVgpuInstanceClearAccountingPids_f)(nvmlVgpuInstance_t vgpuInstance) ;
typedef nvmlReturn_t (*nvmlGetExcludedDeviceInfoByIndex_f)(unsigned int index, nvmlExcludedDeviceInfo_t *info) ;
typedef nvmlReturn_t (*nvmlGetVgpuVersion_f)(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current) ;
typedef nvmlReturn_t (*nvmlSetVgpuVersion_f)(nvmlVgpuVersion_t *vgpuVersion) ;
typedef nvmlReturn_t (*nvmlDeviceGetHostVgpuMode_f)(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode) ;
typedef nvmlReturn_t (*nvmlDeviceSetMigMode_f)(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus) ;
typedef nvmlReturn_t (*nvmlDeviceGetGpuInstanceProfileInfo_f)(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info) ;
typedef nvmlReturn_t (*nvmlDeviceGetGpuInstanceProfileInfoV_f)(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t *info) ;
typedef nvmlReturn_t (*nvmlDeviceGetGpuInstancePossiblePlacements_f)(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count) ;
typedef nvmlReturn_t (*nvmlDeviceCreateGpuInstance_f)(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance) ;
typedef nvmlReturn_t (*nvmlDeviceCreateGpuInstanceWithPlacement_f)(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement, nvmlGpuInstance_t *gpuInstance) ;
typedef nvmlReturn_t (*nvmlGpuInstanceDestroy_f)(nvmlGpuInstance_t gpuInstance) ;
typedef nvmlReturn_t (*nvmlDeviceGetGpuInstances_f)(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *instances, unsigned int *count) ;
typedef nvmlReturn_t (*nvmlGpuInstanceGetInfo_f)(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info) ;
typedef nvmlReturn_t (*nvmlGpuInstanceGetComputeInstanceProfileInfo_f)(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info) ;
typedef nvmlReturn_t (*nvmlGpuInstanceGetComputeInstanceProfileInfoV_f)(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t *info) ;
typedef nvmlReturn_t (*nvmlGpuInstanceGetComputeInstanceRemainingCapacity_f)(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count) ;
typedef nvmlReturn_t (*nvmlGpuInstanceCreateComputeInstance_f)(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance) ;
typedef nvmlReturn_t (*nvmlComputeInstanceDestroy_f)(nvmlComputeInstance_t computeInstance) ;
typedef nvmlReturn_t (*nvmlGpuInstanceGetComputeInstances_f)(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstances, unsigned int *count) ;
typedef nvmlReturn_t (*nvmlComputeInstanceGetInfo_f)(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info) ;
typedef nvmlReturn_t (*nvmlDeviceGetMigDeviceHandleByIndex_f)(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice) ;
typedef nvmlReturn_t (*nvmlSystemGetConfComputeState_f)(nvmlConfComputeSystemState_t *ccMode) ;
typedef nvmlReturn_t (*nvmlDeviceGetDeviceHandleFromMigDeviceHandle_f)(nvmlDevice_t migDevice, nvmlDevice_t *device) ;
typedef nvmlReturn_t (*nvmlDeviceGetAttributes_f)(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes) ;
typedef nvmlReturn_t (*nvmlDeviceGetRemappedRows_f)(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred) ;
typedef nvmlReturn_t (*nvmlDeviceGetRowRemapperHistogram_f)(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values) ;
typedef nvmlReturn_t (*nvmlDeviceGetBusType_f)(nvmlDevice_t device, nvmlBusType_t *type) ;
typedef nvmlReturn_t (*nvmlDeviceGetPowerSource_f)(nvmlDevice_t device, nvmlPowerSource_t *powerSource) ;
typedef nvmlReturn_t (*nvmlDeviceGetDynamicPstatesInfo_f)(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t *pDynamicPstatesInfo) ;
typedef nvmlReturn_t (*nvmlDeviceGetThermalSettings_f)(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t *pThermalSettings) ;
typedef nvmlReturn_t (*nvmlDeviceGetMinMaxClockOfPState_f)(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int *minClockMHz, unsigned int *maxClockMHz) ;
typedef nvmlReturn_t (*nvmlDeviceGetSupportedPerformanceStates_f)(nvmlDevice_t device, nvmlPstates_t *pstates, unsigned int size) ;
typedef nvmlReturn_t (*nvmlDeviceGetGpcClkVfOffset_f)(nvmlDevice_t device, int *offset) ;
typedef nvmlReturn_t (*nvmlDeviceSetGpcClkVfOffset_f)(nvmlDevice_t device, int offset) ;
typedef nvmlReturn_t (*nvmlGpmMetricsGet_f)(nvmlGpmMetricsGet_t *metricsGet) ;
typedef nvmlReturn_t (*nvmlGpmSampleAlloc_f)(nvmlGpmSample_t *gpmSample) ;
typedef nvmlReturn_t (*nvmlGpmSampleFree_f)(nvmlGpmSample_t gpmSample) ;
typedef nvmlReturn_t (*nvmlGpmSampleGet_f)(nvmlDevice_t device, nvmlGpmSample_t gpmSample) ;
typedef nvmlReturn_t (*nvmlGpmMigSampleGet_f)(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample) ;
typedef nvmlReturn_t (*nvmlGpmQueryDeviceSupport_f)(nvmlDevice_t device, nvmlGpmSupport_t *gpmSupport) ;
typedef nvmlReturn_t (*nvmlDeviceGetArchitecture_f)(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch) ;
