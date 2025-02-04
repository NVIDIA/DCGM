//
// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#ifndef __NVSDM_API_MAP_H__
#define __NVSDM_API_MAP_H__

#include "nvsdm.h"

NVSDM_FUNCTION(nvsdmRet_t, nvsdmInitialize, (void));
NVSDM_FUNCTION(nvsdmRet_t, nvsdmFinalize, (void));
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDiscoverTopology, (char *srcCA, int srcPort), srcCA, srcPort);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetAllDevices, (nvsdmDeviceIter_t *iter), iter);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetAllDevicesOfType, (int type, nvsdmDeviceIter_t *iter), type, iter);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetDeviceGivenGUID, (uint64_t guid, nvsdmDevice_t *dev), guid, dev);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetDeviceGivenLID,
                (uint16_t lid, nvsdmDevice_t *dev),
                lid, dev);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetNextDevice, (nvsdmDeviceIter_t iter, nvsdmDevice_t *device), iter, device);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmIterateDevices, (nvsdmDeviceIter_t iter, nvsdmRet_t (*callback)(nvsdmDevice_t const, void *), void *cbData), iter, callback, cbData);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmResetDeviceList, (nvsdmDeviceIter_t iter), iter);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetPorts, (nvsdmDevice_t const device, nvsdmPortIter_t *iter), device, iter);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortGetDevice, (nvsdmPort_t const port, nvsdmDevice_t *device), port, device);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetAllPorts, (nvsdmPortIter_t *iter), iter);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetNextPort, (nvsdmPortIter_t iter, nvsdmPort_t *port), iter, port);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmIteratePorts, (nvsdmPortIter_t iter, nvsdmRet_t (*callback)(nvsdmPort_t const, void *), void *cbData), iter, callback, cbData);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortGetRemote, (nvsdmPort_t const port, nvsdmPort_t *remote), port, remote);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortGetInfo, (nvsdmPort_t const port, nvsdmPortInfo_t *info), port, info);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetConnectedDevices, (nvsdmDevice_t const device, nvsdmDeviceIter_t *iter), device, iter);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetConnectedPorts, (nvsdmDevice_t const device, nvsdmPortIter_t *iter), device, iter);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmResetPortList, (nvsdmPortIter_t iter), iter);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetSwitchInfo,
                (nvsdmDevice_t const device, nvsdmSwitchInfo_t *info),
                device, info);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortGetTelemetryValues, (nvsdmPort_t const port, nvsdmTelemParam_t *param), port, param);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetTelemetryValues,
               (nvsdmDevice_t const device, nvsdmTelemParam_t *param),
               device, param);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetName, (nvsdmDevice_t const device, char str[], unsigned int strSize), device, str, strSize);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetShortName, (nvsdmDevice_t const device, char str[], unsigned int strSize), device, str, strSize);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortGetNum, (nvsdmPort_t const port, unsigned int *num), port, num);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortGetLID, (nvsdmPort_t const port, uint16_t *lid), port, lid);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortGetGUID, (nvsdmPort_t const port, uint64_t *guid), port, guid);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortGetGID, (nvsdmPort_t const port, uint8_t gid[16]), port, gid);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortToString, (nvsdmPort_t const port, char str[], unsigned int strSize), port, str, strSize);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmPortExecuteMAD, (nvsdmPort_t const dest, int mgmtClass, uint8_t *outBuff, uint8_t const *mad), dest, mgmtClass, outBuff, mad);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetGUID, (nvsdmDevice_t const device, uint64_t *guid), device, guid);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetLID,
                (nvsdmDevice_t const device, uint16_t *lid),
                device, lid);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetVendorID, (nvsdmDevice_t const device, uint32_t *vendorID), device, vendorID);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetDevID, (nvsdmDevice_t const device, uint16_t *devID), device, devID);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceToString, (nvsdmDevice_t const device, char str[], unsigned int strSize), device, str, strSize);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetType, (nvsdmDevice_t const device, unsigned int *type), device, type);

NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetVersion, (uint64_t *version), version);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetFirmwareVersion,
               (nvsdmDevice_t const device, nvsdmVersionInfo_t *version),
               device, version);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetSupportedInterfaceVersionRange, (uint64_t *versionFrom, uint64_t *versionTo), versionFrom, versionTo);

NVSDM_FUNCTION(nvsdmRet_t, nvsdmGetLogLevel, (uint32_t *logLevel), logLevel);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmSetLogLevel, (uint32_t logLevel), logLevel);
NVSDM_FUNCTION(nvsdmRet_t, nvsdmSetLogFile, (char const *logFile), logFile);

NVSDM_FUNCTION(nvsdmRet_t, nvsdmDeviceGetHealthStatus,
               (nvsdmDevice_t const device, nvsdmDeviceHealthStatus_t *status), device, status);

#endif // __NVSDM_API_MAP_H__
