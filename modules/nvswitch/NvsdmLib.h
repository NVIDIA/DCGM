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

#include <string>
#include <unordered_map>
#include <vector>

#include <nvsdm.h>

namespace DcgmNs
{

class NvsdmBase
{
public:
    virtual ~NvsdmBase() = default;

    virtual nvsdmRet_t nvsdmInitialize();
    virtual nvsdmRet_t nvsdmFinalize();
    virtual nvsdmRet_t nvsdmDiscoverTopology(char *srcCA, int srcPort);
    virtual nvsdmRet_t nvsdmGetAllDevices(nvsdmDeviceIter_t *iter);
    virtual nvsdmRet_t nvsdmGetAllDevicesOfType(int type, nvsdmDeviceIter_t *iter);
    virtual nvsdmRet_t nvsdmGetDeviceGivenGUID(uint64_t guid, nvsdmDevice_t *dev);
    virtual nvsdmRet_t nvsdmGetNextDevice(nvsdmDeviceIter_t iter, nvsdmDevice_t *device);
    virtual nvsdmRet_t nvsdmIterateDevices(nvsdmDeviceIter_t iter,
                                           nvsdmRet_t (*callback)(nvsdmDevice_t const, void *),
                                           void *cbData);
    virtual nvsdmRet_t nvsdmResetDeviceList(nvsdmDeviceIter_t iter);
    virtual nvsdmRet_t nvsdmDeviceGetPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter);
    virtual nvsdmRet_t nvsdmGetAllPorts(nvsdmPortIter_t *iter);
    virtual nvsdmRet_t nvsdmGetNextPort(nvsdmPortIter_t iter, nvsdmPort_t *port);
    virtual nvsdmRet_t nvsdmIteratePorts(nvsdmPortIter_t iter,
                                         nvsdmRet_t (*callback)(nvsdmPort_t const, void *),
                                         void *cbData);
    virtual nvsdmRet_t nvsdmPortGetDevice(nvsdmPort_t const port, nvsdmDevice_t *device);
    virtual nvsdmRet_t nvsdmPortGetRemote(nvsdmPort_t const port, nvsdmPort_t *remote);
    virtual nvsdmRet_t nvsdmPortGetInfo(nvsdmPort_t const port, nvsdmPortInfo_t *info);
    virtual nvsdmRet_t nvsdmDeviceGetConnectedDevices(nvsdmDevice_t const device, nvsdmDeviceIter_t *iter);
    virtual nvsdmRet_t nvsdmDeviceGetConnectedPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter);
    virtual nvsdmRet_t nvsdmResetPortList(nvsdmPortIter_t iter);
    virtual nvsdmRet_t nvsdmDeviceGetSwitchInfo(nvsdmDevice_t const device, nvsdmSwitchInfo_t *info);
    virtual nvsdmRet_t nvsdmPortGetTelemetryValues(nvsdmPort_t const port, nvsdmTelemParam_t *param);
    virtual nvsdmRet_t nvsdmDeviceGetTelemetryValues(nvsdmDevice_t const device, nvsdmTelemParam_t *param);
    virtual nvsdmRet_t nvsdmPortResetTelemetryCounters(nvsdmPort_t const port, nvsdmTelemParam_t *param);
    virtual nvsdmRet_t nvsdmPortExecuteMAD(nvsdmPort_t const dest,
                                           uint8_t *outBuff,
                                           unsigned int *bytesRead,
                                           uint8_t const *mad);
    virtual nvsdmRet_t nvsdmDeviceToString(nvsdmDevice_t const device, char str[], unsigned int strSize);
    virtual nvsdmRet_t nvsdmDeviceGetName(nvsdmDevice_t const device, char str[], unsigned int strSize);
    virtual nvsdmRet_t nvsdmDeviceGetShortName(nvsdmDevice_t const device, char str[], unsigned int strSize);
    virtual nvsdmRet_t nvsdmPortToString(nvsdmPort_t const port, char str[], unsigned int strSize);
    virtual nvsdmRet_t nvsdmPortGetNum(nvsdmPort_t const port, unsigned int *num);
    virtual nvsdmRet_t nvsdmPortGetLID(nvsdmPort_t const port, uint16_t *lid);
    virtual nvsdmRet_t nvsdmPortGetGID(nvsdmPort_t const port, uint8_t gid[16]);
    virtual nvsdmRet_t nvsdmPortGetGUID(nvsdmPort_t const port, uint64_t *guid);
    virtual nvsdmRet_t nvsdmDeviceGetDevID(nvsdmDevice_t const device, uint16_t *devID);
    virtual nvsdmRet_t nvsdmDeviceGetVendorID(nvsdmDevice_t const device, uint32_t *vendorID);
    virtual nvsdmRet_t nvsdmDeviceGetGUID(nvsdmDevice_t const device, uint64_t *guid);
    virtual nvsdmRet_t nvsdmDeviceGetType(nvsdmDevice_t const device, unsigned int *type);
    virtual nvsdmRet_t nvsdmDeviceGetHealthStatus(nvsdmDevice_t const device, nvsdmDeviceHealthStatus_t *status);
    virtual nvsdmRet_t nvsdmGetVersion(uint64_t *version);
    virtual nvsdmRet_t nvsdmGetSupportedInterfaceVersionRange(uint64_t *versionFrom, uint64_t *versionTo);
    virtual nvsdmRet_t nvsdmSetLogLevel(uint32_t logLevel);
    virtual char const *nvsdmGetErrorString(nvsdmRet_t ret);
    virtual nvsdmRet_t nvsdmDumpPortTelemValues(int lid, int portNum);
};

class NvsdmLib final : public NvsdmBase
{
public:
    virtual ~NvsdmLib() = default;

    virtual nvsdmRet_t nvsdmInitialize() override;
    virtual nvsdmRet_t nvsdmFinalize() override;
    virtual nvsdmRet_t nvsdmDiscoverTopology(char *srcCA, int srcPort) override;
    virtual nvsdmRet_t nvsdmGetAllDevices(nvsdmDeviceIter_t *iter) override;
    virtual nvsdmRet_t nvsdmGetAllDevicesOfType(int type, nvsdmDeviceIter_t *iter) override;
    virtual nvsdmRet_t nvsdmGetDeviceGivenGUID(uint64_t guid, nvsdmDevice_t *dev) override;
    virtual nvsdmRet_t nvsdmGetNextDevice(nvsdmDeviceIter_t iter, nvsdmDevice_t *device) override;
    virtual nvsdmRet_t nvsdmIterateDevices(nvsdmDeviceIter_t iter,
                                           nvsdmRet_t (*callback)(nvsdmDevice_t const, void *),
                                           void *cbData) override;
    virtual nvsdmRet_t nvsdmResetDeviceList(nvsdmDeviceIter_t iter) override;
    virtual nvsdmRet_t nvsdmDeviceGetPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter) override;
    virtual nvsdmRet_t nvsdmGetAllPorts(nvsdmPortIter_t *iter) override;
    virtual nvsdmRet_t nvsdmGetNextPort(nvsdmPortIter_t iter, nvsdmPort_t *port) override;
    virtual nvsdmRet_t nvsdmIteratePorts(nvsdmPortIter_t iter,
                                         nvsdmRet_t (*callback)(nvsdmPort_t const, void *),
                                         void *cbData) override;
    virtual nvsdmRet_t nvsdmPortGetDevice(nvsdmPort_t const port, nvsdmDevice_t *device) override;
    virtual nvsdmRet_t nvsdmPortGetRemote(nvsdmPort_t const port, nvsdmPort_t *remote) override;
    virtual nvsdmRet_t nvsdmPortGetInfo(nvsdmPort_t const port, nvsdmPortInfo_t *info) override;
    virtual nvsdmRet_t nvsdmDeviceGetConnectedDevices(nvsdmDevice_t const device, nvsdmDeviceIter_t *iter) override;
    virtual nvsdmRet_t nvsdmDeviceGetConnectedPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter) override;
    virtual nvsdmRet_t nvsdmResetPortList(nvsdmPortIter_t iter) override;
    virtual nvsdmRet_t nvsdmDeviceGetSwitchInfo(nvsdmDevice_t const device, nvsdmSwitchInfo_t *info) override;
    virtual nvsdmRet_t nvsdmPortGetTelemetryValues(nvsdmPort_t const port, nvsdmTelemParam_t *param) override;
    virtual nvsdmRet_t nvsdmDeviceGetTelemetryValues(nvsdmDevice_t const device, nvsdmTelemParam_t *param) override;
    virtual nvsdmRet_t nvsdmPortResetTelemetryCounters(nvsdmPort_t const port, nvsdmTelemParam_t *param) override;
    virtual nvsdmRet_t nvsdmPortExecuteMAD(nvsdmPort_t const dest,
                                           uint8_t *outBuff,
                                           unsigned int *bytesRead,
                                           uint8_t const *mad) override;
    virtual nvsdmRet_t nvsdmDeviceToString(nvsdmDevice_t const device, char str[], unsigned int strSize) override;
    virtual nvsdmRet_t nvsdmDeviceGetName(nvsdmDevice_t const device, char str[], unsigned int strSize) override;
    virtual nvsdmRet_t nvsdmDeviceGetShortName(nvsdmDevice_t const device, char str[], unsigned int strSize) override;
    virtual nvsdmRet_t nvsdmPortToString(nvsdmPort_t const port, char str[], unsigned int strSize) override;
    virtual nvsdmRet_t nvsdmPortGetNum(nvsdmPort_t const port, unsigned int *num) override;
    virtual nvsdmRet_t nvsdmPortGetLID(nvsdmPort_t const port, uint16_t *lid) override;
    virtual nvsdmRet_t nvsdmPortGetGID(nvsdmPort_t const port, uint8_t gid[16]) override;
    virtual nvsdmRet_t nvsdmPortGetGUID(nvsdmPort_t const port, uint64_t *guid) override;
    virtual nvsdmRet_t nvsdmDeviceGetDevID(nvsdmDevice_t const device, uint16_t *devID) override;
    virtual nvsdmRet_t nvsdmDeviceGetVendorID(nvsdmDevice_t const device, uint32_t *vendorID) override;
    virtual nvsdmRet_t nvsdmDeviceGetGUID(nvsdmDevice_t const device, uint64_t *guid) override;
    virtual nvsdmRet_t nvsdmDeviceGetType(nvsdmDevice_t const device, unsigned int *type) override;
    virtual nvsdmRet_t nvsdmDeviceGetHealthStatus(nvsdmDevice_t const device,
                                                  nvsdmDeviceHealthStatus_t *status) override;
    virtual nvsdmRet_t nvsdmGetVersion(uint64_t *version) override;
    virtual nvsdmRet_t nvsdmGetSupportedInterfaceVersionRange(uint64_t *versionFrom, uint64_t *versionTo) override;
    virtual nvsdmRet_t nvsdmSetLogLevel(uint32_t logLevel) override;
    virtual char const *nvsdmGetErrorString(nvsdmRet_t ret) override;
    virtual nvsdmRet_t nvsdmDumpPortTelemValues(int lid, int portNum) override;
};

class NvsdmMockPort
{
public:
    NvsdmMockPort(unsigned int portNum, uint16_t lid);
    ~NvsdmMockPort() = default;

    unsigned int GetPortNum() const;
    uint16_t GetLid() const;
    std::string GetGid() const;
    uint64_t GetGuid() const;
    nvsdmPortInfo_t GetInfo() const;

    void SetPortState(nvsdmPortState_t const state);

    void SetFieldValue(nvsdmPortTelemCounter_t const field, nvsdmTelem_v1_t const value);
    nvsdmTelem_v1_t GetFieldValue(nvsdmPortTelemCounter_t const field) const;

    void SetFieldValue(nvsdmPlatformTelemCounter_t const field, nvsdmTelem_v1_t const value);
    nvsdmTelem_v1_t GetFieldValue(nvsdmPlatformTelemCounter_t const field) const;

private:
    unsigned int m_portNum;
    uint16_t m_lid;
    nvsdmPortState_t m_portState = NVSDM_PORT_STATE_ACTIVE;
    std::unordered_map<nvsdmPortTelemCounter_t, nvsdmTelem_v1_t> m_portFieldValues;
    std::unordered_map<nvsdmPlatformTelemCounter_t, nvsdmTelem_v1_t> m_platformFieldValues;
};

class NvsdmMockDevice
{
public:
    NvsdmMockDevice(nvsdmDevType type, uint16_t devID, uint32_t vendorID, uint32_t healthState);
    ~NvsdmMockDevice() = default;

    nvsdmDevType GetType() const;
    uint16_t GetDevID() const;
    uint32_t GetVendorID() const;
    uint64_t GetGuid() const;
    uint32_t GetHealthState() const;
    nvsdmPortIter_t GetPortsIterator();

    void AddPort(NvsdmMockPort const &port);

    void SetFieldValue(nvsdmConnectXTelemCounter_t const field, nvsdmTelem_v1_t const value);
    nvsdmTelem_v1_t GetFieldValue(nvsdmConnectXTelemCounter_t const field) const;

private:
    nvsdmDevType m_type;
    uint16_t m_devID;
    uint32_t m_vendorID;
    uint32_t m_healthState;
    std::vector<NvsdmMockPort> m_ports;
    std::unordered_map<nvsdmConnectXTelemCounter_t, nvsdmTelem_v1_t> m_fieldValues;
};

class NvsdmMock final : public NvsdmBase
{
public:
    virtual ~NvsdmMock() = default;

    virtual nvsdmRet_t nvsdmInitialize() override;
    virtual nvsdmRet_t nvsdmFinalize() override;
    virtual nvsdmRet_t nvsdmDiscoverTopology(char * /* srcCA */, int /* srcPort */) override;
    virtual nvsdmRet_t nvsdmGetAllDevices(nvsdmDeviceIter_t *iter) override;
    virtual nvsdmRet_t nvsdmIterateDevices(nvsdmDeviceIter_t iter,
                                           nvsdmRet_t (*callback)(nvsdmDevice_t const, void *),
                                           void *cbData) override;
    virtual nvsdmRet_t nvsdmPortGetTelemetryValues(nvsdmPort_t const port, nvsdmTelemParam_t *param) override;
    virtual nvsdmRet_t nvsdmDeviceGetTelemetryValues(nvsdmDevice_t const device, nvsdmTelemParam_t *param) override;
    virtual nvsdmRet_t nvsdmDeviceToString(nvsdmDevice_t const device, char str[], unsigned int strSize) override;
    virtual nvsdmRet_t nvsdmDeviceGetType(nvsdmDevice_t const device, unsigned int *type) override;
    virtual nvsdmRet_t nvsdmDeviceGetDevID(nvsdmDevice_t const device, uint16_t *devID) override;
    virtual nvsdmRet_t nvsdmDeviceGetVendorID(nvsdmDevice_t const device, uint32_t *vendorID) override;
    virtual nvsdmRet_t nvsdmDeviceGetGUID(nvsdmDevice_t const device, uint64_t *guid) override;
    virtual nvsdmRet_t nvsdmDeviceGetHealthStatus(nvsdmDevice_t const device,
                                                  nvsdmDeviceHealthStatus_t *status) override;
    virtual nvsdmRet_t nvsdmDeviceGetPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter) override;
    virtual nvsdmRet_t nvsdmIteratePorts(nvsdmPortIter_t iter,
                                         nvsdmRet_t (*callback)(nvsdmPort_t const, void *),
                                         void *cbData) override;
    virtual nvsdmRet_t nvsdmPortGetNum(nvsdmPort_t const port, unsigned int *num) override;
    virtual nvsdmRet_t nvsdmPortGetLID(nvsdmPort_t const port, uint16_t *lid) override;
    virtual nvsdmRet_t nvsdmPortGetGID(nvsdmPort_t const port, uint8_t gid[16]) override;
    virtual nvsdmRet_t nvsdmPortGetGUID(nvsdmPort_t const port, uint64_t *guid) override;
    virtual nvsdmRet_t nvsdmPortGetInfo(nvsdmPort_t const port, nvsdmPortInfo_t *info) override;

    void InjectDevice(NvsdmMockDevice const &dev);
    bool LoadYaml(std::string const &path);

private:
    bool m_init = false;
    std::vector<NvsdmMockDevice> m_devices;
};

} //namespace DcgmNs