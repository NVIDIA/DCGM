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

#include "NvsdmLib.h"
#include "DcgmStringHelpers.h"
#include "nvsdm.h"

#include <DcgmLogging.h>

#include <fmt/format.h>
#include <string>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

namespace
{

std::optional<nvsdmTelem_v1_t> ParseFieldValue(YAML::Node const &node)
{
    nvsdmTelem_v1_t val;
    try
    {
        val.telemType = static_cast<nvsdmTelemType_t>(node["type"].as<unsigned int>());
        val.telemCtr  = node["field"].as<unsigned int>();
        val.valType   = static_cast<nvsdmValType_t>(node["value_type"].as<unsigned int>());
        if (val.valType == NVSDM_VAL_TYPE_DOUBLE)
        {
            val.val.dVal = node["value"].as<double>();
        }
        else if (val.valType == NVSDM_VAL_TYPE_FLOAT)
        {
            val.val.fVal = node["value"].as<float>();
        }
        else if (val.valType == NVSDM_VAL_TYPE_UINT64)
        {
            val.val.u64Val = node["value"].as<uint64_t>();
        }
        else
        {
            val.val.u32Val = node["value"].as<unsigned int>();
        }
        val.status = static_cast<nvsdmRet_t>(node["status"].as<unsigned int>());
    }
    catch (const std::exception &e)
    {
        log_error("failed to parse field value, err: [{}]", e.what());
        return std::nullopt;
    }
    return val;
}

std::optional<DcgmNs::NvsdmMockPort> ParsePort(unsigned int portNum, YAML::Node const &node)
{
    try
    {
        auto lid = node["lid"].as<unsigned int>();
        DcgmNs::NvsdmMockPort port(portNum, lid);
        for (auto const &field : node["fields"])
        {
            auto parsedFieldValue = ParseFieldValue(field);
            if (!parsedFieldValue)
            {
                continue;
            }
            if (parsedFieldValue.value().telemType == NVSDM_TELEM_TYPE_PORT)
            {
                port.SetFieldValue(static_cast<nvsdmPortTelemCounter_t>(parsedFieldValue.value().telemCtr),
                                   parsedFieldValue.value());
            }
            else if (parsedFieldValue.value().telemType == NVSDM_TELEM_TYPE_PLATFORM)
            {
                port.SetFieldValue(static_cast<nvsdmPlatformTelemCounter_t>(parsedFieldValue.value().telemCtr),
                                   parsedFieldValue.value());
            }
        }
        return port;
    }
    catch (const std::exception &e)
    {
        log_error("failed to parse port, err: [{}]", e.what());
        return std::nullopt;
    }
}

} //namespace

namespace DcgmNs
{

nvsdmRet_t NvsdmBase::nvsdmInitialize()
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmFinalize()
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDiscoverTopology(char *, int)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmGetAllDevices(nvsdmDeviceIter_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmGetAllDevicesOfType(int, nvsdmDeviceIter_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmGetDeviceGivenGUID(uint64_t, nvsdmDevice_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmGetNextDevice(nvsdmDeviceIter_t, nvsdmDevice_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmIterateDevices(nvsdmDeviceIter_t, nvsdmRet_t (*)(nvsdmDevice_t const, void *), void *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmResetDeviceList(nvsdmDeviceIter_t)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetPorts(nvsdmDevice_t const, nvsdmPortIter_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmGetAllPorts(nvsdmPortIter_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmGetNextPort(nvsdmPortIter_t, nvsdmPort_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmIteratePorts(nvsdmPortIter_t, nvsdmRet_t (*)(nvsdmPort_t const, void *), void *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortGetDevice(nvsdmPort_t const, nvsdmDevice_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortGetRemote(nvsdmPort_t const, nvsdmPort_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortGetInfo(nvsdmPort_t const, nvsdmPortInfo_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetConnectedDevices(nvsdmDevice_t const, nvsdmDeviceIter_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetConnectedPorts(nvsdmDevice_t const, nvsdmPortIter_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmResetPortList(nvsdmPortIter_t)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetSwitchInfo(nvsdmDevice_t const, nvsdmSwitchInfo_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortGetTelemetryValues(nvsdmPort_t const, nvsdmTelemParam_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetTelemetryValues(nvsdmDevice_t const, nvsdmTelemParam_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortResetTelemetryCounters(nvsdmPort_t const, nvsdmTelemParam_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortExecuteMAD(nvsdmPort_t const, uint8_t *, unsigned int *, uint8_t const *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceToString(nvsdmDevice_t const, char[], unsigned int)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetName(nvsdmDevice_t const, char[], unsigned int)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetShortName(nvsdmDevice_t const, char[], unsigned int)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortToString(nvsdmPort_t const, char[], unsigned int)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortGetNum(nvsdmPort_t const, unsigned int *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortGetLID(nvsdmPort_t const, uint16_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortGetGID(nvsdmPort_t const, uint8_t[16])
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmPortGetGUID(nvsdmPort_t const, uint64_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetDevID(nvsdmDevice_t const, uint16_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetVendorID(nvsdmDevice_t const, uint32_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetGUID(nvsdmDevice_t const, uint64_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetType(nvsdmDevice_t const, unsigned int *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmDeviceGetHealthStatus(nvsdmDevice_t const, nvsdmDeviceHealthStatus_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmGetVersion(uint64_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmGetSupportedInterfaceVersionRange(uint64_t *, uint64_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmBase::nvsdmSetLogLevel(uint32_t)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

char const *NvsdmBase::nvsdmGetErrorString(nvsdmRet_t)
{
    return "NvsdmBase::nvsdmGetErrorString";
}

nvsdmRet_t NvsdmBase::nvsdmDumpPortTelemValues(int, int)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmLib::nvsdmInitialize()
{
    return ::nvsdmInitialize();
}

nvsdmRet_t NvsdmLib::nvsdmFinalize()
{
    return ::nvsdmFinalize();
}

nvsdmRet_t NvsdmLib::nvsdmDiscoverTopology(char *srcCA, int srcPort)
{
    return ::nvsdmDiscoverTopology(srcCA, srcPort);
}

nvsdmRet_t NvsdmLib::nvsdmGetAllDevices(nvsdmDeviceIter_t *iter)
{
    return ::nvsdmGetAllDevices(iter);
}

nvsdmRet_t NvsdmLib::nvsdmGetAllDevicesOfType(int type, nvsdmDeviceIter_t *iter)
{
    return ::nvsdmGetAllDevicesOfType(type, iter);
}

nvsdmRet_t NvsdmLib::nvsdmGetDeviceGivenGUID(uint64_t guid, nvsdmDevice_t *dev)
{
    return ::nvsdmGetDeviceGivenGUID(guid, dev);
}

nvsdmRet_t NvsdmLib::nvsdmGetNextDevice(nvsdmDeviceIter_t iter, nvsdmDevice_t *device)
{
    return ::nvsdmGetNextDevice(iter, device);
}

nvsdmRet_t NvsdmLib::nvsdmIterateDevices(nvsdmDeviceIter_t iter,
                                         nvsdmRet_t (*callback)(nvsdmDevice_t const, void *),
                                         void *cbData)
{
    return ::nvsdmIterateDevices(iter, callback, cbData);
}

nvsdmRet_t NvsdmLib::nvsdmResetDeviceList(nvsdmDeviceIter_t iter)
{
    return ::nvsdmResetDeviceList(iter);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter)
{
    return ::nvsdmDeviceGetPorts(device, iter);
}

nvsdmRet_t NvsdmLib::nvsdmGetAllPorts(nvsdmPortIter_t *iter)
{
    return ::nvsdmGetAllPorts(iter);
}

nvsdmRet_t NvsdmLib::nvsdmGetNextPort(nvsdmPortIter_t iter, nvsdmPort_t *port)
{
    return ::nvsdmGetNextPort(iter, port);
}

nvsdmRet_t NvsdmLib::nvsdmIteratePorts(nvsdmPortIter_t iter,
                                       nvsdmRet_t (*callback)(nvsdmPort_t const, void *),
                                       void *cbData)
{
    return ::nvsdmIteratePorts(iter, callback, cbData);
}

nvsdmRet_t NvsdmLib::nvsdmPortGetDevice(nvsdmPort_t const port, nvsdmDevice_t *device)
{
    return ::nvsdmPortGetDevice(port, device);
}

nvsdmRet_t NvsdmLib::nvsdmPortGetRemote(nvsdmPort_t const port, nvsdmPort_t *remote)
{
    return ::nvsdmPortGetRemote(port, remote);
}

nvsdmRet_t NvsdmLib::nvsdmPortGetInfo(nvsdmPort_t const port, nvsdmPortInfo_t *info)
{
    return ::nvsdmPortGetInfo(port, info);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetConnectedDevices(nvsdmDevice_t const device, nvsdmDeviceIter_t *iter)
{
    return ::nvsdmDeviceGetConnectedDevices(device, iter);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetConnectedPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter)
{
    return ::nvsdmDeviceGetConnectedPorts(device, iter);
}

nvsdmRet_t NvsdmLib::nvsdmResetPortList(nvsdmPortIter_t iter)
{
    return ::nvsdmResetPortList(iter);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetSwitchInfo(nvsdmDevice_t const device, nvsdmSwitchInfo_t *info)
{
    return ::nvsdmDeviceGetSwitchInfo(device, info);
}

nvsdmRet_t NvsdmLib::nvsdmPortGetTelemetryValues(nvsdmPort_t const port, nvsdmTelemParam_t *param)
{
    return ::nvsdmPortGetTelemetryValues(port, param);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetTelemetryValues(nvsdmDevice_t const device, nvsdmTelemParam_t *param)
{
    return ::nvsdmDeviceGetTelemetryValues(device, param);
}

nvsdmRet_t NvsdmLib::nvsdmPortResetTelemetryCounters(nvsdmPort_t const, nvsdmTelemParam_t *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmLib::nvsdmPortExecuteMAD(nvsdmPort_t const, uint8_t *, unsigned int *, uint8_t const *)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t NvsdmLib::nvsdmDeviceToString(nvsdmDevice_t const device, char str[], unsigned int strSize)
{
    return ::nvsdmDeviceToString(device, str, strSize);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetName(nvsdmDevice_t const device, char str[], unsigned int strSize)
{
    return ::nvsdmDeviceGetName(device, str, strSize);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetShortName(nvsdmDevice_t const device, char str[], unsigned int strSize)
{
    return ::nvsdmDeviceGetShortName(device, str, strSize);
}

nvsdmRet_t NvsdmLib::nvsdmPortToString(nvsdmPort_t const port, char str[], unsigned int strSize)
{
    return ::nvsdmPortToString(port, str, strSize);
}

nvsdmRet_t NvsdmLib::nvsdmPortGetNum(nvsdmPort_t const port, unsigned int *num)
{
    return ::nvsdmPortGetNum(port, num);
}

nvsdmRet_t NvsdmLib::nvsdmPortGetLID(nvsdmPort_t const port, uint16_t *lid)
{
    return ::nvsdmPortGetLID(port, lid);
}

nvsdmRet_t NvsdmLib::nvsdmPortGetGID(nvsdmPort_t const port, uint8_t gid[16])
{
    return ::nvsdmPortGetGID(port, gid);
}

nvsdmRet_t NvsdmLib::nvsdmPortGetGUID(nvsdmPort_t const port, uint64_t *guid)
{
    return ::nvsdmPortGetGUID(port, guid);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetDevID(nvsdmDevice_t const device, uint16_t *devID)
{
    return ::nvsdmDeviceGetDevID(device, devID);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetVendorID(nvsdmDevice_t const device, uint32_t *vendorID)
{
    return ::nvsdmDeviceGetVendorID(device, vendorID);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetGUID(nvsdmDevice_t const device, uint64_t *guid)
{
    return ::nvsdmDeviceGetGUID(device, guid);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetType(nvsdmDevice_t const device, unsigned int *type)
{
    return ::nvsdmDeviceGetType(device, type);
}

nvsdmRet_t NvsdmLib::nvsdmDeviceGetHealthStatus(nvsdmDevice_t const device, nvsdmDeviceHealthStatus_t *status)
{
    return ::nvsdmDeviceGetHealthStatus(device, status);
}

nvsdmRet_t NvsdmLib::nvsdmGetVersion(uint64_t *version)
{
    return ::nvsdmGetVersion(version);
}

nvsdmRet_t NvsdmLib::nvsdmGetSupportedInterfaceVersionRange(uint64_t *versionFrom, uint64_t *versionTo)
{
    return ::nvsdmGetSupportedInterfaceVersionRange(versionFrom, versionTo);
}

nvsdmRet_t NvsdmLib::nvsdmSetLogLevel(uint32_t logLevel)
{
    return ::nvsdmSetLogLevel(logLevel);
}

char const *NvsdmLib::nvsdmGetErrorString(nvsdmRet_t ret)
{
    return ::nvsdmGetErrorString(ret);
}

nvsdmRet_t NvsdmLib::nvsdmDumpPortTelemValues(int, int)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

NvsdmMockPort::NvsdmMockPort(unsigned int portNum, uint16_t lid)
    : m_portNum(portNum)
    , m_lid(lid)
{}

unsigned int NvsdmMockPort::GetPortNum() const
{
    return m_portNum;
}

uint16_t NvsdmMockPort::GetLid() const
{
    return m_lid;
}

std::string NvsdmMockPort::GetGid() const
{
    return fmt::format("NvsdmPort-{}", m_portNum);
}

uint64_t NvsdmMockPort::GetGuid() const
{
    int const guidPortNumLshift = 32;
    int const guidPortLidLshift = 16;
    uint64_t guid               = (static_cast<uint64_t>(m_portNum) << guidPortNumLshift)
                    | (static_cast<uint64_t>(m_lid) << guidPortLidLshift) | m_portNum;
    return guid;
}
nvsdmPortInfo_t NvsdmMockPort::GetInfo() const
{
    nvsdmPortInfo_t info {};
    info.portState = m_portState;
    return info;
}

void NvsdmMockPort::SetPortState(nvsdmPortState_t const state)
{
    m_portState = state;
}

void NvsdmMockPort::SetFieldValue(nvsdmPortTelemCounter_t const field, nvsdmTelem_v1_t const value)
{
    m_portFieldValues[field] = value;
}

nvsdmTelem_v1_t NvsdmMockPort::GetFieldValue(nvsdmPortTelemCounter_t const field) const
{
    auto it = m_portFieldValues.find(field);
    if (it != m_portFieldValues.end())
    {
        return it->second;
    }

    nvsdmTelem_v1_t val {};
    val.telemType = NVSDM_TELEM_TYPE_PORT;
    val.telemCtr  = field;
    val.status    = NVSDM_ERROR_INVALID_CTR;
    return val;
}

void NvsdmMockPort::SetFieldValue(nvsdmPlatformTelemCounter_t const field, nvsdmTelem_v1_t const value)
{
    m_platformFieldValues[field] = value;
}

nvsdmTelem_v1_t NvsdmMockPort::GetFieldValue(nvsdmPlatformTelemCounter_t const field) const
{
    auto it = m_platformFieldValues.find(field);
    if (it != m_platformFieldValues.end())
    {
        return it->second;
    }

    nvsdmTelem_v1_t val {};
    val.telemType = NVSDM_TELEM_TYPE_PLATFORM;
    val.telemCtr  = field;
    val.status    = NVSDM_ERROR_INVALID_CTR;
    return val;
}

NvsdmMockDevice::NvsdmMockDevice(nvsdmDevType type, uint16_t devID, uint32_t vendorID, uint32_t healthState)
    : m_type(type)
    , m_devID(devID)
    , m_vendorID(vendorID)
    , m_healthState(healthState)
{}

nvsdmDevType NvsdmMockDevice::GetType() const
{
    return m_type;
}

uint16_t NvsdmMockDevice::GetDevID() const
{
    return m_devID;
}

uint32_t NvsdmMockDevice::GetVendorID() const
{
    return m_vendorID;
}

uint64_t NvsdmMockDevice::GetGuid() const
{
    return (static_cast<uint64_t>(m_vendorID) << 32) | (static_cast<uint64_t>(m_devID) << 16) | m_type;
}

uint32_t NvsdmMockDevice::GetHealthState() const
{
    return m_healthState;
}

nvsdmPortIter_t NvsdmMockDevice::GetPortsIterator()
{
    return reinterpret_cast<nvsdmPortIter_t>(&m_ports);
}

void NvsdmMockDevice::AddPort(NvsdmMockPort const &port)
{
    m_ports.push_back(port);
}


void NvsdmMockDevice::SetFieldValue(nvsdmConnectXTelemCounter_t const field, nvsdmTelem_v1_t const value)
{
    m_fieldValues[field] = value;
}

nvsdmTelem_v1_t NvsdmMockDevice::GetFieldValue(nvsdmConnectXTelemCounter_t const field) const
{
    auto it = m_fieldValues.find(field);
    if (it != m_fieldValues.end())
    {
        return it->second;
    }

    nvsdmTelem_v1_t val {};
    val.telemType = NVSDM_TELEM_TYPE_CONNECTX;
    val.telemCtr  = field;
    val.status    = NVSDM_ERROR_INVALID_CTR;
    return val;
}

nvsdmRet_t NvsdmMock::nvsdmInitialize()
{
    m_init = true;
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmFinalize()
{
    m_init = false;
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDiscoverTopology(char *srcCA, int /* srcPort */)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    if (srcCA != nullptr && !std::string_view(srcCA).starts_with("mlx5_"))
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmGetAllDevices(nvsdmDeviceIter_t *iter)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    *iter = reinterpret_cast<nvsdmDeviceIter_t>(0);
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmIterateDevices(nvsdmDeviceIter_t iter,
                                          nvsdmRet_t (*callback)(nvsdmDevice_t const, void *),
                                          void *cbData)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    auto idx = reinterpret_cast<uint64_t>(iter);
    for (; idx < m_devices.size(); ++idx)
    {
        nvsdmDevice_t dev = reinterpret_cast<nvsdmDevice_t>(&m_devices[idx]);
        callback(dev, cbData);
    }
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDeviceToString(nvsdmDevice_t const device, char str[], unsigned int strSize)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }

    if (!device || strSize < 64)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    NvsdmMockDevice *mockDev = reinterpret_cast<NvsdmMockDevice *>(device);
    std::string type;
    switch (mockDev->GetType())
    {
        case NVSDM_DEV_TYPE_SWITCH:
            type = "SW";
            break;
        case NVSDM_DEV_TYPE_CA:
            type = "IB-CX";
            break;
        default:
            // Unimplemented
            return NVSDM_ERROR_FUNCTION_NOT_FOUND;
    }

    std::string temp = fmt::format("{}-{}", type, mockDev->GetDevID());
    strncpy(str, temp.c_str(), strSize);
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDeviceGetType(nvsdmDevice_t const device, unsigned int *type)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    NvsdmMockDevice *mockDev = reinterpret_cast<NvsdmMockDevice *>(device);
    if (!mockDev || !type)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    *type = mockDev->GetType();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDeviceGetDevID(nvsdmDevice_t const device, uint16_t *devID)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    NvsdmMockDevice *mockDev = reinterpret_cast<NvsdmMockDevice *>(device);
    if (!mockDev || !devID)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    *devID = mockDev->GetDevID();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDeviceGetVendorID(nvsdmDevice_t const device, uint32_t *vendorID)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    NvsdmMockDevice *mockDev = reinterpret_cast<NvsdmMockDevice *>(device);
    if (!mockDev || !vendorID)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    *vendorID = mockDev->GetVendorID();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDeviceGetGUID(nvsdmDevice_t const device, uint64_t *guid)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    NvsdmMockDevice *mockDev = reinterpret_cast<NvsdmMockDevice *>(device);
    if (!mockDev || !guid)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    *guid = mockDev->GetGuid();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDeviceGetHealthStatus(nvsdmDevice_t const device, nvsdmDeviceHealthStatus_t *status)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    NvsdmMockDevice *mockDev = reinterpret_cast<NvsdmMockDevice *>(device);
    if (!mockDev || !status)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    status->state = mockDev->GetHealthState();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDeviceGetPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    NvsdmMockDevice *mockDev = reinterpret_cast<NvsdmMockDevice *>(device);
    if (!mockDev)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    *iter = mockDev->GetPortsIterator();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmIteratePorts(nvsdmPortIter_t iter,
                                        nvsdmRet_t (*callback)(nvsdmPort_t const, void *),
                                        void *cbData)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    auto &ports = *reinterpret_cast<std::vector<NvsdmMockPort> *>(iter);
    for (unsigned int i = 0; i < ports.size(); ++i)
    {
        nvsdmPort_t port = reinterpret_cast<nvsdmPort_t>(&ports[i]);
        callback(port, cbData);
    }
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmPortGetNum(nvsdmPort_t const port, unsigned int *num)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    if (!num)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    NvsdmMockPort *mockPort = reinterpret_cast<NvsdmMockPort *>(port);
    *num                    = mockPort->GetPortNum();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmPortGetLID(nvsdmPort_t const port, uint16_t *lid)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    if (!lid)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    NvsdmMockPort *mockPort = reinterpret_cast<NvsdmMockPort *>(port);
    *lid                    = mockPort->GetLid();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmPortGetGID(nvsdmPort_t const port, uint8_t gid[16])
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    NvsdmMockPort *mockPort = reinterpret_cast<NvsdmMockPort *>(port);
    auto gidStr             = mockPort->GetGid();
    std::memset(gid, 0, 16);
    fmt::format_to_n(gid, 16, "{}\0", gidStr);
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmPortGetGUID(nvsdmPort_t const port, uint64_t *guid)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    if (!guid)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    NvsdmMockPort *mockPort = reinterpret_cast<NvsdmMockPort *>(port);
    *guid                   = mockPort->GetGuid();
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmPortGetInfo(nvsdmPort_t const port, nvsdmPortInfo_t *info)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    if (!info)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    NvsdmMockPort *mockPort = reinterpret_cast<NvsdmMockPort *>(port);
    *info                   = mockPort->GetInfo();
    return NVSDM_SUCCESS;
}

void NvsdmMock::InjectDevice(NvsdmMockDevice const &dev)
{
    m_devices.push_back(dev);
}

nvsdmRet_t NvsdmMock::nvsdmPortGetTelemetryValues(nvsdmPort_t const port, nvsdmTelemParam_t *param)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    if (!param)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    if (param->numTelemEntries < 1)
    {
        param->numTelemEntries = 1;
        return NVSDM_ERROR_INSUFFICIENT_SIZE;
    }
    NvsdmMockPort *mockPort = reinterpret_cast<NvsdmMockPort *>(port);
    if (!mockPort)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    if (param->telemValsArray[0].telemType == NVSDM_TELEM_TYPE_PORT)
    {
        param->telemValsArray[0]
            = mockPort->GetFieldValue(static_cast<nvsdmPortTelemCounter_t>(param->telemValsArray[0].telemCtr));
    }
    else if (param->telemValsArray[0].telemType == NVSDM_TELEM_TYPE_PLATFORM)
    {
        param->telemValsArray[0]
            = mockPort->GetFieldValue(static_cast<nvsdmPlatformTelemCounter_t>(param->telemValsArray[0].telemCtr));
    }
    else
    {
        return NVSDM_ERROR_NOT_SUPPORTED;
    }
    return NVSDM_SUCCESS;
}

nvsdmRet_t NvsdmMock::nvsdmDeviceGetTelemetryValues(nvsdmDevice_t const device, nvsdmTelemParam_t *param)
{
    if (!m_init)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }
    if (!param)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    if (param->numTelemEntries < 1)
    {
        param->numTelemEntries = 1;
        return NVSDM_ERROR_INSUFFICIENT_SIZE;
    }
    NvsdmMockDevice *mockDev = reinterpret_cast<NvsdmMockDevice *>(device);
    if (!mockDev)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }
    param->telemValsArray[0]
        = mockDev->GetFieldValue(static_cast<nvsdmConnectXTelemCounter_t>(param->telemValsArray[0].telemCtr));
    return NVSDM_SUCCESS;
}

bool NvsdmMock::LoadYaml(std::string const &path)
{
    YAML::Node root;

    try
    {
        root = YAML::LoadFile(path);
        if (!root["devices"] || !root["devices"].IsSequence())
        {
            log_debug("devices field is wrong.");
            return false;
        }

        for (auto const &device : root["devices"])
        {
            auto type     = static_cast<nvsdmDevType>(device["type"].as<unsigned int>());
            auto devId    = device["dev_id"].as<unsigned int>();
            auto vendorId = device["vendor_id"].as<unsigned int>();
            auto health   = device["health_state"].as<unsigned int>();
            NvsdmMockDevice mockDev(type, devId, vendorId, health);
            for (auto const field : device["fields"])
            {
                auto parsedFieldOpt = ParseFieldValue(field);
                if (!parsedFieldOpt)
                {
                    YAML::Emitter out;
                    out << field;
                    log_error("failed to parse [{}].", out.c_str());
                    continue;
                }
                mockDev.SetFieldValue(static_cast<nvsdmConnectXTelemCounter_t>(parsedFieldOpt.value().telemCtr),
                                      *parsedFieldOpt);
            }

            unsigned int portNum = 0;
            for (auto const port : device["ports"])
            {
                auto parsedPortOpt = ParsePort(portNum, port);
                if (!parsedPortOpt)
                {
                    YAML::Emitter out;
                    out << port;
                    log_error("failed to parse [{}].", out.c_str());
                    continue;
                }
                mockDev.AddPort(parsedPortOpt.value());
                portNum += 1;
            }
            InjectDevice(mockDev);
        }
    }
    catch (const std::exception &e)
    {
        log_error("failed to YAML load [%s], reason [%s]", path.c_str(), e.what());
        return false;
    }
    return true;
}

} //namespace DcgmNs