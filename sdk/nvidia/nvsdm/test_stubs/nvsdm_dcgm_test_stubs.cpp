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

#include "nvsdm.h"

#include <fmt/core.h>
#include <list>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

struct nvsdmDevice {
    unsigned int id;
    nvsdmDevType type;
};

struct nvsdmPort {
    unsigned int id;
};

static std::mutex stubLibraryMutex;
static std::vector<nvsdmDevice> stubDevices = {nvsdmDevice{0, NVSDM_DEV_TYPE_SWITCH}, nvsdmDevice{1, NVSDM_DEV_TYPE_SWITCH}};
static std::vector<nvsdmPort> stubPorts = {nvsdmPort{0}, nvsdmPort{1}};
static bool stubLibraryInit = false;

#pragma GCC visibility push(default)

nvsdmRet_t nvsdmInitialize()
{
    stubLibraryInit = true;
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmFinalize()
{
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmDiscoverTopology(char *srcCA, int srcPort)
{
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmGetAllDevices(nvsdmDeviceIter_t *iter)
{
    *iter = reinterpret_cast<nvsdmDeviceIter_t>(&stubDevices[0]);
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmGetAllDevicesOfType(int type, nvsdmDeviceIter_t *iter)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmGetDeviceGivenGUID(uint64_t guid, nvsdmDevice_t *dev)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmGetNextDevice(nvsdmDeviceIter_t iter, nvsdmDevice_t *device)
{
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmIterateDevices(nvsdmDeviceIter_t iter,
                                 nvsdmRet_t (*callback)(nvsdmDevice_t const, void *),
                                 void *cbData)
{
    nvsdmDevice_t it = reinterpret_cast<nvsdmDevice_t>(iter);
    nvsdmDevice_t endIndex = &stubDevices[0] + stubDevices.size();
    if (!it || it >= endIndex)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    for (;it < endIndex; it++)
    {
        callback(it, cbData);
    }
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmResetDeviceList(nvsdmDeviceIter_t iter)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmDeviceGetPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter)
{
    *iter = reinterpret_cast<nvsdmPortIter_t>(&stubPorts[0]);
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmGetAllPorts(nvsdmPortIter_t *iter)
{
    *iter = reinterpret_cast<nvsdmPortIter_t>(&stubPorts[0]);
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmGetNextPort(nvsdmPortIter_t iter, nvsdmPort_t *port)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmIteratePorts(nvsdmPortIter_t iter, nvsdmRet_t (*callback)(nvsdmPort_t const, void *), void *cbData)
{
    nvsdmPort_t it = reinterpret_cast<nvsdmPort_t>(iter);
    nvsdmPort_t endIndex = &stubPorts[0] + stubPorts.size();
    if (!it || it >= endIndex)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    for (;it < endIndex; it++)
    {
        callback(it, cbData);
    }
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmPortGetDevice(nvsdmPort_t const port, nvsdmDevice_t *device)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmPortGetRemote(nvsdmPort_t const port, nvsdmPort_t *remote)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmPortGetInfo(nvsdmPort_t const port, nvsdmPortInfo_t *info)
{
    if (!port || port->id >= stubPorts.size() || !info)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    if (info->version != nvsdmPortInfo_v1)
    {
        return NVSDM_ERROR_VERSION_NOT_SUPPORTED;
    }

    /* Populate port info. For now only portState is populated. */
    info->portState = NVSDM_PORT_STATE_ACTIVE;

    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmDeviceGetConnectedDevices(nvsdmDevice_t const device, nvsdmDeviceIter_t *iter)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmDeviceGetConnectedPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmResetPortList(nvsdmPortIter_t iter)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmDeviceGetSwitchInfo(nvsdmDevice_t const device, nvsdmSwitchInfo_t *info)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmPortGetTelemetryValues(nvsdmPort_t const port, nvsdmTelemParam_t *param)
{
    if (!stubLibraryInit)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }

    if (!port || !param || !param->telemValsArray)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    if (param->version != nvsdmTelemParam_v1)
    {
        return NVSDM_ERROR_VERSION_NOT_SUPPORTED;
    }

    auto now = std::chrono::duration_cast<std::chrono::seconds> (std::chrono::system_clock::now().time_since_epoch()).count();
    for (unsigned int i = 0; i < param->numTelemEntries; i++)
    {
        /* For simplicity use same value type. Otherwise we can have a switch cases based on telemCtr. */
        param->telemValsArray[i].valType    = NVSDM_VAL_TYPE_UINT64;
        param->telemValsArray[i].val.u64Val = i + param->telemValsArray[i].telemCtr;
        param->telemValsArray[i].status     = NVSDM_SUCCESS;
        param->telemValsArray[i].timestamp  = now;
    }

    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmPortResetTelemetryCounters(nvsdmPort_t const port, nvsdmTelemParam_t *param)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmPortExecuteMAD(nvsdmPort_t const dest, uint8_t *outBuff, unsigned int *bytesRead, uint8_t const *mad)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmDeviceToString(nvsdmDevice_t const device, char str[], unsigned int strSize)
{
    if (!device || strSize < 64) {
        return NVSDM_ERROR_INVALID_ARG;
    }

    std::string type;

    switch(device->type)
    {
    case NVSDM_DEV_TYPE_SWITCH:
        type = "SW";
        break;
    default:
        // Unimplemented
        return NVSDM_ERROR_FUNCTION_NOT_FOUND;
    }

    std::string temp = fmt::format("{}-{}", type, device->id);

    strncpy(str, temp.c_str(), strSize);

    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmDeviceGetName(nvsdmDevice_t const device, char str[], unsigned int strSize)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmDeviceGetShortName(nvsdmDevice_t const device, char str[], unsigned int strSize)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmPortToString(nvsdmPort_t const port, char str[], unsigned int strSize)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmPortGetNum(nvsdmPort_t const port, unsigned int *num)
{
    if (!port || !num || port->id >= stubPorts.size())
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    /* Stub assumes port number is same as port id. */
    *num = port->id;
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmPortGetLID(nvsdmPort_t const port, uint16_t *lid)
{
    if (!stubLibraryInit)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }

    if (!port || !lid || port->id >= stubPorts.size())
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    /* Stub assumes each port belongs to link id 1. */
    uint16_t const nvsdmPortLID = 1;
    *lid = nvsdmPortLID;
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmPortGetGID(nvsdmPort_t const port, uint8_t gid[16])
{
    if (!stubLibraryInit)
    {
        return NVSDM_ERROR_UNINITIALIZED;
    }

    if (!port || !gid || port->id >= stubPorts.size())
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    auto result = fmt::format_to_n(gid, 15, "NvsdmPort-{}", port->id);
    *result.out = '\0';

    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmPortGetGUID(nvsdmPort_t const port, uint64_t *guid)
{
    if (!port || !guid || port->id >= stubPorts.size())
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    unsigned int nvsdmPortNum = 0;
    nvsdmPortGetNum(port, &nvsdmPortNum);

    uint16_t nvsdmPortLid = 0;
    nvsdmPortGetLID(port, &nvsdmPortLid);

    uint8_t gid[16] = {};
    nvsdmPortGetGID(port, gid);

    int const guidPortNumLshift = 32;
    int const guidPortLidLshift = 16;
    *guid = ((uint64_t)(nvsdmPortNum) << guidPortNumLshift) | ((uint64_t)(nvsdmPortLid) << guidPortLidLshift) | port->id;

    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmDeviceGetDevID(nvsdmDevice_t const device, uint16_t *devID)
{
    if (!device || !devID || device->id >= stubDevices.size())
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    *devID = device->id;
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmDeviceGetVendorID(nvsdmDevice_t const device, uint32_t *vendorID)
{
    if (!device || !vendorID || device->id >= stubDevices.size())
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    uint32_t const nvsdmSwitchVendorID = 0xbaca;
    switch (device->type)
    {
        case NVSDM_DEV_TYPE_SWITCH:
            *vendorID = nvsdmSwitchVendorID;
            break;
        default:
            return NVSDM_ERROR_INVALID_ARG;
    }

    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmDeviceGetGUID(nvsdmDevice_t const device, uint64_t *guid)
{
    if (!device || !guid || device->id >= stubDevices.size())
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    uint16_t nvsdmDeviceDevID = 0;
    nvsdmDeviceGetDevID(device, &nvsdmDeviceDevID);

    uint32_t nvsdmDeviceVendorID = 0;
    nvsdmDeviceGetVendorID(device, &nvsdmDeviceVendorID);

    int const guidDeviceVendorIDLshift = 32;
    int const guidDeviceDevIDLshift = 16;
    *guid = ((uint64_t)(nvsdmDeviceVendorID) << guidDeviceVendorIDLshift) | ((uint64_t)(nvsdmDeviceDevID) << guidDeviceDevIDLshift) | device->type;

    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmDeviceGetType(nvsdmDevice_t const device, unsigned int *type)
{
    if (!device || !type || device->id >= stubDevices.size())
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    *type = device->type;
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmDeviceGetHealthStatus(nvsdmDevice_t const device, nvsdmDeviceHealthStatus_t *status)
{
    if (!device || device->id >= stubDevices.size() || !status)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    if (device->type != NVSDM_DEV_TYPE_SWITCH)
    {
        return NVSDM_ERROR_NOT_SUPPORTED;
    }

    status->state = NVSDM_DEVICE_STATE_HEALTHY;
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmGetVersion(uint64_t *version)
{
    if (!version)
    {
        return NVSDM_ERROR_INVALID_ARG;
    }

    *version = NVSDM_CURR_VERSION;
    return NVSDM_SUCCESS;
}

nvsdmRet_t nvsdmGetSupportedInterfaceVersionRange(uint64_t *versionFrom, uint64_t *versionTo)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

nvsdmRet_t nvsdmSetLogLevel(uint32_t logLevel)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

char const *nvsdmGetErrorString(nvsdmRet_t ret)
{
    static std::unordered_map <nvsdmRet_t, char const *, std::hash<int>> const errorStrings = {
        {NVSDM_SUCCESS,                             "Success"},
        {NVSDM_ERROR_UNINITIALIZED,                 "Uninitialized"},
        {NVSDM_ERROR_NOT_SUPPORTED,                 "Not Supported"},
        {NVSDM_ERROR_INVALID_ARG,                   "Invalid Argument"},
        {NVSDM_ERROR_INSUFFICIENT_SIZE,             "Insufficient Size"},
        {NVSDM_ERROR_VERSION_NOT_SUPPORTED,         "Version Not Supported"},
        {NVSDM_ERROR_MEMORY,                        "Insufficient Memory"},
        {NVSDM_ERROR_DEVICE_DISCOVERY_FAILURE,      "Device Discovery Failure"},
        {NVSDM_ERROR_LIBRARY_LOAD,                  "Error Loading Library"},
        {NVSDM_ERROR_FUNCTION_NOT_FOUND,            "Function Not Found"},
        {NVSDM_ERROR_INVALID_CTR,                   "Invalid Telemetry Counter"},
        {NVSDM_ERROR_TELEMETRY_READ,                "Telemetry Read Error"},
        {NVSDM_ERROR_DEVICE_NOT_FOUND,              "Device Not Found"},
        {NVSDM_ERROR_UMAD_INIT,                     "UMAD Init Error"},
        {NVSDM_ERROR_UMAD_LIB_CALL,                 "UMAD Library Call Error"},
        {NVSDM_ERROR_MAD_LIB_CALL,                  "MAD Library Call Error"},
        {NVSDM_ERROR_NLSOCKET_OPEN_FAILED,          "Netlink Socket Open Error"},
        {NVSDM_ERROR_NLSOCKET_BIND_FAILED,          "Netlink Socket Bind Error"},
        {NVSDM_ERROR_NLSOCKET_SEND_FAILED,          "Netlink Socket Send Error"},
        {NVSDM_ERROR_NLSOCKET_RECV_FAILED,          "Netlink Socket Recv Error"},
        {NVSDM_ERROR_UNKNOWN,                       "Unknown Error"},
    };
    char const *str = "N/A";
    auto it = errorStrings.find(ret);
    if (it != errorStrings.end())
    {
        str = it->second;
    }
    return str;
}

nvsdmRet_t nvsdmDumpPortTelemValues(int lid, int portNum)
{
    return NVSDM_ERROR_FUNCTION_NOT_FOUND;
}

#pragma GCC visibility pop
#pragma GCC diagnostic pop
