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
#pragma once

#include <map>
#include <string>
#include <vector>

#include <nvml.h>

#include "AttributeHolder.h"
#include "CompoundValue.h"
#include "FieldHelpers.h"
#include "InjectionArgument.h"

typedef struct
{
    std::string pciBusId;
    std::string uuid;
    std::string serial;
    unsigned int index;
    nvmlDevice_t device;
} nvmlDeviceWithIdentifiers;

class InjectedNvml
{
public:
    /*****************************************************************************/
    static InjectedNvml *Init();

    /*****************************************************************************/
    static InjectedNvml *GetInstance();

    /*****************************************************************************/
    bool IsGetter(const std::string &funcname) const;

    /*****************************************************************************/
    bool IsSetter(const std::string &funcname) const;

    /*****************************************************************************/
    nvmlReturn_t DeviceGetWrapper(const std::string &funcname,
                                  const std::string &key,
                                  nvmlDevice_t nvmlDevice,
                                  std::vector<InjectionArgument> &args);

    /*****************************************************************************/
    nvmlReturn_t GetWrapper(const std::string &funcname, std::vector<InjectionArgument> &args) const;

    /*****************************************************************************/
    nvmlReturn_t DeviceSetWrapper(const std::string &funcname,
                                  const std::string &key,
                                  nvmlDevice_t nvmlDevice,
                                  std::vector<InjectionArgument> &args);

    /*****************************************************************************/
    nvmlReturn_t SetWrapper(const std::string &funcname, std::vector<InjectionArgument> &args);

    /*****************************************************************************/
    InjectionArgument SimpleDeviceGet(nvmlDevice_t nvmlDevice, const std::string &key);

    /*****************************************************************************/
    nvmlDevice_t GetNvmlDevice(InjectionArgument &arg, const std::string &identifier);

    /*****************************************************************************/
    InjectionArgument ObjectlessGet(const std::string &key);

    /*****************************************************************************/
    void ObjectlessSet(const std::string &key, const InjectionArgument &value);

    /*****************************************************************************/
    nvmlReturn_t GetCompoundValue(nvmlDevice_t nvmlDevice, const std::string &key, CompoundValue &cv);

    /*****************************************************************************/
    unsigned int GetClockInfo(nvmlDevice_t nvmlDevice, const std::string &key, nvmlClockType_t clockType);

    /*****************************************************************************/
    unsigned int GetClock(nvmlDevice_t nvmlDevice, nvmlClockType_t clockType, nvmlClockId_t clockId);

    /*****************************************************************************/
    std::string GetString(InjectionArgument &arg, const std::string &key);

    /*****************************************************************************/
    InjectionArgument DeviceGetWithExtraKey(nvmlDevice_t nvmlDevice,
                                            const std::string &key,
                                            const InjectionArgument &arg);

    /*****************************************************************************/
    nvmlReturn_t SimpleDeviceSet(nvmlDevice_t nvmlDevice, const std::string &key, InjectionArgument &value);

    /*****************************************************************************/
    nvmlReturn_t DeviceSetCompoundValue(nvmlDevice_t nvmlDevice, const std::string &key, const CompoundValue &cv);

    /*****************************************************************************/
    nvmlReturn_t DeviceSetWithExtraKey(nvmlDevice_t nvmlDevice,
                                       const std::string &key,
                                       const InjectionArgument &extraKey,
                                       InjectionArgument &value);

    /*****************************************************************************/
    nvmlReturn_t SimpleDeviceCreate(const std::string &key, InjectionArgument &value);

    /*****************************************************************************/
    InjectionArgument VgpuInstanceGet(nvmlVgpuInstance_t vgpuInstance, const std::string &key);

    /*****************************************************************************/
    InjectionArgument UnitGet(nvmlUnit_t unit, const std::string &key);

    /*****************************************************************************/
    InjectionArgument GetByVgpuTypeId(nvmlVgpuTypeId_t vgpuType, const std::string &key);

    /*****************************************************************************/
    nvmlReturn_t SetFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t &fieldValue);

    /*****************************************************************************/
    nvmlReturn_t GetFieldValues(nvmlDevice_t nvmlDevice, int valuesCount, nvmlFieldValue_t *values);

    /*****************************************************************************/
    unsigned int GetGpuCount();

private:
    static InjectedNvml *m_injectedNvmlInstance;

    /*****************************************************************************/
    InjectedNvml();

    unsigned int m_nextDeviceId;
    static const unsigned int m_nvmlDeviceStart = 0xA0A0;

    std::map<nvmlVgpuInstance_t, AttributeHolder<nvmlVgpuInstance_t>> m_vgpuInstances;
    std::map<nvmlVgpuTypeId_t, AttributeHolder<nvmlVgpuTypeId_t>> m_vgpuTypeIds;
    std::map<nvmlUnit_t, AttributeHolder<nvmlUnit_t>> m_units;
    std::map<nvmlDevice_t, AttributeHolder<nvmlDevice_t>> m_devices;

    std::map<std::string, AttributeHolder<nvmlDevice_t>> m_busIdToDevice;
    std::map<std::string, AttributeHolder<nvmlDevice_t>> m_uuidToDevice;
    std::map<std::string, AttributeHolder<nvmlDevice_t>> m_serialToDevice;
    std::map<unsigned int, AttributeHolder<nvmlDevice_t>> m_indexToDevice;

    std::map<std::string, InjectionArgument> m_globalAttributes;

    FieldHelpers m_fieldHelpers;

    nvmlReturn_t IncrementDeviceCount();

    void InitializeGlobalValues();

    void InitializeGpuDefaults(nvmlDevice_t device, unsigned int index, AttributeHolder<nvmlDevice_t> &ah);
};
