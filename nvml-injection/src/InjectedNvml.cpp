/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <InjectedNvml.h>
#include <InjectionKeys.h>
#include <TimestampedData.h>

#include <cstring>

InjectedNvml *InjectedNvml::m_injectedNvmlInstance = nullptr;

InjectedNvml *InjectedNvml::Init()
{
    if (m_injectedNvmlInstance == nullptr)
    {
        m_injectedNvmlInstance = new InjectedNvml();
    }

    return m_injectedNvmlInstance;
}

InjectedNvml *InjectedNvml::GetInstance()
{
    return m_injectedNvmlInstance;
}

InjectedNvml::InjectedNvml()
    : m_nextDeviceId(0)
    , m_fieldHelpers()
{
    m_injectedNvmlInstance = this;

    InitializeGlobalValues();
}

/*****************************************************************************/
bool InjectedNvml::IsGetter(const std::string &funcname) const
{
    return false;
}

/*****************************************************************************/
bool InjectedNvml::IsSetter(const std::string &funcname) const
{
    return false;
}

/*****************************************************************************/
nvmlReturn_t InjectedNvml::DeviceGetWrapper(const std::string &funcname,
                                            const std::string &key,
                                            nvmlDevice_t nvmlDevice,
                                            std::vector<InjectionArgument> &args)
{
    if (funcname == "nvmlDeviceGetRemappedRows")
    {
        CompoundValue cv = m_devices[nvmlDevice].GetCompoundAttribute(key);
        // Arg types will be checked in SetInjectionArguments()
        return (cv.SetInjectionArguments(args));
    }
    else if (funcname == "nvmlDeviceGetInforomVersion")
    {
        CompoundValue cv = m_devices[nvmlDevice].GetCompoundAttribute(key, args[1]);
        // Arg types will be checked in SetString()
        return cv.SetString(args[2], args[3]);
    }
    else if (funcname == "nvmlDeviceGetDetailedEccErrors" || funcname == "nvmlDeviceGetTotalEccErrors")
    {
        static const std::string DETAILED  = "Detailed";
        static const std::string GET_TOTAL = "GetTotal";

        if (args.size() != 4 || args[1].GetType() != INJECTION_MEMORYERRORTYPE
            || args[2].GetType() != INJECTION_ECCCOUNTERTYPE)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        std::string trimmedKey;
        if (key.rfind(DETAILED, 0) == 0)
        {
            trimmedKey = key.substr(DETAILED.size());
        }
        else if (key.rfind(GET_TOTAL, 0) == 0)
        {
            trimmedKey = key.substr(GET_TOTAL.size());
        }
        else
        {
            trimmedKey = key;
        }

        nvmlEccErrorCounts_t *errorCounts = nullptr;
        unsigned long long *total         = nullptr;
        if (funcname == "nvmlDeviceGetDetailedEccErrors")
        {
            if (args[3].GetType() != INJECTION_ECCERRORCOUNTS_PTR)
            {
                return NVML_ERROR_INVALID_ARGUMENT;
            }
            errorCounts = args[3].AsEccErrorCountsPtr();

            if (errorCounts == nullptr)
            {
                return NVML_ERROR_INVALID_ARGUMENT;
            }
        }
        else
        {
            if (args[3].GetType() != INJECTION_ULONG_LONG_PTR)
            {
                return NVML_ERROR_INVALID_ARGUMENT;
            }

            total = args[3].AsULongLongPtr();
            if (total == nullptr)
            {
                return NVML_ERROR_INVALID_ARGUMENT;
            }
        }

        for (unsigned int i = 0; i <= NVML_MEMORY_LOCATION_REGISTER_FILE; i++)
        {
            nvmlMemoryErrorType_t errorType  = args[1].AsMemoryErrorType();
            nvmlEccCounterType_t counterType = args[2].AsEccCounterType();
            unsigned int fieldId = m_fieldHelpers.GetFieldId(errorType, counterType, (nvmlMemoryLocation_t)i);

            nvmlFieldValue_t fieldValue = m_devices[nvmlDevice].GetFieldValue(fieldId);

            if (errorCounts != nullptr)
            {
                switch (static_cast<nvmlMemoryLocation_t>(i))
                {
                    case NVML_MEMORY_LOCATION_L1_CACHE:
                        errorCounts->l1Cache = m_fieldHelpers.GetULongLong(fieldValue.value);
                        break;
                    case NVML_MEMORY_LOCATION_L2_CACHE:
                        errorCounts->l2Cache = m_fieldHelpers.GetULongLong(fieldValue.value);
                        break;
                    case NVML_MEMORY_LOCATION_DRAM:
                        errorCounts->deviceMemory = m_fieldHelpers.GetULongLong(fieldValue.value);
                        break;
                    case NVML_MEMORY_LOCATION_REGISTER_FILE:
                        errorCounts->registerFile = m_fieldHelpers.GetULongLong(fieldValue.value);
                        break;
                    // coverity[dead_error_begin]
                    default: // NOT-REACHED due to loop conditions
                        break;
                }
            }
            else
            {
                *total += m_fieldHelpers.GetULongLong(fieldValue.value);
            }
        }

        return NVML_SUCCESS;
    }
    else if (funcname == "nvmlDeviceGetMemoryErrorCounter")
    {
        if (args.size() != 5 || args[1].GetType() != INJECTION_MEMORYERRORTYPE
            || args[2].GetType() != INJECTION_ECCCOUNTERTYPE || args[3].GetType() != INJECTION_MEMORYLOCATION
            || args[4].GetType() != INJECTION_ULONG_LONG_PTR)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        unsigned long long value = m_devices[nvmlDevice].GetEccErrorCount(
            args[1].AsMemoryErrorType(), args[2].AsEccCounterType(), args[3].AsMemoryLocation());
        return args[4].SetValueFrom(value);
    }
    else if (funcname == "nvmlDeviceClearEccErrorCounts")
    {
        if (args.size() != 2 || args[1].GetType() != INJECTION_MEMORYERRORTYPE)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        nvmlReturn_t overallRet = NVML_SUCCESS;
        for (unsigned int i = 0; i < NVML_MEMORY_ERROR_TYPE_COUNT; i++)
        {
            nvmlMemoryErrorType_t etype = (nvmlMemoryErrorType_t)i;
            InjectionArgument arg(etype);
            // 2nd arg = counterType
            nvmlReturn_t ret = m_devices[nvmlDevice].ClearAttribute("EccErrors", arg, args[1]);
            if (ret != NVML_SUCCESS)
            {
                overallRet = ret;
            }
        }

        return overallRet;
    }
    else if (funcname == "nvmlDeviceValidateInforom")
    {
        InjectionArgument value = m_devices[nvmlDevice].GetAttribute(key);
        if (value.IsEmpty())
        {
            // If not injected, default to a valid inforom
            return NVML_SUCCESS;
        }
        return static_cast<nvmlReturn_t>(value.AsUInt());
    }
    else if (funcname == "nvmlDeviceGetRetiredPages_v2")
    {
        if (args.size() != 5)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        if (args[1].GetType() != INJECTION_PAGERETIREMENTCAUSE || args[2].GetType() != INJECTION_UINT_PTR
            || args[3].GetType() != INJECTION_ULONG_LONG_PTR || args[4].GetType() != INJECTION_ULONG_LONG_PTR)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        return m_devices[nvmlDevice].GetRetiredPages(
            args[1].AsPageRetirementCause(), args[2].AsUIntPtr(), args[3].AsULongLongPtr(), args[4].AsULongLongPtr());
    }
    else if (funcname == "nvmlDeviceGetNvLinkErrorCounter")
    {
        if (args.size() != 4 || args[1].GetType() != INJECTION_UINT || args[2].GetType() != INJECTION_NVLINKERRORCOUNTER
            || args[3].GetType() != INJECTION_ULONG_LONG_PTR)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        return args[3].SetValueFrom(m_devices[nvmlDevice].GetAttribute(key, args[1], args[2]));
    }
    else if (funcname == "nvmlDeviceGetProcessUtilization")
    {
        if (args.size() != 4 || args[1].GetType() != INJECTION_PROCESSUTILIZATIONSAMPLE_PTR
            || args[1].AsProcessUtilizationSamplePtr() == nullptr || args[2].GetType() != INJECTION_UINT_PTR
            || args[2].AsUIntPtr() == nullptr || args[3].GetType() != INJECTION_ULONG_LONG)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        std::vector<TimestampedData> data = m_devices[nvmlDevice].GetDataAfter(args[3].AsULongLong(), key);

        unsigned int max     = *args[2].AsUIntPtr();
        unsigned int count   = 0;
        unsigned int loopEnd = max;
        nvmlReturn_t ret     = NVML_SUCCESS;

        if (data.size() == 0)
        {
            return NVML_ERROR_NOT_FOUND;
        }

        if (data.size() < max)
        {
            loopEnd = data.size();
        }
        else if (data.size() > max)
        {
            ret = NVML_ERROR_INSUFFICIENT_SIZE;
        }

        nvmlProcessUtilizationSample_t *samples = args[1].AsProcessUtilizationSamplePtr();

        for (unsigned int i = 0; i < loopEnd; i++, count++)
        {
            InjectionArgument val = data[i].GetData();
            memcpy(samples + i, val.AsProcessUtilizationSamplePtr(), sizeof(samples[i]));
        }

        return ret;
    }


    // If we reach here, the function hasn't been supported yet
    return NVML_ERROR_NOT_SUPPORTED;
}

/*****************************************************************************/
nvmlReturn_t InjectedNvml::GetWrapper(const std::string &funcname, std::vector<InjectionArgument> &args) const
{
    return NVML_SUCCESS;
}

/*****************************************************************************/
nvmlReturn_t InjectedNvml::DeviceSetWrapper(const std::string &funcname,
                                            const std::string &key,
                                            nvmlDevice_t nvmlDevice,
                                            std::vector<InjectionArgument> &args)
{
    if (funcname == "nvmlDeviceResetNvLinkErrorCounters")
    {
        if (args.size() != 2 || args[1].GetType() != INJECTION_UINT)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        nvmlReturn_t overallRet = NVML_SUCCESS;

        for (unsigned int i = 0; i < NVML_NVLINK_ERROR_COUNT; i++)
        {
            InjectionArgument arg(static_cast<nvmlNvLinkErrorCounter_t>(i));
            nvmlReturn_t ret = m_devices[nvmlDevice].ClearAttribute(key, args[1], arg);
            if (ret != NVML_SUCCESS)
            {
                overallRet = ret;
            }
        }

        return overallRet;
    }

    return NVML_ERROR_NOT_SUPPORTED;
}

/*****************************************************************************/
nvmlReturn_t InjectedNvml::SetWrapper(const std::string &funcname, std::vector<InjectionArgument> &args)
{
    return NVML_SUCCESS;
}

/*****************************************************************************/
InjectionArgument InjectedNvml::SimpleDeviceGet(nvmlDevice_t nvmlDevice, const std::string &key)
{
    return m_devices[nvmlDevice].GetAttribute(key);
}

/*****************************************************************************/
nvmlDevice_t InjectedNvml::GetNvmlDevice(InjectionArgument &arg, const std::string &identifier)
{
    if (arg.GetType() == INJECTION_UINT)
    {
        return m_indexToDevice[arg.AsUInt()].GetIdentifier();
    }
    else if (identifier == "UUID")
    {
        return m_uuidToDevice[arg.AsStr()].GetIdentifier();
    }
    else if (identifier == "Serial")
    {
        return m_serialToDevice[arg.AsStr()].GetIdentifier();
    }
    else if (identifier == "PciBusId")
    {
        return m_busIdToDevice[arg.AsStr()].GetIdentifier();
    }

    return (nvmlDevice_t)0;
}

InjectionArgument InjectedNvml::ObjectlessGet(const std::string &key)
{
    return m_globalAttributes[key];
}

void InjectedNvml::ObjectlessSet(const std::string &key, const InjectionArgument &value)
{
    m_globalAttributes[key] = value;
}

nvmlReturn_t InjectedNvml::GetCompoundValue(nvmlDevice_t nvmlDevice, const std::string &key, CompoundValue &cv)
{
    return cv.SetValueFrom(m_devices[nvmlDevice].GetCompoundAttribute(key));
}

std::string InjectedNvml::GetString(InjectionArgument &arg, const std::string &key)
{
    switch (arg.GetType())
    {
        case INJECTION_DEVICE:
            return m_devices[arg.AsDevice()].GetAttribute(key).AsString();
            break;
        default:
            break;
    }
    return "";
}

InjectionArgument InjectedNvml::DeviceGetWithExtraKey(nvmlDevice_t nvmlDevice,
                                                      const std::string &key,
                                                      const InjectionArgument &arg)
{
    return m_devices[nvmlDevice].GetAttribute(key, arg);
}

nvmlReturn_t InjectedNvml::SimpleDeviceSet(nvmlDevice_t nvmlDevice, const std::string &key, InjectionArgument &value)
{
    m_devices[nvmlDevice].SetAttribute(key, value);
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::IncrementDeviceCount()
{
    unsigned int count = ObjectlessGet(INJECTION_COUNT_KEY).AsUInt();
    count++;
    ObjectlessSet(INJECTION_COUNT_KEY, InjectionArgument(count));
    return NVML_SUCCESS;
}

unsigned int InjectedNvml::GetGpuCount()
{
    return ObjectlessGet(INJECTION_COUNT_KEY).AsUInt();
}

void InjectedNvml::InitializeGpuDefaults(nvmlDevice_t device, unsigned int index, AttributeHolder<nvmlDevice_t> &ah)
{
    char attribute[512];
    snprintf(attribute, sizeof(attribute), "GPU-1feed7b9-beef-fade-6d19-e5ce8489eb%02d", index);
    std::string paramAttr(attribute);

    InjectionArgument uuid(paramAttr);
    m_uuidToDevice[paramAttr] = ah;
    m_devices[device].SetAttribute(INJECTION_UUID_KEY, uuid);

    snprintf(attribute, sizeof(attribute), "03207190049%02d", index);
    paramAttr = attribute;
    InjectionArgument serial(paramAttr);
    m_serialToDevice[paramAttr] = ah;
    m_devices[device].SetAttribute(INJECTION_SERIAL_KEY, serial);

    snprintf(attribute, sizeof(attribute), "00000000:%02d:00.0", 3 * index + 1);
    paramAttr = attribute;
    InjectionArgument pciBusId(paramAttr);
    m_busIdToDevice[paramAttr] = ah;
    m_devices[device].SetAttribute(INJECTION_PCIBUSID_KEY, pciBusId);

    nvmlBrandType_t tBrand = NVML_BRAND_TESLA;
    InjectionArgument brand(tBrand);
    SimpleDeviceSet(device, INJECTION_BRAND_KEY, brand);

    std::string name("V100");
    InjectionArgument devName(name);
    SimpleDeviceSet(device, INJECTION_NAME_KEY, devName);

    int major = 7;
    int minor = 6;
    std::vector<InjectionArgument> values;
    values.push_back(InjectionArgument(major));
    values.push_back(InjectionArgument(minor));
    CompoundValue cv(values);

    DeviceSetCompoundValue(device, INJECTION_CUDACOMPUTECAPABILITY_KEY, cv);

    m_indexToDevice[index] = ah;
}

nvmlReturn_t InjectedNvml::SimpleDeviceCreate(const std::string &key, InjectionArgument &value)
{
    unsigned int deviceInt = m_nvmlDeviceStart + m_devices.size();
    bool valid             = false;
    nvmlDevice_t device;

    memset(&device, 0, sizeof(device));
    memcpy(&device, &deviceInt, sizeof(deviceInt));

    std::string identifier;

    AttributeHolder<nvmlDevice_t> ah(device);

    if (value.GetType() == INJECTION_UINT && key == INJECTION_INDEX_KEY)
    {
        if (m_indexToDevice.count(value.AsUInt()) > 0)
        {
            // This GPU already exists
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        valid = true;
        m_devices[device].SetAttribute(key, value);

        InitializeGpuDefaults(device, value.AsUInt(), ah);
    }
    else if (value.GetType() == INJECTION_CONST_CHAR_PTR)
    {
        identifier = value.AsConstStr();
    }
    else if (value.GetType() == INJECTION_CHAR_PTR)
    {
        identifier = value.AsStr();
    }

    if (identifier.size() > 0)
    {
        if (key == INJECTION_UUID_KEY)
        {
            if (m_uuidToDevice.count(identifier) > 0)
            {
                // This GPU already exists
                return NVML_ERROR_INVALID_ARGUMENT;
            }

            m_uuidToDevice[identifier] = ah;
        }
        else if (key == INJECTION_SERIAL_KEY)
        {
            if (m_serialToDevice.count(identifier) > 0)
            {
                // This GPU already exists
                return NVML_ERROR_INVALID_ARGUMENT;
            }

            m_serialToDevice[identifier] = ah;
        }
        else if (key == INJECTION_PCIBUSID_KEY)
        {
            if (m_busIdToDevice.count(identifier) > 0)
            {
                // This GPU already exists
                return NVML_ERROR_INVALID_ARGUMENT;
            }

            m_busIdToDevice[identifier] = ah;
        }

        InitializeGpuDefaults(device, deviceInt, ah);
        m_devices[device].SetAttribute(key, value);

        valid             = true;
        m_devices[device] = ah;
    }

    if (valid)
    {
        IncrementDeviceCount();

        return NVML_SUCCESS;
    }
    else
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
}

nvmlReturn_t InjectedNvml::DeviceSetWithExtraKey(nvmlDevice_t nvmlDevice,
                                                 const std::string &key,
                                                 const InjectionArgument &extraKey,
                                                 InjectionArgument &value)
{
    m_devices[nvmlDevice].SetAttribute(key, extraKey, value);
    return NVML_SUCCESS;
}

unsigned int InjectedNvml::GetClockInfo(nvmlDevice_t nvmlDevice, const std::string &key, nvmlClockType_t clockType)
{
    return GetClock(nvmlDevice, clockType, NVML_CLOCK_ID_CURRENT);
}

/*****************************************************************************/
unsigned int InjectedNvml::GetClock(nvmlDevice_t nvmlDevice, nvmlClockType_t clockType, nvmlClockId_t clockId)
{
    // TODO
    return 0;
}

InjectionArgument InjectedNvml::VgpuInstanceGet(nvmlVgpuInstance_t vgpuInstance, const std::string &key)
{
    return m_vgpuInstances[vgpuInstance].GetAttribute(key);
}

InjectionArgument InjectedNvml::UnitGet(nvmlUnit_t unit, const std::string &key)
{
    return m_units[unit].GetAttribute(key);
}

InjectionArgument InjectedNvml::GetByVgpuTypeId(nvmlVgpuTypeId_t vgpuType, const std::string &key)
{
    return m_vgpuTypeIds[vgpuType].GetAttribute(key);
}

nvmlReturn_t InjectedNvml::DeviceSetCompoundValue(nvmlDevice_t nvmlDevice,
                                                  const std::string &key,
                                                  const CompoundValue &cv)
{
    m_devices[nvmlDevice].SetAttribute(key, cv);
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::SetFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t &fieldValue)
{
    m_devices[nvmlDevice].SetFieldValue(fieldValue);
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::GetFieldValues(nvmlDevice_t nvmlDevice, int valuesCount, nvmlFieldValue_t *values)
{
    if (values == nullptr)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    for (int i = 0; i < valuesCount; i++)
    {
        nvmlFieldValue_t val = m_devices[nvmlDevice].GetFieldValue(values[i].fieldId);
        memcpy(&values[i], &val, sizeof(values[i]));
    }

    return NVML_SUCCESS;
}

void InjectedNvml::InitializeGlobalValues()
{
    // Set baseline global data
    std::string nvmlVersion("11.0");
    std::string driverVersion("520.49");
    ObjectlessSet(INJECTION_NVMLVERSION_KEY, InjectionArgument(nvmlVersion));
    ObjectlessSet(INJECTION_DRIVERVERSION_KEY, InjectionArgument(driverVersion));
    InjectionArgument cudaDriverVersion(11010);
    ObjectlessSet(INJECTION_CUDADRIVERVERSION_KEY, cudaDriverVersion);

    // Create one GPU because DCGM quits if there are no GPUs
    InjectionArgument indexArg((unsigned int)0);
    SimpleDeviceCreate(INJECTION_INDEX_KEY, indexArg);
}
