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

#include "NvmlFuncReturn.h"
#include "nvml.h"
#include "nvml_injection_structs.h"
#include <InjectedNvml.h>
#include <InjectionKeys.h>
#include <NvmlLogging.h>
#include <NvmlReturnDeserializer.h>
#include <TimestampedData.h>

#include <cstring>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <yaml-cpp/node/node.h>

namespace
{

bool IsDeviceFunc(const std::string &funcname, const std::vector<InjectionArgument> &args)
{
    return (funcname.starts_with("nvmlDeviceGet") || funcname.starts_with("nvmlGpmQueryDevice")
            || funcname == "nvmlDeviceValidateInforom")
           && args.size() >= 1 && args[0].GetType() == INJECTION_DEVICE;
}

bool IsGpuInstanceFunc(const std::string &funcname, const std::vector<InjectionArgument> &args)
{
    return funcname.starts_with("nvmlGpuInstanceGet") && args.size() >= 1 && args[0].GetType() == INJECTION_GPUINSTANCE;
}

bool IsComputeInstanceFunc(const std::string &funcname, const std::vector<InjectionArgument> &args)
{
    return funcname.starts_with("nvmlComputeInstanceGet") && args.size() >= 1
           && args[0].GetType() == INJECTION_COMPUTEINSTANCE;
}

bool IsVgpuTypeFunc(const std::string &funcname, const std::vector<InjectionArgument> &args)
{
    return funcname.starts_with("nvmlVgpuTypeGet") && args.size() >= 1;
}

bool IsVgpuInstanceFunc(const std::string &funcname, const std::vector<InjectionArgument> &args)
{
    return funcname.starts_with("nvmlVgpuInstanceGet") && args.size() >= 1;
}

template <typename underlyType, typename resultType>
std::optional<InjectionArgument> BasicKeyParser(const YAML::Node &node)
{
    if (!node)
    {
        return std::nullopt;
    }
    return static_cast<resultType>(node.as<underlyType>());
}

bool MemoryErrorCounterParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node)
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    for (YAML::const_iterator layer1 = node.begin(); layer1 != node.end(); ++layer1)
    {
        auto layer1Key = BasicKeyParser<int, nvmlMemoryErrorType_t>(layer1->first);
        auto layer1Val = layer1->second;

        if (!layer1Key)
        {
            return false;
        }
        for (YAML::const_iterator layer2 = layer1Val.begin(); layer2 != layer1Val.end(); ++layer2)
        {
            auto layer2Key = BasicKeyParser<int, nvmlEccCounterType_t>(layer2->first);
            auto layer2Val = layer2->second;

            if (!layer2Key)
            {
                return false;
            }
            for (YAML::const_iterator layer3 = layer2Val.begin(); layer3 != layer2Val.end(); ++layer3)
            {
                auto layer3Key = BasicKeyParser<int, nvmlMemoryLocation_t>(layer3->first);
                auto value     = layer3->second;

                if (!layer3Key)
                {
                    return false;
                }

                if (!value[FunctionReturn])
                {
                    ah.SetAttribute(key,
                                    layer1Key.value(),
                                    layer2Key.value(),
                                    layer3Key.value(),
                                    NvmlFuncReturn(NVML_ERROR_UNKNOWN));
                }
                auto ret = static_cast<nvmlReturn_t>(value[FunctionReturn].as<int>());
                if (ret != NVML_SUCCESS)
                {
                    ah.SetAttribute(key, layer1Key.value(), layer2Key.value(), layer3Key.value(), NvmlFuncReturn(ret));
                    continue;
                }
                auto counter = value[ReturnValue].as<unsigned long long>();
                ah.SetAttribute(
                    key, layer1Key.value(), layer2Key.value(), layer3Key.value(), NvmlFuncReturn(ret, counter));
            }
        }
    }
    return true;
}

bool RemappedRowsParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node || !node[FunctionReturn])
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    auto ret = static_cast<nvmlReturn_t>(node[FunctionReturn].as<int>());
    if (ret != NVML_SUCCESS)
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    if (!node[ReturnValue] || !node[ReturnValue]["corrRows"] || !node[ReturnValue]["uncRows"]
        || !node[ReturnValue]["isPending"] || !node[ReturnValue]["failureOccurred"])
    {
        return false;
    }
    std::vector<InjectionArgument> args;
    args.emplace_back(node[ReturnValue]["corrRows"].as<unsigned int>());
    args.emplace_back(node[ReturnValue]["uncRows"].as<unsigned int>());
    args.emplace_back(node[ReturnValue]["isPending"].as<unsigned int>());
    args.emplace_back(node[ReturnValue]["failureOccurred"].as<unsigned int>());
    ah.SetAttribute(key, NvmlFuncReturn(ret, args));
    return true;
}

bool VmIdParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlVgpuInstance_t> &ah)
{
    if (!node || !node[FunctionReturn])
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    auto ret = static_cast<nvmlReturn_t>(node[FunctionReturn].as<int>());
    if (ret != NVML_SUCCESS)
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    if (!node[ReturnValue] || !node[ReturnValue]["vmId"] || !node[ReturnValue]["vmIdType"])
    {
        return false;
    }
    std::vector<InjectionArgument> args;
    args.emplace_back(node[ReturnValue]["vmId"].as<std::string>());
    args.emplace_back(static_cast<nvmlVgpuVmIdType_t>(node[ReturnValue]["vmIdType"].as<int>()));
    ah.SetAttribute(key, NvmlFuncReturn(ret, args));
    return true;
}

bool ProcessUtilizationParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node || !node[0])
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    // we record ProcessUtilization using timestamp 0
    auto processes = node[0];
    auto ret       = static_cast<nvmlReturn_t>(processes[FunctionReturn].as<int>(NVML_ERROR_UNKNOWN));
    if (ret != NVML_SUCCESS || !processes[ReturnValue])
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    ah.SetAttribute(key, NvmlFuncReturn(NVML_SUCCESS));
    for (YAML::const_iterator it = processes[ReturnValue].begin(); it != processes[ReturnValue].end(); ++it)
    {
        auto process = *it;
        if (!process["decUtil"] || !process["encUtil"] || !process["memUtil"] || !process["pid"] || !process["smUtil"]
            || !process["timeStamp"])
        {
            NVML_LOG_ERR("process info misses expectation");
            return false;
        }

        nvmlProcessUtilizationSample_t sample;
        sample.decUtil   = process["decUtil"].as<unsigned int>();
        sample.encUtil   = process["encUtil"].as<unsigned int>();
        sample.memUtil   = process["memUtil"].as<unsigned int>();
        sample.pid       = process["pid"].as<unsigned int>();
        sample.smUtil    = process["smUtil"].as<unsigned int>();
        sample.timeStamp = process["timeStamp"].as<unsigned long long>();
        ah.AddProcessUtilizationRecord(sample.timeStamp, sample);
    }
    return true;
}

bool VgpuProcessUtilizationParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node || !node[0])
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    // we record ProcessUtilization using timestamp 0
    auto processes = node[0];
    auto ret       = static_cast<nvmlReturn_t>(processes[FunctionReturn].as<int>(NVML_ERROR_UNKNOWN));
    if (ret != NVML_SUCCESS || !processes[ReturnValue])
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    ah.SetAttribute(key, NvmlFuncReturn(NVML_SUCCESS));
    for (YAML::const_iterator it = processes[ReturnValue].begin(); it != processes[ReturnValue].end(); ++it)
    {
        auto process = *it;
        if (!process["decUtil"] || !process["encUtil"] || !process["memUtil"] || !process["pid"] || !process["smUtil"]
            || !process["timeStamp"] || !process["processName"] || !process["vgpuInstance"])
        {
            NVML_LOG_ERR("process info misses expectation");
            return false;
        }

        nvmlVgpuProcessUtilizationSample_t sample;
        sample.decUtil   = process["decUtil"].as<unsigned int>();
        sample.encUtil   = process["encUtil"].as<unsigned int>();
        sample.memUtil   = process["memUtil"].as<unsigned int>();
        sample.pid       = process["pid"].as<unsigned int>();
        sample.smUtil    = process["smUtil"].as<unsigned int>();
        sample.timeStamp = process["timeStamp"].as<unsigned long long>();
        snprintf(
            sample.processName, sizeof(sample.processName), "%s", process["processName"].as<std::string>().c_str());
        sample.vgpuInstance = process["vgpuInstance"].as<unsigned int>();
        ah.AddVgpuProcessUtilizationRecord(sample.timeStamp, sample);
    }
    return true;
}

bool VgpuInstanceUtilizationParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node || !node[0])
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    // we record ProcessUtilization using timestamp 0
    auto instances = node[0];
    auto ret       = static_cast<nvmlReturn_t>(instances[FunctionReturn].as<int>(NVML_ERROR_UNKNOWN));
    if (ret != NVML_SUCCESS || !instances[ReturnValue])
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    ah.SetAttribute(key, NvmlFuncReturn(NVML_SUCCESS));
    for (YAML::const_iterator it = instances[ReturnValue].begin(); it != instances[ReturnValue].end(); ++it)
    {
        auto instance = *it;
        if (!instance["decUtil"] || !instance["encUtil"] || !instance["memUtil"] || !instance["smUtil"]
            || !instance["timeStamp"] || !instance["vgpuInstance"])
        {
            NVML_LOG_ERR("instance info misses expectation");
            return false;
        }

        nvmlVgpuInstanceUtilizationSample_t sample;
        sample.decUtil.uiVal = instance["decUtil"].as<unsigned int>();
        sample.encUtil.uiVal = instance["encUtil"].as<unsigned int>();
        sample.memUtil.uiVal = instance["memUtil"].as<unsigned int>();
        sample.smUtil.uiVal  = instance["smUtil"].as<unsigned int>();
        sample.timeStamp     = instance["timeStamp"].as<unsigned long long>();
        sample.vgpuInstance  = instance["vgpuInstance"].as<unsigned int>();
        ah.AddVgpuInstanceUtilizationRecord(sample.timeStamp, { NVML_VALUE_TYPE_UNSIGNED_INT, sample });
    }
    return true;
}

} //namespace

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
{
    m_injectedNvmlInstance = this;
}

InjectedNvml::~InjectedNvml()
{
    for (auto &[_, ah] : m_vgpuInstances)
    {
        ah.Clear();
    }
    for (auto &[_, ah] : m_vgpuTypeIds)
    {
        ah.Clear();
    }
    for (auto &[_, ah] : m_gpuInstances)
    {
        ah.Clear();
    }
    for (auto &[_, ah] : m_computeInstances)
    {
        ah.Clear();
    }
    for (auto &ah : m_deviceCollection)
    {
        ah.Clear();
    }
    for (auto &ah : m_migDeviceCollection)
    {
        ah.Clear();
    }
    for (auto &[_, ret] : m_globalAttributes)
    {
        ret.Clear();
    }
}

void InjectedNvml::Reset()
{
    m_injectedNvmlInstance = nullptr;
}

bool InjectedNvml::DeviceOrderParser(const YAML::Node &node)
{
    for (const std::string &deviceUUID : node.as<std::vector<std::string>>())
    {
        nvmlDevice_t device = GenNextNvmlDevice();

        m_deviceCollection.emplace_back(device);
        m_deviceCollection.back().SetAttribute(INJECTION_UUID_KEY, NvmlFuncReturn(NVML_SUCCESS, deviceUUID));
        m_devices[device]          = std::prev(m_deviceCollection.end());
        m_uuidToDevice[deviceUUID] = std::prev(m_deviceCollection.end());
        m_indexToDevice.emplace_back(std::prev(m_deviceCollection.end()));
    }
    return true;
}

bool InjectedNvml::ParseGlobal(const YAML::Node &global)
{
    NvmlReturnDeserializer nvmlReturnDeserializer;
    std::unordered_map<std::string, std::function<bool(const YAML::Node &)>> handlers {
        { "DeviceOrder", std::bind(&InjectedNvml::DeviceOrderParser, this, std::placeholders::_1) },
    };

    if (!global)
    {
        return false;
    }

    for (YAML::const_iterator it = global.begin(); it != global.end(); ++it)
    {
        auto key         = it->first.as<std::string>();
        YAML::Node value = it->second;

        if (handlers.contains(key))
        {
            // DeviceOrder is a fake key for recording the order of device.
            // We don't need to store it just execute it for preparing internal members.
            handlers[key](value);
            continue;
        }
        auto parsedResultOpt = nvmlReturnDeserializer.GeneralHandle(key, value);
        if (!parsedResultOpt)
        {
            NVML_LOG_ERR("failed to parse key [%s]", key.c_str());
            return false;
        }
        m_globalAttributes[key] = parsedResultOpt.value();
    }
    return true;
}

bool InjectedNvml::ActiveVgpusParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node || !node[FunctionReturn])
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    auto ret = static_cast<nvmlReturn_t>(node[FunctionReturn].as<int>(NVML_ERROR_UNKNOWN));
    if (ret != NVML_SUCCESS || !node[ReturnValue])
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    std::vector<InjectionArgument> injectedArgs;
    auto elements    = node[ReturnValue].as<std::vector<unsigned>>();
    auto numElements = static_cast<unsigned>(elements.size());
    injectedArgs.emplace_back(numElements);
    auto *elementsPtr = reinterpret_cast<nvmlVgpuInstance_t *>(malloc(numElements * sizeof(nvmlVgpuInstance_t)));
    for (unsigned i = 0; i < numElements; ++i)
    {
        elementsPtr[i]                  = static_cast<nvmlVgpuInstance_t>(elements[i]);
        m_vgpuInstances[elementsPtr[i]] = AttributeHolder<nvmlVgpuInstance_t>();
    }
    injectedArgs.emplace_back(elementsPtr, numElements);
    ah.SetAttribute(key, NvmlFuncReturn(ret, std::move(injectedArgs)));
    return true;
}

bool InjectedNvml::GpuInstancesParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node)
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
    {
        auto key2    = it->first.as<unsigned int>();
        auto profile = it->second;
        if (!profile[FunctionReturn])
        {
            NVML_LOG_ERR("profile lacks of FunctionReturn");
            return false;
        }
        auto profileRet = static_cast<nvmlReturn_t>(profile[FunctionReturn].as<int>());
        if (profileRet != NVML_SUCCESS)
        {
            ah.SetAttribute(key, key2, NvmlFuncReturn(profileRet));
            continue;
        }
        auto profileVale     = profile[ReturnValue].as<std::vector<std::string>>();
        auto *gpuInstanceArr = reinterpret_cast<nvmlGpuInstance_t *>(malloc(sizeof(&profileVale) * profileVale.size()));
        unsigned int idx     = 0;
        for (const auto &gpuInstance : profileVale)
        {
            m_gpuInstanceCollection.emplace_back(gpuInstance);
            gpuInstanceArr[idx]                 = reinterpret_cast<nvmlGpuInstance_t>(&m_gpuInstanceCollection.back());
            m_gpuInstances[gpuInstanceArr[idx]] = AttributeHolder<nvmlGpuInstance_t>();
            idx += 1;
        }
        std::vector<InjectionArgument> args;
        args.emplace_back(gpuInstanceArr);
        args.emplace_back(idx);
        ah.SetAttribute(key, key2, NvmlFuncReturn(profileRet, args));
    }
    return true;
}

bool InjectedNvml::TopologyCommonAncestorParser(const std::string &key,
                                                const YAML::Node &node,
                                                AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node)
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
    {
        auto device2UUID     = it->first.as<std::string>();
        nvmlDevice_t device2 = m_uuidToDevice[device2UUID]->GetIdentifier();
        auto ancestor        = it->second;
        if (!ancestor[FunctionReturn])
        {
            NVML_LOG_ERR("ancestor lacks of FunctionReturn");
            return false;
        }
        auto nvmlRet = static_cast<nvmlReturn_t>(ancestor[FunctionReturn].as<int>());
        if (nvmlRet != NVML_SUCCESS)
        {
            ah.SetAttribute(key, device2, NvmlFuncReturn(nvmlRet));
            continue;
        }
        nvmlGpuTopologyLevel_t gpuTopologyLevel = static_cast<nvmlGpuTopologyLevel_t>(ancestor[ReturnValue].as<int>());
        ah.SetAttribute(key, device2, NvmlFuncReturn(nvmlRet, gpuTopologyLevel));
    }
    return true;
}

bool InjectedNvml::MigDeviceUUIDParser(const std::string &, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node)
    {
        return true;
    }
    unsigned int idx    = 0;
    auto migDeviceUUIDs = node.as<std::vector<std::string>>();
    for (const auto &migDeviceUUID : migDeviceUUIDs)
    {
        nvmlDevice_t *newDevicePtr = reinterpret_cast<nvmlDevice_t *>(malloc(sizeof(nvmlDevice_t)));
        *newDevicePtr              = GenNextNvmlDevice();
        m_migDeviceCollection.emplace_back(AttributeHolder<nvmlDevice_t>());
        m_uuidToDevice[migDeviceUUID] = std::prev(m_migDeviceCollection.end());
        m_devices[*newDevicePtr]      = std::prev(m_migDeviceCollection.end());
        ah.SetAttribute("MigDeviceHandleByIndex", idx, NvmlFuncReturn(NVML_SUCCESS, newDevicePtr));
        idx += 1;
    }
    return true;
}

bool InjectedNvml::OnSameBoardParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node)
    {
        return true;
    }
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
    {
        auto device2UUID     = it->first.as<std::string>();
        nvmlDevice_t device2 = m_uuidToDevice[device2UUID]->GetIdentifier();
        auto value           = it->second;
        if (!value[FunctionReturn])
        {
            NVML_LOG_ERR("ancestor lacks of FunctionReturn");
            return false;
        }
        auto nvmlRet = static_cast<nvmlReturn_t>(value[FunctionReturn].as<int>());
        if (nvmlRet != NVML_SUCCESS)
        {
            ah.SetAttribute(key, device2, NvmlFuncReturn(nvmlRet));
            continue;
        }
        int onSameBoard = static_cast<int>(value[ReturnValue].as<bool>());
        ah.SetAttribute(key, device2, NvmlFuncReturn(nvmlRet, onSameBoard));
    }
    return true;
}

bool InjectedNvml::TopologyNearestGpuParser(const std::string &key,
                                            const YAML::Node &node,
                                            AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node)
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
    {
        auto level = static_cast<nvmlGpuTopologyLevel_t>(it->first.as<int>());
        auto gpus  = it->second;
        if (!gpus[FunctionReturn])
        {
            NVML_LOG_ERR("gpus lacks of FunctionReturn");
            return false;
        }
        auto nvmlRet = static_cast<nvmlReturn_t>(gpus[FunctionReturn].as<int>());
        if (nvmlRet != NVML_SUCCESS)
        {
            ah.SetAttribute(key, level, NvmlFuncReturn(nvmlRet));
            continue;
        }
        unsigned int size = gpus[ReturnValue].size();
        int idx           = 0;
        auto *deviceArr   = reinterpret_cast<nvmlDevice_t *>(malloc(sizeof(nvmlDevice_t) * size));
        for (YAML::const_iterator gpuIt = gpus[ReturnValue].begin(); gpuIt != gpus[ReturnValue].end(); ++gpuIt)
        {
            int deviceIdx       = gpuIt->as<int>();
            nvmlDevice_t device = m_indexToDevice[deviceIdx]->GetIdentifier();
            deviceArr[idx++]    = device;
        }
        std::vector<InjectionArgument> args;
        args.emplace_back(size);
        args.emplace_back(deviceArr, size);
        ah.SetAttribute(key, level, NvmlFuncReturn(nvmlRet, args));
    }
    return true;
}

bool InjectedNvml::FieldValuesParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah)
{
    if (!node)
    {
        ah.SetAttribute(key, NvmlFuncReturn(NVML_ERROR_UNKNOWN));
        return false;
    }
    auto ret = static_cast<nvmlReturn_t>(node[FunctionReturn].as<int>(NVML_ERROR_UNKNOWN));
    if (ret != NVML_SUCCESS || !node[ReturnValue])
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    ah.SetAttribute(key, NvmlFuncReturn(NVML_SUCCESS));
    for (YAML::const_iterator it = node[ReturnValue].begin(); it != node[ReturnValue].end(); ++it)
    {
        auto fieldValue = it->second;
        if (!fieldValue["fieldId"] || !fieldValue["scopeId"] || !fieldValue["timestamp"] || !fieldValue["latencyUsec"]
            || !fieldValue["valueType"] || !fieldValue["nvmlReturn"] || !fieldValue["value"])
        {
            NVML_LOG_ERR("fieldValue misses expectation");
            return false;
        }

        nvmlFieldValue_t value;
        value.fieldId     = fieldValue["fieldId"].as<unsigned int>();
        value.scopeId     = fieldValue["scopeId"].as<unsigned int>();
        value.timestamp   = fieldValue["timestamp"].as<long long>();
        value.latencyUsec = fieldValue["latencyUsec"].as<long long>();
        value.valueType   = static_cast<nvmlValueType_t>(fieldValue["valueType"].as<int>());
        value.nvmlReturn  = static_cast<nvmlReturn_t>(fieldValue["nvmlReturn"].as<int>());
        switch (value.valueType)
        {
            case NVML_VALUE_TYPE_DOUBLE:
                value.value.dVal = fieldValue["value"].as<double>();
                break;
            case NVML_VALUE_TYPE_UNSIGNED_INT:
                value.value.uiVal = fieldValue["value"].as<unsigned int>();
                break;
            case NVML_VALUE_TYPE_UNSIGNED_LONG:
                value.value.ulVal = fieldValue["value"].as<unsigned long>();
                break;
            case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
                value.value.ullVal = fieldValue["value"].as<unsigned long long>();
                break;
            case NVML_VALUE_TYPE_SIGNED_LONG_LONG:
                value.value.sllVal = fieldValue["value"].as<signed long long>();
                break;
            default:
                NVML_LOG_ERR("not handled value type: %d", value.valueType);
                return false;
        }

        ah.SetFieldValue(value);
    }
    return true;
}

bool InjectedNvml::ParseOneDevice(const YAML::Node &device, AttributeHolder<nvmlDevice_t> &ah)
{
    NvmlReturnDeserializer nvmlReturnDeserializer;
    // key -> value parser
    std::unordered_map<std::string,
                       std::function<bool(const std::string &, const YAML::Node &, AttributeHolder<nvmlDevice_t> &)>>
        handlers {
            { INJECTION_ACTIVEVGPUS_KEY,
              std::bind(&InjectedNvml::ActiveVgpusParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
            { INJECTION_GPUINSTANCES_KEY,
              std::bind(&InjectedNvml::GpuInstancesParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
            { INJECTION_TOPOLOGYCOMMONANCESTOR_KEY,
              std::bind(&InjectedNvml::TopologyCommonAncestorParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
            { INJECTION_TOPOLOGYNEARESTGPUS_KEY,
              std::bind(&InjectedNvml::TopologyNearestGpuParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
            { "MigDeviceUUID",
              std::bind(&InjectedNvml::MigDeviceUUIDParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
            { INJECTION_ONSAMEBOARD_KEY,
              std::bind(&InjectedNvml::OnSameBoardParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
            { INJECTION_FIELDVALUES_KEY,
              std::bind(&InjectedNvml::FieldValuesParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
            { INJECTION_MEMORYERRORCOUNTER_KEY, MemoryErrorCounterParser },
            { INJECTION_REMAPPEDROWS_KEY, RemappedRowsParser },
            { INJECTION_PROCESSUTILIZATION_KEY, ProcessUtilizationParser },
            { INJECTION_VGPUPROCESSUTILIZATION_KEY, VgpuProcessUtilizationParser },
            { INJECTION_VGPUUTILIZATION_KEY, VgpuInstanceUtilizationParser },
        };

    // key -> [key2 parser, value parser]
    std::unordered_map<std::string,
                       std::tuple<std::function<std::optional<InjectionArgument>(const YAML::Node &)>,
                                  std::function<std::optional<NvmlFuncReturn>(const YAML::Node &, nvmlReturn_t)>>>
        extraKeyHandlers {

        };

    if (!device)
    {
        return false;
    }

    for (YAML::const_iterator it = device.begin(); it != device.end(); ++it)
    {
        auto key         = it->first.as<std::string>();
        YAML::Node value = it->second;

        if (!value)
        {
            // Some key may not always be recorded (e.g. nvmlDeviceGetAccountingStats)
            continue;
        }

        if (handlers.contains(key))
        {
            if (!handlers[key](key, value, ah))
            {
                NVML_LOG_ERR("failed to handle key [%s].", key.c_str());
            }
            continue;
        }

        auto parsedResultOpt = nvmlReturnDeserializer.DeviceHandle(key, value);
        if (parsedResultOpt)
        {
            ah.SetAttribute(key, parsedResultOpt.value());
            continue;
        }
        auto extraKeyParsedRetOpt = nvmlReturnDeserializer.DeviceExtraKeyHandle(key, value);
        if (extraKeyParsedRetOpt)
        {
            for (const auto &[key2, ret] : extraKeyParsedRetOpt.value())
            {
                ah.SetAttribute(key, key2, ret);
            }
            continue;
        }
        auto threeKeysParsedRetOpt = nvmlReturnDeserializer.DeviceThreeKeysHandle(key, value);
        if (threeKeysParsedRetOpt)
        {
            for (const auto &[key2, key3, ret] : threeKeysParsedRetOpt.value())
            {
                ah.SetAttribute(key, key2, key3, ret);
            }
            continue;
        }
    }
    return true;
}

bool InjectedNvml::ParseDevices(const YAML::Node &devices)
{
    if (!devices)
    {
        return true;
    }

    for (auto iter = m_deviceCollection.begin(); iter != m_deviceCollection.end(); iter++)
    {
        auto &ah = *iter;
        std::string deviceUUID
            = ah.GetAttribute(INJECTION_UUID_KEY).GetCompoundValue().AsInjectionArgument().AsString();

        if (!devices[deviceUUID])
        {
            NVML_LOG_ERR("missing UUID [%s] in device section", deviceUUID.c_str());
            return false;
        }
        if (!ParseOneDevice(devices[deviceUUID], ah))
        {
            NVML_LOG_ERR("failed to parse UUID [%s] in device section", deviceUUID.c_str());
            return false;
        }

        std::string deviceSerial
            = ah.GetAttribute(INJECTION_SERIAL_KEY).GetCompoundValue().AsInjectionArgument().AsString();
        if (!deviceSerial.empty()) // Some GPUs do not support serial number
        {
            m_serialToDevice[deviceSerial] = iter;
        }
        nvmlPciInfo_t *pciInfo
            = ah.GetAttribute(INJECTION_PCIINFO_KEY).GetCompoundValue().AsInjectionArgument().AsPciInfoPtr();
        if (pciInfo != nullptr)
        {
            m_busIdToDevice[pciInfo->busId] = iter;
        }
    }

    return true;
}

bool InjectedNvml::ComputeInstancesParser(const std::string &key,
                                          const YAML::Node &node,
                                          AttributeHolder<nvmlGpuInstance_t> &ah)
{
    if (!node)
    {
        return true;
    }
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
    {
        auto profileId            = it->first.as<unsigned int>();
        auto fakeComputeInstances = it->second.as<std::vector<std::string>>();
        auto *computeInstanceArr  = reinterpret_cast<nvmlComputeInstance_t *>(
            malloc(sizeof(nvmlComputeInstance_t) * fakeComputeInstances.size()));
        int idx = 0;
        for (const auto &computeInstance : fakeComputeInstances)
        {
            m_computeInstanceCollection.emplace_back(computeInstance);
            computeInstanceArr[idx] = reinterpret_cast<nvmlComputeInstance_t>(&m_computeInstanceCollection.back());
            m_computeInstances[computeInstanceArr[idx]] = AttributeHolder<nvmlComputeInstance_t>();
            idx += 1;
        }
        std::vector<InjectionArgument> args;
        args.emplace_back(computeInstanceArr);
        args.emplace_back(idx);
        ah.SetAttribute(key, profileId, NvmlFuncReturn(NVML_SUCCESS, args));
    }
    return true;
}

bool InjectedNvml::GpuInstanceInfoParser(const std::string &key,
                                         const YAML::Node &node,
                                         AttributeHolder<nvmlGpuInstance_t> &ah)
{
    if (!node || !node[FunctionReturn])
    {
        return false;
    }
    auto ret = static_cast<nvmlReturn_t>(node[FunctionReturn].as<int>(NVML_ERROR_UNKNOWN));
    if (!node[ReturnValue])
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    if (!node[ReturnValue]["device"] || !node[ReturnValue]["id"] || !node[ReturnValue]["placement"]
        || !node[ReturnValue]["profileId"] || !node[ReturnValue]["placement"]["size"]
        || !node[ReturnValue]["placement"]["start"])
    {
        NVML_LOG_ERR("failed to parse GPU instance info due to missing entries");
        return false;
    }
    auto *info            = reinterpret_cast<nvmlGpuInstanceInfo_t *>(malloc(sizeof(nvmlGpuInstanceInfo_t)));
    info->device          = m_uuidToDevice[node[ReturnValue]["device"].as<std::string>()]->GetIdentifier();
    info->id              = node[ReturnValue]["id"].as<unsigned int>();
    info->profileId       = node[ReturnValue]["id"].as<unsigned int>();
    info->placement.size  = node[ReturnValue]["placement"]["size"].as<unsigned int>();
    info->placement.start = node[ReturnValue]["placement"]["start"].as<unsigned int>();
    ah.SetAttribute(key, NvmlFuncReturn(ret, info));
    return true;
}

bool InjectedNvml::ComputeInstanceInfoParser(const std::string &key,
                                             const YAML::Node &node,
                                             AttributeHolder<nvmlComputeInstance_t> &ah)
{
    if (!node || !node[FunctionReturn])
    {
        return false;
    }
    auto ret = static_cast<nvmlReturn_t>(node[FunctionReturn].as<int>(NVML_ERROR_UNKNOWN));
    if (!node[ReturnValue])
    {
        ah.SetAttribute(key, NvmlFuncReturn(ret));
        return true;
    }
    if (!node[ReturnValue]["device"] || !node[ReturnValue]["gpuInstance"] || !node[ReturnValue]["id"]
        || !node[ReturnValue]["profileId"] || !node[ReturnValue]["placement"]["size"]
        || !node[ReturnValue]["placement"]["start"])
    {
        NVML_LOG_ERR("failed to parse GPU instance info due to missing entries");
        return false;
    }

    nvmlGpuInstance_t gpuInstance = nullptr;
    for (auto &gpuInstanceFakeId : m_gpuInstanceCollection)
    {
        if (gpuInstanceFakeId == node[ReturnValue]["gpuInstance"].as<std::string>())
        {
            gpuInstance = reinterpret_cast<nvmlGpuInstance_t>(&gpuInstanceFakeId);
        }
    }

    if (!gpuInstance)
    {
        return false;
    }
    auto *info            = reinterpret_cast<nvmlComputeInstanceInfo_t *>(malloc(sizeof(nvmlComputeInstanceInfo_t)));
    info->device          = m_uuidToDevice[node[ReturnValue]["device"].as<std::string>()]->GetIdentifier();
    info->gpuInstance     = gpuInstance;
    info->id              = node[ReturnValue]["id"].as<unsigned int>();
    info->profileId       = node[ReturnValue]["id"].as<unsigned int>();
    info->placement.size  = node[ReturnValue]["placement"]["size"].as<unsigned int>();
    info->placement.start = node[ReturnValue]["placement"]["start"].as<unsigned int>();
    ah.SetAttribute(key, NvmlFuncReturn(ret, info));
    return true;
}

bool InjectedNvml::ParseOneGpuInstance(const YAML::Node &gpuInstance, AttributeHolder<nvmlGpuInstance_t> &ah)
{
    NvmlReturnDeserializer nvmlReturnDeserializer;
    // key -> value parser
    std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const YAML::Node &, AttributeHolder<nvmlGpuInstance_t> &)>>
        handlers {
            { INJECTION_COMPUTEINSTANCES_KEY,
              std::bind(&InjectedNvml::ComputeInstancesParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
            { INJECTION_INFO_KEY,
              std::bind(&InjectedNvml::GpuInstanceInfoParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
        };

    for (YAML::const_iterator it = gpuInstance.begin(); it != gpuInstance.end(); ++it)
    {
        auto key          = it->first.as<std::string>();
        auto const &value = it->second;

        if (handlers.contains(key))
        {
            if (!handlers[key](key, value, ah))
            {
                NVML_LOG_ERR("failed to handle key [%s]", key.c_str());
            }
            continue;
        }

        auto parsedResultOpt = nvmlReturnDeserializer.GpuInstanceHandle(key, value);
        if (parsedResultOpt)
        {
            ah.SetAttribute(key, parsedResultOpt.value());
            continue;
        }
        auto extraKeyParsedRetOpt = nvmlReturnDeserializer.GpuInstanceExtraKeyHandle(key, value);
        if (extraKeyParsedRetOpt)
        {
            for (const auto &[key2, ret] : extraKeyParsedRetOpt.value())
            {
                ah.SetAttribute(key, key2, ret);
            }
            continue;
        }
        auto threeKeysParsedRetOpt = nvmlReturnDeserializer.GpuInstanceThreeKeysHandle(key, value);
        if (threeKeysParsedRetOpt)
        {
            for (const auto &[key2, key3, ret] : threeKeysParsedRetOpt.value())
            {
                ah.SetAttribute(key, key2, key3, ret);
            }
            continue;
        }
    }
    return true;
}

bool InjectedNvml::ParseGpuInstances(const YAML::Node &gpuInstances)
{
    if (!gpuInstances)
    {
        // some devices may not have it
        return true;
    }

    for (auto &gpuInstanceFakeId : m_gpuInstanceCollection)
    {
        AttributeHolder<nvmlGpuInstance_t> &ah
            = m_gpuInstances[reinterpret_cast<nvmlGpuInstance_t>(&gpuInstanceFakeId)];

        if (!gpuInstances[gpuInstanceFakeId])
        {
            NVML_LOG_ERR("missing GPU instance [%s] in GpuInstance section", gpuInstanceFakeId.c_str());
            return false;
        }
        if (!ParseOneGpuInstance(gpuInstances[gpuInstanceFakeId], ah))
        {
            NVML_LOG_ERR("failed to parse GPU instance [%s] in GpuInstance section", gpuInstanceFakeId.c_str());
            return false;
        }
    }

    return true;
}

bool InjectedNvml::ParseOneComputeInstance(const YAML::Node &computeInstance,
                                           AttributeHolder<nvmlComputeInstance_t> &ah)
{
    NvmlReturnDeserializer nvmlReturnDeserializer;
    // key -> value parser
    std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const YAML::Node &, AttributeHolder<nvmlComputeInstance_t> &)>>
        handlers {
            { INJECTION_INFO_KEY,
              std::bind(&InjectedNvml::ComputeInstanceInfoParser,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3) },
        };

    for (YAML::const_iterator it = computeInstance.begin(); it != computeInstance.end(); ++it)
    {
        auto key          = it->first.as<std::string>();
        auto const &value = it->second;

        if (handlers.contains(key))
        {
            if (!handlers[key](key, value, ah))
            {
                NVML_LOG_ERR("failed to handle key [%s]", key.c_str());
            }
            continue;
        }
        auto parsedResultOpt = nvmlReturnDeserializer.ComputeInstanceHandle(key, value);
        if (parsedResultOpt)
        {
            ah.SetAttribute(key, parsedResultOpt.value());
            continue;
        }
    }
    return true;
}

bool InjectedNvml::ParseComputeInstances(const YAML::Node &computeInstances)
{
    if (!computeInstances)
    {
        // some devices may not have it
        return true;
    }

    for (auto &computeInstanceFakeId : m_computeInstanceCollection)
    {
        AttributeHolder<nvmlComputeInstance_t> &ah
            = m_computeInstances[reinterpret_cast<nvmlComputeInstance_t>(&computeInstanceFakeId)];

        if (!computeInstances[computeInstanceFakeId])
        {
            NVML_LOG_ERR("missing compute instance [%s] in ComputeInstance section", computeInstanceFakeId.c_str());
            return false;
        }
        if (!ParseOneComputeInstance(computeInstances[computeInstanceFakeId], ah))
        {
            NVML_LOG_ERR("failed to parse compute instance [%s] in ComputeInstance section",
                         computeInstanceFakeId.c_str());
            return false;
        }
    }

    return true;
}

bool InjectedNvml::ParseOneVgpuType(const YAML::Node &vgpuType, AttributeHolder<nvmlVgpuTypeId_t> &ah)
{
    NvmlReturnDeserializer nvmlReturnDeserializer;

    for (YAML::const_iterator it = vgpuType.begin(); it != vgpuType.end(); ++it)
    {
        auto key          = it->first.as<std::string>();
        auto const &value = it->second;

        auto parsedResultOpt = nvmlReturnDeserializer.VgpuTypeHandle(key, value);
        if (parsedResultOpt)
        {
            ah.SetAttribute(key, parsedResultOpt.value());
            continue;
        }
        auto extraKeyParsedRetOpt = nvmlReturnDeserializer.VgpuTypeExtraKeyHandle(key, value);
        if (extraKeyParsedRetOpt)
        {
            for (const auto &[key2, ret] : extraKeyParsedRetOpt.value())
            {
                ah.SetAttribute(key, key2, ret);
            }
            continue;
        }
    }
    return true;
}

bool InjectedNvml::ParseVgpuTypes(const YAML::Node &vgpuType)
{
    if (!vgpuType)
    {
        // some devices may not have it
        return true;
    }

    for (YAML::const_iterator it = vgpuType.begin(); it != vgpuType.end(); ++it)
    {
        auto vgpuTypeId           = static_cast<nvmlVgpuTypeId_t>(it->first.as<unsigned int>());
        m_vgpuTypeIds[vgpuTypeId] = AttributeHolder<nvmlVgpuTypeId_t>();

        if (!ParseOneVgpuType(it->second, m_vgpuTypeIds[vgpuTypeId]))
        {
            NVML_LOG_ERR("failed to parse vGPU type [%u] in vGPUType section", vgpuTypeId);
            return false;
        }
    }

    return true;
}

bool InjectedNvml::ParseOneVgpuInstance(const YAML::Node &vgpuInstance, AttributeHolder<nvmlVgpuInstance_t> &ah)
{
    NvmlReturnDeserializer nvmlReturnDeserializer;
    // key -> value parser
    std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const YAML::Node &, AttributeHolder<nvmlVgpuInstance_t> &)>>
        handlers {
            { INJECTION_VMID_KEY, VmIdParser },
        };

    for (YAML::const_iterator it = vgpuInstance.begin(); it != vgpuInstance.end(); ++it)
    {
        auto key          = it->first.as<std::string>();
        auto const &value = it->second;

        if (handlers.contains(key))
        {
            if (!handlers[key](key, value, ah))
            {
                NVML_LOG_ERR("failed to handle key [%s]", key.c_str());
            }
            continue;
        }
        auto parsedResultOpt = nvmlReturnDeserializer.VgpuInstanceHandle(key, value);
        if (parsedResultOpt)
        {
            ah.SetAttribute(key, parsedResultOpt.value());
            continue;
        }
    }
    return true;
}

bool InjectedNvml::ParseVgpuInstances(const YAML::Node &vgpuInstances)
{
    if (!vgpuInstances)
    {
        // some devices may not have it
        return true;
    }

    for (YAML::const_iterator it = vgpuInstances.begin(); it != vgpuInstances.end(); ++it)
    {
        auto vgpuInstance             = static_cast<nvmlVgpuInstance_t>(it->first.as<unsigned int>());
        m_vgpuInstances[vgpuInstance] = AttributeHolder<nvmlVgpuInstance_t>();

        if (!ParseOneVgpuInstance(it->second, m_vgpuInstances[vgpuInstance]))
        {
            NVML_LOG_ERR("failed to parse vGPU instance [%u] in vGPUType section", vgpuInstance);
            return false;
        }
    }

    return true;
}

bool InjectedNvml::ParseMigDevices(const YAML::Node &migDevices)
{
    if (!migDevices)
    {
        // some devices may not have it
        return true;
    }

    for (YAML::const_iterator it = migDevices.begin(); it != migDevices.end(); ++it)
    {
        auto migDeviceUUID = it->first.as<std::string>();
        auto &ah           = m_uuidToDevice[migDeviceUUID];

        if (!ParseOneDevice(it->second, *ah))
        {
            NVML_LOG_ERR("failed to parse mig device UUID [%s] in MigDevice section", migDeviceUUID.c_str());
            return false;
        }
    }

    return true;
}

/*****************************************************************************/
bool InjectedNvml::LoadFromFile(const std::string &path)
{
    YAML::Node root;

    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const std::exception &e)
    {
        NVML_LOG_ERR("failed to YAML load [%s], reason [%s]", path.c_str(), e.what());
        return false;
    }

    if (!LoadFromYamlNode(root))
    {
        NVML_LOG_ERR("failed to parse file [%s]", path.c_str());
        return false;
    }
    return true;
}

/*****************************************************************************/
bool InjectedNvml::LoadFromString(const std::string &yamlContent)
{
    YAML::Node root;

    try
    {
        root = YAML::Load(yamlContent);
    }
    catch (const std::exception &e)
    {
        NVML_LOG_ERR("failed to YAML load [%s], reason [%s]", yamlContent.c_str(), e.what());
        return false;
    }

    if (!LoadFromYamlNode(root))
    {
        NVML_LOG_ERR("failed to parse content [%s]", yamlContent.c_str());
        return false;
    }
    return true;
}

/*****************************************************************************/
bool InjectedNvml::LoadFromYamlNode(const YAML::Node &root)
{
    if (!ParseGlobal(root[GLOBAL]))
    {
        NVML_LOG_ERR("failed to parse global part");
        return false;
    }

    if (!ParseDevices(root[DEVICE]))
    {
        NVML_LOG_ERR("failed to parse device part");
        return false;
    }

    if (!ParseGpuInstances(root[GPU_INASTANCE]))
    {
        NVML_LOG_ERR("failed to parse GPU instance part");
        return false;
    }

    if (!ParseComputeInstances(root[COMPUTE_INASTANCE]))
    {
        NVML_LOG_ERR("failed to parse vGPU instance part");
        return false;
    }

    if (!ParseVgpuTypes(root[VGPU_TYPE]))
    {
        NVML_LOG_ERR("failed to parse vGPU type part");
        return false;
    }

    if (!ParseVgpuInstances(root[VGPU_INSTANCE]))
    {
        NVML_LOG_ERR("failed to parse vGPU instance part");
        return false;
    }

    if (!ParseMigDevices(root[MIG_DEVICE]))
    {
        NVML_LOG_ERR("failed to parse vGPU instance part");
        return false;
    }
    return true;
}

/*****************************************************************************/
bool InjectedNvml::IsGetter(const std::string &funcname) const
{
    if (funcname.starts_with("nvmlDeviceGet") || funcname.starts_with("nvmlGpuInstanceGet")
        || funcname == "nvmlEventSetWait_v2" || funcname.starts_with("nvmlComputeInstanceGet")
        || funcname.starts_with("nvmlVgpuInstanceGet") || funcname.starts_with("nvmlVgpuTypeGet")
        || funcname.starts_with("nvmlDeviceWorkloadPowerProfileGet") || funcname == "nvmlDeviceValidateInforom")
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
bool InjectedNvml::IsSetter(const std::string & /* funcname */) const
{
    return false;
}

/*****************************************************************************/
std::optional<nvmlReturn_t> InjectedNvml::GetWrapperSpecialCase(const std::string &funcname,
                                                                const std::string &key,
                                                                std::vector<InjectionArgument> &args,
                                                                std::vector<InjectionArgument> &values)
{
    // funcname -> return NVML_ERROR_INSUFFICIENT_SIZE when has element
    std::unordered_map<std::string, bool> queryArraySizeFuncs {
        { "nvmlDeviceGetSupportedVgpus", true },         { "nvmlDeviceGetActiveVgpus", true },
        { "nvmlDeviceGetFBCSessions", false },           { "nvmlDeviceGetCreatableVgpus", true },
        { "nvmlVgpuInstanceGetEncoderSessions", false }, { "nvmlVgpuInstanceGetFBCSessions", false },
    };
    auto queryArraySizeHandler = [&](const bool returnInsufficientSizeWhenHasValue) -> std::optional<nvmlReturn_t> {
        if (IsDeviceFunc(funcname, args) && !m_devices.contains(args[0].AsDevice()))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
        if (IsVgpuInstanceFunc(funcname, args) && !m_vgpuInstances.contains(args[0].AsUInt()))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
        if (values.size() != 2 || values[0].GetType() != INJECTION_UINT_PTR)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
        if (*values[0].AsUIntPtr() == 0)
        {
            NvmlFuncReturn nvmlFuncReturn;
            if (IsDeviceFunc(funcname, args))
            {
                nvmlFuncReturn = m_devices[args[0].AsDevice()]->GetAttribute(key);
            }
            else
            {
                nvmlFuncReturn = m_vgpuInstances[args[0].AsUInt()].GetAttribute(key);
            }
            if (!nvmlFuncReturn.IsNvmlSucces())
            {
                return nvmlFuncReturn.GetRet();
            }
            if (nvmlFuncReturn.GetCompoundValue().RawValues().size() < 1)
            {
                return NVML_ERROR_UNKNOWN;
            }
            *values[0].AsUIntPtr() = nvmlFuncReturn.GetCompoundValue().RawValues()[0].AsUInt();
            if (*values[0].AsUIntPtr())
            {
                return returnInsufficientSizeWhenHasValue ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS;
            }
            return NVML_SUCCESS;
        }
        return std::nullopt;
    };

    auto it = queryArraySizeFuncs.find(funcname);
    if (it != queryArraySizeFuncs.end())
    {
        return queryArraySizeHandler(it->second);
    }
    if (funcname == "nvmlEventSetWait_v2")
    {
        return NVML_ERROR_TIMEOUT;
    }
    if (funcname == "nvmlGpuInstanceGetComputeInstances")
    {
        nvmlGpuInstance_t gpuInstance           = args[0].AsGpuInstance();
        nvmlComputeInstance_t *computeInstances = values[0].AsComputeInstancePtr();
        unsigned int *count                     = values[1].AsUIntPtr();
        NvmlFuncReturn nvmlFuncReturn
            = m_gpuInstances[gpuInstance].GetAttribute(INJECTION_COMPUTEINSTANCES_KEY, args[1]);
        if (!nvmlFuncReturn.IsNvmlSucces())
        {
            return nvmlFuncReturn.GetRet();
        }
        nvmlComputeInstance_t *actualComputeInstances
            = nvmlFuncReturn.GetCompoundValue().RawValues()[0].AsComputeInstancePtr();
        unsigned actualCount = nvmlFuncReturn.GetCompoundValue().RawValues()[1].AsUInt();
        for (unsigned i = 0; i < actualCount; ++i)
        {
            std::memcpy(&computeInstances[i], &actualComputeInstances[i], sizeof(nvmlComputeInstance_t));
        }
        *count = actualCount;
        return NVML_SUCCESS;
    }
    if (funcname == "nvmlDeviceGetGpuInstances")
    {
        if (args.size() != 2 || args[0].GetType() != INJECTION_DEVICE || args[1].GetType() != INJECTION_UINT
            || values.size() != 2 || values[0].GetType() != INJECTION_GPUINSTANCE_PTR
            || values[0].AsGpuInstancePtr() == nullptr || values[1].GetType() != INJECTION_UINT_PTR
            || values[1].AsUIntPtr() == nullptr || !m_devices.contains(args[0].AsDevice()))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        NvmlFuncReturn nvmlFuncRet = m_devices[args[0].AsDevice()]->GetAttribute(key, args[1].AsUInt());
        if (!nvmlFuncRet.IsNvmlSucces())
        {
            return nvmlFuncRet.GetRet();
        }
        for (unsigned i = 0; i < nvmlFuncRet.GetCompoundValue().RawValues()[1].AsUInt(); ++i)
        {
            std::memcpy(&values[0].AsGpuInstancePtr()[i],
                        &nvmlFuncRet.GetCompoundValue().RawValues()[0].AsGpuInstancePtr()[i],
                        sizeof(nvmlGpuInstance_t));
        }
        values[1].SetValueFrom(nvmlFuncRet.GetCompoundValue().RawValues()[1]);
        return NVML_SUCCESS;
    }
    if (funcname == "nvmlVgpuInstanceGetVmID")
    {
        if (args.size() != 2 || values.size() != 2 || values[0].GetType() != INJECTION_CHAR_PTR
            || !m_vgpuInstances.contains(args[0].AsUInt()))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
        auto nvmlFuncReturn = m_vgpuInstances[args[0].AsUInt()].GetAttribute(key);
        if (!nvmlFuncReturn.IsNvmlSucces())
        {
            return nvmlFuncReturn.GetRet();
        }
        if (nvmlFuncReturn.GetCompoundValue().RawValues().size() < 2)
        {
            return NVML_ERROR_UNKNOWN;
        }
        auto vmId = nvmlFuncReturn.GetCompoundValue().RawValues()[0].AsString();
        if (vmId.size() > args[1].AsUInt())
        {
            return NVML_ERROR_INSUFFICIENT_SIZE;
        }
        snprintf(values[0].AsStr(), vmId.size(), "%s", vmId.c_str());
        values[1].SetValueFrom(nvmlFuncReturn.GetCompoundValue().RawValues()[1]);
        return NVML_SUCCESS;
    }
    if (funcname == "nvmlDeviceGetProcessUtilization")
    {
        if (args.size() != 2 || args[0].GetType() != INJECTION_DEVICE || args[1].GetType() != INJECTION_ULONG_LONG
            || values.size() != 2 || values[0].GetType() != INJECTION_PROCESSUTILIZATIONSAMPLE_PTR
            || values[1].GetType() != INJECTION_UINT_PTR || values[1].AsUIntPtr() == nullptr
            || !m_devices.contains(args[0].AsDevice()))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
        auto funcRet = m_devices[args[0].AsDevice()]->GetAttribute(key);
        if (!funcRet.IsNvmlSucces())
        {
            return funcRet.GetRet();
        }
        auto records = m_devices[args[0].AsDevice()]->GetProcessUtilizationRecord(args[1].AsULongLong());
        // special case for querying size
        if (values[0].AsProcessUtilizationSamplePtr() == nullptr || *values[1].AsUIntPtr() < records.size())
        {
            *values[1].AsUIntPtr() = records.size();
            return NVML_ERROR_INSUFFICIENT_SIZE;
        }
        *values[1].AsUIntPtr() = records.size();
        for (size_t i = 0; i < records.size(); ++i)
        {
            std::memcpy(&values[0].AsProcessUtilizationSamplePtr()[i], &records[i], sizeof(records[i]));
        }
        return NVML_SUCCESS;
    }
    if (funcname == "nvmlDeviceGetVgpuProcessUtilization")
    {
        if (args.size() != 2 || args[0].GetType() != INJECTION_DEVICE || args[1].GetType() != INJECTION_ULONG_LONG
            || values.size() != 2 || values[0].GetType() != INJECTION_UINT_PTR
            || values[1].GetType() != INJECTION_VGPUPROCESSUTILIZATIONSAMPLE_PTR || values[0].AsUIntPtr() == nullptr
            || !m_devices.contains(args[0].AsDevice()))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
        auto funcRet = m_devices[args[0].AsDevice()]->GetAttribute(key);
        if (!funcRet.IsNvmlSucces())
        {
            return funcRet.GetRet();
        }
        auto records = m_devices[args[0].AsDevice()]->GetVgpuProcessUtilizationRecord(args[1].AsULongLong());
        // special case for querying size
        if (values[1].AsVgpuProcessUtilizationSamplePtr() == nullptr || *values[0].AsUIntPtr() < records.size())
        {
            *values[0].AsUIntPtr() = records.size();
            return NVML_ERROR_INSUFFICIENT_SIZE;
        }
        *values[0].AsUIntPtr() = records.size();
        for (size_t i = 0; i < records.size(); ++i)
        {
            std::memcpy(&values[1].AsVgpuProcessUtilizationSamplePtr()[i], &records[i], sizeof(records[i]));
        }
        return NVML_SUCCESS;
    }
    if (funcname == "nvmlDeviceGetVgpuUtilization")
    {
        if (args.size() != 2 || args[0].GetType() != INJECTION_DEVICE || args[1].GetType() != INJECTION_ULONG_LONG
            || values.size() != 3 || values[0].GetType() != INJECTION_VALUETYPE_PTR
            || values[1].GetType() != INJECTION_UINT_PTR || values[1].AsUIntPtr() == nullptr
            || values[2].GetType() != INJECTION_VGPUINSTANCEUTILIZATIONSAMPLE_PTR
            || !m_devices.contains(args[0].AsDevice()))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
        auto funcRet = m_devices[args[0].AsDevice()]->GetAttribute(key);
        if (!funcRet.IsNvmlSucces())
        {
            return funcRet.GetRet();
        }
        auto records = m_devices[args[0].AsDevice()]->GetVgpuInstanceUtilizationRecord(args[1].AsULongLong());
        // special case for querying size
        if (values[2].AsVgpuInstanceUtilizationSamplePtr() == nullptr || *values[1].AsUIntPtr() < records.size())
        {
            *values[1].AsUIntPtr() = records.size();
            if (records.size() == 0)
            {
                return NVML_SUCCESS;
            }
            return NVML_ERROR_INSUFFICIENT_SIZE;
        }
        *values[1].AsUIntPtr() = records.size();
        for (size_t i = 0; i < records.size(); ++i)
        {
            auto &[valueType, sample]   = records[i];
            *values[0].AsValueTypePtr() = valueType;
            std::memcpy(&values[2].AsVgpuInstanceUtilizationSamplePtr()[i], &sample, sizeof(sample));
        }
        return NVML_SUCCESS;
    }
    if (funcname == "nvmlDeviceGetMemoryInfo")
    {
        if (args.size() != 1 || args[0].GetType() != INJECTION_DEVICE || values.size() != 1
            || values[0].GetType() != INJECTION_MEMORY_PTR || !m_devices.contains(args[0].AsDevice()))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        NvmlFuncReturn nvmlFuncRet = m_devices[args[0].AsDevice()]->GetAttribute(key);
        if (!nvmlFuncRet.IsNvmlSucces())
        {
            return nvmlFuncRet.GetRet();
        }
        if (nvmlFuncRet.GetCompoundValue().RawValues().size() < 1)
        {
            return NVML_ERROR_UNKNOWN;
        }
        values[0].AsMemoryPtr()->free  = nvmlFuncRet.GetCompoundValue().RawValues()[0].AsMemory_v2Ptr()->free;
        values[0].AsMemoryPtr()->used  = nvmlFuncRet.GetCompoundValue().RawValues()[0].AsMemory_v2Ptr()->used;
        values[0].AsMemoryPtr()->total = nvmlFuncRet.GetCompoundValue().RawValues()[0].AsMemory_v2Ptr()->total;
        return NVML_SUCCESS;
    }
    return std::nullopt;
}

NvmlFuncReturn InjectedNvml::DeviceGetWrapper(const std::string &key, std::vector<InjectionArgument> &args)
{
    if (args.size() < 1 || args[0].GetType() != INJECTION_DEVICE || !m_devices.contains(args[0].AsDevice()))
    {
        return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
    }
    if (args.size() == 1)
    {
        return m_devices[args[0].AsDevice()]->GetAttribute(key);
    }
    if (args.size() == 2)
    {
        return m_devices[args[0].AsDevice()]->GetAttribute(key, args[1]);
    }
    if (args.size() == 3)
    {
        return m_devices[args[0].AsDevice()]->GetAttribute(key, args[1], args[2]);
    }
    if (args.size() == 4)
    {
        return m_devices[args[0].AsDevice()]->GetAttribute(key, args[1], args[2], args[3]);
    }
    return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
}

NvmlFuncReturn InjectedNvml::GpuInstanceGetWrapper(const std::string &key, std::vector<InjectionArgument> &args)
{
    if (args.size() < 1 || args[0].GetType() != INJECTION_GPUINSTANCE
        || !m_gpuInstances.contains(args[0].AsGpuInstance()))
    {
        return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
    }
    if (args.size() == 1)
    {
        return m_gpuInstances[args[0].AsGpuInstance()].GetAttribute(key);
    }
    if (args.size() == 2)
    {
        return m_gpuInstances[args[0].AsGpuInstance()].GetAttribute(key, args[1]);
    }
    if (args.size() == 3)
    {
        return m_gpuInstances[args[0].AsGpuInstance()].GetAttribute(key, args[1], args[2]);
    }
    return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
}

NvmlFuncReturn InjectedNvml::ComputeInstanceGetWrapper(const std::string &key, std::vector<InjectionArgument> &args)
{
    if (args.size() < 1 || args[0].GetType() != INJECTION_COMPUTEINSTANCE
        || !m_computeInstances.contains(args[0].AsComputeInstance()))
    {
        return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
    }
    if (args.size() == 1)
    {
        return m_computeInstances[args[0].AsComputeInstance()].GetAttribute(key);
    }
    if (args.size() == 2)
    {
        return m_computeInstances[args[0].AsComputeInstance()].GetAttribute(key, args[1]);
    }
    if (args.size() == 3)
    {
        return m_computeInstances[args[0].AsComputeInstance()].GetAttribute(key, args[1], args[2]);
    }
    return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
}

NvmlFuncReturn InjectedNvml::VgpuTypeIdGetWrapper(const std::string &key, std::vector<InjectionArgument> &args)
{
    if (args.size() < 1 || !m_vgpuTypeIds.contains(args[0].AsUInt()))
    {
        return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
    }
    if (args.size() == 1)
    {
        return m_vgpuTypeIds[args[0].AsUInt()].GetAttribute(key);
    }
    return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
}

NvmlFuncReturn InjectedNvml::VgpuInstanceGetWrapper(const std::string &key, std::vector<InjectionArgument> &args)
{
    if (args.size() < 1 || !m_vgpuInstances.contains(args[0].AsUInt()))
    {
        return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
    }
    if (args.size() == 1)
    {
        return m_vgpuInstances[args[0].AsUInt()].GetAttribute(key);
    }
    return NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
}

/*****************************************************************************/
nvmlReturn_t InjectedNvml::GetWrapper(const std::string &funcname,
                                      const std::string &key,
                                      std::vector<InjectionArgument> &args,
                                      std::vector<InjectionArgument> &values)
{
    // dcgm may try to call the following functions for testing, using the list to avoid misleading infomation
    std::unordered_set<std::string> allowNotInjectedFuncs {
        "nvmlGpuInstanceGetComputeInstanceProfileInfo",
        "nvmlDeviceGetGpuInstanceProfileInfo",
        "nvmlDeviceGetMigDeviceHandleByIndex",
    };
    std::lock_guard<std::mutex> guard(m_mutex);

    std::optional<nvmlReturn_t> nvmlRetOpt = GetWrapperSpecialCase(funcname, key, args, values);
    if (nvmlRetOpt.has_value())
    {
        return nvmlRetOpt.value();
    }

    NvmlFuncReturn nvmlFuncReturn;
    if (IsDeviceFunc(funcname, args))
    {
        nvmlFuncReturn = DeviceGetWrapper(key, args);
    }
    else if (IsGpuInstanceFunc(funcname, args))
    {
        nvmlFuncReturn = GpuInstanceGetWrapper(key, args);
    }
    else if (IsComputeInstanceFunc(funcname, args))
    {
        nvmlFuncReturn = ComputeInstanceGetWrapper(key, args);
    }
    else if (IsVgpuTypeFunc(funcname, args))
    {
        nvmlFuncReturn = VgpuTypeIdGetWrapper(key, args);
    }
    else if (IsVgpuInstanceFunc(funcname, args))
    {
        nvmlFuncReturn = VgpuInstanceGetWrapper(key, args);
    }
    else
    {
        NVML_LOG_ERR("Calling function [%s] not injected.", funcname.c_str());
        nvmlFuncReturn = NvmlFuncReturn(NVML_ERROR_INVALID_ARGUMENT);
    }

    if (!nvmlFuncReturn.HasValue())
    {
        if (!allowNotInjectedFuncs.contains(funcname))
        {
            NVML_LOG_ERR("calling a function [%s] without injection.", funcname.c_str());
        }
    }
    if (!nvmlFuncReturn.IsNvmlSucces())
    {
        return nvmlFuncReturn.GetRet();
    }
    if (values.size() > nvmlFuncReturn.GetCompoundValue().RawValues().size())
    {
        NVML_LOG_ERR("value of key [%s] is not expected", key.c_str());
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < values.size(); ++i)
    {
        auto nvmlRet = values[i].SetValueFrom(nvmlFuncReturn.GetCompoundValue().RawValues()[i]);
        if (nvmlRet != NVML_SUCCESS)
        {
            return nvmlRet;
        }
    }
    return NVML_SUCCESS;
}

/*****************************************************************************/
nvmlReturn_t InjectedNvml::DeviceSetWrapper(const std::string &funcname,
                                            const std::string &key,
                                            nvmlDevice_t nvmlDevice,
                                            std::vector<InjectionArgument> &args)
{
    std::lock_guard<std::mutex> guard(m_mutex);

    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

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
            nvmlReturn_t ret = m_devices[nvmlDevice]->ClearAttribute(key, args[1], arg);
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
nvmlReturn_t InjectedNvml::SetWrapper(const std::string &funcname,
                                      const std::string & /* key */,
                                      std::vector<InjectionArgument> & /* args */,
                                      std::vector<InjectionArgument> & /* values */)
{
    // let event related flow control success to prevent failing on starting process.
    // we will return NVML_ERROR_TIMEOUT for nvmlEventSetWait_v2.
    if (funcname == "nvmlEventSetCreate" || funcname == "nvmlEventSetFree" || funcname == "nvmlDeviceRegisterEvents")
    {
        return NVML_SUCCESS;
    }
    NVML_LOG_ERR("Calling function [%s] not injected.", funcname.c_str());
    return NVML_ERROR_NOT_SUPPORTED;
}

/*****************************************************************************/
void InjectedNvml::AddFuncCallCount(std::string_view funcName)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    auto const &mapIt = m_nvmlFuncCallCounts.find(funcName);
    if (mapIt != m_nvmlFuncCallCounts.end())
    {
        mapIt->second++;
        return;
    }
    m_nvmlFuncCallCounts.emplace(funcName, 1);
}

/*****************************************************************************/
void InjectedNvml::ResetFuncCallCounts()
{
    std::lock_guard<std::mutex> guard(m_mutex);
    m_nvmlFuncCallCounts.clear();
}

/*****************************************************************************/
funcCallMap_t InjectedNvml::GetFuncCallCounts()
{
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_nvmlFuncCallCounts;
}

/*****************************************************************************/
nvmlDevice_t InjectedNvml::GetNvmlDevice(InjectionArgument &arg, const std::string &identifier)
{
    std::lock_guard<std::mutex> guard(m_mutex);

    if (arg.GetType() == INJECTION_UINT)
    {
        auto index = arg.AsUInt();
        if (index >= m_indexToDevice.size())
        {
            return (nvmlDevice_t)0;
        }
        return m_indexToDevice[index]->GetIdentifier();
    }
    else if (identifier == "UUID")
    {
        if (!m_uuidToDevice.contains(arg.AsString()))
        {
            return (nvmlDevice_t)0;
        }
        return m_uuidToDevice[arg.AsString()]->GetIdentifier();
    }
    else if (identifier == "Serial")
    {
        if (!m_serialToDevice.contains(arg.AsString()))
        {
            return (nvmlDevice_t)0;
        }
        return m_serialToDevice[arg.AsString()]->GetIdentifier();
    }
    else if (identifier == "PciBusId")
    {
        if (!m_busIdToDevice.contains(arg.AsString()))
        {
            return (nvmlDevice_t)0;
        }
        return m_busIdToDevice[arg.AsString()]->GetIdentifier();
    }

    return (nvmlDevice_t)0;
}

InjectionArgument InjectedNvml::ObjectlessGet(const std::string &key)
{
    std::lock_guard<std::mutex> guard(m_mutex);

    return m_globalAttributes[key].GetCompoundValue().AsInjectionArgument();
}

InjectionArgument InjectedNvml::ObjectlessGetNoLock(const std::string &key)
{
    return m_globalAttributes[key].GetCompoundValue().AsInjectionArgument();
}

void InjectedNvml::ObjectlessSet(const std::string &key, const InjectionArgument &value)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    m_globalAttributes[key] = NvmlFuncReturn(NVML_SUCCESS, value);
}

void InjectedNvml::ObjectlessSetNoLock(const std::string &key, const InjectionArgument &value)
{
    m_globalAttributes[key] = NvmlFuncReturn(NVML_SUCCESS, value);
}

nvmlReturn_t InjectedNvml::GetCompoundValue(nvmlDevice_t nvmlDevice, const std::string &key, CompoundValue &cv)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return cv.SetValueFrom(m_devices[nvmlDevice]->GetAttribute(key).GetCompoundValue());
}

std::pair<nvmlReturn_t, std::string> InjectedNvml::GetString(InjectionArgument &arg, const std::string &key)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    switch (arg.GetType())
    {
        case INJECTION_DEVICE:
        {
            if (!m_devices.contains(arg.AsDevice()))
            {
                return { NVML_ERROR_INVALID_ARGUMENT, "" };
            }
            auto nvmlFuncRet = m_devices[arg.AsDevice()]->GetAttribute(key);
            return { nvmlFuncRet.GetRet(), nvmlFuncRet.GetCompoundValue().AsInjectionArgument().AsString() };
            break;
        }
        default:
            break;
    }
    return { NVML_ERROR_INVALID_ARGUMENT, "" };
}

std::pair<nvmlReturn_t, std::string> InjectedNvml::GetString(InjectionArgument &arg,
                                                             const std::string &key,
                                                             const InjectionArgument &key2)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    switch (arg.GetType())
    {
        case INJECTION_DEVICE:
        {
            if (!m_devices.contains(arg.AsDevice()))
            {
                return { NVML_ERROR_INVALID_ARGUMENT, "" };
            }
            auto nvmlFuncRet = m_devices[arg.AsDevice()]->GetAttribute(key, key2);
            return { nvmlFuncRet.GetRet(), nvmlFuncRet.GetCompoundValue().AsInjectionArgument().AsString() };
            break;
        }
        default:
            break;
    }
    return { NVML_ERROR_INVALID_ARGUMENT, "" };
}

nvmlReturn_t InjectedNvml::DeviceSet(nvmlDevice_t nvmlDevice,
                                     const std::string &key,
                                     const std::vector<InjectionArgument> &extraKeys,
                                     const NvmlFuncReturn &nvmlFunRet)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    return DeviceSetNoLock(nvmlDevice, key, extraKeys, nvmlFunRet);
}

nvmlReturn_t InjectedNvml::DeviceSetNoLock(nvmlDevice_t nvmlDevice,
                                           const std::string &key,
                                           const std::vector<InjectionArgument> &extraKeys,
                                           const NvmlFuncReturn &nvmlFunRet)
{
    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (extraKeys.empty())
    {
        m_devices[nvmlDevice]->SetAttribute(key, nvmlFunRet);
    }
    else if (extraKeys.size() == 1)
    {
        m_devices[nvmlDevice]->SetAttribute(key, extraKeys[0], nvmlFunRet);
    }
    else if (extraKeys.size() == 2)
    {
        m_devices[nvmlDevice]->SetAttribute(key, extraKeys[0], extraKeys[1], nvmlFunRet);
    }
    else if (extraKeys.size() == 3)
    {
        m_devices[nvmlDevice]->SetAttribute(key, extraKeys[0], extraKeys[1], extraKeys[2], nvmlFunRet);
    }
    else
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::DeviceInject(nvmlDevice_t nvmlDevice,
                                        const std::string &key,
                                        const std::vector<InjectionArgument> &extraKeys,
                                        const NvmlFuncReturn &nvmlFunRet)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (extraKeys.empty())
    {
        m_devices[nvmlDevice]->InjectAttribute(key, false, { nvmlFunRet });
    }
    else if (extraKeys.size() == 1)
    {
        m_devices[nvmlDevice]->InjectAttribute(key, extraKeys[0], false, { nvmlFunRet });
    }
    else if (extraKeys.size() == 2)
    {
        m_devices[nvmlDevice]->InjectAttribute(key, extraKeys[0], extraKeys[1], false, { nvmlFunRet });
    }
    else if (extraKeys.size() == 3)
    {
        m_devices[nvmlDevice]->InjectAttribute(key, extraKeys[0], extraKeys[1], extraKeys[2], false, { nvmlFunRet });
    }
    else
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::DeviceInjectForFollowingCalls(nvmlDevice_t nvmlDevice,
                                                         const std::string &key,
                                                         const std::vector<InjectionArgument> &extraKeys,
                                                         const std::list<NvmlFuncReturn> &nvmlFunRets)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (extraKeys.empty())
    {
        m_devices[nvmlDevice]->InjectAttribute(key, true, nvmlFunRets);
    }
    else if (extraKeys.size() == 1)
    {
        m_devices[nvmlDevice]->InjectAttribute(key, extraKeys[0], true, nvmlFunRets);
    }
    else if (extraKeys.size() == 2)
    {
        m_devices[nvmlDevice]->InjectAttribute(key, extraKeys[0], extraKeys[1], true, nvmlFunRets);
    }
    else if (extraKeys.size() == 3)
    {
        m_devices[nvmlDevice]->InjectAttribute(key, extraKeys[0], extraKeys[1], extraKeys[2], true, nvmlFunRets);
    }
    else
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::DeviceReset(nvmlDevice_t nvmlDevice)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    m_devices[nvmlDevice]->ResetInjectedAttribute();
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::IncrementDeviceCount()
{
    unsigned int count = ObjectlessGetNoLock(INJECTION_COUNT_KEY).AsUInt();
    count++;
    ObjectlessSetNoLock(INJECTION_COUNT_KEY, InjectionArgument(count));
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::DecrementDeviceCount()
{
    unsigned int count = ObjectlessGetNoLock(INJECTION_COUNT_KEY).AsUInt();
    if (count == 0)
    {
        return NVML_ERROR_UNKNOWN;
    }
    count--;
    ObjectlessSetNoLock(INJECTION_COUNT_KEY, InjectionArgument(count));
    return NVML_SUCCESS;
}

unsigned int InjectedNvml::GetGpuCount()
{
    std::lock_guard<std::mutex> guard(m_mutex);
    return ObjectlessGetNoLock(INJECTION_COUNT_KEY).AsUInt();
}

nvmlDevice_t InjectedNvml::GenNextNvmlDevice() const
{
    unsigned int deviceInt = m_nvmlDeviceStart + m_deviceCollection.size() + m_migDeviceCollection.size();
    nvmlDevice_t device;

    memset(&device, 0, sizeof(device));
    memcpy(&device, &deviceInt, sizeof(deviceInt));
    return device;
}

void InjectedNvml::InitializeGpuDefaults(nvmlDevice_t device, unsigned int index)
{
    char attribute[512];
    snprintf(attribute, sizeof(attribute), "GPU-1feed7b9-beef-fade-6d19-e5ce8489eb%02d", index);
    std::string paramAttr(attribute);

    m_deviceCollection.emplace_back(device);
    m_devices[device] = std::prev(m_deviceCollection.end());
    m_indexToDevice.emplace_back(std::prev(m_deviceCollection.end()));

    InjectionArgument uuid(paramAttr);
    m_deviceCollection.back().SetAttribute(INJECTION_UUID_KEY, NvmlFuncReturn(NVML_SUCCESS, uuid));
    m_uuidToDevice[paramAttr] = std::prev(m_deviceCollection.end());

    snprintf(attribute, sizeof(attribute), "03207190049%02d", index);
    paramAttr = attribute;
    InjectionArgument serial(paramAttr);
    m_deviceCollection.back().SetAttribute(INJECTION_SERIAL_KEY, NvmlFuncReturn(NVML_SUCCESS, serial));
    m_serialToDevice[paramAttr] = std::prev(m_deviceCollection.end());

    snprintf(attribute, sizeof(attribute), "00000000:%02d:00.0", 3 * index + 1);
    paramAttr = attribute;
    InjectionArgument pciBusId(paramAttr);
    m_deviceCollection.back().SetAttribute(INJECTION_PCIBUSID_KEY, NvmlFuncReturn(NVML_SUCCESS, pciBusId));
    m_busIdToDevice[paramAttr] = std::prev(m_deviceCollection.end());

    nvmlBrandType_t tBrand = NVML_BRAND_TESLA;
    InjectionArgument brand(tBrand);
    DeviceSetNoLock(device, INJECTION_BRAND_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, brand));

    std::string name("V100");
    InjectionArgument devName(name);
    DeviceSetNoLock(device, INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, devName));

    int major = 7;
    int minor = 6;
    std::vector<InjectionArgument> values;
    values.push_back(InjectionArgument(major));
    values.push_back(InjectionArgument(minor));
    CompoundValue cv(values);
    DeviceSetNoLock(device, INJECTION_CUDACOMPUTECAPABILITY_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, cv));

    DeviceSetNoLock(device, INJECTION_MIGMODE_KEY, {}, NvmlFuncReturn(NVML_ERROR_NOT_SUPPORTED));
}

nvmlReturn_t InjectedNvml::SimpleDeviceCreate(const std::string &key, InjectionArgument &value)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    unsigned int const nextDeviceIndex = m_indexToDevice.size();
    bool valid                         = false;
    nvmlDevice_t device                = GenNextNvmlDevice();

    std::string identifier;

    if (value.GetType() == INJECTION_UINT && key == INJECTION_INDEX_KEY)
    {
        auto givenIndex = value.AsUInt();
        if (givenIndex != nextDeviceIndex)
        {
            // Either this GPU already exists, or the user is trying to insert a GPU
            // out-of-order. Do not allow out-of-order device insertions. It is
            // complicated to track the missing indices/devices. Keep it simple.
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        valid = true;
        InitializeGpuDefaults(device, givenIndex);
        m_devices[device]->SetAttribute(key, NvmlFuncReturn(NVML_SUCCESS, value));
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
        }
        else if (key == INJECTION_SERIAL_KEY)
        {
            if (m_serialToDevice.count(identifier) > 0)
            {
                // This GPU already exists
                return NVML_ERROR_INVALID_ARGUMENT;
            }
        }
        else if (key == INJECTION_PCIBUSID_KEY)
        {
            // The YAML file does not seem to contain this entry.
            // This is available only if the user injects the value.
            if (m_busIdToDevice.count(identifier) > 0)
            {
                // This GPU already exists
                return NVML_ERROR_INVALID_ARGUMENT;
            }
        }

        InitializeGpuDefaults(device, nextDeviceIndex);
        m_devices[device]->SetAttribute(key, NvmlFuncReturn(NVML_SUCCESS, value));
        valid = true;
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

nvmlReturn_t InjectedNvml::SetFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t &fieldValue)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    m_devices[nvmlDevice]->SetAttribute(INJECTION_FIELDVALUES_KEY, NvmlFuncReturn(NVML_SUCCESS));
    m_devices[nvmlDevice]->SetFieldValue(fieldValue);
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::InjectFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t &fieldValue)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    m_devices[nvmlDevice]->InjectAttribute(INJECTION_FIELDVALUES_KEY, false, { NvmlFuncReturn(NVML_SUCCESS) });
    m_devices[nvmlDevice]->InjectFieldValue(fieldValue);
    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::GetFieldValues(nvmlDevice_t nvmlDevice, int valuesCount, nvmlFieldValue_t *values)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_devices.contains(nvmlDevice))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (values == nullptr)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    auto funcRet = m_devices[nvmlDevice]->GetAttribute(INJECTION_FIELDVALUES_KEY);
    if (!funcRet.IsNvmlSucces())
    {
        return funcRet.GetRet();
    }

    for (int i = 0; i < valuesCount; i++)
    {
        nvmlFieldValue_t val = m_devices[nvmlDevice]->GetFieldValue(values[i].fieldId);
        memcpy(&values[i], &val, sizeof(values[i]));
    }

    return NVML_SUCCESS;
}

void InjectedNvml::SetupDefaultEnv()
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

nvmlReturn_t InjectedNvml::RemoveGpu(std::string const &uuid)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_uuidToDevice.contains(uuid))
    {
        NVML_LOG_ERR("Provided uuid [%s] does not exist.", uuid.c_str());
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    // Remove device from all cached containers
    auto ahIter = m_uuidToDevice[uuid];
    auto device = ahIter->GetIdentifier();
    m_devices.erase(device);
    m_uuidToDevice.erase(uuid);

    std::string serial = ahIter->GetAttribute(INJECTION_SERIAL_KEY).GetCompoundValue().AsInjectionArgument().AsString();
    unsigned int index = ahIter->GetAttribute(INJECTION_INDEX_KEY).GetCompoundValue().AsInjectionArgument().AsUInt();
    nvmlPciInfo_t *pciInfo
        = ahIter->GetAttribute(INJECTION_PCIINFO_KEY).GetCompoundValue().AsInjectionArgument().AsPciInfoPtr();

    m_indexToDevice.erase(m_indexToDevice.begin() + index);
    m_serialToDevice.erase(serial);
    m_busIdToDevice.erase(pciInfo->busId);

    // Update index attribute of remaining devices based on new index
    for (unsigned int i = 0; i < m_indexToDevice.size(); i++)
    {
        auto curAhIter = m_indexToDevice[i];
        curAhIter->SetAttribute(INJECTION_INDEX_KEY, NvmlFuncReturn(NVML_SUCCESS, i));
    }

    nvmlDeviceWithIdentifiers devIds = {
        .pciBusId = pciInfo->busId,
        .uuid     = uuid,
        .serial   = std::move(serial),
        .index    = index,
        .ah       = std::move(*ahIter),
    };

    // Store removed device info for restoration later
    m_removedGpus[uuid] = std::move(devIds);
    m_deviceCollection.erase(ahIter);
    DecrementDeviceCount();

    return NVML_SUCCESS;
}

nvmlReturn_t InjectedNvml::RestoreGpu(std::string const &uuid)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_removedGpus.contains(uuid))
    {
        NVML_LOG_ERR("Provided uuid [%s] does not exist.", uuid.c_str());
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    auto &devIds                = m_removedGpus[uuid];
    auto &ah                    = devIds.ah;
    unsigned int const newIndex = m_indexToDevice.size();

    ah.SetAttribute(INJECTION_INDEX_KEY, NvmlFuncReturn(NVML_SUCCESS, newIndex));
    auto const &device = ah.GetIdentifier();

    auto const &deviceIter = m_deviceCollection.emplace(m_deviceCollection.end(), std::move(ah));

    m_devices[device]    = deviceIter;
    m_uuidToDevice[uuid] = deviceIter;
    m_indexToDevice.emplace_back(deviceIter);
    if (!devIds.serial.empty())
    {
        m_serialToDevice[devIds.serial] = deviceIter;
    }
    m_busIdToDevice[devIds.pciBusId] = deviceIter;

    m_removedGpus.erase(uuid);
    IncrementDeviceCount();
    return NVML_SUCCESS;
}
