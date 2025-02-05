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

#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nvml.h>
#include <yaml-cpp/yaml.h>

#include "AttributeHolder.h"
#include "CompoundValue.h"
#include "InjectionArgument.h"
#include "NvmlFuncReturn.h"

typedef struct
{
    std::string pciBusId;
    std::string uuid;
    std::string serial;
    unsigned int index;
    AttributeHolder<nvmlDevice_t> ah;
} nvmlDeviceWithIdentifiers;

struct StringHash : std::hash<std::string_view>
{
    using is_transparent = void;
};

using funcCallMap_t = std::unordered_map<std::string, uint32_t, StringHash, std::equal_to<>>;

#define GLOBAL            "Global"
#define DEVICE            "Device"
#define GPU_INASTANCE     "GpuInstance"
#define COMPUTE_INASTANCE "ComputeInstance"
#define VGPU_TYPE         "vGPUType"
#define VGPU_INSTANCE     "vGPUInstance"
#define MIG_DEVICE        "MigDevice"
#define FunctionReturn    "FunctionReturn"
#define ReturnValue       "ReturnValue"

class InjectedNvml
{
public:
    /*****************************************************************************/
    ~InjectedNvml();

    /*****************************************************************************/
    static InjectedNvml *Init();

    /*****************************************************************************/
    static InjectedNvml *GetInstance();

    static void Reset();

    /*****************************************************************************/
    bool LoadFromFile(const std::string &path);

    /*****************************************************************************/
    bool LoadFromString(const std::string &yamlContent);

    /*****************************************************************************/
    void SetupDefaultEnv();

    /*****************************************************************************/
    bool IsGetter(const std::string &funcname) const;

    /*****************************************************************************/
    bool IsSetter(const std::string &funcname) const;

    /*****************************************************************************/
    nvmlReturn_t GetWrapper(const std::string &funcname,
                            const std::string &key,
                            std::vector<InjectionArgument> &args,
                            std::vector<InjectionArgument> &values);

    /*****************************************************************************/
    nvmlReturn_t DeviceSetWrapper(const std::string &funcname,
                                  const std::string &key,
                                  nvmlDevice_t nvmlDevice,
                                  std::vector<InjectionArgument> &args);

    /*****************************************************************************/
    nvmlReturn_t SetWrapper(const std::string &funcname,
                            const std::string &key,
                            std::vector<InjectionArgument> &args,
                            std::vector<InjectionArgument> &values);

    /*****************************************************************************/
    nvmlDevice_t GetNvmlDevice(InjectionArgument &arg, const std::string &identifier);

    /*****************************************************************************/
    InjectionArgument ObjectlessGet(const std::string &key);

    /*****************************************************************************/
    void ObjectlessSet(const std::string &key, const InjectionArgument &value);

    /*****************************************************************************/
    nvmlReturn_t GetCompoundValue(nvmlDevice_t nvmlDevice, const std::string &key, CompoundValue &cv);

    /*****************************************************************************/
    std::pair<nvmlReturn_t, std::string> GetString(InjectionArgument &arg, const std::string &key);

    /*****************************************************************************/
    std::pair<nvmlReturn_t, std::string> GetString(InjectionArgument &arg,
                                                   const std::string &key,
                                                   const InjectionArgument &key2);

    /**
     * Change the function return to the key for a specific device. The changed value CANNOT be restored by DeviceReset.
     * This function is mainly exposed for nvml functions.
     * @param[in]  nvmlDevice           Target device
     * @param[in]  key                  Target key
     * @param[in]  extraKeys            Extra keys if need
     * @param[in]  nvmlFunRet           Expected return for this function after calling corresponding nvml function
     */
    nvmlReturn_t DeviceSet(nvmlDevice_t nvmlDevice,
                           const std::string &key,
                           const std::vector<InjectionArgument> &extraKeys,
                           const NvmlFuncReturn &nvmlFunRet);

    /**
     * Inject the function return to the key for a specific device. The injected value can be restored by DeviceReset.
     * This function is mainly exposed for test cases.
     * @param[in]  nvmlDevice           Target device
     * @param[in]  key                  Target key
     * @param[in]  extraKeys            Extra keys if need
     * @param[in]  nvmlFunRet           Expected return for∂ corresponding nvml function
     */
    nvmlReturn_t DeviceInject(nvmlDevice_t nvmlDevice,
                              const std::string &key,
                              const std::vector<InjectionArgument> &extraKeys,
                              const NvmlFuncReturn &nvmlFunRet);

    /**
     * Indicate the next nvmlFunRets.size() times return of a nvml function for a specific device.
     * After nvmlFunRets.size() times of calling nvml function, the device returns back to loaded state.
     * The behavior can be restored by DeviceReset.
     * This function is mainly exposed for test cases.
     * @param[in]  nvmlDevice           Target device
     * @param[in]  key                  Target key
     * @param[in]  extraKeys            Extra keys if need
     * @param[in]  nvmlFunRets          Expected returns corresponding nvml function for next several times
     */
    nvmlReturn_t DeviceInjectForFollowingCalls(nvmlDevice_t nvmlDevice,
                                               const std::string &key,
                                               const std::vector<InjectionArgument> &extraKeys,
                                               const std::list<NvmlFuncReturn> &nvmlFunRets);

    /**
     * Reset the device to loaded state. Which remove all the injected value by DeviceInject.
     * @param[in]  nvmlDevice           Target device
     */
    nvmlReturn_t DeviceReset(nvmlDevice_t nvmlDevice);

    /*****************************************************************************/
    nvmlReturn_t SimpleDeviceCreate(const std::string &key, InjectionArgument &value);

    /*****************************************************************************/
    nvmlReturn_t SetFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t &fieldValue);

    /**
     * Inject the field value for a specific device. The injected value can be restored by DeviceReset.
     * This function is mainly exposed for test cases.
     * @param[in]  nvmlDevice           Target device
     * @param[in]  fieldValue                  Target fieldValue
     */
    nvmlReturn_t InjectFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t &fieldValue);

    /*****************************************************************************/
    nvmlReturn_t GetFieldValues(nvmlDevice_t nvmlDevice, int valuesCount, nvmlFieldValue_t *values);

    /*****************************************************************************/
    unsigned int GetGpuCount();

    void AddFuncCallCount(std::string_view funcName);
    void ResetFuncCallCounts();
    funcCallMap_t GetFuncCallCounts();

    nvmlReturn_t RemoveGpu(std::string const &uuid);
    nvmlReturn_t RestoreGpu(std::string const &uuid);


private:
    static InjectedNvml *m_injectedNvmlInstance;

    /*****************************************************************************/
    InjectedNvml();

    std::mutex m_mutex;
    static const unsigned int m_nvmlDeviceStart = 0xA0A0;

    std::map<nvmlVgpuInstance_t, AttributeHolder<nvmlVgpuInstance_t>> m_vgpuInstances;
    std::map<nvmlVgpuTypeId_t, AttributeHolder<nvmlVgpuTypeId_t>> m_vgpuTypeIds;
    std::map<nvmlDevice_t, std::list<AttributeHolder<nvmlDevice_t>>::iterator> m_devices;
    std::map<nvmlGpuInstance_t, AttributeHolder<nvmlGpuInstance_t>> m_gpuInstances;
    std::map<nvmlComputeInstance_t, AttributeHolder<nvmlComputeInstance_t>> m_computeInstances;

    std::unordered_map<std::string, std::list<AttributeHolder<nvmlDevice_t>>::iterator> m_busIdToDevice;
    std::unordered_map<std::string, std::list<AttributeHolder<nvmlDevice_t>>::iterator> m_uuidToDevice;
    std::unordered_map<std::string, std::list<AttributeHolder<nvmlDevice_t>>::iterator> m_serialToDevice;
    std::vector<std::list<AttributeHolder<nvmlDevice_t>>::iterator> m_indexToDevice;

    // actual place to hold the device attributes.
    // other mapping (budId, uuid, serial, index) only maps the identifier to the iterator in this list
    std::list<AttributeHolder<nvmlDevice_t>> m_deviceCollection;

    // actual place to hold the MIG device attributes.
    // other mapping (uuid) only maps the identifier to the iterator in this list
    std::list<AttributeHolder<nvmlDevice_t>> m_migDeviceCollection;

    // This list collects all "fake" GPU instances. Each element is {DeviceUUID}_{idx} which represents as a GPU
    // instance of {deviceUUID}. nvml.h dose not define nvmlGpuInstance_t, use the pointer of element in this list to
    // represent it
    std::list<std::string> m_gpuInstanceCollection;

    // This list collects all "fake" compute instances. Each element is {fake_gpu_instance}_{idx} which represents as a
    // compute instance of GPU istance {fake_gpu_instance}. nvml.h dose not define nvmlComputeInstance_t, use the
    // pointer of element in this list to represent it
    std::list<std::string> m_computeInstanceCollection;

    // This map contains the UUIDs of the GPUs removed with the RemoveGPU API, which will be used to restore the GPUs
    // with the RestoreGPU API using the saved device values.
    std::unordered_map<std::string, nvmlDeviceWithIdentifiers> m_removedGpus;

    std::unordered_map<std::string, NvmlFuncReturn> m_globalAttributes;

    funcCallMap_t m_nvmlFuncCallCounts;

    nvmlDevice_t GenNextNvmlDevice() const;

    /*****************************************************************************/
    bool LoadFromYamlNode(const YAML::Node &root);

    // The following functions used for parsing YAML file and preparing behavior of NVML
    bool ParseGlobal(const YAML::Node &global);
    bool ParseOneDevice(const YAML::Node &device, AttributeHolder<nvmlDevice_t> &ah);
    bool ParseDevices(const YAML::Node &devices);
    bool ParseOneGpuInstance(const YAML::Node &gpuInstance, AttributeHolder<nvmlGpuInstance_t> &ah);
    bool ParseGpuInstances(const YAML::Node &gpuInstances);
    bool ParseOneComputeInstance(const YAML::Node &computeInstance, AttributeHolder<nvmlComputeInstance_t> &ah);
    bool ParseComputeInstances(const YAML::Node &computeInstances);
    bool ParseVgpuTypes(const YAML::Node &vgpuTypes);
    bool ParseOneVgpuType(const YAML::Node &vgpuType, AttributeHolder<nvmlVgpuTypeId_t> &ah);
    bool ParseVgpuInstances(const YAML::Node &vgpuInstances);
    bool ParseOneVgpuInstance(const YAML::Node &vgpuInstance, AttributeHolder<nvmlVgpuInstance_t> &ah);
    bool ParseMigDevices(const YAML::Node &migDevices);

    // The following parsers will touch the class member
    bool DeviceOrderParser(const YAML::Node &node);
    bool ActiveVgpusParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah);
    bool GpuInstancesParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah);
    bool TopologyCommonAncestorParser(const std::string &key,
                                      const YAML::Node &node,
                                      AttributeHolder<nvmlDevice_t> &ah);
    bool TopologyNearestGpuParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah);
    bool MigDeviceUUIDParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah);
    bool ComputeInstancesParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlGpuInstance_t> &ah);
    bool GpuInstanceInfoParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlGpuInstance_t> &ah);
    bool OnSameBoardParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah);
    bool FieldValuesParser(const std::string &key, const YAML::Node &node, AttributeHolder<nvmlDevice_t> &ah);
    bool ComputeInstanceInfoParser(const std::string &key,
                                   const YAML::Node &node,
                                   AttributeHolder<nvmlComputeInstance_t> &ah);

    std::optional<nvmlReturn_t> GetWrapperSpecialCase(const std::string &funcname,
                                                      const std::string &key,
                                                      std::vector<InjectionArgument> &args,
                                                      std::vector<InjectionArgument> &values);
    NvmlFuncReturn DeviceGetWrapper(const std::string &key, std::vector<InjectionArgument> &args);
    NvmlFuncReturn GpuInstanceGetWrapper(const std::string &key, std::vector<InjectionArgument> &args);
    NvmlFuncReturn ComputeInstanceGetWrapper(const std::string &key, std::vector<InjectionArgument> &args);
    NvmlFuncReturn VgpuTypeIdGetWrapper(const std::string &key, std::vector<InjectionArgument> &args);
    NvmlFuncReturn VgpuInstanceGetWrapper(const std::string &key, std::vector<InjectionArgument> &args);

    // The following functions used for modified the NVML behavior
    nvmlReturn_t IncrementDeviceCount();
    nvmlReturn_t DecrementDeviceCount();
    void InitializeGpuDefaults(nvmlDevice_t device, unsigned int index);


    nvmlReturn_t DeviceSetNoLock(nvmlDevice_t nvmlDevice,
                                 const std::string &key,
                                 const std::vector<InjectionArgument> &extraKeys,
                                 const NvmlFuncReturn &nvmlFunRet);
    InjectionArgument ObjectlessGetNoLock(const std::string &key);
    void ObjectlessSetNoLock(const std::string &key, const InjectionArgument &value);
};
