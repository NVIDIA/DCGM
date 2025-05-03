# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# our build image does not have pynvml, skip this file

from dcgm_nvml import *
from typing import List
import yaml
import re
import time
import uuid

# constant defines
NVML_SUCCESS = 0

# constant defines
MAX_NUM_INSTANCE = 8

# common elements
FUNC_RETURN = "FunctionReturn"
RETURN_VAL = "ReturnValue"

# key defines
GLOBAL_KEY = "Global"
COUNT = "Count"
DEVICE_ORDER = "DeviceOrder"
MIG_DEVICE_ORDER = "MigDeviceOrder"
DEVICE = "Device"
VGPU_TYPE = "vGPUType"
VGPU_INSTANCE = "vGPUInstance"
EXCLUDED_DEVICE = "ExcludedDevice"
GPU_INSTANCE = "GpuInstance"
COMPUTE_INSTANCE = "ComputeInstance"
MIG_DEVICE = "MigDevice"
GPM = "GPM"

def run_nvml_func(func, *args, **kwargs):
    try:
        return NVML_SUCCESS, func(*args, **kwargs)
    except NVMLError as error:
        return error.value, 0

def run_nvml_func_str(func_str, *args, **kwargs):
    try:
        return NVML_SUCCESS, eval(f"{func_str}(*args, **kwargs)")
    except NVMLError as error:
        return error.value, 0

def run_nvml_no_output_func_str(func_str, *args, **kwargs):
    try:
        eval(f"{func_str}(*args, **kwargs)")
        return NVML_SUCCESS
    except NVMLError as error:
        return error.value

def run_nvml_two_attrs_func_str(func_str, *args, **kwargs):
    try:
        ret_val1, ret_val2 = eval(f"{func_str}(*args, **kwargs)")
        return NVML_SUCCESS, ret_val1, ret_val2
    except NVMLError as error:
        return error.value, 0, 0

def run_nvml_three_attrs_func_str(func_str, *args, **kwargs):
    try:
        ret_val1, ret_val2, ret_val3 = eval(f"{func_str}(*args, **kwargs)")
        return NVML_SUCCESS, ret_val1, ret_val2, ret_val3
    except NVMLError as error:
        return error.value, 0, 0, 0

def run_nvml_four_attrs_func_str(func_str, *args, **kwargs):
    try:
        ret_val1, ret_val2, ret_val3, ret_val4 = eval(f"{func_str}(*args, **kwargs)")
        return NVML_SUCCESS, ret_val1, ret_val2, ret_val3, ret_val4
    except NVMLError as error:
        return error.value, 0, 0, 0, 0

def nvmlDeviceGetMemoryInfo_v2(handle):
    return nvmlDeviceGetMemoryInfo(handle, version=2)

def nvml_func_key_suffix_extract(func_str):
    key_prefixes = [
        "nvmlDeviceGetHandleBy",
        "nvmlDeviceGet",
        "nvmlSystemGet",
        "nvmlDeviceSet",
        "nvmlUnitGet",
        "nvmlUnitSet",
        "nvmlVgpuTypeGet",
        "nvmlVgpuInstanceGet",
        "nvmlVgpuInstanceSet",
        "nvmlGet",
        "nvmlSet",
        "nvmlGpuInstanceGet",
        "nvmlComputeInstanceGet",
        "nvmlDeviceClear",
        "nvmlDeviceFreeze",
        "nvmlDeviceModify",
        "nvmlDeviceQuery",
        "nvmlDeviceCreate",
        "nvmlDeviceReset",
        "nvmlDeviceIs",
        "nvmlDevice",
        "nvmlGpm",
    ]
    for prefix in key_prefixes:
        if func_str[:len(prefix)] != prefix:
            continue
        if prefix != "nvmlGpm":
            suffix = func_str[len(prefix):]
            return suffix
        else:
            if func_str[-3:] == "Get":
                return func_str[7:-3]
    raise RuntimeError(f"Unknown function: {func_str}")

def basic_type_value_parser(value):
    if type(value) is str:
        quoted_str = f'{value}'
        return quoted_str
    return value

def arr_0_value_parser(value):
    return value[0]

def basic_c_type_value_parser(value):
    return value.value

def memory_info_parser(value):
    return {
        "total": value.total,
        "free": value.free,
        "used": value.used,
    }

def memory_info_v2_parser(value):
    return {
        "total": value.total,
        "free": value.free,
        "used": value.used,
        "reserved": value.reserved,
        "version": value.version,
    }

def pci_info_parser(value):
    return {
        "busIdLegacy": value.busIdLegacy,
        "domain": value.domain,
        "bus": value.bus,
        "device": value.device,
        "pciDeviceId": value.pciDeviceId,
        "pciSubSystemId": value.pciSubSystemId,
        "busId": value.busId,
    }

def utilization_parser(value):
    return {
        "gpu": value.gpu,
        "memory": value.memory,
    }

def fbc_stats_parser(value):
    return {
        "sessionsCount": value.sessionsCount,
        "averageFPS": value.averageFPS,
        "averageLatency": value.averageLatency,
    }

def cuda_capability_parser(major, minor):
    return {
        "major": major,
        "minor": minor,
    }

def current_pending_parser(current, pending):
    return {
        "current": current,
        "pending": pending,
    }

def current_pending_mode_parser(current, pending):
    return {
        "currentMode": current,
        "pendingMode": pending,
    }

def affinity_parser(value):
    ret = []
    for i in range(8):
        ret.append(value[i])
    return ret

def bar1_memory_info_parser(value):
    return {
        "bar1Total": value.bar1Total,
        "bar1Free": value.bar1Free,
        "bar1Used": value.bar1Used,
    }

def violation_status_parser(value):
    return {
        "referenceTime": value.referenceTime,
        "violationTime": value.violationTime,
    }

def codec_utilization_parser(utilization, sampling_period_us):
    return {
        "utilization": utilization,
        "samplingPeriodUs": sampling_period_us,
    }

def clock_value_parser(min_clock, max_clock):
    return {
        "minGpuClockMHz": min_clock,
        "maxGpuClockMHz": max_clock,
    }

def auto_boost_enable_parser(is_enabled, default_is_enabled):
    return {
        "isEnabled": is_enabled,
        "defaultIsEnabled": default_is_enabled,
    }

def power_management_limit_constraints_parser(min_limit, max_limit):
    return {
        "minLimit": min_limit,
        "maxLimit": max_limit,
    }

def accounting_stat_parser(value):
    return {
        "gpuUtilization": value.gpuUtilization,
        "memoryUtilization": value.memoryUtilization,
        "maxMemoryUsage": value.maxMemoryUsage,
        "time": value.time,
        "startTime": value.startTime,
        "isRunning": value.isRunning,
        # skip reserved as it will failed in yaml dump
        # "reserved": value.reserved,
    }

def sample_parser(values):
    _, samples = values
    ret = []
    for value in samples:
        ret.append({"timestamp": value.timeStamp, "sampleValue": value.sampleValue.uiVal})
    return ret

def topology_value_parser(devices):
    ret = []
    for device in devices:
        device_idx = nvmlDeviceGetIndex(device)
        ret.append(device_idx)
    return ret

def device_id_parser(device_id, subsystem_id):
    return {
        "deviceID": device_id,
        "subsystemID": subsystem_id,
    }

def resolution_parser(xdim, ydim):
    return {
        "xdim": xdim,
        "ydim": ydim,
    }

def vm_id_parser(vm_id, vm_id_type):
    return {
        "vmId": vm_id.decode("utf-8"),
        "vmIdType": vm_id_type,
    }

def vgpu_expire_value_parser(value):
    return {
        "year": value.year,
        "month": value.month,
        "day": value.day,
        "hour": value.hour,
        "min": value.min,
        "sec": value.sec,
        "status": value.status,
    }

def vgpu_license_parser(value):
    return {
        "isLicensed": value.isLicensed,
        "licenseExpiry": vgpu_expire_value_parser(value.licenseExpiry),
        "currentState": value.currentState,
    }

def vgpu_utilization_parser(values):
    ret = []
    for value in values:
        ret.append({
            "vgpuInstance": value.vgpuInstance,
            "timeStamp": value.timeStamp,
            "smUtil": value.smUtil.uiVal,
            "memUtil": value.memUtil.uiVal,
            "encUtil": value.encUtil.uiVal,
            "decUtil": value.decUtil.uiVal,
        })
    return ret

def vgpu_process_utilization_parser(values):
    ret = []
    for value in values:
        ret.append({
            "vgpuInstance": value.vgpuInstance,
            "pid": value.pid,
            "processName": value.processName,
            "timeStamp": value.timeStamp,
            "smUtil": value.smUtil,
            "memUtil": value.memUtil,
            "encUtil": value.encUtil,
            "decUtil": value.decUtil,
        })
    return ret

def process_utilization_parser(values):
    ret = []
    for value in values:
        ret.append({
            "pid": value.pid,
            "timeStamp": value.timeStamp,
            "smUtil": value.smUtil,
            "memUtil": value.memUtil,
            "encUtil": value.encUtil,
            "decUtil": value.decUtil,
        })
    return ret

def vgpu_metadata_parser(value):
    return {
        "version": value.version,
        "revision": value.revision,
        "guestInfoState": value.guestInfoState,
        "guestDriverVersion": value.guestDriverVersion,
        "hostDriverVersion": value.hostDriverVersion,
        # Skip reserve, otherwise yaml dump may fail
        # "reserved": value.reserved,
        "vgpuVirtualizationCaps": value.vgpuVirtualizationCaps,
        "guestVgpuVersion": value.guestVgpuVersion,
        "opaqueDataSize": value.opaqueDataSize,
        # Skip value.opaqueData, otherwise yaml dump may fail
        # "opaqueData": value.opaqueData,
    }

def gsp_firmware_mode_parser(is_enabled, default_mode):
    return {
        "isEnabled": is_enabled,
        "defaultMode": default_mode,
    }

def vgpu_host_supported_range_paser(value):
    return {
        "minVersion": value.minVersion,
        "maxVersion": value.maxVersion,
    }

def vgpu_pgpu_metadata_parser(value):
    return {
        "version": value.version,
        "revision": value.revision,
        "hostDriverVersion": value.hostDriverVersion,
        "pgpuVirtualizationCaps": value.pgpuVirtualizationCaps,
        # Skip reserve, otherwise yaml dump may fail
        # "reserved": value.reserved,
        "hostSupportedVgpuRange": vgpu_host_supported_range_paser(value.hostSupportedVgpuRange),
        "opaqueDataSize": value.opaqueDataSize,
        # Skip value.opaqueData, otherwise yaml dump may fail
        # "opaqueData": value.opaqueData,
    }

def license_expiry_parser(value):
    return {
        "year": value.year,
        "month": value.month,
        "day": value.day,
        "hour": value.hour,
        "min": value.min,
        "sec": value.sec,
        "status": value.status,
    }

def grid_license_parser(value):
    ret = {
        "isGridLicenseSupported": value.isGridLicenseSupported,
        "licensableFeaturesCount": value.licensableFeaturesCount,
    }
    ret["gridLicensableFeatures"] = []
    for i in range(value.licensableFeaturesCount):
        feature = {
            "featureCode": value.gridLicensableFeatures[i].featureCode,
            "featureState": value.gridLicensableFeatures[i].featureState,
            "licenseInfo": value.gridLicensableFeatures[i].licenseInfo,
            "productName": value.gridLicensableFeatures[i].productName,
            "featureEnabled": value.gridLicensableFeatures[i].featureEnabled,
            "licenseExpiry": license_expiry_parser(value.gridLicensableFeatures[i].licenseExpiry),
        }
        ret["gridLicensableFeatures"].append(feature)
    return ret

def encoder_stats_parser(session_count, average_fps, average_latency):
    return {
        "sessionCount": session_count,
        "averageFps": average_fps,
        "averageLatency": average_latency
    }

def encoder_sessions_parser(sessions):
    ret = []
    for session in sessions:
        ret.append({
            "sessionId": session.sessionId,
            "pid": session.pid,
            "vgpuInstance": session.vgpuInstance,
            "codecType": session.codecType,
            "hResolution": session.hResolution,
            "vResolution": session.vResolution,
            "averageFps": session.averageFps,
            "encodeLatency": session.encodeLatency,
        })
    return ret

def fbc_sessions_parser(sessions):
    ret = []
    for session in sessions:
        ret.append({
            "sessionId": session.sessionId,
            "pid": session.pid,
            "vgpuInstance": session.vgpuInstance,
            "displayOrdinal": session.displayOrdinal,
            "sessionType": session.sessionType,
            "sessionFlags": session.sessionFlags,
            "hMaxResolution": session.hMaxResolution,
            "vMaxResolution": session.vMaxResolution,
            "hResolution": session.hResolution,
            "vResolution": session.vResolution,
            "averageFPS": session.averageFPS,
            "averageLatency": session.averageLatency,
        })
    return ret

def field_value_parser(field_value):
    ret = {
        "fieldId": field_value.fieldId,
        "scopeId": field_value.scopeId,
        "timestamp": field_value.timestamp,
        "latencyUsec": field_value.latencyUsec,
        "valueType": field_value.valueType,
        "nvmlReturn": field_value.nvmlReturn,
    }
    if field_value.valueType == NVML_VALUE_TYPE_DOUBLE:
        ret["value"] = field_value.value.dVal
    elif field_value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT:
        ret["value"] = field_value.value.uiVal
    elif field_value.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG:
        ret["value"] = field_value.value.ulVal
    elif field_value.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        ret["value"] = field_value.value.ullVal
    elif field_value.valueType == NVML_VALUE_TYPE_SIGNED_LONG_LONG:
        ret["value"] = field_value.value.sllVal
    else:
        print("strange filedValue returned, give 0 as default")
        ret["value"] = 0
    return ret

def accounting_pid_parser(count, pids):
    ret = []
    for i in range(count.value):
        ret.append(pids[i].value)
    return ret

def instance_profile_parser(value):
    return {
        "version": value.version,
        "id": value.id,
        "isP2pSupported": value.isP2pSupported,
        "sliceCount": value.sliceCount,
        "instanceCount": value.instanceCount,
        "multiprocessorCount": value.multiprocessorCount,
        "copyEngineCount": value.copyEngineCount,
        "decoderCount": value.decoderCount,
        "encoderCount": value.encoderCount,
        "jpegCount": value.jpegCount,
        "ofaCount": value.ofaCount,
        "memorySizeMB": value.memorySizeMB,
        "name": value.name,
    }

def placement_parser(value):
    return {
        "start": value.start,
        "size": value.size,
    }

def instance_info_parser(parent_device_uuid, value):
    return {
        "device": parent_device_uuid,
        "id": value.id,
        "profileId": value.profileId,
        "placement": placement_parser(value.placement),
    }

def ci_profile_info_parser(ci_info):
    return {
        "version": ci_info.version,
        "id": ci_info.id,
        "sliceCount": ci_info.sliceCount,
        "instanceCount": ci_info.instanceCount,
        "multiprocessorCount": ci_info.multiprocessorCount,
        "sharedCopyEngineCount": ci_info.sharedCopyEngineCount,
        "sharedDecoderCount": ci_info.sharedDecoderCount,
        "sharedEncoderCount": ci_info.sharedEncoderCount,
        "sharedJpegCount": ci_info.sharedJpegCount,
        "sharedOfaCount": ci_info.sharedOfaCount,
        "name": ci_info.name,
    }

def ci_info_parser(parent_device_uuid, fake_gpu_instance, ci_info):
    return {
        "device": parent_device_uuid,
        "gpuInstance": fake_gpu_instance,
        "id": ci_info.id,
        "profileId": ci_info.profileId,
        "placement": placement_parser(ci_info.placement),
    }

def attributes_parser(value):
    return {
        "multiprocessorCount": value.multiprocessorCount,
        "sharedCopyEngineCount": value.sharedCopyEngineCount,
        "sharedDecoderCount": value.sharedDecoderCount,
        "sharedEncoderCount": value.sharedEncoderCount,
        "sharedJpegCount": value.sharedJpegCount,
        "sharedOfaCount": value.sharedOfaCount,
        "gpuInstanceSliceCount": value.gpuInstanceSliceCount,
        "computeInstanceSliceCount": value.computeInstanceSliceCount,
        "memorySizeMB": value.memorySizeMB,
    }

def remapped_row_parser(corr_rows, unc_rows, is_pending, failure_occured):
    return {
        "corrRows": corr_rows,
        "uncRows": unc_rows,
        "isPending": is_pending,
        "failureOccurred": failure_occured,
    }

def row_remapper_histogram_parser(value):
    return {
        "max": value.max,
        "high": value.high,
        "partial": value.partial,
        "low": value.low,
        "none": value.none,
    }

def conf_compute_state_parser(conf_compute_state):
    return {
        "environment": conf_compute_state.environment,
        "ccFeature": conf_compute_state.ccFeature,
        "devToolsMode": conf_compute_state.devToolsMode,
    }

def runing_process_parser(processes):
    ret = []
    for process in processes:
        ret.append({
            "computeInstanceId": process.computeInstanceId,
            "gpuInstanceId": process.gpuInstanceId,
            "pid": process.pid,
            "usedGpuMemory": process.usedGpuMemory,
        })
    return ret

def ecc_error_count_parser(value):
    return {
        "l1Cache": value.l1Cache,
        "l2Cache": value.l2Cache,
        "deviceMemory": value.deviceMemory,
        "registerFile": value.registerFile,
    }

def bridge_chip_info_parser(value):
    ret = {
        "bridgeCount": value.bridgeCount,
    }
    ret["bridgeChipInfo"] = []
    for i in range(value.bridgeCount):
        ret.append({
            "type": value.bridgeChipInfo[i].type,
            "fwVersion": value.bridgeChipInfo[i].fwVersion,
        })
    return ret

class NVMLSimpleFunc(object):
    def __init__(self, func_str, value_parser):
        self._func_str = func_str
        self._value_parser = value_parser

class NVMLSimpleVersionFunc(object):
    def __init__(self, func_str, version, value_parser):
        self._func_str = func_str
        self._version = version
        self._value_parser = value_parser

class NVMLSimpleArrayOutputFunc(object):
    def __init__(self, func_str, arr_size, value_parser):
        self._func_str = func_str
        self._arr_size = arr_size
        self._value_parser = value_parser

class NVMLExtraKeyArrayOutputFunc(object):
    def __init__(self, func_str, arr_size, possible_inputs: List[int], value_parser):
        self._func_str = func_str
        self._arr_size = arr_size
        self._possible_inputs = possible_inputs
        self._value_parser = value_parser

class NVMLExtraKeyFunc(object):
    def __init__(self, func_str, possible_inputs: List[int], value_parser):
        self._func_str = func_str
        self._possible_inputs = possible_inputs
        self._value_parser = value_parser

class NVMLTwoKeysFunc(object):
    def __init__(self, func_str, key1_possible_inputs: List[int], key2_possible_inputs: List[int], value_parser):
        self._func_str = func_str
        self._key1_possible_inputs = key1_possible_inputs
        self._key2_possible_inputs = key2_possible_inputs
        self._value_parser = value_parser

class NVMLThreeKeysFunc(object):
    def __init__(self, func_str, key1_possible_inputs: List[int], key2_possible_inputs: List[int], key3_possible_inputs: List[int], value_parser):
        self._func_str = func_str
        self._key1_possible_inputs = key1_possible_inputs
        self._key2_possible_inputs = key2_possible_inputs
        self._key3_possible_inputs = key3_possible_inputs
        self._value_parser = value_parser

def fabric_info_parser(value):
    dev_uuid = uuid.UUID(int=int.from_bytes(value.clusterUuid, 'big'))
    return {
        "clusterUuid": str(dev_uuid),
        "status": value.status,
        "cliqueId": value.cliqueId,
        "state": value.state,
    }

def nvmlDeviceGetGpuFabricInfo(device):
    import dcgm_nvml as pynvml
    import nvml_injection_structs
    c_fabricInfo = nvml_injection_structs.c_nvmlGpuFabricInfo_t_dcgm_ver()
    pynvml.nvmlDeviceGetGpuFabricInfo(device, byref(c_fabricInfo))
    return c_fabricInfo

def fabric_infov_parser(value):
    dev_uuid = uuid.UUID(int=int.from_bytes(value.clusterUuid, 'big'))
    return {
        "version": value.version,
        "clusterUuid": str(dev_uuid),
        "status": value.status,
        "cliqueId": value.cliqueId,
        "state": value.state,
        "healthMask": value.healthMask,
    }

def nvmlDeviceGetGpuFabricInfoV(device):
    import dcgm_nvml as pynvml
    import nvml_injection_structs
    c_fabricInfo = nvml_injection_structs.c_nvmlGpuFabricInfoV_t_dcgm_ver()
    c_fabricInfo.version = pynvml.nvmlGpuFabricInfo_v2
    pynvml.nvmlDeviceGetGpuFabricInfoV(device, byref(c_fabricInfo))
    return c_fabricInfo

def sram_ecc_error_status_parser(value):
    return {
        "version": value.version,
        "aggregateUncParity": value.aggregateUncParity,
        "aggregateUncSecDed": value.aggregateUncSecDed,
        "aggregateCor": value.aggregateCor,
        "volatileUncParity": value.volatileUncParity,
        "volatileUncSecDed": value.volatileUncSecDed,
        "volatileCor": value.volatileCor,
        "aggregateUncBucketL2": value.aggregateUncBucketL2,
        "aggregateUncBucketSm": value.aggregateUncBucketSm,
        "aggregateUncBucketPcie": value.aggregateUncBucketPcie,
        "aggregateUncBucketMcu": value.aggregateUncBucketMcu,
        "aggregateUncBucketOther": value.aggregateUncBucketOther,
        "bThresholdExceeded": value.bThresholdExceeded,
    }

def nvmlDeviceGetSramEccErrorStatus(device):
    import dcgm_nvml as pynvml
    c_sramErrStatus = pynvml.c_nvmlEccSramErrorStatus_v1_t()
    c_sramErrStatus.version = pynvml.nvmlEccSramErrorStatus_v1
    pynvml.nvmlDeviceGetSramEccErrorStatus(device, byref(c_sramErrStatus))
    return c_sramErrStatus


class NVMLApiRecorder(object):
    _attrs = {}
    # mapping all fake nvmlGpuInstance_t to (parent device uuid, real nvmlGpuInstance_t)
    _gpu_instances_mapping = {}
    # mapping all fake nvmlComputeInstance_t to (parent device uuid, fake nvmlGpuInstance_t, real nvmlComputeInstance_t)
    _compute_instance_mapping = {}
    # an array of (mig device uuid, mig device)
    _mig_devices_collector = []
    _nvml_global_attr_funcs = [
        NVMLSimpleFunc("nvmlDeviceGetCount", basic_type_value_parser),
        NVMLSimpleFunc("nvmlSystemGetDriverVersion", basic_type_value_parser),
        NVMLSimpleFunc("nvmlSystemGetNVMLVersion", basic_type_value_parser),
        NVMLSimpleFunc("nvmlSystemGetCudaDriverVersion", basic_type_value_parser),
        NVMLSimpleFunc("nvmlSystemGetCudaDriverVersion_v2", basic_type_value_parser),
        NVMLSimpleFunc("nvmlGetExcludedDeviceCount", basic_type_value_parser),
        NVMLSimpleFunc("nvmlSystemGetConfComputeState", conf_compute_state_parser),
    ]

    # which has one device input and no output
    _nvml_device_no_output_funcs = [
        "nvmlDeviceValidateInforom"
    ]

    # which has one device input and one output
    _nvml_device_attr_funcs = [
        NVMLSimpleFunc("nvmlDeviceGetComputeMode", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetDisplayMode", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetDefaultEccMode", arr_0_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetBoardId", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetMultiGpuBoard", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetBrand", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPersistenceMode", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPowerState", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPerformanceState", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPowerUsage", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetTotalEnergyConsumption", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPowerManagementMode", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPowerManagementLimit", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetFanSpeed", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetNumFans", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetMaxPcieLinkGeneration", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetMaxPcieLinkWidth", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetCurrPcieLinkGeneration", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetCurrPcieLinkWidth", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetBridgeChipInfo", bridge_chip_info_parser),
        NVMLSimpleFunc("nvmlDeviceGetSupportedEventTypes", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetInforomConfigurationChecksum", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetDisplayActive", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPowerManagementDefaultLimit", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetCurrentClocksThrottleReasons", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetSupportedClocksThrottleReasons", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetIndex", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetAccountingMode", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetAccountingBufferSize", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetRetiredPagesPendingStatus", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetMinorNumber", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetEnforcedPowerLimit", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPcieReplayCounter", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetHostVgpuMode", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetGpuInstanceId", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetComputeInstanceId", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetMaxMigDeviceCount", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetAttributes", attributes_parser),
        NVMLSimpleFunc("nvmlDeviceGetRowRemapperHistogram", row_remapper_histogram_parser),
        NVMLSimpleFunc("nvmlDeviceGetBusType", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetIrqNum", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetNumGpuCores", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPowerSource", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetMemoryBusWidth", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPcieLinkMaxSpeed", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetAdaptiveClockInfoStatus", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPcieSpeed", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetGpcClkVfOffset", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetMemClkVfOffset", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetArchitecture", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceIsMigDeviceHandle", basic_c_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetPciInfo", pci_info_parser),
        NVMLSimpleFunc("nvmlDeviceGetUtilizationRates", utilization_parser),
        NVMLSimpleFunc("nvmlDeviceGetFBCStats", fbc_stats_parser),
        NVMLSimpleFunc("nvmlDeviceGetInforomImageVersion", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetName", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetSerial", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetBoardPartNumber", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetUUID", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetBAR1MemoryInfo", bar1_memory_info_parser),
        NVMLSimpleFunc("nvmlDeviceGetVbiosVersion", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetComputeRunningProcesses", runing_process_parser),
        NVMLSimpleFunc("nvmlDeviceGetGraphicsRunningProcesses", runing_process_parser),
        NVMLSimpleFunc("nvmlDeviceGetMPSComputeRunningProcesses", runing_process_parser),
        NVMLSimpleFunc("nvmlDeviceGetSupportedMemoryClocks", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetAccountingPids", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetVirtualizationMode", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetSupportedVgpus", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetCreatableVgpus", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetActiveVgpus", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetVgpuMetadata", vgpu_pgpu_metadata_parser),
        NVMLSimpleFunc("nvmlDeviceGetGridLicensableFeatures", grid_license_parser),
        NVMLSimpleFunc("nvmlDeviceGetEncoderSessions", encoder_sessions_parser),
        NVMLSimpleFunc("nvmlDeviceGetFBCSessions", fbc_sessions_parser),
        NVMLSimpleFunc("nvmlDeviceGetGpuFabricInfo", fabric_info_parser),
        NVMLSimpleFunc("nvmlDeviceGetGpuFabricInfoV", fabric_infov_parser),
        NVMLSimpleFunc("nvmlDeviceGetSramEccErrorStatus", sram_ecc_error_status_parser),        
    ]

    # which has one device input, version and produces one output
    _nvml_device_with_version_attr_funcs = [
    ]

    # which has one device input and two outputs
    _nvml_device_two_attrs_funcs = [
        NVMLSimpleFunc("nvmlDeviceGetCudaComputeCapability", cuda_capability_parser),
        NVMLSimpleFunc("nvmlDeviceGetDriverModel", current_pending_parser),
        NVMLSimpleFunc("nvmlDeviceGetEccMode", current_pending_parser),
        NVMLSimpleFunc("nvmlDeviceGetEncoderUtilization", codec_utilization_parser),
        NVMLSimpleFunc("nvmlDeviceGetDecoderUtilization", codec_utilization_parser),
        NVMLSimpleFunc("nvmlDeviceGetGpuOperationMode", current_pending_parser),
        NVMLSimpleFunc("nvmlDeviceGetAutoBoostedClocksEnabled", auto_boost_enable_parser),
        NVMLSimpleFunc("nvmlDeviceGetPowerManagementLimitConstraints", power_management_limit_constraints_parser),
        NVMLSimpleFunc("nvmlDeviceGetMigMode", current_pending_mode_parser),
    ]

    # which has one device input and three outputs
    _nvml_device_three_attrs_funcs = [
        NVMLSimpleFunc("nvmlDeviceGetEncoderStats", encoder_stats_parser),
    ]

    # which has one device input and four outputs
    _nvml_device_four_attrs_funcs = [
        NVMLSimpleFunc("nvmlDeviceGetRemappedRows", remapped_row_parser),
    ]

    # which has one device input and another input indicated in possible_inputs
    _nvml_device_extra_key_attr_funcs = [
        NVMLExtraKeyFunc("nvmlDeviceGetClockInfo", range(NVML_CLOCK_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetMaxClockInfo", range(NVML_CLOCK_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetInforomVersion", range(NVML_INFOROM_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetViolationStatus", [NVML_PERF_POLICY_POWER, NVML_PERF_POLICY_THERMAL, NVML_PERF_POLICY_SYNC_BOOST, NVML_PERF_POLICY_BOARD_LIMIT, NVML_PERF_POLICY_LOW_UTILIZATION, NVML_PERF_POLICY_RELIABILITY, NVML_PERF_POLICY_TOTAL_APP_CLOCKS, NVML_PERF_POLICY_TOTAL_BASE_CLOCKS], violation_status_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetTemperature", range(NVML_TEMPERATURE_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetTemperatureThreshold", range(NVML_TEMPERATURE_THRESHOLD_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetApplicationsClock", range(NVML_CLOCK_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetMaxCustomerBoostClock", range(NVML_CLOCK_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetDefaultApplicationsClock", range(NVML_CLOCK_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetRetiredPages", range(NVML_PAGE_RETIREMENT_CAUSE_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetRetiredPages_v2", range(NVML_PAGE_RETIREMENT_CAUSE_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetAPIRestriction", range(NVML_RESTRICTED_API_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetPcieThroughput", range(NVML_PCIE_UTIL_COUNT), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetTopologyNearestGpus", [NVML_TOPOLOGY_INTERNAL, NVML_TOPOLOGY_SINGLE, NVML_TOPOLOGY_MULTIPLE, NVML_TOPOLOGY_HOSTBRIDGE, NVML_TOPOLOGY_NODE, NVML_TOPOLOGY_SYSTEM], topology_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetNvLinkState", range(NVML_NVLINK_MAX_LINKS), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetNvLinkVersion", range(NVML_NVLINK_MAX_LINKS), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetNvLinkRemotePciInfo", range(NVML_NVLINK_MAX_LINKS), pci_info_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetNvLinkRemoteDeviceType", range(NVML_NVLINK_MAX_LINKS), basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetVgpuUtilization", [0], vgpu_utilization_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetEncoderCapacity", [NVML_ENCODER_QUERY_H264, NVML_ENCODER_QUERY_HEVC, NVML_ENCODER_QUERY_AV1], basic_type_value_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetVgpuProcessUtilization", [0], vgpu_process_utilization_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetProcessUtilization", [0], process_utilization_parser),
        NVMLExtraKeyFunc("nvmlDeviceGetGpuInstanceProfileInfo", range(NVML_GPU_INSTANCE_PROFILE_COUNT), instance_profile_parser),
    ]

    # which has one device input and another two input indicated in key1_possible_inputs, key2_possible_inputs
    _nvml_device_two_keys_attr_funcs = [
        NVMLTwoKeysFunc("nvmlDeviceGetDetailedEccErrors", range(NVML_MEMORY_ERROR_TYPE_COUNT), range(NVML_ECC_COUNTER_TYPE_COUNT), ecc_error_count_parser),
        NVMLTwoKeysFunc("nvmlDeviceGetTotalEccErrors", range(NVML_MEMORY_ERROR_TYPE_COUNT), range(NVML_ECC_COUNTER_TYPE_COUNT), basic_type_value_parser),
        NVMLTwoKeysFunc("nvmlDeviceGetMemoryAffinity", [8], range(2), affinity_parser),
        NVMLTwoKeysFunc("nvmlDeviceGetCpuAffinityWithinScope", [8], range(2), affinity_parser),
        NVMLTwoKeysFunc("nvmlDeviceGetClock", range(NVML_CLOCK_COUNT), range(NVML_CLOCK_ID_COUNT), basic_type_value_parser),
        NVMLTwoKeysFunc("nvmlDeviceGetSamples", range(NVML_SAMPLINGTYPE_COUNT), [0], sample_parser),
        NVMLTwoKeysFunc("nvmlDeviceGetNvLinkCapability", range(NVML_NVLINK_MAX_LINKS), range(NVML_NVLINK_CAP_COUNT), basic_type_value_parser),
        NVMLTwoKeysFunc("nvmlDeviceGetNvLinkErrorCounter", range(NVML_NVLINK_MAX_LINKS), range(NVML_NVLINK_ERROR_COUNT), basic_type_value_parser),
        NVMLTwoKeysFunc("nvmlDeviceGetNvLinkUtilizationCounter", range(NVML_NVLINK_MAX_LINKS), range(2), basic_type_value_parser),
    ]

    _nvml_device_three_keys_attr_funcs = [
        NVMLThreeKeysFunc("nvmlDeviceGetMemoryErrorCounter", range(NVML_MEMORY_ERROR_TYPE_COUNT), range(NVML_ECC_COUNTER_TYPE_COUNT), range(NVML_MEMORY_LOCATION_COUNT), basic_type_value_parser),
    ]

    # which has one device input along with expected array size
    _nvml_device_array_size_as_input_funcs = [
        NVMLSimpleArrayOutputFunc("nvmlDeviceGetCpuAffinity", 8, affinity_parser),
    ]

    # which has one device input along with expected array size with extra key
    _nvml_device_array_size_as_input_extra_key_funcs = [
        NVMLExtraKeyArrayOutputFunc("nvmlDeviceGetMemoryAffinity", 8, range(2), affinity_parser),
        NVMLExtraKeyArrayOutputFunc("nvmlDeviceGetCpuAffinityWithinScope", 8, range(2), affinity_parser),
    ]

    # which has two devices input, framework will iterate all possible combination for produing results
    _nvml_device_two_devices_input_funcs = [
        NVMLSimpleFunc("nvmlDeviceOnSameBoard", basic_type_value_parser),
        NVMLSimpleFunc("nvmlDeviceGetTopologyCommonAncestor", basic_type_value_parser),
    ]

    # which takes one vGPU type and produces one result
    _nvml_vgpu_type_simple_funcs = [
        NVMLSimpleFunc("nvmlVgpuTypeGetClass", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuTypeGetName", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuTypeGetGpuInstanceProfileId", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuTypeGetFramebufferSize", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuTypeGetNumDisplayHeads", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuTypeGetLicense", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuTypeGetFrameRateLimit", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuTypeGetMaxInstancesPerVm", basic_type_value_parser),
    ]

    # which takes one vGPU type and a possible_inputs. It then produces one result
    _nvml_vgpu_type_extra_key_attrs_funcs = [
        NVMLExtraKeyFunc("nvmlVgpuTypeGetCapabilities", range(NVML_VGPU_CAP_COUNT), basic_type_value_parser),
    ]

    # which takes one vGPU type and a possible_inputs. It then produces two result
    _nvml_vgpu_type_extra_key_two_attrs_funcs = [
    ]

    # which takes one vGPU type and produces two attributes
    _nvml_vgpu_type_two_attrs_funcs = [
        NVMLSimpleFunc("nvmlVgpuTypeGetDeviceID", device_id_parser),
        NVMLSimpleFunc("nvmlVgpuTypeGetResolution", resolution_parser),
    ]

    # functions which are handled but is not visiable in global context
    _nvml_funcs_handled = {
        # In PyNVML the function without version covers it
        "nvmlDeviceGetPciInfo_v2",
        "nvmlDeviceGetPciInfo_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetGraphicsRunningProcesses_v2",
        "nvmlDeviceGetGraphicsRunningProcesses_v3",
        "nvmlDeviceGetMPSComputeRunningProcesses_v2",
        "nvmlDeviceGetMPSComputeRunningProcesses_v3",
        "nvmlDeviceGetNvLinkRemotePciInfo_v2",
        "nvmlVgpuInstanceGetLicenseInfo_v2",
        "nvmlDeviceGetGridLicensableFeatures_v2",
        "nvmlDeviceGetGridLicensableFeatures_v3",
        "nvmlDeviceGetGridLicensableFeatures_v4",
        "nvmlDeviceGetGpuInstanceProfileInfoV",
        "nvmlGpuInstanceGetComputeInstanceProfileInfoV",
        "nvmlComputeInstanceGetInfo_v2",
        "nvmlDeviceGetAttributes_v2",
        "nvmlDeviceGetCount_v2",

        # handled in _record_device_fan_speed_func
        "nvmlDeviceGetFanSpeed_v2",
        "nvmlDeviceGetTargetFanSpeed",

        # handled in _record_device_supported_graphics_clocks
        "nvmlDeviceGetSupportedGraphicsClocks",

        # handled in _record_device_accounting
        "nvmlDeviceGetAccountingStats",

        # handled in _record_device_vgpu_type_max_count
        "nvmlVgpuTypeGetMaxInstances",

        # handled in _record_device_field_values
        "nvmlDeviceGetFieldValues",

        # handled in _record_vgpu_instnace_accounting
        "nvmlVgpuInstanceGetAccountingStats",

        # handled in _record_device_instance
        "nvmlDeviceGetGpuInstanceRemainingCapacity",
        "nvmlDeviceGetGpuInstances",
        "nvmlGpuInstanceGetInfo",

        # handled in _record_gpu_instances
        "nvmlGpuInstanceGetComputeInstanceProfileInfo",
        "nvmlGpuInstanceGetComputeInstanceRemainingCapacity",

        # handled in _record_compute_instance
        "nvmlComputeInstanceGetInfo",

        # handled in _record_gpm
        "nvmlGpmSampleAlloc",
        "nvmlGpmSampleFree",
        "nvmlGpmSampleGet",
        "nvmlGpmMigSampleGet",
        "nvmlGpmQueryDeviceSupport",
        "nvmlGpmMetricsGet",
    }

    _nvml_not_captured_funcs = [
        # event related
        "nvmlEventSetCreate",
        "nvmlDeviceRegisterEvents",
        "nvmlEventSetWait",
        "nvmlEventSetWait_v2",
        "nvmlEventSetFree",

        # setter
        "nvmlDeviceSetComputeMode",
        "nvmlDeviceSetDriverModel",
        "nvmlDeviceSetEccMode",
        "nvmlDeviceClearEccErrorCounts",
        "nvmlDeviceSetCpuAffinity",
        "nvmlDeviceClearCpuAffinity",
        "nvmlDeviceSetPersistenceMode",
        "nvmlDeviceSetPowerMode",
        "nvmlDeviceSetTemperatureThreshold",
        "nvmlDeviceSetGpuOperationMode",
        "nvmlDeviceSetGpuLockedClocks",
        "nvmlDeviceResetGpuLockedClocks",
        "nvmlDeviceSetMemoryLockedClocks",
        "nvmlDeviceResetMemoryLockedClocks",
        "nvmlDeviceSetApplicationsClocks",
        "nvmlDeviceSetAutoBoostedClocksEnabled",
        "nvmlDeviceSetDefaultAutoBoostedClocksEnabled",
        "nvmlDeviceSetPowerManagementLimit",
        "nvmlDeviceResetApplicationsClocks",
        "nvmlDeviceSetAccountingMode",
        "nvmlDeviceClearAccountingPids",
        "nvmlDeviceResetNvLinkErrorCounters",
        "nvmlDeviceSetNvLinkUtilizationControl",
        "nvmlDeviceFreezeNvLinkUtilizationCounter",
        "nvmlDeviceResetNvLinkUtilizationCounter",
        "nvmlDeviceSetAPIRestriction",
        "nvmlDeviceSetVirtualizationMode",
        "nvmlVgpuInstanceSetEncoderCapacity",
        "nvmlDeviceModifyDrainState",
        "nvmlDeviceRemoveGpu",
        "nvmlDeviceRemoveGpu_v2",
        "nvmlVgpuInstanceClearAccountingPids",
        "nvmlGetExcludedDeviceInfoByIndex",
        "nvmlSetVgpuVersion",
        "nvmlDeviceSetMigMode",
        "nvmlDeviceCreateGpuInstance",
        "nvmlDeviceCreateGpuInstanceWithPlacement",
        "nvmlGpuInstanceDestroy",
        "nvmlGpuInstanceCreateComputeInstance",
        "nvmlComputeInstanceDestroy",
        "nvmlDeviceSetFanSpeed_v2",
        "nvmlDeviceSetDefaultFanSpeed_v2",
        "nvmlDeviceSetGpcClkVfOffset",
        "nvmlDeviceSetMemClkVfOffset",

        # handle/instance getter
        "nvmlDeviceGetHandleByIndex",
        "nvmlDeviceGetHandleBySerial",
        "nvmlDeviceGetHandleByUUID",
        "nvmlDeviceGetHandleByPciBusId",
        "nvmlDeviceGetHandleByPciBusId_v2",
        "nvmlDeviceGetGpuInstanceById",
        "nvmlGpuInstanceGetComputeInstances",
        "nvmlGpuInstanceGetComputeInstanceById",
        "nvmlDeviceGetMigDeviceHandleByIndex",
        "nvmlDeviceGetDeviceHandleFromMigDeviceHandle",
        "nvmlDeviceGetHandleByIndex_v2",

        # function not found
        # in PyNVML, version is passed by parameter but it still shows not found
        "nvmlDeviceGetMemoryInfo_v2",
        "nvmlDeviceGetPowerMode",
        "nvmlDeviceGetSupportedPowerModes",

        # deprecated functions
        "nvmlUnitGetCount",
        "nvmlUnitGetFanSpeedInfo",
        "nvmlUnitGetHandleByIndex",
        "nvmlUnitGetLedState",
        "nvmlUnitSetLedState",
        "nvmlUnitGetPsuInfo",
        "nvmlUnitGetTemperature",
        "nvmlUnitGetUnitInfo",
        "nvmlUnitGetDevices",

        # not used
        "nvmlDeviceGetNvLinkUtilizationControl",
        "nvmlSystemGetHicVersion",
        "nvmlSystemGetTopologyGpuSet",
        "nvmlDeviceGetP2PStatus",
        "nvmlGetVgpuCompatibility",
        "nvmlDeviceGetPgpuMetadataString",
        "nvmlDeviceGetGspFirmwareVersion",
        "nvmlDeviceGetGspFirmwareMode",
        "nvmlDeviceQueryDrainState",
        "nvmlDeviceDiscoverGpus",
        "nvmlGetBlacklistDeviceCount",
        "nvmlGetBlacklistDeviceInfoByIndex",
        "nvmlGetVgpuVersion",
        "nvmlDeviceGetGpuInstancePossiblePlacements",
        "nvmlDeviceGetGpuInstancePossiblePlacements_v2",
        "nvmlDeviceGetDynamicPstatesInfo",
        "nvmlDeviceGetThermalSettings",
        "nvmlDeviceGetMinMaxClockOfPState",
        "nvmlDeviceGetSupportedPerformanceStates",
        "nvmlDeviceGetMinMaxFanSpeed",
        "nvmlDeviceGetGpcClkMinMaxVfOffset",
        "nvmlDeviceGetMemClkMinMaxVfOffset",

        # system current status, does not make sense to capture it
        "nvmlSystemGetProcessName",
    ]

    # which takes one vGPU instance and produces one result
    _nvml_vgpu_instance_simple_funcs = [
        NVMLSimpleFunc("nvmlVgpuInstanceGetUUID", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetMdevUUID", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetVmDriverVersion", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetFbUsage", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetLicenseStatus", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetLicenseInfo", vgpu_license_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetType", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetFrameRateLimit", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetEccMode", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetEncoderCapacity", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetMetadata", vgpu_metadata_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetGpuPciId", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetGpuInstanceId", basic_type_value_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetEncoderSessions", encoder_sessions_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetFBCStats", fbc_stats_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetFBCSessions", fbc_sessions_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetAccountingMode", basic_type_value_parser),
    ]

    # which takes one vGPU instance and produces two attributes
    _nvml_vgpu_instance_two_attrs_funcs = [
        NVMLSimpleFunc("nvmlVgpuInstanceGetVmID", vm_id_parser),
        NVMLSimpleFunc("nvmlVgpuInstanceGetAccountingPids", accounting_pid_parser),
    ]

    # which takes one vGPU instance and produces three attributes
    _nvml_vgpu_instance_three_attrs_funcs = [
        NVMLSimpleFunc("nvmlVgpuInstanceGetEncoderStats", encoder_stats_parser),
    ]

    def __enter__(self):
        nvmlInit()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        nvmlShutdown()

    def _record_global_attr(self):
        self._attrs[GLOBAL_KEY] = {}
        for global_func in self._nvml_global_attr_funcs:
            suffix_key = nvml_func_key_suffix_extract(global_func._func_str)
            self._attrs[GLOBAL_KEY][suffix_key] = {}
            func_ret, ret_val = run_nvml_func_str(global_func._func_str)
            self._attrs[GLOBAL_KEY][suffix_key][FUNC_RETURN] = func_ret
            if func_ret == NVML_SUCCESS:
                self._attrs[GLOBAL_KEY][suffix_key][RETURN_VAL] = global_func._value_parser(ret_val)

    # Some GPU does not support nvmlDeviceGetMemoryInfo_v2, add rollback layer
    def _record_device_memory_info_func(self, nvml_device, device_attr):
        func_ret, ret_val = run_nvml_func(nvmlDeviceGetMemoryInfo_v2, nvml_device)
        suffix_key = "MemoryInfo"
        device_attr[suffix_key] = {}
        device_attr[suffix_key][FUNC_RETURN] = func_ret
        if func_ret == NVML_SUCCESS:
            device_attr[suffix_key][RETURN_VAL] = memory_info_v2_parser(ret_val)
            return
        func_ret, ret_val = run_nvml_func(nvmlDeviceGetMemoryInfo, nvml_device)
        device_attr[suffix_key][FUNC_RETURN] = func_ret
        if func_ret == NVML_SUCCESS:
            device_attr[suffix_key][RETURN_VAL] = memory_info_parser(ret_val)

    def _record_device_fan_speed_func(self, nvml_device, device_attr):
        if nvmlDeviceIsMigDeviceHandle(nvml_device):
            return
        num_fans = nvmlDeviceGetNumFans(nvml_device)
        fan_speed_funcs = [
            NVMLExtraKeyFunc("nvmlDeviceGetFanSpeed_v2", range(num_fans), basic_type_value_parser),
            NVMLExtraKeyFunc("nvmlDeviceGetTargetFanSpeed", range(num_fans), basic_type_value_parser),
        ]
        for extra_key_func in fan_speed_funcs:
            suffix_key = nvml_func_key_suffix_extract(extra_key_func._func_str)
            device_attr[suffix_key] = {}
            for input in extra_key_func._possible_inputs:
                func_ret, ret_val = run_nvml_func_str(extra_key_func._func_str, nvml_device, input)
                device_attr[suffix_key][input] = {}
                device_attr[suffix_key][input][FUNC_RETURN] = func_ret
                device_attr[suffix_key][input][RETURN_VAL] = extra_key_func._value_parser(ret_val)

    def _record_device_supported_graphics_clocks(self, nvml_device, device_attr):
        func_ret, memory_clocks = run_nvml_func(nvmlDeviceGetSupportedMemoryClocks, nvml_device)
        suffix_key = nvml_func_key_suffix_extract("nvmlDeviceGetSupportedGraphicsClocks")
        device_attr[suffix_key] = {}
        if func_ret != NVML_SUCCESS:
            return
        for memory_clock in memory_clocks:
            device_attr[suffix_key][memory_clock] = {}
            func_ret, ret_val = run_nvml_func(nvmlDeviceGetSupportedGraphicsClocks, nvml_device, memory_clock)
            device_attr[suffix_key][memory_clock][FUNC_RETURN] = func_ret
            device_attr[suffix_key][memory_clock][RETURN_VAL] = ret_val

    def _record_device_accounting(self, nvml_device, device_attr):
        func_ret, pids = run_nvml_func(nvmlDeviceGetAccountingPids, nvml_device)
        suffix_key = nvml_func_key_suffix_extract("nvmlDeviceGetAccountingStats")
        if func_ret != NVML_SUCCESS:
            return
        device_attr[suffix_key] = {}
        for pid in pids:
            func_ret, ret_val = run_nvml_func(nvmlDeviceGetAccountingStats, nvml_device, pid)
            device_attr[suffix_key][pid] = {}
            device_attr[suffix_key][pid][RETURN_VAL] = func_ret
            device_attr[suffix_key][pid][RETURN_VAL] = accounting_stat_parser(ret_val)

    def _record_device_vgpu_type_max_count(self, nvml_device, device_attr):
        func_ret, supported_vgpu_types = run_nvml_func(nvmlDeviceGetSupportedVgpus, nvml_device)
        if func_ret != NVML_SUCCESS:
            return
        suffix_key = nvml_func_key_suffix_extract("nvmlVgpuTypeGetMaxInstances")
        device_attr[suffix_key] = {}
        for supported_vgpu_type in supported_vgpu_types:
            device_attr[suffix_key][supported_vgpu_type] = {}
            func_ret, ret_val = run_nvml_func(nvmlVgpuTypeGetMaxInstances, nvml_device, supported_vgpu_type)
            device_attr[suffix_key][supported_vgpu_type][FUNC_RETURN] = func_ret
            device_attr[suffix_key][supported_vgpu_type][RETURN_VAL] = basic_type_value_parser(ret_val)

    def _record_device_field_values(self, nvml_device, device_attr):
        func_ret, values = run_nvml_func(nvmlDeviceGetFieldValues, nvml_device, range(NVML_FI_MAX))
        suffix_key = nvml_func_key_suffix_extract("nvmlDeviceGetFieldValues")
        device_attr[suffix_key] = {}
        device_attr[suffix_key][FUNC_RETURN] = func_ret
        if func_ret != NVML_SUCCESS:
            return
        device_attr[suffix_key][RETURN_VAL] = {}
        for value in values:
            parsed_value = field_value_parser(value)
            device_attr[suffix_key][RETURN_VAL][value.fieldId] = parsed_value

    def _record_device_instance(self, nvml_device, device_attr, device_uuid):
        func_ret, current, _ = run_nvml_two_attrs_func_str("nvmlDeviceGetMigMode", nvml_device)
        if func_ret != NVML_SUCCESS or current != NVML_DEVICE_MIG_ENABLE:
            return
        for profile in range(NVML_GPU_INSTANCE_PROFILE_COUNT):
            func_ret, profile_info = run_nvml_func(nvmlDeviceGetGpuInstanceProfileInfo, nvml_device, profile)
            if func_ret != NVML_SUCCESS:
                continue
            suffix_key = nvml_func_key_suffix_extract("nvmlDeviceGetGpuInstanceRemainingCapacity")
            if suffix_key not in device_attr:
                device_attr[suffix_key] = {}
            func_ret, count = run_nvml_func(nvmlDeviceGetGpuInstanceRemainingCapacity, nvml_device, profile_info.id)
            device_attr[suffix_key][profile_info.id] = {}
            device_attr[suffix_key][profile_info.id][FUNC_RETURN] = func_ret
            if func_ret != NVML_SUCCESS:
                continue
            device_attr[suffix_key][profile_info.id][RETURN_VAL] = count
            InstancesArray = c_nvmlGpuInstance_t * MAX_NUM_INSTANCE
            c_count = c_uint()
            c_profile_instances = InstancesArray()
            suffix_key = nvml_func_key_suffix_extract("nvmlDeviceGetGpuInstances")
            if suffix_key not in device_attr:
                device_attr[suffix_key] = {}
            func_ret = run_nvml_no_output_func_str("nvmlDeviceGetGpuInstances", nvml_device, profile_info.id, c_profile_instances, byref(c_count))
            device_attr[suffix_key][profile_info.id] = {}
            device_attr[suffix_key][profile_info.id][FUNC_RETURN] = func_ret
            if func_ret != NVML_SUCCESS:
                continue
            device_attr[suffix_key][profile_info.id][RETURN_VAL] = []
            for i in range(c_count.value):
                func_ret, instance_info = run_nvml_func(nvmlGpuInstanceGetInfo, c_profile_instances[i])
                if func_ret != NVML_SUCCESS:
                    continue
                # create fake_gpu_instance so that we can index it in c++
                fake_gpu_instance = f"{device_uuid}_{instance_info.id}"
                device_attr[suffix_key][profile_info.id][RETURN_VAL].append(fake_gpu_instance)
                self._gpu_instances_mapping[fake_gpu_instance] = {}
                self._gpu_instances_mapping[fake_gpu_instance] = (device_uuid, c_profile_instances[i])

    def _record_mig_devices_uuid(self, nvml_device, device_attr):
        if nvmlDeviceIsMigDeviceHandle(nvml_device):
            return
        func_ret, mig_count = run_nvml_func(nvmlDeviceGetMaxMigDeviceCount, nvml_device)
        if func_ret != NVML_SUCCESS:
            return
        # Actual NVML APIs don't provide this, we add this one for the fake mig device handle in c++
        suffix = "MigDeviceUUID"
        device_attr[suffix] = []
        for mig_idx in range(mig_count):
            func_ret, nvml_mig_device = run_nvml_func(nvmlDeviceGetMigDeviceHandleByIndex, nvml_device, mig_idx)
            if func_ret != NVML_SUCCESS:
                continue
            mig_device_uuid = nvmlDeviceGetUUID(nvml_mig_device)
            device_attr[suffix].append(mig_device_uuid)
            self._mig_devices_collector.append((mig_device_uuid, nvml_mig_device))

    def _get_two_samples_mig_device(self, nvml_device):
        func_ret, instance_id = run_nvml_func(nvmlDeviceGetGpuInstanceId, nvml_device)
        if func_ret != NVML_SUCCESS:
            return None, None
        sample1 = nvmlGpmSampleAlloc()
        func_ret, sample1 = run_nvml_func(nvmlGpmMigSampleGet, nvml_device, instance_id, sample1)
        if func_ret != NVML_SUCCESS:
            return None, None
        time.sleep(1)
        sample2 = nvmlGpmSampleAlloc()
        func_ret, sample2 = run_nvml_func(nvmlGpmMigSampleGet, nvml_device, instance_id, sample2)
        if func_ret != NVML_SUCCESS:
            return None, None
        return sample1, sample2

    def _get_two_samples_normal_device(self, nvml_device):
        sample1 = nvmlGpmSampleAlloc()
        func_ret, sample1 = run_nvml_func(nvmlGpmSampleGet, nvml_device, sample1)
        if func_ret != NVML_SUCCESS:
            return None, None
        time.sleep(1)
        sample2 = nvmlGpmSampleAlloc()
        func_ret, sample2 = run_nvml_func(nvmlGpmSampleGet, nvml_device, sample2)
        if func_ret != NVML_SUCCESS:
            return None, None
        return sample1, sample2

    def _get_two_gpm_samples(self, nvml_device):
        if nvmlDeviceIsMigDeviceHandle(nvml_device):
            return self._get_two_samples_mig_device(nvml_device)
        else:
            return self._get_two_samples_normal_device(nvml_device)

    def _record_gpm(self, nvml_device, device_attr):
        func_ret, support = run_nvml_func(nvmlGpmQueryDeviceSupport, nvml_device)
        device_attr["QueryDeviceSupport"] = {}
        device_attr["QueryDeviceSupport"][FUNC_RETURN] = func_ret
        if func_ret != NVML_SUCCESS:
            return
        device_attr["QueryDeviceSupport"][RETURN_VAL] = {}
        device_attr["QueryDeviceSupport"][RETURN_VAL]["isSupportedDevice"] = support.isSupportedDevice
        device_attr["QueryDeviceSupport"][RETURN_VAL]["version"] = support.version
        if not support:
            return
        sample1, sample2 = self._get_two_gpm_samples(nvml_device)
        if sample1 == None or sample2 == None:
            return
        device_attr["GpmMetrics"] = {}
        mg = c_nvmlGpmMetricsGet_t()
        for id in range(1, NVML_GPM_METRIC_MAX):
            mg.version = NVML_GPM_METRICS_GET_VERSION
            mg.numMetrics = NVML_GPM_METRIC_MAX
            mg.sample1 = sample1
            mg.sample2 = sample2
            mg.metrics[id - 1].metricId = id
        func_ret, mg = run_nvml_func(nvmlGpmMetricsGet, mg)
        device_attr["GpmMetrics"][FUNC_RETURN] = func_ret
        if func_ret != NVML_SUCCESS:
            return
        device_attr["GpmMetrics"][RETURN_VAL] = {}
        for id in range(1, NVML_GPM_METRIC_MAX):
            device_attr["GpmMetrics"][RETURN_VAL][id] = {}
            device_attr["GpmMetrics"][RETURN_VAL][id][FUNC_RETURN] = mg.metrics[id - 1].nvmlReturn
            device_attr["GpmMetrics"][RETURN_VAL][id][RETURN_VAL] = {}
            device_attr["GpmMetrics"][RETURN_VAL][id][RETURN_VAL]["value"] = mg.metrics[id - 1].value
            device_attr["GpmMetrics"][RETURN_VAL][id][RETURN_VAL]["metricInfo"] = {}
            device_attr["GpmMetrics"][RETURN_VAL][id][RETURN_VAL]["metricInfo"]["shortName"] = mg.metrics[id - 1].metricInfo.shortName
            device_attr["GpmMetrics"][RETURN_VAL][id][RETURN_VAL]["metricInfo"]["longName"] = mg.metrics[id - 1].metricInfo.longName
            device_attr["GpmMetrics"][RETURN_VAL][id][RETURN_VAL]["metricInfo"]["unit"] = mg.metrics[id - 1].metricInfo.unit

        nvmlGpmSampleFree(sample1)
        nvmlGpmSampleFree(sample2)

    def _get_device_attrs(self, nvml_device, device_uuid):
        device_attr = {}
        for simple_func in self._nvml_device_attr_funcs:
            func_ret, ret_val = run_nvml_func_str(simple_func._func_str, nvml_device)
            suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
            device_attr[suffix_key] = {}
            device_attr[suffix_key][FUNC_RETURN] = func_ret
            if func_ret == NVML_SUCCESS:
                device_attr[suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val)

        for simple_func in self._nvml_device_with_version_attr_funcs:
            func_ret, ret_val = run_nvml_func_str(simple_func._func_str, nvml_device, version=simple_func._version)
            suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
            device_attr[suffix_key] = {}
            device_attr[suffix_key][FUNC_RETURN] = func_ret
            if func_ret == NVML_SUCCESS:
                device_attr[suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val)

        for simple_func in self._nvml_device_two_attrs_funcs:
            func_ret, ret_val1, ret_val2 = run_nvml_two_attrs_func_str(simple_func._func_str, nvml_device)
            suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
            device_attr[suffix_key] = {}
            device_attr[suffix_key][FUNC_RETURN] = func_ret
            if func_ret == NVML_SUCCESS:
                device_attr[suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val1, ret_val2)

        for simple_func in self._nvml_device_three_attrs_funcs:
            func_ret, ret_val1, ret_val2, ret_val3 = run_nvml_three_attrs_func_str(simple_func._func_str, nvml_device)
            suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
            device_attr[suffix_key] = {}
            device_attr[suffix_key][FUNC_RETURN] = func_ret
            if func_ret == NVML_SUCCESS:
                device_attr[suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val1, ret_val2, ret_val3)

        for simple_func in self._nvml_device_four_attrs_funcs:
            func_ret, ret_val1, ret_val2, ret_val3, ret_val4 = run_nvml_four_attrs_func_str(simple_func._func_str, nvml_device)
            suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
            device_attr[suffix_key] = {}
            device_attr[suffix_key][FUNC_RETURN] = func_ret
            if func_ret == NVML_SUCCESS:
                device_attr[suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val1, ret_val2, ret_val3, ret_val4)

        for extra_key_func in self._nvml_device_extra_key_attr_funcs:
            suffix_key = nvml_func_key_suffix_extract(extra_key_func._func_str)
            device_attr[suffix_key] = {}
            for input in extra_key_func._possible_inputs:
                func_ret, ret_val = run_nvml_func_str(extra_key_func._func_str, nvml_device, input)
                device_attr[suffix_key][input] = {}
                device_attr[suffix_key][input][FUNC_RETURN] = func_ret
                if func_ret == NVML_SUCCESS:
                    device_attr[suffix_key][input][RETURN_VAL] = extra_key_func._value_parser(ret_val)

        for two_keys_func in self._nvml_device_two_keys_attr_funcs:
            suffix_key = nvml_func_key_suffix_extract(two_keys_func._func_str)
            device_attr[suffix_key] = {}
            for key1 in two_keys_func._key1_possible_inputs:
                device_attr[suffix_key][key1] = {}
                for key2 in two_keys_func._key2_possible_inputs:
                    func_ret, ret_val = run_nvml_func_str(two_keys_func._func_str, nvml_device, key1, key2)
                    device_attr[suffix_key][key1][key2] = {}
                    device_attr[suffix_key][key1][key2][FUNC_RETURN] = func_ret
                    if func_ret == NVML_SUCCESS:
                        device_attr[suffix_key][key1][key2][RETURN_VAL] = two_keys_func._value_parser(ret_val)

        for three_keys_func in self._nvml_device_three_keys_attr_funcs:
            suffix_key = nvml_func_key_suffix_extract(three_keys_func._func_str)
            device_attr[suffix_key] = {}
            for key1 in three_keys_func._key1_possible_inputs:
                device_attr[suffix_key][key1] = {}
                for key2 in three_keys_func._key2_possible_inputs:
                    device_attr[suffix_key][key1][key2] = {}
                    for key3 in three_keys_func._key3_possible_inputs:
                        func_ret, ret_val = run_nvml_func_str(three_keys_func._func_str, nvml_device, key1, key2, key3)
                        device_attr[suffix_key][key1][key2][key3] = {}
                        device_attr[suffix_key][key1][key2][key3][FUNC_RETURN] = func_ret
                        if func_ret == NVML_SUCCESS:
                            device_attr[suffix_key][key1][key2][key3][RETURN_VAL] = three_keys_func._value_parser(ret_val)

        for simple_array_out_func in self._nvml_device_array_size_as_input_funcs:
            suffix_key = nvml_func_key_suffix_extract(simple_array_out_func._func_str)
            device_attr[suffix_key] = {}
            func_ret, ret_val = run_nvml_func_str(simple_array_out_func._func_str, nvml_device, simple_array_out_func._arr_size)
            device_attr[suffix_key][FUNC_RETURN] = func_ret
            if func_ret == NVML_SUCCESS:
                device_attr[suffix_key][RETURN_VAL] = simple_array_out_func._value_parser(ret_val)

        for extra_key_array_out_func in self._nvml_device_array_size_as_input_extra_key_funcs:
            suffix_key = nvml_func_key_suffix_extract(extra_key_array_out_func._func_str)
            device_attr[suffix_key] = {}
            for input in extra_key_array_out_func._possible_inputs:
                func_ret, ret_val = run_nvml_func_str(extra_key_array_out_func._func_str, nvml_device, extra_key_array_out_func._arr_size, input)
                device_attr[suffix_key][input] = {}
                device_attr[suffix_key][input][FUNC_RETURN] = func_ret
                if func_ret == NVML_SUCCESS:
                    device_attr[suffix_key][input][RETURN_VAL] = extra_key_array_out_func._value_parser(ret_val)

        for j in range(self._attrs[GLOBAL_KEY][COUNT][RETURN_VAL]):
            nvml_device_j = nvmlDeviceGetHandleByIndex(j)
            device_uuid_j = nvmlDeviceGetUUID(nvml_device_j)
            for two_devices_input_func in self._nvml_device_two_devices_input_funcs:
                suffix_key = nvml_func_key_suffix_extract(two_devices_input_func._func_str)
                func_ret, ret_val = run_nvml_func_str(two_devices_input_func._func_str, nvml_device, nvml_device_j)
                if suffix_key not in device_attr:
                    device_attr[suffix_key] = {}
                device_attr[suffix_key][device_uuid_j] = {}
                device_attr[suffix_key][device_uuid_j][FUNC_RETURN] = func_ret
                if func_ret == NVML_SUCCESS:
                    device_attr[suffix_key][device_uuid_j][RETURN_VAL] = two_devices_input_func._value_parser(ret_val)

        for func_str in self._nvml_device_no_output_funcs:
            suffix_key = nvml_func_key_suffix_extract(func_str)
            func_ret = run_nvml_no_output_func_str(func_str, nvml_device)
            device_attr[suffix_key] = {}
            if func_ret == NVML_SUCCESS:
                device_attr[suffix_key][FUNC_RETURN] = func_ret

        self._record_device_memory_info_func(nvml_device, device_attr)
        self._record_device_fan_speed_func(nvml_device, device_attr)
        self._record_device_supported_graphics_clocks(nvml_device, device_attr)
        self._record_device_accounting(nvml_device, device_attr)
        self._record_device_vgpu_type_max_count(nvml_device, device_attr)
        self._record_device_field_values(nvml_device, device_attr)
        self._record_device_instance(nvml_device, device_attr, device_uuid)
        self._record_mig_devices_uuid(nvml_device, device_attr)
        self._record_gpm(nvml_device, device_attr)
        return device_attr

    def _record_devices_funcs(self):
        self._attrs[DEVICE] = {}
        self._attrs[GLOBAL_KEY][DEVICE_ORDER] = []
        for device_idx in range(self._attrs[GLOBAL_KEY][COUNT][RETURN_VAL]):
            nvml_device = nvmlDeviceGetHandleByIndex(device_idx)
            device_uuid = nvmlDeviceGetUUID(nvml_device)
            self._attrs[GLOBAL_KEY][DEVICE_ORDER].append(device_uuid)
            self._attrs[DEVICE][device_uuid] = self._get_device_attrs(nvml_device, device_uuid)

    def _record_vgpu_types_funcs(self):
        self._attrs[VGPU_TYPE] = {}
        vgpu_type_ids = []
        for idx in range(self._attrs[GLOBAL_KEY][COUNT][RETURN_VAL]):
            nvml_device = nvmlDeviceGetHandleByIndex(idx)
            func_ret, ids = run_nvml_func(nvmlDeviceGetSupportedVgpus, nvml_device)
            if func_ret != NVML_SUCCESS:
                continue
            vgpu_type_ids.extend(ids)

        for vgpu_type_id in vgpu_type_ids:
            self._attrs[VGPU_TYPE][vgpu_type_id] = {}
            for simple_func in self._nvml_vgpu_type_simple_funcs:
                func_ret, ret_val = run_nvml_func_str(simple_func._func_str, vgpu_type_id)
                suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
                self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key] = {}
                self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][FUNC_RETURN] = func_ret
                if func_ret == NVML_SUCCESS:
                    self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val)

            for extra_key_func in self._nvml_vgpu_type_extra_key_attrs_funcs:
                suffix_key = nvml_func_key_suffix_extract(extra_key_func._func_str)
                self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key] = {}
                for input in extra_key_func._possible_inputs:
                    func_ret, ret_val = run_nvml_func_str(extra_key_func._func_str, nvml_device, input)
                    self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][input] = {}
                    self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][input][FUNC_RETURN] = func_ret
                    if func_ret == NVML_SUCCESS:
                        self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][input][RETURN_VAL] = extra_key_func._value_parser(ret_val)

            for simple_func in self._nvml_vgpu_type_two_attrs_funcs:
                func_ret, ret_val1, ret_val2 = run_nvml_two_attrs_func_str(simple_func._func_str, vgpu_type_id)
                suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
                self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key] = {}
                self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][FUNC_RETURN] = func_ret
                if func_ret == NVML_SUCCESS:
                    self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val1, ret_val2)

            for extra_key_func in self._nvml_vgpu_type_extra_key_two_attrs_funcs:
                suffix_key = nvml_func_key_suffix_extract(extra_key_func._func_str)
                self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key] = {}
                for input in extra_key_func._possible_inputs:
                    func_ret, ret_val1, ret_val2 = run_nvml_two_attrs_func_str(extra_key_func._func_str, vgpu_type_id, input)
                    self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][input] = {}
                    self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][input][FUNC_RETURN] = func_ret
                    if func_ret == NVML_SUCCESS:
                        self._attrs[VGPU_TYPE][vgpu_type_id][suffix_key][input][RETURN_VAL] = extra_key_func._value_parser(ret_val1, ret_val2)

    def _record_vgpu_instnace_accounting(self, vgpu_instance):
        count, c_pids = nvmlVgpuInstanceGetAccountingPids(vgpu_instance)
        pids = accounting_pid_parser(count, c_pids)
        suffix_key = nvml_func_key_suffix_extract("nvmlVgpuInstanceGetAccountingStats")
        self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key] = {}
        for pid in pids:
            self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][pid] = {}
            func_ret, ret_val = run_nvml_func(nvmlVgpuInstanceGetAccountingStats, vgpu_instance, pid)
            self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][pid][FUNC_RETURN] = func_ret
            self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][pid][RETURN_VAL] = accounting_stat_parser(ret_val)

    def _record_vgpu_instances_funcs(self):
        self._attrs[VGPU_INSTANCE] = {}
        vgpu_instances = []
        for idx in range(self._attrs[GLOBAL_KEY][COUNT][RETURN_VAL]):
            nvml_device = nvmlDeviceGetHandleByIndex(idx)
            func_ret, instances = run_nvml_func(nvmlDeviceGetActiveVgpus, nvml_device)
            if func_ret != NVML_SUCCESS:
                continue
            vgpu_instances.extend(instances)

        for vgpu_instance in vgpu_instances:
            self._attrs[VGPU_INSTANCE][vgpu_instance] = {}
            for simple_func in self._nvml_vgpu_instance_simple_funcs:
                func_ret, ret_val = run_nvml_func_str(simple_func._func_str, vgpu_instance)
                suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
                self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key] = {}
                self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][FUNC_RETURN] = func_ret
                if func_ret == NVML_SUCCESS:
                    self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val)

            for simple_func in self._nvml_vgpu_instance_two_attrs_funcs:
                func_ret, ret_val1, ret_val2 = run_nvml_two_attrs_func_str(simple_func._func_str, vgpu_instance)
                suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
                self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key] = {}
                self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][FUNC_RETURN] = func_ret
                if func_ret == NVML_SUCCESS:
                    self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val1, ret_val2)

            for simple_func in self._nvml_vgpu_instance_three_attrs_funcs:
                func_ret, ret_val1, ret_val2, ret_val3 = run_nvml_three_attrs_func_str(simple_func._func_str, vgpu_instance)
                suffix_key = nvml_func_key_suffix_extract(simple_func._func_str)
                self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key] = {}
                self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][FUNC_RETURN] = func_ret
                if func_ret == NVML_SUCCESS:
                    self._attrs[VGPU_INSTANCE][vgpu_instance][suffix_key][RETURN_VAL] = simple_func._value_parser(ret_val1, ret_val2, ret_val3)

            self._record_vgpu_instnace_accounting(vgpu_instance)

    def _record_excluded_devices(self):
        excluded_device_count = nvmlGetExcludedDeviceCount()
        self._attrs[EXCLUDED_DEVICE] = {}
        for idx in range(excluded_device_count):
            _, excluded_device_info = run_nvml_func(nvmlGetExcludedDeviceInfoByIndex, idx)
            self._attrs[EXCLUDED_DEVICE][excluded_device_info.uuid] = {}
            self._attrs[EXCLUDED_DEVICE][excluded_device_info.uuid]["pci"] = pci_info_parser(excluded_device_info.pci)

    def _record_compute_instance(self):
        self._attrs[COMPUTE_INSTANCE] = {}
        for fake_ci, instance_obj in self._compute_instance_mapping.items():
            self._attrs[COMPUTE_INSTANCE][fake_ci] = {}
            parent_device_uuid, fake_gpu_instance, real_ci = instance_obj
            suffix_key = nvml_func_key_suffix_extract("nvmlComputeInstanceGetInfo")
            self._attrs[COMPUTE_INSTANCE][fake_ci][suffix_key] = {}
            func_ret, ci_info = run_nvml_func(nvmlComputeInstanceGetInfo, real_ci)
            self._attrs[COMPUTE_INSTANCE][fake_ci][suffix_key][FUNC_RETURN] = func_ret
            if func_ret != NVML_SUCCESS:
                continue
            self._attrs[COMPUTE_INSTANCE][fake_ci][suffix_key][RETURN_VAL] = ci_info_parser(parent_device_uuid, fake_gpu_instance, ci_info)

    def _record_gpu_instances(self):
        self._attrs[GPU_INSTANCE] = {}
        for fake_gpu_instance, instance_obj in self._gpu_instances_mapping.items():
            parent_device_uuid, real_gpu_instance = instance_obj
            self._attrs[GPU_INSTANCE][fake_gpu_instance] = {}
            suffix_key = nvml_func_key_suffix_extract("nvmlGpuInstanceGetInfo")
            self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key] = {}
            func_ret, info = run_nvml_func_str("nvmlGpuInstanceGetInfo", real_gpu_instance)
            self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][FUNC_RETURN] = func_ret
            if func_ret != NVML_SUCCESS:
                continue
            self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][RETURN_VAL] = instance_info_parser(parent_device_uuid, info)
            suffix_key = nvml_func_key_suffix_extract("nvmlGpuInstanceGetComputeInstanceProfileInfo")
            self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key] = {}
            all_ci_profile_ids = []
            for ci_instance_profile in range(NVML_COMPUTE_INSTANCE_PROFILE_COUNT):
                self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_instance_profile] = {}
                for ci_engine_profile in range(NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT):
                    self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_instance_profile][ci_engine_profile] = {}
                    func_ret, ci_profile_info = run_nvml_func(
                                nvmlGpuInstanceGetComputeInstanceProfileInfo, real_gpu_instance, ci_instance_profile, ci_engine_profile)
                    self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_instance_profile][ci_engine_profile][FUNC_RETURN] = func_ret
                    if func_ret != NVML_SUCCESS:
                        continue
                    self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_instance_profile][ci_engine_profile][RETURN_VAL] = ci_profile_info_parser(ci_profile_info)
                    all_ci_profile_ids.append(ci_profile_info.id)

            suffix_key = nvml_func_key_suffix_extract("nvmlGpuInstanceGetComputeInstanceRemainingCapacity")
            self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key] = {}
            for ci_profile_id in all_ci_profile_ids:
                func_ret, count = run_nvml_func(nvmlGpuInstanceGetComputeInstanceRemainingCapacity, real_gpu_instance, ci_profile_id)
                self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_profile_id] = {}
                self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_profile_id][FUNC_RETURN] = func_ret
                if func_ret != NVML_SUCCESS:
                    continue
                self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_profile_id][RETURN_VAL] = count

            suffix_key = nvml_func_key_suffix_extract("nvmlGpuInstanceGetComputeInstances")
            self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key] = {}
            for ci_profile_id in all_ci_profile_ids:
                self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_profile_id] = []
                InstancesArray = c_nvmlComputeInstance_t * MAX_NUM_INSTANCE
                c_count = c_uint()
                c_profile_instances = InstancesArray()
                func_ret = run_nvml_no_output_func_str(
                        "nvmlGpuInstanceGetComputeInstances", real_gpu_instance, ci_profile_id, c_profile_instances, byref(c_count))
                if func_ret != NVML_SUCCESS:
                    continue
                for i in range(c_count.value):
                    func_ret, ci_info = run_nvml_func(nvmlComputeInstanceGetInfo, c_profile_instances[i])
                    if func_ret != NVML_SUCCESS:
                        continue
                    fake_ci = f"{fake_gpu_instance}_{ci_info.id}"
                    self._compute_instance_mapping[fake_ci] = (parent_device_uuid, fake_gpu_instance, c_profile_instances[i])
                    self._attrs[GPU_INSTANCE][fake_gpu_instance][suffix_key][ci_profile_id].append(fake_ci)

    def _record_mig_devices_funcs(self):
        self._attrs[MIG_DEVICE] = {}
        for mig_device_uuid, mig_device in self._mig_devices_collector:
            self._attrs[MIG_DEVICE][mig_device_uuid] = self._get_device_attrs(mig_device, mig_device_uuid)

    def record(self, out_file_path):
        self._record_global_attr()
        self._record_devices_funcs()
        self._record_vgpu_types_funcs()
        self._record_vgpu_instances_funcs()
        self._record_excluded_devices()
        self._record_gpu_instances()
        self._record_compute_instance()
        self._record_mig_devices_funcs()

        # print(self._attrs)
        with open(out_file_path, "w") as outfile:
            yaml.dump(self._attrs, outfile)

    def captured_funcs_list(self):
        funcs = []
        for func in self._nvml_global_attr_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_no_output_funcs:
            funcs.append(func)
        for func in self._nvml_device_attr_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_with_version_attr_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_two_attrs_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_three_attrs_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_four_attrs_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_extra_key_attr_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_two_keys_attr_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_three_keys_attr_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_array_size_as_input_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_array_size_as_input_extra_key_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_device_two_devices_input_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_vgpu_type_simple_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_vgpu_type_extra_key_attrs_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_vgpu_type_extra_key_two_attrs_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_vgpu_type_two_attrs_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_vgpu_instance_simple_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_vgpu_instance_two_attrs_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_vgpu_instance_three_attrs_funcs:
            funcs.append(func._func_str)
        for func in self._nvml_funcs_handled:
            funcs.append(func)
        return funcs

    def known_but_skipped_funcs(self):
        return self._nvml_not_captured_funcs

    def all_funcs_in_entry_points(self, entry_points_path):
        funcs = []
        with open(entry_points_path, 'r') as file:
            content = file.read()

            pattern = re.compile(r'NVML_ENTRY_POINT\((.*), .*,')
            matches = re.findall(pattern, content)

            for match in matches:
                funcs.append(match)
        return funcs

    def has_not_handled_funcs(self, entry_points_path):
        all_handled = self.captured_funcs_list()
        known_but_skipped = self.known_but_skipped_funcs()
        all_handled.extend(known_but_skipped)
        all_funcs = self.all_funcs_in_entry_points(entry_points_path)
        all_handled_set = set(all_handled)
        has_not_handled = False
        for func in all_funcs:
            if func not in all_handled_set:
                print(func)
                has_not_handled = True
        return has_not_handled

if __name__ == "__main__":
    with NVMLApiRecorder() as recorder:
        recorder.record("out.yaml")
