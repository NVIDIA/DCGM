# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import ctypes
import dcgm_structs
import dcgm_agent_internal
import dcgmvalue
import pydcgm
import test_utils
import dcgm_agent
import nvml_injection
import nvml_injection_structs
import dcgm_nvml
from _test_helpers import skip_test_if_no_dcgm_nvml
from dcgm_structs import dcgmExceptionClass


def helper_get_blank_dcgm_config_for_workload_power_profiles():
    config_values = dcgm_structs.c_dcgmDeviceConfig_v2()
    config_values.version = dcgm_structs.dcgmDeviceConfig_version2
    config_values.gpuId = dcgmvalue.DCGM_INT32_BLANK
    config_values.mEccMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.memClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK

    return config_values


def helper_verify_target_config(groupObj, expectedMask):
    # Get the target configuration
    getConfigValues = groupObj.config.Get(
        dcgm_structs.DCGM_CONFIG_TARGET_STATE)
    assert len(
        getConfigValues) > 0, "Failed to get configuration using dcgmConfigGet"
    numGpuIds = len(groupObj.GetGpuIds())

    # Verify that the target power profile matches the new mask.
    for gpuId in range(numGpuIds):
        for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
            assert getConfigValues[gpuId].mWorkloadPowerProfiles[bitmapIndex] == expectedMask[bitmapIndex], \
                f"Workload power profile at index {bitmapIndex} is {getConfigValues[gpuId].mWorkloadPowerProfiles[bitmapIndex]}, expected {expectedMask[bitmapIndex]}"


def helper_verify_current_config(groupObj, expectedMask):
    # Get the current configuration
    getConfigValues = groupObj.config.Get(
        dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
    assert len(
        getConfigValues) > 0, "Failed to get configuration using dcgmConfigGet"
    numGpuIds = len(groupObj.GetGpuIds())

    # Verify that the current power profile matches the expected mask.
    for gpuId in range(numGpuIds):
        for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
            assert getConfigValues[gpuId].mWorkloadPowerProfiles[bitmapIndex] == expectedMask[bitmapIndex], \
                f"Workload power profile at index {bitmapIndex} is {getConfigValues[gpuId].mWorkloadPowerProfiles[bitmapIndex]}, expected {expectedMask[bitmapIndex]}"


def helper_initialize_config(initialMask):
    setConfigValues = helper_get_blank_dcgm_config_for_workload_power_profiles()
    for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
        setConfigValues.mWorkloadPowerProfiles[bitmapIndex] = initialMask[bitmapIndex]
    return setConfigValues


def helper_set_workload_power_profiles_with_new_nvml_api(handle, gpuIds, setMechanism):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds("testGroup", gpuIds)
    groupId = groupObj.GetId()

    # First get the target configuration and verify the initial state in DCGM is all blanks.
    blankMask = [dcgmvalue.DCGM_INT32_BLANK] * \
        dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE
    helper_verify_target_config(groupObj, blankMask)

    workloadPowerProfiles = [0] * \
        dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE

    setInjectedRet = nvml_injection.c_injectNvmlRet_t()
    setInjectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    setInjectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_WORKLOADPOWERPROFILEUPDATEPROFILES_V1
    setInjectedRet.values[0].value.WorkloadPowerProfileUpdateProfiles_v1.operation = 0
    nvmlMask255 = dcgm_nvml.c_nvmlMask255_t()
    mask = [0, 0, 0, 0, 0, 0, 0, 0]
    nvmlMask255.mask = (ctypes.c_uint * len(mask))(*mask)
    setInjectedRet.values[0].value.WorkloadPowerProfileUpdateProfiles_v1.updateProfilesMask = nvmlMask255
    setInjectedRet.valueCount = 1
    for gpuId in gpuIds:
        dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "WorkloadPowerProfileUpdateProfiles", None, 0, setInjectedRet)

    # Before setting the config, current GPU config is read from NVML using WorkloadPowerProfileGetCurrentProfiles.
    # WorkloadPowerProfileGetCurrentProfiles is not injected here, and will return an error, which is
    # translated to a mask of blanks in the current config. This works for the initial configuration.
    if setMechanism == "dcgmConfigSet":
        # Initialize the new config to all blanks and update only the workload power profiles to all 0s.
        setConfigValues = helper_get_blank_dcgm_config_for_workload_power_profiles()
        for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
            setConfigValues.mWorkloadPowerProfiles[bitmapIndex] = workloadPowerProfiles[bitmapIndex]
        groupObj.config.Set(setConfigValues)
    elif setMechanism == "dcgmConfigSetWorkloadPowerProfile":
        workloadPowerProfile = dcgm_structs.c_dcgmWorkloadPowerProfile_t()
        workloadPowerProfile.version = dcgm_structs.dcgmWorkloadPowerProfile_version
        workloadPowerProfile.groupId = groupId
        workloadPowerProfile.profileMask = (ctypes.c_uint * len(mask))(*mask)
        dcgm_agent.dcgmConfigSetWorkloadPowerProfile(
            handle, workloadPowerProfile)

    # Inject the current GPU config and verify that it is read correctly, along with the saved target config in DCGM.
    getInjectedRet = nvml_injection.c_injectNvmlRet_t()
    getInjectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    getInjectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_WORKLOADPOWERPROFILECURRENTPROFILES
    getInjectedRet.values[0].value.WorkloadPowerProfileCurrentProfiles.requestedProfilesMask = nvmlMask255
    getInjectedRet.valueCount = 1
    for gpuId in gpuIds:
        dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "WorkloadPowerProfileGetCurrentProfiles", None, 0, getInjectedRet)

    helper_verify_current_config(groupObj, workloadPowerProfiles)

    helper_verify_target_config(groupObj, workloadPowerProfiles)

    # Saved target config is now all 0s.
    # Set a new config mask and verify that target config is successfully merged.

    newMask = [0, 1, 1, 0, 0, 1, 0, 0]

    if setMechanism == "dcgmConfigSet":
        for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
            setConfigValues.mWorkloadPowerProfiles[bitmapIndex] = newMask[bitmapIndex]
        groupObj.config.Set(setConfigValues)
    elif setMechanism == "dcgmConfigSetWorkloadPowerProfile":
        workloadPowerProfile.profileMask = (
            ctypes.c_uint * len(newMask))(*newMask)
        dcgm_agent.dcgmConfigSetWorkloadPowerProfile(
            handle, workloadPowerProfile)

    helper_verify_target_config(groupObj, newMask)

    groupObj.Delete()


def helper_set_workload_power_profiles_with_old_nvml_api(handle, gpuIds, setMechanism):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds("testGroup", gpuIds)
    groupId = groupObj.GetId()

    # First get the target configuration and verify the initial state in DCGM is all blanks.
    blankMask = [dcgmvalue.DCGM_INT32_BLANK] * \
        dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE
    helper_verify_target_config(groupObj, blankMask)

    # Initialize the new config to all blanks and update only the workload power profiles to all 0s.
    zeroMask = [0] * dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE
    setConfigValues = helper_initialize_config(zeroMask)

    # Inject a NVML_ERROR_NOT_SUPPORTED return value for WorkloadPowerProfileUpdateProfiles.
    # This is to ensure that we fall back to the old API.
    updateProfileInjectedRet = nvml_injection.c_injectNvmlRet_t()
    updateProfileInjectedRet.nvmlRet = dcgm_nvml.NVML_ERROR_NOT_SUPPORTED
    for gpuId in gpuIds:
        dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "WorkloadPowerProfileUpdateProfiles", None, 0, updateProfileInjectedRet)

    # Inject a NVML_SUCCESS return value for WorkloadPowerProfileClearRequestedProfiles.
    # Clear profiles will be called with dcgmConfigSet because all profiles are set to 0.
    clearRequestedProfileInjectedRet = nvml_injection.c_injectNvmlRet_t()
    clearRequestedProfileInjectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    clearRequestedProfileInjectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_WORKLOADPOWERPROFILEREQUESTEDPROFILES
    mask = [0, 0, 0, 0, 0, 0, 0, 0]
    nvmlMask255 = dcgm_nvml.c_nvmlMask255_t()
    nvmlMask255.mask = (ctypes.c_uint * len(mask))(*mask)
    # irrelevant. Just needs to be set.
    clearRequestedProfileInjectedRet.values[0].value.WorkloadPowerProfileRequestedProfiles.requestedProfilesMask = nvmlMask255
    clearRequestedProfileInjectedRet.valueCount = 1

    for gpuId in gpuIds:
        dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "WorkloadPowerProfileClearRequestedProfiles", None, 0, clearRequestedProfileInjectedRet)
        # Inject a NVML_SUCCESS return value for WorkloadPowerProfileSetRequestedProfiles.
        # This will be called with dcgmConfigSetWorkloadPowerProfile.
        setRequestedProfileInjectedRet = clearRequestedProfileInjectedRet
        dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "WorkloadPowerProfileSetRequestedProfiles", None, 0, setRequestedProfileInjectedRet)

    # Before setting the config, current GPU config is read from NVML using WorkloadPowerProfileGetCurrentProfiles.
    # WorkloadPowerProfileGetCurrentProfiles is not injected here, and will return an error, which is
    # translated to a mask of blanks in the current config. This works for the initial configuration.
    if setMechanism == "dcgmConfigSet":
        # Initialize the new config to all blanks and update only the workload power profiles to all 0s.
        setConfigValues = helper_get_blank_dcgm_config_for_workload_power_profiles()
        for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
            setConfigValues.mWorkloadPowerProfiles[bitmapIndex] = zeroMask[bitmapIndex]
        groupObj.config.Set(setConfigValues)
    elif setMechanism == "dcgmConfigSetWorkloadPowerProfile":
        workloadPowerProfile = dcgm_structs.c_dcgmWorkloadPowerProfile_t()
        workloadPowerProfile.version = dcgm_structs.dcgmWorkloadPowerProfile_version
        workloadPowerProfile.groupId = groupId
        workloadPowerProfile.profileMask = (ctypes.c_uint * len(mask))(*mask)
        dcgm_agent.dcgmConfigSetWorkloadPowerProfile(
            handle, workloadPowerProfile)

    # Inject the current GPU config and verify that it is read correctly
    getInjectedRet = nvml_injection.c_injectNvmlRet_t()
    getInjectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    getInjectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_WORKLOADPOWERPROFILECURRENTPROFILES
    getInjectedRet.values[0].value.WorkloadPowerProfileCurrentProfiles.requestedProfilesMask = nvmlMask255
    getInjectedRet.valueCount = 1
    for gpuId in gpuIds:
        dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "WorkloadPowerProfileGetCurrentProfiles", None, 0, getInjectedRet)

    helper_verify_current_config(groupObj, zeroMask)

    # Verify that the target configuration is set correctly to all 0s.
    helper_verify_target_config(groupObj, zeroMask)

    # Saved target config is now all 0s.
    # Set a new config mask and verify that target config is successfully merged.
    # Since WorkloadPowerProfileGetCurrentProfiles, WorkloadPowerProfileSetRequestedProfiles and WorkloadPowerProfileUpdateProfiles
    # are already injected, we do not need to inject them again.

    newMask = [0, 1, 1, 0, 0, 1, 0, 0]

    if setMechanism == "dcgmConfigSet":
        for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
            setConfigValues.mWorkloadPowerProfiles[bitmapIndex] = newMask[bitmapIndex]
        groupObj.config.Set(setConfigValues)
    elif setMechanism == "dcgmConfigSetWorkloadPowerProfile":
        workloadPowerProfile.profileMask = (
            ctypes.c_uint * len(newMask))(*newMask)
        dcgm_agent.dcgmConfigSetWorkloadPowerProfile(
            handle, workloadPowerProfile)

    helper_verify_target_config(groupObj, newMask)

    groupObj.Delete()


def helper_set_workload_power_profiles_error_with_new_nvml_api(handle, gpuIds, setMechanism):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds("testGroup", gpuIds)
    groupId = groupObj.GetId()

    # Initialize the new config to all blanks and update only the workload power profiles.
    newMask = [0, 1, 1, 0, 0, 1, 0, 0]
    setConfigValues = helper_initialize_config(newMask)

    setInjectedRet = nvml_injection.c_injectNvmlRet_t()
    setInjectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    setInjectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_WORKLOADPOWERPROFILEUPDATEPROFILES_V1
    setInjectedRet.values[0].value.WorkloadPowerProfileUpdateProfiles_v1.operation = 0
    nvmlMask255 = dcgm_nvml.c_nvmlMask255_t()
    nvmlMask255.mask = (ctypes.c_uint * len(newMask))(*newMask)
    setInjectedRet.values[0].value.WorkloadPowerProfileUpdateProfiles_v1.updateProfilesMask = nvmlMask255
    setInjectedRet.valueCount = 1

    # Inject an NVML error return value for WorkloadPowerProfileUpdateProfiles on the last GPU.
    setInjectedRetError = setInjectedRet
    setInjectedRetError.nvmlRet = dcgm_nvml.NVML_ERROR_INVALID_STATE
    errorGpuId = gpuIds[-1]
    for gpuId in gpuIds:
        if gpuId == errorGpuId:
            dcgm_agent_internal.dcgmInjectNvmlDevice(
                handle, gpuId, "WorkloadPowerProfileUpdateProfiles", None, 0, setInjectedRetError)
        else:
            dcgm_agent_internal.dcgmInjectNvmlDevice(
                handle, gpuId, "WorkloadPowerProfileUpdateProfiles", None, 0, setInjectedRet)

    if setMechanism == "dcgmConfigSet":
        with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NVML_ERROR)):
            groupObj.config.Set(setConfigValues)
    elif setMechanism == "dcgmConfigSetWorkloadPowerProfile":
        workloadPowerProfile = dcgm_structs.c_dcgmWorkloadPowerProfile_t()
        workloadPowerProfile.version = dcgm_structs.dcgmWorkloadPowerProfile_version
        workloadPowerProfile.groupId = groupId
        workloadPowerProfile.profileMask = (
            ctypes.c_uint * len(newMask))(*newMask)
        with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NVML_ERROR)):
            dcgm_agent.dcgmConfigSetWorkloadPowerProfile(
                handle, workloadPowerProfile)

    groupObj.Delete()


def helper_set_workload_power_profiles_error_with_old_nvml_api(handle, gpuIds, setMechanism):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds("testGroup", gpuIds)
    groupId = groupObj.GetId()

    # Initialize the new config to all blanks and update only the workload power profiles.
    newMask = [0, 1, 1, 0, 0, 1, 0, 0]
    setConfigValues = helper_initialize_config(newMask)

    # Inject a NVML_ERROR_NOT_SUPPORTED return value for WorkloadPowerProfileUpdateProfiles.
    # This is to ensure that we fall back to the old API.
    updateProfileInjectedRet = nvml_injection.c_injectNvmlRet_t()
    updateProfileInjectedRet.nvmlRet = dcgm_nvml.NVML_ERROR_NOT_SUPPORTED
    for gpuId in gpuIds:
        dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "WorkloadPowerProfileUpdateProfiles", None, 0, updateProfileInjectedRet)

    setRequestedProfileInjectedRet = nvml_injection.c_injectNvmlRet_t()
    setRequestedProfileInjectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    setRequestedProfileInjectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_WORKLOADPOWERPROFILEREQUESTEDPROFILES
    nvmlMask255 = dcgm_nvml.c_nvmlMask255_t()
    nvmlMask255.mask = (ctypes.c_uint * len(newMask))(*newMask)
    setRequestedProfileInjectedRet.values[0].value.WorkloadPowerProfileRequestedProfiles.requestedProfilesMask = nvmlMask255
    setRequestedProfileInjectedRet.valueCount = 1

    # Inject an NVML error return value for WorkloadPowerProfileSetRequestedProfiles on one of the GPUs.
    setRequestedProfileInjectedRetError = setRequestedProfileInjectedRet
    setRequestedProfileInjectedRetError.nvmlRet = dcgm_nvml.NVML_ERROR_INVALID_STATE
    # Set error on the last GPU.
    errorGpuId = gpuIds[-1]
    for gpuId in gpuIds:
        if gpuId == errorGpuId:
            dcgm_agent_internal.dcgmInjectNvmlDevice(
                handle, gpuId, "WorkloadPowerProfileSetRequestedProfiles", None, 0, setRequestedProfileInjectedRetError)
        else:
            dcgm_agent_internal.dcgmInjectNvmlDevice(
                handle, gpuId, "WorkloadPowerProfileSetRequestedProfiles", None, 0, setRequestedProfileInjectedRet)

    if setMechanism == "dcgmConfigSet":
        with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NVML_ERROR)):
            groupObj.config.Set(setConfigValues)
    elif setMechanism == "dcgmConfigSetWorkloadPowerProfile":
        workloadPowerProfile = dcgm_structs.c_dcgmWorkloadPowerProfile_t()
        workloadPowerProfile.version = dcgm_structs.dcgmWorkloadPowerProfile_version
        workloadPowerProfile.groupId = groupId
        workloadPowerProfile.profileMask = (
            ctypes.c_uint * len(newMask))(*newMask)
        with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NVML_ERROR)):
            dcgm_agent.dcgmConfigSetWorkloadPowerProfile(
                handle, workloadPowerProfile)

    groupObj.Delete()


def helper_verify_profile_merged_with_target_config(handle, gpuIds, setMechanism):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds("testGroup", gpuIds)
    groupId = groupObj.GetId()

    # First get the target configuration and verify the initial state in DCGM is all blanks.
    blankMask = [dcgmvalue.DCGM_INT32_BLANK] * \
        dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE
    helper_verify_target_config(groupObj, blankMask)

    # Inject current config with a different mask. This is to establish a difference in target config and current config.
    nvmlMask255 = dcgm_nvml.c_nvmlMask255_t()
    mask = [0, 0, 1, 0, 1, 0, 0, 0]
    nvmlMask255.mask = (ctypes.c_uint * len(mask))(*mask)
    getInjectedRet = nvml_injection.c_injectNvmlRet_t()
    getInjectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    getInjectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_WORKLOADPOWERPROFILECURRENTPROFILES
    getInjectedRet.values[0].value.WorkloadPowerProfileCurrentProfiles.requestedProfilesMask = nvmlMask255
    getInjectedRet.valueCount = 1
    for gpuId in gpuIds:
        dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "WorkloadPowerProfileGetCurrentProfiles", None, 0, getInjectedRet)

    if setMechanism == "dcgmConfigSet":
        setConfigValues = helper_get_blank_dcgm_config_for_workload_power_profiles()
        for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
            setConfigValues.mWorkloadPowerProfiles[bitmapIndex] = mask[bitmapIndex]
        groupObj.config.Set(setConfigValues)
    elif setMechanism == "dcgmConfigSetWorkloadPowerProfile":
        setInjectedRet = nvml_injection.c_injectNvmlRet_t()
        setInjectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
        setInjectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_WORKLOADPOWERPROFILEUPDATEPROFILES_V1
        setInjectedRet.values[0].value.WorkloadPowerProfileUpdateProfiles_v1.operation = 0
        setInjectedRet.values[0].value.WorkloadPowerProfileUpdateProfiles_v1.updateProfilesMask = nvmlMask255
        setInjectedRet.valueCount = 1
        for gpuId in gpuIds:
            dcgm_agent_internal.dcgmInjectNvmlDevice(
                handle, gpuId, "WorkloadPowerProfileUpdateProfiles", None, 0, setInjectedRet)

        workloadPowerProfile = dcgm_structs.c_dcgmWorkloadPowerProfile_t()
        workloadPowerProfile.version = dcgm_structs.dcgmWorkloadPowerProfile_version
        workloadPowerProfile.groupId = groupId
        workloadPowerProfile.profileMask = (ctypes.c_uint * len(mask))(*mask)
        dcgm_agent.dcgmConfigSetWorkloadPowerProfile(
            handle, workloadPowerProfile)

    helper_verify_target_config(groupObj, mask)

    groupObj.Delete()


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_config_set_workload_power_profiles_with_new_nvml_api(handle, gpuIds):
    """
    Verifies that the correct nvml calls are made when the new api is available,
    and that the target config is set correctly.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_set_workload_power_profiles_with_new_nvml_api(
        handle, gpuIds, "dcgmConfigSet")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_config_set_workload_power_profiles_with_old_nvml_api(handle, gpuIds):
    """
    Verifies that the correct nvml calls are made when the new api is not available,
    and that the target config is set correctly.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_set_workload_power_profiles_with_old_nvml_api(
        handle, gpuIds, "dcgmConfigSet")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_set_workload_power_profiles_api_with_new_nvml_api(handle, gpuIds):
    """
    Verifies that the correct nvml calls are made when the new api is available,
    and that the target config is set correctly.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_set_workload_power_profiles_with_new_nvml_api(
        handle, gpuIds, "dcgmConfigSetWorkloadPowerProfile")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_set_workload_power_profiles_api_with_old_nvml_api(handle, gpuIds):
    """
    Verifies that the correct nvml calls are made when the new api is available,
    and that the target config is set correctly.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_set_workload_power_profiles_with_old_nvml_api(
        handle, gpuIds, "dcgmConfigSetWorkloadPowerProfile")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_config_set_workload_power_profiles_error_with_old_nvml_api(handle, gpuIds):
    """
    Verifies that when one of the GPUs returns an error, the error is propagated back.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_set_workload_power_profiles_error_with_old_nvml_api(
        handle, gpuIds, "dcgmConfigSet")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_set_workload_power_profiles_error_with_old_nvml_api(handle, gpuIds):
    """
    Verifies that when one of the GPUs returns an error, the error is propagated back.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_set_workload_power_profiles_error_with_old_nvml_api(
        handle, gpuIds, "dcgmConfigSetWorkloadPowerProfile")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_config_set_workload_power_profiles_error_with_new_nvml_api(handle, gpuIds):
    """
    Verifies that when one of the GPUs returns an error, the error is propagated back.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_set_workload_power_profiles_error_with_new_nvml_api(
        handle, gpuIds, "dcgmConfigSet")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_set_workload_power_profiles_error_with_new_nvml_api(handle, gpuIds):
    """
    Verifies that when one of the GPUs returns an error, the error is propagated back.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_set_workload_power_profiles_error_with_new_nvml_api(
        handle, gpuIds, "dcgmConfigSetWorkloadPowerProfile")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_set_workload_power_profiles_with_new_nvml_api_invalid_group_id(handle, gpuIds):
    """
    Verifies that when the group id is invalid, the error is propagated back.
    """
    newMask = [0, 1, 1, 0, 0, 1, 0, 0]
    workloadPowerProfile = dcgm_structs.c_dcgmWorkloadPowerProfile_t()
    workloadPowerProfile.version = dcgm_structs.dcgmWorkloadPowerProfile_version
    workloadPowerProfile.groupId = 5
    workloadPowerProfile.profileMask = (ctypes.c_uint * len(newMask))(*newMask)
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        dcgm_agent.dcgmConfigSetWorkloadPowerProfile(
            handle, workloadPowerProfile)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_set_config_verify_profile_merged_with_target_config(handle, gpuIds):
    """
    Verifies that the profile is merged with the target config correctly.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_verify_profile_merged_with_target_config(
        handle, gpuIds, "dcgmConfigSet")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_set_workload_power_profile_verify_profile_merged_with_target_config(handle, gpuIds):
    """
    Verifies that the profile is merged with the target config correctly.
    """
    gpuIds = [gpuIds[0], gpuIds[1]]
    helper_verify_profile_merged_with_target_config(
        handle, gpuIds, "dcgmConfigSetWorkloadPowerProfile")
