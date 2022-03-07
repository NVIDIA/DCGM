# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
# test the policy manager for DCGM

import pydcgm
import dcgm_structs
import dcgm_agent_internal
import dcgm_structs_internal
import dcgm_agent
import logger
import test_utils
import dcgm_fields
import dcgmvalue
import time
import inspect
import apps
from subprocess import check_output


def helper_inject_vgpu_configuration(handle, gpuId, eccModeVal, powerLimitVal, computeModeVal):
    """
    Helper method to inject configuration to Cachemanager
    """
    if (eccModeVal != None):
         # inject an error into Ecc Mode
        eccMode = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        eccMode.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        eccMode.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_CURRENT
        eccMode.status = 0
        eccMode.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
        eccMode.ts = int((time.time()+1) * 1000000.0) # set the injected data into the future
        eccMode.value.i64 = eccModeVal

        ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, eccMode)
        assert (ret == dcgm_structs.DCGM_ST_OK)

    if (powerLimitVal != None):
        powerLimit = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        powerLimit.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        powerLimit.fieldId = dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT
        powerLimit.status = 0
        powerLimit.fieldType = ord(dcgm_fields.DCGM_FT_DOUBLE)
        powerLimit.ts = int((time.time()+1) * 1000000.0) # set the injected data into the future
        powerLimit.value.dbl = powerLimitVal

        ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, powerLimit)
        assert (ret == dcgm_structs.DCGM_ST_OK)


    if (computeModeVal != None):
        computeMode = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        computeMode.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        computeMode.fieldId = dcgm_fields.DCGM_FI_DEV_COMPUTE_MODE
        computeMode.status = 0
        computeMode.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
        computeMode.ts = int((time.time()+1) * 1000000.0) # set the injected data into the future
        computeMode.value.i64 = computeModeVal

        ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, computeMode)
        assert (ret == dcgm_structs.DCGM_ST_OK)

def helper_get_status_list(statusHandle):
    """
    Helper method to get status list from the provided status handle
    """
    errorList = list()

    errorInfo = dcgm_agent.dcgmStatusPopError(statusHandle)

    while (errorInfo != None):
        errorList.append(errorInfo)
        errorInfo = dcgm_agent.dcgmStatusPopError(statusHandle)

    return errorList

'''
def helper_investigate_status(statusHandle):
    """
    Helper method to investigate status handle
    """
    errorCount = 0;
    errorInfo = dcgm_agent.dcgmStatusPopError(statusHandle)

    while (errorInfo != None):
        errorCount += 1
        print errorCount
        print("  GPU Id: %d" % errorInfo.gpuId)
        print("  Field ID: %d" % errorInfo.fieldId)
        print("  Error: %d" % errorInfo.status)
        errorInfo = dcgm_agent.dcgmStatusPopError(statusHandle)
'''

@test_utils.run_with_embedded_host_engine()
def test_dcgm_vgpu_config_embedded_get_devices(handle):
    """
    Verifies that DCGM Engine returns list of devices
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert len(gpuIdList) >= 0, "Not able to find devices on the node for embedded case"

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_vgpu_config_standalone_get_devices(handle):
    """
    Verifies that DCGM Engine returns list of devices
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert	len(gpuIdList) >= 0, "Not able to find devices for standalone case"

def helper_dcgm_vgpu_config_get_attributes(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    gpuIdList = groupObj.GetGpuIds()

    for gpuId in gpuIdList:
        attributes = systemObj.discovery.GetGpuAttributes(gpuId)
        assert (attributes.identifiers.deviceName != dcgmvalue.DCGM_STR_NOT_SUPPORTED
            and attributes.identifiers.deviceName != dcgmvalue.DCGM_STR_NOT_FOUND
            and attributes.identifiers.deviceName != dcgmvalue.DCGM_STR_NOT_SUPPORTED
            and attributes.identifiers.deviceName != dcgmvalue.DCGM_STR_NOT_PERMISSIONED), "Not able to find attributes"

@test_utils.run_with_embedded_host_engine()
def test_dcgm_vgpu_config_embedded_get_attributes(handle):
    """
    Get Device attributes for each GPU ID
    """
    helper_dcgm_vgpu_config_get_attributes(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_vgpu_config_standalone_get_attributes(handle):
    """
	Get Device attributes for each GPU ID
	"""
    helper_dcgm_vgpu_config_get_attributes(handle)

'''
def helper_dcgm_vgpu_config_set(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    vgpu_config_values = dcgm_structs.c_dcgmDeviceVgpuConfig_v1()
    vgpu_config_values.SetBlank()

    #Will throw an exception on error
    groupObj.vgpu.Set(vgpu_config_values)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_set_embedded(handle):
    """
    Verifies that the configuration can be set for a group
    """
    helper_dcgm_vgpu_config_set(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_set_standalone(handle):
    """
    Verifies that the vGPU configuration can be set for a group
    """
    helper_dcgm_vgpu_config_set(handle)

def helper_dcgm_vgpu_config_get(handle):

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    ## Set the configuration first
    vgpu_config_values = dcgm_structs.c_dcgmDeviceVgpuConfig_v1()
    vgpu_config_values.SetBlank()

    #Will throw exception on error
    groupObj.vgpu.Set(vgpu_config_values)

    ## Get the target vGPU configuration to make sure that it's exact same as the one configured
    config_values = groupObj.vgpu.Get(dcgm_structs.DCGM_CONFIG_TARGET_STATE)

    gpuIds = groupObj.GetGpuIds()

    ## Loop through config_values to to check for correctness of values fetched from the hostengine
    for x in xrange(0, len(gpuIds)):
        assert config_values[x].mEccMode == dcgmvalue.DCGM_INT32_BLANK, "Failed to get matching value for ecc mode. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_BLANK, vgpu_config_values[x].mEccMode)
        assert config_values[x].mPowerLimit.val == dcgmvalue.DCGM_INT32_BLANK, "Failed to get matching value for power limit. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_BLANK, vgpu_config_values[x].mPowerLimit.val)
        assert config_values[x].mComputeMode == dcgmvalue.DCGM_INT32_BLANK, "Failed to get matching value for power limit. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_BLANK, vgpu_config_values[x].mComputeMode)
        pass

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_get_embedded(handle, gpuIds):
    """
    Verifies "Get vGPU Configuration" Basic functionality
    """
    helper_dcgm_vgpu_config_get(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_get_standalone(handle, gpuIds):
    """
    Verifies "Get vGPU Configuration" Basic functionality
    """
    helper_dcgm_vgpu_config_get(handle)

def helper_dcgm_vgpu_config_enforce(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    vgpu_config_values = dcgm_structs.c_dcgmDeviceVgpuConfig_v1()
    vgpu_config_values.SetBlank()

    #Will throw exception on error
    groupObj.vgpu.Set(vgpu_config_values)

    groupObj.vgpu.Enforce()


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_enforce_embedded(handle, gpuIds):
    """
    Verifies that the vGPU configuration can be enforced for a group
    """
    helper_dcgm_vgpu_config_enforce(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_enforce_standalone(handle, gpuIds):
    """
    Verifies that the vGPU configuration can be enforced for a group
    """
    helper_dcgm_vgpu_config_enforce(handle)

def helper_dcgm_vgpu_config_injection(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuIds = groupObj.GetGpuIds()

    ## Loop through vgpu_config_values to to check for correctness of values fetched from the hostengine
    for x in xrange(0, len(gpuIds)):
        helper_inject_vgpu_configuration(handle, gpuIds[x], dcgmvalue.DCGM_INT32_NOT_SUPPORTED,
            dcgmvalue.DCGM_FP64_NOT_SUPPORTED, dcgmvalue.DCGM_INT32_NOT_SUPPORTED)
        pass

    ## Get the target vGPU configuration to make sure that it's exact same as the one configured
    vgpu_config_values = groupObj.vgpu.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)

    assert len(vgpu_config_values) > 0, "Failed to get vGPU configuration using groupObj.vgpu.Get"

    ## Loop through vgpu_config_values to to check for correctness of values fetched from the hostengine
    for x in xrange(0, len(gpuIds)):
        assert vgpu_config_values[x].mEccMode == dcgmvalue.DCGM_INT32_NOT_SUPPORTED, "Failed to get matching value for ecc mode. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_NOT_SUPPORTED, vgpu_config_values[x].mEccMode)
        assert vgpu_config_values[x].mComputeMode == dcgmvalue.DCGM_INT32_NOT_SUPPORTED, "Failed to get matching value for compute mode. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_NOT_SUPPORTED, vgpu_config_values[x].mComputeMode)
        assert vgpu_config_values[x].mPowerLimit.val == dcgmvalue.DCGM_INT32_NOT_SUPPORTED, "Failed to get matching value for power limit. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_NOT_SUPPORTED, vgpu_config_values[x].mPowerLimit.val)
        pass


    valToInsert = 100

    ## Loop through vgpu_config_values to to check for correctness of values fetched from the hostengine
    for x in xrange(0, len(gpuIds)):
        helper_inject_vgpu_configuration(handle, gpuIds[x], valToInsert,
            valToInsert, valToInsert)
        pass

    ## Get the target vGPU configuration to make sure that it's exact same as the one configured
    vgpu_config_values = groupObj.vgpu.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)

    assert len(vgpu_config_values) > 0, "Failed to get vGPU configuration using dcgmClientVgpuConfigGet"

    ## Loop through vgpu_config_values to to check for correctness of values fetched from the hostengine
    for x in xrange(0, len(gpuIds)):
        assert vgpu_config_values[x].mEccMode == valToInsert, "Failed to get matching value for ecc mode. Expected: %d Received: %d" % (valToInsert, vgpu_config_values[x].mEccMode)
        assert vgpu_config_values[x].mComputeMode == valToInsert, "Failed to get matching value for Compute mode. Expected: %d Received: %d" % (valToInsert, vgpu_config_values[x].mComputeMode)
        assert vgpu_config_values[x].mPowerLimit.val == valToInsert, "Failed to get matching value for power limit. Expected: %d Received: %d" % (valToInsert, vgpu_config_values[x].mPowerLimit.val)
        pass

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_injection_embedded(handle, gpuIds):
    """
    Injects values to the Cache manager and verifies if Config Manager can fetch those values
    """
    helper_dcgm_vgpu_config_injection(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_injection_standalone(handle, gpuIds):
    """
    Injects values to the Cache manager and verifies if Config Manager can fetch those values
    """
    helper_dcgm_vgpu_config_injection(handle)

def helper_dcgm_vgpu_config_powerbudget(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    ## Add first GPU to the group
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against

    ## Get Min and Max Power limit on the group
    attributes = dcgm_agent.dcgmGetDeviceAttributes(handle, gpuIds[0])

    ## Verify that power is supported on the GPUs in the group
    if dcgmvalue.DCGM_INT32_IS_BLANK(attributes.powerLimits.maxPowerLimit):
        test_utils.skip_test("Needs Power limit to be supported on the GPU")

    powerLimit = (attributes.powerLimits.maxPowerLimit + attributes.powerLimits.minPowerLimit)/2

    vgpu_config_values = dcgm_structs.c_dcgmDeviceVgpuConfig_v1()
    vgpu_config_values.SetBlank()
    vgpu_config_values.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_BUDGET_GROUP
    vgpu_config_values.mPowerLimit.val = powerLimit * len(gpuIds) #Assumes homogenous GPUs

    groupObj.vgpu.Set(vgpu_config_values)

    vgpu_config_values = groupObj.vgpu.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
    assert len(vgpu_config_values) > 0, "Failed to get vGPU configuration using groupObj.vgpu.Get"

    for x in xrange(0, len(gpuIds)):
        if (vgpu_config_values[x].mPowerLimit.val != dcgmvalue.DCGM_INT32_NOT_SUPPORTED):
            assert vgpu_config_values[x].mPowerLimit.type == dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL, "The power limit type for gpuId %d is incorrect. Returned: %d Expected :%d" % (x, vgpu_config_values[x].mPowerLimit.type, dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL)
            assert vgpu_config_values[x].mPowerLimit.val == powerLimit, "The power limit value for gpuID %d is incorrect. Returned: %d Expected: %s" % (x, vgpu_config_values[x].mPowerLimit.val, powerLimit)
        pass

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_powerbudget_embedded(handle, gpuIds):
    """
    This method verfies setting power budget for a group of GPUs
    """
    helper_dcgm_vgpu_config_powerbudget(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_powerbudget_standalone(handle, gpuIds):
    """
    This method verfies setting power budget for a group of GPUs
    """
    helper_dcgm_vgpu_config_powerbudget(handle, gpuIds)

def helper_verify_power_value_standalone(groupObj, expected_power):
    """
    Helper Method to verify power value
    """
    gpuIds = groupObj.GetGpuIds()
    vgpu_config_values = groupObj.vgpu.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
    assert len(vgpu_config_values) > 0, "Failed to get vGPU configuration using dcgmClientVgpuConfigGet"

    for x in xrange(0, len(gpuIds)):
        if (vgpu_config_values[x].mPowerLimit.val != dcgmvalue.DCGM_INT32_NOT_SUPPORTED):
            assert vgpu_config_values[x].mPowerLimit.type == dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL, \
                                    "The power limit type for gpuId %d is incorrect. Returned: %d Expected :%d" \
                                    % (x, vgpu_config_values[x].mPowerLimit.type, dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL)
            assert vgpu_config_values[x].mPowerLimit.val == expected_power, "The power limit value for gpuID %d is incorrect. Returned: %d Expected: %d" \
                                    % (x, vgpu_config_values[x].mPowerLimit.val, expected_power)
        pass

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_config_power_enforce_standalone(handle, gpuIds):
    """
    Checks if DCGM can enforce the power settings if it's changed behind the scenes
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    ## Add first GPU to the group
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against
    gpuId = gpuIds[0]

    #Make sure that the power management limit is updating fast enough to look at
    fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handleObj.handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT)
    sleepTime = 1.2 * (fieldInfo.monitorFrequencyUsec / 1000000.0)

    ## Get Min and Max Power limit on the group
    attributes = systemObj.discovery.GetGpuAttributes(gpuId)

    ## Verify that power is supported on the GPUs in the group
    if dcgmvalue.DCGM_INT32_IS_BLANK(attributes.powerLimits.maxPowerLimit):
        test_utils.skip_test("Needs Power limit to be supported on the GPU")

    powerLimit_set_dcgmi = (attributes.powerLimits.maxPowerLimit + attributes.powerLimits.minPowerLimit)/2
    powerLimit_set_nvsmi = attributes.powerLimits.maxPowerLimit

    vgpu_config_values = dcgm_structs.c_dcgmDeviceVgpuConfig_v1()
    vgpu_config_values.SetBlank()
    vgpu_config_values.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL
    vgpu_config_values.mPowerLimit.val = powerLimit_set_dcgmi

    groupObj.vgpu.Set(vgpu_config_values)

    logger.info("Verify if dcgmi configured value has taken effect")
    helper_verify_power_value_standalone(groupObj, powerLimit_set_dcgmi)

    ## Change Power limit to max from external entity like nvidia-smi
    assert 0 == apps.NvidiaSmiApp(["-pl", str(powerLimit_set_nvsmi), "-i", str(gpuIds[0])]).run(), \
        "Nvidia smi couldn't set the power limit"

    systemObj.UpdateAllFields(1)

    logger.info("Sleeping for %f seconds to allow the power limit to update in the cache" % sleepTime)
    time.sleep(sleepTime)

    logger.info("Verify if nvsmi configured value has taken effect")
    helper_verify_power_value_standalone(groupObj, powerLimit_set_nvsmi)

    groupObj.vgpu.Enforce()


    logger.info("Verify if dcgmi enforced value has taken effect")
    helper_verify_power_value_standalone(groupObj, powerLimit_set_dcgmi)

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_default_status_handler(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    ## Add first GPU to the group
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against

    vgpu_config_values = dcgm_structs.c_dcgmDeviceVgpuConfig_v1()
    vgpu_config_values.SetBlank()

    groupObj.vgpu.Set(vgpu_config_values)

    vgpu_config_values = groupObj.vgpu.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
    assert len(vgpu_config_values) > 0, "Failed to work with NULL status handle"

    vgpu_config_values = groupObj.vgpu.Enforce()

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED
    newPolicy.parms[2].tag = 1
    newPolicy.parms[2].val.llval = 5

    ret = dcgm_agent.dcgmPolicySet(handle, groupObj.GetId(), newPolicy, 0)
    assert (ret != dcgm_structs.DCGM_ST_BADPARAM), "Failed to work with NULL status handle: %d" % ret

    policy = dcgm_agent.dcgmPolicyGet(handle, groupObj.GetId(), len(gpuIds), 0)
'''

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_configure_ecc_mode(handle, gpuIds):
    test_utils.skip_test("Skipping this test until bug 200377294 is fixed")

    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_EMPTY, "test1")

    validDevice = -1
    for x in gpuIds:
        fvSupported = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, x, [dcgm_fields.DCGM_FI_DEV_RETIRED_DBE])
        if (fvSupported[0].value.i64 != dcgmvalue.DCGM_INT64_NOT_SUPPORTED):
            validDevice = x
        break

    if (validDevice == -1):
        test_utils.skip_test("Can only run if at least one GPU with ECC is present")

    ret = dcgm_agent.dcgmGroupAddDevice(handle, groupId, validDevice)
    assert (ret == dcgm_structs.DCGM_ST_OK),"Failed to add a device to the group %d. Return %d" % (groupId.value, ret)

    groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId)

    #Create a status handle
    status_handle = dcgm_agent.dcgmStatusCreate()

    ## Get original ECC mode on the device
    vgpu_config_values = dcgm_agent_internal.dcgmVgpuConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle)
    assert len(vgpu_config_values) > 0, "Failed to work with NULL status handle"

    eccmodeOnGroupExisting = vgpu_config_values[0].mEccMode
    if eccmodeOnGroupExisting == 0:
        eccmodeOnGroupToSet = 1
    else:
        eccmodeOnGroupToSet = 0

    #print eccmodeOnGroupExisting
    #print eccmodeOnGroupToSet

    ## Toggle the ECC mode on the group
    vgpu_config_values = dcgm_structs.c_dcgmDeviceVgpuConfig_v1()
    vgpu_config_values.mEccMode = eccmodeOnGroupToSet
    vgpu_config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
    vgpu_config_values.mPowerLimit.type = dcgmvalue.DCGM_INT32_BLANK
    vgpu_config_values.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK




    #Clear the status handle to log the errors while setting the config
    ret = dcgm_agent.dcgmStatusClear(status_handle)
    assert ret == dcgm_structs.DCGM_ST_OK, "Failed to clear the status handle. Return %d" %ret

    try:
        ret = dcgm_agent_internal.dcgmVgpuConfigSet(handle, groupId, vgpu_config_values, status_handle)
    except dcgm_structs.DCGMError as e:
        pass

    errors = helper_get_status_list(status_handle)

    if len(errors) > 0:
        for error in errors:
            if error.status == dcgm_structs.DCGM_ST_RESET_REQUIRED:
                test_utils.skip_test("Skipping the test - Unable to reset the Gpu, FieldId - %d, Return - %d" % (error.fieldId, error.status))
            else:
                test_utils.skip_test("Skipping the test - Unable to set the ECC mode. FieldId - %d, Return %d" % (error.fieldId,error.status))

    #Sleep after reset and then apply update for it to occur
    time.sleep(2)

    dcgm_agent.dcgmUpdateAllFields(handle, 1)
    
    #Clear the status handle to log the errors while setting the config
    ret = dcgm_agent.dcgmStatusClear(status_handle)
    assert ret == dcgm_structs.DCGM_ST_OK, "Failed to clear the status handle. Return %d" %ret

    #Get the current configuration
    config_values = dcgm_agent_internal.dcgmVgpuConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle)
    assert len(config_values) > 0, "Failed to get configuration using dcgmiVgpuConfigGet"

    assert config_values[0].mEccMode == (eccmodeOnGroupToSet), "ECC mode different from the set value"

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_vgpu_attributes(handle, gpuIds):
    """
    Verifies that vGPU attributes are properly queried
    """
    vgpuAttributes = dcgm_agent_internal.dcgmGetVgpuDeviceAttributes(handle, gpuIds[0])
    assert  vgpuAttributes.activeVgpuInstanceCount >= 0, "Active vGPU instance count is negative!"

    if (vgpuAttributes.activeVgpuInstanceCount > 0):
        vgpuInstanceAttributes = dcgm_agent_internal.dcgmGetVgpuInstanceAttributes(handle, vgpuAttributes.activeVgpuInstanceIds[0])
        assert  len(vgpuInstanceAttributes.vmName) > 0, "Active vGPU VM name is blank!"


