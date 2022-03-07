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
def test_dcgm_config_embedded_get_devices(handle):
    """
    Verifies that DCGM Engine returns list of devices
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert len(gpuIdList) >= 0, "Not able to find devices on the node for embedded case"

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_config_standalone_get_devices(handle):
    """
    Verifies that DCGM Engine returns list of devices
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    gpuIdList = systemObj.discovery.GetAllGpuIds()
    assert	len(gpuIdList) >= 0, "Not able to find devices for standalone case"

def helper_dcgm_config_get_attributes(handle):
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

        #We used to assert that attributes.clockSets.count was > 0. This was because the NVML internal API that provided it
        #bypassed the SKU check. If nvidia-smi -q -d SUPPORTED_CLOCKS returns N/A, we will no longer have clockSets.
        
        for i in range(attributes.clockSets.count):
            memClock = attributes.clockSets.clockSet[i].memClock
            smClock = attributes.clockSets.clockSet[i].smClock

            assert memClock > 0 and memClock < 10000, "gpuId %d got memClock out of range 0 - 10000: %d" % (gpuId, memClock)
            assert smClock > 0 and smClock < 10000, "gpuId %d got smClock out of range 0 - 10000: %d" % (gpuId, smClock)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_config_embedded_get_attributes(handle):
    """
    Get Device attributes for each GPU ID
    """
    helper_dcgm_config_get_attributes(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_config_standalone_get_attributes(handle):
    """
	Get Device attributes for each GPU ID
	"""
    helper_dcgm_config_get_attributes(handle)

def helper_dcgm_config_set(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode =    dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost  =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.memClock  =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK

    #Will throw an exception on error
    groupObj.config.Set(config_values)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_as_root()
def test_dcgm_config_set_embedded(handle):
    """
    Verifies that the configuration can be set for a group
    """
    helper_dcgm_config_set(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_as_root()
def test_dcgm_config_set_standalone(handle):
    """
    Verifies that the configuration can be set for a group
    """
    helper_dcgm_config_set(handle)

def helper_dcgm_config_get(handle):

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    ## Set the configuration first
    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode =    dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost  =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.memClock  =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK

    #Will throw exception on error
    groupObj.config.Set(config_values)

    ## Get the target configuration to make sure that it's exact same as the one configured
    config_values = groupObj.config.Get(dcgm_structs.DCGM_CONFIG_TARGET_STATE)

    gpuIds = groupObj.GetGpuIds()

    ## Loop through config_values to to check for correctness of values fetched from the hostengine
    for x in range(0, len(gpuIds)):
        assert config_values[x].mEccMode == dcgmvalue.DCGM_INT32_BLANK, "Failed to get matching value for ecc mode. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_BLANK, config_values[x].mEccMode)
        assert config_values[x].mPerfState.targetClocks.memClock == dcgmvalue.DCGM_INT32_BLANK, "Failed to get matching value for mem app clk. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_BLANK, config_values[x].mPerfState.targetClocks.memClock)
        assert config_values[x].mPerfState.targetClocks.smClock == dcgmvalue.DCGM_INT32_BLANK, "Failed to get matching value for proc app clk. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_BLANK, config_values[x].mPerfState.targetClocks.smClock)
        assert config_values[x].mPowerLimit.val == dcgmvalue.DCGM_INT32_BLANK, "Failed to get matching value for power limit. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_BLANK, config_values[x].mPowerLimit.val)
        assert config_values[x].mComputeMode == dcgmvalue.DCGM_INT32_BLANK, "Failed to get matching value for power limit. Expected: %d Received: %d" % (dcgmvalue.DCGM_INT32_BLANK, config_values[x].mComputeMode)
        pass

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_config_get_embedded(handle, gpuIds):
    """
    Verifies "Get Configuration" Basic functionality
    """
    helper_dcgm_config_get(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_config_get_standalone(handle, gpuIds):
    """
    Verifies "Get Configuration" Basic functionality
    """
    helper_dcgm_config_get(handle)

def helper_dcgm_config_enforce(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode    =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost  =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.memClock  =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock =   dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.val =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK

    #Will throw exception on error
    groupObj.config.Set(config_values)

    groupObj.config.Enforce()


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_config_enforce_embedded(handle, gpuIds):
    """
    Verifies that the configuration can be enforced for a group
    """
    helper_dcgm_config_enforce(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_config_enforce_standalone(handle, gpuIds):
    """
    Verifies that the configuration can be enforced for a group
    """
    helper_dcgm_config_enforce(handle)

def helper_dcgm_config_powerbudget(handle, gpuIds):
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

    powerLimit = int((attributes.powerLimits.maxPowerLimit + attributes.powerLimits.minPowerLimit)/2)

    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.memClock =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_BUDGET_GROUP
    config_values.mPowerLimit.val = powerLimit * len(gpuIds) #Assumes homogenous GPUs

    groupObj.config.Set(config_values)

    config_values = groupObj.config.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
    assert len(config_values) > 0, "Failed to get configuration using groupObj.config.Get"

    for x in range(0, len(gpuIds)):
        if (config_values[x].mPowerLimit.val != dcgmvalue.DCGM_INT32_NOT_SUPPORTED):
            assert config_values[x].mPowerLimit.type == dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL, "The power limit type for gpuId %d is incorrect. Returned: %d Expected :%d" % (x, config_values[x].mPowerLimit.type, dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL)
            assert config_values[x].mPowerLimit.val == powerLimit, "The power limit value for gpuID %d is incorrect. Returned: %d Expected: %s" % (x, config_values[x].mPowerLimit.val, powerLimit)
        pass

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_config_powerbudget_embedded(handle, gpuIds):
    """
    This method verfies setting power budget for a group of GPUs
    """
    helper_dcgm_config_powerbudget(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_config_powerbudget_standalone(handle, gpuIds):
    """
    This method verfies setting power budget for a group of GPUs
    """
    helper_dcgm_config_powerbudget(handle, gpuIds)

def helper_verify_power_value(groupObj, expected_power):
    """
    Helper Method to verify power value
    """
    gpuIds = groupObj.GetGpuIds()
    config_values = groupObj.config.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
    assert len(config_values) > 0, "Failed to get configuration using dcgmClientConfigGet"

    for x in range(0, len(gpuIds)):
        if (config_values[x].mPowerLimit.val != dcgmvalue.DCGM_INT32_NOT_SUPPORTED):
            assert config_values[x].mPowerLimit.type == dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL, \
                                    "The power limit type for gpuId %d is incorrect. Returned: %d Expected :%d" \
                                    % (x, config_values[x].mPowerLimit.type, dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL)
            assert config_values[x].mPowerLimit.val == expected_power, "The power limit value for gpuID %d is incorrect. Returned: %d Expected: %d" \
                                    % (x, config_values[x].mPowerLimit.val, expected_power)
        pass

def helper_test_config_config_power_enforce(handle, gpuIds):
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

    ## Get Min and Max Power limit on the group
    attributes = systemObj.discovery.GetGpuAttributes(gpuId)

    ## Verify that power is supported on the GPUs in the group
    if dcgmvalue.DCGM_INT32_IS_BLANK(attributes.powerLimits.maxPowerLimit):
        test_utils.skip_test("Needs Power limit to be supported on the GPU")

    powerLimit_set_dcgmi = int((attributes.powerLimits.maxPowerLimit + attributes.powerLimits.minPowerLimit)/2)
    powerLimit_set_nvsmi = attributes.powerLimits.maxPowerLimit

    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.memClock =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL
    config_values.mPowerLimit.val = powerLimit_set_dcgmi

    groupObj.config.Set(config_values)

    logger.info("Verify if dcgmi configured value has taken effect")
    helper_verify_power_value(groupObj, powerLimit_set_dcgmi)

    ## Change Power limit to max from external entity like nvidia-smi
    assert 0 == apps.NvidiaSmiApp(["-pl", str(powerLimit_set_nvsmi), "-i", str(gpuIds[0])]).run(), \
        "Nvidia smi couldn't set the power limit"

    logger.info("Verify if nvsmi configured value has taken effect")
    helper_verify_power_value(groupObj, powerLimit_set_nvsmi)

    groupObj.config.Enforce()

    logger.info("Verify if dcgmi enforced value has taken effect")
    helper_verify_power_value(groupObj, powerLimit_set_dcgmi)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_config_power_enforce_embedded(handle, gpuIds):
    helper_test_config_config_power_enforce(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_config_power_enforce_standalone(handle, gpuIds):
    helper_test_config_config_power_enforce(handle, gpuIds)
    
@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_default_status_handler(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    ## Add first GPU to the group
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against
    
    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode =    dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.memClock =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL
    config_values.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK

    groupObj.config.Set(config_values)


    config_values = groupObj.config.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
    assert len(config_values) > 0, "Failed to work with NULL status handle"

    groupObj.config.Enforce()
    
    #No need to test policy set/get with default status here. this is covered by test_policy.py that passes None as the status handler
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_configure_ecc_mode(handle, gpuIds):
    test_utils.skip_test("Skipping this test until bug 200377294 is fixed")

    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_EMPTY, "test1")
    
    validDevice = -1
    for x in gpuIds:
        fvSupported = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, x, [dcgm_fields.DCGM_FI_DEV_ECC_CURRENT, ])
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
    config_values = dcgm_agent.dcgmConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle)
    assert len(config_values) > 0, "Failed to get configuration using dcgmConfigGet"
    
    eccmodeOnGroupExisting = config_values[0].mEccMode
    if eccmodeOnGroupExisting == 0:
        eccmodeOnGroupToSet = 1
    else:
        eccmodeOnGroupToSet = 0

    #print eccmodeOnGroupExisting
    #print eccmodeOnGroupToSet

    ## Toggle the ECC mode on the group
    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode = eccmodeOnGroupToSet
    config_values.mPerfState.syncBoost =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.memClock =  dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.type = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK        

    #Clear the status handle to log the errors while setting the config
    ret = dcgm_agent.dcgmStatusClear(status_handle)
    assert ret == dcgm_structs.DCGM_ST_OK, "Failed to clear the status handle. Return %d" %ret

    try:
        ret = dcgm_agent.dcgmConfigSet(handle,groupId,config_values, status_handle)
    except dcgm_structs.DCGMError as e:
        pass
        
    errors = helper_get_status_list(status_handle)

    if len(errors) > 0:
        for error in errors:
            if error.status == dcgm_structs.DCGM_ST_RESET_REQUIRED:
                test_utils.skip_test("Skipping the test - Unable to reset the Gpu, FieldId - %d, Return - %d" % (error.fieldId, error.status))
            else:
                test_utils.skip_test("Skipping the test - Unable to set the ECC mode. FieldId - %d, Return %d" % (error.fieldId,error.status))

    #Sleep after reset
    time.sleep(2)
    
    #Clear the status handle to log the errors while setting the config
    ret = dcgm_agent.dcgmStatusClear(status_handle)
    assert ret == dcgm_structs.DCGM_ST_OK, "Failed to clear the status handle. Return %d" %ret

    #Get the current configuration
    config_values = dcgm_agent.dcgmConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle)
    assert len(config_values) > 0, "Failed to get configuration using dcgmConfigGet"

    fvs = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, validDevice, [dcgm_fields.DCGM_FI_DEV_ECC_PENDING, dcgm_fields.DCGM_FI_DEV_ECC_CURRENT])
    if fvs[0].value.i64 != fvs[1].value.i64:
        logger.warning("Pending ECC %d != Current ECC %d for gpuId %d. Box probably needs a reboot" % (fvs[0].value.i64, fvs[1].value.i64, validDevice))
    else:
        assert config_values[0].mEccMode == (eccmodeOnGroupToSet), "ECC mode %d different from the set value %d" % \
                                                                   (config_values[0].mEccMode, eccmodeOnGroupToSet)
        
@test_utils.run_with_standalone_host_engine(20, ["--port", "5545"])
@test_utils.run_with_initialized_client("127.0.0.1:5545")
@test_utils.run_only_with_live_gpus()
def test_dcgm_port_standalone(handle, gpuIds):
    """
    Verifies that DCGM Engine works on different port
    """
    gpuIdList = dcgm_agent.dcgmGetAllDevices(handle) 
    assert    len(gpuIdList) >= 0, "Standalone host engine using different port number failed."

def helper_dcgm_verify_sync_boost_single_gpu(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    
    ## Add first GPU to the group
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against
        
    ## Set the sync boost for the group
    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost = 1
    config_values.mPerfState.targetClocks.memClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.type = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK

    #Config Set must return DCGM_ST_BADPARAM since we only have a single GPU
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        groupObj.config.Set(config_values)
    
    groupObj.Delete()

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_verify_sync_boost_single_gpu_standalone(handle, gpuIds):
    helper_dcgm_verify_sync_boost_single_gpu(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgm_verify_sync_boost_single_gpu_embedded(handle, gpuIds):
    helper_dcgm_verify_sync_boost_single_gpu(handle, gpuIds)


def helper_dcgm_verify_sync_boost_multi_gpu(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    
    if len(gpuIds) < 2:
        test_utils.skip_test("This test only works with 2 or more identical GPUs")

    ## Add all identical GPUs to the group
    for gpuId in gpuIds:
        groupObj.AddGpu(gpuId)
    
    gpuIds = groupObj.GetGpuIds() #Only reference GPUs we are testing against
    
    ## Set the sync boost for the group
    config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
    config_values.mEccMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.syncBoost = 1
    config_values.mPerfState.targetClocks.memClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
    config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.type = dcgmvalue.DCGM_INT32_BLANK
    config_values.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK
    
    #Enable sync boost - Will throw an exception on error
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED)):
        groupObj.config.Set(config_values)

    config_values.mPerfState.syncBoost = 0

    #Disable sync boost - Will throw an exception on error
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED)):
        groupObj.config.Set(config_values)
    
    groupObj.Delete()

@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_verify_sync_boost_multi_gpu_standalone(handle, gpuIds):
    helper_dcgm_verify_sync_boost_multi_gpu(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_verify_sync_boost_multi_gpu_embedded(handle, gpuIds):
    helper_dcgm_verify_sync_boost_multi_gpu(handle, gpuIds)
