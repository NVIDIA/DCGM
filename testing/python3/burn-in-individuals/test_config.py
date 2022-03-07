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
import sys
sys.path.insert(0,'..')
import dcgm_agent
import dcgm_agent_internal
import dcgm_structs
import dcgm_fields
import dcgmvalue
import dcgm_structs_internal
from ctypes import *
import time

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



def helper_verify_config_values_standalone(handle, groupId, expected_power, expected_ecc, \
                                            expected_proc_clock, expected_mem_clock, expected_compute_mode, \
                                            expected_sync_boost, expected_auto_boost):
    """
    Helper Method to verify all the values for the current configuration are as expected
    """

    groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId, dcgm_structs.c_dcgmGroupInfo_version2)
    status_handle = dcgm_agent.dcgmStatusCreate()
    
    config_values = dcgm_agent.dcgmConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle)
    assert len(config_values) > 0, "Failed to get configuration using dcgmConfigGet"

    for x in range(0,groupInfo.count):
            assert config_values[x].mPowerLimit.type == dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL, \
                                    "The power limit type for gpuId %d is incorrect. Returned: %d Expected :%d" \
                                    % (x, config_values[x].mPowerLimit.type, dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL)
            assert config_values[x].mPowerLimit.val == expected_power, "The power limit value for gpuID %d is incorrect. Returned: %d Expected: %d" \
                                    % (x, config_values[x].mPowerLimit.val, expected_power)

            assert config_values[x].mPerfState.syncBoost == expected_sync_boost, "The syncboost value for gpuID %d is incorrect."\
                                    " Returned: %d Expected: %d" \
                                    % (x, config_values[x].mPerfState.syncBoost, expected_sync_boost)
                                    
            assert config_values[x].mPerfState.autoBoost == expected_auto_boost, "The autoboost value for gpuID %d is incorrect."\
                                    " Returned: %d Expected: %d" \
                                    % (x, config_values[x].mPerfState.autoBoost, expected_auto_boost)
                                    
            assert config_values[x].mPerfState.minVPState.memClk == expected_mem_clock, "The min mem clock value for gpuID %d is incorrect."\
                                    " Returned: %d Expected: %d" \
                                    % (x, config_values.mPerfState.minVPState.memClk , expected_mem_clock)
                                    
            assert config_values[x].mPerfState.minVPState.procClk  == expected_proc_clock, "The min proc clock value for gpuID %d is incorrect."\
                                    " Returned: %d Expected: %d" \
                                    % (x, config_values[x].mPerfState.minVPState.procClk , expected_proc_clock)
                                    
            assert config_values[x].mComputeMode  == expected_compute_mode, "The compute mode value for gpuID %d is incorrect."\
                                    " Returned: %d Expected: %d" \
                                    % (x, config_values[x].mComputeMode, expected_compute_mode)

            assert config_values[x].mEccMode  == expected_ecc, "The ecc mode value for gpuID %d is incorrect."\
                                    " Returned: %d Expected: %d" \
                                    % (x, config_values[x].mEccMode, expected_ecc)
            pass
    
    ret = dcgm_agent.dcgmStatusDestroy(status_handle)
    assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to remove status handler, error: %s" % ret


dcgm_structs._LoadDcgmLibrary()
handle = dcgm_agent.dcgmInit()

devices = dcgm_agent.dcgmGetAllDevices(handle)
validDevices = list()
for x in devices:
    fvSupported = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, x, [dcgm_fields.DCGM_FI_DEV_RETIRED_DBE, ])
    if (fvSupported[0].value.i64 != dcgmvalue.DCGM_INT64_NOT_SUPPORTED):
        validDevices.append(x)

if (len(validDevices) == 0):
    print("Can only run if at least one GPU with ECC is present")
    sys.exit(1)
    
print("Number of valid devices: %d" % len(validDevices))

groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_EMPTY, "test1")
statusHandle = dcgm_agent.dcgmStatusCreate()


for device in validDevices:
    ret = dcgm_agent.dcgmGroupAddDevice(handle, groupId, device)
    assert (ret == dcgm_structs.DCGM_ST_OK)


## Get attributes for all the devices
attributesForDevices = list()
for device in validDevices:
    attributes = dcgm_agent.dcgmGetDeviceAttributes(handle, device)
    attributesForDevices.append(attributes)

assert len(attributesForDevices) != 0, "Can't get attributes for all the devices"

device0_name = attributesForDevices[0].identifiers.deviceName
for attribute in attributesForDevices:
    if attribute.identifiers.deviceName != device0_name:
        print("Can only run test if all the GPUs are same")
        sys.exit(1)
        

powerLimit_set = dcgmvalue.DCGM_INT32_BLANK
fvSupported = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, x, [dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT, ])
if (fvSupported[0].value.i64 != dcgmvalue.DCGM_INT64_NOT_SUPPORTED):
    powerLimit_set = (attributesForDevices[0].powerLimits.maxPowerLimit + attributesForDevices[0].powerLimits.minPowerLimit)/2
    print("configure power limit")

autoBoost_set = dcgmvalue.DCGM_INT32_BLANK
fvSupported = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, x, [dcgm_fields.DCGM_FI_DEV_AUTOBOOST, ])
if (fvSupported[0].value.i64 != dcgmvalue.DCGM_INT64_NOT_SUPPORTED):
    autoBoost_set = 1
    print("configure autobost")

assert attributesForDevices[0].vpStates.count > 0, "Can't find clocks for the device"
total_clocks = attributesForDevices[0].vpStates.count
proc_clk_set = attributesForDevices[0].vpStates.vpState[total_clocks/2].procClk
mem_clk_set = attributesForDevices[0].vpStates.vpState[total_clocks/2].memClk


## Always Switch the ecc mode
ecc_set = 1
groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId, dcgm_structs.c_dcgmGroupInfo_version2)
config_values = dcgm_agent.dcgmConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, 0)
assert len(config_values) > 0, "Failed to work with NULL status handle"
eccmodeOnGroupExisting = config_values[0].mEccMode

if eccmodeOnGroupExisting == 0:
    ecc_set = 1
else:
    ecc_set = 0

syncboost_set = 1
compute_set = dcgm_structs.DCGM_CONFIG_COMPUTEMODE_DEFAULT

config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
config_values.mEccMode = ecc_set
config_values.mPerfState.syncBoost =  syncboost_set
config_values.mPerfState.autoBoost =  autoBoost_set
config_values.mPerfState.minVPState.memClk =  mem_clk_set
config_values.mPerfState.minVPState.procClk = proc_clk_set
config_values.mPerfState.maxVPState.memClk =  mem_clk_set
config_values.mPerfState.maxVPState.procClk = proc_clk_set
config_values.mComputeMode = compute_set
config_values.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL
config_values.mPowerLimit.val = powerLimit_set

## Set Config and verify the value
status_handle = dcgm_agent.dcgmStatusCreate()
ret = dcgm_agent.dcgmConfigSet(handle, groupId, config_values, statusHandle)
errors = helper_get_status_list(status_handle)

ecc_to_verify = ecc_set
if len(errors) > 0:
    ## Possible that reset failed. Check the error codes
    for error in errors:
        if error.fieldId == dcgm_fields.DCGM_FI_DEV_ECC_CURRENT:
            ecc_to_verify = eccmodeOnGroupExisting

#assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to set configuration for the group: %s" % ret
dcgm_agent.dcgmStatusClear(statusHandle)
helper_verify_config_values_standalone(handle, groupId, powerLimit_set, ecc_to_verify, proc_clk_set, mem_clk_set, compute_set, syncboost_set, autoBoost_set)

print("Verification Successful")

ret = dcgm_agent.dcgmGroupDestroy(handle, groupId)
assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to remove the test group, error: %s" % ret 

ret = dcgm_agent.dcgmStatusDestroy(statusHandle)
assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to remove status handler, error: %s" % ret

dcgm_agent.dcgmShutdown()
