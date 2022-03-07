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

global callbackCalled
callbackCalled = False

C_FUNC = CFUNCTYPE(None, c_void_p)


def helper_verify_power_value_standalone(handle, groupId, expected_power):
    """
    Helper Method to verify power value
    """
    groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId, dcgm_structs.c_dcgmGroupInfo_version2)
    status_handle = dcgm_agent.dcgmStatusCreate()
    
    config_values = dcgm_agent.dcgmConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle)
    assert len(config_values) > 0, "Failed to get configuration using dcgmConfigGet"

    for x in range(0,groupInfo.count):
        if (config_values[x].mPowerLimit.val != dcgmvalue.DCGM_INT32_NOT_SUPPORTED):
            assert config_values[x].mPowerLimit.type == dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL, \
                                    "The power limit type for gpuId %d is incorrect. Returned: %d Expected :%d" \
                                    % (x, config_values[x].mPowerLimit.type, dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL)
            assert config_values[x].mPowerLimit.val == expected_power, "The power limit value for gpuID %d is incorrect. Returned: %d Expected: %d" \
                                    % (x, config_values[x].mPowerLimit.val, expected_power)
        pass
    
    ret = dcgm_agent.dcgmStatusDestroy(status_handle)
    assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to remove status handler, error: %s" % ret
    


def callback_function(data):
    global callbackCalled
    callbackCalled = True

c_callback = C_FUNC(callback_function)

dcgm_structs._LoadDcgmLibrary()

newPolicy = dcgm_structs.c_dcgmPolicy_v1()
handle = dcgm_agent.dcgmInit()

newPolicy.version = dcgm_structs.dcgmPolicy_version1
newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED
newPolicy.action = dcgm_structs.DCGM_POLICY_ACTION_GPURESET
newPolicy.validation = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
newPolicy.parms[2].tag = 1
newPolicy.parms[2].val.llval = 5


# find a GPU that supports retired pages (otherwise internal test will ignore it)
devices = dcgm_agent.dcgmGetAllDevices(handle)
validDevice = -1
for x in devices:
    fvSupported = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, x, [dcgm_fields.DCGM_FI_DEV_RETIRED_DBE, ])
    if (fvSupported[0].value.i64 != dcgmvalue.DCGM_INT64_NOT_SUPPORTED):
        validDevice = x
        break

if (validDevice == -1):
    print("Can only run if at least one GPU with ECC is present")
    sys.exit(1)

groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_EMPTY, "test1")
statusHandle = dcgm_agent.dcgmStatusCreate()

ret = dcgm_agent.dcgmGroupAddDevice(handle, groupId, validDevice)
assert (ret == dcgm_structs.DCGM_ST_OK)

## Add Configuration to be different than default
## Get Min and Max Power limit on the group
attributes = dcgm_agent.dcgmGetDeviceAttributes(handle, validDevice)
    
## Verify that power is supported on the GPUs in the group
powerLimit_set = (attributes.powerLimits.maxPowerLimit + attributes.powerLimits.minPowerLimit)/2
config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
config_values.mEccMode =    dcgmvalue.DCGM_INT32_BLANK
config_values.mPerfState.syncBoost =  dcgmvalue.DCGM_INT32_BLANK
config_values.mPerfState.autoBoost =  dcgmvalue.DCGM_INT32_BLANK
config_values.mPerfState.minVPState.memClk =  dcgmvalue.DCGM_INT32_BLANK
config_values.mPerfState.minVPState.procClk = dcgmvalue.DCGM_INT32_BLANK
config_values.mPerfState.maxVPState.memClk =  dcgmvalue.DCGM_INT32_BLANK
config_values.mPerfState.maxVPState.procClk = dcgmvalue.DCGM_INT32_BLANK
config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
config_values.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL
config_values.mPowerLimit.val = powerLimit_set

## Set Config and verify the value
ret = dcgm_agent.dcgmConfigSet(handle, groupId, config_values, statusHandle)
assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to set configuration for the group: %s" % ret
dcgm_agent.dcgmStatusClear(statusHandle)
helper_verify_power_value_standalone(handle, groupId, powerLimit_set)

ret = dcgm_agent.dcgmPolicySet(handle, groupId, newPolicy, statusHandle)
assert (ret == dcgm_structs.DCGM_ST_OK)

time.sleep(5) # give the policy manager a chance to start

requestId = dcgm_agent.dcgmPolicyRegister(handle, groupId, dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED, c_callback, c_callback)
assert(requestId != None)

# inject an error into page retirement
field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
field.fieldId = dcgm_fields.DCGM_FI_DEV_RETIRED_DBE
field.status = 0
field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
field.ts = int((time.time()+11) * 1000000.0) # set the injected data into the future
field.value.i64 = 10

ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, validDevice, field)
assert (ret == dcgm_structs.DCGM_ST_OK)

now = time.time()

while not callbackCalled:
    if time.time() == now + 60: # wait 60 seconds
        print("Timeout waiting for callback")
        sys.exit(1)

print("Callback successfully received.")

## Verify that configuration is auto-enforced after GPU reset
#dcgm_agent.dcgmStatusClear(statusHandle)
#ret = dcgm_agent.dcgmConfigEnforce(handle, groupId, statusHandle)
helper_verify_power_value_standalone(handle, groupId, powerLimit_set)

print("Config enforce verification successful")

ret = dcgm_agent.dcgmGroupDestroy(handle, groupId)
assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to remove the test group, error: %s" % ret 

ret = dcgm_agent.dcgmStatusDestroy(statusHandle)
assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to remove status handler, error: %s" % ret

dcgm_agent.dcgmShutdown()
