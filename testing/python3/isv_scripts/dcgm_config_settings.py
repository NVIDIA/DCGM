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
sys.path.insert(0, '../')
import dcgm_structs
import dcgm_fields
import dcgm_agent
import dcgmvalue
from threading import Thread
from time import sleep

## Look at __name__ == "__main__" for entry point to the script

class RunDCGM():
    
    def __init__(self, ip, opMode):
        self.ip = ip
        self.opMode = opMode
    
    def __enter__(self):
        dcgm_structs._dcgmInit()
        self.handle = dcgm_agent.dcgmInit()
        return self.handle
        
    def __exit__(self, eType, value, traceback):
        dcgm_agent.dcgmShutdown()

## Helper method to convert DCGM value to string
def convert_value_to_string(value):
    v = dcgmvalue.DcgmValue(value)

    try:
        if (v.IsBlank()):
            return "N/A"
        else:
            return v.__str__()
    except:
        ## Exception is generally thorwn when int32 is
        ## passed as an input. Use additional methods to fix it
        sys.exc_clear()
        v = dcgmvalue.DcgmValue(0)
        v.SetFromInt32(value)

        if (v.IsBlank()):
            return "N/A"
        else:
            return v.__str__()

## Helper method to investigate the status handler
def helper_investigate_status(statusHandle):
    """
    Helper method to investigate status handle
    """
    errorCount = 0;
    errorInfo = dcgm_agent.dcgmStatusPopError(statusHandle)

    while (errorInfo != None):
        errorCount += 1
        print("Error%d" % errorCount)
        print(("  GPU Id: %d" % errorInfo.gpuId))
        print(("  Field ID: %d" % errorInfo.fieldId))
        print(("  Error: %d" % errorInfo.status))
        errorInfo = dcgm_agent.dcgmStatusPopError(statusHandle)


## Worker Function to get Configuration for a dcgm group
def agent_worker_function(handle, groupId, groupInfo, status_handle):
    NUM_ITERATIONS = 5
    count = 0    

    while True:
        dcgm_agent.dcgmUpdateAllFields(handle, 1)

        ## Get the current configuration for the group
        config_values = dcgm_agent.dcgmConfigGet(handle, groupId, dcgm_structs.DCGM_CONFIG_CURRENT_STATE, groupInfo.count, status_handle)
    
        ## Since this is a group operation, Check for the status codes if any of the property failed    
        helper_investigate_status(status_handle)
        dcgm_agent.dcgmStatusClear(status_handle)
        
        ## Display current configuration for the group
        for x in range(0,groupInfo.count):
            print("GPU Id      : %d" % (config_values[x].gpuId))
            print("Ecc  Mode   : %s" % (convert_value_to_string(config_values[x].mEccMode)))
            print("Auto Boost  : %s" % (convert_value_to_string(config_values[x].mPerfState.autoBoost)))
            print("Sync Boost  : %s" % (convert_value_to_string(config_values[x].mPerfState.autoBoost)))
            print("Mem Clock   : %s" % (convert_value_to_string(config_values[x].mPerfState.minVPState.memClk)))
            print("SM  Clock   : %s" % (convert_value_to_string(config_values[x].mPerfState.minVPState.procClk)))
            print("Power Limit : %s" % (convert_value_to_string(config_values[x].mPowerLimit.val)))
            print("Compute Mode: %s" % (convert_value_to_string(config_values[x].mComputeMode)))
            print("\n")
        
        count = count + 1
        
        if count == NUM_ITERATIONS:
            break

        sleep(2)


## Entry point for this script
if __name__ == "__main__":
    
    ## Initialize the DCGM Engine as manual operation mode. This implies that it's execution is 
    ## controlled by the monitoring agent. The user has to periodically call APIs such as 
    ## dcgmEnginePolicyTrigger and dcgmEngineUpdateAllFields which tells DCGM to wake up and 
    ## perform data collection and operations needed for policy management.
    with RunDCGM('127.0.0.1', dcgm_structs.DCGM_OPERATION_MODE_MANUAL) as handle:
    
        ## Create a default group. (Default group is comprised of all the GPUs on the node)
        ## Let's call the group as "all_gpus_group". The method returns an opaque handle (groupId) to
        ## identify the newly created group.
        groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "all_gpus_group")
        
        ## Invoke method to get information on the newly created group
        groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId)
        
        ## Create reference to DCGM status handler which can be used to get the statuses for multiple 
        ## operations on one or more devices present in the group
        status_handle = dcgm_agent.dcgmStatusCreate()
        
        ## The worker function can be executed as a separate thread or as part of the main thread.
        ## Executed as a separate thread here
        thread = Thread(target = agent_worker_function, args = (handle, groupId, groupInfo, status_handle))
        thread.start()
    
        ##########################################
        # Any other useful work can be placed here
        ##########################################
        
        thread.join()
        print("Worker thread completed")
        
        ## Destroy the group
        ret = dcgm_agent.dcgmGroupDestroy(handle, groupId)
        assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to remove the test group, error: %s" % ret 
    
        ## Destroy the status handle
        ret = dcgm_agent.dcgmStatusDestroy(status_handle)
        assert(ret == dcgm_structs.DCGM_ST_OK), "Failed to remove status handler, error: %s" % ret
        
    
