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
from ctypes import *

C_FUNC = CFUNCTYPE(None, c_void_p)

def callback_function(data):
    print("Received a callback from the policy manager")

c_callback = C_FUNC(callback_function)

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

## Main
if __name__ == "__main__":
    
    ## Initilaize the DCGM Engine as manual operation mode. This implies that it's execution is 
    ## controlled by the monitoring agent. The user has to periodically call APIs such as 
    ## dcgmEnginePolicyTrigger and dcgmEngineUpdateAllFields which tells DCGM to wake up and 
    ## perform data collection and operations needed for policy management.
    with RunDCGM('127.0.0.1', dcgm_structs.DCGM_OPERATION_MODE_MANUAL) as handle:
       
        # The validate information should be packed in the dcgmRunDiag object 
        runDiagInfo = dcgm_structs.c_dcgmRunDiag_v7()
        runDiagInfo.version = dcgm_structs.dcgmRunDiag_version7
    
        ## Create a default group. (Default group is comprised of all the GPUs on the node)
        ## Let's call the group as "all_gpus_group". The method returns an opaque handle (groupId) to
        ## identify the newly created group.
        runDiagInfo.groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "all_gpus_group")
        
        ## Invoke method to get information on the newly created group
        groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, runDiagInfo.groupId)
        
        ## define the actions and validations for those actions to take place 
        runDiagInfo.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    
        ## This will go ahead and perform a "prologue" diagnostic 
        ## to make sure everything is ready to run
        ## currently this calls an outside diagnostic binary but eventually
        ## that binary will be merged into the DCGM framework 
        ## The "response" is a dcgmDiagResponse structure that can be parsed for errors
        response = dcgm_agent.dcgmActionValidate_v2(handle, runDiagInfo)
    
        ## This will perform an "eiplogue" diagnostic that will stress the system
        ## Currently commented out because it takes several minutes to execute
        # runDiagInfo.validate = dcgm_structs.DCGM_POLICY_VALID_SV_LONG
        #response = dcgm_agent.dcgmActionValidate_v2(handle, dcgmRunDiagInfo)
    
        ## prime the policy manager to look for ECC, PCIe events
        ## if a callback occurs the function above is called. Currently the data returned
        ## corresponds to the error that occurred (PCI, DBE, etc.) but in the future it will be a 
        ## dcgmPolicyViolation_t or similar
        ret = dcgm_agent.dcgmPolicyRegister(handle, runDiagInfo.groupId, dcgm_structs.DCGM_POLICY_COND_PCI | dcgm_structs.DCGM_POLICY_COND_DBE, None, c_callback)
    
        ## trigger the policy loop
        ## typically this would be looped in a separate thread or called on demand
        ret = dcgm_agent.dcgmPolicyTrigger(handle)
        
        ## Destroy the group
        try:
            dcgm_agent.dcgmGroupDestroy(handle, runDiagInfo.groupId)
        except dcgm_structs.DCGMError as e:
            print("Failed to remove the test group, error: %s" % e, file=sys.stderr) 
            sys.exit(1)
    
