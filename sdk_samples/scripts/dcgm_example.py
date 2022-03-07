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
import os
import sys
import argparse

try:
    from dcgm_structs import dcgmExceptionClass
    import pydcgm
    import dcgm_structs
    import dcgm_fields
    import dcgm_agent
    import dcgmvalue
except:
    pass
    print("Unable to find python bindings, please refer to the exmaple below: ")
    print("PYTHONPATH=/usr/local/dcgm/bindings python dcgm_example.py")
    sys.exit(1)

## Look at __name__ == "__main__" for entry point to the script

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
    errorCount = 0
    errorInfo = dcgm_agent.dcgmStatusPopError(statusHandle)

    while (errorInfo != None):
        errorCount += 1
        print("Error%d" % errorCount)
        print(("  GPU Id: %d" % errorInfo.gpuId))
        print(("  Field ID: %d" % errorInfo.fieldId))
        print(("  Error: %d" % errorInfo.status))
        errorInfo = dcgm_agent.dcgmStatusPopError(statusHandle)


## Helper method to convert enum to system name
def helper_convert_system_enum_to_sytem_name(system):
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_PCIE):
        return "PCIe"

    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_NVLINK):
        return "NvLink"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_PMU):
        return "PMU"

    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_MCU):
        return "MCU"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_MEM):
        return "MEM"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_SM):
        return "SM"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_INFOROM):
        return "Inforom"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_THERMAL):
        return "Thermal"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_POWER):
        return "Power"   
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_DRIVER):
        return "Driver"
    
    
## helper method to convert helath return to a string for display purpose        
def convert_overall_health_to_string(health):
    if health == dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        return "Pass"
    elif health == dcgm_structs.DCGM_HEALTH_RESULT_WARN:
        return "Warn"
    elif  health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL:
        return "Fail"
    else :
        return "N/A"

def nvvs_installed():
    return os.path.isfile('/usr/share/nvidia-validation-suite/nvvs')

def dcgm_diag_test_didnt_pass(rc):
    if rc == dcgm_structs.DCGM_HEALTH_RESULT_FAIL or rc == dcgm_structs.DCGM_HEALTH_RESULT_WARN:
        return True
    else:
        return False

def dcgm_diag_test_index_to_name(index):
    if index == dcgm_structs.DCGM_SWTEST_BLACKLIST:
        return "blacklist"
    elif index == dcgm_structs.DCGM_SWTEST_NVML_LIBRARY:
        return "nvmlLibrary"
    elif index == dcgm_structs.DCGM_SWTEST_CUDA_MAIN_LIBRARY:
        return "cudaMainLibrary"
    elif index == dcgm_structs.DCGM_SWTEST_CUDA_RUNTIME_LIBRARY:
        return "cudaRuntimeLibrary"
    elif index == dcgm_structs.DCGM_SWTEST_PERMISSIONS:
        return "permissions"
    elif index == dcgm_structs.DCGM_SWTEST_PERSISTENCE_MODE:
        return "persistenceMode"
    elif index == dcgm_structs.DCGM_SWTEST_ENVIRONMENT:
        return "environment"
    elif index == dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT:
        return "pageRetirement"
    elif index == dcgm_structs.DCGM_SWTEST_GRAPHICS_PROCESSES:
        return "graphicsProcesses"
    elif index == dcgm_structs.DCGM_SWTEST_INFOROM:
        return "inforom"
    else:
        raise dcgm_structs.DCGMError(dcgm_structs.DCGM_ST_BADPARAM)

# Returns true if the error here should be ignored
def should_ignore_error(diagException):
    if diagException.info:
        if diagException.info.find("MIG configuration is incompatible with the diagnostic because it prevents access to the entire GPU."
    ) != -1:
            return True

        if diagException.info.find("Cannot run diagnostic: CUDA does not support enumerating GPUs with MIG mode enabled") == 0:
            return True

    return False

def main(manualOpMode=False, embeddedHostengine=True):
    
    if manualOpMode:
        ## Initialize the DCGM Engine as manual operation mode. This implies that it's execution is 
        ## controlled by the monitoring agent. The user has to periodically call APIs such as 
        ## dcgmEnginePolicyTrigger and dcgmEngineUpdateAllFields which tells DCGM to wake up and 
        ## perform data collection and operations needed for policy management.
        ## Manual operation mode is only possible on an "embedded" hostengine.
        opMode = dcgm_structs.DCGM_OPERATION_MODE_MANUAL
    else:
        ## Initialize the DCGM Engine as automatic operation mode. This is required when connecting
        ## to a "standalone" hostengine (one that is running separately) but can also be done on an 
        ## embedded hostengine.  In this mode, fields are updated
        ## periodically based on their configured frequency.  When watching new fields you must still manually
        ## trigger an update if you wish to view these new fields' values right away.
        opMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO
    
    if embeddedHostengine:
        print(("Running an embedded hostengine with %s opmode..." % 
              ('manual' if manualOpMode else 'auto')))
        
        ## create embedded hostengine by leaving ipAddress as None
        dcgmHandle = pydcgm.DcgmHandle(opMode=opMode)
    
    else:
        print(("Connecting to a standalone hostengine with %s opmode..." % 
              ('manual' if manualOpMode else 'auto')))
        
        dcgmHandle = pydcgm.DcgmHandle(ipAddress='127.0.0.1', opMode=opMode)
    print("")

    ## Get a handle to the system level object for DCGM
    dcgmSystem = dcgmHandle.GetSystem()
    supportedGPUs = dcgmSystem.discovery.GetAllSupportedGpuIds()
    
    
    ## Create an empty group. Let's call the group as "one_gpus_group". 
    ## We will add the first supported GPU in the system to this group. 
    dcgmGroup = pydcgm.DcgmGroup(dcgmHandle, groupName="one_gpu_group", groupType=dcgm_structs.DCGM_GROUP_EMPTY)
    
    #Skip the test if no supported gpus are available
    if len(supportedGPUs) < 1:
        print("Unable to find supported GPUs on this system")
        sys.exit(0)
        
    dcgmGroup.AddGpu(supportedGPUs[0])

    ## Invoke method to get gpu IDs of the members of the newly-created group
    groupGpuIds = dcgmGroup.GetGpuIds()
    
    ## Trigger field updates since we just started DCGM (always necessary in MANUAL mode to get recent values)
    dcgmSystem.UpdateAllFields(waitForUpdate=True)

    ## Get the current configuration for the group
    config_values = dcgmGroup.config.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
    
    ## Display current configuration for the group
    for x in range(0, len(groupGpuIds)):
        print("GPU Id      : %d" % (config_values[x].gpuId))
        print("Ecc  Mode   : %s" % (convert_value_to_string(config_values[x].mEccMode)))
        print("Sync Boost  : %s" % (convert_value_to_string(config_values[x].mPerfState.syncBoost)))
        print("Mem Clock   : %s" % (convert_value_to_string(config_values[x].mPerfState.targetClocks.memClock)))
        print("SM  Clock   : %s" % (convert_value_to_string(config_values[x].mPerfState.targetClocks.smClock)))
        print("Power Limit : %s" % (convert_value_to_string(config_values[x].mPowerLimit.val)))
        print("Compute Mode: %s" % (convert_value_to_string(config_values[x].mComputeMode)))
        print("\n")

    ## Add the health watches
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_ALL)
    
    ## Ensure that the newly watched health fields are updated since we wish to access them right away.
    ## Needed in manual mode and only needed in auto mode if we want to see the values right away
    dcgmSystem.UpdateAllFields(waitForUpdate=True)

    ## Invoke Health checks
    try:
        group_health = dcgmGroup.health.Check()
        print("Overall Health for the group: %s" % convert_overall_health_to_string(group_health.overallHealth))

        for index in range (0, group_health.incidentCount):
            print("GPU ID : %d" % group_health.incidents[index].entityInfo.entityId)

            print("system tested     : %d" % group_health.incidents[index].system)
            print("system health     : %s" % convert_overall_health_to_string(group_health.incidents[index].health))
            print("system health err : %s" % group_health.incidents[index].error.msg)
            print("\n")
    except dcgm_structs.DCGMError as e:
        errorCode = e.value
        print("dcgmHealthCheck returned error %d: %s" % (errorCode, e))
        sys.exc_clear()

    print("")
        
    if nvvs_installed():
        ## This will go ahead and perform a "prologue" diagnostic 
        ## to make sure everything is ready to run
        ## currently this calls an outside diagnostic binary but eventually
        ## that binary will be merged into the DCGM framework 
        ## The "response" is a dcgmDiagResponse structure that can be parsed for errors. 
        try:
            response = dcgmGroup.action.RunDiagnostic(dcgm_structs.DCGM_DIAG_LVL_SHORT)
        except dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED):
            print("One of the GPUs on your system is not supported by NVVS")
        except dcgmExceptionClass(dcgm_structs.DCGM_ST_GROUP_INCOMPATIBLE):
            print("GPUs in the group are not compatible with each other for running diagnostics")
        except dcgmExceptionClass(dcgm_structs.DCGM_ST_NVVS_ERROR) as e:
            if not should_ignore_error(e):
               raise(e)
            else:
                print(str(e))
        else:
            isHealthy = True
            
            for i in range(0, response.levelOneTestCount):
                if dcgm_diag_test_didnt_pass(response.levelOneResults[i].result):
                    print("group failed validation check for %s" % dcgm_diag_test_index_to_name(i))
                    isHealthy = False
                
            if not isHealthy:
                print("System is not healthy")
    else:
        print("not running short group validation because NVIDIA Validation Suite is not installed")
    print("")
    
    ## Add process watches so that DCGM can start watching process info
    dcgmGroup.stats.WatchPidFields(1000000, 3600, 0)
    
    ####################################################################
    # Start a CUDA process at this point and get the PID for the process
    ## Wait until it completes
    ## dcgmGroup.health.Check() is a low overhead check and can be performed 
    ## in parallel to the job without impacting application's performance
    ####################################################################

    # Initialized to 0 for now. Change it to PID of the CUDA process if there is a process to run
    pid = 0

    try:
        pidInfo = dcgmGroup.stats.GetPidInfo(pid)
    
        ## Display some process statistics (more may be desired)
        print("Process ID      : %d" % pid)
        print("Start time      : %d" % pidInfo.summary.startTime)
        print("End time        : %d" % pidInfo.summary.endTime)
        print("Energy consumed : %d" % pidInfo.summary.energyConsumed)
        print("Max GPU Memory  : %d" % pidInfo.summary.maxGpuMemoryUsed)
        print("Avg. SM util    : %d" % pidInfo.summary.smUtilization.average)
        print("Avg. mem util   : %d" % pidInfo.summary.memoryUtilization.average)
    except:
        print("There was no CUDA job running to collect the stats")
        pass
    
    # Nvidia Validation Suite is required when performing "validate" actions
    if nvvs_installed():
        ## Now that the process has completed we perform an "epilogue" diagnostic that will stress the system
        try:
            response = dcgmGroup.action.RunDiagnostic(dcgm_structs.DCGM_DIAG_LVL_MED)
        except dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED):
            print("One of the GPUs on your system is not supported by NVVS")
        except dcgmExceptionClass(dcgm_structs.DCGM_ST_NVVS_ERROR) as e:
            if not should_ignore_error(e):
               raise(e)
            else:
                print(str(e))
        else:
            ## Check the response and do any actions desired based on the results. 
            pass
        
    else:
        print("not running medium group validation because NVIDIA Validation Suite is not installed")
    print("")
    
    ## Delete the group
    dcgmGroup.Delete()
    del(dcgmGroup)
    dcgmGroup = None

    ## disconnect from the hostengine by deleting the DcgmHandle object
    del(dcgmHandle)
    dcgmHandle = None

## Entry point for this script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for showing off how to use DCGM python bindings')
    parser.add_argument('-o', '--opmode', 
                        choices=['manual', 'auto'], 
                        default='manual',
                        help='Operation mode for the hostengine. Must be auto if a standalone hostengine ' +
                              'is used. Defaults to auto.')
    
    parser.add_argument('-t', '--type',
                        choices=['embedded', 'standalone'], 
                        default='embedded',
                        help='Type of hostengine.  Embedded mode starts a hostengine within the ' +
                             'same process. Standalone means that a separate hostengine process ' +
                             'is already running that will be connected to. '
                        )
    
    args = parser.parse_args()
    manualOpMode = args.opmode == 'manual'
    embeddedHostengine = args.type == 'embedded'
    
    main(manualOpMode, embeddedHostengine)
