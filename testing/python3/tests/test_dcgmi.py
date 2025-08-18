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
import pydcgm
import dcgm_structs
import dcgm_agent_internal
import dcgm_agent
import logger
import test_utils
import dcgm_fields
import dcgm_errors
import apps
import dcgmvalue
import DcgmSystem
import DcgmDiag
from dcgm_structs import dcgmExceptionClass, DCGM_ST_NOT_CONFIGURED
import dcgm_structs_internal
import dcgm_internal_helpers
import utils
import threading
import option_parser
import shutil
import string
import time
import tempfile
from ctypes import *
import sys
import os
import pprint
from subprocess import Popen, PIPE
from sys import stdout
import json
import subprocess
import glob
from contextlib import closing
import ctypes
from _test_helpers import skip_test_if_no_dcgm_nvml
from dcgm_field_injection_helpers import inject_nvml_value, inject_value
import dcgm_field_helpers
import nvml_injection
import nvml_injection_structs
import dcgm_nvml

def _run_dcgmi_command(args):
    ''' run a command then return (retcode, stdout_lines, stderr_lines) '''
    dcgmi = apps.DcgmiApp(args)
    # Some commands (diag -r 2) can take a minute or two
    dcgmi.start(250)
    retValue = dcgmi.wait()
    dcgmi.validate()
    return retValue, dcgmi.stdout_lines, dcgmi.stderr_lines

def _is_eris_diag_inforom_failure(args, stdout_lines):
    INFOROM_FAILURE_STRING = 'nvmlDeviceValidateInforom for nvml device'
    if not option_parser.options.eris:
        # This is used to skip diag tests. We only want to do that on Eris
        return False
    if len(args) > 0 and args[0] == 'diag' and INFOROM_FAILURE_STRING in stdout_lines:
        return True
    return False

def _assert_valid_dcgmi_results(args, retValue, stdout_lines, stderr_lines):
    if (len(stdout_lines) == 0) and (len(stderr_lines) > 0):
        logger.error('stderr: "%s"' % (stderr_lines)) 
    assert (len(stdout_lines) > 0), 'No output detected for args "%s"' % ' '.join(args[1:])

    if _is_eris_diag_inforom_failure(args, stdout_lines):
        # If we see inforom corruption, the test should not fail
        test_utils.skip_test('Detected corrupt inforom for diag test')
        return

    output = ''
    for line in stdout_lines:
        output = output + line + ' '
    if test_utils.is_mig_incompatible_failure(output):
        test_utils.skip_test("Skipping this test because MIG is configured incompatibly (preventing access to the whole GPU)")
    
    if retValue != c_ubyte(dcgm_structs.DCGM_ST_OK).value:
        logger.error('Valid test - Function returned error code: %s . Args used: "%s"' % (retValue, ' '.join(args[1:]))) 
        logger.error('Stdout:')
        for line in stdout_lines:
            logger.error('\t'+line)
        logger.error('Stderr:')
        for line in stderr_lines:
            logger.error('\t'+line)
        assert False, "See errors above."
    
    errLines = _lines_with_errors(stdout_lines)
    assert len(errLines) == 0, "Found errors in output.  Offending lines: \n%s" % '\n'.join(errLines)
        

def _assert_invalid_dcgmi_results(args, retValue, stdout_lines, stderr_lines):
    assert retValue != c_ubyte(dcgm_structs.DCGM_ST_OK).value, \
           'Invalid test - Function returned error code: %s . Args used: "%s"' \
           % (retValue, ', '.join(args[0:]))
            
    assert len(_lines_with_errors(stderr_lines + stdout_lines)) >= 1, \
            'Function did not display error message for args "%s". Returned: %s\nstdout: %s\nstderr: %s' \
            % (' '.join(args[1:]), retValue, '\n'.join(stdout_lines), '\n'.join(stderr_lines))

def _lines_with_errors(lines):
    errorLines = []

    errorStrings = [
        'error',
        'invalid',
        'incorrect',
        'unexpected'
    ]
    exceptionStrings = [
        'nvlink error',
        'flit error',
        'data error',
        'replay error',
        'recovery error',
        'ecc error',
        'xid error',
        '| error detail',
        'malformed packet error',
        'buffer overrun error',
        'rx error',
        'rx remote error',
        'rx general error',
        'link integrity error',
        'rx symbol error',
        'effective error'
    ]

    for line in lines:
        lineLower = line.lower()

        for errorString in errorStrings:
            if not errorString in lineLower:
                continue

            wasExcepted = False
            for exceptionString in exceptionStrings:
                if exceptionString in lineLower:
                    wasExcepted = True
                    break
            if wasExcepted:
                continue

            errorLines.append(line)

    return errorLines

def _create_dcgmi_group(groupType=dcgm_structs.DCGM_GROUP_EMPTY):
    ''' Create an empty group and return its group ID '''
    createGroupArgs = ["group", "-c", "test_group"]

    if groupType == dcgm_structs.DCGM_GROUP_DEFAULT:
        createGroupArgs.append('--default')
    elif groupType == dcgm_structs.DCGM_GROUP_DEFAULT_NVSWITCHES:
        createGroupArgs.append('--defaultnvswitches')
    
    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(createGroupArgs)
    _assert_valid_dcgmi_results(createGroupArgs, retValue, stdout_lines, stderr_lines)
    
    # dcgmi "group -c" outputs a line like 'Successfully created group "test_group" with a group ID of 2'
    # so we capture the last word as the group ID (it doesn't seem like there's a better way)
    # convert to int so that if it's not an int, an exception is raised
    return int(stdout_lines[0].strip().split()[-1])

def _test_valid_args(argsList):
    for args in argsList:
        retValue, stdout_lines, stderr_lines = _run_dcgmi_command(args)
        _assert_valid_dcgmi_results(args, retValue, stdout_lines, stderr_lines)
        
def _test_invalid_args(argsList):
    for args in argsList:
        retValue, stdout_lines, stderr_lines = _run_dcgmi_command(args)
        _assert_invalid_dcgmi_results(args, retValue, stdout_lines, stderr_lines)
        
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus(2) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_with_injection_gpu_instances(2)
@test_utils.run_with_injection_gpu_compute_instances(2)
def test_dcgmi_group(handle, gpuIds, instanceIds, ciIds):
    """
    Test DCGMI group
    """
     
    DCGM_ALL_GPUS = dcgm_structs.DCGM_GROUP_ALL_GPUS
     
    groupId = str(_create_dcgmi_group())
    
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["group", "-l", ""],                            # list groups
            ["group", "-g", groupId, "-i"],                 # get info on created group
            ["group", "-g", groupId, "-a", str(gpuIds[0])], # add gpu to group
            ["group", "-g", groupId, "-r", str(gpuIds[0])], # remove that gpu from the group
            ["group", "-g", groupId, "-a", "instance:" + str(instanceIds[0])], # add instance to group
            ["group", "-g", groupId, "-r", "instance:" + str(instanceIds[0])], # remove instance from group
            ["group", "-g", groupId, "-a", "ci:" + str(ciIds[0])], # add CI to group
            ["group", "-g", groupId, "-r", "ci:" + str(ciIds[0])], # remove CI from group
            ["group", "-g", groupId, "-a", "gpu:" + str(gpuIds[0])], # add gpu to group with gpu tag
            ["group", "-g", groupId, "-r", "gpu:" + str(gpuIds[0])], # remove that gpu from the group with gpu tag
            ["group", "-g", groupId, "-a", "instance:" + str(instanceIds[0])], # add instance to group with instance tag
            ["group", "-g", groupId, "-r", "instance:" + str(instanceIds[0])], # remove instance from the group with instance tag
            ["group", "-g", groupId, "-a", "ci:" + str(instanceIds[0])], # add CI to group with compute instance tag
            ["group", "-g", groupId, "-r", "ci:" + str(instanceIds[0])], # remove CI from the group with compute instace tag
            # Testing Cuda/MIG formats for entity ids.
            # Fake GPUs have GPU-00000000-0000-0000-0000-000000000000 UUID
            ["group", "-g", groupId, "-a", "00000000-0000-0000-0000-000000000000"],  # add a GPU to the group
            ["group", "-g", groupId, "-r", "00000000-0000-0000-0000-000000000000"],  # remove the GPU from the group
            ["group", "-g", groupId, "-a", "GPU-00000000-0000-0000-0000-000000000000"],  # add a GPU to the group
            ["group", "-g", groupId, "-r", "GPU-00000000-0000-0000-0000-000000000000"],  # remove the GPU from the group
            ["group", "-g", groupId, "-a", "MIG-GPU-00000000-0000-0000-0000-000000000000"],  # add a GPU to the group
            ["group", "-g", groupId, "-r", "MIG-GPU-00000000-0000-0000-0000-000000000000"],  # remove the GPU from the group
            ["group", "-g", groupId, "-a", "GPU-00000000-0000-0000-0000-000000000000/0"],  # add a GPU Instance to the group
            ["group", "-g", groupId, "-r", "GPU-00000000-0000-0000-0000-000000000000/0"],  # remove the GPU Instance from the group
            ["group", "-g", groupId, "-a", "GPU-00000000-0000-0000-0000-000000000000/%d/%d" % (0, 0)],  # add a CI to the group
            ["group", "-g", groupId, "-r", "GPU-00000000-0000-0000-0000-000000000000/%d/%d" % (0, 0)],  # remove the CI from the group
            ["group", "-g", groupId, "-a", "GPU-00000000-0000-0000-0000-000000000000/%d/%d" % (1, 0)],  # add another CI to the group
            ["group", "-g", groupId, "-r", "GPU-00000000-0000-0000-0000-000000000000/%d/%d" % (1, 0)],  # remove the CI from the group
            ["group", "-g", groupId, "-a", "MIG-GPU-00000000-0000-0000-0000-000000000000/%d/%d" % (0, 0)],  # add a CI to the group
            ["group", "-g", groupId, "-r", "MIG-GPU-00000000-0000-0000-0000-000000000000/%d/%d" % (0, 0)],  # remove the CI from the group
            ["group", "-g", groupId, "-a", "MIG-GPU-00000000-0000-0000-0000-000000000000/%d/%d" % (1, 0)],  # add another CI to the group
            ["group", "-g", groupId, "-r", "MIG-GPU-00000000-0000-0000-0000-000000000000/%d/%d" % (1, 0)],  # remove the CI from the group
            ["group", "-g", groupId, "-a", "MIG-GPU-00000000-0000-0000-0000-000000000000/0/*"],  # add all CIs for InstanceId_0
            ["group", "-g", groupId, "-r", "MIG-GPU-00000000-0000-0000-0000-000000000000/0/0"],  # remove CI_0
            # This one disabled as the run_with_injection_gpu_compute_instances decorator does not inject hierarchy for now.
            # ["group", "-g", groupId, "-r", "MIG-GPU-00000000-0000-0000-0000-000000000000/0/1"],  # remove CI_1
            ["group", "-g", groupId, "-a", "MIG-GPU-00000000-0000-0000-0000-000000000000/*/0"],  # add all CI_0 for Instances 0 and 1
            ["group", "-g", groupId, "-r", "MIG-GPU-00000000-0000-0000-0000-000000000000/0/0"],  # remove CI_0 for Instance 0
            ["group", "-g", groupId, "-r", "MIG-GPU-00000000-0000-0000-0000-000000000000/1/0"],  # remove CI_0 for Instance 1
            ["group", "-g", groupId, "-a", "*"],  # add all GPUs
            ["group", "-g", groupId, "-r", "*"],  # remove all GPUs
            ["group", "-g", groupId, "-a", "*/*"],  # add all GPU instances
            ["group", "-g", groupId, "-r", "*/*"],  # remove all GPU instances
            ["group", "-g", groupId, "-a", "*/*/*"],  # add all CIs
            ["group", "-g", groupId, "-r", "*/*/*"],  # remove all CIs
            ["group", "-g", groupId, "-a", "*,*/*/*"],  # add all GPUs and CIs
            ["group", "-g", groupId, "-r", "*,*/*/*"],  # remove all GPUs and CIs
            ["group", "-d", groupId, ],                     # delete the group
            ["group", "-g", "0", "-i"],                     # Default group can be fetched by ID as long as group IDs start at 0
    ])
         
    nonExistentGroupId = str(int(groupId) + 10)
     
    groupId = str(_create_dcgmi_group())

    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["group", "-c", "--default"],                       # Can't create a group called --default
            ["group", "-c", "--add"],                           # Can't create a group called --add
            ["group", "-c", "-a"],                              # Can't create a group called -a
            ["group", "-g", nonExistentGroupId, "-a", str(gpuIds[0])], # Can't add to a group that doesn't exist
            ["group", "-g", groupId, "-a", "129"],              # Can't add a GPU that doesn't exist
            ["group", "-g", groupId, "-r", "129"],              # Can't remove a GPU that doesn't exist
            ["group", "-g", groupId, "-a", "instance:2000"],     # Can't add an instance that doesn't exist
            ["group", "-g", groupId, "-r", "instance:2000"],     # Can't remove an instance that doesn't exist
            ["group", "-g", groupId, "-a", "ci:2000"],           # Can't add a CI that doesn't exist
            ["group", "-g", groupId, "-r", "ci:2000"],           # Can't remove a CI that doesn't exist
            ["group", "-g", nonExistentGroupId, "-r", str(gpuIds[0])],  # Can't remove from a group that does't exist
            ["group", "-g", "0", "-r", "0"],                    # Can't remove from the default group (ID 0)
            ["group", "-g", str(DCGM_ALL_GPUS), "-r", str(gpuIds[0])], # Can't remove from the default group w/ handle
            ["group", "-d", "0"],                               # Can't delete the default group (ID 0)
            ["group", "-d", str(DCGM_ALL_GPUS)],                # Can't delete the default group w/ handle
            ["group", "-d", nonExistentGroupId],                # Can't delete a group that doesnt exist
            ["group", "-g", groupId, "-a", "11111111-1111-1111-1111-111111111111"],  # add a GPU to the group
            ["group", "-g", groupId, "-r", "11111111-1111-1111-1111-111111111111"],  # remove the GPU from the group
            ["group", "-g", groupId, "-a", "GPU-11111111-1111-1111-1111-111111111111"],  # add a GPU to the group
            ["group", "-g", groupId, "-r", "GPU-11111111-1111-1111-1111-111111111111"],  # remove the GPU from the group
            ["group", "-g", groupId, "-a", "MIG-GPU-11111111-1111-1111-1111-111111111111"],  # add a GPU to the group
            ["group", "-g", groupId, "-r", "MIG-GPU-11111111-1111-1111-1111-111111111111"],  # remove the GPU from the group
            ["group", "-g", groupId, "-a", "%d/%d" % (129, instanceIds[0])],  # Can't add an instance that doesn't exits
            ["group", "-g", groupId, "-r", "%d/%d" % (129, instanceIds[0])],  # Can't remove an instance that doesn't exist
            ["group", "-g", groupId, "-a", "%d/%d/%d" % (129, instanceIds[0], ciIds[0])],  # Can't add a CI that doesn't exist
            ["group", "-g", groupId, "-r", "%d/%d/%d" % (129, instanceIds[0], ciIds[0])],  # Can't remove a CI that doesn't exist
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_live_cpus()
def test_dcgmi_cpu_group(handle, gpuIds, cpuIds):
    gpuStr = ''
    cpuStr = ''
    for gpuId in gpuIds:
        if gpuStr == '':
            gpuStr = str(gpuId)
        else:
            gpuStr = gpuStr + ",%s" % str(gpuId)

    for cpuId in cpuIds:
        if cpuStr == '':
            cpuStr = "cpu:%s" % str(cpuId)
        else:
            cpuStr = cpuStr + ",cpu:%s" % str(cpuId)
    entityStr = "%s,%s" % (gpuStr, cpuStr)
    createGroupArgs = ["group", "-c", "cpu_test_group", "-a", entityStr ]
    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(createGroupArgs)
    _assert_valid_dcgmi_results(createGroupArgs, retValue, stdout_lines, stderr_lines)
 
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_nvswitches(2)
def test_dcgmi_group_nvswitch(handle, switchIds):

    groupId = str(_create_dcgmi_group(groupType=dcgm_structs.DCGM_GROUP_DEFAULT_NVSWITCHES))

    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["group", "-g", groupId, "-i"],                 # get info on created group
            ["group", "-g", groupId, "-r", "nvswitch:%s" % str(switchIds[0])], # remove a switch from the group
            ["group", "-g", groupId, "-a", "nvswitch:%s" % str(switchIds[0])], # add a switch to group
            ["group", "-g", groupId, "-r", "nvswitch:%s" % str(switchIds[1])], # remove a 2nd switch from the group
            ["group", "-g", groupId, "-a", "nvswitch:%s" % str(switchIds[1])], # add a 2nd switch to group
            ["group", "-g", groupId, "-r", "nvswitch:%s,nvswitch:%s" % (str(switchIds[0]), str(switchIds[1]))], # remove both switches at once
            ["group", "-g", groupId, "-a", "nvswitch:%s,nvswitch:%s" % (str(switchIds[0]), str(switchIds[1]))], # add both switches at once
    ])
         
    nonExistantGroupId = str(int(groupId) + 10)
     
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["group", "-g", groupId, "-r", "taco:%s" % str(switchIds[0])], # remove a switch from an invalid entityGroupId
            ["group", "-g", groupId, "-a", "taco:%s" % str(switchIds[0])], # add a switch to an invalid entityGroupId
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_config(handle, gpuIds):
    """
    Test DCGMI config
    """
    assert len(gpuIds) > 0, "Failed to get devices from the node"

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    # Getting GPU power limits
    for gpuId in gpuIds:
        gpuAttrib = dcgmSystem.discovery.GetGpuAttributes(gpuId)
        dft_pwr = str(gpuAttrib.powerLimits.defaultPowerLimit)
        max_pwr = str(gpuAttrib.powerLimits.maxPowerLimit)

    groupId = str(_create_dcgmi_group())

    ## Keep args in order in all dcgmi command sequence arrays below.

    ## This is the list of dcgmi command arguments that are valid as root
    ## and non-root, and effect setup for additional tests. We use a list in
    ## case we want to add more than one.
    setupArgsTestList = [
        ["group", "-g", groupId, "-a", str(gpuIds[0])],        # add gpu to group
    ]

    _test_valid_args(setupArgsTestList)

    ## This is the list of dcgmi command arguments that are valid as root,
    ## and invalid as non-root.
    validArgsTestList = [
            ["config", "--get", "-g", groupId],                    # get default group configuration
            ["config", "--get", "-g", "0"],                        # get default group configuration by ID. This will work as long as group IDs start at 0
            ["config", "-g", groupId, "--set", "-P", dft_pwr],     # set default power limit
            ["config", "-g", groupId, "--set", "-P", max_pwr],     # set max power limit
            ["config", "--get", "-g", groupId, "--verbose"],       # get verbose default group configuration
            ["config", "--enforce", "-g", groupId],                # enforce default group configuration
            ["config", "--enforce", "-g", "0" ]                    # enforce group configuration on default group by ID
    ]

    # Setting the compute mode is only supported when MIG mode is not enabled.
    if not test_utils.is_mig_mode_enabled():
        # set group configuration on default group by ID
        validArgsTestList.append(["config", "--set", "-c", "0", "-g", "0" ])

    #Config management only works when the host engine is running as root
    if utils.is_root():
        _test_valid_args(validArgsTestList)
    else:
        _test_invalid_args(validArgsTestList)

    ## This is the list of invalid dcgmi command arguments.
    _test_invalid_args([
            ["config", "--get", "-g", "9999"],                 # Can't get config of group that doesn't exist
            ["config", "--get", "-g", "9999", "--verbose"],    # Can't get config of group that doesn't exist
            ["config", "--set", ""],                        # Can't set group configuration to nothing
            ["config", "--set", "-c", "5"],                 # Can't set an invalid compute mode
            ["config", "--enforce", "-g", "9999"]           # Can't enforce a configuration of group that doesn't exist
    ])
 
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus() #Use injected GPUs for policy so this doesn't fail on GeForce and Quadro
def test_dcgmi_policy(handle, gpuIds):
    """
     Test DCGMI policy
    """
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    groupObj = dcgmSystem.GetGroupWithGpuIds("testgroup", gpuIds)
    groupIdStr = str(groupObj.GetId().value)

    DCGM_ALL_GPUS = dcgm_structs.DCGM_GROUP_ALL_GPUS
     
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["policy", "--get", "", "-g", groupIdStr],                 # get default group policy
            ["policy", "--get", "-g", "0"],                            # get default group policy by ID. this will fail if groupIds ever start from > 0
            ["policy", "--get", "--verbose", "-g", groupIdStr],        # get verbose default group policy
            ["policy", "--set", "0,0", "-p", "-e", "-g", groupIdStr],  # set default group policy 
            ["policy", "--set", "1,2", "-p", "-e", "-g", groupIdStr],  # set default group policy 
            ["policy", "--set", "1,0", "-x", "-g", groupIdStr],        # set monitoring of xid errors
            ["policy", "--set", "1,0", "-x", "-n", "-g", groupIdStr],  # set monitoring of xid errors and nvlink errors
            #["policy", "--reg", ""]                                   # register default group policy (causes timeout)     
    ])
     
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["policy", "--get", "-g", "1000"],               # Can't get policy of group that doesn't exist
            ["policy", "--get", "-g", "1000", "--verbose"],  # Can't get policy of group that doesn't exist
            ["policy", "--set", "-p"],                       # Can't set group policy w/ no action/validaion
            ["policy", "--set", "0,0"],                      # Can't set group policy w/ no watches
            ["policy", "--set", "0,0", "-p", "-g", "1000" ], # Can't set group policy on group that doesn't exist
            ["policy", "--reg", "-g", "1000"]                # Can't register a policy of group that doesn't exist
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgmi_health(handle, gpuIds):
    """
      Test DCGMI Health
    """
     
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["health", "--fetch", ""],                  # get default group health
            ["health", "--set", "pmit"],                # set group health
            ["health", "--clear", ""]                   # clear group health watches
    ])
                
    #Create group for testing
    groupId = str(_create_dcgmi_group())
    nonExistantGroupId = str(int(groupId) + 10)
      
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["health", "--fetch", "-g", nonExistantGroupId],    # Can't get health of group that doesn't exist
            ["health", "--set", "a", "-g", nonExistantGroupId], # Can't get health of group that doesn't exist
            ["health", "--set", "pp"],                          # Can't set health of group with multiple of same tag
            ["health", "--get", "ap"],                          # Can't set health to all plus another tag 
            ["health", "--set", ""],                            # Can't set group health w/ no arguments
            ["health", "--check", "-g", nonExistantGroupId],    # Can't check health of group that doesn't exist
            ["health", "--check", "-g", groupId]                # Can't check health of group that has no watches enabled
    ])
 
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgmi_discovery(handle, gpuIds):
    """
    Test DCGMI discovery 
    """
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["discovery", "--list", ""],                   # list gpus
            ["discovery", "--info", "aptc"],               # check group info
            ["discovery", "--info", "aptc", "--verbose"]  # checl group info verbose
    ])
    
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["discovery", "--info", "a", "-g", "2"],              # Cant check info on group that doesn't exist
            ["discovery", "--info", "a", "--gpuid", "123"]        # Cant check info on gpu that doesn't exist
    ])

@test_utils.run_only_on_numa_systems()
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_live_cpus()
def test_dcgmi_discovery_cpus(handle, gpuIds, cpuIds):
    """
    Test DCGMI discovery 
    """
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["discovery", "--list", ""],
            ["discovery", "--info", "a", "--cpuid", "0"]     # check cpu can be specified
    ])

def helper_dcgmi_discovery_can_list_cx(numCxCards):
    expectedStr = f"{numCxCards} ConnectX found."
    _, stdoutLines, _ = _run_dcgmi_command(["discovery", "--list"])
    found = False
    for line in stdoutLines:
        if expectedStr in line:
            found = True
            break
    assert found, f"Cannot find expected output: [{expectedStr}]"

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_cx()
def test_dcgmi_discovery_can_list_cx_live(handle, cxIds):
    helper_dcgmi_discovery_can_list_cx(len(cxIds))

@test_utils.run_with_nvsdm_mock_config("one_cx.yaml")
@test_utils.run_with_standalone_host_engine(30)
@test_utils.run_with_nvsdm_mocked_cx()
def test_dcgmi_discovery_can_list_cx_mocked(handle, cxIds):
    helper_dcgmi_discovery_can_list_cx(len(cxIds))

def get_nvidia_cpu_count():
    try:
        with open("/sys/devices/soc0/soc_id") as f:
            socId = f.read()
    except:
        return 0

    if not socId.startswith("jep106:036b:"):
        return 0

    coreSiblingsSet = set()
    coreSiblingsPaths = glob.iglob("/sys/devices/system/cpu/cpu*/topology/core_siblings")
    for coreSiblingsPath in coreSiblingsPaths:
        try:
            with open(coreSiblingsPath) as f:
                coreSiblings = f.read()
                coreSiblingsSet.add(coreSiblings)
        except:
            return 0

    return len(coreSiblingsSet)

@test_utils.run_with_persistence_mode_on()
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_diag(handle, gpuIds):
    """
    Test DCGMI diagnostics
    """
    allGpusCsv = ",".join(map(str,gpuIds))
    ## keep args in this order. Changing it may break the test

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    createdGroup = dcgmSystem.GetGroupWithGpuIds("capoo_as_group_name", gpuIds)
    createdGroupId = createdGroup.GetId().value
    gpu0Info = dcgmSystem.discovery.GetGpuAttributes(gpuIds[0])
    nvidiaCpuCount = get_nvidia_cpu_count()

    pciTestParameters = "pcie.h2d_d2h_single_unpinned.min_pci_width=1"
    pciTestParameters += ";pcie.h2d_d2h_single_pinned.min_pci_width=1"
    #Need to skip checks for down NvLinks or QA will file bugs
    if test_utils.are_any_nvlinks_down:
        pciTestParameters += ";pcie.test_nvlink_status=false"

    pciTestCmdLineArgs = ["diag", "--run", "pcie", "-p", pciTestParameters, "-i", str(gpuIds[0])]

    validArgsTestList = [
           ["diag", "--run", "1", "-i", allGpusCsv], # run diagnostic other settings currently run for too long
           ["diag", "--run", "1", "-i", str(gpuIds[0]), "--debugLogFile aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.txt"], # Test that we can pass a long debugLogFile
           ["diag", "--run", "1", "-i", allGpusCsv, "--parameters", "diagnostic.test_duration=30", "--fail-early"], # verifies --fail-early option
           ["diag", "--run", "1", "-i", allGpusCsv, "--parameters", "diagnostic.test_duration=30", "--fail-early", "--check-interval", "3"], # verifies --check-interval option
           ["diag", "--run", "1", "-i", allGpusCsv, "--parameters", "memtest.test_duration=10;diagnostic.test_duration=10"], # verifies multiple parameters with different test name but same variable name
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "HW_SLOWDOWN"], # verifies that --throttle-mask with HW_SLOWDOWN reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "SW_THERMAL"], # verifies that --throttle-mask with SW_THERMAL reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "HW_THERMAL"], # verifies that --throttle-mask with HW_THERMAL reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "HW_POWER_BRAKE"], # verifies that --throttle-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "HW_SLOWDOWN,SW_THERMAL,HW_POWER_BRAKE"], # verifies that --throttle-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "SW_THERMAL,HW_THERMAL,HW_SLOWDOWN"], # verifies that --throttle-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "40"], # verifies that --throttle-mask with HW_SLOWDOWN (8) and SW_THERMAL (32) reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "96"], # verifies that --throttle-mask with SW_THERMAL (32) and HW_THERMAL (64) reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--throttle-mask", "232"], # verifies that --throttle-mask with ALL reasons to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "HW_SLOWDOWN"], # verifies that --clocksevent-mask with HW_SLOWDOWN reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "SW_THERMAL"], # verifies that --clocksevent-mask with SW_THERMAL reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "HW_THERMAL"], # verifies that --clocksevent-mask with HW_THERMAL reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "HW_POWER_BRAKE"], # verifies that --clocksevent-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "HW_SLOWDOWN,SW_THERMAL,HW_POWER_BRAKE"], # verifies that --clocksevent-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "SW_THERMAL,HW_THERMAL,HW_SLOWDOWN"], # verifies that --clocksevent-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "40"], # verifies that --clocksevent-mask with HW_SLOWDOWN (8) and SW_THERMAL (32) reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "96"], # verifies that --clocksevent-mask with SW_THERMAL (32) and HW_THERMAL (64) reason to be ignored
           ["diag", "--run", "1", "-i", allGpusCsv, "--clocksevent-mask", "232"], # verifies that --clocksevent-mask with ALL reasons to be ignored

           ["diag", "--run", "1", "--gpuList", ",".join(str(x) for x in gpuIds)], # verifies --gpuList option accepts and validates list of GPUs passed in
           ["diag", "--run", "1", "--entity-id", "gpu:"+",".join(str(x) for x in gpuIds)], # verifies --entity-id option accepts and validates list of GPUs passed in
           ["diag", "--run", "1", "--entity-id", "gpu:"+"{" + str(gpuIds[0]) + "}"], # verifies --entity-id option accepts and validates list of GPUs passed in
           ["diag", "--run", "1", "--entity-id", "gpu:"+"{" + str(gpuIds[0]) + "-" + str(gpuIds[0]) + "}"], # verifies --entity-id option accepts and validates list of GPUs passed in
           ["diag", "--run", "1", "--entity-id", gpu0Info.identifiers.uuid], # verifies --entity-id option accepts GPU UUID
           ["diag", "--run", "1", "-g", str(createdGroupId)], # verifies --group can work
           pciTestCmdLineArgs,
    ]

    ## keep args in this order. Changing it may break the test
    invalidArgsTestList = [
            ["diag", "--run", "-g", "2"],           # Can't run on group that doesn't exist
            ["diag", "--run", "5"],                 # Can't run with a test number that doesn't exist
            ["diag", "--run", "\"roshar stress\""], # Can't run a non-existent test name
            ["diag", "--run", "\"pcie\"", "--entity-id", "cpu:0"], # Can't run an existing test but not for CPU
            ["diag", "--run", "3", "--parameters", "dianarstic.test_duration=40"],
            ["diag", "--run", "3", "--parameters", "diagnostic.test_durration=40"],
            ["diag", "--run", "3", "--parameters", "pcie.h2d_d2h_singgle_pinned.iterations=4000"],
            ["diag", "--run", "3", "--parameters", "pcie.h2d_d2h_single_pinned.itterations=4000"],
            ["diag", "--run", "3", "--parameters", "bob.tom=maybe"],
            ["diag", "--run", "3", "--parameters", "truck=slow"],
            ["diag", "--run", "3", "--parameters", "now this is a story all about how"],
            ["diag", "--run", "3", "--parameters", "my=life=got=flipped=turned=upside=down"],
            ["diag", "--run", "3", "--parameters", "and.i'd.like.to.take.a=minute=just.sit=right=there"],
            ["diag", "--run", "3", "--parameters", "i'll tell you=how.I.became the=prince of .a town called"],
            ["diag", "--run", "3", "--parameters", "Bel-Air"],
            ["diag", "--train"], # ensure that training is no longer supported
            ["diag", "--run", "1", "-i", allGpusCsv, "--parameters", "diagnostic.test_duration=30", "--fail-early 10"], # verifies --fail-early does not accept parameters
            ["diag", "--run", "1", "--parameters", "diagnostic.test_duration=30", "--fail-early", "--check-interval -1"], # no negative numbers allowed
            ["diag", "--run", "1", "--parameters", "diagnostic.test_duration=30", "--fail-early", "--check-interval 350"], # no numbers > 300 allowed
            ["diag", "--run", "1", "--parameters", "diagnostic.test_duration=30", "--check-interval 10"], # requires --fail-early parameter
            # The tests below are disabled until bug http://nvbugs/2672193 is fixed
            ["diag", "--run", "1", "--throttle-mask", "HW_ZSLOWDOWN"], # verifies that --throttle-mask incorrect reason does not work
            ["diag", "--run", "1", "--throttle-mask", "SW_TRHERMAL"], # verifies that --throttle-mask incorrect reason does not work
            ["diag", "--run", "1", "--throttle-mask", "HWT_THERMAL"], # verifies that --throttle-mask incorrect reason does not work
            ["diag", "--run", "1", "--throttle-mask", "HW_POWER_OUTBRAKE"], # verifies that --throttle-mask incorrect reason does not work
            ["diag", "--run", "1", "--throttle-mask -10"], # verifies that --throttle-mask does not accept incorrect values for any reasons to be ignored
            ["diag", "--run", "1", "--clocksevent-mask", "HW_ZSLOWDOWN"], # verifies that --clocksevent-mask incorrect reason does not work
            ["diag", "--run", "1", "--clocksevent-mask", "SW_TRHERMAL"], # verifies that --clocksevent-mask incorrect reason does not work
            ["diag", "--run", "1", "--clocksevent-mask", "HWT_THERMAL"], # verifies that --clocksevent-mask incorrect reason does not work
            ["diag", "--run", "1", "--clocksevent-mask", "HW_POWER_OUTBRAKE"], # verifies that --clocksevent-mask incorrect reason does not work
            ["diag", "--run", "1", "--clocksevent-mask -10"], # verifies that --clocksevent-mask does not accept incorrect values for any reasons to be ignored
            ["diag", "--run", "1", "--plugin-path", "/usr/libexec/datacenter-gpu-manager-4/unplugins"], # verifies --plugin-path fails if the plugins path is not specified correctly
            # ["diag", "--run", "1", "--gpuList", "0-1-2-3-4-5"], # verifies --gpuList option accepts and validates list of GPUs passed in (disabled until http://nvbugs/2733071 is fixed)
            ["diag", "--run", "1", "--gpuList", "-1,-2,-3,-4,-5"], # verifies --gpuList option accepts and validates list of GPUs passed in
            ["diag", "--run", "1", "--gpuList", "a,b,c,d,e"], # verifies --gpuList option accepts and validates list of GPUs passed in
            ["diag", "--run", "1", "-i", "0-1-2-3-4"], # Make sure -i is a comma-separated list of integers
            ["diag", "--run", "1", "-i", "roshar"], # Make sure -i is a comma-separated list of integers
            ["diag", "--run", "1", "-i", "a,b,c,d,e,f"], # Make sure -i is a comma-separated list of integers
            ["diag", "--run", "1", "-i", allGpusCsv, "--plugin-path", "./apps/nvvs/plugins"], # verifies --plugin-path no longer works (removed)
            ["diag", "--run", "1", "--iterations", "0"], # We cannot run 0 iterations
            ["diag", "--run", "1", "--iterations", r"\-1"], # We cannot run negative iterations
            ["diag", "--run", "1", "--parameters", "diagnostic.test_duration=30;diagnostic.test_duration=40"], # multiple defined parameters not allowed
            ["diag", "--run", "diagnostic", "--parameters", "diagnostic.test_duration=10", "timeout", "10"], # test_duration equals the timeout
            ["diag", "--run", "diagnostic,targeted_power", "--parameters", "diagnostic.test_duration=10,targeted_power.test_duration=10", "timeout", "15"], # combined test_duration exceeds the timeout
            ["diag", "--run", "1", "--entity-id", "a,b,c,d,e"], # Test --entity-id option with invalid gpu id
            ["diag", "--run", "1", "--entity-id", "-1,-2,-3,-4,-5"], # Test --entity-id option with invalid gpu id
            ["diag", "--run", "1", "--entity-id", "dcgm:0,1"], # Test --entity-id option with invalid gpu id
            ["diag", "--run", "1", "--entity-id", "gpu:{0"], # Test --entity-id option with invalid gpu id
            ["diag", "--run", "1", "--entity-id", "GPU-00000000-0000-0000-0000-000000000000"], # Test --entity-id with invalid GPU UUID
            ["diag", "--run", "1", "--entity-id", "core:0"], # Test --entity-id with non-GPU and non-CPU entity
            ["diag", "--run", "1", "--entity-id", "cpu:0,cpu:0"], # Test --entity-id with duplicated CPU entities
            ["diag", "--run", "1", "--entity-id", "core:*"], # Test --entity-id with non-supported wildcard entity type
            ["diag", "--run", "1", "--entity-id", ""], # Test empty --entity-id
            ["diag", "--run", "1", "--entity-id", "6" * (dcgm_structs.DCGM_ENTITY_ID_LIST_LEN + 1)], # Test too large --entity-id
            ["diag", "--run", "1", "--entity-id", str(gpuIds[0]), "--gpuList", str(gpuIds[0])], # Test --entity-id and --gpuList cannot use in the same time
            ["diag", "--run", "1", "--entity-id", str(gpuIds[0]), "--group", str(createdGroupId)], # Test --entity-id and --group cannot use in the same time
            ["diag", "--run", "1", "--gpuList", str(gpuIds[0]), "--group", str(createdGroupId)], # Test --gpuList and --group cannot use in the same time
            ["diag", "--run", "3", "--parameters", f"pcie.h2d_d2h_singgle_pinned.iterations={'6'*(dcgm_structs.DCGM_MAX_TEST_PARMS_LEN_V2 - len('pcie.h2d_d2h_singgle_pinned.iterations=') - 1)}"], # Test case parameter length limit exceeded
            ["diag", "--run", "1", "--expectedNumEntities", "gpu:{0"], # Test --expectedNumEntities option with invalid num gpus
            ["diag", "--run", "1", "--expectedNumEntities", "gpu:"], # Test --expectedNumEntities option with invalid num gpus
            ["diag", "--run", "1", "--expectedNumEntities", "gpu:0,cpu:1"], # Test --expectedNumEntities option with invalid num gpus
            ["diag", "--run", "1", "--expectedNumEntities", "gpu:-1"], # Test --expectedNumEntities option with invalid num gpus
            ["diag", "--run", "1", "-f", "0,1", "--expectedNumEntities", f"gpu:{len(gpuIds)}"], # Test --expectedNumEntities option cannot be used with fake gpus
            ["diag", "--run", "1", "--gpuList", allGpusCsv, "--expectedNumEntities", f"gpu:{len(gpuIds)}"], # Test --expectedNumEntities option cannot be used with gpu list
    ]

    if nvidiaCpuCount != 0:
        entityIdArg = f"gpu:{str(gpuIds[0])}"
        for cpuId in range(nvidiaCpuCount):
            entityIdArg += f",cpu:{cpuId}"
        # Test --entity-id with GPU and CPU entity
        # Since Grace EUD lacks an option to exclude the CPU in testing scenarios,
        # it becomes necessary to enumerate and indicate all processors within the system to initiating test.
        # Skip testing this case if we fail to find out the number of physical CPUs.
        validArgsTestList.append(["diag", "--run", "1", "--entity-id", f"{entityIdArg}"])

        # Test --entity-id with non-existed CPU entity
        # The presence of a CPU is only verified within Nvidia environment.
        invalidArgsTestList.append(["diag", "--run", "1", "--entity-id", "cpu:5566"])
    else:
        # Test --entity-id with non Nvidia CPU entity
        invalidArgsTestList.append(["diag", "--run", "1", "--entity-id", "cpu:0"])

    # for_all_same_sku_gpus() protected us from running on heterogeneous GPUs in a single diag run.
    # We can detect that this decorator is in effect by comparing the gpuIds we're running against vs all of the gpuIds in the system
    # Don't test the --entity-id * if it will just return DCGM_ST_GROUP_INCOMPATIBLE
    allGpusInSystem = dcgmSystem.discovery.GetAllGpuIds()
    homogeneousSystemTestList = [
        ["diag", "--run", "1", "--entity-id", "*"], # verifies --entity-id option accepts all GPUs via wildcard
        ["diag", "--run", "1", "--expectedNumEntities", f"gpu:{len(gpuIds)}"], # verifies --expectedNumEntities works when all GPUs are present
        ["diag", "--run", "1", "-g", str(dcgm_structs.DCGM_GROUP_ALL_GPUS), "--expectedNumEntities", f"gpu:{len(gpuIds)}"], # verifies --expectedNumEntities works with the DCGM_GROUP_ALL_GPUS groupId
        ["diag", "--run", "1", "--expectedNumEntities", f"gpu:0"], # verifies --expectedNumEntities with value 0 is ignored and does not error
    ]
    if len(allGpusInSystem) == len(gpuIds):
        validArgsTestList.extend(homogeneousSystemTestList)
    else:
        invalidArgsTestList.extend(homogeneousSystemTestList)

    _test_valid_args(validArgsTestList)
    _test_invalid_args(invalidArgsTestList)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_persistence_mode_on()
@test_utils.run_with_injection_nvml_using_specific_sku('A100x4-and-DGX.yaml')
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_diag_invalid_test_specified(handle, gpuIds):

    def verifyCliOutput(testNames):
        testNames = ','.join(testNames)
        cliArgs = [ 'diag', '--run', testNames, '-i', ','.join(map(str, gpuIds)) ]
        _, stdoutLines, stderrLines = _run_dcgmi_command(cliArgs)
        combinedResult = '\n'.join(stdoutLines + stderrLines)
        assert 'Unable to complete diagnostic' in combinedResult
        assert 'Error: requested test(s)' in combinedResult and 'were not found' in combinedResult
        for testName in testNames.split(','):
            assert testName in combinedResult, f'{testName} not found in {combinedResult}'

    def verifyJsonOutput(testNames):
        testNames = ','.join(testNames)
        jsonArgs = [ 'diag', '--run', testNames, '-j', '-v', '-i', ','.join(map(str, gpuIds)) ]
        _, stdoutLines, stderrLines = _run_dcgmi_command(jsonArgs)
        combinedResult = '\n'.join(stdoutLines + stderrLines)
        jsondata = json.loads(combinedResult)
        assert 'DCGM Diagnostic' in jsondata
        assert 'runtime_error' in jsondata['DCGM Diagnostic']
        runtimeError = jsondata['DCGM Diagnostic']['runtime_error']
        assert 'Unable to complete diagnostic' in runtimeError
        assert 'Error: requested test(s)' in runtimeError and 'were not found' in runtimeError
        for testName in testNames.split(','):
            assert testName in runtimeError, f'{testName} not found in {runtimeError}'

    verifyCliOutput(['roshar_stress'])
    verifyCliOutput(['roshar_stress', 'targeted_moash'])

    verifyJsonOutput(['capoo_power'])
    verifyJsonOutput(['capoo_power', 'dogdog_stress'])

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100x4-and-DGX.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_diag_on_heterogeneous_env(handle, gpuIds):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    allGpusInSystem = dcgmSystem.discovery.GetAllGpuIds()

    # for_all_same_sku_gpus() protected us from running on heterogeneous GPUs in a single diag run.
    validArgsTestList = [
        ["diag", "--run", "1", "--gpuList", ",".join(str(x) for x in gpuIds)],
        ["diag", "--run", "1", "--entity-id", ",".join(str(x) for x in gpuIds)],
    ]
    for args in validArgsTestList:
        _, stdoutLines, stderrLines = _run_dcgmi_command(args)
        allOutput = (' '.join(stdoutLines + stderrLines)).lower()
        hasError = "error" in allOutput
        hasHomogeneous = "homogeneous" in allOutput
        assert (not hasError or not hasHomogeneous), "Should work in homogeneous group of GPUs."

    invalidArgsTestList = [
        ["diag", "--run", "1", "--gpuList", ",".join(str(x) for x in allGpusInSystem)],
        ["diag", "--run", "1", "--entity-id", ",".join(str(x) for x in allGpusInSystem)],
        ["diag", "--run", "1", "--entity-id", "*"],
        ["diag", "--run", "1"],
    ]
    for args in invalidArgsTestList:
        _, stdoutLines, stderrLines = _run_dcgmi_command(args)
        # expected allOutput: error: diagnostic can only be performed on a homogeneous group of gpus.
        allOutput = (' '.join(stdoutLines + stderrLines)).lower()
        hasError = "error" in allOutput
        hasHomogeneous = "homogeneous" in allOutput
        assert (hasError and hasHomogeneous), "Should not work in heterogeneous group of GPUs."

@test_utils.run_with_persistence_mode_on()
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_multiple_iterations(handle, gpuIds):
    allGpusCsv = ",".join(map(str,gpuIds))
    args = ["diag", "-r", "1", "-j", "-i", allGpusCsv, "--iterations", "3"]
    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(args)

    assert retValue == 0, "Expected successful execution, but got %d" % retValue
    rawtext = ""
    for line in stdout_lines:
        rawtext = rawtext + line + "\n"
    
    try:        
        jsondict = json.loads(rawtext)
        overallResult = jsondict["Overall Result"]
        assert overallResult != None, "Didn't find a populated value for the overall result!"
        iterationArray = jsondict["iterations"]
        for i in range(0,2):
            assert iterationArray[i] != None, "Didn't find a populated result for run %d" % i+1
    except ValueError as e:
        assert False, ("Couldn't parse json from '%s'" % rawtext)
    

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgmi_stats(handle, gpuIds):
    """
     Test DCGMI Stats
    """
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["stats", "--enable"],                  # Enable watches
            #["stats", "--pid", "100"],                  # check pid
            #["stats", "--pid", "100", "--verbose"],     # check pid verbose (run test process and enable these if wanted)

            ["stats", "--jstart", "1"],             #start a job with Job ID 1
            ["stats", "--jstop", "1"],              #Stop the job
            ["stats", "--job", "1"],                #Print stats for the job
            ["stats", "--jremove", "1"],            #Remove the job the job
            ["stats", "--jremoveall"],              #Remove all jobs
            ["stats", "--disable"],                 #disable watches

            ["stats", "--jstart", "1"],             #start another job with Job ID 1. This should work due to jremove above. Also, setup the jstart failure below
    ])

    #Create group for testing
    groupId = str(_create_dcgmi_group())
    nonExistantGroupId = str(int(groupId) + 10)
      
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["stats", "--pid", "100", "-g", groupId],            # Can't view stats with out watches enabled
            ["stats", "enable", "-g", nonExistantGroupId],       # Can't enable watches on group that doesn't exist
            ["stats", "disable", "-g", nonExistantGroupId],      # Can't disable watches on group that doesn't exist
            ["stats", "--jstart", "1"],                          # Cant start the job with a job ID which is being used
            ["stats", "--jstop", "3"],                           # Stop an invalid job id
            ["stats", "--jremove", "3"],                         # Remove an invalid job id
            ["stats", "--job", "3"]                              # Get stats for an invalid job id
    ])
        
@test_utils.run_with_standalone_host_engine(20, "127.0.0.1:5545", ["--port", "5545"])
def test_dcgmi_port(handle):
    """
    Test DCGMI port - does dcgmi group testing using port 5545
    """
    
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["group", "--host", "localhost:5545", "-l", ""],      # list groups
    ])
    
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["group", "--host", "localhost:5545", "-c", "--default"],      # Can't create a group called --default
    ])

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_nvml()
def test_dcgmi_field_groups(handle):
    """
    Test DCGMI field groups - test the dcgmi commands under "fieldgroup"
    """

    _test_valid_args([
        ["fieldgroup", "-l"],
        ["fieldgroup", "-i", "-g", "1"],                    # show internal field group
        ["fieldgroup", "-c", "my_group", "-f", "1,2,3"],    # Add a field group
    ])

    _test_invalid_args([
        ["fieldgroup", "-c", "my_group", "-f", "1,2,3"],      # Add a duplicate group
        ["fieldgroup", "-c", "bad_fieldids", "-f", "999999"], # Pass bad fieldIds 
        ["introspect", "-d", "-g", "1"],                      # Delete internal group. Bad
        ["introspect", "-i", "-g", "100000"],                 # Info for invalid group
    ])

@test_utils.run_with_standalone_host_engine()
def test_dcgmi_introspect(handle):
    """
    Test DCGMI introspection - test the dcgmi commands under "introspection"
    """
    
    _test_valid_args([
        ["introspect", "--show", "--hostengine"],           # show hostengine
        ["introspect", "-s", "-H"],                         # short form
    ])
    
    _test_invalid_args([
        ["introspect", "--show"],         # "show" without "--hostengine" should fail
    ])
    
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_only_with_live_gpus()
def test_dcgmi_nvlink(handle, gpuIds):
    """
    Test dcgmi to display nvlink error counts
    """
    
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["nvlink", "-e", "-g", str(gpuIds[0])],  # run the working nvlink command for gpuId[0]
           ["nvlink", "-s"]                         # Link status should work without parameters
    ])
      
    _test_invalid_args([
           ["nvlink","-e"],                         # -e option requires -g option
           ["nvlink","-e -s"]                       # -e and -s are mutually exclusive
    ])


def helper_make_switch_string(switchId):
    return "nvswitch:" + str(switchId)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus(2) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_with_injection_nvswitches(2)
@test_utils.run_with_injection_gpu_instances(2)
@test_utils.run_with_injection_gpu_compute_instances(2)
@test_utils.run_with_injection_cpus(1)
@test_utils.run_with_injection_cpu_cores(1)
def test_dcgmi_dmon(handle, gpuIds, switchIds, instanceIds, ciIds, cpuIds, coreIds):
    """
    Test dcgmi to display dmon values
    """
    gpuGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT))
    switchGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT_NVSWITCHES))

    logger.info("Injected switch IDs:" + str(switchIds))

    # Creates a comma separated list of gpus
    allGpusCsv = ",".join(map(str,gpuIds))
    allInstancesCsv = ",".join([("instance:" + str(x)) for x in instanceIds])
    # All compute instances
    allCisCsv = ",".join([("ci:" + str(x)) for x in ciIds])
    #Same for switches but predicate each one with nvswitch
    allSwitchesCsv = ",".join(map(helper_make_switch_string,switchIds))

    switchFieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT
    cpuFields = dcgm_fields.DCGM_FI_DEV_CPU_UTIL_USER

    #Inject a value for a field for each switch so we can retrieve it
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = switchFieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()-5) * 1000000.0) #5 seconds ago
    field.value.i64 = 0
    for switchId in switchIds:
        linkId = (dcgm_fields.DCGM_FE_SWITCH << 0) | (switchId << 40) | (1 << 8)
        ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_LINK, linkId, field)

    _test_valid_args([
        ["dmon", "-e", "150,155","-c","1"],                                             # run the dmon for default gpu group.
        ["dmon", "-e", "150,155","-c","1","-g",gpuGroupId],                             # run the dmon for a specified gpu group
        ["dmon", "-e", "150,155","-c","1","-g",'all_gpus'],                             # run the dmon for a specified group
        ["dmon", "-e", str(switchFieldId),"-c","1","-g",'all_nvswitches'],              # run the dmon for a specified group - Reenable after DCGM-413 is fixed
        ["dmon", "-e", str(switchFieldId),"-c","1","-g",switchGroupId],                 # run the dmon for a specified group
        ["dmon", "-e", "150,155","-c","1","-d","2000"],                                 # run the dmon for delay mentioned and default gpu group.
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i",allGpusCsv],                 # run the dmon for devices mentioned and mentioned delay.
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i",allInstancesCsv],
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i",allCisCsv],
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i",allGpusCsv + "," + allInstancesCsv + "," + allCisCsv],
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i","*"],                        # run the dmon for all GPUs via wildcard
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i","*/*"],                      # run the dmon for all GPU Instances via wildcards
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i","*/*/*"],                    # run the dmon for all Compute Instances via wildcards
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i","*,*/*,*/*/*"],              # run the dmon for all entities via wildcards
        ["dmon", "-e", str(switchFieldId),"-c","1","-d","2000","-i",allSwitchesCsv],    # run the dmon for devices mentioned and mentioned delay.
    ])

    #Run tests that take a gpuId as an argument
    for gpu in gpuIds:
        _test_valid_args([
               ["dmon", "-e", "150","-c","1","-i",str(gpu)],        # run the dmon for one gpu.
               ["dmon", "-e", "150","-c","1","-i",'gpu:'+str(gpu)], # run the dmon for one gpu, tagged as gpu:.
               ["dmon", "-e", "150","-c","1","-i",str(gpu)],        # run the dmon for mentioned devices and count value.
               ["dmon", "-e", "150,155","-c","1","-i",str(gpu)],    # run the dmon for devices mentioned, default delay and field values that are provided.
        ])
    
    #Run tests that take a nvSwitch as an argument
    for switchId in switchIds:
        _test_valid_args([
               ["dmon", "-e", str(switchFieldId),"-c","1","-i",'nvswitch:'+str(switchId)], # run the dmon for one nvswitch, tagged as nvswitch:.
        ])

    hugeGpuCsv = ",".join(map(str,list(range(0, dcgm_structs.DCGM_MAX_NUM_DEVICES*2, 1))))

    _test_invalid_args([
           ["dmon","-c","1"],                                                    # run without required fields.
           ["dmon", "-e", "-150","-c","1","-i","1"],                             # run with invalid field id.
           ["dmon", "-e", "150","-c","1","-i","-2"],                             # run with invalid gpu id.
           ["dmon", "-e", "150","-c","1","-i","gpu:999"],                        # run with invalid gpu id.
           ["dmon", "-e", "150","-c","1","-g","999"],                            # run with invalid group id.
           ["dmon", "-i", hugeGpuCsv, "-e", "150", "-c", "1"],                   # run with invalid number of devices.
           ["dmon", "-i", "instance:2000", "-e", "150", "-c", "1"],              # run with invalid gpu_i
           ["dmon", "-i", "ci:2000", "-e", "150", "-c", "1"],                    # run with invalid gpu_ci
           ["dmon", "-e", "150","f","0","-c","1","-i","0,1,765"],                # run with invalid device id (non existing id).
           ["dmon", "-e", "150","-c","-1","-i","1"],                             # run with invalid count value.
           ["dmon", "-e", "150","-c","1","-i","1","-d","-1"],                    # run with invalid delay (negative value).
           ["dmon", "-f", "-9","-c","1","-i","1","-d","10000"],                  # run with invalid field Id.
           ["dmon", "-f", "150","-c", "1", "-i","0", "-d", "99" ],               # run with invalid delay value.
           ["dmon", "-e", str(cpuFields), "-i", "cpu:0", "-g", "0", "-c", "1"],  # run dmon for CPUs and a group
           ["dmon", "--gpu-id", "1"],                                            # run with invalid and obsolete --gpu-id arg.
           ["dmon", "--group-id", "1", "--entity-id", "1", "--field-id", "1"],   # run with both group id and entity id
    ])

    # Run tests that take several entities
    entityStr = "%d,nvswitch:%d,cpu:%d,core:%d" % (gpuIds[0], switchIds[0], cpuIds[0], coreIds[0])
    _test_valid_args([
           ["dmon", "-e", "150,%s" % str(cpuFields), "-i", entityStr, "-c", "1"], # run dmon for CPUs and a group
    ])

@test_utils.run_only_on_numa_systems()
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_injection_gpus(2) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_only_with_live_cpus()
def test_dcgmi_dmon_cpu(handle, gpuIds, cpuIds):
    cpuFields = dcgm_fields.DCGM_FI_DEV_CPU_UTIL_USER

    allGpusCsv = ",".join(map(str,gpuIds))
    cpuGpuCsv = ",".join(["cpu:0", allGpusCsv])

    _test_valid_args([
        ["dmon", "-e", str(cpuFields), "-i", "cpu:0", "-c", "1"],   # run dmon for CPUs
        ["dmon", "-e", str(cpuFields), "-i", cpuGpuCsv, "-c", "1"], # run dmon for CPUs and GPUs
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.run_with_injection_nvswitches(2)
def test_dcgmi_nvlink_nvswitches(handle, gpuIds, switchIds):
    """
    Test dcgmi to display dmon values
    """
    gpuGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT))
    switchGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT_NVSWITCHES))

    logger.info("Injected switch IDs:" + str(switchIds))

    _test_valid_args([
           ["nvlink", "-s"]                       # Link status should work without parameters
    ])

    # Creates a comma separated list of gpus
    allGpusCsv = ",".join(map(str,gpuIds))
    #Same for switches but predicate each one with nvswitch
    allSwitchesCsv = ",".join(map(helper_make_switch_string,switchIds))

    switchFieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX

    #Inject a value for a field for each switch so we can retrieve it
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = switchFieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()-5) * 1000000.0) #5 seconds ago
    field.value.i64 = 0
    for switchId in switchIds:
        ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH, 
                                                             switchId, field)

    _test_valid_args([
        ["dmon", "-e", "150,155","-c","1"],                          # run the dmon for default gpu group.
        ["dmon", "-e", "150,155","-c","1","-g",gpuGroupId],          # run the dmon for a specified gpu group
        ["dmon", "-e", "150,155","-c","1","-g",'all_gpus'],          # run the dmon for a specified group
        ["dmon", "-e", str(switchFieldId),"-c","1","-g",'all_nvswitches'], # run the dmon for a specified group - Reenable after DCGM-413 is fixed
        ["dmon", "-e", str(switchFieldId),"-c","1","-g",switchGroupId], # run the dmon for a specified group
        ["dmon", "-e", "150,155","-c","1","-d","2000"],              # run the dmon for delay mentioned and default gpu group. 
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i",allGpusCsv], # run the dmon for devices mentioned and mentioned delay.
        ["dmon", "-e", str(switchFieldId),"-c","1","-d","2000","-i",allSwitchesCsv] # run the dmon for devices mentioned and mentioned delay.
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgmi_modules(handle, gpuIds):
    """
    Test DCGMI modules 
    """

    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["modules", "--list"],
           ["modules", "--denylist", "5"],
           ["modules", "--denylist", "policy"],
    ])
      
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["modules", "--list", "4"],
            ["modules", "--denylist", "20"],
            ["modules", "--denylist", "notamodule"],
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_profile(handle, gpuIds):
    """
    Test DCGMI "profile" subcommand
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    # Creates a comma separated list of gpus
    allGpusCsv = ",".join(map(str,gpuIds))

    #See if these GPUs even support profiling. This will bail out for non-Tesla or Pascal or older SKUs
    try:
        supportedMetrics = dcgmGroup.profiling.GetSupportedMetricGroups()
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_PROFILING_NOT_SUPPORTED) as e:
        test_utils.skip_test("Profiling is not supported for gpuIds %s" % str(gpuIds))
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED) as e:
        test_utils.skip_test("The profiling module could not be loaded")
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED) as e:
        test_utils.skip_test("The profiling module is not supported")

    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["profile", "--list", "-i", allGpusCsv],
           ["profile", "--list", "-g", str(dcgmGroup.GetId().value)],
    ])
      
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["profile", "--list", "--pause", "--resume"], #mutually exclusive flags
            ["profile", "--pause", "--resume"], #mutually exclusive flags
            ["profile", "--list", "-i", "999"], #Invalid gpuID
            ["profile", "--list", "-i", allGpusCsv + ",taco"], #Invalid gpu at end
            ["profile", "--list", "-g", "999"], #Invalid group
            ["profile", "--gpu-id", "1"], #Invalid and obsolete --gpu-id
            ["profile", "--group-id", "1", "--entity-id", "1", "--list"], #Both group id and entity id together
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_profile_affected_by_gpm(handle, gpuIds):
    """
    Test DCGMI "profile" subcommands that are affected by if GPM works or not
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    #See if these GPUs even support profiling. This will bail out for non-Tesla or Pascal or older SKUs
    try:
        supportedMetrics = dcgmGroup.profiling.GetSupportedMetricGroups()
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_PROFILING_NOT_SUPPORTED) as e:
        test_utils.skip_test("Profiling is not supported for gpuIds %s" % str(gpuIds))
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED) as e:
        test_utils.skip_test("The profiling module could not be loaded")
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED) as e:
        test_utils.skip_test("The profiling module is not supported")

    ## keep args in this order. Changing it may break the test
    pauseResumeArgs = [
        ["profile", "--pause"], #Pause followed by resume
        ["profile", "--resume"],
        ["profile", "--pause"], #Double pause and double resume should be fine
        ["profile", "--pause"],
        ["profile", "--resume"],
        ["profile", "--resume"],
    ]

    #GPM GPUs don't support pause/resume since monitoring and profiling aren't mutually exclusive anymore
    if test_utils.gpu_supports_gpm(handle, gpuIds[0]):
        _test_invalid_args(pauseResumeArgs)
    else:
        _test_valid_args(pauseResumeArgs)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgmi_test_introspect(handle, gpuIds):
    """
    Test "dcgmi test --introspect"
    """
    oneGpuIdStr = str(gpuIds[0])
    gpuGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT))
    gpuGroupIdStr = str(gpuGroupId)

    fieldIdStr = str(dcgm_fields.DCGM_FI_DEV_ECC_CURRENT) #Use this field because it's watched by default in the host engine

    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["test", "--introspect", "--gpuid", oneGpuIdStr, "--field", fieldIdStr],
           ["test", "--introspect", "-g", gpuGroupIdStr, "--field", fieldIdStr],
           ["test", "--introspect", "-g", gpuGroupIdStr, "--field", fieldIdStr],
    ])
      
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["test", "--introspect", "--inject"], #mutually exclusive flags
            ["test", "--introspect", "--gpuid", oneGpuIdStr], #Missing --field
            ["test", "--introspect", "-g", gpuGroupIdStr, "--gpuid", oneGpuIdStr],
            ["test", "--introspect", "--gpuid", "11001001"],
            ["test", "--introspect", "-g", "11001001"],
            ["test", "--introspect", "--group", "11001001"],
            ["test", "--introspect", "-g", gpuGroupIdStr, "--field", "10000000"], #Bad fieldId
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgmi_test_inject(handle, gpuIds):
    """
    Test "dcgmi test --inject"
    """
    oneGpuIdStr = str(gpuIds[0])
    gpuGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT))
    gpuGroupIdStr = str(gpuGroupId)

    fieldIdStr = str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP) 

    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["test", "--inject", "--gpuid", oneGpuIdStr, "--field", fieldIdStr, "-v", '45'],
           ["test", "--inject", "--gpuid", oneGpuIdStr, "--field", fieldIdStr, "--value", '45'],
    ])
      
    ## keep args in this order. Changing it may break the test
    _test_invalid_args([
            ["test", "--inject", "--introspect"], #mutually exclusive flags
            ["test", "--inject", "-g", gpuGroupIdStr], #group ID is not supported
            ["test", "--inject", "--gpuid", oneGpuIdStr], #Missing --field
            ["test", "--inject", "--gpuid", oneGpuIdStr, "--field", fieldIdStr], #Missing --value
            ["test", "--inject", "-g", gpuGroupIdStr, "--gpuid", oneGpuIdStr],
            ["test", "--inject", "--gpuid", "11001001", "--field", fieldIdStr, "--value", '45'], #Bad gpuId
            ["test", "--inject", "-g", "11001001"],
            ["test", "--inject", "--group", "11001001"],
            ["test", "--inject", "--gpuid", oneGpuIdStr, "--field", "10000000", "--value", '45'], #Bad fieldId
    ])


@test_utils.run_with_standalone_host_engine(20)
def test_dcgmi_dmon_pause_resume(handle):
    _test_valid_args([
        ['test', '--pause'],
        ['test', '--resume'],
    ])


@test_utils.run_with_logging_on()
def test_dcgmi_settings_logging_severity():
    if test_utils.loggingLevel != 'DEBUG':
        test_utils.skip_test("Detected logLevel != DEBUG. This test requires DEBUG. Likely cause: --eris option")

    # Env var is automatically set in NvHostEngineApp
    app = apps.NvHostEngineApp()
    app.start(timeout=10)

    contents = None

    # Try for 5 seconds
    for i in range(25):
        time.sleep(0.2)
        with closing(open(app.dcgm_trace_fname, encoding='utf-8')) as f:
            # pylint: disable=no-member
            contents = f.read()
            logger.debug("Read %d bytes from %s" % (len(contents), app.dcgm_trace_fname))
            if 'DEBUG' in contents:
                break

    set_severity_args = ['set', '--logging-severity', 'VERB']
    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(set_severity_args)

    assert retValue == 0, f'retValue = {retValue}, stdout={stdout_lines}'

    passed = False

    for i in range(25):
        time.sleep(0.2)
        with closing(open(app.dcgm_trace_fname, encoding='utf-8')) as f:
            # pylint: disable=no-member
            contents = f.read()
            logger.debug("Read %d bytes from %s" % (len(contents), app.dcgm_trace_fname))
            if 'VERB' in contents:
                passed = True
                break

    # Cleaning up
    app.terminate()
    app.validate()

    assert passed, "Unable to find 'VERB' in log file"

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus(5) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_with_injection_gpu_instances(2,1)
@test_utils.run_with_injection_gpu_instances(2,3)
@test_utils.run_with_injection_gpu_compute_instances(2)
@test_utils.run_with_injection_gpu_compute_instances(2,2)
@test_utils.run_with_injection_nvswitches(2)
def test_dcgmi_global_and_others(handle, gpuIds, instanceIds, ciIds, switchIds):
    """
    Test DCGMI group

    This tests the retrieval of field metrics for both GPU MIG and non-MIG entities, as well as globals.

    Historically, dcgmi would ignore non-MIG entities when a MIG Hierarchy existed.
    """
    # Creates a comma separated list of all gpus
    allGpusCsv = ",".join(map(str,gpuIds))
    # Creates a comma separated list of all GPU instances
    allInstancesCsv = ",".join([("instance:" + str(x)) for x in instanceIds])
    # Creates a comma separated list of all compute instances
    allCisCsv = ",".join([("ci:" + str(x)) for x in ciIds])
    # Creates a comma separated list of all nvswitches
    allNvSwitchCsv= ",".join([("nvswitch:" + str(x)) for x in switchIds])

    # Create a field object to insert into globals, GPUs, GIs, and CIs

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.status = 0
    field.ts = int((time.time()-5) * 1000000.0) #5 seconds ago

    # Insert GLOBALS into a single GPU -- will appear in all of them

    globalData = []

    globalData.append({ "type" : ord(dcgm_fields.DCGM_FT_STRING),
                        "fieldId" : dcgm_fields.DCGM_FI_DRIVER_VERSION,
                        "value" : "535.27" })

    globalData.append({ "type" : ord(dcgm_fields.DCGM_FT_STRING),
                        "fieldId" : dcgm_fields.DCGM_FI_NVML_VERSION,
                        "value" : "12.535.27" })

    globalData.append({ "type" :  ord(dcgm_fields.DCGM_FT_STRING),
                        "fieldId" : dcgm_fields.DCGM_FI_PROCESS_NAME,
                        "value" : "./apps/amd64/nv-hostengine" })

    globalData.append({ "type" : ord(dcgm_fields.DCGM_FT_INT64),
                        "fieldId" : dcgm_fields.DCGM_FI_DEV_COUNT,
                        "value" : 8 })

    globalData.append({ "type" : ord(dcgm_fields.DCGM_FT_INT64),
                        "fieldId" : dcgm_fields.DCGM_FI_CUDA_DRIVER_VERSION,
                        "value" : 12020 })

    for data in globalData:
        field.fieldType = data["type"]
        field.fieldId = data["fieldId"]

        if field.fieldType == ord(dcgm_fields.DCGM_FT_STRING):
            field.value.str = data["value"]

        if field.fieldType == ord(dcgm_fields.DCGM_FT_INT64):
            field.value.i64 = data["value"]

        ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_NONE, gpuIds[0], field)

    # Insert non-GLOBAL data into non-MIG GPUs

    nonGlobalData = []

    """
    We have commented nonGlobalData stuff out since these fields are not
    supported on all GPUs, and may want to selectively add them later for
    those GPUs.

    nonGlobalData.append({ "type" : ord(dcgm_fields.DCGM_FT_DOUBLE),
                           "fieldId" : dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
                           "value" : 50.000 })

    nonGlobalData.append({ "type" : ord(dcgm_fields.DCGM_FT_INT64),
                           "fieldId" : dcgm_fields.DCGM_FI_DEV_COMPUTE_MODE,
                           "value" : 0 })

    nonGlobalData.append({ "type" : ord(dcgm_fields.DCGM_FT_DOUBLE),
                           "fieldId" : dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
                           "value" : 90.000 })

    for gpu in [gpuIds[0], gpuIds[2], gpuIds[4]]:
        for data in nonGlobalData:
            field.fieldType = data["type"]
            field.fieldId = data["fieldId"]

            if field.fieldType == ord(dcgm_fields.DCGM_FT_DOUBLE):
                field.value.dbl = data["value"]

            if field.fieldType == ord(dcgm_fields.DCGM_FT_INT64):
                field.value.i64 = data["value"]

            ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_GPU, gpu, field)
    """

    gpuArg = str(gpuIds[0])
    gpuArg += ","+str(gpuIds[1])
    gpuArg += ","+str(gpuIds[2])
    gpuArg += ","+str(gpuIds[3])
    gpuArg += ","+str(gpuIds[4])
    gpuArg += ","+str(gpuIds[1]) + "/*"
    gpuArg += ","+str(gpuIds[1]) + "/*/*"
    gpuArg += ","+str(gpuIds[3]) + "/*"
    gpuArg += ","+str(gpuIds[3]) + "/*/*"
    gpuArg += ","+allNvSwitchCsv

    fieldArg = str(dcgm_fields.DCGM_FI_DRIVER_VERSION)
    fieldArg += ","+str(dcgm_fields.DCGM_FI_NVML_VERSION)
    fieldArg += ","+str(dcgm_fields.DCGM_FI_PROCESS_NAME)
    fieldArg += ","+str(dcgm_fields.DCGM_FI_DEV_COUNT)
    fieldArg += ","+str(dcgm_fields.DCGM_FI_CUDA_DRIVER_VERSION)
    """
    fieldArg += ","+str(dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE)
    fieldArg += ","+str(dcgm_fields.DCGM_FI_DEV_COMPUTE_MODE)
    fieldArg += ","+str(dcgm_fields.DCGM_FI_PROF_SM_ACTIVE)
    """

    cmdArgs = ["dmon", "-i", gpuArg, "-e", fieldArg, "-c", "1"]

    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(cmdArgs)
    _assert_valid_dcgmi_results(cmdArgs, retValue, stdout_lines, stderr_lines)

    for line in stdout_lines:
        logger.info("stdout: " + line)

    for line in stderr_lines:
        logger.info("stderr: " + line)

    assert len(stderr_lines) == 0, "stderr is not empty"

    lastMatchPosition = stdout_lines[0].find("#Entity")
    assert lastMatchPosition == 0, "#Entity not found starting header"

    outputFormat = dcgm_fields.c_dcgm_field_output_format_t()

    defaultFieldPosition = 11 # should match dcgmi dmon
    outputPadding = 8  # should match dcgmi dmon
    fieldPosition = defaultFieldPosition

    for data in globalData + nonGlobalData:
        fieldId = data["fieldId"]
        fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
        memmove(addressof(outputFormat), fieldMeta.valueFormat, sizeof(outputFormat))
        assert stdout_lines[0].find(outputFormat.shortName) == fieldPosition, outputFormat.shortName + " not in proper place on header"
        assert stdout_lines[1][fieldPosition:].find(outputFormat.unit) == 0, outputFormat.shortName + " unit not in proper place on header"
        fieldPosition += outputFormat.width + outputPadding

    lineHeaders = [ "GPU " + str(gpuIds[0]),
                    "GPU " + str(gpuIds[1]),
                    "GPU-I " + str(instanceIds[0]),
                    "GPU-CI " + str(ciIds[0]),
                    "GPU-I " + str(instanceIds[1]),
                    "GPU-CI " + str(ciIds[1]),
                    "GPU " + str(gpuIds[2]),
                    "GPU " + str(gpuIds[3]),
                    "GPU-I " + str(instanceIds[2]),
                    "GPU-CI " + str(ciIds[2]),
                    "GPU-I " + str(instanceIds[3]),
                    "GPU-CI " + str(ciIds[3]),
                    "GPU " + str(gpuIds[4]),
                    "Switch " + str(switchIds[0]),
                    "Switch " + str(switchIds[1])

    ]

    assert len(stdout_lines) == (len(lineHeaders) + 2), "Should have " + str(len(lineHeaders) + 2) + " output lines, have " + str(len(stdout_lines))

    for index in range(len(lineHeaders)):
        assert stdout_lines[index + 2].find(lineHeaders[index]) == 0, lineHeaders[index] + " not present"

    #  Match MIG GPU data

    firstDataLine = stdout_lines[2]
    fieldPosition = defaultFieldPosition

    for data in globalData:
        fieldId = data["fieldId"]
        fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
        memmove(addressof(outputFormat), fieldMeta.valueFormat, sizeof(outputFormat))
        assert firstDataLine[fieldPosition:].find(str(data["value"])[:outputFormat.width]) == 0, "Field " + outputFormat.shortName + " has wrong value"
        fieldPosition += outputFormat.width + outputPadding

    for data in nonGlobalData:
        type = data["type"]

        if type == ord(dcgm_fields.DCGM_FT_DOUBLE):
            matchValue = "0.000"

        if type == ord(dcgm_fields.DCGM_FT_INT64):
            matchValue = "0"

        fieldId = data["fieldId"]
        fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
        memmove(addressof(outputFormat), fieldMeta.valueFormat, sizeof(outputFormat))
        assert firstDataLine[fieldPosition:].find(matchValue) == 0, "Field " + outputFormat.shortName + " has wrong value"
        fieldPosition += outputFormat.width + outputPadding

    fieldPosition = defaultFieldPosition

    for line in [4, 5, 6, 7, 8, 10, 11, 12, 13, 14]:
        assert firstDataLine[fieldPosition:] == stdout_lines[line][fieldPosition:], "line " + str(line) + " does not match other MIG GPUs"

    firstDataLine = stdout_lines[3]
    fieldPosition = defaultFieldPosition

    # Match non-MIG GPU data.

    for data in globalData:
        fieldId = data["fieldId"]
        fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
        memmove(addressof(outputFormat), fieldMeta.valueFormat, sizeof(outputFormat))
        assert firstDataLine[fieldPosition:].find(str(data["value"])[:outputFormat.width]) == 0, "Field " + outputFormat.shortName + " has wrong value"
        fieldPosition += outputFormat.width + outputPadding

    for data in nonGlobalData:
        type = data["type"]

        if type == ord(dcgm_fields.DCGM_FT_DOUBLE):
            matchValue = "0.000"

        if type == ord(dcgm_fields.DCGM_FT_INT64):
            matchValue = "0"

        fieldId = data["fieldId"]
        fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
        memmove(addressof(outputFormat), fieldMeta.valueFormat, sizeof(outputFormat))
        assert firstDataLine[fieldPosition:].find(str(data["value"])[:outputFormat.width]) == 0, "Field " + outputFormat.shortName + " has wrong value"
        fieldPosition += outputFormat.width + outputPadding

    fieldPosition = defaultFieldPosition

    for line in [9, 15]:
        assert firstDataLine[fieldPosition:] == stdout_lines[line][fieldPosition:], "line " + str(line) + " does not match other non-MIG GPUs"

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_diag_expected_num_entities(handle, gpuIds):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    allGpusInSystem = dcgmSystem.discovery.GetAllGpuIds()

    def create_dcgmi_group_with_entities(entityIds):
        dcgmSystem = dcgmHandle.GetSystem()
        groupObj = dcgmSystem.GetGroupWithGpuIds("testgroup", entityIds)
        return str(groupObj.GetId().value)

    def run_diag_with_args(args, expectError = False, errorString = ""):
        _, stdoutLines, stderrLines = _run_dcgmi_command(args)
        allOutput = (' '.join(stdoutLines + stderrLines)).lower()
        if not expectError:
            assert "successfully ran diagnostic" in allOutput, f"Output of diag command '{args}': {allOutput}"
            return
        hasErrorString = errorString in allOutput
        assert hasErrorString, f"Output of diag command '{args}': {allOutput}"

    newAllGpusGroupId = create_dcgmi_group_with_entities(allGpusInSystem)
    expectedNumEntitiesStr = f"gpu:{str(len(newAllGpusGroupId))}"
    invalidArgsTestList = [
        ["diag", "--run", "1", "--entity-id", "0,1", "--expectedNumEntities", "gpu:2"],
        ["diag", "--run", "1", "--entity-id", "0,1,2,3,4,5,6,7", "--expectedNumEntities", "gpu:8"],
        ["diag", "--run", "1", "--entity-id", "*", "--expectedNumEntities", "gpu:8"],
        ["diag", "--run", "1", "--group", newAllGpusGroupId, "--expectedNumEntities", "gpu:9"],
        ["diag", "--run", "1", "--group", newAllGpusGroupId, "--expectedNumEntities", "gpu:3"],
        ["diag", "--run", "1", "--group", newAllGpusGroupId, "--expectedNumEntities", expectedNumEntitiesStr]
    ]

    for args in invalidArgsTestList:
        run_diag_with_args(args, "error occurred trying to parse the command line")

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_diag_missing_gpu_expected_num_entities(handle, gpuIds):
    """
    This test removes and restores injected nvml GPUs, and verifies that
    - blank field values are inserted when a GPU goes missing
    - expectedNumEntities errors when a GPU goes missing
    """
    fakeCurrentTemperature = 1000
    injectionOffset = 1
    removeGpuId = gpuIds[1]
    updateFreq = 1000000
    maxKeepAge = 600.0 #10 minutes
    maxKeepEntries = 0 #no limit
    fieldId = dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION
    expectedAllGpusStr = "gpu:" + str(len(gpuIds))
    handleObj = pydcgm.DcgmHandle(handle=handle)
    dcgmDiscovery = DcgmSystem.DcgmSystemDiscovery(handleObj)

    # Devices in the H200 YAML are indexed from 0-7
    dev1Uuid = "GPU-1ae4048a-9b19-f6c5-a7ed-1160943cdd18"
    dev6Uuid = "GPU-8c52e150-ab3b-77f7-34c0-107fb2163182"

    def _run_diag_with_expected_entities(expectedEntities, expectError = False, error = dcgm_structs.DCGM_ST_OK):
        runDiagInfo = dcgm_structs.c_dcgmRunDiag_v9()
        runDiagInfo.version = dcgm_structs.dcgmRunDiag_version9
        runDiagInfo.groupId = dcgm_structs.DCGM_GROUP_NULL
        runDiagInfo.entityIds = "*,cpu:*"
        runDiagInfo.expectedNumEntities = expectedEntities
        runDiagInfo.validate = 1

        if expectError:
            with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(error)):
                response = test_utils.action_validate_wrapper(runDiagInfo, handle, dcgm_structs.dcgmRunDiag_version9)
        else:
            response = test_utils.action_validate_wrapper(runDiagInfo, handle, dcgm_structs.dcgmRunDiag_version9)
            assert response, "Should have received a response"
            assert response.tests[0].name == "software", \
                f"The response should have contained the 'software' plugin result, instead got {response.tests[0].name}"

    def _wait_for_gpu_status_change(gpuId, expectedGpuStatus):
        maxWait = 2
        checkInterval = 0.1
        currentGpuStatus = dcgm_structs_internal.DcgmEntityStatusUnknown
        start = time.time()
        while (time.time() - start) < maxWait:
            currentGpuStatus = dcgmDiscovery.GetGpuStatus(gpuId)
            if currentGpuStatus == expectedGpuStatus:
                return
            time.sleep(checkInterval)
        assert False, f"Timeout waiting for GPU {gpuId} status to update to {expectedGpuStatus}. Last GPU status was {currentGpuStatus}"

    def _wait_for_field_value_update(gpuId, fieldId, expectedValue):
        maxWait = 2
        checkInterval = 0.25
        lastValue = 0
        start = time.time()
        while (time.time() - start) < maxWait:
            dcgm_agent.dcgmUpdateAllFields(handle, 1)
            values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fieldId])
            lastValue = values[0].value.i64
            if lastValue == expectedValue:
                return
            time.sleep(checkInterval)
        assert False, f"Timeout waiting for GPU {gpuId} field value to update to {lastValue}. Last GPU status was {expectedValue}"

    # Verify injection before removing GPU
    inject_nvml_value(handle, removeGpuId, fieldId, fakeCurrentTemperature, injectionOffset)
    dcgm_agent_internal.dcgmWatchFieldValue(handle, removeGpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)
    _wait_for_field_value_update(removeGpuId, fieldId, fakeCurrentTemperature)

    # Verify blank value for the field after removing the GPU
    dcgm_agent_internal.dcgmRemoveNvmlInjectedGpu(handle, dev1Uuid)
    _wait_for_gpu_status_change(1, dcgm_structs_internal.DcgmEntityStatusLost)
    _run_diag_with_expected_entities(expectedAllGpusStr, True, dcgm_structs.DCGM_ST_BADPARAM) 
    _wait_for_field_value_update(removeGpuId, fieldId, dcgmvalue.DCGM_INT64_BLANK)

    # Inject the temperature value again, and verify that the blank values
    # are replaced by the injected values now that the GPU is back online.
    dcgm_agent_internal.dcgmRestoreNvmlInjectedGpu(handle, dev1Uuid)
    _wait_for_gpu_status_change(1, dcgm_structs_internal.DcgmEntityStatusOk)
    _run_diag_with_expected_entities(expectedAllGpusStr)
    inject_nvml_value(handle, removeGpuId, fieldId, fakeCurrentTemperature, injectionOffset)
    _wait_for_field_value_update(removeGpuId, fieldId, fakeCurrentTemperature)

    dcgm_agent_internal.dcgmRemoveNvmlInjectedGpu(handle, dev6Uuid)
    _wait_for_gpu_status_change(6, dcgm_structs_internal.DcgmEntityStatusLost)
    _run_diag_with_expected_entities(expectedAllGpusStr, True, dcgm_structs.DCGM_ST_BADPARAM)
    dcgm_agent_internal.dcgmRestoreNvmlInjectedGpu(handle, dev6Uuid)
    _wait_for_gpu_status_change(6, dcgm_structs_internal.DcgmEntityStatusOk)
    _run_diag_with_expected_entities(expectedAllGpusStr)

    dcgm_agent_internal.dcgmRemoveNvmlInjectedGpu(handle, dev1Uuid)
    dcgm_agent_internal.dcgmRemoveNvmlInjectedGpu(handle, dev6Uuid)
    _wait_for_gpu_status_change(1, dcgm_structs_internal.DcgmEntityStatusLost)
    _wait_for_gpu_status_change(6, dcgm_structs_internal.DcgmEntityStatusLost)
    _run_diag_with_expected_entities("gpu:" + str(len(gpuIds) - 2))
    dcgm_agent_internal.dcgmRestoreNvmlInjectedGpu(handle, dev1Uuid)
    dcgm_agent_internal.dcgmRestoreNvmlInjectedGpu(handle, dev6Uuid)
    _wait_for_gpu_status_change(1, dcgm_structs_internal.DcgmEntityStatusOk)
    _wait_for_gpu_status_change(6, dcgm_structs_internal.DcgmEntityStatusOk)
    _run_diag_with_expected_entities(expectedAllGpusStr)

TEST_NAME_TO_RESULT_FIELD_ID = {
    "memory": dcgm_fields.DCGM_FI_DEV_DIAG_MEMORY_RESULT,        
    "diagnostic": dcgm_fields.DCGM_FI_DEV_DIAG_DIAGNOSTIC_RESULT,
    "pcie": dcgm_fields.DCGM_FI_DEV_DIAG_PCIE_RESULT,
    "targeted_stress": dcgm_fields.DCGM_FI_DEV_DIAG_TARGETED_STRESS_RESULT,
    "targeted_power": dcgm_fields.DCGM_FI_DEV_DIAG_TARGETED_POWER_RESULT,
    "memory_bandwidth": dcgm_fields.DCGM_FI_DEV_DIAG_MEMORY_BANDWIDTH_RESULT,
    "memtest": dcgm_fields.DCGM_FI_DEV_DIAG_MEMTEST_RESULT,
    "pulse_test": dcgm_fields.DCGM_FI_DEV_DIAG_PULSE_TEST_RESULT,
    "eud": dcgm_fields.DCGM_FI_DEV_DIAG_EUD_RESULT,
    "cpu_eud": dcgm_fields.DCGM_FI_DEV_DIAG_CPU_EUD_RESULT,
    "software": dcgm_fields.DCGM_FI_DEV_DIAG_SOFTWARE_RESULT,
    "nvbandwidth": dcgm_fields.DCGM_FI_DEV_DIAG_NVBANDWIDTH_RESULT,
}

def get_diag_status_struct(blob):
    diagStatus = dcgm_structs.c_dcgmDiagStatus_v1()
    ctypes.memmove(ctypes.addressof(diagStatus), blob, diagStatus.FieldsSizeof())
    return diagStatus

def helper_dcgmi_diag_status(handle, dcgmiArgs, dcgmiTimeout, tests, errorCodes = {}):
    dcgmi = apps.DcgmiApp(dcgmiArgs)
    start = time.time()
    dcgmi.start(dcgmiTimeout)
    dcgmi.wait()
    dcgmi.validate()
    logger.debug(f"The dcgmi output is {dcgmi.stdout_lines}")

    fields = []
    summaryResultFieldId = dcgm_fields.DCGM_FI_DEV_DIAG_STATUS
    fields.append(summaryResultFieldId)
    for test in tests:
        testResultFieldId = TEST_NAME_TO_RESULT_FIELD_ID[test]
        fields.append(testResultFieldId)

    startTs = int(start*1000000) # start time of dcgmi in us
    endTs = 0
    totalTestCount = len(tests)
    order = dcgm_structs.DCGM_ORDER_ASCENDING
    fvFetched = {}
    for field in fields:
        expectedValuesCount = 1
        if field == dcgm_fields.DCGM_FI_DEV_DIAG_STATUS:
            expectedValuesCount = totalTestCount
        fvFetched[field] = dcgm_agent_internal.dcgmGetMultipleValuesForField(handle, 0, field, totalTestCount, startTs, endTs, order)
        assert len(fvFetched[field]) == expectedValuesCount, f"Expected {expectedValuesCount} values for field {field}. Got {len(fvFetched[field])}"

    # For each test, verify that the diag status field updates with the
    # result of the test. Also, verify that the test result field is
    # updated with the same values.
    for i in range(totalTestCount):
        diagStatus = get_diag_status_struct(fvFetched[dcgm_fields.DCGM_FI_DEV_DIAG_STATUS][i].value.blob)
        testResult = fvFetched[TEST_NAME_TO_RESULT_FIELD_ID[tests[i]]][0].value.i64

        assert diagStatus.totalTests == totalTestCount, f"Actual value of totalTests is {diagStatus.totalTests}, expected {totalTestCount}"
        assert diagStatus.testName == tests[i], f"Actual test name is {diagStatus.testName}, expected {tests[i]}"
        assert diagStatus.errorCode == testResult, f"diagStatus error code for test {diagStatus.testName} is {diagStatus.errorCode}, test result field value is {testResult}"
        assert diagStatus.completedTests == i+1, f"Actual value of completedTests is {diagStatus.completedTests}, expected {i+1}"
        # Verify the error codes for the tests if provided
        if diagStatus.testName in errorCodes:
            assert diagStatus.errorCode == errorCodes[diagStatus.testName], f"Actual error code for test {diagStatus.testName} is {diagStatus.errorCode}, expected {errorCodes[diagStatus.testName]}"

        logger.debug(f"Verified status of test {tests[i]}")

@test_utils.run_with_standalone_host_engine(240)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_status_r1(handle, gpuIds):
    """
    This test validates the diag status field and the individual plugin result
    fields as the test progresses.
    """
    args = ["diag", "-i", str(gpuIds[0]), "--run", "1"]
    helper_dcgmi_diag_status(handle, args, 10, ["software"], {"software": dcgm_errors.DCGM_FR_OK})

@test_utils.run_with_standalone_host_engine(240)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_status_r2(handle, gpuIds):
    # Skip the memory and pcie tests to reduce test time, and to verify the
    # new skipped error code
    args = ["diag", "-i", str(gpuIds[0]), "--run", "2", "-p", "memory.is_allowed=0;pcie.is_allowed=0"]
    helper_dcgmi_diag_status(handle, args, 60, ["software", "memory", "pcie"], {"software": dcgm_errors.DCGM_FR_OK, "pcie": dcgm_errors.DCGM_FR_TEST_SKIPPED, "memory": dcgm_errors.DCGM_FR_TEST_SKIPPED})

@test_utils.run_with_standalone_host_engine(240)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_status_r_memory(handle, gpuIds):
    args = ["diag", "-i", str(gpuIds[0]), "--run", "memory", "-p", "memory.is_allowed=0"]
    helper_dcgmi_diag_status(handle, args, 60, ["software", "memory"], {"software": dcgm_errors.DCGM_FR_OK, "memory": dcgm_errors.DCGM_FR_TEST_SKIPPED}) # the software plugin is always run

@test_utils.run_with_standalone_host_engine(240)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_status_r_memory_error(handle, gpuIds):
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ECC_CURRENT, 0, 0, repeatCount=5)
    args = ["diag", "-i", str(gpuIds[0]), "--run", "memory", "-p", "memory.is_allowed=true"]
    helper_dcgmi_diag_status(handle, args, 120, ["software", "memory"], {"software": dcgm_errors.DCGM_FR_OK, "memory": dcgm_errors.DCGM_FR_TEST_SKIPPED})

@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(240)
@test_utils.run_only_if_mig_is_disabled()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_diag_status_with_injected_gpu_r1(handle, gpuIds):
    """
    This test verifies the plugin error codes in the plugin result fields
    on diag error.
    """
    # Inject fabric manager error.
    injectedRet = nvml_injection.c_injectNvmlRet_t()
    injectedRet.nvmlRet = dcgm_nvml.NVML_ERROR_UNKNOWN
    injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_GPUFABRICINFOV
    injectedRet.values[0].value.GpuFabricInfoV.state = dcgm_nvml.NVML_GPU_FABRIC_STATE_COMPLETED
    injectedRet.values[0].value.GpuFabricInfoV.status = dcgm_nvml.NVML_SUCCESS
    injectedRet.valueCount = 1
    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuIds[0], "GpuFabricInfoV", None, 0, injectedRet)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    args = ["diag", "-i", str(gpuIds[0]), "--run", "1"]
    # Assume that the software plugin will error with the Fabric Manager
    # error on every run with the injected GPUs.
    # Note that the error code may need to be updated when a new test is added
    # to the software test suite, and the first error code is from that test
    # instead.
    helper_dcgmi_diag_status(handle, args, 10, ["software"], {"software": dcgm_errors.DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR})

def _run_diag_with_ignore_error_codes(handle, ignoreErrorCodes, error = None):
    dd = DcgmDiag.DcgmDiag(ignoreErrorCodesStr=ignoreErrorCodes)

    if error:
        with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(error)):
            response = test_utils.diag_execute_wrapper(dd, handle)
    else:
        response = test_utils.diag_execute_wrapper(dd, handle)
        assert response, "Should have received a response"
        assert response.tests[0].name == "software", \
            f"The response should have contained the 'software' plugin result, instead got {response.tests[0].name}"

@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_diag_ignore_error_codes_with_injected_gpus(handle, gpuIds):
    ignoreErrorCodes = "*:*"
    _run_diag_with_ignore_error_codes(handle, ignoreErrorCodes)

    ignoreErrorCodes = "gpu0:101"
    _run_diag_with_ignore_error_codes(handle, ignoreErrorCodes)

    ignoreErrorCodes = "gpu3:0"
    _run_diag_with_ignore_error_codes(handle, ignoreErrorCodes, dcgm_structs.DCGM_ST_NVVS_ERROR)

    ignoreErrorCodes = "gpu:*"
    _run_diag_with_ignore_error_codes(handle, ignoreErrorCodes, dcgm_structs.DCGM_ST_NVVS_ERROR)

def _get_run_diag_info(gpuIds, ignoreErrorCodes, testNameStr):
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v10()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version10
    runDiagInfo.groupId = dcgm_structs.DCGM_GROUP_NULL       
    runDiagInfo.entityIds = ",".join(map(str,gpuIds))
    runDiagInfo.ignoreErrorCodes = ignoreErrorCodes

    # The following is to ensure that the tests return as early as
    # possible, and do not take too long to run.
    runDiagInfo.flags = dcgm_structs.DCGM_RUN_FLAGS_FAIL_EARLY
    runDiagInfo.failCheckInterval = 1 # 1s
    testDurationStr = None
    if testNameStr not in ["nvbandwidth", "memory_bandwidth"]:
        testDurationStr = f"{testNameStr}.test_duration=2"

    index = 0
    for c in testNameStr:
        runDiagInfo.testNames[0][index] = ord(c)
        index += 1 
    index = 0
    if testDurationStr:
        for c in testDurationStr:
            runDiagInfo.testParms[0][index] = ord(c)
            index += 1

    return runDiagInfo

def _assert_diag_result_with_error_check(resultObj, expectedResults, entityId, errorsList, errorCode):
    if resultObj.result not in expectedResults:
        # Check if the failure is due to the specific error code
        errorFound = False
        for err in errorsList:
            if err.code == errorCode and err.entity.entityId == entityId:
                errorFound = True
                break

        if not errorFound:
            # If the failure is not due to the expected error, then we can skip the test
            test_utils.skip_test(f"Test failed with unexpected result {resultObj.result} and does not contain the expected {errorCode} error, skipping test.")
        else:
            # If the expected error is found, then the test should have had expected results.
            assert False, f"Expected result {expectedResults}, got {resultObj.result}."

def _run_diag_with_ignore_error_codes_error_check(handle, runDiagInfo, gpuId, expectedResults, testNameStr, errorCode = None):
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_XID_ERRORS, 97, 0, repeatCount=3, repeatOffset=5)

    response = test_utils.action_validate_wrapper(runDiagInfo, handle, dcgm_structs.dcgmRunDiag_version10)
    assert response.numTests == 2
    assert response, "Should have received a response"
    assert response.tests[0].name == "software", \
        f"The response should have contained the 'software' plugin result, instead got {response.tests[0].name}"
    assert response.tests[1].name == testNameStr, \
        f"The response should have contained the '{testNameStr}' plugin result, instead got {response.tests[1].name}"
     
    assert response.numResults == 2 

    # Skip checking software test results
    assert response.results[1].entity.entityId == gpuId, f"Expected GPU ID {gpuId}, got {response.results[1].entity.entityId}"
    errorFound = False
    if errorCode:
        assert response.results[1].result in expectedResults, f"Expected result in {expectedResults}, got {response.results[1].result}"
    else:
        _assert_diag_result_with_error_check(response.results[1], expectedResults, gpuId, response.errors, dcgm_errors.DCGM_FR_XID_ERROR)
        errorFound = True
        assert response.numErrors == 0, f"Expected 0 errors, got {response.numErrors} errors"
        # Verify that the ignored error is in the info array
        assert response.numInfo > 0
        infoFound = False
        for i in range(response.numInfo):
            if response.info[i].msg.startswith("Suppressed error:"):
                infoFound = True
        assert infoFound

    for i in range(response.numErrors):
        if response.errors[i].code == errorCode:
            errorFound = True
        logger.debug(f"Error {response.errors[i].code} found.")
    assert errorFound, f"Expected error code {errorCode} in response."

@test_utils.run_with_standalone_host_engine(360)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_ignore_error_codes(handle, gpuIds):
    testNames = ["memory"]
    for test in testNames:
        logger.debug(f"Running {test} test with injected error - blank ignoreErrorCodes")
        ignoreErrorCodes = "" 
        runDiagInfo = _get_run_diag_info([gpuIds[0]], ignoreErrorCodes, test)
        _run_diag_with_ignore_error_codes_error_check(handle, runDiagInfo, gpuIds[0], [dcgm_structs.DCGM_DIAG_RESULT_FAIL], test, dcgm_errors.DCGM_FR_XID_ERROR)

        logger.debug(f"Running {test} test with injected error - ignoreErrorCodes set to different error code")
        ignoreErrorCodes = f"gpu{gpuIds[0]}:{dcgm_errors.DCGM_FR_THERMAL_VIOLATIONS}"
        runDiagInfo = _get_run_diag_info([gpuIds[0]], ignoreErrorCodes, test)
        _run_diag_with_ignore_error_codes_error_check(handle, runDiagInfo, gpuIds[0], [dcgm_structs.DCGM_DIAG_RESULT_FAIL], test, dcgm_errors.DCGM_FR_XID_ERROR)

        logger.debug(f"Running {test} test with injected error - ignoreErrorCodes set to all possible gpus and error codes")
        ignoreErrorCodes = "*:*"
        runDiagInfo = _get_run_diag_info([gpuIds[0]], ignoreErrorCodes, test)
        _run_diag_with_ignore_error_codes_error_check(handle, runDiagInfo, gpuIds[0], [dcgm_structs.DCGM_DIAG_RESULT_PASS, dcgm_structs.DCGM_DIAG_RESULT_SKIP], test)

        logger.debug(f"Running {test} test with injected error - ignoreErrorCodes set to same gpu and error code")
        ignoreErrorCodes = f"gpu{gpuIds[0]}:{dcgm_errors.DCGM_FR_XID_ERROR}"
        runDiagInfo = _get_run_diag_info([gpuIds[0]], ignoreErrorCodes, test)
        _run_diag_with_ignore_error_codes_error_check(handle, runDiagInfo, gpuIds[0], [dcgm_structs.DCGM_DIAG_RESULT_PASS, dcgm_structs.DCGM_DIAG_RESULT_SKIP], test)

@test_utils.run_with_standalone_host_engine(240)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_ignore_error_codes_multiple_gpus_only_one_passes(handle, gpuIds):
    if len(gpuIds) <= 1:
        test_utils.skip_test("This test can be run only when there is more than 1 GPU.")
    
    test = "memory"
    # Inject error in both GPUs, but ignore error only on one GPU
    ignoreErrorCodes = f"gpu{gpuIds[0]}:{dcgm_errors.DCGM_FR_XID_ERROR}"
    runDiagInfo = _get_run_diag_info([gpuIds[0], gpuIds[1]], ignoreErrorCodes, test)
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_XID_ERRORS, 97, 0, repeatCount=3, repeatOffset=5)
    inject_value(handle, gpuIds[1], dcgm_fields.DCGM_FI_DEV_XID_ERRORS, 97, 0, repeatCount=3, repeatOffset=5)

    response = test_utils.action_validate_wrapper(runDiagInfo, handle, dcgm_structs.dcgmRunDiag_version10)
    assert response.numTests == 2, f"Expected 2 tests, got {response.numTests}"
    assert response, "Should have received a response"
    assert response.tests[0].name == "software", \
        f"The response should have contained the 'software' plugin result, instead got {response.tests[0].name}"
    assert response.tests[1].name == test, \
        f"The response should have contained the '{test}' plugin result, instead got {response.tests[0].name}"
     
    assert response.numResults == 4, f"Expected 4 results, got {response.numResults}"

    # Skip checking software test results
    assert response.results[2].entity.entityId == gpuIds[0], f"Expected GPU ID {gpuIds[0]}, got {response.results[2].entity.entityId}"
    _assert_diag_result_with_error_check(response.results[2], [dcgm_structs.DCGM_DIAG_RESULT_PASS, dcgm_structs.DCGM_DIAG_RESULT_SKIP], gpuIds[0], response.errors, dcgm_errors.DCGM_FR_XID_ERROR)

    assert response.results[3].entity.entityId == gpuIds[1], f"Expected GPU ID {gpuIds[1]}, got {response.results[3].entity.entityId}"
    assert response.results[3].result in [dcgm_structs.DCGM_DIAG_RESULT_FAIL], f"Got result {response.results[3].result}"

    errorFound = False
    for i in range(response.numErrors):
        if response.errors[i].code == dcgm_errors.DCGM_FR_XID_ERROR and response.errors[i].entity.entityId == gpuIds[1]:
            errorFound = True
        logger.debug(f"Error {response.errors[i].code} for entity {response.errors[i].entity.entityId} found.")
    assert errorFound, f"Expected error code {dcgm_errors.DCGM_FR_XID_ERROR} in response for entity {gpuIds[1]}."

@test_utils.run_with_standalone_host_engine(240)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_ignore_error_codes_multiple_gpus_all_pass(handle, gpuIds):
    if len(gpuIds) <= 1:
        test_utils.skip_test("This test can be run only when there is more than 1 GPU.")
    
    test = "pcie"
    # Inject error and ignore error on both GPUs
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_XID_ERRORS, 97, 0, repeatCount=3, repeatOffset=5)
    inject_value(handle, gpuIds[1], dcgm_fields.DCGM_FI_DEV_XID_ERRORS, 97, 0, repeatCount=3, repeatOffset=5)
    
    ignoreErrorCodes = f"gpu{gpuIds[0]}:{dcgm_errors.DCGM_FR_XID_ERROR};gpu{gpuIds[1]}:{dcgm_errors.DCGM_FR_XID_ERROR}"
    runDiagInfo = _get_run_diag_info([gpuIds[0], gpuIds[1]], ignoreErrorCodes, test)
    response = test_utils.action_validate_wrapper(runDiagInfo, handle, dcgm_structs.dcgmRunDiag_version10)
    assert response.numTests == 2, f"Expected 2 tests, got {response.numTests}"
    assert response, "Should have received a response"
    assert response.tests[0].name == "software", \
        f"The response should have contained the 'software' plugin result, instead got {response.tests[0].name}"
    assert response.tests[1].name == test, \
        f"The response should have contained the '{test}' plugin result, instead got {response.tests[0].name}"
     
    assert response.numResults == 4, f"Expected 4 results, got {response.numResults}"

    # Skip checking software test results
    assert response.results[2].entity.entityId == gpuIds[0], f"Expected GPU ID {gpuIds[0]}, got {response.results[2].entity.entityId}"
    _assert_diag_result_with_error_check(response.results[2], [dcgm_structs.DCGM_DIAG_RESULT_PASS, dcgm_structs.DCGM_DIAG_RESULT_SKIP], gpuIds[0], response.errors, dcgm_errors.DCGM_FR_XID_ERROR)
    assert response.results[3].entity.entityId == gpuIds[1], f"Expected GPU ID {gpuIds[1]}, got {response.results[3].entity.entityId}"
    _assert_diag_result_with_error_check(response.results[3], [dcgm_structs.DCGM_DIAG_RESULT_PASS, dcgm_structs.DCGM_DIAG_RESULT_SKIP], gpuIds[1], response.errors, dcgm_errors.DCGM_FR_XID_ERROR)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_diag_ignore_error_codes_software(handle, gpuIds):
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v10()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version10
    runDiagInfo.groupId = dcgm_structs.DCGM_GROUP_NULL
    runDiagInfo.entityIds = str(gpuIds[0])
    runDiagInfo.ignoreErrorCodes = f"gpu{gpuIds[0]}:{dcgm_errors.DCGM_FR_ROW_REMAP_FAILURE}"
    runDiagInfo.validate = 1

    fieldId = dcgm_fields.DCGM_FI_DEV_ROW_REMAP_FAILURE
    injected_value = 20 # random non-zero number
    inject_nvml_value(handle, gpuIds[0], fieldId, injected_value, 0)

    response = test_utils.action_validate_wrapper(runDiagInfo, handle, dcgm_structs.dcgmRunDiag_version10)

    suppressedErrorFound = False
    for i in range(response.numInfo):
        if response.info[i].msg.startswith(f"Suppressed error:  Page Retirement/Row Remap: GPU {gpuIds[0]} had uncorrectable memory errors and row remapping failed"):
            suppressedErrorFound = True
            break
        logger.debug(f"Info msg: {response.info[i].msg}")

    assert suppressedErrorFound, f"Suppressed error info message not found for error code {dcgm_errors.DCGM_FR_ROW_REMAP_FAILURE}"
