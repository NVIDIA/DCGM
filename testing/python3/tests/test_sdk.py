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
import dcgm_structs
import dcgm_agent_internal
import dcgm_agent
import logger
import utils
import test_utils
import dcgm_fields
import apps
import dcgmvalue
from apps.app_runner import AppRunner

import string
import time
from ctypes import *
import sys
import os
import subprocess
from subprocess import PIPE
import pprint
from sys import stdout

paths = {
            "Linux_32bit": "./apps/x86/",
            "Linux_64bit": "./apps/amd64/",
            "Linux_ppc64le": "./apps/ppc64le/",
            "Linux_aarch64": "./apps/aarch64/",
            "Windows_64bit": "./apps/amd64/"
            }

sdk_path = paths[utils.platform_identifier]

sdk_sample_scripts_path = "./sdk_samples/scripts"

# the sample scripts can potentially take a long time to run since they perform 
# a health check
SAMPLE_SCRIPT_TIMEOUT = 120.0 


def initialize_sdk(fileName):
    
    sdk_executable = sdk_path + fileName
    
    if utils.is_linux():
        if os.path.exists(sdk_executable):
            # On linux, for binaries inside the package (not just commands in the path) test that they have +x
            # e.g. if package is extracted on windows and copied to Linux, the +x privileges will be lost
            assert os.access(sdk_executable, os.X_OK), \
                "SDK binary %s is not executable! Make sure that the testing archive has been correctly extracted." \
                % sdk_executable
    
    return subprocess.Popen( sdk_executable, stdout=PIPE, stdin=PIPE)
    

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
@test_utils.run_only_if_mig_is_disabled() # We cannot set the compute mode when MIG is enabled
@test_utils.for_all_same_sku_gpus()
def test_sdk_configuration_sample_embedded(handle, gpuIds):
    """
    Test SDK configuration sample
    """
    
    sdk_subprocess = initialize_sdk("configuration_sample")
    
    sdk_stdout = sdk_subprocess.communicate(input=b'0')[0]  #input 0 through stdin (embeddded)
    
    ss = ""
    
    for line in sdk_stdout.decode():
        ss += line

    assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss
        
    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK, "SDK sample encountered an error. Return code: %d" % sdk_subprocess.returncode
    
    
    
@test_utils.run_only_if_mig_is_disabled() # We cannot set the compute mode when MIG is enabled
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
@test_utils.for_all_same_sku_gpus()
def test_sdk_configuration_sample_standalone(handle, gpuIds):
    """
    Test SDK configuration sample
    """
    
    sdk_subprocess = initialize_sdk("configuration_sample")
    
    sdk_stdout = sdk_subprocess.communicate(input=b'1\n127.0.0.1')[0]
  
    ss = ""
    
    for line in sdk_stdout.decode():
        ss += line

    assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss
        
    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK, "SDK sample encountered an error. Return code: %d" % sdk_subprocess.returncode
     
     
     
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_sdk_health_sample_embedded(handle, gpuIds):
    """
    Test SDK health sample
    """
    gpuGroups = test_utils.group_gpu_ids_by_sku(handle, gpuIds)
    if len(gpuGroups) > 1:
        test_utils.skip_test("Test only works for gpus with same sku")

    sdk_subprocess = initialize_sdk("health_sample")
     
    sdk_stdout = sdk_subprocess.communicate(input=b'0')[0]
    
    ss = ""
    
    for line in sdk_stdout.decode():
        ss += line

    assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss
         
    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK or sdk_subprocess.returncode == dcgm_structs.DCGM_ST_NO_DATA, "SDK sample encountered an error. Return code: %d. stdout: %s" % (sdk_subprocess.returncode, sdk_stdout.decode('utf-8'))



@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_sdk_health_sample_standalone(handle, gpuIds):
    """
    Test SDK health sample
    """
    gpuGroups = test_utils.group_gpu_ids_by_sku(handle, gpuIds)
    if len(gpuGroups) > 1:
        test_utils.skip_test("Test only works for gpus with same sku")

    sdk_subprocess = initialize_sdk("health_sample")
     
    sdk_stdout = sdk_subprocess.communicate(input=b'1\n127.0.0.1')[0]
    
    ss = ""
    
    for line in sdk_stdout.decode():
        ss += line
        
    assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss
         
    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK or sdk_subprocess.returncode == dcgm_structs.DCGM_ST_NO_DATA, "SDK sample encountered an error. Return code: %d" % sdk_subprocess.returncode


     
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_sdk_policy_sample_embedded(handle, gpuIds):
    """
    Test SDK policy sample
    """
    sdk_subprocess = initialize_sdk("policy_sample")
     
    sdk_stdout = sdk_subprocess.communicate(input=b'0')[0]
    
    ss = ""
    
    for line in sdk_stdout.decode():
        ss += line

    assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss
         
    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK, "SDK sample encountered an error. Return code: %d" % sdk_subprocess.returncode
    
    
    
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_sdk_policy_sample_standalone(handle, gpuIds):
    """
    Test SDK policy sample
    """
    sdk_subprocess = initialize_sdk("policy_sample")
     
    sdk_stdout = sdk_subprocess.communicate(input=b'1\n127.0.0.1')[0]
    
    ss = ""
    
    for line in sdk_stdout.decode():
        ss += line

    assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss
         
    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK, "SDK sample encountered an error. Return code: %d" % sdk_subprocess.returncode

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_sdk_field_value_sample_embedded(handle, gpuIds):
    """
    Test SDK field value sample
    """
    sdk_subprocess = initialize_sdk("field_value_sample")

    sdk_stdout = sdk_subprocess.communicate(input=b'0')[0]

    ss = ""

    for line in sdk_stdout.decode():
        ss += line

    assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss

    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK, "SDK sample encountered an error. Return code: %d. Output %s" % (sdk_subprocess.returncode, ss)



@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_sdk_field_value_sample_standalone(handle, gpuIds):
    """
    Test SDK policy sample
    """
    sdk_subprocess = initialize_sdk("field_value_sample")

    sdk_stdout = sdk_subprocess.communicate(input=b'1\n127.0.0.1')[0]

    ss = ""

    for line in sdk_stdout.decode():
        ss += line

    assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss

    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK, "SDK sample encountered an error. Return code: %d. Output: %s" % (sdk_subprocess.returncode, ss)

    
@test_utils.run_with_standalone_host_engine(timeout=SAMPLE_SCRIPT_TIMEOUT)
@test_utils.run_only_as_root()
def test_sdk_example_script_smoke_standalone_auto():
    """
    Smoke test ensuring that the example script for using dcgm does not fail
    for a standalone hostengine with auto operation mode
    """
    env = {'PYTHONPATH': ':'.join(sys.path)}
    script = os.path.join(sdk_sample_scripts_path, 'dcgm_example.py')
    example = AppRunner(sys.executable, [script, '--opmode=auto', '--type=standalone'], env=env)
    example.run(timeout=SAMPLE_SCRIPT_TIMEOUT)

@test_utils.run_only_as_root()
def test_sdk_example_script_smoke_embedded_auto():
    """
    Smoke test ensuring that the example script for using dcgm does not fail
    for an embedded hostengine with auto operation mode
    """
    env = {'PYTHONPATH': ':'.join(sys.path)}
    script = os.path.join(sdk_sample_scripts_path, 'dcgm_example.py')
    example = AppRunner(sys.executable, [script, '--opmode=auto', '--type=embedded'], env=env)
    example.run(timeout=SAMPLE_SCRIPT_TIMEOUT)

@test_utils.run_only_as_root()
def test_sdk_example_script_smoke_embedded_manual():
    """
    Smoke test ensuring that the example script for using dcgm does not fail 
    for an embedded hostengine with manual operation mode
    """
    env = {'PYTHONPATH': ':'.join(sys.path)}
    script = os.path.join(sdk_sample_scripts_path, 'dcgm_example.py')
    example = AppRunner(sys.executable, [script, '--opmode=manual', '--type=embedded'], env=env)
    example.run(timeout=SAMPLE_SCRIPT_TIMEOUT)

""" 
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_sdk_process_stats_sample_embedded(handle, gpuIds):
    ""
    Test SDK process stats sample
    ""

    devices = dcgm_agent.dcgmGetAllDevices(handle)
    for gpu in devices:
        device = dcgm_agent.dcgmGetDeviceAttributes(handle, gpu)
        if device is None:
            continue
        else:
            break
    
    if device == None:
        test_utils.skip_test("No GPU to run on")
    
    dcgm_agent.dcgmWatchFields(handle, dcgm_structs.DCGM_GROUP_ALL_GPUS, dcgm_fields.DCGM_FC_PROCESSINFO, 1000000, 3600, 0)
    
    ctx = apps.CudaCtxCreateAdvancedApp(
        [
            "--ctxCreate", device.identifiers.pciBusId,
            "--busyGpu", device.identifiers.pciBusId, "5000", # keep GPU busy (100% utilization) for 15s
            "--ctxDestroy", device.identifiers.pciBusId,
        ])

    ctx.start(10000)
    cuda_pid = ctx.getpid()
    ctx.wait()
    time.sleep(1.0)
    ctx.validate()
    
    dcgm_agent.dcgmUpdateAllFields(handle, 1) #force update
    
    sdk_subprocess = initialize_sdk("process_stats_sample")
     
    sdk_stdout = sdk_subprocess.communicate(input= b"0\n" + str(cuda_pid) )[0]

    ss = ""

    for line in sdk_stdout.decode():
        ss += line

    #assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss
        
    # Right now it returns no-data since we are passing in a random PID
    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK or sdk_subprocess.returncode == dcgm_structs.DCGM_ST_NO_DATA, "SDK sample encountered an error. Return code: %d" % sdk_subprocess.returncode
    


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_sdk_process_stats_sample_standalone(handle, gpuIds):
    ""
    Test SDK process stats sample
    ""
    
    devices = dcgm_agent.dcgmGetAllDevices(handle)
    for gpu in devices:
        device = dcgm_agent.dcgmGetDeviceAttributes(handle, gpu)
        if device is None:
            continue
        else:
            break
    
    if device == None:
        test_utils.skip_test("No GPU to run on")
    
    dcgm_agent.dcgmWatchFields(handle, dcgm_structs.DCGM_GROUP_ALL_GPUS, dcgm_fields.DCGM_FC_PROCESSINFO, 1000000, 3600, 0)
    
    ctx = apps.CudaCtxCreateAdvancedApp(
        [
            "--ctxCreate", device.identifiers.pciBusId,
            "--busyGpu", device.identifiers.pciBusId, "5000", # keep GPU busy (100% utilization) for 5s
            "--ctxDestroy", device.identifiers.pciBusId,
        ])

    ctx.start(10000)
    cuda_pid = ctx.getpid()
    ctx.wait()
    time.sleep(1.0)
    ctx.validate()
    
    
    dcgm_agent.dcgmUpdateAllFields(handle, 1) #force update
    
    time.sleep(1.0)
    
    sdk_subprocess = initialize_sdk("process_stats_sample")
     
    sdk_stdout = sdk_subprocess.communicate(input= b"1\n127.0.0.1\n" + str(cuda_pid) )[0]
    
    ss = ""
    
    for line in sdk_stdout.decode():
        ss += line

    #assert "error" not in ss.lower(), "Error detected in SDK sample. Output: %s" % ss
         
    # Right now it returns no-data since we are passing in a random PID
    assert sdk_subprocess.returncode == dcgm_structs.DCGM_ST_OK or sdk_subprocess.returncode == dcgm_structs.DCGM_ST_NO_DATA, "SDK sample encountered an error. Return code: %d" % sdk_subprocess.returncode
    
    #ctx.wait()
    #ctx.validate()
"""

     
