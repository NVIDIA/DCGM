# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import logger
import test_utils
import dcgm_fields
import dcgm_internal_helpers
import DcgmDiag
import option_parser
import os
import subprocess

import signal
import threading
import time

from dcgm_field_injection_helpers import inject_value
from shutil import which as find_executable

injection_offset = 3

################# General helpers #################
def check_diag_result_fail(response, gpuIndex, testIndex):
    return response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL

def check_diag_result_pass(response, gpuIndex, testIndex):
    return response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_PASS

################# General tests #################

##### Fail early behavior tests
def verify_early_fail_checks_for_test(handle, gpuId, test_name, testIndex, extraTestInfo):
    """
    Helper method for verifying the fail early checks for the specified test.
    """
    duration = 2 if testIndex != dcgm_structs.DCGM_TARGETED_POWER_INDEX else 30 # Prevent false failures due to min
                                                                                # duration requirements for Targeted Power
    paramsStr = "%s.test_duration=%s" % (test_name, duration)

    ###
    # First verify that the given test passes for the gpu.
    # If it doesn't pass, skip test and add note to check GPU health
    logger.info("Checking whether %s test passes on GPU %s" % (test_name, gpuId))
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=test_name, paramsStr=paramsStr)
    test_name_no_spaces = test_name.replace(" ", "_")
    logname = '/tmp/nv_' + test_name_no_spaces + '%s.log'
    dd.SetDebugLogFile(logname % 1)
    dd.SetDebugLevel(5)
    response = test_utils.diag_execute_wrapper(dd, handle)
    if not check_diag_result_pass(response, gpuId, testIndex):
        logger.info("Not testing %s because GPU %s does not pass. "
                             "Please verify whether the GPU is healthy." % (test_name, gpuId))

    ###
    # Verify fail early behavior by inserting an error.
    # Setup test parameters
    
    # We will be exiting early so the following duration is just how long we allow the test
    # to run before we kill it due to a suspected test failure.
    # Note that this has been increased from 20 -> 60 because some platforms are egregiously
    # slow for even small context create + smaller cuda malloc. 
    # If this test fails, it will take the full duration.
    duration = 60
    
    paramsStr = "%s.test_duration=%s" % (test_name, duration)
    response = None
    test_names = test_name
    if extraTestInfo:
        test_names += "," + extraTestInfo[0]
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=test_names, paramsStr=paramsStr)
    dd.SetFailEarly(checkInterval=2) # enable fail early checks
    dd.SetDebugLogFile(logname % 3)

    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 150, 1, True, repeatCount=10)

    # launch the diagnostic
    start = time.time()
    response = test_utils.diag_execute_wrapper(dd, handle)
    end = time.time()

    total_time = end - start
    assert total_time < duration, \
        "Expected %s test to exit early. Test took %ss to complete.\nGot result: %s (\ninfo: %s,\n warning: %s)" \
            % (test_name, total_time,
               response.perGpuResponses[gpuId].results[testIndex].result,
               response.perGpuResponses[gpuId].results[testIndex].info,
               response.perGpuResponses[gpuId].results[testIndex].error[0].msg)
    
    # Verify the test failed
    assert check_diag_result_fail(response, gpuId, testIndex), \
        "Expected %s test to fail due to injected dbes.\nGot result: %s (\ninfo: %s,\n warning: %s)" % \
            (test_name, response.perGpuResponses[gpuId].results[testIndex].result,
             response.perGpuResponses[gpuId].results[testIndex].info,
             response.perGpuResponses[gpuId].results[testIndex].error[0].msg)

    if extraTestInfo:
        extraTestResult = response.perGpuResponses[gpuId].results[extraTestInfo[1]].result
        assert extraTestResult == dcgm_structs.DCGM_DIAG_RESULT_SKIP, \
            "Expected the extra test to be skipped since the first test failed.\nGot results: %s" % \
            (extraTestResult)

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvvs_plugin_fail_early_diagnostic_standalone(handle, gpuIds):
    verify_early_fail_checks_for_test(handle, gpuIds[0], "diagnostic", dcgm_structs.DCGM_DIAGNOSTIC_INDEX, None)

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvvs_plugin_fail_early_targeted_stress_standalone(handle, gpuIds):
    verify_early_fail_checks_for_test(handle, gpuIds[0], "targeted stress", dcgm_structs.DCGM_TARGETED_STRESS_INDEX, None)

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvvs_plugin_fail_early_targeted_power_standalone(handle, gpuIds):
    verify_early_fail_checks_for_test(handle, gpuIds[0], "targeted power", dcgm_structs.DCGM_TARGETED_POWER_INDEX, None)

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvvs_plugin_fail_early_two_tests_standalone(handle, gpuIds):
    extraTestInfo = [ "pcie", dcgm_structs.DCGM_PCI_INDEX ]
    verify_early_fail_checks_for_test(handle, gpuIds[0], "diagnostic", dcgm_structs.DCGM_DIAGNOSTIC_INDEX, extraTestInfo)

################# Software plugin tests #################
def check_software_result_pass(response, index):
    assert 0 <= index < dcgm_structs.LEVEL_ONE_MAX_RESULTS
    return response.levelOneResults[index].result == dcgm_structs.DCGM_DIAG_RESULT_PASS

def check_software_result_pass_all(response):
    for result in response.levelOneResults:
        # ignore tests that are not run
        if result.result != dcgm_structs.DCGM_DIAG_RESULT_PASS \
                and result.result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
            return False
    return True

def check_software_result_fail(response, index):
    assert 0 <= index < dcgm_structs.LEVEL_ONE_MAX_RESULTS
    return response.levelOneResults[index].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL

def check_software_result_fail_all(response):
    for result in response.levelOneResults:
        # ignore tests that are not run
        if result.result != dcgm_structs.DCGM_DIAG_RESULT_FAIL \
                and result.result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
            return False
    return True


def helper_check_software_page_retirements_fail_on_pending_retirements(handle, gpuId):
    """
    Ensure that the software test for page retirements fails when there are pending page retirements.
    """
    # First verify that the software test passes for the gpu.
    # If it doesn't pass, skip test and add note to check GPU health
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId])
    dd.UseFakeGpus()
    response = test_utils.diag_execute_wrapper(dd, handle)
    if not check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT):
        test_utils.skip_test("Skipping because GPU %s does not pass software page retirement test. "
                             "Please verify whether the GPU is healthy." % gpuId)

    # Inject some pending page retirements
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 1, injection_offset, True)
    response = test_utils.diag_execute_wrapper(dd, handle)
    # Ensure software test failed due to pending page retirments
    assert check_software_result_fail(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
        "Expected software test to fail due to pending page retirements in the GPU"

    # Reset injected value
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 0, injection_offset, True)
    # Ensure diag passes now
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
        "Expected software test to pass"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_nvvs_plugin_software_pending_page_retirements_standalone(handle, gpuIds):
    # Injection tests can only work with the standalone host engine
    helper_check_software_page_retirements_fail_on_pending_retirements(handle, gpuIds[0])


def helper_check_software_page_retirements_fail_total_retirements(handle, gpuId):
    """
    Ensure that the software test for page retirements fails when there are mroe than 60 page retirements.
    """
    # First verify that the software test passes for the gpu. If it doesn't pass, skip test and add note to check GPU health
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId])
    dd.UseFakeGpus()
    response = test_utils.diag_execute_wrapper(dd, handle)
    if not check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT):
        test_utils.skip_test("Skipping because GPU %s does not pass software page retirement test. "
                             "Please verify whether the GPU is healthy." % gpuId)

    # Inject enough page retirements to cause failure
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE, 33, injection_offset, True)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 33, injection_offset, True)
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert check_software_result_fail(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
           "Expected software test to fail due to 60 total page retirements in the GPU"

    # Ensure 59 pages pass injected value
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 25, injection_offset, True)
    # Ensure diag passes now
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
           "Expected software test to pass since there are less than 60 total retired pages"

    # Reset retired pages count and verify pass
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE, 0, injection_offset, True)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 0, injection_offset, True)
    # Ensure diag still passes
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
           "Expected software test to pass since there are no retired pages"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_nvvs_plugin_software_total_page_retirements_standalone(handle, gpuIds):
    # Injection tests can only work with the standalone host engine
    helper_check_software_page_retirements_fail_total_retirements(handle, gpuIds[0])

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_nvvs_plugin_software_inforom_embedded(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="short")
    response = test_utils.diag_execute_wrapper(dd, handle)
    for gpuId in gpuIds:
        result = response.levelOneResults[dcgm_structs.DCGM_SWTEST_INFOROM].result
        assert(result == dcgm_structs.DCGM_DIAG_RESULT_PASS or result == dcgm_structs.DCGM_DIAG_RESULT_SKIP)

def test_nvvs_plugins_required_symbols():
    nmPath = find_executable('nm')
    if nmPath is None:
        test_utils.skip_test("'nm' is not installed on the system.")
    
    pluginPath = os.path.join(os.environ['NVVS_BIN_PATH'], 'plugins')
    numErrors = 0

    requiredSymbols = [
        'GetPluginInterfaceVersion', 
        'GetPluginInfo',
        'InitializePlugin',
        'RunTest',
        'RetrieveCustomStats',
        'RetrieveResults'
    ]

    skipLibraries = [
        'libpluginCommon.so',
        'libcurand.so'
    ]
    
    for cudaDirName in os.listdir(pluginPath):
        cudaPluginPath = os.path.join(pluginPath, cudaDirName)
        for soName in os.listdir(cudaPluginPath):
            soPath = os.path.join(cudaPluginPath, soName)
            #Skip symlinks
            if os.path.islink(soPath):
                continue
            #Skip non-libraries
            if not ".so" in soPath:
                continue
            
            #Skip some helper libraries that aren't plugin entry points
            skip = False
            for sl in skipLibraries:
                if sl in soPath:
                    skip = True
            if skip:
                continue
            
            args = [nmPath, soPath]
            output = str(subprocess.check_output(args, stderr=subprocess.STDOUT))

            if ': no symbols' in output:
                test_utils.skip_test("The installed nm is unable to see symbols within our plugins.")
            
            for rs in requiredSymbols:
                if not rs in output:
                    logger.error("library %s is missing symbol %s" % (soPath, rs))
                    numErrors += 1
    
    assert numErrors == 0, "Some plugins were missing symbols. See errors above."

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_nvvs_plugin_skip_memtest_if_page_retirement_row_remap_present(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="memtest", paramsStr="memtest.test_duration=10")
    dd.UseFakeGpus()

    # Inject Row remap failure and check for memtest to be skipped.
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ROW_REMAP_FAILURE, 1, 10, True, repeatCount=10)
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.levelOneResults[dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL,\
        f"Actual result: [{response.levelOneResults[dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT].result}]"
    assert response.perGpuResponses[gpuIds[0]].results[dcgm_structs.DCGM_MEMTEST_INDEX].result == dcgm_structs.DCGM_DIAG_RESULT_SKIP,\
        f"Actual result is {response.perGpuResponses[gpuIds[0]].results[dcgm_structs.DCGM_MEMTEST_INDEX].result}"
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ROW_REMAP_FAILURE, 0, 10, True, repeatCount=10) # reset the value

    # Inject pending page retired failure and check for memtest to be skipped.
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 1, 10, True, repeatCount=10)
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.levelOneResults[dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL,\
        f"Actual result: [{response.levelOneResults[dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT].result}]"
    assert response.perGpuResponses[gpuIds[0]].results[dcgm_structs.DCGM_MEMTEST_INDEX].result == dcgm_structs.DCGM_DIAG_RESULT_SKIP,\
        f"Actual result is {response.perGpuResponses[gpuIds[0]].results[dcgm_structs.DCGM_MEMTEST_INDEX].result}"
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 0, 10, True, repeatCount=10) # reset the value

    # After Row remap and pending page retired are reset i.e. not present, memtest's shouldn't be skipped.
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.perGpuResponses[gpuIds[0]].results[dcgm_structs.DCGM_MEMTEST_INDEX].result != dcgm_structs.DCGM_DIAG_RESULT_SKIP,\
        f"Actual result is {response.perGpuResponses[gpuIds[0]].results[dcgm_structs.DCGM_MEMTEST_INDEX].result}"
