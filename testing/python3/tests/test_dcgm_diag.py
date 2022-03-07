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
import pydcgm
import dcgm_structs
import dcgm_agent_internal
import dcgm_agent
import logger
import test_utils
import dcgm_fields
import dcgm_internal_helpers
import option_parser
import DcgmDiag
import dcgm_errors

import threading
import time
import sys
import os
import signal
import utils
import json
import tempfile
import shutil

from ctypes import *
from apps.app_runner import AppRunner
from apps.dcgmi_app import DcgmiApp
from dcgm_internal_helpers import inject_value

# Most injection tests use SmStress plugin, which also sleeps for 3 seconds
injection_offset = 3

def injection_wrapper(handle, gpuId, fieldId, value, isInt):
    # Sleep 1 second so that the insertion happens after the test run begins while not prolonging things
    time.sleep(1)
    if isInt:
        ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, fieldId, value, 0)
        assert ret == dcgm_structs.DCGM_ST_OK
        ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, fieldId, value, 5)
        assert ret == dcgm_structs.DCGM_ST_OK
        ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, fieldId, value, 10)
        assert ret == dcgm_structs.DCGM_ST_OK
    else:
        ret = dcgm_internal_helpers.inject_field_value_fp64(handle, gpuId, fieldId, value, 0)
        assert ret == dcgm_structs.DCGM_ST_OK
        ret = dcgm_internal_helpers.inject_field_value_fp64(handle, gpuId, fieldId, value, 5)
        assert ret == dcgm_structs.DCGM_ST_OK
        ret = dcgm_internal_helpers.inject_field_value_fp64(handle, gpuId, fieldId, value, 10)
        assert ret == dcgm_structs.DCGM_ST_OK

def check_diag_result_fail(response, gpuIndex, testIndex):
    return response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL

def check_diag_result_pass(response, gpuIndex, testIndex):
    return response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_PASS

def diag_result_assert_fail(response, gpuIndex, testIndex, msg, errorCode):
    # Instead of checking that it failed, just make sure it didn't pass because we want to ignore skipped
    # tests or tests that did not run.
    assert response.perGpuResponses[gpuIndex].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_PASS, msg
    if response.version == dcgm_structs.dcgmDiagResponse_version6:
        codeMsg = "Failing test expected error code %d, but found %d" % \
                    (errorCode, response.perGpuResponses[gpuIndex].results[testIndex].error.code)
        assert response.perGpuResponses[gpuIndex].results[testIndex].error.code == errorCode, codeMsg

def diag_result_assert_pass(response, gpuIndex, testIndex, msg):
    # Instead of checking that it passed, just make sure it didn't fail because we want to ignore skipped
    # tests or tests that did not run.
    assert response.perGpuResponses[gpuIndex].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_FAIL, msg
    if response.version == dcgm_structs.dcgmDiagResponse_version6:
        codeMsg = "Passing test somehow has a non-zero error code!"
        assert response.perGpuResponses[gpuIndex].results[testIndex].error.code == 0, codeMsg

def helper_check_diag_empty_group(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_t()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version
    runDiagInfo.groupId = groupObj.GetId()
    runDiagInfo.validate = 1

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GROUP_IS_EMPTY)):
        response = test_utils.action_validate_wrapper(runDiagInfo, handle)

    # Now make sure everything works well with a group
    groupObj.AddGpu(gpuIds[0])
    response = test_utils.action_validate_wrapper(runDiagInfo, handle)
    assert response, "Should have received a response now that we have a non-empty group"

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_helper_embedded_check_diag_empty_group(handle, gpuIds):
    helper_check_diag_empty_group(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_helper_standalone_check_diag_empty_group(handle, gpuIds):
    helper_check_diag_empty_group(handle, gpuIds)

def diag_assert_error_found(response, gpuId, testIndex, errorStr):
    if response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_SKIP and \
       response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
        
        warningFound = response.perGpuResponses[gpuId].results[testIndex].error.msg

        assert warningFound.find(errorStr) != -1, "Expected to find '%s' as a warning, but found '%s'" % (errorStr, warningFound)

def diag_assert_error_not_found(response, gpuId, testIndex, errorStr):
    if response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_SKIP and \
       response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
        
        warningFound = response.perGpuResponses[gpuId].results[testIndex].error.msg
        assert warningFound.find(errorStr) == -1, "Expected not to find '%s' as a warning, but found it: '%s'" % (errorStr, warningFound)

def helper_check_diag_thermal_violation(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr='diagnostic', paramsStr='diagnostic.test_duration=10')

    # kick off a thread to inject the failing value while I run the diag
    diag_thread = threading.Thread(target=injection_wrapper,
                                   args =[handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
                                          9223372036854775792, True])
    diag_thread.start()
    response = test_utils.diag_execute_wrapper(dd, handle)
    diag_thread.join()

    assert response.gpuCount == len(gpuIds), "Expected %d gpus, but found %d reported" % (len(gpuIds), response.gpuCount)
    for gpuIndex in range(response.gpuCount):
        diag_assert_error_not_found(response, gpuIndex, dcgm_structs.DCGM_DIAGNOSTIC_INDEX, "Thermal violations")

def helper_check_diag_high_temp_fail(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr='diagnostic', paramsStr='diagnostic.test_duration=10')

    # kick off a thread to inject the failing value while I run the diag
    diag_thread = threading.Thread(target=injection_wrapper,
                                   args =[handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 120, True])
    diag_thread.start()
    response = test_utils.diag_execute_wrapper(dd, handle)
    diag_thread.join()

    assert response.gpuCount == len(gpuIds), "Expected %d gpus, but found %d reported" % (len(gpuIds), response.gpuCount)
    diag_result_assert_fail(response, gpuIds[0], dcgm_structs.DCGM_DIAGNOSTIC_INDEX, "Expected a failure due to 120 degree inserted temp.", dcgm_errors.DCGM_FR_TEMP_VIOLATION)

def helper_check_dcgm_run_diag_backwards_compatibility(handle, gpuId):
    """
    Verifies that the dcgmActionValidate_v2 API supports older versions of the dcgmRunDiag struct
    by using the old structs to run a short validation test.
    """

    def test_dcgm_run_diag(drd, version):
        drd.validate = 1 # run a short test
        drd.gpuList = str(gpuId)
        # This will throw an exception on error
        response = test_utils.action_validate_wrapper(drd, handle, version)

    # Test version 6
    drd = dcgm_structs.c_dcgmRunDiag_v7()
    test_dcgm_run_diag(drd, dcgm_structs.dcgmRunDiag_version7)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_run_diag_backwards_compatibility_embedded(handle, gpuIds):
    helper_check_dcgm_run_diag_backwards_compatibility(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_run_diag_backwards_compatibility_standalone(handle, gpuIds):
    helper_check_dcgm_run_diag_backwards_compatibility(handle, gpuIds[0])

checked_gpus = {} # Used to track that a GPU has been verified as passing
# Makes sure a very basic diagnostic passes and returns a DcgmDiag object
def helper_verify_diag_passing(handle, gpuIds, testNames="SM Stress", testIndex=dcgm_structs.DCGM_SM_STRESS_INDEX, params="sm stress.test_duration=15", version=dcgm_structs.dcgmRunDiag_version, useFakeGpus=False):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testNames, paramsStr=params, version=version)
    dd.SetThrottleMask(0) # We explicitly want to fail for throttle reasons since this test inserts throttling errors 
                          # for verification
    if useFakeGpus:
        dd.UseFakeGpus()

    # If we've already chchecked this GPU, then use the previous result
    runDiag = False
    for gpuId in gpuIds:
        if gpuId in checked_gpus:
            if checked_gpus[gpuId] == False:
                test_utils.skip_test("Skipping because GPU %s does not pass SM Perf test. "
                                     "Please verify whether the GPU is supported and healthy." % gpuId)
        else:
            runDiag = True

    if runDiag == False:
        return dd

    response = test_utils.diag_execute_wrapper(dd, handle)
    for gpuId in gpuIds:
        if not check_diag_result_pass(response, gpuId, testIndex):
            checked_gpus[gpuId] = False
            test_utils.skip_test("Skipping because GPU %s does not pass SM Perf test. "
                                 "Please verify whether the GPU is supported and healthy." % gpuId)
        else:
            checked_gpus[gpuId] = True

    return dd

def find_throttle_failure(response, gpuId, pluginIndex):
    if response.perGpuResponses[gpuId].results[pluginIndex].result != dcgm_structs.DCGM_DIAG_RESULT_PASS:
        error = response.perGpuResponses[gpuId].results[pluginIndex].error.msg
        if error.find('clock throttling') != -1:
            return True, "%s (%s)" % (error, response.perGpuResponses[gpuId].results[pluginIndex].error.msg)
        else:
            return False, error

    return False, ""

def helper_test_thermal_violations_in_seconds(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr='diagnostic', paramsStr='diagnostic.test_duration=10')
    dd.UseFakeGpus()
    fieldId = dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION
    injected_value = 2344122048
    inject_value(handle, gpuIds[0], fieldId, injected_value, 10, True)

    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuIds[0], fieldId, injected_value, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    # Start the diag
    response = dd.Execute(handle)

    testIndex = dcgm_structs.DCGM_DIAGNOSTIC_INDEX
    errmsg = response.perGpuResponses[gpuIds[0]].results[testIndex].error.msg
    # Check for hermal instead of thermal because sometimes it's capitalized
    if errmsg.find("hermal violations") != -1:
        foundError = True
        assert errmsg.find("totaling 2.3 seconds") != -1, \
            "Expected 2.3 seconds of thermal violations but found %s" % errmsg
    else:
        # Didn't find an error
        assert False, "Thermal violations were injected but not found in error message: '%s'." % errmsg

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_thermal_violations_in_seconds_standalone(handle, gpuIds):
    helper_test_thermal_violations_in_seconds(handle, gpuIds)

#####
# Helper method for inserting errors and performing the diag
def perform_diag_with_throttle_mask_and_verify(dd, handle, gpuId, inserted_error, throttle_mask, shouldPass, failureMsg):
    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS
    interval = 0.1
    if throttle_mask is not None:
        dd.SetThrottleMask(throttle_mask)

    inject_value(handle, gpuId, fieldId, inserted_error, injection_offset, True)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, inserted_error, checkInterval=interval, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    # Start the diag
    response = test_utils.diag_execute_wrapper(dd, handle)

    # Check for pass or failure as per the shouldPass parameter
    throttled, errMsg = find_throttle_failure(response, gpuId, dcgm_structs.DCGM_SM_STRESS_INDEX)
    if shouldPass:    
        assert throttled == False, "Expected to not have a throttling error but found %s" % errMsg
    else:
        assert throttled == True, "Expected to find a throttling error but did not (%s)" % errMsg


def helper_test_throttle_mask_fail_hw_slowdown(handle, gpuId):
    """
    Verifies that the throttle ignore mask ignores the masked throttling reasons.
    """
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    #####
    # Insert a throttling error and verify that the test fails
    perform_diag_with_throttle_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask=0, shouldPass=False, failureMsg="Expected test to fail because of throttling"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttle_mask_fail_hw_slowdown(handle, gpuIds):
    helper_test_throttle_mask_fail_hw_slowdown(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_run_injection(handle, gpuIds):
    helper_test_throttle_mask_fail_hw_slowdown(handle, gpuIds[0])

def helper_test_throttle_mask_ignore_hw_slowdown(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert throttling error and set throttle mask to ignore it (as integer value)
    perform_diag_with_throttle_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN, shouldPass=True, 
        failureMsg="Expected test to pass because throttle mask (interger bitmask) ignores the throttle reason"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttle_mask_ignore_hw_slowdown(handle, gpuIds):
    helper_test_throttle_mask_ignore_hw_slowdown(handle, gpuIds[0])

def helper_test_throttle_mask_ignore_hw_slowdown_string(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert throttling error and set throttle mask to ignore it (as string name)
    perform_diag_with_throttle_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask="HW_SLOWDOWN", shouldPass=True, 
        failureMsg="Expected test to pass because throttle mask (named reason) ignores the throttle reason"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttle_mask_ignore_hw_slowdown_string(handle, gpuIds):
    helper_test_throttle_mask_ignore_hw_slowdown_string(handle, gpuIds[0])

def helper_test_throttle_mask_fail_double_inject_ignore_one(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert two throttling errors and set throttle mask to ignore only one (as integer)
    perform_diag_with_throttle_mask_and_verify(
        dd, handle, gpuId,
        inserted_error=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN | dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL, 
        throttle_mask=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN, shouldPass=False, 
        failureMsg="Expected test to fail because throttle mask (interger bitmask) ignores one of the throttle reasons"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttle_mask_fail_double_inject_ignore_one(handle, gpuIds):
    helper_test_throttle_mask_fail_double_inject_ignore_one(handle, gpuIds[0])

def helper_test_throttle_mask_fail_double_inject_ignore_one_string(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert two throttling errors and set throttle mask to ignore only one (as string name)
    perform_diag_with_throttle_mask_and_verify(
        dd, handle, gpuId,
        inserted_error=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN | dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL, 
        throttle_mask="HW_SLOWDOWN", shouldPass=False, 
        failureMsg="Expected test to fail because throttle mask (named reason) ignores one of the throttle reasons"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttle_mask_fail_double_inject_ignore_one_string(handle, gpuIds):
    helper_test_throttle_mask_fail_double_inject_ignore_one_string(handle, gpuIds[0])

def helper_test_throttle_mask_fail_ignore_different_throttle(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert throttling error and set throttle mask to ignore a different reason (as integer value)
    perform_diag_with_throttle_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE, shouldPass=False, 
        failureMsg="Expected test to fail because throttle mask (interger bitmask) ignores different throttle reason"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttle_mask_fail_ignore_different_throttle(handle, gpuIds):
    helper_test_throttle_mask_fail_ignore_different_throttle(handle, gpuIds[0])

def helper_test_throttle_mask_fail_ignore_different_throttle_string(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert throttling error and set throttle mask to ignore a different reason (as string name)
    perform_diag_with_throttle_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask="HW_POWER_BRAKE", shouldPass=False, 
        failureMsg="Expected test to fail because throttle mask (named reason) ignores different throttle reason"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttle_mask_fail_ignore_different_throttle_string(handle, gpuIds):
    helper_test_throttle_mask_fail_ignore_different_throttle_string(handle, gpuIds[0])

def helper_test_throttle_mask_pass_no_throttle(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Clear throttling reasons and mask to verify test passes
    dd.SetThrottleMask("")
    perform_diag_with_throttle_mask_and_verify(
        dd, handle, gpuId, inserted_error=0, throttle_mask=None, shouldPass=True,
        failureMsg="Expected test to pass because there is no throttling"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttle_mask_pass_no_throttle(handle, gpuIds):
    helper_test_throttle_mask_pass_no_throttle(handle, gpuIds[0])

def helper_check_diag_stop_on_interrupt_signals(handle, gpuId):
    """
    Verifies that a launched diag is stopped when the dcgmi executable recieves a SIGINT, SIGHUP, SIGQUIT, or SIGTERM
    signal.
    """
    # First check whether the GPU is healthy/supported
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=2",
                           version=dcgm_structs.dcgmRunDiag_version7)
    response = test_utils.diag_execute_wrapper(dd, handle)
    if not check_diag_result_pass(response, gpuId, dcgm_structs.DCGM_SM_STRESS_INDEX):
        test_utils.skip_test("Skipping because GPU %s does not pass SM Stress test. "
                             "Please verify whether the GPU is supported and healthy." % gpuId)

    # paths to dcgmi executable
    paths = {
        "Linux_32bit": "./apps/x86/dcgmi",
        "Linux_64bit": "./apps/amd64/dcgmi",
        "Linux_ppc64le": "./apps/ppc64le/dcgmi",
        "Linux_aarch64": "./apps/aarch64/dcgmi"
    }
    # Verify test is running on a supported platform
    if utils.platform_identifier not in paths:
        test_utils.skip_test("Dcgmi is not supported on the current platform.")
    dcgmi_path = paths[utils.platform_identifier]

    def verify_exit_code_on_signal(signum):
        # Ensure that host engine is ready to launch a new diagnostic
        dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr='1')
        success = False
        start = time.time()
        while not success and (time.time() - start) <= 3:
            try:
                response = test_utils.diag_execute_wrapper(dd, handle)
                success = True
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_DIAG_ALREADY_RUNNING):
                # Only acceptable error due to small race condition between the nvvs process exiting and 
                # hostengine actually processing the exit. We try for a maximum of 3 seconds since this 
                # should be rare and last only for a short amount of time
                time.sleep(1.5)

        diagApp = AppRunner(dcgmi_path, args=["diag", "-r", "SM Stress", "-i", "%s" % gpuId,
                                              "-d", "INFO", "--debugLogFile", "/tmp/nvvs.log"])
        # Start the diag
        diagApp.start(timeout=40)
        logger.info("Launched dcgmi process with pid: %s" % diagApp.getpid())
        
        # Ensure diag is running before sending interrupt signal
        running, debug_output = dcgm_internal_helpers.check_nvvs_process(want_running=True, attempts=50)
        assert running, "The nvvs process did not start within 25 seconds: %s" % (debug_output)
        # There is a small race condition here - it is possible that the hostengine sends a SIGTERM before the 
        # nvvs process has setup a signal handler, and so the nvvs process does not stop when SIGTERM is sent. 
        # We sleep for 1 second to reduce the possibility of this scenario
        time.sleep(1)
        diagApp.signal(signum)
        retCode = diagApp.wait()
        # Check the return code and stdout/stderr output before asserting for better debugging info
        if retCode != (signum + 128):
            logger.error("Got retcode '%s' from launched diag." % retCode)
            if diagApp.stderr_lines or diagApp.stdout_lines:
                logger.info("dcgmi output:")
                for line in diagApp.stdout_lines:
                    logger.info(line)
                for line in diagApp.stderr_lines:
                    logger.error(line)
        assert retCode == (signum + 128)
        # Since the app returns a non zero exit code, we call the validate method to prevent false
        # failures from the test framework
        diagApp.validate()
        # Give the launched nvvs process 15 seconds to terminate.
        not_running, debug_output = dcgm_internal_helpers.check_nvvs_process(want_running=False, attempts=50)
        assert not_running, "The launched nvvs process did not terminate within 25 seconds. pgrep output:\n%s" \
                % debug_output

    # Verify return code on SIGINT
    # We simply verify the return code because explicitly checking whether the nvvs process has terminated is
    # clunky and error-prone
    logger.info("Testing stop on SIGINT")
    verify_exit_code_on_signal(signal.SIGINT)

    # Verify return code on SIGHUP
    logger.info("Testing stop on SIGHUP")
    verify_exit_code_on_signal(signal.SIGHUP)

    # Verify return code on SIGQUIT
    logger.info("Testing stop on SIGQUIT")
    verify_exit_code_on_signal(signal.SIGQUIT)

    # Verify return code on SIGTERM
    logger.info("Testing stop on SIGTERM")
    verify_exit_code_on_signal(signal.SIGTERM)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_stop_on_signal_embedded(handle, gpuIds):
    if not option_parser.options.developer_mode:
        # This test can run into a race condition when using embedded host engine, which can cause nvvs to 
        # take >60 seconds to terminate after receiving a SIGTERM.
        test_utils.skip_test("Skip test for more debugging")
    helper_check_diag_stop_on_interrupt_signals(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_stop_on_signal_standalone(handle, gpuIds):
    helper_check_diag_stop_on_interrupt_signals(handle, gpuIds[0])

def helper_verify_log_file_creation(handle, gpuIds):
    dd = helper_verify_diag_passing(handle, gpuIds, testNames="targeted stress", testIndex=dcgm_structs.DCGM_TARGETED_STRESS_INDEX, params="targeted stress.test_duration=10", useFakeGpus=True)
    logname = '/tmp/tmp_test_debug_log'
    dd.SetDebugLogFile(logname)
    dd.SetDebugLevel(5)
    response = test_utils.diag_execute_wrapper(dd, handle)
    
    if len(response.systemError.msg) == 0:
        skippedAll = True
        passedCount = 0
        errors = ""
        for gpuId in gpuIds:
            resultType = response.perGpuResponses[gpuId].results[dcgm_structs.DCGM_TARGETED_STRESS_INDEX].result
            if resultType not in [dcgm_structs.DCGM_DIAG_RESULT_SKIP, dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN]:
                skippedAll = False
                if resultType == dcgm_structs.DCGM_DIAG_RESULT_PASS:
                    passedCount = passedCount + 1
                else:
                    warning = response.perGpuResponses[gpuId].results[dcgm_structs.DCGM_TARGETED_STRESS_INDEX].error.msg
                    if len(warning):
                        errors = "%s, GPU %d failed: %s" % (errors, gpuId, warning)

        if skippedAll == False:
            detailedMsg = "passed on %d of %d GPUs" % (passedCount, response.gpuCount)
            if len(errors):
                detailedMsg = "%s and had these errors: %s" % (detailedMsg, errors)
                logger.info(detailedMsg)
            assert os.path.isfile(logname), "Logfile '%s' was not created and %s" % (logname, detailedMsg)
        else:
            logger.info("The diagnostic was skipped, so we cannot run this test.")
    else:
        logger.info("The diagnostic had a problem when executing, so we cannot run this test.")


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_verify_log_file_creation_standalone(handle, gpuIds):
    helper_verify_log_file_creation(handle, gpuIds)

def helper_throttling_masking_failures(handle, gpuId):
    #####
    # First check whether the GPU is healthy
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=2",
                           version=dcgm_structs.dcgmRunDiag_version)
    dd.SetThrottleMask(0) # We explicitly want to fail for throttle reasons since this test inserts throttling errors 
                          # for verification
    dd.UseFakeGpus()
    response = test_utils.diag_execute_wrapper(dd, handle)
    if not check_diag_result_pass(response, gpuId, dcgm_structs.DCGM_SM_STRESS_INDEX):
        test_utils.skip_test("Skipping because GPU %s does not pass SM Perf test. "
                             "Please verify whether the GPU is supported and healthy." % gpuId)
    
    #####
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=15",
                           version=dcgm_structs.dcgmRunDiag_version)
    dd.SetThrottleMask(0)
    dd.UseFakeGpus()

    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS
    insertedError = dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN
    interval = 0.1

    logger.info("Injecting benign errors")
    inject_value(handle, gpuId, fieldId, 3, 1, True)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, 3, checkInterval=interval, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    
    logger.info("Injecting actual errors")
    inject_value(handle, gpuId, fieldId, insertedError, injection_offset, True)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True)

    logger.info("Started diag")
    response = test_utils.diag_execute_wrapper(dd, handle)
    # Verify that the inserted values are visible in DCGM
    # Max wait of 8 is because of 5 second offset + 2 seconds required for 20 matches + 1 second buffer.
    assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, insertedError, checkInterval=0.1, numMatches=1, maxWait=8), \
            "Expected inserted errors to be visible in DCGM"
                
    throttled, errMsg = find_throttle_failure(response, gpuId, dcgm_structs.DCGM_SM_STRESS_INDEX)
    assert throttled, "Expected to find throttling failure, but did not: (%s)" % errMsg

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_throttling_masking_failures_standalone(handle, gpuIds):
    helper_throttling_masking_failures(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_handle_concurrency_standalone(handle, gpuIds):
    '''
    Test that we can use a DCGM handle concurrently with a diagnostic running
    '''
    diagDuration = 10

    gpuId = gpuIds[0]

    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=%d" % diagDuration,
                           version=dcgm_structs.dcgmRunDiag_version)

    dd.UseFakeGpus()
    
    response = [None]
    def run(dd, response):
        response = test_utils.diag_execute_wrapper(dd, handle)
    
    diagStartTime = time.time()
    threadObj = threading.Thread(target=run, args=[dd, response])
    threadObj.start()

    #Give threadObj a head start on its 10 second run
    time.sleep(1.0)

    firstReturnedRequestLatency = None
    numConcurrentCompleted = 0
    sleepDuration = 1.0

    while threadObj.is_alive():
        #Make another request on the handle concurrently
        moduleStatuses = dcgm_agent.dcgmModuleGetStatuses(handle)
        secondRequestLatency = time.time() - diagStartTime
        numConcurrentCompleted += 1

        if firstReturnedRequestLatency is None:
            firstReturnedRequestLatency = secondRequestLatency
        
        time.sleep(sleepDuration)
    
    diagThreadEndTime = time.time()
    diagDuration = diagThreadEndTime - diagStartTime
    
    if firstReturnedRequestLatency is None:
        test_utils.skip_test("Diag returned instantly. It is probably not supported for gpuId %u" % gpuId)
    
    logger.info("Completed %d concurrent requests. Diag ran for %.1f seconds" % (numConcurrentCompleted, diagDuration))
    
    #We should have been able to complete a request every 2 seconds if we slept for 1 (conservatively)
    numShouldHaveCompleted = int((diagDuration / sleepDuration) / 2.0)
    assert numConcurrentCompleted >= numShouldHaveCompleted, "Expected at least %d concurrent tests completed. Got %d" % (numShouldHaveCompleted, numConcurrentCompleted)

def helper_per_gpu_responses_api(handle, gpuIds, testDir):
    """
    Verify that pass/fail status for diagnostic tests are reported on a per GPU basis via dcgmActionValidate API call
    """
    failGpuId = gpuIds[0]
    dd = helper_verify_diag_passing(handle, gpuIds, useFakeGpus=True)


    dd = DcgmDiag.DcgmDiag(gpuIds=[failGpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=15", version=dcgm_structs.dcgmRunDiag_version)
    dd.SetThrottleMask(0) # We explicitly want to fail for throttle reasons since this test inserts throttling errors 
                          # for verification
    dd.UseFakeGpus()
    dd.SetStatsPath(testDir)
    dd.SetStatsOnFail(1)

    # Setup injection app    
    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS
    insertedError = dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN
    interval = 0.1
    # Use an offset to make these errors start after the benign values
    inject_value(handle, failGpuId, fieldId, insertedError, injection_offset, True)
    inject_value(handle, failGpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(failGpuId, fieldId, insertedError, checkInterval=interval, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    response = test_utils.diag_execute_wrapper(dd, handle)
    logger.info("Started diag")
                                                                                                     
    # Verify that responses are reported on a per gpu basis. Ensure the first GPU failed, and all others passed
    for gpuId in gpuIds:
        throttled, errMsg = find_throttle_failure(response, gpuId, dcgm_structs.DCGM_SM_STRESS_INDEX)
        if gpuId == failGpuId:
            assert throttled, "Expected throttling error but found none (%s)" % errMsg
        else:
            assert not throttled, "Expected not to find a throttling error but found '%s'" % errMsg

def helper_per_gpu_responses_dcgmi(handle, gpuIds):
    """
    Verify that pass/fail status for diagnostic tests are reported on a per GPU basis via dcgmi (for both normal stdout 
    and JSON output).
    """
    def get_stdout(app):
        output = ''
        for line in app.stdout_lines:
            output = output + line + " "
        return output
    def print_output(app):
        logger.info(get_stdout(app))
        for line in app.stderr_lines:
            logger.error(line)

    def verify_successful_dcgmi_run(app):
        app.start(timeout=40)
            
        logger.info("Started dcgmi diag with pid %s" % app.getpid())
        retcode = app.wait()

        if test_utils.is_mig_incompatible_failure(get_stdout(app)):
            app.validate()
            test_utils.skip_test("Skipping this test because MIG is configured incompatibly (preventing access to the whole GPU)")

        # dcgm returns DCGM_ST_NVVS_ERROR on diag failure (which is expected here).
        expected_retcode = c_uint8(dcgm_structs.DCGM_ST_NVVS_ISOLATE_ERROR).value
        if retcode != expected_retcode:
            if app.stderr_lines or app.stdout_lines:
                    logger.info("dcgmi output:")
                    print_output(app)
        assert retcode == expected_retcode, \
            "Expected dcgmi diag to have retcode %s. Got return code %s" % (expected_retcode, retcode)
        app.validate() # non-zero exit code must be validated

    #helper_verify_diag_passing(handle, gpuIds, useFakeGpus=True)

    # Setup injection app
    interval = 0.1
    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS
    insertedError = dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN
    # Use an offset to make these errors start after the benign values
    inject_value(handle, gpuIds[0], fieldId, insertedError, injection_offset, True)
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuIds[0], fieldId, insertedError, checkInterval=interval, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    # Verify dcgmi output
    gpuIdStrings = list(map(str, gpuIds))
    gpuList = ",".join(gpuIdStrings)
    args = ["diag", "-r", "SM Stress", "-p", "sm stress.test_duration=5,pcie.max_pcie_replays=1", "-f", gpuList, "--throttle-mask", "0"]
    dcgmiApp = DcgmiApp(args=args)

    logger.info("Verifying stdout output")
    verify_successful_dcgmi_run(dcgmiApp)
    # Verify dcgmi output shows per gpu results (crude approximation of verifying correct console output)
    stress_header_found = False
    fail_gpu_found = False
    fail_gpu_text = "Fail - GPU: %s" % gpuIds[0]
    check_for_warning = False
    warning_found = False
    for line in dcgmiApp.stdout_lines:
        if not stress_header_found:
            if "Stress" not in line:
                continue
            stress_header_found = True
            continue
        if not fail_gpu_found:
            if fail_gpu_text not in line:
                continue
            fail_gpu_found = True
            check_for_warning = True
            continue
        if check_for_warning:
            if "Warning" in line:
                warning_found = True
            break

    if not (stress_header_found and fail_gpu_found and warning_found):
        logger.info("dcgmi output:")
        print_output(dcgmiApp)

    assert stress_header_found, "Expected to see 'Stress' header in output"
    assert fail_gpu_found, "Expected to see %s in output" % fail_gpu_text
    assert warning_found, "Expected to see 'Warning' in output after GPU failure text"

    inject_value(handle, gpuIds[0], fieldId, insertedError, injection_offset, True)
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuIds[0], fieldId, insertedError, checkInterval=interval, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    # Verify JSON output
    logger.info("Verifying JSON output")
    args.append("-j")
    dcgmiApp = DcgmiApp(args=args)
    verify_successful_dcgmi_run(dcgmiApp)

    # Stop error insertion
    logger.info("Stopped error injection")

    # Verify per GPU results
    json_output = "\n".join(dcgmiApp.stdout_lines)
    output = json.loads(json_output)
    verifed = False
    if (len(output.get("DCGM GPU Diagnostic", {}).get("test_categories", [])) == 2
            and output["DCGM GPU Diagnostic"]["test_categories"][1].get("category", None) == "Stress"
            and output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["name"] == "SM Stress"
            and len(output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["results"]) >= 2
            and output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["gpu_ids"] == str(gpuIds[0])
            and output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["status"] == "Fail"
            and output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["results"][1]["status"] == "Pass"):
        verifed = True

    if not verifed:
        print_output(dcgmiApp)

    assert verifed, "dcgmi JSON output did not pass verification"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_per_gpu_responses_standalone_api(handle, gpuIds):
    if len(gpuIds) < 2:
        test_utils.skip_test("Skipping because this test requires 2 or more GPUs with same SKU")

    if test_utils.is_throttling_masked_by_nvvs(handle, gpuIds[0], dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN):
        test_utils.skip_test("Skipping because this SKU ignores the throttling we inject for this test")

    logger.info("Starting test for per gpu responses (API call)")
    outputFile = "stats_sm_stress.json"
    try:
        testDirectory = tempfile.mkdtemp()
    except OSError:
        test_utils.skip_test("Unable to create the test directory")
    else:
        try:
            helper_per_gpu_responses_api(handle, gpuIds, testDirectory)
            assert os.path.isfile(os.path.join(testDirectory, outputFile)), "Expected stats file {} was not created".format(os.path.join(testDirectory, outputFile))
        finally:
             shutil.rmtree(testDirectory, ignore_errors=True)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_per_gpu_responses_standalone_dcgmi(handle, gpuIds):
    if len(gpuIds) < 2:
        test_utils.skip_test("Skipping because this test requires 2 or more GPUs with same SKU")

    if test_utils.is_throttling_masked_by_nvvs(handle, gpuIds[0], dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN):
        test_utils.skip_test("Skipping because this SKU ignores the throttling we inject for this test")

    logger.info("Starting test for per gpu responses (dcgmi output)")
    helper_per_gpu_responses_dcgmi(handle, gpuIds)

def helper_test_diagnostic_config_usage(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="diagnostic", paramsStr="diagnostic.test_duration=10")
    dd.SetConfigFileContents("%YAML 1.2\n\ncustom:\n- custom:\n    diagnostic:\n      max_sbe_errors: 1")
    
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, 1000, injection_offset, True)
    
    response = test_utils.diag_execute_wrapper(dd, handle)
    
    assert response.perGpuResponses[gpuIds[0]].results[dcgm_structs.DCGM_DIAGNOSTIC_INDEX].result != dcgm_structs.DCGM_DIAG_RESULT_PASS, \
                "Should have a failure due to injected SBEs, but got passing result"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(2)
def test_diagnostic_config_usage_standalone(handle, gpuIds):
    helper_test_diagnostic_config_usage(handle, gpuIds)

def helper_test_dcgm_short_diagnostic_run(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="diagnostic", paramsStr="diagnostic.test_duration=15")
    response = test_utils.diag_execute_wrapper(dd, handle)
    for gpuId in gpuIds:
        if response.perGpuResponses[gpuId].results[dcgm_structs.DCGM_DIAGNOSTIC_INDEX].result == dcgm_structs.DCGM_DIAG_RESULT_SKIP:
            logger.info("Got status DCGM_DIAG_RESULT_SKIP for gpuId %d. This is expected if this GPU does not support the Diagnostic test." % gpuId)
            continue

        assert response.perGpuResponses[gpuId].results[dcgm_structs.DCGM_DIAGNOSTIC_INDEX].result == dcgm_structs.DCGM_DIAG_RESULT_PASS, \
                    "Should have passed the 15 second diagnostic for all GPUs"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_short_diagnostic_run(handle, gpuIds):
    helper_test_dcgm_short_diagnostic_run(handle, gpuIds)
