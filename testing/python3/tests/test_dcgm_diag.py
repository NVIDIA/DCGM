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
import dcgm_internal_helpers
import option_parser
import DcgmDiag
from DcgmDiag import check_diag_result_non_failing, check_diag_result_non_passing, check_diag_result_non_running, check_diag_result_pass, check_diag_result_fail, \
    GetEntityCount, GetGpuCount
import dcgm_errors
import dcgmvalue

import threading
import time
import sys
import os
import signal
import utils
import json
import tempfile
import shutil
import subprocess

from ctypes import *
from apps.app_runner import AppRunner
from apps.dcgmi_app import DcgmiApp
from dcgm_field_injection_helpers import inject_value, inject_nvml_value
from _test_helpers import skip_test_if_no_dcgm_nvml
import dcgm_field_injection_helpers

# Most injection tests use memtest plugin, which also sleeps for 3 seconds

# These are used on all architectures but are specific to each.
injection_offset = 3

TEST_DIAGNOSTIC = "diagnostic"
TEST_MEMORY = "memory"
TEST_MEMTEST = "memtest"

g_latestDiagResponseVer = dcgm_structs.dcgmDiagResponse_version12
g_latestDiagRunVer = dcgm_structs.dcgmRunDiag_version10

def diag_result_assert_fail(response, gpuIndex, testName, msg, errorCode):
    # Raises AssertError when there is one or more passing result associated with gpuIndex and testName.
    assert response.version == g_latestDiagResponseVer
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuIndex )
    assert check_diag_result_non_passing(response, entityPair, testName), msg

def diag_result_assert_pass(response, gpuIndex, testName, msg):
    # Raises AssertError when there is one or more failure result associated with gpuIndex and testName.
    assert response.version == g_latestDiagResponseVer
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuIndex )
    assert check_diag_result_non_failing(response, entityPair, testName), msg

def diag_result_assert_fail_v1(response, gpuIndex, testIndex, msg, errorCode):
    # Deprecated. For use with diagResponse_version9 and earlier.
    # Instead of checking that it failed, just make sure it didn't pass because we want to ignore skipped
    # tests or tests that did not run.
    assert response.perGpuResponses[gpuIndex].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_PASS, msg
    if response.version < g_latestDiagResponseVer:
        codeMsg = "Failing test expected error code %d, but found %d" % \
                    (errorCode, response.perGpuResponses[gpuIndex].results[testIndex].error[0].code)
        assert response.perGpuResponses[gpuIndex].results[testIndex].error[0].code == errorCode, codeMsg

def diag_result_assert_pass_v1(response, gpuIndex, testIndex, msg):
    # Deprecated. For use with diagResponse_version9 and earlier.
    # Instead of checking that it passed, just make sure it didn't fail because we want to ignore skipped
    # tests or tests that did not run.
    assert response.perGpuResponses[gpuIndex].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_FAIL, msg
    if response.version < g_latestDiagResponseVer:
        codeMsg = "Passing test somehow has a non-zero error code!"
        assert response.perGpuResponses[gpuIndex].results[testIndex].error[0].code == 0, codeMsg

def helper_check_diag_empty_group(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v10()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version10
    runDiagInfo.groupId = groupObj.GetId()
    runDiagInfo.validate = 1

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GROUP_IS_EMPTY)):
        response = test_utils.action_validate_wrapper(runDiagInfo, handle)

    # Now make sure everything works well with a group
    groupObj.AddGpu(gpuIds[0])
    response = test_utils.action_validate_wrapper(runDiagInfo, handle)
    assert response, "Should have received a response now that we have a non-empty group"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus(2)
def test_helper_standalone_check_diag_empty_group(handle, gpuIds):
    helper_check_diag_empty_group(handle, gpuIds)

# This function isn't currently used and can be removed.
def diag_assert_error_found(response, entityPair, testName, errorStr):
    # Raises AssertError if errorStr is not found associated with entityPair and testName.
    # This currently asserts on the first matching error and can be made more robust by searching all
    # errors associated with entityPair and testName.
    assert response.version == g_latestDiagResponseVer, "Version %d is not handled." % response.version
    if type(entityPair) == int:
        gpuId = entityPair
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuId )
    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.name == testName:
            break
    assert test.name == testName, "Expected to find '%s' as an error for test '%s' but no results were found" % (errorStr, testName)
    err = next(filter(lambda cur: cur.entity == entityPair, map(lambda errIdx: response.errors[errIdx],
                                                                test.errorIndices[:min(test.numErrors, dcgm_structs.DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)])), True)
    assert err, "Expected to find '%s' as an error for test '%s' but none were found." % (errorStr, testName)
    assert err.msg.find(errorStr) != -1, "Expected to find '%s' as an error for test '%s', but found '%s'" % (errorStr, testName, err.msg)

# This function isn't currently used and can be removed.
def diag_assert_error_found_v1(response, gpuId, testIndex, errorStr):
    # Deprecated. For use with diagResponse_v9 and earlier.
    if response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_SKIP and \
       response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
        
        warningFound = response.perGpuResponses[gpuId].results[testIndex].error[0].msg

        assert warningFound.find(errorStr) != -1, "Expected to find '%s' as a warning, but found '%s'" % (errorStr, warningFound)

def diag_assert_error_not_found(response, entityPair, testName, errorStr):
    # Raises AssertError when the specified errorStr is found associated with entity and testName.
    # This currently asserts on the first matching error and can be made more robust by searching all
    # errors associated with entityPair and testName.
    assert response.version == g_latestDiagResponseVer, "Version %d is not handled." % response.version
    if type(entityPair) == int:
        gpuId = entityPair
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuId )
    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.name == testName:
            break
    if test.name != testName:
        # No results were found, so the error was not found.
        return
    err = next(filter(lambda cur: cur.entity == entityPair, map(lambda errIdx: response.errors[errIdx], test.errorIndices[:min(test.numErrors, dcgm_structs.DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)])), None)
    assert not err or err.msg.find(errorStr) == -1, "Expected not to find '%s' as an error, but found it: '%s'" % (errorStr, err.msg)

def diag_assert_error_not_found_v1(response, gpuId, testIndex, errorStr):
    # Deprecated. For use with diagResponse_v9 and earlier.
    if response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_SKIP and \
       response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
        
        warningFound = response.perGpuResponses[gpuId].results[testIndex].error[0].msg
        assert warningFound.find(errorStr) == -1, "Expected not to find '%s' as a warning, but found it: '%s'" % (errorStr, warningFound)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus(2)
def test_multiple_xid_errors(handle, gpuIds):
    testName = TEST_DIAGNOSTIC
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testName, paramsStr='diagnostic.test_duration=10')

    gpuId = gpuIds[0]
    # kick off a thread to inject the failing value while I run the diag
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_XID_ERRORS, 97, 0, repeatCount=3, repeatOffset=5)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_XID_ERRORS, 98, 0, repeatCount=3, repeatOffset=5)

    response = test_utils.diag_execute_wrapper(dd, handle)

    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.name == testName:
            break
    assert test.name == testName, "Expected 2 errors but no results for test %s were found" % testName
    num_errors = sum(1 for err in filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and
                                         cur.entity.entityId == gpuId and cur.code == dcgm_errors.DCGM_FR_XID_ERROR,
                                         map(lambda errIdx: response.errors[errIdx], test.errorIndices[:min(test.numErrors, dcgm_structs.DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)])))
    assert num_errors == 2, "Expected 2 errors but found %d" % (num_errors)

"""
@test_utils.run_with_injection_nvml()
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_if_mig_is_disabled()
def TODO: add the injection nvml test here
"""

def helper_check_diag_high_temp_fail(handle, gpuIds):
    testName = TEST_DIAGNOSTIC
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testName, paramsStr='diagnostic.test_duration=10')

    # kick off a thread to inject the failing value while I run the diag
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 120, 0, repeatCount=3, repeatOffset=5)
    response = test_utils.diag_execute_wrapper(dd, handle)

    assert response.gpuCount == len(gpuIds), "Expected %d gpus, but found %d reported" % (len(gpuIds), response.gpuCount)
    diag_result_assert_fail(response, gpuIds[0], testName, "Expected a failure due to 120 degree inserted temp.", dcgm_errors.DCGM_FR_TEMP_VIOLATION)

def helper_check_dcgm_run_diag_backwards_compatibility(handle, gpuId):
    """
    Verifies that the dcgmActionValidate_v2 API supports older versions of the dcgmRunDiag struct
    by using the old structs to run a short validation test.
    """

    def localDcgmActionValidate_v2(dcgm_handle, runDiagInfo, response):
        fn = dcgm_structs._dcgmGetFunctionPointer("dcgmActionValidate_v2")
        ret = fn(dcgm_handle, byref(runDiagInfo), byref(response))
        return ret

    def _test_run_diag_v9_and_v10():
        runDiagVersions = {
            dcgm_structs.dcgmRunDiag_version9: dcgm_structs.c_dcgmRunDiag_v9(),
            dcgm_structs.dcgmRunDiag_version10: dcgm_structs.c_dcgmRunDiag_v10(),
        }
        diagResponseVersions = {
            dcgm_structs.dcgmRunDiag_version9: [ dcgm_structs.c_dcgmDiagResponse_v11(), dcgm_structs.dcgmDiagResponse_version11 ],
            dcgm_structs.dcgmRunDiag_version10: [ dcgm_structs.c_dcgmDiagResponse_v12(), dcgm_structs.dcgmDiagResponse_version12 ],
        }

        for runDiagVer, runDiag in runDiagVersions.items():
            drd = runDiag
            drd.version = runDiagVer
            drd.entityIds = str(gpuId)
            drd.groupId = dcgm_structs.DCGM_GROUP_NULL
            drd.validate = 1
            response, responseVer = diagResponseVersions[runDiagVer]
            response.version = responseVer
            response.numTests = 0
            ret = localDcgmActionValidate_v2(handle, drd, response)
            assert ret == dcgm_structs.DCGM_ST_OK, f"ret: [{ret}] for RunDiag version {runDiagVer}"
            assert response.version == responseVer, f"expected {responseVer:x} actual {response.version:x}"
            assert response.numTests == 1, f"response.numTests: [{response.numTests}] for RunDiag version {runDiagVer}"
            assert response.tests[0].name == "software", f"response.tests[0].name: [{response.tests[0].name}] for RunDiag version {runDiagVer}"
    
    # Test runDiag_v9 and runDiag_v10 (may be redundant with some tests below)
    _test_run_diag_v9_and_v10()

    # Test dcgmRunDiag_v8 with dcgmDiagResponse_v10
    drd = dcgm_structs.c_dcgmRunDiag_v8()
    drd.version = dcgm_structs.dcgmRunDiag_version8
    drd.gpuList = str(gpuId)
    drd.validate = 1
    response = dcgm_structs.c_dcgmDiagResponse_v10()
    response.version = dcgm_structs.dcgmDiagResponse_version10
    response.levelOneTestCount = 0
    ret = localDcgmActionValidate_v2(handle, drd, response)
    assert ret == dcgm_structs.DCGM_ST_OK, f"ret: [{ret}]"
    assert response.version == dcgm_structs.dcgmDiagResponse_version10
    assert response.levelOneTestCount != 0, f"response.levelOneTestCount: [{response.levelOneTestCount}]"

    # Test dcgmRunDiag_v7 with dcgmDiagResponse_v10
    drd = dcgm_structs.c_dcgmRunDiag_v7()
    drd.version = dcgm_structs.dcgmRunDiag_version7
    drd.gpuList = str(gpuId)
    drd.validate = 1
    response = dcgm_structs.c_dcgmDiagResponse_v10()
    response.version = dcgm_structs.dcgmDiagResponse_version10
    response.levelOneTestCount = 0
    ret = localDcgmActionValidate_v2(handle, drd, response)
    assert ret == dcgm_structs.DCGM_ST_OK, f"ret: [{ret}]"
    assert response.version == dcgm_structs.dcgmDiagResponse_version10
    assert response.levelOneTestCount != 0, f"response.levelOneTestCount: [{response.levelOneTestCount}]"

    # Test dcgmRunDiag_v7 with dcgmDiagResponse_v9
    drd = dcgm_structs.c_dcgmRunDiag_v7()
    drd.version = dcgm_structs.dcgmRunDiag_version7
    drd.gpuList = str(gpuId)
    drd.validate = 1
    response = dcgm_structs.c_dcgmDiagResponse_v9()
    response.version = dcgm_structs.dcgmDiagResponse_version9
    response.levelOneTestCount = 0
    ret = localDcgmActionValidate_v2(handle, drd, response)
    assert ret == dcgm_structs.DCGM_ST_OK, f"ret: [{ret}]"
    assert response.version == dcgm_structs.dcgmDiagResponse_version9
    assert response.levelOneTestCount != 0, f"response.levelOneTestCount: [{response.levelOneTestCount}]"

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100x4-and-DGX.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_run_diag_backwards_compatibility_standalone(handle, gpuIds):
    helper_check_dcgm_run_diag_backwards_compatibility(handle, gpuIds[0])

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100x4-and-DGX.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_run_diagnostic_backwards_compatibility(handle, gpuIds):
    def localDcgmRunDiagnostic(dcgm_handle, group_id, response):
        fn = dcgm_structs._dcgmGetFunctionPointer("dcgmRunDiagnostic")
        ret = fn(dcgm_handle, group_id, dcgm_structs.DCGM_DIAG_LVL_SHORT, byref(response))
        return ret

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    createdGroup = dcgmSystem.GetGroupWithGpuIds("capoo_as_group_name", [gpuIds[0]])
    createdGroupId = createdGroup.GetId().value

    # Test "latest" (may be redundant with some tests below)
    response = dcgm_structs.c_dcgmDiagResponse_v12()
    response.version = dcgm_structs.dcgmDiagResponse_version12
    response.numTests = 0
    ret = localDcgmRunDiagnostic(handle, createdGroupId, response)
    assert ret == dcgm_structs.DCGM_ST_OK, f"ret: [{ret}]"
    assert response.version == dcgm_structs.dcgmDiagResponse_version12
    assert response.numTests == 1, f"response.numTests: [{response.numTests}]"
    assert response.tests[0].name == "software", f"response.tests[0].name: [{response.tests[0].name}]"

    # Test dcgmDiagResponse_v11
    response = dcgm_structs.c_dcgmDiagResponse_v11()
    response.version = dcgm_structs.dcgmDiagResponse_version11
    response.numTests = 0
    ret = localDcgmRunDiagnostic(handle, createdGroupId, response)
    assert ret == dcgm_structs.DCGM_ST_OK, f"ret: [{ret}]"
    assert response.version == dcgm_structs.dcgmDiagResponse_version11
    assert response.numTests == 1, f"response.numTests: [{response.numTests}]"
    assert response.tests[0].name == "software", f"response.tests[0].name: [{response.tests[0].name}]"

    # Test dcgmDiagResponse_v10
    response = dcgm_structs.c_dcgmDiagResponse_v10()
    response.version = dcgm_structs.dcgmDiagResponse_version10
    response.levelOneTestCount = 0
    ret = localDcgmRunDiagnostic(handle, createdGroupId, response)
    assert ret == dcgm_structs.DCGM_ST_OK, f"ret: [{ret}]"
    assert response.version == dcgm_structs.dcgmDiagResponse_version10
    assert response.levelOneTestCount != 0, f"response.levelOneTestCount: [{response.levelOneTestCount}]"

    # Test dcgmDiagResponse_v9
    response = dcgm_structs.c_dcgmDiagResponse_v9()
    response.version = dcgm_structs.dcgmDiagResponse_version9
    response.levelOneTestCount = 0
    ret = localDcgmRunDiagnostic(handle, createdGroupId, response)
    assert ret == dcgm_structs.DCGM_ST_OK, f"ret: [{ret}]"
    assert response.version == dcgm_structs.dcgmDiagResponse_version9
    assert response.levelOneTestCount != 0, f"response.levelOneTestCount: [{response.levelOneTestCount}]"

checked_gpus = {} # Used to track that a GPU has been verified as passing
# Makes sure a very basic diagnostic passes and returns a DcgmDiag object
def helper_verify_diag_passing(handle, gpuIds, testNames=TEST_MEMTEST, params="memtest.test_duration=15", version=g_latestDiagRunVer, useFakeGpus=False):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testNames, paramsStr=params, version=version)
    dd.SetClocksEventMask(0) # We explicitly want to fail for throttle reasons since this test inserts throttling errors 
                             # for verification
    if useFakeGpus:
        dd.UseFakeGpus()

    # If we've already chchecked this GPU, then use the previous result
    runDiag = False
    for gpuId in gpuIds:
        if gpuId in checked_gpus:
            if checked_gpus[gpuId] == False:
                test_utils.skip_test("Skipping because GPU %s does not pass memtest Perf test. "
                                     "Please verify whether the GPU is supported and healthy." % gpuId)
        else:
            runDiag = True

    if runDiag == False:
        return dd

    response = test_utils.diag_execute_wrapper(dd, handle)
    for gpuId in gpuIds:
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuId )
        if not check_diag_result_pass(response, entityPair, testNames):
            checked_gpus[gpuId] = False
            test_utils.skip_test("Skipping because GPU %s does not pass SM Perf test. "
                                 "Please verify whether the GPU is supported and healthy." % gpuId)
        else:
            checked_gpus[gpuId] = True

    return dd

def find_any_error_matching(response, entityPairs, testName, msg):
    # Return `True`, `foundMsg` if any test matches `entityPair`, `testName` and `msg`, `False`, "" otherwise.
    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.name == testName:
            break
    if test.name == testName:
        matchingErr = next(filter(lambda cur: any((cur.entity.entityId == ep.entityId and cur.entity.entityGroupId == ep.entityGroupId) for ep in entityPairs) and msg in cur.msg,
                           map(lambda errIdx: response.errors[errIdx], test.errorIndices[:min(test.numErrors, dcgm_structs.DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)])), None)
        if matchingErr:
            return True, "%s" % matchingErr.msg
    # No result was found, so no error was found
    return False, ""

def find_clocks_event_failure(response, gpuId, testName):
    if type(testName) != str:
        raise TypeError("usage: find_clocks_event_failure(response, gpuId:int, testName:str)")
    return find_any_error_matching(response, [ dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId) ], testName,
                                   'clocks event')

# Inject a thermal violation and excessive temperature, demonstrating the thermal violation is reported
def helper_test_thermal_violations(handle, gpuIds):
    gpuId = gpuIds[0]
    testName = TEST_DIAGNOSTIC
    injected_value = 2344122048

    dcgm_field_injection_helpers.inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
                                              injected_value, 180, verifyInsertion=True)

    dcgm_field_injection_helpers.inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
                                              199, 180, verifyInsertion=True)

    # Start the diag
    dd = DcgmDiag.DcgmDiag([gpuId], testNamesStr=testName, paramsStr='diagnostic.test_duration=10')
    dd.UseFakeGpus()
    response = dd.Execute(handle)
    foundError, errmsg = find_any_error_matching(response, [ dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId) ],
                                                 testName, "hermal violations")
    if foundError:
        import re
        match = re.search(r"totaling.*?seconds", errmsg)
        assert match, "Expected to find 'totaling <seconds> seconds' in error message but found %s" % errmsg
    else:
        # Didn't find an error
        assert False, "Thermal violations were injected but no errors were found"

    assert response.numTests == 2
    assert response.numResults == 2
    assert response.results[0].result == dcgm_structs.DCGM_DIAG_RESULT_PASS
    assert response.results[1].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL

# Inject a thermal violation without other error condition, demonstrating the thermal violation is NOT reported
def helper_test_silent_thermal_violations(handle, gpuIds):
    gpuId = gpuIds[0]
    testName = TEST_DIAGNOSTIC

    # First verify the test passes without any injected values
    dd = DcgmDiag.DcgmDiag([gpuId], testNamesStr=testName, paramsStr='diagnostic.test_duration=5')
    dd.UseFakeGpus()
    response = dd.Execute(handle)
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId)
    assert check_diag_result_pass(response, entityPair, testName), \
                                  "Diagnostic should pass prior to error injection."

    injected_value = 2344122048

    dcgm_field_injection_helpers.inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
                                              injected_value, 180, verifyInsertion=True)

    # Start the diag
    dd = DcgmDiag.DcgmDiag([gpuId], testNamesStr=testName, paramsStr='diagnostic.test_duration=5')
    dd.UseFakeGpus()
    response = dd.Execute(handle)
    foundError, errmsg = find_any_error_matching(response, [ dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId) ],
                                                 testName, "hermal violations")
    assert not foundError, "Unexpected thermal violation error: '%s'" % errmsg

    assert response.numTests == 2
    assert response.numResults == 2
    assert response.results[0].result == dcgm_structs.DCGM_DIAG_RESULT_PASS
    # PASS should be expected here, but this reflects a current behavior
    assert response.results[1].result != dcgm_structs.DCGM_DIAG_RESULT_FAIL

@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_thermal_violations_standalone(handle, gpuIds):
    helper_test_thermal_violations(handle, gpuIds)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_silent_thermal_violations_standalone(handle, gpuIds):
    helper_test_silent_thermal_violations(handle, gpuIds)


#####
# Helper method for inserting errors and performing the diag
def perform_diag_with_clocks_event_mask_and_verify(dd, handle, gpuId, inserted_error, clocks_event_mask, shouldPass, failureMsg):
    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCKS_EVENT_REASONS
    interval = 0.1
    if clocks_event_mask is not None:
        dd.SetClocksEventMask(clocks_event_mask)

    inject_value(handle, gpuId, fieldId, inserted_error, injection_offset, True, repeatCount=5)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True, repeatCount=5)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, inserted_error, checkInterval=interval, maxWait=15, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    # Start the diag
    response = test_utils.diag_execute_wrapper(dd, handle)

    # Check for pass or failure as per the shouldPass parameter
    clocks_event, errMsg = find_clocks_event_failure(response, gpuId, TEST_MEMTEST)
    if shouldPass:    
        assert clocks_event == False, "Expected to not have a clocks event error but found %s" % errMsg
    else:
        assert clocks_event == True, "Expected to find a clocks event error but did not (%s)" % errMsg


def helper_test_clocks_event_mask_fail_hw_slowdown(handle, gpuId):
    """
    Verifies that the clocks event ignore mask ignores the masked clocks event reasons.
    """
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    #####
    # Insert a clocks event error and verify that the test fails
    perform_diag_with_clocks_event_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN,
        clocks_event_mask=0, shouldPass=False, failureMsg="Expected test to fail because of clocks event"
    )

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_mask_fail_hw_slowdown(handle, gpuIds):
    helper_test_clocks_event_mask_fail_hw_slowdown(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus(2)
def test_run_injection(handle, gpuIds):
    helper_test_clocks_event_mask_fail_hw_slowdown(handle, gpuIds[0])

def helper_test_clocks_event_mask_ignore_hw_slowdown(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert clocks event error and set clocks event mask to ignore it (as integer value)
    perform_diag_with_clocks_event_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN,
        clocks_event_mask=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN, shouldPass=True, 
        failureMsg="Expected test to pass because clocks event mask (integer bitmask) ignores the clocks event reason"
    )

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_mask_ignore_hw_slowdown(handle, gpuIds):
    helper_test_clocks_event_mask_ignore_hw_slowdown(handle, gpuIds[0])

def helper_test_clocks_event_mask_ignore_hw_slowdown_string(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert clocks event error and set clocks event mask to ignore it (as string name)
    perform_diag_with_clocks_event_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN,
        clocks_event_mask="HW_SLOWDOWN", shouldPass=True, 
        failureMsg="Expected test to pass because clocks event mask (named reason) ignores the clocks event reason"
    )

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_mask_ignore_hw_slowdown_string(handle, gpuIds):
    helper_test_clocks_event_mask_ignore_hw_slowdown_string(handle, gpuIds[0])

def helper_test_clocks_event_mask_fail_double_inject_ignore_one(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert two clocks event errors and set clocks event mask to ignore only one (as integer)
    perform_diag_with_clocks_event_mask_and_verify(
        dd, handle, gpuId,
        inserted_error=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN | dcgm_fields.DCGM_CLOCKS_EVENT_REASON_SW_THERMAL, 
        clocks_event_mask=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN, shouldPass=False, 
        failureMsg="Expected test to fail because clocks event mask (integer bitmask) ignores one of the clocks event reasons"
    )

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_mask_fail_double_inject_ignore_one(handle, gpuIds):
    helper_test_clocks_event_mask_fail_double_inject_ignore_one(handle, gpuIds[0])

def helper_test_clocks_event_mask_fail_double_inject_ignore_one_string(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert two throttling errors and set throttle mask to ignore only one (as string name)
    perform_diag_with_clocks_event_mask_and_verify(
        dd, handle, gpuId,
        inserted_error=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN | dcgm_fields.DCGM_CLOCKS_EVENT_REASON_SW_THERMAL, 
        clocks_event_mask="HW_SLOWDOWN", shouldPass=False, 
        failureMsg="Expected test to fail because clocks event mask (named reason) ignores one of the clocks event reasons"
    )

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_context_create(handle, gpuIds):
    helper_verify_diag_passing(handle, gpuIds, "context_create", params="")

@test_utils.run_with_standalone_host_engine(120, heEnv={'__DCGM_PCIE_AER_COUNT' : '100'})
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_pcie_failure(handle, gpuIds):
    if test_utils.get_build_type() != "Debug":
        test_utils.skip_test("Debug test only")
    testName = "pcie"
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr=testName, paramsStr="pcie.test_duration=60;pcie.test_with_gemm=true;pcie.is_allowed=true",
                           version=g_latestDiagRunVer)
    response = test_utils.diag_execute_wrapper(dd, handle)
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuIds[0] )
    assert check_diag_result_fail(response, entityPair, testName), "No failure detected in diagnostic"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_pcie_failure_effective_ber(handle, gpuIds):
    testName = "pcie"
    gpuId = gpuIds[0]
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER, 1000, 600, True, repeatCount=5)

    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=testName, paramsStr="pcie.test_duration=4;pcie.is_allowed=true;pcie.test_with_gemm=false;pcie.test_broken_p2p=false", version=g_latestDiagRunVer)
    dd.SetWatchFrequency(5000000)
    response = test_utils.diag_execute_wrapper(dd, handle)

    # Verify that the effective BER is detected
    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.name == testName:
            break
    assert test.name == testName, f"Expected fail result for test {testName} but none was found"
    assert test.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, f"Expected fail result for test {testName} but got {test.result}"
    assert test.numErrors > 0, f"Expected at least 1 error for test {testName}"
    foundEffectiveBer = False
    for i in range(test.numErrors):
        errIdx = test.errorIndices[i]
        if response.errors[errIdx].code == dcgm_errors.DCGM_FR_NVLINK_EFFECTIVE_BER_THRESHOLD:
            foundEffectiveBer = True
            break
    assert foundEffectiveBer, f"Expected DCGM_FR_NVLINK_EFFECTIVE_BER_THRESHOLD for test {testName} but none was found"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_pcie_failure_symbol_errors(handle, gpuIds):
    testName = "pcie"
    gpuId = gpuIds[0]
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS, 1, 600, True, repeatCount=5)

    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=testName, paramsStr="pcie.test_duration=4;pcie.is_allowed=true;pcie.test_with_gemm=false;pcie.test_broken_p2p=false", version=g_latestDiagRunVer)
    dd.SetWatchFrequency(5000000)
    response = test_utils.diag_execute_wrapper(dd, handle)

    foundError, _ = find_any_error_matching(response, [ dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId) ],
                                                 testName, "nvlink_symbol_err")
    assert foundError, "DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS was injected but no errors were found"


@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_mask_fail_double_inject_ignore_one_string(handle, gpuIds):
    helper_test_clocks_event_mask_fail_double_inject_ignore_one_string(handle, gpuIds[0])

def helper_test_clocks_event_mask_fail_ignore_different_clocks_event(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert throttling error and set throttle mask to ignore a different reason (as integer value)
    perform_diag_with_clocks_event_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN,
        clocks_event_mask=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_POWER_BRAKE, shouldPass=False, 
        failureMsg="Expected test to fail because clocks event mask (integer bitmask) ignores different clocks event reason"
    )

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_mask_fail_ignore_different_clocks_event(handle, gpuIds):
    helper_test_clocks_event_mask_fail_ignore_different_clocks_event(handle, gpuIds[0])

def helper_test_clocks_event_mask_fail_ignore_different_clocks_event_string(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Insert throttling error and set throttle mask to ignore a different reason (as string name)
    perform_diag_with_clocks_event_mask_and_verify(
        dd, handle, gpuId, inserted_error=dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN,
        clocks_event_mask="HW_POWER_BRAKE", shouldPass=False, 
        failureMsg="Expected test to fail because clocks event mask (named reason) ignores different clocks event reason"
    )

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_mask_fail_ignore_different_clocks_event_string(handle, gpuIds):
    helper_test_clocks_event_mask_fail_ignore_different_clocks_event_string(handle, gpuIds[0])

def helper_test_clocks_event_mask_pass_no_clocks_event(handle, gpuId):
    dd = helper_verify_diag_passing(handle, [gpuId], useFakeGpus=True)

    # Clear throttling reasons and mask to verify test passes
    dd.SetClocksEventMask("")
    perform_diag_with_clocks_event_mask_and_verify(
        dd, handle, gpuId, inserted_error=0, clocks_event_mask=None, shouldPass=True,
        failureMsg="Expected test to pass because there is no clocks event"
    )

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_mask_pass_no_clocks_event(handle, gpuIds):
    helper_test_clocks_event_mask_pass_no_clocks_event(handle, gpuIds[0])

def wait_host_engine_ready(handle, gpuId):
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr='1')
    success = False
    start = time.time()
    while not success and (time.time() - start) <= 3:
        try:
            _ = test_utils.diag_execute_wrapper(dd, handle)
            success = True
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_DIAG_ALREADY_RUNNING):
            # Only acceptable error due to small race condition between the nvvs process exiting and
            # hostengine actually processing the exit. We try for a maximum of 3 seconds since this
            # should be rare and last only for a short amount of time
            time.sleep(1.5)

def get_dcgmi_path():
    # paths to dcgmi executable
    paths = {
        "Linux_32bit": "./apps/x86/dcgmi",
        "Linux_64bit": "./apps/amd64/dcgmi",
        "Linux_aarch64": "./apps/aarch64/dcgmi"
    }
    # Verify test is running on a supported platform
    if utils.platform_identifier not in paths:
        test_utils.skip_test("Dcgmi is not supported on the current platform.")
    return paths[utils.platform_identifier]

def helper_check_diag_stop_on_interrupt_signals(handle, gpuId):
    """
    Verifies that a launched diag is stopped when the dcgmi executable recieves a SIGINT, SIGHUP, SIGQUIT, or SIGTERM
    signal.
    """
    # First check whether the GPU is healthy/supported
    testName = "memtest"
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=testName, paramsStr="memtest.test_duration=2",
                           version=g_latestDiagRunVer)
    start_time = time.time()
    response = test_utils.diag_execute_wrapper(dd, handle)
    end_time = time.time()
    logger.info(f"{end_time - start_time:.2f} seconds spent on checking whether the GPU is healthy/supported.")
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuId )
    if not check_diag_result_pass(response, entityPair, testName):
        test_utils.skip_test("Skipping because GPU %s does not pass memtest. "
                             "Please verify whether the GPU is supported and healthy." % gpuId)

    dcgmi_path = get_dcgmi_path()
    def verify_exit_code_on_signal(signum):
        # Ensure that host engine is ready to launch a new diagnostic
        wait_host_engine_ready(handle, gpuId)

        diagApp = AppRunner(dcgmi_path, args=["diag", "-r", "memtest", "-i", "%s" % gpuId,
                                              "-d", "INFO", "--debugLogFile", "/tmp/nvvs.log"])
        # Start the diag
        diagApp.start(timeout=60)
        logger.info("Launched dcgmi process with pid: %s" % diagApp.getpid())

        # Ensure diag is running before sending interrupt signal
        running, debug_output = dcgm_internal_helpers.check_nvvs_process(want_running=True, delay=0.1, attempts=250)
        assert running, "The nvvs process did not start within 25 seconds: %s" % (debug_output)
        # There is a small race condition here - it is possible that the hostengine sends a SIGTERM before the
        # nvvs process has setup a signal handler, and so the nvvs process does not stop when SIGTERM is sent.
        # We sleep for 0.1 second to reduce the possibility of this scenario
        time.sleep(0.1)
        start_time = time.time()
        diagApp.signal(signum)
        retCode = diagApp.wait()
        end_time = time.time()
        # Check the return code and stdout/stderr output before asserting for better debugging info
        if retCode == 0:
            logger.error("Got retcode '%s' from launched diag." % retCode)
            if diagApp.stderr_lines or diagApp.stdout_lines:
                logger.info("dcgmi output:")
                for line in diagApp.stdout_lines:
                    logger.info(line)
                for line in diagApp.stderr_lines:
                    logger.error(line)
        assert retCode != 0, "Expected a non-zero exit code, but got 0"
        assert diagApp.retvalue() != AppRunner.RETVALUE_TIMEOUT, "The process should not be killed by timeout."
        logger.info(f"{end_time - start_time:.2f} seconds spent on handling signal.")
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

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus() #CC makes this too slow
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_stop_on_signal_standalone(handle, gpuIds):
    helper_check_diag_stop_on_interrupt_signals(handle, gpuIds[0])

def test_dcgm_diag_stop_on_interrupt_signals_dcgmi_embedded_itself():
    # Don't proceed if there's a residual nvvs process running
    not_running, debug_output = dcgm_internal_helpers.check_nvvs_process(want_running=False, delay=0.0, attempts=1)
    assert not_running, "A residual nvvs process is already executing and should be terminated before running this test again. pgrep output:\n%s" \
        % debug_output

    dcgmi_path = get_dcgmi_path()
    diagApp = AppRunner(dcgmi_path, args=["diag", "-r", "memtest", "-i", "0",
                                                "-p", "memtest.test_duration=2",
                                                "-d", "INFO", "--debugLogFile", "/tmp/nvvs.log"], env=test_utils.smallFbModeEnv)
    start_time = time.time()
    diagApp.start(timeout=None)
    retCode = diagApp.wait()
    end_time = time.time()
    logger.info(f"{end_time - start_time:.2f} seconds spent on checking whether the GPU is healthy/supported.")
    if retCode != 0:
        diagApp.validate()
        test_utils.skip_test("Skip test due to basic memtest failed.")

    def verify_exit_code_on_signal(signum):
        diagApp = AppRunner(dcgmi_path, args=["diag", "-r", "memtest", "-i", "0",
                                                "-d", "INFO", "--debugLogFile", "/tmp/nvvs.log"], env=test_utils.smallFbModeEnv)
        # Start the diag
        diagApp.start(timeout=60)
        logger.info("Launched dcgmi process with pid: %s" % diagApp.getpid())

        # Ensure diag is running before sending interrupt signal
        running, debug_output = dcgm_internal_helpers.check_nvvs_process(want_running=True, delay=0.1, attempts=250)
        assert running, "The nvvs process did not start within 25 seconds: %s" % (debug_output)
        # There is a small race condition here - it is possible that the we sends a signal before the
        # nvvs process has setup a signal handler.
        # We sleep for 0.1 second to reduce the possibility of this scenario
        time.sleep(0.1)
        start_time = time.time()
        diagApp.signal(signum)
        retCode = diagApp.wait()
        end_time = time.time()
        # Check the return code and stdout/stderr output before asserting for better debugging info
        if retCode == 0:
            logger.error("Got retcode '%s' from launched diag." % retCode)
            if diagApp.stderr_lines or diagApp.stdout_lines:
                logger.info("dcgmi output:")
                for line in diagApp.stdout_lines:
                    logger.info(line)
                for line in diagApp.stderr_lines:
                    logger.error(line)
        assert retCode != 0, "Expected a non-zero exit code, but got 0"
        assert diagApp.retvalue() != AppRunner.RETVALUE_TIMEOUT, "The process should not be killed by timeout."
        logger.info(f"{end_time - start_time:.2f} seconds spent on handling signal.")
        # Since the app returns a non zero exit code, we call the validate method to prevent false
        # failures from the test framework
        diagApp.validate()
        # Give the launched nvvs process 15 seconds to terminate.
        not_running, debug_output = dcgm_internal_helpers.check_nvvs_process(want_running=False, attempts=50)
        assert not_running, "The launched nvvs process did not terminate within 25 seconds. pgrep output:\n%s" \
                % debug_output

    # Verify return code on SIGINT
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

# See test_diag_stats.py: helper_test_stats_file_basics for refactoring and multi-maintenance opportunity
def helper_verify_log_file_creation(handle, gpuIds):
    testName = TEST_MEMTEST
    dd = helper_verify_diag_passing(handle, gpuIds, testNames=testName, params="memtest.test_duration=10", useFakeGpus=True)
    logname = '/tmp/tmp_test_debug_log'
    dd.SetDebugLogFile(logname)
    dd.SetDebugLevel(5)
    response = test_utils.diag_execute_wrapper(dd, handle)

    sysErrs = sum(1 for err in filter(lambda cur: cur.testId == dcgm_structs.DCGM_DIAG_RESPONSE_SYSTEM_ERROR,
                                      response.errors[:min(response.numErrors, dcgm_structs.DCGM_DIAG_RESPONSE_ERRORS_MAX)]))
    if sysErrs == 0:
        skippedAll = True
        passedCount = 0
        errmsg = ""

        # For each result
        #   if any result is pass, fail, or warn, skippedAll = False
        #   count passes
        #   count gpus
        #   record errors and their associated entity

        for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
            if test.name == testName:
                break
        assert test.name == testName, "Expected results for test %s, but none were found" % testName
        results = map(lambda resIdx: response.results[resIdx], test.resultIndices[:min(test.numResults, dcgm_structs.DCGM_DIAG_TEST_RUN_RESULTS_MAX)])
        for result in results:
            if result.result not in { dcgm_structs.DCGM_DIAG_RESULT_SKIP, dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN }:
                skippedAll = False
                if result.result == dcgm_structs.DCGM_DIAG_RESULT_PASS:
                    passedCount += 1
                else:
                    # Errors/warnings will be collected separately
                    pass

        # Collect errors/warnings
        errors = map(lambda errIdx: response.errors[errIdx], test.errorIndices[:min(test.numErrors, dcgm_structs.DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)])
        for error in errors:
            if error.entity.entityGroupId == dcgm_fields.DCGM_FE_NONE:
                errmsg = "%s %s" % (errmsg, error.msg)
            elif error.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU:
                errmsg = "%s GPU %d failed: %s " % (errmsg, error.entity.entityId, error.msg)
            else:
                errmsg = "%s Entity (grp=%d, id=%d) failed: %s " % (errmsg, error.entity.entityGroupId, error.entity.entityId, error.msg)

        gpuCount = sum(1 for gpu in filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU,
                                           response.entities[:min(response.numEntities, dcgm_structs.DCGM_DIAG_RESPONSE_ENTITIES_MAX)]))

        if skippedAll == False:
            detailedMsg = "passed on %d of %d GPUs" % (passedCount, gpuCount)
            if errmsg:
                detailedMsg = "%s and had these errors: %s" % (detailedMsg, errmsg)
                logger.info("%s when running the %s test" % (detailedMsg, testName))
            assert os.path.isfile(logname), "Logfile '%s' was not created and %s" % (logname, detailedMsg)
        else:
            logger.info("The diagnostic was skipped, so we cannot run this test.")
    else:
        logger.info("The diagnostic had a problem when executing, so we cannot run this test.")


@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_verify_log_file_creation_standalone(handle, gpuIds):
    helper_verify_log_file_creation(handle, gpuIds)

def helper_clocks_event_masking_failures(handle, gpuId):
    #####
    testName = "memtest"
    # First check whether the GPU is healthy
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=testName, paramsStr="memtest.test_duration=2",
                           version=dcgm_structs.dcgmRunDiag_version10)
    dd.SetClocksEventMask(0) # We explicitly want to fail for clocks event reasons since this test inserts those errors 
                          # for verification
    dd.UseFakeGpus()
    response = test_utils.diag_execute_wrapper(dd, handle)
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuId )
    if not check_diag_result_pass(response, entityPair, testName):
        test_utils.skip_test("Skipping because GPU %s does not pass %s. "
                             "Please verify whether the GPU is supported and healthy." % (gpuId, testName))
    
    #####
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="memtest", paramsStr="memtest.test_duration=15",
                           version=dcgm_structs.dcgmRunDiag_version10)
    dd.SetClocksEventMask(0)
    dd.UseFakeGpus()

    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCKS_EVENT_REASONS
    insertedError = dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN
    interval = 0.1

    logger.info("Injecting benign errors")
    inject_value(handle, gpuId, fieldId, 3, 1, True)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, 3, checkInterval=interval, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    
    logger.info("Injecting actual errors")
    inject_value(handle, gpuId, fieldId, insertedError, injection_offset, True, repeatCount=5)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True, repeatCount=5)

    logger.info("Started diag")
    response = test_utils.diag_execute_wrapper(dd, handle)
    # Verify that the inserted values are visible in DCGM
    # Max wait of 8 is because of 5 second offset + 2 seconds required for 20 matches + 1 second buffer.
    assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, insertedError, checkInterval=0.1, numMatches=1, maxWait=8), \
            "Expected inserted errors to be visible in DCGM"
                
    clocks_event, errMsg = find_clocks_event_failure(response, gpuId, testName)
    assert clocks_event, "Expected to find clocks event error, but did not: (%s)" % errMsg

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_dcgm_diag_clocks_event_masking_failures_standalone(handle, gpuIds):
    helper_clocks_event_masking_failures(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
@test_utils.run_only_with_nvml()
def test_dcgm_diag_handle_concurrency_standalone(handle, gpuIds):
    '''
    Test that we can use a DCGM handle concurrently with a diagnostic running
    '''
    diagDuration = 10

    gpuId = gpuIds[0]

    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="memtest", paramsStr="memtest.test_duration=%d" % diagDuration,
                           version=dcgm_structs.dcgmRunDiag_version10)

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

    dd = DcgmDiag.DcgmDiag(gpuIds=[failGpuId], testNamesStr="memtest", paramsStr="memtest.test_duration=15", version=dcgm_structs.dcgmRunDiag_version10)
    dd.SetClocksEventMask(0) # We explicitly want to fail for clocks event reasons since this test inserts those errors 
                          # for verification
    dd.UseFakeGpus()
    dd.SetStatsPath(testDir)
    dd.SetStatsOnFail(1)

    # Setup injection app    
    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCKS_EVENT_REASONS
    insertedError = dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN
    interval = 0.1
    # Use an offset to make these errors start after the benign values
    inject_value(handle, failGpuId, fieldId, insertedError, injection_offset, True, repeatCount=5)
    inject_value(handle, failGpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True, repeatCount=5)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(failGpuId, fieldId, insertedError, checkInterval=interval, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    response = test_utils.diag_execute_wrapper(dd, handle)
    logger.info("Started diag")
                                                                                                     
    # Verify that responses are reported on a per gpu basis. Ensure the first GPU failed, and all others passed
    for gpuId in gpuIds:
        clocks_event, errMsg = find_clocks_event_failure(response, gpuId, "memtest")
        if gpuId == failGpuId:
            assert clocks_event, "Expected clocks event error but found none (%s)" % errMsg
        else:
            assert not clocks_event, "Expected not to find a clocks event error but found '%s'" % errMsg

def helper_per_gpu_responses_dcgmi(handle, gpuIds, testName, testParams):
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
        expected_retcode = c_uint8(dcgm_structs.DCGM_ST_NVVS_ERROR).value
        if retcode != expected_retcode:
            if app.stderr_lines or app.stdout_lines:
                    logger.info("dcgmi output:")
                    print_output(app)
        assert retcode == expected_retcode, \
            "Expected dcgmi diag to have retcode %s. Got return code %s" % (c_int8(expected_retcode).value, c_int8(retcode).value)
        app.validate() # non-zero exit code must be validated

    # Setup injection app
    interval = 0.1
    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCKS_EVENT_REASONS
    insertedError = dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN
    # Use an offset to make these errors start after the benign values
    inject_value(handle, gpuIds[0], fieldId, insertedError, injection_offset, True, repeatCount=5)
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True, repeatCount=5)
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuIds[0], fieldId, insertedError, checkInterval=interval, maxWait=5, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    # Verify dcgmi output
    gpuIdStrings = list(map(str, gpuIds))
    gpuList = ",".join(gpuIdStrings)
    args = ["diag", "-r", testName, "-p", testParams, "-f", gpuList, "--clocksevent-mask", "0"]
    dcgmiApp = DcgmiApp(args=args)

    logger.info("Verifying stdout output")
    verify_successful_dcgmi_run(dcgmiApp)
    # Verify dcgmi output shows per gpu results (crude approximation of verifying correct console output)
    test_header_found = False
    fail_gpu_found = False
    fail_gpu_text = "GPU%s: Fail" % gpuIds[0]
    check_for_warning = False
    warning_found = False
    for line in dcgmiApp.stdout_lines:
        if not test_header_found:
            if testName not in line:
                continue
            test_header_found = True
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

    if not (test_header_found and fail_gpu_found and warning_found):
        logger.info("dcgmi output:")
        print_output(dcgmiApp)

    assert test_header_found, "Expected to see '%s' header in output" % testName
    assert fail_gpu_found, "Expected to see %s in output" % fail_gpu_text
    assert warning_found, "Expected to see 'Warning' in output after GPU failure text"

    inject_value(handle, gpuIds[0], fieldId, insertedError, injection_offset, True, repeatCount=5)
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 1000, injection_offset, True, repeatCount=5)
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
    test_utils.diag_verify_json(output)

    try:
        assert len(output.get("DCGM Diagnostic", {}).get("test_categories", [])) == 2
        assert output["DCGM Diagnostic"]["test_categories"][1].get("category", None) == "Stress"
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["name"] == testName
        assert len(output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["results"]) >= 2
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["test_summary"]["status"] == "Fail"
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["entity_group"] == "GPU"
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["entity_id"] == gpuIds[0]
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["status"] == "Fail"
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["warnings"][0]["error_id"] == dcgm_errors.DCGM_FR_TEMP_VIOLATION
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["warnings"][0]["error_category"] == dcgm_errors.DCGM_FR_EC_HARDWARE_THERMAL
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["warnings"][0]["error_severity"] == dcgm_errors.DCGM_ERROR_MONITOR
        assert output["DCGM Diagnostic"]["test_categories"][1]["tests"][0]["results"][1]["status"] == "Pass"
    except AssertionError:
        print_output(dcgmiApp)
        raise AssertionError("dcgmi JSON output did not pass verification")

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
@test_utils.run_only_with_nvml()
def test_dcgm_diag_per_gpu_responses_standalone_api(handle, gpuIds):
    if len(gpuIds) < 2:
        test_utils.skip_test("Skipping because this test requires 2 or more GPUs with same SKU")

    if test_utils.is_clocks_event_masked_by_nvvs(handle, gpuIds[0], dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN):
        test_utils.skip_test("Skipping because this SKU ignores the clocks event we inject for this test")

    logger.info("Starting test for per gpu responses (API call)")
    outputFile = "stats_memtest.json"
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


@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
@test_utils.run_only_with_nvml()
def test_dcgm_diag_per_gpu_responses_standalone_dcgmi(handle, gpuIds):
    if len(gpuIds) < 2:
        test_utils.skip_test("Skipping because this test requires 2 or more GPUs with same SKU")

    if test_utils.is_clocks_event_masked_by_nvvs(handle, gpuIds[0], dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN):
        test_utils.skip_test("Skipping because this SKU ignores the clocks event we inject for this test")

    logger.info("Starting test for per gpu responses (dcgmi output)")
    helper_per_gpu_responses_dcgmi(handle, gpuIds, TEST_MEMTEST, "memtest.test_duration=5,pcie.max_pcie_replays=1")

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
@test_utils.run_only_with_nvml()
def test_dcgm_diag_memtest_fails_standalone_dcgmi(handle, gpuIds):
    if len(gpuIds) < 2:
        test_utils.skip_test("Skipping because this test requires 2 or more GPUs with same SKU")

    if test_utils.is_clocks_event_masked_by_nvvs(handle, gpuIds[0], dcgm_fields.DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN):
        test_utils.skip_test("Skipping because this SKU ignores the clocks event we inject for this test")

    logger.info("Starting test for per gpu responses (dcgmi output)")
    helper_per_gpu_responses_dcgmi(handle, gpuIds, TEST_MEMTEST, "memtest.test_duration=15")

def helper_test_diagnostic_config_usage(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="diagnostic", paramsStr="diagnostic.test_duration=10")
    dd.SetConfigFileContents("%YAML 1.2\n\ncustom:\n- custom:\n    diagnostic:\n      max_sbe_errors: 1")
    
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, 1000, injection_offset, True, repeatCount=5)
    
    response = test_utils.diag_execute_wrapper(dd, handle)
    
    assert check_diag_result_non_passing(response, dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuIds[0]), "diagnostic"), \
        "Should have a failure due to injected SBEs, but got passing result"

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_diagnostic_config_usage_standalone(handle, gpuIds):
    helper_test_diagnostic_config_usage(handle, gpuIds)

def helper_test_dcgm_short_diagnostic_run(handle, gpuIds):
    testName = TEST_DIAGNOSTIC
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testName, paramsStr="diagnostic.test_duration=15")
    response = test_utils.diag_execute_wrapper(dd, handle)
    for gpuId in gpuIds:
        gpuEntity = dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId)
        if check_diag_result_non_running(response, gpuEntity, testName):
            logger.info("Got status DCGM_DIAG_RESULT_SKIP for gpuId %d. This is expected if this GPU does not support the Diagnostic test." % gpuId)
            continue

        assert check_diag_result_non_failing(response, gpuEntity, testName), \
            "Should have passed the 15 second diagnostic for all GPUs"

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(2)
def test_memtest_failures_standalone(handle, gpuIds):
    testName=TEST_MEMTEST
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testName, paramsStr="memtest.test_duration=10")

    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, 1000, injection_offset, True, repeatCount=5)

    response = test_utils.diag_execute_wrapper(dd, handle)

    assert check_diag_result_non_passing(response, dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuIds[0]), testName), \
        "Should have a failure due to injected DBEs, but got passing result"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_nvbandwidth_failure(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="nvbandwidth", paramsStr="nvbandwidth.is_allowed=true;nvbandwidth.testcases=0,1")

    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL, 99, 0, True, repeatCount=5)

    response = test_utils.diag_execute_wrapper(dd, handle)
    nvbandwidth_test_idx = -1
    for i in range(response.numTests):
        if response.tests[i].name == "nvbandwidth":
            nvbandwidth_test_idx = i
            break
    assert nvbandwidth_test_idx != -1, "Should have nvbandwidth test result."
    resultIdx = response.tests[nvbandwidth_test_idx].resultIndices[0]
    assert response.results[resultIdx].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU, f"Should have entity group as DCGM_FE_GPU, but got {response.results[resultIdx].entity.entityGroupId}"
    assert response.results[resultIdx].entity.entityId == gpuIds[0], f"Should have entityId as {gpuIds[0]}, but got {response.results[resultIdx].entity.entityId}"
    assert response.results[resultIdx].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, \
                "Should have a failure due to high memory copy utilitization, but got passing result"

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_short_memtest_run(handle, gpuIds):
    testName = TEST_MEMTEST
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testName, paramsStr="memtest.test_duration=10;memtest.test10=false")
    response = test_utils.diag_execute_wrapper(dd, handle)
    for gpuId in gpuIds:
        gpuEntity = dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, gpuId )
        if check_diag_result_non_running(response, gpuEntity, testName):
            logger.info("Got status DCGM_DIAG_RESULT_SKIP for gpuId %d. This is expected if this GPU does not support the Diagnostic test." % gpuId)
            continue

        assert check_diag_result_non_failing(response, gpuEntity, testName), \
            "Should have passed the 15 second diagnostic for all GPUs"

@test_utils.run_with_diag_small_fb_mode() #Needs to be before host engine start
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_output(handle, gpuIds):
    if len(gpuIds) <= 1:
        test_utils.skip_test("Skipping because test requires >1 live gpus")

    try:
        testName = TEST_MEMTEST
        os.environ['__DCGM_DIAG_MEMTEST_FAIL_GPU'] = "1"
        dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testName, paramsStr="memtest.test_duration=10;memtest.test10=false")
        response = test_utils.diag_execute_wrapper(dd, handle)

        assert check_diag_result_non_failing(response, dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, 0 ), testName), \
                                            "GPU %d should have passed the 15 second diagnostic" % 0

        assert check_diag_result_non_passing(response, dcgm_structs.c_dcgmGroupEntityPair_t( dcgm_fields.DCGM_FE_GPU, 1 ), testName), \
                                            "GPU %d should NOT have passed the 15 second diagnostic" % 1

        assert response.numEntities != 0, "Should have found entities."
        for i in range(response.numEntities):
            if response.entities[i].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU:
                assert response.entities[i].serialNum != dcgmvalue.DCGM_STR_BLANK, \
                    f"Invalid gpu serial detected for gpu: {response.entities[i].entity.entityId}"
            else:
                assert response.entities[i].serialNum == dcgmvalue.DCGM_STR_BLANK, \
                    f"Invalid serial detected for entity: {response.entities[i].entity.entityId}"
    finally:
        del os.environ['__DCGM_DIAG_MEMTEST_FAIL_GPU']

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_short_diagnostic_run(handle, gpuIds):
    helper_test_dcgm_short_diagnostic_run(handle, gpuIds)


def helper_test_dcgm_diag_paused(handle, gpuIds):
    """
    Test that DCGM_ST_PAUSED is returned when the host engine is paused
    """
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="1")
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    try:
        dcgmSystem.PauseTelemetryForDiag()
        response = test_utils.diag_execute_wrapper(dd, handle)
        dcgmSystem.ResumeTelemetryForDiag()

    except dcgm_structs.DCGMError as e:
        assert e.value == dcgm_structs.DCGM_ST_PAUSED, "Expected DCGM_ST_PAUSED error"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus(1)
@test_utils.run_only_with_nvml()
def test_dcgm_diag_paused_standalone(handle, gpuIds):
    helper_test_dcgm_diag_paused(handle, gpuIds)

def helper_hbm_temperature_check(handle, gpuIds, flag):
    # ------------------------
    testsToCheck = (
                    TEST_MEMORY,
                    TEST_DIAGNOSTIC
    )

    # create new diag object
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="memory,diagnostic", paramsStr="diagnostic.test_duration=10")
    dd.UseFakeGpus()
    
    # ------------------------
    # set the temperature
    if flag == "fail":
        # set fake hbm temperature greater than max
        fake_curr_temperature = 1000
    
    else: 
        # set fake hbm temperature less than max
        fake_curr_temperature = 10

    # ------------------------
    # inject the temperature
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP, fake_curr_temperature, injection_offset, repeatCount=3, repeatOffset=5)

    # ------------------------
    # Verify that the inserted values are visible in DCGM before starting the diag        
    assert dcgm_internal_helpers.verify_field_value(gpuIds[0], dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP, fake_curr_temperature, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    # ------------------------
    # run the diag
    response = test_utils.diag_execute_wrapper(dd, handle)
    logger.info("Started diag")

    # ------------------------
    # Verify that the inserted values are visible in DCGM after running the diag
    assert dcgm_internal_helpers.verify_field_value(gpuIds[0], dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP, fake_curr_temperature, numMatches=1), \
        "Expected inserted values to be visible in DCGM"

    # ------------------------
    # check if the memory diagnostic fails/passes
    gpuCount = GetGpuCount(response)
    assert gpuCount == len(gpuIds), "Expected %d gpus, but found %d reported" % (len(gpuIds), gpuCount)
    
    # ------------------------
    logger.info("Checking HBM Temperature")

    for key in testsToCheck:
        for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
            if test.name == key:
                break
        if test.name != key:
            logger.info("Diag test: {} had no results in the response".format(key))
        else:
            wasSkipped = False
            if test.result == dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
                logger.info("Diag test: {} did not run".format(key))
                wasSkipped = True
            elif test.result == dcgm_structs.DCGM_DIAG_RESULT_SKIP:
                logger.info("Diag test: {} did not run and was skipped".format(key))
                wasSkipped = True
            else:
                result = next(filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and cur.entity.entityId == gpuIds[0] and
                                 cur.result in { dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN, dcgm_structs.DCGM_DIAG_RESULT_SKIP },
                                 map(lambda resIdx: response.results[resIdx], test.resultIndices[:min(test.numResults, dcgm_structs.DCGM_DIAG_TEST_RUN_RESULTS_MAX)])), None)
                if result:
                    if result.result == dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
                        logger.info("Diag test: {} did not run for gpu {}".format(key, gpuIds[0]))
                        wasSkipped = True
                    else:
                        logger.info("Diag test: {} did not run and was skipped for gpu {}".format(key, gpuIds[0]))
                        wasSkipped = True
            if not wasSkipped:
                if flag=="fail":
                    diag_result_assert_fail(response, gpuIds[0], key, "Expected a failure in test: {} due to 1000 degree inserted temp.".format(key), dcgm_errors.DCGM_FR_TEMP_VIOLATION)
                else:
                    diag_result_assert_pass(response, gpuIds[0], key, "Expected a pass in test {}".format(key))


@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(1)
@test_utils.run_only_with_nvml()
def test_dcgm_diag_hbm_temperature_fail(handle, gpuIds):
    logger.info("Starting test")
    helper_hbm_temperature_check(handle, gpuIds, "fail")


@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_with_injection_gpus(1)
@test_utils.run_only_with_nvml()
def test_dcgm_diag_hbm_temperature_pass(handle, gpuIds):
    logger.info("Starting test")
    helper_hbm_temperature_check(handle, gpuIds, "pass")

def helper_test_dcgm_diag_timing_out(handle, gpuIds, version):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="pcie", timeout=1, version=version)
    try:
        test_utils.diag_execute_wrapper(dd, handle)
    except Exception as e:
        assert isinstance(e, dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_TIMEOUT))
    else:
        test_utils.skip_test('Skip due to rapid pace of code execution.')

    for i in range(0,3):
        try:
            dcgm_agent_internal.dcgmStopDiagnostic(handle)
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_CHILD_NOT_KILLED) as e:
            if i == 2:
                nvvsPids = map(int, subprocess.check_output(["pidof", "nvvs"]).split())
                for childPid in nvvsPids:
                    logger.info("Force killing the NVVS process %d", childPid)
                    os.kill(childPid, 9)
            else:
                pass

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_timing_out(handle, gpuIds):
    helper_test_dcgm_diag_timing_out(handle, gpuIds, dcgm_structs.dcgmRunDiag_version9)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_timing_out_v7(handle, gpuIds):
    helper_test_dcgm_diag_timing_out(handle, gpuIds, dcgm_structs.dcgmRunDiag_version7)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_serial_nums(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="1")
    response = test_utils.diag_execute_wrapper(dd, handle)

    assert response.numEntities != 0, "Should have found entities."
    for i in range(response.numEntities):
        if response.entities[i].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU:
            assert response.entities[i].serialNum != dcgmvalue.DCGM_STR_BLANK, \
                f"Invalid gpu serial detected for gpu: {response.entities[i].entity.entityId}"
        else:
            assert response.entities[i].serialNum == dcgmvalue.DCGM_STR_BLANK, \
                f"Invalid serial detected for entity: {response.entities[i].entity.entityId}"

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_dev_ids(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="1")
    response = test_utils.diag_execute_wrapper(dd, handle)

    assert response.numEntities != 0, "Should have found entities."
    for i in range(response.numEntities):
        if response.entities[i].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU:
            assert response.entities[i].skuDeviceId != dcgmvalue.DCGM_STR_BLANK, \
                f"Invalid sku device id detected for gpu: {response.entities[i].entity.entityId}"
        else:
            assert response.entities[i].skuDeviceId == dcgmvalue.DCGM_STR_BLANK, \
                f"Invalid sku device id detected for entity: {response.entities[i].entity.entityId}"

@test_utils.run_only_as_root()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.run_only_with_live_cpus()
def test_dcgm_diag_will_fill_cpu_serials(handle, gpuIds, cpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], cpuIds=cpuIds, testNamesStr="1")
    response = test_utils.diag_execute_wrapper(dd, handle)
    hasCpuEntities = False
    for i in range(response.numEntities):
        if response.entities[i].entity.entityGroupId != dcgm_fields.DCGM_FE_CPU:
            continue
        hasCpuEntities = True
        assert response.entities[i].serialNum.find(dcgmvalue.DCGM_STR_BLANK) == -1,\
                f"Missing serial for cpu: [{response.entities[i].entity.entityId}]"
    assert hasCpuEntities, "Missing CPU entities"

@test_utils.run_only_as_root()
@test_utils.run_with_standalone_host_engine(120, heEnv={'DCGM_SUPPORT_NON_NVIDIA_CPU': '1'})
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_will_display_num_cpu(handle, gpuIds):
    dcgmiPath = get_dcgmi_path()
    diagApp = AppRunner(dcgmiPath, args=["diag", "-r", "1", "-i", f"{str(gpuIds[0])},cpu:*"])
    diagApp.start(timeout=60)
    retCode = diagApp.wait()
    if retCode != 0:
        logger.error(f"Got retcode [{retCode}] from launched diag.")
        logger.error(f"stdout: [{diagApp.stdout_lines}], stderr: [{diagApp.stderr_lines}]")
    assert retCode == 0, f"Actual ret code: [{retCode}]"
    expectedNumCpuStr = "Number of CPUs Detected"
    found = False
    for stdoutLine in diagApp.stdout_lines:
        if expectedNumCpuStr in stdoutLine:
            found = True
    assert found, f"Actual stdout: [{diagApp.stdout_lines}]"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_diag_nvbandwidth_env_var_restoration(handle, gpuIds):
    # Store original value if it exists
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        # Run nvbandwidth test
        dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="nvbandwidth", paramsStr="nvbandwidth.is_allowed=true;nvbandwidth.testcases=0,1")
        response = test_utils.diag_execute_wrapper(dd, handle)
        
        # Verify test results
        nvbandwidth_test_idx = -1
        for i in range(response.numTests):
            if response.tests[i].name == "nvbandwidth":
                nvbandwidth_test_idx = i
                break
        assert nvbandwidth_test_idx != -1, "Should have nvbandwidth test result."
        
        # Verify CUDA_VISIBLE_DEVICES was restored
        current_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if original_cuda_visible_devices:
            assert current_cuda_visible_devices == original_cuda_visible_devices, \
                f"CUDA_VISIBLE_DEVICES should be restored to {original_cuda_visible_devices}, but got {current_cuda_visible_devices}"
        else:
            assert current_cuda_visible_devices is None, \
                "CUDA_VISIBLE_DEVICES should be unset, but it was set"
            
    finally:
        # Restore original value in case test fails
        if original_cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

def helper_check_nvbandwidth_log_file_creation(handle, gpuIds):
    """
    Helper function to verify that the nvbandwidth log file is created when running the nvbandwidth test.
    """
    def validate_json_content(lines, source_name):
        """
        Helper function to validate JSON content in the lines.
        Skips warning lines and checks for the expected JSON format.
        
        Args:
            lines: List of lines to check
            source_name: Name of the source (for error messages)
            
        Returns:
            True if valid JSON content is found, False otherwise
        """

        def is_valid_json_content(json_lines):
            try:
                json.loads('\n'.join(json_lines))
                return True
            except json.JSONDecodeError:
                return False
        
        # Skip any warning lines until we find the first line that starts with "{"
        json_start_idx = -1
        for i, line in enumerate(lines):
            if line.startswith("{"):
                json_start_idx = i
                break
        
        if json_start_idx != -1:
            # Extract the JSON content starting from the "{" line
            json_lines = lines[json_start_idx:]

            # For nvvs.log, json_lines may contain other logging information
            if source_name != "nvvs.log":
                assert is_valid_json_content(json_lines), f"Invalid JSON content in {source_name}"                
            
            if len(json_lines) >= 2:
                # Check that the first two lines match the expected format
                assert json_lines[0] == '{', f"First JSON line in {source_name} should be '{{', but got '{json_lines[0]}'"
                assert json_lines[1].startswith('"nvbandwidth"'), f"Second JSON line in {source_name} should start with '\"nvbandwidth\"', but got '{json_lines[1]}'"
                return True
            else:
                assert False, f"Found JSON start in {source_name} but fewer than 2 lines: {json_lines}"
        else:
            assert False, f"Could not find JSON content starting with '{{' in {source_name}. Lines found: {lines}"
        
    
    # Determine the log directory
    log_dir = os.environ.get("DCGM_HOME_DIR", "/var/log/nvidia-dcgm")
    
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create log directory {log_dir}: {str(e)}")
            # If we can't create the directory, we should skip the test
            test_utils.skip_test(f"Skipping test because log directory {log_dir} doesn't exist and couldn't be created")
    
    # Check if we have write permissions to the log directory
    if not os.access(log_dir, os.W_OK):
        test_utils.skip_test(f"Skipping test because we don't have write permissions to log directory {log_dir}")
    
    log_file_path = os.path.join(log_dir, "dcgm_nvbandwidth.log")
    nvvs_log_path = os.path.join(log_dir, "nvvs.log")
    
    # Remove the log file if it exists to ensure we're testing the creation of a new file
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
        except Exception as e:
            logger.warning(f"Failed to remove existing log file {log_file_path}: {str(e)}")
    
    # Run the nvbandwidth test
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="nvbandwidth", paramsStr="nvbandwidth.is_allowed=true;nvbandwidth.testcases=0,1")
    
    # Execute the diagnostic
    response = test_utils.diag_execute_wrapper(dd, handle)
    
    # Verify the test ran
    nvbandwidth_test_idx = -1
    for i in range(response.numTests):
        if response.tests[i].name == "nvbandwidth":
            nvbandwidth_test_idx = i
            break
    assert nvbandwidth_test_idx != -1, "Should have nvbandwidth test result."
    
    # Check if the main log file exists
    if os.path.isfile(log_file_path) and os.path.getsize(log_file_path) > 0:
        # Check that the first two lines (after removing spaces) are '{' and '"nvbandwidth":'
        with open(log_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            validate_json_content(lines, f"log file {log_file_path}")
    else:    
        # Check nvvs.log for the "External command stdout:" line
        if os.path.isfile(nvvs_log_path):
            # Look for the "External command stdout:" line which indicates stdout was written to nvvs.log
            external_cmd_stdout_found = False
            
            with open(nvvs_log_path, 'r') as f:
                nvvs_log_content = f.read()
                
                # Find the last occurrence of "External command stdout:"
                external_cmd_stdout_pattern = "External command stdout:"
                last_position = nvvs_log_content.rfind(external_cmd_stdout_pattern)
                
                if last_position != -1:
                    external_cmd_stdout_found = True
                    
                    # Get content after the last occurrence
                    content_after = nvvs_log_content[last_position + len(external_cmd_stdout_pattern):]
                    
                    # Get the non-empty lines after the pattern
                    lines_after_stdout = [line.strip() for line in content_after.splitlines() if line.strip()]
                    validate_json_content(lines_after_stdout, "nvvs.log after 'External command stdout:'")
            
            # If the stdout log file doesn't exist, then "External command stdout:" must be found in nvvs.log
            assert external_cmd_stdout_found, f"'External command stdout:' was not found in nvvs.log"

@test_utils.run_only_as_root()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_nvbandwidth_log_file_creation(handle, gpuIds):
    """
    Test to verify that the nvbandwidth log file is created when running the nvbandwidth test.
    """
    helper_check_nvbandwidth_log_file_creation(handle, gpuIds)
