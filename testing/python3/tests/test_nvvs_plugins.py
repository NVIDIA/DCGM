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
import logger
import test_utils
import dcgm_fields
import dcgm_agent_internal
import dcgm_internal_helpers
import DcgmDiag
from DcgmDiag import check_diag_result_fail, check_diag_result_pass
import option_parser
import os
import subprocess

import signal
import threading
import time

import nvml_injection
import nvml_injection_structs
import dcgm_nvml
from _test_helpers import skip_test_if_no_dcgm_nvml
from apps.app_runner import AppRunner

from dcgm_field_injection_helpers import inject_value
from shutil import which as find_executable

TEST_DIAGNOSTIC = "diagnostic"
TEST_TARGETED_POWER = "targeted_power"
TEST_TARGETED_STRESS = "targeted_stress"
TEST_PCIE = "pcie"

# Most injection tests use memtest plugin, which also sleeps for 3 seconds

injection_offset = 3

################# General tests #################

##### Fail early behavior tests
def verify_early_fail_checks_for_test(handle, gpuId, test_name, extraTestInfo):
    # Helper method for verifying the fail early checks for the specified test.
    duration = 2 if test_name not in { TEST_TARGETED_POWER, "targeted power" } else 30 # Prevent false failures due to min
                                                                                # duration requirements for Targeted Power
    paramsStr = "%s.test_duration=%s" % (test_name, duration)

    ###
    # First verify that the given test passes for the gpu.
    # If it doesn't pass, skip test and add note to check GPU health
    logger.info("Checking whether %s test passes on GPU %s" % (test_name, gpuId))
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=test_name, paramsStr=paramsStr)
    response = test_utils.diag_execute_wrapper(dd, handle)
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId)
    if not check_diag_result_pass(response, entityPair, test_name):
        test_utils.skip_test("Not testing %s because GPU %s does not pass. "
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

    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 150, 15, True, repeatCount=10)

    # launch the diagnostic
    start = time.time()
    response = test_utils.diag_execute_wrapper(dd, handle)
    end = time.time()
    total_time = end - start

    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.name == test_name:
            break
    assert test.name == test_name, "Expected results for test %s, but none were found" % test_name

    result = next(filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and cur.entity.entityId == gpuId,
                         map(lambda resIdx: response.results[resIdx], test.resultIndices[:min(test.numResults, dcgm_structs.DCGM_DIAG_TEST_RUN_RESULTS_MAX)])), None)
    assert result, "Expected a result for gpu %d, test %s but none were found" % (gpuId, test_name)
    
    info = next(filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and cur.entity.entityId == gpuId,
                         map(lambda infoIdx: response.info[infoIdx], test.infoIndices[:min(test.numInfo, dcgm_structs.DCGM_DIAG_TEST_RUN_INFO_INDICES_MAX)])), None)
    error = next(filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and cur.entity.entityId == gpuId,
                         map(lambda errorIdx: response.errors[errorIdx], test.errorIndices[:min(test.numErrors, dcgm_structs.DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)])), None)

    exitEarlyMsg = "Expected %s test to exit early. Test took %ss to complete." % (test_name, total_time)
    if result:
        exitEarlyMsg = "%s\nGot result: %s" % (exitEarlyMsg, result.result)
        delim = ""
        infoMsg = ""
        errMsg = ""
        if info:
            infoMsg = "info: %s" % info.msg
            delim = ",\n "
        if error:
            errMsg = "error: %s" % error.msg
        if infoMsg or errMsg:
            exitEarlyMsg = "%s (\n%s%s%s)" % (exitEarlyMsg, info, delim, errMsg)

    assert total_time < duration, exitEarlyMsg

    # Verify the test failed
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId)
    injectedDbesMsg = "Expected %s test to fail due to injected dbes.\nGot result: %s" % (test_name, result.result)
    if infoMsg or errMsg:
        injectedDbesMsg = "%s (\n%s%s%s)" % (injectedDbesMsg, info, delim, errMsg)
    assert check_diag_result_fail(response, entityPair, test_name), injectedDbesMsg

    if extraTestInfo:
        for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
            if test.name == extraTestInfo[0]:
                break
        assert test.name != extraTestInfo[0] or test.name == extraTestInfo[0] and \
            test.result in [dcgm_structs.DCGM_DIAG_RESULT_SKIP, dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN], \
                "Expected extra test %s to be skipped since the first test failed.\nGot over result: %s" % (test.name, test.result)

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvvs_plugin_fail_early_diagnostic_standalone(handle, gpuIds):
    verify_early_fail_checks_for_test(handle, gpuIds[0], TEST_DIAGNOSTIC, None)

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvvs_plugin_fail_early_targeted_stress_standalone(handle, gpuIds):
    verify_early_fail_checks_for_test(handle, gpuIds[0], TEST_TARGETED_STRESS, None)

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvvs_plugin_fail_early_targeted_power_standalone(handle, gpuIds):
    verify_early_fail_checks_for_test(handle, gpuIds[0], TEST_TARGETED_POWER, None)

@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_nvvs_plugin_fail_early_two_tests_standalone(handle, gpuIds):
    extraTestInfo = [ TEST_PCIE ]
    verify_early_fail_checks_for_test(handle, gpuIds[0], TEST_DIAGNOSTIC, extraTestInfo)

################# Software plugin tests #################

def pageRetirementErrorsPresent(response):
    """Returns `True` when known errors associated with page retirement are present, `False` otherwise."""
    import re
    assert response.numTests == 1
    for error in response.errors[:min(response.numErrors, dcgm_structs.DCGM_DIAG_RESPONSE_ERRORS_MAX)]:
        unwantedErrors = \
            r'retired page' + \
            r'|retired_pages_' + \
            r'|DCGM_FI_DEV_RETIRED_PENDING' + \
            r'|Pending page retirements together with a DBE were detected'
        if re.search(unwantedErrors, error.msg):
            return True
    return False

def expectTestFailures(response):
    """Returns `True` if there is one or more test failure, `False` otherwise."""
    assert response.numTests == 1
    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL:
            return True
    for result in response.results[:min(response.numResults, dcgm_structs.DCGM_DIAG_RESPONSE_RESULTS_MAX)]:
        if result.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL:
            return True
    if response.numErrors != 0:
        raise RuntimeError("No tests failed, however, one or more errors were reported.")
    return False

def helper_check_software_page_retirements_fail_on_pending_retirements(handle, gpuId):
    """
    Ensure that the software test for page retirements fails when there are pending page retirements.
    """
    # First verify that the software test passes for the gpu.
    # If it doesn't pass, skip test and add note to check GPU health
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId])
    dd.UseFakeGpus()
    response = test_utils.diag_execute_wrapper(dd, handle)
    if pageRetirementErrorsPresent(response):
        test_utils.skip_test("Skipping because GPU %s does not pass software page retirement test. "
                             "Please verify whether the GPU is healthy." % gpuId)

    # Inject some pending page retirements
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 1, injection_offset, True)
    response = test_utils.diag_execute_wrapper(dd, handle)
    # Ensure software test failed due to pending page retirments
    assert expectTestFailures(response), "Expected software test to fail due to pending page retirements in the GPU"
    assert pageRetirementErrorsPresent(response), "Expected software test to fail due to pending page retirements in the GPU"

    # Reset injected value
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 0, injection_offset, True)
    # Ensure diag passes now
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert not expectTestFailures(response), "Expected software test to pass"

@test_utils.run_with_standalone_host_engine(120)
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
    if pageRetirementErrorsPresent(response):
        test_utils.skip_test("Skipping because GPU %s does not pass software page retirement test. "
                             "Please verify whether the GPU is healthy." % gpuId)

    # Inject enough page retirements to cause failure
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE, 33, injection_offset, True)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 33, injection_offset, True)
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert pageRetirementErrorsPresent(response), "Expected software test to fail due to 60 total page retirements in the GPU"

    # Ensure 59 pages pass injected value
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 25, injection_offset, True)
    # Ensure diag passes now
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert not expectTestFailures(response), "Expected software test to pass since there are less than 60 total retired pages"

    # Reset retired pages count and verify pass
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE, 0, injection_offset, True)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 0, injection_offset, True)
    # Ensure diag still passes
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert not pageRetirementErrorsPresent(response), \
           "Expected software test to pass since there are no retired pages"

@test_utils.run_with_standalone_host_engine(120)
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
    # Prior code explicitly looked for the Inforom test. Currently, only the results for
    # the software test are available.
    assert response.numTests == 1

    for gpuId in gpuIds:
        foundResults = list(filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and cur.entity.entityId == gpuId and
                              cur.result in [dcgm_structs.DCGM_DIAG_RESULT_PASS, dcgm_structs.DCGM_DIAG_RESULT_SKIP],
                              response.results[:min(response.numResults, dcgm_structs.DCGM_DIAG_RESPONSE_RESULTS_MAX)]))
        assert len(foundResults) > 0

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

@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_current_system_injection_nvml()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_software_on_fabric_manager(handle, gpuIds):
    class Case:
        def __init__(self):
            self.nvmlFunRet = dcgm_nvml.NVML_SUCCESS
            self.fbState = dcgm_nvml.NVML_GPU_FABRIC_STATE_COMPLETED
            self.fbStatus = dcgm_nvml.NVML_SUCCESS
            self.isPass = True
            self.expectedErrorMsg = ""
            self.caseName = ""

        def SetNvmlFunRet(self, ret):
            self.nvmlFunRet = ret

        def SetFbState(self, state):
            self.fbState = state

        def SetFbStatus(self, status):
            self.fbStatus = status

        def SetIsPass(self, isPass):
            self.isPass = isPass

        def SetExpectedErrorMsg(self, msg):
            self.expectedErrorMsg = msg

        def SetCaseName(self, name):
            self.caseName = name

        def GetIsPass(self):
            return self.isPass

        def GetExpectedErrorMsg(self):
            return self.expectedErrorMsg

        def GetCaseName(self):
            return self.caseName

        def GenerateInjectedStructure(self):
            injectedRet = nvml_injection.c_injectNvmlRet_t()
            injectedRet.nvmlRet = self.nvmlFunRet
            injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_GPUFABRICINFOV
            injectedRet.values[0].value.GpuFabricInfoV.state = self.fbState
            injectedRet.values[0].value.GpuFabricInfoV.status = self.fbStatus
            injectedRet.valueCount = 1
            return injectedRet

    caseList = []
    fbFunctionErrorCase = Case()
    fbFunctionErrorCase.SetCaseName("fbFunctionErrorCase")
    fbFunctionErrorCase.SetNvmlFunRet(dcgm_nvml.NVML_ERROR_UNKNOWN)
    fbFunctionErrorCase.SetIsPass(False)
    fbFunctionErrorCase.SetExpectedErrorMsg("Ensure that the FabricManager is running without errors.")
    caseList.append(fbFunctionErrorCase)

    functionNotFoundCase = Case()
    functionNotFoundCase.SetCaseName("functionNotFoundCase")
    functionNotFoundCase.SetNvmlFunRet(dcgm_nvml.NVML_ERROR_FUNCTION_NOT_FOUND)
    functionNotFoundCase.SetIsPass(True)
    caseList.append(functionNotFoundCase)

    gpuNotSupportCase = Case()
    gpuNotSupportCase.SetCaseName("gpuNotSupportCase")
    gpuNotSupportCase.SetNvmlFunRet(dcgm_nvml.NVML_SUCCESS)
    gpuNotSupportCase.SetIsPass(True)
    gpuNotSupportCase.SetFbState(dcgm_nvml.NVML_GPU_FABRIC_STATE_NOT_SUPPORTED)
    caseList.append(gpuNotSupportCase)

    notStartedCase = Case()
    notStartedCase.SetCaseName("notStartedCase")
    notStartedCase.SetNvmlFunRet(dcgm_nvml.NVML_SUCCESS)
    notStartedCase.SetIsPass(False)
    notStartedCase.SetFbState(dcgm_nvml.NVML_GPU_FABRIC_STATE_NOT_STARTED)
    notStartedCase.SetExpectedErrorMsg("training not started")
    caseList.append(notStartedCase)

    inProgressCase = Case()
    inProgressCase.SetCaseName("inProgressCase")
    inProgressCase.SetNvmlFunRet(dcgm_nvml.NVML_SUCCESS)
    inProgressCase.SetIsPass(False)
    inProgressCase.SetFbState(dcgm_nvml.NVML_GPU_FABRIC_STATE_IN_PROGRESS)
    inProgressCase.SetExpectedErrorMsg("training in progress")
    caseList.append(inProgressCase)

    completeButErrorCase = Case()
    completeButErrorCase.SetCaseName("completeButErrorCase")
    completeButErrorCase.SetNvmlFunRet(dcgm_nvml.NVML_SUCCESS)
    completeButErrorCase.SetIsPass(False)
    completeButErrorCase.SetFbState(dcgm_nvml.NVML_GPU_FABRIC_STATE_COMPLETED)
    completeButErrorCase.SetFbStatus(dcgm_nvml.NVML_ERROR_UNKNOWN)
    completeButErrorCase.SetExpectedErrorMsg("Training completed with an error")
    caseList.append(completeButErrorCase)

    completeAndSuccessCase = Case()
    completeAndSuccessCase.SetCaseName("completeAndSuccessCase")
    completeAndSuccessCase.SetNvmlFunRet(dcgm_nvml.NVML_SUCCESS)
    completeAndSuccessCase.SetIsPass(True)
    completeAndSuccessCase.SetFbState(dcgm_nvml.NVML_GPU_FABRIC_STATE_COMPLETED)
    completeAndSuccessCase.SetFbStatus(dcgm_nvml.NVML_SUCCESS)
    caseList.append(completeAndSuccessCase)

    for currentCase in caseList:
        injectedRet = currentCase.GenerateInjectedStructure()
        ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuIds[0], "GpuFabricInfoV", None, 0, injectedRet)
        assert (ret == dcgm_structs.DCGM_ST_OK)

        dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]])
        response = test_utils.diag_execute_wrapper(dd, handle)
        assert response.numTests == 1, f"Case: [{currentCase.GetCaseName()}], actual number of tests: [{response.numTests}]"
        assert response.tests[0].name == "software", f"Case: [{currentCase.GetCaseName()}], actual test name: [{response.tests[0].name}]"
        if currentCase.GetIsPass():
            assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_PASS, f"Case: [{currentCase.GetCaseName()}], actual result: [{response.tests[0].result}]"
            assert response.numErrors == 0, f"Case: [{currentCase.GetCaseName()}], actual number of errors: [{response.numErrors}]"
        else:
            assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, f"Case: [{currentCase.GetCaseName()}], actual result: [{response.tests[0].result}]"
            assert response.numErrors == 1, f"Case: [{currentCase.GetCaseName()}], actual number of errors: [{response.numErrors}]"
            assert response.errors[0].msg.find(currentCase.GetExpectedErrorMsg()) != -1, f"Case: [{currentCase.GetCaseName()}], expected error: [{currentCase.GetExpectedErrorMsg()}], actual error message: [{response.errors[0].msg}]"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus(2)
def test_nvvs_plugin_skip_memtest_if_page_retirement_row_remap_present(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="memtest", paramsStr="memtest.test_duration=10")
    dd.UseFakeGpus()

    # Inject Row remap failure and check for memtest to be skipped.
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ROW_REMAP_FAILURE, 1, 10, True, repeatCount=10)
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.numErrors > 0
    assert response.tests[1].result == dcgm_structs.DCGM_DIAG_RESULT_SKIP, f"Actual result is {response.tests[1].result}"
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ROW_REMAP_FAILURE, 0, 10, True, repeatCount=10) # reset the value

    # Inject pending page retired failure and check for memtest to be skipped.
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 1, 10, True, repeatCount=10)
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.numErrors > 0
    assert response.tests[1].result == dcgm_structs.DCGM_DIAG_RESULT_SKIP, f"Actual result is {response.tests[1].result}"
    inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 0, 10, True, repeatCount=10) # reset the value

    # After Row remap and pending page retired are reset i.e. not present, memtest's shouldn't be skipped.
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.tests[1].result != dcgm_structs.DCGM_DIAG_RESULT_SKIP

@test_utils.run_only_when_path_exists(os.path.abspath("./apps/nvvs/nvvs"))
@test_utils.run_with_persistence_mode_on()
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_nvvs_executes_directly(handle, gpuIds):
    nvvs = AppRunner("./apps/nvvs/nvvs", ["--entity-id", str(gpuIds[0]), "--specifiedtest", "short"])
    nvvs.start(timeout=60)
    retValue = nvvs.wait()
    assert retValue == 0, f"Actual ret: [{retValue}]"

@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_current_system_injection_nvml()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_software_parameters(handle, gpuIds):
    def mock_persistence_mode_off():
        injectedRet = nvml_injection.c_injectNvmlRet_t()
        injectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
        injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_ENABLESTATE
        injectedRet.values[0].value.EnableState = 0
        injectedRet.valueCount = 1
        ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuIds[0], "PersistenceMode", None, 0, injectedRet)
        assert (ret == dcgm_structs.DCGM_ST_OK)

    mock_persistence_mode_off()
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]])
    dd.AddParameter("software.require_persistence_mode=true")
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.numTests == 1, f"actual number of tests: [{response.numTests}]"
    assert response.tests[0].name == "software", f"actual test name: [{response.tests[0].name}]"
    assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL, f"actual result: [{response.tests[0].result}]"
    assert response.numErrors == 1, f"actual number of errors: [{response.numErrors}]"
    assert response.errors[0].msg.find("Persistence Mode") != -1, f"actual error: [{response.errors[0].msg}]"

    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]])
    dd.AddParameter("software.require_persistence_mode=false")
    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.numTests == 1, f"actual number of tests: [{response.numTests}]"
    assert response.tests[0].name == "software", f"actual test name: [{response.tests[0].name}]"
    assert response.tests[0].result == dcgm_structs.DCGM_DIAG_RESULT_PASS, f"actual result: [{response.tests[0].result}]"
    assert response.numErrors == 0, f"actual number of errors: [{response.numErrors}]"
