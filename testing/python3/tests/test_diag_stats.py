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
import dcgm_fields
import dcgm_structs
import dcgm_agent
from dcgm_structs import dcgmExceptionClass
import test_utils
import dcgm_fields
import logger
import os
import option_parser
import DcgmDiag
from DcgmDiag import check_diag_result_fail, check_diag_result_pass
import json
import os.path

TEST_DIAGNOSTIC="diagnostic"
TEST_TARGETED_POWER="targeted_power"
TEST_TARGETED_STRESS="targeted_stress"
TEST_PCIE="pcie"

def load_json_stats_file(filename, logContentsOnError=True):
    with open(filename) as json_file:
        data = json.load(json_file)

        if data == None and logContentsOnError:
            logger.error("Unable to load json file %s. File contents: %s" % (filename, json_file.read()))
        return data

    raise "Couldn't open stats file %s" % filename

def helper_basic_stats_file_check(statsFile, gpuIds, statName):
    try:
        json_data = load_json_stats_file(statsFile)
    finally:
        os.remove(statsFile);
        
    assert json_data != None, "Could not load json data from the stats file"
    foundGpuIds = []
    foundStatName = False
    if not statName:
        foundStatName = True
    
    for stat_names in json_data["GPUS"]:
        for stat in stat_names:
            # Make sure the requested stat name is found
            if foundStatName == False:
                foundStatName = statName == stat

            prev_timestamps = []
            gpuId = stat_names["gpuId"]
            if stat == "gpuId":
                foundGpuIds.append(int(gpuId))
                continue

            for value in stat_names[stat]:
                # Make sure no timestamps are repeated
                if len(prev_timestamps) > 0:
                    assert value['timestamp'] > prev_timestamps[-1], \
                        "GPU %s, field %s has out of order timestamps: %s then %s" % \
                        (gpuId, stat, prev_timestamps[-1], value['timestamp'])
                
                prev_timestamps.append(value['timestamp'])

    assert foundStatName == True, "Expected to find stat '%s', but it was not present in the stats file" % statName

    # Make sure we found each expected GPU id
    for gpuId in gpuIds:
        assert gpuId in foundGpuIds, "Couldn't find GPU %d in the stats file (found %s)" % (gpuId, str(foundGpuIds))


# See test_dcgm_diag.py: helper_verify_log_file_creation for refactoring and multi-maintenance opportunity
def helper_test_stats_file_basics(handle, gpuIds, statsAsString, testName, paramStr, statName=None, watchFrequency=5000000):
    testName = testName.replace(' ', '_')
    #Run on a single GPU since we're just testing the stats file output
    gpuIds = [gpuIds[0], ]

    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=testName, paramsStr=paramStr) # was 20

    dd.SetStatsPath('/tmp/')
    
    dd.SetWatchFrequency(watchFrequency)

    # Make sure a stats file was created
    statsfile = '/tmp/stats_%s.json' % (testName)

    if statsAsString == True:
        dd.SetConfigFileContents("%YAML 1.2\n\nglobals:\n  logfile_type: text\n")

    response = test_utils.diag_execute_wrapper(dd, handle)

    try:
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
            assert test.name == testName, "Expected results for test %s but none were found" % testName
            results = map(lambda resIdx: response.results[resIdx], test.resultIndices[:min(test.numResults, dcgm_structs.DCGM_DIAG_TEST_RUN_RESULTS_MAX)])
            for result in results:
                if result.result not in { dcgm_structs.DCGM_DIAG_RESULT_SKIP, dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN }:
                    skippedAll = False
                    if result.result == dcgm_structs.DCGM_DIAG_RESULT_PASS:
                        passedCount += 1
                    else:
                        # Errors/warnings will be collected separately
                        pass

            errors = map(lambda errIdx: response.errors[errIdx], test.errorIndices[:min(test.numErrors, dcgm_structs.DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)])
            for error in errors:
                if error.entity.entityGroupId == dcgm_fields.DCGM_FE_NONE:
                    errmsg = "%s %s" % (errmsg, error.msg)
                elif error.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU:
                    errmsg = "%s GPU %d failed: %s " % (errmsg, error.entity.entityId, error.msg)
                else:
                    errmsg = "%s Entity (grp=%d, id=%d) failed: %s " % (errmsg, error.entity.entityGroupId, error.entity.entityId, error.msg)

                if error.msg.startswith("API call cudaMalloc failed"):
                    test_utils.skip_test("The %s plugin had a problem when executing: '%s'" % (testName, error.msg))

            gpuCount = sum(1 for gpu in filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU,
                                               response.entities[:min(response.numEntities, dcgm_structs.DCGM_DIAG_RESPONSE_ENTITIES_MAX)]))

            if skippedAll == False:
                detailedMsg = "passed on %d of %d GPUs" % (passedCount, gpuCount)
                if errmsg:
                    detailedMsg = "%s and had these errors: %s" % (detailedMsg, errmsg)
                    logger.info("%s when running the %s plugin" % (detailedMsg, testName))
                    assert os.path.isfile(statsfile), "Statsfile '%s' was not created as expected and %s" % (statsfile, detailedMsg)

                if not statsAsString:
                    helper_basic_stats_file_check(statsfile, gpuIds, statName)
            elif passedCount == 0:
                test_utils.skip_test("Unable to pass any of these short runs for plugin %s." % testName)
            else:
                test_utils.skip_test("The %s plugin was skipped, so we cannot run this test." % testName)
        else:
            test_utils.skip_test("The %s plugin had a problem when executing, so we cannot run this test." % testName)
    finally:
        if os.path.exists(statsfile):
            os.remove(statsfile)

@test_utils.run_only_as_root()
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_standalone_host_engine(20, heArgs=['--service-account', 'dcgm-tests-service-account'])
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_file_present_standalone_with_service_account(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, TEST_DIAGNOSTIC, "diagnostic.test_duration=10", statName='perf_gflops')

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_file_present_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, TEST_DIAGNOSTIC, "diagnostic.test_duration=10", statName='perf_gflops')

@test_utils.run_only_as_root()
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_standalone_host_engine(20, heArgs=['--service-account', 'dcgm-tests-service-account'])
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_string_stats_file_present_standalone_with_service_account(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, True, TEST_DIAGNOSTIC, "diagnostic.test_duration=10")

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_string_stats_file_present_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, True, TEST_DIAGNOSTIC, "diagnostic.test_duration=10")

@test_utils.run_only_as_root()
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_standalone_host_engine(20, heArgs=['--service-account', 'dcgm-tests-service-account'])
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_targeted_power_standalone_with_service_account(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, TEST_TARGETED_POWER, "targeted_power.test_duration=10")

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_targeted_power_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, TEST_TARGETED_POWER, "targeted_power.test_duration=10")

@test_utils.run_only_as_root()
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_standalone_host_engine(20, heArgs=['--service-account', 'dcgm-tests-service-account'])
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_targeted_stress_standalone_with_service_account(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, TEST_TARGETED_STRESS, "targeted_stress.test_duration=10", statName='flops_per_op')

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_targeted_stress_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, TEST_TARGETED_STRESS, "targeted_stress.test_duration=10", statName='flops_per_op')

@test_utils.run_only_as_root()
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_standalone_host_engine(20, heArgs=['--service-account', 'dcgm-tests-service-account'])
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_pcie_standalone_with_service_account(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, TEST_PCIE, "pcie.test_duration=10;pcie.test_with_gemm=true", statName='perf_gflops')

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_pcie_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, TEST_PCIE, "pcie.test_duration=10;pcie.test_with_gemm=true", statName='perf_gflops')

@test_utils.run_only_as_root()
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_standalone_host_engine(20, heArgs=['--service-account', 'dcgm-tests-service-account'])
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_nvbandwidth_standalone_with_service_account(handle, gpuIds):
    # Set the watch frequency for nvbandwidth to 0.5 seconds as it runs too fast on some systems
    helper_test_stats_file_basics(handle, gpuIds, False, 'nvbandwidth', "nvbandwidth.is_allowed=true;nvbandwidth.testcases=0,1,2,3,4,5,6,7,8,9", watchFrequency=500000)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_nvbandwidth_standalone(handle, gpuIds):
    # Set the watch frequency for nvbandwidth to 0.5 seconds as it runs too fast on some systems
    helper_test_stats_file_basics(handle, gpuIds, False, 'nvbandwidth', "nvbandwidth.is_allowed=true;nvbandwidth.testcases=0,1,2,3,4,5,6,7,8,9", watchFrequency=500000)


def helper_test_bad_statspath(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=TEST_DIAGNOSTIC, paramsStr='diagnostic.test_duration=20')
    dd.SetStatsPath('/fake/superfake/notreal/')
    failed = False
    try:
        response = test_utils.diag_execute_wrapper(dd, handle)
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NVVS_ERROR) as e:
        failed = True
        assert str(e).find('cannot access statspath') != -1, "Should have received a statspath error but got %s" % str(e)

    assert failed, "We must fail when attempting to access a fake dir"

    filename = '/tmp/not_a_file'
    if not os.path.isfile(filename):
        # create the file
        with open(filename, 'w') as f:
            f.write('lorem ipsum')

        failed = False
        dd.SetStatsPath(filename)
        try:
            response = test_utils.diag_execute_wrapper(dd, handle)
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NVVS_ERROR) as e:
            failed = True
            assert str(e).find('is not a directory') != -1, "Should have received a statspath error but got %s" % str(e)
        assert failed, "We must fail when attempting to set statspath to a file"

        # Remove the file to clean up after ourselves
        os.remove(filename)

@test_utils.run_only_as_root()
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_standalone_host_engine(20, heArgs=['--service-account', 'dcgm-tests-service-account'])
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_stats_bad_statspath_standalone_with_service_account(handle, gpuIds):
    helper_test_bad_statspath(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_stats_bad_statspath_standalone(handle, gpuIds):
    helper_test_bad_statspath(handle, gpuIds)
