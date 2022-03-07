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
import dcgm_agent
from dcgm_structs import dcgmExceptionClass
import test_utils
import logger
import os
import option_parser
import DcgmDiag
import json
import os.path

def load_json_stats_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data

    raise "Couldn't open stats file %s" % filename

def helper_basic_stats_file_check(statsFile, gpuIds, statName):
    json_data = load_json_stats_file(statsFile)
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

def helper_test_stats_file_basics(handle, gpuIds, statsAsString, pluginName, pluginIndex, statName=None):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=pluginName, paramsStr='%s.test_duration=20' % pluginName)

    dd.SetStatsPath('/tmp/')

    # Make sure a stats file was created
    statsfile = '/tmp/stats_%s.json' % (pluginName.replace(' ', '_'))

    if statsAsString == True:
        dd.SetConfigFileContents("%YAML 1.2\n\nglobals:\n  logfile_type: text\n")
    
    response = test_utils.diag_execute_wrapper(dd, handle)

    skippedAll = True

    if len(response.systemError.msg) == 0:
        passedCount = 0
        errors = ""
        for gpuIndex in range(response.gpuCount):
            resultType = response.perGpuResponses[gpuIndex].results[pluginIndex].result
            if resultType != dcgm_structs.DCGM_DIAG_RESULT_SKIP \
                             and resultType != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
                skippedAll = False
                if resultType == dcgm_structs.DCGM_DIAG_RESULT_PASS:
                    passedCount = passedCount + 1
                else:
                    warning = response.perGpuResponses[gpuIndex].results[pluginIndex].error.msg
                    if len(warning):
                        errors = "%s GPU %d failed: %s" % (errors, gpuIndex, warning)

        if skippedAll == False and passedCount > 0:
            detailedMsg = "passed on %d of %d GPUs" % (passedCount, response.gpuCount)
            if len(errors):
                detailedMsg = "%s and had these errors: %s" % (detailedMsg, errors)
                logger.info("%s when running the %s plugin" % (detailedMsg, pluginName))

            assert os.path.isfile(statsfile), "Statsfile '%s' was not created as expected and %s" % (statsfile, detailedMsg)

            if not statsAsString:
                helper_basic_stats_file_check(statsfile, gpuIds, statName)
        elif passedCount == 0:
            test_utils.skip_test("Unable to pass any of these short runs for plugin %s." % pluginName)
        else:
            test_utils.skip_test("The %s plugin was skipped, so we cannot run this test." % pluginName)
    else:
        test_utils.skip_test("The %s plugin had a problem when executing, so we cannot run this test." % pluginName)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_file_present_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, 'diagnostic', dcgm_structs.DCGM_DIAGNOSTIC_INDEX, statName='perf_gflops')

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_file_present_embedded(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, 'diagnostic', dcgm_structs.DCGM_DIAGNOSTIC_INDEX, statName='perf_gflops')

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_string_stats_file_present_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, True, 'diagnostic', dcgm_structs.DCGM_DIAGNOSTIC_INDEX)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_string_stats_file_present_embedded(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, True, 'diagnostic', dcgm_structs.DCGM_DIAGNOSTIC_INDEX)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_targeted_power_embedded(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, 'targeted power', dcgm_structs.DCGM_TARGETED_POWER_INDEX)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_targeted_power_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, 'targeted power', dcgm_structs.DCGM_TARGETED_POWER_INDEX)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_targeted_stress_embedded(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, 'targeted stress', dcgm_structs.DCGM_TARGETED_STRESS_INDEX, statName='flops_per_op')

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_targeted_stress_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, 'targeted stress', dcgm_structs.DCGM_TARGETED_STRESS_INDEX, statName='flops_per_op')

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_sm_stress_embedded(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, 'sm stress', dcgm_structs.DCGM_SM_STRESS_INDEX, statName='perf_gflops')

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_stats_basics_sm_stress_standalone(handle, gpuIds):
    helper_test_stats_file_basics(handle, gpuIds, False, 'sm stress', dcgm_structs.DCGM_SM_STRESS_INDEX, statName='perf_gflops')

def helper_test_bad_statspath(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr='diagnostic', paramsStr='diagnostic.test_duration=20')
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

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_stats_bad_statspath_embedded(handle, gpuIds):
    helper_test_bad_statspath(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_stats_bad_statspath_standalone(handle, gpuIds):
    helper_test_bad_statspath(handle, gpuIds)
