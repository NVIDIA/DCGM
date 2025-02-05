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
import dcgm_agent
from dcgm_structs import DCGM_ST_NOT_SUPPORTED, dcgmExceptionClass
import test_utils
import logger
import os
import option_parser
import time
import dcgm_fields
import dcgm_structs_internal
import dcgm_agent_internal
import DcgmReader
import random
import dcgm_field_helpers
import apps

g_profNotSupportedErrorStr = "Continuous mode profiling is not supported for this GPU group. Either libnvperf_dcgm_host.so isn't in your LD_LIBRARY_PATH or it is not the NDA version that supports DC profiling"
g_moduleNotLoadedErrorStr = "Continuous mode profiling is not supported for this system because the profiling module could not be loaded. This is likely due to libnvperf_dcgm_host.so not being in LD_LIBRARY_PATH"
g_profGroupIsEmpty = "No GPUs suitable for testing."
g_noMigSlicesErrorStr = "GPU(s) is(are) in MIG mode, but no MIG CI partitions are defined."

DLG_MAX_METRIC_GROUPS = 5 #This is taken from modules/profiling/DcgmLopConfig.h. These values need to be in sync for multipass tests to pass


def helper_check_profiling_environment(dcgmGroup):
    try:
        dcgmGroup.profiling.GetSupportedMetricGroups()
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GROUP_IS_EMPTY) as e:
        test_utils.skip_test(g_profGroupIsEmpty)
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_PROFILING_NOT_SUPPORTED) as e:
        test_utils.skip_test(g_profNotSupportedErrorStr)
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED) as e:
        test_utils.skip_test(g_moduleNotLoadedErrorStr)
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED) as e:
        test_utils.skip_test(g_profNotSupportedErrorStr)

def helper_get_supported_field_ids(dcgmGroup):
    '''
    Get a list of the supported fieldIds for the provided DcgmGroup object.

    It's important to query this dynamically, as field IDs can vary from chip to chip
    and driver version to driver version
    '''
    fieldIds = []

    metricGroups = dcgmGroup.profiling.GetSupportedMetricGroups()
    for i in range(metricGroups.numMetricGroups):
        for j in range(metricGroups.metricGroups[i].numFieldIds):
            fieldIds.append(metricGroups.metricGroups[i].fieldIds[j])

    return fieldIds

def helper_get_multipass_field_ids(dcgmGroup):
    '''
    Get a list of the supported fieldIds for the provided DcgmGroup object that 
    require multiple passes in the hardware

    Returns None if no such combination exists. Otherwise a list of lists
    where the first dimension is groups of fields that are exclusive with each other.
    the second dimension are the fieldIds within an exclusive group.
    '''
    exclusiveFields = {} #Key by major ID

    #First, look for two metric groups that have the same major version but different minor version
    #That is the sign of being multi-pass
    metricGroups = dcgmGroup.profiling.GetSupportedMetricGroups()
    for i in range(metricGroups.numMetricGroups):

        majorId = metricGroups.metricGroups[i].majorId
        if majorId not in exclusiveFields:
            exclusiveFields[majorId] = []

        fieldIds = metricGroups.metricGroups[i].fieldIds[0:metricGroups.metricGroups[i].numFieldIds]
        exclusiveFields[majorId].append(fieldIds)

    #See if any groups have > 1 element. Volta and turing only have one multi-pass group, so we
    #can just return one if we find it
    for group in list(exclusiveFields.values()):
        if len(group) > 1:
            return group

    return None

def helper_get_single_pass_field_ids(dcgmGroup):
    '''
    Get a list of the supported fieldIds for the provided DcgmGroup object that can
    be watched at the same time

    Returns None if no field IDs are supported
    '''
    fieldIds = []

    #Try to return the largest single-pass group
    largestMetricGroupIndex = None
    largestMetricGroupCount = 0

    metricGroups = dcgmGroup.profiling.GetSupportedMetricGroups()
    for i in range(metricGroups.numMetricGroups):
        if metricGroups.metricGroups[i].numFieldIds > largestMetricGroupCount:
            largestMetricGroupIndex = i
            largestMetricGroupCount = metricGroups.metricGroups[i].numFieldIds

    if largestMetricGroupIndex is None:
        return None

    for i in range(metricGroups.metricGroups[largestMetricGroupIndex].numFieldIds):
        fieldIds.append(metricGroups.metricGroups[largestMetricGroupIndex].fieldIds[i])

    return fieldIds

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_prof_get_supported_metric_groups_sanity(handle, gpuIds):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_prof_watch_fields_sanity(handle, gpuIds):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    fieldIds = helper_get_single_pass_field_ids(dcgmGroup)
    assert fieldIds is not None

    logger.info("Single pass field IDs: " + str(fieldIds))

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", fieldIds)

    dcgmGroup.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)

    #Throws an exception on error
    dcgmGroup.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)
    
    #Cleanup
    dcgmGroup.samples.UnwatchFields(fieldGroup)
    dcgmGroup.Delete()

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.run_only_as_root()
@test_utils.run_clearing_gpus() # Remove real GPUs.
@test_utils.run_with_injection_gpus(gpuCount=2)  # Injecting fake GPUs to simulate not supported SKUs
@test_utils.run_for_each_gpu_individually()
def test_dcgm_prof_all_supported_fields_watchable(handle, gpuId):
    '''
    Verify that all fields that are reported as supported are watchable and 
    that values can be returned for them
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', [gpuId])

    helper_check_profiling_environment(dcgmGroup)

    fieldIds = helper_get_supported_field_ids(dcgmGroup)
    assert fieldIds is not None

    watchFreq = 1000 #1 ms
    maxKeepAge = 60.0
    maxKeepSamples = 0
    maxAgeUsec = int(maxKeepAge) * watchFreq

    entityPairList = [dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId)]

    for fieldId in fieldIds:
        fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group_%d" % fieldId, [fieldId, ])

        # If there are only one unsupported SKUs in the group, WatchFields should return an error.
        # If at least one GPU in the group is supported, WatchFields will be successful.
        # The described logic is used to skip unsupported or fake SKUs.
        try:
            dcgmGroup.samples.WatchFields(fieldGroup, watchFreq, maxKeepAge, maxKeepSamples)
        except:
            fieldGroup.Delete()
            test_utils.skip_test_supported("DCP")

        # Sending a request to the profiling manager guarantees that an update cycle has happened since 
        # the last request
        dcgmGroup.profiling.GetSupportedMetricGroups()

        # validate watch freq, quota, and watched flags
        cmfi = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, dcgm_fields.DCGM_FE_GPU, fieldId)
        assert (cmfi.flags & dcgm_structs_internal.DCGM_CMI_F_WATCHED) != 0, "gpuId %u, fieldId %u not watched" % (gpuId, fieldId)
        assert cmfi.numSamples > 0
        assert cmfi.numWatchers == 1, "numWatchers %d" % cmfi.numWatchers
        assert cmfi.monitorIntervalUsec == watchFreq, "monitorIntervalUsec %u != watchFreq %u" % (cmfi.monitorIntervalUsec, watchFreq)
        assert cmfi.lastStatus == dcgm_structs.DCGM_ST_OK, "lastStatus %u != DCGM_ST_OK" % (cmfi.lastStatus)

        fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entityPairList, [fieldId, ], 0)

        for i, fieldValue in enumerate(fieldValues):
            logger.debug(str(fieldValue))
            assert(fieldValue.status == dcgm_structs.DCGM_ST_OK), "idx %d status was %d" % (i, fieldValue.status)
            assert(fieldValue.ts != 0), "idx %d timestamp was 0" % (i)

        dcgmGroup.samples.UnwatchFields(fieldGroup)
        fieldGroup.Delete()

        #Validate watch flags after unwatch
        cmfi = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, dcgm_fields.DCGM_FE_GPU, fieldId)
        assert (cmfi.flags & dcgm_structs_internal.DCGM_CMI_F_WATCHED) == 0, "gpuId %u, fieldId %u still watched. flags x%X" % (gpuId, fieldId, cmfi.flags)
        assert cmfi.numWatchers == 0, "numWatchers %d" % cmfi.numWatchers

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_prof_watch_multipass(handle, gpuIds):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    mpFieldIds = helper_get_multipass_field_ids(dcgmGroup)
    if mpFieldIds is None:
        test_utils.skip_test("No multipass profiling fields exist for the gpu group")

    logger.info("Multipass fieldIds: " + str(mpFieldIds))

    #Make sure that multipass watching up to DLG_MAX_METRIC_GROUPS groups works
    for i in range(min(len(mpFieldIds), DLG_MAX_METRIC_GROUPS)):
        fieldIds = []
        for j in range(i+1):
            fieldIds.extend(mpFieldIds[j])

        logger.info("Positive testing multipass fieldIds %s" % str(fieldIds))

        fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group_%d" % i, fieldIds)

        dcgmGroup.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)
        dcgmGroup.samples.UnwatchFields(fieldGroup)
        fieldGroup.Delete()

    if len(mpFieldIds) <= DLG_MAX_METRIC_GROUPS:
        test_utils.skip_test("Skipping multipass failure test since there are %d <= %d multipass groups." %
                             (len(mpFieldIds), DLG_MAX_METRIC_GROUPS))

    for i in range(DLG_MAX_METRIC_GROUPS+1, len(mpFieldIds)+1):
        fieldIds = []
        for j in range(i):
            fieldIds.extend(mpFieldIds[j])

        logger.info("Negative testing multipass fieldIds %s" % str(fieldIds))

        fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group_%d" % i, fieldIds)

        with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_PROFILING_MULTI_PASS)):    
            dcgmGroup.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)
            dcgmGroup.samples.UnwatchFields(fieldGroup)

        fieldGroup.Delete()

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_prof_watch_fields_multi_user(handle, gpuIds):
    dcgmHandle = pydcgm.DcgmHandle(ipAddress="127.0.0.1")
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    dcgmHandle2 = pydcgm.DcgmHandle(ipAddress="127.0.0.1")
    dcgmSystem2 = dcgmHandle2.GetSystem()
    dcgmGroup2 = dcgmSystem2.GetGroupWithGpuIds('mygroup2', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    fieldIds = helper_get_single_pass_field_ids(dcgmGroup)
    assert fieldIds is not None

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group_0", fieldIds)
    fieldGroup2 = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group_2", fieldIds)

    # Take ownership of the profiling watches
    dcgmGroup.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)

    dcgmGroup2.samples.WatchFields(fieldGroup2, 1000000, 3600.0, 0)

    # Release the watches
    dcgmGroup2.samples.UnwatchFields(fieldGroup2)
    dcgmGroup.samples.UnwatchFields(fieldGroup)

    # Now dcgmHandle2 owns the watches
    dcgmGroup2.samples.WatchFields(fieldGroup2, 1000000, 3600.0, 0)

    # connection 1 should not fail to acquire the watches
    dcgmGroup.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)

    dcgmGroup2.samples.UnwatchFields(fieldGroup2)
    dcgmGroup.samples.UnwatchFields(fieldGroup)

    fieldGroup.Delete()
    fieldGroup2.Delete()

    dcgmHandle.Shutdown()
    dcgmHandle2.Shutdown()


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_prof_with_dcgmreader(handle, gpuIds):
    """ 
    Verifies that we can access profiling data with DcgmReader, which is the 
    base class for dcgm exporters
    """
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()

    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    fieldIds = helper_get_single_pass_field_ids(dcgmGroup)

    updateFrequencyUsec = 200000 # 200ms
    sleepTime = updateFrequencyUsec / 1000000 * 2 # Convert to seconds and sleep twice as long; ensures fresh sample

    dr = DcgmReader.DcgmReader(fieldIds=fieldIds, updateFrequency=updateFrequencyUsec, maxKeepAge=30.0, gpuIds=gpuIds)
    dr.SetHandle(handle)

    for i in range(5):
        time.sleep(sleepTime)

        latest = dr.GetLatestGpuValuesAsFieldIdDict()
        logger.info(str(latest))

        for gpuId in gpuIds:
            if len(latest[gpuId]) != len(fieldIds):
                missingFieldIds = []
                extraFieldIds = []
                for fieldId in fieldIds:
                    if fieldId not in latest[gpuId]:
                        missingFieldIds.append(fieldId)

                for fieldId in latest[gpuId]:
                    if fieldId not in fieldIds:
                        extraFieldIds.append(fieldId)

                errmsg = "i=%d, gpuId %d, len %d != %d" % (i, gpuId, len(latest[gpuId]), len(fieldIds))
                if len(missingFieldIds) > 0:
                    errmsg = errmsg + " GPU is missing entries for fields %s" % str(missingFieldIds)
                if len(extraFieldIds) > 0:
                    errmsg = errmsg + " GPU has extra entries for fields %s" % str(extraFieldIds)

                assert len(latest[gpuId]) == len(fieldIds), errmsg


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_prof_initial_valid_record(handle, gpuIds):
    '''
    Test that we can retrieve a valid FV for a profiling field immediately after watching
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    fieldIds = helper_get_single_pass_field_ids(dcgmGroup)
    assert fieldIds is not None

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group_0", fieldIds)

    #Set watches using a large interval so we don't get a record for 10 seconds in the bug case
    dcgmGroup.samples.WatchFields(fieldGroup, 10000000, 3600.0, 0)

    gpuId = gpuIds[0]

    fieldValues = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, gpuId, fieldIds)
    assert len(fieldValues) == len(fieldIds), "%d != %d" % (len(fieldValues), len(fieldIds))

    for i, fieldValue in enumerate(fieldValues):
        logger.info(str(fieldValue))
        assert(fieldValue.version != 0), "idx %d Version was 0" % i
        assert(fieldValue.fieldId == fieldIds[i]), "idx %d fieldValue.fieldId %d != fieldIds[i] %d" % (i, fieldValue.fieldId, fieldIds[i])
        assert(fieldValue.status == dcgm_structs.DCGM_ST_OK), "idx %d status was %d" % (i, fieldValue.status)
        #The following line catches the bug in Jira DCGM-1357. Previously, a record would be returned with a
        #0 timestamp
        assert(fieldValue.ts != 0), "idx %d timestamp was 0" % i

    #Cleanup
    dcgmGroup.samples.UnwatchFields(fieldGroup)
    fieldGroup.Delete()

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_prof_multi_pause_resume(handle, gpuIds):
    '''
    Test that we can pause and resume profiling over and over without error
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)    

    #GPM-enabled GPUs don't support pause/resume
    if test_utils.gpu_supports_gpm(handle, gpuIds[0]):
        with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED)):
             dcgmSystem.profiling.Pause()
        with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED)):
             dcgmSystem.profiling.Resume()
        return

    helper_check_profiling_environment(dcgmGroup)

    #We should never get an error back from pause or resume. Pause and Resume throw exceptions on error
    numPauses = 0
    numResumes = 0

    for i in range(100):
        #Flip a coin and pause if we get 0. unpause otherwise (1)
        coin = random.randint(0,1)
        if coin == 0:
            dcgmSystem.profiling.Pause()
            numPauses += 1
        else:
            dcgmSystem.profiling.Resume()
            numResumes += 1

    logger.info("Got %d pauses and %d resumes" % (numPauses, numResumes))

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_prof_pause_resume_values(handle, gpuIds):
    '''
    Test that we get valid values when profiling is resumed and BLANK values when profiling is paused
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    #GPM-enabled GPUs don't support pause/resume
    if test_utils.gpu_supports_gpm(handle, gpuIds[0]):
        with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED)):
             dcgmSystem.profiling.Pause()
        with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED)):
             dcgmSystem.profiling.Resume()
        return

    helper_check_profiling_environment(dcgmGroup)

    fieldIds = helper_get_single_pass_field_ids(dcgmGroup)
    assert fieldIds is not None

    #10 ms watches so we can test quickly
    watchIntervalUsec = 10000
    sleepIntervalSec = 0.1 * len(gpuIds) #100 ms per GPU
    #Start paused. All the other tests start unpaused
    dcgmSystem.profiling.Pause()

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group_0", fieldIds)
    dcgmGroup.samples.WatchFields(fieldGroup, watchIntervalUsec, 60.0, 0)

    gpuId = gpuIds[0]

    fieldValues = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, gpuId, fieldIds)
    assert len(fieldValues) == len(fieldIds), "%d != %d" % (len(fieldValues), len(fieldIds))

    #All should be blank
    for i, fieldValue in enumerate(fieldValues):
        fv = dcgm_field_helpers.DcgmFieldValue(fieldValue)
        assert fv.isBlank, "Got nonblank fv index %d" % i

    #Resume. All should be valid
    dcgmSystem.profiling.Resume()

    time.sleep(sleepIntervalSec)

    fieldValues = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, gpuId, fieldIds)
    assert len(fieldValues) == len(fieldIds), "%d != %d" % (len(fieldValues), len(fieldIds))

    #All should be non-blank
    for i, fieldValue in enumerate(fieldValues):
        fv = dcgm_field_helpers.DcgmFieldValue(fieldValue)
        assert not fv.isBlank, "Got blank fv index %d" % i

    #Pause again. All should be blank
    dcgmSystem.profiling.Pause()

    time.sleep(sleepIntervalSec)

    fieldValues = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, gpuId, fieldIds)
    assert len(fieldValues) == len(fieldIds), "%d != %d" % (len(fieldValues), len(fieldIds))

    #All should be blank
    for i, fieldValue in enumerate(fieldValues):
        fv = dcgm_field_helpers.DcgmFieldValue(fieldValue)
        assert fv.isBlank, "Got nonblank fv index %d" % i

    #This shouldn't fail
    dcgmSystem.profiling.Resume()

    dcgmGroup.samples.UnwatchFields(fieldGroup)
    fieldGroup.Delete()

def helper_test_dpt_sync_count(handle, gpuIds, fieldIdsStr, extraArgs = None):
    '''
    Test that dcgmproftester passes for non-validation run.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    cudaDriverVersion = test_utils.get_cuda_driver_version(handle, gpuIds[0])

    supportedFieldIds = helper_get_supported_field_ids(dcgmGroup)

    # Find the first testable GPU of our SKU. Other tests will cover multiple
    # GPUs.
    
    slices = 0;

    for gpuId in gpuIds:
        slices = test_utils.get_gpu_slices(handle, gpuId)

        if slices == 0:
            continue

    if slices == 0:
        test_utils.skip_test(g_noMigSlicesErrorStr)

    useGpuIds = [ gpuId ]
        
    duration = 5.0 * slices
    rate = .25 * slices

    args = ["-d", str(duration), "-r", str(rate), "-t", fieldIdsStr]

    # MIG requires slightly looser tolerances than 10%
    if ((fieldIdsStr == str(dcgm_fields.DCGM_FI_PROF_SM_ACTIVE)) or (fieldIdsStr == str(dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY))) and (slices > 1):
        args.extend(["--percent-tolerance", "15"])

    if extraArgs is not None:
        args.extend(extraArgs)

    app = apps.DcgmProfTesterApp(cudaDriverMajorVersion=cudaDriverVersion[0], gpuIds=useGpuIds, args=args)
    app.start(timeout=120.0 * len(gpuIds)) #Account for slow systems but still add an upper bound
    app.wait()

def helper_test_dpt_field_ids(handle, gpuIds, fieldIdsStr, fast = False, extraArgs = None):
    '''
    Test that dcgmproftester passes for validation run.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    cudaDriverVersion = test_utils.get_cuda_driver_version(handle, gpuIds[0])

    supportedFieldIds = helper_get_supported_field_ids(dcgmGroup)

    # Find the first testable GPU of our SKU. Other tests will cover multiple
    # GPUs.
    
    slices = 0;

    for gpuId in gpuIds:
        slices = test_utils.get_gpu_slices(handle, gpuId)

        if slices == 0:
            continue

    if slices == 0:
        test_utils.skip_test(g_noMigSlicesErrorStr)

    if fast:
        duration = 1.0 * slices
    else:
        duration = 5.0 * slices

    rate = 0.25 * slices
    useGpuIds = [ gpuId ]

    args = ["--target-max-value", "--no-dcgm-validation", "--dvs", "--reset", "--mode", "validate", "-d", str(duration), "-r", str(rate), "--sync-count", "5", "-w", "5", "-t", fieldIdsStr]

    # MIG requires slightly looser tolerances than 10%
    if ((fieldIdsStr == str(dcgm_fields.DCGM_FI_PROF_SM_ACTIVE)) or (fieldIdsStr == str(dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY))) and (slices > 1):
        args.extend(["--percent-tolerance", "15"])

    if extraArgs is not None:
        args.extend(extraArgs)

    app = apps.DcgmProfTesterApp(cudaDriverMajorVersion=cudaDriverVersion[0], gpuIds=useGpuIds, args=args)
    app.start(timeout=120.0 * len(gpuIds)) #Account for slow systems but still add an upper bound
    app.wait()

def helper_test_dpt_field_id(handle, gpuIds, fieldId, fast = False, extraArgs = None):
    '''
    Test that dcgmproftester passes.
    '''
    helper_test_dpt_field_ids(handle, gpuIds, str(fieldId), extraArgs)

def helper_test_dpt_field_fast_id(handle, gpuIds, fieldId, fast = False, extraArgs = None):
    '''
    Test that dcgmproftester passes in fast mode.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    cudaDriverVersion = test_utils.get_cuda_driver_version(handle, gpuIds[0])

    supportedFieldIds = helper_get_supported_field_ids(dcgmGroup)

    # Find the first testable GPU of our SKU. Other tests will cover multiple
    # GPUs.
    
    slices = 0;

    for gpuId in gpuIds:
        slices = test_utils.get_gpu_slices(handle, gpuId)

        if slices == 0:
            continue

    if slices == 0:
        test_utils.skip_test(g_noMigSlicesErrorStr)

    if fast:
        duration = 1.0 * slices
        rate = 0.25 * slices
    else:
        duration = 15.0 * slices
        rate = 1.0 * slices

    useGpuIds = [ gpuId ]

    args = ["--target-max-value", "--no-dcgm-validation", "--dvs", "--reset", "--mode", "validate,fast", "-d", str(duration), "-r", str(rate), "--sync-count", "5", "-w", "5", "-t", str(fieldId)]

    # MIG requires slightly looser tolerances than 10%
    if ((fieldId == dcgm_fields.DCGM_FI_PROF_SM_ACTIVE) or (fieldId == dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY)) and (slices > 1):
        args.extend(["--percent-tolerance", "15"])

    if extraArgs is not None:
        args.extend(extraArgs)

    app = apps.DcgmProfTesterApp(cudaDriverMajorVersion=cudaDriverVersion[0], gpuIds=useGpuIds, args=args)
    app.start(timeout=120.0 * len(gpuIds)) #Account for slow systems but still add an upper bound
    app.wait()

def helper_test_dpt_h(handle, gpuIds):
    '''
    Test that -h command line argument works.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    cudaDriverVersion = test_utils.get_cuda_driver_version(handle, gpuIds[0])

    supportedFieldIds = helper_get_supported_field_ids(dcgmGroup)

    #Just test the first GPU of our SKU. Other tests will cover multiple SKUs
    useGpuIds = [gpuIds[0], ]

    args = ["-h"]
    app = apps.DcgmProfTesterApp(cudaDriverMajorVersion=cudaDriverVersion[0], gpuIds=useGpuIds, args=args)
    app.start(timeout=120.0 * len(gpuIds)) #Account for slow systems but still add an upper bound
    app.wait()

def helper_test_dpt_help(handle, gpuIds):
    '''
    Test that command line --help argument works.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    cudaDriverVersion = test_utils.get_cuda_driver_version(handle, gpuIds[0])

    supportedFieldIds = helper_get_supported_field_ids(dcgmGroup)

    #Just test the first GPU of our SKU. Other tests will cover multiple SKUs
    useGpuIds = [gpuIds[0], ]

    args = ["--help"]
    app = apps.DcgmProfTesterApp(cudaDriverMajorVersion=cudaDriverVersion[0], gpuIds=useGpuIds, args=args)
    app.start(timeout=120.0 * len(gpuIds)) #Account for slow systems but still add an upper bound
    app.wait()

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_non_validation(handle, gpuIds):
    helper_test_dpt_sync_count(handle, gpuIds, str(dcgm_fields.DCGM_FI_PROF_GR_ENGINE_ACTIVE))

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_gr_active(handle, gpuIds):
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_GR_ENGINE_ACTIVE)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_h(handle, gpuIds):
    helper_test_dpt_h(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_help(handle, gpuIds):
    helper_test_dpt_help(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_sm_active(handle, gpuIds):
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_SM_ACTIVE)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_sm_occupancy(handle, gpuIds):
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.exclude_non_compute_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.filter_sku("2329 2328 26B7 26B8 26BA 27B6 2322 1FB2 1FF2") # poor or no Tensor math
@test_utils.run_only_as_root()
def test_dcgmproftester_tensor_active(handle, gpuIds):
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, True)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.filter_sku("2322 2324 20F5 20F3")
@test_utils.run_only_if_gpus_available()
@test_utils.run_only_as_root()
def test_dcgmproftester_fp64_active(handle, gpuIds):
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE, True)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_fp32_active(handle, gpuIds):
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE, True)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_fp32_active_cublas(handle, gpuIds):
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE, True, ["--cublas"])

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_fp16_active(handle, gpuIds):
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE, True)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_pcie_rx(handle, gpuIds):
    helper_test_dpt_field_fast_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES, True, ["--percent-tolerance", "20.0"])

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_pcie_tx(handle, gpuIds):
    helper_test_dpt_field_fast_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES, True)

def dont_test_slower_gpus(handle, gpuIds):
    # These GPU ids don't need to be tested
    lower_bandwidth_ids = [ 0x20f5, 0x20f6 ]
    for gpuId in gpuIds:
        deviceId = test_utils.get_device_id(handle, gpuId)
        if deviceId in lower_bandwidth_ids:
            test_utils.skip_test("Skipping the nvlink bandwidth tests for device id: '%s'" % deviceId)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmproftester_nvlink_rx(handle, gpuIds):
    dont_test_slower_gpus(handle, gpuIds)
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES, True)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmproftester_nvlink_tx(handle, gpuIds):
    dont_test_slower_gpus(handle, gpuIds)
    helper_test_dpt_field_id(handle, gpuIds, dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES, True)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmproftester_nvlink_and_other(handle, gpuIds):
    '''
    This added to verify the fix for
    https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3903747
    '''
    dont_test_slower_gpus(handle, gpuIds)
    helper_test_dpt_field_ids(handle, gpuIds, str(dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE) + "," + str(dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES), True)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgmproftester_parallel_gpus(handle, gpuIds):
    '''
    Test that we can successfully read dcgmproftester metrics multiple concurrent GPUs

    This tests a few things:
    1. That metrics work for more than GPU 0
    2. That metrics work for multiple GPUs at a time
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("Skipping multi-GPU test since there's only one of this SKU")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    helper_check_profiling_environment(dcgmGroup)

    cudaDriverVersion = test_utils.get_cuda_driver_version(handle, gpuIds[0])

    #FP16 works for every GPU that supports DCP. It also works reliably even under heavy concurrecy
    fieldIds = "1008"

    args = ["--mode", "validate", "--sync-count", "5", "-w", "10", "-t", fieldIds]

    foundGpu = False;
    
    for gpuId in gpuIds:
        slices = test_utils.get_gpu_slices(handle, gpuId)

        if slices == 0:
            continue

        duration = 1.0 * slices
        rate = .25 * slices

        foundGpu = True;

        args.extend(["-d", str(duration), "-r", str(rate)])

        # MIG requires slightly looser tolerances than 10%
        if ((fieldIds == str(dcgm_fields.DCGM_FI_PROF_SM_ACTIVE)) or (fieldIds == str(dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY))) and (slices > 1):
            args.extend(["--percent-tolerance", "15"])

        args.extend(["-i", str(gpuId)])

    if not foundGpu:
        test_utils.skip_test(g_noMigSlicesErrorStr)

    app = apps.DcgmProfTesterApp(cudaDriverMajorVersion=cudaDriverVersion[0], args=args)
    app.start(timeout=120.0 * len(gpuIds)) #Account for slow systems but still add an upper bound
    app.wait()
    app.validate() #Validate here so that errors are printed when they occur instead of at the end of the test


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_gpus_available()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_prof_global_pause_resume_values(handle, gpuIds):
    """
    Test that we get valid values when DCGM is resumed and BLANK values when DCGM is paused
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    # GPM-enabled GPUs would get DCP metrics from the NVML instead of the paused profiling module and will return
    # valid values when DCGM is paused until we implement full driver detach/reattach on pause/resume.
    if test_utils.gpu_supports_gpm(handle, gpuIds[0]):
        test_utils.skip_test("Skipping test for GPM-enabled GPUs")

    helper_check_profiling_environment(dcgmGroup)

    fieldIds = helper_get_single_pass_field_ids(dcgmGroup)
    assert fieldIds is not None

    # 10 ms watches so we can test quickly
    watchIntervalUsec = 10000
    sleepIntervalSec = 0.1 * len(gpuIds)  # 100 ms per GPU
    # Start paused. All the other tests start unpaused
    dcgmSystem.PauseTelemetryForDiag()

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group_0", fieldIds)
    dcgmGroup.samples.WatchFields(fieldGroup, watchIntervalUsec, 60.0, 0)

    gpuId = gpuIds[0]

    fieldValues = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, gpuId, fieldIds)
    assert len(fieldValues) == len(fieldIds), "%d != %d" % (len(fieldValues), len(fieldIds))

    # All should be blank
    for i, fieldValue in enumerate(fieldValues):
        fv = dcgm_field_helpers.DcgmFieldValue(fieldValue)
        assert fv.isBlank, "Got nonblank fv index %d" % i

    # Resume. All should be valid
    dcgmSystem.ResumeTelemetryForDiag()

    time.sleep(sleepIntervalSec)

    fieldValues = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, gpuId, fieldIds)
    assert len(fieldValues) == len(fieldIds), "%d != %d" % (len(fieldValues), len(fieldIds))

    # All should be non-blank
    for i, fieldValue in enumerate(fieldValues):
        fv = dcgm_field_helpers.DcgmFieldValue(fieldValue)
        assert not fv.isBlank, "Got blank fv index %d" % i

    # Pause again. All should be blank
    dcgmSystem.PauseTelemetryForDiag()

    time.sleep(sleepIntervalSec)

    fieldValues = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, gpuId, fieldIds)
    assert len(fieldValues) == len(fieldIds), "%d != %d" % (len(fieldValues), len(fieldIds))

    # All should be blank
    for i, fieldValue in enumerate(fieldValues):
        fv = dcgm_field_helpers.DcgmFieldValue(fieldValue)
        assert fv.isBlank, "Got nonblank fv index %d" % i

    # This shouldn't fail
    dcgmSystem.ResumeTelemetryForDiag()

    dcgmGroup.samples.UnwatchFields(fieldGroup)
    fieldGroup.Delete()
