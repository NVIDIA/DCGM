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
# test the metadata API calls for DCGM
import time
from sys import float_info

import dcgm_structs
import dcgm_agent_internal
import dcgm_field_helpers
import dcgm_fields
import pydcgm
import logger
import test_utils
import stats
import option_parser

def _watch_field_group_basic(fieldGroup, handle, groupId, updateFreq=1000):
    return dcgm_field_helpers.DcgmFieldGroupWatcher(handle,
                                                    groupId,
                                                    fieldGroup=fieldGroup,
                                                    operationMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO,
                                                    updateFreq=updateFreq,
                                                    maxKeepAge=86400.0,
                                                    maxKeepSamples=1000,
                                                    startTimestamp=0)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_memory_get_field_sane(handle):
    '''
    Sanity test for API that gets memory usage of a single field
    '''
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skipping developer test.")
    handleObj = pydcgm.DcgmHandle(handle=handle)
    fieldIds = [dcgm_fields.DCGM_FI_DEV_GPU_TEMP, ]
    fieldGroup = pydcgm.DcgmFieldGroup(handleObj, "my_field_group", fieldIds)

    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(handle), groupName="test-metadata", 
                             groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    system = pydcgm.DcgmSystem(pydcgm.DcgmHandle(handle))
    
    _watch_field_group_basic(fieldGroup, handle, group.GetId())
    system.introspect.UpdateAll()

    memoryInfo = dcgm_agent_internal.dcgmIntrospectGetFieldMemoryUsage(handle, fieldIds[0])

    logger.debug("field %s using %.2f KB" % (fieldIds[0], memoryInfo.aggregateInfo.bytesUsed / 1024.))
    
    # 0+ to 200 KB
    assert(0 < memoryInfo.aggregateInfo.bytesUsed < 1024*200), \
        'bytes used to store field was unreasonable for ID %s, bytes: %s' \
        % (fieldIds[0], memoryInfo.aggregateInfo.bytesUsed)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_memory_get_field_group_sane(handle):
    '''
    Sanity test for API that gets memory usage of a single field group
    '''
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skipping developer test.")
    handle = pydcgm.DcgmHandle(handle)
    group = pydcgm.DcgmGroup(handle, groupName='test-metadata', groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    system = pydcgm.DcgmSystem(handle)

    fieldIds = [dcgm_fields.DCGM_FI_DEV_GPU_TEMP, dcgm_fields.DCGM_FI_DEV_POWER_USAGE]

    fieldGroup = pydcgm.DcgmFieldGroup(handle, "my_field_group", fieldIds)
    
    # ensure that the field group is watched
    _watch_field_group_basic(fieldGroup, handle.handle, group.GetId())
    system.introspect.UpdateAll()
    
    memoryInfo = system.introspect.memory.GetForFieldGroup(fieldGroup)
    
    logger.debug("field group %s is using %.2f KB" % (fieldGroup.name, memoryInfo.aggregateInfo.bytesUsed / 1024.))

    # 0+ to 20 MB
    assert(0 < memoryInfo.aggregateInfo.bytesUsed < 1024*1024*20), \
        'bytes used to store field was unreasonable for field group %s, bytes: %s' \
        % (fieldGroup.name, memoryInfo.aggregateInfo.bytesUsed)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_memory_get_all_fields_sane(handle):
    """
    Sanity test for API that gets memory usage of all fields together
    """
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skipping developer test.")
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    group = pydcgm.DcgmGroup(handle, groupName="metadata-test", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    
    # watch a ton of fields so that we know that some are being stored
    test_utils.watch_all_fields(handle.handle, group.GetGpuIds(), updateFreq=1000)
    system.introspect.UpdateAll() 
    
    memoryInfo = system.introspect.memory.GetForAllFields().aggregateInfo
    
    logger.debug('All fields in hostengine are using %.2f MB' % (memoryInfo.bytesUsed / 1024. / 1024.))

    assert(1024*20 < memoryInfo.bytesUsed < 100*1024*1024), memoryInfo.bytesUsed        # 20 KB to 100 MB
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_memory_aggregate_is_sum_of_gpu_and_global(handle):
    """
    Ensure that when memory info is retrieved relating to fields that the "global" and "gpu" 
    values add up to the "aggregate" value
    """
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    group = pydcgm.DcgmGroup(handle, groupName="metadata-test", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    
    # watch a ton of fields so that we know that some are being stored
    test_utils.watch_all_fields(handle.handle, group.GetGpuIds(), updateFreq=100000)
    system.introspect.UpdateAll() 
    
    memoryInfo = system.introspect.memory.GetForAllFields()
    
    gpuMemory = sum(
        mem.bytesUsed 
        for mem in memoryInfo.gpuInfo[:memoryInfo.gpuInfoCount]
    )
    
    globalMemory = memoryInfo.globalInfo.bytesUsed if memoryInfo.hasGlobalInfo else 0

    if (memoryInfo.hasGlobalInfo):
        logger.debug('global mem info: %s' % (memoryInfo.globalInfo))

    for i in range(memoryInfo.gpuInfoCount):
        logger.debug('gpu mem info %s: %s' % (i, memoryInfo.gpuInfo[i]))

    assert(memoryInfo.aggregateInfo.bytesUsed == gpuMemory + globalMemory), (
           'aggregate for all fields reports %s bytes but a sum of GPU and global reports %s bytes. '
           % (memoryInfo.aggregateInfo.bytesUsed, gpuMemory + globalMemory)
           + ' These values should be equal.')
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_exectime_aggregate_is_sum_of_gpu_and_global(handle):
    """
    Ensure that when execution time is retrieved relating to fields that the "global" and "gpu" 
    values add up to the "aggregate" value
    """
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    group = pydcgm.DcgmGroup(handle, groupName="metadata-test", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    
    # watch a ton of fields so that we know that some are being stored
    test_utils.watch_all_fields(handle.handle, group.GetGpuIds(), updateFreq=100000)
    system.introspect.UpdateAll() 
    
    execTimeInfo = system.introspect.execTime.GetForAllFields()
    
    gpuExecTime = sum(
        info.totalEverUpdateUsec
        for info in execTimeInfo.gpuInfo[:execTimeInfo.gpuInfoCount]
    )
    
    if execTimeInfo.hasGlobalInfo:
        globalExecTime = execTimeInfo.globalInfo.totalEverUpdateUsec
    else:
        globalExecTime = 0

    assert(execTimeInfo.aggregateInfo.totalEverUpdateUsec == globalExecTime + gpuExecTime), (
           'aggregate for all fields reports %s usec but GPUs report %s usec and global reports %s usec. '
           % (execTimeInfo.aggregateInfo.totalEverUpdateUsec, gpuExecTime, globalExecTime)
           + ' GPUs + global should sum to aggregate.')
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_mean_update_frequency(handle):
    """
    Ensure that mean update frequency is being calculated properly
    """
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    group = pydcgm.DcgmGroup(handle, groupName="metadata-test", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    
    # these frequencies must have a perfect integer mean or the last assertion will fail
    updateFreqs = {
        dcgm_fields.DCGM_FI_DEV_POWER_USAGE: 10000, 
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP: 20000,
    }
    meanUpdateFreq = stats.mean(list(updateFreqs.values()))
    
    gpuId = group.GetGpuIds()[0]
    fieldIds = []
    
    for fieldId, freqUsec in list(updateFreqs.items()):
        fieldIds.append(fieldId)
        dcgm_agent_internal.dcgmWatchFieldValue(handle.handle, gpuId, 
                                                fieldId, 
                                                freqUsec, 
                                                100000, 
                                                10)
        
    system.UpdateAllFields(True)
    system.introspect.UpdateAll()

    fieldGroup = pydcgm.DcgmFieldGroup(handle, "my_field_group", fieldIds)
    execTime = system.introspect.execTime.GetForFieldGroup(fieldGroup)
    
    resultGpuIndex = -1
    for i in range(execTime.gpuInfoCount):
        if execTime.gpuIdsForGpuInfo[i] == gpuId:
            resultGpuIndex = i
            break
        
    assert(resultGpuIndex >= 0), "no results returned for the watched GPU"
    
    actualMeanUpdateFreq = execTime.gpuInfo[resultGpuIndex].meanUpdateFreqUsec
    assert(actualMeanUpdateFreq == meanUpdateFreq), "expected %s, got %s" \
        % (meanUpdateFreq, actualMeanUpdateFreq)
    
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_with_introspection_enabled()
def test_dcgm_standalone_metadata_memory_get_hostengine_sane(handle):
    """
    Sanity test for API that gets memory usage of the hostengine process
    """
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skipping developer test.")
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    system.introspect.UpdateAll() 
    
    bytesUsed = system.introspect.memory.GetForHostengine().bytesUsed 
    
    logger.debug('the hostengine process is using %.2f MB' % (bytesUsed / 1024. / 1024.))
    
    assert(1*1024*1024 < bytesUsed < 100*1024*1024), bytesUsed        # 1MB to 100MB
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_memory_get_aggregate_fields_equals_total(handle):
    system = pydcgm.DcgmSystem(pydcgm.DcgmHandle(handle))
    
    _metadata_get_aggregate_fields_equals_total(
        handle, 
        dcgm_agent_internal.dcgmIntrospectGetFieldMemoryUsage, 
        system.introspect.memory.GetForAllFields, 
        'bytesUsed'
    )

def _metadata_get_aggregate_fields_equals_total(handle, getForFieldFn, getForAllFieldsFn, metaAttr):
    '''
    Test that the sum of API calls for metadata for single fields equals the 
    API call the same metadata used by all fields.  
    '''
    aggregateVal, startVal, endVal = _sumMetadata(handle, getForFieldFn, getForAllFieldsFn, metaAttr)
    
    assert(startVal <= aggregateVal <= endVal), \
        'sum of "%s" for all individual fields was not reasonably close to the actual total. ' % metaAttr \
        + 'Got %s, expected to be between %s and %s'\
        % (aggregateVal, startVal, endVal)
    
@test_utils.run_with_embedded_host_engine()
def test_dcgm_embedded_metadata_disabled(handle):
    _assert_metadata_not_configured_failure(handle)
    
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
def test_dcgm_standalone_metadata_disabled(handle):
    _assert_metadata_not_configured_failure(handle)
    
def _assert_metadata_not_configured_failure(handle):
    """
    Verifies that:
    1. metadata gathering is disabled by default 
    2. an appropriate error is raised when metadata APIs are accessed but 
       metadata gathering is disabled.
    """
    system = pydcgm.DcgmSystem(pydcgm.DcgmHandle(handle))

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        memoryInfo = system.introspect.memory.GetForAllFields()
        
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_exectime_get_all_fields_sane(handle):
    """
    Sanity test for API that gets execution time of all fields together
    """
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skipping developer test.")
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    group = pydcgm.DcgmGroup(handle, groupName="metadata-test", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    
    # watch a ton of fields so that we know that some are being stored
    updateFreqUsec = 1000
    test_utils.watch_all_fields(handle.handle, group.GetGpuIds(), updateFreq=updateFreqUsec)
    system.introspect.UpdateAll()
    
    execTime = system.introspect.execTime.GetForAllFields().aggregateInfo
    
    perGpuSane = 300*1000 # 300 ms
    activeGpuCount = test_utils.get_live_gpu_count(handle.handle)
    saneLimit = perGpuSane*activeGpuCount
    
    # test that all struct fields in the API response have reasonable values
    assert(100 < execTime.totalEverUpdateUsec < saneLimit), (
        'execution time seems way too long for a system with %s gpus. Took %s ms. Sane limit: %s ms' 
        % (activeGpuCount, execTime.totalEverUpdateUsec/1000, saneLimit/1000))
    
    assert(100 < execTime.recentUpdateUsec < saneLimit), (
        'recent update time seems way too long for a system with %s gpus. Took %s ms. Sane limit: %s ms' 
        % (activeGpuCount, execTime.recentUpdateUsec/1000, saneLimit/1000))
    
    assert(updateFreqUsec-1 <= execTime.meanUpdateFreqUsec <= updateFreqUsec+1), execTime.meanUpdateFreqUsec
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_exectime_get_field_group_sane(handle):
    """
    Sanity test for API that gets execution time of all fields together
    """
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skipping developer test.")
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    group = pydcgm.DcgmGroup(handle, groupName="metadata-test", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)

    fieldIds = [dcgm_fields.DCGM_FI_DEV_POWER_USAGE, dcgm_fields.DCGM_FI_DEV_SM_CLOCK, dcgm_fields.DCGM_FI_DEV_GPU_TEMP]
    fieldGroup = pydcgm.DcgmFieldGroup(handle, "my_field_group", fieldIds)

    updateFreqUsec = 1000
    _watch_field_group_basic(fieldGroup, handle.handle, group.GetId(), updateFreq=updateFreqUsec)
    system.introspect.UpdateAll()
    
    execTime = system.introspect.execTime.GetForFieldGroup(fieldGroup).aggregateInfo
    
    # test that all struct fields in the API response have reasonable values
    assert(100 < execTime.totalEverUpdateUsec < 100*1000), execTime.totalEverUpdateUsec
    assert(100 < execTime.recentUpdateUsec < 100*1000), execTime.recentUpdateUsec
    assert(updateFreqUsec == execTime.meanUpdateFreqUsec), execTime.meanUpdateFreqUsec
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_exectime_get_field_sane(handle):
    """
    Sanity test for API that gets execution time of a single field
    """
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skipping developer test.")
    
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    group = pydcgm.DcgmGroup(handle, groupName="metadata-test", groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    
    updateFreqUsec = 1000
    dcgm_agent_internal.dcgmWatchFieldValue(handle.handle, group.GetGpuIds()[0], 
                                            dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 
                                            updateFreqUsec, 
                                            100000, 
                                            10)
    system.UpdateAllFields(True)
    system.introspect.UpdateAll()
    
    execTime = dcgm_agent_internal.dcgmIntrospectGetFieldExecTime(
        handle.handle, dcgm_fields.DCGM_FI_DEV_GPU_TEMP).aggregateInfo
     
    # test that all struct fields in the API response have reasonable values
    assert(100 < execTime.totalEverUpdateUsec < 100*1000), execTime.totalEverUpdateUsec
    assert(100 < execTime.recentUpdateUsec < 100*1000), execTime.recentUpdateUsec
    assert(updateFreqUsec == execTime.meanUpdateFreqUsec), execTime.meanUpdateFreqUsec
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_exectime_get_aggregate_fields_equals_total(handle):
    '''
    Test that the aggregate of execution time across all fields is within 5% of the total value.
    '''
    system = pydcgm.DcgmSystem(pydcgm.DcgmHandle(handle))
    
    _metadata_get_aggregate_fields_equals_total(
        handle, 
        dcgm_agent_internal.dcgmIntrospectGetFieldExecTime, 
        system.introspect.execTime.GetForAllFields, 
        'totalEverUpdateUsec'
    )   
    
def _sumMetadata(handle, getForFieldFn, getForAllFieldsFn, metaAttr):
    '''
    Return a 3-tuple where the first entry is the aggregate of summing the metadata for every 
    field, the second entry is the total before aggregating and the third entry is the total after aggregating.
    '''
    system = pydcgm.DcgmSystem(pydcgm.DcgmHandle(handle))
    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(handle), groupName="test-metadata", 
                             groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
    
    # watch every field possible on all GPUs
    watchedFields = test_utils.watch_all_fields(handle, group.GetGpuIds())
    
    system.introspect.UpdateAll() 
                
    # Get the total before and after to accomodate for any slight changes 
    # in total memory usage while the individual field amounts are being summed
    startVal = getForAllFieldsFn().aggregateInfo.__getattribute__(metaAttr)
    
    aggregateVal = sum(
        getForFieldFn(handle, fieldId).aggregateInfo.__getattribute__(metaAttr)
        for fieldId in watchedFields
    )
    
    endVal = getForAllFieldsFn().aggregateInfo.__getattribute__(metaAttr)
    
    return aggregateVal, startVal, endVal
    
def _metadata_get_aggregate_fields_approx_equals_total(handle, getForFieldFn, getForAllFieldsFn, metaAttr):
    '''
    Test that the aggregate of a type of metadata across all fields is within 5% of the total for that metadata.
    '''
    
    aggregateVal, startVal, endVal = _sumMetadata(handle, getForFieldFn, getForAllFieldsFn, metaAttr)
    approxTotal = (startVal + endVal) / 2.0
    
    assert(0.95 * approxTotal <= aggregateVal <= 1.05 * approxTotal), \
        'sum of "%s" for all individual fields was not reasonably close to the actual total. ' % metaAttr \
        + 'Got %s, expected to be between %s and %s'\
        % (aggregateVal, startVal, endVal)
        
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_introspection_enabled()
def test_dcgm_embedded_metadata_cpuutil_get_hostengine_sane(handle):
    """
    Sanity test for API that gets CPU Utilization of the hostengine process.
    """
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)

    # wait up to 1 second for CPU utilization to be averaged properly (not be 0)
    for attempt in range(100):
        cpuUtil = system.introspect.cpuUtil.GetForHostengine()
        assert(0 <= cpuUtil.total <= 1), cpuUtil.total

        if (0 < cpuUtil.total < 1):
            break
        time.sleep(0.010)

    # 0+% to 50% CPU utilization
    assert(0.00001 < cpuUtil.total < 0.50), cpuUtil.total

    # test that user and kernel add to total (with rough float accuracy)
    assert(abs(cpuUtil.total - (cpuUtil.user + cpuUtil.kernel)) <= 4*float_info.epsilon), \
           'CPU kernel and user utilization did not add up to total. Kernel: %f, User: %f, Total: %f' \
           % (cpuUtil.kernel, cpuUtil.user, cpuUtil.total)

