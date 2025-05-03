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
from DcgmReader import *
import pydcgm
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_fields
import dcgm_fields_internal
from dcgm_structs import dcgmExceptionClass
import logger
import test_utils
import time

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgm_reader_default(handle, gpuIds):
    # pylint: disable=undefined-variable
    dr = DcgmReader()
    dr.SetHandle(handle)
    latest = dr.GetLatestGpuValuesAsFieldNameDict()

    for gpuId in latest:
        # latest data might be less than the list, because blank values aren't included
        # Defined in DcgmReader
        # pylint: disable=undefined-variable
        assert len(latest[gpuId]) <= len(defaultFieldIds)

        # Make sure we get strings
        for key in latest[gpuId]:
            assert isinstance(key, str)

    sample = dr.GetLatestGpuValuesAsFieldIdDict()

    for gpuId in sample:
        # Defined in DcgmReader
        # pylint: disable=undefined-variable
        assert len(sample[gpuId]) <= len(defaultFieldIds)
        
        # Make sure we get valid integer field ids
        for fieldId in sample[gpuId]:
            assert isinstance(fieldId, int)
            assert dcgm_fields.DcgmFieldGetById(fieldId) != None

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgm_reader_specific_fields(handle, gpuIds):
    specificFields = [dcgm_fields.DCGM_FI_DEV_POWER_USAGE, dcgm_fields.DCGM_FI_DEV_XID_ERRORS]
    # pylint: disable=undefined-variable
    dr = DcgmReader(fieldIds=specificFields)
    dr.SetHandle(handle)
    latest = dr.GetLatestGpuValuesAsFieldNameDict()

    for gpuId in latest:
        assert len(latest[gpuId]) <= len(specificFields)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_reading_specific_data(handle, gpuIds):
    """ 
    Verifies that we can inject specific data and get that same data back
    """

    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()

    specificFieldIds = [ dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                         dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
                         dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
                       ]
    fieldValues = [ 1,
                    1000,
                    9000,
                  ]
    
    for i in range(0, len(specificFieldIds)):
        field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        field.fieldId = specificFieldIds[i]
        field.status = 0
        field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
        field.ts = int((time.time()+10) * 1000000.0) # set the injected data into the future
        field.value.i64 = fieldValues[i]
        ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
        assert (ret == dcgm_structs.DCGM_ST_OK)
    
    # pylint: disable=undefined-variable
    dr = DcgmReader(fieldIds=specificFieldIds)
    dr.SetHandle(handle)
    latest = dr.GetLatestGpuValuesAsFieldIdDict()

    assert len(latest[gpuIds[0]]) == len(specificFieldIds)

    for i in range(0, len(specificFieldIds)):
        assert latest[gpuIds[0]][specificFieldIds[i]] == fieldValues[i]
        
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_on_non_mig_gpus()
@test_utils.run_with_non_mig_cuda_visible_devices()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_with_cuda_app()
def test_reading_pid_fields(handle, gpuIds, cudaApp):
    """
    Verifies that we can decode PID structs
    """
    fieldTag = dcgm_fields_internal.DCGM_FI_DEV_COMPUTE_PIDS
    pids = []

    # pylint: disable=undefined-variable
    dr = DcgmReader(fieldIds=[ fieldTag ], updateFrequency=100000)
    logger.debug("Trying for 5 seconds")
    exit_loop = False
    for _ in range(150):
        if (exit_loop):
            break

        data = dr.GetLatestGpuValuesAsFieldIdDict()
        assert len(data) > 0

        for gpuId in data:
            gpuData = data[gpuId]
            if fieldTag in gpuData:
                pids.append(gpuData[fieldTag].pid)
                if gpuData[fieldTag].pid == cudaApp.getpid():
                    # Found our PID. Exit the loop
                    exit_loop = True
        time.sleep(0.2)

    logger.debug("PIDs: %s. cudaApp PID: %d" % (str(pids), cudaApp.getpid()))
    assert cudaApp.getpid() in pids, "could not find cudaApp PID"

def util_dcgm_reader_all_since_last_call(handle, flag, repeat):
    """
    Test to ensure GetAllValuesAsDictSinceLastCall behaves. It was first used
    for collectd integration to ensure it does not crash and also checks that
    no unexpected fields are returned.
    
    Arguments:
        handle: DCGM handle
        flag:   argument for GetAllGpuValuesAsDictSinceLastCall
        repeat: whether to repeat GetAllGpuValuesAsDictsSinceLastCall call
    """
    specificFields = [dcgm_fields.DCGM_FI_DEV_POWER_USAGE, dcgm_fields.DCGM_FI_DEV_XID_ERRORS]
    # pylint: disable=undefined-variable
    dr = DcgmReader(fieldIds=specificFields)
    dr.SetHandle(handle)
    latest = dr.GetAllGpuValuesAsDictSinceLastCall(flag)

    if repeat:
        latest = dr.GetAllGpuValuesAsDictSinceLastCall(flag)

    if flag == False:
        dcgmHandle = pydcgm.DcgmHandle(handle)
        dcgmSystem = dcgmHandle.GetSystem()
        fieldTags = []

        for fieldId in specificFields:
            fieldTags.append(dcgmSystem.fields.GetFieldById(fieldId).tag)

    for gpuId in latest:
        # Latest data might be less than the list, because blank values aren't
        # included. We basically try to ensure there is no crash and we don't
        # return something absurd.
        assert len(latest[gpuId]) <= len(specificFields)

        for key in latest[gpuId].keys():
            if flag == False:
                assert key in fieldTags
            else:
                assert key in specificFields
                    
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgm_reader_all_since_last_call_false(handle, gpuIds):
    util_dcgm_reader_all_since_last_call(handle, False, False)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgm_reader_all_since_last_call_true(handle, gpuIds):
    util_dcgm_reader_all_since_last_call(handle, True, False)
        
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgm_reader_all_since_last_call_false_repeat(handle, gpuIds):
    util_dcgm_reader_all_since_last_call(handle, False, True)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_dcgm_reader_all_since_last_call_true_repeat(handle, gpuIds):
    util_dcgm_reader_all_since_last_call(handle, True, True)

def helper_mig_init_field_values(handle, ciIds, fieldIds, fieldValues):
    """
    Helper to inititialize MIG CI field value tests.
    """
    
    # Create a field object to insert into globals, GPUs, GIs, and CIs
    
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.status = 0
    field.ts = int((time.time()) * 1000000.0) # now

    # Insert MIC CI data into MIG GPUs

    fieldIds.extend( [ dcgm_fields.DCGM_FI_DEV_FB_FREE,
                       dcgm_fields.DCGM_FI_DEV_FB_USED,
                       dcgm_fields.DCGM_FI_DEV_FB_TOTAL ]
    )

    """
    These have to be unique and match the field Id above with the same index.
    """
    fieldValues.extend([ 30, 40, 70])

    nonGlobalData = []

    for i in range(0, len(fieldIds)):
        nonGlobalData.append({ "type" : ord(dcgm_fields.DCGM_FT_INT64),
                               "fieldId" : fieldIds[i],
                               "value" : fieldValues[i]})

    for ci in ciIds:
        for data in nonGlobalData:
            field.fieldType = data["type"]
            field.fieldId = data["fieldId"]
            field.value.i64 = data["value"]

            ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_GPU_CI, ci, field)
            assert (ret == dcgm_structs.DCGM_ST_OK)

    time.sleep(0.050)

def helper_dcgm_reader_latest_mig_ci_fields(handle, gpuIds, instanceIds, ciIds, mapById = True):
    """
    Test DcgmiReader for MIG CI fields.
    """

    fieldIds = []
    fieldValues = []

    helper_mig_init_field_values(handle, ciIds, fieldIds, fieldValues)
    
    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(handle), groupName='allMigs', groupType=dcgm_structs.DCGM_GROUP_DEFAULT_COMPUTE_INSTANCES)
    dr = DcgmReader(fieldIds=fieldIds, entities=group.GetEntities())
    dr.SetHandle(handle)

    """
    Initialize dictionary
    """
    
    if mapById:
        latest = dr.GetLatestEntityValuesAsFieldIdDict()
    else:
        latest = dr.GetLatestEntityValuesAsFieldNameDict()

    """
        latest = dr.GetLatestEntityValuesAsDict(mapById)
    """

    foundValues = 0;

    for entityGroupId in latest:
        if entityGroupId != dcgm_fields.DCGM_FE_GPU_CI:
            continue

        for entityId in latest[entityGroupId]:
            if entityId not in ciIds:
                continue;

            foundValues = 0;
            
            for fieldId in latest[entityGroupId][entityId]:
                value = latest[entityGroupId][entityId][fieldId]

                if not mapById:
                    fieldId = dcgm_fields.DcgmFieldGetIdByTag(fieldId)

                if fieldId not in fieldIds:
                    continue
                
                assert(fieldValues[fieldIds.index(fieldId)] == value)
                foundValues += 1

            assert(foundValues == len(fieldValues))

def helper_dcgm_reader_all_mig_ci_fields(handle, gpuIds, instanceIds, ciIds, mapById = True):
    """
    Test DcgmiReader for MIG CI fields.
    """

    fieldIds = []
    fieldValues = []

    helper_mig_init_field_values(handle, ciIds, fieldIds, fieldValues)
    
    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(handle), groupName='allMigs', groupType=dcgm_structs.DCGM_GROUP_DEFAULT_COMPUTE_INSTANCES)
    dr = DcgmReader(fieldIds=fieldIds, entities=group.GetEntities())
    dr.SetHandle(handle)

    """
    Initialize dictionary
    """
    
    if mapById:
        dr.GetAllEntityValuesAsFieldIdDictSinceLastCall()
        latest = dr.GetAllEntityValuesAsFieldIdDictSinceLastCall()
    else:
        dr.GetAllEntityValuesAsFieldNameDictSinceLastCall()
        latest = dr.GetAllEntityValuesAsFieldNameDictSinceLastCall()

    """
        dr.GetAllEntityValuesAsDictSinceLastCall(mapById)
        latest = dr.GetAllEntityValuesAsDictSinceLastCall(mapById)
    """

    foundValues = 0;

    for entityGroupId in latest:
        if entityGroupId != dcgm_fields.DCGM_FE_GPU_CI:
            continue

        for entityId in latest[entityGroupId]:
            if entityId not in ciIds:
                continue;

            foundValues = 0;
            
            for fieldId in latest[entityGroupId][entityId]:
                fieldId2 = fieldId

                if not mapById:
                    fieldId2 = dcgm_fields.DcgmFieldGetIdByTag(fieldId2)

                if fieldId2 not in fieldIds:
                    continue
                
                for value in latest[entityGroupId][entityId][fieldId]:
                    if value.value not in fieldValues:
                        continue

                    assert(fieldValues[fieldIds.index(fieldId2)] == value.value)
                    foundValues += 1
                    break

            assert(foundValues == len(fieldValues))

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus(2) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_with_injection_gpu_instances(1)
@test_utils.run_with_injection_gpu_instances(1, 1)
@test_utils.run_with_injection_gpu_compute_instances(1, 1)
def test_dcgm_reader_wildcard_gi(handle, gpuIds, instanceIds, ciIds):
    """
    Test DcgmCacheManager for removal of wildcard GI Fake Ci insertion.
    """

    for i in gpuIds:
        logger.debug("GPU ID: %d" % (i))

    for i in instanceIds:
        logger.debug("GPU Instance ID: %d" % (i))
        
    for i in ciIds:
        logger.debug("Compute Instance ID: %d" % (i))

    # We inject two GPUs so the offset of the second GI (and this first CI) is
    # at least one GPU's worth of max GIs. If we injected one, and had no real
    # GPUs, # we'd be dividing by gpuIds[0], which would be 0.
        
    assert (ciIds[0] / gpuIds[1]) == dcgm_structs.DCGM_MAX_COMPUTE_INSTANCES_PER_GPU, "CIs Expected value %d, observed value: %d"% (dcgm_structs.DCGM_MAX_COMPUTE_INSTANCES_PER_GPU, ciIds[0] / gpuIds[1])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus(1) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_with_injection_gpu_instances(1)
@test_utils.run_with_injection_gpu_compute_instances(4)
def test_dcgm_reader_latest_mig_ci_fields_by_id(handle, gpuIds, instanceIds, ciIds):
    """
    Test DcgmiReader for MIG CI fields.
    """

    helper_dcgm_reader_latest_mig_ci_fields(handle, gpuIds, instanceIds, ciIds, True)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus(1) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_with_injection_gpu_instances(1)
@test_utils.run_with_injection_gpu_compute_instances(4)
def test_dcgm_reader_all_mig_ci_fields_by_id(handle, gpuIds, instanceIds, ciIds):
    """
    Test DcgmiReader for MIG CI fields.
    """

    helper_dcgm_reader_all_mig_ci_fields(handle, gpuIds, instanceIds, ciIds, True)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus(1) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_with_injection_gpu_instances(1)
@test_utils.run_with_injection_gpu_compute_instances(4)
def test_dcgm_reader_latest_mig_ci_fields_by_tag(handle, gpuIds, instanceIds, ciIds):
    """
    Test DcgmiReader for MIG CI fields.
    """

    helper_dcgm_reader_latest_mig_ci_fields(handle, gpuIds, instanceIds, ciIds, False)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus(1) #Injecting compute instances only works with live ampere or injected GPUs
@test_utils.run_with_injection_gpu_instances(1)
@test_utils.run_with_injection_gpu_compute_instances(4)
def test_dcgm_reader_all_mig_ci_fields_by_tag(handle, gpuIds, instanceIds, ciIds):
    """
    Test DcgmiReader for MIG CI fields.
    """

    helper_dcgm_reader_all_mig_ci_fields(handle, gpuIds, instanceIds, ciIds, False)
