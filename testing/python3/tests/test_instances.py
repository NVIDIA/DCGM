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
import test_utils
import dcgm_internal_helpers
import dcgm_agent_internal
import dcgm_structs_internal
import dcgm_agent
import dcgm_fields
import dcgm_structs
import dcgm_errors
import DcgmHandle
import subprocess
import time
import logger
import dcgmvalue

def create_fake_gpu_instances(handle, gpuId, instanceCount):
    cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
    cfe.numToCreate = 0
    fakeInstanceMap = {}

    if instanceCount > 0:
        for i in range(0, instanceCount):
            cfe.entityList[cfe.numToCreate].parent.entityGroupId = dcgm_fields.DCGM_FE_GPU
            cfe.entityList[cfe.numToCreate].parent.entityId = gpuId
            cfe.entityList[cfe.numToCreate].entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_I
            cfe.numToCreate += 1

        # Create the instances first so we can control which GPU the compute instances are placed on
        updated = dcgm_agent_internal.dcgmCreateFakeEntities(handle, cfe)
        for i in range(0, updated.numToCreate):
            if updated.entityList[i].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
                fakeInstanceMap[updated.entityList[i].entity.entityId] = updated.entityList[i].parent.entityId

    return fakeInstanceMap

def create_fake_compute_instances(handle, parentIds, ciCount):
    fakeCIMap = {}
    if ciCount > 0:
        cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
        instanceIndex = 0
        for i in range(0, ciCount):
            cfe.entityList[cfe.numToCreate].parent.entityGroupId = dcgm_fields.DCGM_FE_GPU_I
            if instanceIndex > len(parentIds):
                instanceIndex = 0
            cfe.entityList[cfe.numToCreate].parent.entityId = parentIds[instanceIndex]
            instanceIndex = instanceIndex + 1
            cfe.entityList[cfe.numToCreate].entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_CI
            cfe.numToCreate += 1

        updated = dcgm_agent_internal.dcgmCreateFakeEntities(handle, cfe)
        for i in range(0, updated.numToCreate):
            if updated.entityList[i].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_CI:
                fakeCIMap[updated.entityList[i].entity.entityId] = updated.entityList[i].parent.entityId

    return fakeCIMap

def ensure_instance_ids(handle, gpuId, minInstances, minCIs):
    hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(handle)
    legalGpu = False
    instanceMap = {}
    ciMap = {}
    legalInstances = []

    for i in range(0, hierarchy.count):
        entity = hierarchy.entityList[i]
        if entity.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
            if entity.parent.entityId == gpuId:
                legalGpu = True
                instanceMap[entity.entity.entityId] = entity.parent.entityId
            else:
                legalGpu = False
        elif entity.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_CI and legalGpu:
            ciMap[entity.entity.entityId] = entity.parent.entityId
            legalInstances.append(entity.parent.entityId)

    instancesNeeded = minInstances - len(instanceMap)
    cisNeeded = minCIs - len(ciMap)

    fakeInstanceMap = create_fake_gpu_instances(handle, gpuId, instancesNeeded)
    for fakeInstance in fakeInstanceMap:
        legalInstances.append(fakeInstance)

    instanceMap.update(fakeInstanceMap)

    fakeCIMap = create_fake_compute_instances(handle, legalInstances, cisNeeded)
    ciMap.update(fakeCIMap)

    return instanceMap, ciMap

def helper_test_inject_instance_fields(handle, gpuIds):
    instances, cis = ensure_instance_ids(handle, gpuIds[0], 1, 1)
    firstInstanceId = list(instances.keys())[0]
    lastCIId = list(cis.keys())[0]

    # Set up the watches on these groups
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_EMPTY, 'tien')
    fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(handle, [dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL], 'kal')

    dcgm_agent.dcgmGroupAddEntity(handle, groupId, dcgm_fields.DCGM_FE_GPU, gpuIds[0])
    dcgm_agent.dcgmGroupAddEntity(handle, groupId, dcgm_fields.DCGM_FE_GPU_I, firstInstanceId)
    dcgm_agent.dcgmGroupAddEntity(handle, groupId, dcgm_fields.DCGM_FE_GPU_CI, lastCIId)
    dcgm_agent.dcgmWatchFields(handle, groupId, fieldGroupId, 1, 100, 100)

    dcgm_internal_helpers.inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                       2, 5, isInt=True, verifyInsertion=True,
                                       entityType=dcgm_fields.DCGM_FE_GPU)

    # Read the values to make sure they were stored properly
    entities = [dcgm_structs.c_dcgmGroupEntityPair_t(), dcgm_structs.c_dcgmGroupEntityPair_t(),
                dcgm_structs.c_dcgmGroupEntityPair_t()]

    entities[0].entityGroupId = dcgm_fields.DCGM_FE_GPU_I
    entities[0].entityId = firstInstanceId
    entities[1].entityGroupId = dcgm_fields.DCGM_FE_GPU_CI
    entities[1].entityId = lastCIId
    entities[2].entityGroupId = dcgm_fields.DCGM_FE_GPU
    entities[2].entityId = gpuIds[0]

    fieldIds = [dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL]

    values = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entities, fieldIds, 0)
    for v in values:
        if v.entityGroupId == dcgm_fields.DCGM_FE_GPU:
            assert v.value.i64 == 2, "Failed to inject value 2 for entity %u from group %u" % (
                v.entityId, v.entityGroupId)
        else:
            from dcgm_structs import DCGM_ST_NO_DATA
            assert (v.status == DCGM_ST_NO_DATA), "Injected meaningless value %u for entity %u from group %u" % (
                v.value.i64, v.entityId, v.entityGroupId)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_inject_instance_fields_standalone(handle, gpuIds):
    helper_test_inject_instance_fields(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_inject_instance_fields_embedded(handle, gpuIds):
    helper_test_inject_instance_fields(handle, gpuIds)

def verify_fake_profile_names(handle, fakeEntities, isGpuInstance):
    fieldIds = [dcgm_fields.DCGM_FI_DEV_NAME]
    entities = []
    for entityId in fakeEntities:
        entity = dcgm_structs.c_dcgmGroupEntityPair_t()
        if isGpuInstance:
            entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_I
        else:
            entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_CI
        entity.entityId = entityId
        entities.append(entity)

    values = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entities, fieldIds, dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)

    if isGpuInstance:
        expectedFakeName = "1fg.4gb"
    else:
        expectedFakeName = "1fc.1g.4gb"
    
    for v in values:
        assert v.value.str == expectedFakeName, "Fake profile name appears to be wrong. Expected '%s', found '%s'" % (
            expectedFakeName, v.value.str)

def verify_profile_names_exist(handle, migEntityList, isGpuInstance):
    fieldIds = [dcgm_fields.DCGM_FI_DEV_NAME]
    entities = []
    for entityId in migEntityList:
        entity = dcgm_structs.c_dcgmGroupEntityPair_t()
        if isGpuInstance:
            entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_I
        else:
            entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_CI
        entity.entityId = entityId
        entities.append(entity)

    values = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entities, fieldIds, dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)

    for v in values:
        assert len(v.value.str) and v.value.str != dcgmvalue.DCGM_STR_BLANK, \
               "Expected a non-empty profile name, but found '%s'" % (v.value.str)

def helper_test_fake_mig_device_profile_names(handle, gpuIds):
    fakeInstanceMap = {}
    for gpuId in gpuIds:
        tmpMap = create_fake_gpu_instances(handle, gpuId, 1)
        fakeInstanceMap.update(tmpMap)
        
    fakeCIMap = create_fake_compute_instances(handle, list(fakeInstanceMap.keys()), len(fakeInstanceMap))

    verify_fake_profile_names(handle, list(fakeInstanceMap.keys()), True)
    verify_fake_profile_names(handle, list(fakeCIMap.keys()), False)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_fake_mig_device_profile_names_standalone(handle, gpuIds):
    helper_test_fake_mig_device_profile_names(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_fake_mig_device_profile_names_embedded(handle, gpuIds):
    helper_test_fake_mig_device_profile_names(handle, gpuIds)

def helper_test_health_check_instances(handle, gpuIds):
    instances, cis = ensure_instance_ids(handle, gpuIds[0], 1, 1)
    instanceId = list(instances.keys())[0]
    ciId = list(cis.keys())[0]
    handleObj = DcgmHandle.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddEntity(dcgm_fields.DCGM_FE_GPU, gpuIds[0])
    groupObj.AddEntity(dcgm_fields.DCGM_FE_GPU_I, instanceId)
    groupObj.AddEntity(dcgm_fields.DCGM_FE_GPU_CI, ciId)

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)
    
    # Verify health prior to testing
    responseV4 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version4)
    if responseV4.incidentCount != 0:
        test_utils.skip_test("Cannot test on unhealthy systems.")

    # Inject one error per system
    dcgm_internal_helpers.inject_value(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                       2, 5, isInt=True, verifyInsertion=True,
                                       entityType=dcgm_fields.DCGM_FE_GPU)

    responseV4 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version4)
    assert (responseV4.incidentCount == 1), "Should have 1 total incidents but found %d" % responseV4.incidentCount

    assert (responseV4.incidents[0].entityInfo.entityId == gpuIds[0])
    assert (responseV4.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV4.incidents[0].error.code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)
    assert (responseV4.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV4.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_health_check_instances_standalone(handle, gpuIds):
    helper_test_health_check_instances(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_instances_embedded(handle, gpuIds):
    helper_test_health_check_instances(handle, gpuIds)

def populate_counts_per_gpu(hierarchy):
    gpuInstances = {}
    gpuCIIds = {}

    # Get counts for each GPU
    for i in range(0, hierarchy.count):
        entity = hierarchy.entityList[i]
        if entity.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
            if entity.parent.entityId not in gpuInstances:
                gpuInstances[entity.parent.entityId] = []
            gpuInstances[entity.parent.entityId].append(entity.entity.entityId)
        elif entity.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_CI:
            for key in gpuInstances:
                if entity.parent.entityId in gpuInstances[key]:
                    if key not in gpuCIIds:
                        gpuCIIds[key] = []
                    gpuCIIds[key].append(entity.entity.entityId)

    return gpuInstances, gpuCIIds

class ExpectedValues(object):
    def __init__(self, instanceCount=0, ciCount=0):
        self.instanceCount = instanceCount
        self.ciCount = ciCount
        self.verified = False

def create_small_mig_objects(handle, gpuIds, numToCreate):
    numInstancesCreated = 0
    for gpuId in gpuIds:
        try:
            dcgm_agent.dcgmCreateMigEntity(handle, gpuId, dcgm_structs.DcgmMigProfileGpuInstanceSlice1, dcgm_structs.DcgmMigCreateGpuInstance, 0)
            numInstancesCreated = numInstancesCreated + 1
            if numInstancesCreated >= numToCreate:
                break
        except:
            # There may not be space; ignore this.
            continue

    return numInstancesCreated

def verifyMigUpdates(handle, oGpuInstances, oGpuCIIds, numInstancesCreated, numCIsCreated, retries=19):
    newGpuInstances = []
    newComputeInstances = []

    if numInstancesCreated == 0 and numCIsCreated == 0:
        return newGpuInstances, newComputeInstances, ''

    errMsg = ''
    while retries >= 0:
        newGpuInstances = []
        newComputeInstances = []
        hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(handle)
        gpuInstances, gpuCIIds = populate_counts_per_gpu(hierarchy)

        # Add any new instances to the map
        for key in gpuInstances:
            if key in oGpuInstances:
                # Compare lists
                for instanceId in gpuInstances[key]:
                    if instanceId not in oGpuInstances[key]:
                        newGpuInstances.append(instanceId)
            else:
                # Add the entire list to the new instances
                for instanceId in gpuInstances[key]:
                    newGpuInstances.append(instanceId)

        # Add any new compute instances to the map
        for key in gpuCIIds:
            if key in oGpuCIIds:
                # Compare lists
                for ciId in gpuCIIds[key]:
                    if ciId not in oGpuCIIds[key]:
                        newComputeInstances.append(ciId)
            else:
                # Add the entire list to the new compute instances
                for ciId in gpuCIIds[key]:
                    newComputeInstances.append(ciId)

        if len(newGpuInstances) >= numInstancesCreated and len(newComputeInstances) >= numCIsCreated:
            errMsg = ''
            break
        elif len(newGpuInstances) < numInstancesCreated and len(newComputeInstances) < numCIsCreated:
            errMsg = 'Expected %d new GPU instances and %d new compute instances but only found %d and %d' % \
                     (numInstancesCreated, numCIsCreated, len(newGpuInstances), len(newComputeInstances))
        elif len(newGpuInstances) < numInstancesCreated:
            errMsg = "Expected %d new GPU instances but only found %d" % (numInstancesCreated, len(newGpuInstances))
        else:
            errMsg = "Expected %d new compute instances but only found %d" % (numCIsCreated, len(newComputeInstances))

        retries = retries - 1
        time.sleep(1)

    return newGpuInstances, newComputeInstances, errMsg    

def verify_entries_are_deleted(deletedMap, detectedMap):
    stillHere = []
    for key in deletedMap:
        for gpu in detectedMap:
            if key in detectedMap[gpu]:
                stillHere.append(key)

    return stillHere

def delete_gpu_instances(handle, newGpuInstances, flags):
    for instanceId in newGpuInstances[:-1]:
        dcgm_agent.dcgmDeleteMigEntity(handle, dcgm_fields.DCGM_FE_GPU_I, instanceId, flags) 

    dcgm_agent.dcgmDeleteMigEntity(handle, dcgm_fields.DCGM_FE_GPU_I, newGpuInstances[-1], 0) 

def delete_gpu_instances_no_fail(handle, newGpuInstances, flags):
    try:
        delete_gpu_instances(handle, newGpuInstances, flags)
    except:
        pass

def create_mig_entities_and_verify(handle, gpuIds, instanceCreateCount, minInstanceCreateCount):
    # get mig hierarchy
    hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(handle)
    oGpuInstances, oGpuCIIds = populate_counts_per_gpu(hierarchy)

    numInstancesCreated = create_small_mig_objects(handle, gpuIds, 3)
    if numInstancesCreated < minInstanceCreateCount:
        test_utils.skip_test("Cannot create any GPU instances, skipping test.")
       
    # Make sure the new instances appear 
    newGpuInstances, newComputeInstances, errMsg = verifyMigUpdates(handle, oGpuInstances, oGpuCIIds, numInstancesCreated, 0)
    assert errMsg == '', errMsg

    # Create new compute instances
    flags = dcgm_structs.DCGM_MIG_RECONFIG_DELAY_PROCESSING
    numCIsCreated = 0
    try:
        for instanceId in newGpuInstances[:-1]:
            dcgm_agent.dcgmCreateMigEntity(handle, instanceId, dcgm_structs.DcgmMigProfileComputeInstanceSlice1, \
                   dcgm_structs.DcgmMigCreateComputeInstance, flags)
            numCIsCreated = numCIsCreated + 1

        # For the last one, send a flag to ask hostengine to process the reconfiguring
        dcgm_agent.dcgmCreateMigEntity(handle, newGpuInstances[-1], dcgm_structs.DcgmMigProfileComputeInstanceSlice1, \
               dcgm_structs.DcgmMigCreateComputeInstance, 0)
        numCIsCreated = numCIsCreated + 1
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_INSUFFICIENT_RESOURCES) as e:
        delete_gpu_instances_no_fail(handle, newGpuInstances, flags)
        test_utils.skip_test("Insufficient resources to run this test")

    # Verify the new compute instances have appeared
    newGpuInstances, newComputeInstances, errMsg = verifyMigUpdates(handle, oGpuInstances, oGpuCIIds, numInstancesCreated, numCIsCreated)
    if errMsg != '':
        delete_gpu_instances_no_fail(handle, newGpuInstances, flags)
        
    assert errMsg == '', errMsg

    return newGpuInstances, newComputeInstances

def delete_compute_instances_and_verify(handle, newComputeInstances):
    errMsg = ''
    flags = dcgm_structs.DCGM_MIG_RECONFIG_DELAY_PROCESSING
    # Delete the new instances
    for ciId in newComputeInstances[:-1]:
        dcgm_agent.dcgmDeleteMigEntity(handle, dcgm_fields.DCGM_FE_GPU_CI, ciId, flags)

    # don't block processing the reconfigure with the last one
    dcgm_agent.dcgmDeleteMigEntity(handle, dcgm_fields.DCGM_FE_GPU_CI, newComputeInstances[-1], 0)

    # verify that the compute instances disappear
    retries = 20
    cisStillHere = newComputeInstances
    while retries > 0:
        hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(handle)
        gpuInstances, gpuCIIds = populate_counts_per_gpu(hierarchy)
        retries = retries - 1
        updated = verify_entries_are_deleted(cisStillHere, gpuCIIds)
        if len(updated) == 0:
            errMsg = ''
            break
        else:
            errMsg = "Compute instances '"
            for item in updated:
                errMsg = "%s %s" % (errMsg, item)
            errMsg = "%s' were not deleted successfully" % errMsg
            cisStillHere = updated

        time.sleep(1)

    return errMsg

def delete_gpu_instances_and_verify(handle, newGpuInstances):
    errMsg = ''
    flags = dcgm_structs.DCGM_MIG_RECONFIG_DELAY_PROCESSING
    delete_gpu_instances(handle, newGpuInstances, flags)

    retries = 20
    gpuInstancesStillHere = newGpuInstances
    while retries > 0:
        hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(handle)
        gpuInstances, gpuCIIds = populate_counts_per_gpu(hierarchy)
        retries = retries - 1
        updated = verify_entries_are_deleted(gpuInstancesStillHere, gpuInstances)
        if len(updated) == 0:
            errMsg = ''
            break
        else:
            errMsg = "GPU instances '"
            for item in updated:
                errMsg = "%s %s" % (errMsg, item)
            errMsg = "%s' were not deleted successfully" % errMsg
            gpuInstancesStillHere = updated

        time.sleep(1)

    return errMsg

def helper_test_mig_reconfigure(handle, gpuIds):
    newGpuInstances, newComputeInstances = create_mig_entities_and_verify(handle, gpuIds, 3, 1)
    
    verify_profile_names_exist(handle, newComputeInstances, False)

    ciFailMsg = delete_compute_instances_and_verify(handle, newComputeInstances)

    # Save this and attempt to cleanup the rest even though we failed here
    if ciFailMsg != '':
        logger.warning("The compute instances didn't clean up correctly, but we'll attempt to clean up the GPU instances anyway")

    instanceFailMsg = delete_gpu_instances_and_verify(handle, newGpuInstances)
    
    assert ciFailMsg == '', ciFailMsg
    assert instanceFailMsg == '', instanceFailMsg

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_enabled()
@test_utils.run_only_as_root()
def test_mig_reconfigure_standalone(handle, gpuIds):
    helper_test_mig_reconfigure(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_enabled()
@test_utils.run_only_as_root()
def test_mig_reconfigure_embedded(handle, gpuIds):
    helper_test_mig_reconfigure(handle, gpuIds)

def helper_test_mig_cuda_visible_devices_string(handle, gpuIds):
    hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(handle)
    gpuPartOfTest = False

    for i in range(0, hierarchy.count):
        entity = hierarchy.entityList[i]

        isInstance = False
        if entity.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
            gpuPartOfTest = entity.parent.entityId in gpuIds
            isInstance = True

        if gpuPartOfTest:
            cuda_vis = test_utils.get_cuda_visible_devices_str(handle, entity.entity.entityGroupId, entity.entity.entityId)
            assert cuda_vis[:4] == 'MIG-', "Expected the CUDA_VISIBLE_DEVICES string to start with 'MIG-', but found '%s" % (cuda_vis)
            firstSlashIndex = cuda_vis.find('/')
            assert firstSlashIndex != -1, "Expected to find '/' in CUDA_VISIBLE_DEVICES, but didn't: '%s'" % (cuda_vis)
            if not isInstance:
                secondSlashIndex = cuda_vis.find('/', firstSlashIndex+1)
                assert secondSlashIndex != -1, "Expected to find two '/' marks in CUDA_VISIBLE_DEVICES, but didn't: '%s'" % (cuda_vis)
            

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_enabled()
@test_utils.run_only_as_root()
def test_mig_cuda_visible_devices_string_embedded(handle, gpuIds):
    helper_test_mig_cuda_visible_devices_string(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_enabled()
@test_utils.run_only_as_root()
def test_mig_cuda_visible_devices_string_standalone(handle, gpuIds):
    helper_test_mig_cuda_visible_devices_string(handle, gpuIds)

def helper_test_mig_value_reporting(handle, gpuIds):
    # These fields should report the same value for GPUs, instances, and compute instances
    sameValueFieldIds = [
        dcgm_fields.DCGM_FI_DEV_COMPUTE_MODE,
        dcgm_fields.DCGM_FI_DEV_MIG_MODE,
        dcgm_fields.DCGM_FI_DEV_SHUTDOWN_TEMP,
    ]

    differentValueFieldIds = [
        dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
    ]
    
    newGpuInstances, newComputeInstances = create_mig_entities_and_verify(handle, gpuIds, 3, 1)

    # Make sure we get the same values for these fields on the GPU, instances, and compute instances

    # Build the entity list
    entities = []
    for gpuId in gpuIds:
        entities.append(dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId))
    for instanceId in newGpuInstances:
        entities.append(dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU_I, instanceId))
    for ciId in newComputeInstances:
        entities.append(dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU_CI, ciId))

    fieldIds = []
    fieldIds.extend(sameValueFieldIds)
    fieldIds.extend(differentValueFieldIds)    
    values = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entities, fieldIds, dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)
    gpuValues = {}

    # Make a map of a map the values reported by the GPUs: gpuId -> fieldId -> value
    for value in values:
        if value.entityGroupId == dcgm_fields.DCGM_FE_GPU:
            if value.entityId not in gpuValues:
                gpuValues[value.entityId] = {}
                gpuValues[value.entityId][value.fieldId] = value.value.i64
            elif value.fieldId not in gpuValues[value.entityId]:
                gpuValues[value.entityId][value.fieldId] = value.value.i64

    errMsg = ''    
    for value in values:
        if value.entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
            gpuId = value.entityId % dcgm_structs.DCGM_MAX_INSTANCES_PER_GPU
            same = gpuValues[gpuId][value.fieldId] == value.value.i64
            if not same and value.fieldId in sameValueFieldIds:
                errMsg = errMsg + "\nExpected %d but found %d for field %d GPU instance %d on GPU %d" \
                          % (gpuValues[gpuId][value.fieldId], value.value.i64, value.fieldId, value.entityId, gpuId)
            elif same and value.fieldId in differentValueFieldIds:
                errMsg = errMsg + "\nExpected different values but found %d for field %d for GPU instance %d on GPU %d" \
                          % (value.value.i64, value.fieldId, value.entityId, gpuId)
        if value.entityGroupId == dcgm_fields.DCGM_FE_GPU_CI:
            gpuId = value.entityId % dcgm_structs.DCGM_MAX_COMPUTE_INSTANCES_PER_GPU
            same = gpuValues[gpuId][value.fieldId] == value.value.i64
            if not same and value.fieldId in sameValueFieldIds:
                errMsg = errMsg + "\nExpected %d but found %d for field %d compute instance %d on GPU %d" \
                          % (gpuValues[gpuId][value.fieldId], value.value.i64, value.fieldId, value.entityId, gpuId)
            elif same and value.fieldId in differentValueFieldIds:
                errMsg = errMsg + "\nExpected different values but found %d for field %d for compute instance %d on GPU %d" \
                          % (value.value.i64, value.fieldId, value.entityId, gpuId)

    ciFailMsg = delete_compute_instances_and_verify(handle, newComputeInstances)
    instanceFailMsg = delete_gpu_instances_and_verify(handle, newGpuInstances)

    if ciFailMsg != '':
        logger.warning("The compute instances didn't clean up correctly: %s" % ciFailMsg)
    if instanceFailMsg != '':
        logger.warning("The GPU instances didn't clean up correctly: %s" % instanceFailMsg)

    assert errMsg == '', errMsg

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_enabled()
@test_utils.run_only_as_root()
def test_mig_value_reporting_embedded(handle, gpuIds):
    helper_test_mig_value_reporting(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_enabled()
@test_utils.run_only_as_root()
def test_mig_value_reporting_standalone(handle, gpuIds):
    helper_test_mig_value_reporting(handle, gpuIds)
