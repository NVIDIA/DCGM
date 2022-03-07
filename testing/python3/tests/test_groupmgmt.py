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
# test the policy manager for DCGM

import dcgm_structs
import dcgm_agent_internal
import dcgm_agent
import logger
import test_utils
import dcgm_fields
import dcgmvalue
import pydcgm
from dcgm_structs import dcgmExceptionClass

import time
import inspect
from subprocess import check_output

def helper_dcgm_group_create_grp(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    groupObj = pydcgm.DcgmGroup(handleObj, groupName="test1")
    groupId = groupObj.GetId()
    assert(groupId != 0)

    #Force the group to be deleted
    del(groupObj)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_group_create_grp_embedded(handle):
    helper_dcgm_group_create_grp(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_group_create_grp_standalone(handle):
    helper_dcgm_group_create_grp(handle)

def helper_dcgm_group_update_grp(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")

    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_EMPTY, "test1")

    gpuIdList = gpuIds
    assert len(gpuIdList) > 0, "Failed to get devices from the node"

    for gpuId in gpuIdList:
        groupObj.AddGpu(gpuId)
        gpuIdListAfterAdd = groupObj.GetGpuIds()
        assert gpuId in gpuIdListAfterAdd, "Expected gpuId %d in %s" % (gpuId, str(gpuIdListAfterAdd))

    for gpuId in gpuIdList:
        groupObj.RemoveGpu(gpuId)
        gpuIdListAfterAdd = groupObj.GetGpuIds()
        assert gpuId not in gpuIdListAfterAdd, "Expected gpuId %d NOT in %s" % (gpuId, str(gpuIdListAfterAdd))

    #Force the group to be deleted
    del(groupObj)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_group_update_grp_embedded(handle, gpuIds):
    helper_dcgm_group_update_grp(handle, gpuIds)


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_group_update_grp_standalone(handle, gpuIds):
    helper_dcgm_group_update_grp(handle, gpuIds)

def helper_dcgm_group_get_grp_info(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")

    gpuIdList = gpuIds
    assert len(gpuIdList) > 0, "Failed to get devices from the node"

    for gpuId in gpuIdList:
        groupObj.AddGpu(gpuId)

    # We used to test fetching negative value throws Bad Param error here.
    # This was only a usecase because we we mixing signed and unsigned values
    # Now we're just testing that passing an invalid group ID results in the
    # expected NOT_CONFIGURED error.
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        ret = dcgm_agent.dcgmGroupGetInfo(handle, -1)
    
    gpuIdListAfterAdd = groupObj.GetGpuIds()
    assert gpuIdList == gpuIdListAfterAdd, "Expected all GPUs from %s to be added. Got %s" % (str(gpuIdList), str(gpuIdListAfterAdd))

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_group_get_grp_info_embedded(handle, gpuIds):
    helper_dcgm_group_get_grp_info(handle, gpuIds)

def helper_dcgm_group_get_grp_info_entities(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")

    gpuIdList = gpuIds
    assert len(gpuIdList) > 0, "Failed to get devices from the node"

    for gpuId in gpuIdList:
        groupObj.AddEntity(dcgm_fields.DCGM_FE_GPU, gpuId)
    
    gpuIdListAfterAdd = groupObj.GetGpuIds()
    assert gpuIdList == gpuIdListAfterAdd, "Expected all GPUs from %s to be added. Got %s" % (str(gpuIdList), str(gpuIdListAfterAdd))

    entityListAfterAdd = groupObj.GetEntities()
    gpuList2 = []
    for entity in entityListAfterAdd:
        assert entity.entityGroupId == dcgm_fields.DCGM_FE_GPU, str(entity.entityGroupId)
        gpuList2.append(entity.entityId)
    assert gpuIdList == gpuList2, "Expected all GPUs from %s to be added. Got %s" % (str(gpuIdList), str(gpuList2))

    #Remove all GPUs
    for gpuId in gpuIdList:
        groupObj.RemoveEntity(dcgm_fields.DCGM_FE_GPU, gpuId)
    entityListAfterRem = groupObj.GetEntities()
    assert len(entityListAfterRem) == 0, str(entityListAfterRem)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_helper_dcgm_group_get_grp_info_entities(handle, gpuIds):
    helper_dcgm_group_get_grp_info_entities(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_group_get_grp_info_standalone(handle, gpuIds):
    helper_dcgm_group_get_grp_info(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_group_get_all_ids_standalone(handle):
    """
    Get all the group IDS configured on the host engine
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    #Get the list of groups before we add ours so that we account for them
    groupIdListBefore = dcgm_agent.dcgmGroupGetAllIds(handle)

    expectedCount = len(groupIdListBefore)
    groupObjs = []

    for index in range(0,10):
        expectedCount += 1
        name = 'Test'
        name += repr(index)
        groupObj = systemObj.GetEmptyGroup(name)
        groupObjs.append(groupObj) #keep reference so it doesn't go out of scope
        pass

    groupIdListAfter = dcgm_agent.dcgmGroupGetAllIds(handle)
    assert len(groupIdListAfter) == expectedCount, "Num of groups less than expected. Expected: %d Returned %d" % (expectedCount, len(groupIdListAfter))

def dcgm_group_test_default_group(handle, gpuIds):
    """
    Test that the default group can not be deleted, or manipulated and is returning all GPUs.

    Note that we're not using groupObj for some tests because it protects against operations on the default group
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    gpuIdList = gpuIds
    assert len(gpuIdList) > 0, "Failed to get devices from the node"

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, 9999)

    groupGpuIdList = groupObj.GetGpuIds()
    assert(gpuIdList == groupGpuIdList), "Expected gpuId list match %s != %s" % (str(gpuIdList), str(groupGpuIdList))
    groupEntityList = groupObj.GetEntities()
    gpuIdList2 = []
    for entity in groupEntityList:
        assert entity.entityGroupId == dcgm_fields.DCGM_FE_GPU, str(entity.entityGroupId)
        gpuIdList2.append(entity.entityId)
    assert gpuIdList == gpuIdList2, "Expected gpuId list to match entity list: %s != %s" % (str(gpuIdList), str(gpuIdList2))

    for gpuId in gpuIdList:
        with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
            ret = dcgm_agent.dcgmGroupRemoveDevice(handle, dcgm_structs.DCGM_GROUP_ALL_GPUS, gpuId)
        with test_utils.assert_raises(pydcgm.DcgmException):
            groupObj.RemoveGpu(gpuId)

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        ret = dcgm_agent.dcgmGroupDestroy(handle, dcgm_structs.DCGM_GROUP_ALL_GPUS)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_group_test_default_group_standalone(handle, gpuIds):
    dcgm_group_test_default_group(handle, gpuIds)
    
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_group_test_default_group_embedded(handle, gpuIds):
    dcgm_group_test_default_group(handle, gpuIds)

def helper_dcgm_group_delete_grp(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    groupObj = pydcgm.DcgmGroup(handleObj, groupName="test1")
    groupId = groupObj.GetId().value

    #Delete the group
    groupObj.Delete()
    
    ids = dcgm_agent.dcgmGroupGetAllIds(handle)
    assert(groupId not in ids), "groupId %d in %s" % (groupId, str(ids))

@test_utils.run_with_embedded_host_engine()
def test_dcgm_group_delete_grp_embedded(handle):
    helper_dcgm_group_delete_grp(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_group_delete_grp_standalone(handle):
    helper_dcgm_group_delete_grp(handle)

