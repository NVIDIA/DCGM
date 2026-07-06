# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import pydcgm
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent
import dcgm_agent_internal
import dcgmvalue
import logger
import test_utils
import dcgm_fields
import time
from ctypes import *
import sys
import os
import pprint
import DcgmSystem
from _test_helpers import skip_test_if_no_dcgm_nvml


HOPPER_GPU_COUNT = 8
HOPPER_NVLINK_COUNT = 18
HOPPER_NVLINK_MASK = (1 << HOPPER_NVLINK_COUNT) - 1
RUBIN_GPU_COUNT = 4
RUBIN_NVLINK_COUNT = 36
DCGM_TOPOLOGY_PATH_NVLINK_MASK = 0xFFFFFFFFFFFFFF00


def helper_dcgm_get_nvlink_link_status_v4(handle):
    linkStatus = dcgm_structs.c_dcgmNvLinkStatus_v4()
    linkStatus.version = dcgm_structs.dcgmNvLinkStatus_version4

    fn = dcgm_agent.dcgmFP("dcgmGetNvLinkLinkStatus")
    ret = fn(handle, byref(linkStatus))
    dcgm_structs._dcgmCheckReturn(ret)

    return linkStatus


def helper_assert_injection_gpu_ids(gpuIds, expectedGpuCount, fixtureName):
    expectedGpuIds = list(range(expectedGpuCount))
    assert sorted(gpuIds) == expectedGpuIds, (
        "%s injection SKU should expose GPU IDs %s, got %s" %
        (fixtureName, expectedGpuIds, sorted(gpuIds)))


def helper_assert_injection_nvlink_status(linkStatus,
                                          gpuIds,
                                          expectedGpuCount,
                                          expectedUpLinkCount,
                                          minLinkSlots,
                                          fixtureName):
    helper_assert_injection_gpu_ids(gpuIds, expectedGpuCount, fixtureName)

    assert linkStatus.numGpus == len(gpuIds), (
        "Expected %d GPUs, got %d" % (len(gpuIds), linkStatus.numGpus))

    seenGpuIds = set()
    for i in range(linkStatus.numGpus):
        gpuStatus = linkStatus.gpus[i]
        seenGpuIds.add(gpuStatus.entityId)

        linkCount = len(gpuStatus.linkState)
        assert linkCount >= minLinkSlots, (
            "Expected at least %d link states for GPU %d, got %d" %
            (minLinkSlots, gpuStatus.entityId, linkCount))

        for linkId in range(min(expectedUpLinkCount, linkCount)):
            assert gpuStatus.linkState[linkId] == dcgm_structs.DcgmNvLinkLinkStateUp, (
                "Expected GPU %d link %d to be Up, got %d" %
                (gpuStatus.entityId, linkId, gpuStatus.linkState[linkId]))

        for linkId in range(expectedUpLinkCount, linkCount):
            assert gpuStatus.linkState[linkId] == dcgm_structs.DcgmNvLinkLinkStateNotSupported, (
                "Expected GPU %d link %d to be NotSupported, got %d" %
                (gpuStatus.entityId, linkId, gpuStatus.linkState[linkId]))

    assert seenGpuIds == set(gpuIds), (
        "Expected GPU IDs %s in link status, got %s" %
        (sorted(gpuIds), sorted(seenGpuIds)))


def helper_assert_hopper_nvlink_status(linkStatus, gpuIds, minLinkSlots):
    helper_assert_injection_nvlink_status(
        linkStatus, gpuIds, HOPPER_GPU_COUNT, HOPPER_NVLINK_COUNT, minLinkSlots, "Hopper")


def helper_assert_rubin_nvlink_status(linkStatus, gpuIds, minLinkSlots):
    helper_assert_injection_nvlink_status(
        linkStatus, gpuIds, RUBIN_GPU_COUNT, RUBIN_NVLINK_COUNT, minLinkSlots, "Rubin")


def helper_nvlink_status_by_gpu(linkStatus):
    return {
        linkStatus.gpus[i].entityId: linkStatus.gpus[i]
        for i in range(linkStatus.numGpus)
    }


def helper_link_ids_from_mask(linkMask):
    return [
        linkId
        for linkId in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU)
        if linkMask & (1 << linkId)
    ]


def helper_dcgm_topology_device_standalone(handle, gpuIds):
    """
    Verifies that the topology get for the default group works
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuIds = groupObj.GetGpuIds()  # Use just the GPUs in our group

    if len(gpuIds) < 2:
        test_utils.skip_test("Needs >= 2 GPUs")

    topologyInfo = systemObj.discovery.GetGpuTopology(gpuIds[0])

    assert (topologyInfo.numGpus == len(gpuIds) -
            1), "Expected %d, received numGpus = %d" % (len(gpuIds) - 1, topologyInfo.numGpus)

    affinity = False

    for bitmapIndex in range(dcgm_structs.DCGM_AFFINITY_BITMASK_ARRAY_SIZE):
        if (topologyInfo.cpuAffinityMask[bitmapIndex] != 0):
            affinity = True
            break

    assert (affinity), "GPU 0 should have *some* affinity"


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_all_supported_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_topology_device_standalone(handle, gpuIds):
    helper_dcgm_topology_device_standalone(handle, gpuIds)


def helper_dcgm_topology_group_single_gpu_standalone(handle, gpuIds):
    """
    Verifies that the topology get for a group works for a single GPU
    """
    # Topology will work for a one-GPU group if there are > 1 GPUs on the
    # system
    if len(gpuIds) < 2:
        test_utils.skip_test("Needs >= 2 GPUs")

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Use just the GPUs in our group

    topologyInfo = groupObj.discovery.GetTopology()

    assert (topologyInfo.numaOptimalFlag >
            0), "with a single GPU, numa is by default optimal"
    assert (topologyInfo.slowestPath ==
            0), "with a single GPU, slowest path shouldn't be set"


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_all_supported_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_topology_group_single_gpu_standalone(handle, gpuIds):
    helper_dcgm_topology_group_single_gpu_standalone(handle, gpuIds)


def helper_dcgm_topology_device_nvlink_standalone(handle, gpuIds):
    """
    Verifies that the topology get for the default group returns valid NVLINK info
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuIds = groupObj.GetGpuIds()  # Use just the GPUs in our group

    if len(gpuIds) < 2:
        test_utils.skip_test("Needs >= 2 GPUs")

    topologyInfo = systemObj.discovery.GetGpuTopology(gpuIds[0])

    if topologyInfo.gpuPaths[0].localNvLinkIds == 0:
        test_utils.skip_test("Needs NVLINK support")

    assert ((topologyInfo.gpuPaths[0].path & 0xFFFFFFFFFFFFFF00) >
            0), "No NVLINK state set when localNvLinkIds is > 0"


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_all_supported_gpus()
def test_dcgm_topology_device_nvlink_standalone(handle, gpuIds):
    helper_dcgm_topology_device_nvlink_standalone(handle, gpuIds)


def helper_test_select_gpus_by_topology(handle, gpuIds):
    '''
    Verifies basic selection of GPUs by topology.
    '''
    handleObj = pydcgm.DcgmHandle(handle=handle)
    discover = DcgmSystem.DcgmSystemDiscovery(handleObj)

    inputList = 0
    gpuBits = {}

    # Create the initial input list
    for gpuId in gpuIds:
        mask = (0x1 << gpuId)
        inputList = inputList | mask
        gpuBits[gpuId] = mask

    # If we ask for all the GPUs then we should get all the GPUs
    numGpus = len(gpuIds)
    # Ignore the health since we don't know if this system is healthy or not
    hints = dcgm_structs.DCGM_TOPO_HINT_F_IGNOREHEALTH

    selectedMask = dcgm_agent.dcgmSelectGpusByTopology(
        handle, inputList, numGpus, hints)
    sysSelectedMask = discover.SelectGpusByTopology(inputList, numGpus, hints)

    assert (selectedMask.value == inputList), "Expected %s but got %s" % (
        str(inputList), str(selectedMask))
    assert (sysSelectedMask.value == selectedMask.value)

    if len(gpuIds) > 2:
        numGpus = len(gpuIds) - 1

        # Make sure we don't select a gpu that isn't in the parameters
        for gpuId in gpuIds:
            intputList = inputList & (~gpuBits[gpuId])

            selectedMask = dcgm_agent.dcgmSelectGpusByTopology(
                handle, inputList, numGpus, hints)
            sysSelectedMask = discover.SelectGpusByTopology(
                inputList, numGpus, hints)

            assert ((selectedMask.value & inputList) ==
                    selectedMask.value), "Selected a GPU outside of the input list"
            assert (sysSelectedMask.value == selectedMask.value)
            intputList = inputList | (gpuBits[gpuId])


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_all_supported_gpus()
@test_utils.run_only_with_nvml()
def test_select_gpus_by_topology_standalone(handle, gpuIds):
    helper_test_select_gpus_by_topology(handle, gpuIds)


def helper_test_dcgm_get_nvlink_link_status_hopper_injection(handle, gpuIds):
    """
    Verify current and legacy NvLink link-status APIs against a deterministic Hopper fixture.
    """
    linkStatusV5 = dcgm_agent.dcgmGetNvLinkLinkStatus(handle)
    helper_assert_hopper_nvlink_status(
        linkStatusV5, gpuIds, dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU)
    assert linkStatusV5.version == dcgm_structs.dcgmNvLinkStatus_version5, (
        "Expected v5 link status, got version %d" % linkStatusV5.version)

    linkStatusV4 = helper_dcgm_get_nvlink_link_status_v4(handle)
    helper_assert_hopper_nvlink_status(
        linkStatusV4, gpuIds, dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU_LEGACY3)
    assert linkStatusV4.version == dcgm_structs.dcgmNvLinkStatus_version4, (
        "Expected v4 link status, got version %d" % linkStatusV4.version)

    statusV5ByGpu = helper_nvlink_status_by_gpu(linkStatusV5)
    statusV4ByGpu = helper_nvlink_status_by_gpu(linkStatusV4)
    assert set(statusV4ByGpu.keys()) == set(statusV5ByGpu.keys()), (
        "Expected v4 GPU IDs %s to match v5 GPU IDs %s" %
        (sorted(statusV4ByGpu.keys()), sorted(statusV5ByGpu.keys())))

    for gpuId in statusV4ByGpu.keys():
        for linkId in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU_LEGACY3):
            assert statusV4ByGpu[gpuId].linkState[linkId] == statusV5ByGpu[gpuId].linkState[linkId], (
                "Mismatch between v4 and v5 for GPU %d link %d" % (gpuId, linkId))


# NO HARDWARE
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku("H200.yaml")
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_get_nvlink_link_status_hopper_injection(handle, gpuIds):
    helper_test_dcgm_get_nvlink_link_status_hopper_injection(handle, gpuIds)


def helper_test_dcgm_get_nvlink_link_status_rubin_injection(handle, gpuIds):
    """
    Verify current and legacy NvLink link-status APIs against a deterministic Rubin fixture.
    """
    linkStatusV5 = dcgm_agent.dcgmGetNvLinkLinkStatus(handle)
    helper_assert_rubin_nvlink_status(
        linkStatusV5, gpuIds, dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU)
    assert linkStatusV5.version == dcgm_structs.dcgmNvLinkStatus_version5, (
        "Expected v5 link status, got version %d" % linkStatusV5.version)

    linkStatusV4 = helper_dcgm_get_nvlink_link_status_v4(handle)
    helper_assert_rubin_nvlink_status(
        linkStatusV4, gpuIds, dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU_LEGACY3)
    assert linkStatusV4.version == dcgm_structs.dcgmNvLinkStatus_version4, (
        "Expected v4 link status, got version %d" % linkStatusV4.version)

    statusV5ByGpu = helper_nvlink_status_by_gpu(linkStatusV5)
    statusV4ByGpu = helper_nvlink_status_by_gpu(linkStatusV4)
    assert set(statusV4ByGpu.keys()) == set(statusV5ByGpu.keys()), (
        "Expected v4 GPU IDs %s to match v5 GPU IDs %s" %
        (sorted(statusV4ByGpu.keys()), sorted(statusV5ByGpu.keys())))

    for gpuId in statusV4ByGpu.keys():
        for linkId in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU_LEGACY3):
            assert statusV4ByGpu[gpuId].linkState[linkId] == statusV5ByGpu[gpuId].linkState[linkId], (
                "Mismatch between v4 and v5 for GPU %d link %d" % (gpuId, linkId))


# NO HARDWARE
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku("Rubin.yaml")
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_get_nvlink_link_status_rubin_injection(handle, gpuIds):
    helper_test_dcgm_get_nvlink_link_status_rubin_injection(handle, gpuIds)


def helper_test_dcgm_get_device_topology_hopper_injection(handle, gpuIds):
    """
    Verify per-GPU topology paths and local NvLink masks against a deterministic Hopper fixture.
    """
    linkStatus = dcgm_agent.dcgmGetNvLinkLinkStatus(handle)
    helper_assert_hopper_nvlink_status(
        linkStatus, gpuIds, dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU)
    linkStatusByGpu = helper_nvlink_status_by_gpu(linkStatus)

    for gpuId in gpuIds:
        topologyInfo = dcgm_agent.dcgmGetDeviceTopology(handle, gpuId)
        assert topologyInfo.version == dcgm_structs.dcgmDeviceTopology_version, (
            "Expected device topology version %d, got %d" %
            (dcgm_structs.dcgmDeviceTopology_version, topologyInfo.version))
        assert topologyInfo.numGpus == len(gpuIds) - 1, (
            "Expected GPU %d to have %d peer paths, got %d" %
            (gpuId, len(gpuIds) - 1, topologyInfo.numGpus))

        peerGpuIds = set()
        for pathIndex in range(topologyInfo.numGpus):
            gpuPath = topologyInfo.gpuPaths[pathIndex]
            peerGpuIds.add(gpuPath.gpuId)

            assert gpuPath.gpuId in gpuIds, (
                "GPU %d topology returned unexpected peer GPU %d" %
                (gpuId, gpuPath.gpuId))
            assert gpuPath.gpuId != gpuId, (
                "GPU %d topology unexpectedly included itself" % gpuId)

            nvlinkPath = gpuPath.path & DCGM_TOPOLOGY_PATH_NVLINK_MASK
            assert nvlinkPath == dcgm_structs.DCGM_TOPOLOGY_NVLINK18, (
                "Expected GPU %d -> GPU %d path to include NVLINK18, got 0x%x" %
                (gpuId, gpuPath.gpuId, gpuPath.path))
            assert gpuPath.localNvLinkIds == HOPPER_NVLINK_MASK, (
                "Expected GPU %d -> GPU %d local NvLink mask 0x%x, got 0x%x" %
                (gpuId, gpuPath.gpuId, HOPPER_NVLINK_MASK, gpuPath.localNvLinkIds))

            localLinkIds = helper_link_ids_from_mask(gpuPath.localNvLinkIds)
            assert localLinkIds == list(range(HOPPER_NVLINK_COUNT)), (
                "Expected GPU %d local link IDs 0-%d, got %s" %
                (gpuId, HOPPER_NVLINK_COUNT - 1, localLinkIds))
            for linkId in localLinkIds:
                assert linkStatusByGpu[gpuId].linkState[linkId] == dcgm_structs.DcgmNvLinkLinkStateUp, (
                    "Topology used GPU %d link %d, but link status was %d" %
                    (gpuId, linkId, linkStatusByGpu[gpuId].linkState[linkId]))

        assert peerGpuIds == set(gpuIds) - {gpuId}, (
            "Expected GPU %d peers %s, got %s" %
            (gpuId, sorted(set(gpuIds) - {gpuId}), sorted(peerGpuIds)))


# NO HARDWARE
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku("H200.yaml")
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_get_device_topology_hopper_injection(handle, gpuIds):
    helper_test_dcgm_get_device_topology_hopper_injection(handle, gpuIds)
