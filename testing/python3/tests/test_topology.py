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

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_all_supported_gpus()
def test_dcgm_topology_device_standalone(handle, gpuIds):
    """
    Verifies that the topology get for the default group works
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuIds = groupObj.GetGpuIds() #Use just the GPUs in our group

    if len(gpuIds) < 2:
        test_utils.skip_test("Needs >= 2 GPUs")

    topologyInfo = systemObj.discovery.GetGpuTopology(gpuIds[0])

    assert (topologyInfo.numGpus == len(gpuIds) - 1), "Expected %d, received numGpus = %d" % (len(gpuIds) - 1, topologyInfo.numGpus)
    assert (topologyInfo.cpuAffinityMask[0] != 0), "GPU 0 should have *some* affinity"

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_all_supported_gpus()
def test_dcgm_topology_group_single_gpu_standalone(handle, gpuIds):
    """
    Verifies that the topology get for a group works for a single GPU
    """
    #Topology will work for a one-GPU group if there are > 1 GPUs on the system
    if len(gpuIds) < 2:
        test_utils.skip_test("Needs >= 2 GPUs")

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Use just the GPUs in our group

    topologyInfo = groupObj.discovery.GetTopology()

    assert (topologyInfo.numaOptimalFlag > 0), "with a single GPU, numa is by default optimal"
    assert (topologyInfo.slowestPath == 0), "with a single GPU, slowest path shouldn't be set"
    
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_all_supported_gpus()
def test_dcgm_topology_device_nvlink_standalone(handle, gpuIds):
    """
    Verifies that the topology get for the default group returns valid NVLINK info
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()
    gpuIds = groupObj.GetGpuIds() #Use just the GPUs in our group

    if len(gpuIds) < 2:
        test_utils.skip_test("Needs >= 2 GPUs")

    topologyInfo = systemObj.discovery.GetGpuTopology(gpuIds[0])

    if topologyInfo.gpuPaths[0].localNvLinkIds == 0:
        test_utils.skip_test("Needs NVLINK support")

    assert ((topologyInfo.gpuPaths[0].path & 0xFFFFFF00) > 0), "No NVLINK state set when localNvLinkIds is > 0"

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

    selectedMask = dcgm_agent.dcgmSelectGpusByTopology(handle, inputList, numGpus, hints)
    sysSelectedMask = discover.SelectGpusByTopology(inputList, numGpus, hints)

    assert (selectedMask.value == inputList), "Expected %s but got %s" % (str(inputList), str(selectedMask))
    assert (sysSelectedMask.value == selectedMask.value)

    if len(gpuIds) > 2:
        numGpus = len(gpuIds) - 1

        # Make sure we don't select a gpu that isn't in the parameters
        for gpuId in gpuIds:
            intputList = inputList & (~gpuBits[gpuId])

            selectedMask = dcgm_agent.dcgmSelectGpusByTopology(handle, inputList, numGpus, hints)
            sysSelectedMask = discover.SelectGpusByTopology(inputList, numGpus, hints)

            assert ((selectedMask.value & inputList) == selectedMask.value), "Selected a GPU outside of the input list"
            assert (sysSelectedMask.value == selectedMask.value)
            intputList = inputList | (gpuBits[gpuId])

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_select_gpus_by_topology_embedded(handle, gpuIds):
    helper_test_select_gpus_by_topology(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_all_supported_gpus()
def test_select_gpus_by_topology_standalone(handle, gpuIds):
    helper_test_select_gpus_by_topology(handle, gpuIds)
