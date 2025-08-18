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
import logger
import pydcgm
import dcgm_agent
import dcgm_structs
import test_utils

@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_live_gpus()
def test_nvlink_p2p_all_status(handle, gpuIds):
    """
    Get nvlink p2p status for each GPU and assert if it's unexpected
    """

    inOutStatus = dcgm_structs.c_dcgmNvLinkP2PStatus_v1()
    inOutStatus.numGpus = 0 # full retrieval.

    dcgm_agent.dcgmGetNvLinkP2PStatus(handle, inOutStatus)

    assert (len(gpuIds) == inOutStatus.numGpus), "Expected %d GPUs, got %d." % (len(gpuIds), inOutStatus.numGpus)

    for i in range(inOutStatus.numGpus):
        assert(inOutStatus.gpus[i].entityId == i), "Expected Entity %d ID %d, got %d." % (i, i, inOutStatus.gpus[i].entityId)
        
        for j in range(inOutStatus.numGpus):
            status = inOutStatus.gpus[i].linkStatus[j]
            logger.info("test_nvlink GPU %d to %d yields %d" % (i, j, status))

            if i == j:
                assert (status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported), "Expected LinkP2PStatus for GPU %d to %d as %d, got %d." % (i, j, dcgm_structs.DcgmNvLinkP2PStatusNotSupported, status)
            else:
                if status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported:
                    test_utils.skip_test("Nvlink p2p status is not supported, skipping test.")
                assert (status == dcgm_structs.DvgmNvLinkP2PStatusOK), "Expected LinkP2PStatus for GPU %d to %d as %d, got %d." % (i, j, dcgm_structs.DvgmNvLinkP2PStatusOK, status)

# This helper is intended to be called for reporting on some, not all GPU Ids.
#
# There was a particular boundary condition when there was only one GPU so
# we have a test specifically for that. (The test was buggy and corrected, not
# the hostengine service.)
#
def helper_nvlink_p2p_specific_status(handle, gpuIds):
    """
    Get nvlink p2p status for specific GPUs and assert if it's unexpected
    """

    inOutStatus = dcgm_structs.c_dcgmNvLinkP2PStatus_v1()
    inOutStatus.numGpus = (len(gpuIds) + 1 ) // 2 # partial retrieval.

    for i in range(inOutStatus.numGpus):
        inOutStatus.gpus[i].entityId = inOutStatus.numGpus - 1 - i # reverse

    dcgm_agent.dcgmGetNvLinkP2PStatus(handle, inOutStatus)

    assert (((len(gpuIds) + 1) // 2) == inOutStatus.numGpus), "Expected %d GPUs, got %d." % ((len(gpuIds) + 1) // 2, inOutStatus.numGpus)

    for i in range(inOutStatus.numGpus):
        gpu = inOutStatus.numGpus - 1 - i
        assert(inOutStatus.gpus[i].entityId == gpu), "Expected Entity %d ID %d, got %d." % (i, gpu, inOutStatus.gpus[i].entityId)
        
        for j in range(len(gpuIds)):
            status = inOutStatus.gpus[i].linkStatus[j]
            logger.info("test_nvlink GPU %d to %d yields %d" % (gpu, j, status))

            if gpu == j:
                assert (status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported), "Expected LinkP2PStatus for GPU %d to %d as %d, got %d." % (gpu, j, dcgm_structs.DcgmNvLinkP2PStatusNotSupported, status)
            else:
                if status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported:
                    test_utils.skip_test("Nvlink p2p status is not supported, skipping test.")
                assert (status == dcgm_structs.DvgmNvLinkP2PStatusOK), "Expected LinkP2PStatus for GPU %d to %d as %d, got %d." % (gpu, j, dcgm_structs.DvgmNvLinkP2PStatusOK, status)

# Test for half the GPUs.
@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_live_gpus()
def test_nvlink_p2p_specific_status(handle, gpuIds):
    helper_nvlink_p2p_specific_status(handle, gpuIds)

# Test for single GPU.
@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_live_gpus()
def test_nvlink_p2p_single_status(handle, gpuIds):
    gpuIds = [ gpuIds[0] ]
    helper_nvlink_p2p_specific_status(handle, gpuIds)
    
