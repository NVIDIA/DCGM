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
import logger
import pydcgm
import dcgm_agent
import dcgm_structs
import test_utils
import nvidia_smi_utils


def helper_nvlink_p2p_all_status(handle, gpuIds):
    """
    Get nvlink p2p status for each GPU and assert if it's unexpected
    """

    inOutStatus = dcgm_structs.c_dcgmNvLinkP2PStatus_v1()
    inOutStatus.numGpus = 0  # full retrieval.

    dcgm_agent.dcgmGetNvLinkP2PStatus(handle, inOutStatus)

    assert (len(gpuIds) == inOutStatus.numGpus), "Expected %d GPUs, got %d." % (
        len(gpuIds), inOutStatus.numGpus)

    for i in range(inOutStatus.numGpus):
        assert (inOutStatus.gpus[i].entityId == i), "Expected Entity %d ID %d, got %d." % (
            i, i, inOutStatus.gpus[i].entityId)

        for j in range(inOutStatus.numGpus):
            status = inOutStatus.gpus[i].linkStatus[j]
            logger.info("test_nvlink GPU %d to %d yields %d" % (i, j, status))

            if i == j:
                assert (status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported), "Expected LinkP2PStatus for GPU %d to %d as %d, got %d." % (
                    i, j, dcgm_structs.DcgmNvLinkP2PStatusNotSupported, status)
            else:
                if status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported:
                    test_utils.skip_test(
                        "Nvlink p2p status is not supported, skipping test.")
                assert (status == dcgm_structs.DcgmNvLinkP2PStatusOK), "Expected LinkP2PStatus for GPU %d to %d as %d, got %d." % (
                    i, j, dcgm_structs.DcgmNvLinkP2PStatusOK, status)

# This helper is intended to be called for reporting on some, not all GPU Ids.
#
# There was a particular boundary condition when there was only one GPU so
# we have a test specifically for that. (The test was buggy and corrected, not
# the hostengine service.)
#


@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_live_gpus()
def test_nvlink_p2p_all_status(handle, gpuIds):
    helper_nvlink_p2p_all_status(handle, gpuIds)


def helper_nvlink_p2p_specific_status(handle, gpuIds):
    """
    Get nvlink p2p status for specific GPUs and assert if it's unexpected
    """

    inOutStatus = dcgm_structs.c_dcgmNvLinkP2PStatus_v1()
    inOutStatus.numGpus = (len(gpuIds) + 1) // 2  # partial retrieval.

    for i in range(inOutStatus.numGpus):
        inOutStatus.gpus[i].entityId = inOutStatus.numGpus - 1 - i  # reverse

    dcgm_agent.dcgmGetNvLinkP2PStatus(handle, inOutStatus)

    assert (((len(gpuIds) + 1) // 2) == inOutStatus.numGpus), "Expected %d GPUs, got %d." % (
        (len(gpuIds) + 1) // 2, inOutStatus.numGpus)

    for i in range(inOutStatus.numGpus):
        gpu = inOutStatus.numGpus - 1 - i
        assert (inOutStatus.gpus[i].entityId == gpu), "Expected Entity %d ID %d, got %d." % (
            i, gpu, inOutStatus.gpus[i].entityId)

        for j in range(len(gpuIds)):
            status = inOutStatus.gpus[i].linkStatus[j]
            logger.info("test_nvlink GPU %d to %d yields %d" %
                        (gpu, j, status))

            if gpu == j:
                assert (status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported), "Expected LinkP2PStatus for GPU %d to %d as %d, got %d." % (
                    gpu, j, dcgm_structs.DcgmNvLinkP2PStatusNotSupported, status)
            else:
                if status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported:
                    test_utils.skip_test(
                        "Nvlink p2p status is not supported, skipping test.")
                assert (status == dcgm_structs.DcgmNvLinkP2PStatusOK), "Expected LinkP2PStatus for GPU %d to %d as %d, got %d." % (
                    gpu, j, dcgm_structs.DcgmNvLinkP2PStatusOK, status)

# Test for half the GPUs.


@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_live_gpus()
def test_nvlink_p2p_specific_status(handle, gpuIds):
    helper_nvlink_p2p_specific_status(handle, gpuIds)

# Test for single GPU.


def helper_nvlink_p2p_single_status(handle, gpuIds):
    gpuIds = [gpuIds[0]]
    helper_nvlink_p2p_specific_status(handle, gpuIds)


@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_live_gpus()
def test_nvlink_p2p_single_status(handle, gpuIds):
    helper_nvlink_p2p_single_status(handle, gpuIds)


def _get_dcgm_nvlink_up_counts(dcgmSystem):
    """
    Returns {gpuId: up_link_count} from the DCGM NvLink status API.

    Counts links in the Up state. This is the same source `dcgmi nvlink -s`
    renders as 'U', via dcgmGetNvLinkLinkStatus.
    """
    linkStatus = dcgmSystem.discovery.GetNvLinkLinkStatus()
    counts = {}
    for i in range(linkStatus.numGpus):
        gpu = linkStatus.gpus[i]
        upCount = 0
        for j in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU):
            if gpu.linkState[j] == dcgm_structs.DcgmNvLinkLinkStateUp:
                upCount += 1
        counts[gpu.entityId] = upCount
    return counts


def helper_nvlink_count_matches_nvidia_smi(handle, gpuIds):
    """
    Verify the active NVLink count per GPU matches between nvidia-smi and DCGM.
    """
    smiCounts = nvidia_smi_utils.get_nvlink_active_counts_by_uuid()
    if not smiCounts:
        test_utils.skip_test(
            "nvidia-smi NVLink information is unavailable, skipping test.")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmCounts = _get_dcgm_nvlink_up_counts(dcgmSystem)

    for gpuId in gpuIds:
        uuid = dcgmSystem.discovery.GetGpuAttributes(gpuId).identifiers.uuid
        assert uuid in smiCounts, \
            "No nvidia-smi NVLink data for GPU %d (%s)." % (gpuId, uuid)
        assert gpuId in dcgmCounts, \
            "No DCGM NVLink data for GPU %d." % gpuId

        smiCount = smiCounts[uuid]
        dcgmCount = dcgmCounts[gpuId]
        logger.info("test_nvlink GPU %d (%s): nvidia-smi=%d, dcgm=%d active NVLinks" % (
            gpuId, uuid, smiCount, dcgmCount))
        assert smiCount == dcgmCount, \
            "NVLink count mismatch for GPU %d (%s): nvidia-smi=%d, dcgm=%d." % (
                gpuId, uuid, smiCount, dcgmCount)


@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
def test_nvlink_count_matches_nvidia_smi(handle, gpuIds):
    helper_nvlink_count_matches_nvidia_smi(handle, gpuIds)
