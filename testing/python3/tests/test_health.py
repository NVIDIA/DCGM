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
# test the health module for DCGM

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
import dcgm_internal_helpers
import dcgm_field_injection_helpers
import dcgm_errors
import dcgm_nvml
import nvml_injection
import nvml_injection_structs
import dcgm_nvml
from _test_helpers import skip_test_if_no_dcgm_nvml, maybe_dcgm_nvml
import random
import math


def skip_test_if_unhealthy(groupObj):
    # Skip the test if the GPU is already failing health checks
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    if responseV5.overallHealth != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        msg = "Skipping health check test because we are already unhealthy: "
        for i in range(0, responseV5.incidentCount):
            if i == 0:
                msg += "%s" % responseV5.incidents[i].error.msg
            else:
                msg += ", %s" % responseV5.incidents[i].error.msg

        test_utils.skip_test(msg)


def helper_dcgm_health_set_pcie(handle):
    """
    Verifies that the set/get path for the health monitor is working
    Checks for call errors are done in the bindings
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    groupObj.health.Set(0)

    currentSystems = groupObj.health.Get()
    assert (currentSystems == 0)

    newSystems = currentSystems | dcgm_structs.DCGM_HEALTH_WATCH_PCIE

    groupObj.health.Set(newSystems)

    currentSystems = groupObj.health.Get()
    assert (currentSystems == newSystems)

    # Set it back to 0 and validate it
    groupObj.health.Set(0)
    currentSystems = groupObj.health.Get()
    assert (currentSystems == 0)


@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_health_set_pcie_standalone(handle):
    helper_dcgm_health_set_pcie(handle)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_health_invalid_group_embedded(handle):
    '''
    Validate that group operations fail if a bogus group ID is provided
    '''
    invalidGroupId = c_void_p(99)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = pydcgm.DcgmGroup(handleObj, groupId=invalidGroupId)

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupObj.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupObj.health.Get()

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)


def helper_dcgm_health_check_pcie(handle, gpuIds, pcieGen, pcieLanes, pcieReplayCounter, expectingPcieIncident, errmsg):
    """
    Verifies that a check error occurs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    # inject PCIe Gen and width/lanes
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_LINK_GEN,
                                                              pcieGen, 0)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_LINK_WIDTH,
                                                              pcieLanes, 0)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    response = groupObj.health.Check()
    # we expect that there will be no data here

    # inject an error into PCI
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                                              pcieReplayCounter, 100)  # set the injected data into the future
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    if expectingPcieIncident:
        assert (responseV5.incidentCount == 1), errmsg
        assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
        assert (responseV5.incidents[0].system ==
                dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
        assert (responseV5.incidents[0].error.code ==
                dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)
    else:
        assert (responseV5.incidentCount == 0), errmsg


def helper_reset_pcie_replay_counter(handle, gpuIds):
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                                              0, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_check_pcie_standalone(handle, gpuIds):
    # PCIe replay rate thresholds for each generation per lane.
    pcieGenReplayRatesPerLane = [
        # Gen1 speed = 2.5 Gbps, (1x10^-12) * (2.5x10^9) * 60 = 0.15 errors/min per lane.
        2.5 / 1000 * 60,
        # Gen2 speed = 5 Gbps, (1x10^-12) * (5x10^9) * 60 = 0.3 errors/min per lane.
        5.0 / 1000 * 60,
        # Gen3 speed = 8 Gbps, (1x10^-12) * (8x10^9) * 60 = 0.48 errors/min per lane.
        8.0 / 1000 * 60,
        # Gen4 speed = 16 Gbps, (1x10^-12) * (16x10^9) * 60 = 0.96 errors/min per lane.
        16.0 / 1000 * 60,
        # Gen5 speed = 32 Gbps, (1x10^-12) * (32x10^9) * 60 = 1.92 errors/min per lane.
        32.0 / 1000 * 60,
        # Gen6 speed = 64 Gbps, (1x10^-12) * (64x10^9) * 60 = 3.84 errors/min per lane.
        64.0 / 1000 * 60
    ]

    # Run it multiple times to cover more combinations of pcieGen and pcieLanes.
    for i in range(3):
        pcieGen = 1
        # For each Gen, randomly select number of lanes and PCIe replay counter to inject to dcgm.
        # If replay counter rate > expected rate, then make sure PCIe incident is present, no incident otherwise.
        for pcieGenReplayRatePerLane in pcieGenReplayRatesPerLane:
            pcieLanes = random.randint(1, 16)
            expectedPcieReplayCounterLimit = math.ceil(
                pcieGenReplayRatePerLane * pcieLanes)
            # Multiply by 2 to have uniform probability for pcieReplayCounter to give success or failure scenario.
            pcieReplayCounter = random.randint(
                1, 2 * expectedPcieReplayCounterLimit)
            expectingPcieIncident = True if pcieReplayCounter > expectedPcieReplayCounterLimit else False
            errmsg = (f"pcieGen={pcieGen} pcieGenReplayRatePerLane={pcieGenReplayRatePerLane} pcieLanes={pcieLanes} "
                      f"expectedPcieReplayCounterLimit={expectedPcieReplayCounterLimit} pcieReplayCounter={pcieReplayCounter} "
                      f"expectingPcieIncident={expectingPcieIncident}")
            helper_dcgm_health_check_pcie(
                handle, gpuIds, pcieGen, pcieLanes, pcieReplayCounter, expectingPcieIncident, errmsg)
            # Reset after testing each failure test case i.e. if expecting PCIe incident.
            if expectingPcieIncident:
                helper_reset_pcie_replay_counter(handle, gpuIds)
            pcieGen += 1


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_health_check_pcie_embedded_using_nvml_injection(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    injectedRetsArray = nvml_injection.c_injectNvmlRet_t * 4
    injectedRets = injectedRetsArray()
    injectedRets[0].nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRets[0].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRets[0].values[0].value.UInt = 40
    injectedRets[0].valueCount = 1
    injectedRets[1].nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRets[1].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRets[1].values[0].value.UInt = 50
    injectedRets[1].valueCount = 1
    injectedRets[2].nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRets[2].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRets[2].values[0].value.UInt = 60
    injectedRets[2].valueCount = 1
    injectedRets[3].nvmlRet = maybe_dcgm_nvml.NVML_SUCCESS
    injectedRets[3].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    injectedRets[3].values[0].value.UInt = 70
    injectedRets[3].valueCount = 1

    ret = dcgm_agent_internal.dcgmInjectNvmlDeviceForFollowingCalls(
        handle, gpuId, "PcieReplayCounter", None, 0, injectedRets, 4)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    # 4 values * 250ms = 1 second to get a hit
    groupObj.health.Set(newSystems, 250000, 3600.0)

    skip_test_if_unhealthy(groupObj)

    response = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    # we expect that there will be no data here
    assert (response.incidentCount == 0)

    # Wait for cache refresh
    for _ in range(10000):
        responseV5 = groupObj.health.Check(
            dcgm_structs.dcgmHealthResponse_version5)
        if responseV5.incidentCount == 1:
            break
        time.sleep(0.001)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)


def helper_test_dcgm_health_check_mem_dbe(handle, gpuIds):
    """
    Verifies that the health check will fail if there's 1 DBE and it continues to be
    reported
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                              2, -50)  # set the injected data to 50 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)

    # Give it the same failure 45 seconds ago and make sure we fail again
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                              2, -45)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)

    # Make the failure count go down to zero. This should clear the error
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                              0, -40)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_check_mem_dbe(handle, gpuIds):
    helper_test_dcgm_health_check_mem_dbe(handle, gpuIds)


def helper_verify_dcgm_health_watch_mem_result(groupObj, errorCode, verifyFail=False, gpuId=0):
    """
    Verify that memory health check result is what was expected. If verifyFail is False, verify a pass result,
    otherwise verify a failure occurred.
    """
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    if not verifyFail:
        assert (responseV5.overallHealth ==
                dcgm_structs.DCGM_HEALTH_RESULT_PASS)
        return

    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_FAIL)


def helper_reset_page_retirements(handle, gpuId=0, reset_sbe=False):
    """
    Helper function to reset non volatile page retirements.
    """
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                                                              0, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)
    if reset_sbe:
        ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
                                                                  0, -30)  # set the injected data to 30 seconds ago
        assert (ret == dcgm_structs.DCGM_ST_OK)


def helper_test_dcgm_health_check_mem_retirements(handle, gpuIds):
    """
    Verifies that the health check will fail when the number of non-volatile page retirements
    match the failure criteria.

    Specifically tests the criteria given in DCGM-458.
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ####### Tests #######
    #### Condition 1 ####
    # Fail if the total number of page retirements (due to DBE or SBE) meets or exceeds 60
    # Test 1: >= 60 page retirements total should fail
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                                                              30, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
                                                              30, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we get a failure
    helper_verify_dcgm_health_watch_mem_result(groupObj, dcgm_errors.DCGM_FR_RETIRED_PAGES_LIMIT, verifyFail=True,
                                               gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    # Test 2: 59 page retirements total should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                                                              10, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
                                                              49, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    #### Condition 2 ####
    # Fail if > 15 page retirement due to DBEs AND more than 1 DBE page retirement in past week
    # Test 1: 15 page retirements due to DBEs should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                                                              15, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    # Test 2: 16 page retirements due to DBE should fail (since all 16 are inserted in current week)
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                                                              16, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we get a failure
    helper_verify_dcgm_health_watch_mem_result(groupObj, dcgm_errors.DCGM_FR_RETIRED_PAGES_DBE_LIMIT,
                                               verifyFail=True, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    # Test 3: 16 page retirements due to SBEs should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
                                                              16, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    # Test 4: 16 page retirements due to DBEs (with first 15 pages inserted more than 1 week ago,
    # and 16th page inserted in current week) should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                                                              15, -604860)  # set the injected data to 7 days and 1 minute ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                                                              1, -30)  # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_check_mem_retirements_standalone(handle, gpuIds):
    helper_test_dcgm_health_check_mem_retirements(handle, gpuIds)


def helper_test_dcgm_health_check_mem(handle, gpuIds):
    """
    Verifies that a check error occurs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
                                                              100, -40)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount ==
            1), "Expected 1 incident but found %d" % responseV5.incidentCount
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_PENDING_PAGE_RETIREMENTS), \
        "Expected %d but found %d" % (dcgm_errors.DCGM_FR_PENDING_PAGE_RETIREMENTS,
                                      responseV5.incidents[0].error.code)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN), \
        "Expected warning but found %d" % responseV5.incidents[0].health

    # Clear the error
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
                                                              0, -35)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we've set the monitor frequency to less than 35 seconds - that will make us around
    # half or less of the 60 seconds we give the data before calling it stale.
    cmFieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(
        handle, gpuId, dcgm_fields.DCGM_FE_GPU, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING)
    assert cmFieldInfo.monitorIntervalUsec < 35000000


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_check_mem_standalone(handle, gpuIds):
    helper_test_dcgm_health_check_mem(handle, gpuIds)


@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_standalone_health_set_thermal(handle):
    """
    Verifies that the set/get path for the health monitor is working
    Checks for call errors are done in the bindings
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    groupObj.health.Set(0)

    currentSystems = groupObj.health.Get()
    assert (currentSystems == 0)

    newSystems = currentSystems | dcgm_structs.DCGM_HEALTH_WATCH_THERMAL

    groupObj.health.Set(newSystems)

    currentSystems = groupObj.health.Get()
    assert (currentSystems == newSystems)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_standalone_health_check_thermal(handle, gpuIds):
    """
    Verifies that a check error occurs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_THERMAL
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuIds[0],
                                                              dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, 0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    # we expect that there will be no data here
    # assert (dcgm_structs.DCGM_ST_OK == result or dcgm_structs.DCGM_ST_NO_DATA == result)

    # inject an error into thermal
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuIds[0],
                                                              dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, 1000, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuIds[0])
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_CLOCK_THROTTLE_THERMAL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_CLOCKS_EVENT_THERMAL)


@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_standalone_health_set_power(handle):
    """
    Verifies that the set/get path for the health monitor is working
    Checks for call errors are done in the bindings
    """

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    groupObj.health.Set(0)

    currentSystems = groupObj.health.Get()
    assert (currentSystems == 0)

    newSystems = currentSystems | dcgm_structs.DCGM_HEALTH_WATCH_POWER

    groupObj.health.Set(newSystems)

    currentSystems = groupObj.health.Get()
    assert (currentSystems == newSystems)


def helper_check_health_response_v4(gpuIds, response):
    numErrors = 0
    if response.version == 0:
        numErrors += 1
        logger.error("bad response.version x%X" % response.version)
    if response.overallHealth != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        numErrors += 1
        logger.error(
            "bad response.overallHealth %d. Are these GPUs really healthy?" % response.overallHealth)
        test_utils.skip_test(
            "bad response.overallHealth %d. Are these GPUs really healthy?" % response.overallHealth)
    if response.incidentCount > 0:
        numErrors += 1
        logger.error("bad response.incidentCount %d > 0" %
                     (response.incidentCount))

    assert numErrors == 0, "Errors were encountered. See above."


def helper_run_dcgm_health_check_sanity(handle, gpuIds, system_to_check):
    """
    Verifies that the DCGM health checks return healthy for all GPUs on live systems.
    """

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    groupObj = systemObj.GetGroupWithGpuIds('testgroup', gpuIds)
    groupObj.health.Set(system_to_check)

    systemObj.UpdateAllFields(1)

    # This will throw an exception on error
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    # Check that our response comes back clean
    helper_check_health_response_v4(gpuIds, responseV5)

# Start health sanity checks
# The health sanity checks verify that that the DCGM health checks return healthy for all GPUs on live systems.
# Note: These tests can fail if a GPU is really unhealthy. We should give detailed feedback so that this is attributed
# to the GPU and not the test


@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_pcie_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(
        handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_PCIE)


@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_mem_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(
        handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_MEM)


@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_inforom_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(
        handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_INFOROM)


@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_thermal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(
        handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)


@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_power_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(
        handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_POWER)


@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_nvlink_standalone(handle, gpuIds):
    # We will get false failures if any nvlinks are down on the GPUs
    test_utils.skip_test_if_any_nvlinks_down(handle)
    helper_run_dcgm_health_check_sanity(
        handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)


@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_nvswitch_nonfatal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(
        handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_NONFATAL)


@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_nvswitch_fatal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(
        handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_FATAL)

# End health sanity checks


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_standalone_health_check_power(handle, gpuIds):
    """
    Verifies that a check error occurs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_POWER
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    skip_test_if_unhealthy(groupObj)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    # we expect that there will be no data here

    # inject an error into power
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
                                                              1000, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuIds[0])
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_CLOCK_THROTTLE_POWER)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_CLOCKS_EVENT_POWER)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_standalone_health_check_nvlink(handle, gpuIds):
    helper_health_check_nvlink_error_counters(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_dcgm_embedded_health_check_nvlink(handle, gpuIds):
    helper_health_check_nvlink_error_counters(handle, gpuIds)


@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_standalone_health_set_nvlink(handle):
    """
    Verifies that the set/get path for the health monitor is working
    Checks for call errors are done in the bindings
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    groupObj.health.Set(0)

    currentSystems = groupObj.health.Get()
    assert (currentSystems == 0)

    newSystems = currentSystems | dcgm_structs.DCGM_HEALTH_WATCH_NVLINK

    groupObj.health.Set(newSystems)

    currentSystems = groupObj.health.Get()
    assert (currentSystems == newSystems)


def helper_health_check_nvlink_error_counters(handle, gpuIds):
    """
    Verifies that a check error occurs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    # We will get false failures if any nvlinks are down on the GPUs
    test_utils.skip_test_if_any_nvlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Capture baseline incident count
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineIncidentCount = responseV5.incidentCount

    # inject an error into NV Link
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                              100, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    # Verify exactly one new incident
    new_incidents = max(0, responseV5.incidentCount - baselineIncidentCount)
    assert new_incidents == 1, f"Expected 1 new incident, got {new_incidents}"

    # Verify the new incident details
    incident = responseV5.incidents[baselineIncidentCount]
    assert incident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU
    assert incident.entityInfo.entityId == gpuId
    assert incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    assert incident.error.code == dcgm_errors.DCGM_FR_NVLINK_ERROR_THRESHOLD
    assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_WARN


def helper_nvlink_check_fatal_errors(handle, gpuIds):
    test_utils.skip_test_if_any_nvlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Capture baseline incident count
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineIncidentCount = responseV5.incidentCount

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                                              1, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

    # Verify exactly one new incident
    new_incidents = max(0, responseV5.incidentCount - baselineIncidentCount)
    assert new_incidents == 1, f"Expected 1 new incident, got {new_incidents}"

    # Verify the new incident details
    incident = responseV5.incidents[baselineIncidentCount]
    assert incident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU
    assert incident.entityInfo.entityId == gpuId
    assert incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    assert incident.error.code == dcgm_errors.DCGM_FR_NVLINK_ERROR_CRITICAL
    assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_standalone_nvlink_fatal(handle, gpuIds):
    helper_nvlink_check_fatal_errors(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_dcgm_embedded_nvlink_fatal(handle, gpuIds):
    helper_nvlink_check_fatal_errors(handle, gpuIds)


def helper_nvlink_crc_fatal_threshold(handle, gpuIds):
    test_utils.skip_test_if_any_nvlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Capture baseline incident count
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineIncidentCount = responseV5.incidentCount

    # Trigger a failure by having more than 100 CRC errors per second
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                              1000000, -20)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

    # Verify exactly one new incident
    new_incidents = max(0, responseV5.incidentCount - baselineIncidentCount)
    assert new_incidents == 1, f"Expected 1 new incident, got {new_incidents}"

    # Verify the new incident details
    incident = responseV5.incidents[baselineIncidentCount]
    assert incident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU
    assert incident.entityInfo.entityId == gpuId
    assert incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    assert incident.error.code == dcgm_errors.DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD
    assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_standalone_nvlink_crc_threshold(handle, gpuIds):
    helper_nvlink_crc_fatal_threshold(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_dcgm_embedded_nvlink_crc_threshold(handle, gpuIds):
    helper_nvlink_crc_fatal_threshold(handle, gpuIds)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_nvlink_symbol_threshold(handle, gpuIds):
    test_utils.skip_test_if_any_nvlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Capture baseline incident count
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineIncidentCount = responseV5.incidentCount

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS,
                                                              1, -20)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

    # Verify exactly one new incident
    new_incidents = max(0, responseV5.incidentCount - baselineIncidentCount)
    assert new_incidents == 1, f"Expected 1 new incident, got {new_incidents}"

    # Verify the new incident details
    incident = responseV5.incidents[baselineIncidentCount]
    assert incident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU
    assert incident.entityInfo.entityId == gpuId
    assert incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    assert incident.error.code == dcgm_errors.DCGM_FR_NVLINK_ERROR_CRITICAL
    assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_nvlink_effective_ber_threshold(handle, gpuIds):
    test_utils.skip_test_if_any_nvlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER,
                                                              4095, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Capture baseline incident count
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineIncidentCount = responseV5.incidentCount

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                              dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER,
                                                              1000, -20)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

    # Verify exactly one new incident
    new_incidents = max(0, responseV5.incidentCount - baselineIncidentCount)
    assert new_incidents == 1, f"Expected 1 new incident, got {new_incidents}"

    # Verify the new incident details
    incident = responseV5.incidents[baselineIncidentCount]
    assert incident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU
    assert incident.entityInfo.entityId == gpuId
    assert incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    assert incident.error.code == dcgm_errors.DCGM_FR_NVLINK_EFFECTIVE_BER_THRESHOLD
    assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL


def helper_health_check_nvlink5_error_counters(handle, gpuIds):
    """
    Test for NVLink 5 error counter health checks.
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    test_fields = [
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_MALFORMED_PACKET_ERRORS,
         "malformed packet errors"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_BUFFER_OVERRUN_ERRORS,
         "buffer overrun errors"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_ERRORS, "rx errors"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_REMOTE_ERRORS, "rx remote errors"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_GENERAL_ERRORS, "rx general errors"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS,
         "local link integrity errors"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS,
         "link recovery events"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS, "effective errors"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER, "symbol BER"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_TOTAL, "ECC data errors"),
    ]

    for field_id, field_name in test_fields:
        # Inject baseline value (healthy state)
        # BER fields are encoded: 4095 = mantissa 15, exponent 255 (1.5e-254, healthy)
        # Error counter fields use 0 as healthy
        healthy_value = 4095 if field_id == dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER else 0
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, field_id, healthy_value, -50)
        assert ret == dcgm_structs.DCGM_ST_OK, f"Failed to inject baseline for {field_name}"

        # Get baseline health check - should pass
        responseV5 = groupObj.health.Check(
            dcgm_structs.dcgmHealthResponse_version5)
        baseline_incident_count = responseV5.incidentCount

        # Inject error value exceeding threshold
        # BER fields: any value != 4095 (unhealthy BER)
        # Error counter fields: 1 or more errors
        error_value = 1000 if field_id == dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER else 1
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, field_id, error_value, -20)
        assert ret == dcgm_structs.DCGM_ST_OK, f"Failed to inject error for {field_name}"

        # Verify health check fails
        responseV5 = groupObj.health.Check(
            dcgm_structs.dcgmHealthResponse_version5)

        new_incidents = max(0, responseV5.incidentCount -
                            baseline_incident_count)
        assert new_incidents == 1, f"Expected 1 new incident for {field_name}, got {new_incidents}"

        # Verify incident details
        incident = responseV5.incidents[baseline_incident_count]
        assert incident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU, f"Wrong entity group for {field_name}"
        assert incident.entityInfo.entityId == gpuId, f"Wrong entity ID for {field_name}"
        assert incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK, f"Wrong system for {field_name}"
        # BER fields use different error code than error counters
        expected_error_code = dcgm_errors.DCGM_FR_NVLINK_SYMBOL_BER_THRESHOLD if field_id == dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER else dcgm_errors.DCGM_FR_NVLINK_ERROR_CRITICAL
        assert incident.error.code == expected_error_code, f"Wrong error code for {field_name}"
        assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL, f"Wrong health result for {field_name}"

        # Cleanup: Reset field to healthy value at a more recent timestamp
        healthy_value = 4095 if field_id == dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER else 0
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, field_id, healthy_value, 0)
        assert ret == dcgm_structs.DCGM_ST_OK, f"Failed to cleanup {field_name}"

        logger.debug(f"Health check test passed for {field_name}")

    # Test multiple simultaneous errors
    test_fields = [
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_ERRORS, "rx errors"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS,
         "link recovery events"),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER, "symbol BER"),
    ]

    # Clear test fields with healthy values at a future timestamp
    for field_id, field_name in test_fields:
        healthy_value = 4095 if field_id == dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER else 0
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, field_id, healthy_value, 10)
        assert ret == dcgm_structs.DCGM_ST_OK, f"Failed to clear {field_name}"

    # Get baseline health check - should pass
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineIncidentCount = responseV5.incidentCount

    # Inject error values exceeding threshold at an even more recent timestamp
    for field_id, field_name in test_fields:
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, field_id, 1000, 20)
        assert ret == dcgm_structs.DCGM_ST_OK, f"Failed to inject error for {field_name}"

    # Verify health check fails
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    # Let newIncidents be the incidents of this health system, that are not already in the baseline [baseline_incident_count:]
    newIncidents = [incident for incident in responseV5.incidents[baselineIncidentCount:]
                    if incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK]
    assert len(newIncidents) == len(
        test_fields), f"Expected {len(test_fields)} new incidents, got {len(newIncidents)}"

    # Verify incident details
    for incident in newIncidents:
        assert incident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU, f"Wrong entity group for {incident.system}"
        assert incident.entityInfo.entityId == gpuId, f"Wrong entity ID for {incident.system}"
        assert incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK, f"Wrong system for {incident.system}"
        assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL, f"Wrong health result for {incident.system}"
        assert incident.error.code in [dcgm_errors.DCGM_FR_NVLINK_SYMBOL_BER_THRESHOLD, dcgm_errors.DCGM_FR_NVLINK_ERROR_CRITICAL], \
            f"Unexpected error code {incident.error.code}"

    # Clean up - restore fields to healthy values at the most recent timestamp
    for field_id, field_name in test_fields:
        # BER fields are encoded: 4095 = mantissa 15, exponent 255 (1.5e-254, healthy)
        # Error counter fields: 0 = healthy
        healthy_value = 4095 if field_id == dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER else 0
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, field_id, healthy_value, 30)
        assert ret == dcgm_structs.DCGM_ST_OK, f"Failed to cleanup {field_name}"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_standalone_nvlink5_error_counters(handle, gpuIds):
    helper_health_check_nvlink5_error_counters(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_dcgm_embedded_nvlink5_error_counters(handle, gpuIds):
    helper_health_check_nvlink5_error_counters(handle, gpuIds)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_standalone_health_large_groupid(handle, gpuIds):
    """
    Verifies that a health check can run on a large groupId
    This verifies the fix for bug 1868821
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    # Make a bunch of groups and delete them right away so our next groupId is large
    for i in range(100):
        groupObj = systemObj.GetEmptyGroup("test_group_%d" % i)
        groupObj.Delete()

    groupObj = systemObj.GetEmptyGroup("test_good_group")
    groupObj.AddGpu(gpuIds[0])
    groupId = groupObj.GetId().value

    assert groupId >= 100, "Expected groupId > 100. got %d" % groupObj.GetId()

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_ALL

    # Any of these will throw an exception on error. Making it past these = success
    groupObj.health.Set(newSystems)
    systemObj.UpdateAllFields(True)
    groupObj.health.Get()
    groupObj.health.Check()


def helper_health_check_nvswitch_errors(handle, switchIds, fieldId, healthSystem, healthResult, errorCode):
    """
    Verifies that a check error occurs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    # This test will fail if any NvLinks are down
    test_utils.skip_test_if_any_nvlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    switchId = switchIds[0]
    groupObj.AddEntity(dcgm_fields.DCGM_FE_SWITCH, switchId)

    newSystems = healthSystem
    groupObj.health.Set(newSystems)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = fieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time() - 5) * 1000000.0)
    field.value.i64 = 0

    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH,
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    field.ts = int((time.time() - 50) * 1000000.0)
    field.value.i64 = 0

    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH,
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # we expect that there will be no data here
    # inject an error into NvSwitch
    # set the injected data for a second ago
    field.ts = int((time.time() - 1) * 1000000.0)
    field.value.i64 = 5

    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH,
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_SWITCH)
    assert (responseV5.incidents[0].entityInfo.entityId == switchId)
    assert (responseV5.incidents[0].health == healthResult)
    assert (responseV5.incidents[0].system == healthSystem)
    assert (responseV5.incidents[0].error.code == errorCode)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_nvswitches()
def test_health_check_nvswitch_fatal_errors_standalone(handle, switchIds):
    helper_health_check_nvswitch_errors(handle, switchIds,
                                        dcgm_fields.DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS,
                                        dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_FATAL,
                                        dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
                                        dcgm_errors.DCGM_FR_NVSWITCH_FATAL_ERROR)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_nvswitches()
def test_health_check_nvswitch_nonfatal_errors_standalone(handle, switchIds):
    helper_health_check_nvswitch_errors(handle, switchIds,
                                        dcgm_fields.DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS,
                                        dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_NONFATAL,
                                        dcgm_structs.DCGM_HEALTH_RESULT_WARN,
                                        dcgm_errors.DCGM_FR_NVSWITCH_NON_FATAL_ERROR)


def helper_health_check_nvlink_link_down_gpu(handle, gpuIds):
    """
    Verifies that a check error occurs when a NvLink link is set to broken
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    # Set all links of our injected GPU to Up
    for linkId in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU):
        dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
            handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId, dcgm_structs.DcgmNvLinkLinkStateUp)

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    # Capture baseline incident count
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineIncidentCount = responseV5.incidentCount

    # Set a link to Down
    linkId = 3
    dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
        handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId, dcgm_structs.DcgmNvLinkLinkStateDown)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    # Verify exactly one new incident
    new_incidents = max(0, responseV5.incidentCount - baselineIncidentCount)
    assert new_incidents == 1, f"Expected 1 new incident, got {new_incidents}"

    # Verify the new incident details
    incident = responseV5.incidents[baselineIncidentCount]
    logger.info("Health String: " + incident.error.msg)

    assert incident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU
    assert incident.entityInfo.entityId == gpuId
    assert incident.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    assert incident.error.code == dcgm_errors.DCGM_FR_NVLINK_DOWN
    assert str(linkId) in incident.error.msg, "Didn't find linkId %d in %s" % (
        linkId, incident.error.msg)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_link_down_gpu_standalone(handle, gpuIds):
    helper_health_check_nvlink_link_down_gpu(handle, gpuIds)


def helper_health_check_nvlink_link_down_nvswitch(handle, switchIds):
    """
    Verifies that a check error occurs when a NvLink link is set to broken
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    switchId = switchIds[0]
    groupObj.AddEntity(dcgm_fields.DCGM_FE_SWITCH, switchId)
    linkId = 17

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_FATAL
    groupObj.health.Set(newSystems)

    # By default, the health check should pass
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert responseV5.incidentCount == 0, "Expected no errors. Got %d entities with errors: %s" % (
        responseV5.incidentCount, responseV5.incidents[0].error.msg)

    # Set a link to Down
    dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
        handle, dcgm_fields.DCGM_FE_SWITCH, switchId, linkId, dcgm_structs.DcgmNvLinkLinkStateDown)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == switchId)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_SWITCH)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_FATAL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_NVLINK_DOWN)
    assert str(linkId) in responseV5.incidents[0].error.msg, "Didn't find linkId %d in %s" % (
        linkId, responseV5.incidents[0].error.msg)


def setupNvLinkHealthTest(handle, gpuIds):
    """
    Common setup for NVLink health tests.

    Args:
        handle: DCGM handle
        gpuIds: List of GPU IDs

    Returns:
        tuple: (groupObj, gpuId, baselineIncidentCount)
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test_nvlink_health")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    # Clear all NVLink error counter values from previous tests
    clearNvlinkFields(handle, gpuId, 5)

    # Set all links to Up state to ensure clean baseline
    for linkId in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU):
        dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
            handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId,
            dcgm_structs.DcgmNvLinkLinkStateUp)

    # Get baseline health state
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineIncidentCount = responseV5.incidentCount

    return groupObj, gpuId, baselineIncidentCount


def helper_health_check_nvlink_all_links_up(handle, gpuIds):
    """Test NVLink health: All links Up -> PASS"""
    # Setup
    groupObj, gpuId, baselineIncidentCount = setupNvLinkHealthTest(
        handle, gpuIds)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    if responseV5.incidentCount > baselineIncidentCount:
        newNvLinkIncidents = [x for x in responseV5.incidents[baselineIncidentCount:responseV5.incidentCount]
                              if x.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK]
        for incident in newNvLinkIncidents:
            logger.debug(
                f"Unexpected new NVLink incident: health={incident.health}, error.code={incident.error.code}, msg='{incident.error.msg}'")
        assert len(
            newNvLinkIncidents) == 0, f"Expected 0 new NVLink incidents with all links Up, got {len(newNvLinkIncidents)}"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_all_links_up_standalone(handle, gpuIds):
    """Test NVLink health: All links Up -> PASS (standalone)"""
    helper_health_check_nvlink_all_links_up(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_all_links_up_embedded(handle, gpuIds):
    """Test NVLink health: All links Up -> PASS (embedded)"""
    helper_health_check_nvlink_all_links_up(handle, gpuIds)


def helper_health_check_nvlink_one_link_down(handle, gpuIds):
    """Test NVLink health: One link Down -> FAIL"""
    # Setup
    groupObj, gpuId, baselineIncidentCount = setupNvLinkHealthTest(
        handle, gpuIds)

    # Set one specific link to Down
    downLinkId = 2
    dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
        handle, dcgm_fields.DCGM_FE_GPU, gpuId, downLinkId, dcgm_structs.DcgmNvLinkLinkStateDown)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    newIncidents = max(0, responseV5.incidentCount - baselineIncidentCount)
    newNvLinkIncidents = [x for x in responseV5.incidents[baselineIncidentCount:responseV5.incidentCount]
                          if x.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK]
    assert len(
        newNvLinkIncidents) == 1, f"Expected exactly 1 NVLink incident for one link Down, got {len(newNvLinkIncidents)} (total new incidents: {newIncidents})"

    nvlinkIncident = newNvLinkIncidents[0]

    # Check that the NVLink incident has the correct error code
    assert nvlinkIncident.error.code == dcgm_errors.DCGM_FR_NVLINK_DOWN, \
        f"Expected DCGM_FR_NVLINK_DOWN, but got error code {nvlinkIncident.error.code}. Incident msg: '{nvlinkIncident.error.msg}'"

    # Check that the error message contains the link ID
    assert str(
        downLinkId) in nvlinkIncident.error.msg, f"Expected error message to contain link ID {downLinkId}, got: {nvlinkIncident.error.msg}"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_one_link_down_standalone(handle, gpuIds):
    """Test NVLink health: One link Down -> FAIL (standalone)"""
    helper_health_check_nvlink_one_link_down(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_one_link_down_embedded(handle, gpuIds):
    """Test NVLink health: One link Down -> FAIL (embedded)"""
    helper_health_check_nvlink_one_link_down(handle, gpuIds)


def helper_health_check_nvlink_not_supported_and_disabled(handle, gpuIds):
    """Test NVLink health: NotSupported and Disabled links -> PASS"""
    # Setup
    groupObj, gpuId, baselineIncidentCount = setupNvLinkHealthTest(
        handle, gpuIds)

    # Set some links to NotSupported and some to Disabled
    for linkId in range(0, 3):
        dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
            handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId, dcgm_structs.DcgmNvLinkLinkStateNotSupported)
    for linkId in range(3, 6):
        dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
            handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId, dcgm_structs.DcgmNvLinkLinkStateDisabled)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    if responseV5.incidentCount > baselineIncidentCount:
        newNvLinkIncidents = [x for x in responseV5.incidents[baselineIncidentCount:responseV5.incidentCount]
                              if x.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK]
        for incident in newNvLinkIncidents:
            logger.debug(
                f"Unexpected NVLink incident: health={incident.health}, error.code={incident.error.code}, msg='{incident.error.msg}'")
        assert len(
            newNvLinkIncidents) == 0, f"Expected 0 new NVLink incidents with NotSupported/Disabled links, got {len(newNvLinkIncidents)}"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_not_supported_and_disabled_standalone(handle, gpuIds):
    """Test NVLink health: NotSupported and Disabled links -> PASS (standalone)"""
    helper_health_check_nvlink_not_supported_and_disabled(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_not_supported_and_disabled_embedded(handle, gpuIds):
    """Test NVLink health: NotSupported and Disabled links -> PASS (embedded)"""
    helper_health_check_nvlink_not_supported_and_disabled(handle, gpuIds)


def helper_health_check_nvlink_mig_down_without_mig(handle, gpuIds):
    """Test MIG-aware NVLink health: Down without MIG -> FAIL"""
    # Setup
    groupObj, gpuId, baselineIncidentCount = setupNvLinkHealthTest(
        handle, gpuIds)

    # Ensure MIG is disabled
    ret = dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpuId, dcgm_fields.DCGM_FI_DEV_MIG_MODE, 0, 0)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Set NVLink to Down
    downLinkId = 2
    dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
        handle, dcgm_fields.DCGM_FE_GPU, gpuId, downLinkId, dcgm_structs.DcgmNvLinkLinkStateDown)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    newIncidents = max(0, responseV5.incidentCount - baselineIncidentCount)
    newNvLinkIncidents = [x for x in responseV5.incidents[baselineIncidentCount:responseV5.incidentCount]
                          if x.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK]
    assert len(
        newNvLinkIncidents) == 1, f"Expected exactly 1 NVLink incident for one link Down without MIG, got {len(newNvLinkIncidents)} (total new incidents: {newIncidents})"

    nvlinkIncident = newNvLinkIncidents[0]
    assert nvlinkIncident.entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU, \
        f"Expected entity group ID {dcgm_fields.DCGM_FE_GPU}, got {nvlinkIncident.entityInfo.entityGroupId}"
    assert nvlinkIncident.entityInfo.entityId == gpuId, \
        f"Expected entity ID {gpuId}, got {nvlinkIncident.entityInfo.entityId}"
    assert nvlinkIncident.error.code == dcgm_errors.DCGM_FR_NVLINK_DOWN, \
        f"Expected error code {dcgm_errors.DCGM_FR_NVLINK_DOWN}, got {nvlinkIncident.error.code}"
    assert str(downLinkId) in nvlinkIncident.error.msg, \
        f"Didn't find downLinkId {downLinkId} in {nvlinkIncident.error.msg}"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_mig_down_without_mig_standalone(handle, gpuIds):
    """Test MIG-aware NVLink health: Down without MIG -> FAIL (standalone)"""
    helper_health_check_nvlink_mig_down_without_mig(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_mig_down_without_mig_embedded(handle, gpuIds):
    """Test MIG-aware NVLink health: Down without MIG -> FAIL (embedded)"""
    helper_health_check_nvlink_mig_down_without_mig(handle, gpuIds)


def helper_health_check_nvlink_mig_down_with_mig_enabled(handle, gpuIds):
    """Test MIG-aware NVLink health: Down with MIG enabled -> PASS"""
    # Setup group and health monitoring
    groupObj, gpuId, baselineIncidentCount = setupNvLinkHealthTest(
        handle, gpuIds)

    # Enable MIG mode BEFORE setting link Down
    ret = dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpuId, dcgm_fields.DCGM_FI_DEV_MIG_MODE, 1, 0)
    assert ret == dcgm_structs.DCGM_ST_OK

    # Wait a moment for MIG mode to take effect
    time.sleep(0.1)

    # Now set link Down - this should NOT generate any incidents due to MIG mode
    downLinkId = 2
    dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(
        handle, dcgm_fields.DCGM_FE_GPU, gpuId, downLinkId, dcgm_structs.DcgmNvLinkLinkStateDown)

    # Wait for health check to process the Down link
    time.sleep(0.2)

    # Check that no incidents were added (Down link should be treated as acceptable in MIG mode)
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    # Verify no NVLink incidents were added
    if responseV5.incidentCount > baselineIncidentCount:
        newNvLinkIncidents = [x for x in responseV5.incidents[baselineIncidentCount:responseV5.incidentCount]
                              if x.system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK]
        for incident in newNvLinkIncidents:
            logger.debug(f"  NVLink incident: {incident.error.msg}")
        assert len(
            newNvLinkIncidents) == 0, f"MIG-aware logic failed: Found {len(newNvLinkIncidents)} NVLink failures with MIG enabled"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_mig_down_with_mig_enabled_standalone(handle, gpuIds):
    """Test MIG-aware NVLink health: Down with MIG enabled -> PASS (standalone)"""
    helper_health_check_nvlink_mig_down_with_mig_enabled(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_nvlink_mig_down_with_mig_enabled_embedded(handle, gpuIds):
    """Test MIG-aware NVLink health: Down with MIG enabled -> PASS (embedded)"""
    helper_health_check_nvlink_mig_down_with_mig_enabled(handle, gpuIds)


def helper_health_check_multiple_failures(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    # We are going to trigger two failures at the same time
    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_PCIE | dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    # inject a PCI Gen and width/lanes
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_LINK_GEN,
                                                              4, 0)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_LINK_WIDTH,
                                                              16, 0)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # inject a PCI error and a memory error, and make sure we report both
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                              4, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                                              100, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.incidentCount == 2)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (
        responseV5.incidents[1].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[1].entityInfo.entityId == gpuId)

    if responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM:
        # The memory error is in position 0 here
        assert (responseV5.incidents[0].error.code ==
                dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)

        # PCIE error is in position 1 here
        assert (responseV5.incidents[1].system ==
                dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
        assert (responseV5.incidents[1].error.code ==
                dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)
    else:
        assert (responseV5.incidents[0].system ==
                dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
        assert (responseV5.incidents[1].system ==
                dcgm_structs.DCGM_HEALTH_WATCH_MEM)

        # Mem is in position 1 now
        assert (responseV5.incidents[1].error.code ==
                dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)
        assert (responseV5.incidents[0].error.code ==
                dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_standalone_multiple_failures(handle, gpuIds):
    helper_health_check_multiple_failures(handle, gpuIds)


def helper_health_check_unreadable_power_usage(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_POWER
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_fp64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
                                                               dcgmvalue.DCGM_FP64_BLANK, 50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_WARN)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_POWER_UNREADABLE)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_standalone_unreadable_power_usage(handle, gpuIds):
    helper_health_check_unreadable_power_usage(handle, gpuIds)


def helper_health_set_version2(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds("test1", gpuIds)
    watchInterval = 999999
    maxKeepAge = 1234.5
    maxKeepAgeUsec = int(maxKeepAge) * 1000000

    fieldId = dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER
    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    groupObj.health.Set(newSystems, watchInterval, maxKeepAge)

    for gpuId in gpuIds:
        cmfi = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(
            handle, gpuId, dcgm_fields.DCGM_FE_GPU, fieldId)
        assert cmfi.flags & dcgm_structs_internal.DCGM_CMI_F_WATCHED, "x%X" % cmfi.flags
        assert cmfi.monitorIntervalUsec == watchInterval, "%d != %d" % (
            cmfi.monitorIntervalUsec, watchInterval)
        assert cmfi.maxAgeUsec == maxKeepAgeUsec, "%d != %d" % (
            cmfi.maxAgeUsec, maxKeepAgeUsec)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus(2)
def test_health_set_version2_standalone(handle, gpuIds):
    helper_health_set_version2(handle, gpuIds)


def helper_test_dcgm_health_check_uncontained_errors(handle, gpuIds):
    """
    Verifies that the health check will fail if we inject an uncontained error
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_ALL
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
                                                              95, 0)  # set the injected data to now
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_ALL)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_UNCONTAINED_ERROR)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_health_check_uncontained_errors(handle, gpuIds):
    helper_test_dcgm_health_check_uncontained_errors(handle, gpuIds)


def helper_test_dcgm_health_check_row_remap_failure(handle, gpuIds):
    """
    Verifies that the health check will fail if we inject an uncontained error
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ROW_REMAP_FAILURE,
                                                              1, 0)  # set the injected data to now
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_ROW_REMAP_FAILURE)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_health_check_row_remap_failure(handle, gpuIds):
    helper_test_dcgm_health_check_row_remap_failure(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_cpus(1)
@test_utils.run_with_injection_cpu_cores(1)
def test_dcgm_health_cpu_thermal(handle, cpuIds, coreIds):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t()
    entityPair.entityGroupId = dcgm_fields.DCGM_FE_CPU
    entityPair.entityId = cpuIds[0]
    systemObj = dcgmHandle.GetSystem()
    dcgmGroup = systemObj.GetEmptyGroup("test1")
    dcgmGroup.AddEntity(entityPair.entityGroupId, entityPair.entityId)
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)

    injection_info = [[dcgm_fields.DCGM_FI_DEV_CPU_TEMP_CURRENT, 150.0],
                      [dcgm_fields.DCGM_FI_DEV_CPU_TEMP_WARNING, 100.1],
                      [dcgm_fields.DCGM_FI_DEV_CPU_TEMP_CRITICAL, 110.1],
                      ]
    for ii in injection_info:
        dcgm_field_injection_helpers.inject_value(handle, entityPair.entityId, ii[0],
                                                  ii[1], -10, verifyInsertion=True,
                                                  entityType=entityPair.entityGroupId)
    for ii in injection_info:
        dcgm_field_injection_helpers.inject_value(handle, entityPair.entityId, ii[0],
                                                  ii[1], 50, verifyInsertion=True,
                                                  entityType=entityPair.entityGroupId)

    responseV5 = dcgmGroup.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 2)
    assert (responseV5.incidents[0].entityInfo.entityId == cpuIds[0])
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_CPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_WARN)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_FIELD_THRESHOLD_DBL)
    assert (responseV5.incidents[1].entityInfo.entityId == cpuIds[0])
    assert (
        responseV5.incidents[1].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_CPU)
    assert (responseV5.incidents[1].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)
    assert (responseV5.incidents[1].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[1].error.code ==
            dcgm_errors.DCGM_FR_FIELD_THRESHOLD_DBL)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_cpus(1)
@test_utils.run_with_injection_cpu_cores(1)
def test_dcgm_health_cpu_power(handle, cpuIds, coreIds):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t()
    entityPair.entityGroupId = dcgm_fields.DCGM_FE_CPU
    entityPair.entityId = cpuIds[0]
    systemObj = dcgmHandle.GetSystem()
    dcgmGroup = systemObj.GetEmptyGroup("test1")
    dcgmGroup.AddEntity(entityPair.entityGroupId, entityPair.entityId)
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_POWER)

    injection_info = [[dcgm_fields.DCGM_FI_DEV_CPU_POWER_LIMIT, 150.0],
                      [dcgm_fields.DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT, 150.1],
                      ]
    for ii in injection_info:
        dcgm_field_injection_helpers.inject_value(handle, entityPair.entityId, ii[0],
                                                  ii[1], 5, verifyInsertion=True,
                                                  entityType=entityPair.entityGroupId)

    responseV5 = dcgmGroup.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == cpuIds[0])
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_CPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_FIELD_THRESHOLD_DBL)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_check_pcie_correctable_errors_field_injection_valid(handle, gpuIds):
    """Test PCIe correctable errors field value injection and retrieval"""
    gpuId = gpuIds[0]
    injection_value = 42

    # Create field value for injection
    fv = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    fv.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    fv.fieldId = dcgm_fields.DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS
    fv.status = 0
    fv.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    fv.ts = int((time.time() - 5) * 1000000.0)
    fv.value.i64 = injection_value

    # Inject the field value
    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fv)
    assert ret == dcgm_structs.DCGM_ST_OK, f"Field injection failed with status {ret}"

    # Verify the injected value can be retrieved
    fieldValues = dcgm_agent_internal.dcgmGetLatestValuesForFields(
        handle, gpuId, [dcgm_fields.DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS])
    assert len(
        fieldValues) == 1, f"Expected 1 field value, got {len(fieldValues)}"
    assert fieldValues[
        0].fieldId == dcgm_fields.DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS, f"Expected fieldId {dcgm_fields.DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS}, got {fieldValues[0].fieldId}"
    assert fieldValues[
        0].status == dcgm_structs.DCGM_ST_OK, f"Expected status OK, got {fieldValues[0].status}"
    assert fieldValues[0].value.i64 == injection_value, f"Expected value {injection_value}, got {fieldValues[0].value.i64}"

    print(
        f"Field injection and retrieval successful: value={fieldValues[0].value.i64}")


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_check_mem_unrepairable_flag(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test_unrepairable")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()  # Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    # Test Case 1: Inject flag = 0 (no unrepairable memory) - should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpuId, dcgm_fields.DCGM_FI_DEV_MEMORY_UNREPAIRABLE_FLAG, 0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)

    # Test Case 2: Inject flag = 1 (unrepairable memory detected) - should fail
    ret = dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpuId, dcgm_fields.DCGM_FI_DEV_MEMORY_UNREPAIRABLE_FLAG, 1, -45)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health ==
            dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_FAULTY_MEMORY)

    # Test Case 3: Clear the flag back to 0 - should pass again
    ret = dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpuId, dcgm_fields.DCGM_FI_DEV_MEMORY_UNREPAIRABLE_FLAG, 0, -40)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
def test_dcgm_health_check_fabric_health_mask(handle, gpuIds):
    """Test passive health checks for fabric health mask"""
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup(
        "test_dcgm_health_check_fabric_health_mask_grp")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()
    gpuId = gpuIds[0]
    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    # Inject healthy state (all bits = 0 or FALSE)
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_FABRIC_HEALTH_MASK,
                                                              0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Verify baseline is actually healthy
    skip_test_if_unhealthy(groupObj)
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.incidentCount ==
            0), "Expected no incidents in healthy state"

    # Inject Route Unhealthy incident (bit 4 set to TRUE = 1)
    route_unhealthy_mask = dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_TRUE << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ROUTE_UNHEALTHY
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_FABRIC_HEALTH_MASK,
                                                              route_unhealthy_mask, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.incidentCount ==
            1), f"Expected 1 incident for Route Unhealthy, got {responseV5.incidentCount}"
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (
        responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_FIELD_VIOLATION)
    assert "Route Unhealthy" in responseV5.incidents[
        0].error.msg, f"Expected 'Route Unhealthy' in error message, got: {responseV5.incidents[0].error.msg}"

    # Inject multiple incidents: Route Unhealthy + Bandwidth Degraded
    degraded_bw_mask = dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_TRUE << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_DEGRADED_BW
    combined_mask = route_unhealthy_mask | degraded_bw_mask
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_FABRIC_HEALTH_MASK,
                                                              combined_mask, 20)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.incidentCount >=
            1), f"Expected at least 1 incident for combined errors, got {responseV5.incidentCount}"

    # Verify error message contains both conditions
    error_messages = ' '.join(
        [responseV5.incidents[i].error.msg for i in range(responseV5.incidentCount)])
    assert "Route Unhealthy" in error_messages, f"Expected 'Route Unhealthy' in error messages, got: {error_messages}"
    assert "Bandwidth Degraded" in error_messages, f"Expected 'Bandwidth Degraded' in error messages, got: {error_messages}"

    # Inject NOT_SUPPORTED state (all fields set to NOT_SUPPORTED)
    not_supported_mask = (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_NOT_SUPPORTED << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_DEGRADED_BW) | \
                         (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_NOT_SUPPORTED << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ROUTE_UNHEALTHY) | \
                         (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_RECOVERY_NOT_SUPPORTED << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ROUTE_RECOVERY) | \
                         (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_ACCESS_TIMEOUT_RECOVERY_NOT_SUPPORTED << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ACCESS_TIMEOUT_RECOVERY) | \
                         (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NOT_SUPPORTED <<
                          dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_INCORRECT_CONFIGURATION)
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_FABRIC_HEALTH_MASK,
                                                              not_supported_mask, 40)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Verify no incidents for NOT_SUPPORTED states
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.incidentCount ==
            0), f"Expected no incidents for NOT_SUPPORTED states, got {responseV5.incidentCount}"

    # Inject healthy state again
    healthy_mask = (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_FALSE << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_DEGRADED_BW) | \
                   (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_FALSE << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ROUTE_UNHEALTHY) | \
                   (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_RECOVERY_FALSE << dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ROUTE_RECOVERY) | \
                   (dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_ACCESS_TIMEOUT_RECOVERY_FALSE <<
                    dcgm_nvml.NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ACCESS_TIMEOUT_RECOVERY)
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_FABRIC_HEALTH_MASK,
                                                              healthy_mask, 30)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Verify no incidents after clearing errors
    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.incidentCount ==
            0), f"Expected no incidents after clearing errors, got {responseV5.incidentCount}"

    # Clear health watches
    groupObj.health.Set(0)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
def test_dcgm_health_check_imex_status(handle, gpuIds):
    """Test IMEX domain and daemon status health checks"""
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test_imex")
    gpuId = gpuIds[0]
    groupObj.AddGpu(gpuId)
    groupObj.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)

    # Helper function to inject IMEX domain status (string field)
    # IMEX fields are global scope (DCGM_FE_NONE), inject with entity 0
    def injectImexDomainStatus(status_str, offset_seconds):
        field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        field.fieldId = dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS
        field.status = 0
        field.fieldType = ord(dcgm_fields.DCGM_FT_STRING)
        field.ts = int((time.time() + offset_seconds) * 1000000.0)
        field.value.str = status_str.encode(
            'utf-8') if isinstance(status_str, str) else status_str
        return dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_NONE, 0, field)

    # Helper function to inject IMEX daemon status (int64 field)
    # IMEX fields are global scope (DCGM_FE_NONE), inject with entity 0
    def injectImexDaemonStatus(status_int, offset_seconds):
        field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        field.fieldId = dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS
        field.status = 0
        field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
        field.ts = int((time.time() + offset_seconds) * 1000000.0)
        field.value.i64 = status_int
        return dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_NONE, 0, field)

    # Helper function to run a single IMEX health test case
    def validateImexHealth(
        timestamp_offset,
        domain_status,
        daemon_status,
        expected_health,
        test_description,
        error_msg_checks=None
    ):
        logger.debug(test_description)

        # Inject statuses
        ret = injectImexDomainStatus(domain_status, timestamp_offset)
        assert ret == dcgm_structs.DCGM_ST_OK, f"Domain injection failed: {ret}"
        ret = injectImexDaemonStatus(daemon_status, timestamp_offset)
        assert ret == dcgm_structs.DCGM_ST_OK, f"Daemon injection failed: {ret}"
        timestamp_offset += 1

        # Check health
        responseV5 = groupObj.health.Check(
            dcgm_structs.dcgmHealthResponse_version5)
        assert responseV5.overallHealth == expected_health, \
            f"{test_description}: Expected {expected_health}, got {responseV5.overallHealth}"

        # Optional error message validation
        if error_msg_checks:
            min_incidents = error_msg_checks.get('min_incidents', 1)
            assert responseV5.incidentCount >= min_incidents, \
                f"Expected at least {min_incidents} incident(s), got {responseV5.incidentCount}"

            if responseV5.incidentCount > 0:
                # Check incident attributes
                assert responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
                assert responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL
                # IMEX fields are global scope (DCGM_FE_NONE) so entityId should be 0
                assert responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_NONE
                assert responseV5.incidents[0].entityInfo.entityId == 0

                # Check error message keywords
                error_msg = responseV5.incidents[0].error.msg
                keywords = error_msg_checks.get('keywords', [])
                for keyword in keywords:
                    assert keyword in error_msg, \
                        f"Expected '{keyword}' in error message: {error_msg}"

        return timestamp_offset, responseV5

    # DcgmImexManager updates the cache every 60 seconds. Using offset >= 0 ensures injected values are
    # treated as the latest sample.
    timestamp_offset = 0

    # Establish healthy baseline to prevent skip_test_if_unhealthy from skipping
    ret = injectImexDomainStatus("UP", timestamp_offset)
    assert ret == dcgm_structs.DCGM_ST_OK, f"Failed to inject baseline domain status: {ret}"
    ret = injectImexDaemonStatus(5, timestamp_offset)  # 5 = READY
    assert ret == dcgm_structs.DCGM_ST_OK, f"Failed to inject baseline daemon status: {ret}"

    # Now check baseline health (should pass with our injected healthy values)
    skip_test_if_unhealthy(groupObj)
    timestamp_offset += 1

    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="UP",
        daemon_status=5,  # READY
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_PASS,
        test_description="Healthy IMEX state"
    )

    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="DOWN",
        daemon_status=5,
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
        test_description="Domain DOWN",
        error_msg_checks={
            'keywords': ["IMEX domain status", "DOWN"],
            'min_incidents': 1
        }
    )

    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="DEGRADED",
        daemon_status=5,
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
        test_description="Domain DEGRADED",
        error_msg_checks={
            'keywords': ["IMEX domain status", "DEGRADED"],
            'min_incidents': 1
        }
    )

    # Daemon Status
    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="UP",
        daemon_status=0,  # INITIALIZING
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
        test_description="Daemon INITIALIZING",
        error_msg_checks={
            'keywords': ["IMEX daemon status", "0"],
            'min_incidents': 1
        }
    )

    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="UP",
        daemon_status=7,  # UNAVAILABLE
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_PASS,
        test_description="Daemon UNAVAILABLE"
    )

    # Both unhealthy - Domain=DOWN, Daemon=INITIALIZING
    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="DOWN",
        daemon_status=0,
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
        test_description="Both unhealthy (Domain=DOWN, Daemon=INITIALIZING)",
        error_msg_checks={
            'min_incidents': 1  # Should have at least 1 incident, possibly 2
        }
    )

    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="NOT_INSTALLED",
        daemon_status=-1,  # NOT_INSTALLED
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_PASS,
        test_description="Domain NOT_INSTALLED state"
    )

    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="NOT_CONFIGURED",
        daemon_status=-2,  # NOT_CONFIGURED
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_PASS,
        test_description="Domain NOT_CONFIGURED state"
    )

    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="UNAVAILABLE",
        daemon_status=-1,
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_PASS,
        test_description="Domain UNAVAILABLE state"
    )

    timestamp_offset, _ = validateImexHealth(
        timestamp_offset=timestamp_offset,
        domain_status="UP",
        daemon_status=5,  # READY
        expected_health=dcgm_structs.DCGM_HEALTH_RESULT_PASS,
        test_description="Recovery to healthy state"
    )

    # Clean up: Clear health watches to avoid impacting other tests
    logger.debug("Cleaning up: clearing health watches")
    groupObj.health.Set(0)


def clearNvlinkFields(handle, gpuId, timestamp):
    nvlink_fields = [
        (dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL, 0),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL, 0),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL, 0),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL, 0),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS, 0),
        (dcgm_fields.DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER,
         dcgmvalue.DCGM_INT64_BLANK),
        (dcgm_fields.DCGM_FI_DEV_FABRIC_HEALTH_MASK, 0),
        (dcgm_fields.DCGM_FI_DEV_FABRIC_MANAGER_STATUS, dcgmvalue.DCGM_INT64_BLANK),
    ]

    for field_id, healthy_value in nvlink_fields:
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, field_id, healthy_value, timestamp)
        assert ret == dcgm_structs.DCGM_ST_OK


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_health_fabric_manager_status(handle, gpuIds):
    """Test passive health checks for fabric manager status"""
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test_fabric_manager_status")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds()
    gpuId = gpuIds[0]

    groupObj.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)

    clearNvlinkFields(handle, gpuId, -60)
    skip_test_if_unhealthy(groupObj)

    test_cases = [
        (3, "Success", 0, None, None),
        (0, "NotSupported", 0, None, None),
        (2, "InProgress", 1, dcgm_structs.DCGM_HEALTH_RESULT_WARN, "In Progress"),
        (4, "Failure", 1, dcgm_structs.DCGM_HEALTH_RESULT_FAIL, None),
        (1, "NotStarted", 1, dcgm_structs.DCGM_HEALTH_RESULT_FAIL, None),
        (5, "Unrecognized", 1, dcgm_structs.DCGM_HEALTH_RESULT_FAIL, None),
        (6, "NvmlTooOld", 1, dcgm_structs.DCGM_HEALTH_RESULT_FAIL, None),
    ]

    for idx, (status_value, status_desc, expected_incidents, expected_health, expected_msg) in enumerate(test_cases):
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, dcgm_fields.DCGM_FI_DEV_FABRIC_MANAGER_STATUS, status_value, idx * 10)
        assert ret == dcgm_structs.DCGM_ST_OK

        responseV5 = groupObj.health.Check(
            dcgm_structs.dcgmHealthResponse_version5)
        assert responseV5.incidentCount == expected_incidents, \
            f"Expected {expected_incidents} incidents for {status_desc}, got {responseV5.incidentCount}"

        if expected_incidents > 0:
            assert responseV5.incidents[0].entityInfo.entityId == gpuId
            assert responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
            assert responseV5.incidents[0].health == expected_health, \
                f"Expected health {expected_health} for {status_desc}, got {responseV5.incidents[0].health}"
            assert responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_FABRIC_PROBE_STATE, \
                f"Expected error code {dcgm_errors.DCGM_FR_FABRIC_PROBE_STATE} for {status_desc}, got {responseV5.incidents[0].error.code}"
            if expected_msg:
                assert expected_msg in responseV5.incidents[0].error.msg, \
                    f"Expected '{expected_msg}' in error message for {status_desc}, got: {responseV5.incidents[0].error.msg}"

    clearNvlinkFields(handle, gpuId, 100)
    groupObj.health.Set(0)


def helper_health_check_gpu_recovery_action(handle, gpuIds, inject_value, expected_health,
                                            expected_error_code=None,
                                            health_watch=dcgm_structs.DCGM_HEALTH_WATCH_DRIVER,
                                            verify_incident=True, cleanup=True):
    """
    Parameterized helper for testing GPU recovery action health checks

    Args:
        handle: DCGM handle
        gpuIds: List of GPU IDs
        inject_value: Value to inject (0-4, or DCGM_INT64_BLANK)
        expected_health: Expected health result (PASS, WARN, or FAIL)
        expected_error_code: Expected error code (if None, inferred from inject_value)
        health_watch: Health watch system to enable (default: DCGM_HEALTH_WATCH_DRIVER)
        verify_incident: Whether to verify incident details (default: True)
        cleanup: Whether to inject healthy value after test (default: True)
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuId = gpuIds[0]

    groupObj.health.Set(health_watch)
    skip_test_if_unhealthy(groupObj)

    baselineResponse = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineCount = baselineResponse.incidentCount

    ret = dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpuId, dcgm_fields.DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION, inject_value, 0)
    assert ret == dcgm_structs.DCGM_ST_OK

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    assert responseV5.overallHealth == expected_health

    if verify_incident and expected_health != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        assert responseV5.incidentCount > baselineCount, \
            f"Expected new incident, baseline={baselineCount}, current={responseV5.incidentCount}"

        # Map inject_value to expected error code if not explicitly provided
        if expected_error_code is None:
            error_code_map = {
                1: dcgm_errors.DCGM_FR_GPU_RECOVERY_RESET,
                2: dcgm_errors.DCGM_FR_GPU_RECOVERY_REBOOT,
                3: dcgm_errors.DCGM_FR_GPU_RECOVERY_DRAIN_P2P,
                4: dcgm_errors.DCGM_FR_GPU_RECOVERY_DRAIN_RESET,
            }
            expected_error_code = error_code_map.get(inject_value)

        # Expected keywords for each recovery action type (for message content verification)
        expected_keywords = {
            1: ["reset", "recover", "fault"],           # GPU_RESET
            2: ["reboot", "inconsistent"],              # NODE_REBOOT
            3: ["peer-to-peer", "quiesced"],            # DRAIN_P2P
            4: ["reduced capacity", "drain"],           # DRAIN_AND_RESET
        }

        found_driver_failure = False
        for i in range(responseV5.incidentCount):
            incident = responseV5.incidents[i]
            if incident.system == dcgm_structs.DCGM_HEALTH_WATCH_DRIVER:
                # Verify error code
                assert incident.error.code == expected_error_code, \
                    f"Expected error code {expected_error_code}, got {incident.error.code}"

                # Verify entity ID
                assert incident.entityInfo.entityId == gpuId, \
                    f"Expected entity ID {gpuId}, got {incident.entityInfo.entityId}"

                # Verify incident's health result matches expected
                assert incident.health == expected_health, \
                    f"Expected incident health {expected_health}, got {incident.health}"

                # Log the error message for debugging
                logger.info(
                    f"GPU Recovery Action incident message: {incident.error.msg}")

                # Verify error message contains GPU ID
                assert str(gpuId) in incident.error.msg, \
                    f"GPU ID {gpuId} not found in error message: {incident.error.msg}"

                # Verify error message contains recovery action value (for non-NONE values)
                if inject_value > 0 and inject_value <= 4:
                    assert str(inject_value) in incident.error.msg, \
                        f"Recovery action value {inject_value} not found in error message: {incident.error.msg}"

                # Verify message contains expected keywords for this recovery action
                if inject_value in expected_keywords:
                    msg_lower = incident.error.msg.lower()
                    for keyword in expected_keywords[inject_value]:
                        assert keyword.lower() in msg_lower, \
                            f"Expected keyword '{keyword}' not found in message: {incident.error.msg}"

                found_driver_failure = True
                break

        assert found_driver_failure, "Driver health failure not detected"

    if cleanup and expected_health != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        ret = dcgm_field_injection_helpers.inject_field_value_i64(
            handle, gpuId, dcgm_fields.DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION, 0, 0)
        assert ret == dcgm_structs.DCGM_ST_OK


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_none_standalone(handle, gpuIds):
    """Test GPU recovery action health check - NONE value (0) should PASS"""
    helper_health_check_gpu_recovery_action(handle, gpuIds, 0, dcgm_structs.DCGM_HEALTH_RESULT_PASS,
                                            verify_incident=False, cleanup=False)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_none_embedded(handle, gpuIds):
    """Test GPU recovery action health check - NONE value (0) should PASS"""
    helper_health_check_gpu_recovery_action(handle, gpuIds, 0, dcgm_structs.DCGM_HEALTH_RESULT_PASS,
                                            verify_incident=False, cleanup=False)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_reset_standalone(handle, gpuIds):
    """Test GPU recovery action health check - GPU_RESET value (1) should FAIL"""
    helper_health_check_gpu_recovery_action(
        handle, gpuIds, 1, dcgm_structs.DCGM_HEALTH_RESULT_FAIL)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_reset_embedded(handle, gpuIds):
    """Test GPU recovery action health check - GPU_RESET value (1) should FAIL"""
    helper_health_check_gpu_recovery_action(
        handle, gpuIds, 1, dcgm_structs.DCGM_HEALTH_RESULT_FAIL)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_reboot_standalone(handle, gpuIds):
    """Test GPU recovery action health check - NODE_REBOOT value (2) should FAIL"""
    helper_health_check_gpu_recovery_action(
        handle, gpuIds, 2, dcgm_structs.DCGM_HEALTH_RESULT_FAIL)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_reboot_embedded(handle, gpuIds):
    """Test GPU recovery action health check - NODE_REBOOT value (2) should FAIL"""
    helper_health_check_gpu_recovery_action(
        handle, gpuIds, 2, dcgm_structs.DCGM_HEALTH_RESULT_FAIL)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_drain_p2p_standalone(handle, gpuIds):
    """Test GPU recovery action health check - DRAIN_P2P value (3) should WARN"""
    helper_health_check_gpu_recovery_action(
        handle, gpuIds, 3, dcgm_structs.DCGM_HEALTH_RESULT_WARN)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_drain_p2p_embedded(handle, gpuIds):
    """Test GPU recovery action health check - DRAIN_P2P value (3) should WARN"""
    helper_health_check_gpu_recovery_action(
        handle, gpuIds, 3, dcgm_structs.DCGM_HEALTH_RESULT_WARN)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_drain_and_reset_standalone(handle, gpuIds):
    """Test GPU recovery action health check - DRAIN_AND_RESET value (4) should WARN"""
    helper_health_check_gpu_recovery_action(
        handle, gpuIds, 4, dcgm_structs.DCGM_HEALTH_RESULT_WARN)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_drain_and_reset_embedded(handle, gpuIds):
    """Test GPU recovery action health check - DRAIN_AND_RESET value (4) should WARN"""
    helper_health_check_gpu_recovery_action(
        handle, gpuIds, 4, dcgm_structs.DCGM_HEALTH_RESULT_WARN)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_blank_standalone(handle, gpuIds):
    """Test GPU recovery action health check - BLANK value should PASS"""
    helper_health_check_gpu_recovery_action(handle, gpuIds, dcgmvalue.DCGM_INT64_BLANK,
                                            dcgm_structs.DCGM_HEALTH_RESULT_PASS,
                                            verify_incident=False, cleanup=False)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_blank_embedded(handle, gpuIds):
    """Test GPU recovery action health check - BLANK value should PASS"""
    helper_health_check_gpu_recovery_action(handle, gpuIds, dcgmvalue.DCGM_INT64_BLANK,
                                            dcgm_structs.DCGM_HEALTH_RESULT_PASS,
                                            verify_incident=False, cleanup=False)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_all_watches_healthy_standalone(handle, gpuIds):
    """Test GPU recovery action with ALL watches enabled - healthy state"""
    helper_health_check_gpu_recovery_action(handle, gpuIds, 0, dcgm_structs.DCGM_HEALTH_RESULT_PASS,
                                            health_watch=dcgm_structs.DCGM_HEALTH_WATCH_ALL,
                                            verify_incident=False, cleanup=False)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_all_watches_healthy_embedded(handle, gpuIds):
    """Test GPU recovery action with ALL watches enabled - healthy state"""
    helper_health_check_gpu_recovery_action(handle, gpuIds, 0, dcgm_structs.DCGM_HEALTH_RESULT_PASS,
                                            health_watch=dcgm_structs.DCGM_HEALTH_WATCH_ALL,
                                            verify_incident=False, cleanup=False)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_all_watches_unhealthy_standalone(handle, gpuIds):
    """Test GPU recovery action with ALL watches enabled - unhealthy state should be detected"""
    helper_health_check_gpu_recovery_action(handle, gpuIds, 1, dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
                                            health_watch=dcgm_structs.DCGM_HEALTH_WATCH_ALL)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_gpu_recovery_action_all_watches_unhealthy_embedded(handle, gpuIds):
    """Test GPU recovery action with ALL watches enabled - unhealthy state should be detected"""
    helper_health_check_gpu_recovery_action(handle, gpuIds, 1, dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
                                            health_watch=dcgm_structs.DCGM_HEALTH_WATCH_ALL)


def helper_health_check_gpu_recovery_action_multiple_gpus(handle, gpuIds):
    """
    Test GPU recovery action with multiple GPUs having different recovery actions.
    Verifies that each GPU's incident is properly tracked with correct error codes and messages.
    """
    if len(gpuIds) < 2:
        test_utils.skip_test("Test requires at least 2 GPUs")

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test_multi_gpu")

    # Add two GPUs to the group
    gpu0 = gpuIds[0]
    gpu1 = gpuIds[1]
    groupObj.AddGpu(gpu0)
    groupObj.AddGpu(gpu1)

    groupObj.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_DRIVER)
    skip_test_if_unhealthy(groupObj)

    baselineResponse = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)
    baselineCount = baselineResponse.incidentCount

    # Inject different recovery actions for each GPU
    # GPU 0: GPU_RESET (1) - should FAIL
    ret = dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpu0, dcgm_fields.DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION, 1, 0)
    assert ret == dcgm_structs.DCGM_ST_OK

    # GPU 1: DRAIN_P2P (3) - should WARN
    ret = dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpu1, dcgm_fields.DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION, 3, 0)
    assert ret == dcgm_structs.DCGM_ST_OK

    responseV5 = groupObj.health.Check(
        dcgm_structs.dcgmHealthResponse_version5)

    # Overall health should be FAIL (worst of the two)
    assert responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL, \
        f"Expected overall health FAIL, got {responseV5.overallHealth}"

    # Filter new DRIVER incidents (exclude old baseline incidents and non-DRIVER incidents)
    newDriverIncidents = [incident for incident in responseV5.incidents[baselineCount:]
                          if incident.system == dcgm_structs.DCGM_HEALTH_WATCH_DRIVER]

    # Should have 2 new DRIVER incidents
    assert len(newDriverIncidents) == 2, \
        f"Expected 2 new DRIVER incidents, got {len(newDriverIncidents)} (baseline={baselineCount}, current={responseV5.incidentCount})"

    # Find and verify both incidents
    found_gpu0_incident = False
    found_gpu1_incident = False

    for incident in newDriverIncidents:
        if incident.entityInfo.entityId == gpu0:
            # GPU 0 should have GPU_RESET error (FAIL)
            assert incident.error.code == dcgm_errors.DCGM_FR_GPU_RECOVERY_RESET, \
                f"GPU {gpu0}: Expected DCGM_FR_GPU_RECOVERY_RESET, got {incident.error.code}"
            assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL, \
                f"GPU {gpu0}: Expected FAIL, got {incident.health}"
            assert str(gpu0) in incident.error.msg, \
                f"GPU {gpu0}: GPU ID not found in message: {incident.error.msg}"
            assert "1" in incident.error.msg, \
                f"GPU {gpu0}: Recovery action value 1 not found in message: {incident.error.msg}"
            msg_lower = incident.error.msg.lower()
            assert "reset" in msg_lower and "fault" in msg_lower, \
                f"GPU {gpu0}: Expected keywords not found in message: {incident.error.msg}"
            logger.info(f"GPU {gpu0} incident verified: {incident.error.msg}")
            found_gpu0_incident = True

        elif incident.entityInfo.entityId == gpu1:
            # GPU 1 should have DRAIN_P2P error (WARN)
            assert incident.error.code == dcgm_errors.DCGM_FR_GPU_RECOVERY_DRAIN_P2P, \
                f"GPU {gpu1}: Expected DCGM_FR_GPU_RECOVERY_DRAIN_P2P, got {incident.error.code}"
            assert incident.health == dcgm_structs.DCGM_HEALTH_RESULT_WARN, \
                f"GPU {gpu1}: Expected WARN, got {incident.health}"
            assert str(gpu1) in incident.error.msg, \
                f"GPU {gpu1}: GPU ID not found in message: {incident.error.msg}"
            assert "3" in incident.error.msg, \
                f"GPU {gpu1}: Recovery action value 3 not found in message: {incident.error.msg}"
            msg_lower = incident.error.msg.lower()
            assert "peer-to-peer" in msg_lower and "quiesced" in msg_lower, \
                f"GPU {gpu1}: Expected keywords not found in message: {incident.error.msg}"
            logger.info(f"GPU {gpu1} incident verified: {incident.error.msg}")
            found_gpu1_incident = True

    assert found_gpu0_incident, f"Incident for GPU {gpu0} not found"
    assert found_gpu1_incident, f"Incident for GPU {gpu1} not found"

    # Cleanup: restore both GPUs to healthy state
    dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpu0, dcgm_fields.DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION, 0, 0)
    dcgm_field_injection_helpers.inject_field_value_i64(
        handle, gpu1, dcgm_fields.DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION, 0, 0)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus(2)
def test_health_check_gpu_recovery_action_multiple_gpus_standalone(handle, gpuIds):
    """Test GPU recovery action with multiple GPUs having different recovery states"""
    helper_health_check_gpu_recovery_action_multiple_gpus(handle, gpuIds)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus(2)
def test_health_check_gpu_recovery_action_multiple_gpus_embedded(handle, gpuIds):
    """Test GPU recovery action with multiple GPUs having different recovery states"""
    helper_health_check_gpu_recovery_action_multiple_gpus(handle, gpuIds)
