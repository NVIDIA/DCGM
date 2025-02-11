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
import nvml_injection
import nvml_injection_structs
from _test_helpers import skip_test_if_no_dcgm_nvml, maybe_dcgm_nvml
import random
import math

def skip_test_if_unhealthy(groupObj):
    # Skip the test if the GPU is already failing health checks
    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
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

    #Set it back to 0 and validate it
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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
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
            pcieReplayCounter, 100) # set the injected data into the future
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    if expectingPcieIncident:
        assert (responseV5.incidentCount == 1), errmsg
        assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
        assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
        assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)
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
        2.5 / 1000 * 60,  # Gen1 speed = 2.5 Gbps, (1x10^-12) * (2.5x10^9) * 60 = 0.15 errors/min per lane.
        5.0 / 1000 * 60,  # Gen2 speed = 5 Gbps, (1x10^-12) * (5x10^9) * 60 = 0.3 errors/min per lane.
        8.0 / 1000 * 60,  # Gen3 speed = 8 Gbps, (1x10^-12) * (8x10^9) * 60 = 0.48 errors/min per lane.
        16.0 / 1000 * 60, # Gen4 speed = 16 Gbps, (1x10^-12) * (16x10^9) * 60 = 0.96 errors/min per lane.
        32.0 / 1000 * 60, # Gen5 speed = 32 Gbps, (1x10^-12) * (32x10^9) * 60 = 1.92 errors/min per lane.
        64.0 / 1000 * 60  # Gen6 speed = 64 Gbps, (1x10^-12) * (64x10^9) * 60 = 3.84 errors/min per lane.
    ]

    # Run it multiple times to cover more combinations of pcieGen and pcieLanes.
    for i in range(3):
        pcieGen = 1
        # For each Gen, randomly select number of lanes and PCIe replay counter to inject to dcgm.
        # If replay counter rate > expected rate, then make sure PCIe incident is present, no incident otherwise.
        for pcieGenReplayRatePerLane in pcieGenReplayRatesPerLane:
            pcieLanes = random.randint(1, 16)
            expectedPcieReplayCounterLimit = math.ceil(pcieGenReplayRatePerLane * pcieLanes)
            # Multiply by 2 to have uniform probability for pcieReplayCounter to give success or failure scenario.
            pcieReplayCounter = random.randint(1, 2 * expectedPcieReplayCounterLimit) 
            expectingPcieIncident = True if pcieReplayCounter > expectedPcieReplayCounterLimit else False
            errmsg = (f"pcieGen={pcieGen} pcieGenReplayRatePerLane={pcieGenReplayRatePerLane} pcieLanes={pcieLanes} "\
                      f"expectedPcieReplayCounterLimit={expectedPcieReplayCounterLimit} pcieReplayCounter={pcieReplayCounter} "\
                      f"expectingPcieIncident={expectingPcieIncident}")
            helper_dcgm_health_check_pcie(handle, gpuIds, pcieGen, pcieLanes, pcieReplayCounter, expectingPcieIncident, errmsg)
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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
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

    ret = dcgm_agent_internal.dcgmInjectNvmlDeviceForFollowingCalls(handle, gpuId, "PcieReplayCounter", None, 0, injectedRets, 4)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    groupObj.health.Set(newSystems, 250000, 3600.0) # 4 values * 250ms = 1 second to get a hit

    skip_test_if_unhealthy(groupObj)

    response = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    # we expect that there will be no data here
    assert (response.incidentCount == 0)

    # Wait for cache refresh
    for _ in range(10000):
        responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
        if responseV5.incidentCount == 1:
            break
        time.sleep(0.001)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)

def helper_test_dcgm_health_check_mem_dbe(handle, gpuIds):
    """
    Verifies that the health check will fail if there's 1 DBE and it continues to be
    reported
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            2, -50) # set the injected data to 50 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)

    # Give it the same failure 45 seconds ago and make sure we fail again
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            2, -45)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)

    # Make the failure count go down to zero. This should clear the error
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            0, -40)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
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
    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    if not verifyFail:
        assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)
        return

    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

def helper_reset_page_retirements(handle, gpuId=0, reset_sbe=False):
    """
    Helper function to reset non volatile page retirements.
    """
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
        0, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)
    if reset_sbe:
        ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            0, -30) # set the injected data to 30 seconds ago
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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ####### Tests #######
    #### Condition 1 ####
    ### Fail if the total number of page retirements (due to DBE or SBE) meets or exceeds 60
    ## Test 1: >= 60 page retirements total should fail
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            30, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            30, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we get a failure
    helper_verify_dcgm_health_watch_mem_result(groupObj, dcgm_errors.DCGM_FR_RETIRED_PAGES_LIMIT, verifyFail=True,
                                               gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    ## Test 2: 59 page retirements total should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            10, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            49, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    #### Condition 2 ####
    ### Fail if > 15 page retirement due to DBEs AND more than 1 DBE page retirement in past week
    ## Test 1: 15 page retirements due to DBEs should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            15, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    ## Test 2: 16 page retirements due to DBE should fail (since all 16 are inserted in current week)
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            16, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we get a failure
    helper_verify_dcgm_health_watch_mem_result(groupObj, dcgm_errors.DCGM_FR_RETIRED_PAGES_DBE_LIMIT,
                                               verifyFail=True, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    ## Test 3: 16 page retirements due to SBEs should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            16, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    ## Test 4: 16 page retirements due to DBEs (with first 15 pages inserted more than 1 week ago,
    # and 16th page inserted in current week) should pass
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            15, -604860) # set the injected data to 7 days and 1 minute ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            1, -30) # set the injected data to 30 seconds ago
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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
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

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1), "Expected 1 incident but found %d" % responseV5.incidentCount
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_PENDING_PAGE_RETIREMENTS),\
            "Expected %d but found %d" % (dcgm_errors.DCGM_FR_PENDING_PAGE_RETIREMENTS, \
                responseV5.incidents[0].error.code)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN),\
            "Expected warning but found %d" % responseV5.incidents[0].health

    # Clear the error
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
            0, -35)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we've set the monitor frequency to less than 35 seconds - that will make us around
    # half or less of the 60 seconds we give the data before calling it stale.
    cmFieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, dcgm_fields.DCGM_FE_GPU, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING)
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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_THERMAL
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuIds[0],
                                                       dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, 0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    # we expect that there will be no data here
    #assert (dcgm_structs.DCGM_ST_OK == result or dcgm_structs.DCGM_ST_NO_DATA == result)

    # inject an error into thermal
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuIds[0],
                                                       dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, 1000, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuIds[0])
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_CLOCK_THROTTLE_THERMAL)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_CLOCKS_EVENT_THERMAL)

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
        logger.error("bad response.overallHealth %d. Are these GPUs really healthy?" % response.overallHealth)
        test_utils.skip_test("bad response.overallHealth %d. Are these GPUs really healthy?" % response.overallHealth)
    if response.incidentCount > 0:
        numErrors += 1
        logger.error("bad response.incidentCount %d > 0" % (response.incidentCount))

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

    #This will throw an exception on error
    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    #Check that our response comes back clean
    helper_check_health_response_v4(gpuIds, responseV5)

################ Start health sanity checks
# The health sanity checks verify that that the DCGM health checks return healthy for all GPUs on live systems.
# Note: These tests can fail if a GPU is really unhealthy. We should give detailed feedback so that this is attributed
# to the GPU and not the test

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_pcie_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_mem_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_MEM)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_inforom_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_INFOROM)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_thermal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_power_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_POWER)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_nvlink_standalone(handle, gpuIds):
    #We will get false failures if any nvlinks are down on the GPUs
    test_utils.skip_test_if_any_nvlinks_down(handle)
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_nvswitch_nonfatal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_NONFATAL)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_nvswitch_fatal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_FATAL)

################ End health sanity checks

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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_POWER
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
            0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    skip_test_if_unhealthy(groupObj)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    # we expect that there will be no data here

    # inject an error into power
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
            1000, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuIds[0])
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_CLOCK_THROTTLE_POWER)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_CLOCKS_EVENT_POWER)

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
    #We will get false failures if any nvlinks are down on the GPUs
    test_utils.skip_test_if_any_nvlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # we expect that there will be no data here
    # inject an error into NV Link
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       100, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_NVLINK_ERROR_THRESHOLD)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN)

def helper_nvlink_check_fatal_errors(handle, gpuIds):
    test_utils.skip_test_if_any_nvlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                                       0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                                       1, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_NVLINK_ERROR_CRITICAL)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    # Trigger a failure by having more than 100 CRC errors per second
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       1000000, -20)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_dcgm_standalone_nvlink_crc_threshold(handle, gpuIds):
    helper_nvlink_crc_fatal_threshold(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_dcgm_embedded_nvlink_crc_threshold(handle, gpuIds):
    helper_nvlink_crc_fatal_threshold(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_standalone_health_large_groupid(handle, gpuIds):
    """
    Verifies that a health check can run on a large groupId
    This verifies the fix for bug 1868821
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    #Make a bunch of groups and delete them right away so our next groupId is large
    for i in range(100):
        groupObj = systemObj.GetEmptyGroup("test_group_%d" % i)
        groupObj.Delete()

    groupObj = systemObj.GetEmptyGroup("test_good_group")
    groupObj.AddGpu(gpuIds[0])
    groupId = groupObj.GetId().value

    assert groupId >= 100, "Expected groupId > 100. got %d" % groupObj.GetId()

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_ALL

    #Any of these will throw an exception on error. Making it past these = success
    groupObj.health.Set(newSystems)
    systemObj.UpdateAllFields(True)
    groupObj.health.Get()
    groupObj.health.Check()


def helper_health_check_nvswitch_errors(handle, switchIds, fieldId, healthSystem, healthResult, errorCode):
    """
    Verifies that a check error occurs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    #This test will fail if any NvLinks are down
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
    field.ts = int((time.time()-5) * 1000000.0)
    field.value.i64 = 0

    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH,
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    field.ts = int((time.time()-50) * 1000000.0)
    field.value.i64 = 0

    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH,
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # we expect that there will be no data here
    # inject an error into NvSwitch
    field.ts = int((time.time() - 1) * 1000000.0) # set the injected data for a second ago
    field.value.i64 = 5

    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH,
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_SWITCH)
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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    #Set all links of our injected GPU to Up
    for linkId in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU):
        dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId, dcgm_structs.DcgmNvLinkLinkStateUp)

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
    groupObj.health.Set(newSystems)

    #By default, the health check should pass

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert responseV5.incidentCount == 0, "Expected no errors. Got %d errors" % responseV5.incidentCount

    #Set a link to Down
    linkId = 3
    dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId, dcgm_structs.DcgmNvLinkLinkStateDown)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    logger.info("Health String: " + responseV5.incidents[0].error.msg)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_NVLINK)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_NVLINK_DOWN)
    assert str(linkId) in (responseV5.incidents[0].error.msg), "Didn't find linkId %d in %s" % (linkId, responseV5.incidents[0].error.msg)

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

    #By default, the health check should pass
    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    assert responseV5.incidentCount == 0, "Expected no errors. Got %d entities with errors: %s" % (responseV5.incidentCount, responseV5.incidents[0].error.msg)

    #Set a link to Down
    dcgm_agent_internal.dcgmSetEntityNvLinkLinkState(handle, dcgm_fields.DCGM_FE_SWITCH, switchId, linkId, dcgm_structs.DcgmNvLinkLinkStateDown)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == switchId)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_SWITCH)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_FATAL)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_NVLINK_DOWN)
    assert str(linkId) in responseV5.incidents[0].error.msg, "Didn't find linkId %d in %s" % (linkId, responseV5.incidents[0].error.msg)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_nvswitches()
def test_health_check_nvlink_link_down_nvswitch_standalone(handle, switchIds):
    helper_health_check_nvlink_link_down_nvswitch(handle, switchIds)

def helper_health_check_multiple_failures(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
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

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.incidentCount == 2)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[1].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[1].entityInfo.entityId == gpuId)

    if responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM:
        # The memory error is in position 0 here
        assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)

        # PCIE error is in position 1 here
        assert (responseV5.incidents[1].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
        assert (responseV5.incidents[1].error.code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)
    else:
        assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
        assert (responseV5.incidents[1].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)

        # Mem is in position 1 now
        assert (responseV5.incidents[1].error.code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)
        assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_injection_gpus()
def test_health_check_standalone_multiple_failures(handle, gpuIds):
    helper_health_check_multiple_failures(handle, gpuIds)

def helper_health_check_unreadable_power_usage(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_POWER
    groupObj.health.Set(newSystems)

    ret = dcgm_field_injection_helpers.inject_field_value_fp64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
            dcgmvalue.DCGM_FP64_BLANK, 50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)

    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_POWER_UNREADABLE)

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
        cmfi = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, dcgm_fields.DCGM_FE_GPU, fieldId)
        assert cmfi.flags & dcgm_structs_internal.DCGM_CMI_F_WATCHED, "x%X" % cmfi.flags
        assert cmfi.monitorIntervalUsec == watchInterval, "%d != %d" % (cmfi.monitorIntervalUsec, watchInterval)
        assert cmfi.maxAgeUsec == maxKeepAgeUsec, "%d != %d" % (cmfi.maxAgeUsec, maxKeepAgeUsec)

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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
            95, 0) # set the injected data to now
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_UNCONTAINED_ERROR)

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
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    skip_test_if_unhealthy(groupObj)

    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ROW_REMAP_FAILURE,
            1, 0) # set the injected data to now
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == gpuId)
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_ROW_REMAP_FAILURE)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus(2)
def test_dcgm_health_check_row_remap_failure(handle, gpuIds):
    helper_test_dcgm_health_check_row_remap_failure(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_cpus(1)
@test_utils.run_with_injection_cpu_cores(1)
def test_dcgm_health_cpu_thermal(handle, cpuIds, coreIds):
    dcgmHandle               = pydcgm.DcgmHandle(handle=handle)
    entityPair               = dcgm_structs.c_dcgmGroupEntityPair_t()
    entityPair.entityGroupId = dcgm_fields.DCGM_FE_CPU
    entityPair.entityId      = cpuIds[0]
    systemObj = dcgmHandle.GetSystem()
    dcgmGroup = systemObj.GetEmptyGroup("test1")
    dcgmGroup.AddEntity(entityPair.entityGroupId, entityPair.entityId)
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)

    injection_info = [ [ dcgm_fields.DCGM_FI_DEV_CPU_TEMP_CURRENT, 150.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_TEMP_WARNING, 100.1 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_TEMP_CRITICAL, 110.1 ],
    ]
    for ii in injection_info:
        dcgm_field_injection_helpers.inject_value(handle, entityPair.entityId, ii[0],
                               ii[1], -10, verifyInsertion=True,
                               entityType=entityPair.entityGroupId)
    for ii in injection_info:
        dcgm_field_injection_helpers.inject_value(handle, entityPair.entityId, ii[0],
                               ii[1], 50, verifyInsertion=True,
                               entityType=entityPair.entityGroupId)

    responseV5 = dcgmGroup.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 2)
    assert (responseV5.incidents[0].entityInfo.entityId == cpuIds[0])
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_CPU)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_FIELD_THRESHOLD_DBL)
    assert (responseV5.incidents[1].entityInfo.entityId == cpuIds[0])
    assert (responseV5.incidents[1].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_CPU)
    assert (responseV5.incidents[1].system == dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)
    assert (responseV5.incidents[1].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[1].error.code == dcgm_errors.DCGM_FR_FIELD_THRESHOLD_DBL)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_cpus(1)
@test_utils.run_with_injection_cpu_cores(1)
def test_dcgm_health_cpu_power(handle, cpuIds, coreIds):
    dcgmHandle               = pydcgm.DcgmHandle(handle=handle)
    entityPair               = dcgm_structs.c_dcgmGroupEntityPair_t()
    entityPair.entityGroupId = dcgm_fields.DCGM_FE_CPU
    entityPair.entityId      = cpuIds[0]
    systemObj = dcgmHandle.GetSystem()
    dcgmGroup = systemObj.GetEmptyGroup("test1")
    dcgmGroup.AddEntity(entityPair.entityGroupId, entityPair.entityId)
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_POWER)

    injection_info = [ [ dcgm_fields.DCGM_FI_DEV_CPU_POWER_LIMIT, 150.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT, 150.1 ],
    ]
    for ii in injection_info:
        dcgm_field_injection_helpers.inject_value(handle, entityPair.entityId, ii[0],
                               ii[1], 5, verifyInsertion=True,
                               entityType=entityPair.entityGroupId)

    responseV5 = dcgmGroup.health.Check(dcgm_structs.dcgmHealthResponse_version5)
    assert (responseV5.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidentCount == 1)
    assert (responseV5.incidents[0].entityInfo.entityId == cpuIds[0])
    assert (responseV5.incidents[0].entityInfo.entityGroupId == dcgm_fields.DCGM_FE_CPU)
    assert (responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV5.incidents[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_FIELD_THRESHOLD_DBL)
