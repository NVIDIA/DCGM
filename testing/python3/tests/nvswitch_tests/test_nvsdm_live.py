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
import pydcgm
import dcgm_agent
from . import nvsdm_helpers
import dcgm_structs
import dcgm_structs_internal
import dcgm_fields
import dcgm_structs
import test_utils
import time
import DcgmReader

@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_nvsdm_live()
def test_nvsdm_live_list_switches(handle):
    """
    Try to list switches using the NVSDM stub
    """
    flags = 0
    switchIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_SWITCH, flags)

    assert len(switchIds) > 0

@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_nvsdm_live()
@test_utils.run_only_with_nvml()
def test_nvsdm_live_port_telemetry(handle):
    """
    Read port telemetry from NVSDM
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    nvsdmPortLID = 1
    numOfNvsdmStubbedPorts = 2

    flags = 0
    linkIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_LINK, flags)
    linkStates = dcgm_agent.dcgmGetNvLinkLinkStatus(handle)
    assert len(linkIds) > 0

    linkEntities = []
    for link in linkIds:
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t()
        entityPair.entityGroupId = dcgm_fields.DCGM_FE_LINK
        entityPair.entityId = link
        linkEntities.append(entityPair)

    dcgmGroup = dcgmSystem.GetGroupWithEntities('linkGroup', linkEntities)

    fieldIds = [ dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX,
                 dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX,
                 dcgm_fields.DCGM_FI_DEV_NVSWITCH_DEVICE_UUID,
               ]

    updateFrequencyUsec = 200000 # 200ms
    sleepTime = updateFrequencyUsec / 1000000 * 2 # Convert to seconds and sleep twice as long; ensures fresh sample

    dr = DcgmReader.DcgmReader(fieldIds=fieldIds, updateFrequency=updateFrequencyUsec, maxKeepAge=30.0, entities=linkEntities)
    dr.SetHandle(handle)

    for i in range(5):
        time.sleep(sleepTime)

        linkLatest = dr.GetLatestEntityValuesAsFieldIdDict()[dcgm_fields.DCGM_FE_LINK]

        for linkId in linkIds:
            if not helper_is_active_link(linkId, linkStates):
                continue
            if len(linkLatest[linkId]) != len(fieldIds):
                missingFieldIds = []
                extraFieldIds = []
                for fieldId in fieldIds:
                    if fieldId not in linkLatest[linkId]:
                        missingFieldIds.append(fieldId)

                for fieldId in linkLatest[linkId]:
                    if fieldId not in fieldIds:
                        extraFieldIds.append(fieldId)

                errmsg = "i=%d, linkId %d, len %d != %d" % (i, linkId, len(linkLatest[linkId]), len(fieldIds))
                if len(missingFieldIds) > 0:
                    errmsg = errmsg + " Link is missing entries for fields %s" % str(missingFieldIds)
                if len(extraFieldIds) > 0:
                    errmsg = errmsg + " Link has extra entries for fields %s" % str(extraFieldIds)

                assert len(linkLatest[linkId]) == len(fieldIds), errmsg

            # Verify the GUID for each NvLink
            receivedGUID = linkLatest[linkId][dcgm_fields.DCGM_FI_DEV_NVSWITCH_DEVICE_UUID]
            nvsdmPortNum = nvsdmPortId = linkId % numOfNvsdmStubbedPorts
            assert receivedGUID != "", "Received empty GUID"

@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_nvsdm_live()
def test_nvsdm_live_platform_telemetry(handle):
    """
    Read platform telemetry from NVSDM
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    nvsdmDeviceTypeSwitch    = 2
    nvsdmSwitchVendorID      = 0xbaca

    flags = 0
    switchIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_SWITCH, flags)
    assert len(switchIds) > 0

    switchEntities = []
    for switch in switchIds:
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t()
        entityPair.entityGroupId = dcgm_fields.DCGM_FE_SWITCH
        entityPair.entityId = switch
        switchEntities.append(entityPair)

    dcgmGroup = dcgmSystem.GetGroupWithEntities('switchGroup', switchEntities)

    fieldIds = [dcgm_fields.DCGM_FI_DEV_NVSWITCH_POWER_VDD,
                 dcgm_fields.DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT,
                 dcgm_fields.DCGM_FI_DEV_NVSWITCH_DEVICE_UUID,
               ]

    updateFrequencyUsec = 200000 # 200ms
    sleepTime = updateFrequencyUsec / 1000000 * 2 # Convert to seconds and sleep twice as long; ensures fresh sample

    dr = DcgmReader.DcgmReader(fieldIds=fieldIds, updateFrequency=updateFrequencyUsec, maxKeepAge=30.0, entities=switchEntities)
    dr.SetHandle(handle)

    for i in range(5):
        time.sleep(sleepTime)

        switchLatest = dr.GetLatestEntityValuesAsFieldIdDict()[dcgm_fields.DCGM_FE_SWITCH]

        for switchId in switchIds:
            if len(switchLatest[switchId]) != len(fieldIds):
                missingFieldIds = []
                extraFieldIds = []
                for fieldId in fieldIds:
                    if fieldId not in switchLatest[switchId]:
                        missingFieldIds.append(fieldId)

                for fieldId in switchLatest[switchId]:
                    if fieldId not in fieldIds:
                        extraFieldIds.append(fieldId)

                errmsg = "i=%d, switchId %d, len %d != %d" % (i, switchId, len(switchLatest[switchId]), len(fieldIds))
                if len(missingFieldIds) > 0:
                    errmsg = errmsg + " Switch is missing entries for fields %s" % str(missingFieldIds)
                if len(extraFieldIds) > 0:
                    errmsg = errmsg + " Switch has extra entries for fields %s" % str(extraFieldIds)

                assert len(switchLatest[switchId]) == len(fieldIds), errmsg

            # Verify the GUID for each NvSwitch.
            receivedGUID = switchLatest[switchId][dcgm_fields.DCGM_FI_DEV_NVSWITCH_DEVICE_UUID]
            assert receivedGUID != "", "Received empty GUID"

@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_nvsdm_live()
def test_nvsdm_live_nvlink_status(handle):
    """
    Get nvlink status for each nvswitch using NVSDM stub and assert if it's down
    """
    linkStatus = dcgm_agent.dcgmGetNvLinkLinkStatus(handle)

    for i in range(linkStatus.numNvSwitches):
        for j in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH):
            if linkStatus.nvSwitches[i].linkState[j] == dcgm_structs.DcgmNvLinkLinkStateDown:
                errmsg = "LinkStatus for NvSwitch=%d's link=%d is down." % (i, j)
                assert(linkStatus.nvSwitches[i].linkState[j] != dcgm_structs.DcgmNvLinkLinkStateDown), errmsg

@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_nvsdm_live()
def test_nvsdm_live_nvswitch_health_check(handle):
    """
    Test NvSwitch health check and assert if overall health result doesn't pass.
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)

    dcgmGroup = pydcgm.DcgmGroup(dcgmHandle, groupName="switchGroup")
    switchIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_SWITCH, 0)
    assert len(switchIds) > 0

    for switchId in switchIds:
        dcgmGroup.AddEntity(dcgm_fields.DCGM_FE_SWITCH, switchId)

    # Add the health watches
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_NVSWITCH_FATAL)

    # Invoke Health check
    group_health = dcgmGroup.health.Check()
    assert(group_health.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)

@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_nvsdm_live()
def test_nvsdm_live_composite_field_telemetry(handle):
    """
    Read composite field telemetry from NVSDM
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    flags = 0
    switchIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_SWITCH, flags)
    assert len(switchIds) > 0

    switchEntities = []
    for switch in switchIds:
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t()
        entityPair.entityGroupId = dcgm_fields.DCGM_FE_SWITCH
        entityPair.entityId = switch
        switchEntities.append(entityPair)

    dcgmGroup = dcgmSystem.GetGroupWithEntities('switchGroup', switchEntities)

    fieldIds = [ dcgm_fields.DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX,
                 dcgm_fields.DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX,
               ]

    updateFrequencyUsec = 200000 # 200ms
    sleepTime = updateFrequencyUsec / 1000000 * 2 # Convert to seconds and sleep twice as long; ensures fresh sample

    dr = DcgmReader.DcgmReader(fieldIds=fieldIds, updateFrequency=updateFrequencyUsec, maxKeepAge=30.0, entities=switchEntities)
    dr.SetHandle(handle)

    for i in range(5):
        time.sleep(sleepTime)

        switchLatest = dr.GetLatestEntityValuesAsFieldIdDict()[dcgm_fields.DCGM_FE_SWITCH]

        for switchId in switchIds:
            if len(switchLatest[switchId]) != len(fieldIds):
                missingFieldIds = []
                extraFieldIds = []
                for fieldId in fieldIds:
                    if fieldId not in switchLatest[switchId]:
                        missingFieldIds.append(fieldId)

                for fieldId in switchLatest[switchId]:
                    if fieldId not in fieldIds:
                        extraFieldIds.append(fieldId)

                errmsg = "i=%d, switchId %d, len %d != %d" % (i, switchId, len(switchLatest[switchId]), len(fieldIds))
                if len(missingFieldIds) > 0:
                    errmsg = errmsg + " Switch is missing entries for fields %s" % str(missingFieldIds)
                if len(extraFieldIds) > 0:
                    errmsg = errmsg + " Switch has extra entries for fields %s" % str(extraFieldIds)

                assert len(switchLatest[switchId]) == len(fieldIds), errmsg

def helper_is_active_link(linkId, linkStates):
    for i in range(linkStates.numNvSwitches):
        if linkStates.nvSwitches[i].linkState[linkId - 1] != dcgm_structs_internal.DcgmEntityStatusOk:
            return False
    return True

@test_utils.run_with_standalone_host_engine(initializedClient=True)
@test_utils.run_only_with_nvsdm_live()
def test_nvsdm_live_pause_resume(handle):
    nvsdm_helpers.helper_nvsdm_pause_resume(handle)