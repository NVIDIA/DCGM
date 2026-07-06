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

import dcgm_agent_internal
import dcgm_fields
import dcgm_agent
import dcgm_structs
import DcgmReader
import logger
import test_utils
import time


def helper_nvsdm_pause_resume(handle):
    flags = 0
    switchIds = dcgm_agent.dcgmGetEntityGroupEntities(
        handle, dcgm_fields.DCGM_FE_SWITCH, flags)
    ibCxIds = dcgm_agent.dcgmGetEntityGroupEntities(
        handle, dcgm_fields.DCGM_FE_CONNECTX, flags)
    portIds = dcgm_agent.dcgmGetEntityGroupEntities(
        handle, dcgm_fields.DCGM_FE_LINK, flags)

    # Exceptions are checked inside the `dcgmPauseTelemetryForDiag` and
    # `dcgmResumeTelemetryForDiag` functions.
    dcgm_agent_internal.dcgmPauseTelemetryForDiag(handle)
    dcgm_agent_internal.dcgmResumeTelemetryForDiag(handle)

    newSwitchIds = dcgm_agent.dcgmGetEntityGroupEntities(
        handle, dcgm_fields.DCGM_FE_SWITCH, flags)
    newIbCxIds = dcgm_agent.dcgmGetEntityGroupEntities(
        handle, dcgm_fields.DCGM_FE_CONNECTX, flags)
    newPortIds = dcgm_agent.dcgmGetEntityGroupEntities(
        handle, dcgm_fields.DCGM_FE_LINK, flags)

    def assertIdListIsTheSame(old, new, name):
        assert len(old) == len(new), (
            f"The length of {name} does not match: "
            f"old length is [{len(old)}], "
            f"new length is [{len(new)}].")
        for i in range(len(old)):
            assert old[i] == new[i], (
                f"The index [{i}] of {name} does not match: "
                f"old: [{old[i]}], new: [{new[i]}]")

    assertIdListIsTheSame(switchIds, newSwitchIds, "Switch")
    assertIdListIsTheSame(ibCxIds, newIbCxIds, "CX")
    assertIdListIsTheSame(portIds, newPortIds, "Port")


_SWITCH_TOPOLOGY_FIELDS_ALL = [
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_PHYSICAL_ID,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_PCIE_BUS,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION,
]

_LINK_TOPOLOGY_FIELDS_ALL = [
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_ID,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_STATUS,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_TYPE,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_ID,
    dcgm_fields.DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID,
]


def helper_nvsdm_switch_topology_fields(handle, fieldIds=None):
    """
    Read switch-level topology/info fields and verify all fields are present
    and non-None. Defaults to all fields (863-864, 866-869, 1524). Pass a custom
    fieldIds list to restrict to a subset (e.g. mock tests that lack PCI info).
    """
    if fieldIds is None:
        fieldIds = _SWITCH_TOPOLOGY_FIELDS_ALL

    switchIds = dcgm_agent.dcgmGetEntityGroupEntities(
        handle, dcgm_fields.DCGM_FE_SWITCH, 0)
    if len(switchIds) == 0:
        test_utils.skip_test(
            "No NvSwitches found, skipping topology field test.")

    switchEntities = []
    for switch in switchIds:
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t()
        entityPair.entityGroupId = dcgm_fields.DCGM_FE_SWITCH
        entityPair.entityId = switch
        switchEntities.append(entityPair)

    updateFrequencyUsec = 200000  # 200ms
    dr = DcgmReader.DcgmReader(
        fieldIds=fieldIds,
        updateFrequency=updateFrequencyUsec,
        maxKeepAge=30.0,
        entities=switchEntities)
    dr.SetHandle(handle)

    time.sleep(updateFrequencyUsec / 1000000 * 2)

    switchLatest = dr.GetLatestEntityValuesAsFieldIdDict()[
        dcgm_fields.DCGM_FE_SWITCH]

    for switchId in switchIds:
        assert switchId in switchLatest, f"Switch {switchId} not in results"
        logger.info(f"Switch {switchId} topology fields:")
        for fieldId in fieldIds:
            assert fieldId in switchLatest[switchId], \
                f"Field {fieldId} missing for switch {switchId}"
            value = switchLatest[switchId][fieldId]
            logger.info(f"  Field {fieldId}: {value}")
            assert value is not None, \
                f"Field {fieldId} returned None for switch {switchId}"


def helper_nvsdm_link_topology_fields(handle, fieldIds=None):
    """
    Read link-level topology/info fields and verify all fields are present
    and non-None. Defaults to all fields (865, 870-877). Pass a custom
    fieldIds list to restrict to a subset (e.g. mock tests that lack remote
    device info).
    """
    if fieldIds is None:
        fieldIds = _LINK_TOPOLOGY_FIELDS_ALL

    linkIds = dcgm_agent.dcgmGetEntityGroupEntities(
        handle, dcgm_fields.DCGM_FE_LINK, 0)
    if len(linkIds) == 0:
        test_utils.skip_test("No NvLinks found, skipping topology field test.")

    linkEntities = []
    for link in linkIds:
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t()
        entityPair.entityGroupId = dcgm_fields.DCGM_FE_LINK
        entityPair.entityId = link
        linkEntities.append(entityPair)

    updateFrequencyUsec = 200000  # 200ms

    dr = DcgmReader.DcgmReader(
        fieldIds=fieldIds,
        updateFrequency=updateFrequencyUsec,
        maxKeepAge=30.0,
        entities=linkEntities,
        ignoreBlank=False)
    dr.SetHandle(handle)

    time.sleep(updateFrequencyUsec / 1000000 * 2)

    linkLatest = dr.GetLatestEntityValuesAsFieldIdDict()[
        dcgm_fields.DCGM_FE_LINK]

    for linkId in linkIds:
        assert linkId in linkLatest, f"Link {linkId} not in results"
        logger.info(f"Link {linkId} topology fields:")
        for fieldId in fieldIds:
            assert fieldId in linkLatest[linkId], \
                f"Field {fieldId} missing for link {linkId}"
            value = linkLatest[linkId][fieldId]
            logger.info(f"  Field {fieldId}: {value}")
            assert value is not None, \
                f"Field {fieldId} returned None for link {linkId}"
