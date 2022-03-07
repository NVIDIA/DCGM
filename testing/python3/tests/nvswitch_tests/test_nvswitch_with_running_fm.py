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
import pydcgm
import dcgm_field_helpers
import dcgm_fields
import dcgm_structs
import shlex
import time
import logger
import subprocess
import test_utils
from . import test_nvswitch_utils

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_nvswitches()
def test_nvswitch_traffic_p2p(handle, switchIds):
    """
    Verifies that fabric can pass p2p read and write traffic successfully
    """

    test_utils.skip_test("Bandwidth field not being updated yet")

    # TX_0 and RX_0 on port 0
    nvSwitchBandwidth0FieldIds = []
    for i in range(dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_0_P00,
                   dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_RX_0_P00 + 1, 1):
        nvSwitchBandwidth0FieldIds.append(i)

    # TX_1 and RX_1 on port 0
    nvSwitchBandwidth1FieldIds = []
    for i in range(dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_1_P00,
                   dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_RX_1_P00 + 1, 1):
        nvSwitchBandwidth1FieldIds.append(i)

    dcgmHandle = pydcgm.DcgmHandle(ipAddress="127.0.0.1")

    groupName = "test_nvswitches"
    allNvSwitchesGroup = pydcgm.DcgmGroup(dcgmHandle, groupName=groupName,
                                          groupType=dcgm_structs.DCGM_GROUP_DEFAULT_NVSWITCHES)

    fgName = "test_nvswitches_bandwidth0"
    nvSwitchBandwidth0FieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, name=fgName,
                                                         fieldIds=nvSwitchBandwidth0FieldIds)

    fgName = "test_nvswitches_bandwidth1"
    nvSwitchBandwidth1FieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, name=fgName,
                                                         fieldIds=nvSwitchBandwidth1FieldIds)

    updateFreq = int(20 / 2.0) * 1000000
    maxKeepAge = 600.0
    maxKeepSamples = 0

    nvSwitchBandwidth0Watcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
        dcgmHandle.handle, allNvSwitchesGroup.GetId(),
        nvSwitchBandwidth0FieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
        updateFreq, maxKeepAge, maxKeepSamples, 0)
    nvSwitchBandwidth1Watcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
        dcgmHandle.handle, allNvSwitchesGroup.GetId(),
        nvSwitchBandwidth1FieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
        updateFreq, maxKeepAge, maxKeepSamples, 0)

    # wait for FM reports and populates stats
    time.sleep(30)

    # read the counters before sending traffic
    nvSwitchBandwidth0Watcher.GetMore()
    nvSwitchBandwidth1Watcher.GetMore()

    for entityGroupId in list(nvSwitchBandwidth0Watcher.values.keys()):
        for entityId in nvSwitchBandwidth0Watcher.values[entityGroupId]:
            bandwidth0FieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_0_P00
            bandwidth1FieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_1_P00

            counter0TxBefore = nvSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[
                -1].value
            bandwidth0FieldId += 1
            counter0RxBefore = nvSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[
                -1].value
            counter1TxBefore = nvSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[
                -1].value
            bandwidth1FieldId += 1
            counter1RxBefore = nvSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[
                -1].value

    # Generate write traffic for the nvswitches
    test_utils.run_p2p_bandwidth_app(test_nvswitch_utils.MEMCPY_DTOD_WRITE_CE_BANDWIDTH)

    # Generate read traffic for the nvswitches
    test_utils.run_p2p_bandwidth_app(test_nvswitch_utils.MEMCPY_DTOD_READ_CE_BANDWIDTH)

    # read the counters again after sending traffic
    nvSwitchBandwidth0Watcher.GetMore()
    nvSwitchBandwidth1Watcher.GetMore()

    for entityGroupId in list(nvSwitchBandwidth0Watcher.values.keys()):
        for entityId in nvSwitchBandwidth0Watcher.values[entityGroupId]:
            bandwidth0FieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_0_P00
            bandwidth1FieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_1_P00

            counter0TxAfter = nvSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[
                -1].value
            bandwidth0FieldId += 1
            counter0RxAfter = nvSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[
                -1].value
            counter1TxAfter = nvSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[
                -1].value
            bandwidth1FieldId += 1
            counter1RxAfter = nvSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[
                -1].value

    assert counter0TxAfter > counter0TxBefore, "Counter0Tx did not increase"
    assert counter0RxAfter > counter0RxBefore, "counter0Rx did not increase"
    assert counter1TxAfter > counter1TxBefore, "Counter1Tx did not increase"
    assert counter1RxAfter > counter1RxBefore, "counter1Rx did not increase"


