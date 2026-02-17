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

import os
import pydcgm
import dcgm_structs_internal
import dcgm_structs
import dcgm_agent
import subprocess
import shutil
import test_utils
import apps
import logger
import time
import re
import DcgmGroup
import dcgm_fields
import dcgm_agent_internal
import skip_test_helpers
import DcgmDiag
import threading
import dcgm_internal_helpers
import nvidia_smi_utils
import dcgm_field_injection_helpers
import dcgm_errors
import dcgmvalue
import nvml_injection
import queue
import nvml_injection
import nvml_injection_structs
import dcgm_nvml
import DcgmSystem
from datetime import datetime
from ctypes import CFUNCTYPE, POINTER, c_uint64, memmove, addressof
from _test_helpers import skip_test_if_no_dcgm_nvml
from ctypes import byref


def in_nvml_injection_mode():
    return os.getenv(test_utils.INJECTION_MODE_VAR, default='False') == 'True'


def trigger_bind_gpu_event(uuids=None, pcibusIds=None):
    '''
    Trigger the bind GPU event.
    Args:
        uuids: The UUIDs of the GPUs to bind. Must be provided in NVML injection mode.
        pcibusIds: The PCI bus IDs of the GPUs to bind. Must be provided in regular mode.
    '''
    if in_nvml_injection_mode():
        if uuids is None:
            test_utils.skip_test(
                "UUIDs for to be attached GPUs are required in NVML injection mode")

        path = os.path.join(test_utils.NVML_INJECTION_FOLDER, "sys-event.yaml")
        with open(path, "w") as f:
            f.write("bind:\n")
            for uuid in uuids:
                f.write(f"  - {uuid}\n")
    else:
        if pcibusIds is None:
            test_utils.skip_test(
                "PCI bus IDs for to be bound GPUs are required in regular mode")
        test_utils.attach_gpus(pcibusIds)


def trigger_unbind_gpu_event(uuids=None, pcibusIds=None):
    '''
    Trigger the unbind GPU event.
    Args:
        uuids: The UUIDs of the GPUs to unbind. Must be provided in NVML injection mode.
        pcibusIds: The PCI bus IDs of the GPUs to unbind. Must be provided in regular mode.
    '''
    if in_nvml_injection_mode():
        if uuids is None:
            test_utils.skip_test(
                "UUIDs for to be detached GPUs are required in NVML injection mode")

        path = os.path.join(test_utils.NVML_INJECTION_FOLDER, "sys-event.yaml")
        with open(path, "w") as f:
            f.write("unbind:\n")
            for uuid in uuids:
                f.write(f"  - {uuid}\n")
    else:
        if pcibusIds is None:
            test_utils.skip_test(
                "PCI bus IDs for to be detached GPUs are required in regular mode")
        test_utils.detach_gpus(pcibusIds)


def get_gpu_uuid(dcgmSystem, gpuId):
    gpuAttributes = dcgmSystem.discovery.GetGpuAttributes(gpuId)
    return gpuAttributes.identifiers.uuid


def wait_gpu_status(dcgmSystem, gpuId, expectedStatus):
    retryTimes = 64
    sleepTime = 0.1
    if not in_nvml_injection_mode():
        # In regular mode, host engine takes longer to attach / detach the GPUs.
        retryTimes = 512
    for _ in range(retryTimes):
        gpuStatus = dcgmSystem.discovery.GetGpuStatus(gpuId)
        if gpuStatus == expectedStatus:
            break
        time.sleep(sleepTime)
    else:
        raise RuntimeError(
            f"Expected GPU {gpuId} to be {expectedStatus}, but it is not")


def bind_gpu_and_wait_for_status_updated(dcgmSystem, gpuIds, gpuUuids=None, pcibusIds=None):
    '''
    Bind the GPUs and wait for the status to be updated.
    It will correctly determine whether NVML Injection mode is used and apply the appropriate parameters to trigger the event.
    Args:
        dcgmSystem: The DCGM system.
        gpuIds: The IDs of the GPUs to bind.
        gpuUuids: The UUIDs of the GPUs to bind. Must be provided in NVML injection mode.
        pcibusIds: The PCI bus IDs of the GPUs to bind. Must be provided in regular mode.
    '''
    if in_nvml_injection_mode():
        if gpuUuids is None:
            test_utils.skip_test(
                "UUIDs for to be bound GPUS are required in NVML injection mode")
        if len(gpuIds) != len(gpuUuids):
            raise RuntimeError(
                f"Expected GPU IDs and UUIDs to have the same length, but got {len(gpuIds)} and {len(gpuUuids)}")
    else:
        if pcibusIds is None:
            test_utils.skip_test(
                "PCI bus IDs for to be bound GPUS are required in regular mode")
        if len(gpuIds) != len(pcibusIds):
            raise RuntimeError(
                f"Expected GPU IDs and PCI bus IDs to have the same length, but got {len(gpuIds)} and {len(pcibusIds)}")

    now = datetime.now().timestamp() * 1000000
    now = int(now)
    trigger_bind_gpu_event(gpuUuids, pcibusIds)

    # nvml system event thread will call nvmlSystemEventSetWait every 1 second,
    # we need to retry for a while (plus some buffer time) to ensure the status is updated
    retryTimes = 64
    if not in_nvml_injection_mode():
        # In regular mode, host engine takes longer to attach / detach the GPUs.
        retryTimes = 512
    for gpuId, gpuIdentifier in zip(gpuIds, gpuUuids if gpuUuids is not None else pcibusIds):
        for _ in range(retryTimes):
            gpuStatus = dcgmSystem.discovery.GetGpuStatus(gpuId)
            if gpuStatus == dcgm_structs_internal.DcgmEntityStatusOk:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                f"Expected GPU {gpuId}, {gpuIdentifier} to be ok, but it is not")
    # Tests wait for GPU status changes, but modules may still be processing in the AttachGpus function when the status becomes OK.
    # We need to check the bind/unbind event to ensure the process is completed.
    helper_check_bind_unbind_event(dcgmSystem._dcgmHandle.handle, now, {
                                   dcgm_structs.DCGM_BU_EVENT_STATE_SYSTEM_REINITIALIZATION_COMPLETED: 1})


def unbind_gpu_and_wait_for_status_updated(dcgmSystem, gpuIds, pcibusIds=None):
    '''
    Unbind the GPUs and wait for the status to be updated.
    It will correctly determine whether NVML Injection mode is used and apply the appropriate parameters to trigger the event.
    Args:
        dcgmSystem: The DCGM system.
        gpuIds: The IDs of the GPUs to unbind.
        pcibusIds: The PCI bus IDs of the GPUs to unbind. Must be provided in regular mode.
    '''

    targetGpus = None
    if in_nvml_injection_mode():
        targetGpus = [get_gpu_uuid(dcgmSystem, gpuId) for gpuId in gpuIds]
    else:
        if pcibusIds is None:
            test_utils.skip_test(
                "PCI bus IDs for to be detached GPUs are required in regular mode")
    now = datetime.now().timestamp() * 1000000
    now = int(now)

    trigger_unbind_gpu_event(targetGpus, pcibusIds)

    # nvml system event thread will call nvmlSystemEventSetWait every 1 second,
    # we need to retry for a while (plus some buffer time) to ensure the status is updated
    retryTimes = 64
    if not in_nvml_injection_mode():
        # In regular mode, host engine takes longer to detach the GPUs.
        retryTimes = 512
    for gpuId in gpuIds:
        for _ in range(retryTimes):
            gpuStatus = dcgmSystem.discovery.GetGpuStatus(gpuId)
            if gpuStatus == dcgm_structs_internal.DcgmEntityStatusDetached:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                f"Expected GPU {gpuId} to be detached, but it is not")
    # Tests wait for GPU status changes, but modules may still be processing in the AttachGpus function when the status becomes OK.
    # We need to check the bind/unbind event to ensure the process is completed.
    helper_check_bind_unbind_event(dcgmSystem._dcgmHandle.handle, now, {
                                   dcgm_structs.DCGM_BU_EVENT_STATE_SYSTEM_REINITIALIZATION_COMPLETED: 1})


def helper_run_dcgmi(args):
    """
    Helper function to run dcgmi commands.
    Returns tuple: (returnCode, stdout_lines, stderr_lines)
    """
    dcgmi = apps.DcgmiApp(args)
    dcgmi.start(250)
    retValue = dcgmi.wait()
    dcgmi.validate()
    return retValue, dcgmi.stdout_lines, dcgmi.stderr_lines


def get_gpu_section(output, gpuId):
    """Extract the section of discovery output for a specific GPU (excludes other device types)."""
    lines = output.split('\n')
    in_gpu_list = False
    gpu_start = None

    for i, line in enumerate(lines):
        if "| GPU ID |" in line:
            in_gpu_list = True
        elif in_gpu_list and " found" in line and "GPU" not in line:
            break  # Reached next device section
        elif in_gpu_list and f"| {gpuId}" in line:
            gpu_start = i
        elif gpu_start is not None and "+--------+" in line:
            return '\n'.join(lines[gpu_start:i])

    return ""


def helper_test_bind_unbind_gpu_status(handle, gpuIds, pcibusIds=None):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    targetGpus = None
    if in_nvml_injection_mode():
        targetGpus = [get_gpu_uuid(dcgmSystem, gpuId) for gpuId in gpuIds]
    trigger_unbind_gpu_event(targetGpus, pcibusIds)

    # nvml system event thread will call nvmlSystemEventSetWait every 1 second,
    # we need to retry for a while (plus some buffer time) to ensure the status is updated
    retryTimes = 64
    if not in_nvml_injection_mode():
        # In regular mode, host engine takes longer to attach / detach the GPUs.
        retryTimes = 512
    for gpuId in gpuIds:
        for _ in range(retryTimes):
            gpuStatus = dcgmSystem.discovery.GetGpuStatus(gpuId)
            if gpuStatus == dcgm_structs_internal.DcgmEntityStatusDetached:
                break
            time.sleep(0.1)
        else:
            assert False, f"Expected GPU {gpuId} to be detached, but it is not"

    backedGpus = []
    for gpuId, gpuIdentifier in zip(gpuIds, targetGpus if targetGpus is not None else pcibusIds):
        if in_nvml_injection_mode():
            trigger_bind_gpu_event(uuids=[gpuIdentifier], pcibusIds=None)
        else:
            trigger_bind_gpu_event(uuids=None, pcibusIds=[gpuIdentifier])
        for _ in range(retryTimes):
            gpuStatus = dcgmSystem.discovery.GetGpuStatus(gpuId)
            if gpuStatus == dcgm_structs_internal.DcgmEntityStatusOk:
                backedGpus.append(gpuId)
                break
            time.sleep(0.1)
        else:
            assert False, f"Expected GPU {gpuId} to be ok, but it is not"

        for id in gpuIds:
            gpuStatus = dcgmSystem.discovery.GetGpuStatus(id)
            if id in backedGpus:
                assert gpuStatus == dcgm_structs_internal.DcgmEntityStatusOk, f"Expected GPU {id} to be ok, but it is not"
            else:
                # Other GPUs should still be in unbind status.
                assert gpuStatus == dcgm_structs_internal.DcgmEntityStatusDetached, f"Expected GPU {id} to be detached, but it is not"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_bind_unbind_gpu_status(handle, gpuIds):
    helper_test_bind_unbind_gpu_status(handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_bind_unbind_gpu_status_live(handle, gpuIds, pcibusIds):
    helper_test_bind_unbind_gpu_status(handle, gpuIds, pcibusIds)


def helper_test_bind_unbind_gpu_index_order_remain_the_same(handle, gpuIds, pcibusIds=None):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    targetGpuId = gpuIds[0]
    targetGpuUuid = get_gpu_uuid(dcgmSystem, targetGpuId)
    targetGpuPciBusId = [pcibusIds[targetGpuId]
                         ] if pcibusIds is not None and targetGpuId < len(pcibusIds) else None
    originalOrder = [get_gpu_uuid(dcgmSystem, gpuId) for gpuId in gpuIds]

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [targetGpuId], targetGpuPciBusId)
    bind_gpu_and_wait_for_status_updated(dcgmSystem, [targetGpuId], [
                                         targetGpuUuid], targetGpuPciBusId)

    newGpuIds = dcgmSystem.discovery.GetAllSupportedGpuIds()
    assert len(newGpuIds) == len(gpuIds), "Expected GPU number to be the same"
    newOrder = [get_gpu_uuid(dcgmSystem, gpuId) for gpuId in newGpuIds]
    assert newOrder == originalOrder, f"Expected GPU order to be the same, but it is {newOrder}"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_bind_unbind_gpu_index_order_remain_the_same(handle, gpuIds):
    helper_test_bind_unbind_gpu_index_order_remain_the_same(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_bind_unbind_gpu_index_order_remain_the_same_live(handle, gpuIds, pcibusIds):
    helper_test_bind_unbind_gpu_index_order_remain_the_same(
        handle, gpuIds, pcibusIds)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_attach_driver_when_nvml_is_loaded(handle, gpuIds):
    try:
        dcgm_agent.dcgmAttachDriver(handle)
    except Exception as e:
        assert False, f"Expected DCGM_ST_OK, but got {e}"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_detach_driver_when_nvml_is_not_loaded(handle, gpuIds):
    # detach driver first to make sure nvml is not loaded
    dcgm_agent.dcgmDetachDriver(handle)
    try:
        dcgm_agent.dcgmDetachDriver(handle)
    except Exception as e:
        assert False, f"Expected DCGM_ST_OK, but got {e}"


def get_host_engine_pid():
    pgrep_path = shutil.which("pgrep")
    if pgrep_path is None:
        test_utils.skip_test("pgrep is not found")
    try:
        output = subprocess.check_output(
            ["pgrep", "-f", "nv-hostengine"], stderr=subprocess.STDOUT)
    except Exception as e:
        test_utils.skip_test(f"failed to get host engine pid: {e}")
    output = output.decode("utf-8")
    lines = output.splitlines()
    for pid in lines:
        try:
            with open(os.path.join('/proc', str(pid), 'stat'), 'r') as f:
                stat = f.read()
                if 'nv-hostengine' not in stat:
                    continue
                tokens = stat.split()
                if len(tokens) < 3:
                    continue
                if tokens[2] == 'Z':
                    continue
                return pid
        except Exception as e:
            continue
    test_utils.skip_test("Unexpected pgrep output: {}".format(output))


def get_lsof_of_host_engine():
    lsof_path = shutil.which("lsof")
    if lsof_path is None:
        test_utils.skip_test("lsof is not found")
    try:
        output = subprocess.check_output(
            ["lsof", "-p", str(get_host_engine_pid())], stderr=subprocess.STDOUT)
    except Exception as e:
        test_utils.skip_test(f"failed to get lsof of host engine: {e}")
    return output.decode("utf-8")


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_attach_detach_driver(handle, gpuIds):
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    dcgm_agent.dcgmDetachDriver(handle)
    lsof_output = get_lsof_of_host_engine()
    pattern = re.compile(r'.*/dev/nvidia\d+')
    assert pattern.search(
        lsof_output) is None, f"Expected /dev/nvidia to not be in lsof output {lsof_output}"

    newGpuIds = dcgmSystem.discovery.GetAllSupportedGpuIds()
    assert len(newGpuIds) == 0, "Expected GPU number to be zero"

    dcgm_agent.dcgmAttachDriver(handle)
    lsof_output = get_lsof_of_host_engine()
    assert pattern.search(
        lsof_output) is not None, f"Expected /dev/nvidia to be in lsof output {lsof_output}"

    newGpuIds = dcgmSystem.discovery.GetAllSupportedGpuIds()
    assert len(newGpuIds) == len(gpuIds), "Expected GPU number to be the same"


def helper_test_dcgm_add_inactive_gpu_to_group_will_fail(handle, gpuIds, pcibusIds=None):
    '''
    Try to add an inactive GPU to a group, and make sure it fails.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    targetUnBindGpuId = gpuIds[0]
    targetGpuPciBusId = pcibusIds[targetUnBindGpuId] if pcibusIds is not None and targetUnBindGpuId < len(
        pcibusIds) else None

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [targetUnBindGpuId], [targetGpuPciBusId])

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_add_inactive_gpu_to_group_will_fail")

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        group.AddGpu(targetUnBindGpuId)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_add_inactive_gpu_to_group_will_fail(handle, gpuIds):
    helper_test_dcgm_add_inactive_gpu_to_group_will_fail(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_add_inactive_gpu_to_group_will_fail_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_add_inactive_gpu_to_group_will_fail(
        handle, gpuIds, pcibusIds)


def helper_test_dcgm_field_watch_inactive_to_active(handle, gpuIds, pcibusIds=None):
    '''
    Watch fields on a group with a GPU that is not active, and make sure the field values are not returned for this GPU even when it backs to active.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    normal_gpu_id = gpuIds[0]
    unbind_gpu_id = gpuIds[1]
    unbind_gpu_uuid = get_gpu_uuid(dcgmSystem, unbind_gpu_id)
    unbind_gpu_pci_bus_id = pcibusIds[unbind_gpu_id] if pcibusIds is not None and unbind_gpu_id < len(
        pcibusIds) else None

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_field_watch_inactive_to_active")
    group.AddGpu(normal_gpu_id)
    group.AddGpu(unbind_gpu_id)

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbind_gpu_id], [unbind_gpu_pci_bus_id])
    wait_gpu_status(dcgmSystem, normal_gpu_id,
                    dcgm_structs_internal.DcgmEntityStatusOk)

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [
                                       dcgm_fields.DCGM_FI_DEV_GPU_TEMP])
    group.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)

    dcgmSystem.UpdateAllFields(1)

    values = group.samples.GetLatest_v2(fieldGroup)
    assert values.values[dcgm_fields.DCGM_FE_GPU][normal_gpu_id][dcgm_fields.DCGM_FI_DEV_GPU_TEMP][0].isBlank == False
    assert unbind_gpu_id not in values.values[dcgm_fields.DCGM_FE_GPU]

    bind_gpu_and_wait_for_status_updated(dcgmSystem, [unbind_gpu_id], [
                                         unbind_gpu_uuid], [unbind_gpu_pci_bus_id])

    dcgmSystem.UpdateAllFields(1)

    values = group.samples.GetLatest_v2(fieldGroup)
    assert values.values[dcgm_fields.DCGM_FE_GPU][normal_gpu_id][dcgm_fields.DCGM_FI_DEV_GPU_TEMP][0].isBlank == False
    assert unbind_gpu_id not in values.values[dcgm_fields.DCGM_FE_GPU]


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_watch_inactive_to_active(handle, gpuIds):
    helper_test_dcgm_field_watch_inactive_to_active(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_field_watch_inactive_to_active_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_field_watch_inactive_to_active(handle, gpuIds, pcibusIds)


def helper_test_dcgm_field_watch_on_meta_group_will_affect_on_new_attached_gpus(handle, gpuIds, pcibusIds=None):
    '''
    Watch fields on a meta group, and make sure the field values are correctly updated when new GPUs are attached.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    unbind_gpu_uuid = None
    unbind_gpu_pci_bus_id = None
    if in_nvml_injection_mode():
        # GPU 3 is GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778 in H200.yaml
        unbind_gpu_uuid = ["GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778"]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        unbind_gpu_pci_bus_id = [pcibusIds[0]]

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)
    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [
                                       dcgm_fields.DCGM_FI_DEV_GPU_TEMP])
    group.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)

    # The first-time attached GPU will be appended to the last one.
    unbind_gpu_id = len(gpuIds)
    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbind_gpu_id], unbind_gpu_uuid, unbind_gpu_pci_bus_id)
    newGpuIds = dcgmSystem.discovery.GetAllSupportedGpuIds()
    assert len(newGpuIds) == len(gpuIds) + \
        1, f"Expected {len(gpuIds) + 1} GPUs, but got {len(newGpuIds)}"

    dcgmSystem.UpdateAllFields(1)
    values = group.samples.GetLatest_v2(fieldGroup)
    assert values.values[dcgm_fields.DCGM_FE_GPU][unbind_gpu_id][dcgm_fields.DCGM_FI_DEV_GPU_TEMP][0].isBlank == False


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120, heEnv={"DCGM_NVML_INJECTION_GPU_DETACHED_BEFORE_HAND": "3"})
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_watch_on_meta_group_will_affect_on_new_attached_gpus(handle, gpuIds):
    helper_test_dcgm_field_watch_on_meta_group_will_affect_on_new_attached_gpus(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus(detachGpusBeforeHand=[0])
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_field_watch_on_meta_group_will_affect_on_new_attached_gpus_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_field_watch_on_meta_group_will_affect_on_new_attached_gpus(
        handle, gpuIds, pcibusIds)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_field_watch_on_meta_group_will_not_add_to_new_attached_gpus_if_unwatch_is_called(handle, gpuIds):
    '''
    Watch fields on a meta group, and make sure the field values are not updated when new GPUs are attached if unwatch is called.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)

    fieldId = dcgm_fields.DCGM_FI_DEV_GPU_TEMP
    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)
    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [fieldId])
    group.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)
    group.samples.UnwatchFields(fieldGroup)

    dcgm_agent.dcgmDetachDriver(handle)
    dcgm_agent.dcgmAttachDriver(handle)

    for gpuId in gpuIds:
        cmfi = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(
            handle, gpuId, dcgm_fields.DCGM_FE_GPU, fieldId)
        assert (
            cmfi.flags & dcgm_structs_internal.DCGM_CMI_F_WATCHED) == 0, f"gpuId {gpuId}, fieldId {fieldId} still watched. flags x{cmfi.flags}"
        assert cmfi.numWatchers == 0, f"numWatchers {cmfi.numWatchers}"


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_field_watch_meta_group_when_existing_gpu_is_attached(handle, gpuIds):
    '''
    Watch fields on a meta group, and make sure the field values are correctly updated when the existing GPU is attached.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)
    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [
                                       dcgm_fields.DCGM_FI_DEV_GPU_TEMP])
    group.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)

    dcgm_agent.dcgmDetachDriver(handle)
    dcgm_agent.dcgmAttachDriver(handle)

    dcgmSystem.UpdateAllFields(1)

    values = group.samples.GetLatest_v2(fieldGroup)
    assert values.values[dcgm_fields.DCGM_FE_GPU][gpuIds[0]
                                                  ][dcgm_fields.DCGM_FI_DEV_GPU_TEMP][0].isBlank == False


def get_diag_entity_result(response, entityGroupId, entityId):
    return next(filter(lambda cur: cur.entity.entityGroupId == entityGroupId and cur.entity.entityId == entityId,
                       response.entities[:min(response.numEntities, dcgm_structs.DCGM_DIAG_RESPONSE_ENTITIES_MAX)]), None)


def helper_test_diag_should_only_run_on_active_gpus(handle, gpuIds, pcibusIds=None):
    '''
    Make sure the diag should only run on active GPUs.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    gpu0Uuid = None
    pcibusid0 = None
    if in_nvml_injection_mode():
        gpu0Uuid = [get_gpu_uuid(dcgmSystem, gpuIds[0])]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        pcibusid0 = [pcibusIds[0]]

    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuIds[0]], pcibusid0)
    for i in range(1, len(gpuIds)):
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    drd = dcgm_structs.c_dcgmRunDiag_v10()
    drd.version = dcgm_structs.dcgmRunDiag_version10
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    drd.groupId = dcgm_structs.DCGM_GROUP_ALL_GPUS
    response = test_utils.action_validate_wrapper(
        drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version10)

    assert response.numEntities == len(
        gpuIds) - 1, f"Expected number of entities to be {len(gpuIds) - 1}, but got {response.numEntities}"
    gpu0_result = get_diag_entity_result(
        response, dcgm_fields.DCGM_FE_GPU, gpuIds[0])
    assert gpu0_result is None, f"Expected GPU {gpuIds[0]} to not be in the response"

    for i in range(1, len(gpuIds)):
        gpu_result = get_diag_entity_result(
            response, dcgm_fields.DCGM_FE_GPU, gpuIds[i])
        assert gpu_result is not None, f"Expected GPU {gpuIds[i]} to be in the response"

    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuIds[0]], gpu0Uuid, pcibusid0)
    response = test_utils.action_validate_wrapper(
        drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version10)

    for i in range(len(gpuIds)):
        gpu_result = get_diag_entity_result(
            response, dcgm_fields.DCGM_FE_GPU, gpuIds[i])
        assert gpu_result is not None, f"Expected GPU {gpuIds[i]} to be in the response"


@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_diag_should_only_run_on_active_gpus(handle, gpuIds):
    helper_test_diag_should_only_run_on_active_gpus(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_diag_should_only_run_on_active_gpus_live(handle, gpuIds, pcibusIds):
    if len(gpuIds) != len(pcibusIds):
        test_utils.skip_test("Skip on mixed SKU setup")
    helper_test_diag_should_only_run_on_active_gpus(handle, gpuIds, pcibusIds)


def check_nvvs_started_running_on_gpu():
    tree = nvidia_smi_utils.helper_nvidia_smi_xml()

    if tree is None:
        test_utils.skip_test("Failed to get nvidia-smi xml")

    for node in tree.iter('gpu'):
        for proc in node.iterfind('processes'):
            for pr in proc.iterfind('process_info'):
                for name in pr.iterfind('process_name'):
                    if "nvvs" in name.text:
                        return True
    return False


def wait_for_nvvs_started_running_on_gpu():
    for _ in range(160):
        if check_nvvs_started_running_on_gpu():
            return
        time.sleep(0.1)
    test_utils.skip_test("NVVS did not start within 10 seconds")


def helper_test_diag_should_be_stopped_when_gpu_is_attached(handle, gpuIds, pcibusIds=None):
    if len(gpuIds) <= 1:
        test_utils.skip_test("Skipping because test requires >1 live gpus")

    for gpuId in gpuIds[:2]:
        # First check whether the GPU is healthy/supported
        if not skip_test_helpers.gpu_is_healthy_and_support_memtest(handle, gpuId):
            test_utils.skip_test("Skipping because GPU %s does not pass memtest. "
                                 "Please verify whether the GPU is supported and healthy." % gpuId)

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    # unbind GPU 0 so that we can bind it back later.
    gpu0Uuid = None
    pcibusid0 = None
    if in_nvml_injection_mode():
        gpu0Uuid = [get_gpu_uuid(dcgmSystem, gpuIds[0])]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        pcibusid0 = [pcibusIds[0]]

    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuIds[0]], pcibusid0)
    wait_gpu_status(dcgmSystem, gpuIds[1],
                    dcgm_structs_internal.DcgmEntityStatusOk)

    def bind_thread_fn():
        running, debug_output = dcgm_internal_helpers.check_nvvs_process(
            want_running=True, attempts=50)
        if not running:
            raise RuntimeError(
                f"The launched nvvs process did not start within 25 seconds. pgrep output: {debug_output}")
        wait_for_nvvs_started_running_on_gpu()
        trigger_bind_gpu_event(gpu0Uuid, pcibusid0)
    thread = threading.Thread(target=bind_thread_fn)
    thread.start()

    testName = "memtest"
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[1]], testNamesStr=testName)
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_DIAG_STOPPED)):
        _ = test_utils.diag_execute_wrapper(dd, handle)

    not_running, debug_output = dcgm_internal_helpers.check_nvvs_process(
        want_running=False, attempts=50)
    assert not_running, f"The launched nvvs process did not terminate within 25 seconds. pgrep output: {debug_output}"

    thread.join(timeout=16)
    assert not thread.is_alive(), "The thread should have terminated"
    wait_gpu_status(dcgmSystem, gpuIds[0],
                    dcgm_structs_internal.DcgmEntityStatusOk)


@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_current_system_injection_nvml(skuFileName="current_test_diag_should_be_stopped_when_gpu_is_attached.yaml")
@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_diag_should_be_stopped_when_gpu_is_attached(handle, gpuIds):
    helper_test_diag_should_be_stopped_when_gpu_is_attached(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_diag_should_be_stopped_when_gpu_is_attached_live(handle, gpuIds, pcibusIds):
    if len(gpuIds) != len(pcibusIds):
        test_utils.skip_test("Skip on mixed SKU setup")
    helper_test_diag_should_be_stopped_when_gpu_is_attached(
        handle, gpuIds, pcibusIds)


@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.run_only_with_nvml()
def test_diag_should_be_stopped_when_detach_driver(handle, gpuIds):
    gpuId = gpuIds[0]
    # First check whether the GPU is healthy/supported
    if not skip_test_helpers.gpu_is_healthy_and_support_memtest(handle, gpuId):
        test_utils.skip_test("Skipping because GPU %s does not pass memtest. "
                             "Please verify whether the GPU is supported and healthy." % gpuId)

    def detach_thread_fn():
        running, debug_output = dcgm_internal_helpers.check_nvvs_process(
            want_running=True, attempts=50)
        if not running:
            raise RuntimeError(
                f"The launched nvvs process did not start within 25 seconds. pgrep output: {debug_output}")
        # wait for a while to make sure the memtest is started
        # if we do not wait, it is possible that nvvs failed to find usable GPUs (as during the unbind process, the GPUs are not available).
        wait_for_nvvs_started_running_on_gpu()
        dcgm_agent.dcgmDetachDriver(handle)
    thread = threading.Thread(target=detach_thread_fn)
    thread.start()

    testName = "memtest"
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=testName)
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_DIAG_STOPPED)):
        _ = test_utils.diag_execute_wrapper(dd, handle)

    not_running, debug_output = dcgm_internal_helpers.check_nvvs_process(
        want_running=False, attempts=50)
    assert not_running, f"The launched nvvs process did not terminate within 25 seconds. pgrep output: {debug_output}"

    thread.join(timeout=16)
    assert not thread.is_alive(), "The thread should have terminated"

    lsof_output = get_lsof_of_host_engine()
    pattern = re.compile(r'.*/dev/nvidia\d+')
    assert pattern.search(
        lsof_output) is None, f"Expected /dev/nvidia to not be in lsof output {lsof_output}"


def helper_test_diag_should_be_stopped_when_gpu_is_detached(handle, gpuIds, pcibusIds=None):
    gpuId = gpuIds[0]
    # First check whether the GPU is healthy/supported
    if not skip_test_helpers.gpu_is_healthy_and_support_memtest(handle, gpuId):
        test_utils.skip_test("Skipping because GPU %s does not pass memtest. "
                             "Please verify whether the GPU is supported and healthy." % gpuId)

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    def unbind_thread_fn():
        running, debug_output = dcgm_internal_helpers.check_nvvs_process(
            want_running=True, attempts=50)
        if not running:
            raise RuntimeError(
                f"The launched nvvs process did not start within 25 seconds. pgrep output: {debug_output}")
        # wait for a while to make sure the memtest is started
        # if we do not wait, it is possible that nvvs failed to find usable GPUs (as during the bind process, the GPUs are not available).
        wait_for_nvvs_started_running_on_gpu()
        if in_nvml_injection_mode():
            trigger_unbind_gpu_event(uuids=[get_gpu_uuid(dcgmSystem, gpuId)])
        else:
            trigger_unbind_gpu_event(pcibusIds=[pcibusIds[gpuId]])
    thread = threading.Thread(target=unbind_thread_fn)
    thread.start()

    testName = "memtest"
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=testName)
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_DIAG_STOPPED)):
        _ = test_utils.diag_execute_wrapper(dd, handle)

    not_running, debug_output = dcgm_internal_helpers.check_nvvs_process(
        want_running=False, attempts=50)
    assert not_running, f"The launched nvvs process did not terminate within 25 seconds. pgrep output: {debug_output}"

    thread.join(timeout=16)
    assert not thread.is_alive(), "The thread should have terminated"
    wait_gpu_status(dcgmSystem, gpuId,
                    dcgm_structs_internal.DcgmEntityStatusDetached)


@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_current_system_injection_nvml(skuFileName="current_test_diag_should_be_stopped_when_gpu_is_detached.yaml")
@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.exclude_confidential_compute_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_should_be_stopped_when_gpu_is_detached(handle, gpuIds):
    helper_test_diag_should_be_stopped_when_gpu_is_detached(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_diag_should_be_stopped_when_gpu_is_detached_live(handle, gpuIds, pcibusIds):
    if len(gpuIds) != len(pcibusIds):
        test_utils.skip_test("Skip on mixed SKU setup")
    helper_test_diag_should_be_stopped_when_gpu_is_detached(
        handle, gpuIds, pcibusIds)


def helper_test_dcgm_health_watch_on_meta_group_will_affect_on_new_attached_gpus(handle, gpuIds, pcibusIds=None):
    '''
    Health watch on a meta group, and make sure the field values are correctly updated when new GPUs are attached.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    unbindGpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        # GPU 3 is GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778 in H200.yaml
        unbindGpuUuid = ["GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778"]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        # We detached NVML index 0 before host engine starts.
        unbindGpuPciBusId = [pcibusIds[0]]

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)
    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    group.health.Set(newSystems)

    # The first-time attached GPU will be appended to the last one.
    unbindGpuId = len(gpuIds)
    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbindGpuId], unbindGpuUuid, unbindGpuPciBusId)

    group.health.Check()

    # inject an error into new attached GPU
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, unbindGpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                                              0xC8763, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV5 = group.health.Check()
    assert (responseV5.incidentCount == 1), "No incident found"
    assert (responseV5.incidents[0].entityInfo.entityId == unbindGpuId)
    assert (responseV5.incidents[0].system ==
            dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
    assert (responseV5.incidents[0].error.code ==
            dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120, heEnv={"DCGM_NVML_INJECTION_GPU_DETACHED_BEFORE_HAND": "3"})
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_health_watch_on_meta_group_will_affect_on_new_attached_gpus(handle, gpuIds):
    helper_test_dcgm_health_watch_on_meta_group_will_affect_on_new_attached_gpus(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus(detachGpusBeforeHand=[0])
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_watch_on_meta_group_will_affect_on_new_attached_gpus_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_health_watch_on_meta_group_will_affect_on_new_attached_gpus(
        handle, gpuIds, pcibusIds)


def helper_test_dcgm_health_will_not_report_incident_for_inactive_gpus(handle, gpuIds, pcibusIds=None):
    '''
    Make sure the health will not report incident for inactive GPUs and when it comes back to active,
    because the watch is removed when the GPU is detached.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    normalGpuId = gpuIds[0]
    unbindGpuId = gpuIds[1]

    unbindGpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        unbindGpuUuid = [get_gpu_uuid(dcgmSystem, unbindGpuId)]
    else:
        if pcibusIds is None or unbindGpuId >= len(pcibusIds):
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        unbindGpuPciBusId = [pcibusIds[unbindGpuId]]

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_health_will_not_report_incident_for_inactive_gpus")
    group.AddGpu(normalGpuId)
    group.AddGpu(unbindGpuId)
    group.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

    group.health.Check()

    for id in [normalGpuId, unbindGpuId]:
        ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, id, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                                                  0xC8763, 100)
        assert (ret == dcgm_structs.DCGM_ST_OK)

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbindGpuId], unbindGpuPciBusId)
    wait_gpu_status(dcgmSystem, normalGpuId,
                    dcgm_structs_internal.DcgmEntityStatusOk)

    responseV5 = group.health.Check()
    assert responseV5.incidentCount == 1, "Expected incident count to be 1 (for normal GPU), received %d" % responseV5.incidentCount
    assert responseV5.incidents[0].entityInfo.entityId == normalGpuId
    assert responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    assert responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE

    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbindGpuId], unbindGpuUuid, unbindGpuPciBusId)

    responseV5 = group.health.Check()
    assert responseV5.incidentCount == 1, "Expected incident count to be 1 (for normal GPU), received %d" % responseV5.incidentCount
    assert responseV5.incidents[0].entityInfo.entityId == normalGpuId
    assert responseV5.incidents[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    assert responseV5.incidents[0].error.code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_health_will_not_report_incident_for_inactive_gpus(handle, gpuIds):
    helper_test_dcgm_health_will_not_report_incident_for_inactive_gpus(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_will_not_report_incident_for_inactive_gpus_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_health_will_not_report_incident_for_inactive_gpus(
        handle, gpuIds, pcibusIds)


def set_power_limit(group, value):
    configValues = dcgm_structs.c_dcgmDeviceConfig_v2()
    configValues.mEccMode = dcgmvalue.DCGM_INT32_BLANK
    configValues.mPerfState.syncBoost = dcgmvalue.DCGM_INT32_BLANK
    configValues.mPerfState.targetClocks.memClock = dcgmvalue.DCGM_INT32_BLANK
    configValues.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
    configValues.mPowerLimit.val = value
    configValues.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
    for bitmapIndex in range(dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE):
        configValues.mWorkloadPowerProfiles[bitmapIndex] = dcgmvalue.DCGM_INT32_BLANK

    # Will throw an exception on error
    group.config.Set(configValues)


def assert_nvml_func_call_count(handle, funcName, expectedCount):
    retryTimes = 32
    callCount = 0
    for _ in range(retryTimes):
        funcCallCounts = nvml_injection.c_injectNvmlFuncCallCounts_t()
        ret = dcgm_agent_internal.dcgmGetNvmlInjectFuncCallCount(
            handle, funcCallCounts)
        assert (ret == dcgm_structs.DCGM_ST_OK)
        for j in range(funcCallCounts.numFuncs):
            if funcCallCounts.funcCallInfo[j].funcName.decode('utf-8') != funcName:
                continue

            # Target function found
            callCount = funcCallCounts.funcCallInfo[j].funcCallCount
            if callCount == expectedCount:
                return
            break
        time.sleep(1)
    raise AssertionError(
        f"{funcName} should have been called {expectedCount} times, but got {callCount}")


def assert_nvml_func_not_called(handle, funcName):
    funcCallCounts = nvml_injection.c_injectNvmlFuncCallCounts_t()
    ret = dcgm_agent_internal.dcgmGetNvmlInjectFuncCallCount(
        handle, funcCallCounts)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    for i in range(funcCallCounts.numFuncs):
        assert funcCallCounts.funcCallInfo[i].funcName.decode(
            'utf-8') != funcName


@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_config_will_re_apply_only_on_attached_gpus(handle, gpuIds):
    '''
    Make sure the config will be re-applied only on the attached GPUs.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    gpuId1 = gpuIds[1]
    gpuId1Uuid = get_gpu_uuid(dcgmSystem, gpuId1)
    gpuId2 = gpuIds[2]
    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_config_will_re_apply_only_on_attached_gpus")
    group.AddGpu(gpuId1)
    group.AddGpu(gpuId2)

    set_power_limit(group, 256)

    ret = dcgm_agent_internal.dcgmResetNvmlInjectFuncCallCount(handle)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuId1])
    for i in range(len(gpuIds)):
        if gpuIds[i] == gpuId1:
            continue
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    # Since we only unbind GPU 1, the config will be re-applied to GPU 2.
    assert_nvml_func_call_count(handle, "nvmlDeviceSetPowerManagementLimit", 1)

    ret = dcgm_agent_internal.dcgmResetNvmlInjectFuncCallCount(handle)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    bind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuId1], [gpuId1Uuid])
    # Even after binding GPU 1 back, the config will only be re-applied to GPU 2.
    assert_nvml_func_call_count(handle, "nvmlDeviceSetPowerManagementLimit", 1)


@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_config_cannot_set_on_detached_gpu(handle, gpuIds):
    '''
    Make sure the config cannot be set on a detached GPU.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    gpuId = gpuIds[1]
    gpuUuid = get_gpu_uuid(dcgmSystem, gpuId)
    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_config_cannot_set_on_detached_gpu")
    group.AddGpu(gpuId)
    # We need to at least 1 active GPU in the group so that we can test this behavior, otherwise the group will be empty after unbinding and fail early.
    group.AddGpu(gpuIds[2])

    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuId])
    for i in range(len(gpuIds)):
        if gpuIds[i] == gpuId:
            continue
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    set_power_limit(group, 256)
    # Since the GPU 1 is detached, nvmlDeviceSetPowerManagementLimit should have been called 1 time for GPU 2
    assert_nvml_func_call_count(handle, "nvmlDeviceSetPowerManagementLimit", 1)

    ret = dcgm_agent_internal.dcgmResetNvmlInjectFuncCallCount(handle)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    bind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuId], [gpuUuid])
    # Only GPU 2 should have the config be applied, even after binding GPU 1 back
    assert_nvml_func_call_count(handle, "nvmlDeviceSetPowerManagementLimit", 1)


@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_config_cannot_acquire_on_detached_gpu(handle, gpuIds):
    '''
    Make sure the config cannot be acquired on a detached GPU.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    gpuId = gpuIds[1]
    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_config_cannot_acquire_on_detached_gpu")
    group.AddGpu(gpuId)
    # We need to have at least 1 active GPU in the group so that we can test this behavior, otherwise the group will be empty after unbinding and fail early.
    group.AddGpu(gpuIds[2])

    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuId])
    for i in range(len(gpuIds)):
        if gpuIds[i] == gpuId:
            continue
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    set_power_limit(group, 256)
    for state in [dcgm_structs.DCGM_CONFIG_TARGET_STATE, dcgm_structs.DCGM_CONFIG_CURRENT_STATE]:
        ret = group.config.Get(state)
        assert len(ret) == 1
        assert ret[0].gpuId == gpuIds[2]
        assert ret[0].mPowerLimit.val == 256


def create_c_callback(queue=None):
    @CFUNCTYPE(None, POINTER(dcgm_structs.c_dcgmPolicyCallbackResponse_v2), c_uint64)
    def c_callback(response, userData):
        if queue:
            # copy data into a python struct so that it is the right format and is not lost when "response" var is lost
            callbackResp = dcgm_structs.c_dcgmPolicyCallbackResponse_v2()
            memmove(addressof(callbackResp), response,
                    callbackResp.FieldsSizeof())
            queue.put(callbackResp)
    return c_callback


def helper_test_dcgm_policy_on_meta_group_will_affect_on_new_attached_gpus(handle, gpuIds, pcibusIds=None):
    '''
    Register and set policy on a meta group, and make sure the callback is correctly triggered when error occurs on new attached GPUs.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    unbindGpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        # GPU 3 is GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778 in H200.yaml
        unbindGpuUuid = ["GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778"]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        # we detached NVML index 0 before host engine starts.
        unbindGpuPciBusId = [pcibusIds[0]]

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean = True

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)

    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_DBE, c_callback, newPolicy)

    # The first-time attached GPU will be appended to the last one.
    unbindGpuId = len(gpuIds)
    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbindGpuId], unbindGpuUuid, unbindGpuPciBusId)
    # Inject an error to new attached GPU
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, unbindGpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_DEV,
                                                              0xC8763, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=15)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (callbackResp.condition == dcgm_structs.DCGM_POLICY_COND_DBE), \
        (f"error callback was not for a DBE error, got: {callbackResp.condition}")
    assert (callbackResp.val.dbe.numerrors ==
            0xC8763), f"Expected 0xC8763 DBE error but got {callbackResp.val.dbe.numerrors}"
    assert (callbackResp.val.dbe.location == dcgm_structs.c_dcgmPolicyConditionDbe_t.LOCATIONS['DEVICE']),  \
        f"got: {callbackResp.val.dbe.location}"

    # check that callback response has correct gpuId for which error was injected.
    assert (callbackResp.gpuId == unbindGpuId), "GPU ID in callback response is incorrect. " \
        f"Received: [{callbackResp.gpuId}], Expected: [{unbindGpuId}]."


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120, heEnv={"DCGM_NVML_INJECTION_GPU_DETACHED_BEFORE_HAND": "3"})
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_policy_on_meta_group_will_affect_on_new_attached_gpus(handle, gpuIds):
    helper_test_dcgm_policy_on_meta_group_will_affect_on_new_attached_gpus(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus(detachGpusBeforeHand=[0])
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_policy_on_meta_group_will_affect_on_new_attached_gpus_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_policy_on_meta_group_will_affect_on_new_attached_gpus(
        handle, gpuIds, pcibusIds)


def helper_test_dcgm_policy_will_affect_on_alive_gpus(handle, gpuIds, pcibusIds=None):
    '''
    Register and set policy on GPUs, detach one of the GPUs, and make sure the callback is correctly triggered when error occurs on GPUs that are not detached.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean = True

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_policy_will_affect_on_alive_gpus")
    group.AddGpu(gpuIds[0])
    group.AddGpu(gpuIds[1])
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)

    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_DBE, c_callback, newPolicy)

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuIds[0]], [pcibusIds[0]] if pcibusIds is not None and 0 < len(pcibusIds) else None)
    for i in range(1, len(gpuIds)):
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    # Inject an error to both GPUs
    for i in range(2):
        ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, gpuIds[i], dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_DEV,
                                                                  0xC8763, 100)
        assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=15)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (callbackResp.condition == dcgm_structs.DCGM_POLICY_COND_DBE), \
        (f"error callback was not for a DBE error, got: {callbackResp.condition}")
    assert (callbackResp.val.dbe.numerrors ==
            0xC8763), f"Expected 0xC8763 DBE error but got {callbackResp.val.dbe.numerrors}"
    assert (callbackResp.val.dbe.location == dcgm_structs.c_dcgmPolicyConditionDbe_t.LOCATIONS['DEVICE']),  \
        f"got: {callbackResp.val.dbe.location}"

    # check that callback response has only active gpuId.
    assert (callbackResp.gpuId == gpuIds[1]), "GPU ID in callback response is incorrect. " \
        f"Received: [{callbackResp.gpuId}], Expected: [{gpuIds[1]}]."

    assert callbackQueue.empty(
    ), "Callback queue should be empty and not include GPU ID of detached GPU"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_policy_will_affect_on_alive_gpus(handle, gpuIds):
    helper_test_dcgm_policy_will_affect_on_alive_gpus(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_policy_will_affect_on_alive_gpus_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_policy_will_affect_on_alive_gpus(
        handle, gpuIds, pcibusIds)


def helper_test_dcgm_policy_on_meta_group_will_not_affect_on_new_attached_gpus_after_unregister(handle, gpuIds, pcibusIds=None):
    '''
    Register and set policy on a meta group, and make sure the callback is not triggered when error occurs on new attached GPUs after unregister.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    unbindGpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        # GPU 3 is GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778 in H200.yaml
        unbindGpuUuid = ["GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778"]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        # we detached NVML index 0 before host engine starts.
        unbindGpuPciBusId = [pcibusIds[0]]

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean = True

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)

    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_DBE, c_callback, newPolicy)
    group.policy.Unregister(dcgm_structs.DCGM_POLICY_COND_DBE)

    # The first-time attached GPU will be appended to the last one.
    unbindGpuId = len(gpuIds)
    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbindGpuId], unbindGpuUuid, unbindGpuPciBusId)
    # Inject an error to new attached GPU
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, unbindGpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_DEV,
                                                              0xC8763, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    queueEmpty = False
    try:
        _ = callbackQueue.get(timeout=15)
    except queue.Empty:
        queueEmpty = True

    assert queueEmpty, "Callback should not be triggered"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120, heEnv={"DCGM_NVML_INJECTION_GPU_DETACHED_BEFORE_HAND": "3"})
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_policy_on_meta_group_will_not_affect_on_new_attached_gpus_after_unregister(handle, gpuIds):
    helper_test_dcgm_policy_on_meta_group_will_not_affect_on_new_attached_gpus_after_unregister(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus(detachGpusBeforeHand=[0])
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_policy_on_meta_group_will_not_affect_on_new_attached_gpus_after_unregister_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_policy_on_meta_group_will_not_affect_on_new_attached_gpus_after_unregister(
        handle, gpuIds, pcibusIds)


def helper_test_dcgm_policy_on_normal_group_will_not_affect_on_new_attached_gpus(handle, gpuIds, pcibusIds=None):
    '''
    Register and set policy on a normal group, and make sure the callback is not triggered when error occurs on new attached GPUs.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    unbindGpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        # GPU 3 is GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778 in H200.yaml
        unbindGpuUuid = ["GPU-3e635e09-4cdc-f86d-7ef5-14a5db271778"]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        # we detached NVML index 0 before host engine starts.
        unbindGpuPciBusId = [pcibusIds[0]]

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean = True

    group = DcgmGroup.DcgmGroup(dcgmHandle, groupName="a_normal_group")
    group.AddGpu(gpuIds[0])
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)

    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_DBE, c_callback, newPolicy)

    # The first-time attached GPU will be appended to the last one.
    unbindGpuId = len(gpuIds)
    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbindGpuId], unbindGpuUuid, unbindGpuPciBusId)
    # Inject an error to new attached GPU
    ret = dcgm_field_injection_helpers.inject_field_value_i64(handle, unbindGpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_DEV,
                                                              0xC8763, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    queueEmpty = False
    try:
        _ = callbackQueue.get(timeout=15)
    except queue.Empty:
        queueEmpty = True

    assert queueEmpty, "Callback should not be triggered"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120, heEnv={"DCGM_NVML_INJECTION_GPU_DETACHED_BEFORE_HAND": "3"})
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_policy_on_normal_group_will_not_affect_on_new_attached_gpus(handle, gpuIds):
    helper_test_dcgm_policy_on_normal_group_will_not_affect_on_new_attached_gpus(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus(detachGpusBeforeHand=[0])
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_policy_on_normal_group_will_not_affect_on_new_attached_gpus_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_policy_on_normal_group_will_not_affect_on_new_attached_gpus(
        handle, gpuIds, pcibusIds)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_policy_will_not_trigger_on_inactive_gpus(handle, gpuIds):
    '''
    Register and set policy on ALL_GPUS, and make sure the callback is not triggered when error occurs on inactive GPUs.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    def mock_ecc_memory_error_counter(handle, gpuId):
        injectedRet = nvml_injection.c_injectNvmlRet_t()
        injectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
        injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_ULONG_LONG
        injectedRet.values[0].value.ULongLong = 0xC8763
        injectedRet.valueCount = 1

        extraKeysType = nvml_injection_structs.c_injectNvmlVal_t * 3
        extraKeys = extraKeysType()
        extraKeys[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_MEMORYERRORTYPE
        extraKeys[0].value.MemoryErrorType = dcgm_nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED
        extraKeys[1].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_ECCCOUNTERTYPE
        extraKeys[1].value.EccCounterType = dcgm_nvml.NVML_VOLATILE_ECC
        extraKeys[2].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_MEMORYLOCATION
        extraKeys[2].value.MemoryLocation = dcgm_nvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY

        ret = dcgm_agent_internal.dcgmInjectNvmlDevice(
            handle, gpuId, "MemoryErrorCounter", extraKeys, 3, injectedRet)
        assert (ret == dcgm_structs.DCGM_ST_OK)

    mock_ecc_memory_error_counter(handle, gpuIds[0])

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean = True

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)

    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuIds[0]])
    for i in range(1, len(gpuIds)):
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_DBE, c_callback, newPolicy)

    queueEmpty = False
    try:
        _ = callbackQueue.get(timeout=15)
    except queue.Empty:
        queueEmpty = True

    assert queueEmpty, "Callback should not be triggered"


def helper_test_dcgm_gpu_index_and_nvml_index_mapping(handle, gpuIds, pcibusIds=None):
    '''
    Make sure the gpu index and nvml index mapping is correct after bind / unbind.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    gpuId = gpuIds[0]

    unbindGpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        unbindGpuUuid = [get_gpu_uuid(dcgmSystem, gpuId)]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        unbindGpuPciBusId = [pcibusIds[0]]

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuId], unbindGpuPciBusId)
    for i in range(len(gpuIds)):
        if gpuIds[i] == gpuId:
            continue
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)
    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [
                                       dcgm_fields.DCGM_FI_DEV_NVML_INDEX])
    group.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)

    values = group.samples.GetLatest_v2(fieldGroup)
    assert values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_DEV_NVML_INDEX][0].isBlank == True
    for i in range(1, len(gpuIds)):
        # minus 1 because gpuIds[0] is detached
        assert values.values[dcgm_fields.DCGM_FE_GPU][gpuIds[i]
                                                      ][dcgm_fields.DCGM_FI_DEV_NVML_INDEX][0].value == i - 1

    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuId], unbindGpuUuid, unbindGpuPciBusId)
    dcgmSystem.UpdateAllFields(1)
    values = group.samples.GetLatest_v2(fieldGroup)
    assert values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_DEV_NVML_INDEX][0].isBlank == False
    if in_nvml_injection_mode():
        # gpuIds[0] is now attached, its nvml index should be the last one
        # In NVML injection mode, the new attached GPU will be appended to the last one.
        assert values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_DEV_NVML_INDEX][0].value == len(
            gpuIds) - 1
        for i in range(1, len(gpuIds)):
            assert values.values[dcgm_fields.DCGM_FE_GPU][gpuIds[i]
                                                          ][dcgm_fields.DCGM_FI_DEV_NVML_INDEX][0].value == i - 1
    else:
        # In real NVML, the new attached GPU will be located to its original index.
        for i in range(len(gpuIds)):
            assert values.values[dcgm_fields.DCGM_FE_GPU][gpuIds[i]
                                                          ][dcgm_fields.DCGM_FI_DEV_NVML_INDEX][0].value == i


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_gpu_index_and_nvml_index_mapping(handle, gpuIds):
    helper_test_dcgm_gpu_index_and_nvml_index_mapping(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_gpu_index_and_nvml_index_mapping_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_gpu_index_and_nvml_index_mapping(
        handle, gpuIds, pcibusIds)


def helper_check_bind_unbind_event(handle, startTs, expectedValues):
    fieldId = dcgm_fields.DCGM_FI_BIND_UNBIND_EVENT
    maxCount = 100
    endTs = 0
    readTs = 0
    readValues = {}
    # Retry getting the bind/unbind event values for a while since the field update
    # timing is indeterminate with respect to the GPUs coming back online.
    retryTimes = 16
    for _ in range(retryTimes):
        values = dcgm_agent_internal.dcgmGetMultipleValuesForField(
            handle, 0, fieldId, maxCount, startTs, endTs, dcgm_structs.DCGM_ORDER_ASCENDING)
        for fieldValue in values:
            bindUnbindNum = fieldValue.value.i64
            if dcgmvalue.DCGM_INT64_IS_BLANK(bindUnbindNum):
                continue
            readTs = fieldValue.ts
            startTs = readTs + 1
            logger.debug(
                "Read Bind/Unbind value {} and timestamp {}".format(bindUnbindNum, readTs))
            if bindUnbindNum in expectedValues:
                if bindUnbindNum in readValues:
                    readValues[bindUnbindNum] += 1
                else:
                    readValues[bindUnbindNum] = 1
        if readValues != expectedValues:
            time.sleep(1)
        else:
            break
    assert readValues == expectedValues, "Bind/Unbind field values - expected {}, got {} in registered Bind/Unbind events".format(
        expectedValues, readValues)
    return readTs


def helper_test_topology_device_with_detached_gpu(handle, gpuIds, pcibusIds=None):
    '''
    Make sure the topology device works correctly when detached GPUs exist.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    detachedGpuId = gpuIds[0]

    unbindGpuPciBusId = None
    if pcibusIds and detachedGpuId < len(pcibusIds):
        unbindGpuPciBusId = [pcibusIds[detachedGpuId]]

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [detachedGpuId], unbindGpuPciBusId)
    for i in range(len(gpuIds)):
        if gpuIds[i] == detachedGpuId:
            continue
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GPU_IS_LOST)):
        _ = dcgmSystem.discovery.GetGpuTopology(detachedGpuId)

    topologyInfo = dcgmSystem.discovery.GetGpuTopology(gpuIds[1])
    # minus 2 because gpuIds[0] is detached, and gpuIds[1] is the target GPU
    assert topologyInfo.numGpus == len(gpuIds) - 2

    for i in range(topologyInfo.numGpus):
        assert topologyInfo.gpuPaths[i].gpuId != detachedGpuId


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_topology_device_with_detached_gpu(handle, gpuIds):
    helper_test_topology_device_with_detached_gpu(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_topology_device_with_detached_gpu_live(handle, gpuIds, pcibusIds):
    helper_test_topology_device_with_detached_gpu(handle, gpuIds, pcibusIds)


def helper_test_select_gpus_by_topology_with_detached_gpu(handle, gpuIds, pcibusIds=None):
    '''
    Make sure the select gpus by topology behave correctly when detached GPUs are included.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    discover = DcgmSystem.DcgmSystemDiscovery(dcgmHandle)

    inputList = 0
    gpuBits = {}

    for gpuId in gpuIds:
        mask = (0x1 << gpuId)
        inputList = inputList | mask
        gpuBits[gpuId] = mask

    detachedGpuId = gpuIds[1]

    unbindGpuPciBusId = None
    if pcibusIds and detachedGpuId < len(pcibusIds):
        unbindGpuPciBusId = [pcibusIds[detachedGpuId]]

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [detachedGpuId], unbindGpuPciBusId)
    for i in range(len(gpuIds)):
        if gpuIds[i] == detachedGpuId:
            continue
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    hints = dcgm_structs.DCGM_TOPO_HINT_F_IGNOREHEALTH
    # cannot achieve len(gpuIds) because detachedGpuId is included in inputList
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_INSUFFICIENT_SIZE)):
        _ = discover.SelectGpusByTopology(inputList, len(gpuIds), hints)

    outputGpus = discover.SelectGpusByTopology(
        inputList, len(gpuIds) - 1, hints)
    assert outputGpus.value & (
        0x1 << detachedGpuId) == 0, f"Expected {detachedGpuId} to be excluded from the output 0x{outputGpus.value:x}"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_select_gpus_by_topology_with_detached_gpu(handle, gpuIds):
    helper_test_select_gpus_by_topology_with_detached_gpu(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_select_gpus_by_topology_with_detached_gpu_live(handle, gpuIds, pcibusIds):
    helper_test_select_gpus_by_topology_with_detached_gpu(
        handle, gpuIds, pcibusIds)


def healper_test_dcgm_field_bind_unbind_event_gpu_is_unbound_bound(handle, gpuIds, pcibusIds=None):
    '''
    Watch the bind/unbind event field, and make sure the field values are correctly updated when a GPU is bound/unbound.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    normal_gpu_id = gpuIds[0]
    unbind_gpu_id = gpuIds[1]

    unbindGpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        unbindGpuUuid = [get_gpu_uuid(dcgmSystem, unbind_gpu_id)]
    else:
        if pcibusIds is None or unbind_gpu_id >= len(pcibusIds):
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        unbindGpuPciBusId = [pcibusIds[unbind_gpu_id]]

    unbindTs = datetime.now().timestamp() * 1000000
    logger.debug("Unbind timestamp: {}".format(unbindTs))
    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbind_gpu_id], unbindGpuPciBusId)
    wait_gpu_status(dcgmSystem, normal_gpu_id,
                    dcgm_structs_internal.DcgmEntityStatusOk)

    updateFreq = 1000000
    maxKeepAge = 600.0  # 10 minutes
    maxKeepEntries = 0  # no limit
    fieldId = dcgm_fields.DCGM_FI_BIND_UNBIND_EVENT
    # GPU ID does not matter for this global field
    dcgm_agent_internal.dcgmWatchFieldValue(
        handle, 0, fieldId, updateFreq, maxKeepAge, maxKeepEntries)

    startTs = int(unbindTs)
    expectedValues = {dcgm_structs.DCGM_BU_EVENT_STATE_SYSTEM_REINITIALIZING: 1,
                      dcgm_structs.DCGM_BU_EVENT_STATE_SYSTEM_REINITIALIZATION_COMPLETED: 1}
    lastReadTs = helper_check_bind_unbind_event(
        handle, startTs, expectedValues)
    logger.debug("Read bind timestamp: {}".format(lastReadTs))

    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [unbind_gpu_id], unbindGpuUuid, unbindGpuPciBusId)
    wait_gpu_status(dcgmSystem, normal_gpu_id,
                    dcgm_structs_internal.DcgmEntityStatusOk)

    startTs = lastReadTs + 1
    helper_check_bind_unbind_event(handle, startTs, expectedValues)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_bind_unbind_event_gpu_is_unbound_bound(handle, gpuIds):
    healper_test_dcgm_field_bind_unbind_event_gpu_is_unbound_bound(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgm_field_bind_unbind_event_gpu_is_unbound_bound_live(handle, gpuIds, pcibusIds):
    healper_test_dcgm_field_bind_unbind_event_gpu_is_unbound_bound(
        handle, gpuIds, pcibusIds)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_field_bind_unbind_event_detach_attach_driver(handle, gpuIds):
    '''
    Watch the bind/unbind event field, and make sure the field values are correctly updated when the driver is detached/attached.
    '''
    detachTs = datetime.now().timestamp() * 1000000
    logger.debug("Detach timestamp: {}".format(detachTs))
    dcgm_agent.dcgmDetachDriver(handle)

    updateFreq = 1000000
    maxKeepAge = 600.0  # 10 minutes
    maxKeepEntries = 0  # no limit
    fieldId = dcgm_fields.DCGM_FI_BIND_UNBIND_EVENT
    # GPU ID does not matter for this global field
    dcgm_agent_internal.dcgmWatchFieldValue(
        handle, 0, fieldId, updateFreq, maxKeepAge, maxKeepEntries)

    expectedValues = {
        dcgm_structs.DCGM_BU_EVENT_STATE_SYSTEM_REINITIALIZING: 1}
    startTs = int(detachTs)
    lastReadTs = helper_check_bind_unbind_event(
        handle, startTs, expectedValues)
    logger.debug("Read bind timestamp: {}".format(lastReadTs))

    dcgm_agent.dcgmAttachDriver(handle)

    expectedValues = {
        dcgm_structs.DCGM_BU_EVENT_STATE_SYSTEM_REINITIALIZATION_COMPLETED: 1}
    startTs = lastReadTs + 1
    helper_check_bind_unbind_event(handle, startTs, expectedValues)


def helper_test_group_manager_will_remove_detached_gpu_from_group(handle, gpuIds, pcibusIds=None):
    '''
    Make sure the group manager will remove detached GPUs from the group.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    detachedGpuId = gpuIds[1]
    unbindGpuPciBusId = None
    if pcibusIds and detachedGpuId < len(pcibusIds):
        unbindGpuPciBusId = [pcibusIds[detachedGpuId]]

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_group_manager_will_remove_detached_gpu_from_group")

    group.AddGpu(gpuIds[0])
    group.AddGpu(detachedGpuId)
    if in_nvml_injection_mode():
        # In H200-With-MIG.yaml, GPU 1 has 2 GI (entity id 7 and 8) and 2 CI (entity id 7 and 8)
        group.AddEntity(dcgm_fields.DCGM_FE_GPU_I, 7)
        group.AddEntity(dcgm_fields.DCGM_FE_GPU_CI, 7)

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [detachedGpuId], unbindGpuPciBusId)
    for i in range(len(gpuIds)):
        if gpuIds[i] == detachedGpuId:
            continue
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    entities = group.GetEntities()
    assert len(entities) == 1
    assert entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU
    assert entities[0].entityId == gpuIds[0]


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200-With-MIG.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_group_manager_will_remove_detached_gpu_from_group(handle, gpuIds):
    helper_test_group_manager_will_remove_detached_gpu_from_group(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_group_manager_will_remove_detached_gpu_from_group_live(handle, gpuIds, pcibusIds):
    helper_test_group_manager_will_remove_detached_gpu_from_group(
        handle, gpuIds, pcibusIds)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_load_module_when_gpu_is_detached(handle, gpuIds):
    '''
    Verify that modules cannot be loaded when the GPUs are detached.
    '''
    dcgm_agent.dcgmDetachDriver(handle)

    drd = dcgm_structs.c_dcgmRunDiag_v10()
    drd.version = dcgm_structs.dcgmRunDiag_version10
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    # Initializing to DCGM_GROUP_NULL in case the constructor above doesn't and entityIds is specified.
    drd.groupId = dcgm_structs.DCGM_GROUP_NULL
    drd.entityIds = str(gpuIds[0])

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GPUS_DETACHED)):
        test_utils.action_validate_wrapper(
            drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version10)

    dcgm_agent.dcgmAttachDriver(handle)

    test_utils.action_validate_wrapper(
        drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version10)


def watch_profiling_field(group, fieldGroup, gpuIds):
    try:
        group.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_PROFILING_NOT_SUPPORTED) as e:
        test_utils.skip_test(f"Profiling is not supported for gpuIds {gpuIds}")
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED) as e:
        test_utils.skip_test("The profiling module could not be loaded")
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED) as e:
        test_utils.skip_test("The profiling module is not supported")


def helper_test_diag_some_requested_gpus_detached(handle, gpuIds, pcibusIds=None):
    """
    Test that diag handles some requested GPUs are detached correctly.
    """
    if len(gpuIds) < 4:
        test_utils.skip_test("This test requires at least 4 GPUs")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    # Detach GPUs 0,1,2 and rest are attached
    detachedGpuIds = gpuIds[:3]
    attachedGpuIds = gpuIds[3:]
    unbindGpuPciBusId = None
    if pcibusIds and len(pcibusIds) >= 3:
        unbindGpuPciBusId = [pcibusIds[i] for i in range(3)]

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, detachedGpuIds, unbindGpuPciBusId)
    for gid in attachedGpuIds:
        wait_gpu_status(dcgmSystem, gid,
                        dcgm_structs_internal.DcgmEntityStatusOk)

    allTestGpuIds = detachedGpuIds + attachedGpuIds
    dd = DcgmDiag.DcgmDiag(gpuIds=allTestGpuIds, testNamesStr="1")
    response = test_utils.diag_execute_wrapper(dd, handle)

    # Version check
    assert response.version in [dcgm_structs.dcgmDiagResponse_version11,
                                dcgm_structs.dcgmDiagResponse_version12], f"Expected version 11 or 12, got {response.version}"

    # Response check
    detachedGpuResults = {gid: False for gid in detachedGpuIds}
    attachedGpuResults = {gid: False for gid in attachedGpuIds}
    for i in range(response.numResults):
        result = response.results[i]
        if result.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU:
            if result.entity.entityId in detachedGpuIds:
                detachedGpuResults[result.entity.entityId] = True
                assert result.result == dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN, f"Detached GPU {result.entity.entityId} should have NOT_RUN result, got {result.result}"
            elif result.entity.entityId in attachedGpuIds:
                attachedGpuResults[result.entity.entityId] = True
                assert result.result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN, f"Attached GPU {result.entity.entityId} should have actually run, got NOT_RUN"

    for gid in detachedGpuIds:
        assert detachedGpuResults[gid], f"Detached GPU {gid} should appear in results with NOT_RUN status"
    for gid in attachedGpuIds:
        assert attachedGpuResults[gid], f"Attached GPU {gid} should have test results"

    foundDetachedMessages = {gid: False for gid in detachedGpuIds}

    for i in range(response.numErrors):
        error = response.errors[i]
        if error.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and error.entity.entityId in detachedGpuIds:
            foundDetachedMessages[error.entity.entityId] = True
            assert 'detached' in error.msg.lower() or 'not accessible' in error.msg.lower(
            ), f"Expected 'detached' or 'not accessible' in error message, got: {error.msg}"

    for gid in detachedGpuIds:
        assert foundDetachedMessages[
            gid], f"Did not find detached GPU error message for GPU {gid}"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_diag_some_requested_gpus_detached(handle, gpuIds):
    helper_test_diag_some_requested_gpus_detached(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_diag_some_requested_gpus_detached_live(handle, gpuIds, pcibusIds):
    if len(gpuIds) != len(pcibusIds):
        test_utils.skip_test("Skip on mixed SKU setup")
    helper_test_diag_some_requested_gpus_detached(handle, gpuIds, pcibusIds)


def helper_test_diag_all_requested_gpus_detached(handle, gpuIds, pcibusIds=None):
    """
    Test that diag handles the case when all requested GPUs are detached.
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    detachedGpuIds = gpuIds
    if pcibusIds:
        # When PCI bus IDs are provided, we are operating in normal mode.
        # Unbinding all GPUs simultaneously can cause the event to fire multiple times (it is normal as NVML may take time to detach the GPUs).
        # The unbind_gpu_and_wait_for_status_updated expects a single bind/unbind event, so it fails if it sees more than one.
        # To avoid this, we unbind the GPUs one by one.
        for gpuId, pcibusId in zip(detachedGpuIds, pcibusIds):
            unbind_gpu_and_wait_for_status_updated(
                dcgmSystem, [gpuId], [pcibusId])
    else:
        unbind_gpu_and_wait_for_status_updated(dcgmSystem, detachedGpuIds)

    dd = DcgmDiag.DcgmDiag(gpuIds=detachedGpuIds, testNamesStr="1")

    dd.runDiagInfo.version = dd.version
    fn = dcgm_agent.dcgmFP("dcgmActionValidate_v2")

    for _ in range(100):
        response = dcgm_structs.c_dcgmDiagResponse_v12()
        response.version = dcgm_structs.dcgmDiagResponse_version12
        ret = fn(handle, byref(dd.runDiagInfo), byref(response))
        if ret == dcgm_structs.DCGM_ST_GPUS_DETACHED:
            time.sleep(0.1)
            continue
        break

    assert ret == dcgm_structs.DCGM_ST_NVVS_ERROR, f"Expected DCGM_ST_NVVS_ERROR, got {ret}"

    assert response.version in [dcgm_structs.dcgmDiagResponse_version11, dcgm_structs.dcgmDiagResponse_version12], \
        f"Expected version 11 or 12, got {response.version}"

    foundSystemError = False
    foundDetachedMessage = False
    for i in range(response.numErrors):
        error = response.errors[i]
        if error.testId == dcgm_structs.DCGM_DIAG_RESPONSE_SYSTEM_ERROR:
            foundSystemError = True
            errorMsg = error.msg.lower()
            if 'detached' in errorMsg or 'inaccessible' in errorMsg:
                foundDetachedMessage = True
                assert str(len(detachedGpuIds)) in error.msg, \
                    f"Expected count {len(detachedGpuIds)} in system error, got: {error.msg}"

    assert foundSystemError, "Should have system error(s)"
    assert foundDetachedMessage, "Should have a system error message about detached/inaccessible GPUs"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_diag_all_requested_gpus_detached(handle, gpuIds):
    helper_test_diag_all_requested_gpus_detached(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_diag_all_requested_gpus_detached_live(handle, gpuIds, pcibusIds):
    if len(gpuIds) != len(pcibusIds):
        test_utils.skip_test("Skip on mixed SKU setup")
    helper_test_diag_all_requested_gpus_detached(handle, gpuIds, pcibusIds)


def helper_test_dcgm_prof_gpu_watch_should_not_be_resumed_when_gpu_is_detached(handle, gpuIds, pcibusIds=None):
    '''
    Make sure the GPU watch should not be resumed when the GPU is detached and then attached back later.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    gpuId = gpuIds[0]
    gpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        gpuUuid = [get_gpu_uuid(dcgmSystem, gpuId)]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        unbindGpuPciBusId = [pcibusIds[0]]

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_prof_gpu_watch_should_not_be_resumed_when_gpu_is_detached")

    group.AddGpu(gpuId)

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [
                                       dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES])
    watch_profiling_field(group, fieldGroup, [gpuId])

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuId], unbindGpuPciBusId)
    for i in range(1, len(gpuIds)):
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)
    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuId], gpuUuid, unbindGpuPciBusId)

    cmfi = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(
        handle, gpuId, dcgm_fields.DCGM_FE_GPU, dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES)
    assert (cmfi.flags & dcgm_structs_internal.DCGM_CMI_F_WATCHED) == 0, \
        f"gpuId {gpuId}, fieldId {dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES} still watched. flags x{cmfi.flags}"
    assert cmfi.numWatchers == 0, f"numWatchers {cmfi.numWatchers}"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_current_system_injection_nvml(skuFileName="current_test_dcgm_prof_gpu_watch_should_not_be_resumed_when_gpu_is_detached.yaml")
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.with_gpu_filter(test_utils.non_gpm_gpus)
def test_dcgm_prof_gpu_watch_should_not_be_resumed_when_gpu_is_detached(handle, gpuIds):
    helper_test_dcgm_prof_gpu_watch_should_not_be_resumed_when_gpu_is_detached(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.with_gpu_filter(test_utils.non_gpm_gpus)
def test_dcgm_prof_gpu_watch_should_not_be_resumed_when_gpu_is_detached_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_prof_gpu_watch_should_not_be_resumed_when_gpu_is_detached(
        handle, gpuIds, pcibusIds)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
@test_utils.with_gpu_filter(test_utils.non_gpm_gpus)
def test_dcgm_prof_gpu_watch_should_be_resumed_when_gpu_is_still_attached(handle, gpuIds):
    '''
    Make sure the GPU watch should be resumed when the GPU is still attached during the reinitialization.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    gpuId = gpuIds[0]
    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_prof_gpu_watch_should_be_resumed_when_gpu_is_still_attached")

    group.AddGpu(gpuId)

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [
                                       dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES])
    watch_profiling_field(group, fieldGroup, [gpuId])

    dcgm_agent.dcgmDetachDriver(handle)
    dcgm_agent.dcgmAttachDriver(handle)

    dcgmSystem.UpdateAllFields(1)
    values = group.samples.GetLatest_v2(fieldGroup)
    assert not values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].isBlank
    txBytes = values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].value

    waitSeconds = 32
    for _ in range(waitSeconds):
        values = group.samples.GetLatest_v2(fieldGroup)
        assert not values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].isBlank
        currentTxBytes = values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].value
        if currentTxBytes != txBytes:
            break
        time.sleep(1)
    else:
        raise AssertionError("The tx bytes should have been updated")


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
@test_utils.with_gpu_filter(test_utils.non_gpm_gpus)
def test_dcgm_prof_field_should_be_blank_when_gpu_is_detached(handle, gpuIds):
    '''
    Make sure the profiling field should be blank when the GPU is detached.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)

    gpuId = gpuIds[0]
    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupName="test_dcgm_prof_field_should_be_blank_when_gpu_is_detached")

    group.AddGpu(gpuId)

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [
                                       dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES])
    watch_profiling_field(group, fieldGroup, [gpuId])

    values = group.samples.GetLatest_v2(fieldGroup)
    assert not values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].isBlank

    dcgm_agent.dcgmDetachDriver(handle)
    values = group.samples.GetLatest_v2(fieldGroup)
    assert values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].isBlank
    dcgm_agent.dcgmAttachDriver(handle)


def helper_test_dcgm_prof_on_all_gpus_should_be_resumed(handle, gpuIds, pcibusIds=None):
    '''
    Make sure that when we watch on all GPUs, the watch should be resumed when the GPU is attached back.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    # Since we will watch on all GPUs, we need to make sure that the number of GPUs in the system is the same as the number of filtered GPUs
    # to prevent some of GPUs are not supported from profiling module.
    allGpusInSystem = dcgmSystem.discovery.GetAllGpuIds()
    if len(allGpusInSystem) != len(gpuIds):
        test_utils.skip_test(
            "Skipping because the number of GPUs in the system is not the same as the number of filtered GPUs")

    gpuId = gpuIds[0]

    gpuUuid = None
    unbindGpuPciBusId = None
    if in_nvml_injection_mode():
        gpuUuid = [get_gpu_uuid(dcgmSystem, gpuId)]
    else:
        if pcibusIds is None or len(pcibusIds) == 0:
            test_utils.skip_test(
                "PCI bus IDs are required in regular mode")
        unbindGpuPciBusId = [pcibusIds[0]]

    group = DcgmGroup.DcgmGroup(
        dcgmHandle, groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS)

    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", [
                                       dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES])
    watch_profiling_field(group, fieldGroup, gpuIds)

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuId], unbindGpuPciBusId)
    for i in range(1, len(gpuIds)):
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)
    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuId], gpuUuid, unbindGpuPciBusId)

    # When GPU status is updated, only means that cache manager is notified, but the watch in profiling module may not be resumed yet.
    retryTimes = 10
    txBytes = 0
    for _ in range(retryTimes):
        dcgmSystem.UpdateAllFields(1)
        values = group.samples.GetLatest_v2(fieldGroup)
        if not values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].isBlank:
            txBytes = values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].value
            break
        time.sleep(1)
    else:
        raise AssertionError("The tx bytes should have been updated")

    waitSeconds = 32
    for _ in range(waitSeconds):
        values = group.samples.GetLatest_v2(fieldGroup)
        assert not values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].isBlank
        currentTxBytes = values.values[dcgm_fields.DCGM_FE_GPU][gpuId][dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES][0].value
        if currentTxBytes != txBytes:
            break
        time.sleep(1)
    else:
        raise AssertionError("The tx bytes should have been updated")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_current_system_injection_nvml(skuFileName="current_test_dcgm_prof_on_all_gpus_should_be_resumed.yaml")
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.with_gpu_filter(test_utils.non_gpm_gpus)
def test_dcgm_prof_on_all_gpus_should_be_resumed(handle, gpuIds):
    helper_test_dcgm_prof_on_all_gpus_should_be_resumed(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.with_gpu_filter(test_utils.non_gpm_gpus)
def test_dcgm_prof_on_all_gpus_should_be_resumed_live(handle, gpuIds, pcibusIds):
    helper_test_dcgm_prof_on_all_gpus_should_be_resumed(
        handle, gpuIds, pcibusIds)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_bind_gpu_back_will_re_register_nvml_events(handle, gpuIds):
    '''
    Make sure that when we bind GPUs back, the NVML events will be re-registered.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    gpuId = gpuIds[0]
    gpuUuid = get_gpu_uuid(dcgmSystem, gpuId)

    ret = dcgm_agent_internal.dcgmResetNvmlInjectFuncCallCount(handle)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuId])
    for i in range(1, len(gpuIds)):
        wait_gpu_status(
            dcgmSystem, gpuIds[i], dcgm_structs_internal.DcgmEntityStatusOk)

    # (All GPUs (8) - Detached one (1)) * 2 (nvmlEventTypeXidCriticalError + nvmlEventMigConfigChange) = 14
    assert_nvml_func_call_count(handle, "nvmlDeviceRegisterEvents", 14)

    ret = dcgm_agent_internal.dcgmResetNvmlInjectFuncCallCount(handle)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    bind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuId], [gpuUuid])
    # All GPUs (8) * 2 (nvmlEventTypeXidCriticalError + nvmlEventMigConfigChange) = 16
    assert_nvml_func_call_count(handle, "nvmlDeviceRegisterEvents", 16)


def helper_test_dcgmi_diag_with_one_gpu_detached(handle, gpuIds, pcibusIds=None):
    """
    Test dcgmi diag command with GPU 0 detached and GPU 1 attached.
    Validates that detached GPU 0 has device_id and serial_num in JSON output.
    """
    import json

    def validate_gpu_entity(entities, gpuId, gpuType):
        entity = next((e for e in entities if e["entity_id"] == gpuId), None)
        assert entity is not None, f"Entity with entity_id {gpuId} ({gpuType} GPU) not found in JSON output"

        deviceId = entity["device_id"]
        serialNum = entity["serial_num"]

        assert deviceId, f"device_id for {gpuType} GPU {gpuId} is null or empty"
        assert serialNum, f"serial_num for {gpuType} GPU {gpuId} is null or empty"

    if len(gpuIds) < 2:
        test_utils.skip_test("Need at least 2 GPUs for this test")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    attachedGpu = gpuIds[0]
    gpuToDetach = gpuIds[1]

    unbindGpuPciBusId = None
    if pcibusIds and gpuToDetach < len(pcibusIds):
        unbindGpuPciBusId = [pcibusIds[gpuToDetach]]

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [gpuToDetach], unbindGpuPciBusId)
    wait_gpu_status(dcgmSystem, gpuToDetach,
                    dcgm_structs_internal.DcgmEntityStatusDetached)

    # Retry logic to wait for proper entity_groups response
    diagResponse = None
    jsonOutput = ""
    for _ in range(100):
        time.sleep(0.1)

        _, stdoutJson, _ = helper_run_dcgmi(
            ["diag", "-r", "1", "-i", f"{gpuToDetach},{attachedGpu}", "-j"])
        jsonOutput = '\n'.join(stdoutJson)

        try:
            diagResponse = json.loads(jsonOutput)
            if "entity_groups" in diagResponse:
                break
        except json.JSONDecodeError:
            pass

    try:
        entities = diagResponse["entity_groups"][0]["entities"]

        # Validate attached and detached GPUs
        validate_gpu_entity(entities, gpuToDetach, "detached")
        validate_gpu_entity(entities, attachedGpu, "attached")

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"Failed to parse or access JSON: {e}")
        logger.error("JSON Output (for debugging):")
        logger.error(jsonOutput)
        raise
    except AssertionError as e:
        logger.error(f"Validation failed: {e}")
        logger.error("JSON Output (for debugging):")
        logger.error(jsonOutput)
        raise


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_diag_with_one_gpu_detached(handle, gpuIds):
    helper_test_dcgmi_diag_with_one_gpu_detached(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_diag_with_one_gpu_detached_live(handle, gpuIds, pcibusIds):
    if len(gpuIds) != len(pcibusIds):
        test_utils.skip_test("Skip on mixed SKU setup")
    helper_test_dcgmi_diag_with_one_gpu_detached(handle, gpuIds, pcibusIds)


def helper_test_dcgmi_discovery_shows_detached_gpu_status(handle, gpuIds, pcibusIds=None):
    """
    Test 'dcgmi discovery' filtering behavior with detached GPUs.

    Tests:
    1. discovery -l -a: Shows detached GPUs with "Status: DETACHED" and cached values
    2. discovery -l: Shows only active GPUs (detached GPUs hidden, no Status field)
    3. After re-attach: All GPUs shown normally without detached indicators
    """
    if len(gpuIds) < 4:
        test_utils.skip_test("Need at least 4 GPUs for this test")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    detachedGpuIds = [gpuIds[0], gpuIds[3]]
    attachedGpuIds = [gpuId for gpuId in gpuIds if gpuId not in detachedGpuIds]

    unbindGpuPciBusId = None
    detachedGpuUuids = None
    if in_nvml_injection_mode():
        # Get UUIDs before detaching (needed for re-attach later)
        detachedGpuUuids = [get_gpu_uuid(dcgmSystem, gpuId)
                            for gpuId in detachedGpuIds]
    else:
        unbindGpuPciBusId = []
        for gpuId in detachedGpuIds:
            if gpuId >= len(pcibusIds):
                test_utils.skip_test(
                    "Skip mismatch between GPU IDs and PCI bus IDs")
            unbindGpuPciBusId.append(pcibusIds[gpuId])

    # Detach GPUs
    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, detachedGpuIds, unbindGpuPciBusId)
    for gpuId in attachedGpuIds:
        wait_gpu_status(dcgmSystem, gpuId,
                        dcgm_structs_internal.DcgmEntityStatusOk)

    # Verify API returns cached values
    for gpuId in detachedGpuIds:
        attrs = dcgmSystem.discovery.GetGpuAttributes(gpuId)
        assert attrs.identifiers.deviceName != dcgmvalue.DCGM_STR_BLANK, f"GPU {gpuId}: cached name missing"
        assert attrs.identifiers.uuid != dcgmvalue.DCGM_STR_BLANK, f"GPU {gpuId}: cached UUID missing"

    # Get discovery output with -a flag to show all GPUs including detached ones
    _, stdout_lines, _ = helper_run_dcgmi(["discovery", "-l", "-a"])
    output = '\n'.join(stdout_lines)
    logger.debug(
        f"discovery output for -l -a with with detached GPUs: {output}")
    assert output, "Failed to get dcgmi discovery output"

    # Verify detached GPUs show status and cached values in discovery output
    for gpuId in detachedGpuIds:
        section = get_gpu_section(output, gpuId)
        assert section, f"GPU {gpuId} not found in output"
        assert "Status: DETACHED" in section, f"GPU {gpuId}: missing 'Status: DETACHED'\n{section}"
        assert dcgmvalue.DCGM_STR_BLANK not in section, f"GPU {gpuId}: has {dcgmvalue.DCGM_STR_BLANK} values\n{section}"

    # Verify attached GPUs do not show status (only non-OK statuses are shown)
    for gpuId in attachedGpuIds:
        section = get_gpu_section(output, gpuId)
        assert "Status:" not in section, f"GPU {gpuId}: should not show status (OK is implicit)\n{section}"

    # Verify that without -a flag, detached GPUs are NOT shown (default behavior shows only active GPUs)
    _, stdout_lines_no_flag, _ = helper_run_dcgmi(["discovery", "-l"])
    output_no_flag = '\n'.join(stdout_lines_no_flag)
    logger.debug(
        f"discovery output for -l with detatched GPUs hidden: {output_no_flag}")
    assert output_no_flag, "Failed to get dcgmi discovery output without -a flag"

    # Verify NO detached status information appears in output without -a flag
    assert "DETACHED" not in output_no_flag, "Output without -a flag should not contain 'DETACHED' at all"

    # Detached GPUs should NOT appear in output without -a flag
    for gpuId in detachedGpuIds:
        section = get_gpu_section(output_no_flag, gpuId)
        assert not section, f"GPU {gpuId} should NOT appear in output without -a flag (detached GPU)\n{output_no_flag}"

    # Attached GPUs should still appear without any status field
    for gpuId in attachedGpuIds:
        section = get_gpu_section(output_no_flag, gpuId)
        assert section, f"GPU {gpuId} should appear in output without -a flag (active GPU)"
        assert "Status:" not in section, f"GPU {gpuId}: should not show status (OK is implicit)"

    # Re-attach the previously detached GPUs
    bind_gpu_and_wait_for_status_updated(
        dcgmSystem, detachedGpuIds, detachedGpuUuids, unbindGpuPciBusId)
    for gpuId in detachedGpuIds:
        wait_gpu_status(dcgmSystem, gpuId,
                        dcgm_structs_internal.DcgmEntityStatusOk)

    # Get discovery output after re-attach (both with and without -a flag should now show all GPUs)
    _, stdout_lines_after, _ = helper_run_dcgmi(["discovery", "-l"])
    output_after = '\n'.join(stdout_lines_after)
    assert output_after, "Failed to get dcgmi discovery output after re-attach"

    # Verify re-attached GPUs no longer show detached status or "(last known)" suffix
    for gpuId in detachedGpuIds:
        section = get_gpu_section(output_after, gpuId)
        assert section, f"GPU {gpuId} not found in output after re-attach"
        assert "Status: DETACHED" not in section, f"GPU {gpuId}: still shows DETACHED after re-attach\n{section}"
        assert "(last known)" not in section, f"GPU {gpuId}: still shows '(last known)' after re-attach\n{section}"
        assert dcgmvalue.DCGM_STR_BLANK not in section, f"GPU {gpuId}: has {dcgmvalue.DCGM_STR_BLANK} values after re-attach\n{section}"


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_discovery_shows_detached_gpu_status(handle, gpuIds):
    helper_test_dcgmi_discovery_shows_detached_gpu_status(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_dcgmi_discovery_shows_detached_gpu_status_live(handle, gpuIds, pcibusIds):
    helper_test_dcgmi_discovery_shows_detached_gpu_status(
        handle, gpuIds, pcibusIds)


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_nvml()
def test_dcgmi_discovery_detached_status_after_detach_driver(handle, gpuIds):
    """Test 'dcgmi discovery -l -a' shows all GPUs with DETACHED status after detach-driver."""
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    # Get UUIDs before detaching
    gpuUuids = {gpuId: get_gpu_uuid(dcgmSystem, gpuId) for gpuId in gpuIds}

    # Detach driver and get discovery output with -a flag to show all GPUs
    dcgm_agent.dcgmDetachDriver(handle)
    _, stdout_lines, _ = helper_run_dcgmi(["discovery", "-l", "-a"])
    output = '\n'.join(stdout_lines)

    # Verify all GPUs show detached status with cached values
    for gpuId in gpuIds:
        section = get_gpu_section(output, gpuId)
        assert section, f"GPU {gpuId} not found in output"
        assert "Status: DETACHED" in section and "(last known)" in section, \
            f"GPU {gpuId}: missing detached indicators\n{section}"
        assert all(field in section for field in ["Name:", "PCI Bus ID:", "Device UUID:"]), \
            f"GPU {gpuId}: missing required fields\n{section}"
        assert dcgmvalue.DCGM_STR_BLANK not in section, \
            f"GPU {gpuId}: has blank values\n{section}"
        assert gpuUuids[gpuId] in section, f"GPU {gpuId}: UUID mismatch\n{section}"

    # Verify that without -a flag, NO GPUs are shown (all are detached)
    _, stdout_lines_no_flag, _ = helper_run_dcgmi(["discovery", "-l"])
    output_no_flag = '\n'.join(stdout_lines_no_flag)
    assert "DETACHED" not in output_no_flag, "Output without -a flag should not contain 'DETACHED' at all"

    for gpuId in gpuIds:
        section = get_gpu_section(output_no_flag, gpuId)
        assert not section, f"GPU {gpuId} should NOT appear without -a flag (all GPUs are detached)\n{output_no_flag}"

    # Re-attach and verify status cleared
    dcgm_agent.dcgmAttachDriver(handle)
    _, stdout_lines_after, _ = helper_run_dcgmi(["discovery", "-l"])
    output_after = '\n'.join(stdout_lines_after)

    for gpuId in gpuIds:
        section = get_gpu_section(output_after, gpuId)
        assert section, f"GPU {gpuId} not found after re-attach"
        assert "Status:" not in section and "(last known)" not in section, \
            f"GPU {gpuId}: still shows detached indicators\n{section}"
        assert all(field in section for field in ["Name:", "PCI Bus ID:", "Device UUID:"]), \
            f"GPU {gpuId}: missing fields after re-attach\n{section}"
        assert dcgmvalue.DCGM_STR_BLANK not in section, \
            f"GPU {gpuId}: has blank values after re-attach\n{section}"


def inject_p2p_status(handle, gpuId, nvmlIndex, p2pStatus):
    injectedRetsArray = nvml_injection.c_injectNvmlRet_t * 1
    injectedRets = injectedRetsArray()
    injectedRets[0].nvmlRet = dcgm_nvml.NVML_SUCCESS
    injectedRets[0].values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_GPUP2PSTATUS
    injectedRets[0].values[0].value.GpuP2PStatus = p2pStatus
    injectedRets[0].valueCount = 1

    extraKeysArray = nvml_injection_structs.c_injectNvmlVal_t * 2
    extraKeys = extraKeysArray()
    extraKeys[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_DEVICE
    # For device handle parameter, we pass the nvmlIndex. The dcgmInjectNvmlDevice will translate it to the device handle.
    extraKeys[0].value.Device = nvmlIndex
    extraKeys[1].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_GPUP2PCAPSINDEX
    extraKeys[1].value.GpuP2PCapsIndex = dcgm_nvml.NVML_P2P_CAPS_INDEX_NVLINK

    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(
        handle, gpuId, "P2PStatus", extraKeys, 2, injectedRets)
    assert (ret == dcgm_structs.DCGM_ST_OK)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_nvlink_p2p_status_field_with_detached_gpu(handle, gpuIds):
    '''
    Test the query of nvlink p2p status with a detached GPU on the system.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    gpuId = gpuIds[0]
    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [gpuIds[1]])

    # nvmlIndex 0 is GPU 0 and since GPU 1 is detached, its nvmlIndex is released.
    # That means nvmlIndex 1 is GPU 2, and so on.
    # Here, we inject the P2P status for GPU 0 to those GPUs with DCGM GPU id 2 to 7.
    # We don't need to inject the P2P status for GPU 0 to GPU 1, because GPU 1 is detached.
    # The codebase must avoid calling NVML functions on this situation.
    # If it does call, the relevant NVML call will fail and return error leads to the test failure.
    for nvmlIndex in range(1, 7):
        inject_p2p_status(handle, gpuId, nvmlIndex,
                          dcgm_nvml.NVML_P2P_STATUS_OK)

    fieldId = dcgm_fields.DCGM_FI_DEV_P2P_NVLINK_STATUS
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgm_agent_internal.dcgmWatchFieldValue(
        dcgmHandle.handle, gpuId, fieldId, 60000000, 3600.0, 0)
    dcgmSystem.UpdateAllFields(1)
    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(
        dcgmHandle.handle, gpuId, [fieldId, ])
    # GPU 0 is itself, GPU 1 is detached.
    assert (values[0].value.i64 ==
            0b11111100), f"Actual value: {values[0].value.i64}"


def assert_nvlink_p2p_status(inOutStatus, detachedGpuId):
    for i in range(inOutStatus.numGpus):
        assert (inOutStatus.gpus[i].entityId ==
                i), f"Expected Entity {i} ID {i}, got {inOutStatus.gpus[i].entityId}."

        if i == detachedGpuId:
            # for detached GPU, its linkStatus to all GPUs should be not supported.
            for j in range(inOutStatus.numGpus):
                status = inOutStatus.gpus[i].linkStatus[j]
                assert (status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported), \
                    f"Expected LinkP2PStatus for GPU {i} to {j} as {dcgm_structs.DcgmNvLinkP2PStatusNotSupported}, got {status}."
        else:
            # for other GPUs, its linkStatus to itself and the detached GPU should be not supported.
            for j in range(inOutStatus.numGpus):
                status = inOutStatus.gpus[i].linkStatus[j]
                if j == i or j == detachedGpuId:
                    assert (status == dcgm_structs.DcgmNvLinkP2PStatusNotSupported), \
                        f"Expected LinkP2PStatus for GPU {i} to {j} as {dcgm_structs.DcgmNvLinkP2PStatusNotSupported}, got  {status}."
                else:
                    assert (status == dcgm_structs.DcgmNvLinkP2PStatusOK), \
                        f"Expected LinkP2PStatus for GPU {i} to {j} as {dcgm_structs.DcgmNvLinkP2PStatusOK}, got  {status}."


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_nvlink_p2p_status_api_with_detached_gpu(handle, gpuIds):
    '''
    Test the dcgmGetNvLinkP2PStatus with a detached GPU on the system.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    detachedGpuId = gpuIds[1]
    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [detachedGpuId])

    gpuIdToNvmlIndex = {}
    # nvmlIndex 0 is GPU 0 and since GPU 1 is detached, its nvmlIndex is released.
    # That means nvmlIndex 1 is GPU 2, and so on.
    gpuIdToNvmlIndex[gpuIds[0]] = 0
    for i in range(2, len(gpuIds)):
        gpuIdToNvmlIndex[gpuIds[i]] = i - 1

    # Let all GPUs be connected to each other.
    for gpuId in gpuIds:
        if gpuId == detachedGpuId:
            continue
        for targetGpuId, targetNvmlIndex in gpuIdToNvmlIndex.items():
            if targetGpuId == gpuId:
                continue
            inject_p2p_status(handle, gpuId, targetNvmlIndex,
                              dcgm_nvml.NVML_P2P_STATUS_OK)

    inOutStatus = dcgm_structs.c_dcgmNvLinkP2PStatus_v1()
    inOutStatus.numGpus = 0  # full retrieval.

    dcgm_agent.dcgmGetNvLinkP2PStatus(handle, inOutStatus)
    assert (len(gpuIds) ==
            inOutStatus.numGpus), f"Expected {len(gpuIds)} GPUs, got {inOutStatus.numGpus}."
    assert_nvlink_p2p_status(inOutStatus, detachedGpuId)

    inOutStatus = dcgm_structs.c_dcgmNvLinkP2PStatus_v1()
    inOutStatus.numGpus = 2  # partial retrieval.

    for i in range(inOutStatus.numGpus):
        inOutStatus.gpus[i].entityId = gpuIds[i]

    dcgm_agent.dcgmGetNvLinkP2PStatus(handle, inOutStatus)
    assert (
        2 == inOutStatus.numGpus), f"Expected 2 GPUs, got {inOutStatus.numGpus}."
    assert_nvlink_p2p_status(inOutStatus, detachedGpuId)


def helper_test_get_nvlink_status_api_with_detached_gpu(handle, gpuIds, pcibusIds=None):
    '''
    Test the dcgmGetNvLinkLinkStatus with a detached GPU on the system.
    '''
    if len(gpuIds) < 2:
        test_utils.skip_test("Need at least 2 GPUs for this test")

    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    detachedGpuId = gpuIds[1]
    unbindGpuPciBusId = None
    if pcibusIds and detachedGpuId < len(pcibusIds):
        unbindGpuPciBusId = [pcibusIds[detachedGpuId]]

    unbind_gpu_and_wait_for_status_updated(
        dcgmSystem, [detachedGpuId], unbindGpuPciBusId)

    linkStatus = dcgm_agent.dcgmGetNvLinkLinkStatus(handle)
    assert (len(gpuIds) - 1 ==
            linkStatus.numGpus), f"Expected {len(gpuIds) - 1} GPUs, got {linkStatus.numGpus}."
    assert (linkStatus.gpus[0].entityId ==
            0), f"Expected Entity 0 ID 0, got {linkStatus.gpus[0].entityId}."
    for i in range(1, linkStatus.numGpus):
        # GPU 1 is detached, so the entities after it should be shifted by 1.
        assert (linkStatus.gpus[i].entityId == i +
                1), f"Expected Entity {i} ID {i}, got {linkStatus.gpus[i].entityId}."


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_get_nvlink_status_api_with_detached_gpu(handle, gpuIds):
    helper_test_get_nvlink_status_api_with_detached_gpu(
        handle, gpuIds, pcibusIds=None)


@test_utils.run_with_developer_mode()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_pci_bus_ids()
@test_utils.auto_restore_detached_gpus()
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
def test_get_nvlink_status_api_with_detached_gpu_live(handle, gpuIds, pcibusIds):
    helper_test_get_nvlink_status_api_with_detached_gpu(
        handle, gpuIds, pcibusIds)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_nvml_injection_folder()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_nvlink_errors_on_detached_gpu(handle, gpuIds):
    '''
    Test the dcgmi nvlink -e -g on a detached GPU.
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    detachedGpuId = gpuIds[0]
    unbind_gpu_and_wait_for_status_updated(dcgmSystem, [detachedGpuId])

    _, stdoutLines, _ = helper_run_dcgmi(
        ["nvlink", "-e", "-g", f"{detachedGpuId}"])
    stdout = '\n'.join(stdoutLines)
    assert "GPU is lost" in stdout, f"Expected 'GPU is lost' in stdout, got {stdout}."
