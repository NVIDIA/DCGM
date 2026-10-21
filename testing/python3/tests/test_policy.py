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

import dcgm_structs
import dcgm_structs_internal
import dcgm_agent
import dcgm_agent_internal
import dcgm_field_injection_helpers
import dcgmvalue
import pydcgm
import logger
import test_utils
import dcgm_fields
from dcgm_structs import dcgmExceptionClass

import gc
import json
import time
from ctypes import *
import queue
import sys
import os
import tempfile
import threading
import uuid

# ---------------
# global variables
# ---------------
# How long to wait for policy callbacks to occur. The loop runs every 10
# seconds in the host engine, so this should be 10 seconds + some fuzz
# time
POLICY_CALLBACK_TIMEOUT_SECS = 15
INJECT_TS_OFFSET_SEC = 60
POLICY_NO_VIOLATION_TIMEOUT_SECS = 2
DCGMI_TEMP_FIELD_NAME = "gpu_temp"


# ---------------
# helpers
# ---------------

def _clear_field_policies(handle):
    dcgm_agent.dcgmPolicyDelete(handle, deleteAll=True)


def _create_temp_policy(handle,
                        gpuIds,
                        enabled=1,
                        channelMask=dcgm_structs.DCGM_POLICY_CHANNEL_CONSOLE,
                        rateLimitSec=1):
    return dcgm_agent.dcgmPolicyCreate(
        handle,
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        "python_temp_policy",
        threshold=70.0,
        policyOperator=dcgm_structs.DCGM_POLICY_OP_GT,
        entities=gpuIds,
        channelMask=channelMask,
        enabled=enabled,
        rateLimitSec=rateLimitSec)


def _inject_temp(handle, gpuId, value):
    dcgm_field_injection_helpers.inject_value(
        handle,
        gpuId,
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        value,
        INJECT_TS_OFFSET_SEC)


def _get_temp_violations(handle, policyId, gpuId, sinceTimestamp=0):
    return [
        violation for violation in dcgm_agent.dcgmPolicyGetViolations(handle, sinceTimestamp)
        if violation.policyId == policyId
        and violation.fieldId == dcgm_fields.DCGM_FI_DEV_GPU_TEMP
        and violation.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU
        and violation.entity.entityId == gpuId
    ]


def _find_temp_violation(handle, policyId, gpuId, timeoutSec=10):
    deadline = time.time() + timeoutSec
    while time.time() < deadline:
        matching = _get_temp_violations(handle, policyId, gpuId)
        if matching:
            return matching[0]
        time.sleep(0.2)
    return None


def _wait_for_temp_violations(handle, policyId, gpuId, count, sinceTimestamp=0, timeoutSec=10):
    deadline = time.time() + timeoutSec
    matching = []
    while time.time() < deadline:
        matching = _get_temp_violations(
            handle, policyId, gpuId, sinceTimestamp)
        if len(matching) >= count:
            return matching
        time.sleep(0.2)
    return matching


def _violation_count_remains(handle, policyId, gpuId, expectedCount, timeoutSec):
    deadline = time.time() + timeoutSec
    while time.time() < deadline:
        if len(_get_temp_violations(handle, policyId, gpuId)) != expectedCount:
            return False
        time.sleep(0.2)
    return True


def _run_dcgmi_policy_json(args):
    retValue, stdoutLines, stderrLines = test_utils.run_dcgmi_command(
        ["policy", "--json"] + args)
    assert retValue == c_ubyte(dcgm_structs.DCGM_ST_OK).value, \
        "dcgmi policy failed: stdout=%s stderr=%s" % (stdoutLines, stderrLines)
    output = "\n".join(stdoutLines).strip()
    assert output, "dcgmi policy returned no JSON output for args %s" % args
    return json.loads(output)


def _run_dcgmi_policy(args):
    retValue, stdoutLines, stderrLines = test_utils.run_dcgmi_command(
        ["policy"] + args)
    assert retValue == c_ubyte(dcgm_structs.DCGM_ST_OK).value, \
        "dcgmi policy failed: stdout=%s stderr=%s" % (stdoutLines, stderrLines)
    return stdoutLines


def _run_dcgmi_policy_expect_failure(args, expectedText):
    retValue, stdoutLines, stderrLines = test_utils.run_dcgmi_command(
        ["policy"] + args, validate=True)
    assert retValue != c_ubyte(dcgm_structs.DCGM_ST_OK).value, \
        "dcgmi policy unexpectedly succeeded: stdout=%s stderr=%s" % (
            stdoutLines, stderrLines)
    assert expectedText in "\n".join(stdoutLines + stderrLines)


def _dcgmi_policy_create_args(fieldId, name, threshold, entities, operator=">", channels="console", rateLimitSec=1):
    return [
        "--create", "--field", str(
            fieldId), "--name", name, "--operator", operator,
        "--threshold", str(threshold), "--entities", str(
            entities), "--channels", channels,
        "--rate-limit", str(rateLimitSec),
    ]


def _create_dcgmi_policy(fieldId, name, threshold, entities, operator=">", channels="console", rateLimitSec=1):
    return _run_dcgmi_policy_json(
        _dcgmi_policy_create_args(fieldId, name, threshold, entities, operator, channels, rateLimitSec))


def _get_dcgmi_policy(fieldId, policyId):
    return _run_dcgmi_policy_json([
        "--get",
        "--field", str(fieldId),
        "--policy-id", str(policyId),
    ])


def _run_concurrent_dcgmi_policy_commands(commands):
    results = [None] * len(commands)
    errors = [None] * len(commands)

    def run_command(index, args):
        try:
            results[index] = _run_dcgmi_policy_json(args)
        except Exception as error:
            errors[index] = error

    threads = [
        threading.Thread(target=run_command, args=(index, args))
        for index, args in enumerate(commands)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for error in errors:
        if error is not None:
            raise error

    return results


def _run_concurrent_dcgmi_policy_commands_allow_failures(commands):
    results = [None] * len(commands)

    def run_command(index, args):
        results[index] = test_utils.run_dcgmi_command(
            ["policy"] + args, validate=True)

    threads = [
        threading.Thread(target=run_command, args=(index, args))
        for index, args in enumerate(commands)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return results


# creates a callback function for dcgmPolicyRegister calls which will push its args
# into "queue" when it is called.  This allows synchronization on the callback by
# retrieving the args from the queue as well as checks of the args via asserts.


def create_c_callback(queue=None):
    @CFUNCTYPE(None,
               POINTER(dcgm_structs.c_dcgmPolicyCallbackResponse_v2),
               c_uint64)
    def c_callback(response, userData):
        if queue:
            # copy data into a python struct so that it is the right format and
            # is not lost when "response" var is lost
            callbackResp = dcgm_structs.c_dcgmPolicyCallbackResponse_v2()
            memmove(addressof(callbackResp), response,
                    callbackResp.FieldsSizeof())
            queue.put(callbackResp)

            callbackUserData = cast(userData, py_object).value
            queue.put(callbackUserData)
    return c_callback


def create_field_policy_callback(queue=None):
    @CFUNCTYPE(c_int32,
               POINTER(dcgm_structs.c_dcgm_policy_violation_t),
               c_uint64)
    def c_callback(violation, userData):
        if queue:
            callbackViolation = dcgm_structs.c_dcgm_policy_violation_t()
            memmove(addressof(callbackViolation), violation,
                    sizeof(callbackViolation))
            queue.put(callbackViolation)
            queue.put(cast(userData, py_object).value)
        return 0
    return c_callback


def _assert_policies_equal(policy1, policy2):
    assert (policy1)  # check if None
    assert (policy2)
    assert (policy1.version == policy2.version)
    assert (policy1.condition == policy2.condition)
    assert (policy1.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag ==
            policy2.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag)
    assert (policy1.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean ==
            policy2.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean)


def helper_dcgm_policy_inject_eccerror(handle, gpuIds):
    """
    Verifies that we can inject an error into the ECC counters and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean = True

    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds(
        "helper_dcgm_policy_inject_eccerror", gpuIds)
    group.policy.Set(newPolicy)

    # the order of the callbacks will change once implementation is complete
    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)

    # send newPolicy as user data for callback.
    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_DBE, c_callback, newPolicy)

    # inject baseline sample first (required for delta detection)
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_DEV
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC) * 1000000.0)
    field.value.i64 = 0  # baseline
    logger.debug("injecting baseline %s for gpuId %d" %
                 (str(field), gpuIds[0]))

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # inject an error into ECC (delta detection will see 0 -> 1)
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC + 5) * 1000000.0)
    field.value.i64 = 1  # new error
    logger.debug("injecting error %s for gpuId %d" % (str(field), gpuIds[0]))

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
        callbackUserData = callbackQueue.get(
            timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (dcgm_structs.DCGM_POLICY_COND_DBE == callbackResp.condition), \
        ("error callback was not for a DBE error, got: %s" % callbackResp.condition)
    assert (1 == callbackResp.val.dbe.numerrors), 'Expected 1 DBE error but got %s' % callbackResp.val.dbe.numerrors
    assert (dcgm_structs.c_dcgmPolicyConditionDbe_t.LOCATIONS['DEVICE'] ==
            callbackResp.val.dbe.location), 'got: %s' % callbackResp.val.dbe.location

    # check that callback occured with the correct user data
    assert (callbackUserData.version == newPolicy.version), "User data at callback is incorrect. " \
        "Received: [%d], Expected: [%d]." % (
            callbackUserData.version, newPolicy.version)

    # check that callback response has correct gpuId for which error was
    # injected.
    assert (callbackResp.gpuId == gpuIds[0]), "GPU ID in callback response is incorrect. " \
        "Received: [%d], Expected: [%d]." % (callbackResp.gpuId, gpuIds[0])


def helper_test_dcgm_policy_inject_xiderror(handle, gpuIds):
    """
    Verifies that we can inject an XID error and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()

    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_XID
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_XID].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_XID].val.boolean = True

    dcgmHandle = pydcgm.DcgmHandle(handle)
    validDeviceId = -1
    devices = gpuIds
    for x in devices:
        fvSupported = dcgm_agent_internal.dcgmGetLatestValuesForFields(
            handle, x, [dcgm_fields.DCGM_FI_DEV_XID_ERROR, ])
        if (fvSupported[0].value.i64 != dcgmvalue.DCGM_INT64_NOT_SUPPORTED):
            validDeviceId = x
            break
    if (validDeviceId == -1):
        test_utils.skip_test(
            "Can only run if at least one GPU that supports XID errors is present")

    group = pydcgm.DcgmGroup(
        dcgmHandle,
        groupName="helper_test_dcgm_policy_inject_xiderror",
        groupType=dcgm_structs.DCGM_GROUP_EMPTY)
    group.AddGpu(validDeviceId)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_XID, c_callback)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_XID_ERROR
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    # set the injected data into the future
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC) * 1000000.0)
    field.value.i64 = 16

    ret = dcgm_agent_internal.dcgmInjectFieldValue(
        handle, validDeviceId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (dcgm_structs.DCGM_POLICY_COND_XID == callbackResp.condition), \
        ("XID error callback was not for a XID error, got: %s" %
         callbackResp.condition)
    assert (
        16 == callbackResp.val.xid.errnum), ('Expected XID error 16 but got %s' %
                                             callbackResp.val.xid.errnum)


def helper_dcgm_policy_inject_pcierror(handle, gpuIds):
    """
    Verifies that we can inject an error into the PCI counters and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()

    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_PCI
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_PCI].tag = 1
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_PCI].val.llval = 0

    gpuId = gpuIds[0]

    group = pydcgm.DcgmGroup(
        pydcgm.DcgmHandle(handle),
        groupName="helper_dcgm_policy_inject_pcierror",
        groupType=dcgm_structs.DCGM_GROUP_EMPTY)
    group.AddGpu(gpuId)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_PCI, c_callback)

    # inject baseline sample first (required for delta detection)
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_TOTAL
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC) * 1000000.0)
    field.value.i64 = 0  # baseline

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # inject an error (delta detection will see 0 -> 1)
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC + 5) * 1000000.0)
    field.value.i64 = 1  # new error

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (dcgm_structs.DCGM_POLICY_COND_PCI == callbackResp.condition), \
        ("PCI error callback was not for a PCI error, got: %s" %
         callbackResp.condition)
    assert (1 == callbackResp.val.pci.counter), 'Expected 1 PCI error but got %s' % callbackResp.val.pci.counter

# ---------------------------------------------------------------------------
# legacy tests
# ---------------------------------------------------------------------------


# NO HARDWARE

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_reg_unreg_for_policy_update_standalone(handle, gpuIds):
    """
    Verifies that the reg/unreg path for the policy manager is working
    """
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds('test', gpuIds)

    # must hold ref so func is not GC'ed before c api uses it
    empty_c_callback = create_c_callback()

    # Register/Unregister will throw exceptions if they fail
    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_DBE, empty_c_callback, None)
    group.policy.Unregister(dcgm_structs.DCGM_POLICY_COND_DBE)


# NO HARDWARE

@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_policy_negative_register_standalone(handle):
    """
    Verifies that the register function does not allow a bad groupId value
    """
    policy = pydcgm.DcgmGroupPolicy(pydcgm.DcgmHandle(handle), 9999, None)
    # must hold ref so func is not GC'ed before c api uses it
    empty_c_callback = create_c_callback()
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        policy.Register(dcgm_structs.DCGM_POLICY_COND_DBE, empty_c_callback)


# NO HARDWARE

@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_policy_negative_unregister_standalone(handle):
    """
    Verifies that the unregister function does not allow a bad groupId value
    """
    policy = pydcgm.DcgmGroupPolicy(pydcgm.DcgmHandle(handle), 9999, None)
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        policy.Unregister(dcgm_structs.DCGM_POLICY_COND_DBE)


# NO HARDWARE

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_policy_set_get_violation_policy_standalone(handle, gpuIds):
    """
    Verifies that set and get violation policy work
    """
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds("test1", gpuIds)

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean = True

    group.policy.Set(newPolicy)
    policies = group.policy.Get()

    _assert_policies_equal(policies[0], newPolicy)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_eccerror_standalone(handle, gpuIds):
    helper_dcgm_policy_inject_eccerror(handle, gpuIds)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_nvlinkerror_standalone(handle, gpuIds):
    """
    Verifies that we can inject an error into the NVLINK error and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()

    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_NVLINK
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_NVLINK].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_NVLINK].val.boolean = True

    # find a GPU that supports nvlink (otherwise internal test will ignore it)
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds(
        'test_dcgm_policy_inject_nvlinkerror_standalone1', gpuIds)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_NVLINK, c_callback)

    # inject baseline sample first (required for delta detection)
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC) * 1000000.0)
    field.value.i64 = 0  # baseline

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # inject an error (delta detection will see 0 -> 1)
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC + 5) * 1000000.0)
    field.value.i64 = 1  # new error

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (dcgm_structs.DCGM_POLICY_COND_NVLINK == callbackResp.condition), \
        ("NVLINK error callback was not for a NVLINK error, got: %s" %
         callbackResp.condition)
    assert (dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL == callbackResp.val.nvlink.fieldId), \
        ("Expected 130 fieldId but got %s" % callbackResp.val.nvlink.fieldId)
    assert (1 == callbackResp.val.nvlink.counter), 'Expected 1 PCI error but got %s' % callbackResp.val.nvlink.counter


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_xiderror_standalone(handle, gpuIds):
    helper_test_dcgm_policy_inject_xiderror(handle, gpuIds)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_pcierror_standalone(handle, gpuIds):
    helper_dcgm_policy_inject_pcierror(handle, gpuIds)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_retiredpages_standalone(handle, gpuIds):
    """
    Verifies that we can inject an error into the retired pages counters and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()

    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_MAX_PAGES_RETIRED].tag = 1
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_MAX_PAGES_RETIRED].val.llval = 5

    # find a GPU that supports ECC and retired pages (otherwise internal test
    # will ignore it)
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds("test1", gpuIds)

    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED, c_callback)

    # inject an error into ECC
    numPages = 10
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    # set the injected data into the future
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC) * 1000000.0)
    field.value.i64 = numPages

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # inject a SBE too so that the health check code gets past its internal
    # checks
    field.fieldId = dcgm_fields.DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED == callbackResp.condition), \
        ("error callback was not for a retired pages, got: %s" %
         callbackResp.condition)
    assert (numPages == callbackResp.val.mpr.dbepages), \
        'Expected %s errors but got %s' % (
            numPages, callbackResp.val.mpr.dbepages)

# NO HARDWARE


@test_utils.run_with_standalone_host_engine(40)
def test_dcgm_policy_get_with_no_gpus_standalone(handle):
    '''
    Test that getting the policies when no GPUs are in the group raises an exception
    '''
    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(
        handle), groupType=dcgm_structs.DCGM_GROUP_EMPTY, groupName="test1")

    with test_utils.assert_raises(pydcgm.DcgmException):
        policies = group.policy.Get()


def helper_dcgm_policy_get_with_some_gpus_standalone(handle, gpuIds):
    '''
    Test that getting the policies returns the correct number of policies as GPUs in the system
    when "count" is not specified for policy.Get
    '''
    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(
        handle), groupType=dcgm_structs.DCGM_GROUP_EMPTY, groupName="test1")

    group.AddGpu(gpuIds[0])
    policies = group.policy.Get()
    assert len(policies) == 1, len(policies)


@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_only_with_live_gpus()
def test_dcgm_policy_get_with_some_gpus_standalone(handle, gpuIds):
    helper_dcgm_policy_get_with_some_gpus_standalone(handle, gpuIds)


def helper_dcgm_policy_inject_powererror(handle, gpuIds):
    """
    Verifies that we can inject an error into the Power level and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_POWER
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_POWER].tag = 1
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_POWER].val.llval = 200

    gpuId = gpuIds[0]

    group = pydcgm.DcgmGroup(
        pydcgm.DcgmHandle(handle),
        groupName="helper_dcgm_policy_inject_powererror",
        groupType=dcgm_structs.DCGM_GROUP_EMPTY)
    group.AddGpu(gpuId)
    group.policy.Set(newPolicy)

    # the order of the callbacks will change once implementation is complete
    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_POWER, c_callback, None)

    # inject an error into Power Violation
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_BOARD_POWER_WATTS
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_DOUBLE)
    # set the injected data into the future
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC) * 1000000.0)
    field.value.dbl = 210.0
    logger.debug("injecting %s for gpuId %d" % (str(field), gpuIds[0]))

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (dcgm_structs.DCGM_POLICY_COND_POWER == callbackResp.condition), \
        ("error callback was not for a Power Violation error, got: %s" %
         callbackResp.condition)
    assert (210 == callbackResp.val.power.powerViolation), \
        'Expected Power Violation at 210 but got %s' % callbackResp.val.power.powerViolation


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_powererror_standalone(handle, gpuIds):
    helper_dcgm_policy_inject_powererror(handle, gpuIds)


def helper_dcgm_policy_inject_thermalerror(handle, gpuIds):
    """
    Verifies that we can inject an error into the Thermal level and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_THERMAL
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_THERMAL].tag = 1
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_THERMAL].val.llval = 90

    gpuId = gpuIds[0]

    group = pydcgm.DcgmGroup(
        pydcgm.DcgmHandle(handle),
        groupName="helper_dcgm_policy_inject_thermalerror",
        groupType=dcgm_structs.DCGM_GROUP_EMPTY)
    group.AddGpu(gpuId)
    group.policy.Set(newPolicy)

    # the order of the callbacks will change once implementation is complete
    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(
        dcgm_structs.DCGM_POLICY_COND_THERMAL, c_callback, None)

    # inject an error into Power Violation
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_GPU_TEMP_CELSIUS
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    # set the injected data into the future
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC) * 1000000.0)
    field.value.i64 = 95
    logger.debug("injecting %s for gpuId %d" % (str(field), gpuIds[0]))

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackResp = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        raise AssertionError("Callback never happened")

    # check that the callback occurred with the correct arguments
    assert (dcgm_structs.DCGM_POLICY_COND_THERMAL == callbackResp.condition), \
        ("error callback was not for a Thermal Violation error, got: %s" %
         callbackResp.condition)
    assert (95 == callbackResp.val.power.powerViolation), \
        'Expected Thermal Violation at 95 but got %s' % callbackResp.val.power.powerViolation


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_thermalerror_standalone(handle, gpuIds):
    helper_dcgm_policy_inject_thermalerror(handle, gpuIds)


# Helper functions for delta detection tests
def helper_inject_field_value(handle, gpuId, field, value, tsOffset):
    """Inject a field value with the given timestamp offset."""
    field.ts = int((time.time() + INJECT_TS_OFFSET_SEC + tsOffset) * 1000000.0)
    field.value.i64 = value
    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, field)
    assert ret == dcgm_structs.DCGM_ST_OK


def helper_assert_callback_received(
        callbackQueue,
        gpuId,
        expectedValue,
        valueGetter,
        msg):
    """Assert that a callback was received with expected gpuId and value.
    Note: create_c_callback puts 2 items per callback (response + userData), so we consume both.
    """
    try:
        callbackResp = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        raise AssertionError(f"{msg}: callback never happened")
    # Consume the userData that create_c_callback also puts in the queue
    try:
        callbackQueue.get(timeout=1)
    except queue.Empty:
        pass
    assert callbackResp.gpuId == gpuId, (
        f"{msg}: wrong gpuId {callbackResp.gpuId}"
    )
    actualValue = valueGetter(callbackResp)
    assert actualValue == expectedValue, f"{msg}: expected {expectedValue}, got {actualValue}"


def helper_assert_no_callback(callbackQueue, msg):
    """Assert that no callback was received."""
    try:
        callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
        raise AssertionError(f"{msg}: got unexpected callback")
    except queue.Empty:
        pass  # Expected


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus(1)
def test_dcgm_policy_pcie_delta_detection(handle, gpuIds):
    """
    Verifies PCIe policy uses delta-based detection.
    Uses GetSamples() from cache - needs 2 samples before delta can be computed.
    1st=5 (no callback - baseline), 2nd=5 (no callback), 3rd=10 (callback), 4th=3 (no callback), 5th=7 (callback)
    """
    gpuId = gpuIds[0]

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_PCI
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_PCI].tag = 1
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_PCI].val.llval = 0

    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(handle),
                             groupName="test_pcie_delta_detection",
                             groupType=dcgm_structs.DCGM_GROUP_EMPTY)
    group.AddGpu(gpuId)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_PCI, c_callback)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_TOTAL
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)

    def valueGetter(r):
        return r.val.pci.counter

    # 1st injection: value=5 -> no callback (establishes baseline, need 2
    # samples for delta)
    helper_inject_field_value(handle, gpuId, field, 5, 0)
    helper_assert_no_callback(callbackQueue, "1st - baseline")

    # 2nd injection: value=5 (same) -> no callback (no delta)
    helper_inject_field_value(handle, gpuId, field, 5, 5)
    helper_assert_no_callback(callbackQueue, "2nd - no change")

    # 3rd injection: value=10 (increased) -> callback (delta=5)
    helper_inject_field_value(handle, gpuId, field, 10, 10)
    helper_assert_callback_received(
        callbackQueue, gpuId, 10, valueGetter, "3rd")

    # 4th injection: value=3 (decreased/reset) -> no callback
    helper_inject_field_value(handle, gpuId, field, 3, 15)
    helper_assert_no_callback(callbackQueue, "4th - reset")

    # 5th injection: value=7 (increased from reset baseline) -> callback
    helper_inject_field_value(handle, gpuId, field, 7, 20)
    helper_assert_callback_received(
        callbackQueue, gpuId, 7, valueGetter, "5th")


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus(1)
def test_dcgm_policy_nvlink_delta_detection(handle, gpuIds):
    """
    Verifies NVLink error policy uses delta-based detection.
    Uses GetSamples() from cache - needs 2 samples before delta can be computed.
    1st=10 (no callback - baseline), 2nd=10 (no callback), 3rd=15 (callback), 4th=5 (no callback), 5th=8 (callback)
    """
    gpuId = gpuIds[0]

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_NVLINK
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_NVLINK].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_NVLINK].val.boolean = True

    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds('test_nvlink_delta', gpuIds)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_NVLINK, c_callback)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)

    def valueGetter(r):
        return r.val.nvlink.counter

    # 1st injection: value=10 -> no callback (establishes baseline, need 2
    # samples for delta)
    helper_inject_field_value(handle, gpuId, field, 10, 0)
    helper_assert_no_callback(callbackQueue, "1st - baseline")

    # 2nd injection: value=10 (same) -> no callback (no delta)
    helper_inject_field_value(handle, gpuId, field, 10, 5)
    helper_assert_no_callback(callbackQueue, "2nd - no change")

    # 3rd injection: value=15 (increased) -> callback (delta=5)
    helper_inject_field_value(handle, gpuId, field, 15, 10)
    helper_assert_callback_received(
        callbackQueue, gpuId, 15, valueGetter, "3rd")

    # 4th injection: value=5 (decreased/reset) -> no callback
    helper_inject_field_value(handle, gpuId, field, 5, 15)
    helper_assert_no_callback(callbackQueue, "4th - reset")

    # 5th injection: value=8 (increased from reset baseline) -> callback
    helper_inject_field_value(handle, gpuId, field, 8, 20)
    helper_assert_callback_received(
        callbackQueue, gpuId, 8, valueGetter, "5th")


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_injection_gpus(1)
def test_dcgm_policy_ecc_delta_detection(handle, gpuIds):
    """
    Verifies ECC error policy uses delta-based detection.
    Uses GetSamples() from cache - needs 2 samples before delta can be computed.
    1st=3 (no callback - baseline), 2nd=3 (no callback), 3rd=7 (callback), 4th=2 (no callback), 5th=4 (callback)
    """
    gpuId = gpuIds[0]

    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].tag = 0
    newPolicy.parms[dcgm_structs.DCGM_POLICY_COND_IDX_DBE].val.boolean = True

    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds('test_ecc_delta', gpuIds)
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_DBE, c_callback)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_DEV
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)

    def valueGetter(r):
        return r.val.dbe.numerrors

    # 1st injection: value=3 -> no callback (establishes baseline, need 2
    # samples for delta)
    helper_inject_field_value(handle, gpuId, field, 3, 0)
    helper_assert_no_callback(callbackQueue, "1st - baseline")

    # 2nd injection: value=3 (same) -> no callback (no delta)
    helper_inject_field_value(handle, gpuId, field, 3, 5)
    helper_assert_no_callback(callbackQueue, "2nd - no change")

    # 3rd injection: value=7 (increased) -> callback (delta=4)
    helper_inject_field_value(handle, gpuId, field, 7, 10)
    helper_assert_callback_received(
        callbackQueue, gpuId, 7, valueGetter, "3rd")

    # 4th injection: value=2 (decreased/reset) -> no callback
    helper_inject_field_value(handle, gpuId, field, 2, 15)
    helper_assert_no_callback(callbackQueue, "4th - reset")

    # 5th injection: value=4 (increased from reset baseline) -> callback
    helper_inject_field_value(handle, gpuId, field, 4, 20)
    helper_assert_callback_received(
        callbackQueue, gpuId, 4, valueGetter, "5th")


# ---------------------------------------------------------------------------
# new tests, dcgm api
# ---------------------------------------------------------------------------

# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_violation(handle, gpuIds):
    """
    Verifies that the new field-based policy API evaluates injected values and
    exposes violations through dcgmPolicyGetViolations.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]
    policyId = _create_temp_policy(handle, [gpuId])
    assert policyId != 0

    _inject_temp(handle, gpuId, 80)

    violation = _find_temp_violation(handle, policyId, gpuId)
    assert violation is not None, "Policy violation never appeared"
    assert violation.currentValue == 80.0
    assert violation.thresholdValue == 70.0
    assert violation.policyOperator == dcgm_structs.DCGM_POLICY_OP_GT


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_disabled(handle, gpuIds):
    """
    Verifies disabled field-based policies do not record recent violations.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]
    policyId = _create_temp_policy(handle, [gpuId], enabled=0)
    assert policyId != 0

    _inject_temp(handle, gpuId, 80)

    violation = _find_temp_violation(handle, policyId, gpuId, timeoutSec=10)
    assert violation is None


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_wrong_gpu(handle, gpuIds):
    """
    Verifies a policy only records violations for targeted GPUs.
    """
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 injected GPUs")

    _clear_field_policies(handle)
    policyId = _create_temp_policy(handle, [gpuIds[0]])
    assert policyId != 0

    _inject_temp(handle, gpuIds[1], 80)

    targetViolation = _find_temp_violation(
        handle, policyId, gpuIds[0], timeoutSec=2)
    injectedViolation = _find_temp_violation(
        handle, policyId, gpuIds[1], timeoutSec=2)
    assert targetViolation is None
    assert injectedViolation is None


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_callback(handle, gpuIds):
    """
    Verifies new field-based callback registration receives the raw policy violation payload.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]
    policyId = _create_temp_policy(
        handle,
        [gpuId],
        channelMask=dcgm_structs.DCGM_POLICY_CHANNEL_CALLBACK,
        rateLimitSec=0)
    assert policyId != 0

    callbackQueue = queue.Queue()
    c_callback = create_field_policy_callback(callbackQueue)
    callbackUserData = {"policyId": policyId}
    dcgm_agent.dcgmPolicyRegister_v3(
        handle,
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        policyId,
        c_callback,
        callbackUserData)
    del c_callback
    del callbackUserData
    gc.collect()

    try:
        _inject_temp(handle, gpuId, 80)
        callbackViolation = callbackQueue.get(
            timeout=POLICY_CALLBACK_TIMEOUT_SECS)
        receivedUserData = callbackQueue.get(timeout=1)
    finally:
        dcgm_agent.dcgmPolicyUnregister_v3(
            handle,
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            policyId)

    assert callbackViolation.policyId == policyId
    assert callbackViolation.fieldId == dcgm_fields.DCGM_FI_DEV_GPU_TEMP
    assert callbackViolation.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU
    assert callbackViolation.entity.entityId == gpuId
    assert callbackViolation.currentValue == 80.0
    assert callbackViolation.timestamp > 0
    assert receivedUserData == {"policyId": policyId}


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_rate_limit_is_preserved(handle, gpuIds):
    """
    Verifies field-policy rate limit is preserved in policy metadata.
    """
    _clear_field_policies(handle)
    policyId = _create_temp_policy(handle, [gpuIds[0]], rateLimitSec=7)
    assert policyId != 0

    policy = dcgm_agent.dcgmPolicyGetOne(
        handle, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)
    assert policy.rateLimitSec == 7

    with test_utils.assert_raises(ValueError):
        _create_temp_policy(handle, [gpuIds[0]], rateLimitSec=-1)
    with test_utils.assert_raises(ValueError):
        _create_temp_policy(handle, [gpuIds[0]], rateLimitSec=0x100000000)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_rate_limit_controls_violation_delivery(handle, gpuIds):
    """
    Verifies a five-second rate limit suppresses one-second violations.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    try:
        rateLimitSec = 5
        rateLimitedPolicyId = _create_temp_policy(
            handle, [gpuId], rateLimitSec=rateLimitSec)
        for value in range(80, 87):
            _inject_temp(handle, gpuId, value)
            time.sleep(1)

        rateLimitedViolations = _wait_for_temp_violations(
            handle, rateLimitedPolicyId, gpuId, 2)
        assert len(rateLimitedViolations) == 2, \
            "Five-second rate limit did not suppress one-second violations"
        timestamps = sorted(
            violation.timestamp for violation in rateLimitedViolations)
        assert timestamps[1] - timestamps[0] >= rateLimitSec * 1000000, \
            "Rate-limited violations were delivered less than five seconds apart"
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_zero_rate_limit_records_every_violation(handle, gpuIds):
    """
    Verifies a zero rate limit records every violating sample.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    try:
        policyId = _create_temp_policy(handle, [gpuId], rateLimitSec=0)
        _inject_temp(handle, gpuId, 80)
        _wait_for_temp_violations(handle, policyId, gpuId, 1)
        _inject_temp(handle, gpuId, 81)
        violations = _wait_for_temp_violations(handle, policyId, gpuId, 2)
        assert len(
            violations) == 2, "Zero-rate-limit policy did not record both violations"
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_enable_disable_and_delete(handle, gpuIds):
    """
    Verifies the Python bindings enable, disable, and delete a field policy.
    """
    _clear_field_policies(handle)

    try:
        policyId = _create_temp_policy(handle, [gpuIds[0]], enabled=0)
        fieldId = dcgm_fields.DCGM_FI_DEV_GPU_TEMP

        policy = dcgm_agent.dcgmPolicyGetOne(handle, fieldId, policyId)
        assert policy.state == dcgm_structs.DCGM_POLICY_STATE_DISABLED

        dcgm_agent.dcgmPolicyEnable(handle, fieldId, policyId)
        policy = dcgm_agent.dcgmPolicyGetOne(handle, fieldId, policyId)
        assert policy.state == dcgm_structs.DCGM_POLICY_STATE_ENABLED

        dcgm_agent.dcgmPolicyDisable(handle, fieldId, policyId)
        policy = dcgm_agent.dcgmPolicyGetOne(handle, fieldId, policyId)
        assert policy.state == dcgm_structs.DCGM_POLICY_STATE_DISABLED

        dcgm_agent.dcgmPolicyDelete(handle, fieldId, policyId)
        policyIds = {
            policy.policyId for policy in dcgm_agent.dcgmPolicyGetAll(handle)}
        assert policyId not in policyIds
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_modify_and_get_all(handle, gpuIds):
    """
    Verifies the Python bindings modify a policy and return all created policies.
    """
    _clear_field_policies(handle)

    try:
        tempPolicyId = _create_temp_policy(handle, [gpuIds[0]])
        powerPolicyId = dcgm_agent.dcgmPolicyCreate(
            handle,
            dcgm_fields.DCGM_FI_DEV_BOARD_POWER_WATTS,
            "python_power_policy",
            threshold=300.0,
            policyOperator=dcgm_structs.DCGM_POLICY_OP_GT,
            entities=[gpuIds[0]],
            channelMask=dcgm_structs.DCGM_POLICY_CHANNEL_CONSOLE,
            rateLimitSec=1)

        policy = dcgm_agent.dcgmPolicyGetOne(
            handle, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, tempPolicyId)
        policy.threshold = 75.0
        policy.policyOperator = dcgm_structs.DCGM_POLICY_OP_GE
        policy.channelMask = (
            dcgm_structs.DCGM_POLICY_CHANNEL_CONSOLE
            | dcgm_structs.DCGM_POLICY_CHANNEL_CALLBACK)
        policy.rateLimitSec = 7
        dcgm_agent.dcgmPolicyModify(handle, policy)

        modifiedPolicy = dcgm_agent.dcgmPolicyGetOne(
            handle, dcgm_fields.DCGM_FI_DEV_GPU_TEMP, tempPolicyId)
        assert modifiedPolicy.threshold == 75.0
        assert modifiedPolicy.policyOperator == dcgm_structs.DCGM_POLICY_OP_GE
        assert modifiedPolicy.channelMask == (
            dcgm_structs.DCGM_POLICY_CHANNEL_CONSOLE
            | dcgm_structs.DCGM_POLICY_CHANNEL_CALLBACK)
        assert modifiedPolicy.rateLimitSec == 7

        policies = dcgm_agent.dcgmPolicyGetAll(handle)
        policyIds = {policy.policyId for policy in policies}
        assert tempPolicyId in policyIds
        assert powerPolicyId in policyIds
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_policy_import(handle, gpuIds):
    """
    Verifies the Python binding imports a field policy from YAML.
    """
    _clear_field_policies(handle)
    importDir = "/var/log/nvidia-dcgm"
    if not os.path.isdir(importDir) or not os.access(importDir, os.W_OK):
        test_utils.skip_test(
            f"This test requires a writable {importDir}")

    importPath = os.path.join(
        importDir, "python_field_policy_import_test.yaml")
    yamlContents = """schema_version: 1
policies:
  - fieldId: %d
    name: "python_imported_policy"
    operator: "GREATER_THAN"
    threshold: 70.0
    entities:
      - { entityGroupId: %d, entityId: %d }
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: 2
""" % (dcgm_fields.DCGM_FI_DEV_GPU_TEMP, dcgm_fields.DCGM_FE_GPU, gpuIds[0])

    try:
        with open(importPath, "w") as yamlFile:
            yamlFile.write(yamlContents)

        dcgm_agent.dcgmPolicyImport(handle, importPath)
        policies = dcgm_agent.dcgmPolicyGetAll(handle)
        importedPolicies = [
            policy for policy in policies
            if policy.name.rstrip("\0") == "python_imported_policy"
        ]
        assert len(importedPolicies) == 1
        assert importedPolicies[0].fieldId == dcgm_fields.DCGM_FI_DEV_GPU_TEMP
        assert importedPolicies[0].entityCount == 1
        assert importedPolicies[0].entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU
        assert importedPolicies[0].entities[0].entityId == gpuIds[0]
        assert importedPolicies[0].rateLimitSec == 2
    finally:
        _clear_field_policies(handle)
        if os.path.exists(importPath):
            os.remove(importPath)


# dcgmi
# ---------------------------------------------------------------------------

# NO HARDWARE


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_create_list_get_delete(handle, gpuIds):
    """
    Verifies dcgmi can create, list, get, and delete field-based policies.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    createOutput = _create_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        "dcgmi_temp_policy",
        70,
        gpuId)
    policyId = createOutput["policyId"]
    assert policyId != 0, "dcgmi create returned invalid policy ID"

    listOutput = _run_dcgmi_policy_json(["--list"])
    matchingPolicies = [
        policy for policy in listOutput
        if policy["policyId"] == policyId
    ]
    assert len(
        matchingPolicies) == 1, "created policy not found in dcgmi list output"
    assert matchingPolicies[0]["fieldId"] == dcgm_fields.DCGM_FI_DEV_GPU_TEMP
    assert matchingPolicies[0]["fieldName"] == DCGMI_TEMP_FIELD_NAME
    assert matchingPolicies[0]["name"] == "dcgmi_temp_policy"
    assert matchingPolicies[0]["entities"] == [
        {"entityGroupId": dcgm_fields.DCGM_FE_GPU, "entityId": gpuId}]

    getOutput = _get_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)
    assert getOutput["policyId"] == policyId
    assert getOutput["fieldName"] == DCGMI_TEMP_FIELD_NAME
    assert getOutput["operator"] == "GREATER_THAN"
    assert getOutput["channels"] == ["CONSOLE"]

    fieldGetOutput = _run_dcgmi_policy_json([
        "--get",
        "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
    ])
    matchingPolicies = [
        policy for policy in fieldGetOutput
        if policy["policyId"] == policyId
    ]
    assert len(
        matchingPolicies) == 1, "field-policy get did not return the created policy"

    _run_dcgmi_policy([
        "--delete",
        "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
        "--policy-id", str(policyId),
    ])

    listOutput = _run_dcgmi_policy_json(["--list"])
    assert all(policy["policyId"] != policyId for policy in listOutput)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_concurrent_creates_different_fields(handle, gpuIds):
    """
    Verifies concurrent dcgmi creates are serialized and all different-field policies persist.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]
    fields = [
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        dcgm_fields.DCGM_FI_DEV_BOARD_POWER_WATTS,
        dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
    ]

    commands = [
        _dcgmi_policy_create_args(
            fieldId,
            "dcgmi_concurrent_%d" % fieldId,
            70 + index,
            gpuId)
        for index, fieldId in enumerate(fields)
    ]

    createOutputs = _run_concurrent_dcgmi_policy_commands(commands)
    createdPolicyIds = [output["policyId"] for output in createOutputs]
    assert all(
        policyId != 0 for policyId in createdPolicyIds), "dcgmi create returned invalid policy ID"

    listOutput = _run_dcgmi_policy_json(["--list"])
    listedFields = {
        policy["fieldId"] for policy in listOutput
        if policy["policyId"] in createdPolicyIds
    }
    assert listedFields == set(
        fields), "concurrent dcgmi creates did not persist all fields"


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_concurrent_creates_same_field_duplicate_rejection(handle, gpuIds):
    """
    Verifies concurrent same-field dcgmi creates serialize to one create and duplicate rejections.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]
    thresholds = [70.0, 71.0, 72.0, 73.0]

    commands = [
        _dcgmi_policy_create_args(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            "dcgmi_concurrent_same_field",
            threshold,
            gpuId)
        for threshold in thresholds
    ]

    results = _run_concurrent_dcgmi_policy_commands_allow_failures(commands)
    okStatus = c_ubyte(dcgm_structs.DCGM_ST_OK).value
    okResults = [result for result in results if result[0] == okStatus]
    duplicateResults = [
        result for result in results
        if "Duplicate Key passed to function" in "\n".join(result[1] + result[2])
    ]

    assert len(
        okResults) == 1, "same-field concurrent creates should have exactly one success"
    assert len(duplicateResults) == len(commands) - 1, \
        "same-field concurrent creates should reject duplicate creates"

    listOutput = _run_dcgmi_policy_json(["--list"])
    tempPolicies = [
        policy for policy in listOutput
        if policy["fieldId"] == dcgm_fields.DCGM_FI_DEV_GPU_TEMP
    ]
    assert len(
        tempPolicies) == 1, "same-field concurrent creates should leave exactly one policy"
    assert tempPolicies[0]["threshold"] in thresholds, "final threshold was not one of the submitted values"
    assert tempPolicies[0]["entities"] == [
        {"entityGroupId": dcgm_fields.DCGM_FE_GPU, "entityId": gpuId}]


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_watch_json(handle, gpuIds):
    """
    Verifies dcgmi can fetch field-policy violations in JSON after injected NVML values.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    createOutput = _create_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        "dcgmi_watch_policy",
        70,
        gpuId,
        rateLimitSec=0)
    policyId = createOutput["policyId"]

    _inject_temp(handle, gpuId, 80)
    directViolation = _find_temp_violation(handle, policyId, gpuId)
    assert directViolation is not None, "Policy violation never appeared before dcgmi watch"

    watchOutput = _run_dcgmi_policy_json([
        "--watch",
        "--once",
        "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
    ])
    assert all(
        violation["fieldId"] == dcgm_fields.DCGM_FI_DEV_GPU_TEMP
        for violation in watchOutput), "dcgmi watch returned violations for an unrequested field"
    matchingViolations = [
        violation for violation in watchOutput
        if violation["policyId"] == policyId
        and violation["fieldId"] == dcgm_fields.DCGM_FI_DEV_GPU_TEMP
        and violation["entityId"] == gpuId
    ]
    assert len(
        matchingViolations) >= 1, "dcgmi watch did not return injected policy violation"
    assert matchingViolations[0]["fieldName"] == DCGMI_TEMP_FIELD_NAME
    assert matchingViolations[0]["currentValue"] == 80.0
    assert matchingViolations[0]["threshold"] == 70.0


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_modify_enable_disable(handle, gpuIds):
    """
    Verifies dcgmi can modify, disable, and enable field-based policies.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    createOutput = _create_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        "dcgmi_modify_policy",
        70,
        gpuId)
    policyId = createOutput["policyId"]

    _run_dcgmi_policy([
        "--modify",
        "--policy-id", str(policyId),
        "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
        "--operator", ">=",
        "--threshold", "75",
        "--channels", "console,callback",
        "--rate-limit", "3",
    ])

    modifiedPolicy = _get_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)
    assert modifiedPolicy["name"] == "dcgmi_modify_policy"
    assert modifiedPolicy["threshold"] == 75.0
    assert modifiedPolicy["operator"] == "GREATER_THAN_OR_EQUAL"
    assert modifiedPolicy["channels"] == ["CONSOLE", "CALLBACK"]
    assert modifiedPolicy["rateLimitSec"] == 3

    _run_dcgmi_policy([
        "--disable",
        "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
        "--policy-id", str(policyId),
    ])
    assert _get_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)["state"] == "DISABLED"

    _run_dcgmi_policy([
        "--enable",
        "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
        "--policy-id", str(policyId),
    ])
    assert _get_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)["state"] == "ENABLED"


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_import_and_delete_all(handle, gpuIds):
    """
    Verifies dcgmi can import field-based policies and delete all policies.
    """
    _clear_field_policies(handle)
    importDir = "/var/log/nvidia-dcgm"
    if not os.path.isdir(importDir) or not os.access(importDir, os.W_OK):
        test_utils.skip_test(
            "This test requires a writable /var/log/nvidia-dcgm")

    importPath = os.path.join(importDir, "dcgmi_field_policy_import_test.yaml")
    yamlContents = """schema_version: 1
policies:
  - fieldId: %d
    name: "dcgmi_imported_policy"
    operator: "GREATER_THAN"
    threshold: 70.0
    entities:
      - { entityGroupId: %d, entityId: %d }
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: 2
""" % (dcgm_fields.DCGM_FI_DEV_GPU_TEMP, dcgm_fields.DCGM_FE_GPU, gpuIds[0])

    try:
        with open(importPath, "w") as yamlFile:
            yamlFile.write(yamlContents)

        _run_dcgmi_policy(["--import", importPath])
        listOutput = _run_dcgmi_policy_json(["--list"])
        matchingPolicies = [
            policy for policy in listOutput
            if policy["name"] == "dcgmi_imported_policy"
        ]
        assert len(
            matchingPolicies) == 1, "imported policy not found in dcgmi list output"
        assert matchingPolicies[0]["fieldId"] == dcgm_fields.DCGM_FI_DEV_GPU_TEMP
        assert matchingPolicies[0]["entities"] == [
            {"entityGroupId": dcgm_fields.DCGM_FE_GPU, "entityId": gpuIds[0]}
        ]
        assert matchingPolicies[0]["rateLimitSec"] == 2

        _run_dcgmi_policy(["--delete", "--all"])
        assert _run_dcgmi_policy_json(["--list"]) == []
    finally:
        if os.path.exists(importPath):
            os.remove(importPath)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_validation_failures(handle, gpuIds):
    """
    Verifies dcgmi rejects invalid field-policy arguments.
    """
    _clear_field_policies(handle)
    fieldId = str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP)

    emptyGetOutput = _run_dcgmi_policy(["--get", "--field", fieldId])
    assert "No field policies configured for field ID" in "\n".join(
        emptyGetOutput)

    _run_dcgmi_policy_expect_failure([
        "--get",
        "--field", fieldId,
        "--policy-id", "-1",
    ], "Expected unsigned integer policy ID")
    _run_dcgmi_policy_expect_failure([
        "--create",
        "--field", fieldId,
        "--name", "bad_operator",
        "--operator", "bogus",
        "--threshold", "70",
    ], "Expected one of >, >=, <, <=, ==, !=, changed")
    _run_dcgmi_policy_expect_failure([
        "--create",
        "--field", fieldId,
        "--name", "bad_channel",
        "--operator", ">",
        "--threshold", "70",
        "--channels", "bogus",
    ], "Expected channels from console,file,callback")
    _run_dcgmi_policy_expect_failure([
        "--create",
        "--field", fieldId,
        "--operator", ">",
        "--threshold", "70",
    ], "--create requires --field, --name, and --operator")
    _run_dcgmi_policy_expect_failure([
        "--create",
        "--field", "0",
        "--name", "invalid_field",
        "--operator", ">",
        "--threshold", "70",
    ], "Cannot create field policy")


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_all_gpus_watch_json(handle, gpuIds):
    """
    Verifies dcgmi --entities all creates an all-GPU policy that watches active GPUs.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    createOutput = _create_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        "dcgmi_all_gpus_policy",
        70,
        "all",
        rateLimitSec=0)
    policyId = createOutput["policyId"]

    policy = _get_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)
    assert policy["entities"] == [
    ], "all-GPU policies should store an empty entity list"

    _inject_temp(handle, gpuId, 80)
    directViolation = _find_temp_violation(handle, policyId, gpuId)
    assert directViolation is not None, "All-GPU policy violation never appeared"

    watchOutput = _run_dcgmi_policy_json(["--watch", "--once"])
    matchingViolations = [
        violation for violation in watchOutput
        if violation["policyId"] == policyId
        and violation["entityId"] == gpuId
    ]
    assert len(
        matchingViolations) >= 1, "dcgmi watch did not return all-GPU policy violation"


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_watch_since_and_max_violations(handle, gpuIds):
    """
    Verifies dcgmi watch honors --since and --max-violations with --once.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    createOutput = _create_dcgmi_policy(
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        "dcgmi_since_policy",
        70,
        gpuId,
        rateLimitSec=0)
    policyId = createOutput["policyId"]

    _inject_temp(handle, gpuId, 80)
    firstViolations = _wait_for_temp_violations(handle, policyId, gpuId, 1)
    assert len(firstViolations) >= 1, "First policy violation never appeared"

    _inject_temp(handle, gpuId, 85)
    violations = _wait_for_temp_violations(handle, policyId, gpuId, 2)
    assert len(violations) >= 2, "Second policy violation never appeared"

    watchOutput = _run_dcgmi_policy_json([
        "--watch",
        "--once",
        "--since", str(firstViolations[0].timestamp + 1),
        "--max-violations", "1",
        "--interval", "1",
    ])
    matchingViolations = [
        violation for violation in watchOutput
        if violation["policyId"] == policyId
        and violation["entityId"] == gpuId
    ]
    assert len(matchingViolations) == 1, "dcgmi watch did not honor max-violations"
    assert matchingViolations[0]["currentValue"] == 85.0


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_import_failure(handle, gpuIds):
    """
    Verifies dcgmi reports import failures for invalid policy file paths.
    """
    _clear_field_policies(handle)
    missingPath = os.path.join(
        tempfile.gettempdir(),
        "dcgmi_field_policy_missing_import_test_%s.yaml" % uuid.uuid4().hex,
    )
    _run_dcgmi_policy_expect_failure([
        "--import",
        missingPath,
    ], "Cannot import field policies")


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_text_list_get_and_watch_once(handle, gpuIds):
    """
    Verifies list, get, and one-shot watch render field-policy information as text.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    try:
        createOutput = _create_dcgmi_policy(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            "dcgmi_text_policy",
            70,
            gpuId,
            rateLimitSec=0)
        policyId = createOutput["policyId"]

        listOutput = "\n".join(_run_dcgmi_policy(["--list"]))
        assert "Policy Monitoring Status" in listOutput
        assert "dcgmi_text_policy" in listOutput

        getOutput = "\n".join(_run_dcgmi_policy([
            "--get",
            "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
            "--policy-id", str(policyId),
        ]))
        assert "Policy Details: dcgmi_text_policy" in getOutput
        assert "Field Name:     %s" % DCGMI_TEMP_FIELD_NAME in getOutput
        assert "Policy ID:      %d" % policyId in getOutput
        assert "Status:         ENABLED" in getOutput

        _inject_temp(handle, gpuId, 80)
        assert _find_temp_violation(handle, policyId, gpuId) is not None

        watchOutput = "\n".join(_run_dcgmi_policy([
            "--watch",
            "--once",
            "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
        ]))
        assert "dcgmi_text_policy" in watchOutput
        assert str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP) in watchOutput
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_create_disabled_then_enable(handle, gpuIds):
    """
    Verifies a policy created disabled ignores violations until dcgmi enables it.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    try:
        createOutput = _run_dcgmi_policy_json([
            "--create",
            "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
            "--name", "dcgmi_disabled_policy",
            "--operator", ">",
            "--threshold", "70",
            "--entities", str(gpuId),
            "--channels", "console",
            "--rate-limit", "0",
            "--disabled",
        ])
        policyId = createOutput["policyId"]
        assert _get_dcgmi_policy(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)["state"] == "DISABLED"

        _inject_temp(handle, gpuId, 80)
        assert _find_temp_violation(
            handle, policyId, gpuId, timeoutSec=2) is None

        _run_dcgmi_policy([
            "--enable",
            "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
            "--policy-id", str(policyId),
        ])
        assert _get_dcgmi_policy(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)["state"] == "ENABLED"

        _inject_temp(handle, gpuId, 81)
        assert _find_temp_violation(handle, policyId, gpuId) is not None
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_create_changed_without_threshold(handle, gpuIds):
    """
    Verifies the changed operator can create a policy without a threshold.
    """
    _clear_field_policies(handle)

    try:
        createOutput = _run_dcgmi_policy_json([
            "--create",
            "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
            "--name", "dcgmi_changed_policy",
            "--operator", "changed",
            "--entities", str(gpuIds[0]),
            "--channels", "console",
        ])
        policyId = createOutput["policyId"]
        policy = _get_dcgmi_policy(dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)
        assert policy["operator"] == "CHANGED"
        assert policy["name"] == "dcgmi_changed_policy"
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_explicit_gpu_entity_syntax(handle, gpuIds):
    """
    Verifies dcgmi accepts the entity parser's explicit gpu:<id> syntax.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    try:
        createOutput = _create_dcgmi_policy(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            "dcgmi_explicit_gpu_policy",
            70,
            "gpu:%d" % gpuId)
        policy = _get_dcgmi_policy(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP, createOutput["policyId"])
        assert policy["entities"] == [
            {"entityGroupId": dcgm_fields.DCGM_FE_GPU, "entityId": gpuId}]
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_modify_name_and_entities(handle, gpuIds):
    """
    Verifies dcgmi modifies a policy name and entity list.
    """
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 injected GPUs")

    _clear_field_policies(handle)
    try:
        createOutput = _create_dcgmi_policy(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            "dcgmi_original_policy",
            70,
            "gpu:%d" % gpuIds[0])
        policyId = createOutput["policyId"]

        _run_dcgmi_policy([
            "--modify",
            "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
            "--policy-id", str(policyId),
            "--name", "dcgmi_renamed_policy",
            "--entities", "gpu:%d" % gpuIds[1],
        ])
        policy = _get_dcgmi_policy(dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)
        assert policy["name"] == "dcgmi_renamed_policy"
        assert policy["entities"] == [
            {"entityGroupId": dcgm_fields.DCGM_FE_GPU, "entityId": gpuIds[1]}]
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_remaining_operator_violations(handle, gpuIds):
    """
    Verifies remaining comparison operators and changed produce injected violations.
    """
    # (operator, threshold, baseline, values, expectedValues, expectedOperator)
    # baseline overrides the SKU default (31 C) for operators that would treat
    # 31 as a violation. changed seeds baseline 31 and waits for async policy
    # eval to settle before opening the violation window.
    cases = [
        ("<", 70, 70, [69], [69], dcgm_structs.DCGM_POLICY_OP_LT),
        ("<=", 70, 71, [70], [70], dcgm_structs.DCGM_POLICY_OP_LE),
        ("==", 70, 71, [70], [70], dcgm_structs.DCGM_POLICY_OP_EQ),
        ("!=", 70, 70, [71], [71], dcgm_structs.DCGM_POLICY_OP_NE),
        ("changed", None, 31, [70, 71], [70, 71],
         dcgm_structs.DCGM_POLICY_OP_CHANGED),
    ]
    gpuId = gpuIds[0]

    try:
        for (operator, threshold, baseline, values,
             expectedValues, expectedOperator) in cases:
            _clear_field_policies(handle)
            createArgs = [
                "--create",
                "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
                "--name", "dcgmi_operator_%s" % expectedOperator,
                "--operator", operator,
                "--entities", str(gpuId),
                "--channels", "console",
                "--rate-limit", "0",
            ]
            if threshold is not None:
                createArgs.extend(["--threshold", str(threshold)])

            policyId = _run_dcgmi_policy_json(createArgs)["policyId"]
            if baseline is not None:
                _inject_temp(handle, gpuId, baseline)
                if expectedOperator == dcgm_structs.DCGM_POLICY_OP_CHANGED:
                    assert _violation_count_remains(
                        handle,
                        policyId,
                        gpuId,
                        0,
                        POLICY_NO_VIOLATION_TIMEOUT_SECS), \
                        "changed baseline produced unexpected violations before test window"

            sinceTimestamp = int(time.time() * 1000000)
            for value in values:
                _inject_temp(handle, gpuId, value)

            violations = _wait_for_temp_violations(
                handle,
                policyId,
                gpuId,
                len(expectedValues),
                sinceTimestamp)
            actualViolations = {
                (violation.currentValue, violation.policyOperator)
                for violation in violations
            }
            expectedViolations = {
                (float(value), expectedOperator)
                for value in expectedValues
            }
            assert actualViolations == expectedViolations, \
                "%s produced unexpected violations: %s" % (
                    operator, actualViolations)
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_multiple_entity_violations(handle, gpuIds):
    """
    Verifies a dcgmi policy evaluates injected violations on every target GPU.
    """
    if len(gpuIds) < 2:
        test_utils.skip_test("This test requires at least 2 injected GPUs")

    _clear_field_policies(handle)
    try:
        entities = "gpu:%d,gpu:%d" % (gpuIds[0], gpuIds[1])
        policyId = _create_dcgmi_policy(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            "dcgmi_multi_entity_policy",
            70,
            entities,
            rateLimitSec=0)["policyId"]
        policy = _get_dcgmi_policy(dcgm_fields.DCGM_FI_DEV_GPU_TEMP, policyId)
        assert policy["entities"] == [
            {"entityGroupId": dcgm_fields.DCGM_FE_GPU, "entityId": gpuIds[0]},
            {"entityGroupId": dcgm_fields.DCGM_FE_GPU, "entityId": gpuIds[1]},
        ]

        for gpuId, value in zip(gpuIds[:2], [80, 81]):
            _inject_temp(handle, gpuId, value)

        for gpuId, value in zip(gpuIds[:2], [80.0, 81.0]):
            violation = _find_temp_violation(handle, policyId, gpuId)
            assert violation is not None, "No violation for targeted GPU %d" % gpuId
            assert violation.currentValue == value, "Unexpected violation value for GPU %d" % gpuId
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_disable_then_enable_controls_violations(handle, gpuIds):
    """
    Verifies disabling an enabled policy suppresses violations until re-enabled.
    """
    _clear_field_policies(handle)
    gpuId = gpuIds[0]

    try:
        policyId = _create_dcgmi_policy(
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            "dcgmi_toggle_policy",
            70,
            gpuId,
            rateLimitSec=0)["policyId"]
        _inject_temp(handle, gpuId, 80)
        initialViolations = _wait_for_temp_violations(
            handle, policyId, gpuId, 1)
        assert len(
            initialViolations) == 1, "Enabled policy did not record its initial violation"

        _run_dcgmi_policy([
            "--disable",
            "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
            "--policy-id", str(policyId),
        ])
        _inject_temp(handle, gpuId, 81)
        assert _violation_count_remains(
            handle,
            policyId,
            gpuId,
            len(initialViolations),
            POLICY_NO_VIOLATION_TIMEOUT_SECS), "Disabled policy recorded a violation"

        _run_dcgmi_policy([
            "--enable",
            "--field", str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP),
            "--policy-id", str(policyId),
        ])
        _inject_temp(handle, gpuId, 82)
        resumedViolations = _wait_for_temp_violations(
            handle, policyId, gpuId, len(initialViolations) + 1)
        assert len(resumedViolations) == len(initialViolations) + 1, \
            "Re-enabled policy did not resume violations"
    finally:
        _clear_field_policies(handle)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_malformed_import_failure(handle, gpuIds):
    """
    Verifies dcgmi rejects malformed YAML from the policy import directory.
    """
    _clear_field_policies(handle)
    temporaryFile = tempfile.NamedTemporaryFile(
        prefix="dcgmi_field_policy_malformed_import_", suffix=".yaml", delete=False)
    importPath = temporaryFile.name
    temporaryFile.close()
    try:
        with open(importPath, "w") as yamlFile:
            yamlFile.write("schema_version: 1\npolicies: [\n")

        _run_dcgmi_policy_expect_failure([
            "--import",
            importPath,
        ], "Cannot import field policies")
    finally:
        if os.path.exists(importPath):
            os.remove(importPath)


# NO HARDWARE

@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_field_policy_invalid_policy_id_failures(handle, gpuIds):
    """
    Verifies dcgmi reports runtime failures for unknown policy IDs.
    """
    _clear_field_policies(handle)
    fieldId = str(dcgm_fields.DCGM_FI_DEV_GPU_TEMP)
    invalidPolicyId = "999999"
    cases = [
        ("--get", "Cannot get field policy"),
        ("--delete", "Cannot delete field policy"),
        ("--enable", "Cannot enable field policy"),
        ("--disable", "Cannot disable field policy"),
    ]

    for action, expectedText in cases:
        _run_dcgmi_policy_expect_failure([
            action,
            "--field", fieldId,
            "--policy-id", invalidPolicyId,
        ], expectedText)
