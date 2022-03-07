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

import dcgm_structs
import dcgm_structs_internal
import dcgm_agent_internal
import dcgmvalue
import pydcgm
import logger
import test_utils
import dcgm_fields
from dcgm_structs import dcgmExceptionClass

import time
from ctypes import *
import queue
import sys
import os

POLICY_CALLBACK_TIMEOUT_SECS = 15 #How long to wait for policy callbacks to occur. The loop runs every 10 seconds in the host engine, so this should be 10 seconds + some fuzz time

# creates a callback function for dcgmPolicyRegister calls which will push its args 
# into "queue" when it is called.  This allows synchronization on the callback by 
# retrieving the args from the queue as well as checks of the args via asserts.
def create_c_callback(queue=None):
    @CFUNCTYPE(None, c_void_p)
    def c_callback(data):
        if queue:
            # copy data into a python struct so that it is the right format and is not lost when "data" var is lost
            callbackData = dcgm_structs.c_dcgmPolicyCallbackResponse_v1()
            memmove(addressof(callbackData), data, callbackData.FieldsSizeof())
            queue.put(callbackData)
    return c_callback

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_reg_unreg_for_policy_update_standalone(handle, gpuIds):
    """
    Verifies that the reg/unreg path for the         policy manager is working
    """
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds('test', gpuIds)
    
    empty_c_callback = create_c_callback()  # must hold ref so func is not GC'ed before c api uses it
    
    # Register/Unregister will throw exceptions if they fail
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_DBE, None, empty_c_callback)
    group.policy.Unregister(dcgm_structs.DCGM_POLICY_COND_DBE)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_policy_negative_register_standalone(handle):
    """
    Verifies that the register function does not allow a bad groupId value
    """
    policy = pydcgm.DcgmGroupPolicy(pydcgm.DcgmHandle(handle), 9999, None)
    empty_c_callback = create_c_callback()  # must hold ref so func is not GC'ed before c api uses it
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        policy.Register(dcgm_structs.DCGM_POLICY_COND_DBE, empty_c_callback)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_policy_negative_unregister_standalone(handle):
    """
    Verifies that the unregister function does not allow a bad groupId value
    """
    policy = pydcgm.DcgmGroupPolicy(pydcgm.DcgmHandle(handle), 9999, None)
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        policy.Unregister(dcgm_structs.DCGM_POLICY_COND_DBE)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
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
    newPolicy.parms[0].tag = 0
    newPolicy.parms[0].val.boolean = True

    group.policy.Set(newPolicy)
    policies = group.policy.Get()
    
    _assert_policies_equal(policies[0], newPolicy)

def _assert_policies_equal(policy1, policy2):
    assert(policy1) # check if None
    assert(policy2)
    assert(policy1.version == policy2.version)
    assert(policy1.condition == policy2.condition)
    assert(policy1.parms[0].tag == policy2.parms[0].tag)
    assert(policy1.parms[0].val.boolean == policy2.parms[0].val.boolean)
    
def helper_dcgm_policy_inject_eccerror(handle, gpuIds):
    """ 
    Verifies that we can inject an error into the ECC counters and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_DBE
    newPolicy.parms[0].tag = 0
    newPolicy.parms[0].val.boolean = True

    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds("test1", gpuIds)
    group.policy.Set(newPolicy)

    # the order of the callbacks will change once implementation is complete
    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_DBE, c_callback, None)

    # inject an error into ECC
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_DEV
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()+60) * 1000000.0) # set the injected data into the future
    field.value.i64 = 1
    logger.debug("injecting %s for gpuId %d" % (str(field), gpuIds[0]))

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackData = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        assert False, "Callback never happened"

    # check that the callback occurred with the correct arguments
    assert(dcgm_structs.DCGM_POLICY_COND_DBE == callbackData.condition), \
            ("error callback was not for a DBE error, got: %s" % callbackData.condition)
    assert(1 == callbackData.val.dbe.numerrors), 'Expected 1 DBE error but got %s' % callbackData.val.dbe.numerrors
    assert(dcgm_structs.c_dcgmPolicyConditionDbe_t.LOCATIONS['DEVICE'] == callbackData.val.dbe.location), \
        'got: %s' % callbackData.val.dbe.location

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_eccerror_embedded(handle, gpuIds):
    helper_dcgm_policy_inject_eccerror(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_eccerror_standalone(handle, gpuIds):
    helper_dcgm_policy_inject_eccerror(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_nvlinkerror_standalone(handle, gpuIds):
    """ 
    Verifies that we can inject an error into the NVLINK error and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_NVLINK
    newPolicy.parms[5].tag = 0
    newPolicy.parms[5].val.boolean = True

    # find a GPU that supports nvlink (otherwise internal test will ignore it)
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds('test1', gpuIds)
    group.policy.Set(newPolicy)
 
    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_NVLINK, finishCallback=c_callback)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()+60) * 1000000.0) # set the injected data into the future
    field.value.i64 = 1
    
    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackData = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        assert False, "Callback never happened"

    # check that the callback occurred with the correct arguments
    assert(dcgm_structs.DCGM_POLICY_COND_NVLINK == callbackData.condition), \
            ("NVLINK error callback was not for a NVLINK error, got: %s" % callbackData.condition)
    assert(dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL == callbackData.val.nvlink.fieldId), \
            ("Expected 130 fieldId but got %s" % callbackData.val.nvlink.fieldId)
    assert(1 == callbackData.val.nvlink.counter), 'Expected 1 PCI error but got %s' % callbackData.val.nvlink.counter

def helper_test_dcgm_policy_inject_xiderror(handle, gpuIds):
    """ 
    Verifies that we can inject an XID error and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_XID
    newPolicy.parms[6].tag = 0
    newPolicy.parms[6].val.boolean = True

    dcgmHandle = pydcgm.DcgmHandle(handle)
    validDeviceId = -1
    devices = gpuIds
    for x in devices:
        fvSupported = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, x, [dcgm_fields.DCGM_FI_DEV_XID_ERRORS, ])
        if (fvSupported[0].value.i64 != dcgmvalue.DCGM_INT64_NOT_SUPPORTED):
            validDeviceId = x
            break
    if (validDeviceId == -1):
        test_utils.skip_test("Can only run if at least one GPU that supports XID errors is present")

    group = pydcgm.DcgmGroup(dcgmHandle, groupName="test1", groupType=dcgm_structs.DCGM_GROUP_EMPTY)
    group.AddGpu(validDeviceId)
    group.policy.Set(newPolicy)
 
    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_XID, finishCallback=c_callback)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_XID_ERRORS
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()+60) * 1000000.0) # set the injected data into the future
    field.value.i64 = 16
    
    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, validDeviceId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackData = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        assert False, "Callback never happened"

    # check that the callback occurred with the correct arguments
    assert(dcgm_structs.DCGM_POLICY_COND_XID == callbackData.condition), \
            ("XID error callback was not for a XID error, got: %s" % callbackData.condition)
    assert(16 == callbackData.val.xid.errnum), ('Expected XID error 16 but got %s' % callbackData.val.xid.errnum)

@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_xiderror_standalone(handle, gpuIds):
    helper_test_dcgm_policy_inject_xiderror(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_xiderror_embedded(handle, gpuIds):
    helper_test_dcgm_policy_inject_xiderror(handle, gpuIds)
    

def helper_dcgm_policy_inject_pcierror(handle, gpuIds):
    """ 
    Verifies that we can inject an error into the PCI counters and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()
    
    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_PCI
    newPolicy.parms[1].tag = 1
    newPolicy.parms[1].val.llval = 0

    gpuId = gpuIds[0]
    
    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(handle), groupName="test1", groupType=dcgm_structs.DCGM_GROUP_EMPTY)
    group.AddGpu(gpuId)
    group.policy.Set(newPolicy)
 
    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_PCI, finishCallback=c_callback)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()+60) * 1000000.0) # set the injected data into the future
    field.value.i64 = 1
    
    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackData = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        assert False, "Callback never happened"

    # check that the callback occurred with the correct arguments
    assert(dcgm_structs.DCGM_POLICY_COND_PCI == callbackData.condition), \
            ("PCI error callback was not for a PCI error, got: %s" % callbackData.condition)
    assert(1 == callbackData.val.pci.counter), 'Expected 1 PCI error but got %s' % callbackData.val.pci.counter

@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_pcierror_standalone(handle, gpuIds):
    helper_dcgm_policy_inject_pcierror(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_pcierror_embedded(handle, gpuIds):
    helper_dcgm_policy_inject_pcierror(handle, gpuIds)


@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_dcgm_policy_inject_retiredpages_standalone(handle, gpuIds):
    """ 
    Verifies that we can inject an error into the retired pages counters and receive a callback
    """
    newPolicy = dcgm_structs.c_dcgmPolicy_v1()

    newPolicy.version = dcgm_structs.dcgmPolicy_version1
    newPolicy.condition = dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED
    newPolicy.parms[2].tag = 1
    newPolicy.parms[2].val.llval = 5

    # find a GPU that supports ECC and retired pages (otherwise internal test will ignore it)
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    group = dcgmSystem.GetGroupWithGpuIds("test1", gpuIds)
    
    group.policy.Set(newPolicy)

    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED, finishCallback=c_callback)

    # inject an error into ECC
    numPages = 10
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_RETIRED_DBE
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()+60) * 1000000.0) # set the injected data into the future
    field.value.i64 = numPages

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    #inject a SBE too so that the health check code gets past its internal checks
    field.fieldId = dcgm_fields.DCGM_FI_DEV_RETIRED_SBE

    ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # wait for the the policy manager to call back
    try:
        callbackData = callbackQueue.get(timeout=POLICY_CALLBACK_TIMEOUT_SECS)
    except queue.Empty:
        assert False, "Callback never happened"

    # check that the callback occurred with the correct arguments
    assert(dcgm_structs.DCGM_POLICY_COND_MAX_PAGES_RETIRED == callbackData.condition), \
            ("error callback was not for a retired pages, got: %s" % callbackData.condition)
    assert(numPages == callbackData.val.mpr.dbepages), \
            'Expected %s errors but got %s' % (numPages, callbackData.val.mpr.dbepages)


@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_initialized_client()
def test_dcgm_policy_get_with_no_gpus_standalone(handle):
    '''
    Test that getting the policies when no GPUs are in the group raises an exception
    '''
    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(handle), groupType=dcgm_structs.DCGM_GROUP_EMPTY, groupName="test")
    
    with test_utils.assert_raises(pydcgm.DcgmException):
        policies = group.policy.Get()
        
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_policy_get_with_some_gpus_standalone(handle, gpuIds):
    '''
    Test that getting the policies returns the correct number of policies as GPUs in the system 
    when "count" is not specified for policy.Get
    '''
    group = pydcgm.DcgmGroup(pydcgm.DcgmHandle(handle), groupType=dcgm_structs.DCGM_GROUP_EMPTY, groupName="test")
    
    group.AddGpu(gpuIds[0])
    policies = group.policy.Get()
    assert len(policies) == 1, len(policies)

    
