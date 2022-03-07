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
import dcgm_structs
import dcgm_structs_internal
import test_utils
import dcgm_errors

def helper_test_dcgm_error_get_priority(handle, gpuIds):
    prio = dcgm_errors.dcgmErrorGetPriorityByCode(dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)
    assert prio == dcgm_errors.DCGM_ERROR_ISOLATE, "DBE errors should be an isolate priority, but found %d" % prio
    prio = dcgm_errors.dcgmErrorGetPriorityByCode(dcgm_errors.DCGM_FR_NVML_API)
    assert prio == dcgm_errors.DCGM_ERROR_MONITOR, "DBE errors should be a monitor priority, but found %d" % prio

    prio = dcgm_errors.dcgmErrorGetPriorityByCode(-1)
    assert prio == dcgm_errors.DCGM_ERROR_UNKNOWN, "Non-existent error should be unknown priority, but found %d" % prio
    prio = dcgm_errors.dcgmErrorGetPriorityByCode(dcgm_errors.DCGM_FR_ERROR_SENTINEL)
    assert prio == dcgm_errors.DCGM_ERROR_UNKNOWN, "The sentinel error error should be unknown priority, but found %d" % prio

def helper_test_dcgm_error_get_msg(handle, gpuIds):
    msg = dcgm_errors.dcgmErrorGetFormatMsgByCode(dcgm_errors.DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD)
    assert msg == dcgm_errors.DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD_MSG, \
           "Expected '%s' as msg, but found '%s'" % (dcgm_errors.DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD_MSG, msg)

    msg = dcgm_errors.dcgmErrorGetFormatMsgByCode(dcgm_errors.DCGM_FR_DEVICE_COUNT_MISMATCH)
    assert msg == dcgm_errors.DCGM_FR_DEVICE_COUNT_MISMATCH_MSG, \
           "Expected '%s' as msg, but found '%s'" % (dcgm_errors.DCGM_FR_DEVICE_COUNT_MISMATCH_MSG, msg)
    
    msg = dcgm_errors.dcgmErrorGetFormatMsgByCode(dcgm_errors.DCGM_FR_ERROR_SENTINEL)
    assert not msg, "The sentinel error error should be empty, but found %s" % msg
    
    msg = dcgm_errors.dcgmErrorGetFormatMsgByCode(-1)
    assert not msg, "Non-existent error should be empty, but found %s" % msg

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_error_get_priority_embedded(handle, gpuIds):
    helper_test_dcgm_error_get_priority(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_error_get_priority_standalone(handle, gpuIds):
    helper_test_dcgm_error_get_priority(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_error_get_msg_embedded(handle, gpuIds):
    helper_test_dcgm_error_get_msg(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_error_get_msg_standalone(handle, gpuIds):
    helper_test_dcgm_error_get_msg(handle, gpuIds)
