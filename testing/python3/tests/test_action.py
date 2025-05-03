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
import dcgm_structs
import dcgm_agent
from dcgm_structs import dcgmExceptionClass
import test_utils
import logger
import os
import option_parser
import DcgmDiag
from _test_helpers import skip_test_if_no_dcgm_nvml

g_latestDiagResponseVersion = dcgm_structs.dcgmDiagResponse_version12

g_allValidations = [dcgm_structs.DCGM_POLICY_VALID_NONE, dcgm_structs.DCGM_POLICY_VALID_SV_SHORT,
                    dcgm_structs.DCGM_POLICY_VALID_SV_MED, dcgm_structs.DCGM_POLICY_VALID_SV_LONG,
                    dcgm_structs.DCGM_POLICY_VALID_SV_XLONG]

def helper_validate_action(groupObj):
    
    if not option_parser.options.developer_mode:
        validations = g_allValidations[0:0] #Just run short for non-developer
    else:
        validations = g_allValidations

    for validation in validations:
        if validation == dcgm_structs.DCGM_POLICY_VALID_NONE:
            #This returns success unconditionally. Not worth checking
            continue

        response = groupObj.action.Validate(validation)

        #Validate the contents
        assert response.version == g_latestDiagResponseVersion, "Version mismatch. Expected %d. got %d" % \
                                                                           (g_latestDiagResponseVersion, response.version)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_with_max_power_limit_set()
def test_dcgm_action_validate_embedded(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds('actiongroup', gpuIds)

    helper_validate_action(groupObj)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_action_validate_remote(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds('actiongroup', gpuIds)

    helper_validate_action(groupObj)

g_allDiagLevels = [dcgm_structs.DCGM_DIAG_LVL_SHORT,
                   dcgm_structs.DCGM_DIAG_LVL_MED,
                   dcgm_structs.DCGM_DIAG_LVL_LONG,
                   dcgm_structs.DCGM_DIAG_LVL_XLONG]

def helper_validate_run_diag(groupObj):
    if not option_parser.options.developer_mode:
        diagLevels = g_allDiagLevels[0:0] #Just run short for non-developer
    else:
        diagLevels = g_allDiagLevels

    for diagLevel in diagLevels:
        logger.info("Running diag level %d. This may take minutes." % diagLevel)
        response = groupObj.action.RunDiagnostic(diagLevel)

        #Validate the contents
        assert response.version == g_latestDiagResponseVersion, "Version mismatch. Expected %d. got %d" % \
                                                                           (g_latestDiagResponseVersion, response.version)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_with_max_power_limit_set()
def test_dcgm_action_run_diag_embedded(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds('actiongroup', gpuIds)

    helper_validate_run_diag(groupObj)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_with_max_power_limit_set()
def test_dcgm_action_run_diag_remote(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds('actiongroup', gpuIds)

    helper_validate_run_diag(groupObj)

def helper_dcgm_action_run_diag_gpu_list(handle, gpuIds):
    '''
    Test that running the DCGM diagnostic works if you provide a GPU ID list rather
    than a groupId.
    '''
    gpuIdStr = ""
    for i, gpuId in enumerate(gpuIds):
        if i > 0:
            gpuIdStr += ","
        gpuIdStr += str(gpuId)

    drd = dcgm_structs.c_dcgmRunDiag_v10()
    drd.version = dcgm_structs.dcgmRunDiag_version10
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    drd.groupId = dcgm_structs.DCGM_GROUP_NULL #Initializing to DCGM_GROUP_NULL in case the constructor above doesn't and entityIds is specified.
    drd.entityIds = gpuIdStr
    #this will throw an exception on error
    response = test_utils.action_validate_wrapper(drd, handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_run_diag_gpu_list_standalone(handle, gpuIds):
    helper_dcgm_action_run_diag_gpu_list(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_action_run_diag_bad_validation(handle, gpuIds):
    gpuIdStr = ""
    for i, gpuId in enumerate(gpuIds):
        if i > 0:
            gpuIdStr += ","
        gpuIdStr += str(gpuId)
    drd = dcgm_structs.c_dcgmRunDiag_v9()
    drd.version = dcgm_structs.dcgmRunDiag_version9
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_XLONG + 1 #use an invalid value
    drd.groupId = dcgm_structs.DCGM_GROUP_NULL #Initializing to DCGM_GROUP_NULL in case the constructor above doesn't and entityIds is specified.
    drd.entityIds = gpuIdStr
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        response = test_utils.action_validate_wrapper(drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version9)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100x4-and-DGX.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_action_run_diag_on_heterogeneous_env(handle, gpuIds):
    # GPUs are specified by entity-id
    drd = dcgm_structs.c_dcgmRunDiag_v9()
    drd.version = dcgm_structs.dcgmRunDiag_version9
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    drd.groupId = dcgm_structs.DCGM_GROUP_NULL #Initializing to DCGM_GROUP_NULL in case the constructor above doesn't and entityIds is specified.
    drd.entityIds = "*"
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GROUP_INCOMPATIBLE)):
        _ = test_utils.action_validate_wrapper(drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version9)

    # GPUs are specified by group id
    drd = dcgm_structs.c_dcgmRunDiag_v10()
    drd.version = dcgm_structs.dcgmRunDiag_version10
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    drd.groupId = dcgm_structs.DCGM_GROUP_ALL_GPUS
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GROUP_INCOMPATIBLE)):
        _ = test_utils.action_validate_wrapper(drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version9)

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100x4-and-DGX.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_action_run_diag_entity_and_group_specified(handle, gpuIds):
    drd = dcgm_structs.c_dcgmRunDiag_v10()
    drd.version = dcgm_structs.dcgmRunDiag_version10
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    drd.groupId = dcgm_structs.DCGM_GROUP_ALL_GPUS
    drd.entityIds = str(gpuIds[0])
    # When both entity id and group are specified, entity id will be ignored.
    # It should return DCGM_ST_GROUP_INCOMPATIBLE as it is on heterogeneous env with group DCGM_GROUP_ALL_GPUS.
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GROUP_INCOMPATIBLE)):
        _ = test_utils.action_validate_wrapper(drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version9)
