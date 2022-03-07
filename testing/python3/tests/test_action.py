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
import dcgm_structs
import dcgm_agent
from dcgm_structs import dcgmExceptionClass
import test_utils
import logger
import os
import option_parser
import DcgmDiag

g_allValidations = [dcgm_structs.DCGM_POLICY_VALID_NONE, dcgm_structs.DCGM_POLICY_VALID_SV_SHORT,
                    dcgm_structs.DCGM_POLICY_VALID_SV_MED, dcgm_structs.DCGM_POLICY_VALID_SV_LONG]

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
        assert response.version == dcgm_structs.dcgmDiagResponse_version6, "Version mismatch. Expected %d. got %d" % \
                                                                           (dcgm_structs.dcgmDiagResponse_version6, response.version)

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
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_action_validate_remote(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds('actiongroup', gpuIds)

    helper_validate_action(groupObj)

g_allDiagLevels = [dcgm_structs.DCGM_DIAG_LVL_SHORT,
                   dcgm_structs.DCGM_DIAG_LVL_MED,
                   dcgm_structs.DCGM_DIAG_LVL_LONG]

def helper_validate_run_diag(groupObj):
    if not option_parser.options.developer_mode:
        diagLevels = g_allDiagLevels[0:0] #Just run short for non-developer
    else:
        diagLevels = g_allDiagLevels

    for diagLevel in diagLevels:
        logger.info("Running diag level %d. This may take minutes." % diagLevel)
        response = groupObj.action.RunDiagnostic(diagLevel)

        #Validate the contents
        assert response.version == dcgm_structs.dcgmDiagResponse_version6, "Version mismatch. Expected %d. got %d" % \
                                                                           (dcgm_structs.dcgmDiagResponse_version6, response.version)

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
@test_utils.run_with_initialized_client()
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

    drd = dcgm_structs.c_dcgmRunDiag_t()
    drd.version = dcgm_structs.dcgmRunDiag_version
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_SHORT
    drd.groupId = 0 #Initializing to 0 in case the constructor above doesn't
    drd.gpuList = gpuIdStr
    #this will throw an exception on error
    response = test_utils.action_validate_wrapper(drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_run_diag_gpu_list_embedded(handle, gpuIds):
    helper_dcgm_action_run_diag_gpu_list(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
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

    drd = dcgm_structs.c_dcgmRunDiag_t()
    drd.version = dcgm_structs.dcgmRunDiag_version
    drd.validate = dcgm_structs.DCGM_POLICY_VALID_SV_LONG + 1 #use an invalid value
    drd.groupId = 0 #Initializing to 0 in case the constructor above doesn't
    drd.gpuList = gpuIdStr

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        response = test_utils.action_validate_wrapper(drd, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version)
