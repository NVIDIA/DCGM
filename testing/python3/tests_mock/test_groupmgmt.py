# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from tests.test_groupmgmt import helper_dcgm_group_create_grp
from tests.test_groupmgmt import helper_dcgm_group_update_grp
from tests.test_groupmgmt import helper_dcgm_group_get_grp_info
from tests.test_groupmgmt import helper_dcgm_group_get_all_ids_standalone
from tests.test_groupmgmt import dcgm_group_test_default_group
from tests.test_groupmgmt import helper_dcgm_group_delete_grp

import test_utils
from test_utils import trampoline  # For introspection to work


@test_utils.run_with_injection_nvml()
@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_group_create_grp_standalone(handle):
    helper_dcgm_group_create_grp(handle)


@test_utils.run_with_injection_nvml()
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_group_update_grp_standalone(handle, gpuIds):
    helper_dcgm_group_update_grp(handle, gpuIds)


@test_utils.run_with_injection_nvml()
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_group_get_grp_info_standalone(handle, gpuIds):
    helper_dcgm_group_get_grp_info(handle, gpuIds)


@test_utils.run_with_injection_nvml()
@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_group_get_all_ids_standalone(handle):
    helper_dcgm_group_get_all_ids_standalone(handle)


@test_utils.run_with_injection_nvml()
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_group_test_default_group_standalone(handle, gpuIds):
    dcgm_group_test_default_group(handle, gpuIds)


@test_utils.run_with_injection_nvml()
@test_utils.run_with_standalone_host_engine(20)
def test_dcgm_group_delete_grp_standalone(handle):
    helper_dcgm_group_delete_grp(handle)
