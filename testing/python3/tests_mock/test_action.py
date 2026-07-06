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

from tests.test_action import helper_dcgm_action_validate_embedded
from tests.test_action import helper_dcgm_action_validate_remote
from tests.test_action import helper_dcgm_action_run_diag_embedded
from tests.test_action import helper_dcgm_action_run_diag_remote
from tests.test_action import helper_dcgm_action_run_diag_gpu_list
from tests.test_action import helper_dcgm_action_run_diag_bad_validation

from _test_helpers import skip_test_if_no_dcgm_nvml

import test_utils
# Explicit so trampoline introspection works.
from test_utils import trampoline


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_with_max_power_limit_set()
def test_dcgm_action_validate_embedded(handle, gpuIds):
    helper_dcgm_action_validate_embedded(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_action_validate_remote(handle, gpuIds):
    helper_dcgm_action_validate_remote(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_with_max_power_limit_set()
def test_dcgm_action_run_diag_embedded(handle, gpuIds):
    helper_dcgm_action_run_diag_embedded(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_with_max_power_limit_set()
def test_dcgm_action_run_diag_remote(handle, gpuIds):
    helper_dcgm_action_run_diag_remote(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('T400.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_action_run_diag_gpu_list_standalone(handle, gpuIds):
    helper_dcgm_action_run_diag_gpu_list(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_action_run_diag_bad_validation(handle, gpuIds):
    helper_dcgm_action_run_diag_bad_validation(handle, gpuIds)


def test_dcgm_action_run_diag_on_heterogeneous_env():
    trampoline()


def test_dcgm_action_run_diag_entity_and_group_specified():
    trampoline()
