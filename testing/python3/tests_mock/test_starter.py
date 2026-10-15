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
# Sample script to test python bindings for DCGM

from tests.test_starter import helper_dcgm_agent_get_values_for_fields
from tests.test_starter import helper_dcgm_engine_watch_field_values
from tests.test_starter import helper_dcgm_engine_unwatch_field_value
from tests.test_starter import helper_unwatch_field_values_public
from tests.test_starter import helper_promote_field_values_watch_public
from tests.test_starter import helper_dcgm_entity_api_sanity
from tests.test_starter import helper_dcgm_hostengine_environment_variable_ok
from tests.test_starter import helper_dcgm_hostengine_environment_variable_not_configured
from tests.test_starter import helper_dcgm_hostengine_environment_variable_badparam_empty
from tests.test_starter import helper_dcgm_hostengine_environment_variable_badparam_invalid
from tests.test_starter import helper_dcgm_hostengine_environment_variable_ver_mismatch
from tests.test_starter import helper_dcgm_nvlink_link_state


import test_utils
from test_utils import trampoline  # For introspection


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.skip_denylisted_gpus(["GeForce GT 640"])
def test_dcgm_agent_get_values_for_fields(handle, gpuIds):
    helper_dcgm_agent_get_values_for_fields(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_engine_watch_field_values(handle, gpuIds):
    helper_dcgm_engine_watch_field_values(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_engine_unwatch_field_value(handle, gpuIds):
    helper_dcgm_engine_unwatch_field_value(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_unwatch_field_values_public_embedded(handle, gpuIds):
    helper_unwatch_field_values_public(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_unwatch_field_values_public_remote(handle, gpuIds):
    helper_unwatch_field_values_public(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_promote_field_values_watch_public_embedded(handle, gpuIds):
    helper_promote_field_values_watch_public(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_promote_field_values_watch_public_remote(handle, gpuIds):
    helper_promote_field_values_watch_public(handle, gpuIds)


def test_dcgm_engine_update_all_fields():
    trampoline()


def test_dcgm_cgroups_device_block():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_entity_api_sanity(handle, gpuIds):
    helper_dcgm_entity_api_sanity(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20,
                                            heEnv={"CUDA_VISIBLE_DEVICES": "0,1,2"})
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_hostengine_environment_variable_ok(handle, gpuIds):
    helper_dcgm_hostengine_environment_variable_ok(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_hostengine_environment_variable_not_configured(handle, gpuIds):
    helper_dcgm_hostengine_environment_variable_not_configured(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20,
                                            heEnv={"CUDA_VISIBLE_DEVICES": "0,1,2"})
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_hostengine_environment_variable_badparam_empty(handle, gpuIds):
    helper_dcgm_hostengine_environment_variable_badparam_empty(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20,
                                            heEnv={"CUDA_VISIBLE_DEVICES": "0,1,2"})
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_hostengine_environment_variable_badparam_invalid(handle, gpuIds):
    helper_dcgm_hostengine_environment_variable_badparam_invalid(
        handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20,
                                            heEnv={"CUDA_VISIBLE_DEVICES": "0,1,2"})
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgm_hostengine_environment_variable_ver_mismatch(handle, gpuIds):
    helper_dcgm_hostengine_environment_variable_ver_mismatch(handle, gpuIds)


def test_dcgm_nvlink_link_state():
    trampoline()
