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

from tests.test_dcgm_versioned_structs import helper_dcgm_connect_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_get_device_attributes_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_group_get_info_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_field_group_get_info_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_field_group_get_all_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_config_set_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_config_get_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_policy_get_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_run_diagnostic_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_get_pid_info_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_health_check_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_job_get_stats_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_introspect_get_hostengine_memory_usage_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_introspect_get_hostengine_cpu_utilization_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_get_vgpu_device_attributes_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_get_vgpu_instance_attributes_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_vgpu_config_set_validate
from tests.test_dcgm_versioned_structs import helper_dcgm_vgpu_config_get_validate

import test_utils
from test_utils import trampoline  # to make introspection work


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_connect_validate(handle, gpuIds):
    helper_dcgm_connect_validate(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_as_root()
def test_dcgm_get_device_attributes_validate(handle, gpuIds):
    helper_dcgm_get_device_attributes_validate(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_group_get_info_validate(handle, gpuIds):
    helper_dcgm_group_get_info_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_field_group_get_info_validate(handle, gpuIds):
    helper_dcgm_field_group_get_info_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_field_group_get_all_validate(handle, gpuIds):
    helper_dcgm_field_group_get_all_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_config_set_validate(handle, gpuIds):
    helper_dcgm_config_set_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_config_get_validate(handle, gpuIds):
    helper_dcgm_config_get_validate(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_policy_get_validate(handle, gpuIds):
    helper_dcgm_policy_get_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
@test_utils.run_with_max_power_limit_set()
def test_dcgm_run_diagnostic_validate(handle, gpuIds):
    helper_dcgm_run_diagnostic_validate(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_get_pid_info_validate(handle, gpuIds):
    helper_dcgm_get_pid_info_validate(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_health_check_validate(handle, gpuIds):
    helper_dcgm_health_check_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_job_get_stats_validate(handle, gpuIds):
    helper_dcgm_job_get_stats_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_introspect_get_hostengine_memory_usage_validate(handle, gpuIds):
    helper_dcgm_introspect_get_hostengine_memory_usage_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_introspect_get_hostengine_cpu_utilization_validate(handle, gpuIds):
    helper_dcgm_introspect_get_hostengine_cpu_utilization_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_get_vgpu_device_attributes_validate(handle, gpuIds):
    helper_dcgm_get_vgpu_device_attributes_validate(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_get_vgpu_instance_attributes_validate(handle, gpuIds):
    helper_dcgm_get_vgpu_instance_attributes_validate(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_vgpu_config_set_validate(handle, gpuIds):
    helper_dcgm_vgpu_config_set_validate(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_vgpu_config_get_validate(handle, gpuIds):
    helper_dcgm_vgpu_config_get_validate(handle)
