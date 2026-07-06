# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from tests.test_dcgmi import helper_dcgmi_diag_env_var_warning_hostengine_set
from tests.test_dcgmi import helper_dcgmi_diag_removed_env_var_positive
from tests.test_dcgmi import helper_dcgmi_diag_removed_env_var_negative
from tests.test_dcgmi import helper_dcgmi_diag_help
from tests.test_dcgmi import helper_dcgmi_diag_h
from tests.test_dcgmi import helper_dcgmi_diag_env_var_warning_hostengine_not_set
from tests.test_dcgmi import helper_dcgmi_diag_ignore_error_codes_multiple_gpus_all_pass
from tests.test_dcgmi import helper_dcgmi_test_inject
from tests.test_dcgmi import helper_dcgmi_test_introspect
from tests.test_dcgmi import helper_dcgmi_modules
from tests.test_dcgmi import helper_dcgmi_nvlink_nvswitches
from tests.test_dcgmi import helper_dcgmi_nvlink
from tests.test_dcgmi import helper_dcgmi_stats
from tests.test_dcgmi import helper_dcgmi_discovery
from tests.test_dcgmi import helper_dcgmi_health
from tests.test_dcgmi import helper_dcgmi_config

import dcgm_structs
import test_utils

from test_utils import trampoline  # To make introspection work.


def test_dcgmi_group():
    trampoline()


def test_dcgmi_group_nvswitch():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_config(handle, gpuIds):
    helper_dcgmi_config(handle, gpuIds)


def test_dcgmi_policy():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_health(handle, gpuIds):
    helper_dcgmi_health(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_discovery(handle, gpuIds):
    helper_dcgmi_discovery(handle, gpuIds)


def test_dcgmi_discovery_can_list_cx_mocked():
    trampoline()


def test_dcgmi_diag_invalid_test_specified():
    trampoline()


def test_dcgmi_diag_on_heterogeneous_env():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_as_root()
def test_dcgmi_stats(handle, gpuIds):
    helper_dcgmi_stats(handle, gpuIds)


def test_dcgmi_port():
    trampoline()


def test_dcgmi_field_groups():
    trampoline()


def test_dcgmi_introspect():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_nvlink(handle, gpuIds):
    helper_dcgmi_nvlink(handle, gpuIds)


def test_dcgmi_nvlink_show_entity_ids_nvswitch():
    trampoline()


def test_dcgmi_nvlink_show_entity_ids():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_with_injection_nvswitches(2)
def test_dcgmi_nvlink_nvswitches(handle, gpuIds, switchIds):
    helper_dcgmi_nvlink_nvswitches(handle, gpuIds, switchIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_modules(handle, gpuIds):
    helper_dcgmi_modules(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_test_introspect(handle, gpuIds):
    helper_dcgmi_test_introspect(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgmi_test_inject(handle, gpuIds):
    helper_dcgmi_test_inject(handle, gpuIds)


def test_dcgmi_dmon_pause_resume():
    trampoline()


def test_dcgmi_settings_logging_severity():
    trampoline()


def test_dcgmi_set():
    trampoline()


def test_dcgmi_global_and_others():
    trampoline()


def test_dcgmi_diag_expected_num_entities():
    trampoline()


def test_dcgmi_diag_missing_gpu_expected_num_entities():
    trampoline()


def test_dcgmi_diag_ignore_error_codes_with_injected_gpus():
    trampoline()


def test_dcgmi_diag_ignore_error_codes_software():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120,
                                            heEnv={"CUDA_VISIBLE_DEVICES": "1,2,3"})
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_env_var_warning_hostengine_set(handle, gpuIds):
    helper_dcgmi_diag_env_var_warning_hostengine_set(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_env_var_warning_hostengine_not_set(handle, gpuIds):
    helper_dcgmi_diag_env_var_warning_hostengine_not_set(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_diag_h(handle, gpuIds):
    helper_dcgmi_diag_h(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_diag_help(handle, gpuIds):
    helper_dcgmi_diag_help(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_removed_env_var_negative(handle, gpuIds):
    helper_dcgmi_diag_removed_env_var_negative(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(120,
                                            heEnv={"DCGM_DIAG_REMOVED": "1"})
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_dcgmi_diag_removed_env_var_positive(handle, gpuIds):
    helper_dcgmi_diag_removed_env_var_positive(handle, gpuIds)


def test_dcgmi_diag_software_memory_health():
    trampoline()
