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
# test the policy manager for DCGM

import test_utils

from tests.test_configmanager import helper_dcgm_error_code_propagation
from tests.test_configmanager import helper_dcgm_verify_sync_boost_multi_gpu
from tests.test_configmanager import helper_dcgm_verify_sync_boost_single_gpu
from tests.test_configmanager import helper_dcgm_port_standalone
from tests.test_configmanager import helper_dcgm_default_status_handler
from tests.test_configmanager import helper_dcgm_config_standalone_get_devices
from tests.test_configmanager import helper_verify_power_value
from tests.test_configmanager import helper_dcgm_config_powerbudget
from tests.test_configmanager import helper_dcgm_config_enforce
from tests.test_configmanager import helper_dcgm_config_get
from tests.test_configmanager import helper_dcgm_config_set
from tests.test_configmanager import helper_dcgm_config_get_attributes

from _test_helpers import skip_test_if_no_dcgm_nvml


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_config_standalone_get_devices(handle, gpuIds):
    helper_dcgm_config_standalone_get_devices(handle)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_config_standalone_get_attributes(handle, gpuIds):
    helper_dcgm_config_get_attributes(handle)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.run_only_as_root()
def test_dcgm_config_set_standalone(handle, gpuIds):
    helper_dcgm_config_set(handle)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.run_only_as_root()
def test_dcgm_config_get_standalone(handle, gpuIds):
    helper_dcgm_config_get(handle)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.run_only_as_root()
def test_dcgm_config_enforce_standalone(handle, gpuIds):
    helper_dcgm_config_enforce(handle)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.run_only_as_root()
def test_dcgm_config_powerbudget_standalone(handle, gpuIds):
    helper_dcgm_config_powerbudget(handle, gpuIds)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.run_only_as_root()
def test_dcgm_default_status_handler(handle, gpuIds):
    helper_dcgm_default_status_handler(handle, gpuIds)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(
    20, "127.0.0.1:5545", ["--port", "5545"])
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_port_standalone(handle, gpuIds):
    helper_dcgm_port_standalone(handle, gpuIds)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.run_only_as_root()
def test_dcgm_verify_sync_boost_single_gpu_standalone(handle, gpuIds):
    helper_dcgm_verify_sync_boost_single_gpu(handle, gpuIds)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_verify_sync_boost_multi_gpu_standalone(handle, gpuIds):
    helper_dcgm_verify_sync_boost_multi_gpu(handle, gpuIds)


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_as_root()
def test_dcgm_error_code_propagation(handle, gpuIds):
    helper_dcgm_error_code_propagation(handle, gpuIds)
