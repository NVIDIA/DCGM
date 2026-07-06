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

from tests.test_topology import helper_dcgm_topology_device_standalone
from tests.test_topology import helper_dcgm_topology_group_single_gpu_standalone
from tests.test_topology import helper_dcgm_topology_device_nvlink_standalone
from tests.test_topology import helper_test_select_gpus_by_topology

import test_utils
from test_utils import trampoline  # For introspection.


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_all_supported_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_topology_device_standalone(handle, gpuIds):
    helper_dcgm_topology_device_standalone(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_all_supported_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_topology_group_single_gpu_standalone(handle, gpuIds):
    helper_dcgm_topology_group_single_gpu_standalone(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_all_supported_gpus()
def test_dcgm_topology_device_nvlink_standalone(handle, gpuIds):
    helper_dcgm_topology_device_nvlink_standalone(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('H200.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_all_supported_gpus()
@test_utils.run_only_with_nvml()
def test_select_gpus_by_topology_standalone(handle, gpuIds):
    helper_test_select_gpus_by_topology(handle, gpuIds)


def test_dcgm_get_nvlink_link_status_hopper_injection():
    trampoline()


def test_dcgm_get_nvlink_link_status_rubin_injection():
    trampoline()


def test_dcgm_get_device_topology_hopper_injection():
    trampoline()
