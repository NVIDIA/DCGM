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

from tests.test_vgpu import helper_dcgm_vgpu_configure_ecc_mode
from tests.test_vgpu import helper_dcgm_vgpu_attributes

import test_utils
from test_utils import trampoline  # For introspection.


def test_dcgm_vgpu_config_standalone_get_devices():
    trampoline()


def test_dcgm_vgpu_config_standalone_get_attributes():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_as_root()
def test_dcgm_vgpu_configure_ecc_mode(handle, gpuIds):
    helper_dcgm_vgpu_configure_ecc_mode(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(60)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_vgpu_attributes(handle, gpuIds):
    helper_dcgm_vgpu_attributes(handle, gpuIds)
