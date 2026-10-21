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

# All these tests are independent of hardware and trampoline to legacy versions
# of these tests under tests.

from tests.test_nvlink_prm import helper_nvlink_prm_manual_link_entity_creation

import dcgm_structs
import test_utils
from test_utils import trampoline  # For introspection


def test_nvlink_prm_field_registration():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('B200.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_minimum_gpu_architecture(
    dcgm_structs.DCGM_CHIP_ARCH_BLACKWELL)
@test_utils.run_only_with_nvml()
def test_nvlink_prm_manual_link_entity_creation(handle, gpuIds):
    helper_nvlink_prm_manual_link_entity_creation(handle, gpuIds)


def test_dcgm_link_t_structure():
    trampoline()


def test_dcgm_link_entity_id_encoding():
    trampoline()


def test_nvlink_prm_single_field_live_then_cached():
    trampoline()


def test_prm_field_caching_with_field_group_watcher():
    trampoline()


def test_prm_field_injection_infrastructure():
    trampoline()


def test_nvlink_prm_counter_port_offset_behavior():
    trampoline()
