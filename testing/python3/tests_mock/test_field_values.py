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

from test_utils import trampoline


def test_fabric_info_v3_support():
    trampoline()


def test_nvlink_count_perlink_injection():
    trampoline()


def test_nvlink_fec_history_fields():
    trampoline()


def test_c2c_link_fields():
    trampoline()


def test_nvlink_count_aggregate_behavior_injection():
    trampoline()


def test_gpu_recovery_action_field_values():
    trampoline()


def test_nvlink_state_field_injection():
    trampoline()


def test_nvlink_state_field():
    trampoline()


def test_unrepairable_memory_query():
    trampoline()


def test_nvlink_error_counter():
    trampoline()


def test_nvlink_error_counter_per_link_fields():
    trampoline()


def test_nvlink_field_values_per_link_throughput_fields():
    trampoline()


def test_nvlink_gpm_per_link_prof_fields():
    trampoline()


def test_nvlink_error_fields():
    trampoline()


def test_nvlink_ber_fields():
    trampoline()


def test_chassis_serial_number_empty_value():
    trampoline()


def test_platform_info_fields_invalid_values():
    trampoline()


def test_platform_info_fields():
    trampoline()


def test_nvswitch_monitoring_standalone():
    trampoline()


def test_dcgm_field_values_since_agent():
    trampoline()
