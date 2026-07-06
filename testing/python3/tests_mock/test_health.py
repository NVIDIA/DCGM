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


def test_dcgmi_health_check_nvlink_arch_specific_fields():
    trampoline()


def test_health_check_gpu_recovery_action_multiple_gpus_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_multiple_gpus_standalone():
    trampoline()


def test_health_check_gpu_recovery_action_all_watches_unhealthy_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_all_watches_unhealthy_standalone():
    trampoline()


def test_health_check_gpu_recovery_action_all_watches_healthy_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_all_watches_healthy_standalone():
    trampoline()


def test_health_check_gpu_recovery_action_blank_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_blank_standalone():
    trampoline()


def test_health_check_gpu_recovery_action_drain_and_reset_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_drain_and_reset_standalone():
    trampoline()


def test_health_check_gpu_recovery_action_drain_p2p_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_drain_p2p_standalone():
    trampoline()


def test_health_check_gpu_recovery_action_reboot_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_reboot_standalone():
    trampoline()


def test_health_check_gpu_recovery_action_reset_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_reset_standalone():
    trampoline()


def test_health_check_gpu_recovery_action_none_embedded():
    trampoline()


def test_health_check_gpu_recovery_action_none_standalone():
    trampoline()


def test_dcgm_health_fabric_manager_status():
    trampoline()


def test_dcgm_health_check_imex_status():
    trampoline()


def test_dcgm_health_check_fabric_health_mask():
    trampoline()


def test_dcgm_health_check_mem_unrepairable_flag():
    trampoline()


def test_dcgm_health_check_pcie_correctable_errors_field_injection_valid():
    trampoline()


def test_dcgm_health_cpu_power():
    trampoline()


def test_dcgm_health_cpu_thermal():
    trampoline()


def test_dcgm_health_check_row_remap_failure():
    trampoline()


def test_dcgm_health_check_contained_error():
    trampoline()


def test_dcgm_health_check_uncontained_errors():
    trampoline()


def test_health_set_version2_standalone():
    trampoline()


def test_health_check_standalone_unreadable_power_usage():
    trampoline()


def test_health_check_standalone_multiple_failures():
    trampoline()


def test_health_check_nvlink_mig_down_with_mig_enabled_embedded():
    trampoline()


def test_health_check_nvlink_mig_down_with_mig_enabled_standalone():
    trampoline()


def test_health_check_nvlink_mig_down_without_mig_embedded():
    trampoline()


def test_health_check_nvlink_mig_down_without_mig_standalone():
    trampoline()


def test_health_check_nvlink_not_supported_and_disabled_embedded():
    trampoline()


def test_health_check_nvlink_not_supported_and_disabled_standalone():
    trampoline()


def test_health_check_nvlink_one_link_down_embedded():
    trampoline()


def test_health_check_nvlink_one_link_down_standalone():
    trampoline()


def test_health_check_nvlink_all_links_up_embedded():
    trampoline()


def test_health_check_nvlink_all_links_up_standalone():
    trampoline()


def test_health_check_nvlink_link_down_gpu_standalone():
    trampoline()


def test_health_check_nvswitch_nonfatal_errors_standalone():
    trampoline()


def test_health_check_nvswitch_fatal_errors_standalone():
    trampoline()


def test_dcgm_embedded_nvlink5_error_counters():
    trampoline()


def test_dcgm_standalone_blackwell_symbol_ber_aggregate_floor_health():
    trampoline()


def test_dcgm_standalone_nvlink5_error_counters():
    trampoline()


def test_dcgm_health_nvlink_effective_ber_threshold():
    trampoline()


def test_dcgm_health_nvlink_symbol_threshold():
    trampoline()


def test_dcgm_embedded_nvlink_crc_threshold():
    trampoline()


def test_dcgm_standalone_nvlink_crc_threshold():
    trampoline()


def test_dcgm_embedded_nvlink_fatal():
    trampoline()


def test_dcgm_standalone_nvlink_fatal():
    trampoline()


def test_dcgm_embedded_health_check_nvlink():
    trampoline()


def test_dcgm_standalone_health_check_nvlink():
    trampoline()


def test_dcgm_standalone_health_check_power():
    trampoline()


def test_dcgm_standalone_health_check_thermal():
    trampoline()


def test_dcgm_health_check_mem_standalone():
    trampoline()


def test_dcgm_health_check_mem_retirements_standalone():
    trampoline()


def test_dcgm_health_check_mem_dbe():
    trampoline()


def test_dcgm_health_check_pcie_embedded_using_nvml_injection():
    trampoline()


def test_dcgm_health_check_pcie_standalone():
    trampoline()


def test_dcgm_health_check_xid93_volta_reported():
    trampoline()


def test_dcgm_health_check_xid93_non_volta_ignored():
    trampoline()
