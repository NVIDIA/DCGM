# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


def test_bind_unbind_gpu_status():
    trampoline()


def test_bind_unbind_gpu_index_order_remain_the_same():
    trampoline()


def test_dcgm_add_inactive_gpu_to_group_will_fail():
    trampoline()


def test_dcgm_field_watch_inactive_to_active():
    trampoline()


def test_dcgm_field_watch_on_meta_group_will_affect_on_new_attached_gpus():
    trampoline()


def test_diag_should_only_run_on_active_gpus():
    trampoline()


def test_dcgm_health_watch_on_meta_group_will_affect_on_new_attached_gpus():
    trampoline()


def test_dcgm_health_will_not_report_incident_for_inactive_gpus():
    trampoline()


def test_dcgm_config_will_re_apply_only_on_attached_gpus():
    trampoline()


def test_dcgm_policy_on_meta_group_will_affect_on_new_attached_gpus():
    trampoline()


def test_dcgm_policy_will_affect_on_alive_gpus():
    trampoline()


def test_dcgm_policy_on_meta_group_will_not_affect_on_new_attached_gpus_after_unregister():
    trampoline()


def test_dcgm_policy_on_normal_group_will_not_affect_on_new_attached_gpus():
    trampoline()


def test_dcgm_policy_will_not_trigger_on_inactive_gpus():
    trampoline()


def test_dcgm_gpu_index_and_nvml_index_mapping():
    trampoline()


def test_topology_device_with_detached_gpu():
    trampoline()


def test_select_gpus_by_topology_with_detached_gpu():
    trampoline()


def test_dcgm_field_bind_unbind_event_gpu_is_unbound_bound():
    trampoline()


def test_group_manager_will_remove_detached_gpu_from_group():
    trampoline()


def test_diag_some_requested_gpus_detached():
    trampoline()


def test_diag_all_requested_gpus_detached():
    trampoline()


def test_bind_gpu_back_will_re_register_nvml_events():
    trampoline()


def test_dcgmi_diag_with_one_gpu_detached():
    trampoline()


def test_dcgmi_discovery_shows_detached_gpu_status():
    trampoline()


def test_nvlink_p2p_status_field_with_detached_gpu():
    trampoline()


def test_nvlink_p2p_status_api_with_detached_gpu():
    trampoline()


def test_get_nvlink_status_api_with_detached_gpu():
    trampoline()


def test_nvlink_errors_on_detached_gpu():
    trampoline()
