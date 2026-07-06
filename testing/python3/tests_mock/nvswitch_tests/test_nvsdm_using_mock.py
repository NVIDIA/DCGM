# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Trampoline legacy tests.
from test_utils import trampoline


def test_nvsdm_list_switches():
    trampoline()


def test_nvsdm_list_ib_cx():
    trampoline()


def test_nvsdm_port_telemetry():
    trampoline()


def test_nvsdm_platform_telemetry():
    trampoline()


def test_nvsdm_nvlink_status():
    trampoline()


def test_nvsdm_nvswitch_health_check():
    trampoline()


def test_nvsdm_composite_field_telemetry():
    trampoline()


def test_nvsdm_mock_pause_resume():
    trampoline()


def test_nvsdm_connectx_error_injection():
    trampoline()
