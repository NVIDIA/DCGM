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


def test_health_check_instances_standalone():
    trampoline()


def test_fake_mig_device_profile_names_standalone():
    trampoline()


def test_inject_instance_fields_standalone():
    trampoline()


def test_instances_fetch_global_fields():
    trampoline()


def test_instances_large_mig_topology_getlatestvalues_v2():
    trampoline()
