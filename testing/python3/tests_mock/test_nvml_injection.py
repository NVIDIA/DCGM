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


def test_inject_with_device_handle_parameter():
    trampoline()


def test_inject_with_extra_keys_for_following_calls_standalone():
    trampoline()


def test_inject_key_with_two_values_for_following_calls_standalone():
    trampoline()


def test_inject_struct_for_following_calls_standalone():
    trampoline()


def test_inject_with_extra_keys_standalone():
    trampoline()


def test_inject_key_with_two_values_standalone():
    trampoline()


def test_inject_struct_standalone():
    trampoline()


def test_gpm_metrics_supported():
    trampoline()
