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

# We trampoline into tests to run legacy tests that do not require GPU hardware.

from test_utils import trampoline


def test_dcgm_cpu_sdc_with_cpu_pass():
    trampoline()


def test_dcgm_cpu_sdc_with_cpu_fail_then_pass():
    trampoline()


def test_dcgm_cpu_sdc_with_system_pass():
    trampoline()


def test_dcgmi_cpu_sdc_missing_package_without_binary_env():
    trampoline()


def test_dcgm_diag_r_3_4_include_cpu_sdc():
    trampoline()


def test_dcgm_diag_r_3_4_include_cpu_sdc_with_service_account():
    trampoline()


def test_dcgm_cpu_sdc_with_system_fail_then_pass():
    trampoline()


def test_dcgm_cpu_sdc_with_cpu_pass_nonzero_exit():
    trampoline()


def test_cpu_sdc_plugin_is_allowed():
    trampoline()


def test_dcgmi_cpu_sdc_json_output_has_sdc_binary_version():
    trampoline()


def test_dcgmi_cpu_sdc_output_skips_sdc_binary_version_without_version_payload():
    trampoline()


def test_dcgmi_cpu_sdc_cli_output_has_sdc_binary_version():
    trampoline()
