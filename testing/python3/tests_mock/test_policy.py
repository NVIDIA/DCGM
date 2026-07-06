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

from tests.test_policy import helper_dcgm_policy_get_with_some_gpus_standalone

import test_utils
from test_utils import trampoline  # For introspection


def test_dcgm_policy_ecc_delta_detection():
    trampoline()


def test_dcgm_policy_nvlink_delta_detection():
    trampoline()


def test_dcgm_policy_pcie_delta_detection():
    trampoline()


def test_dcgm_policy_inject_thermalerror_standalone():
    trampoline()


def test_dcgm_policy_inject_powererror_standalone():
    trampoline()


def test_dcgm_policy_inject_retiredpages_standalone():
    trampoline()


def test_dcgm_policy_inject_pcierror_standalone():
    trampoline()


def test_dcgm_policy_inject_xiderror_standalone():
    trampoline()


def test_dcgm_policy_inject_nvlinkerror_standalone():
    trampoline()


def test_dcgm_policy_inject_eccerror_standalone():
    trampoline()


def test_dcgm_policy_set_get_violation_policy_standalone():
    trampoline()


def test_dcgm_policy_reg_unreg_for_policy_update_standalone():
    trampoline()


def test_dcgm_policy_negative_register_standalone():
    trampoline()


def test_dcgm_policy_negative_unregister_standalone():
    trampoline()


def test_dcgm_policy_get_with_no_gpus_standalone():
    trampoline()


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(40)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_policy_get_with_some_gpus_standalone(handle, gpuIds):
    helper_dcgm_policy_get_with_some_gpus_standalone(handle, gpuIds)
