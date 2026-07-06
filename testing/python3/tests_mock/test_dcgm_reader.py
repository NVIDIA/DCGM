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

from tests.test_dcgm_reader import helper_dcgm_reader_default
from tests.test_dcgm_reader import helper_dcgm_reader_specific_fields
from tests.test_dcgm_reader import helper_reading_specific_data
from tests.test_dcgm_reader import helper_dcgm_reader_all_since_last_call_false
from tests.test_dcgm_reader import helper_dcgm_reader_all_since_last_call_true
from tests.test_dcgm_reader import helper_dcgm_reader_all_since_last_call_false_repeat
from tests.test_dcgm_reader import helper_dcgm_reader_all_since_last_call_true_repeat

import test_utils
from test_utils import trampoline  # Explicit to make introspection work.


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_reader_default(handle, gpuIds):
    helper_dcgm_reader_default(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_reader_specific_fields(handle, gpuIds):
    helper_dcgm_reader_specific_fields(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_reading_specific_data(handle, gpuIds):
    helper_reading_specific_data(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_reader_all_since_last_call_false(handle, gpuIds):
    helper_dcgm_reader_all_since_last_call_false(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_reader_all_since_last_call_true(handle, gpuIds):
    helper_dcgm_reader_all_since_last_call_true(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_reader_all_since_last_call_false_repeat(handle, gpuIds):
    helper_dcgm_reader_all_since_last_call_false_repeat(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_dcgm_reader_all_since_last_call_true_repeat(handle, gpuIds):
    helper_dcgm_reader_all_since_last_call_true_repeat(handle, gpuIds)


def test_dcgm_reader_wildcard_gi():
    trampoline()


def test_dcgm_reader_latest_mig_ci_fields_by_id():
    trampoline()


def test_dcgm_reader_all_mig_ci_fields_by_id():
    trampoline()


def test_dcgm_reader_latest_mig_ci_fields_by_tag():
    trampoline()


def test_dcgm_reader_all_mig_ci_fields_by_tag():
    trampoline()
