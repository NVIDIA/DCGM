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

from tests.test_imex_fields import helper_imex_fields_basic_retrieval
from tests.test_imex_fields import helper_imex_domain_status_values
from tests.test_imex_fields import helper_imex_daemon_status_values
from tests.test_imex_fields import helper_imex_fields_consistency
from tests.test_imex_fields import helper_imex_fields_update_frequency

import test_utils


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_imex_fields_basic_retrieval(handle, gpuIds):
    helper_imex_fields_basic_retrieval(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_imex_domain_status_values(handle, gpuIds):
    helper_imex_domain_status_values(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_imex_daemon_status_values(handle, gpuIds):
    helper_imex_daemon_status_values(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_imex_fields_consistency(handle, gpuIds):
    """Test that IMEX field values are consistent across multiple GPUs"""
    helper_imex_fields_consistency(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_imex_fields_update_frequency(handle, gpuIds):
    """Test that IMEX fields can be updated at different frequencies"""
    helper_imex_fields_update_frequency(handle, gpuIds)
