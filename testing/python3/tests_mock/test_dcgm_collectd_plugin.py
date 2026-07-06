# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from tests.test_dcgm_collectd_plugin import helper_collectd_basic_integration
from tests.test_dcgm_collectd_plugin import helper_collectd_config_integration
from tests.test_dcgm_collectd_plugin import helper_collectd_config_bad_alpha_field
from tests.test_dcgm_collectd_plugin import helper_collectd_config_bad_numeric_field
from tests.test_dcgm_collectd_plugin import helper_collectd_config_no_fields

import test_utils


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_collectd_basic_integration(handle, gpuIds):
    helper_collectd_basic_integration(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_collectd_config_integration(handle, gpuIds):
    helper_collectd_config_integration(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_collectd_config_bad_alpha_field(handle, gpuIds):
    helper_collectd_config_bad_alpha_field(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_collectd_config_bad_numeric_field(handle, gpuIds):
    helper_collectd_config_bad_numeric_field(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_collectd_config_no_fields(handle, gpuIds):
    helper_collectd_config_no_fields(handle, gpuIds)
