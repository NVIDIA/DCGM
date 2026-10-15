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


from test_globals import SAMPLE_SCRIPT_TIMEOUT
from tests.test_sdk import helper_sdk_configuration_sample_standalone
from tests.test_sdk import helper_sdk_health_sample_standalone
from tests.test_sdk import helper_sdk_policy_sample_standalone
from tests.test_sdk import helper_sdk_field_value_sample_standalone
from tests.test_sdk import helper_sdk_example_script_smoke_standalone_auto
from tests.test_sdk import helper_sdk_example_script_smoke_embedded_auto
from tests.test_sdk import helper_sdk_example_script_smoke_embedded_manual


import test_utils


# We cannot set the compute mode when MIG is enabled
@test_utils.run_only_if_mig_is_disabled()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_as_root()
def test_sdk_configuration_sample_standalone(handle, gpuIds):
    helper_sdk_configuration_sample_standalone(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_sdk_health_sample_standalone(handle, gpuIds):
    helper_sdk_health_sample_standalone(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_sdk_policy_sample_standalone(handle, gpuIds):
    helper_sdk_policy_sample_standalone(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
def test_sdk_field_value_sample_standalone(handle, gpuIds):
    helper_sdk_field_value_sample_standalone(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(timeout=SAMPLE_SCRIPT_TIMEOUT)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_as_root()
@test_utils.run_only_with_nvml()
def test_sdk_example_script_smoke_standalone_auto(handle, gpuIds):
    helper_sdk_example_script_smoke_standalone_auto(handle)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_as_root()
@test_utils.run_only_with_nvml()
def test_sdk_example_script_smoke_embedded_auto(handle, gpuIds):
    helper_sdk_example_script_smoke_embedded_auto()


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_as_root()
@test_utils.run_only_with_nvml()
def test_sdk_example_script_smoke_embedded_manual(handle, gpuIds):
    helper_sdk_example_script_smoke_embedded_manual()
