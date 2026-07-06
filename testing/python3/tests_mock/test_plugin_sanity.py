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


from test_globals import DEV_MODE_MSG

from tests.test_plugin_sanity import no_errors_run
from tests.test_plugin_sanity import with_error_run


import test_utils
from test_utils import trampoline  # For introspection

# NO HARDWARE


def test_create_output_dir():
    trampoline()


@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_short_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "short", "short")


@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_medium_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "medium", "medium")


@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
def test_long_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "long", "long")
