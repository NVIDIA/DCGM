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

from tests.test_diag_stats import helper_test_bad_statspath

import test_utils


@test_utils.run_only_as_root()
@test_utils.with_service_account('dcgm-tests-service-account')
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20,
                                            heArgs=['--service-account',
                                                    'dcgm-tests-service-account'])
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_stats_bad_statspath_standalone_with_service_account(
        handle,
        gpuIds):
    helper_test_bad_statspath(handle, gpuIds)


@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.for_all_same_sku_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_diag_stats_bad_statspath_standalone(handle, gpuIds):
    helper_test_bad_statspath(handle, gpuIds)
