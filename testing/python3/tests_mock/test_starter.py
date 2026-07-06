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

from tests.test_starter import helper_dcgm_nvlink_link_state

import test_utils
from test_utils import trampoline  # for introspection


@test_utils.run_with_injection_nvml()
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_all_supported_gpus()
@test_utils.skip_denylisted_gpus(["GeForce GT 640"])
@test_utils.run_with_injection_nvswitches(2)
def test_dcgm_nvlink_link_state(handle, gpuIds, switchIds):
    helper_dcgm_nvlink_link_state(handle, gpuIds, switchIds)
