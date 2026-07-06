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
# Sample script to test python bindings for DCGM

import test_utils

# We need to do this explicitly because introspection looks for a
# call to trampoline() and not test_utils.trampoline().
from test_utils import trampoline

from tests.test_dcgm_apprunners import helper_dcgm_unittests_app


def test_nv_hostengine_app():
    trampoline()


def test_dcgmi_app():
    trampoline()


@test_utils.run_with_injection_nvml()
@test_utils.run_only_on_linux()
@test_utils.run_only_with_nvml()
def test_dcgm_unittests_app(*args, **kwargs):
    helper_dcgm_unittests_app(['-f'])
