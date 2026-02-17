# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import test_utils

_HE_ENV = {
    "PATH": test_utils.get_updated_env_path_variable(),
    "DCGM_MNDIAG_MPIRUN_PATH": test_utils.get_mpirun_path(),
    "DCGM_MNDIAG_MNUBERGEMM_PATH": test_utils.get_mock_mnubergemm_path(),
    "DCGM_MNDIAG_SUPPORTED_SKUS": test_utils.get_saved_skus_env,
    "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT": "1",
}


def HE_Env():
    # Lazily evaluate to capture any runtime changes, if desired
    return {k: (v() if callable(v) else v) for k, v in _HE_ENV.items()}
