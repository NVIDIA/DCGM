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

import os

import dcgm_nvml

_MOCK_DIR = os.path.abspath("mocks")
_LIVE_NCCL_BASE = os.path.abspath("apps/nvvs/nccl")


def _get_cuda_major_version():
    try:
        dcgm_nvml.nvmlInit()
        try:
            cuda_version = dcgm_nvml.nvmlSystemGetCudaDriverVersion_v2()
            return cuda_version // 1000
        finally:
            # Shutdown only if init succeeded
            dcgm_nvml.nvmlShutdown()
    except:
        return 13


def _get_live_nccl_dir():
    major = _get_cuda_major_version()
    return os.path.join(_LIVE_NCCL_BASE, f"cuda{major}")


_COMMON_ENV = {
    "DCGM_NCCL_TESTS_BIN_PATH": _MOCK_DIR,
    "DCGM_NCCL_TESTS_SKIP_BIN_PERMISSION_CHECK": "1"
}

_HE_ENV = {
    "pass": {
        **_COMMON_ENV,
        "MOCK_NCCL_OUTPUT_MODE": "pass"
    },
    "pass_exit_1": {
        **_COMMON_ENV,
        "MOCK_NCCL_OUTPUT_MODE": "pass_exit1"
    },
    "fail_oob": {
        **_COMMON_ENV,
        "MOCK_NCCL_OUTPUT_MODE": "fail_oob"
    },
    "fail_bw": {
        **_COMMON_ENV,
        "MOCK_NCCL_OUTPUT_MODE": "fail_bw"
    },
    "none_exit_0": {
        **_COMMON_ENV,
        "MOCK_NCCL_OUTPUT_MODE": "none",
        "MOCK_NCCL_EXIT_CODE": "0"
    },
    "none_exit_1": {
        **_COMMON_ENV,
        "MOCK_NCCL_OUTPUT_MODE": "none",
        "MOCK_NCCL_EXIT_CODE": "1"
    }
}


def HE_Env(env):
    return _HE_ENV[env].copy()


def live_binary_path():
    return os.path.join(_get_live_nccl_dir(), "all_reduce_perf")


def is_live_binary_available():
    return os.path.isfile(live_binary_path())


def HE_Env_Live(extra_env=None):
    env = {
        "DCGM_NCCL_TESTS_BIN_PATH": _get_live_nccl_dir(),
        "DCGM_NCCL_TESTS_SKIP_BIN_PERMISSION_CHECK": "1",
        # Only needed for packaged libnccl to prevent loading net plugin
        "NCCL_NET": "Socket",
    }
    if extra_env:
        env.update(extra_env)
    return env
