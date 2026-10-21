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

import os


MOCKED_SDC_PATH = os.path.abspath("tests/mocked_cpu_sdc")
TEST_FILES_DIR = os.path.abspath("tests/cpu_sdc_test_files")

_RESULT_FILES = {
    "cpu0_pass": "cpu0_pass.jsonl",
    "cpu0_fail_then_pass": "cpu0_fail_then_pass.jsonl",
    "system_pass": "system_pass.jsonl",
    "system_pass_no_version": "system_pass_no_version.jsonl",
    "system_fail_then_pass": "system_fail_then_pass.jsonl",
}


def result_file_path(result_name):
    return os.path.join(TEST_FILES_DIR, _RESULT_FILES[result_name])


def HE_Env(
        result_name,
        exit_code=None,
        heartbeat_timeout_disabled=None,
        heartbeat_timeout_seconds=None,
        mocked_missed_heartbeat_interval_sec=None):
    env = {
        "DCGM_CPU_SDC_BIN_PATH": MOCKED_SDC_PATH,
        "DCGM_MOCKED_SDC_RETURN_PATH": result_file_path(result_name),
        "DCGM_CPU_SDC_SKIP_BIN_PERMISSION_CHECK": "1",
        "DCGM_SUPPORT_NON_NVIDIA_CPU": "1",
    }
    if exit_code is not None:
        env["DCGM_MOCKED_SDC_EXIT_CODE"] = str(exit_code)
    if heartbeat_timeout_disabled is not None:
        env["DCGM_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED"] = str(
            heartbeat_timeout_disabled)
    if heartbeat_timeout_seconds is not None:
        env["DCGM_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS"] = str(
            heartbeat_timeout_seconds)
    if mocked_missed_heartbeat_interval_sec is not None:
        env["DCGM_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC"] = str(
            mocked_missed_heartbeat_interval_sec)
    return env
