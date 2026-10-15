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
import subprocess

import test_utils
import utils


# NO HARDWARE
@test_utils.run_only_on_linux()
def test_module_log_isolation():
    """
    Verifies per-cdylib module logging isolation via the Rust dlmopen harness.
    """
    apps_dir = utils.get_testing_framework_library_path()
    host = os.path.abspath(os.path.join(apps_dir, "module_log_test_host"))
    plugin = os.path.abspath(os.path.join(apps_dir, "libplugin_log_test.so"))

    if not os.path.exists(host):
        test_utils.skip_test(f"Skipping: [{host}] does not exist.")
    if not os.path.exists(plugin):
        test_utils.skip_test(f"Skipping: [{plugin}] does not exist.")

    result = subprocess.run(
        [host, "--plugin-path", plugin],
        cwd=os.path.dirname(host),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        f"module_log_test_host failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
