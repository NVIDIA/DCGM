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
import shlex

from MockMndiagController import MockMndiagController


class MockMnnvbandwidthController(MockMndiagController):
    """
    Controller for mock mnnvbandwidth plain-text output.
    """

    BINARY_PATH_ENV = "DCGM_MNDIAG_MNNVBANDWIDTH_PATH"

    @staticmethod
    def _write_lines(lines, destination):
        if not lines:
            return []

        redirect = " >&2" if destination == "stderr" else ""
        return [
            f"printf '%s\\n' {shlex.quote(line)}{redirect}"
            for line in lines
        ]

    def _set_output(self, config=None, output_path=None):
        if config is None:
            config = self.config

        stdout_lines = config.get("stdout_lines", [])
        stderr_lines = config.get("stderr_lines", [])
        exit_code = config.get("exit_code", 0)
        run_seconds = config.get("run_seconds", 0)
        if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code < 0 or exit_code > 255:
            raise ValueError(f"Invalid mock exit code: {exit_code}")
        if isinstance(run_seconds, bool) or not isinstance(run_seconds, int) or run_seconds < 0:
            raise ValueError(f"Invalid mock run duration: {run_seconds}")

        commands = self._write_lines(stdout_lines, "stdout")
        commands.extend(self._write_lines(stderr_lines, "stderr"))
        command_block = "\n".join(commands) if commands else ":"
        required_args = config.get("required_args", [])
        argument_checks = "\n".join(
            "if ! printf '%s\\n' \"$@\" | grep -Fx -- "
            f"{shlex.quote(required_arg)} >/dev/null; then\n"
            f"    printf 'Missing required mnnvbandwidth argument: %s\\n' {shlex.quote(required_arg)} >&2\n"
            "    exit 1\n"
            "fi"
            for required_arg in required_args)
        script = f"""#!/bin/bash

{argument_checks}
rank="${{OMPI_COMM_WORLD_RANK:-${{PMIX_RANK:-0}}}}"
if [ "$rank" -eq 0 ]; then
{command_block}
fi
sleep {run_seconds}
exit {exit_code}
"""

        path = output_path or self.mockPath
        try:
            with open(path, "w") as mock_file:
                mock_file.write(script)
            os.chmod(path, 0o755)
        except IOError as error:
            raise RuntimeError(
                f"Failed to create mock script at {path}: {error}")
