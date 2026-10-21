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

from MockMndiagController import MockMndiagController


class MockMnubergemmController(MockMndiagController):
    """
    Controller for mock mnubergemm behavior and output.
    Inherits SSH setup, cleanup, and file deployment from MockMndiagController.
    Provides mnubergemm-specific output format.
    """

    BINARY_PATH_ENV = "DCGM_MNDIAG_MNUBERGEMM_PATH"

    def _get_bash_script_template(self):
        """Return the bash script template for the mock mnubergemm."""
        return '''#!/bin/bash

{echo_commands}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time_to_run=*)
            time_to_run="${{1#*=}}"
            shift
            ;;
        --time_to_run)
            if [ -n "$2" ] && [ ${{2:0:1}} != "-" ]; then
                time_to_run="$2"
                shift 2
            else
                echo "Error: --time_to_run requires a value"
                exit 1
            fi
            ;;
        *)
            shift
            ;;
    esac
done

# Add a sleep to simulate execution time
sleep $time_to_run
'''

    def _create_mock_script(self, lines, output_path=None):
        """Create the mock shell script."""
        path = output_path or self.mockPath
        try:
            safe_lines = [
                line.replace('"', r'\"').replace('`', r'\`').replace(
                    '$', r'\$') for line in lines]

            echo_commands = '\n'.join(f'echo "{line}"' for line in safe_lines)
            script_content = self._get_bash_script_template().format(
                echo_commands=echo_commands)

            with open(path, 'w') as f:
                f.write(script_content)

            os.chmod(path, 0o755)

        except IOError as e:
            raise RuntimeError(
                f"Failed to create mock script at {path}: {e}")

    def _set_output(self, config=None, output_path=None):
        """
        Generate mnubergemm-format output with per-entityId info and error messages.
        """
        if config is None:
            config = self.config
        lines = []
        hostList = config.get('hostList', [])
        gpu_required_info = config.get('gpu_required_info', {})
        gpu_required_error = config.get('gpu_required_error', {})
        for host in hostList:
            for entityId, info_msgs in gpu_required_info.items():
                for msg in info_msgs:
                    lines.append(
                        f"MNUB [I] G: {entityId} {host} L: {entityId} {msg}")
            for entityId, error_msgs in gpu_required_error.items():
                for msg in error_msgs:
                    lines.append(
                        f"MNUB [E] G: {entityId} {host} L: {entityId} T:N {msg}")

        self._create_mock_script(lines, output_path=output_path)
