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
import tempfile

from MockMndiagController import MockMndiagController


class MockNvloomController(MockMndiagController):
    """
    Controller for mock nvloom_cli behavior and output.
    Inherits SSH setup, cleanup, and file deployment from MockMndiagController.
    Provides nvloom_cli-specific output format with globally unique MPI ranks.
    """

    BINARY_PATH_ENV = "DCGM_MNDIAG_NVLOOM_PATH"

    def __init__(self, mockPath, mockNvidiaSmiPath):
        super().__init__(mockPath, mockNvidiaSmiPath)
        fd, self._args_capture_file = tempfile.mkstemp(
            prefix="dcgm_mock_nvloom_args_")
        os.close(fd)

    def setup_multinode_mock_environment(self, config):
        """Override to compute and pass rank_offset to each node's _set_output.

        In real mpirun output, ranks are globally unique across all nodes.
        The head node gets ranks 0..N-1, worker-1 gets N..2N-1, etc.
        We store offsets in a shared dict keyed by IP so that _set_output
        can look up the correct offset for its hostList.
        """
        gpus_per_host = config.get('gpus_per_host', 4)
        test_nodes = [n for n in config['test_nodes']
                      if n.get("ip", "") != "localhost"]

        rank_offsets = {'localhost': 0}
        for i, node in enumerate(test_nodes):
            rank_offsets[node['ip']] = (i + 1) * gpus_per_host
        config['_rank_offsets'] = rank_offsets
        config['rank_offset'] = 0

        super().setup_multinode_mock_environment(config)

    def _get_bash_script_template(self):
        """Return the bash script template for the mock nvloom_cli."""
        args_file = self._args_capture_file
        return '''#!/bin/bash

{echo_commands}

# Capture full argument list for test inspection
echo "$@" > ''' + args_file + '''

# Parse command line arguments
duration=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration=*)
            duration="${{1#*=}}"
            shift
            ;;
        --duration)
            if [ -n "$2" ] && [ ${{2:0:1}} != "-" ]; then
                duration="$2"
                shift 2
            else
                shift
            fi
            ;;
        --suite|--suite=*|-s)
            if [ -n "$2" ] && [ ${{2:0:1}} != "-" ]; then
                shift 2
            else
                shift
            fi
            ;;
        *)
            shift
            ;;
    esac
done

sleep $duration
'''

    def get_captured_args(self):
        """Read the argument list captured by the mock script.

        Returns the raw argument string, or None if the file does not exist.
        """
        try:
            with open(self._args_capture_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    def get_captured_duration(self):
        """Extract the --duration value from the captured arguments.

        Returns the duration as a string, or None if not found.
        """
        args = self.get_captured_args()
        if args is None:
            return None
        for token in args.split():
            if token.startswith("--duration="):
                return token.split("=", 1)[1]
        tokens = args.split()
        for i, token in enumerate(tokens):
            if token == "--duration" and i + 1 < len(tokens):
                return tokens[i + 1]
        return None

    def cleanup_captured_args(self):
        """Remove the captured args file."""
        try:
            os.remove(self._args_capture_file)
        except FileNotFoundError:
            pass

    def _create_mock_script(self, lines, output_path=None):
        """Create the mock shell script.

        Accepts lines as a list of strings (output lines) or as a tuple
        of (lines, exit_code) for backward compatibility (exit_code is ignored).
        """
        path = output_path or self.mockPath
        if isinstance(lines, tuple):
            output_lines, _ = lines
        else:
            output_lines = lines

        try:
            safe_lines = [
                line.replace('"', r'\"').replace('`', r'\`').replace(
                    '$', r'\$') for line in output_lines]

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
        Generate nvloom_cli-format output from the test configuration.

        Config keys:
          hostList: list of hostnames
          gpus_per_host: number of GPUs per host (default 4)
          gpu_required_info: dict of entityId -> [info messages]
          gpu_required_error: dict of entityId -> [error messages]
          test_suites: list of sub-test names (default: one test)
        """
        if config is None:
            config = self.config

        stdout_lines = []
        hostList = list(config.get('hostList', []))
        gpus_per_host = config.get('gpus_per_host', 4)
        gpu_required_error = dict(config.get('gpu_required_error', {}))
        test_suites = config.get(
            'test_suites',
            ['gpu_to_rack_device_to_device_bidir_read_ce'])

        # Determine rank offset: prefer per-IP lookup, fall back to flat key
        rank_offsets = config.get('_rank_offsets', {})
        if hostList and hostList[0] in rank_offsets:
            rank_offset = rank_offsets[hostList[0]]
        else:
            rank_offset = config.get('rank_offset', 0)

        # Header
        stdout_lines.append("nvloom_cli 2.0.0")
        stdout_lines.append("git commit:")
        stdout_lines.append("NVCC version: 13.0")
        stdout_lines.append("CUDA version: 13.4")
        stdout_lines.append("Driver version: 615.27")
        stdout_lines.append("Allocation strategy: reuse")
        stdout_lines.append("Buffer size: 536870912 bytes")
        stdout_lines.append("Iteration count: 16")

        # Process info lines (ranks must be globally unique across all nodes)
        pci_addrs = [
            "00000008:01:00", "00000009:01:00",
            "00000018:01:00", "00000019:01:00",
            "00000028:01:00", "00000029:01:00",
            "00000038:01:00", "00000039:01:00",
        ]
        rank = rank_offset
        for host_idx, host in enumerate(hostList):
            rack_guid = f"123456789000{host_idx + 1}"
            for dev_idx in range(gpus_per_host):
                pci = pci_addrs[dev_idx % len(pci_addrs)]
                stdout_lines.append(
                    f"Process {rank} ({host}): device {dev_idx}: "
                    f"NVIDIA GB200 ({pci}); rackGuid: {rack_guid}")
                rank += 1

        total_ranks = rank - rank_offset
        stdout_lines.append("")

        # Sub-test blocks
        for test_name in test_suites:
            stdout_lines.append(f"Running {test_name}")

            bw_values = []
            for r in range(total_ranks):
                bw = 165.0 + (r * 0.1)
                bw_values.append(bw)
                stdout_lines.append(f"{r}\t{bw:.2f}")

            stdout_lines.append("")

            if bw_values:
                stdout_lines.append(
                    f"Min {test_name} {min(bw_values):.2f} GB/s")
                avg_bw = sum(bw_values) / len(bw_values)
                stdout_lines.append(
                    f"Avg {test_name} {avg_bw:.2f} GB/s")
                stdout_lines.append(
                    f"Max {test_name} {max(bw_values):.2f} GB/s")
                stdout_lines.append(
                    f"Total {test_name} {sum(bw_values):.2f} GB/s")

            stdout_lines.append("")
            stdout_lines.append(f"ExecutionTime {test_name} 5.0 s")
            stdout_lines.append(f"Done {test_name}")
            stdout_lines.append("")

        # Error lines use [hostname:localDeviceIdx] format. nvloom emits the
        # local CUDA device index (not the global MPI rank) in error messages.
        for host in hostList:
            for entityId_str, error_msgs in gpu_required_error.items():
                entityId = int(entityId_str)
                for msg in error_msgs:
                    stdout_lines.append(f"[{host}:{entityId}]: {msg}")

        self._create_mock_script(stdout_lines, output_path=output_path)
