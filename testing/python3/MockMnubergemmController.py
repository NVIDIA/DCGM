# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logger
import stat
import test_utils


class MockMnubergemmController:
    """
    Controller class to manage mock mnubergemm behavior and output
    """
    def __init__(self, mockPath, mockNvidiaSmiPath):
        self.mockPath = mockPath
        self.mockNvidiaSmiPath = mockNvidiaSmiPath
        self.originalEnv = {}
        self.config = None
        self.envFile = "/tmp/multinode_env.sh"
        self.counter_file = "/tmp/nvidia_smi_call_count"
        
    def setup_localhost_mock_environment(self, config):
        # Set the environment variables
        self.originalEnv = os.environ.copy()
        self.config = config

        # Set nvidia-smi overwrite process detection sh
        self._generate_mock_nvidia_smi_local()

        # Set expected output
        self._set_output()

    def setup_multinode_mock_environment(self, config):
        # Get test nodes, localhost is head node
        test_nodes = config['test_nodes']

        # Remove localhost from test nodes
        test_nodes = [node for node in test_nodes if node.get("ip", "") != "localhost"]
        if not test_nodes:
            raise ValueError("No test nodes specified")

        # Setup remote mock env
        for node in test_nodes:
            hostname = node["hostname"]
            ip = node["ip"]
            ssh_cmd = f"ssh {hostname}@{ip}"
            env_lines = []

            # 1. Check/update mpirun and set DCGM_MNDIAG_MPIRUN_PATH
            self._check_and_update_mpirun_path(ssh_cmd, ip, env_lines)

            # 2. Generate and copy mock mnubergemm binary to /tmp/mock_mnubergemm on remote nodes and set env 
            self._generate_mock_mnubergemm_remote(ssh_cmd, config, node, hostname, ip, env_lines)

            # 3. Generate and copy mock nvidia-smi on remote nodes
            self._generate_mock_nvidia_smi_remote(ssh_cmd, hostname, ip, env_lines)

            # 4. Add SKU and Save env file into the remote node
            env_lines.append(f"export DCGM_MNDIAG_SUPPORTED_SKUS={node['sku']} ")
            self._save_env_file(hostname, ip, env_lines)

            # 5. Run hostengine on the remote node
            self._run_hostengine(ssh_cmd, ip, node["hostengine_path"])

        # Setup localhost mock env
        self.setup_localhost_mock_environment(config)

    def cleanup_mock_environment(self):
        # Restore the original environment if it was saved
        if hasattr(self, 'originalEnv') and self.originalEnv:
            os.environ.clear()
            os.environ.update(self.originalEnv)

        # Delete mock file if it exists
        if os.path.exists(self.mockPath):
            os.remove(self.mockPath)

        # Restore original nvidia-smi
        self._restore_nvidia_smi()

    def cleanup_multinode_mock_environment(self, config):
        test_nodes = config['test_nodes']
        test_nodes = [node for node in test_nodes if node.get("ip", "") != "localhost"]

        for node in test_nodes:
            hostname = node["hostname"]
            ip = node["ip"]
            ssh_cmd = f"ssh {hostname}@{ip}"

            # Remove env variable lines from env file
            envs = (
                'DCGM_MNDIAG_MPIRUN_PATH',
                'DCGM_MNDIAG_MNUBERGEMM_PATH', 
                'DCGM_MNDIAG_SUPPORTED_SKUS')
            command = ';'.join((f'/{env}/d' for env in envs))
            os.system(f"""{ssh_cmd} 'sed -i "{command}" {self.envFile}'""")

            # Remove mock mnubergemm file
            os.system(f"{ssh_cmd} 'rm -f {self.mockPath}'")

            # Stop hostengine
            os.system(f"{ssh_cmd} '{node['hostengine_path']} -t'")

            # Remove mock nvidia-smi and counter file
            os.system(f"{ssh_cmd} 'rm -f {self.mockNvidiaSmiPath}'")
            os.system(f"{ssh_cmd} 'rm -f {self.counter_file}'")

        # Cleanup head node
        self.cleanup_mock_environment()

    def get_error_entities(self):
        error_entities = {}
        for host in self.config["hostList"]:
            for entityId, msgs in self.config["gpu_required_error"].items():
                if len(msgs) > 0:  # Only if there are error messages
                    key = f"{host}:{entityId}"
                    error_entities[key] = msgs

        for node in self.config["test_nodes"]:
            for entityId, msgs in node["gpu_required_error"].items():
                if len(msgs) > 0:  # Only if there are error messages
                    key = f"{node['ip']}:{entityId}"
                    error_entities[key] = msgs
                    
        return error_entities

    def get_driver_versions(self, config):
        driver_versions = set()
        
        # Get driver version for head node (localhost)
        try:
            result = os.popen("nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader").read().strip()
            if result:
                driver_versions.add(result)
        except Exception as e:
            logger.error(f"Failed to get driver version for head node: {e}")

        # Get driver versions for remote nodes
        for node in config.get("test_nodes", []):
            try:
                hostname = node["hostname"]
                ip = node["ip"]
                ssh_cmd = f"ssh {hostname}@{ip}"
                
                # Run nvidia-smi on remote node
                result = os.popen(f"{ssh_cmd} 'nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader'").read().strip()
                if result:
                    driver_versions.add(result)
            except Exception as e:
                logger.error(f"Failed to get driver version for node {ip}: {e}")

        return driver_versions

    # -------------------------------------
    def _get_bash_script_template(self):
        """Return the bash script template for the mock."""
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

    def _create_mock_script(self, lines):
        """Create the mock shell script."""
        try:
            # Sanitize lines to prevent shell injection
            safe_lines = [line.replace('"', r'\"').replace('`', r'\`').replace('$', r'\$') 
                         for line in lines]
            
            echo_commands = '\n'.join(f'echo "{line}"' for line in safe_lines)
            script_content = self._get_bash_script_template().format(echo_commands=echo_commands)
            
            with open(self.mockPath, 'w') as f:
                f.write(script_content)
                
            # Make the script executable
            os.chmod(self.mockPath, 0o755)
            
        except IOError as e:
            raise RuntimeError(f"Failed to create mock script at {self.mockPath}: {e}")

    def _set_output(self):
        """
        Generate and set the output for the mock mnubergemm, supporting per-entityId info and error messages
        """
        config = self.config
        lines = []
        hostList = config.get('hostList', [])
        gpu_required_info = config.get('gpu_required_info', {})
        gpu_required_error = config.get('gpu_required_error', {})
        for host in hostList:
            # Info lines
            for entityId, info_msgs in gpu_required_info.items():
                for msg in info_msgs:
                    lines.append(f"MNUB [I] G: {entityId} {host} L: {entityId} {msg}")
            # Error lines
            for entityId, error_msgs in gpu_required_error.items():
                for msg in error_msgs:
                    lines.append(f"MNUB [E] G: {entityId} {host} L: {entityId} T:N {msg}")

        # Create the mock script using the template approach
        self._create_mock_script(lines)

    # ----------------
    def _generate_mock_nvidia_smi_local(self):
        """
        Create temporary mock nvidia-smi script that outputs hardcoded process info
        Uses a counter file to track number of calls and return different outputs
        """
        
        # Create mock nvidia-smi script with counter-based output
        mock_script_content = f'''#!/bin/bash
# Mock nvidia-smi for DCGM tests

# Counter file to track number of calls
COUNTER_FILE="/tmp/nvidia_smi_call_count"

# Initialize counter if it doesn't exist
if [ ! -f "$COUNTER_FILE" ]; then
    echo "0" > "$COUNTER_FILE"
fi

if [[ "$1" == "--query-compute-apps=pid,process_name" && "$2" == "--format=csv,noheader,nounits" ]]; then
    # Read and increment counter
    CALL_COUNT=$(($(cat "$COUNTER_FILE") + 1))
    echo "$CALL_COUNT" > "$COUNTER_FILE"
    
    # First call should return no processes
    if [ "$CALL_COUNT" -eq 1 ]; then
        exit 0
    fi
    
    # Subsequent calls return the mock process
    echo "3505755, {self.mockPath}"
    exit 0
fi

# For other calls, try to find and call real nvidia-smi
REAL_NVIDIA_SMI=$(PATH="{os.environ.get('PATH', '')}" which nvidia-smi 2>/dev/null | grep -v "{self.mockNvidiaSmiPath}" | head -n1)
if [[ -n "$REAL_NVIDIA_SMI" ]]; then
    exec "$REAL_NVIDIA_SMI" "$@"
else
    echo "nvidia-smi: command not found" >&2
    exit 127
fi
'''
        
        # Write the script
        with open(self.mockNvidiaSmiPath, 'w') as f:
            f.write(mock_script_content)
        
        # Make it executable
        os.chmod(self.mockNvidiaSmiPath, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    # ----------------
    def _restore_nvidia_smi(self):
        """
        Restore original nvidia-smi environment by:
        1. Removing the mock nvidia-smi directory from PATH
        2. Cleaning up the temporary directory and counter file
        """
        # Remove counter file
        if os.path.exists(self.counter_file):
            os.remove(self.counter_file)

        if hasattr(self, 'mockNvidiaSmiPath'):
            # Remove our mock directory from PATH if it exists
            if 'PATH' in os.environ:
                paths = os.environ['PATH'].split(':')
                paths = [p for p in paths if p != self.mockNvidiaSmiPath]
                os.environ['PATH'] = ':'.join(paths)
                
            # Clean up the attributes
            delattr(self, 'mockNvidiaSmiPath')

    # ----------------
    def _run_hostengine(self, ssh_cmd, ip, hostengine_path):
        # Check if hostengine binary exists
        check_bin_cmd = f"{ssh_cmd} 'test -x {hostengine_path} && echo OK || echo FAIL'"
        bin_check_result = os.popen(check_bin_cmd).read().strip()
        if bin_check_result != "OK":
            raise RuntimeError(f"Failed to find hostengine binary on {ip}")

        # Execute
        os.system(f"{ssh_cmd} 'source {self.envFile} && {hostengine_path} --log-level debug'")

    # ----------------
    def _save_env_file(self, hostname, ip, env_lines):
        with open(self.envFile, "w") as f:
            f.write("\n".join(env_lines) + "\n")
        os.system(f"scp {self.envFile} {hostname}@{ip}:{self.envFile}")
        os.remove(self.envFile)

    # ----------------
    def _generate_mock_mnubergemm_remote(self, ssh_cmd, config, node, hostname, ip, env_lines):
        node_config = config.copy()
        node_config['hostList'] = [ip]
        node_config['gpu_required_info'] = node.get('gpu_required_info', {})
        node_config['gpu_required_error'] = node.get('gpu_required_error', {})
        self.config = node_config
        self._set_output()

        os.system(f"{ssh_cmd} 'rm -f {self.mockPath}'")
        os.system(f"scp {self.mockPath} {hostname}@{ip}:{self.mockPath}")
        
        # Check if mock mnubergemm file exists
        check_out_cmd = f"{ssh_cmd} 'test -f {self.mockPath} && echo OK || echo FAIL'"
        out_check_result = os.popen(check_out_cmd).read().strip()
        if out_check_result != "OK":
            raise RuntimeError(f"Failed to create mock mnubergemm file on {hostname}@{ip}:{self.mockPath}")

        env_lines.append(f"export DCGM_MNDIAG_MNUBERGEMM_PATH={self.mockPath} ")
        os.remove(self.mockPath)

    # ----------------
    def _check_and_update_mpirun_path(self, ssh_cmd, ip, env_lines):
        mpirun_path_cmd = f"{ssh_cmd} 'which mpirun'"
        mpirun_path = os.popen(mpirun_path_cmd).read().strip().split('\n')[-1]
        if not mpirun_path:
            test_utils.skip_test(f"Failed to find mpirun on {ip}")
        
        env_lines.append(f"export DCGM_MNDIAG_MPIRUN_PATH={mpirun_path} ")

    # ----------------
    def _generate_mock_nvidia_smi_remote(self, ssh_cmd, hostname, ip, env_lines):
        self._generate_mock_nvidia_smi_local()

        # Copy mock nvidia-smi to remote node
        os.system(f"scp {self.mockNvidiaSmiPath} {hostname}@{ip}:{self.mockNvidiaSmiPath}")

        # Make it executable on remote node
        os.system(f"{ssh_cmd} 'chmod +x {self.mockNvidiaSmiPath}'")

        env_lines.append(f"export PATH={os.path.dirname(self.mockNvidiaSmiPath)}:$PATH")
