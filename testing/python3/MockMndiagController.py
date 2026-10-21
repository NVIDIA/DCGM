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

import concurrent.futures
import os
import shlex
import stat
import subprocess
import tempfile
import time

import logger
import test_utils


class MockMndiagController:
    """
    Base controller for mock mndiag test binaries.

    Provides all generic infrastructure: SSH setup/cleanup, file deployment
    to remote nodes, nvidia-smi mocking, and parallel node management.

    Subclasses override BINARY_PATH_ENV, _set_output(), _get_bash_script_template(),
    and _create_mock_script() to produce test-binary-specific output.
    """

    MAX_PARALLEL_WORKERS = 64
    SSH_MAX_RETRIES = 3
    SSH_RETRY_DELAY = 5
    DRIVER_VERSION_QUERY_TIMEOUT_SECONDS = 10
    SSH_MUX_OPTS = [
        "-o", "ControlMaster=auto",  # relingo_ignore: OpenSSH keyword
        "-o", "ControlPath=~/.ssh/dcgm-mux-%r@%h:%p",
        "-o", "ControlPersist=10",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
    ]

    BINARY_PATH_ENV = None  # Subclasses must set this

    @staticmethod
    def _run_with_retry(cmd, retries=3, delay=5, **kwargs):
        """Run a subprocess command with exponential-backoff retries."""
        for attempt in range(1, retries + 1):
            try:
                return subprocess.run(cmd, **kwargs)
            except subprocess.CalledProcessError:
                if attempt == retries:
                    raise
                time.sleep(delay * attempt)
            except OSError:
                if attempt == retries:
                    raise
                time.sleep(delay * attempt)

    def __init__(self, mockPath, mockNvidiaSmiPath):
        """Initialize with paths for the mock test binary and mock nvidia-smi."""
        self.mockPath = mockPath
        self.mockNvidiaSmiPath = mockNvidiaSmiPath
        self.originalEnv = {}
        self.config = None
        self._instance_id = os.getpid()
        self.envFile = f"/tmp/multinode_env_{self._instance_id}.sh"
        self.counter_file = f"/tmp/nvidia_smi_call_count_{self._instance_id}"

    def setup_localhost_mock_environment(self, config):
        """Set up mock binaries and environment variables on the local host."""
        self.originalEnv = os.environ.copy()
        self.config = config
        self._generate_mock_nvidia_smi_local()
        self._set_output()

    def setup_multinode_mock_environment(self, config):
        """Deploy mock binaries and start host engines on all remote nodes."""
        test_nodes = config['test_nodes']
        test_nodes = [node for node in test_nodes if node.get(
            "ip", "") != "localhost"]
        if not test_nodes:
            raise ValueError("No test nodes specified")

        self._generate_mock_nvidia_smi_local()

        def setup_node(node):
            """Deploy mock scripts, env file, and start host engine on one node."""
            username = node["username"]
            ip = node["ip"]
            ssh_base = ["ssh"] + self.SSH_MUX_OPTS + [f"{username}@{ip}"]
            scp_base = ["scp"] + self.SSH_MUX_OPTS

            setup_res = self._run_with_retry(
                ssh_base + [
                    "d=$(mktemp -d /tmp/mndiag_setup.XXXXXXXX) && "
                    "echo $d && { which mpirun || true; }"],
                retries=self.SSH_MAX_RETRIES, delay=self.SSH_RETRY_DELAY,
                check=True, capture_output=True, text=True)
            lines = setup_res.stdout.strip().split('\n')
            remote_tmp = lines[-2] if len(lines) >= 2 else lines[-1]
            mpirun_path = lines[-1] if len(lines) >= 2 else ""
            if not mpirun_path or mpirun_path.startswith("/tmp/mndiag_setup"):
                test_utils.skip_test(f"Failed to find mpirun on {ip}")

            nvidia_smi_dir = os.path.dirname(self.mockNvidiaSmiPath)
            env_lines = [
                f"export DCGM_MNDIAG_MPIRUN_PATH={mpirun_path} ",
                f"export {self.BINARY_PATH_ENV}={self.mockPath} ",
                f"export PATH={nvidia_smi_dir}:$PATH",
                f"export DCGM_MNDIAG_SUPPORTED_SKUS={node['sku']} ",
                'export NVML_INJECTION_MODE=True',
                f'export NVML_YAML_FILE={node["nvml_injection_yaml_file_path"]}',
            ]

            node_config = config.copy()
            node_config['hostList'] = [ip]
            node_config['gpu_required_info'] = node.get(
                'gpu_required_info', {})
            node_config['gpu_required_error'] = node.get(
                'gpu_required_error', {})
            if 'exit_code' in node:
                node_config['exit_code'] = node['exit_code']

            fd_mock, local_mock = tempfile.mkstemp(
                prefix=f"mock_{ip}_")
            os.close(fd_mock)
            fd_env, local_env = tempfile.mkstemp(prefix=f"mndiag_env_{ip}_")
            os.close(fd_env)
            try:
                self._set_output(config=node_config, output_path=local_mock)
                with open(local_env, "w") as f:
                    f.write("\n".join(env_lines) + "\n")

                self._run_with_retry(
                    scp_base + [local_mock, self.mockNvidiaSmiPath, local_env,
                                f"{username}@{ip}:{remote_tmp}/"],
                    retries=self.SSH_MAX_RETRIES, delay=self.SSH_RETRY_DELAY,
                    check=True)
            finally:
                for p in (local_mock, local_env):
                    if os.path.exists(p):
                        os.remove(p)

            hostengine_path = node["hostengine_path"]
            q_he = shlex.quote(hostengine_path)
            q_mock = shlex.quote(self.mockPath)
            q_nvsmi = shlex.quote(self.mockNvidiaSmiPath)
            q_env = shlex.quote(self.envFile)
            q_tmp = shlex.quote(remote_tmp)
            mock_basename = os.path.basename(local_mock)
            env_basename = os.path.basename(local_env)
            nvsmi_basename = os.path.basename(self.mockNvidiaSmiPath)
            self._run_with_retry(
                ssh_base + [
                    f"set -e; "
                    f"mv {q_tmp}/{shlex.quote(mock_basename)} {q_mock}; "
                    f"mv {q_tmp}/{shlex.quote(nvsmi_basename)} {q_nvsmi}; "
                    f"mv {q_tmp}/{shlex.quote(env_basename)} {q_env}; "
                    f"rm -rf {q_tmp}; "
                    f"chmod +x {q_mock} {q_nvsmi}; "
                    f"test -f {q_mock}; "
                    f"test -x {q_he}"],
                retries=self.SSH_MAX_RETRIES, delay=self.SSH_RETRY_DELAY,
                check=True)

            verbose = os.environ.get("DCGM_MNDIAG_VERBOSE", "0") == "1"
            self._run_with_retry(
                ssh_base + [
                    f". {q_env} && {q_he} --log-level debug"],
                retries=self.SSH_MAX_RETRIES, delay=self.SSH_RETRY_DELAY,
                check=True, stdout=None if verbose else subprocess.DEVNULL)

        try:
            self._parallel_over_nodes(setup_node, test_nodes)
            self.setup_localhost_mock_environment(config)
        except Exception:
            try:
                self.cleanup_multinode_mock_environment(config)
            except Exception as cleanup_err:
                logger.warning(
                    f"Rollback after setup failure also failed: {cleanup_err}")
            raise

    def cleanup_mock_environment(self):
        """Remove local mock binaries and restore the original environment."""
        if hasattr(self, 'originalEnv') and self.originalEnv:
            os.environ.clear()
            os.environ.update(self.originalEnv)

        if os.path.exists(self.mockPath):
            os.remove(self.mockPath)

        self._restore_nvidia_smi()

    def cleanup_multinode_mock_environment(self, config):
        """Stop host engines and remove mock files on all remote nodes."""
        test_nodes = config['test_nodes']
        test_nodes = [node for node in test_nodes if node.get(
            "ip", "") != "localhost"]

        mock_path = self.mockPath
        nvsmi_path = getattr(self, 'mockNvidiaSmiPath', '/tmp/mock_nvidia_smi')
        counter_file = self.counter_file
        env_file = self.envFile

        def cleanup_node(node):
            """Stop host engine and remove mock artifacts on one node."""
            username = node["username"]
            ip = node["ip"]
            hostengine_path = node["hostengine_path"]
            ssh_base = ["ssh"] + self.SSH_MUX_OPTS + [f"{username}@{ip}"]

            envs = (
                'DCGM_MNDIAG_MPIRUN_PATH',
                self.BINARY_PATH_ENV,
                'DCGM_MNDIAG_SUPPORTED_SKUS',
                'NVML_INJECTION_MODE',
                'NVML_YAML_FILE')
            sed_command = ';'.join((f'/{env}/d' for env in envs))

            q_he = shlex.quote(hostengine_path)
            q_mock = shlex.quote(mock_path)
            q_nvsmi = shlex.quote(nvsmi_path)
            q_counter = shlex.quote(counter_file)
            q_env = shlex.quote(env_file)
            last_err = None
            for attempt in range(1, self.SSH_MAX_RETRIES + 1):
                result = subprocess.run(
                    ssh_base + [
                        f"rc=0; "
                        f"{q_he} -t || rc=$?; "
                        f"for _i in $(seq 1 20); do pgrep -x nv-hostengine >/dev/null 2>&1 || break; sleep 0.5; done; "
                        f"rm -f {q_mock} {q_nvsmi} {q_counter} || rc=$?; "
                        f'if [ -f {q_env} ]; then sed -i "{sed_command}" {q_env} || rc=$?; fi; '
                        f"exit $rc"],
                    capture_output=True, text=True)
                if result.returncode == 0:
                    last_err = None
                    break
                last_err = (result.returncode, result.stderr.strip())
                if attempt < self.SSH_MAX_RETRIES:
                    time.sleep(self.SSH_RETRY_DELAY * attempt)
            if last_err is not None:
                raise RuntimeError(
                    f"Cleanup failed on {username}@{ip} "
                    f"(exit {last_err[0]}): "
                    f"stderr={last_err[1]!r}")

        try:
            self._parallel_over_nodes(cleanup_node, test_nodes)
        finally:
            self.cleanup_mock_environment()

    def get_error_entities(self):
        """Return a dict mapping 'hostname:entityId' to expected error messages."""
        error_entities = {}
        for host in self.config.get("hostList", []):
            for entityId, msgs in self.config.get(
                    "gpu_required_error", {}).items():
                if len(msgs) > 0:
                    key = f"{host}:{entityId}"
                    error_entities[key] = msgs

        for node in self.config.get("test_nodes", []):
            for entityId, msgs in node.get("gpu_required_error", {}).items():
                if len(msgs) > 0:
                    key = f"{node['ip']}:{entityId}"
                    error_entities[key] = msgs

        return error_entities

    def get_driver_versions(self, config):
        """Query nvidia-smi on the head node and all test nodes for driver versions."""
        driver_versions = set()

        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", "0", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                driver_versions.add(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get driver version for head node: {e}")

        for node in config.get("test_nodes", []):
            try:
                username = node["username"]
                ip = node["ip"]
                result = subprocess.run(
                    ["ssh",
                     "-o", "ControlPath=none",
                     "-o", f"ConnectTimeout={self.DRIVER_VERSION_QUERY_TIMEOUT_SECONDS}",
                     f"{username}@{ip}",
                     "nvidia-smi -i 0 --query-gpu=driver_version "
                     "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=self.DRIVER_VERSION_QUERY_TIMEOUT_SECONDS)
                if result.returncode == 0 and result.stdout.strip():
                    driver_versions.add(result.stdout.strip())
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Timed out querying driver version on node {ip} after "
                    f"{self.DRIVER_VERSION_QUERY_TIMEOUT_SECONDS} seconds")
            except Exception as e:
                logger.error(
                    f"Failed to get driver version for node {ip}: {e}")

        return driver_versions

    # ------------------------------------------------------------------
    # Subclass interface: override these methods
    # ------------------------------------------------------------------

    def _get_bash_script_template(self):
        """Return the bash script template. Must be overridden."""
        raise NotImplementedError

    def _create_mock_script(self, lines, output_path=None):
        """Create the mock shell script. Must be overridden."""
        raise NotImplementedError

    def _set_output(self, config=None, output_path=None):
        """Generate mock output and write the script. Must be overridden."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _generate_mock_nvidia_smi_local(self):
        """Create temporary mock nvidia-smi script."""
        mock_script_content = f'''#!/bin/bash
COUNTER_FILE="{self.counter_file}"

if [ ! -f "$COUNTER_FILE" ]; then
    echo "0" > "$COUNTER_FILE"
fi

if [[ "$1" == "--query-compute-apps=pid,process_name" && "$2" == "--format=csv,noheader,nounits" ]]; then
    CALL_COUNT=$(($(cat "$COUNTER_FILE") + 1))
    echo "$CALL_COUNT" > "$COUNTER_FILE"
    if [ "$CALL_COUNT" -eq 1 ]; then
        exit 0
    fi
    echo "3505755, {self.mockPath}"
    exit 0
fi

REAL_NVIDIA_SMI=$(PATH="{os.environ.get('PATH', '')}" which nvidia-smi 2>/dev/null | grep -v "{self.mockNvidiaSmiPath}" | head -n1)
if [[ -n "$REAL_NVIDIA_SMI" ]]; then
    exec "$REAL_NVIDIA_SMI" "$@"
else
    echo "nvidia-smi: command not found" >&2
    exit 127
fi
'''
        with open(self.mockNvidiaSmiPath, 'w') as f:
            f.write(mock_script_content)

        os.chmod(self.mockNvidiaSmiPath, stat.S_IRWXU | stat.S_IRGRP |
                 stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    def _restore_nvidia_smi(self):
        """Restore original nvidia-smi environment."""
        if os.path.exists(self.counter_file):
            os.remove(self.counter_file)

        if hasattr(self, 'mockNvidiaSmiPath'):
            if os.path.exists(self.mockNvidiaSmiPath):
                os.remove(self.mockNvidiaSmiPath)

            if 'PATH' in os.environ:
                mock_dir = os.path.dirname(self.mockNvidiaSmiPath)
                paths = os.environ['PATH'].split(':')
                paths = [p for p in paths if p != mock_dir]
                os.environ['PATH'] = ':'.join(paths)

            delattr(self, 'mockNvidiaSmiPath')

    def _parallel_over_nodes(self, fn, nodes, max_workers=None):
        """Run fn(node) in parallel across nodes."""
        if not nodes:
            return
        if max_workers is None:
            max_workers = self.MAX_PARALLEL_WORKERS
        workers = min(len(nodes), max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fn, n): n for n in nodes}
            errors = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except test_utils.TestSkipped:
                    raise
                except Exception as e:
                    errors.append((futures[future], e))
            if errors:
                msgs = [f"  {n.get('ip', n)}: {e}" for n, e in errors]
                raise RuntimeError(
                    f"{len(errors)} node(s) failed:\n" + "\n".join(msgs))
