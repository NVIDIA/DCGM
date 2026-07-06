#!/bin/sh
#
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

# Host prerequisites
#   - Docker (with BuildKit support): for `docker buildx bake` to build the cluster image
#   - Docker Compose v2             : for `docker compose up/down/exec/ps` cluster lifecycle
#   - jq                            : generates dcgm_mndiag_test_config.json from NUM_WORKERS
#   - moreutils (optional)          : provides `ts` for the -t timestamp flag

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure containers and config file are always cleaned up
trap 'docker compose down -v 2>/dev/null || true; rm -f dcgm_mndiag_test_config.json' EXIT

# Default test filter and host-side log output path
INTERACTIVE=0
TEST_FILTER="test_dcgm_mndiag"
BASE_IMAGE="ubuntu:24.04"
SKIP_BUILD=0
NUM_WORKERS=3
TIMESTAMP=0

# Default is architecture-specific app directory inside test tarball where core DCGM binaries are
if [ -z "$DCGM_INSTALL_DIR" ]; then
    if [ -d "../apps/amd64" ]; then
        DCGM_INSTALL_DIR="../apps/amd64"
    elif [ -d "../apps/aarch64" ]; then
        DCGM_INSTALL_DIR="../apps/aarch64"
    else
        echo "Error: Could not find DCGM binaries." >&2
        exit 1
    fi
fi

usage() {
    echo "Usage: $0 [-i] [-f test_filter] [-b base_image] [-s] [-t] [-n num_workers]"
    echo "  -i    Interactive mode (drops into head node shell without running tests)"
    echo "  -f    Regex filter for test selection (default: test_dcgm_mndiag)"
    echo "  -b    Base image: ubuntu:24.04 or manually set base image"
    echo "  -s    Skip image build (use a pre-loaded mndiag:latest)"
    echo "  -t    Prefix test output with timestamps (requires moreutils on host)"
    echo "  -n    Number of worker containers to spin up (default: 3 workers + 1 head node)"
    echo "  -h    Show this help message"
    echo ""
    echo "Environment Variable:"
    echo "  DCGM_INSTALL_DIR    Path to the DCGM installation directory (default: ../apps/{amd64,aarch64})"
    exit 1
}

# Parse optional flags (e.g. -f to run a specific test)
while getopts "if:b:stn:h" opt; do
    case $opt in
        i) INTERACTIVE=1 ;;
        f) TEST_FILTER="$OPTARG" ;;
        b) BASE_IMAGE="$OPTARG" ;;
        s) SKIP_BUILD=1 ;;
        t) TIMESTAMP=1 ;;
        n) NUM_WORKERS="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Validate that jq is available, required for generating test config
if ! command -v jq >/dev/null 2>&1; then
    echo "Error: 'jq' is required to generate the test config" >&2
    exit 1
fi

# Validate that ts command is available if timestamp flag is set
if [ "$TIMESTAMP" -eq 1 ] && ! command -v ts >/dev/null 2>&1; then
    echo "Error: -t flag requires 'ts' command from moreutils package" >&2
    exit 1
fi

# Generate dcgm_mndiag_test_config.json from NUM_WORKERS so it always matches the cluster
generate_test_config() {
    jq -n --argjson n "$NUM_WORKERS" '
        [range(1; $n + 1) | {
            username: "root",
            ip: "mndiag-worker-\(.)",
            sku: "2941",
            hostengine_path: "/usr/bin/nv-hostengine",
            nvml_injection_yaml_file_path: "/workspace/testing/SKUs/GB200.yaml"
        }] | {test_nodes: .}'
}
generate_test_config > dcgm_mndiag_test_config.json

# Tear down any existing cluster to start clean
docker compose down -v 2>/dev/null || true

# Build the container image and start virtual cluster
export BASE_IMAGE DCGM_INSTALL_DIR
if [ "$SKIP_BUILD" -eq 0 ]; then
    docker buildx bake --load
fi
docker compose up -d --scale worker=$NUM_WORKERS

# Verify all containers are running
docker compose ps

# Pre-warm SSH host keys for workers so test output doesn't print SSH warnings
docker exec head-node bash -c \
    "ssh-keyscan -t ed25519 \$(seq -f 'mndiag-worker-%g' 1 $NUM_WORKERS) >> ~/.ssh/known_hosts 2>/dev/null" || true

# Interactive mode block if i flag enabled
if [ "$INTERACTIVE" -eq 1 ]; then
    {
        echo ""
        echo "Cluster is live in interactive mode."
        echo "Dropping you into the head node. Type 'exit' to leave."
        echo ""
    } >&2

    docker exec -it head-node bash
    exit $?
fi

# Non-interactive mode
{
    echo ""
    echo "Running tests with filter: $TEST_FILTER"
    echo ""
} >&2

# Run tests on the head node, capturing exit code so log collection always runs
# We write the return code to a tmp file to avoid losing it when running with -t flag
run_tests() {
    docker exec -w /workspace/testing head-node python3 -u main.py -f "$TEST_FILTER"
    echo $? > /tmp/test_return_code
}

# When -t is set, pipe through `ts` to prefix each line with a timestamp
set +e
if [ "$TIMESTAMP" -eq 0 ]; then
    run_tests
else
    run_tests 2>&1 | ts '[%Y-%m-%d %H:%M:%S]'
fi
TEST_RC=$(cat /tmp/test_return_code)
rm -f /tmp/test_return_code
set -e

# Copy head node logs to the standard test output location on the host
mkdir -p "../_out_runLogs/"
docker cp head-node:/workspace/testing/_out_runLogs/. "../_out_runLogs/" 2>/dev/null || true

echo ""
echo "Logs saved to: ../_out_runLogs/"

exit $TEST_RC
