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

# Ensure containers, config file, and sysctl overrides are always cleaned up
cleanup() {
    docker compose down -v 2>/dev/null || true
    rm -f dcgm_mndiag_test_config.json
    if [ -n "$SYSCTL_RESTORED" ]; then
        return
    fi
    if [ -n "$ORIG_GC1" ]; then
        sudo sysctl -qw \
            "net.ipv4.neigh.default.gc_thresh1=$ORIG_GC1" \
            "net.ipv4.neigh.default.gc_thresh2=$ORIG_GC2" \
            "net.ipv4.neigh.default.gc_thresh3=$ORIG_GC3" 2>/dev/null || true
        SYSCTL_RESTORED=1
        echo "Restored ARP cache limits to gc_thresh1=$ORIG_GC1 gc_thresh2=$ORIG_GC2 gc_thresh3=$ORIG_GC3" >&2
    fi
}
trap cleanup EXIT INT TERM

# Default test filter and host-side log output path
INTERACTIVE=0
TEST_FILTER="test_dcgm_mndiag"
BASE_IMAGE=""
OPENMPI_MAJOR=4
SKIP_BUILD=0
NUM_WORKERS=3
TIMESTAMP=0
FORCE_LOGGING=0

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
    echo "Usage: $0 [-i] [-f test_filter] [-b base_image] [-m openmpi_major] [-s] [-t] [-l] [-n num_workers]"
    echo "  -i    Interactive mode (drops into head node shell without running tests)"
    echo "  -f    Regex filter for test selection (default: test_dcgm_mndiag)"
    echo "  -b    Override the base image selected for the OpenMPI environment"
    echo "  -m    OpenMPI major version environment: 4 (Ubuntu 24.04) or 5 (Debian 13; default: 4)"
    echo "  -s    Skip image build (use the matching pre-loaded mndiag:openmpi{4,5} image)"
    echo "  -t    Prefix test output with timestamps (requires moreutils on host)"
    echo "  -l    Enable DEBUG-level hostengine logging (--force-logging)"
    echo "  -n    Number of worker containers to spin up (default: 3 workers + 1 head node)"
    echo "  -h    Show this help message"
    echo ""
    echo "Environment Variable:"
    echo "  DCGM_INSTALL_DIR    Path to the DCGM installation directory (default: ../apps/{amd64,aarch64})"
    exit 1
}

# Parse optional flags (e.g. -f to run a specific test)
while getopts "if:b:m:stln:h" opt; do
    case $opt in
        i) INTERACTIVE=1 ;;
        f) TEST_FILTER="$OPTARG" ;;
        b) BASE_IMAGE="$OPTARG" ;;
        m) OPENMPI_MAJOR="$OPTARG" ;;
        s) SKIP_BUILD=1 ;;
        t) TIMESTAMP=1 ;;
        l) FORCE_LOGGING=1 ;;
        n) NUM_WORKERS="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

case "$OPENMPI_MAJOR" in
    4)
        DEFAULT_BASE_IMAGE="ubuntu:24.04"
        ;;
    5)
        DEFAULT_BASE_IMAGE="debian:13"
        ;;
    *)
        echo "Error: OpenMPI major version must be 4 or 5." >&2
        exit 1
        ;;
esac

if [ -z "$BASE_IMAGE" ]; then
    BASE_IMAGE="$DEFAULT_BASE_IMAGE"
fi

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

# For large clusters, raise ARP cache limits so the kernel doesn't evict
# neighbor entries while containers are in use. The EXIT trap restores originals.
ARP_THRESH=256
if [ "$NUM_WORKERS" -gt "$ARP_THRESH" ]; then
    ORIG_GC1=$(sysctl -n net.ipv4.neigh.default.gc_thresh1)
    ORIG_GC2=$(sysctl -n net.ipv4.neigh.default.gc_thresh2)
    ORIG_GC3=$(sysctl -n net.ipv4.neigh.default.gc_thresh3)

    NEED_GC1=$((NUM_WORKERS * 4))
    NEED_GC2=$((NUM_WORKERS * 8))
    NEED_GC3=$((NUM_WORKERS * 16))

    # Only raise; never lower below current values
    [ "$NEED_GC1" -le "$ORIG_GC1" ] && NEED_GC1="$ORIG_GC1"
    [ "$NEED_GC2" -le "$ORIG_GC2" ] && NEED_GC2="$ORIG_GC2"
    [ "$NEED_GC3" -le "$ORIG_GC3" ] && NEED_GC3="$ORIG_GC3"

    if [ "$NEED_GC3" -gt "$ORIG_GC3" ] || [ "$NEED_GC2" -gt "$ORIG_GC2" ] || [ "$NEED_GC1" -gt "$ORIG_GC1" ]; then
        echo "Raising ARP cache limits for $NUM_WORKERS workers (gc_thresh: $NEED_GC1/$NEED_GC2/$NEED_GC3)..." >&2
        sudo sysctl -qw \
            "net.ipv4.neigh.default.gc_thresh1=$NEED_GC1" \
            "net.ipv4.neigh.default.gc_thresh2=$NEED_GC2" \
            "net.ipv4.neigh.default.gc_thresh3=$NEED_GC3"
    fi
fi

# Tear down any existing cluster to start clean
docker compose down -v 2>/dev/null || true

# Build the container image and start virtual cluster
export BASE_IMAGE DCGM_INSTALL_DIR OPENMPI_MAJOR
if [ "$SKIP_BUILD" -eq 0 ]; then
    docker buildx bake --load
fi
docker compose up -d --scale worker=$NUM_WORKERS

# The test command below passes "-u nobody" so the non-root caller regression
# tests can switch users. At startup, the Python framework validates that this
# user can create a probe file in its test root; the bind-mounted install tree
# is normally 0755 and would fail that check before any tests run. CI prepares
# the extracted tree on the host, while a local rootful Docker daemon can grant
# access through the bind mount.
if ! docker exec -u nobody head-node test -w /workspace/testing; then
    echo "Granting nobody write access to the mounted test directory..." >&2
    docker exec head-node chmod a+rwx /workspace/testing
fi
if ! docker exec -u nobody head-node test -w /workspace/testing; then
    echo "Error: User 'nobody' cannot write to the mounted test directory." >&2
    echo "For rootless Docker, run 'chmod -R a+rwX ../' before this script." >&2
    exit 1
fi

# Wait for all worker containers to be running
echo "Waiting for $NUM_WORKERS worker containers to start..." >&2
READY_TIMEOUT=120
READY_ELAPSED=0
while true; do
    RUNNING=$(docker compose ps --status running --format json | grep -c '"worker"' || true)
    if [ "$RUNNING" -ge "$NUM_WORKERS" ]; then
        break
    fi
    if [ "$READY_ELAPSED" -ge "$READY_TIMEOUT" ]; then
        echo "Error: Only $RUNNING/$NUM_WORKERS workers running after ${READY_TIMEOUT}s" >&2
        docker compose ps >&2
        exit 1
    fi
    sleep 2
    READY_ELAPSED=$((READY_ELAPSED + 2))
done
echo "All $NUM_WORKERS workers running (${READY_ELAPSED}s)" >&2

# Verify SSH connectivity to every worker from the head node.
# ssh-keyscan doubles as both a host-key warmup and a reachability probe.
# Workers whose sshd hasn't started yet are retried up to SSH_PROBE_RETRIES times.
echo "Probing SSH connectivity to $NUM_WORKERS workers..." >&2
SSH_PROBE_RETRIES=5
SSH_PROBE_DELAY=5

docker exec head-node bash -c "
set -e
rm -f ~/.ssh/known_hosts
BATCH=50
RETRIES=$SSH_PROBE_RETRIES
DELAY=$SSH_PROBE_DELAY
TOTAL=$NUM_WORKERS

remaining=\$(seq 1 \$TOTAL)
for attempt in \$(seq 1 \$RETRIES); do
    # Run ssh-keyscan in batches over the remaining workers
    for host in \$remaining; do
        printf 'mndiag-worker-%d\n' \"\$host\"
    done | xargs -n \$BATCH sh -c '
        ssh-keyscan -T 10 -t ed25519 \"\$@\" >> ~/.ssh/known_hosts 2>/dev/null
    ' _

    # Find workers still missing from known_hosts
    failed=\"\"
    for w in \$remaining; do
        if ! grep -q \"mndiag-worker-\${w} \" ~/.ssh/known_hosts 2>/dev/null; then
            failed=\"\$failed \$w\"
        fi
    done
    failed=\$(echo \$failed | xargs)

    if [ -z \"\$failed\" ]; then
        echo \"All \$TOTAL workers SSH-ready (probe attempt \$attempt)\" >&2
        break
    fi

    fail_count=\$(echo \$failed | wc -w)
    if [ \"\$attempt\" -eq \"\$RETRIES\" ]; then
        echo \"Error: \$fail_count workers not SSH-reachable after \$RETRIES attempts\" >&2
        echo \"Failed workers: \$failed\" >&2
        exit 1
    fi

    echo \"  Attempt \$attempt: \$fail_count workers not yet reachable, retrying in \${DELAY}s...\" >&2
    remaining=\"\$failed\"
    sleep \$DELAY
done
"

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
    LOGGING_FLAG=""
    if [ "$FORCE_LOGGING" -eq 1 ]; then
        LOGGING_FLAG="--force-logging"
    fi
    docker exec -w /workspace/testing head-node python3 -u main.py $LOGGING_FLAG -u nobody -f "$TEST_FILTER" --fail-on-test-failure
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
