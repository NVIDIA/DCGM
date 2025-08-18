#!/usr/bin/env bash

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

set -euo pipefail

SCRIPTPATH="$(realpath "$(cat /proc/$$/cmdline | cut --delimiter="" --fields=2)")"
export SCRIPTDIR="$(dirname "$SCRIPTPATH")"

source "$SCRIPTDIR/_common.sh"

LONG_OPTS=help,no-build
! PARSED=$(getopt --options= --longoptions=${LONG_OPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments. Consult with $0 --help"
    exit 1
fi
eval set -- "${PARSED}"

function usage() {
cat <<EOF
Usage:
$0 [--no-build]

Options:
  - --no-build: skip deleting _out and building packages

This script
  - deletes _out
  - creates a fresh DCGM build for all platforms
  - generates package lists for all platforms (both deb and rpm)
EOF
}

NOBUILD=0

while true; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --no-build)
            NOBUILD=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unrecognized argument: $1"
            exit 1
    esac
done

cd "$DCGM_DIR"

if [[ "$NOBUILD" -eq 0 ]]; then
    read -p "This will delete the contents of _out. Do you want to proceed? [y/N]? " -n 1 -r
    echo

    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Did not receive user confirmation. Aborting"
        exit 1
    fi

    echo 'Deleting _out'
    rm -rf "_out"

    if [[ -z "${BUILD_NUMBER:-}" ]]; then
        export BUILD_NUMBER=999999
    fi

    xargs --delimiter=' ' \
          --max-args=1 \
          --max-procs=2 \
          -- \
          ./build.sh --release --deb --rpm --no-test --no-install --arch \
          <<< "amd64 aarch64"
fi

find _out \
    -path '*/build/*' -prune , \
    \( -name '*.rpm' -o -name '*.deb' \) \
    -exec scripts/verify_package_contents/generate_package_contents.sh {} +
