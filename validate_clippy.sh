#!/usr/bin/env bash

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

set -e -o pipefail -o nounset

DIR=$(dirname "$(realpath "${0}")")

function usage() {
    cat <<-EOF >&2
	Usage: ${0} [options]
	    Where options are:
	        -h --help       : This information

EOF
}

if ! PARSED=$(getopt --options=h --longoptions=help --name "$0" -- "$@"); then
    echo "Failed to parse arguments" >&2
    usage
    exit 1
fi

eval set -- "$PARSED"

while true; do
    case "$1" in
        -h|--help)
            echo 'Clippy validation script for DCGM Rust workspace'
            usage
            echo 'This script will run cargo clippy when staged Rust workspace files changed.'
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unrecognized argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

git diff --cached --diff-filter=d --quiet \
         -- 'Cargo.toml' 'Cargo.lock' '.cargo/config.toml' 'rust/**/*.rs' 'rust/**/*.toml' \
&& exit 0


if [[ ${DCGM_BUILD_INSIDE_DOCKER:-0} -eq 0 ]]; then
    "${DIR}/intodocker.sh" --env CARGO_HOME=_out/cargo-home -- "${0}" "$@"
    exit $?
fi

export CARGO_HOME="${CARGO_HOME:-${DIR}/_out/cargo-home}"
mkdir --parents "${CARGO_HOME}"

cargo clippy \
    --manifest-path "${DIR}/Cargo.toml" \
    --workspace \
    --locked \
    --all-targets \
    --all-features \
    --no-deps \
    -- \
    -D warnings
