#!/usr/bin/env bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

DIR="$(dirname $(realpath $0))"
source "$DIR/_common.sh"

run_in_dcgmbuild "$@"

LONG_OPTS=help,deb:,rpm:

! PARSED=$(getopt --options= --longoptions=${LONG_OPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments"
    exit 1
fi

eval set -- "${PARSED}"

function usage() {
   echo "$0 [--deb <debfile>|--rpm <rpmfile>]"
}

function verify_deb() {
    DEB="$1"
    CONTENTS="$(normalize_deb_name "$DEB")"
    diff <(sort "$CONTENTS") <(list_deb_contents "$1" | sort)
}

function verify_rpm() {
    RPM="$1"
    CONTENTS="$(normalize_rpm_name "$RPM")"
    diff <(sort "$CONTENTS") <(list_rpm_contents "$1" | sort)
}

while true; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --deb)
            verify_deb "$2"
            echo "Passed: $2"
            shift 2
            ;;
        --rpm)
            verify_rpm "$2"
            echo "Passed: $2"
            shift 2
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

