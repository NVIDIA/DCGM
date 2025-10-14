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
SCRIPTDIR="$(dirname "$SCRIPTPATH")"

source "$SCRIPTDIR/_common.sh"

run_in_dcgmbuild "$@"

function generalize() {
    local package_name=$1
    local package_name_regex=$(echo $package_name | sed -E 's/(.)/[\1]/g')
    local package_version=${2##*:}
    local package_version_regex=$(echo $package_version | sed -E 's/(.)/[\1]/g')
    local base_package_version=${package_version%%-*}
    local base_package_version_regex=$(echo $base_package_version | sed -E 's/(.)/[\1]/g')
    local software_version=${base_package_version%%~*}
    local software_version_regex=$(echo $software_version | sed -E 's/(.)/[\1]/g')
    local major_version=${software_version%%.*}
    local substitution=$(echo $package_name | sed "s/$major_version/<MAJORVERSION>/")

    sed -E "
        s/$package_version_regex/<PACKAGEVERSION>/g;
        s/$base_package_version_regex/<BASEPACKAGEVERSION>/g;
        s/$software_version_regex/<SOFTWAREVERSION>/g;
        s/$package_name_regex/$substitution/;
        s/[.]$major_version$/.<MAJORVERSION>/;
        s/datacenter-gpu-manager-$major_version/datacenter-gpu-manager-<MAJORVERSION>/g" $3
}

function write_template_deb() {
    local path="$1"
    local filename=$(basename "$path")
    local package=$(dpkg-deb --field "$path" Package)
    local version=$(dpkg-deb --field "$path" Version)

    generalize $package $version <(dpkg --contents "$path" | awk '{print $6}' | sort) \
    > "$SCRIPTDIR/$(template_deb $filename)"
}

function write_template_rpm() {
    local path="$1"
    local filename=$(basename "$path")
    local package=$(rpm --query --queryformat '%{NAME}' "$path")
    local version=$(rpm --query --queryformat '%{EPOCH}:%{VERSION}-%{RELEASE}' "$path")

    generalize $package $version <(rpm --query --list --package "$path" | sed -E '/.*\/[.]build-id.*/d' | sort) \
    > "$SCRIPTDIR/$(template_rpm $filename)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        *.deb)
            write_template_deb $1
            >&2 echo "Generated contents for: $1"
            shift 1
            ;;
        *.rpm)
            write_template_rpm $1
            >&2 echo "Generated contents for: $1"
            shift 1
            ;;
        *)
            echo "Unrecognized argument: $1"
            exit 1
    esac
done
