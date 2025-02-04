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

SCRIPTPATH=$(realpath $(cat /proc/$$/cmdline | cut --delimiter="" --fields=2))
export SCRIPTDIR=$(dirname $SCRIPTPATH)

source $SCRIPTDIR/_common.sh

run_in_dcgmbuild "$@"

function usage() {
   echo "$0 [[<deb file>] [<rpm file>]]..."
}

function specialize() {
    local filepath="$1"
    local package_version=$2
    local software_version
    software_version=${package_version%%~*}
    software_version=${software_version##*:}

    local major_version=${software_version%%.*}

    sed "
        s/<PACKAGEVERSION>/$package_version/g;
        s/<SOFTWAREVERSION>/$software_version/g;
        s/<MAJORVERSION>/$major_version/g;" "$filepath"
}

function verify() {
    local path="$1"
    local filename=$(basename "$path")
    local version

    case $filename in
        *.deb)
            package_version=$(dpkg-deb --field "$path" Version)
            diff \
            <(specialize "$SCRIPTDIR/$(template_deb $filename)" $package_version) \
            <(dpkg --contents "$path" | awk '{print $6}' | sort)
            ;;
        *.rpm)
            package_version=$(rpm --query --queryformat '%{VERSION}' "$path")
            diff \
            <(specialize "$SCRIPTDIR/$(template_rpm $filename)" $package_version) \
            <(rpm --query --list --package "$path" | sed -E '/.*\/[.]build-id.*/d' | sort)
            ;;
        *)
            >&2 echo "Unrecognized argument: $1"
            return 1
    esac

    if [[ $? -eq 0 ]]
    then
        >&2 echo "Passed: $path"
    else
        >&2 echo "Failed: $path"
        return 1
    fi
}

if [[ $# -eq 0 ]] || [[ $1 == '-h' ]] || [[ $1 == '--help' ]]
then
    usage
    exit 0
fi

export SCRIPTDIR
export -f verify specialize template_deb template_rpm

printf "%s\n" $@ | xargs --max-args=1 bash -c 'verify "$@"' _ 
