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

# This is not a script. It's shared functionality loaded by the
# verify_package_contents scripts

DCGM_DIR="${DCGM_DIR:-$(realpath "$SCRIPTDIR/../..")}"

SCRIPTPATH_RELATIVE=$(realpath "--relative-to=$DCGM_DIR" "$SCRIPTPATH")

function run_in_dcgmbuild() {
    if ! [[ -f "$DCGM_DIR/intodocker.sh" ]]
    then
        >&2 echo "Could not find intodocker.sh. Make sure DCGM_DIR is properly configured"
    fi

    if [[ "${DCGM_BUILD_INSIDE_DOCKER:-0}" -ne 1 ]]
    then
        "$DCGM_DIR/intodocker.sh" -- bash -c "$SCRIPTPATH_RELATIVE $*"
        exit $?
    fi
}

function template_deb() {
    local filename=$1

    local package=${filename%%_*}
    local prefix=$(echo $package | sed -E 's/-[0-9]+//')
    local suffix=${filename##*_}.txt

    echo ${prefix}_$suffix
} 

function template_rpm() {
    local filename=$1
    local components=$(echo $filename | sed -E 's/(.*)-[0-9]+([.][0-9]+){2,3}(~[0-9]+)?-[0-9]+[.](.*)/\1:\4/')

    local package=${components%%:*}
    local prefix=$(echo $package | sed -E 's/-[0-9]+//')
    local suffix=${components##*:}.txt

    echo ${prefix}_$suffix
} 
