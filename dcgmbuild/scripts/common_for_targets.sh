#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#

## Do NOT include this script if you want to build something for the host
## This script changes build toolset to cross-build to the specified target

set -ex -o nounset -o pipefail

export CROSS_PREFIX=${CROSS_PREFIX:-/opt/cross}
export TARGET=${TARGET:-x86_64-linux-gnu}
export PREFIX=${PREFIX:-${CROSS_PREFIX}/${TARGET}}
export CC=${CROSS_PREFIX}/bin/${TARGET}-gcc
export CPP=${CROSS_PREFIX}/bin/${TARGET}-cpp
export CXX=${CROSS_PREFIX}/bin/${TARGET}-g++
export LD=${CROSS_PREFIX}/bin/${TARGET}-ld
export AS=${CROSS_PREFIX}/bin/${TARGET}-as

hide_output() {
  set +x
  bash -c "while true; do sleep 30; echo \$(date) - building ...; done" &
  PING_LOOP_PID=$!

  on_err="
kill $PING_LOOP_PID
echo ERROR: An error was encountered with the build.
cat /tmp/build.log
exit 1
"
  trap "$on_err" ERR SIGINT SIGTERM SIGKILL
  $@ &> /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  set -x
}

die() {
    echo "$@" >&2
    exit 1
}

# Download a file
# Params: (url, output_file)
download_url(){
    local url=$1; shift
    local outputfile=$1; shift

    #
    # Cannot use curl until 7.68: https://github.com/curl/curl/pull/4623
    #
    # curl -fL "${url}" -o "${outputfile}"
    #
    wget -O "${outputfile}" -nv -t10 --unlink --no-check-certificate "${url}"
}
