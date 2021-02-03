#!/usr/bin/env bash

## Do NOT include this script if you want to build something for the host
## This script changes build toolset to cross-build to the specified target

set -ex -o nounset

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
