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

set -o errexit -o pipefail -o noclobber -o nounset

if [[ ${DEBUG_BUILD_SCRIPT:-0} -eq 1 ]]
then
    PS4='$LINENO: ' # to see line numbers
    set -xv
fi

# Architecture suffix will be appended to the image name
# -x86_64 for x86_64
# -aarch64 for aarch64
DCGM_DOCKER_IMAGE=${DCGM_DOCKER_IMAGE:-dcgmbuild}

function usage() {
    echo "Handling DCGM docker build image
Usage: ${0} [options] [command]
    Where options are:
        -a | --arch NAME       : Architecture to build for (amd64|x86_64, aarch64|arm64)
        -e | --env VAR=VALUE   : Environment variables for the launched container
        -n | --name            : Print docker image name and exit
        -h | --help            : Print this help and exit

    Command will be run inside the docker container
"
}

LONG_OPTS=name,arch:,env:,help
SHORT_OPTS=n,a:,e:,h

! PARSED=$(getopt --options=${SHORT_OPTS} --longoptions=${LONG_OPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    >&2 echo "Failed to parse arguments"
    exit 1
fi

eval set -- "${PARSED}"

PRINT_NAME_ONLY=0
TARGET_ARCH=$(uname -m) # Default to host architecture

declare -a docker_args

while true; do
    case $1 in
    -a | --arch)
        TARGET_ARCH=$2
        shift 2
        ;;
    -e | --env)
        docker_args+=(--env "$2")
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    -n | --name)
        PRINT_NAME_ONLY=1
        shift
        ;;
    --)
        shift
        break
        ;;
    *) ;;
    esac
done

if [[ ${DCGM_BUILD_INSIDE_DOCKER:-0} -ne 0 ]]; then
    exit
fi

image=${DCGM_DOCKER_IMAGE}

case "${TARGET_ARCH,,}" in
amd64 | x86_64 | x64)
    TARGET_ARCH="x86_64"
    image="${image}-x86_64"
    ;;
aarch64 | arm64 | arm)
    TARGET_ARCH="aarch64"
    image="${image}-aarch64"
    ;;
*)
    >&2 echo "Unknown architecture $TARGET_ARCH"

    exit 1
    ;;
esac


if [[ -z "$image" ]]; then
    >&2 echo "Error: Cannot find a proper docker image to use"
    exit 1
fi

if [[ $PRINT_NAME_ONLY -eq 1 ]]; then
    echo $image
    exit
fi

SCRIPTPATH="$(realpath "$(cat /proc/$$/cmdline | cut --delimiter="" --fields=2)")"
DIR="$(dirname "$SCRIPTPATH")"
PROJECT="$(basename "$DIR")"
REMOTE_DIR="/workspaces/$PROJECT"

docker_args+=(
    --net=host
    --rm
    --volume "$DIR:$REMOTE_DIR"
    --workdir "$REMOTE_DIR")

if [[ -t 0 ]]; then
    docker_args+=(--interactive --tty)
fi

if [[ -f "$DIR/.git" ]]; then
    GITDIR="$(git rev-parse --path-format=absolute --git-common-dir)"

    if [[ "$REMOTE_DIR" != "$DIR" ]]; then
        docker_args+=( --mount "source=$DIR,target=$DIR,type=bind" )
    fi

    if [[ "$REMOTE_DIR" != "$(dirname $GITDIR)" ]]; then
        docker_args+=( --mount "source=$GITDIR,target=$GITDIR,type=bind" )
    fi
fi

if ! (docker info -f '{{println .SecurityOptions}}' | grep rootless); then
    docker_args+=(--user $(id -u):$(id -g))
fi

docker run "${docker_args[@]}" $image "$@"
