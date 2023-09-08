#!/usr/bin/env bash

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

set -e -o pipefail -o nounset

DCGM_DOCKER_IMAGE=${DCGM_DOCKER_IMAGE:-dcgmbuild}

ABSPATH=`realpath ${0}`
DIR=$(dirname ${ABSPATH})
PROJECT=$(basename ${DIR})
REMOTE_DIR=/workspaces/${PROJECT}
MOUNT_OPTIONS=""

function usage() {
    echo "Handling DCGM docker build image
Usage: ${0} [options] [command]
    Where options are:
        -a | --arch            : Architecture to build for (amd64|x86_64, aarch64|arm64, ppc64le|powerpc64le|ppc)
        -n | --name            : Print docker image name and exit
        -h | --help            : Print this help and exit

    Command will be run inside the docker container
"
}

LONG_OPTS=name,arch:
SHORT_OPTS=n,a:

! PARSED=$(getopt --options=${SHORT_OPTS} --longoptions=${LONG_OPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments"
    exit 1
fi

eval set -- "${PARSED}"

TARGET_ARCH=$(uname -m) # Default to host architecture
PRINT_NAME_ONLY=0
while true; do
    case "${1}" in
        -h | --help)
            usage
            exit 0
            ;;
        -n | --name)
            PRINT_NAME_ONLY=1
            shift
            ;;
        -a | --arch)
            TARGET_ARCH="${2}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
    esac
done

if [[ -t 0 ]]; then
    DOCKER_ARGS=-it
fi

case "${TARGET_ARCH}" in
    amd64|x86_64|x64)
        TARGET_ARCH="x86_64"
        ;;
    aarch64|arm64|arm)
        TARGET_ARCH="aarch64"
        ;;
    ppc64le|powerpc64le|ppc)
        TARGET_ARCH="powerpc64le"
        ;;
    *)
        echo "Unknown architecture ${TARGET_ARCH}"
        exit 1
        ;;

esac

DCGM_DOCKER_IMAGE=${DCGM_DOCKER_IMAGE}-${TARGET_ARCH}

if [[ ${PRINT_NAME_ONLY} -eq 1 ]]; then
    echo ${DCGM_DOCKER_IMAGE}
    exit
fi

if [[ ${DCGM_BUILD_INSIDE_DOCKER:-} -ne 1 ]]; then
    docker run --rm -u "$(id -u)":"$(id -g)" ${DOCKER_ARGS:-} \
        --group-add $(stat -c '%g' /var/run/docker.sock) \
        -v "${DIR}":"${REMOTE_DIR}" \
        -v /run/docker.sock:/run/docker.sock:rw \
        -w "${REMOTE_DIR}" \
        ${MOUNT_OPTIONS} \
        -e "DCGM_BUILD_INSIDE_DOCKER=1" \
        -e "NPROC=$(nproc)" \
        ${DCGM_DOCKER_IMAGE} "$@"
else
    eval "$@"
fi
