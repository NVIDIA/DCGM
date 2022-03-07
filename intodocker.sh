#!/usr/bin/env bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
        -n | --name            : Print docker image name and exit
        -h | --help            : Print this help and exit

    Command will be run inside the docker container
"
}

LONG_OPTS=name
SHORT_OPTS=n

! PARSED=$(getopt --options=${SHORT_OPTS} --longoptions=${LONG_OPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments"
    exit 1
fi

eval set -- "${PARSED}"


STATIC_ANALYSIS=0
while true; do
    case "${1}" in
        -h | --help)
            usage
            exit 0
            ;;
        -n | --name)
            echo ${DCGM_DOCKER_IMAGE}
            exit 0
            ;;
        --)
            shift
            break
            ;;
    esac
done

if [[ ${STATIC_ANALYSIS} -eq 1 ]]; then
    MOUNT_OPTIONS="--mount source=dcgm_coverity,target=/coverity"
fi

if [[ -t 0 ]]; then
    DOCKER_ARGS=-it
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
