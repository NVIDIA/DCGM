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
set -euxo pipefail
DIR=$(dirname $(realpath $0))

BASE_DOCKERFILE=base.dockerfile
DCGMBUILD_IMAGE_NAME=${DCGMBUILD_IMAGE_NAME:-dcgmbuild}

####################
#### Versioning ####
####################

DCGMBUILD_BASE_VERSION=gcc11-$(sha256sum ${BASE_DOCKERFILE} | head -c6)
DCGMBUILD_BASE_TAG="${DCGMBUILD_IMAGE_NAME}:base-${DCGMBUILD_BASE_VERSION}"

###############
#### Logic ####
###############

DOCKER_BUILD_OPTIONS="--compress --platform=linux/amd64 --load --network=host"

function docker_build() {
    docker buildx build ${DOCKER_BUILD_OPTIONS} "$@"
}

function image_exists() {
    docker inspect --type=image --format="ignore" "$1" >/dev/null 2>&1
    return $?
}

function get_image_sha256() {
    docker inspect --type=image --format="{{.Id}}" "$1" 2>/dev/null
}

default_targets=("x86_64" "aarch64" "powerpc64le")

if [[ $# -gt 0 ]]; then
    targets=("$@")
else
    targets=("${default_targets[@]}")
fi

echo "Building for targets: ${targets[@]}"

for target in "${targets[@]}"; do
    CURRENT_BASE_TAG=${DCGMBUILD_BASE_TAG}-${target}
    CURRENT_CHILD_TAG=${DCGMBUILD_IMAGE_NAME}-${target}

    if ! image_exists ${CURRENT_BASE_TAG}; then
        echo Unable to find ${CURRENT_BASE_TAG}. Building ...
        docker_build -t ${CURRENT_BASE_TAG} \
            --build-arg TARGET=${target} \
            -f ${BASE_DOCKERFILE} \
            ${DIR}
    else
        echo ${CURRENT_BASE_TAG} already exists locally: $(get_image_sha256 ${CURRENT_BASE_TAG})
    fi

    BASE_IMAGE=${CURRENT_BASE_TAG}

    docker_build -t ${CURRENT_CHILD_TAG} \
        --build-arg BASE_IMAGE=${BASE_IMAGE} \
        --build-arg BASE_IMAGE_TARGET=${target} \
        ${DIR}
done
