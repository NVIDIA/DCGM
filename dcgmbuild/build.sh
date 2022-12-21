#!/usr/bin/env bash
#
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
#
set -euxo pipefail
DIR=$(dirname $(realpath $0))

BASE_DOCKERFILE=base.dockerfile
DCGMBUILD_IMAGE_NAME=${DCGMBUILD_IMAGE_NAME:-dcgmbuild}
DCGMBUILD_BASE_VERSION=gcc11-$(sha256sum ${BASE_DOCKERFILE} | head -c6)
DCGMBUILD_BASE_TAG="${DCGMBUILD_IMAGE_NAME}:base-${DCGMBUILD_BASE_VERSION}"

DOCKER_BUILD_OPTIONS=--compress

function docker_build() {
   docker buildx build ${DOCKER_BUILD_OPTIONS} "$@"
}

function image_exists() {
   [[ -n $(docker images -q "$1" 2> /dev/null) ]]
}

if ! image_exists ${DCGMBUILD_BASE_TAG}; then
    docker_build -t ${DCGMBUILD_BASE_TAG} -f ${BASE_DOCKERFILE} ${DIR}
fi

docker_build -t ${DCGMBUILD_IMAGE_NAME} --build-arg BASE_IMAGE=${DCGMBUILD_BASE_TAG} ${DIR}
