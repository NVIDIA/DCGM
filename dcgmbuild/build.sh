#!/usr/bin/env bash

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
