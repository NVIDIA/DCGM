#!/usr/bin/env bash

# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

usage() {
    echo "Usage: $0 [--no-clone] [--multinode] --src-dir=<user_provided_folder_path> [-- <cuda local repo url>...]"
    echo "Example: $0 --no-clone --multinode --src-dir=/path/to/source-directory -- https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_arm64.deb"
    exit 1
}

# Parse arguments
if ! OPTS=$(getopt -o '' --long no-clone,multinode,src-dir: -- "$@"); then
    usage
fi

eval set -- "$OPTS"

# Default values
NVBANDWIDTH_BUILD_BUILD_ID_SUFFIX=${NVBANDWIDTH_BUILD_BUILD_ID_SUFFIX:-}
NVBANDWIDTH_BUILD_IMAGE_NAME=${NVBANDWIDTH_BUILD_IMAGE_NAME:-nvbandwidthbuild}
NVBANDWIDTH_REPO_URL="${NVBANDWIDTH_REPO_URL:-https://github.com/NVIDIA/nvbandwidth}"
MULTINODE=0

declare -a cuda_local_repo_urls=(
    'https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-ubuntu2004-12-9-local_12.9.1-575.57.08-1_amd64.deb'
    'https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-ubuntu2004-12-9-local_12.9.1-575.57.08-1_arm64.deb'
    'https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb'
    'https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_arm64.deb'
)

while true; do
    case "$1" in
        --no-clone)
            NOCLONE=
            shift
            ;;
        --multinode)
            MULTINODE=1
            shift
            ;;
        --src-dir)
            SRC_DIR="$2"
            shift 2
            ;;
        --)
            shift
            if [ "$#" -ne 0 ]; then
                cuda_local_repo_urls=( "$@" )
            fi
            break
            ;;
        *)
            usage
            ;;
    esac
done

# Validate mandatory arguments
if [ ! -v SRC_DIR ]; then
    usage
fi

# Constants
# Using Ubuntu 20.04 for glibc compatibility on older systems
# Note cuda 13.0.0 doesn't have ubuntu 20.04 support, but we can use cuda 13.0.0-ubuntu 24.04 for the build
UBUNTU_VERSION="${UBUNTU_VERSION:-20.04}"
declare -A PLATFORMS=( ['x86_64']='amd64' ['aarch64']='arm64' )
declare -A BASE_IMAGES=( ['amd64']="ubuntu:${UBUNTU_VERSION}" ['arm64']="arm64v8/ubuntu:${UBUNTU_VERSION}" )
BUILD_PLATFORM="linux/${PLATFORMS[$(uname -m)]}"
DIR=$(dirname "$(realpath "$0")")
REMOTE_DIR="/usr/local/src/nvbandwidth_src"
if [ -v NOCLONE ]; then
    if [ ! -d "${SRC_DIR}" ]; then
        >&2 echo "Error: Source directory ${SRC_DIR} does not exist."
        exit 1
    fi
    SRC_DIR=$(realpath -- "${SRC_DIR}")
else
    SRC_DIR=$(realpath -m -- "${SRC_DIR}")
fi
OUTPUT_BINARY_NAME=nvbandwidth
BUILD_DIR_SUFFIX=
if [ "$MULTINODE" -eq 1 ]; then
    NVBANDWIDTH_VERSION="${NVBANDWIDTH_VERSION:-v0.9}"
    OUTPUT_BINARY_NAME=mnnvbandwidth
    BUILD_DIR_SUFFIX=-multinode
else
    NVBANDWIDTH_VERSION="${NVBANDWIDTH_VERSION:-V0.8}"
fi

# Clone the repository if --no-clone is not set
if [ ! -v NOCLONE ]; then
    if [ -d "$SRC_DIR" ]; then
        >&2 echo "Error: Trying to clone from $NVBANDWIDTH_REPO_URL to $SRC_DIR"
        >&2 echo "Error: Directory $SRC_DIR already exists."
        exit 1
    fi

    if ! git clone --depth 1 \
                   --branch "$NVBANDWIDTH_VERSION" \
                   "$NVBANDWIDTH_REPO_URL" \
                   "$SRC_DIR"
    then
        >&2 echo "Failed to clone repository from $NVBANDWIDTH_REPO_URL to $SRC_DIR"
        exit 1
    fi
fi

# Check if local registry is running
if ! docker ps | awk '{print $NF}' | grep --quiet 'nvbandwidth-local-registry'
then
    >&2 echo "Starting local Docker registry..."
    docker run --detach \
               --name nvbandwidth-local-registry \
               --publish 0:5000 \
               --rm \
               registry:2
else
    >&2 echo "Local Docker registry is already running."
fi
 
LOCAL_REGISTRY="localhost:$(
    docker container port nvbandwidth-local-registry 5000 \
    | head -n1 \
    | cut -d: -f2)"

for url in "${cuda_local_repo_urls[@]}";
do
    cuda_repo_deb=$(basename "$url")
    [ -f "$cuda_repo_deb" ] || curl --location --output "$cuda_repo_deb" "$url"
    ln -sf "$cuda_repo_deb" cuda-repo.deb

    CUDA_VERSION=$(dpkg-deb --field cuda-repo.deb Version)
    TARGET_ARCHITECTURE=$(dpkg-deb --field cuda-repo.deb Architecture)

    BUILD_TAG="$NVBANDWIDTH_BUILD_IMAGE_NAME:$NVBANDWIDTH_VERSION-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}-$TARGET_ARCHITECTURE"
    
    if ! docker image inspect "$LOCAL_REGISTRY/$BUILD_TAG" >/dev/null; then
        docker buildx rm mybuilder || true
    
        TARGET_PLATFORM="linux/$TARGET_ARCHITECTURE"
    
        # We've assumed the host platform and target platform are the same
        if [ "$BUILD_PLATFORM" != "$TARGET_PLATFORM" ]; then
            # Set up QEMU emulation
            docker run --privileged \
                       --rm \
                       multiarch/qemu-user-static --reset \
                                                  --credentials yes \
                                                  --persistent yes
                                                  

            docker buildx create --name mybuilder \
                                 --use \
                                 --platform "$BUILD_PLATFORM,$TARGET_PLATFORM"
        else
            docker buildx create --name mybuilder --use
        fi
    
        docker buildx inspect mybuilder --bootstrap
        docker buildx build --build-arg "BASE_IMAGE=${BASE_IMAGES[$TARGET_ARCHITECTURE]}" \
                            --builder mybuilder \
                            --file Dockerfile \
                            --load \
                            --platform "$TARGET_PLATFORM" \
                            --tag "$LOCAL_REGISTRY/$BUILD_TAG" \
                            "$DIR"
    
        docker push "$LOCAL_REGISTRY/$BUILD_TAG"
        docker buildx rm mybuilder
        docker container prune --force || true
        docker image prune --force || true
    fi

    CMAKE_BINARY_DIR=_out/build/Linux-$TARGET_ARCHITECTURE-release/cuda$CUDA_VERSION$BUILD_DIR_SUFFIX

    # Multinode CUDA 13.2 arm64 builds of nvbandwidth v0.9/v0.10.0 fail if CMakeLists.txt
    # explicitly adds MPI_INCLUDE_PATH because it injects -isystem=/usr/include. Remove
    # that include_directories line and any existing CMake build directory before retrying;
    # MPI::MPI_CXX already supplies the required MPI include directories.
    # Launch a container to build the NVBandwidth
    docker run \
        --rm \
        --user "$(id -u)":"$(id -g)" \
        --volume "${SRC_DIR}":"${REMOTE_DIR}" \
        --workdir "${REMOTE_DIR}" \
        ${DOCKER_ARGS:-} \
        "$LOCAL_REGISTRY/$BUILD_TAG" \
        bash -c "
        set -ex;
        cmake -S . \
              -B \"$CMAKE_BINARY_DIR\" \
              -DMULTINODE=$MULTINODE \
              -DCMAKE_BUILD_TYPE=Release \
              '-DCMAKE_CUDA_ARCHITECTURES=all' \
              '-DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++ -static-libgcc' \
              '-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc';
        cmake --build \"$CMAKE_BINARY_DIR\" --verbose;
        if [ \"$OUTPUT_BINARY_NAME\" != \"nvbandwidth\" ]; then
            cp \"$CMAKE_BINARY_DIR/nvbandwidth\" \"$CMAKE_BINARY_DIR/$OUTPUT_BINARY_NAME\";
        fi;
        strip \"$CMAKE_BINARY_DIR/$OUTPUT_BINARY_NAME\""
done
