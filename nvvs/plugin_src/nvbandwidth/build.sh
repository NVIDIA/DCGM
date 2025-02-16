#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 [--no-clone] --src-dir=<user_provided_folder_path> [-- <cuda local repo url>...]"
    echo "Example: $0 --no-clone --src-dir=/path/to/source-directory -- https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_arm64.deb"
    exit 1
}

# Parse arguments
OPTS=$(getopt -o '' --long no-clone,src-dir: -- "$@")
if [ $? != 0 ]; then
    usage
fi

eval set -- "$OPTS"

# Default values
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-90;100}"
NVBANDWIDTH_BUILD_BUILD_ID_SUFFIX=${NVBANDWIDTH_BUILD_BUILD_ID_SUFFIX:-}
NVBANDWIDTH_BUILD_IMAGE_NAME=${NVBANDWIDTH_BUILD_IMAGE_NAME:-nvbandwidthbuild}
NVBANDWIDTH_REPO_URL="${NVBANDWIDTH_REPO_URL:-https://github.com/NVIDIA/nvbandwidth}"
NVBANDWIDTH_VERSION="${NVBANDWIDTH_VERSION:-v0.5}"

declare -a cuda_local_repo_urls=(
    'https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb'
    'https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_arm64.deb'
)

while true; do
    case "$1" in
        --no-clone)
            NOCLONE=
            shift
            ;;
        --src-dir)
            SRC_DIR="$2"
            shift 2
            ;;
        --)
            shift
            test -z "$@" || cuda_local_repo_urls=( "$@" )
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
declare -A PLATFORMS=( ['x86_64']='amd64' ['aarch64']='arm64' )
declare -A BASE_IMAGES=( ['amd64']='ubuntu:20.04' ['arm64']='arm64v8/ubuntu:20.04' )
BUILD_PLATFORM="linux/${PLATFORMS[$(uname -m)]}"
DIR=$(dirname $(realpath $0))
REMOTE_DIR="/usr/local/src/nvbandwidth_src"
SRC_DIR=$(realpath ${SRC_DIR})

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
    [ -f $(basename "$url") ] || curl --location --output $(basename "$url") "$url"
    ln -sf $(basename "$url") cuda-repo.deb

    CUDA_VERSION=$(dpkg-deb --field cuda-repo.deb Version)
    TARGET_ARCHITECTURE=$(dpkg-deb --field cuda-repo.deb Architecture)

    BUILD_TAG="$NVBANDWIDTH_BUILD_IMAGE_NAME:$NVBANDWIDTH_VERSION-gcc11-cuda${CUDA_VERSION}-$TARGET_ARCHITECTURE"
    
    if ! docker image inspect "$LOCAL_REGISTRY/$BUILD_TAG" >/dev/null; then
        docker buildx rm mybuilder || true
    
        TARGET_PLATFORM="linux/$TARGET_ARCHITECTURE"
    
        # We've assumed the host platform and target platform are the same
        if [ $BUILD_PLATFORM != $TARGET_PLATFORM ]; then
            # Set up QEMU emulation
            docker run --privileged \
                       --rm \
                       multiarch/qemu-user-static --reset \
                                                  --credentials yes \
                                                  --persistent yes
                                                  

            docker buildx create --name mybuilder \
                                 --use \
                                 --platform $BUILD_PLATFORM,$TARGET_PLATFORM
        else
            docker buildx create --name mybuilder --use
        fi
    
        docker buildx inspect mybuilder --bootstrap
        docker buildx build --build-arg "BASE_IMAGE=${BASE_IMAGES[$TARGET_ARCHITECTURE]}" \
                            --builder mybuilder \
                            --file Dockerfile \
                            --load \
                            --platform $TARGET_PLATFORM \
                            --tag "$LOCAL_REGISTRY/$BUILD_TAG" \
                            "$DIR"
    
        docker push "$LOCAL_REGISTRY/$BUILD_TAG"
        docker buildx rm mybuilder
        docker container prune --force || true
        docker image prune --force || true
    fi

    CMAKE_BINARY_DIR=_out/build/Linux-$TARGET_ARCHITECTURE-release/cuda$CUDA_VERSION

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
              -B $CMAKE_BINARY_DIR \
              -DMULTINODE=1 \
              '-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}' \
              -DCMAKE_BUILD_TYPE=Release \
              '-DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++ -static-libgcc' \
              '-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc';
        cmake --build $CMAKE_BINARY_DIR"
done
