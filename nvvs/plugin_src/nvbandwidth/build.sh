#!/usr/bin/env bash

set -euxo pipefail

usage() {
    echo "Usage: $0 [--no_clone] --src_dir=<user_provided_folder_path>"
    echo "Example: $0 --no_clone --src_dir=/path/to/source-directory"
    exit 1
}

# Default values
NO_CLONE=false
SRC_DIR=""

# Parse arguments
OPTS=$(getopt -o '' --long no_clone,src_dir: -- "$@")
if [ $? != 0 ]; then
    usage
fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        --no_clone)
            NO_CLONE=true; shift ;;
        --src_dir)
            SRC_DIR="$2"; shift 2 ;;
        --)
            shift; break ;;
        *)
            usage ;;
    esac
done

# Validate mandatory arguments
if [ -z "$SRC_DIR" ]; then
    usage
fi

echo "NO_CLONE: $NO_CLONE"
echo "SRC_DIR: $SRC_DIR"

# Clone the repository if --no_clone is not set
if [ "$NO_CLONE" = false ]; then
    REPO_URL="https://github.com/NVIDIA/nvbandwidth"
    if [ -d "$SRC_DIR" ]; then
        echo "Error: Trying to clone from $REPO_URL to $SRC_DIR"
        echo "Error: Directory $SRC_DIR already exists."
        exit 1
    fi
    git clone --depth 1 --branch v0.5 "$REPO_URL" "$SRC_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to clone repository from $REPO_URL to $SRC_DIR"
        exit 1
    fi
fi

# Change to the target directory
SRC_DIR=$(realpath ${SRC_DIR})
REMOTE_DIR="/usr/local/src/nvbandwidth_src"
DIR=$(dirname $(realpath $0))

BASE_DOCKERFILE=Dockerfile

NVBANDWIDTH_BUILD_BUILD_ID_SUFFIX=${NVBANDWIDTH_BUILD_BUILD_ID_SUFFIX:-}
NVBANDWIDTH_BUILD_IMAGE_NAME=${NVBANDWIDTH_BUILD_IMAGE_NAME:-nvbandwidthbuild}
echo ${NVBANDWIDTH_BUILD_IMAGE_NAME}

NVBANDWIDTH_VERSION=0.5
LOCAL_REGISTRY="localhost:5050"

NVBANDWIDTH_BUILD_GIT_COMMIT=$(git rev-parse HEAD)
NVBANDWIDTH_BUILD_GIT_ORIGIN=$(git config --get remote.origin.url)

DEFAULT_ARCH="x86_64"
targets=("x86_64" "aarch64")
dockerfiles=("Dockerfile.x86_64" "Dockerfile.aarch64")
cuda_repo_suffixes=("x86_64" "sbsa")

echo "Building for targets: ${targets[@]}"

# Find a local image for the specified architecture
# find_local_image [ARCH]
# If ARCH is not specified, use the host architecture
# If a local image is found, print its name
# If no local image is found, nothing is printed
function find_local_image() {
    function check_local_image() {
        docker image inspect "$1" >/dev/null
    }
    local target_arch=${1:-${DEFAULT_ARCH}}
    local LOCAL_IMAGE="${LOCAL_REGISTRY}/${NVBANDWIDTH_BUILD_TAG}-${target_arch}-cuda${CUDA_VERSION_MAJOR}"
    if check_local_image "${LOCAL_IMAGE}"; then
        echo "${LOCAL_IMAGE}"
        return 0
    fi
    return 0
}

function build_local_image() {
    local arch=${1}; shift
    local dockerfile=${1}; shift
    local CURRENT_BUILD_TAG=${NVBANDWIDTH_BUILD_TAG}-${arch}-cuda${CUDA_VERSION_MAJOR}

    # Check if local registry is running
    if ! docker ps | grep -q "registry:2"; then
        echo "Starting local Docker registry..."
        docker run -d -p 5050:5000 --name local-registry registry:2
    else
        echo "Local Docker registry is already running."
    fi

    docker buildx rm mybuilder || true
    if [ "$arch" = "aarch64" ]; then
        # Set up QEMU emulation
        docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
        docker buildx create --name mybuilder --use --platform linux/amd64,linux/arm64
        local platform="linux/arm64"
    else
        docker buildx create --name mybuilder --use
        local platform="linux/amd64"
    fi
    docker buildx inspect mybuilder --bootstrap
    docker buildx build --builder mybuilder --platform ${platform} -t "${LOCAL_REGISTRY}/${CURRENT_BUILD_TAG}" --load \
        --build-arg GIT_COMMIT=${NVBANDWIDTH_BUILD_GIT_COMMIT} \
        --build-arg GIT_ORIGIN=${NVBANDWIDTH_BUILD_GIT_ORIGIN} \
        --build-arg TARGETARCH=${arch} \
        --build-arg CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} \
        --build-arg CUDA_VERSION_MINOR=${CUDA_VERSION_MINOR} \
        --build-arg CUDA_VERSION_PATCH=${CUDA_VERSION_PATCH} \
        --build-arg CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        --build-arg CUDA_PUB_KEY_LINK=${CUDA_PUB_KEY_LINK} \
        --build-arg CUDA_PACKAGE_LINK=${CUDA_PACKAGE_LINK} \
        -f ${dockerfile} \
        ${DIR}
    docker push "${LOCAL_REGISTRY}/${CURRENT_BUILD_TAG}"
    docker buildx rm mybuilder
    docker container prune -f || true
    docker image prune -f || true
}

function nvbandwidth_build_using_docker() {
    local arch=${1}; shift
    local dockerfile=${1}; shift
    local nvbandwidth_docker_image=$(find_local_image ${arch})

    # The docker image is not found, build it
    if [[ -z "$nvbandwidth_docker_image" ]]; then
        build_local_image ${arch} ${dockerfile}
    fi
    nvbandwidth_docker_image=$(find_local_image ${arch})
    
    local release_suffix=Linux-${arch}-release
    local outdir_release=_out/build/${release_suffix}/cuda${CUDA_VERSION_MAJOR}
    # Launch a container to build the NVBandwidth
    docker run --rm -u "$(id -u)":"$(id -g)" \
        ${DOCKER_ARGS:-} \
        -v "${SRC_DIR}":"${REMOTE_DIR}" \
        -w "${REMOTE_DIR}" \
        ${nvbandwidth_docker_image} \
        bash -c "set -ex; mkdir -p ${outdir_release} && pushd ${outdir_release} && cmake -DMULTINODE=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DCMAKE_BUILD_TYPE=Release ${REMOTE_DIR} -DCMAKE_EXE_LINKER_FLAGS=\"-static-libstdc++ -static-libgcc\" -DCMAKE_SHARED_LINKER_FLAGS=\"-static-libstdc++ -static-libgcc\" && cmake --build ."
}

echo "### Docker debug info ###"
docker info
docker buildx ls
docker context ls
docker buildx inspect

num_arch=${#targets[@]}

for (( i=0; i<$num_arch; i++ )); do
    arch="${targets[$i]}"
    dockerfile="${dockerfiles[$i]}"
    repo_suffix="${cuda_repo_suffixes[$i]}"

    NVBANDWIDTH_DOCKERFILE_SHA256SUM_VALUE=$(sha256sum ${dockerfile} | head -c6)
    NVBANDWIDTH_BUILD_TAG="${NVBANDWIDTH_BUILD_IMAGE_NAME}:v${NVBANDWIDTH_VERSION}-gcc11-${NVBANDWIDTH_DOCKERFILE_SHA256SUM_VALUE}"

    CUDA_PUB_KEY_LINK="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${repo_suffix}/3bf863cc.pub"
    CUDA_PACKAGE_LINK="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${repo_suffix}"
    CUDA_VERSION_MAJOR="12"
    CUDA_VERSION_MINOR="0"
    CUDA_VERSION_PATCH="0"
    CUDA_ARCHITECTURES="90"

    nvbandwidth_build_using_docker "${arch}" "${dockerfile}" "$@"

done
