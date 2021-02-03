#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

mkdir -p ${HOME}/.build/${TARGET}/cuda
pushd ${HOME}/.build/${TARGET}/cuda

PARTS_TO_UNPACK=(
    cuda-cublas-dev
    cuda-cudart-dev
    cuda-cufft-dev
    cuda-cupti
    cuda-curand-dev
    cuda-driver-dev
    cuda-misc-headers
    cuda-nvml-dev
    cuda-license
    cuda-nvcc
    libcublas-dev
    )

download_cuda_package(){
    local url=${1}; shift;
    local outdir=${1}; shift;
    local filename=${1}; shift;
    local chksum=${1:-}; shift
    download_url "${url}" "${outdir}/${filename}"
    if [[ -n "${chksum}" ]]; then
        echo "${chksum}  ${outdir}/${filename}" | sha512sum -c -
    fi
}

process_cuda_package(){
# Once unpacked listing file is in /etc/apt/sources.list.d/*.list
#   format is: deb file:///var/cuda-repo-10-0-local-10.0.130-410.48 /
# cat *.list | sed -e 's/^deb file:\/\/\///' | sed -e 's/\s*\/$//'
    local dir=${1}; shift
    pushd ${dir}
    dpkg -x package.deb .
    local subdir=$(cat etc/apt/sources.list.d/*.list | sed -e 's/^deb file:\/\/\///' | sed -e 's/\s*\/$//')
    cd ${subdir}
    for cuda_file in ${PARTS_TO_UNPACK[@]}; do
        find . -iname "${cuda_file}*" -exec dpkg -x {} ${PREFIX} \;
    done
    popd
}

case ${TARGET} in
    x86_64-linux-gnu)
        CUDA9_URL=https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64
        CUDA9_URL_CHKSUM=2cc5e9c9eda82de89ceddd3dfd91df15ac8330e3660242cd2b232e8f2cde0a8a3541f92219e71bf43d09595a95a94df9ab07d0a7591aa9244e6f11303fc4389d

        CUDA9PATCH_URL=https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda-repo-ubuntu1604-9-2-148-local-patch-1_1.0-1_amd64
        CUDA9PATCH_URL_CHKSUM=4e161caed4b78e2a91bb7587e3889bdbfecaed8a1d6a6ac13eb3db71493f9f0cb15e5166ac248d1e20090f8b93213f4a47a8edc5490c304fd7c130d5247420e8

        CUDA10_URL=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64
        CUDA10_URL_CHKSUM=65bc98980ecdb275dcd55304425cbb6ed5a8afe0505e778caa4333bf4ab9d3310639bc69c302e32207e624a7217e684318f0ddc4f121b56c0c2c845f6ccae5d6

        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
        CUDA11_URL_CHKSUM=12661466cacd450cc7e36c9abda59cae91573655bded31a58e8cfa073464fa8bab47dae9a43ca50e0f52e837d13c83079812240ba71628a2b7d1a440d2f73357
        ;;
    powerpc64le-linux-gnu)
        CUDA9_URL=https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.148-1_ppc64el
        CUDA9_URL_CHKSUM=f0678f0b162bdd28a77ea2c58f7da0081246a0e99d25f19b1dbc4d6aa494100eae69901560be71c26fc1389d6af7040adedc4a043912eef2623806471b8c530e

        CUDA9PATCH_URL=https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda-repo-ubuntu1604-9-2-148-local-patch-1_1.0-1_ppc64el
        CUDA9PATCH_URL_CHKSUM=928dd070074d49bd44417663842224c1b72c0d9922f8e62f161518511ff0ff3efdaf4a2d3aa6dd950ea8597c5d26aa5295f632ea07a9e65f3d2267f8c03555c5

        CUDA10_URL=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_ppc64el
        CUDA10_URL_CHKSUM=950be9a5022907774739c52c5a25779253e9b8bad8bc7e05db91e02afcbe0d6c68c2bf32dffaa79b12b9fbfb93dd0e14cd56269e13f01915d0106645efd6451c

        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_ppc64el.deb
        CUDA11_URL_CHKSUM=85f5640912e44d97007ab1e61a02d23e1a3f36339609f245f1d67872b578acb05a7ed02352e49c82b1f072d48c69eaba1be00ce70f63db8204f8aacca9646987
        ;;
    aarch64-linux-gnu)
        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_arm64.deb
        CUDA11_URL_CHKSUM=0f8b961cd62dd0b3d5ff89a7aa639b630cbbb08a51cc3e72c79289dfb1b7f87e842d52c16d731e2fad4fd7375567723ef65ea079962c7d4c104303cb64bbe4e8
        ;;
    *)
        die "Unknown architecture"
        ;;
esac

if [[ "${TARGET}" != "aarch64-linux-gnu" ]]; then
    mkdir cuda{9,9patch,10}
    download_cuda_package "${CUDA9_URL}" cuda9 package.deb "${CUDA9_URL_CHKSUM}"
    download_cuda_package "${CUDA9PATCH_URL}" cuda9patch package.deb "${CUDA9PATCH_URL_CHKSUM}"
    download_cuda_package "${CUDA10_URL}" cuda10 package.deb "${CUDA10_URL_CHKSUM}"

    process_cuda_package cuda9
    process_cuda_package cuda9patch
    process_cuda_package cuda10
fi

mkdir cuda11
download_cuda_package "${CUDA11_URL}" cuda11 package.deb "${CUDA11_URL_CHKSUM}"
process_cuda_package cuda11

popd
