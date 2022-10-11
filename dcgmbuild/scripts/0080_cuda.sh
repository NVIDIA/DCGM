#!/usr/bin/env bash

set -ex -o nounset

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
    libcufft-dev
    libcurand-dev
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

unpack_rpm_to() {
    local filename=$(realpath ${1}); shift
    local dest_dir=$(realpath ${1}); shift
    rpm2cpio "${filename}" | cpio -idm -D "${dest_dir}"
}

process_rpm_package() {
    local tmpdir=$(realpath ${1}); shift
    pushd ${tmpdir}
    unpack_rpm_to package.rpm "${tmpdir}"
    for cuda_file in ${PARTS_TO_UNPACK[@]}; do
        find . -iname "${cuda_file}*" -print0 | while IFS= read -r -d '' file; do unpack_rpm_to "${file}" "${PREFIX}"; done
    done
    popd
}

process_cuda_package(){
# Once unpacked listing file is in /etc/apt/sources.list.d/*.list
#   old format is: deb file:///var/cuda-repo-10-0-local-10.0.130-410.48 /
#   new format: deb [signature related info] file:///var/cuda-repo-10-0-local-10.0.130-410.48 /
# we need to support both as Cuda <11.8 use the former and >=11.8 use the latter.
    local dir=${1}; shift
    pushd ${dir}
    dpkg -x package.deb .
    local subdir=$(cat etc/apt/sources.list.d/*.list | sed -r -e 's!^deb\s+(\[.*\]\s+)?file:\/\/\/!!' | sed -r -e 's/\s*\/$//')
    cd ${subdir}
    for cuda_file in ${PARTS_TO_UNPACK[@]}; do
        find . -iname "${cuda_file}*" -exec dpkg -x {} ${PREFIX} \;
    done
    popd
}

case ${TARGET} in
    x86_64-linux-gnu)
        CUDA10_URL=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64
        CUDA10_URL_CHKSUM=65bc98980ecdb275dcd55304425cbb6ed5a8afe0505e778caa4333bf4ab9d3310639bc69c302e32207e624a7217e684318f0ddc4f121b56c0c2c845f6ccae5d6

        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        CUDA11_URL_CHKSUM=9bed302b7fdeebbc2c7e73392f86ac895b20566de8992f494201c732aca33209d2f7fdad81749cb361b4e8b7001601c6789ea42316bc8074a5cf31651143d0dd
        ;;
    powerpc64le-linux-gnu)
        CUDA10_URL=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_ppc64el
        CUDA10_URL_CHKSUM=950be9a5022907774739c52c5a25779253e9b8bad8bc7e05db91e02afcbe0d6c68c2bf32dffaa79b12b9fbfb93dd0e14cd56269e13f01915d0106645efd6451c

        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-rhel8-11-8-local-11.8.0_520.61.05-1.ppc64le.rpm
        CUDA11_URL_CHKSUM=9138b6123ac4526d95896d86d542269d6d1229601dc280a81c9db8010b78034285454d31d32f56824a2d236634a97b71b4440d176420cd3b1880ff7d04627ee5
        ;;
    aarch64-linux-gnu)
        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_arm64.deb
        CUDA11_URL_CHKSUM=f0c5288e6a51d82852dac1edf70fda8d994c84cbf5491fa12f097fcdec6b294918bbd388a94a386e9d92cd2e51926a982c59e01494189187ef8d656e244bda4a
        ;;
    *)
        die "Unknown architecture"
        ;;
esac

if [[ "${TARGET}" != "aarch64-linux-gnu" ]]; then
    mkdir cuda10
    download_cuda_package "${CUDA10_URL}" cuda10 package.deb "${CUDA10_URL_CHKSUM}"

    process_cuda_package cuda10
fi

mkdir cuda11
if [[ "${TARGET}" != "powerpc64le-linux-gnu" ]]; then
    download_cuda_package "${CUDA11_URL}" cuda11 package.deb "${CUDA11_URL_CHKSUM}"
    process_cuda_package cuda11
else
    download_cuda_package "${CUDA11_URL}" cuda11 package.rpm "${CUDA11_URL_CHKSUM}"
    process_rpm_package cuda11
fi

popd
