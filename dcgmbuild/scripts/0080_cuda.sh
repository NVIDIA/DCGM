#!/usr/bin/env bash
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
set -ex -o nounset

source $(dirname $(realpath ${0}))/common_for_targets.sh

mkdir -p ${HOME}/.build/${TARGET}/cuda
pushd ${HOME}/.build/${TARGET}/cuda

PARTS_TO_UNPACK=(
    cuda-cccl
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
        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        CUDA11_URL_CHKSUM=9bed302b7fdeebbc2c7e73392f86ac895b20566de8992f494201c732aca33209d2f7fdad81749cb361b4e8b7001601c6789ea42316bc8074a5cf31651143d0dd

        CUDA12_URL=https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.1-525.85.12-1_amd64.deb
        CUDA12_URL_CHKSUM=4a4be5447650d649ca4fdd5b6cf2ef6e9355fe4057c179fbbd5784418ff3455482194500f4eab89bca46a67dc8e3b4a17f6674308f3d3ea7d3d90d31ee752198
        ;;
    powerpc64le-linux-gnu)
        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-rhel8-11-8-local-11.8.0_520.61.05-1.ppc64le.rpm
        CUDA11_URL_CHKSUM=9138b6123ac4526d95896d86d542269d6d1229601dc280a81c9db8010b78034285454d31d32f56824a2d236634a97b71b4440d176420cd3b1880ff7d04627ee5

        CUDA12_URL=https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda-repo-rhel8-12-0-local-12.0.1_525.85.12-1.ppc64le.rpm
        CUDA12_URL_CHKSUM=7374225637a74ee67a0bff5ec21e75233d2bd91fa327d081dedf09e4aa0ecd9d9b5193da4222aa84ecb0533d610ced645a67e525f83e714798143a4b4952e123
        ;;
    aarch64-linux-gnu)
        CUDA11_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_arm64.deb
        CUDA11_URL_CHKSUM=f0c5288e6a51d82852dac1edf70fda8d994c84cbf5491fa12f097fcdec6b294918bbd388a94a386e9d92cd2e51926a982c59e01494189187ef8d656e244bda4a

        CUDA12_URL=https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.1-525.85.12-1_arm64.deb
        CUDA12_URL_CHKSUM=f7664be9ec13617a54d7480eb9b54e5ca7763682cb0a9350ecbeb8b55c0840143ff61db6864ffbf9649709d28533c8e16ef4f5ece4842092a811de3ae0b68007
        ;;
    *)
        die "Unknown architecture"
        ;;
esac

mkdir cuda11 cuda12
if [[ "${TARGET}" != "powerpc64le-linux-gnu" ]]; then
    download_cuda_package "${CUDA11_URL}" cuda11 package.deb "${CUDA11_URL_CHKSUM}"
    process_cuda_package cuda11
    download_cuda_package "${CUDA12_URL}" cuda12 package.deb "${CUDA12_URL_CHKSUM}"
    process_cuda_package cuda12
else
    download_cuda_package "${CUDA11_URL}" cuda11 package.rpm "${CUDA11_URL_CHKSUM}"
    process_rpm_package cuda11
    download_cuda_package "${CUDA12_URL}" cuda12 package.rpm "${CUDA12_URL_CHKSUM}"
    process_rpm_package cuda12
fi

popd
