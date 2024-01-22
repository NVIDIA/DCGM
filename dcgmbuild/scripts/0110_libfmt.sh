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
set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

PKGNAME=fmt
PKGVER=9.1.0
PKG="${PKGNAME}-${PKGVER}"
PKGDIR="${HOME}/.build/${TARGET}/${PKG}"
SOURCE="https://github.com/fmtlib/fmt/archive/refs/tags/${PKGVER}.tar.gz"
SHA512SUM="a18442042722dd48e20714ec034a12fcc0576c9af7be5188586970e2edf47529825bdc99af366b1d5891630c8dbf6f63bfa9f012e77ab3d3ed80d1a118e3b2be"

mkdir -p ${PKGDIR}_{build,src}
mkdir -p ${PKGDIR}_build_{Release,Debug,RelWithDebInfo}

function build_libfmt(){
    local build_type=$1
    pushd ${PKGDIR}_build_${build_type}
    cmake \
        -DCMAKE_BUILD_TYPE=${build_type} \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_STATIC_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        "${PKGDIR}_src"

    cmake --build . -j`nproc`
    cmake --build . --target install
    popd
}

pushd ${PKGDIR}_build
download_url "${SOURCE}" "${PKG}.tar.gz"
echo "${SHA512SUM}  ${PKG}.tar.gz" | sha512sum -c -
tar xf "${PKG}.tar.gz" -C "${PKGDIR}_src" --strip-components=1

build_libfmt Debug
build_libfmt Release
build_libfmt RelWithDebInfo
