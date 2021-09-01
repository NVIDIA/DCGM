#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

PKGNAME=fmt
PKGVER=8.0.0
PKG="${PKGNAME}-${PKGVER}"
PKGDIR="${HOME}/.build/${TARGET}/${PKG}"
SOURCE="https://github.com/fmtlib/fmt/archive/refs/tags/${PKGVER}.tar.gz"
SHA512SUM="61768bf8b64c430f11536800985509ce436bbbe05cbe1dfb6045cfaf2f859af98eae1019ef602af8fec6946ae25e4d8adb589f0f738666b20beb3afe65ee760c"

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
tar xf "${PKG}.tar.gz" -C "${PKGDIR}_src" --strip-components=1

build_libfmt Debug
build_libfmt Release
build_libfmt RelWithDebInfo
