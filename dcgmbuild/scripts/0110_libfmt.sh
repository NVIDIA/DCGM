#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

PKGNAME=fmt
PKGVER=8.0.1
PKG="${PKGNAME}-${PKGVER}"
PKGDIR="${HOME}/.build/${TARGET}/${PKG}"
SOURCE="https://github.com/fmtlib/fmt/archive/refs/tags/${PKGVER}.tar.gz"
SHA512SUM="643e68d5b2e0e9c83231ab2b0036596a6297b1d9ed6bd7b1172bee4ff134c8af8f09174c06c94225132c1b635b0977ea4ce783748d7bd76a9a0b5ad597456c84"

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
