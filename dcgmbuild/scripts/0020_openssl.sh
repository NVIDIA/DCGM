#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=1.1.1i

mkdir -p ${HOME}/.build/${TARGET}
pushd ${HOME}/.build/${TARGET}

case ${TARGET} in
    x86_64-linux-gnu)
        OPENSSL_ARCH=linux-x86_64
        ;;
    powerpc64le-linux-gnu)
        OPENSSL_ARCH=linux-ppc64le
        ;;
    aarch64-linux-gnu)
        OPENSSL_ARCH=linux-aarch64
        ;;
    *)
        die "Unknown target architecture"
        ;;
esac


download_url "https://www.openssl.org/source/openssl-${VERSION}.tar.gz" openssl.tar.gz

echo "fe12e0ab9e1688f24dd862ac633d0ab703b499c0f34b53c3560aa0d3879d81d647aa0678ed517dda5efb2711f669fcb1a1e0e24f6eac2efc2cf4eae6b62014d8 openssl.tar.gz" | sha512sum -c -
mkdir openssl_src
tar xf openssl.tar.gz -C openssl_src --strip-components=1
mkdir openssl_build
cd openssl_build
../openssl_src/Configure --prefix=${PREFIX} -fPIC ${OPENSSL_ARCH}
make -j`nproc`
make install

popd
