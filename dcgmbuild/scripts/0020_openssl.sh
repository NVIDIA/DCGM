#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=1.1.1l

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

echo "d9611f393e37577cca05004531388d3e0ebbf714894cab9f95f4903909cd4f45c214faab664c0cbc3ad3cca309d500b9e6d0ecbf9a0a0588d1677dc6b047f9e0 openssl.tar.gz" | sha512sum -c -
mkdir openssl_src
tar xf openssl.tar.gz -C openssl_src --strip-components=1
mkdir openssl_build
cd openssl_build
../openssl_src/Configure --prefix=${PREFIX} -fPIC ${OPENSSL_ARCH}
make -j`nproc`
make install

popd
