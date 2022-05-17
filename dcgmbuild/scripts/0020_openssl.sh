#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=1.1.1o

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

echo "75b2f1499cb4640229eb6cd35d85cbff2e19db17b959ac4d04b60f1b395b73567f9003521452a0fcfeea9b31b26de0a7bccf476ecf9caae02298f3647cfb7e23 openssl.tar.gz" | sha512sum -c -
mkdir openssl_src
tar xf openssl.tar.gz -C openssl_src --strip-components=1
mkdir openssl_build
cd openssl_build
../openssl_src/Configure --prefix=${PREFIX} -fPIC ${OPENSSL_ARCH}
make -j`nproc`
make install

popd
