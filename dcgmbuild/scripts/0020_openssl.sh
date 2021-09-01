#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=1.1.1k

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

echo "73cd042d4056585e5a9dd7ab68e7c7310a3a4c783eafa07ab0b560e7462b924e4376436a6d38a155c687f6942a881cfc0c1b9394afcde1d8c46bf396e7d51121 openssl.tar.gz" | sha512sum -c -
mkdir openssl_src
tar xf openssl.tar.gz -C openssl_src --strip-components=1
mkdir openssl_build
cd openssl_build
../openssl_src/Configure --prefix=${PREFIX} -fPIC ${OPENSSL_ARCH}
make -j`nproc`
make install

popd
