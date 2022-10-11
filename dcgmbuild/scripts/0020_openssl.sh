#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=1.1.1q

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

echo "cb9f184ec4974a3423ef59c8ec86b6bf523d5b887da2087ae58c217249da3246896fdd6966ee9c13aea9e6306783365239197e9f742c508a0e35e5744e3e085f openssl.tar.gz" | sha512sum -c -
mkdir openssl_src
tar xf openssl.tar.gz -C openssl_src --strip-components=1
mkdir openssl_build
cd openssl_build
../openssl_src/Configure --prefix=${PREFIX} -fPIC ${OPENSSL_ARCH}
make -j`nproc`
make install

popd
