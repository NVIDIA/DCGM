#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=2.1.8

mkdir ${HOME}/.build/${TARGET}/libevent_{build,src}

download_url "https://github.com/libevent/libevent/releases/download/release-${VERSION}-stable/libevent-${VERSION}-stable.tar.gz" libevent.tar.gz
echo "a2fd3dd111e73634e4aeb1b29d06e420b15c024d7b47778883b5f8a4ff320b5057a8164c6d50b53bd196c79d572ce2639fe6265e03a93304b09c22b41e4c2a17  libevent.tar.gz" | sha512sum -c -
tar xzf libevent.tar.gz -C ${HOME}/.build/${TARGET}/libevent_src --strip-components=1

pushd ${HOME}/.build/${TARGET}/libevent_src
./autogen.sh
popd

pushd ${HOME}/.build/${TARGET}/libevent_build
CFLAGS="${CFLAGS:-} -Wno-cast-function-type -Wno-implicit-fallthrough -fPIC" \
    CXXFLAGS="${CXXFLAGS:-} -Wno-cast-function-type -Wno-implicit-fallthrough -fPIC" \
    ${HOME}/.build/${TARGET}/libevent_src/configure --disable-shared --with-pic --prefix=${PREFIX} --host=${TARGET}
make -j`nproc`
make install

popd

