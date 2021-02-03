#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath $0))/common_for_targets.sh

VERSION=1.2.11

mkdir -p ${HOME}/.build/${TARGET}
pushd ${HOME}/.build/${TARGET}

download_url "https://www.zlib.net/zlib-${VERSION}.tar.gz" zlib.tar.gz
echo "73fd3fff4adeccd4894084c15ddac89890cd10ef105dd5e1835e1e9bbb6a49ff229713bd197d203edfa17c2727700fce65a2a235f07568212d820dca88b528ae  zlib.tar.gz" | sha512sum -c -
mkdir -p zlib_src
tar xf zlib.tar.gz -C zlib_src --strip-components=1

mkdir -p zlib_build
cd zlib_build
../zlib_src/configure --prefix=${PREFIX}
make -j`nproc`
make install
ldconfig

popd
