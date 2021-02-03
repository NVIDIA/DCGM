#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

mkdir -p ${HOME}/.build/doxygen/{build,src}
pushd ${HOME}/.build/doxygen

VERSION=1_8_16

download_url "https://github.com/doxygen/doxygen/archive/Release_${VERSION}.tar.gz" doxygen.tar.gz
echo "546ceaae949cf5cc8162309bc804dad7a00b970c14a2dd66d27111e03f2fbebfec299ddc559f8c915ed00776217d2c439843806d2b6c15b58cf29f490979bd8f  doxygen.tar.gz" | sha512sum -c -

tar xzf doxygen.tar.gz -C src --strip-components=1
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-Wno-return-local-addr" -DCMAKE_CXX_FLAGS="-Wno-return-local-addr" ../src
cmake --build . -j`nproc`
cmake --build . --target install

popd

