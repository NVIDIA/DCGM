#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=0.6.2

mkdir -p ${HOME}/.build/${TARGET}/yaml_{build,src}
pushd ${HOME}/.build/${TARGET}/yaml_build

download_url "https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-${VERSION}.tar.gz" yaml-cpp.tar.gz
echo "fea8ce0a20a00cbc75023d1db442edfcd32d0ac57a3c41b32ec8d56f87cc1d85d7dd7a923ce662f5d3a315f91a736d6be0d649997acd190915c1d68cc93795e4  yaml-cpp.tar.gz" | sha512sum -c -
tar xzf yaml-cpp.tar.gz -C ${HOME}/.build/${TARGET}/yaml_src --strip-components=1

cmake -E env CXXFLAGS='-Wno-unused-variable' CFLAGS='-Wno-unused-variable' \
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON  \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_STATIC_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    ${HOME}/.build/${TARGET}/yaml_src

cmake --build . -j`nproc`
cmake --build . --target install

popd

