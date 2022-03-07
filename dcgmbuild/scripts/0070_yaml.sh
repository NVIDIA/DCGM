#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=0.6.3

mkdir -p ${HOME}/.build/${TARGET}/yaml_{build,src}
pushd ${HOME}/.build/${TARGET}/yaml_build

download_url "https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-${VERSION}.tar.gz" yaml-cpp.tar.gz
echo "68b9ce987cabc1dec79382f922de20cc2c222cb9c090ecb93dc686b048da5c917facf4fce6d8f72feea44b61e5a6770ed3b0c199c4cd4e6bde5b6245c09f8e49  yaml-cpp.tar.gz" | sha512sum -c -
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

