#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=1.8.4

mkdir -p ${HOME}/.build/${TARGET}/jsoncpp_{build,src}
pushd ${HOME}/.build/${TARGET}/jsoncpp_build

download_url "https://github.com/open-source-parsers/jsoncpp/archive/${VERSION}.tar.gz" jsoncpp.tar.gz
echo "f70361a3263dd8b9441374a9a409462be1426c0d6587c865171a80448ab73b3f69de2b4d70d2f0c541764e1e6cccc727dd53178347901f625ec6fb54fb94f4f1  jsoncpp.tar.gz" | sha512sum -c -
tar xzf jsoncpp.tar.gz -C ${HOME}/.build/${TARGET}/jsoncpp_src --strip-components=1

cmake -DJSONCPP_WITH_TEST=ON \
    -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
    -DJSONCPP_WITH_WARNINGS_AS_ERROR=OFF \
    -DJSONCPP_WITH_STRICT_ISO=OFF \
    -DJSONCPP_WITH_CMAKE_PACKAGE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_STATIC_LIBS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    ${HOME}/.build/${TARGET}/jsoncpp_src

cmake --build . -j`nproc`
cmake --build . --target install

popd

