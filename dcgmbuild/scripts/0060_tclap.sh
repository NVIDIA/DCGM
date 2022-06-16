#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=1.4

mkdir -p ${HOME}/.build/${TARGET}/tclap_src

pushd ${HOME}/.build/${TARGET}/tclap_src
git init
git remote add origin git://git.code.sf.net/p/tclap/code
git fetch --depth 1 origin 82abdc943e23b2f91678fb86c436962182ca9a29
git checkout FETCH_HEAD

mkdir -p docs/html || true
touch docs/manual.html

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON  \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOC=OFF \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    .

cmake --build . -j`nproc`
cmake --build . --target install

popd

