#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

mkdir -p ${HOME}/.build/doxygen/{build,src}
pushd ${HOME}/.build/doxygen

VERSION=1_9_1

download_url "https://github.com/doxygen/doxygen/archive/Release_${VERSION}.tar.gz" doxygen.tar.gz
echo "1b835701f3d76a968442ac3912842c7ee3e24bbce89517bef9c81ff54c4866f35bf693ad6dd235519adffdb5f0bfe418956ddc096edddc1c53c0d3030a86c1e7  doxygen.tar.gz" | sha512sum -c -

tar xzf doxygen.tar.gz -C src --strip-components=1
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-Wno-return-local-addr" -DCMAKE_CXX_FLAGS="-Wno-return-local-addr" ../src
cmake --build . -j`nproc`
cmake --build . --target install

popd

