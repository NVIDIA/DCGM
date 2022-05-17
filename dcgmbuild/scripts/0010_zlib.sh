#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath $0))/common_for_targets.sh

VERSION=1.2.11

mkdir -p ${HOME}/.build/${TARGET}
pushd ${HOME}/.build/${TARGET}

download_url "https://github.com/madler/zlib/archive/refs/tags/v${VERSION}.tar.gz" zlib.tar.gz
echo "104c62ed1228b5f1199bc037081861576900eb0697a226cafa62a35c4c890b5cb46622e399f9aad82ee5dfb475bae26ae75e2bd6da3d261361b1c8b996970faf zlib.tar.gz" | sha512sum -c -
mkdir -p zlib_src
tar xf zlib.tar.gz -C zlib_src --strip-components=1

mkdir -p zlib_build
cd zlib_build
../zlib_src/configure --prefix=${PREFIX}
make -j`nproc`
make install
ldconfig

popd
