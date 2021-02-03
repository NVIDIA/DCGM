#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=3.7.1

mkdir -p ${HOME}/.build/protobuf_{build,src}
pushd ${HOME}/.build/protobuf_build

download_url "https://github.com/protocolbuffers/protobuf/releases/download/v${VERSION}/protobuf-cpp-${VERSION}.tar.gz" protobuf.tar.gz
echo "6e6873f83dc6e8cf565874c428799b0dd86f7eb83130d7eec26225f640560bb74e1604fda16ffd507e0416567236831922fbd9f5354308abee945b5b83ca153e  protobuf.tar.gz" | sha512sum -c -
tar xzf protobuf.tar.gz -C ${HOME}/.build/protobuf_src --strip-components=1

CFLAGS="${CFLAGS:-} -Wno-sign-compare -fwrapv -fPIC" \
    CXXFLAGS="${CXXFLAGS:-} -Wno-sign-compare -fwrapv -fPIC" \
    ${HOME}/.build/protobuf_src/configure --with-pic --prefix=/opt/cross
make -j`nproc`
make install

popd
