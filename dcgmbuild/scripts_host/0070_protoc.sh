#!/usr/bin/env bash
#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
