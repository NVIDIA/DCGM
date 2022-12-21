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

source $(dirname $(realpath $0))/common_for_targets.sh

VERSION=1.2.13

mkdir -p ${HOME}/.build/${TARGET}
pushd ${HOME}/.build/${TARGET}

download_url "https://github.com/madler/zlib/archive/refs/tags/v${VERSION}.tar.gz" zlib.tar.gz
echo "1525952a0a567581792613a9723333d7f8cc20b87a81f920fb8bc7e3f2251428 zlib.tar.gz" | sha256sum -c -
mkdir -p zlib_src
tar xf zlib.tar.gz -C zlib_src --strip-components=1

mkdir -p zlib_build
cd zlib_build
../zlib_src/configure --prefix=${PREFIX}
make -j`nproc`
make install
ldconfig

popd
