#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=2.1.12

mkdir ${HOME}/.build/${TARGET}/libevent_{build,src}

pushd ${HOME}/.build/${TARGET}/libevent_src

download_url "https://github.com/libevent/libevent/releases/download/release-${VERSION}-stable/libevent-${VERSION}-stable.tar.gz" libevent.tar.gz
echo "88d8944cd75cbe78bc4e56a6741ca67c017a3686d5349100f1c74f8a68ac0b6410ce64dff160be4a4ba0696ee29540dfed59aaf3c9a02f0c164b00307fcfe84f libevent.tar.gz" | sha512sum -c -
tar xzf libevent.tar.gz -C ${HOME}/.build/${TARGET}/libevent_src --strip-components=1

./autogen.sh
popd

pushd ${HOME}/.build/${TARGET}/libevent_build
CFLAGS="${CFLAGS:-} -Wno-cast-function-type -Wno-implicit-fallthrough -fPIC" \
    CXXFLAGS="${CXXFLAGS:-} -Wno-cast-function-type -Wno-implicit-fallthrough -fPIC" \
    ${HOME}/.build/${TARGET}/libevent_src/configure --disable-shared --disable-openssl --with-pic --prefix=${PREFIX} --host=${TARGET}
make -j`nproc`
make install

popd

