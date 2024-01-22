#!/usr/bin/env bash
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

VERSION=1.9.5

mkdir -p ${HOME}/.build/${TARGET}/jsoncpp_{build,src}
pushd ${HOME}/.build/${TARGET}/jsoncpp_build

download_url "https://github.com/open-source-parsers/jsoncpp/archive/${VERSION}.tar.gz" jsoncpp.tar.gz
echo "1d06e044759b1e1a4cc4960189dd7e001a0a4389d7239a6d59295af995a553518e4e0337b4b4b817e70da5d9731a4c98655af90791b6287870b5ff8d73ad8873  jsoncpp.tar.gz" | sha512sum -c -
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

