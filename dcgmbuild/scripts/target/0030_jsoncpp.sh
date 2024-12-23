#!/usr/bin/env bash
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

read -r _ URL SHA512SUM <<<$(grep '^jsoncpp ' $1)

curl --location --fail --output jsoncpp.tar.gz $URL
echo "$SHA512SUM jsoncpp.tar.gz" | sha512sum --check -

mkdir --parents jsoncpp/src
tar xf jsoncpp.tar.gz -C jsoncpp/src --strip-components=1

cmake \
    -B jsoncpp/build \
    -S jsoncpp/src \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    -D BUILD_OBJECT_LIBS=OFF \
    -D BUILD_STATIC_LIBS=ON \
    -D JSONCPP_WITH_CMAKE_PACKAGE=ON \
    -D JSONCPP_WITH_EXAMPLE=OFF \
    -D JSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
    -D JSONCPP_WITH_STRICT_ISO=OFF \
    -D JSONCPP_WITH_TESTS=OFF \

cmake --build jsoncpp/build
cmake --install jsoncpp/build

rm -rf jsoncpp jsoncpp.tar.gz
