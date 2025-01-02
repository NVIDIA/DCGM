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

read -r _ URL SHA512SUM <<<$(grep '^yaml-cpp ' $1)

curl --location --fail --output yaml-cpp.tar.gz $URL
echo "$SHA512SUM yaml-cpp.tar.gz" | sha512sum --check -

mkdir --parents yaml-cpp/src
tar xf yaml-cpp.tar.gz -C yaml-cpp/src --strip-components=1

cmake \
    -S yaml-cpp/src \
    -B yaml-cpp/build \
    -D BUILD_SHARED_LIBS=OFF \
    -D BUILD_TESTING=OFF \
    -D CMAKE_CXX_FLAGS='-Wno-unused-variable' \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON  \
    -D YAML_CPP_FORMAT_SOURCE=OFF

cmake --build yaml-cpp/build
cmake --install yaml-cpp/build

rm -rf yaml-cpp yaml-cpp.tar.gz
