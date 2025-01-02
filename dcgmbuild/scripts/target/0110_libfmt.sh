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

read -r _ URL SHA512SUM <<<$(grep '^fmt ' $1)

curl --location --fail --output fmt.tar.gz $URL
echo "$SHA512SUM fmt.tar.gz" | sha512sum --check -

mkdir --parents fmt/src
tar xf fmt.tar.gz -C fmt/src --strip-components=1

cmake \
    -S fmt/src \
    -B fmt/build \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    -D FMT_DOC=OFF \
    -D FMT_TEST=OFF

cmake --build fmt/build
cmake --install fmt/build

rm -rf fmt fmt.tar.gz
