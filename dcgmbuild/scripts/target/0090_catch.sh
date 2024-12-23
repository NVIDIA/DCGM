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

read -r _ URL SHA512SUM <<<$(grep '^catch2 ' $1)

curl --location --fail --output catch2.tar.gz $URL
echo "$SHA512SUM catch2.tar.gz" | sha512sum --check -

mkdir --parents catch2/src
tar xf catch2.tar.gz -C catch2/src --strip-components=1

# Unit tests are not installed for client consumption. As such, it unnecessary
# to build it as a static library or with position independent code
cmake \
    -S catch2/src \
    -B catch2/build \
    -D BUILD_TESTING=OFF \
    -D CATCH_INSTALL_DOCS=OFF

cmake --build catch2/build
cmake --install catch2/build

rm -rf catch2 catch2.tar.gz
