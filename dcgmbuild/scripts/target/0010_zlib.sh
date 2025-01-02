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

read -r _ URL SHA512SUM <<<$(grep '^zlib ' $1)

curl --location --fail --output zlib.tar.gz $URL
echo "$SHA512SUM zlib.tar.gz" | sha512sum --check -

mkdir --parents zlib/{src,build}
tar xf zlib.tar.gz -C zlib/src --strip-components=1

pushd zlib/build

export CFLAGS="${CFLAGS:-} -fPIC -Wno-error"
../src/configure --prefix=$CMAKE_INSTALL_PREFIX

make
make install
popd

rm -rf zlib zlib.tar.gz
