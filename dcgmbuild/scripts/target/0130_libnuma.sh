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

read -r _ URL SHA512SUM <<<$(grep '^libnuma ' $1)

curl --location --fail --output libnuma.tar.gz $URL
echo "$SHA512SUM libnuma.tar.gz" | sha512sum --check -

mkdir --parents libnuma/src
tar xf libnuma.tar.gz -C libnuma/src --strip-components=1

mkdir --parents $CMAKE_INSTALL_PREFIX/include
install --mode 644 libnuma/src/numa.h $CMAKE_INSTALL_PREFIX/include/

mkdir --parents $CMAKE_INSTALL_PREFIX/share/licenses/libnuma
install --mode 644 libnuma/src/LICENSE.LGPL2.1 $CMAKE_INSTALL_PREFIX/share/licenses/libnuma/

rm -rf libnuma libnuma.tar.gz
