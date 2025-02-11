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

read -r _ URL SHA512SUM <<<$(grep '^plog ' $1)

curl --location --fail --output plog.tar.gz $URL
echo "$SHA512SUM plog.tar.gz" | sha512sum --check -

mkdir --parents plog/src
tar xf plog.tar.gz -C plog/src --strip-components=1

cmake -S plog/src -B plog/build -D PLOG_BUILD_SAMPLES=OFF
cmake --install plog/build

rm -rf plog plog.tar.gz
