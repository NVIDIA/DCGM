#!/usr/bin/env bash
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

read -r _ URL SHA512SUM <<<$(grep '^ccache ' $1)

curl --location --fail --output ccache.tar.xz --retry 5 $URL
echo "$SHA512SUM ccache.tar.xz" | sha512sum --check -

mkdir ccache
tar xf ccache.tar.xz --strip-components=1 -C ccache
make -C ccache install

rm -rf ccache.tar.xz ccache
