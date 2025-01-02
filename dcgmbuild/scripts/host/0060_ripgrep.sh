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

read -r _ URL SHA512SUM <<<$(grep '^ripgrep ' $1)

curl --location --fail --output ripgrep.tar.gz $URL
echo "$SHA512SUM ripgrep.tar.gz" | sha512sum --check -

mkdir ripgrep
tar xzf ripgrep.tar.gz -C ripgrep --strip-components=1

cp -a ripgrep/rg /usr/local/bin/
rm -rf ripgrep ripgrep.tar.gz
