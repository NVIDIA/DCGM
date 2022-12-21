#!/usr/bin/env bash
#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=13.0.0
mkdir -p ${HOME}/.build/ripgrep
pushd ${HOME}/.build/

download_url "https://github.com/BurntSushi/ripgrep/releases/download/${VERSION}/ripgrep-${VERSION}-x86_64-unknown-linux-musl.tar.gz" ripgrep.tar.gz
echo "cdc18bd31019fc7b8509224c2f52b230be33dee36deea2e4db1ee8c78ace406c7cd182814d056f4ce65ee533290a674822432777b61c2b4bc8cc4a4ea107cfde  ripgrep.tar.gz" | sha512sum -c -

tar xzf ripgrep.tar.gz -C ripgrep --strip-components=1

cp -a ripgrep/rg /usr/local/bin/

popd