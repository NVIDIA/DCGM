#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
set -ex -o pipefail -o nounset

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=1.15

mkdir -p ${HOME}/.build/lcov
pushd ${HOME}/.build/lcov

download_url "https://github.com/linux-test-project/lcov/archive/v${VERSION}.tar.gz" lcov.tar.gz
echo "0612853a39ba484c8ecc540412c5a85d105a51278a81fad0831f0ede263870b235d9bc702d7958931a5e7e5867060d0d9d277576d5acae8ebb22d0214005aa88 lcov.tar.gz" | sha512sum -c -
tar xf lcov.tar.gz -C ${HOME}/.build/lcov --strip-components=1

make install

popd
