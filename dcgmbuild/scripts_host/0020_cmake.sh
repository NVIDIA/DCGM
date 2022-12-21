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
set -ex -o pipefail

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=3.21.3
CHKSUM=2b6d7a4f966a7b1c4b4a8dd11ffbdb073169522e424aae118e7c947edb186026bb078a7c926c3fb96cb27131a496f4be6fdc64d28abf3631469f5bc7c0ee9d30

mkdir -p ${HOME}/.build/cmake
pushd ${HOME}/.build/cmake

download_url "https://github.com/Kitware/CMake/releases/download/v${VERSION}/cmake-${VERSION}-Linux-x86_64.sh" cmake.sh
echo "${CHKSUM} cmake.sh" | sha512sum -c -

chmod +x cmake.sh
./cmake.sh --prefix=/usr --skip-license

/usr/bin/cmake --version | grep -i ${VERSION}

if [ "$?" != "0" ]; then
    echo "Failed to verify cmake version" >&2
    exit 1
fi
