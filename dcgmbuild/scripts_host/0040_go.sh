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

mkdir -p /usr/build/
pushd /usr/build

VERSION=1.17.1
CHKSUM=8addbda7af20c5fb415e5f8402a3ec5edd3d61dca7861ac0b7d94f626ecd34896ae145b8eddabfdd1041a7ebe85cb7d8c75edf10c70b518ad5a23b451d8359d9

download_url "https://dl.google.com/go/go${VERSION}.linux-amd64.tar.gz" go.tar.gz
echo "${CHKSUM} go.tar.gz" | sha512sum -c -
tar xf go.tar.gz -C /usr/local/

export PATH=/usr/local/go/bin:$PATH

popd