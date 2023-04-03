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
set -ex

source $(dirname $(realpath $0))/common_for_host.sh

mkdir -p ${HOME}/.build/clang
pushd ${HOME}/.build/clang

export DEBIAN_FRONTEND=noninteractive

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 15 all

find /usr/bin -iname 'clang*15' | while read -r file; do
    ln -f $file /usr/bin/$(basename $file | sed 's/-15//')
done

popd
