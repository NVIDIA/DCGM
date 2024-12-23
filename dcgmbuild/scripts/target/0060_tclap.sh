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

read -r _ URL COMMIT_SHA <<<$(grep '^tclap ' $1)
mkdir --parents tclap/src

pushd tclap/src
git init
git remote add origin $URL
git fetch --depth 1 origin $COMMIT_SHA
git checkout FETCH_HEAD
popd

cmake \
    -S tclap/src \
    -B tclap/build \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_UNITTESTS=OFF \
    -D BUILD_DOC=OFF

cmake --install tclap/build --component lib

rm -rf tclap tclap.tar.gz
