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

VERSION=18
read -r _ URL SHA512SUM <<<$(grep '^clang ' $1)

export DEBIAN_FRONTEND=noninteractive
apt install --quiet --assume-yes software-properties-common gpg lsb-release wget

curl --location --fail --output llvm.sh $URL
echo "$SHA512SUM llvm.sh" | sha512sum --check -

chmod +x llvm.sh
./llvm.sh -n $(lsb_release -c | awk '{print $2}') $VERSION all

apt install --quiet --assume-yes \
            clang \
            clang-format \
            clang-tidy \
            clang-tools \
            python3-clang

find /usr/bin -regex-type posix-extended -regex "(clang|llvm).*-$VERSION$" \
| while read -r FILEPATH
do
    LINK=${FILEPATH//-*}
    update-alternatives --install $LINK $(basename $LINK) $FILEPATH 100
done

rm llvm.sh
