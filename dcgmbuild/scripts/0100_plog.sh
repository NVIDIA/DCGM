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

source $(dirname $(realpath ${0}))/common_for_targets.sh

PKGNAME=plog
PKGVER=1.1.8
PKG="$PKGNAME-$PKGVER"
PKGDIR="${HOME}/.build/${TARGET}/${PKG}"
SOURCE="https://github.com/SergiusTheBest/plog/archive/$PKGVER.tar.gz"
SHA512SUM="09bf6e0cae7f20c1b42e68a174b4cd6a2fb8751db9758efb87449cbff48375708e43c147c72b7ed17fb9334acaf7802441f61578356284a8ed337fd886a45e79"

mkdir -p "$PKGDIR"

pushd "$PKGDIR"
download_url "${SOURCE}" "${PKG}.tar.gz"

echo "$SHA512SUM $PKG.tar.gz" > "$PKG.tar.gz.sha512sum"
sha512sum -c "$PKG.tar.gz.sha512sum"

tar xf "$PKG.tar.gz" -C "$PKGDIR" --strip-components=1

# plog is header-only
cp -r include/plog -t "$PREFIX/include/"
popd
