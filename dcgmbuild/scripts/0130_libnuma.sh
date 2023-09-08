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
source $(dirname $(realpath ${0}))/common_for_targets.sh

mkdir -p libnuma
pushd libnuma

PKG_AARCH64="http://ports.ubuntu.com/pool/main/n/numactl/libnuma-dev_2.0.16-1_arm64.deb"
PKG_PPC64LE="http://ports.ubuntu.com/pool/main/n/numactl/libnuma-dev_2.0.16-1_ppc64el.deb"
PKG_AMD64="http://archive.ubuntu.com/ubuntu/pool/main/n/numactl/libnuma-dev_2.0.16-1_amd64.deb"

case $TARGET in
  x86_64-linux-gnu)
    download_url ${PKG_AMD64} libnuma.deb
    ;;

  aarch64-linux-gnu)
    download_url ${PKG_AARCH64} libnuma.deb
    ;;

  powerpc64le-linux-gnu)
    download_url ${PKG_PPC64LE} libnuma.deb
    ;;

  *)
    die "Unknown architecture"
esac

dpkg -x libnuma.deb .
cp -a usr/include/numa.h /opt/cross/${TARGET}/include/
cp -a usr/lib/${TARGET}/libnuma.a /opt/cross/${TARGET}/lib/
mkdir /opt/cross/share/licenses/libnuma
cp -a usr/share/doc/libnuma-dev/copyright /opt/cross/share/licenses/libnuma/

popd
rm -rf libnuma