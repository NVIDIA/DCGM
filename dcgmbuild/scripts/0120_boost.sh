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

source $(dirname $(realpath $0))/common_for_targets.sh

PKGDIR="${HOME}/.build/${TARGET}/boost"

mkdir -p ${PKGDIR}
pushd ${PKGDIR}

VERSION="1.81.0"
VERSION_EXTENSION=$(echo "$VERSION" | sed 's/\./_/g')
SHA512SUM="8f18972314e8dd5c952825fc52ca49d17b0e0b31db12bcc1cd0ba42c2d71c4f6ce5f5062fdbb65db029ec2c58ca93a32c32d0cdce62329556200dc8650a03fbf"
download_url "https://boostorg.jfrog.io/artifactory/main/release/${VERSION}/source/boost_${VERSION_EXTENSION}.tar.gz" boost.tar.gz
echo "${SHA512SUM} boost.tar.gz" | sha512sum -c -

tar xf boost.tar.gz -C ${PKGDIR} --strip-components=1

cat <<EOF > ${PKGDIR}/user-config.jam
using gcc : : /opt/cross/bin/${TARGET}-gcc ;
EOF

sed -i 's!#include <limits>!#include <limits>\n#include <utime.h>\n!' ${PKGDIR}/libs/filesystem/src/operations.cpp

./bootstrap.sh --prefix=/opt/cross/${TARGET} --without-libraries=regex,mpi,graph,graph_parallel,python,contract

sed -i "s!using gcc ;!using gcc : : /opt/cross/bin/${TARGET}-gcc ;!" ${PKGDIR}/project-config.jam

./b2 --prefix=/opt/cross/${TARGET} --layout=system --without-python cxxflags=-fPIC cflags=-fPIC variant=release link=static install

popd

