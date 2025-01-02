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

read -r _ URL SHA512SUM <<<$(grep '^boost ' $1)

curl --location --fail --output boost.tar.gz $URL
echo "$SHA512SUM boost.tar.gz" | sha512sum --check -

mkdir --parents boost/src
tar xf boost.tar.gz -C boost/src --strip-components=1

cat <<EOF > boost/src/user-config.jam
using gcc : $ARCHITECTURE : $CC ;
EOF

pushd boost/src
./bootstrap.sh --prefix=$CMAKE_INSTALL_PREFIX \
               --without-libraries=regex,mpi,graph,graph_parallel,python,contract

CFLAGS="${CFLAGS:+$CFLAGS }-fPIC"
CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-fPIC"

#statx is only available since glibc 2.28 and we are using 2.27
./b2 --user-config=./user-config.jam \
     --prefix=$CMAKE_INSTALL_PREFIX \
     --layout=system \
     --without-python \
     cflags=$CFLAGS \
     cxxflags=$CXXFLAGS \
     define=BOOST_FILESYSTEM_DISABLE_STATX \
     link=static \
     toolset=gcc-$ARCHITECTURE \
     variant=release \
     install
popd

rm -rf boost boost.tar.gz
