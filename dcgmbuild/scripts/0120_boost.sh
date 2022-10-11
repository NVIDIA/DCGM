#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_targets.sh

PKGDIR="${HOME}/.build/${TARGET}/boost"

mkdir -p ${PKGDIR}
pushd ${PKGDIR}

VERSION="1.80.0"
VERSION_EXTENSION=$(echo "$VERSION" | sed 's/\./_/g')
SHA512SUM="f62aba05b1b6864e0403f20d06e64da567cd81825d0c75f8aa12004658313747d94e4259879f74975d3a19561fa202cf926ebb523c5e83dc1034076483902276"

download_url "https://boostorg.jfrog.io/artifactory/main/release/${VERSION}/source/boost_${VERSION_EXTENSION}.tar.gz" boost.tar.gz
echo "${SHA512SUM} boost.tar.gz" | sha512sum -c -

tar xf boost.tar.gz -C ${PKGDIR} --strip-components=1

cat <<EOF > ${PKGDIR}/user-config.jam
using gcc : : /opt/cross/bin/${TARGET}-gcc ;
EOF

sed -i 's!#include <limits>!#include <limits>\n#include <utime.h>\n!' ${PKGDIR}/libs/filesystem/src/operations.cpp

./bootstrap.sh --prefix=/opt/cross/${TARGET} --without-libraries=regex,mpi,graph,graph_parallel,python

sed -i "s!using gcc ;!using gcc : : /opt/cross/bin/${TARGET}-gcc ;!" ${PKGDIR}/project-config.jam

./b2 --prefix=/opt/cross/${TARGET} --layout=system --without-python cxxflags=-fPIC cflags=-fPIC variant=release link=static install

popd

