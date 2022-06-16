#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_targets.sh

PKGDIR="${HOME}/.build/${TARGET}/boost"

mkdir -p ${PKGDIR}
pushd ${PKGDIR}

VERSION="1.79.0"
VERSION_EXTENSION=$(echo "$VERSION" | sed 's/\./_/g')
SHA512SUM="ae2e7304fb808bd3a9e6c56bce05a3d0ad8ac98d0e901be4a02cf98530a8765989926ef9d85f7eaf5392a9301b84ed37bf802f70a0d0db72a2a53569c950fa46"

download_url "https://boostorg.jfrog.io/artifactory/main/release/${VERSION}/source/boost_${VERSION_EXTENSION}.tar.gz" boost.tar.gz
echo "${SHA512SUM} boost.tar.gz" | sha512sum -c -

HEADER_INSTALL_LOCATION="/opt/cross/${TARGET}/include/boost${VERSION_EXTENSION}"
mkdir -p ${HEADER_INSTALL_LOCATION}

tar xf boost.tar.gz -C ${PKGDIR} --strip-components=1
# copy the headers since it is headers only
cp -R ${PKGDIR}/boost ${HEADER_INSTALL_LOCATION}

popd

