#!/usr/bin/env bash
set -ex -o pipefail

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=3.16.4

mkdir -p ${HOME}/.build/cmake
pushd ${HOME}/.build/cmake

download_url "https://github.com/Kitware/CMake/releases/download/v${VERSION}/cmake-${VERSION}-Linux-x86_64.sh" cmake.sh
echo "96f3d3cbbe7c74426ead189e06932363ff1d301ece640dff504345999cc2438e074c48d282a5f22fe1756c7efd63e9cabb4fd6138940a59373fe5c00d4fb5681  cmake.sh" | sha512sum -c -

chmod +x cmake.sh
./cmake.sh --prefix=/usr --skip-license

/usr/bin/cmake --version | grep -i ${VERSION}

if [ "$?" != "0" ]; then
    echo "Failed to verify cmake version" >&2
    exit 1
fi
