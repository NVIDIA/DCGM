#!/usr/bin/env bash
set -ex -o pipefail

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=3.20.4

mkdir -p ${HOME}/.build/cmake
pushd ${HOME}/.build/cmake

download_url "https://github.com/Kitware/CMake/releases/download/v${VERSION}/cmake-${VERSION}-Linux-x86_64.sh" cmake.sh
echo "7b71845d99f07c3bce5922befaa7f32f979e36a517f813cbf79b8b442aec457f44826fd1f9beb9a939ef183880e69027e3aa71ee0b6ac223f824237048202e75  cmake.sh" | sha512sum -c -

chmod +x cmake.sh
./cmake.sh --prefix=/usr --skip-license

/usr/bin/cmake --version | grep -i ${VERSION}

if [ "$?" != "0" ]; then
    echo "Failed to verify cmake version" >&2
    exit 1
fi
