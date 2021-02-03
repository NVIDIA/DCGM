#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

mkdir -p ${HOME}/.build/clang
pushd ${HOME}/.build/clang

CLANG_VERSION=9.0.0
OS_DISTRIBUTION=$(lsb_release -is | awk '{print tolower($0)}')
OS_RELEASE=$(lsb_release -rs)
OS_ARCH=$(uname -m)

#curl -fL http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
download_url "http://releases.llvm.org/${CLANG_VERSION}/clang+llvm-${CLANG_VERSION}-${OS_ARCH}-linux-gnu-${OS_DISTRIBUTION}-${OS_RELEASE}.tar.xz" clang.tar.gz
echo "863d62ff7b5a17a36f4787b37d247b865cccb86423cfd667bba27dd0d7469a288ba12fbdfa0c8b1875ff7a810614ad5ae2b940ce623e437849a93f16ef4aa5f4  clang.tar.gz" | sha512sum -c -
tar xf clang.tar.gz -C /usr --strip-components=1

popd
