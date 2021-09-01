#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

mkdir -p ${HOME}/.build/clang
pushd ${HOME}/.build/clang

CLANG_VERSION=12.0.0
OS_DISTRIBUTION=$(lsb_release -is | awk '{print tolower($0)}')
OS_RELEASE=$(lsb_release -rs)
OS_ARCH=$(uname -m)

#curl -fL https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.0/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
download_url "https://github.com/llvm/llvm-project/releases/download/llvmorg-${CLANG_VERSION}/clang+llvm-${CLANG_VERSION}-${OS_ARCH}-linux-gnu-${OS_DISTRIBUTION}-${OS_RELEASE}.tar.xz" clang.tar.xz
echo "a65099aa46d8a030fb478c4e9a940838f5cef70905dbe3449e082478749da6fcddd8f4de7078850e187e8e557584fd3eda92401f8c6cc4c61f1f2d15917553b1  clang.tar.xz" | sha512sum -c -
tar xf clang.tar.xz -C /usr --strip-components=1

popd
