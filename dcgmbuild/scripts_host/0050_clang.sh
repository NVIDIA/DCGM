#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

mkdir -p ${HOME}/.build/clang
pushd ${HOME}/.build/clang

CLANG_VERSION=13.0.0
OS_DISTRIBUTION=$(lsb_release -is | awk '{print tolower($0)}')
OS_RELEASE=$(lsb_release -rs)
OS_ARCH=$(uname -m)

#curl -fL https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.0/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
download_url "https://github.com/llvm/llvm-project/releases/download/llvmorg-${CLANG_VERSION}/clang+llvm-${CLANG_VERSION}-${OS_ARCH}-linux-gnu-${OS_DISTRIBUTION}-${OS_RELEASE}.tar.xz" clang.tar.xz
echo "4fe7a52c6e1dc2fd584ec83ac92401b3bee2c181e69ea93cd66ff321b56364e58758307da8745ae05db31fabdbf6e537b9cc3ea5f192ef40ca27ddb0d1a86b11  clang.tar.xz" | sha512sum -c -
tar xf clang.tar.xz -C /usr --strip-components=1

popd
