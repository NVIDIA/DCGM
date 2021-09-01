#!/usr/bin/env bash

set -ex -o pipefail -o nounset

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=1.15

mkdir -p ${HOME}/.build/lcov
pushd ${HOME}/.build/lcov

download_url "https://github.com/linux-test-project/lcov/archive/v${VERSION}.tar.gz" lcov.tar.gz
echo "0612853a39ba484c8ecc540412c5a85d105a51278a81fad0831f0ede263870b235d9bc702d7958931a5e7e5867060d0d9d277576d5acae8ebb22d0214005aa88 lcov.tar.gz" | sha512sum -c -
tar xf lcov.tar.gz -C ${HOME}/.build/lcov --strip-components=1

make install

popd
