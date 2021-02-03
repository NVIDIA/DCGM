#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=11.0.2
mkdir -p ${HOME}/.build/ripgrep
pushd ${HOME}/.build/

download_url "https://github.com/BurntSushi/ripgrep/releases/download/${VERSION}/ripgrep-${VERSION}-x86_64-unknown-linux-musl.tar.gz" ripgrep.tar.gz
echo "384b284ca1b57cb2a42f89573db2eede51c9e3fb297c5de096fda683bf8de81ac5841fad120b321d076e8560c0aaaf052018651c792cf9fce49bb6baaebf288f  ripgrep.tar.gz" | sha512sum -c -

tar xzf ripgrep.tar.gz -C ripgrep --strip-components=1

cp -a ripgrep/rg /usr/local/bin/

popd