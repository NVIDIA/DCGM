#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=13.0.0
mkdir -p ${HOME}/.build/ripgrep
pushd ${HOME}/.build/

download_url "https://github.com/BurntSushi/ripgrep/releases/download/${VERSION}/ripgrep-${VERSION}-x86_64-unknown-linux-musl.tar.gz" ripgrep.tar.gz
echo "cdc18bd31019fc7b8509224c2f52b230be33dee36deea2e4db1ee8c78ace406c7cd182814d056f4ce65ee533290a674822432777b61c2b4bc8cc4a4ea107cfde  ripgrep.tar.gz" | sha512sum -c -

tar xzf ripgrep.tar.gz -C ripgrep --strip-components=1

cp -a ripgrep/rg /usr/local/bin/

popd