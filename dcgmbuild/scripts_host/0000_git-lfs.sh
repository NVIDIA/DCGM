#!/usr/bin/env bash

set -ex -o pipefail

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=2.13.3

mkdir -p ${HOME}/.build/git-lfs
pushd ${HOME}/.build/git-lfs

curl -fL https://github.com/git-lfs/git-lfs/releases/download/v${VERSION}/git-lfs-linux-amd64-v${VERSION}.tar.gz -o git-lfs.tar.gz
echo "110906644dd558705b996271066c18cc3e017035ceecf6dcea8a600691914513204f25a6d1549d20ca398d9bab78993d08ef66cc1744d1df8c74fadbdec965e7  git-lfs.tar.gz" | sha512sum -c -
tar xf git-lfs.tar.gz
./install.sh

popd
