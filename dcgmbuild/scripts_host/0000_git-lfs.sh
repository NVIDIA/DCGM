#!/usr/bin/env bash

set -ex -o pipefail

source $(dirname $(realpath $0))/common_for_host.sh

VERSION=2.8.0

mkdir -p ${HOME}/.build/git-lfs
pushd ${HOME}/.build/git-lfs

curl -fL https://github.com/git-lfs/git-lfs/releases/download/v${VERSION}/git-lfs-linux-amd64-v${VERSION}.tar.gz -o git-lfs.tar.gz
echo "acca1360df73910720a0e85cc4403c0959f8a9279a13e28e003d5463d7066b502434712a9e8f15fe2d7619d13ad01e03d8860b7d4ac964fbe5d77804c2e5e873  git-lfs.tar.gz" | sha512sum -c -
tar xf git-lfs.tar.gz
./install.sh

popd
