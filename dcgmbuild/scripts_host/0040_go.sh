#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

mkdir -p /usr/build/
pushd /usr/build

VERSION=1.17

download_url "https://dl.google.com/go/go${VERSION}.linux-amd64.tar.gz" go.tar.gz
echo "7fccc832d68623146470c7a1ae03639675ff46b3d422bc6ba0b3f637b04581025d180ea51bd4444de9f68634ada732e1ea103681201cd79ce7ec274ec0e05846  go.tar.gz" | sha512sum -c -
tar xf go.tar.gz -C /usr/local/

export PATH=/usr/local/go/bin:$PATH

popd