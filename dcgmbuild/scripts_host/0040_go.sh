#!/usr/bin/env bash

set -ex

source $(dirname $(realpath $0))/common_for_host.sh

mkdir -p /usr/build/
pushd /usr/build

VERSION=1.13

download_url "https://dl.google.com/go/go${VERSION}.linux-amd64.tar.gz" go.tar.gz
echo "15306a63e5ebfa679c44254f5776e7265ec4c675df3dbf1ca8d41bb51aef7fc951267c7ab4bf376a2b586fc55c1d060e0ebd5dd7feb9bb1ed9b7cd4dd6ffd67d  go.tar.gz" | sha512sum -c -
tar xf go.tar.gz -C /usr/local/

export PATH=/usr/local/go/bin:$PATH

popd