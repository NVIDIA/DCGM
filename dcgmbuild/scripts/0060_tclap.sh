#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

VERSION=1.2.2

mkdir -p ${HOME}/.build/${TARGET}/tclap_{build,src}
pushd ${HOME}/.build/${TARGET}/tclap_build

download_url "https://astuteinternet.dl.sourceforge.net/project/tclap/tclap-${VERSION}.tar.gz" tclap.tar.gz
echo "516ec17f82a61277922bc8c0ed66973300bf42a738847fbbd2912c6405c34f94a13e47dc964854a5b26a9a9f1f518cce682ca54e769d6016851656c647866107  tclap.tar.gz" | sha512sum -c -
tar xzf tclap.tar.gz -C ${HOME}/.build/${TARGET}/tclap_src --strip-components=1

${HOME}/.build/${TARGET}/tclap_src/configure --disable-shared --prefix=${PREFIX} --host=${TARGET}
make -j`nproc`
make install

popd

