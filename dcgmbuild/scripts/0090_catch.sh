#!/usr/bin/env bash

set -ex -o nounset

source $(dirname $(realpath ${0}))/common_for_targets.sh

PKGNAME=catch2
PKGVER=v2.9.2
PKG="$PKGNAME-$PKGVER"
PKGDIR="${HOME}/.build/${TARGET}/${PKG}"
BUILDDIR="${PKGDIR}/build"
SRCDIR="${PKGDIR}/src"
SOURCE="https://github.com/catchorg/Catch2/archive/$PKGVER.tar.gz"
SHA512SUM="06430322dbeb637902f3bdc1c4df04e2525bc3ad9aea47aaf284b311401f26f489092971a2822d5a54041ef1d01d1b1bda3eedea2ba5041ae89903d8e56db121"

mkdir -p "$BUILDDIR"
mkdir -p "$SRCDIR"

pushd "$PKGDIR"

download_url "${SOURCE}" "$PKG.tar.gz"

echo "$SHA512SUM $PKG.tar.gz" > "$PKG.tar.gz.sha512sum"
sha512sum -c "$PKG.tar.gz.sha512sum"

tar xf "$PKG.tar.gz" -C "$SRCDIR" --strip-components=1

popd

pushd "$BUILDDIR"
cmake "$SRCDIR" -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_TESTING=OFF
cmake --build . --target install
popd
