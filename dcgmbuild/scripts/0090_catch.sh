#!/usr/bin/env bash

set -ex -o nounset

source $(dirname $(realpath ${0}))/common_for_targets.sh

PKGNAME=catch2
PKGVER=v2.13.9
PKG="$PKGNAME-$PKGVER"
PKGDIR="${HOME}/.build/${TARGET}/${PKG}"
BUILDDIR="${PKGDIR}/build"
SRCDIR="${PKGDIR}/src"
SOURCE="https://github.com/catchorg/Catch2/archive/$PKGVER.tar.gz"
SHA512SUM="4a254a20a1d916c14ffa072daa3d371d9ad4b5eb4d3c9257301c7c2ae9171116701cca2438a66ab731604c5b7a9cf4336197a31e32b8ca9bcf93db64bbba344b"

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
