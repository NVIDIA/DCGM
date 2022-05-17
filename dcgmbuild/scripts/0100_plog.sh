#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

PKGNAME=plog
PKGVER=1.1.5
PKG="$PKGNAME-$PKGVER"
PKGDIR="${HOME}/.build/${TARGET}/${PKG}"
SOURCE="https://github.com/SergiusTheBest/plog/archive/$PKGVER.tar.gz"
SHA512SUM="c16b428e1855c905c486130c8610d043962bedc2b40d1d986c250c8f7fd7139540164a3cbb408ed08298370aa150d5937f358c13ccae2728ce8ea47fa897fd0b"

mkdir -p "$PKGDIR"

pushd "$PKGDIR"
download_url "${SOURCE}" "${PKG}.tar.gz"

echo "$SHA512SUM $PKG.tar.gz" > "$PKG.tar.gz.sha512sum"
sha512sum -c "$PKG.tar.gz.sha512sum"

tar xf "$PKG.tar.gz" -C "$PKGDIR" --strip-components=1

# plog is header-only
cp -r include/plog -t "$PREFIX/include/"
popd
