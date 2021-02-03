#!/usr/bin/env bash

set -ex

source $(dirname $(realpath ${0}))/common_for_targets.sh

PKGNAME=plog
PKGVER=1.1.4
PKG="$PKGNAME-$PKGVER"
PKGDIR="${HOME}/.build/${TARGET}/${PKG}"
SOURCE="https://github.com/SergiusTheBest/plog/archive/$PKGVER.tar.gz"
SHA512SUM="7af75af8343460d62e04cc0c27d4cf86373b136df73d2312d19a2e57fa309e916cef8625b8eed1b7270b93aa5d1ff27aee6edb74beb138e3a21c06a3c3debb41"

mkdir -p "$PKGDIR"

pushd "$PKGDIR"
download_url "${SOURCE}" "${PKG}.tar.gz"

echo "$SHA512SUM $PKG.tar.gz" > "$PKG.tar.gz.sha512sum"
sha512sum -c "$PKG.tar.gz.sha512sum"

tar xf "$PKG.tar.gz" -C "$PKGDIR" --strip-components=1

# plog is header-only
cp -r include/plog -t "$PREFIX/include/"
popd
