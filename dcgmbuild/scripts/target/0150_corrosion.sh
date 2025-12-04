#!/usr/bin/env bash

set -ex

read -r _ URL SHA512SUM <<<$(grep '^corrosion ' $1)

curl --location --fail --output corrosion.tar.gz $URL
echo "$SHA512SUM corrosion.tar.gz" | sha512sum --check -

mkdir corrosion
tar xzf corrosion.tar.gz -C corrosion --strip-components=1

export PATH=$RUST_INSTALL_PREFIX/bin:$PATH

cmake -Scorrosion -Bbuild -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX \
    -DRust_COMPILER=$RUST_INSTALL_PREFIX/bin/rustc
cmake --build build --config Release
cmake --install build --config Release

rm -rf corrosion corrosion.tar.gz