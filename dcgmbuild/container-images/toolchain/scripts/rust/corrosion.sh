#!/usr/bin/env bash

set -ex

read -r _ URL SHA512SUM <<<$(grep '^corrosion ' $1)

curl --location --fail --output corrosion.tar.gz --retry 5 $URL
echo "$SHA512SUM corrosion.tar.gz" | sha512sum --check -

mkdir corrosion
tar xzf corrosion.tar.gz -C corrosion --strip-components=1

unset CMAKE_TOOLCHAIN_FILE
export PATH=$RUST_INSTALL_PREFIX/bin:$PATH

cmake -Scorrosion -Bcorrosion/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DRust_COMPILER=$RUST_INSTALL_PREFIX/bin/rustc
cmake --build corrosion/build
cmake --install corrosion/build

rm -rf corrosion corrosion.tar.gz
