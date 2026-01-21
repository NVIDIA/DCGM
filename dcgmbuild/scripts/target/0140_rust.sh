#!/usr/bin/env bash

set -ex

# First install Rust x86_64-unknown-linux-gnu for all targets.
# This is required to install host tools for cargo.

read -r _ URL SHA512SUM <<<$(grep '^rustc ' $1)

curl --location --fail --output rust.tar.xz $URL
echo "$SHA512SUM rust.tar.xz" | sha512sum --check -

mkdir rust_install
tar xf rust.tar.xz -C rust_install --strip-components=1

# This installs into /usr/local/bin
./rust_install/install.sh --prefix=$RUST_INSTALL_PREFIX

rm -rf rust_install rust.tar.xz

# Now install rust-std for the target if it's different from x86_64-unknown-linux-gnu
# The target location must match where the rustc above was installed

read -r _ URL SHA512SUM <<<$(grep "^rust-std-$TARGET " $1)

# Check if URL was found for this target
if [ -n "$URL" ]; then
    curl --location --fail --output rust-std.tar.xz $URL
    echo "$SHA512SUM rust-std.tar.xz" | sha512sum --check -

    mkdir rust-std
    tar xf rust-std.tar.xz -C rust-std --strip-components=1

    ./rust-std/install.sh --prefix=$RUST_INSTALL_PREFIX

    rm -rf rust-std rust-std.tar.xz
else
    echo "No rust-std URL found for target: $TARGET"
fi


# Now install rust-src (solely for the rust-analyzer proper work for now)
read -r _ URL SHA512SUM <<<$(grep "^rust-src " $1)

# Check if URL was found for this target
if [ -n "$URL" ]; then
    curl --location --fail --output rust-src.tar.xz $URL
    echo "$SHA512SUM rust-src.tar.xz" | sha512sum --check -

    mkdir rust-src
    tar xf rust-src.tar.xz -C rust-src --strip-components=1

    ./rust-src/install.sh --prefix=$RUST_INSTALL_PREFIX --components=rust-src

    rm -rf rust-src rust-src.tar.xz
else
    echo "No rust-src URL found for target: $TARGET"
fi
