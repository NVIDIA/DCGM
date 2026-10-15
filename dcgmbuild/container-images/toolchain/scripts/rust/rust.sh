#!/usr/bin/env bash

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -ex

HOST_TRIPLE=x86_64-unknown-linux-gnu
RUST_TARGET=$ARCHITECTURE-unknown-linux-gnu
RUST_TARGETS=$HOST_TRIPLE

if [[ $RUST_TARGET != "$HOST_TRIPLE" ]]; then
    RUST_TARGETS+=",$RUST_TARGET"
fi

mkdir rust
tar xf /tmp/downloads/rust.tar.xz -C rust --strip-components=1
cd rust

# Use Rust's distribution profile, but install only the tools needed by the build image.
cat > bootstrap.toml <<EOF
profile = "dist"
change-id = "ignore"

[build]
profiler = true
tools = [
    "cargo",
    "clippy",
    "rustdoc",
    "rustfmt",
    "rust-analyzer",
    "rust-analyzer-proc-macro-srv",
    "src",
    "wasm-component-ld",
    "llvm-bitcode-linker",
]

[install]
prefix = "$RUST_INSTALL_PREFIX"

[rust]
codegen-backends = ["llvm"]
llvm-bitcode-linker = true
llvm-tools = true

[target.$HOST_TRIPLE]
cc = "/usr/bin/clang"
cxx = "/usr/bin/clang++"
linker = "/usr/bin/clang"
llvm-libunwind = "in-tree"
EOF

if [[ $ARCHITECTURE != "x86_64" ]]; then
cat >> bootstrap.toml <<EOF

[target.$RUST_TARGET]
cc = "/opt/cross/bin/$TARGET-gcc"
cxx = "/opt/cross/bin/$TARGET-g++"
ar = "/opt/cross/bin/$TARGET-ar"
linker = "/opt/cross/bin/$TARGET-gcc"
llvm-libunwind = "in-tree"
EOF
fi

unset CMAKE_TOOLCHAIN_FILE
# Observed: Clang 22 can crash sporadically when compiling Cargo's bundled SQLite at -O3.
# Limit only bootstrap's C dependencies to -O2; Rust and LLVM keep the dist profile's settings.
export CFLAGS_x86_64_unknown_linux_gnu=-O2
if [[ $RUST_TARGET == "aarch64-unknown-linux-gnu" ]]
then
    # profiler_builtins uses legacy __sync atomics whose GCC outline helpers are not provided by
    # Rust's compiler_builtins. Keep the profiler runtime self-contained instead of linking libgcc.
    export CFLAGS_aarch64_unknown_linux_gnu=-mno-outline-atomics
fi
RUST_VERSION=$(<src/version)
./x.py install \
    --build "$HOST_TRIPLE" \
    --host "$HOST_TRIPLE" \
    --target "$RUST_TARGETS"

RUSTC=$RUST_INSTALL_PREFIX/bin/rustc
COMPONENTS=$RUST_INSTALL_PREFIX/lib/rustlib/components

"$RUSTC" --version | grep -q "^rustc $RUST_VERSION "
"$RUSTC" -vV | grep -qx "host: $HOST_TRIPLE"
test -d "$("$RUSTC" --target "$RUST_TARGET" --print target-libdir)"
test -f "$COMPONENTS"
if grep -q '^rustc-dev' "$COMPONENTS"
then
    echo "ERROR: rustc-dev was installed unexpectedly" >&2
    exit 1
fi

cd ..
rm -rf rust

mkdir -p /.cargo
cat <<EOF > /.cargo/config.toml
[target.$ARCHITECTURE-unknown-linux-gnu]
linker = "/opt/cross/bin/$ARCHITECTURE-linux-gnu-gcc"
EOF

if [[ $ARCHITECTURE != "x86_64" ]]
then
cat <<EOF >> /.cargo/config.toml
runner = "/usr/bin/qemu-$ARCHITECTURE -L /opt/cross/$ARCHITECTURE-linux-gnu/sysroot"

[target.x86_64-unknown-linux-gnu]
linker = "/usr/bin/clang"
EOF
fi

PROBE_DIR=$(mktemp -d)
mkdir -p "$PROBE_DIR/src"

cat > "$PROBE_DIR/Cargo.toml" <<EOF
[package]
name = "rust-unwind-probe"
version = "0.0.0"
edition = "2024"
publish = false
EOF

cat > "$PROBE_DIR/src/main.rs" <<'EOF'
fn assert_unwinds() {
    assert!(std::panic::catch_unwind(|| panic!("unwind probe")).is_err());
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    assert_unwinds();
}

#[cfg(test)]
mod tests {
    #[test]
    fn panic_unwinds() {
        std::panic::set_hook(Box::new(|_| {}));
        super::assert_unwinds();
    }
}
EOF

IFS=',' read -r -a RUST_TARGET_LIST <<< "$RUST_TARGETS"
for RUST_PROBE_TARGET in "${RUST_TARGET_LIST[@]}"
do
    RUST_PROBE_LINKER=/usr/bin/clang
    if [[ $RUST_PROBE_TARGET != "$HOST_TRIPLE" ]]
    then
        RUST_PROBE_LINKER=/opt/cross/bin/$TARGET-gcc
    fi

    RUSTFLAGS="-C instrument-coverage" \
    CARGO_TARGET_DIR="$PROBE_DIR/target" \
        "$RUST_INSTALL_PREFIX/bin/cargo" build \
        --quiet \
        --manifest-path "$PROBE_DIR/Cargo.toml" \
        --config "target.$RUST_PROBE_TARGET.linker=\"$RUST_PROBE_LINKER\"" \
        --target "$RUST_PROBE_TARGET"

    PROBE_BINARY="$PROBE_DIR/target/$RUST_PROBE_TARGET/debug/rust-unwind-probe"
    PROBE_DYNAMIC=$(/usr/bin/llvm-readelf --dynamic "$PROBE_BINARY")
    if grep -q libgcc_s <<< "$PROBE_DYNAMIC"
    then
        echo "ERROR: $RUST_PROBE_TARGET Rust binary depends on libgcc_s" >&2
        printf '%s\n' "$PROBE_DYNAMIC" >&2
        exit 1
    fi

    RUST_PROBE_RUN_PROFILE="$PROBE_DIR/$RUST_PROBE_TARGET-run.profraw"
    LLVM_PROFILE_FILE="$RUST_PROBE_RUN_PROFILE" \
    RUSTFLAGS="-C instrument-coverage" \
    CARGO_TARGET_DIR="$PROBE_DIR/target" \
        "$RUST_INSTALL_PREFIX/bin/cargo" run \
        --quiet \
        --manifest-path "$PROBE_DIR/Cargo.toml" \
        --config "target.$RUST_PROBE_TARGET.linker=\"$RUST_PROBE_LINKER\"" \
        --target "$RUST_PROBE_TARGET"

    test -s "$RUST_PROBE_RUN_PROFILE"

    RUST_PROBE_TEST_PROFILE="$PROBE_DIR/$RUST_PROBE_TARGET-test.profraw"
    LLVM_PROFILE_FILE="$RUST_PROBE_TEST_PROFILE" \
    RUSTFLAGS="-C instrument-coverage" \
    CARGO_TARGET_DIR="$PROBE_DIR/target" \
        "$RUST_INSTALL_PREFIX/bin/cargo" test \
        --verbose \
        --manifest-path "$PROBE_DIR/Cargo.toml" \
        --config "target.$RUST_PROBE_TARGET.linker=\"$RUST_PROBE_LINKER\"" \
        --target "$RUST_PROBE_TARGET"

    test -s "$RUST_PROBE_TEST_PROFILE"
done

rm -rf "$PROBE_DIR"
