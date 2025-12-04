# syntax=docker/dockerfile:1.4
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG BASE_IMAGE=dcgm/toolchain-x86_64:latest

FROM $BASE_IMAGE AS builder

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt install --quiet --assume-yes autoconf \
                                     automake \
                                     libtool; \
    apt autoremove --purge --quiet --assume-yes; \
    apt clean --quiet --assume-yes; \
    rm -rf /var/lib/apt/lists/*

COPY cmake/$TARGET-toolchain.cmake /tmp/$TARGET.cmake

ENV CC=/opt/cross/bin/$TARGET-gcc
ENV CPP=/opt/cross/bin/$TARGET-cpp
ENV CXX=/opt/cross/bin/$TARGET-g++
ENV LD=/opt/cross/bin/$TARGET-ld
ENV AS=/opt/cross/bin/$TARGET-as

ENV CROSS_COMPILATION_SYSROOT=/tmp/$TARGET
ENV CMAKE_INSTALL_PREFIX=$CROSS_COMPILATION_SYSROOT/usr/local
ENV CMAKE_BUILD_TYPE=RelWithDebInfo
ENV CMAKE_TOOLCHAIN_FILE=/tmp/$TARGET.cmake
ENV RUST_INSTALL_PREFIX=/tmp/rust

COPY scripts/target /tmp/scripts/target
ARG JOBS
RUN set -ex; \
    echo $CC; \
    $CC --version; \
    mkdir --parents $CMAKE_INSTALL_PREFIX; \
    export MAKEFLAGS="--jobs=${JOBS:-$(nproc)}"; \
    export RUST_INSTALL_PREFIX=$RUST_INSTALL_PREFIX; \
    find /tmp/scripts/target -name '*.sh' | sort | while read -r SCRIPT; \
    do $SCRIPT /tmp/scripts/target/urls.txt; \
    done;

ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY --from=builder --chown=root:root /tmp/$TARGET /opt/cross/$TARGET/sysroot
COPY --from=builder --chown=root:root /tmp/rust/ /usr/local/

ENV DCGM_BUILD_INSIDE_DOCKER=1

RUN bash -c "set -ex && \
cargo install --root=/usr/local/ --force --locked cargo-auditable@0.6.7 && \
cargo install --root=/usr/local/ --force --locked cargo-audit@0.21.2 && \
cargo install --root=/usr/local/ --force --locked cargo-about@0.7.1 && \
cargo install --root=/usr/local/ --force --locked cargo-deny@0.18.3 && \
cargo install --root=/usr/local/ --force --locked cargo-make@0.37.24 && \
cargo install --root=/usr/local/ --force --locked cargo-vet@0.10.1 && \
cargo install --root=/usr/local/ --force --locked bindgen-cli@0.72.0 && \
cargo install --root=/usr/local/ --force --locked bacon && \
rm -rf $HOME/.cargo"

#ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="/opt/cross/bin/x86_64-linux-gnu-gcc"
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="/opt/cross/bin/aarch64-linux-gnu-gcc"
RUN bash -c 'echo "export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/opt/cross/bin/aarch64-linux-gnu-gcc" >> /etc/bash.bashrc'
RUN bash -c 'echo "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/opt/cross/bin/aarch64-linux-gnu-gcc" >> /etc/environment'
RUN set -ex; \
    echo "Debug: $TARGET"; \
    if [ "$TARGET" = "x86_64-linux-gnu" ]; then \
        echo "export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/opt/cross/bin/x86_64-linux-gnu-gcc" >> /etc/bash.bashrc; \
        echo "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/opt/cross/bin/x86_64-linux-gnu-gcc" >> /etc/environment; \
    fi
