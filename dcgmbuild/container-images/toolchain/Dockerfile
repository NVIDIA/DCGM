# syntax=docker/dockerfile:1.6

# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG ARCHITECTURE=x86_64

FROM scratch AS downloads

ARG CORROSION_URL
ARG CROSSTOOL_URL
ARG CROSSTOOL_BINUTILS_URL
ARG CROSSTOOL_GCC_URL
ARG CROSSTOOL_GDB_URL
ARG CROSSTOOL_GETTEXT_URL
ARG CROSSTOOL_GLIBC_URL
ARG CROSSTOOL_GMP_URL
ARG CROSSTOOL_ISL_URL
ARG CROSSTOOL_LIBICONV_URL
ARG CROSSTOOL_LINUX_URL
ARG CROSSTOOL_MOLD_URL
ARG CROSSTOOL_MPC_URL
ARG CROSSTOOL_MPFR_URL
ARG CROSSTOOL_NCURSES_URL
ARG CROSSTOOL_ZLIB_URL
ARG CROSSTOOL_ZSTD_URL
ARG RUST_URL

ARG CORROSION_SHA256SUM
ARG CROSSTOOL_SHA256SUM
ARG CROSSTOOL_BINUTILS_SHA256SUM
ARG CROSSTOOL_GCC_SHA256SUM
ARG CROSSTOOL_GDB_SHA256SUM
ARG CROSSTOOL_GETTEXT_SHA256SUM
ARG CROSSTOOL_GLIBC_SHA256SUM
ARG CROSSTOOL_GMP_SHA256SUM
ARG CROSSTOOL_ISL_SHA256SUM
ARG CROSSTOOL_LIBICONV_SHA256SUM
ARG CROSSTOOL_LINUX_SHA256SUM
ARG CROSSTOOL_MOLD_SHA256SUM
ARG CROSSTOOL_MPC_SHA256SUM
ARG CROSSTOOL_MPFR_SHA256SUM
ARG CROSSTOOL_NCURSES_SHA256SUM
ARG CROSSTOOL_ZLIB_SHA256SUM
ARG CROSSTOOL_ZSTD_SHA256SUM
ARG RUST_SHA256SUM

ADD --checksum=${CORROSION_SHA256SUM}                  ${CORROSION_URL}                  /corrosion.tar.gz
ADD --checksum=${CROSSTOOL_SHA256SUM}                  ${CROSSTOOL_URL}                  /crosstool-ng.tar.gz
ADD --checksum=${CROSSTOOL_BINUTILS_SHA256SUM}         ${CROSSTOOL_BINUTILS_URL}         /crosstool-ng/binutils-2.42.tar.xz
ADD --checksum=${CROSSTOOL_GCC_SHA256SUM}              ${CROSSTOOL_GCC_URL}              /crosstool-ng/gcc-14.3.0.tar.xz
ADD --checksum=${CROSSTOOL_GDB_SHA256SUM}              ${CROSSTOOL_GDB_URL}              /crosstool-ng/gdb-14.2.tar.xz
ADD --checksum=${CROSSTOOL_GETTEXT_SHA256SUM}          ${CROSSTOOL_GETTEXT_URL}          /crosstool-ng/gettext-0.22.5.tar.xz
ADD --checksum=${CROSSTOOL_GLIBC_SHA256SUM}            ${CROSSTOOL_GLIBC_URL}            /crosstool-ng/glibc-2.27.tar.xz
ADD --checksum=${CROSSTOOL_GMP_SHA256SUM}              ${CROSSTOOL_GMP_URL}              /crosstool-ng/gmp-6.2.1.tar.xz
ADD --checksum=${CROSSTOOL_ISL_SHA256SUM}              ${CROSSTOOL_ISL_URL}              /crosstool-ng/isl-0.26.tar.xz
ADD --checksum=${CROSSTOOL_LIBICONV_SHA256SUM}         ${CROSSTOOL_LIBICONV_URL}         /crosstool-ng/libiconv-1.16.tar.gz
ADD --checksum=${CROSSTOOL_LINUX_SHA256SUM}            ${CROSSTOOL_LINUX_URL}            /crosstool-ng/linux-4.20.17.tar.xz
ADD --checksum=${CROSSTOOL_MOLD_SHA256SUM}             ${CROSSTOOL_MOLD_URL}             /crosstool-ng/v2.40.4.tar.gz
ADD --checksum=${CROSSTOOL_MPC_SHA256SUM}              ${CROSSTOOL_MPC_URL}              /crosstool-ng/mpc-1.3.1.tar.gz
ADD --checksum=${CROSSTOOL_MPFR_SHA256SUM}             ${CROSSTOOL_MPFR_URL}             /crosstool-ng/mpfr-4.2.2.tar.xz
ADD --checksum=${CROSSTOOL_NCURSES_SHA256SUM}          ${CROSSTOOL_NCURSES_URL}          /crosstool-ng/ncurses-6.4.tar.gz
ADD --checksum=${CROSSTOOL_ZLIB_SHA256SUM}             ${CROSSTOOL_ZLIB_URL}             /crosstool-ng/zlib-1.3.1.tar.xz
ADD --checksum=${CROSSTOOL_ZSTD_SHA256SUM}             ${CROSSTOOL_ZSTD_URL}             /crosstool-ng/zstd-1.5.5.tar.gz
ADD --checksum=${RUST_SHA256SUM}                       ${RUST_URL}                       /rust.tar.xz

FROM scratch AS cuda-downloads

ARG CUDA11_AARCH64_LINUX_GNU_URL
ARG CUDA11_CROSS_AARCH64_LINUX_GNU_URL
ARG CUDA11_X86_64_LINUX_GNU_URL
ARG CUDA12_AARCH64_LINUX_GNU_URL
ARG CUDA12_CROSS_AARCH64_LINUX_GNU_URL
ARG CUDA12_X86_64_LINUX_GNU_URL
ARG CUDA13_AARCH64_LINUX_GNU_URL
ARG CUDA13_CROSS_AARCH64_LINUX_GNU_URL
ARG CUDA13_X86_64_LINUX_GNU_URL

ARG CUDA11_AARCH64_LINUX_GNU_SHA256SUM
ARG CUDA11_CROSS_AARCH64_LINUX_GNU_SHA256SUM
ARG CUDA11_X86_64_LINUX_GNU_SHA256SUM
ARG CUDA12_AARCH64_LINUX_GNU_SHA256SUM
ARG CUDA12_CROSS_AARCH64_LINUX_GNU_SHA256SUM
ARG CUDA12_X86_64_LINUX_GNU_SHA256SUM
ARG CUDA13_AARCH64_LINUX_GNU_SHA256SUM
ARG CUDA13_CROSS_AARCH64_LINUX_GNU_SHA256SUM
ARG CUDA13_X86_64_LINUX_GNU_SHA256SUM

ADD --checksum=${CUDA11_AARCH64_LINUX_GNU_SHA256SUM}       ${CUDA11_AARCH64_LINUX_GNU_URL}       /cuda11-aarch64.deb
ADD --checksum=${CUDA11_CROSS_AARCH64_LINUX_GNU_SHA256SUM} ${CUDA11_CROSS_AARCH64_LINUX_GNU_URL} /cuda11-cross-aarch64.deb
ADD --checksum=${CUDA11_X86_64_LINUX_GNU_SHA256SUM}        ${CUDA11_X86_64_LINUX_GNU_URL}        /cuda11-x86_64.deb
ADD --checksum=${CUDA12_AARCH64_LINUX_GNU_SHA256SUM}       ${CUDA12_AARCH64_LINUX_GNU_URL}       /cuda12-aarch64.deb
ADD --checksum=${CUDA12_CROSS_AARCH64_LINUX_GNU_SHA256SUM} ${CUDA12_CROSS_AARCH64_LINUX_GNU_URL} /cuda12-cross-aarch64.deb
ADD --checksum=${CUDA12_X86_64_LINUX_GNU_SHA256SUM}        ${CUDA12_X86_64_LINUX_GNU_URL}        /cuda12-x86_64.deb
ADD --checksum=${CUDA13_AARCH64_LINUX_GNU_SHA256SUM}       ${CUDA13_AARCH64_LINUX_GNU_URL}       /cuda13-aarch64.deb
ADD --checksum=${CUDA13_CROSS_AARCH64_LINUX_GNU_SHA256SUM} ${CUDA13_CROSS_AARCH64_LINUX_GNU_URL} /cuda13-cross-aarch64.deb
ADD --checksum=${CUDA13_X86_64_LINUX_GNU_SHA256SUM}        ${CUDA13_X86_64_LINUX_GNU_URL}        /cuda13-x86_64.deb

FROM base_image AS crosstool-ng

ARG ARCHITECTURE

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt full-upgrade --quiet --assume-yes; \
    apt install --quiet --assume-yes --no-install-recommends \
        autoconf \
        automake \
        bison \
        build-essential \
        cmake \
        curl \
        file \
        flex \
        gawk \
        git \
        gperf \
        help2man \
        libexpat1-dev \
        libncurses5-dev \
        libtool \
        libtool-bin \
        python3 \
        python3-dev \
        subversion \
        texinfo \
        unzip \
        wget; \
    apt install --reinstall --quiet --assume-yes ca-certificates; \
    update-ca-certificates -f; \
    apt autoremove --purge --quiet --assume-yes; \
    apt clean --quiet --assume-yes; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN --mount=type=bind,from=downloads,source=/,target=/tmp/downloads \
    set -ex; \
    mkdir -p crosstool-ng; \
    tar xf /tmp/downloads/crosstool-ng.tar.gz -C crosstool-ng --strip-components=1; \
    cd crosstool-ng; \
    ./bootstrap; \
    ./configure --prefix=/opt/crosstool-ng; \
    make -j "$(nproc)"; \
    make install; \
    rm -rf /root/crosstool-ng

COPY crosstool-ng/$ARCHITECTURE.config /root/$ARCHITECTURE/.config
WORKDIR /root/$ARCHITECTURE

RUN --mount=type=bind,from=downloads,source=/,target=/tmp/downloads \
    set -ex; \
    CT_PREFIX=/opt/cross /opt/crosstool-ng/bin/ct-ng build; \
    rm -rf .build build.log

FROM common-host-software

ARG ARCHITECTURE
ENV ARCHITECTURE=$ARCHITECTURE
ENV TARGET=$ARCHITECTURE-linux-gnu

COPY --from=crosstool-ng --chown=root:root /opt/cross/$TARGET /opt/cross

RUN tee /opt/cross/$TARGET/toolchain-gcc.cmake <<EOF
#
# Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR $ARCHITECTURE)

set(CMAKE_LIBRARY_ARCHITECTURE "\${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
set(CMAKE_SYSROOT "/opt/cross/\${CMAKE_LIBRARY_ARCHITECTURE}/sysroot")

set(CMAKE_C_COMPILER "/opt/cross/bin/\${CMAKE_LIBRARY_ARCHITECTURE}-gcc")
set(CMAKE_C_COMPILER_TARGET "\${CMAKE_LIBRARY_ARCHITECTURE}")

set(CMAKE_CXX_COMPILER "/opt/cross/bin/\${CMAKE_LIBRARY_ARCHITECTURE}-g++")
set(CMAKE_CXX_COMPILER_TARGET "\${CMAKE_LIBRARY_ARCHITECTURE}")
set(CMAKE_CXX_FLAGS_INIT "-static-libstdc++")

set(CMAKE_C_LINKER "/opt/cross/\${CMAKE_LIBRARY_ARCHITECTURE}/bin/mold")
set(CMAKE_CXX_LINKER "/opt/cross/\${CMAKE_LIBRARY_ARCHITECTURE}/bin/mold")
set(CMAKE_LINKER_TYPE "MOLD")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

if (NOT CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL CMAKE_SYSTEM_PROCESSOR)
  set(CMAKE_CROSSCOMPILING_EMULATOR
      "/usr/bin/qemu-$ARCHITECTURE"
      "-L"
      "\${CMAKE_SYSROOT}")
endif()

# Hints for CLion. Derived from the following commands.
#
# C standard includes:
# echo \
# | /opt/cross/bin/$ARCHITECTURE-linux-gnu-gcc -E -Wp,-v - 2>&1 \
# | awk '/#include <...> search starts here/{ f = 1; next } /End of search list/{ f = 0 } f'
#
# C++ standard includes:
#
# echo \
# | /opt/cross/bin/$ARCHITECTURE-linux-gnu-gcc -E -Wp,-v -x c++ - 2>&1 \
# | awk '/#include <...> search starts here/{ f = 1; next } /End of search list/{ f = 0 } f'
#
set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES
$(echo \
| /opt/cross/bin/$ARCHITECTURE-linux-gnu-gcc -E -Wp,-v - 2>&1 \
| awk '/#include <...> search starts here/{ f = 1; next } /End of search list/{ f = 0 } f'))

set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
$(echo \
| /opt/cross/bin/$ARCHITECTURE-linux-gnu-gcc -E -Wp,-v -x c++ - 2>&1 \
| awk '/#include <...> search starts here/{ f = 1; next } /End of search list/{ f = 0 } f'))
EOF

RUN tee /opt/cross/$TARGET/toolchain-clang.cmake <<EOF
#
# Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR $ARCHITECTURE)

set(CMAKE_LIBRARY_ARCHITECTURE "\${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
set(CMAKE_SYSROOT "/opt/cross/\${CMAKE_LIBRARY_ARCHITECTURE}/sysroot")

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_C_COMPILER_TARGET "${CMAKE_LIBRARY_ARCHITECTURE}")
# When clang is used as the frontend for linking binaries that only contain
# object files containing C code (in other words, when the LINKER_LANGUAGE for
# that binary is C), clang issues an -Wunused-command-line-argument finding for
# the -static-libstdc++ flag. Although CMake provides mechanisms for using flags
# based on LINKER_LANGUAGE, they cannot be used in toolchain files. Silencing
# that warning is a compromise, as other unused flags will not be reported
set(CMAKE_C_FLAGS_INIT "--gcc-toolchain=/opt/cross/ -Wno-unused-command-line-argument")

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_COMPILER_TARGET "${CMAKE_LIBRARY_ARCHITECTURE}")
set(CMAKE_CXX_FLAGS_INIT "--gcc-toolchain=/opt/cross/")

set(CMAKE_SHARED_LINKER_FLAGS_INIT "-static-libstdc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static-libstdc++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

if (NOT CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL CMAKE_SYSTEM_PROCESSOR)
  set(CMAKE_CROSSCOMPILING_EMULATOR
      "/usr/bin/qemu-$ARCHITECTURE"
      "-L"
      "\${CMAKE_SYSROOT}")
endif()
EOF

ENV CMAKE_TOOLCHAIN_FILE=/opt/cross/$TARGET/toolchain-gcc.cmake

COPY <<EOF /opt/cross/$TARGET/toolchain-gcc.meson.txt
#
# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

[constants]
architecture = '$ARCHITECTURE'
target = '$TARGET'
crossroot = '/opt/cross'
sysroot = crossroot / target / 'sysroot'
mold = crossroot / target / 'bin' / 'mold'
linker_paths = [
  '-L' + sysroot / 'usr' / 'lib',
  '-L' + sysroot / 'usr' / 'lib' / target,
  '-L' + sysroot / 'usr' / 'local' / 'lib'
]

[binaries]
ar = crossroot / 'bin' / target + '-ar'
c = crossroot / 'bin' / target + '-gcc'
cpp = crossroot / 'bin' / target + '-g++'
ld = mold
strip = crossroot / 'bin' / target + '-strip'
pkg-config = 'pkg-config'

[host_machine]
system = 'linux'
cpu_family = architecture
cpu = architecture
endian = 'little'

[properties]
sys_root = sysroot
pkg_config_libdir = [
  sysroot / 'usr' / 'lib' / 'pkgconfig',
  sysroot / 'usr' / 'local' / 'lib' / 'pkgconfig'
]

[built-in options]
c_ld = 'mold'
c_link_args = linker_paths
cpp_ld = 'mold'
cpp_link_args = ['-static-libstdc++'] + linker_paths
EOF

RUN --mount=type=bind,from=cuda-downloads,source=/,target=/tmp/downloads \
    --mount=type=bind,source=scripts/cuda,target=/tmp/scripts/cuda \
    set -ex; \
    /tmp/scripts/cuda/cuda.sh

ENV RUST_INSTALL_PREFIX=/usr/local

RUN --mount=type=bind,from=downloads,source=/,target=/tmp/downloads \
    --mount=type=bind,source=scripts/rust,target=/tmp/scripts/rust \
 set -ex \
 && export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install --quiet --assume-yes --no-install-recommends build-essential \
 && export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/clang \
 && /tmp/scripts/rust/rust.sh \
 && RUST_MIN_STACK=33554432 CC=/usr/bin/clang CXX=/usr/bin/clang++ cargo install --force \
                  --locked \
                  --root=$RUST_INSTALL_PREFIX \
                  cargo-audit@0.22.2 \
                  cargo-auditable@0.7.5 \
                  cargo-deny@0.20.2 \
                  cargo-make@0.37.24 \
                  cargo-vet@0.10.2 \
                  bindgen-cli@0.72.1 \
                  cargo-llvm-cov@0.8.7 \
                  bacon@3.24.0 \
 && CC=/usr/bin/clang CXX=/usr/bin/clang++ cargo install --force \
                  --locked \
                  --root=$RUST_INSTALL_PREFIX \
                  --features=cli \
                  cargo-about@0.9.1 \
 && CC=/usr/bin/clang CXX=/usr/bin/clang++ cargo install --force \
                  --locked \
                  --root=$RUST_INSTALL_PREFIX \
                  --git=https://github.com/coreos/cargo-vendor-filterer \
                  --rev=26e53809accc52a628d028fe485670cdfecf8c65 \
                  cargo-vendor-filterer \
 && rm -rf $HOME/.cargo \
 && /tmp/scripts/rust/corrosion.sh \
 && apt remove --quiet --assume-yes build-essential \
 && apt autoremove --purge --quiet --assume-yes \
 && apt clean --quiet --assume-yes \
 && rm -rf /var/lib/apt/lists/*

ENV DCGM_BUILD_INSIDE_DOCKER=1
