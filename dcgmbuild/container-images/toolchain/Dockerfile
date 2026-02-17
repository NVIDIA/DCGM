# syntax=docker/dockerfile:1.4

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

FROM base_image AS crosstool-ng

ARG ARCHITECTURE
ARG CROSSTOOL_SHA512SUM=45736acb5ead4e50b7565d328739108cd0200838b9d6e4437d1718ec93bc12c404a3c3780386a409a7 ebf4418c6c18844bd4bfd1ad17c9301533d5e89b1a50e2
ARG CROSSTOOL_URL=https://github.com/crosstool-ng/crosstool-ng/archive/ed12fa68402f58e171a6f79500f73f4781fdc9e5.tar.gz

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt full-upgrade --quiet --assume-yes; \
    apt install --quiet --assume-yes --no-install-recommends \
        autoconf \
        automake \
        bison \
        build-essential \
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

RUN set -ex; \
    mkdir -p crosstool-ng; \
    wget $CROSSTOOL_URL -O crosstool-ng.tar.gz; \
    echo "$CROSSTOOL_SHA512SUM  crosstool-ng.tar.gz" | sha512sum -c -; \
    tar xf crosstool-ng.tar.gz -C crosstool-ng --strip-components=1; \
    cd crosstool-ng; \
    ./bootstrap; \
    ./configure --prefix=/opt/crosstool-ng; \
    make -j12; \
    make install; \
    rm -rf /root/crosstool-ng

RUN useradd --create-home builder \
 && mkdir /opt/cross \
 && chown builder:builder /opt/cross

USER builder
COPY --chown=builder:builder crosstool-ng/$ARCHITECTURE.config /home/builder/$ARCHITECTURE/.config
WORKDIR /home/builder/$ARCHITECTURE

RUN CT_PREFIX=/opt/cross /opt/crosstool-ng/bin/ct-ng build

FROM common-host-software

ARG ARCHITECTURE
ENV ARCHITECTURE=$ARCHITECTURE
ENV TARGET=$ARCHITECTURE-linux-gnu

COPY --from=crosstool-ng --chown=root:root /opt/cross/$TARGET /opt/cross

RUN tee /opt/cross/$TARGET/toolchain.cmake <<EOF
#
# Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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

ENV CMAKE_TOOLCHAIN_FILE=/opt/cross/$TARGET/toolchain.cmake

COPY <<EOF /opt/cross/$TARGET/toolchain.meson.txt
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
linker_paths = [
  '-L' + sysroot / 'usr' / 'lib',
  '-L' + sysroot / 'usr' / 'lib' / target,
  '-L' + sysroot / 'usr' / 'local' / 'lib'
]

[binaries]
ar = crossroot / 'bin' / target + '-ar'
c = crossroot / 'bin' / target + '-gcc'
cpp = crossroot / 'bin' / target + '-g++'
ld = crossroot / 'bin' / target + '-ld'
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
c_ld = crossroot / 'bin' / target + '-ld'
c_link_args = linker_paths
cpp_ld = crossroot / 'bin' / target + '-ld'
cpp_link_args = ['-static-libstdc++'] + linker_paths
EOF

RUN --mount=type=bind,source=scripts/cuda,target=/tmp/scripts/cuda \
 set -ex \
 && /tmp/scripts/cuda/cuda.sh /tmp/scripts/cuda/urls.txt

ENV RUST_INSTALL_PREFIX=/usr/local

RUN --mount=type=bind,source=scripts/rust,target=/tmp/scripts/rust \
 set -ex \
 && export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/clang \
 && /tmp/scripts/rust/rust.sh /tmp/scripts/rust/urls.txt \
 && CC=/usr/bin/clang CXX=/usr/bin/clang++ cargo install --force \
                  --locked \
                  --root=$RUST_INSTALL_PREFIX \
                  cargo-audit@0.21.2 \
                  cargo-auditable@0.7.0 \
                  cargo-about@0.7.1 \
                  cargo-deny@0.18.4 \
                  cargo-make@0.37.24 \
                  cargo-vet@0.10.1 \
                  bindgen-cli@0.72.1 \
                  bacon \
 && rm -rf $HOME/.cargo \
 && /tmp/scripts/rust/corrosion.sh /tmp/scripts/rust/urls.txt

ENV DCGM_BUILD_INSIDE_DOCKER=1
