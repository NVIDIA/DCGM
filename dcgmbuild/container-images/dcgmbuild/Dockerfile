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

FROM toolchain AS builder

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt install --quiet --assume-yes \
        autoconf \
        automake \
        libtool; \
    apt autoremove --purge --quiet --assume-yes; \
    apt clean --quiet --assume-yes; \
    rm -rf /var/lib/apt/lists/*

ENV CC=/opt/cross/bin/$TARGET-gcc
ENV CPP=/opt/cross/bin/$TARGET-cpp
ENV CXX=/opt/cross/bin/$TARGET-g++
ENV LD=/opt/cross/bin/$TARGET-ld
ENV AS=/opt/cross/bin/$TARGET-as

ENV CMAKE_STAGING_PREFIX=/tmp/$TARGET/usr/local
ENV CMAKE_INSTALL_PREFIX=/usr/local
ENV CMAKE_BUILD_TYPE=RelWithDebInfo

ARG JOBS
RUN --mount=type=bind,source=scripts,target=/tmp/scripts \
    set -ex; \
    mkdir --parents $CMAKE_STAGING_PREFIX; \
    export MAKEFLAGS="--jobs=${JOBS:-$(nproc)}"; \
    find /tmp/scripts -name '*.sh' | sort | while read -r SCRIPT; \
    do $SCRIPT /tmp/scripts/urls.txt; \
    done;

FROM toolchain

COPY --from=builder --chown=root:root /tmp/$TARGET /opt/cross/$TARGET/sysroot
