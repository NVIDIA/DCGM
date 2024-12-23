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

ENV CMAKE_INSTALL_PREFIX=/tmp/$TARGET
ENV CMAKE_BUILD_TYPE=RelWithDebInfo
ENV CMAKE_TOOLCHAIN_FILE=/tmp/$TARGET.cmake

COPY scripts/target /tmp/scripts/target
ARG JOBS
RUN set -ex; \
    echo $CC; \
    $CC --version; \
    mkdir --parents $CMAKE_INSTALL_PREFIX; \
    export MAKEFLAGS="--jobs=${JOBS:-$(nproc)}"; \
    find /tmp/scripts/target -name '*.sh' | sort | while read -r SCRIPT; \
    do $SCRIPT /tmp/scripts/target/urls.txt; \
    done;

ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY --from=builder --chown=root:root /tmp/$TARGET /opt/cross/$TARGET

ENV DCGM_BUILD_INSIDE_DOCKER=1
