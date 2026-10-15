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

FROM scratch AS downloads

ARG CCACHE_URL
ARG CLANG_URL
ARG CMAKE_URL
ARG GIT_URL
ARG GIT_LFS_URL
ARG LCOV_URL
ARG RIPGREP_URL
ARG SCCACHE_URL

ARG CCACHE_SHA256SUM
ARG CLANG_SHA256SUM
ARG CMAKE_SHA256SUM
ARG GIT_SHA256SUM
ARG GIT_LFS_SHA256SUM
ARG LCOV_SHA256SUM
ARG RIPGREP_SHA256SUM
ARG SCCACHE_SHA256SUM

ADD --checksum=${CCACHE_SHA256SUM}  ${CCACHE_URL}  /ccache.tar.xz
ADD --checksum=${CLANG_SHA256SUM}   ${CLANG_URL}   /llvm.sh
ADD --checksum=${CMAKE_SHA256SUM}   ${CMAKE_URL}   /cmake.sh
ADD --checksum=${GIT_SHA256SUM}     ${GIT_URL}     /git.tar.xz
ADD --checksum=${GIT_LFS_SHA256SUM} ${GIT_LFS_URL} /git-lfs.tar.gz
ADD --checksum=${LCOV_SHA256SUM}    ${LCOV_URL}    /lcov.tar.gz
ADD --checksum=${RIPGREP_SHA256SUM} ${RIPGREP_URL} /ripgrep.tar.gz
ADD --checksum=${SCCACHE_SHA256SUM} ${SCCACHE_URL} /sccache.tar.gz

FROM base_image

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt full-upgrade --quiet --assume-yes; \
    apt install --quiet --assume-yes --no-install-recommends \
        bzip2 \
        build-essential \
        cpio \
        curl \
        dwz \
        elfutils \
        file \
        gettext \
        graphviz \
        libcurl4-openssl-dev \
        libedit-dev \
        libssl-dev \
        make \
        ninja-build \
        parallel \
        patch \
        pkg-config \
        pylint \
        python3 \
        python3-dev \
        python3-pip \
        qemu-system-arm \
        qemu-user \
        rpm \
        rsync \
        software-properties-common \
        unzip \
        vim \
        wget \
        xz-utils \
        yq; \
    pip3 install --break-system-packages \
                 --no-cache-dir \
                 --quiet \
                 --requirement /tmp/requirements.txt; \
    apt remove --quiet --assume-yes python3-pip; \
    apt autoremove --purge --quiet --assume-yes; \
    apt clean --quiet --assume-yes; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,from=downloads,source=/,target=/tmp/downloads \
    --mount=type=bind,source=scripts,target=/tmp/scripts \
    set -ex; \
    find /tmp/scripts -name '*.sh' | sort | while read -r SCRIPT; \
    do $SCRIPT; \
    done

# Remove packages used solely for building
RUN set -ex; \
    apt remove -y \
        build-essential; \
    apt autoremove -y --purge;
