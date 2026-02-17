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

FROM base_image

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt full-upgrade --quiet --assume-yes; \
    apt install --quiet --assume-yes --no-install-recommends \
        bzip2 \
        cpio \
        curl \
        dwz \
        elfutils \
        file \
        gettext \
        graphviz \
        libedit-dev \
        make \
        ninja-build \
        patch \
        pkg-config \
        pylint \
        python3 \
        python3-dev \
        python3-distro \
        python3-pathspec \
        python3-pip \
        python3-requests \
        qemu-system-arm \
        qemu-user \
        rpm \
        software-properties-common \
        unzip \
        vim \
        wget \
        xz-utils \
        yq; \
    add-apt-repository ppa:git-core; \
    apt update; \
    apt install --quiet --assume-yes --no-install-recommends git; \
    pip3 install --break-system-packages \
                 --no-cache-dir \
                 --quiet \
                 --requirement /tmp/requirements.txt; \
    apt remove --quiet --assume-yes python3-pip; \
    apt autoremove --purge --quiet --assume-yes; \
    apt clean --quiet --assume-yes; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=scripts,target=/tmp/scripts \
    set -ex; \
    find /tmp/scripts -name '*.sh' | sort | while read -r SCRIPT; \
    do $SCRIPT /tmp/scripts/urls.txt; \
    done;
