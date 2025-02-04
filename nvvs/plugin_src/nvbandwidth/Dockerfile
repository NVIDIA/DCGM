# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

ARG BASE_IMAGE=ubuntu:20.04
FROM $BASE_IMAGE

ARG CMAKE_VERSION=3.20.6

ENV NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility"

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg2 \
        curl \
        ca-certificates \
        libxml2 \
        build-essential \
        libboost-program-options-dev \
        numactl && \
    apt clean && \
    rm --recursive --force /var/lib/apt/lists/*

# ubuntu 20.04 has default cmake 3.16, nvbandwidth requires cmake >= 3.20
# Download and install CMake
RUN curl --location \
         --output cmake.sh \
         https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-$(uname -p).sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh

RUN --mount=type=bind,source=cuda-repo.deb,target=/mnt/host/cuda-repo.deb \
    export DEBIAN_FRONTEND=noninteractive && \
    PACKAGE=$(dpkg-deb --field /mnt/host/cuda-repo.deb Package) && \
    dpkg --install /mnt/host/cuda-repo.deb && \
    cp /var/$PACKAGE/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        $(apt-cache search cuda-cudart | awk '{print $1}') \
        $(apt-cache search cuda-nvcc | awk '{print $1}') \
        $(apt-cache search cuda-nvml | awk '{print $1}') && \
    apt remove --yes $PACKAGE && \
    rm --recursive --force /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"
