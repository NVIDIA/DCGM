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

ARG ARCHITECTURE=x86_64
ARG CROSSTOOL_BASE_IMAGE=ubuntu:24.04
ARG TOOLCHAIN_BASE_IMAGE=dcgm/common-host-software:latest

FROM $CROSSTOOL_BASE_IMAGE AS crosstool-ng

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

ARG TOOLCHAIN_BASE_IMAGE
FROM $TOOLCHAIN_BASE_IMAGE

ARG ARCHITECTURE
ENV ARCHITECTURE=$ARCHITECTURE
ENV TARGET=$ARCHITECTURE-linux-gnu

COPY --from=crosstool-ng --chown=root:root /opt/cross/$TARGET /opt/cross
