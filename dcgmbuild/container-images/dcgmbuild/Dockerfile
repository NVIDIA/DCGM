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

FROM scratch AS dcgmbuild-downloads

ARG BOOST_URL
ARG CATCH2_URL
ARG FMT_URL
ARG JSONCPP_URL
ARG LIBEVENT_URL
ARG LIBNUMA_URL
ARG PLOG_URL
ARG TCLAP_URL
ARG YAML_CPP_URL
ARG ZLIB_URL

ARG BOOST_SHA256SUM
ARG CATCH2_SHA256SUM
ARG FMT_SHA256SUM
ARG JSONCPP_SHA256SUM
ARG LIBEVENT_SHA256SUM
ARG LIBNUMA_SHA256SUM
ARG PLOG_SHA256SUM
ARG TCLAP_TAG
ARG YAML_CPP_SHA256SUM
ARG ZLIB_SHA256SUM

ADD --checksum=${BOOST_SHA256SUM}    ${BOOST_URL}              /boost.tar.gz
ADD --checksum=${CATCH2_SHA256SUM}   ${CATCH2_URL}             /catch2.tar.gz
ADD --checksum=${FMT_SHA256SUM}      ${FMT_URL}                /fmt.tar.gz
ADD --checksum=${JSONCPP_SHA256SUM}  ${JSONCPP_URL}            /jsoncpp.tar.gz
ADD --checksum=${LIBEVENT_SHA256SUM} ${LIBEVENT_URL}           /libevent.tar.gz
ADD --checksum=${LIBNUMA_SHA256SUM}  ${LIBNUMA_URL}            /libnuma.tar.gz
ADD --checksum=${PLOG_SHA256SUM}     ${PLOG_URL}               /plog.tar.gz
ADD                                  ${TCLAP_URL}#${TCLAP_TAG} /tclap
ADD --checksum=${YAML_CPP_SHA256SUM} ${YAML_CPP_URL}           /yaml-cpp.tar.gz
ADD --checksum=${ZLIB_SHA256SUM}     ${ZLIB_URL}               /zlib.tar.gz

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
ENV AS=/opt/cross/bin/$TARGET-as
ENV LD=/opt/cross/$TARGET/bin/mold

ENV CMAKE_STAGING_PREFIX=/tmp/$TARGET/usr/local
ENV CMAKE_INSTALL_PREFIX=/usr/local
ENV CMAKE_BUILD_TYPE=RelWithDebInfo

ARG JOBS
RUN --mount=type=bind,from=dcgmbuild-downloads,source=/,target=/tmp/downloads \
    --mount=type=bind,source=scripts,target=/tmp/scripts \
    set -ex; \
    mkdir --parents $CMAKE_STAGING_PREFIX; \
    export MAKEFLAGS="--jobs=${JOBS:-$(nproc)}"; \
    find /tmp/scripts -name '*.sh' | sort | while read -r SCRIPT; \
    do $SCRIPT; \
    done

FROM toolchain

COPY --from=builder --chown=root:root /tmp/$TARGET /opt/cross/$TARGET/sysroot
