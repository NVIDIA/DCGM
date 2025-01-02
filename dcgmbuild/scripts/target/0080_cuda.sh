#!/usr/bin/env bash
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

set -ex

for CUDA in cuda11 cuda12
do
    read -r _ URL SHA512SUM <<<$(grep "^$CUDA-$TARGET" $1)

    curl --location --fail --output $CUDA.deb $URL
    echo "$SHA512SUM $CUDA.deb" | sha512sum --check -

    mkdir $CUDA
    dpkg --extract $CUDA.deb $CUDA
    CACHE=$(sed -r 's|.*file:///([^ ]+).*|\1|' $CUDA/etc/apt/sources.list.d/cuda-*.list)
    for PACKAGE in cuda-cccl \
                   cuda-cublas-dev \
                   cuda-cudart-dev \
                   cuda-cufft-dev \
                   cuda-cupti \
                   cuda-curand-dev \
                   cuda-crt \
                   cuda-driver-dev \
                   cuda-license \
                   cuda-misc-headers \
                   cuda-nvcc \
                   cuda-nvml-dev \
                   libcublas-dev \
                   libcufft-dev \
                   libcurand-dev
    do
        find $CUDA/$CACHE -name "$PACKAGE*" -exec dpkg --extract {} $CMAKE_INSTALL_PREFIX \;
    done

    rm -rf $CUDA $CUDA.deb
done
