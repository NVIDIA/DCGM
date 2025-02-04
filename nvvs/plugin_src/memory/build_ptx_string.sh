#!/bin/bash

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

# This script generates l1tag.cu and l1tag.ptx
# l1tag.ptx is then converted to l1tag_ptx_string as a hexified string
# Last, symbol names are added to l1tag_ptx_string by find_ptx_symbols.py
#
# sm_30 is used here for Kepler or newer

CUDA_IMAGE=nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04
docker run \
    --rm \
    -v $(pwd):/work \
    -w /work \
    ${CUDA_IMAGE} \
    /bin/bash -c "/usr/local/cuda/bin/nvcc -ptx -m64  -arch=sm_30 -o l1tag.ptx l1tag.cu || die 'Failed to compile l1tag.cu'; \
                    /usr/local/cuda/bin/bin2c l1tag.ptx --padd 0 --name l1tag_ptx_string > l1tag_ptx_string.h; chmod a+w l1tag_ptx_string.h;"
python find_ptx_symbols.py l1tag.ptx l1tag_ptx_string.h
