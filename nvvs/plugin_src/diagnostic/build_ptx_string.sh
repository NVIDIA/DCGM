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

# This script generates compare.cubin and gpuburn.ptx
# gpuburn.ptx is then converted to gpuburn_ptx_string.h as a hexified string
# Last, symbol names are added to gpuburn_ptx_string.h by find_ptx_symbols.py
#
# sm_30 is used here for Kepler or newer

#/usr/local/cuda/bin/nvcc -arch=sm_50 -ptx -keep compare.cu
CUDA_IMAGE=nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04

docker run \
    --rm \
    -v $(pwd):/work \
    -w /work \
    ${CUDA_IMAGE} \
    /bin/bash -c "/usr/local/cuda/bin/nvcc -ptx -m64 -arch=sm_30 -o compare.ptx compare.cu || die 'Failed to compile compare.cu'; \
                  /usr/local/cuda/bin/bin2c compare.ptx --padd 0 --name gpuburn_ptx_string > gpuburn_ptx_string.h; chmod a+w gpuburn_ptx_string.h"
python find_ptx_symbols.py compare.ptx gpuburn_ptx_string.h
