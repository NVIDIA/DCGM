
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

# This script generates memtest_kernel.ptx and l1tag.ptx
# ptx files are then converted to ptx_string.h as a hexified string
# Last, symbol names are added to ptx_string.h by find_ptx_symbols.py
#
# sm_30 is used here for Kepler or newer

CUDA_IMAGE=nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04

targets=(l1tag memtest_kernel)
for target in "${targets[@]}"; do
    docker run \
        --rm \
        -v "$(pwd)":/work \
        -w /work \
        ${CUDA_IMAGE} \
        /bin/bash -c "/usr/local/cuda/bin/nvcc -ptx -m64  -arch=sm_30 -o ${target}.ptx ${target}.cu || { echo 'Failed to compile ${target}.cu'; exit 1; }; \
                        /usr/local/cuda/bin/bin2c ${target}.ptx --padd 0 --name ${target}_ptx_string > ${target}_ptx_string.h; chmod a+w ${target}_ptx_string.h;"
    python find_ptx_symbols.py ${target}.ptx ${target}_ptx_string.h
done