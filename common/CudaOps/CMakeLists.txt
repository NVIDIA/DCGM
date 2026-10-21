# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

macro(define_cuda_ops CUDA_VER)
    # Link against the interface target if you do not need to/cannot link
    # against the real CUDA library. This is effectively required for unit tests
    target_include_directories(cuda_ops_interface${CUDA_VER} INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CUDA${CUDA_VER}_INCLUDE_DIR}")

    target_compile_definitions(cuda_ops_interface${CUDA_VER} INTERFACE
        CUDA_VERSION_USED=${CUDA_VER})

    # Link against the regular target if you need the real CUDA library
    target_link_libraries(cuda_ops${CUDA_VER}
        PUBLIC
            cuda_ops_interface${CUDA_VER}
        PRIVATE
            ${CUDA${CUDA_VER}_LIBS}
            ${CUDA${CUDA_VER}_STATIC_LIBS})

    target_sources(cuda_ops${CUDA_VER} PRIVATE
        CudaOps.hpp
        CudaOps.cpp)
endmacro()

define_cuda_ops(11)
define_cuda_ops(12)
define_cuda_ops(13)

add_subdirectory(mock)
