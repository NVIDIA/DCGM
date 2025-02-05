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

add_subdirectory(mock)
macro(define_dcgm_cuda_lib cuda_version)
    set(dcgm_cuda_lib dcgm_cuda_lib${cuda_version})
    target_compile_definitions(${dcgm_cuda_lib} PRIVATE
        CUDA_VERSION_USED=${cuda_version})

    target_include_directories(${dcgm_cuda_lib} PRIVATE
        "${CUDA${cuda_version}_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}")

    target_include_directories(${dcgm_cuda_lib} INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}")

    target_link_libraries(${dcgm_cuda_lib} PRIVATE
        ${CUDA${cuda_version}_LIBS}
        ${CUDA${cuda_version}_STATIC_LIBS}
        dcgm_common
        dcgm_cublas_proxy${cuda_version}
        dcgm_logging
        cuda_lib_base_interface
        dl
        rt)

    target_sources(${dcgm_cuda_lib} PRIVATE
        CudaLib.cpp
        CudaLib.h)
endmacro()

define_dcgm_cuda_lib(11)
define_dcgm_cuda_lib(12)

target_include_directories(cuda_lib_base_interface INTERFACE
    "${CUDA12_INCLUDE_DIR}")
target_include_directories(cuda_lib_base_interface INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(cuda_lib_base_interface INTERFACE
    dcgm_common
    dcgm_logging
    fmt::fmt
    dl
    rt)

target_sources(cuda_lib_base_interface INTERFACE
    CudaLibBase.cpp
    CudaLibBase.h)