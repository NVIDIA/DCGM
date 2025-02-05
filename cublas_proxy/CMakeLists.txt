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

include(utils)

macro(define_proxy_lib CUDA_VER)
    set(lib_name dcgm_cublas_proxy${CUDA_VER})

    target_compile_definitions(${lib_name} PRIVATE CUDA_VERSION_USED=${CUDA_VER})
    target_link_options(${lib_name} PRIVATE
        -Wl,--no-undefined
        -Wl,--exclude-libs,ALL
        -Xlinker "--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../cublas_proxy.linux_def")

    target_sources(${lib_name} PRIVATE ../cublas_proxy.cpp)

    target_include_directories(${lib_name} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CUDA${CUDA_VER}_INCLUDE_DIR}>)

    target_link_libraries(${lib_name} PRIVATE
        ${CUDA${CUDA_VER}_LIBS}
        ${CUDA${CUDA_VER}_STATIC_CUBLAS_LIBS}
        ${CUDA${CUDA_VER}_STATIC_LIBS}
        dcgm dcgm_common dcgm_logging dcgm_mutex
        dl
        rt)
endmacro()

add_subdirectory(Cuda11)
add_subdirectory(Cuda12)
