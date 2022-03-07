# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
find_package(Cuda REQUIRED)
include(utils)

macro(define_proxy_lib CUDA_VER)
    set(lib_name dcgm_cublas_proxy${CUDA_VER})
    add_library(${lib_name} SHARED)

    target_compile_definitions(${lib_name} PRIVATE CUDA_VERSION_USED=${CUDA_VER})
    target_compile_options(${lib_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_link_options(${lib_name} PRIVATE -Wl,--exclude-libs,ALL)
    target_link_options(${lib_name} PRIVATE -Xlinker --version-script=${CMAKE_CURRENT_SOURCE_DIR}/../cublas_proxy.linux_def)

    target_sources(${lib_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cublas_proxy.cpp)
    set(cublas_proxy_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/..")
    get_absolute_path(${cublas_proxy_include_dir} cublas_proxy_include_dir)
    target_include_directories(${lib_name} PUBLIC $<BUILD_INTERFACE:${cublas_proxy_include_dir}>)
    target_include_directories(${lib_name} PUBLIC $<BUILD_INTERFACE:${CUDA${CUDA_VER}_INCLUDE_DIR}>)
    target_link_libraries(${lib_name} PRIVATE dcgm_common dcgm_logging dcgm_mutex)
    target_link_libraries(${lib_name} PRIVATE ${CUDA${CUDA_VER}_STATIC_CUBLAS_LIBS})
    target_link_libraries(${lib_name} PRIVATE ${CUDA${CUDA_VER}_STATIC_LIBS})
    target_link_libraries(${lib_name} PRIVATE ${CUDA${CUDA_VER}_LIBS})

    set(LOCAL_CUBLAS_PROXY "${LOCAL_CUBLAS_PROXY};${lib_name}" PARENT_SCOPE)

endmacro()

set(LOCAL_CUBLAS_PROXY "")

add_subdirectory(Cuda11)
if (NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_subdirectory(Cuda9)
    add_subdirectory(Cuda10)
endif()

set(DCGM_CUBLAS_PROXY ${LOCAL_CUBLAS_PROXY} PARENT_SCOPE)

