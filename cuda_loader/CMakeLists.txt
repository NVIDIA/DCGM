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
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(CUDA_VERSION 11)
else()
    set(CUDA_VERSION 10)
endif()
find_package(Cuda REQUIRED)

add_library(cuda_loader_interface INTERFACE)
target_link_libraries(cuda_loader_interface INTERFACE sdk_nvml_interface)
target_include_directories(cuda_loader_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cuda_loader_interface INTERFACE ${CUDA${CUDA_VERSION}_INCLUDE_DIR})

add_library(cuda_loader_objects STATIC)
target_link_libraries(cuda_loader_objects PUBLIC cuda_loader_interface dl)
target_compile_definitions(cuda_loader_objects PRIVATE __CUDA_API_VERSION_INTERNAL)
target_sources(
    cuda_loader_objects
    PRIVATE
    cuda-entrypoints.h
    cuda-hook.h
    cuda-loader.c
    cuda-loader.h
)

target_link_libraries(cuda_loader_objects PUBLIC ${CUDA${CUDA_VERSION}_STATIC_LIBS})
