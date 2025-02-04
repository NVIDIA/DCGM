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

add_subdirectory(tests)
target_compile_definitions(cuda_mock_interface INTERFACE
    CUDA_VERSION_USED=12)

target_include_directories(cuda_mock_interface INTERFACE
    "${CUDA12_INCLUDE_DIR}")
target_include_directories(cuda_mock_interface INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(cuda_mock_interface INTERFACE
    dcgm
    dcgm_common
    dcgm_logging
    fmt::fmt
    cuda_lib_base_interface
    dl
    rt)

target_sources(cuda_mock_interface INTERFACE
    CudaLibMock.cpp
    CudaLibMock.h)