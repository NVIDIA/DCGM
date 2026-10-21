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

if (NOT BUILD_TESTING)
    return()
endif()

add_library(mock_dcgmproftester STATIC)
target_sources(mock_dcgmproftester
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/MockDistributedCudaContextRuntime.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MockDistributedCudaContextRuntimeFactory.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MockDistributedCudaContextRuntime.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MockDcgmProfTester.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MockPhysicalGpu.hpp")

target_include_directories(mock_dcgmproftester PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(mock_dcgmproftester PUBLIC
    dcgmproftester_interface
    mock_dcgm_cuda_worker
    PRIVATE
        Catch2::Catch2)
