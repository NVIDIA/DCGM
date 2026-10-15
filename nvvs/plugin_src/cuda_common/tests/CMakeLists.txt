# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

add_executable(cudacommontests)
target_sources(cudacommontests
    PRIVATE
        CudaCommonTests.cpp
        ../CudaCommon.cpp
)

target_link_libraries(cudacommontests
    PRIVATE
        Catch2::Catch2WithMain
        nvvs_plugins_cudaCommon_interface
        mock_cuda_ops
        mock_cuda_common
        mock_cublas_ops
)
target_link_libraries(cudacommontests PRIVATE yaml-cpp::yaml-cpp jsoncpp_static rt dl)
target_link_libraries(cudacommontests PRIVATE Threads::Threads)
target_link_libraries(cudacommontests PRIVATE sdk_nvml_loader)
target_link_libraries(cudacommontests
    PRIVATE
        sdk_nvml_essentials_objects
        dcgm_common
        dcgm_logging
        nvvs_interface
        nvvs_plugins_common_objects
        nvvs_without_main_objects
)
# target_link_libraries(cudacommontests PUBLIC )

target_link_options(cudacommontests PRIVATE -rdynamic)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(cudacommontests EXTRA_ARGS --colour-mode ansi)
endif()
