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

add_executable(dcgmproftester_tests)

target_sources(dcgmproftester_tests
    PRIVATE
        ArgumentsTests.cpp
        DcgmProfTesterTests.cpp
        DistributedCudaContextTests.cpp
        EntityTests.cpp
        PhysicalGpuTests.cpp
)

target_link_libraries(dcgmproftester_tests
    PRIVATE
        Catch2::Catch2WithMain
        common_interface
        dcgm
        dcgm_common
        dcgm_interface  # Add dcgm_interface for DistributedCudaContext
        dcgm_logging
        dcgm_mutex
        dcgmproftester_lib12
        fmt::fmt
        mock_dcgmproftester
        mock_dcgm_ops
        mock_cublas_ops
        mock_cuda_ops
        sdk_nvml_essentials_objects  # Add NVML objects for DistributedCudaContext
        transport_objects  # Add transport objects for DistributedCudaContext
        test-utils
        Threads::Threads
        dl  # Add dl for dynamic linking
        rt  # Add rt for real-time extensions
)

# Install test binary
install(
    TARGETS dcgmproftester_tests
    RUNTIME
        DESTINATION "${DCGM_TESTS_APP_DIR}/${DCGM_TESTS_ARCH}"
        COMPONENT Tests)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(dcgmproftester_tests
        PROPERTIES SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
        EXTRA_ARGS --colour-mode ansi)
endif()
