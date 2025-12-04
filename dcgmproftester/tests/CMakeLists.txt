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

if (NOT BUILD_TESTING)
    return()
endif()

add_executable(dcgmproftester_tests)

target_sources(dcgmproftester_tests
    PRIVATE
        DistributedCudaContextTests.cpp
        PhysicalGpuStub.cpp  # Minimal stub for PhysicalGpu inheritance
        ../DistributedCudaContext.cpp
        ../Entity.cpp
        ../Arguments.cpp
        DcgmProfTesterStub.cpp  # Stub for g_signalCaught and other globals
        CudaWorkerThreadStub.cpp  # Stub for CudaWorkerThread without CUDA dependencies
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
        fmt::fmt
        testing_cuda_stubs 
        testing_dcgm_cublas_stubs      # Use cublas stubs instead of real cublas for tests
        sdk_nvml_essentials_objects  # Add NVML objects for DistributedCudaContext
        transport_objects  # Add transport objects for DistributedCudaContext
        ${CMAKE_THREAD_LIBS_INIT}
        dl  # Add dl for dynamic linking
        rt  # Add rt for real-time extensions
)

target_include_directories(dcgmproftester_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..  # Include parent directory for dcgmproftester headers
        ${CUDA12_INCLUDE_DIR}          # Include CUDA headers for compilation
        ${PROJECT_SOURCE_DIR}/cublas_proxy  # Include cublas_proxy headers
        ${PROJECT_SOURCE_DIR}/common   # Include common headers
        ${PROJECT_SOURCE_DIR}/common/CudaWorker  # Include CudaWorker headers
        ${PROJECT_SOURCE_DIR}/dcgmlib  # Include dcgmlib headers
        ${PROJECT_SOURCE_DIR}/logging  # Include logging headers
)

target_compile_definitions(dcgmproftester_tests
    PRIVATE
        # Disable actual CUDA initialization for unit tests
        DCGM_TESTING_MODE=1
        # Use CUDA stubs instead of real CUDA
        DCGM_USE_CUDA_STUBS=1
)

# Install test binary
install(
    TARGETS dcgmproftester_tests
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}"
        COMPONENT Tests)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(dcgmproftester_tests
        PROPERTIES SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
        EXTRA_ARGS --colour-mode ansi)
endif()
