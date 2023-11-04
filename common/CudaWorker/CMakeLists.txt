# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

macro(define_dcgm_cuda_worker cuda_version)
    set(dcgm_cuda_worker_lib dcgm_cuda_worker${cuda_version})
    add_library(${dcgm_cuda_worker_lib} STATIC)
    target_compile_definitions(${dcgm_cuda_worker_lib} PRIVATE CUDA_VERSION_USED=${cuda_version})
    target_include_directories(${dcgm_cuda_worker_lib} PRIVATE ${CUDA${cuda_version}_INCLUDE_DIR})
    target_link_libraries(${dcgm_cuda_worker_lib} PRIVATE dl rt)
    target_link_libraries(${dcgm_cuda_worker_lib} PRIVATE dcgm_common)
    target_link_libraries(${dcgm_cuda_worker_lib} PRIVATE dcgm_logging)
    target_link_libraries(${dcgm_cuda_worker_lib} PRIVATE dcgm_cublas_proxy${cuda_version})
    target_link_libraries(${dcgm_cuda_worker_lib} PRIVATE ${CUDA${cuda_version}_STATIC_LIBS})
    target_link_libraries(${dcgm_cuda_worker_lib} PRIVATE ${CUDA${cuda_version}_LIBS})
    target_sources(
        ${dcgm_cuda_worker_lib}
        PRIVATE
        CudaWorkerThread.cpp
        CudaWorkerThread.hpp
    )

    #    if (${cuda_version} GREATER 10)
        target_sources(${dcgm_cuda_worker_lib} PRIVATE DcgmDgemm.cpp)
    #else()
    #    target_compile_options(${dcgm_cuda_worker_lib} PRIVATE -Wno-volatile)
    #endif()

    set(LOCAL_DCGM_CUDA_WORKER "${LOCAL_DCGM_CUDA_WORKER};${dcgm_cuda_worker_lib}" PARENT_SCOPE)
endmacro()

set(LOCAL_DCGM_CUDA_WORKER "")

#if (NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
#    define_dcgm_cuda_worker(10)
#endif()
#define_dcgm_cuda_worker(11)
define_dcgm_cuda_worker(12)

set(DCGM_CUDA_WORKER ${LOCAL_DCGM_CUDA_WORKER} PARENT_SCOPE)
