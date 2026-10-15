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

macro(define_dcgm_cuda_worker cuda_version)
    set(dcgm_cuda_worker_lib dcgm_cuda_worker${cuda_version})

    target_include_directories(${dcgm_cuda_worker_lib}
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}")

    target_link_libraries(${dcgm_cuda_worker_lib}
        PUBLIC
            cublas_ops_interface${cuda_version}
            common_interface
            cuda_ops_interface${cuda_version}
            dcgm_logging
        PRIVATE
            dcgm_common
            dl
            rt)

    target_sources(${dcgm_cuda_worker_lib} PRIVATE
        CudaWorkerThread.cpp
        CudaWorkerThread.hpp
        CudaWorkerThreadAnyUniquePtr.hpp
        CudaWorkerThreadFactory.cpp
        CudaWorkerThreadFactory.hpp
        CudaWorkerThreadFactoryAnyPtr.hpp
        CudaWorkerThreadFactoryAnyUniquePtr.hpp
        DcgmDgemm.cpp
        DcgmDgemm.hpp
        FieldWorkerFactory.cpp
        FieldWorkerFactory.hpp
        FieldWorkerFactoryAnyUniquePtr.hpp
        FieldWorkers.hpp)
endmacro()

define_dcgm_cuda_worker(11)
define_dcgm_cuda_worker(12)
define_dcgm_cuda_worker(13)

add_subdirectory(mock)
add_subdirectory(tests)
