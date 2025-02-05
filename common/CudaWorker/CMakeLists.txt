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

macro(define_dcgm_cuda_worker cuda_version)
    set(dcgm_cuda_worker_lib dcgm_cuda_worker${cuda_version})
    target_compile_definitions(${dcgm_cuda_worker_lib} PRIVATE
        CUDA_VERSION_USED=${cuda_version})

    target_include_directories(${dcgm_cuda_worker_lib} PRIVATE
        "${CUDA${cuda_version}_INCLUDE_DIR}")

    target_link_libraries(${dcgm_cuda_worker_lib} PRIVATE
        ${CUDA${cuda_version}_LIBS}
        ${CUDA${cuda_version}_STATIC_LIBS}
        dcgm_common
        dcgm_cublas_proxy${cuda_version}
        dcgm_logging
        dl
        rt)

    target_sources(${dcgm_cuda_worker_lib} PRIVATE
        CudaWorkerThread.cpp
        CudaWorkerThread.hpp
        DcgmDgemm.cpp)
endmacro()

define_dcgm_cuda_worker(11)
define_dcgm_cuda_worker(12)
