# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

if (BUILD_TESTING)
    add_library(testing_cuda_stubs STATIC)
    target_sources(testing_cuda_stubs PRIVATE CudaCommonStub.cpp CudaStub.cpp cu_stubs.cpp)
    target_include_directories(testing_cuda_stubs PRIVATE ${CUDA12_INCLUDE_DIR})
    target_link_libraries(testing_cuda_stubs PUBLIC nvvs_interface)
    target_link_libraries(testing_cuda_stubs PRIVATE nvvs_plugins_common_interface)

    add_library(testing_cuda_stubs_interface INTERFACE)
    target_include_directories(testing_cuda_stubs_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(testing_cuda_stubs_interface INTERFACE ${CUDA12_INCLUDE_DIR})

    add_library(testing_dcgm_cublas_stubs STATIC)
    target_sources(testing_dcgm_cublas_stubs PRIVATE CublasStub.cpp)
    target_include_directories(testing_dcgm_cublas_stubs PRIVATE ${DCGM_ROOT_DIR}/cublas_proxy)
    target_include_directories(testing_dcgm_cublas_stubs PRIVATE ${CUDA12_INCLUDE_DIR})
endif()
