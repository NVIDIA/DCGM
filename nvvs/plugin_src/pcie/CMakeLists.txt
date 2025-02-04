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

declare_nvvs_plugin(busgrind .)

set(PCIE_PLUGIN_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

macro(define_bw_checker CHECKER_NAME CUDA_VER)
    target_sources(${CHECKER_NAME}_${CUDA_VER}
        PRIVATE
            "${PCIE_PLUGIN_BASE_DIR}/bw_checker/BwCheckerMain.cpp")

    target_include_directories(${CHECKER_NAME}_${CUDA_VER}
        PRIVATE
            "${PCIE_PLUGIN_BASE_DIR}/bw_checker"
            "${JSONCPP_INCLUDE_DIR}"
            "${CUDA${CUDA_VER}_INCLUDE_DIR}")

    target_link_libraries(${CHECKER_NAME}_${CUDA_VER}
        PRIVATE
            ${CUDA${CUDA_VER}_STATIC_LIBS}
            common_interface
            dcgm_cublas_proxy${CUDA_VER}
            dcgm_cuda_worker12
            dcgm_thread dcgm_common dcgm
        PUBLIC
            ${CUDA${CUDA_VER}_LIBS}
            ${JSONCPP_STATIC_LIBS}
            dcgmi_interface)
endmacro()

foreach(CUDA_VER IN LISTS SUPPORTED_CUDA_VERSIONS)
    add_subdirectory(Cuda${CUDA_VER})
endforeach()

add_subdirectory(bw_checker)
