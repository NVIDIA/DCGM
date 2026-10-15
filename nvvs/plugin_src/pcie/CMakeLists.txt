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
add_subdirectory(tests)

declare_nvvs_plugin(busgrind .)

set(PCIE_PLUGIN_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(pcie_bw_checker_interface INTERFACE)
target_sources(pcie_bw_checker_interface INTERFACE
    "${PCIE_PLUGIN_BASE_DIR}/bw_checker/BwCheckerMain.h")
target_include_directories(pcie_bw_checker_interface INTERFACE
    "${PCIE_PLUGIN_BASE_DIR}/bw_checker")

add_library(pcie_bw_checker STATIC)
target_sources(pcie_bw_checker PRIVATE
    "${PCIE_PLUGIN_BASE_DIR}/bw_checker/BwChecker.cpp")
target_link_libraries(pcie_bw_checker
    PUBLIC
        jsoncpp_static
        pcie_bw_checker_interface)

macro(define_bw_checker CHECKER_NAME CUDA_VER)
    target_sources(${CHECKER_NAME}_${CUDA_VER}
        PRIVATE
            "${PCIE_PLUGIN_BASE_DIR}/bw_checker/BwCheckerMain.cpp"
            "${PCIE_PLUGIN_BASE_DIR}/bw_checker/BwCheckerMain.h")

    target_include_directories(${CHECKER_NAME}_${CUDA_VER}
        PRIVATE
            "${PCIE_PLUGIN_BASE_DIR}/bw_checker")

    target_link_libraries(${CHECKER_NAME}_${CUDA_VER}
        PRIVATE
            common_interface
            dcgm_cuda_worker${CUDA_VER}
            cublas_ops${CUDA_VER}
            cuda_ops${CUDA_VER}
            pcie_bw_checker
            pcie_bw_checker_interface
            dcgm_thread dcgm_common dcgm
        PUBLIC
            jsoncpp_static
            dcgmi_interface)
endmacro()

foreach(CUDA_VER IN LISTS SUPPORTED_CUDA_VERSIONS)
    add_subdirectory(Cuda${CUDA_VER})
    target_link_libraries(Pcie_${CUDA_VER}
        PRIVATE
            cublas_ops${CUDA_VER})
endforeach()

add_subdirectory(bw_checker)
