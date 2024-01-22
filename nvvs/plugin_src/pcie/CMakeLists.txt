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
add_subdirectory(tests)

find_package(Cuda REQUIRED)
find_package(Jsoncpp REQUIRED)

declare_nvvs_plugin(busgrind .)

set(LOCAL_NVVS_PLUGINS "")
set(LOCAL_BW_CHECKERS "")

set(PCIE_PLUGIN_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

macro(define_bw_checker CHECKER_NAME CUDA_VER)

    add_executable(${CHECKER_NAME}_${CUDA_VER})
    target_sources(${CHECKER_NAME}_${CUDA_VER}
        PRIVATE
            ${PCIE_PLUGIN_BASE_DIR}/bw_checker/BwCheckerMain.cpp
    )

    target_link_libraries(${CHECKER_NAME}_${CUDA_VER} PRIVATE dcgm_cuda_worker12)
    target_include_directories(${CHECKER_NAME}_${CUDA_VER} PRIVATE ${PCIE_PLUGIN_BASE_DIR}/bw_checker)
    target_include_directories(${CHECKER_NAME}_${CUDA_VER} PRIVATE ${JSONCPP_INCLUDE_DIR})
    target_include_directories(${CHECKER_NAME}_${CUDA_VER} PRIVATE ${CUDA${CUDA_VER}_INCLUDE_DIR})
    target_link_libraries(${CHECKER_NAME}_${CUDA_VER} PRIVATE dcgm_cublas_proxy${CUDA_VER})

    target_link_libraries(${CHECKER_NAME}_${CUDA_VER} PRIVATE ${CUDA${CUDA_VER}_STATIC_LIBS})
    target_link_libraries(${CHECKER_NAME}_${CUDA_VER} PUBLIC ${CUDA${CUDA_VER}_LIBS})
    target_link_libraries(${CHECKER_NAME}_${CUDA_VER} PUBLIC ${JSONCPP_STATIC_LIBS})
    target_link_libraries(${CHECKER_NAME}_${CUDA_VER} PRIVATE common_interface)
    target_link_libraries(${CHECKER_NAME}_${CUDA_VER} PUBLIC dcgmi_interface)
    target_link_libraries(${CHECKER_NAME}_${CUDA_VER} PRIVATE dcgm_thread dcgm_common)

    set_target_properties(${CHECKER_NAME}_${CUDA_VER} PROPERTIES INSTALL_TO_DIR ${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda${CUDA_VER})
    
    if ((${CUDA_VER} EQUAL 10))
        target_compile_options(${CHECKER_NAME}_${CUDA_VER} PRIVATE -Wno-volatile)
    endif()
    
    set(LOCAL_BW_CHECKERS "${LOCAL_BW_CHECKERS};${CHECKER_NAME}_${CUDA_VER}" PARENT_SCOPE)
endmacro()

foreach(CUDA_VER IN LISTS SUPPORTED_CUDA_VERSIONS)
    add_subdirectory(Cuda${CUDA_VER})
endforeach()

add_subdirectory(bw_checker)

set(NVVS_PLUGINS "${NVVS_PLUGINS};${LOCAL_NVVS_PLUGINS}" PARENT_SCOPE)
set(BW_CHECKERS "${LOCAL_BW_CHECKERS}" PARENT_SCOPE)
